//! L2 reranker: rerank L1 candidates by exact dense vector distance on stored vectors
//!
//! Optimized for throughput:
//! - Candidates grouped by segment for batched I/O
//! - Flat indexes sorted for sequential mmap access (OS readahead)
//! - Bounded SIMD batches (not per-candidate scoring or unbounded raw buffers)
//! - Reusable per-segment scratch buffers
//! - unit_norm fast path: skip per-vector norm when vectors are pre-normalized

use futures::{StreamExt, TryStreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use crate::dsl::Field;

use super::{MultiValueCombiner, ScoredPosition, SearchResult, compare_search_results_desc};

/// Maximum stored vectors expanded from the document candidate set by one L2
/// rerank request. Candidate documents are bounded separately by the server;
/// this closes the multi-value/ordinal multiplier.
const MAX_L2_RERANK_VECTORS: usize = 500_000;
const MAX_L2_RERANK_VECTOR_BYTES: usize = 512 * 1024 * 1024;
const RERANK_SCORE_BATCH: usize = 4_096;
const MAX_RERANK_RAW_BATCH_BYTES: usize = 8 * 1024 * 1024;
const MAX_CONCURRENT_RERANK_SEGMENTS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RerankerKind {
    Dense,
    Binary,
}

fn validate_reranker_config<D: crate::directories::Directory + 'static>(
    searcher: &crate::index::Searcher<D>,
    config: &RerankerConfig,
) -> crate::error::Result<RerankerKind> {
    if !config.rrf_k.is_finite() || config.rrf_k < 0.0 {
        return Err(crate::Error::Query(format!(
            "reranker rrf_k must be finite and non-negative, got {}",
            config.rrf_k
        )));
    }
    config.combiner.validate().map_err(crate::Error::Query)?;
    if config.vector.is_empty() == config.binary_vector.is_empty() {
        return Err(crate::Error::Query(
            "reranker must provide exactly one of vector or binary_vector".to_string(),
        ));
    }

    let entry = searcher
        .schema()
        .get_field_entry(config.field)
        .ok_or_else(|| crate::Error::FieldNotFound(config.field.0.to_string()))?;
    if !config.binary_vector.is_empty() {
        if entry.field_type != crate::dsl::FieldType::BinaryDenseVector {
            return Err(crate::Error::InvalidFieldType {
                expected: "binary_dense_vector".to_string(),
                got: format!("{:?}", entry.field_type),
            });
        }
        let field_config = entry.binary_dense_vector_config.as_ref().ok_or_else(|| {
            crate::Error::Schema(format!(
                "binary dense vector field '{}' has no configuration",
                entry.name
            ))
        })?;
        if field_config.dim == 0 || !field_config.dim.is_multiple_of(8) {
            return Err(crate::Error::Schema(format!(
                "binary dense vector field '{}' has invalid dimension {}",
                entry.name, field_config.dim
            )));
        }
        if config.binary_vector.len() != field_config.byte_len() {
            return Err(crate::Error::Query(format!(
                "reranker binary vector byte length {} does not match field '{}' byte length {}",
                config.binary_vector.len(),
                entry.name,
                field_config.byte_len()
            )));
        }
        if config.matryoshka_dims.is_some() {
            return Err(crate::Error::Query(
                "reranker matryoshka_dims is not supported for binary vectors".to_string(),
            ));
        }
        return Ok(RerankerKind::Binary);
    }

    if entry.field_type != crate::dsl::FieldType::DenseVector {
        return Err(crate::Error::InvalidFieldType {
            expected: "dense_vector".to_string(),
            got: format!("{:?}", entry.field_type),
        });
    }
    let field_config = entry.dense_vector_config.as_ref().ok_or_else(|| {
        crate::Error::Schema(format!(
            "dense vector field '{}' has no configuration",
            entry.name
        ))
    })?;
    if config.vector.len() != field_config.dim {
        return Err(crate::Error::Query(format!(
            "reranker vector dimension {} does not match field '{}' dimension {}",
            config.vector.len(),
            entry.name,
            field_config.dim
        )));
    }
    if let Some((index, value)) = config
        .vector
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(crate::Error::Query(format!(
            "reranker vector contains non-finite value {value} at index {index}"
        )));
    }
    if config.unit_norm != field_config.unit_norm {
        return Err(crate::Error::Query(format!(
            "reranker unit_norm={} does not match field '{}' unit_norm={}",
            config.unit_norm, entry.name, field_config.unit_norm
        )));
    }
    if let Some(dims) = config.matryoshka_dims
        && (dims == 0 || dims > field_config.dim)
    {
        return Err(crate::Error::Query(format!(
            "reranker matryoshka_dims must be in 1..={}, got {dims}",
            field_config.dim
        )));
    }
    Ok(RerankerKind::Dense)
}

fn reserve_rerank_vectors(
    vector_budget: &AtomicUsize,
    byte_budget: &AtomicUsize,
    count: usize,
    vector_byte_size: usize,
) -> crate::error::Result<()> {
    let bytes = count.checked_mul(vector_byte_size).ok_or_else(|| {
        crate::Error::Query("reranker stored-vector byte budget overflow".to_string())
    })?;
    byte_budget
        .fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |used| {
            used.checked_add(bytes)
                .filter(|&next| next <= MAX_L2_RERANK_VECTOR_BYTES)
        })
        .map_err(|used| {
            crate::Error::Query(format!(
                "reranker reads more than {MAX_L2_RERANK_VECTOR_BYTES} stored vector bytes \
                 (already reserved {used}, next document needs {bytes})"
            ))
        })?;

    vector_budget
        .fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |used| {
            used.checked_add(count)
                .filter(|&next| next <= MAX_L2_RERANK_VECTORS)
        })
        .map(|_| ())
        .map_err(|used| {
            crate::Error::Query(format!(
                "reranker expands to more than {MAX_L2_RERANK_VECTORS} stored vectors \
                 (already reserved {used}, next document has {count})"
            ))
        })
}

#[inline]
fn rerank_batch_len(vector_byte_size: usize) -> usize {
    RERANK_SCORE_BATCH.min((MAX_RERANK_RAW_BATCH_BYTES / vector_byte_size.max(1)).max(1))
}

/// Precomputed query data for dense reranking (computed once, reused across segments).
struct PrecompQuery<'a> {
    query: &'a [f32],
    inv_norm_q: f32,
    query_f16: &'a [u16],
}

/// Batch SIMD scoring with precomputed query norm + f16 query.
#[inline]
#[allow(clippy::too_many_arguments)]
fn score_batch_precomp(
    pq: &PrecompQuery<'_>,
    raw: &[u8],
    quant: crate::dsl::DenseVectorQuantization,
    dim: usize,
    scores: &mut [f32],
    unit_norm: bool,
) -> crate::error::Result<()> {
    let query = pq.query;
    let inv_norm_q = pq.inv_norm_q;
    let query_f16 = pq.query_f16;
    use crate::dsl::DenseVectorQuantization;
    use crate::structures::simd;
    let element_size = quant.element_size();
    let required_bytes = scores
        .len()
        .checked_mul(dim)
        .and_then(|elements| elements.checked_mul(element_size))
        .ok_or_else(|| {
            crate::Error::Corruption("dense reranker batch size overflow".to_string())
        })?;
    if raw.len() < required_bytes {
        return Err(crate::Error::Corruption(format!(
            "dense reranker batch is truncated: need {required_bytes} bytes, got {}",
            raw.len()
        )));
    }
    if matches!(
        quant,
        DenseVectorQuantization::F32 | DenseVectorQuantization::F16
    ) && required_bytes > 0
        && !(raw.as_ptr() as usize).is_multiple_of(element_size)
    {
        return Err(crate::Error::Corruption(format!(
            "dense reranker {:?} data is not {}-byte aligned",
            quant, element_size
        )));
    }
    match (quant, unit_norm) {
        (DenseVectorQuantization::F32, false) => {
            let num_floats = scores.len() * dim;
            // Safety: Vec<u8> from the global allocator is guaranteed to be at least
            // 8-byte aligned on 64-bit platforms (aligned to max_align_t). Assert at
            // runtime to guard against custom allocators with weaker guarantees.
            let vectors: &[f32] =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
            simd::batch_cosine_scores_precomp(query, vectors, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::F32, true) => {
            let num_floats = scores.len() * dim;
            let vectors: &[f32] =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
            simd::batch_dot_scores_precomp(query, vectors, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::F16, false) => {
            simd::batch_cosine_scores_f16_precomp(query_f16, raw, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::F16, true) => {
            simd::batch_dot_scores_f16_precomp(query_f16, raw, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::UInt8, false) => {
            simd::batch_cosine_scores_u8_precomp(query, raw, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::UInt8, true) => {
            simd::batch_dot_scores_u8_precomp(query, raw, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::Binary, _) => {
            return Err(crate::Error::InvalidFieldType {
                expected: "non-binary dense vector".to_string(),
                got: "binary dense vector".to_string(),
            });
        }
    }
    Ok(())
}

/// Configuration for L2 dense/binary vector reranking
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Vector field (dense or binary dense)
    pub field: Field,
    /// Query vector (f32, for dense fields)
    pub vector: Vec<f32>,
    /// Query vector (packed bits, for binary dense fields).
    /// When non-empty, Hamming distance scoring is used instead of cosine.
    pub binary_vector: Vec<u8>,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
    /// Whether stored vectors are pre-normalized to unit L2 norm.
    /// When true, scoring uses dot-product only (skips per-vector norm — ~40% faster).
    /// Ignored for binary fields.
    pub unit_norm: bool,
    /// Matryoshka pre-filter: number of leading dimensions to use for cheap
    /// approximate scoring before full-dimension exact reranking.
    /// Ignored for binary fields.
    pub matryoshka_dims: Option<usize>,
    /// Reciprocal Rank Fusion k parameter. When > 0, fuses L1 (first-stage) and
    /// L2 (reranker) rankings: `score(d) = 1/(k + rank_L1) + 1/(k + rank_L2)`.
    /// Typical value: 60. When 0, RRF is disabled and only L2 scores are used.
    pub rrf_k: f32,
}

/// Score a single document against the query vector (used by tests).
#[cfg(test)]
use crate::structures::simd::cosine_similarity;
#[cfg(test)]
fn score_document(
    doc: &crate::dsl::Document,
    config: &RerankerConfig,
) -> Option<(f32, Vec<ScoredPosition>)> {
    let query_dim = config.vector.len();
    let mut values: Vec<(u32, f32)> = doc
        .get_all(config.field)
        .filter_map(|fv| fv.as_dense_vector())
        .enumerate()
        .filter_map(|(ordinal, vec)| {
            if vec.len() != query_dim {
                return None;
            }
            let score = cosine_similarity(&config.vector, vec);
            Some((ordinal as u32, score))
        })
        .collect();

    if values.is_empty() {
        return None;
    }

    let combined = config.combiner.combine(&values);

    // Sort ordinals by score descending (best chunk first)
    values.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let positions: Vec<ScoredPosition> = values
        .into_iter()
        .map(|(ordinal, score)| ScoredPosition::new(ordinal, score))
        .collect();

    Some((combined, positions))
}

/// Apply Reciprocal Rank Fusion to combine L1 and L2 rankings.
///
/// `candidates` is sorted by L1 score descending (first-stage query output).
/// `scored` is sorted by L2 score descending (reranker output).
/// Replaces each result's score with the RRF fused score, re-sorts, and truncates.
///
/// Formula (Cormack, Clarke, Buettcher 2009):
///   `RRF(d) = 1/(k + rank_L1(d)) + 1/(k + rank_L2(d))`
/// where ranks are 1-based.
fn apply_rrf(
    candidates: &[SearchResult],
    scored: &mut Vec<SearchResult>,
    k: f32,
    final_limit: usize,
) {
    // Build L1 rank map: (segment_id, doc_id) → 1-based rank
    let l1_ranks: FxHashMap<(u128, u32), usize> = candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| ((c.segment_id, c.doc_id), idx + 1))
        .collect();

    // scored is sorted by L2 score desc → enumerate index + 1 = L2 rank
    for (l2_idx, result) in scored.iter_mut().enumerate() {
        let l1_rank = l1_ranks
            .get(&(result.segment_id, result.doc_id))
            .copied()
            .unwrap_or(candidates.len() + 1);
        result.score = super::fusion::rrf_contribution(k, l1_rank)
            + super::fusion::rrf_contribution(k, l2_idx + 1);
    }

    scored.sort_unstable_by(compare_search_results_desc);
    scored.truncate(final_limit);
}

/// Rerank L1 candidates by exact dense vector distance.
///
/// Groups candidates by segment for batched I/O, sorts flat indexes for
/// sequential mmap access, and scores vectors in bounded SIMD batches.
/// Scratch memory remains independent of the candidate count.
///
/// When `unit_norm` is set in the config, scoring uses dot-product only
/// (skips per-vector norm computation — ~40% less work).
pub async fn rerank<D: crate::directories::Directory + 'static>(
    searcher: &crate::index::Searcher<D>,
    candidates: &[SearchResult],
    config: &RerankerConfig,
    final_limit: usize,
) -> crate::error::Result<Vec<SearchResult>> {
    // Validate before empty-result early returns so malformed requests do not
    // succeed or fail depending on whether the first-stage query found hits.
    let kind = validate_reranker_config(searcher, config)?;
    if final_limit == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Dispatch: binary vector → Hamming, f32 vector → cosine/dot.
    if kind == RerankerKind::Binary {
        return rerank_binary(searcher, candidates, config, final_limit).await;
    }

    let t0 = std::time::Instant::now();
    let field_id = config.field.0;
    let query = &config.vector;
    let query_dim = query.len();
    let segments = searcher.segment_readers();
    let seg_by_id = searcher.segment_map();

    // Precompute query inverse-norm and f16 query once (reused across all segments)
    use crate::structures::simd;
    let norm_q_sq = simd::dot_product_f32(query, query, query_dim);
    let inv_norm_q = if norm_q_sq < f32::EPSILON {
        0.0
    } else {
        simd::fast_inv_sqrt(norm_q_sq)
    };
    let query_f16: Vec<u16> = query.iter().map(|&v| simd::f32_to_f16(v)).collect();
    let pq = PrecompQuery {
        query,
        inv_norm_q,
        query_f16: &query_f16,
    };

    // ── Phase 1: Group candidates by segment ──────────────────────────────
    let mut segment_groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut skipped = 0u32;

    for (ci, candidate) in candidates.iter().enumerate() {
        if let Some(&si) = seg_by_id.get(&candidate.segment_id) {
            segment_groups.entry(si).or_default().push(ci);
        } else {
            skipped += 1;
        }
    }

    // ── Phase 2: Per-segment batched resolve + read + score (concurrent) ──
    // Bound the fan-out so many immutable segments cannot each retain a raw
    // scoring buffer and candidate scratch at the same time.
    let query_ref = pq.query;
    let inv_norm_q_val = pq.inv_norm_q;
    let query_f16_ref = pq.query_f16;
    let vector_budget = Arc::new(AtomicUsize::new(0));
    let byte_budget = Arc::new(AtomicUsize::new(0));

    let segment_futs = futures::stream::iter(segment_groups.into_iter().map(
        |(si, candidate_indices)| {
            #[allow(clippy::redundant_locals)]
            let segments = &segments;
            #[allow(clippy::redundant_locals)]
            let candidates = candidates;
            #[allow(clippy::redundant_locals)]
            let query_ref = query_ref;
            #[allow(clippy::redundant_locals)]
            let query_f16_ref = query_f16_ref;
            #[allow(clippy::redundant_locals)]
            let config = config;
            let vector_budget = Arc::clone(&vector_budget);
            let byte_budget = Arc::clone(&byte_budget);
            async move {
                let mut scores: Vec<(usize, u32, f32)> = Vec::new();
                let mut vectors = 0usize;
                let mut seg_skipped = 0u32;

                let Some(lazy_flat) = segments[si].flat_vectors().get(&field_id) else {
                    return Ok::<_, crate::error::Error>((
                        scores,
                        vectors,
                        candidate_indices.len() as u32,
                    ));
                };
                if lazy_flat.dim != query_dim {
                    return Err(crate::Error::Corruption(format!(
                        "dense reranker field {field_id} stores dimension {}, expected {query_dim}",
                        lazy_flat.dim
                    )));
                }
                if lazy_flat.quantization == crate::dsl::DenseVectorQuantization::Binary {
                    return Err(crate::Error::Corruption(format!(
                        "dense reranker field {field_id} unexpectedly uses binary storage"
                    )));
                }

                let vbs = lazy_flat.vector_byte_size();
                let quant = lazy_flat.quantization;

                // Resolve flat indexes for all candidates in this segment
                let mut resolved: Vec<(usize, usize, u32)> = Vec::new();
                for &ci in &candidate_indices {
                    let local_doc_id = candidates[ci].doc_id;
                    let (start, count) = lazy_flat.flat_indexes_for_doc_range(local_doc_id);
                    if count == 0 {
                        seg_skipped += 1;
                        continue;
                    }
                    reserve_rerank_vectors(&vector_budget, &byte_budget, count, vbs)?;
                    for j in 0..count {
                        let (_, ordinal) = lazy_flat.get_doc_id(start + j);
                        resolved.push((ci, start + j, ordinal as u32));
                    }
                }

                if resolved.is_empty() {
                    return Ok((scores, vectors, seg_skipped));
                }

                let n = resolved.len();
                vectors = n;

                // Sort by flat_idx for sequential mmap access. Prefetch only
                // each bounded scoring chunk below; advising the entire
                // candidate set at once can flood the page cache.
                resolved.sort_unstable_by_key(|&(_, flat_idx, _)| flat_idx);

                let batch_len = rerank_batch_len(vbs);
                let max_batch = batch_len.min(n);
                let max_raw_len = max_batch.checked_mul(vbs).ok_or_else(|| {
                    crate::Error::Query("dense reranker buffer size overflow".into())
                })?;
                let mut raw_buf = vec![0u8; max_raw_len];

                // Reconstruct PrecompQuery from captured components
                let pq = PrecompQuery {
                    query: query_ref,
                    inv_norm_q: inv_norm_q_val,
                    query_f16: query_f16_ref,
                };

                // Matryoshka pre-filter
                if let Some(mdims) = config.matryoshka_dims
                    && mdims < query_dim
                    && n > crate::query::max_candidate_limit(final_limit)
                {
                    let trunc_dim = mdims;
                    let trunc_pq = PrecompQuery {
                        query: &query_ref[..trunc_dim],
                        inv_norm_q: {
                            let nq = simd::dot_product_f32(
                                &query_ref[..trunc_dim],
                                &query_ref[..trunc_dim],
                                trunc_dim,
                            );
                            if nq < f32::EPSILON {
                                0.0
                            } else {
                                simd::fast_inv_sqrt(nq)
                            }
                        },
                        query_f16: &query_f16_ref[..trunc_dim],
                    };
                    let trunc_vbs = trunc_dim * quant.element_size();
                    let mut scores_buf = vec![0.0f32; n];
                    for (chunk_idx, chunk) in resolved.chunks(batch_len).enumerate() {
                        // Pack only the Matryoshka prefix for this batch. The
                        // previous full-vector reads pulled every unused tail
                        // into the page cache and then reread surviving vectors.
                        let raw_len = chunk.len().checked_mul(trunc_vbs).ok_or_else(|| {
                            crate::Error::Query("dense reranker buffer size overflow".into())
                        })?;
                        let raw = &mut raw_buf[..raw_len];
                        for (buf_idx, &(_, flat_idx, _)) in chunk.iter().enumerate() {
                            lazy_flat
                                .read_vector_prefix_raw_into(
                                    flat_idx,
                                    trunc_vbs,
                                    &mut raw[buf_idx * trunc_vbs..(buf_idx + 1) * trunc_vbs],
                                )
                                .await
                                .map_err(crate::error::Error::Io)?;
                        }
                        let score_base = chunk_idx * batch_len;
                        searcher.install_search_cpu(|| {
                            score_batch_precomp(
                                &trunc_pq,
                                raw,
                                quant,
                                trunc_dim,
                                &mut scores_buf[score_base..score_base + chunk.len()],
                                config.unit_norm,
                            )
                        })?;
                    }

                    // Rank approximate *documents*, using every stored value
                    // and the configured combiner. Selecting individual
                    // vectors here can discard the other values needed by
                    // Avg/Sum/LSE and can choose the wrong Max document.
                    let mut approximate_ordinals: FxHashMap<usize, Vec<(u32, f32)>> =
                        FxHashMap::default();
                    for (resolved_index, &(ci, _, ordinal)) in resolved.iter().enumerate() {
                        approximate_ordinals
                            .entry(ci)
                            .or_default()
                            .push((ordinal, scores_buf[resolved_index]));
                    }
                    let mut ranked: Vec<(usize, f32)> = approximate_ordinals
                        .into_iter()
                        .map(|(ci, ordinals)| (ci, config.combiner.combine(&ordinals)))
                        .collect();
                    searcher.install_search_cpu(|| {
                        ranked.sort_unstable_by(|a, b| {
                            b.1.total_cmp(&a.1)
                                .then_with(|| candidates[a.0].doc_id.cmp(&candidates[b.0].doc_id))
                        });
                    });
                    let approximate_docs = ranked.len();
                    let survivor_doc_limit =
                        crate::query::max_candidate_limit(final_limit).min(approximate_docs);
                    let survivor_docs: FxHashSet<usize> = ranked
                        .into_iter()
                        .take(survivor_doc_limit)
                        .map(|(ci, _)| ci)
                        .collect();
                    // Full-score every value belonging to each surviving doc;
                    // the final combiner must never see a truncated value set.
                    let mut survivor_entries: Vec<_> = resolved
                        .iter()
                        .copied()
                        .filter(|(ci, _, _)| survivor_docs.contains(ci))
                        .collect();
                    survivor_entries.sort_unstable_by_key(|&(_, flat_idx, _)| flat_idx);
                    let mut full_scores = vec![0.0f32; max_batch.min(survivor_entries.len())];
                    scores.reserve(survivor_entries.len());
                    for chunk in survivor_entries.chunks(batch_len) {
                        #[cfg(feature = "native")]
                        lazy_flat.prefetch_vectors(
                            chunk.iter().map(|&(_, flat_idx, _)| flat_idx),
                        );
                        let raw_len = chunk.len().checked_mul(vbs).ok_or_else(|| {
                            crate::Error::Query("dense reranker buffer size overflow".into())
                        })?;
                        let raw = &mut raw_buf[..raw_len];
                        for (buf_idx, &(_, flat_idx, _)) in chunk.iter().enumerate() {
                            lazy_flat
                                .read_vector_raw_into(
                                    flat_idx,
                                    &mut raw[buf_idx * vbs..(buf_idx + 1) * vbs],
                                )
                                .await
                                .map_err(crate::error::Error::Io)?;
                        }
                        searcher.install_search_cpu(|| {
                            score_batch_precomp(
                                &pq,
                                raw,
                                quant,
                                query_dim,
                                &mut full_scores[..chunk.len()],
                                config.unit_norm,
                            )
                        })?;
                        for (buf_idx, &(ci, _, ordinal)) in chunk.iter().enumerate() {
                            scores.push((ci, ordinal, full_scores[buf_idx]));
                        }
                    }

                    let survivor_vectors = survivor_entries.len();
                    log::debug!(
                        "[reranker] matryoshka pre-filter: {}/{} dims, {}/{} docs and {}/{} vectors survived",
                        trunc_dim,
                        query_dim,
                        survivor_docs.len(),
                        approximate_docs,
                        survivor_vectors,
                        n,
                    );
                } else {
                    let mut scores_buf = vec![0.0f32; max_batch];
                    scores.reserve(n);
                    for chunk in resolved.chunks(batch_len) {
                        #[cfg(feature = "native")]
                        lazy_flat.prefetch_vectors(
                            chunk.iter().map(|&(_, flat_idx, _)| flat_idx),
                        );
                        let raw_len = chunk.len().checked_mul(vbs).ok_or_else(|| {
                            crate::Error::Query("dense reranker buffer size overflow".into())
                        })?;
                        let raw = &mut raw_buf[..raw_len];
                        for (buf_idx, &(_, flat_idx, _)) in chunk.iter().enumerate() {
                            lazy_flat
                                .read_vector_raw_into(
                                    flat_idx,
                                    &mut raw[buf_idx * vbs..(buf_idx + 1) * vbs],
                                )
                                .await
                                .map_err(crate::error::Error::Io)?;
                        }
                        searcher.install_search_cpu(|| {
                            score_batch_precomp(
                                &pq,
                                raw,
                                quant,
                                query_dim,
                                &mut scores_buf[..chunk.len()],
                                config.unit_norm,
                            )
                        })?;
                        for (buf_idx, &(ci, _, ordinal)) in chunk.iter().enumerate() {
                            scores.push((ci, ordinal, scores_buf[buf_idx]));
                        }
                    }
                }

                Ok((scores, vectors, seg_skipped))
            }
        },
    ))
    .buffer_unordered(MAX_CONCURRENT_RERANK_SEGMENTS);
    futures::pin_mut!(segment_futs);

    let mut all_scores: Vec<(usize, u32, f32)> = Vec::new();
    let mut total_vectors = 0usize;
    while let Some((scores, vectors, seg_skipped)) = segment_futs.try_next().await? {
        all_scores.extend(scores);
        total_vectors = total_vectors.saturating_add(vectors);
        skipped = skipped.saturating_add(seg_skipped);
    }

    let read_score_elapsed = t0.elapsed();

    if total_vectors == 0 {
        log::debug!(
            "[reranker] field {}: {} candidates, all skipped (no flat vectors)",
            field_id,
            candidates.len()
        );
        return Ok(Vec::new());
    }

    // ── Phase 3: Combine scores and build results ─────────────────────────
    // Sort flat buffer by candidate_idx so contiguous runs belong to the same doc
    all_scores.sort_unstable_by_key(|&(ci, _, _)| ci);

    let mut scored: Vec<SearchResult> = Vec::with_capacity(
        candidates
            .len()
            .min(crate::query::max_candidate_limit(final_limit)),
    );
    let mut ordinal_pairs: Vec<(u32, f32)> = Vec::new();
    let mut i = 0;
    while i < all_scores.len() {
        let ci = all_scores[i].0;
        let run_start = i;
        while i < all_scores.len() && all_scores[i].0 == ci {
            i += 1;
        }
        let run = &mut all_scores[run_start..i];

        // Build (ordinal, score) slice for combiner (reuses hoisted buffer)
        ordinal_pairs.clear();
        ordinal_pairs.extend(run.iter().map(|&(_, ord, s)| (ord, s)));
        let combined = config.combiner.combine(&ordinal_pairs);

        // Sort positions by score descending (best chunk first)
        run.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));
        let positions: Vec<ScoredPosition> = run
            .iter()
            .map(|&(_, ord, score)| ScoredPosition::new(ord, score))
            .collect();

        scored.push(SearchResult {
            doc_id: candidates[ci].doc_id,
            score: combined,
            segment_id: candidates[ci].segment_id,
            positions: vec![(field_id, positions)],
        });
    }

    scored.sort_unstable_by(compare_search_results_desc);

    if config.rrf_k > 0.0 {
        apply_rrf(candidates, &mut scored, config.rrf_k, final_limit);
    } else {
        scored.truncate(final_limit);
    }

    log::debug!(
        "[reranker] field {}: {} candidates -> {} results (skipped {}, {} vectors, unit_norm={}, rrf_k={}): read+score={:.1}ms total={:.1}ms",
        field_id,
        candidates.len(),
        scored.len(),
        skipped,
        total_vectors,
        config.unit_norm,
        config.rrf_k,
        read_score_elapsed.as_secs_f64() * 1000.0,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(scored)
}

/// Rerank L1 candidates by exact Hamming distance on stored binary vectors.
async fn rerank_binary<D: crate::directories::Directory + 'static>(
    searcher: &crate::index::Searcher<D>,
    candidates: &[SearchResult],
    config: &RerankerConfig,
    final_limit: usize,
) -> crate::error::Result<Vec<SearchResult>> {
    if config.binary_vector.is_empty() || candidates.is_empty() {
        return Ok(Vec::new());
    }

    let t0 = std::time::Instant::now();
    let field_id = config.field.0;
    let query = &config.binary_vector;
    let byte_len = query.len();
    let segments = searcher.segment_readers();
    let seg_by_id = searcher.segment_map();

    // Group candidates by segment
    let mut segment_groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (ci, cand) in candidates.iter().enumerate() {
        if let Some(&seg_idx) = seg_by_id.get(&cand.segment_id) {
            let reader = &segments[seg_idx];
            if reader.flat_vectors().contains_key(&field_id) {
                segment_groups.entry(seg_idx).or_default().push(ci);
            }
        }
    }

    // Bounded concurrent per-segment scoring (same pattern as dense reranker).
    let vector_budget = Arc::new(AtomicUsize::new(0));
    let byte_budget = Arc::new(AtomicUsize::new(0));
    let segment_futs = futures::stream::iter(segment_groups.into_iter().map(
        |(seg_idx, cand_indices)| {
            #[allow(clippy::redundant_locals)]
            let segments = &segments;
            #[allow(clippy::redundant_locals)]
            let candidates = candidates;
            let vector_budget = Arc::clone(&vector_budget);
            let byte_budget = Arc::clone(&byte_budget);
            async move {
                let mut scores: Vec<(usize, u32, f32)> = Vec::new();

                let Some(lazy_flat) = segments[seg_idx].flat_vectors().get(&field_id) else {
                    return Ok::<_, crate::error::Error>(scores);
                };
                if lazy_flat.quantization != crate::dsl::DenseVectorQuantization::Binary
                    || !lazy_flat.dim.is_multiple_of(8)
                {
                    return Err(crate::Error::Corruption(format!(
                        "binary reranker field {field_id} has invalid flat-vector metadata"
                    )));
                }
                let vbs = lazy_flat.vector_byte_size();
                if vbs != byte_len {
                    return Err(crate::Error::Corruption(format!(
                        "binary reranker field {field_id} stores {vbs} bytes/vector, expected {byte_len}"
                    )));
                }

                // Resolve flat indexes
                let mut resolved: Vec<(usize, usize)> = Vec::new();
                for &ci in &cand_indices {
                    let doc_id = candidates[ci].doc_id;
                    let (start, count) = lazy_flat.flat_indexes_for_doc_range(doc_id);
                    reserve_rerank_vectors(&vector_budget, &byte_budget, count, vbs)?;
                    for j in 0..count {
                        resolved.push((ci, start + j));
                    }
                }
                if resolved.is_empty() {
                    return Ok(scores);
                }

                resolved.sort_unstable_by_key(|&(_, flat_idx)| flat_idx);

                let n = resolved.len();
                let batch_len = rerank_batch_len(vbs);
                let max_batch = batch_len.min(n);
                let max_raw_len = max_batch.checked_mul(vbs).ok_or_else(|| {
                    crate::Error::Query("binary reranker buffer size overflow".into())
                })?;
                let mut raw_buf = vec![0u8; max_raw_len];
                let mut scores_buf = vec![0f32; max_batch];
                scores.reserve(n);

                for chunk in resolved.chunks(batch_len) {
                    let raw_len = chunk.len().checked_mul(vbs).ok_or_else(|| {
                        crate::Error::Query("binary reranker buffer size overflow".into())
                    })?;
                    let raw = &mut raw_buf[..raw_len];
                    for (buf_idx, &(_, flat_idx)) in chunk.iter().enumerate() {
                        lazy_flat
                            .read_vector_raw_into(
                                flat_idx,
                                &mut raw[buf_idx * vbs..(buf_idx + 1) * vbs],
                            )
                            .await
                            .map_err(crate::error::Error::Io)?;
                    }
                    searcher.install_search_cpu(|| {
                        crate::structures::simd::batch_hamming_scores(
                            query,
                            raw,
                            byte_len,
                            lazy_flat.dim,
                            &mut scores_buf[..chunk.len()],
                        );
                    });

                    for (buf_idx, &(ci, flat_idx)) in chunk.iter().enumerate() {
                        let (_, ordinal) = lazy_flat.get_doc_id(flat_idx);
                        scores.push((ci, ordinal as u32, scores_buf[buf_idx]));
                    }
                }

                Ok(scores)
            }
        },
    ))
    .buffer_unordered(MAX_CONCURRENT_RERANK_SEGMENTS);
    futures::pin_mut!(segment_futs);

    // Combine ordinal scores per candidate and apply combiner
    let mut cand_ordinal_scores: FxHashMap<usize, Vec<(u32, f32)>> = FxHashMap::default();
    while let Some(scores) = segment_futs.try_next().await? {
        for (ci, ordinal, score) in scores {
            cand_ordinal_scores
                .entry(ci)
                .or_default()
                .push((ordinal, score));
        }
    }

    let total_vectors = cand_ordinal_scores.len();
    let mut scored: Vec<SearchResult> = Vec::with_capacity(total_vectors);
    for (ci, ordinal_scores) in cand_ordinal_scores {
        let combined = config.combiner.combine(&ordinal_scores);
        let positions: Vec<ScoredPosition> = ordinal_scores
            .iter()
            .map(|&(ord, s)| ScoredPosition::new(ord, s))
            .collect();
        scored.push(SearchResult {
            doc_id: candidates[ci].doc_id,
            score: combined,
            segment_id: candidates[ci].segment_id,
            positions: vec![(field_id, positions)],
        });
    }

    scored.sort_unstable_by(compare_search_results_desc);

    if config.rrf_k > 0.0 {
        apply_rrf(candidates, &mut scored, config.rrf_k, final_limit);
    } else {
        scored.truncate(final_limit);
    }

    log::debug!(
        "[reranker-binary] field {}: {} candidates -> {} results ({} docs scored, {} bytes/vec, rrf_k={}): {:.1}ms",
        field_id,
        candidates.len(),
        scored.len(),
        total_vectors,
        byte_len,
        config.rrf_k,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::{Document, Field};

    fn make_config(vector: Vec<f32>, combiner: MultiValueCombiner) -> RerankerConfig {
        RerankerConfig {
            field: Field(0),
            vector,
            binary_vector: Vec::new(),
            combiner,
            unit_norm: false,
            matryoshka_dims: None,
            rrf_k: 0.0,
        }
    }

    #[test]
    fn rerank_batches_are_bounded_by_bytes() {
        assert_eq!(rerank_batch_len(1), RERANK_SCORE_BATCH);
        assert_eq!(
            rerank_batch_len(MAX_RERANK_RAW_BATCH_BYTES),
            1,
            "one very wide vector must still make progress"
        );
        assert!(
            rerank_batch_len(4_096) * 4_096 <= MAX_RERANK_RAW_BATCH_BYTES,
            "normal batches must stay within the raw scratch budget"
        );
    }

    #[test]
    fn rerank_budget_bounds_count_and_bytes() {
        let vectors = AtomicUsize::new(0);
        let bytes = AtomicUsize::new(0);
        reserve_rerank_vectors(&vectors, &bytes, 2, 32).unwrap();
        assert_eq!(vectors.load(AtomicOrdering::Relaxed), 2);
        assert_eq!(bytes.load(AtomicOrdering::Relaxed), 64);

        let vectors = AtomicUsize::new(0);
        let bytes = AtomicUsize::new(0);
        assert!(reserve_rerank_vectors(&vectors, &bytes, 2, MAX_L2_RERANK_VECTOR_BYTES).is_err());
    }

    #[test]
    fn test_score_document_single_value() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]);

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let (score, positions) = score_document(&doc, &config).unwrap();
        // cosine([1,0,0], [1,0,0]) = 1.0
        assert!((score - 1.0).abs() < 1e-6);
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].position, 0); // ordinal 0
    }

    #[test]
    fn test_score_document_orthogonal() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![0.0, 1.0, 0.0]);

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let (score, _) = score_document(&doc, &config).unwrap();
        // cosine([1,0,0], [0,1,0]) = 0.0
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_score_document_multi_value_max() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]); // cos=1.0 (same direction)
        doc.add_dense_vector(Field(0), vec![0.0, 1.0, 0.0]); // cos=0.0 (orthogonal)

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        let (score, positions) = score_document(&doc, &config).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
        // Best chunk first
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].position, 0); // ordinal 0 scored highest
        assert!((positions[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_multi_value_avg() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]); // cos=1.0
        doc.add_dense_vector(Field(0), vec![0.0, 1.0, 0.0]); // cos=0.0

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Avg);
        let (score, _) = score_document(&doc, &config).unwrap();
        // avg(1.0, 0.0) = 0.5
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_document_missing_field() {
        let mut doc = Document::new();
        // Add to field 1, not field 0
        doc.add_dense_vector(Field(1), vec![1.0, 0.0, 0.0]);

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        assert!(score_document(&doc, &config).is_none());
    }

    #[test]
    fn test_score_document_wrong_field_type() {
        let mut doc = Document::new();
        doc.add_text(Field(0), "not a vector");

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max);
        assert!(score_document(&doc, &config).is_none());
    }

    #[test]
    fn test_score_document_dimension_mismatch() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0]); // 2D

        let config = make_config(vec![1.0, 0.0, 0.0], MultiValueCombiner::Max); // 3D query
        assert!(score_document(&doc, &config).is_none());
    }

    #[test]
    fn test_score_document_empty_query_vector() {
        let mut doc = Document::new();
        doc.add_dense_vector(Field(0), vec![1.0, 0.0, 0.0]);

        let config = make_config(vec![], MultiValueCombiner::Max);
        // Empty query can't match any stored vector (dimension mismatch)
        assert!(score_document(&doc, &config).is_none());
    }

    fn make_result(doc_id: u32, score: f32, segment_id: u128) -> SearchResult {
        SearchResult {
            doc_id,
            score,
            segment_id,
            positions: Vec::new(),
        }
    }

    #[test]
    fn test_rrf_basic_fusion() {
        // L1 ranking: doc A(rank 1), B(rank 2), C(rank 3)
        let candidates = vec![
            make_result(1, 10.0, 1), // A: L1 rank 1
            make_result(2, 8.0, 1),  // B: L1 rank 2
            make_result(3, 5.0, 1),  // C: L1 rank 3
        ];

        // L2 ranking (reversed): C(rank 1), B(rank 2), A(rank 3)
        let mut scored = vec![
            make_result(3, 0.9, 1), // C: L2 rank 1
            make_result(2, 0.7, 1), // B: L2 rank 2
            make_result(1, 0.3, 1), // A: L2 rank 3
        ];

        let k = 60.0;
        apply_rrf(&candidates, &mut scored, k, 10);

        // B should win: rank 2 in both → 2/(k+2) vs split ranks for A and C
        // A: 1/(61) + 1/(63) = 0.01639 + 0.01587 = 0.03226
        // B: 1/(62) + 1/(62) = 0.01613 + 0.01613 = 0.03226
        // C: 1/(63) + 1/(61) = 0.01587 + 0.01639 = 0.03226
        // All equal! (symmetric: rank sum = 4 for each)
        // Actually: A: 1/61 + 1/63, B: 1/62 + 1/62, C: 1/63 + 1/61
        // A = C by symmetry, B is slightly different
        // 1/61 + 1/63 = (63+61)/(61*63) = 124/3843 = 0.032267
        // 1/62 + 1/62 = 2/62 = 1/31 = 0.032258
        // So A = C > B (very slightly). Top result should be doc 3 (C) or doc 1 (A)
        // since they have the same RRF score but C appeared first in scored.

        assert_eq!(scored.len(), 3);
        // All three should have very similar RRF scores
        let spread = scored[0].score - scored[2].score;
        assert!(
            spread < 0.001,
            "All docs have near-equal RRF scores, spread={spread}"
        );
    }

    #[test]
    fn test_rrf_clear_winner() {
        // Doc X is rank 1 in both L1 and L2 → should clearly win
        let candidates = vec![
            make_result(1, 10.0, 1), // X: L1 rank 1
            make_result(2, 8.0, 1),  // Y: L1 rank 2
            make_result(3, 5.0, 1),  // Z: L1 rank 3
        ];

        // L2 ranking: X still rank 1
        let mut scored = vec![
            make_result(1, 0.95, 1), // X: L2 rank 1
            make_result(3, 0.50, 1), // Z: L2 rank 2
            make_result(2, 0.30, 1), // Y: L2 rank 3
        ];

        let k = 60.0;
        apply_rrf(&candidates, &mut scored, k, 10);

        // X: 1/(61) + 1/(61) = 2/61 = 0.03279 (best)
        // Y: 1/(62) + 1/(63) = 0.03200 (worst)
        // Z: 1/(63) + 1/(62) = 0.03200 (same as Y by symmetry)
        assert_eq!(scored[0].doc_id, 1, "Doc 1 (rank 1 in both) should be top");
        assert!(scored[0].score > scored[1].score);
    }

    #[test]
    fn test_rrf_truncation() {
        let candidates = vec![
            make_result(1, 10.0, 1),
            make_result(2, 8.0, 1),
            make_result(3, 5.0, 1),
            make_result(4, 3.0, 1),
            make_result(5, 1.0, 1),
        ];

        let mut scored = vec![
            make_result(5, 0.9, 1),
            make_result(4, 0.8, 1),
            make_result(3, 0.7, 1),
            make_result(2, 0.6, 1),
            make_result(1, 0.5, 1),
        ];

        apply_rrf(&candidates, &mut scored, 60.0, 3);
        assert_eq!(scored.len(), 3, "Should truncate to final_limit=3");
    }

    #[test]
    fn test_rrf_missing_l1_candidate() {
        // L1 has docs 1, 2. L2 scored doc 3 which wasn't in L1 candidates.
        let candidates = vec![make_result(1, 10.0, 1), make_result(2, 8.0, 1)];

        let mut scored = vec![
            make_result(3, 0.9, 1), // not in L1 → gets worst L1 rank
            make_result(1, 0.5, 1),
        ];

        apply_rrf(&candidates, &mut scored, 60.0, 10);

        // Doc 1: L1 rank 1 → 1/61, L2 rank 2 → 1/62  = 0.03252
        // Doc 3: L1 rank 3 (fallback) → 1/63, L2 rank 1 → 1/61 = 0.03226
        // Doc 1 should win because it has a better L1 rank
        assert_eq!(scored[0].doc_id, 1);
    }

    #[test]
    fn test_rrf_small_k() {
        // With small k, rank differences matter more
        let candidates = vec![make_result(1, 10.0, 1), make_result(2, 8.0, 1)];

        let mut scored = vec![
            make_result(2, 0.9, 1), // L2 rank 1
            make_result(1, 0.5, 1), // L2 rank 2
        ];

        apply_rrf(&candidates, &mut scored, 1.0, 10);

        // k=1: Doc 1: 1/(1+1) + 1/(1+2) = 0.5 + 0.333 = 0.833
        //       Doc 2: 1/(1+2) + 1/(1+1) = 0.333 + 0.5 = 0.833
        // With k=1 and symmetric ranks, scores are equal
        let diff = (scored[0].score - scored[1].score).abs();
        assert!(
            diff < 1e-6,
            "Symmetric ranks should produce equal RRF scores"
        );
    }
}
