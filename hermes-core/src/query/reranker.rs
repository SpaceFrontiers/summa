//! L2 reranker: rerank L1 candidates by exact dense vector distance on stored vectors
//!
//! Optimized for throughput:
//! - Candidates grouped by segment for batched I/O
//! - Flat indexes sorted for sequential mmap access (OS readahead)
//! - Single SIMD batch-score call per segment (not per candidate)
//! - Reusable buffers across segments (no per-candidate heap allocation)
//! - unit_norm fast path: skip per-vector norm when vectors are pre-normalized

use rustc_hash::FxHashMap;

use crate::dsl::Field;

use super::{MultiValueCombiner, ScoredPosition, SearchResult};

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
) {
    let query = pq.query;
    let inv_norm_q = pq.inv_norm_q;
    let query_f16 = pq.query_f16;
    use crate::dsl::DenseVectorQuantization;
    use crate::structures::simd;
    match (quant, unit_norm) {
        (DenseVectorQuantization::F32, false) => {
            let num_floats = scores.len() * dim;
            // Safety: Vec<u8> from the global allocator is guaranteed to be at least
            // 8-byte aligned on 64-bit platforms (aligned to max_align_t). Assert at
            // runtime to guard against custom allocators with weaker guarantees.
            assert!(
                (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
                "f32 vector data not 4-byte aligned"
            );
            let vectors: &[f32] =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
            simd::batch_cosine_scores_precomp(query, vectors, dim, scores, inv_norm_q);
        }
        (DenseVectorQuantization::F32, true) => {
            let num_floats = scores.len() * dim;
            assert!(
                (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()),
                "f32 vector data not 4-byte aligned"
            );
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
            unreachable!("Binary quantization should not reach score_batch_precomp");
        }
    }
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
            .unwrap_or(candidates.len() + 1) as f32;
        let l2_rank = (l2_idx + 1) as f32;
        result.score = 1.0 / (k + l1_rank) + 1.0 / (k + l2_rank);
    }

    scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    scored.truncate(final_limit);
}

/// Rerank L1 candidates by exact dense vector distance.
///
/// Groups candidates by segment for batched I/O, sorts flat indexes for
/// sequential mmap access, and scores all vectors in a single SIMD batch
/// per segment. Reuses buffers across segments to avoid per-candidate
/// heap allocation.
///
/// When `unit_norm` is set in the config, scoring uses dot-product only
/// (skips per-vector norm computation — ~40% less work).
pub async fn rerank<D: crate::directories::Directory + 'static>(
    searcher: &crate::index::Searcher<D>,
    candidates: &[SearchResult],
    config: &RerankerConfig,
    final_limit: usize,
) -> crate::error::Result<Vec<SearchResult>> {
    // Dispatch: binary vector → Hamming, f32 vector → cosine/dot
    if !config.binary_vector.is_empty() {
        return rerank_binary(searcher, candidates, config, final_limit).await;
    }

    if config.vector.is_empty() || candidates.is_empty() {
        return Ok(Vec::new());
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
    // Each segment runs independently: resolve flat indexes, read vectors,
    // and score — all overlapping I/O across segments via join_all.
    let query_ref = pq.query;
    let inv_norm_q_val = pq.inv_norm_q;
    let query_f16_ref = pq.query_f16;

    let segment_futs: Vec<_> = segment_groups
        .into_iter()
        .map(|(si, candidate_indices)| {
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
                    return Ok((scores, vectors, candidate_indices.len() as u32));
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

                // Sort by flat_idx for sequential mmap access
                resolved.sort_unstable_by_key(|&(_, flat_idx, _)| flat_idx);

                let first_idx = resolved[0].1;
                let last_idx = resolved[n - 1].1;
                let span = last_idx - first_idx + 1;

                let mut raw_buf: Vec<u8> = vec![0u8; n * vbs];

                if span <= n * 4 {
                    let range_bytes = lazy_flat
                        .read_vectors_batch(first_idx, span)
                        .await
                        .map_err(crate::error::Error::Io)?;
                    let rb = range_bytes.as_slice();
                    for (buf_idx, &(_, flat_idx, _)) in resolved.iter().enumerate() {
                        let rel = flat_idx - first_idx;
                        let src = &rb[rel * vbs..(rel + 1) * vbs];
                        raw_buf[buf_idx * vbs..(buf_idx + 1) * vbs].copy_from_slice(src);
                    }
                } else {
                    for (buf_idx, &(_, flat_idx, _)) in resolved.iter().enumerate() {
                        lazy_flat
                            .read_vector_raw_into(
                                flat_idx,
                                &mut raw_buf[buf_idx * vbs..(buf_idx + 1) * vbs],
                            )
                            .await
                            .map_err(crate::error::Error::Io)?;
                    }
                }

                // Reconstruct PrecompQuery from captured components
                let pq = PrecompQuery {
                    query: query_ref,
                    inv_norm_q: inv_norm_q_val,
                    query_f16: query_f16_ref,
                };

                let mut scores_buf: Vec<f32> = vec![0.0; n];

                // Matryoshka pre-filter
                if let Some(mdims) = config.matryoshka_dims
                    && mdims < query_dim
                    && n > final_limit * 2
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
                    for i in 0..n {
                        let vec_start = i * vbs;
                        score_batch_precomp(
                            &trunc_pq,
                            &raw_buf[vec_start..vec_start + trunc_vbs],
                            quant,
                            trunc_dim,
                            &mut scores_buf[i..i + 1],
                            config.unit_norm,
                        );
                    }

                    let per_doc_cap: usize = match &config.combiner {
                        super::MultiValueCombiner::Max => 1,
                        super::MultiValueCombiner::WeightedTopK { k, .. } => *k,
                        _ => usize::MAX,
                    };

                    let mut ranked: Vec<(usize, f32)> =
                        (0..n).map(|i| (i, scores_buf[i])).collect();
                    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

                    let mut survivors: Vec<(usize, f32)> =
                        Vec::with_capacity(n.min(final_limit * 4));
                    let mut doc_vector_counts: FxHashMap<usize, usize> = FxHashMap::default();
                    let mut unique_docs = 0usize;

                    for &(orig_idx, score) in &ranked {
                        let ci = resolved[orig_idx].0;
                        let count = doc_vector_counts.entry(ci).or_insert(0);

                        if *count >= per_doc_cap {
                            continue;
                        }
                        if *count == 0 {
                            unique_docs += 1;
                        }
                        *count += 1;
                        survivors.push((orig_idx, score));

                        if unique_docs >= final_limit && survivors.len() >= final_limit * 2 {
                            break;
                        }
                    }

                    scores.reserve(survivors.len());
                    for &(orig_idx, _) in &survivors {
                        let vec_start = orig_idx * vbs;
                        let mut score = 0.0f32;
                        score_batch_precomp(
                            &pq,
                            &raw_buf[vec_start..vec_start + vbs],
                            quant,
                            query_dim,
                            std::slice::from_mut(&mut score),
                            config.unit_norm,
                        );
                        let (ci, _, ordinal) = resolved[orig_idx];
                        scores.push((ci, ordinal, score));
                    }

                    let filtered = n - survivors.len();
                    log::debug!(
                        "[reranker] matryoshka pre-filter: {}/{} dims, {}/{} vectors survived from {} unique docs (filtered {}, per_doc_cap={})",
                        trunc_dim,
                        query_dim,
                        survivors.len(),
                        n,
                        unique_docs,
                        filtered,
                        per_doc_cap
                    );
                } else {
                    score_batch_precomp(
                        &pq,
                        &raw_buf[..n * vbs],
                        quant,
                        query_dim,
                        &mut scores_buf[..n],
                        config.unit_norm,
                    );

                    scores.reserve(n);
                    for (buf_idx, &(ci, _, ordinal)) in resolved.iter().enumerate() {
                        scores.push((ci, ordinal, scores_buf[buf_idx]));
                    }
                }

                Ok((scores, vectors, seg_skipped))
            }
        })
        .collect();

    let results = futures::future::join_all(segment_futs).await;

    let mut all_scores: Vec<(usize, u32, f32)> = Vec::new();
    let mut total_vectors = 0usize;
    for result in results {
        let (scores, vectors, seg_skipped) = result?;
        all_scores.extend(scores);
        total_vectors += vectors;
        skipped += seg_skipped;
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

    let mut scored: Vec<SearchResult> = Vec::with_capacity(candidates.len().min(final_limit * 2));
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

    scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

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

    // Concurrent per-segment scoring (same pattern as dense reranker)
    let segment_futs: Vec<_> = segment_groups
        .into_iter()
        .map(|(seg_idx, cand_indices)| {
            #[allow(clippy::redundant_locals)]
            let segments = &segments;
            #[allow(clippy::redundant_locals)]
            let candidates = candidates;
            async move {
                let mut scores: Vec<(usize, u32, f32)> = Vec::new();

                let Some(lazy_flat) = segments[seg_idx].flat_vectors().get(&field_id) else {
                    return Ok::<_, crate::error::Error>(scores);
                };
                let vbs = lazy_flat.vector_byte_size();
                if vbs != byte_len {
                    return Ok(scores);
                }

                // Resolve flat indexes
                let mut resolved: Vec<(usize, usize)> = Vec::new();
                for &ci in &cand_indices {
                    let doc_id = candidates[ci].doc_id;
                    let (start, count) = lazy_flat.flat_indexes_for_doc_range(doc_id);
                    for j in 0..count {
                        resolved.push((ci, start + j));
                    }
                }
                if resolved.is_empty() {
                    return Ok(scores);
                }

                resolved.sort_unstable_by_key(|&(_, flat_idx)| flat_idx);

                let n = resolved.len();
                let first_idx = resolved[0].1;
                let last_idx = resolved[n - 1].1;
                let span = last_idx - first_idx + 1;

                let mut raw_buf = vec![0u8; n * vbs];

                if span <= n * 4 {
                    let range_bytes = lazy_flat
                        .read_vectors_batch(first_idx, span)
                        .await
                        .map_err(crate::error::Error::Io)?;
                    let rb = range_bytes.as_slice();
                    for (buf_idx, &(_, flat_idx)) in resolved.iter().enumerate() {
                        let rel = flat_idx - first_idx;
                        raw_buf[buf_idx * vbs..(buf_idx + 1) * vbs]
                            .copy_from_slice(&rb[rel * vbs..(rel + 1) * vbs]);
                    }
                } else {
                    for (buf_idx, &(_, flat_idx)) in resolved.iter().enumerate() {
                        lazy_flat
                            .read_vector_raw_into(
                                flat_idx,
                                &mut raw_buf[buf_idx * vbs..(buf_idx + 1) * vbs],
                            )
                            .await
                            .map_err(crate::error::Error::Io)?;
                    }
                }

                // Batch Hamming scoring
                let dim_bits = lazy_flat.dim;
                let mut scores_buf = vec![0f32; n];
                crate::structures::simd::batch_hamming_scores(
                    query,
                    &raw_buf,
                    byte_len,
                    dim_bits,
                    &mut scores_buf,
                );

                for (buf_idx, &(ci, flat_idx)) in resolved.iter().enumerate() {
                    let (_, ordinal) = lazy_flat.get_doc_id(flat_idx);
                    scores.push((ci, ordinal as u32, scores_buf[buf_idx]));
                }

                Ok(scores)
            }
        })
        .collect();

    let results = futures::future::join_all(segment_futs).await;

    // Combine ordinal scores per candidate and apply combiner
    let mut cand_ordinal_scores: FxHashMap<usize, Vec<(u32, f32)>> = FxHashMap::default();
    for result in results {
        for (ci, ordinal, score) in result? {
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

    scored.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

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
