//! Async segment reader with lazy loading

pub(crate) mod bmp;
pub(crate) mod loader;
mod types;

pub use bmp::BmpIndex;
#[cfg(feature = "native")]
pub(crate) use types::DimRawData;
pub use types::{SparseIndex, VectorIndex, VectorSearchResult};

/// Bound vocabulary and posting expansion before a prefix query starts loading
/// posting payloads. These are per-segment limits; callers should use exact-term
/// or a more selective prefix when they are exceeded.
const MAX_PREFIX_TERMS: usize = 1_024;
const MAX_PREFIX_POSTINGS: u64 = 5_000_000;
/// Bound ANN result/resolution vectors per segment. Exact vector bytes are
/// scored in smaller batches below, so peak rerank scratch stays predictable.
const MAX_DENSE_CANDIDATES_PER_SEGMENT: usize = 200_000;
const MAX_ANN_ORDINAL_OVERFETCH: usize = 32;
/// Preferred vector count; wide vectors reduce it to stay under the byte cap.
const DENSE_SCORE_BATCH: usize = 4_096;
const BINARY_SCORE_BATCH: usize = 8_192;
const MAX_VECTOR_SCORE_BATCH_BYTES: usize = 8 * 1024 * 1024;

/// Memory statistics for a single segment
#[derive(Debug, Clone, Default)]
pub struct SegmentMemoryStats {
    /// Segment ID
    pub segment_id: u128,
    /// Number of documents in segment
    pub num_docs: u32,
    /// Term dictionary block cache bytes
    pub term_dict_cache_bytes: usize,
    /// Document store block cache bytes
    pub store_cache_bytes: usize,
    /// Sparse vector index bytes (in-memory posting lists)
    pub sparse_index_bytes: usize,
    /// Dense vector index bytes (cluster assignments, quantized codes)
    pub dense_index_bytes: usize,
    /// Bloom filter bytes
    pub bloom_filter_bytes: usize,
    /// Hot metadata bytes actually pinned (mlock/heap-copy) at open
    pub pinned_metadata_bytes: u64,
    /// Hot metadata bytes eligible for pinning (gap vs pinned = budget
    /// exhausted or mlock failures — operator-visible)
    pub pin_intended_bytes: u64,
}

impl SegmentMemoryStats {
    /// Total estimated memory for this segment
    pub fn total_bytes(&self) -> usize {
        self.term_dict_cache_bytes
            + self.store_cache_bytes
            + self.sparse_index_bytes
            + self.dense_index_bytes
            + self.bloom_filter_bytes
    }
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::vector_data::LazyFlatVectorData;
use crate::directories::{Directory, FileHandle};
use crate::dsl::{DenseVectorQuantization, Document, Field, Schema};
use crate::query::{MAX_DENSE_NPROBE, MAX_DENSE_RERANK_FACTOR};
use crate::structures::{
    AsyncSSTableReader, BlockPostingList, CoarseCentroids, IVFPQIndex, IVFRaBitQIndex, PQCodebook,
    RaBitQIndex, SSTableStats, TermInfo,
};
use crate::{DocId, Error, Result};

use super::store::{AsyncStoreReader, RawStoreBlock};
use super::types::{SegmentFiles, SegmentId, SegmentMeta};

/// Combine per-ordinal (doc_id, ordinal, score) triples into VectorSearchResults,
/// applying the multi-value combiner, sorting by score desc, and truncating to `limit`.
///
/// Fast path: when all ordinals are 0 (single-valued field), skips the HashMap
/// grouping entirely and just sorts + truncates the raw results.
pub(crate) fn combine_ordinal_results(
    raw: impl IntoIterator<Item = (u32, u16, f32)>,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Vec<VectorSearchResult> {
    let collected: Vec<(u32, u16, f32)> = raw.into_iter().collect();

    let num_raw = collected.len();
    if log::log_enabled!(log::Level::Debug) {
        let mut ids: Vec<u32> = collected.iter().map(|(d, _, _)| *d).collect();
        ids.sort_unstable();
        ids.dedup();
        log::debug!(
            "combine_ordinal_results: {} raw entries, {} unique docs, combiner={:?}, limit={}",
            num_raw,
            ids.len(),
            combiner,
            limit
        );
    }

    // Fast path: all ordinals are 0 → no grouping needed, skip HashMap
    let all_single = collected.iter().all(|&(_, ord, _)| ord == 0);
    if all_single {
        let mut results: Vec<VectorSearchResult> = collected
            .into_iter()
            .map(|(doc_id, _, score)| VectorSearchResult::new(doc_id, score, vec![(0, score)]))
            .collect();
        results.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results.truncate(limit);
        return results;
    }

    // Slow path: multi-valued field — group by doc_id, apply combiner
    let mut doc_ordinals: rustc_hash::FxHashMap<DocId, Vec<(u32, f32)>> =
        rustc_hash::FxHashMap::default();
    for (doc_id, ordinal, score) in collected {
        doc_ordinals
            .entry(doc_id as DocId)
            .or_default()
            .push((ordinal as u32, score));
    }
    let mut results: Vec<VectorSearchResult> = doc_ordinals
        .into_iter()
        .map(|(doc_id, ordinals)| {
            let combined_score = combiner.combine(&ordinals);
            VectorSearchResult::new(doc_id, combined_score, ordinals)
        })
        .collect();
    results.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    results.truncate(limit);
    results
}

/// Heap entry used by exact flat-vector search after all values belonging to
/// one document have been combined. Keeping the heap at document granularity
/// prevents several strong values from one document from crowding other
/// documents out of the raw vector top-k.
struct HeapVectorResult(VectorSearchResult);

impl PartialEq for HeapVectorResult {
    fn eq(&self, other: &Self) -> bool {
        self.0.score.to_bits() == other.0.score.to_bits() && self.0.doc_id == other.0.doc_id
    }
}

impl Eq for HeapVectorResult {}

impl Ord for HeapVectorResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap top is the worst retained document: lower score, then
        // larger doc ID for deterministic equal-score eviction.
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.doc_id.cmp(&other.0.doc_id))
    }
}

impl PartialOrd for HeapVectorResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Incrementally combine a flat vector stream sorted by `(doc_id, ordinal)`
/// and retain only the best `limit` documents. Scratch is O(values in the
/// current document + retained output), independent of the segment size.
struct FlatDocumentCollector {
    heap: BinaryHeap<HeapVectorResult>,
    limit: usize,
    combiner: crate::query::MultiValueCombiner,
    current_doc: Option<DocId>,
    current_ordinals: Vec<(u32, f32)>,
}

impl FlatDocumentCollector {
    fn new(limit: usize, combiner: crate::query::MultiValueCombiner) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(limit.min(8 * 1024)),
            limit,
            combiner,
            current_doc: None,
            current_ordinals: Vec::new(),
        }
    }

    fn push(&mut self, doc_id: DocId, ordinal: u16, score: f32) {
        if self.current_doc.is_some_and(|current| current != doc_id) {
            self.finish_current();
        }
        self.current_doc = Some(doc_id);
        self.current_ordinals.push((ordinal as u32, score));
    }

    fn finish_current(&mut self) {
        let Some(doc_id) = self.current_doc.take() else {
            return;
        };
        let score = self.combiner.combine(&self.current_ordinals);
        let should_retain = self.heap.len() < self.limit
            || self.heap.peek().is_some_and(|worst| {
                HeapVectorResult(VectorSearchResult::new(doc_id, score, Vec::new()))
                    .cmp(worst)
                    .is_lt()
            });

        if !should_retain {
            // The overwhelmingly common path once the heap is full. Reuse
            // the ordinal scratch instead of allocating a fresh Vec for
            // every rejected document in a flat scan.
            self.current_ordinals.clear();
            return;
        }

        let ordinals = std::mem::take(&mut self.current_ordinals);
        let entry = HeapVectorResult(VectorSearchResult::new(doc_id, score, ordinals));
        if self.heap.len() < self.limit {
            self.heap.push(entry);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            // Recycle the evicted result's allocation as the next document's
            // scratch. PeekMut restores heap order when it is dropped.
            let mut evicted = std::mem::replace(&mut worst.0, entry.0);
            evicted.ordinals.clear();
            self.current_ordinals = evicted.ordinals;
        }
    }

    fn into_results(mut self) -> Vec<VectorSearchResult> {
        self.finish_current();
        let mut results: Vec<_> = self.heap.into_iter().map(|entry| entry.0).collect();
        results.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results
    }
}

/// Collect a stream already grouped by document (the layout produced by flat
/// storage expansion) without rebuilding a hash table for every candidate.
fn combine_grouped_ordinal_results(
    raw: impl IntoIterator<Item = RawVectorCandidate>,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Vec<VectorSearchResult> {
    let mut collector = FlatDocumentCollector::new(limit, combiner);
    for (doc_id, ordinal, score) in raw {
        collector.push(doc_id, ordinal, score);
    }
    collector.into_results()
}

#[derive(Clone, Copy)]
struct DenseSearchParams {
    dim: usize,
    nprobe: usize,
    unit_norm: bool,
}

/// Compute the ANN candidate count without relying on saturating float casts.
fn checked_dense_fetch_k(k: usize, rerank_factor: f32) -> Result<usize> {
    if !rerank_factor.is_finite() || !(1.0..=MAX_DENSE_RERANK_FACTOR).contains(&rerank_factor) {
        return Err(Error::Query(format!(
            "dense rerank_factor must be finite and in [1, {MAX_DENSE_RERANK_FACTOR}], got {rerank_factor}"
        )));
    }

    let fetch = (k as f64) * (rerank_factor as f64);
    if !fetch.is_finite()
        || fetch > usize::MAX as f64
        || fetch > MAX_DENSE_CANDIDATES_PER_SEGMENT as f64
    {
        return Err(Error::Query(format!(
            "dense candidate count exceeds the per-segment maximum of \
             {MAX_DENSE_CANDIDATES_PER_SEGMENT}: k={k}, rerank_factor={rerank_factor}"
        )));
    }
    Ok(fetch.ceil() as usize)
}

fn ann_ordinal_fetch_k(fetch_k: usize, num_vectors: usize, num_docs: usize) -> usize {
    if num_vectors == 0 || num_docs == 0 {
        return 0;
    }
    let average_values_per_doc = num_vectors
        .div_ceil(num_docs)
        .clamp(1, MAX_ANN_ORDINAL_OVERFETCH);
    fetch_k
        .saturating_mul(average_values_per_doc)
        .min(MAX_DENSE_CANDIDATES_PER_SEGMENT)
        .min(num_vectors)
}

/// Search vector-level ANN progressively until it yields enough distinct
/// documents, or until the probed candidate population / safety cap is
/// exhausted. A fixed average-values overfetch is insufficient for skewed
/// fields where one document owns most of the best vectors.
fn progressive_ann_search<F>(
    target_docs: usize,
    initial_fetch: usize,
    max_vectors: usize,
    mut search: F,
) -> Result<Vec<RawVectorCandidate>>
where
    F: FnMut(usize) -> Vec<RawVectorCandidate>,
{
    let max_fetch = max_vectors.min(MAX_DENSE_CANDIDATES_PER_SEGMENT);
    if target_docs == 0 || max_fetch == 0 {
        return Ok(Vec::new());
    }

    let target_docs = target_docs.min(max_fetch);
    let mut fetch = initial_fetch.max(target_docs).min(max_fetch);
    loop {
        let results = search(fetch);
        let mut docs = rustc_hash::FxHashSet::with_capacity_and_hasher(
            target_docs.min(results.len()),
            Default::default(),
        );
        for &(doc_id, _, _) in &results {
            docs.insert(doc_id);
            if docs.len() >= target_docs {
                return Ok(results);
            }
        }
        if results.len() < fetch || fetch == max_vectors {
            return Ok(results);
        }
        if fetch == max_fetch {
            return Err(Error::Query(format!(
                "ANN search reached the per-segment candidate limit of \
                 {MAX_DENSE_CANDIDATES_PER_SEGMENT} with only {} of {target_docs} requested \
                 documents; reduce k or the number of vector values per document",
                docs.len()
            )));
        }

        let next_fetch = fetch.saturating_mul(2).min(max_fetch);
        if next_fetch == fetch {
            return Ok(results);
        }
        fetch = next_fetch;
    }
}

/// Single-scan variant of [`progressive_ann_search`] for indexes whose
/// `search(k)` cost does not depend on `k`.
///
/// Binary IVF scans every vector of every probed cluster regardless of the
/// collector size, so each doubling retry of [`progressive_ann_search`]
/// repeated the identical full scan (~nprobe/num_clusters of the segment's
/// codes) just to grow the collector. Here the closure runs ONCE at the
/// fetch cap and the doubling schedule is replayed over prefixes of the
/// single ranked list, returning exactly what the retry loop would have
/// returned (a `search(f)` that returns the top-`f` of a fixed ranking makes
/// the two observationally identical, including the candidate-limit error).
fn single_scan_ann_search<F>(
    target_docs: usize,
    initial_fetch: usize,
    max_vectors: usize,
    search: F,
) -> Result<Vec<RawVectorCandidate>>
where
    F: FnOnce(usize) -> Vec<RawVectorCandidate>,
{
    let max_fetch = max_vectors.min(MAX_DENSE_CANDIDATES_PER_SEGMENT);
    if target_docs == 0 || max_fetch == 0 {
        return Ok(Vec::new());
    }
    let target_docs = target_docs.min(max_fetch);

    let mut results = search(max_fetch);
    results.truncate(max_fetch);

    // Ranked-list length at which `target_docs` distinct documents are first
    // covered (None if the whole list falls short).
    let mut docs = rustc_hash::FxHashSet::with_capacity_and_hasher(target_docs, Default::default());
    let mut satisfied_len = None;
    for (i, &(doc_id, _, _)) in results.iter().enumerate() {
        docs.insert(doc_id);
        if docs.len() >= target_docs {
            satisfied_len = Some(i + 1);
            break;
        }
    }

    let mut fetch = initial_fetch.max(target_docs).min(max_fetch);
    loop {
        let round_len = fetch.min(results.len());
        if satisfied_len.is_some_and(|len| len <= round_len) {
            results.truncate(round_len);
            return Ok(results);
        }
        if results.len() < fetch || fetch == max_vectors {
            return Ok(results);
        }
        if fetch == max_fetch {
            return Err(Error::Query(format!(
                "ANN search reached the per-segment candidate limit of \
                 {MAX_DENSE_CANDIDATES_PER_SEGMENT} with only {} of {target_docs} requested \
                 documents; reduce k or the number of vector values per document",
                docs.len()
            )));
        }
        let next_fetch = fetch.saturating_mul(2).min(max_fetch);
        if next_fetch == fetch {
            return Ok(results);
        }
        fetch = next_fetch;
    }
}

#[inline]
fn bounded_vector_score_batch(vector_byte_size: usize, preferred: usize) -> usize {
    preferred.min((MAX_VECTOR_SCORE_BATCH_BYTES / vector_byte_size.max(1)).max(1))
}

fn checked_file_range(
    offset: u64,
    length: u64,
    file_length: u64,
    description: &str,
) -> Result<std::ops::Range<u64>> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::Corruption(format!("{description} byte range overflows u64")))?;
    if end > file_length {
        return Err(Error::Corruption(format!(
            "{description} byte range {offset}..{end} exceeds file length {file_length}"
        )));
    }
    Ok(offset..end)
}

type RawVectorCandidate = (u32, u16, f32);
type ResolvedVectorCandidate = (usize, usize); // (result index, flat-vector index)

/// ANN operates on individual vectors, but document combiners require every
/// stored value for a candidate document. Expand the deduplicated ANN document
/// union before exact reranking, with a hard per-segment vector budget.
fn expand_ann_candidate_documents(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
) -> Result<(Vec<RawVectorCandidate>, Vec<ResolvedVectorCandidate>)> {
    let mut candidate_docs: Vec<DocId> = ann_results.iter().map(|candidate| candidate.0).collect();
    candidate_docs.sort_unstable();
    candidate_docs.dedup();

    let mut expanded = Vec::new();
    let mut resolved = Vec::new();
    for doc_id in candidate_docs {
        let (start, count) = flat.flat_indexes_for_doc_range(doc_id);
        if count == 0 {
            return Err(Error::Corruption(format!(
                "ANN candidate document {doc_id} is missing from flat vector storage"
            )));
        }
        let next_len = expanded
            .len()
            .checked_add(count)
            .ok_or_else(|| Error::Query("ANN candidate vector expansion overflow".to_string()))?;
        if next_len > MAX_DENSE_CANDIDATES_PER_SEGMENT {
            return Err(Error::Query(format!(
                "ANN candidate documents expand to more than \
                 {MAX_DENSE_CANDIDATES_PER_SEGMENT} vectors in one segment"
            )));
        }
        expanded.reserve(count);
        resolved.reserve(count);
        let end = start
            .checked_add(count)
            .ok_or_else(|| Error::Corruption("flat vector range overflow".to_string()))?;
        for flat_index in start..end {
            let (stored_doc_id, ordinal) = flat.get_doc_id(flat_index);
            if stored_doc_id != doc_id {
                return Err(Error::Corruption(format!(
                    "flat vector doc map is not contiguous for document {doc_id}"
                )));
            }
            let result_index = expanded.len();
            expanded.push((doc_id, ordinal, 0.0));
            resolved.push((result_index, flat_index));
        }
    }
    Ok((expanded, resolved))
}

/// Reuse the probe's scores for (doc, ordinal) pairs it already returned.
///
/// Binary IVF clusters store the same packed codes as flat storage and score
/// them with the same exact SIMD Hamming kernel, so re-reading those vectors
/// from flat storage recomputes an identical score at the cost of one random
/// read (and potential page fault on cold mmap) per candidate. Only the
/// candidate documents' remaining ordinals still need flat reads. The
/// per-query map is justified by removing one flat read per reused entry.
fn fill_probe_scores_and_prune_resolved(
    ann_results: &[RawVectorCandidate],
    expanded: &mut [RawVectorCandidate],
    resolved: Vec<ResolvedVectorCandidate>,
) -> Vec<ResolvedVectorCandidate> {
    let mut probe_scores: FxHashMap<(DocId, u16), f32> =
        FxHashMap::with_capacity_and_hasher(ann_results.len(), Default::default());
    for &(doc_id, ordinal, score) in ann_results {
        probe_scores.insert((doc_id, ordinal), score);
    }
    resolved
        .into_iter()
        .filter(|&(result_index, _)| {
            let (doc_id, ordinal, _) = expanded[result_index];
            match probe_scores.get(&(doc_id, ordinal)) {
                Some(&score) => {
                    expanded[result_index].2 = score;
                    false
                }
                None => true,
            }
        })
        .collect()
}

async fn exact_score_binary_candidate_documents(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
) -> Result<Vec<RawVectorCandidate>> {
    let (mut expanded, resolved) = expand_ann_candidate_documents(ann_results, flat)?;
    let resolved = fill_probe_scores_and_prune_resolved(ann_results, &mut expanded, resolved);
    let vector_byte_size = flat.vector_byte_size();
    let batch_len = bounded_vector_score_batch(vector_byte_size, BINARY_SCORE_BATCH);
    let raw_capacity = batch_len
        .checked_mul(vector_byte_size)
        .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
    let mut raw = vec![0u8; raw_capacity];
    let mut scores = vec![0.0f32; batch_len];

    for chunk in resolved.chunks(batch_len) {
        #[cfg(feature = "native")]
        flat.prefetch_vectors(chunk.iter().map(|&(_, flat_index)| flat_index));
        let raw_len = chunk
            .len()
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
        let raw = &mut raw[..raw_len];
        for (buffer_index, &(_, flat_index)) in chunk.iter().enumerate() {
            flat.read_vector_raw_into(
                flat_index,
                &mut raw[buffer_index * vector_byte_size..(buffer_index + 1) * vector_byte_size],
            )
            .await
            .map_err(Error::Io)?;
        }
        crate::structures::simd::batch_hamming_scores(
            query,
            raw,
            vector_byte_size,
            dim_bits,
            &mut scores[..chunk.len()],
        );
        for (buffer_index, &(result_index, _)) in chunk.iter().enumerate() {
            expanded[result_index].2 = scores[buffer_index];
        }
    }
    Ok(expanded)
}

#[cfg(feature = "sync")]
fn exact_score_binary_candidate_documents_sync(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
) -> Result<Vec<RawVectorCandidate>> {
    let (mut expanded, resolved) = expand_ann_candidate_documents(ann_results, flat)?;
    let resolved = fill_probe_scores_and_prune_resolved(ann_results, &mut expanded, resolved);
    let vector_byte_size = flat.vector_byte_size();
    let batch_len = bounded_vector_score_batch(vector_byte_size, BINARY_SCORE_BATCH);
    let raw_capacity = batch_len
        .checked_mul(vector_byte_size)
        .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
    let mut raw = vec![0u8; raw_capacity];
    let mut scores = vec![0.0f32; batch_len];

    for chunk in resolved.chunks(batch_len) {
        let raw_len = chunk
            .len()
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
        let raw = &mut raw[..raw_len];
        for (buffer_index, &(_, flat_index)) in chunk.iter().enumerate() {
            flat.read_vector_raw_into_sync(
                flat_index,
                &mut raw[buffer_index * vector_byte_size..(buffer_index + 1) * vector_byte_size],
            )
            .map_err(Error::Io)?;
        }
        crate::structures::simd::batch_hamming_scores(
            query,
            raw,
            vector_byte_size,
            dim_bits,
            &mut scores[..chunk.len()],
        );
        for (buffer_index, &(result_index, _)) in chunk.iter().enumerate() {
            expanded[result_index].2 = scores[buffer_index];
        }
    }
    Ok(expanded)
}

fn validate_coarse_centroids(centroids: &CoarseCentroids, dim: usize) -> Result<()> {
    let expected = (centroids.num_clusters as usize)
        .checked_mul(dim)
        .ok_or_else(|| Error::Corruption("coarse centroid size overflow".into()))?;
    if centroids.num_clusters == 0
        || centroids.dim != dim
        || centroids.centroids.len() != expected
        || centroids.centroids.iter().any(|value| !value.is_finite())
    {
        return Err(Error::Corruption(format!(
            "invalid coarse centroids: clusters={}, dim={}, values={} (expected dim={dim}, values={expected})",
            centroids.num_clusters,
            centroids.dim,
            centroids.centroids.len()
        )));
    }
    Ok(())
}

/// Async segment reader with lazy loading
///
/// - Term dictionary: only index loaded, blocks loaded on-demand
/// - Postings: loaded on-demand per term via HTTP range requests
/// - Document store: only index loaded, blocks loaded on-demand via HTTP range requests
pub struct SegmentReader {
    meta: SegmentMeta,
    /// Term dictionary with lazy block loading
    term_dict: Arc<AsyncSSTableReader<TermInfo>>,
    /// Postings file handle - fetches ranges on demand
    postings_handle: FileHandle,
    /// Document store with lazy block loading
    store: Arc<AsyncStoreReader>,
    schema: Arc<Schema>,
    /// Dense vector indexes per field (RaBitQ or IVF-RaBitQ) — for search
    vector_indexes: FxHashMap<u32, VectorIndex>,
    /// Lazy flat vectors per field — for reranking and merge (doc_ids in memory, vectors via mmap)
    flat_vectors: FxHashMap<u32, LazyFlatVectorData>,
    /// Per-field coarse centroids for IVF/ScaNN search
    coarse_centroids: FxHashMap<u32, Arc<CoarseCentroids>>,
    /// Sparse vector indexes per field (MaxScore format)
    sparse_indexes: FxHashMap<u32, SparseIndex>,
    /// BMP sparse vector indexes per field (BMP format)
    bmp_indexes: FxHashMap<u32, BmpIndex>,
    /// Position file handle for phrase queries (lazy loading)
    positions_handle: Option<FileHandle>,
    /// Fast-field columnar readers per field_id
    fast_fields: FxHashMap<u32, crate::structures::fast_field::FastFieldReader>,
    /// Per-segment MaxScore threshold (f32 stored as AtomicU32 bits).
    /// Allows per-field MaxScore groups within a single query to share thresholds:
    /// field A's result seeds field B's pruning on the same segment.
    /// Hot-metadata pin accounting (see `segment::pin`)
    #[cfg(feature = "native")]
    pin_report: crate::segment::pin::PinReport,
}

impl SegmentReader {
    /// Open a segment with lazy loading
    pub async fn open<D: Directory>(
        dir: &D,
        segment_id: SegmentId,
        schema: Arc<Schema>,
        cache_blocks: usize,
    ) -> Result<Self> {
        Self::open_with_cache_blocks(dir, segment_id, schema, cache_blocks, cache_blocks).await
    }

    /// Open a segment with independent term-dictionary and document-store caches.
    ///
    /// [`Self::open`] keeps the historical single-capacity API for standalone
    /// callers. Native indexes use this method so `IndexConfig::store_cache_blocks`
    /// is not silently replaced by the (usually much larger) term cache capacity.
    pub async fn open_with_cache_blocks<D: Directory>(
        dir: &D,
        segment_id: SegmentId,
        schema: Arc<Schema>,
        term_cache_blocks: usize,
        store_cache_blocks: usize,
    ) -> Result<Self> {
        let files = SegmentFiles::new(segment_id.0);

        // Read metadata (small, always loaded)
        let meta_slice = dir.open_read(&files.meta).await?;
        let meta_bytes = meta_slice.read_bytes().await?;
        let meta = SegmentMeta::deserialize(meta_bytes.as_slice())?;
        debug_assert_eq!(meta.id, segment_id.0);

        // Open term dictionary with lazy loading (fetches ranges on demand)
        let term_dict_handle = dir.open_lazy(&files.term_dict).await?;
        let term_dict = AsyncSSTableReader::open(term_dict_handle, term_cache_blocks).await?;

        // Get postings file handle (lazy - fetches ranges on demand)
        let postings_handle = dir.open_lazy(&files.postings).await?;

        // Open store with lazy loading
        let store_handle = dir.open_lazy(&files.store).await?;
        let store = AsyncStoreReader::open(store_handle, store_cache_blocks).await?;

        // Load dense vector indexes from unified .vectors file
        let vectors_data = loader::load_vectors_file(dir, &files, &schema, meta.num_docs).await?;
        let vector_indexes = vectors_data.indexes;
        let flat_vectors = vectors_data.flat_vectors;

        // Fields served by an ANN index only touch flat vectors for scattered
        // rerank reads — disable readahead for them once at open. Flat-only
        // fields keep default advice: brute-force scans them sequentially.
        // Advice is sticky on the mapping, so per-query re-advising is wasted.
        #[cfg(feature = "native")]
        for (field_id, lazy_flat) in &flat_vectors {
            if vector_indexes.contains_key(field_id) {
                lazy_flat.advise_random_access();
            }
        }

        // Load sparse vector indexes from .sparse file (MaxScore + BMP)
        let sparse_data = loader::load_sparse_file(dir, &files, meta.num_docs, &schema).await?;
        let sparse_indexes = sparse_data.maxscore_indexes;
        let bmp_indexes = sparse_data.bmp_indexes;

        // Open positions file handle (if exists) - offsets are now in TermInfo
        let positions_handle = loader::open_positions_file(dir, &files, &schema).await?;

        // Load fast-field columns from .fast file
        let fast_fields = loader::load_fast_fields_file(dir, &files, &schema).await?;

        // Log segment loading stats
        {
            let mut parts = vec![format!(
                "[segment] loaded {:016x}: docs={}",
                segment_id.0, meta.num_docs
            )];
            if !vector_indexes.is_empty() || !flat_vectors.is_empty() {
                parts.push(format!(
                    "dense: {} ann + {} flat fields",
                    vector_indexes.len(),
                    flat_vectors.len()
                ));
            }
            for (field_id, idx) in &sparse_indexes {
                parts.push(format!(
                    "sparse field {}: {} dims, ~{:.1} KB",
                    field_id,
                    idx.num_dimensions(),
                    idx.num_dimensions() as f64 * 24.0 / 1024.0
                ));
            }
            for (field_id, idx) in &bmp_indexes {
                parts.push(format!(
                    "bmp field {}: {} dims, {} blocks",
                    field_id,
                    idx.dims(),
                    idx.num_blocks
                ));
            }
            if !fast_fields.is_empty() {
                parts.push(format!("fast: {} fields", fast_fields.len()));
            }
            log::debug!("{}", parts.join(", "));
        }

        #[allow(unused_mut)]
        let mut reader = Self {
            meta,
            term_dict: Arc::new(term_dict),
            postings_handle,
            store: Arc::new(store),
            schema,
            vector_indexes,
            flat_vectors,
            coarse_centroids: FxHashMap::default(),
            sparse_indexes,
            bmp_indexes,
            positions_handle,
            fast_fields,
            #[cfg(feature = "native")]
            pin_report: Default::default(),
        };

        // Pin hot metadata per the process-wide policy (no-op when disabled)
        #[cfg(feature = "native")]
        reader.apply_pin_policy(&crate::segment::pin::pin_policy().to_owned());

        Ok(reader)
    }

    /// Pin per-query-mandatory metadata sections in priority order until the
    /// budget is exhausted (see `segment::pin` and docs/hot-metadata-pinning.md).
    ///
    /// Priority: BMP block-offset tables → sparse skip sections → doc-id maps
    /// → BMP superblock grids. Bulk data (4-bit grid, block data, raw vectors)
    /// is never pinned. Fail-loud: budget exhaustion and mlock failures are
    /// logged and visible via `SegmentMemoryStats::{pin_intended_bytes,
    /// pinned_metadata_bytes}`.
    #[cfg(feature = "native")]
    pub(crate) fn apply_pin_policy(&mut self, policy: &crate::segment::pin::PinPolicy) {
        use crate::segment::pin::PinReport;

        if !policy.is_enabled() {
            return;
        }
        let mut remaining = policy.budget_bytes;
        let mut report = PinReport::default();

        // Priority 1: BMP block-offset tables
        for bmp in self.bmp_indexes.values_mut() {
            bmp.pin_block_starts(policy.mode, &mut remaining, &mut report);
        }
        // Priority 2: sparse skip sections
        for sparse in self.sparse_indexes.values_mut() {
            sparse.pin_skip_section(policy.mode, &mut remaining, &mut report);
        }
        // Priority 3: doc-id maps
        for flat in self.flat_vectors.values_mut() {
            flat.pin_doc_ids(policy.mode, &mut remaining, &mut report);
        }
        for bmp in self.bmp_indexes.values_mut() {
            bmp.pin_doc_maps(policy.mode, &mut remaining, &mut report);
        }
        // Priority 4: BMP superblock grids
        for bmp in self.bmp_indexes.values_mut() {
            bmp.pin_sb_grid(policy.mode, &mut remaining, &mut report);
        }

        if report.skipped_budget_bytes > 0 || report.failed_bytes > 0 {
            log::warn!(
                "[pin] segment {:016x}: pinned {}/{} bytes (budget skipped {}, mlock failed {}) —                  raise HERMES_PIN_METADATA_BUDGET_MB or RLIMIT_MEMLOCK for full coverage",
                self.meta.id,
                report.pinned_bytes,
                report.intended_bytes,
                report.skipped_budget_bytes,
                report.failed_bytes,
            );
        } else if report.pinned_bytes > 0 {
            log::info!(
                "[pin] segment {:016x}: pinned {} bytes of hot metadata ({:?})",
                self.meta.id,
                report.pinned_bytes,
                policy.mode,
            );
        }
        self.pin_report = report;
    }

    // NOTE: cross-group MaxScore threshold seeding is query-execution-local
    // (a Cell in the boolean planner) — it must never live on the shared
    // SegmentReader, where concurrent queries would leak thresholds into
    // each other and wrongly prune results.

    pub fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    pub fn num_docs(&self) -> u32 {
        self.meta.num_docs
    }

    /// Get average field length for BM25F scoring
    pub fn avg_field_len(&self, field: Field) -> f32 {
        self.meta.avg_field_len(field)
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get sparse indexes for all fields
    pub fn sparse_indexes(&self) -> &FxHashMap<u32, SparseIndex> {
        &self.sparse_indexes
    }

    /// Get sparse index for a specific field (MaxScore format)
    pub fn sparse_index(&self, field: Field) -> Option<&SparseIndex> {
        self.sparse_indexes.get(&field.0)
    }

    /// Get BMP index for a specific field
    pub fn bmp_index(&self, field: Field) -> Option<&BmpIndex> {
        self.bmp_indexes.get(&field.0)
    }

    /// Get all BMP indexes
    pub fn bmp_indexes(&self) -> &FxHashMap<u32, BmpIndex> {
        &self.bmp_indexes
    }

    /// Get vector indexes for all fields
    pub fn vector_indexes(&self) -> &FxHashMap<u32, VectorIndex> {
        &self.vector_indexes
    }

    /// Get lazy flat vectors for all fields (for reranking and merge)
    pub fn flat_vectors(&self) -> &FxHashMap<u32, LazyFlatVectorData> {
        &self.flat_vectors
    }

    /// Get a fast-field reader for a specific field.
    pub fn fast_field(
        &self,
        field_id: u32,
    ) -> Option<&crate::structures::fast_field::FastFieldReader> {
        self.fast_fields.get(&field_id)
    }

    /// Get all fast-field readers.
    pub fn fast_fields(&self) -> &FxHashMap<u32, crate::structures::fast_field::FastFieldReader> {
        &self.fast_fields
    }

    /// Get term dictionary stats for debugging
    pub fn term_dict_stats(&self) -> SSTableStats {
        self.term_dict.stats()
    }

    /// Estimate memory usage of this segment reader
    pub fn memory_stats(&self) -> SegmentMemoryStats {
        let term_dict_stats = self.term_dict.stats();

        // Report actual decompressed heap retention. Both caches use variable
        // boundary blocks, so multiplying a block count by a guessed size can
        // materially under-report resident memory.
        let term_dict_cache_bytes = self.term_dict.cached_bytes();
        let store_cache_bytes = self.store.cached_bytes();

        // Sparse index: SoA dim table + OwnedBytes skip section + BMP grids
        let sparse_index_bytes: usize = self
            .sparse_indexes
            .values()
            .map(|s| s.estimated_memory_bytes())
            .sum::<usize>()
            + self
                .bmp_indexes
                .values()
                .map(|b| b.estimated_memory_bytes())
                .sum::<usize>();

        // Dense index: vectors are memory-mapped, but we track index structures
        // RaBitQ/IVF indexes have cluster assignments in memory
        let dense_index_bytes: usize = self
            .vector_indexes
            .values()
            .map(|v| v.estimated_memory_bytes())
            .sum();

        #[cfg(feature = "native")]
        let (pinned_metadata_bytes, pin_intended_bytes) =
            (self.pin_report.pinned_bytes, self.pin_report.intended_bytes);
        #[cfg(not(feature = "native"))]
        let (pinned_metadata_bytes, pin_intended_bytes) = (0u64, 0u64);

        SegmentMemoryStats {
            segment_id: self.meta.id,
            num_docs: self.meta.num_docs,
            term_dict_cache_bytes,
            store_cache_bytes,
            sparse_index_bytes,
            dense_index_bytes,
            bloom_filter_bytes: term_dict_stats.bloom_filter_size,
            pinned_metadata_bytes,
            pin_intended_bytes,
        }
    }

    /// Get posting list for a term (async - loads on demand)
    ///
    /// For small posting lists (1-3 docs), the data is inlined in the term dictionary
    /// and no additional I/O is needed. For larger lists, reads from .post file.
    pub async fn get_postings(
        &self,
        field: Field,
        term: &[u8],
    ) -> Result<Option<BlockPostingList>> {
        log::debug!(
            "SegmentReader::get_postings field={} term_len={}",
            field.0,
            term.len()
        );

        // Build key: field_id + term
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);

        // Look up in term dictionary
        let term_info = match self.term_dict.get(&key).await? {
            Some(info) => {
                log::debug!("SegmentReader::get_postings found term_info");
                info
            }
            None => {
                log::debug!("SegmentReader::get_postings term not found");
                return Ok(None);
            }
        };

        // Check if posting list is inlined
        if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
            // Build BlockPostingList from inline data (no I/O needed!)
            let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
            for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs) {
                posting_list.push(doc_id, tf);
            }
            let block_list = BlockPostingList::from_posting_list(&posting_list)?;
            return Ok(Some(block_list));
        }

        // External posting list - read from postings file handle (lazy - HTTP range request)
        let (posting_offset, posting_len) = term_info.external_info().ok_or_else(|| {
            Error::Corruption("TermInfo has neither inline nor external data".to_string())
        })?;

        let range = checked_file_range(
            posting_offset,
            posting_len,
            self.postings_handle.len(),
            "posting",
        )?;
        let posting_bytes = self.postings_handle.read_bytes_range(range).await?;
        let block_list = BlockPostingList::deserialize_zero_copy(posting_bytes)?;

        Ok(Some(block_list))
    }

    /// Get all posting lists for terms that start with `prefix` in the given field.
    pub async fn get_prefix_postings(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<BlockPostingList>> {
        if prefix.is_empty() {
            return Err(Error::Query("prefix must not be empty".into()));
        }
        // Build composite key prefix: field_id ++ prefix
        let mut key_prefix = Vec::with_capacity(4 + prefix.len());
        key_prefix.extend_from_slice(&field.0.to_le_bytes());
        key_prefix.extend_from_slice(prefix);

        let (entries, truncated) = self
            .term_dict
            .prefix_scan_limited(&key_prefix, MAX_PREFIX_TERMS)
            .await?;
        if truncated {
            return Err(Error::Query(format!(
                "prefix expands to more than {MAX_PREFIX_TERMS} terms"
            )));
        }
        let posting_count: u64 = entries
            .iter()
            .map(|(_, term_info)| term_info.doc_freq() as u64)
            .sum();
        if posting_count > MAX_PREFIX_POSTINGS {
            return Err(Error::Query(format!(
                "prefix expands to {posting_count} postings (maximum {MAX_PREFIX_POSTINGS})"
            )));
        }
        let mut results = Vec::with_capacity(entries.len());

        for (_key, term_info) in entries {
            if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
                let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
                for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs) {
                    posting_list.push(doc_id, tf);
                }
                results.push(BlockPostingList::from_posting_list(&posting_list)?);
            } else if let Some((posting_offset, posting_len)) = term_info.external_info() {
                let range = checked_file_range(
                    posting_offset,
                    posting_len,
                    self.postings_handle.len(),
                    "prefix posting",
                )?;
                let posting_bytes = self.postings_handle.read_bytes_range(range).await?;
                results.push(BlockPostingList::deserialize_zero_copy(posting_bytes)?);
            }
        }

        Ok(results)
    }

    /// Get document by local doc_id (async - loads on demand).
    ///
    /// Dense vector fields are hydrated from LazyFlatVectorData (not stored in .store).
    /// Uses binary search on sorted doc_ids for O(log N) lookup.
    pub async fn doc(&self, local_doc_id: DocId) -> Result<Option<Document>> {
        self.doc_with_fields(local_doc_id, None).await
    }

    /// Get document by local doc_id, hydrating only the specified fields.
    ///
    /// If `fields` is `None`, all fields (including dense vectors) are hydrated.
    /// If `fields` is `Some(set)`, only dense vector fields in the set are hydrated,
    /// skipping expensive mmap reads + dequantization for unrequested vector fields.
    pub async fn doc_with_fields(
        &self,
        local_doc_id: DocId,
        fields: Option<&rustc_hash::FxHashSet<u32>>,
    ) -> Result<Option<Document>> {
        let mut doc = match fields {
            Some(set) => {
                let field_ids: Vec<u32> = set.iter().copied().collect();
                match self
                    .store
                    .get_fields(local_doc_id, &self.schema, &field_ids)
                    .await
                {
                    Ok(Some(d)) => d,
                    Ok(None) => return Ok(None),
                    Err(e) => return Err(Error::from(e)),
                }
            }
            None => match self.store.get(local_doc_id, &self.schema).await {
                Ok(Some(d)) => d,
                Ok(None) => return Ok(None),
                Err(e) => return Err(Error::from(e)),
            },
        };

        // Hydrate dense vector fields from flat vector data
        for (&field_id, lazy_flat) in &self.flat_vectors {
            // Skip vector fields not in the requested set
            if let Some(set) = fields
                && !set.contains(&field_id)
            {
                continue;
            }

            let is_binary = lazy_flat.quantization == DenseVectorQuantization::Binary;
            let (start, entries) = lazy_flat.flat_indexes_for_doc(local_doc_id);
            for (j, &(_doc_id, _ordinal)) in entries.iter().enumerate() {
                let flat_idx = start + j;
                if is_binary {
                    let vbs = lazy_flat.vector_byte_size();
                    let mut raw = vec![0u8; vbs];
                    match lazy_flat.read_vector_raw_into(flat_idx, &mut raw).await {
                        Ok(()) => {
                            doc.add_binary_dense_vector(Field(field_id), raw);
                        }
                        Err(e) => {
                            log::warn!("Failed to hydrate binary vector field {}: {}", field_id, e);
                        }
                    }
                } else {
                    match lazy_flat.get_vector(flat_idx).await {
                        Ok(vec) => {
                            doc.add_dense_vector(Field(field_id), vec);
                        }
                        Err(e) => {
                            log::warn!("Failed to hydrate vector field {}: {}", field_id, e);
                        }
                    }
                }
            }
        }

        Ok(Some(doc))
    }

    /// Prefetch term dictionary blocks for a key range
    pub async fn prefetch_terms(
        &self,
        field: Field,
        start_term: &[u8],
        end_term: &[u8],
    ) -> Result<()> {
        let mut start_key = Vec::with_capacity(4 + start_term.len());
        start_key.extend_from_slice(&field.0.to_le_bytes());
        start_key.extend_from_slice(start_term);

        let mut end_key = Vec::with_capacity(4 + end_term.len());
        end_key.extend_from_slice(&field.0.to_le_bytes());
        end_key.extend_from_slice(end_term);

        self.term_dict.prefetch_range(&start_key, &end_key).await?;
        Ok(())
    }

    /// Check if store uses dictionary compression (incompatible with raw merging)
    pub fn store_has_dict(&self) -> bool {
        self.store.has_dict()
    }

    /// Get store reference for merge operations
    pub fn store(&self) -> &super::store::AsyncStoreReader {
        &self.store
    }

    /// Get raw store blocks for optimized merging
    pub fn store_raw_blocks(&self) -> Vec<RawStoreBlock> {
        self.store.raw_blocks()
    }

    /// Get store data slice for raw block access
    pub fn store_data_slice(&self) -> &FileHandle {
        self.store.data_slice()
    }

    /// Get all terms from this segment (for merge)
    pub async fn all_terms(&self) -> Result<Vec<(Vec<u8>, TermInfo)>> {
        self.term_dict.all_entries().await.map_err(Error::from)
    }

    /// Get all terms with parsed field and term string (for statistics aggregation)
    ///
    /// Returns (field, term_string, doc_freq) for each term in the dictionary.
    /// Skips terms that aren't valid UTF-8.
    pub async fn all_terms_with_stats(&self) -> Result<Vec<(Field, String, u32)>> {
        let entries = self.term_dict.all_entries().await?;
        let mut result = Vec::with_capacity(entries.len());

        for (key, term_info) in entries {
            // Key format: field_id (4 bytes little-endian) + term bytes
            if key.len() > 4 {
                let field_id = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
                let term_bytes = &key[4..];
                if let Ok(term_str) = std::str::from_utf8(term_bytes) {
                    result.push((Field(field_id), term_str.to_string(), term_info.doc_freq()));
                }
            }
        }

        Ok(result)
    }

    /// Get streaming iterator over term dictionary (for memory-efficient merge)
    pub fn term_dict_iter(&self) -> crate::structures::AsyncSSTableIterator<'_, TermInfo> {
        self.term_dict.iter()
    }

    /// Prefetch all term dictionary blocks in a single bulk I/O call.
    ///
    /// Call before merge iteration to eliminate per-block cache misses.
    pub async fn prefetch_term_dict(&self) -> crate::Result<()> {
        self.term_dict
            .prefetch_all_data_bulk()
            .await
            .map_err(crate::Error::from)
    }

    /// Read raw posting bytes at offset
    pub async fn read_postings(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let range = checked_file_range(offset, len, self.postings_handle.len(), "posting")?;
        let bytes = self.postings_handle.read_bytes_range(range).await?;
        Ok(bytes.to_vec())
    }

    /// Read raw position bytes at offset (for merge)
    pub async fn read_position_bytes(&self, offset: u64, len: u64) -> Result<Option<Vec<u8>>> {
        let handle = match &self.positions_handle {
            Some(h) => h,
            None => return Ok(None),
        };
        let range = checked_file_range(offset, len, handle.len(), "position")?;
        let bytes = handle.read_bytes_range(range).await?;
        Ok(Some(bytes.to_vec()))
    }

    /// Check if this segment has a positions file
    pub fn has_positions_file(&self) -> bool {
        self.positions_handle.is_some()
    }

    /// Validate all caller-controlled dense-search inputs before touching ANN
    /// structures or entering SIMD code. This is deliberately repeated at the
    /// segment boundary so non-server users receive the same safety guarantees.
    fn validate_dense_search_request(
        &self,
        field: Field,
        query: &[f32],
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<DenseSearchParams> {
        let entry = self
            .schema
            .get_field_entry(field)
            .ok_or_else(|| Error::FieldNotFound(field.0.to_string()))?;
        if entry.field_type != crate::dsl::FieldType::DenseVector {
            return Err(Error::InvalidFieldType {
                expected: "dense_vector".to_string(),
                got: format!("{:?}", entry.field_type),
            });
        }
        let config = entry.dense_vector_config.as_ref().ok_or_else(|| {
            Error::Schema(format!(
                "dense vector field '{}' has no dense vector configuration",
                entry.name
            ))
        })?;

        if query.is_empty() {
            return Err(Error::Query(format!(
                "dense query vector for field '{}' must not be empty",
                entry.name
            )));
        }
        if query.len() != config.dim {
            return Err(Error::Query(format!(
                "dense query vector dimension {} does not match field '{}' dimension {}",
                query.len(),
                entry.name,
                config.dim
            )));
        }
        if let Some((index, value)) = query
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(Error::Query(format!(
                "dense query vector for field '{}' contains non-finite value {value} at index {index}",
                entry.name
            )));
        }

        // A zero query override means "use the schema". Legacy schemas may
        // contain zero for flat fields, so retain 32 as a final ANN fallback.
        let nprobe = match (nprobe, config.nprobe) {
            (0, 0) => 32,
            (0, schema_nprobe) => schema_nprobe,
            (query_nprobe, _) => query_nprobe,
        };
        if nprobe > MAX_DENSE_NPROBE {
            return Err(Error::Query(format!(
                "dense nprobe must be at most {MAX_DENSE_NPROBE}, got {nprobe}"
            )));
        }

        // Validate the factor here even for empty segments. Otherwise malformed
        // requests would succeed or fail depending on segment contents.
        checked_dense_fetch_k(0, rerank_factor)?;
        combiner.validate().map_err(Error::Query)?;

        Ok(DenseSearchParams {
            dim: config.dim,
            nprobe,
            unit_norm: config.unit_norm,
        })
    }

    fn validate_binary_search_request(&self, field: Field, query: &[u8]) -> Result<usize> {
        let entry = self
            .schema
            .get_field_entry(field)
            .ok_or_else(|| Error::FieldNotFound(field.0.to_string()))?;
        if entry.field_type != crate::dsl::FieldType::BinaryDenseVector {
            return Err(Error::InvalidFieldType {
                expected: "binary_dense_vector".to_string(),
                got: format!("{:?}", entry.field_type),
            });
        }
        let config = entry.binary_dense_vector_config.as_ref().ok_or_else(|| {
            Error::Schema(format!(
                "binary dense vector field '{}' has no configuration",
                entry.name
            ))
        })?;
        if config.dim == 0 || !config.dim.is_multiple_of(8) {
            return Err(Error::Schema(format!(
                "binary dense vector field '{}' has invalid dimension {}",
                entry.name, config.dim
            )));
        }
        if query.len() != config.byte_len() {
            return Err(Error::Query(format!(
                "binary query byte length {} does not match field '{}' byte length {}",
                query.len(),
                entry.name,
                config.byte_len()
            )));
        }
        Ok(config.dim)
    }

    /// Batch cosine scoring on raw quantized bytes.
    ///
    /// Dispatches to the appropriate SIMD scorer based on quantization type.
    /// Vectors file uses data-first layout (offset 0) with 8-byte padding between
    /// fields, so mmap slices are always properly aligned for f32/f16/u8 access.
    fn score_quantized_batch(
        query: &[f32],
        raw: &[u8],
        quant: crate::dsl::DenseVectorQuantization,
        dim: usize,
        scores: &mut [f32],
        unit_norm: bool,
    ) -> Result<()> {
        use crate::dsl::DenseVectorQuantization;
        use crate::structures::simd;

        if query.len() != dim {
            return Err(Error::Query(format!(
                "dense SIMD query dimension {} does not match vector dimension {dim}",
                query.len()
            )));
        }
        let element_size = match quant {
            DenseVectorQuantization::F32 => std::mem::size_of::<f32>(),
            DenseVectorQuantization::F16 => std::mem::size_of::<u16>(),
            DenseVectorQuantization::UInt8 => 1,
            DenseVectorQuantization::Binary => {
                return Err(Error::InvalidFieldType {
                    expected: "non-binary dense vector".to_string(),
                    got: "binary dense vector".to_string(),
                });
            }
        };
        let required_bytes = scores
            .len()
            .checked_mul(dim)
            .and_then(|elements| elements.checked_mul(element_size))
            .ok_or_else(|| Error::Corruption("dense vector batch byte length overflow".into()))?;
        if raw.len() < required_bytes {
            return Err(Error::Corruption(format!(
                "dense vector batch is truncated: need {required_bytes} bytes, got {}",
                raw.len()
            )));
        }
        if quant == DenseVectorQuantization::F16
            && required_bytes > 0
            && !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>())
        {
            return Err(Error::Corruption(
                "f16 vector data is not 2-byte aligned".to_string(),
            ));
        }

        match (quant, unit_norm) {
            (DenseVectorQuantization::F32, false) => {
                let num_floats = scores.len() * dim;
                if !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                    return Err(Error::Corruption(
                        "f32 vector data is not 4-byte aligned".to_string(),
                    ));
                }
                let vectors: &[f32] =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
                simd::batch_cosine_scores(query, vectors, dim, scores);
            }
            (DenseVectorQuantization::F32, true) => {
                let num_floats = scores.len() * dim;
                if !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                    return Err(Error::Corruption(
                        "f32 vector data is not 4-byte aligned".to_string(),
                    ));
                }
                let vectors: &[f32] =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
                simd::batch_dot_scores(query, vectors, dim, scores);
            }
            (DenseVectorQuantization::F16, false) => {
                simd::batch_cosine_scores_f16(query, raw, dim, scores);
            }
            (DenseVectorQuantization::F16, true) => {
                simd::batch_dot_scores_f16(query, raw, dim, scores);
            }
            (DenseVectorQuantization::UInt8, false) => {
                simd::batch_cosine_scores_u8(query, raw, dim, scores);
            }
            (DenseVectorQuantization::UInt8, true) => {
                simd::batch_dot_scores_u8(query, raw, dim, scores);
            }
            (DenseVectorQuantization::Binary, _) => unreachable!("validated above"),
        }
        Ok(())
    }

    /// Search dense vectors using RaBitQ
    ///
    /// Returns VectorSearchResult with ordinal tracking for multi-value fields.
    /// Doc IDs are segment-local.
    /// For multi-valued documents, scores are combined using the specified combiner.
    pub async fn search_dense_vector(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        let params =
            self.validate_dense_search_request(field, query, nprobe, rerank_factor, combiner)?;
        let fetch_k = checked_dense_fetch_k(k, rerank_factor)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let ann_index = self.vector_indexes.get(&field.0);
        let lazy_flat = self.flat_vectors.get(&field.0);
        let ann_fetch_k = lazy_flat.map_or(fetch_k, |flat| {
            ann_ordinal_fetch_k(fetch_k, flat.num_vectors, flat.num_docs_with_vectors())
        });

        // No vectors at all for this field
        if ann_index.is_none() && lazy_flat.is_none() {
            return Ok(Vec::new());
        }

        if ann_index.is_some() && lazy_flat.is_none() {
            return Err(Error::Corruption(format!(
                "dense ANN field {} is missing flat vector storage",
                field.0
            )));
        }

        if let Some(flat) = lazy_flat
            && flat.dim != params.dim
        {
            return Err(Error::Corruption(format!(
                "dense vector field {} has schema dimension {} but flat storage dimension {}",
                field.0, params.dim, flat.dim
            )));
        }

        // Results are (doc_id, ordinal, score) where score = similarity (higher = better)
        let t0 = std::time::Instant::now();
        let mut flat_results = None;
        let mut results: Vec<(u32, u16, f32)> = if let Some(index) = ann_index {
            // ANN search (RaBitQ, IVF, ScaNN)
            match index {
                VectorIndex::RaBitQ(lazy) => {
                    let rabitq = lazy.get().ok_or_else(|| {
                        Error::Schema("RaBitQ index deserialization failed".to_string())
                    })?;
                    if rabitq.codebook.config.dim != params.dim {
                        return Err(Error::Corruption(format!(
                            "RaBitQ index dimension {} does not match schema dimension {}",
                            rabitq.codebook.config.dim, params.dim
                        )));
                    }
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    progressive_ann_search(
                        fetch_k.min(flat.num_docs_with_vectors()),
                        ann_fetch_k,
                        rabitq.len().min(flat.num_vectors),
                        |candidate_k| {
                            rabitq
                                .search(query, candidate_k)
                                .into_iter()
                                .map(|(doc_id, ordinal, dist)| {
                                    (doc_id, ordinal, 1.0 / (1.0 + dist))
                                })
                                .collect()
                        },
                    )?
                }
                VectorIndex::IVF(lazy) => {
                    let (index, codebook) = lazy.get().ok_or_else(|| {
                        Error::Schema("IVF index deserialization failed".to_string())
                    })?;
                    let centroids = self.coarse_centroids.get(&field.0).ok_or_else(|| {
                        Error::Schema(format!(
                            "IVF index requires coarse centroids for field {}",
                            field.0
                        ))
                    })?;
                    validate_coarse_centroids(centroids, params.dim)?;
                    if index.config.dim != params.dim
                        || codebook.config.dim != params.dim
                        || index.centroids_version != centroids.version
                        || index.codebook_version != codebook.version
                        || index
                            .clusters
                            .iter()
                            .any(|(cluster_id, _)| cluster_id >= centroids.num_clusters)
                    {
                        return Err(Error::Corruption(format!(
                            "IVF index/codebook/centroid metadata does not match schema dimension {}",
                            params.dim
                        )));
                    }
                    let effective_nprobe = params.nprobe.min(centroids.num_clusters as usize);
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    progressive_ann_search(
                        fetch_k.min(flat.num_docs_with_vectors()),
                        ann_fetch_k,
                        index.len().min(flat.num_vectors),
                        |candidate_k| {
                            index
                                .search(
                                    centroids,
                                    codebook,
                                    query,
                                    candidate_k,
                                    Some(effective_nprobe),
                                )
                                .into_iter()
                                .map(|(doc_id, ordinal, dist)| {
                                    (doc_id, ordinal, 1.0 / (1.0 + dist))
                                })
                                .collect()
                        },
                    )?
                }
                VectorIndex::ScaNN(lazy) => {
                    let (index, codebook) = lazy.get().ok_or_else(|| {
                        Error::Schema("ScaNN index deserialization failed".to_string())
                    })?;
                    let centroids = self.coarse_centroids.get(&field.0).ok_or_else(|| {
                        Error::Schema(format!(
                            "ScaNN index requires coarse centroids for field {}",
                            field.0
                        ))
                    })?;
                    validate_coarse_centroids(centroids, params.dim)?;
                    if index.config.dim != params.dim
                        || codebook.config.dim != params.dim
                        || index.centroids_version != centroids.version
                        || index.codebook_version != codebook.version
                        || index
                            .clusters
                            .iter()
                            .any(|(cluster_id, _)| cluster_id >= centroids.num_clusters)
                    {
                        return Err(Error::Corruption(format!(
                            "ScaNN index/codebook/centroid metadata does not match schema dimension {}",
                            params.dim
                        )));
                    }
                    let effective_nprobe = params.nprobe.min(centroids.num_clusters as usize);
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    progressive_ann_search(
                        fetch_k.min(flat.num_docs_with_vectors()),
                        ann_fetch_k,
                        index.len().min(flat.num_vectors),
                        |candidate_k| {
                            index
                                .search(
                                    centroids,
                                    codebook,
                                    query,
                                    candidate_k,
                                    Some(effective_nprobe),
                                )
                                .into_iter()
                                .map(|(doc_id, ordinal, dist)| {
                                    (doc_id, ordinal, 1.0 / (1.0 + dist))
                                })
                                .collect()
                        },
                    )?
                }
                VectorIndex::BinaryIvf(_) => {
                    // Binary IVF serves Hamming queries only (BinaryDenseVectorQuery)
                    Vec::new()
                }
            }
        } else if let Some(lazy_flat) = lazy_flat {
            // Batched brute-force from lazy flat vectors (native-precision SIMD scoring).
            // Combine every value of a document before document-level top-k;
            // vector-level top-k loses documents on multi-valued fields.
            log::debug!(
                "[search_dense] field {}: brute-force on {} vectors (dim={}, quant={:?})",
                field.0,
                lazy_flat.num_vectors,
                lazy_flat.dim,
                lazy_flat.quantization
            );
            let dim = lazy_flat.dim;
            let n = lazy_flat.num_vectors;
            let quant = lazy_flat.quantization;
            let batch_len =
                bounded_vector_score_batch(lazy_flat.vector_byte_size(), DENSE_SCORE_BATCH);
            let mut collector = FlatDocumentCollector::new(fetch_k.min(n), combiner);
            let mut scores = vec![0f32; batch_len];

            for batch_start in (0..n).step_by(batch_len) {
                let batch_count = batch_len.min(n - batch_start);
                let batch_bytes = lazy_flat
                    .read_vectors_batch(batch_start, batch_count)
                    .await
                    .map_err(crate::Error::Io)?;
                let raw = batch_bytes.as_slice();

                Self::score_quantized_batch(
                    query,
                    raw,
                    quant,
                    dim,
                    &mut scores[..batch_count],
                    params.unit_norm,
                )?;

                for (i, &score) in scores.iter().enumerate().take(batch_count) {
                    let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                    collector.push(doc_id, ordinal, score);
                }
            }

            flat_results = Some(collector.into_results());
            Vec::new()
        } else {
            return Ok(Vec::new());
        };
        let l1_elapsed = t0.elapsed();
        {
            let kind = match ann_index {
                Some(VectorIndex::RaBitQ(_)) => "rabitq",
                Some(VectorIndex::IVF(_)) => "ivf_rabitq",
                Some(VectorIndex::ScaNN(_)) => "scann",
                Some(VectorIndex::BinaryIvf(_)) => "binary_ivf",
                None => "flat",
            };
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                kind,
                l1_elapsed.as_secs_f64(),
                flat_results.as_ref().map_or(results.len(), Vec::len),
            );
        }
        log::debug!(
            "[search_dense] field {}: L1 returned {} candidates in {:.1}ms",
            field.0,
            flat_results.as_ref().map_or(results.len(), Vec::len),
            l1_elapsed.as_secs_f64() * 1000.0
        );

        if let Some(results) = flat_results {
            return Ok(results);
        }

        // Rerank ANN candidates using raw vectors from lazy flat (binary search lookup)
        // Uses native-precision SIMD scoring on quantized bytes — no dequantization overhead.
        if ann_index.is_some()
            && !results.is_empty()
            && let Some(lazy_flat) = lazy_flat
        {
            let t_rerank = std::time::Instant::now();
            let dim = lazy_flat.dim;
            let quant = lazy_flat.quantization;
            let vbs = lazy_flat.vector_byte_size();

            // Deduplicate ANN hits by document, then exact-score every value
            // of each candidate document so the configured combiner sees a
            // complete value set.
            let (expanded, mut resolved) = expand_ann_candidate_documents(&results, lazy_flat)?;
            results = expanded;

            let t_resolve = t_rerank.elapsed();
            if !resolved.is_empty() {
                // Sort by flat_idx for sequential mmap access (better page locality)
                resolved.sort_unstable_by_key(|&(_, flat_idx)| flat_idx);

                // Read and score bounded chunks. A legal high-recall request
                // can resolve hundreds of thousands of candidates; allocating
                // all raw vectors at once used to require gigabytes per segment.
                let batch_len = bounded_vector_score_batch(vbs, DENSE_SCORE_BATCH);
                let max_batch = batch_len.min(resolved.len());
                let max_raw_len = max_batch
                    .checked_mul(vbs)
                    .ok_or_else(|| Error::Query("dense rerank buffer size overflow".into()))?;
                let mut raw_buf = vec![0u8; max_raw_len];
                let mut scores = vec![0f32; max_batch];
                let mut read_elapsed = std::time::Duration::ZERO;
                let mut score_elapsed = std::time::Duration::ZERO;

                for chunk in resolved.chunks(batch_len) {
                    // Advise only the bounded chunk about to be consumed. A
                    // whole-candidate MADV_WILLNEED can evict the rest of the
                    // working set long before those pages are read.
                    #[cfg(feature = "native")]
                    lazy_flat.prefetch_vectors(chunk.iter().map(|&(_, flat_idx)| flat_idx));
                    let raw_len = chunk
                        .len()
                        .checked_mul(vbs)
                        .ok_or_else(|| Error::Query("dense rerank buffer size overflow".into()))?;
                    let raw = &mut raw_buf[..raw_len];

                    let t_read = std::time::Instant::now();
                    for (buf_idx, &(_, flat_idx)) in chunk.iter().enumerate() {
                        lazy_flat
                            .read_vector_raw_into(
                                flat_idx,
                                &mut raw[buf_idx * vbs..(buf_idx + 1) * vbs],
                            )
                            .await
                            .map_err(crate::Error::Io)?;
                    }
                    read_elapsed += t_read.elapsed();

                    let t_score = std::time::Instant::now();
                    Self::score_quantized_batch(
                        query,
                        raw,
                        quant,
                        dim,
                        &mut scores[..chunk.len()],
                        params.unit_norm,
                    )?;
                    score_elapsed += t_score.elapsed();

                    for (buf_idx, &(ri, _)) in chunk.iter().enumerate() {
                        results[ri].2 = scores[buf_idx];
                    }
                }

                crate::observe::dense_rerank(
                    self.schema.index_label(),
                    self.schema.get_field_name(field).unwrap_or("?"),
                    t_rerank.elapsed().as_secs_f64(),
                    t_resolve.as_secs_f64(),
                    read_elapsed.as_secs_f64(),
                    resolved.len(),
                );
                log::debug!(
                    "[search_dense] field {}: rerank {} vectors (dim={}, quant={:?}, {}B/vec): resolve={:.1}ms read={:.1}ms score={:.1}ms",
                    field.0,
                    resolved.len(),
                    dim,
                    quant,
                    vbs,
                    t_resolve.as_secs_f64() * 1000.0,
                    read_elapsed.as_secs_f64() * 1000.0,
                    score_elapsed.as_secs_f64() * 1000.0,
                );
            }

            log::debug!(
                "[search_dense] field {}: rerank total={:.1}ms",
                field.0,
                t_rerank.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(combine_grouped_ordinal_results(results, combiner, k))
    }

    /// Query-time nprobe for a binary field from its schema config (None = index default).
    fn binary_ivf_nprobe(&self, field: Field) -> Option<usize> {
        self.schema
            .get_field_entry(field)
            .and_then(|e| e.binary_dense_vector_config.as_ref())
            .map(|c| c.nprobe)
            .filter(|&n| n > 0)
    }

    /// Search binary dense vectors using brute-force Hamming distance.
    ///
    /// Always flat brute-force (no ANN). Returns VectorSearchResult with ordinal tracking.
    pub async fn search_binary_dense_vector(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        let schema_dim = self.validate_binary_search_request(field, query)?;
        combiner.validate().map_err(Error::Query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let t0 = crate::observe::Timer::start();
        // Binary IVF finds candidate documents. Expand each candidate to all
        // of its stored values and exact-score them before applying the
        // document combiner.
        if let Some(VectorIndex::BinaryIvf(lazy)) = self.vector_indexes.get(&field.0)
            && let Some(ivf) = lazy.get()
        {
            if ivf.config.dim_bits != schema_dim {
                return Err(Error::Corruption(format!(
                    "binary IVF field {} has schema dimension {} but index dimension {}",
                    field.0, schema_dim, ivf.config.dim_bits
                )));
            }
            let flat = self.flat_vectors.get(&field.0).ok_or_else(|| {
                Error::Corruption(format!(
                    "binary IVF field {} is missing flat vector storage",
                    field.0
                ))
            })?;
            if flat.dim != schema_dim
                || flat.quantization != crate::dsl::DenseVectorQuantization::Binary
            {
                return Err(Error::Corruption(format!(
                    "binary IVF field {} has inconsistent flat vector metadata",
                    field.0
                )));
            }
            let nprobe = self.binary_ivf_nprobe(field);
            let initial_fetch =
                ann_ordinal_fetch_k(k, flat.num_vectors, flat.num_docs_with_vectors());
            // Single scan: BinaryIvfIndex::search scans all probed clusters
            // regardless of candidate_k, so progressive doubling would repeat
            // the identical scan on every retry.
            let ann_results = single_scan_ann_search(
                k.min(flat.num_docs_with_vectors()),
                initial_fetch,
                ivf.len().min(flat.num_vectors),
                |candidate_k| ivf.search(query, candidate_k, nprobe),
            )?;
            let results =
                exact_score_binary_candidate_documents(&ann_results, flat, query, schema_dim)
                    .await?;
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "binary_ivf",
                t0.secs(),
                results.len(),
            );
            return Ok(combine_grouped_ordinal_results(results, combiner, k));
        }

        let lazy_flat = match self.flat_vectors.get(&field.0) {
            Some(f) => f,
            None => return Ok(Vec::new()),
        };

        let dim_bits = lazy_flat.dim;
        let byte_len = lazy_flat.vector_byte_size();
        let n = lazy_flat.num_vectors;

        if dim_bits != schema_dim {
            return Err(Error::Corruption(format!(
                "binary vector field {} has schema dimension {} but flat storage dimension {}",
                field.0, schema_dim, dim_bits
            )));
        }

        if byte_len != query.len() {
            return Err(Error::Schema(format!(
                "Binary query vector byte length {} != field byte length {}",
                query.len(),
                byte_len
            )));
        }

        let batch_len = bounded_vector_score_batch(byte_len, BINARY_SCORE_BATCH);
        let mut collector = FlatDocumentCollector::new(k, combiner);
        let mut scores = vec![0f32; batch_len];

        for batch_start in (0..n).step_by(batch_len) {
            let batch_count = batch_len.min(n - batch_start);
            let batch_bytes = lazy_flat
                .read_vectors_batch(batch_start, batch_count)
                .await
                .map_err(crate::Error::Io)?;
            let raw = batch_bytes.as_slice();

            crate::structures::simd::batch_hamming_scores(
                query,
                raw,
                byte_len,
                dim_bits,
                &mut scores[..batch_count],
            );

            for (i, &score) in scores.iter().enumerate().take(batch_count) {
                let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                collector.push(doc_id, ordinal, score);
            }
        }

        let results = collector.into_results();

        crate::observe::dense_l1(
            self.schema.index_label(),
            self.schema.get_field_name(field).unwrap_or("?"),
            "binary_flat",
            t0.secs(),
            results.len(),
        );
        Ok(results)
    }

    /// Check if this segment has dense vectors for the given field
    pub fn has_dense_vector_index(&self, field: Field) -> bool {
        self.vector_indexes.contains_key(&field.0) || self.flat_vectors.contains_key(&field.0)
    }

    /// Get the dense vector index for a field (if available)
    pub fn get_dense_vector_index(&self, field: Field) -> Option<Arc<RaBitQIndex>> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::RaBitQ(lazy)) => lazy.get().cloned(),
            _ => None,
        }
    }

    /// Get the IVF vector index for a field (if available)
    pub fn get_ivf_vector_index(
        &self,
        field: Field,
    ) -> Option<(Arc<IVFRaBitQIndex>, Arc<crate::structures::RaBitQCodebook>)> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::IVF(lazy)) => lazy.get().map(|(i, c)| (i.clone(), c.clone())),
            _ => None,
        }
    }

    /// Get coarse centroids for a field
    pub fn coarse_centroids(&self, field_id: u32) -> Option<&Arc<CoarseCentroids>> {
        self.coarse_centroids.get(&field_id)
    }

    /// Set per-field coarse centroids from index-level trained structures
    pub fn set_coarse_centroids(&mut self, centroids: FxHashMap<u32, Arc<CoarseCentroids>>) {
        self.coarse_centroids = centroids;
    }

    /// Get the ScaNN vector index for a field (if available)
    pub fn get_scann_vector_index(
        &self,
        field: Field,
    ) -> Option<(Arc<IVFPQIndex>, Arc<PQCodebook>)> {
        match self.vector_indexes.get(&field.0) {
            Some(VectorIndex::ScaNN(lazy)) => lazy.get().map(|(i, c)| (i.clone(), c.clone())),
            _ => None,
        }
    }

    /// Get the vector index type for a field
    pub fn get_vector_index(&self, field: Field) -> Option<&VectorIndex> {
        self.vector_indexes.get(&field.0)
    }

    /// Get positions for a term (for phrase queries)
    ///
    /// Position offsets are now embedded in TermInfo, so we first look up
    /// the term to get its TermInfo, then use position_info() to get the offset.
    pub async fn get_positions(
        &self,
        field: Field,
        term: &[u8],
    ) -> Result<Option<crate::structures::PositionPostingList>> {
        // Get positions handle
        let handle = match &self.positions_handle {
            Some(h) => h,
            None => return Ok(None),
        };

        // Build key: field_id + term
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);

        // Look up term in dictionary to get TermInfo with position offset
        let term_info = match self.term_dict.get(&key).await? {
            Some(info) => info,
            None => return Ok(None),
        };

        // Get position offset from TermInfo
        let (offset, length) = match term_info.position_info() {
            Some((o, l)) => (o, l),
            None => return Ok(None),
        };

        // Read the position data only after validating untrusted offsets from
        // the term dictionary. Direct `offset + length` can wrap in release
        // builds and alias an unrelated range.
        let range = checked_file_range(offset, length, handle.len(), "position list")?;
        let slice = handle.slice(range);
        let data = slice.read_bytes().await?;

        // Deserialize
        let pos_list = crate::structures::PositionPostingList::deserialize(data.as_slice())?;

        Ok(Some(pos_list))
    }

    /// Check if positions are available for a field
    pub fn has_positions(&self, field: Field) -> bool {
        // Check schema for position mode on this field
        if let Some(entry) = self.schema.get_field_entry(field) {
            entry.positions.is_some()
        } else {
            false
        }
    }
}

// ── Synchronous search methods (mmap/RAM only) ─────────────────────────────
#[cfg(feature = "sync")]
impl SegmentReader {
    /// Synchronous posting list lookup — requires Inline (mmap/RAM) file handles.
    pub fn get_postings_sync(&self, field: Field, term: &[u8]) -> Result<Option<BlockPostingList>> {
        // Build key: field_id + term
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);

        // Look up in term dictionary (sync)
        let term_info = match self.term_dict.get_sync(&key)? {
            Some(info) => info,
            None => return Ok(None),
        };

        // Check if posting list is inlined
        if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
            let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
            for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs) {
                posting_list.push(doc_id, tf);
            }
            let block_list = BlockPostingList::from_posting_list(&posting_list)?;
            return Ok(Some(block_list));
        }

        // External posting list — sync range read
        let (posting_offset, posting_len) = term_info.external_info().ok_or_else(|| {
            Error::Corruption("TermInfo has neither inline nor external data".to_string())
        })?;

        let range = checked_file_range(
            posting_offset,
            posting_len,
            self.postings_handle.len(),
            "posting",
        )?;
        let posting_bytes = self.postings_handle.read_bytes_range_sync(range)?;
        let block_list = BlockPostingList::deserialize_zero_copy(posting_bytes)?;

        Ok(Some(block_list))
    }

    /// Synchronous prefix posting list lookup — requires Inline (mmap/RAM) file handles.
    pub fn get_prefix_postings_sync(
        &self,
        field: Field,
        prefix: &[u8],
    ) -> Result<Vec<BlockPostingList>> {
        if prefix.is_empty() {
            return Err(Error::Query("prefix must not be empty".into()));
        }
        let mut key_prefix = Vec::with_capacity(4 + prefix.len());
        key_prefix.extend_from_slice(&field.0.to_le_bytes());
        key_prefix.extend_from_slice(prefix);

        let (entries, truncated) = self
            .term_dict
            .prefix_scan_limited_sync(&key_prefix, MAX_PREFIX_TERMS)?;
        if truncated {
            return Err(Error::Query(format!(
                "prefix expands to more than {MAX_PREFIX_TERMS} terms"
            )));
        }
        let posting_count: u64 = entries
            .iter()
            .map(|(_, term_info)| term_info.doc_freq() as u64)
            .sum();
        if posting_count > MAX_PREFIX_POSTINGS {
            return Err(Error::Query(format!(
                "prefix expands to {posting_count} postings (maximum {MAX_PREFIX_POSTINGS})"
            )));
        }
        let mut results = Vec::with_capacity(entries.len());

        for (_key, term_info) in entries {
            if let Some((doc_ids, term_freqs)) = term_info.decode_inline() {
                let mut posting_list = crate::structures::PostingList::with_capacity(doc_ids.len());
                for (doc_id, tf) in doc_ids.into_iter().zip(term_freqs) {
                    posting_list.push(doc_id, tf);
                }
                results.push(BlockPostingList::from_posting_list(&posting_list)?);
            } else if let Some((posting_offset, posting_len)) = term_info.external_info() {
                let range = checked_file_range(
                    posting_offset,
                    posting_len,
                    self.postings_handle.len(),
                    "prefix posting",
                )?;
                let posting_bytes = self.postings_handle.read_bytes_range_sync(range)?;
                results.push(BlockPostingList::deserialize_zero_copy(posting_bytes)?);
            }
        }

        Ok(results)
    }

    /// Synchronous position list lookup — requires Inline (mmap/RAM) file handles.
    pub fn get_positions_sync(
        &self,
        field: Field,
        term: &[u8],
    ) -> Result<Option<crate::structures::PositionPostingList>> {
        let handle = match &self.positions_handle {
            Some(h) => h,
            None => return Ok(None),
        };

        // Build key: field_id + term
        let mut key = Vec::with_capacity(4 + term.len());
        key.extend_from_slice(&field.0.to_le_bytes());
        key.extend_from_slice(term);

        // Look up term in dictionary (sync)
        let term_info = match self.term_dict.get_sync(&key)? {
            Some(info) => info,
            None => return Ok(None),
        };

        let (offset, length) = match term_info.position_info() {
            Some((o, l)) => (o, l),
            None => return Ok(None),
        };

        let range = checked_file_range(offset, length, handle.len(), "position list")?;
        let slice = handle.slice(range);
        let data = slice.read_bytes_sync()?;

        let pos_list = crate::structures::PositionPostingList::deserialize(data.as_slice())?;
        Ok(Some(pos_list))
    }

    /// Synchronous dense vector search — ANN indexes are already sync,
    /// brute-force uses sync mmap reads.
    pub fn search_dense_vector_sync(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        let params =
            self.validate_dense_search_request(field, query, nprobe, rerank_factor, combiner)?;
        let fetch_k = checked_dense_fetch_k(k, rerank_factor)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let ann_index = self.vector_indexes.get(&field.0);
        let lazy_flat = self.flat_vectors.get(&field.0);
        let ann_fetch_k = lazy_flat.map_or(fetch_k, |flat| {
            ann_ordinal_fetch_k(fetch_k, flat.num_vectors, flat.num_docs_with_vectors())
        });

        if ann_index.is_none() && lazy_flat.is_none() {
            return Ok(Vec::new());
        }

        if ann_index.is_some() && lazy_flat.is_none() {
            return Err(Error::Corruption(format!(
                "dense ANN field {} is missing flat vector storage",
                field.0
            )));
        }

        if let Some(flat) = lazy_flat
            && flat.dim != params.dim
        {
            return Err(Error::Corruption(format!(
                "dense vector field {} has schema dimension {} but flat storage dimension {}",
                field.0, params.dim, flat.dim
            )));
        }

        let mut results: Vec<(u32, u16, f32)> = if let Some(index) = ann_index {
            // ANN search (already sync)
            match index {
                VectorIndex::RaBitQ(lazy) => {
                    let rabitq = lazy.get().ok_or_else(|| {
                        Error::Schema("RaBitQ index deserialization failed".to_string())
                    })?;
                    if rabitq.codebook.config.dim != params.dim {
                        return Err(Error::Corruption(format!(
                            "RaBitQ index dimension {} does not match schema dimension {}",
                            rabitq.codebook.config.dim, params.dim
                        )));
                    }
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    progressive_ann_search(
                        fetch_k.min(flat.num_docs_with_vectors()),
                        ann_fetch_k,
                        rabitq.len().min(flat.num_vectors),
                        |candidate_k| {
                            rabitq
                                .search(query, candidate_k)
                                .into_iter()
                                .map(|(doc_id, ordinal, dist)| {
                                    (doc_id, ordinal, 1.0 / (1.0 + dist))
                                })
                                .collect()
                        },
                    )?
                }
                VectorIndex::IVF(lazy) => {
                    let (index, codebook) = lazy.get().ok_or_else(|| {
                        Error::Schema("IVF index deserialization failed".to_string())
                    })?;
                    let centroids = self.coarse_centroids.get(&field.0).ok_or_else(|| {
                        Error::Schema(format!(
                            "IVF index requires coarse centroids for field {}",
                            field.0
                        ))
                    })?;
                    validate_coarse_centroids(centroids, params.dim)?;
                    if index.config.dim != params.dim
                        || codebook.config.dim != params.dim
                        || index.centroids_version != centroids.version
                        || index.codebook_version != codebook.version
                        || index
                            .clusters
                            .iter()
                            .any(|(cluster_id, _)| cluster_id >= centroids.num_clusters)
                    {
                        return Err(Error::Corruption(format!(
                            "IVF index/codebook/centroid metadata does not match schema dimension {}",
                            params.dim
                        )));
                    }
                    let effective_nprobe = params.nprobe.min(centroids.num_clusters as usize);
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    progressive_ann_search(
                        fetch_k.min(flat.num_docs_with_vectors()),
                        ann_fetch_k,
                        index.len().min(flat.num_vectors),
                        |candidate_k| {
                            index
                                .search(
                                    centroids,
                                    codebook,
                                    query,
                                    candidate_k,
                                    Some(effective_nprobe),
                                )
                                .into_iter()
                                .map(|(doc_id, ordinal, dist)| {
                                    (doc_id, ordinal, 1.0 / (1.0 + dist))
                                })
                                .collect()
                        },
                    )?
                }
                VectorIndex::ScaNN(lazy) => {
                    let (index, codebook) = lazy.get().ok_or_else(|| {
                        Error::Schema("ScaNN index deserialization failed".to_string())
                    })?;
                    let centroids = self.coarse_centroids.get(&field.0).ok_or_else(|| {
                        Error::Schema(format!(
                            "ScaNN index requires coarse centroids for field {}",
                            field.0
                        ))
                    })?;
                    validate_coarse_centroids(centroids, params.dim)?;
                    if index.config.dim != params.dim
                        || codebook.config.dim != params.dim
                        || index.centroids_version != centroids.version
                        || index.codebook_version != codebook.version
                        || index
                            .clusters
                            .iter()
                            .any(|(cluster_id, _)| cluster_id >= centroids.num_clusters)
                    {
                        return Err(Error::Corruption(format!(
                            "ScaNN index/codebook/centroid metadata does not match schema dimension {}",
                            params.dim
                        )));
                    }
                    let effective_nprobe = params.nprobe.min(centroids.num_clusters as usize);
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    progressive_ann_search(
                        fetch_k.min(flat.num_docs_with_vectors()),
                        ann_fetch_k,
                        index.len().min(flat.num_vectors),
                        |candidate_k| {
                            index
                                .search(
                                    centroids,
                                    codebook,
                                    query,
                                    candidate_k,
                                    Some(effective_nprobe),
                                )
                                .into_iter()
                                .map(|(doc_id, ordinal, dist)| {
                                    (doc_id, ordinal, 1.0 / (1.0 + dist))
                                })
                                .collect()
                        },
                    )?
                }
                VectorIndex::BinaryIvf(_) => {
                    // Binary IVF serves Hamming queries only (BinaryDenseVectorQuery)
                    Vec::new()
                }
            }
        } else if let Some(lazy_flat) = lazy_flat {
            // Batched brute-force (sync mmap reads)
            let dim = lazy_flat.dim;
            let n = lazy_flat.num_vectors;
            let quant = lazy_flat.quantization;
            let batch_len =
                bounded_vector_score_batch(lazy_flat.vector_byte_size(), DENSE_SCORE_BATCH);
            let mut collector = FlatDocumentCollector::new(fetch_k.min(n), combiner);
            let mut scores = vec![0f32; batch_len];

            for batch_start in (0..n).step_by(batch_len) {
                let batch_count = batch_len.min(n - batch_start);
                let batch_bytes = lazy_flat
                    .read_vectors_batch_sync(batch_start, batch_count)
                    .map_err(crate::Error::Io)?;
                let raw = batch_bytes.as_slice();

                Self::score_quantized_batch(
                    query,
                    raw,
                    quant,
                    dim,
                    &mut scores[..batch_count],
                    params.unit_norm,
                )?;

                for (i, &score) in scores.iter().enumerate().take(batch_count) {
                    let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                    collector.push(doc_id, ordinal, score);
                }
            }

            return Ok(collector.into_results());
        } else {
            return Ok(Vec::new());
        };

        // Rerank ANN candidates using raw vectors (sync)
        if ann_index.is_some()
            && !results.is_empty()
            && let Some(lazy_flat) = lazy_flat
        {
            let dim = lazy_flat.dim;
            let quant = lazy_flat.quantization;
            let vbs = lazy_flat.vector_byte_size();

            let (expanded, mut resolved) = expand_ann_candidate_documents(&results, lazy_flat)?;
            results = expanded;

            if !resolved.is_empty() {
                resolved.sort_unstable_by_key(|&(_, flat_idx)| flat_idx);
                let batch_len = bounded_vector_score_batch(vbs, DENSE_SCORE_BATCH);
                let max_batch = batch_len.min(resolved.len());
                let max_raw_len = max_batch
                    .checked_mul(vbs)
                    .ok_or_else(|| Error::Query("dense rerank buffer size overflow".into()))?;
                let mut raw_buf = vec![0u8; max_raw_len];
                let mut scores = vec![0f32; max_batch];

                for chunk in resolved.chunks(batch_len) {
                    let raw_len = chunk
                        .len()
                        .checked_mul(vbs)
                        .ok_or_else(|| Error::Query("dense rerank buffer size overflow".into()))?;
                    let raw = &mut raw_buf[..raw_len];
                    for (buf_idx, &(_, flat_idx)) in chunk.iter().enumerate() {
                        lazy_flat
                            .read_vector_raw_into_sync(
                                flat_idx,
                                &mut raw[buf_idx * vbs..(buf_idx + 1) * vbs],
                            )
                            .map_err(crate::Error::Io)?;
                    }

                    Self::score_quantized_batch(
                        query,
                        raw,
                        quant,
                        dim,
                        &mut scores[..chunk.len()],
                        params.unit_norm,
                    )?;

                    for (buf_idx, &(ri, _)) in chunk.iter().enumerate() {
                        results[ri].2 = scores[buf_idx];
                    }
                }
            }
        }

        Ok(combine_grouped_ordinal_results(results, combiner, k))
    }

    /// Synchronous binary dense vector search (mmap/RAM only).
    ///
    /// Mirrors [`Self::search_binary_dense_vector`] for the rayon-parallel
    /// sync scorer path used by multi-threaded runtimes.
    #[cfg(feature = "sync")]
    pub fn search_binary_dense_vector_sync(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        let schema_dim = self.validate_binary_search_request(field, query)?;
        combiner.validate().map_err(Error::Query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let t0 = crate::observe::Timer::start();
        // Binary IVF candidate union followed by complete document scoring.
        if let Some(VectorIndex::BinaryIvf(lazy)) = self.vector_indexes.get(&field.0)
            && let Some(ivf) = lazy.get()
        {
            if ivf.config.dim_bits != schema_dim {
                return Err(Error::Corruption(format!(
                    "binary IVF field {} has schema dimension {} but index dimension {}",
                    field.0, schema_dim, ivf.config.dim_bits
                )));
            }
            let flat = self.flat_vectors.get(&field.0).ok_or_else(|| {
                Error::Corruption(format!(
                    "binary IVF field {} is missing flat vector storage",
                    field.0
                ))
            })?;
            if flat.dim != schema_dim
                || flat.quantization != crate::dsl::DenseVectorQuantization::Binary
            {
                return Err(Error::Corruption(format!(
                    "binary IVF field {} has inconsistent flat vector metadata",
                    field.0
                )));
            }
            let nprobe = self.binary_ivf_nprobe(field);
            let initial_fetch =
                ann_ordinal_fetch_k(k, flat.num_vectors, flat.num_docs_with_vectors());
            // Single scan: BinaryIvfIndex::search scans all probed clusters
            // regardless of candidate_k, so progressive doubling would repeat
            // the identical scan on every retry.
            let ann_results = single_scan_ann_search(
                k.min(flat.num_docs_with_vectors()),
                initial_fetch,
                ivf.len().min(flat.num_vectors),
                |candidate_k| ivf.search(query, candidate_k, nprobe),
            )?;
            let results =
                exact_score_binary_candidate_documents_sync(&ann_results, flat, query, schema_dim)?;
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "binary_ivf",
                t0.secs(),
                results.len(),
            );
            return Ok(combine_grouped_ordinal_results(results, combiner, k));
        }

        let lazy_flat = match self.flat_vectors.get(&field.0) {
            Some(f) => f,
            None => return Ok(Vec::new()),
        };

        let dim_bits = lazy_flat.dim;
        let byte_len = lazy_flat.vector_byte_size();
        let n = lazy_flat.num_vectors;

        if dim_bits != schema_dim {
            return Err(Error::Corruption(format!(
                "binary vector field {} has schema dimension {} but flat storage dimension {}",
                field.0, schema_dim, dim_bits
            )));
        }

        if byte_len != query.len() {
            return Err(Error::Schema(format!(
                "Binary query vector byte length {} != field byte length {}",
                query.len(),
                byte_len
            )));
        }

        let batch_len = bounded_vector_score_batch(byte_len, BINARY_SCORE_BATCH);
        let mut collector = FlatDocumentCollector::new(k, combiner);
        let mut scores = vec![0f32; batch_len];

        for batch_start in (0..n).step_by(batch_len) {
            let batch_count = batch_len.min(n - batch_start);
            let batch_bytes = lazy_flat
                .read_vectors_batch_sync(batch_start, batch_count)
                .map_err(crate::Error::Io)?;
            let raw = batch_bytes.as_slice();

            crate::structures::simd::batch_hamming_scores(
                query,
                raw,
                byte_len,
                dim_bits,
                &mut scores[..batch_count],
            );

            for (i, &score) in scores.iter().enumerate().take(batch_count) {
                let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                collector.push(doc_id, ordinal, score);
            }
        }

        let results = collector.into_results();

        crate::observe::dense_l1(
            self.schema.index_label(),
            self.schema.get_field_name(field).unwrap_or("?"),
            "binary_flat",
            t0.secs(),
            results.len(),
        );
        Ok(results)
    }
}

#[cfg(test)]
mod dense_search_safety_tests {
    use super::*;

    #[test]
    fn dense_fetch_count_rejects_non_finite_and_unbounded_factors() {
        for factor in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.5,
            MAX_DENSE_RERANK_FACTOR + 1.0,
        ] {
            assert!(
                checked_dense_fetch_k(10, factor).is_err(),
                "factor={factor}"
            );
        }
    }

    #[test]
    fn flat_document_collector_does_not_let_one_multivalue_doc_crowd_out_others() {
        let mut collector = FlatDocumentCollector::new(2, crate::query::MultiValueCombiner::Max);
        collector.push(1, 0, 1.0);
        collector.push(1, 1, 0.9);
        collector.push(2, 0, 0.8);

        let results = collector.into_results();
        assert_eq!(
            results
                .iter()
                .map(|result| result.doc_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(results[0].ordinals.len(), 2);
    }

    #[test]
    fn flat_document_collector_evicts_by_score_then_doc_id() {
        let mut collector = FlatDocumentCollector::new(2, crate::query::MultiValueCombiner::Max);
        collector.push(1, 0, 0.5);
        collector.push(3, 0, 0.8);
        collector.push(2, 0, 0.9);
        let results = collector.into_results();
        assert_eq!(
            results
                .iter()
                .map(|result| result.doc_id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );

        let mut tied = FlatDocumentCollector::new(1, crate::query::MultiValueCombiner::Max);
        tied.push(2, 0, 1.0);
        tied.push(1, 0, 1.0);
        let results = tied.into_results();
        assert_eq!(results[0].doc_id, 1);
    }

    #[test]
    fn dense_fetch_count_rounds_up_and_detects_overflow() {
        assert_eq!(checked_dense_fetch_k(3, 1.5).unwrap(), 5);
        assert_eq!(checked_dense_fetch_k(50_000, 3.0).unwrap(), 150_000);
        assert!(checked_dense_fetch_k(50_000, 32.0).is_err());
        assert!(checked_dense_fetch_k(usize::MAX, 2.0).is_err());
    }

    #[test]
    fn ann_fetch_depth_accounts_for_multivalue_density_and_stays_bounded() {
        assert_eq!(ann_ordinal_fetch_k(100, 1_000, 100), 1_000);
        assert_eq!(
            ann_ordinal_fetch_k(100_000, usize::MAX, 1),
            MAX_DENSE_CANDIDATES_PER_SEGMENT
        );
        assert_eq!(ann_ordinal_fetch_k(100, 0, 10), 0);
    }

    #[test]
    fn ann_search_deepens_until_skewed_results_contain_enough_documents() {
        let ranked = [
            (1, 0, 1.0),
            (1, 1, 0.99),
            (1, 2, 0.98),
            (1, 3, 0.97),
            (1, 4, 0.96),
            (1, 5, 0.95),
            (2, 0, 0.9),
            (3, 0, 0.8),
        ];
        let mut fetches = Vec::new();
        let results = progressive_ann_search(2, 2, ranked.len(), |fetch| {
            fetches.push(fetch);
            ranked.iter().copied().take(fetch).collect()
        })
        .unwrap();

        assert_eq!(fetches, vec![2, 4, 8]);
        assert!(results.iter().any(|&(doc_id, _, _)| doc_id == 2));
    }

    #[test]
    fn ann_search_stops_when_probed_population_is_exhausted() {
        let ranked = [(1, 0, 1.0), (1, 1, 0.9)];
        let mut calls = 0;
        let results = progressive_ann_search(3, 4, 100, |fetch| {
            calls += 1;
            ranked.iter().copied().take(fetch).collect()
        })
        .unwrap();

        assert_eq!(calls, 1);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn file_ranges_reject_overflow_and_truncation() {
        assert_eq!(checked_file_range(4, 3, 7, "test").unwrap(), 4..7);
        assert!(checked_file_range(u64::MAX, 1, u64::MAX, "test").is_err());
        assert!(checked_file_range(5, 3, 7, "test").is_err());
    }

    /// The single-scan replay must return exactly what the doubling retry
    /// loop returns for any (target, initial fetch, population) combination
    /// when `search(f)` yields the top-`f` prefix of one fixed ranking —
    /// while invoking the search closure exactly once.
    #[test]
    fn single_scan_ann_search_matches_progressive_with_one_search_call() {
        // Skewed ranking: doc 1 owns the best six vectors, then a sparse tail.
        let ranked: Vec<RawVectorCandidate> = vec![
            (1, 0, 1.0),
            (1, 1, 0.99),
            (1, 2, 0.98),
            (1, 3, 0.97),
            (1, 4, 0.96),
            (1, 5, 0.95),
            (2, 0, 0.9),
            (3, 0, 0.8),
            (2, 1, 0.7),
            (4, 0, 0.6),
            (5, 0, 0.5),
        ];

        for target_docs in 0..=6 {
            for initial_fetch in [0, 1, 2, 3, 8, 16, 64] {
                for max_vectors in [0, 1, 2, 8, ranked.len(), 100] {
                    let progressive =
                        progressive_ann_search(target_docs, initial_fetch, max_vectors, |fetch| {
                            ranked.iter().copied().take(fetch).collect()
                        });
                    let mut calls = 0;
                    let single =
                        single_scan_ann_search(target_docs, initial_fetch, max_vectors, |fetch| {
                            calls += 1;
                            ranked.iter().copied().take(fetch).collect()
                        });
                    let case =
                        format!("target={target_docs} initial={initial_fetch} max={max_vectors}");
                    match (progressive, single) {
                        (Ok(a), Ok(b)) => assert_eq!(a, b, "{case}"),
                        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{case}"),
                        (a, b) => panic!("verdict mismatch for {case}: {a:?} vs {b:?}"),
                    }
                    assert!(calls <= 1, "{case}: search must run at most once");
                }
            }
        }
    }

    /// Probe-returned (doc, ordinal) pairs keep their probe scores and are
    /// pruned from the flat-read list; only the candidate documents'
    /// remaining ordinals still need reads.
    #[test]
    fn probe_scored_candidates_skip_flat_rereads() {
        let ann_results: Vec<RawVectorCandidate> = vec![(7, 0, 0.75), (9, 2, 0.5)];
        // Expansion of docs 7 and 9 to all their ordinals (scores unfilled).
        let mut expanded: Vec<RawVectorCandidate> =
            vec![(7, 0, 0.0), (7, 1, 0.0), (9, 0, 0.0), (9, 2, 0.0)];
        let resolved: Vec<ResolvedVectorCandidate> = vec![(0, 10), (1, 11), (2, 20), (3, 22)];

        let remaining = fill_probe_scores_and_prune_resolved(&ann_results, &mut expanded, resolved);

        // Probe-scored pairs (7,0) and (9,2) are filled and pruned.
        assert_eq!(expanded[0].2, 0.75);
        assert_eq!(expanded[3].2, 0.5);
        // Unseen ordinals (7,1) and (9,0) still need flat reads.
        assert_eq!(remaining, vec![(1, 11), (2, 20)]);
        assert_eq!(expanded[1].2, 0.0);
        assert_eq!(expanded[2].2, 0.0);
    }
}
