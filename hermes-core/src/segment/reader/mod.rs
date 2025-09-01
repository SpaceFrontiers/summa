//! Async segment reader with lazy loading

pub(crate) mod bmp;
pub(crate) mod loader;
mod types;

pub use bmp::{BmpDimStats, BmpIndex};
#[cfg(feature = "native")]
pub(crate) use types::DimRawData;
pub use types::{SparseIndex, VectorIndex, VectorOrdinals, VectorSearchResult};

/// Bound vocabulary and posting expansion before a prefix query starts loading
/// posting payloads. These are per-segment limits; callers should use exact-term
/// or a more selective prefix when they are exceeded.
const MAX_PREFIX_TERMS: usize = 1_024;
const MAX_PREFIX_POSTINGS: u64 = 5_000_000;
/// Hard guard for explicitly requested dense candidate documents. Values of
/// those documents are exact-scored through bounded streaming batches, so a
/// valid multi-valued document is not rejected merely for owning many values.
const MAX_DENSE_CANDIDATES_PER_SEGMENT: usize = 20_000;
/// Preferred vector count; wide vectors reduce it to stay under the byte cap.
const DENSE_SCORE_BATCH: usize = 4_096;
const BINARY_SCORE_BATCH: usize = 8_192;
const MAX_VECTOR_SCORE_BATCH_BYTES: usize = 8 * 1024 * 1024;

/// Runtime memory accounting for a single segment.
///
/// Heap, file-backed address space, and pinned residency are deliberately
/// separate: file-backed bytes are not resident merely because they are
/// mapped, and pinned bytes are a subset rather than an additive allocation.
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
    /// Sparse-vector lookup structures retained on the heap.
    pub sparse_heap_bytes: usize,
    /// Dense-vector ANN lookup structures retained on the heap.
    pub dense_heap_bytes: usize,
    /// File-backed term-dictionary bloom-filter bytes.
    pub term_bloom_file_bytes: u64,
    /// Logical `.sparse` file bytes retained by the reader.
    pub sparse_file_backed_bytes: u64,
    /// Logical `.vectors` file bytes retained by the reader.
    pub dense_file_backed_bytes: u64,
    /// Hot metadata bytes actually pinned (mlock/heap-copy) at open
    pub pinned_metadata_bytes: u64,
    /// Hot metadata bytes eligible for pinning (gap vs pinned = budget
    /// exhausted or mlock failures — operator-visible)
    pub pin_intended_bytes: u64,
    /// Sparse-vector subset of `pinned_metadata_bytes`.
    pub sparse_pinned_metadata_bytes: u64,
    /// Sparse-vector bytes eligible for pinning.
    pub sparse_pin_intended_bytes: u64,
    /// Dense-vector subset of `pinned_metadata_bytes`.
    pub dense_pinned_metadata_bytes: u64,
    /// Dense-vector bytes eligible for pinning.
    pub dense_pin_intended_bytes: u64,
}

impl SegmentMemoryStats {
    /// Total estimated heap retained by this segment reader.
    pub fn estimated_heap_bytes(&self) -> usize {
        self.term_dict_cache_bytes
            + self.store_cache_bytes
            + self.sparse_heap_bytes
            + self.dense_heap_bytes
    }

    /// Total logical bytes in the explicitly accounted file-backed sections.
    ///
    /// This is mapped address space for `MmapDirectory`, not resident memory.
    pub fn file_backed_bytes(&self) -> u64 {
        self.term_bloom_file_bytes
            .saturating_add(self.sparse_file_backed_bytes)
            .saturating_add(self.dense_file_backed_bytes)
    }
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::vector_data::LazyFlatVectorData;
use crate::directories::{Directory, FileHandle};
use crate::dsl::{DenseVectorQuantization, Document, Field, Schema};
use crate::observe::DenseAnnScanStats;
use crate::query::{MAX_DENSE_NPROBE, MAX_DENSE_RERANK_FACTOR};
use crate::structures::{
    AsyncSSTableReader, BlockPostingList, CoarseCentroids, SSTableStats, TermInfo,
};
use crate::{DocId, Error, Result};

use super::store::{AsyncStoreReader, RawStoreBlock};
use super::types::{SegmentFiles, SegmentId, SegmentMeta};

/// Combine per-ordinal (doc_id, ordinal, score) triples into VectorSearchResults,
/// applying the multi-value combiner, sorting by score desc, and truncating to `limit`.
///
/// Fast path: when all ordinals are 0 (single-valued field), skips grouping
/// entirely and just sorts + truncates the raw results; each result keeps its
/// single ordinal inline (no per-result allocation).
///
/// Slow path: a stable sort by doc id followed by one run-grouping pass — no
/// hash map and no per-document `Vec`; the encounter order of a document's
/// ordinals (what the combiner and the returned `ordinals` see) is preserved.
pub(crate) fn combine_ordinal_results(
    raw: impl IntoIterator<Item = (u32, u16, f32)>,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Vec<VectorSearchResult> {
    let mut collected: Vec<(u32, u16, f32)> = raw.into_iter().collect();

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

    // Fast path: all ordinals are 0 → no grouping needed
    let all_single = collected.iter().all(|&(_, ord, _)| ord == 0);
    if all_single {
        let mut results: Vec<VectorSearchResult> = collected
            .into_iter()
            .map(|(doc_id, _, score)| VectorSearchResult::single(doc_id, score))
            .collect();
        results.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results.truncate(limit);
        return results;
    }

    // Slow path: multi-valued field — group by doc_id, apply combiner. The
    // sort is stable so a document's ordinals keep their encounter order.
    collected.sort_by_key(|&(doc_id, _, _)| doc_id);
    let mut results: Vec<VectorSearchResult> = Vec::new();
    let mut index = 0;
    while index < collected.len() {
        let doc_id = collected[index].0;
        let mut ordinals = super::VectorOrdinals::new();
        while index < collected.len() && collected[index].0 == doc_id {
            let (_, ordinal, score) = collected[index];
            ordinals.push((ordinal as u32, score));
            index += 1;
        }
        let combined_score = combiner.combine(&ordinals);
        results.push(VectorSearchResult::with_ordinals(
            doc_id as DocId,
            combined_score,
            ordinals,
        ));
    }
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
    current_ordinals: super::VectorOrdinals,
}

impl FlatDocumentCollector {
    fn new(limit: usize, combiner: crate::query::MultiValueCombiner) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(limit.min(8 * 1024)),
            limit,
            combiner,
            current_doc: None,
            current_ordinals: super::VectorOrdinals::new(),
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
                HeapVectorResult(VectorSearchResult::with_ordinals(
                    doc_id,
                    score,
                    super::VectorOrdinals::new(),
                ))
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
        let entry = HeapVectorResult(VectorSearchResult::with_ordinals(doc_id, score, ordinals));
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

/// Query-derived state shared by every native-precision scoring batch in one
/// flat scan or exact rerank operation.
///
/// Computing the query norm is O(dim), and f16 scoring additionally quantizes
/// the query. Keeping both here avoids repeating that work for every bounded
/// vector batch.
struct PreparedDenseScoreQuery<'a> {
    query: &'a [f32],
    query_f16: Vec<u16>,
    inv_norm_q: f32,
    quantization: DenseVectorQuantization,
    dim: usize,
    unit_norm: bool,
}

impl<'a> PreparedDenseScoreQuery<'a> {
    fn new(
        query: &'a [f32],
        quantization: DenseVectorQuantization,
        dim: usize,
        unit_norm: bool,
    ) -> Result<Self> {
        use crate::structures::simd;

        if query.len() != dim {
            return Err(Error::Query(format!(
                "dense SIMD query dimension {} does not match vector dimension {dim}",
                query.len()
            )));
        }
        if quantization == DenseVectorQuantization::Binary {
            return Err(Error::InvalidFieldType {
                expected: "non-binary dense vector".to_string(),
                got: "binary dense vector".to_string(),
            });
        }

        let norm_q_sq = simd::dot_product_f32(query, query, dim);
        let inv_norm_q = if norm_q_sq < f32::EPSILON {
            0.0
        } else {
            simd::fast_inv_sqrt(norm_q_sq)
        };
        let query_f16 = if quantization == DenseVectorQuantization::F16 {
            query.iter().map(|&value| simd::f32_to_f16(value)).collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            query,
            query_f16,
            inv_norm_q,
            quantization,
            dim,
            unit_norm,
        })
    }

    fn score_batch(&self, raw: &[u8], scores: &mut [f32]) -> Result<()> {
        use crate::structures::simd;

        let element_size = match self.quantization {
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
            .checked_mul(self.dim)
            .and_then(|elements| elements.checked_mul(element_size))
            .ok_or_else(|| Error::Corruption("dense vector batch byte length overflow".into()))?;
        if raw.len() < required_bytes {
            return Err(Error::Corruption(format!(
                "dense vector batch is truncated: need {required_bytes} bytes, got {}",
                raw.len()
            )));
        }
        if self.quantization == DenseVectorQuantization::F16
            && required_bytes > 0
            && !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>())
        {
            return Err(Error::Corruption(
                "f16 vector data is not 2-byte aligned".to_string(),
            ));
        }
        if self.quantization == DenseVectorQuantization::F32
            && !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        {
            return Err(Error::Corruption(
                "f32 vector data is not 4-byte aligned".to_string(),
            ));
        }

        // The legacy batch scorers leave the destination untouched for empty
        // dimensions or batches. Retain that boundary behavior before calling
        // the precomputed kernels.
        if self.dim == 0 || scores.is_empty() {
            return Ok(());
        }

        match (self.quantization, self.unit_norm) {
            (DenseVectorQuantization::F32, false) => {
                let num_floats = scores.len() * self.dim;
                let vectors: &[f32] =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
                simd::batch_cosine_scores_precomp(
                    self.query,
                    vectors,
                    self.dim,
                    scores,
                    self.inv_norm_q,
                );
            }
            (DenseVectorQuantization::F32, true) => {
                let num_floats = scores.len() * self.dim;
                let vectors: &[f32] =
                    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats) };
                simd::batch_dot_scores_precomp(
                    self.query,
                    vectors,
                    self.dim,
                    scores,
                    self.inv_norm_q,
                );
            }
            (DenseVectorQuantization::F16, false) => {
                simd::batch_cosine_scores_f16_precomp(
                    &self.query_f16,
                    raw,
                    self.dim,
                    scores,
                    self.inv_norm_q,
                );
            }
            (DenseVectorQuantization::F16, true) => {
                simd::batch_dot_scores_f16_precomp(
                    &self.query_f16,
                    raw,
                    self.dim,
                    scores,
                    self.inv_norm_q,
                );
            }
            (DenseVectorQuantization::UInt8, false) => {
                simd::batch_cosine_scores_u8_precomp(
                    self.query,
                    raw,
                    self.dim,
                    scores,
                    self.inv_norm_q,
                );
            }
            (DenseVectorQuantization::UInt8, true) => {
                simd::batch_dot_scores_u8_precomp(
                    self.query,
                    raw,
                    self.dim,
                    scores,
                    self.inv_norm_q,
                );
            }
            (DenseVectorQuantization::Binary, _) => unreachable!("validated during preparation"),
        }
        Ok(())
    }
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

/// Binary queries do not expose a configurable rerank factor. Use the shared
/// query-level oversubscription policy while retaining the same hard
/// per-segment candidate bound as float-vector reranking. Reject a result
/// window larger than that bound instead of silently returning fewer than
/// requested; candidate oversampling itself may safely clamp at the bound.
#[inline]
fn checked_binary_combined_fetch_k(k: usize) -> Result<usize> {
    if k > MAX_DENSE_CANDIDATES_PER_SEGMENT {
        return Err(Error::Query(format!(
            "binary dense result count exceeds the per-segment maximum of \
             {MAX_DENSE_CANDIDATES_PER_SEGMENT}: k={k}"
        )));
    }
    Ok(crate::query::max_candidate_limit(k).min(MAX_DENSE_CANDIDATES_PER_SEGMENT))
}

#[inline]
fn bounded_vector_score_batch(vector_byte_size: usize, preferred: usize) -> usize {
    preferred.min((MAX_VECTOR_SCORE_BATCH_BYTES / vector_byte_size.max(1)).max(1))
}

#[inline]
fn bounded_rerank_batch(vector_byte_size: usize, preferred: usize, vector_count: usize) -> usize {
    bounded_vector_score_batch(vector_byte_size, preferred).min(vector_count.max(1))
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
type CandidateVectorRef = (DocId, u16, usize); // (doc ID, ordinal, flat-vector index)

#[derive(Clone, Copy)]
struct CandidateDocumentRange {
    doc_id: DocId,
    start: usize,
    end: usize,
}

struct AnnCandidateDocuments {
    ranges: Vec<CandidateDocumentRange>,
    vector_count: usize,
}

/// Resolve the document union returned by ANN to compact flat-vector ranges.
///
/// The document union is bounded by ANN document top-k, while the number of
/// values those documents own is intentionally not capped. A valid
/// multi-valued document may have many ordinals;
/// materializing one result and one flat-index entry per ordinal used to turn
/// that into a spurious query error at 20,000 vectors. Callers stream these
/// ranges through a fixed-size score buffer instead.
fn ann_candidate_document_ranges(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
) -> Result<AnnCandidateDocuments> {
    ann_candidate_document_ranges_from_ids(ann_results.iter().map(|candidate| candidate.0), flat)
}

fn ann_candidate_document_ranges_from_ids(
    doc_ids: impl IntoIterator<Item = DocId>,
    flat: &LazyFlatVectorData,
) -> Result<AnnCandidateDocuments> {
    let mut candidate_docs: Vec<DocId> = doc_ids.into_iter().collect();
    candidate_docs.sort_unstable();
    candidate_docs.dedup();

    let mut ranges = Vec::with_capacity(candidate_docs.len());
    let mut vector_count = 0usize;
    for doc_id in candidate_docs {
        let (start, count) = flat.flat_indexes_for_doc_range(doc_id);
        if count == 0 {
            return Err(Error::Corruption(format!(
                "ANN candidate document {doc_id} is missing from flat vector storage"
            )));
        }
        vector_count = vector_count
            .checked_add(count)
            .ok_or_else(|| Error::Query("ANN candidate vector expansion overflow".to_string()))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| Error::Corruption("flat vector range overflow".to_string()))?;
        if end > flat.num_vectors {
            return Err(Error::Corruption(format!(
                "flat vector range {start}..{end} for document {doc_id} exceeds {} vectors",
                flat.num_vectors
            )));
        }
        ranges.push(CandidateDocumentRange { doc_id, start, end });
    }
    Ok(AnnCandidateDocuments {
        ranges,
        vector_count,
    })
}

/// Validate the no-rerank binary IVF fast path against exact flat metadata.
///
/// Binary IVF stores the original packed codes, so a single-valued field does
/// not need vector-data I/O to recompute scores. It still needs the same
/// ANN/flat consistency checks the rerank path provided: every candidate must
/// name the field's sole stored ordinal. Deduplicate by document as well so a
/// malformed ANN payload cannot surface the same document more than once.
fn validate_binary_single_value_ann_results(
    ann_results: Vec<RawVectorCandidate>,
    flat: &LazyFlatVectorData,
) -> Result<Vec<RawVectorCandidate>> {
    let mut seen_docs = FxHashSet::default();
    let mut validated = Vec::with_capacity(ann_results.len());
    for (doc_id, ordinal, score) in ann_results {
        let (start, count) = flat.flat_indexes_for_doc_range(doc_id);
        if count == 0 {
            return Err(Error::Corruption(format!(
                "ANN candidate document {doc_id} is missing from flat vector storage"
            )));
        }
        if count != 1 {
            return Err(Error::Corruption(format!(
                "binary ANN single-valued candidate document {doc_id} has {count} flat vectors"
            )));
        }
        let (stored_doc_id, stored_ordinal) = flat.get_doc_id(start);
        if stored_doc_id != doc_id {
            return Err(Error::Corruption(format!(
                "flat vector doc map is not contiguous for document {doc_id}"
            )));
        }
        if stored_ordinal != ordinal {
            return Err(Error::Corruption(format!(
                "binary ANN candidate document {doc_id} ordinal {ordinal} is missing from flat vector storage"
            )));
        }
        if seen_docs.insert(doc_id) {
            validated.push((doc_id, ordinal, score));
        }
    }
    Ok(validated)
}

struct CandidateVectorCursor<'a> {
    ranges: &'a [CandidateDocumentRange],
    range_index: usize,
    flat_index: usize,
}

impl<'a> CandidateVectorCursor<'a> {
    fn new(ranges: &'a [CandidateDocumentRange]) -> Self {
        Self {
            ranges,
            range_index: 0,
            flat_index: ranges.first().map_or(0, |range| range.start),
        }
    }

    /// Fill `batch` in `(doc_id, ordinal)` order. The cursor validates the
    /// contiguity promise made by the flat doc map while it streams, avoiding
    /// an O(all candidate ordinals) validation allocation.
    fn fill_batch(
        &mut self,
        flat: &LazyFlatVectorData,
        batch: &mut Vec<CandidateVectorRef>,
        limit: usize,
    ) -> Result<bool> {
        batch.clear();
        while batch.len() < limit && self.range_index < self.ranges.len() {
            let range = self.ranges[self.range_index];
            if self.flat_index == range.end {
                self.range_index += 1;
                if let Some(next) = self.ranges.get(self.range_index) {
                    self.flat_index = next.start;
                }
                continue;
            }
            let (stored_doc_id, ordinal) = flat.get_doc_id(self.flat_index);
            if stored_doc_id != range.doc_id {
                return Err(Error::Corruption(format!(
                    "flat vector doc map is not contiguous for document {}",
                    range.doc_id
                )));
            }
            batch.push((range.doc_id, ordinal, self.flat_index));
            self.flat_index += 1;
        }
        Ok(!batch.is_empty())
    }
}

#[derive(Clone, Copy)]
struct VectorReadRun {
    buffer_start: usize,
    flat_start: usize,
    count: usize,
}

/// Coalesce an ordered set of selected flat indexes into contiguous reads.
/// Multi-valued document bodies are stored consecutively, so this turns the
/// common case from one range lookup per value into one lookup per bounded
/// run while retaining a packed score buffer.
fn plan_vector_read_runs(indexes: &[usize], runs: &mut Vec<VectorReadRun>) -> Result<()> {
    runs.clear();
    for (buffer_index, &flat_index) in indexes.iter().enumerate() {
        if let Some(run) = runs.last_mut()
            && run
                .flat_start
                .checked_add(run.count)
                .is_some_and(|next| next == flat_index)
        {
            run.count += 1;
            continue;
        }
        if buffer_index > 0 && flat_index <= indexes[buffer_index - 1] {
            return Err(Error::Corruption(
                "candidate flat-vector indexes are not strictly ordered".into(),
            ));
        }
        runs.push(VectorReadRun {
            buffer_start: buffer_index,
            flat_start: flat_index,
            count: 1,
        });
    }
    Ok(())
}

/// Plan contiguous raw-vector reads and initiate page-in before either the
/// synchronous or asynchronous reader starts copying. Keeping prefetch here
/// prevents the two execution paths from drifting.
fn prepare_vector_read_runs(
    flat: &LazyFlatVectorData,
    indexes: &[usize],
    runs: &mut Vec<VectorReadRun>,
) -> Result<()> {
    plan_vector_read_runs(indexes, runs)?;
    #[cfg(feature = "native")]
    flat.prefetch_vectors(indexes.iter().copied());
    #[cfg(not(feature = "native"))]
    let _ = flat;
    Ok(())
}

async fn read_vector_runs(
    flat: &LazyFlatVectorData,
    indexes: &[usize],
    runs: &mut Vec<VectorReadRun>,
    output: &mut [u8],
) -> Result<()> {
    prepare_vector_read_runs(flat, indexes, runs)?;
    let vector_byte_size = flat.vector_byte_size();
    for run in runs {
        let bytes = flat
            .read_vectors_batch(run.flat_start, run.count)
            .await
            .map_err(Error::Io)?;
        let start = run
            .buffer_start
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("dense rerank buffer offset overflow".into()))?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::Query("dense rerank buffer range overflow".into()))?;
        let destination = output
            .get_mut(start..end)
            .ok_or_else(|| Error::Corruption("dense rerank buffer is too short".into()))?;
        destination.copy_from_slice(bytes.as_slice());
    }
    Ok(())
}

#[cfg(feature = "sync")]
fn read_vector_runs_sync(
    flat: &LazyFlatVectorData,
    indexes: &[usize],
    runs: &mut Vec<VectorReadRun>,
    output: &mut [u8],
) -> Result<()> {
    prepare_vector_read_runs(flat, indexes, runs)?;
    let vector_byte_size = flat.vector_byte_size();
    for run in runs {
        let bytes = flat
            .read_vectors_batch_sync(run.flat_start, run.count)
            .map_err(Error::Io)?;
        let start = run
            .buffer_start
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("dense rerank buffer offset overflow".into()))?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::Query("dense rerank buffer range overflow".into()))?;
        let destination = output
            .get_mut(start..end)
            .ok_or_else(|| Error::Corruption("dense rerank buffer is too short".into()))?;
        destination.copy_from_slice(bytes.as_slice());
    }
    Ok(())
}

#[derive(Default)]
struct DenseRerankStats {
    vector_count: usize,
    resolve_elapsed: std::time::Duration,
    read_elapsed: std::time::Duration,
    score_elapsed: std::time::Duration,
}

/// Per-thread reusable buffers for the dense rerank and flat-scan paths
/// (model: `query::bmp::BmpScratch`). Every batch read fully overwrites the
/// bytes it scores, so sizing `raw` once per thread replaces a zeroed
/// allocation per rerank call; the index vectors likewise keep their
/// capacity across queries. Growth is bounded by the batch byte cap.
#[derive(Default)]
struct DenseScratch {
    raw: Vec<u8>,
    scores: Vec<f32>,
    batch_scores: Vec<f32>,
    batch: Vec<CandidateVectorRef>,
    flat_indexes: Vec<usize>,
    read_runs: Vec<VectorReadRun>,
    unresolved: Vec<(usize, usize)>,
}

thread_local! {
    static DENSE_SCRATCH: std::cell::RefCell<Option<Box<DenseScratch>>> =
        const { std::cell::RefCell::new(None) };
}

impl DenseScratch {
    /// Borrow this thread's scratch for one operation. The guard hands it
    /// back on drop — also after an `.await` that resumes on another thread,
    /// where the scratch simply migrates to the resuming thread.
    fn take() -> DenseScratchGuard {
        let scratch = DENSE_SCRATCH
            .with(|slot| slot.borrow_mut().take())
            .unwrap_or_default();
        DenseScratchGuard(Some(scratch))
    }

    /// Grow the byte and score buffers to the batch shape (never shrinks)
    /// and clear the index vectors.
    fn prepare(&mut self, raw_bytes: usize, batch_len: usize) {
        if self.raw.len() < raw_bytes {
            self.raw.resize(raw_bytes, 0);
        }
        if self.scores.len() < batch_len {
            self.scores.resize(batch_len, 0.0);
        }
        if self.batch_scores.len() < batch_len {
            self.batch_scores.resize(batch_len, 0.0);
        }
        self.batch.clear();
        self.flat_indexes.clear();
        self.read_runs.clear();
        self.unresolved.clear();
    }
}

struct DenseScratchGuard(Option<Box<DenseScratch>>);

impl std::ops::Deref for DenseScratchGuard {
    type Target = DenseScratch;
    fn deref(&self) -> &DenseScratch {
        self.0
            .as_deref()
            .expect("dense scratch is present until drop")
    }
}

impl std::ops::DerefMut for DenseScratchGuard {
    fn deref_mut(&mut self) -> &mut DenseScratch {
        self.0
            .as_deref_mut()
            .expect("dense scratch is present until drop")
    }
}

impl Drop for DenseScratchGuard {
    fn drop(&mut self) {
        if let Some(scratch) = self.0.take() {
            // A thread already tearing down its TLS simply frees the buffers.
            let _ = DENSE_SCRATCH.try_with(|slot| *slot.borrow_mut() = Some(scratch));
        }
    }
}

async fn exact_score_dense_candidate_documents(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[f32],
    unit_norm: bool,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<(Vec<VectorSearchResult>, DenseRerankStats)> {
    let resolve_started = std::time::Instant::now();
    let documents = ann_candidate_document_ranges(ann_results, flat)?;
    let mut stats = DenseRerankStats {
        vector_count: documents.vector_count,
        resolve_elapsed: resolve_started.elapsed(),
        ..Default::default()
    };
    let vector_byte_size = flat.vector_byte_size();
    let batch_len =
        bounded_rerank_batch(vector_byte_size, DENSE_SCORE_BATCH, documents.vector_count);
    let raw_capacity = batch_len
        .checked_mul(vector_byte_size)
        .ok_or_else(|| Error::Query("dense rerank buffer size overflow".to_string()))?;
    let prepared_query =
        PreparedDenseScoreQuery::new(query, flat.quantization, flat.dim, unit_norm)?;
    let mut scratch = DenseScratch::take();
    scratch.prepare(raw_capacity, batch_len);
    let DenseScratch {
        raw,
        scores,
        batch,
        flat_indexes,
        read_runs,
        ..
    } = &mut *scratch;
    let mut cursor = CandidateVectorCursor::new(&documents.ranges);
    let mut collector = FlatDocumentCollector::new(limit, combiner);
    let mut scored = 0usize;

    while cursor.fill_batch(flat, batch, batch_len)? {
        flat_indexes.clear();
        flat_indexes.extend(batch.iter().map(|&(_, _, flat_index)| flat_index));
        let raw_len = batch
            .len()
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("dense rerank buffer size overflow".to_string()))?;
        let raw = &mut raw[..raw_len];

        let read_started = std::time::Instant::now();
        read_vector_runs(flat, flat_indexes, read_runs, raw).await?;
        stats.read_elapsed += read_started.elapsed();

        let score_started = std::time::Instant::now();
        prepared_query.score_batch(raw, &mut scores[..batch.len()])?;
        stats.score_elapsed += score_started.elapsed();
        for (buffer_index, &(doc_id, ordinal, _)) in batch.iter().enumerate() {
            collector.push(doc_id, ordinal, scores[buffer_index]);
        }
        scored += batch.len();
    }
    debug_assert_eq!(scored, documents.vector_count);
    Ok((collector.into_results(), stats))
}

#[cfg(feature = "sync")]
fn exact_score_dense_candidate_documents_sync(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[f32],
    unit_norm: bool,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let documents = ann_candidate_document_ranges(ann_results, flat)?;
    let vector_byte_size = flat.vector_byte_size();
    let batch_len =
        bounded_rerank_batch(vector_byte_size, DENSE_SCORE_BATCH, documents.vector_count);
    let raw_capacity = batch_len
        .checked_mul(vector_byte_size)
        .ok_or_else(|| Error::Query("dense rerank buffer size overflow".to_string()))?;
    let prepared_query =
        PreparedDenseScoreQuery::new(query, flat.quantization, flat.dim, unit_norm)?;
    let mut scratch = DenseScratch::take();
    scratch.prepare(raw_capacity, batch_len);
    let DenseScratch {
        raw,
        scores,
        batch,
        flat_indexes,
        read_runs,
        ..
    } = &mut *scratch;
    let mut cursor = CandidateVectorCursor::new(&documents.ranges);
    let mut collector = FlatDocumentCollector::new(limit, combiner);
    let mut scored = 0usize;

    while cursor.fill_batch(flat, batch, batch_len)? {
        flat_indexes.clear();
        flat_indexes.extend(batch.iter().map(|&(_, _, flat_index)| flat_index));
        let raw_len = batch
            .len()
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("dense rerank buffer size overflow".to_string()))?;
        let raw = &mut raw[..raw_len];
        read_vector_runs_sync(flat, flat_indexes, read_runs, raw)?;
        prepared_query.score_batch(raw, &mut scores[..batch.len()])?;
        for (buffer_index, &(doc_id, ordinal, _)) in batch.iter().enumerate() {
            collector.push(doc_id, ordinal, scores[buffer_index]);
        }
        scored += batch.len();
    }
    debug_assert_eq!(scored, documents.vector_count);
    Ok(collector.into_results())
}

async fn exact_score_binary_candidate_documents(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let documents = ann_candidate_document_ranges(ann_results, flat)?;
    let probe_scores = sorted_probe_scores(ann_results);
    exact_score_binary_resolved_documents(
        documents,
        &probe_scores,
        flat,
        query,
        dim_bits,
        combiner,
        limit,
    )
    .await
}

async fn exact_score_binary_candidate_document_ids(
    candidate_doc_ids: Vec<DocId>,
    probed_ordinal_scores: &[(u32, u16, f32)],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let documents = ann_candidate_document_ranges_from_ids(candidate_doc_ids, flat)?;
    // Binary leaves hold the original packed codes, so probed ordinals already
    // have exact scores; only ordinals outside the probed leaves are read back.
    let probe_scores = sorted_probe_scores(probed_ordinal_scores);
    exact_score_binary_resolved_documents(
        documents,
        &probe_scores,
        flat,
        query,
        dim_bits,
        combiner,
        limit,
    )
    .await
}

/// Probe scores in `(doc_id, ordinal)` order for the merge-join rerank. The
/// combined binary scan already emits them sorted (so this borrows); any
/// other producer is sorted into an owned copy.
fn sorted_probe_scores(
    scores: &[RawVectorCandidate],
) -> std::borrow::Cow<'_, [RawVectorCandidate]> {
    if scores.is_sorted_by_key(|&(doc_id, ordinal, _)| (doc_id, ordinal)) {
        std::borrow::Cow::Borrowed(scores)
    } else {
        let mut sorted = scores.to_vec();
        sorted.sort_unstable_by_key(|&(doc_id, ordinal, _)| (doc_id, ordinal));
        std::borrow::Cow::Owned(sorted)
    }
}

/// Merge-join cursor over probe scores sorted by `(doc_id, ordinal)`. The
/// candidate cursor streams flat vectors in that same order, so one forward
/// pass replaces a hash map keyed by every probed posting.
struct ProbeScoreCursor<'a> {
    scores: &'a [RawVectorCandidate],
    position: usize,
}

impl<'a> ProbeScoreCursor<'a> {
    fn new(scores: &'a [RawVectorCandidate]) -> Self {
        Self {
            scores,
            position: 0,
        }
    }

    /// Probe score for `(doc_id, ordinal)`, if any. Keys must be requested in
    /// non-decreasing order.
    #[inline]
    fn advance_to(&mut self, doc_id: DocId, ordinal: u16) -> Option<f32> {
        while let Some(&(probed_doc, probed_ordinal, _)) = self.scores.get(self.position)
            && (probed_doc, probed_ordinal) < (doc_id, ordinal)
        {
            self.position += 1;
        }
        match self.scores.get(self.position) {
            Some(&(probed_doc, probed_ordinal, score))
                if probed_doc == doc_id && probed_ordinal == ordinal =>
            {
                Some(score)
            }
            _ => None,
        }
    }
}

async fn exact_score_binary_resolved_documents(
    documents: AnnCandidateDocuments,
    probe_scores: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let vector_byte_size = flat.vector_byte_size();
    let batch_len =
        bounded_rerank_batch(vector_byte_size, BINARY_SCORE_BATCH, documents.vector_count);
    let raw_capacity = batch_len
        .checked_mul(vector_byte_size)
        .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
    let mut scratch = DenseScratch::take();
    scratch.prepare(raw_capacity, batch_len);
    let DenseScratch {
        raw,
        scores,
        batch_scores,
        batch,
        flat_indexes: unresolved_flat_indexes,
        read_runs,
        unresolved,
    } = &mut *scratch;
    let mut cursor = CandidateVectorCursor::new(&documents.ranges);
    let mut probe_cursor = ProbeScoreCursor::new(probe_scores);
    let mut collector = FlatDocumentCollector::new(limit, combiner);
    let mut scored = 0usize;

    while cursor.fill_batch(flat, batch, batch_len)? {
        unresolved.clear();
        for (batch_index, &(doc_id, ordinal, flat_index)) in batch.iter().enumerate() {
            match probe_cursor.advance_to(doc_id, ordinal) {
                Some(score) => batch_scores[batch_index] = score,
                None => unresolved.push((batch_index, flat_index)),
            }
        }
        unresolved_flat_indexes.clear();
        unresolved_flat_indexes.extend(unresolved.iter().map(|&(_, flat_index)| flat_index));
        let raw_len = unresolved
            .len()
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
        let raw = &mut raw[..raw_len];
        read_vector_runs(flat, unresolved_flat_indexes, read_runs, raw).await?;
        crate::structures::simd::batch_hamming_scores(
            query,
            raw,
            vector_byte_size,
            dim_bits,
            &mut scores[..unresolved.len()],
        );
        for (buffer_index, &(batch_index, _)) in unresolved.iter().enumerate() {
            batch_scores[batch_index] = scores[buffer_index];
        }
        for (batch_index, &(doc_id, ordinal, _)) in batch.iter().enumerate() {
            collector.push(doc_id, ordinal, batch_scores[batch_index]);
        }
        scored += batch.len();
    }
    debug_assert_eq!(scored, documents.vector_count);
    Ok(collector.into_results())
}

#[cfg(feature = "sync")]
fn exact_score_binary_candidate_documents_sync(
    ann_results: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let documents = ann_candidate_document_ranges(ann_results, flat)?;
    let probe_scores = sorted_probe_scores(ann_results);
    exact_score_binary_resolved_documents_sync(
        documents,
        &probe_scores,
        flat,
        query,
        dim_bits,
        combiner,
        limit,
    )
}

#[cfg(feature = "sync")]
fn exact_score_binary_candidate_document_ids_sync(
    candidate_doc_ids: Vec<DocId>,
    probed_ordinal_scores: &[(u32, u16, f32)],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let documents = ann_candidate_document_ranges_from_ids(candidate_doc_ids, flat)?;
    let probe_scores = sorted_probe_scores(probed_ordinal_scores);
    exact_score_binary_resolved_documents_sync(
        documents,
        &probe_scores,
        flat,
        query,
        dim_bits,
        combiner,
        limit,
    )
}

#[cfg(feature = "sync")]
fn exact_score_binary_resolved_documents_sync(
    documents: AnnCandidateDocuments,
    probe_scores: &[RawVectorCandidate],
    flat: &LazyFlatVectorData,
    query: &[u8],
    dim_bits: usize,
    combiner: crate::query::MultiValueCombiner,
    limit: usize,
) -> Result<Vec<VectorSearchResult>> {
    let vector_byte_size = flat.vector_byte_size();
    let batch_len =
        bounded_rerank_batch(vector_byte_size, BINARY_SCORE_BATCH, documents.vector_count);
    let raw_capacity = batch_len
        .checked_mul(vector_byte_size)
        .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
    let mut scratch = DenseScratch::take();
    scratch.prepare(raw_capacity, batch_len);
    let DenseScratch {
        raw,
        scores,
        batch_scores,
        batch,
        flat_indexes: unresolved_flat_indexes,
        read_runs,
        unresolved,
    } = &mut *scratch;
    let mut cursor = CandidateVectorCursor::new(&documents.ranges);
    let mut probe_cursor = ProbeScoreCursor::new(probe_scores);
    let mut collector = FlatDocumentCollector::new(limit, combiner);
    let mut scored = 0usize;

    while cursor.fill_batch(flat, batch, batch_len)? {
        unresolved.clear();
        for (batch_index, &(doc_id, ordinal, flat_index)) in batch.iter().enumerate() {
            match probe_cursor.advance_to(doc_id, ordinal) {
                Some(score) => batch_scores[batch_index] = score,
                None => unresolved.push((batch_index, flat_index)),
            }
        }
        unresolved_flat_indexes.clear();
        unresolved_flat_indexes.extend(unresolved.iter().map(|&(_, flat_index)| flat_index));
        let raw_len = unresolved
            .len()
            .checked_mul(vector_byte_size)
            .ok_or_else(|| Error::Query("binary candidate buffer size overflow".to_string()))?;
        let raw = &mut raw[..raw_len];
        read_vector_runs_sync(flat, unresolved_flat_indexes, read_runs, raw)?;
        crate::structures::simd::batch_hamming_scores(
            query,
            raw,
            vector_byte_size,
            dim_bits,
            &mut scores[..unresolved.len()],
        );
        for (buffer_index, &(batch_index, _)) in unresolved.iter().enumerate() {
            batch_scores[batch_index] = scores[buffer_index];
        }
        for (batch_index, &(doc_id, ordinal, _)) in batch.iter().enumerate() {
            collector.push(doc_id, ordinal, batch_scores[batch_index]);
        }
        scored += batch.len();
    }
    debug_assert_eq!(scored, documents.vector_count);
    Ok(collector.into_results())
}

/// Flat vectors scanned in parallel above this count (sync path only). Same
/// bar as the TQ flat fan-out: below it, Rayon scheduling and per-worker
/// collectors cost more than they save.
#[cfg(feature = "sync")]
const FLAT_PARALLEL_SCAN_MIN_VECTORS: usize = 65_536;

/// Batch boundaries over the flat vector map that never split a document,
/// so a per-worker `FlatDocumentCollector` always sees complete documents.
#[cfg(feature = "sync")]
fn document_aligned_batches(flat: &LazyFlatVectorData, batch_len: usize) -> Vec<(usize, usize)> {
    let n = flat.num_vectors;
    let batch_len = batch_len.max(1);
    let mut batches = Vec::with_capacity(n.div_ceil(batch_len));
    let mut start = 0usize;
    while start < n {
        let mut end = (start + batch_len).min(n);
        while end < n && flat.get_doc_id(end).0 == flat.get_doc_id(end - 1).0 {
            end += 1;
        }
        batches.push((start, end));
        start = end;
    }
    batches
}

/// Exact brute-force scan of every flat vector (sync mmap reads). Segments
/// above [`FLAT_PARALLEL_SCAN_MIN_VECTORS`] fan out over document-aligned
/// batches with a collector per worker and merge by the collector's own
/// order (score descending, doc ID ascending), so the result is identical to
/// the sequential scan.
#[cfg(feature = "sync")]
fn brute_force_flat_scan_sync(
    flat: &LazyFlatVectorData,
    query: &[f32],
    unit_norm: bool,
    limit: usize,
    combiner: crate::query::MultiValueCombiner,
) -> Result<(Vec<VectorSearchResult>, DenseAnnScanStats)> {
    let n = flat.num_vectors;
    let batch_len = bounded_vector_score_batch(flat.vector_byte_size(), DENSE_SCORE_BATCH);
    let prepared_query =
        PreparedDenseScoreQuery::new(query, flat.quantization, flat.dim, unit_norm)?;
    let mut stats = DenseAnnScanStats {
        posting_count: n,
        ..DenseAnnScanStats::default()
    };

    #[cfg(feature = "native")]
    if n >= FLAT_PARALLEL_SCAN_MIN_VECTORS && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let batches = document_aligned_batches(flat, batch_len);
        stats.scored_blocks = batches.len();
        stats.parallel = true;
        let merge = |mut left: Vec<VectorSearchResult>, mut right: Vec<VectorSearchResult>| {
            left.append(&mut right);
            left.sort_unstable_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.doc_id.cmp(&b.doc_id))
            });
            left.truncate(limit);
            left
        };
        let results = batches
            .par_iter()
            .try_fold(
                || {
                    (
                        FlatDocumentCollector::new(limit, combiner),
                        Vec::<f32>::new(),
                    )
                },
                |(mut collector, mut scores), &(start, end)| {
                    let count = end - start;
                    if scores.len() < count {
                        scores.resize(count, 0.0);
                    }
                    let batch_bytes = flat
                        .read_vectors_batch_sync(start, count)
                        .map_err(Error::Io)?;
                    prepared_query.score_batch(batch_bytes.as_slice(), &mut scores[..count])?;
                    for (i, &score) in scores.iter().enumerate().take(count) {
                        let (doc_id, ordinal) = flat.get_doc_id(start + i);
                        collector.push(doc_id, ordinal, score);
                    }
                    Ok::<_, Error>((collector, scores))
                },
            )
            .map(|folded| folded.map(|(collector, _)| collector.into_results()))
            .try_reduce(Vec::new, |left, right| Ok(merge(left, right)))?;
        return Ok((results, stats));
    }

    let mut collector = FlatDocumentCollector::new(limit, combiner);
    let mut scratch = DenseScratch::take();
    scratch.prepare(0, batch_len);
    let scores = &mut scratch.scores;
    for batch_start in (0..n).step_by(batch_len) {
        let batch_count = batch_len.min(n - batch_start);
        let batch_bytes = flat
            .read_vectors_batch_sync(batch_start, batch_count)
            .map_err(Error::Io)?;
        prepared_query.score_batch(batch_bytes.as_slice(), &mut scores[..batch_count])?;
        stats.scored_blocks += 1;
        for (i, &score) in scores.iter().enumerate().take(batch_count) {
            let (doc_id, ordinal) = flat.get_doc_id(batch_start + i);
            collector.push(doc_id, ordinal, score);
        }
    }
    Ok((collector.into_results(), stats))
}

/// Whether every `(doc_id, ordinal)` key an ANN payload can emit is unique:
/// the field is single-valued and the payload holds exactly one posting per
/// stored vector (no SOAR spill). Only then may the heap-only collector run.
fn ann_keys_are_unique(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    flat: &LazyFlatVectorData,
) -> bool {
    flat.num_vectors == flat.num_docs_with_vectors()
        && index.header().vector_count == flat.num_vectors
}

fn dense_ann_kind_label(ann_index: Option<&VectorIndex>) -> &'static str {
    match ann_index {
        Some(VectorIndex::BinaryIvf(_)) => "binary_ivf",
        Some(VectorIndex::Tq { .. }) => "tq_flat",
        Some(VectorIndex::IvfTq { .. }) => "ivf_tq",
        Some(VectorIndex::ScannAh(_)) => "scann_ah",
        Some(VectorIndex::ScannBinary(_)) => "scann_binary",
        None => "flat",
    }
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

/// Per-query dense plan caches, shared by every segment scorer the query
/// spawns. Both members are query-global: the IVF-TQ probe route and its
/// LUTs depend only on the query and index-level artifacts, and the TQ
/// LUTs depend only on the query and the schema dimension.
#[derive(Debug, Default)]
pub struct DensePlanCache {
    pub(crate) tq: std::sync::Mutex<Option<std::sync::Arc<crate::structures::TqQueryPlan>>>,
    pub(crate) ivf_tq: std::sync::Mutex<Option<std::sync::Arc<crate::structures::TqIvfQueryPlan>>>,
    scann: std::sync::Mutex<Option<ScannPlanCacheEntry>>,
}

#[derive(Debug)]
struct ScannPlanCacheEntry {
    artifact_id: u64,
    probes: usize,
    query_bits: Vec<u32>,
    plan: std::sync::Arc<crate::structures::vector::scann::FloatScannQuery>,
}

/// Search one segment's TQ payload, reusing the per-query plan across
/// segments: the codec is a pure function of the schema dimension, so the
/// LUTs are identical for every segment of the field (mirrors the IVF-PQ
/// `probe_cache` hot-path rule — no repeated per-segment plan allocation).
#[allow(clippy::too_many_arguments)]
fn search_tq_segment(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    codec: &crate::structures::TqCodec,
    query: &[f32],
    fetch_k: usize,
    document_combiner: Option<crate::query::MultiValueCombiner>,
    field: Field,
    dim: usize,
    plan_cache: Option<&std::sync::Mutex<Option<std::sync::Arc<crate::structures::TqQueryPlan>>>>,
    unique_keys: bool,
) -> Result<(Vec<RawVectorCandidate>, DenseAnnScanStats)> {
    validate_tq_ann(index, codec, dim, field)?;
    let plan = cached_tq_query_plan(codec, query, plan_cache)?;
    match document_combiner {
        Some(combiner) => index
            .search_tq_combined_documents(fetch_k, &plan, combiner)
            .map(|candidates| {
                (
                    candidates
                        .into_iter()
                        // Exact dense reranking consumes only the document ID.
                        // Use a zero placeholder so the document aggregate can
                        // never be mistaken for an ordinal score.
                        .map(|candidate| (candidate.doc_id, 0, 0.0))
                        .collect(),
                    DenseAnnScanStats {
                        posting_count: index.header().vector_count,
                        ..DenseAnnScanStats::default()
                    },
                )
            }),
        None => index.search_tq_distinct_with_stats(fetch_k, &plan, unique_keys),
    }
    .map_err(|error| {
        Error::Corruption(format!("invalid TQ payload for field {}: {error}", field.0))
    })
}

/// Return the query-global flat-TQ plan, rebuilding it whenever either the
/// codec generation or the exact query bits differ.
fn cached_tq_query_plan(
    codec: &crate::structures::TqCodec,
    query: &[f32],
    plan_cache: Option<&std::sync::Mutex<Option<std::sync::Arc<crate::structures::TqQueryPlan>>>>,
) -> Result<std::sync::Arc<crate::structures::TqQueryPlan>> {
    Ok(match plan_cache {
        Some(cache) => {
            let mut cached = cache
                .lock()
                .map_err(|_| Error::Internal("TQ plan cache is poisoned".into()))?;
            match cached.as_ref() {
                Some(plan)
                    if plan.fingerprint() == codec.fingerprint() && plan.matches_query(query) =>
                {
                    std::sync::Arc::clone(plan)
                }
                _ => {
                    let plan =
                        std::sync::Arc::new(crate::structures::TqQueryPlan::build(codec, query));
                    *cached = Some(std::sync::Arc::clone(&plan));
                    plan
                }
            }
        }
        None => std::sync::Arc::new(crate::structures::TqQueryPlan::build(codec, query)),
    })
}

fn validate_tq_ann(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    codec: &crate::structures::TqCodec,
    dim: usize,
    field: Field,
) -> Result<()> {
    let header = index.header();
    if header.dim != dim
        || codec.dim() != dim
        || header.code_size != codec.code_size()
        || header.quantizer_version != codec.fingerprint()
        || header.codebook_version != 0
        || header.num_clusters != 1
    {
        return Err(Error::Corruption(format!(
            "TQ payload for field {} does not match the codec derived from schema dimension {dim}",
            field.0,
        )));
    }
    Ok(())
}

/// Search one segment's IVF-TQ payload. The probe route, the `⟨q̂,c⟩`
/// scalars, and the TQ LUTs are all query-global, so the plan is cached and
/// shared across every segment of the field.
#[allow(clippy::too_many_arguments)]
fn search_ivf_tq_segment(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    centroids: &CoarseCentroids,
    codec: &crate::structures::TqCodec,
    query: &[f32],
    fetch_k: usize,
    document_combiner: Option<crate::query::MultiValueCombiner>,
    field: Field,
    nprobe: usize,
    routing: crate::dsl::IvfRoutingMode,
    plan_cache: Option<
        &std::sync::Mutex<Option<std::sync::Arc<crate::structures::TqIvfQueryPlan>>>,
    >,
    unique_keys: bool,
) -> Result<(Vec<RawVectorCandidate>, DenseAnnScanStats)> {
    let effective_nprobe = nprobe.clamp(1, centroids.num_clusters as usize);
    if effective_nprobe != nprobe {
        log::debug!(
            "[search_ivf_tq] field {}: nprobe {nprobe} clamped to {effective_nprobe} (codebook has {} leaves)",
            field.0,
            centroids.num_clusters
        );
    }
    let request_fingerprint = crate::structures::TqIvfQueryPlan::request_fingerprint_for(
        centroids,
        query,
        effective_nprobe,
        routing,
    );
    let build = || {
        std::sync::Arc::new(crate::structures::TqIvfQueryPlan::build(
            centroids,
            codec,
            query,
            effective_nprobe,
            routing,
        ))
    };
    let plan = match plan_cache {
        Some(cache) => {
            let mut cached = cache
                .lock()
                .map_err(|_| Error::Internal("IVF-TQ plan cache is poisoned".into()))?;
            match cached.as_ref() {
                Some(plan)
                    if plan.quantizer_version == centroids.version
                        && plan.fingerprint == codec.fingerprint()
                        && plan.request_fingerprint == request_fingerprint
                        && plan.cluster_ids.len() == effective_nprobe =>
                {
                    std::sync::Arc::clone(plan)
                }
                _ => {
                    let plan = build();
                    *cached = Some(std::sync::Arc::clone(&plan));
                    plan
                }
            }
        }
        None => build(),
    };
    let candidates = match document_combiner {
        Some(combiner) => index
            .search_ivf_tq_combined_documents_with_stats(fetch_k, &plan, combiner)
            .map(|(documents, stats)| {
                (
                    documents
                        .into_iter()
                        // The compressed score aggregates a whole document. Exact
                        // dense reranking consumes only its ID; a zero placeholder
                        // prevents accidental reuse as an ordinal score.
                        .map(|candidate| (candidate.doc_id, 0, 0.0))
                        .collect(),
                    stats,
                )
            }),
        None => index.search_ivf_tq_distinct_with_stats(fetch_k, &plan, unique_keys),
    };
    candidates.map_err(|error| {
        Error::Corruption(format!(
            "invalid IVF-TQ payload for field {}: {error}",
            field.0
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn search_scann_ah_segment(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    artifact: &crate::segment::ScannTrainedArtifactBytes,
    query: &[f32],
    fetch_k: usize,
    combiner: crate::query::MultiValueCombiner,
    field: Field,
    nprobe: usize,
    plan_cache: Option<&std::sync::Mutex<Option<ScannPlanCacheEntry>>>,
) -> Result<Vec<RawVectorCandidate>> {
    index
        .validate_scann_generation(
            artifact.config(),
            artifact.generation(),
            artifact.artifact_id(),
        )
        .map_err(|error| {
            Error::Corruption(format!(
                "ScaNN generation mismatch for field {}: {error}",
                field.0
            ))
        })?;
    let probes = nprobe.clamp(1, artifact.config().num_leaves as usize);
    if probes != nprobe {
        log::debug!(
            "[search_scann_ah] field {}: nprobe {nprobe} clamped to {probes} (model has {} leaves)",
            field.0,
            artifact.config().num_leaves
        );
    }
    let build = || {
        let mut normalized_query = query.to_vec();
        crate::structures::vector::ivf::routing::normalize_cosine_in_place(&mut normalized_query);
        artifact
            .float_model()
            .map_err(Error::Io)?
            .prepare_query(&normalized_query, probes)
            .map(std::sync::Arc::new)
            .map_err(|error| Error::Query(format!("invalid ScaNN query: {error}")))
    };
    let plan = match plan_cache {
        Some(cache) => {
            let mut cached = cache
                .lock()
                .map_err(|_| Error::Internal("ScaNN plan cache is poisoned".into()))?;
            match cached.as_ref() {
                // Compare exact bits in place; the owned key is built only on
                // a miss (one per query, not one per segment).
                Some(entry)
                    if entry.artifact_id == artifact.artifact_id()
                        && entry.probes == probes
                        && entry.query_bits.len() == query.len()
                        && entry
                            .query_bits
                            .iter()
                            .zip(query)
                            .all(|(&bits, value)| bits == value.to_bits()) =>
                {
                    std::sync::Arc::clone(&entry.plan)
                }
                _ => {
                    let plan = build()?;
                    *cached = Some(ScannPlanCacheEntry {
                        artifact_id: artifact.artifact_id(),
                        probes,
                        query_bits: query.iter().map(|value| value.to_bits()).collect(),
                        plan: std::sync::Arc::clone(&plan),
                    });
                    plan
                }
            }
        }
        None => build()?,
    };
    index
        .search_scann_ah_combined_documents(fetch_k, &plan, combiner)
        .map(|documents| {
            documents
                .into_iter()
                .map(|candidate| (candidate.doc_id, 0, 0.0))
                .collect()
        })
        .map_err(|error| {
            Error::Corruption(format!(
                "invalid ScaNN AH payload for field {}: {error}",
                field.0
            ))
        })
}

fn validate_ivf_tq_ann(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    centroids: &CoarseCentroids,
    codec: &crate::structures::TqCodec,
    dim: usize,
    routing: crate::dsl::IvfRoutingMode,
    field: Field,
) -> Result<()> {
    let header = index.header();
    if !crate::structures::is_ivf_tq_cosine_generation(centroids.version)
        || !crate::structures::is_ivf_tq_cosine_generation(header.quantizer_version)
    {
        return Err(Error::Corruption(format!(
            "IVF-TQ field {} uses a legacy unmarked raw-vector generation that cannot \
             preserve cosine candidate semantics; rebuild the index with a current \
             Hermes version",
            field.0,
        )));
    }
    if header.dim != dim
        || codec.dim() != dim
        || header.code_size != codec.code_size()
        || header.num_clusters != centroids.num_clusters
        || header.quantizer_version != centroids.version
        || header.codebook_version != codec.fingerprint()
        || header.routing != routing
    {
        return Err(Error::Corruption(format!(
            "IVF-TQ payload for field {} does not match its quantizer/codec generation",
            field.0,
        )));
    }
    Ok(())
}

fn validate_binary_ann(
    index: &crate::segment::ann_disk::AnnDiskIndex,
    quantizer: &crate::structures::BinaryCoarseQuantizer,
    config: &crate::dsl::BinaryDenseVectorConfig,
    dim: usize,
    field: Field,
) -> Result<()> {
    let header = index.header();
    if header.dim != dim
        || header.code_size != config.byte_len()
        || header.num_clusters != quantizer.num_clusters
        || header.quantizer_version != quantizer.version
        || header.codebook_version != 0
        || header.routing != config.ivf_routing
        || quantizer.dim_bits != dim
    {
        return Err(Error::Corruption(format!(
            "binary IVF field {} does not match its quantizer/schema generation",
            field.0,
        )));
    }
    Ok(())
}

fn binary_probe_clusters(
    quantizer: &crate::structures::BinaryCoarseQuantizer,
    query: &[u8],
    nprobe: usize,
    routing: crate::dsl::IvfRoutingMode,
    cache: Option<&std::sync::Mutex<Option<crate::structures::IvfProbePlan>>>,
) -> Result<std::sync::Arc<[u32]>> {
    let effective_nprobe = nprobe.clamp(1, quantizer.num_clusters as usize);
    let request_fingerprint = quantizer.request_fingerprint(query, effective_nprobe, routing);
    if let Some(cache) = cache {
        let mut cached = cache
            .lock()
            .map_err(|_| Error::Internal("binary IVF probe cache is poisoned".into()))?;
        if let Some(plan) = cached.as_ref()
            && plan.quantizer_version == quantizer.version
            && plan.request_fingerprint == request_fingerprint
            && plan.cluster_ids.len() == effective_nprobe
        {
            return Ok(std::sync::Arc::clone(&plan.cluster_ids));
        }
        let plan = quantizer
            .probe(query, effective_nprobe, routing)
            .map_err(|error| Error::Query(format!("binary IVF routing failed: {error}")))?;
        let clusters = std::sync::Arc::clone(&plan.cluster_ids);
        *cached = Some(plan);
        return Ok(clusters);
    }
    Ok(quantizer
        .probe(query, effective_nprobe, routing)
        .map_err(|error| Error::Query(format!("binary IVF routing failed: {error}")))?
        .cluster_ids)
}

thread_local! {
    /// Binary ScaNN routing scratch: beam buffers plus the retained-hit set,
    /// reused across every probe on this thread instead of being allocated
    /// per segment probe.
    static BINARY_SCANN_ROUTING_SCRATCH: std::cell::RefCell<
        crate::structures::vector::scann::BinaryScannSearchScratch,
    > = std::cell::RefCell::new(Default::default());
}

fn probe_binary_scann_with_scratch(
    model: &crate::structures::vector::scann::QuantizedBinaryScannModelView<'_>,
    query: &[u8],
    probes: usize,
    routing_beam: usize,
) -> Result<crate::structures::vector::scann::BinaryScannProbePlan> {
    BINARY_SCANN_ROUTING_SCRATCH.with(|cell| {
        let mut scratch = cell
            .try_borrow_mut()
            .map_err(|_| Error::Internal("binary ScaNN routing scratch is busy".into()))?;
        model
            .probe(query, probes, routing_beam, &mut scratch)
            .map_err(|error| Error::Query(format!("binary ScaNN routing failed: {error}")))
    })
}

fn binary_scann_probe_clusters(
    model: &crate::structures::vector::scann::QuantizedBinaryScannModelView<'_>,
    query: &[u8],
    nprobe: usize,
    cache: Option<&std::sync::Mutex<Option<crate::structures::IvfProbePlan>>>,
) -> Result<std::sync::Arc<[u32]>> {
    let probes = nprobe.clamp(1, model.num_leaves() as usize);
    // Start with a bounded recall beam. The model widens intermediate levels
    // when necessary so every requested terminal leaf remains reachable.
    let routing_beam = probes.min(64);
    let mut fingerprint = 0xcbf2_9ce4_8422_2325u64;
    for byte in query.iter().copied().chain((probes as u64).to_le_bytes()) {
        fingerprint ^= u64::from(byte);
        fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(cache) = cache {
        let mut cached = cache
            .lock()
            .map_err(|_| Error::Internal("binary ScaNN probe cache is poisoned".into()))?;
        if let Some(plan) = cached.as_ref()
            && plan.quantizer_version == model.fingerprint()
            && plan.request_fingerprint == fingerprint
            && plan.cluster_ids.len() == probes
        {
            return Ok(std::sync::Arc::clone(&plan.cluster_ids));
        }
        let routed = probe_binary_scann_with_scratch(model, query, probes, routing_beam)?;
        let plan =
            crate::structures::IvfProbePlan::new(model.fingerprint(), fingerprint, routed.leaf_ids);
        let leaves = std::sync::Arc::clone(&plan.cluster_ids);
        *cached = Some(plan);
        return Ok(leaves);
    }
    probe_binary_scann_with_scratch(model, query, probes, routing_beam)
        .map(|plan| plan.leaf_ids.into())
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
    /// Per-segment ANN payloads.
    vector_indexes: FxHashMap<u32, VectorIndex>,
    /// Lazy flat vectors per field — document maps and vectors stay file-backed.
    flat_vectors: FxHashMap<u32, LazyFlatVectorData>,
    /// Logical size of the retained `.vectors` file handle.
    dense_file_backed_bytes: u64,
    /// One immutable generation of all index-global ANN artifacts.
    trained_vectors: Arc<crate::segment::TrainedVectorStructures>,
    /// Sparse vector indexes per field (MaxScore format)
    sparse_indexes: FxHashMap<u32, SparseIndex>,
    /// BMP sparse vector indexes per field (BMP format)
    bmp_indexes: FxHashMap<u32, BmpIndex>,
    /// Logical size of the retained `.sparse` file handle.
    sparse_file_backed_bytes: u64,
    /// Position file handle for phrase queries (lazy loading)
    positions_handle: Option<FileHandle>,
    /// Fast-field columnar readers per field_id
    fast_fields: FxHashMap<u32, crate::structures::fast_field::FastFieldReader>,
    /// Virtual-id maps of chunked text fields per field_id
    chunk_maps: FxHashMap<u32, super::chunk_map::ChunkMap>,
    /// Dense-vector hot-metadata pin accounting (see `segment::pin`).
    #[cfg(feature = "native")]
    dense_pin_report: crate::segment::pin::PinReport,
    /// Sparse-vector hot-metadata pin accounting (see `segment::pin`).
    #[cfg(feature = "native")]
    sparse_pin_report: crate::segment::pin::PinReport,
}

impl SegmentReader {
    /// Open a segment with lazy loading
    pub async fn open<D: Directory>(
        dir: &D,
        segment_id: SegmentId,
        schema: Arc<Schema>,
        term_cache_blocks: usize,
    ) -> Result<Self> {
        Self::open_with_store_cache(
            dir,
            segment_id,
            schema,
            term_cache_blocks,
            dir as *const D as usize,
            Arc::new(super::SharedStoreCache::new(0)),
        )
        .await
    }

    /// Open a search segment against the process-wide document-store cache.
    pub(crate) async fn open_with_store_cache<D: Directory>(
        dir: &D,
        segment_id: SegmentId,
        schema: Arc<Schema>,
        term_cache_blocks: usize,
        store_cache_directory_namespace: usize,
        store_cache: Arc<super::SharedStoreCache>,
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
        let store = AsyncStoreReader::open(
            store_handle,
            store_cache_directory_namespace,
            segment_id.0,
            store_cache,
        )
        .await?;

        // Load dense vector indexes from unified .vectors file
        let vectors_data = loader::load_vectors_file(dir, &files, &schema, meta.num_docs).await?;
        let dense_file_backed_bytes = vectors_data.file_backed_bytes;
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
        let sparse_file_backed_bytes = sparse_data.file_backed_bytes;
        let sparse_indexes = sparse_data.maxscore_indexes;
        let bmp_indexes = sparse_data.bmp_indexes;

        // Open positions file handle (if exists) - offsets are now in TermInfo
        let positions_handle = loader::open_positions_file(dir, &files, &schema).await?;

        // Load fast-field columns from .fast file
        let fast_fields = loader::load_fast_fields_file(dir, &files, &schema).await?;

        // Load chunk maps of chunked text fields from .chunks file
        let chunk_maps = loader::load_chunk_maps_file(dir, &files, &schema).await?;

        // Log segment loading stats
        {
            let mut parts = vec![format!(
                "[segment] loaded {:016x}: docs={}",
                segment_id.0, meta.num_docs
            )];
            if !vector_indexes.is_empty() || !flat_vectors.is_empty() {
                parts.push(format!(
                    "dense vectors: {} ANN + {} flat fields",
                    vector_indexes.len(),
                    flat_vectors.len()
                ));
            }
            for (field_id, idx) in &sparse_indexes {
                parts.push(format!(
                    "sparse vector field {}: {} dims, ~{}",
                    field_id,
                    idx.num_dimensions(),
                    crate::format_bytes(idx.num_dimensions() as u64 * 24)
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
            for (field_id, map) in &chunk_maps {
                parts.push(format!(
                    "chunked text field {}: {} chunks",
                    field_id,
                    map.num_chunks()
                ));
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
            dense_file_backed_bytes,
            trained_vectors: Arc::new(crate::segment::TrainedVectorStructures::default()),
            sparse_indexes,
            bmp_indexes,
            sparse_file_backed_bytes,
            positions_handle,
            fast_fields,
            chunk_maps,
            #[cfg(feature = "native")]
            dense_pin_report: Default::default(),
            #[cfg(feature = "native")]
            sparse_pin_report: Default::default(),
        };

        // Pin hot metadata per the process-wide policy (no-op when disabled)
        #[cfg(feature = "native")]
        reader.apply_pin_policy(&crate::segment::pin::pin_policy().to_owned());

        // Structural ANN health from the already-parsed run directories —
        // O(runs) per field, no payload reads. This is the passive tier of
        // `docs/diagnostics.md`: leaf collapse and extent fragmentation warn
        // here instead of surfacing as unexplained latency.
        for (&field_id, vector_index) in &reader.vector_indexes {
            match vector_index {
                VectorIndex::BinaryIvf(index)
                | VectorIndex::IvfTq { index, .. }
                | VectorIndex::ScannAh(index)
                | VectorIndex::ScannBinary(index) => {
                    index.get().report_health(
                        reader.schema.index_label(),
                        field_id,
                        reader.meta.id,
                    );
                }
                // TQ flat payloads have no cluster structure; skew and
                // fragmentation metrics would be meaningless there.
                VectorIndex::Tq { .. } => {}
            }
        }

        Ok(reader)
    }

    /// Structural health of one field's IVF payload, if it has one.
    ///
    /// Cheap (O(runs) over in-memory data); exposed for `hermes-tool diagnose`.
    pub fn ann_health(&self, field: Field) -> Option<crate::segment::ann_disk::AnnHealth> {
        match self.vector_indexes.get(&field.0)? {
            VectorIndex::BinaryIvf(index)
            | VectorIndex::IvfTq { index, .. }
            | VectorIndex::ScannAh(index)
            | VectorIndex::ScannBinary(index) => Some(index.get().health()),
            VectorIndex::Tq { .. } => None,
        }
    }

    /// Pin per-query-mandatory metadata sections in priority order until the
    /// budget is exhausted (see `segment::pin` and docs/hot-metadata-pinning.md).
    ///
    /// Priority: ANN run directories → BMP block-offset tables → sparse skip
    /// sections → doc-id maps → BMP E offsets + coarse H. Bulk data (ANN codes,
    /// D/E grid payloads, block data, raw vectors) is never pinned. Fail-loud: budget
    /// exhaustion and mlock failures are
    /// logged and visible via `SegmentMemoryStats::{pin_intended_bytes,
    /// pinned_metadata_bytes}`.
    #[cfg(feature = "native")]
    pub(crate) fn apply_pin_policy(&mut self, policy: &crate::segment::pin::PinPolicy) {
        use crate::segment::pin::PinReport;

        // With a zero budget nothing is pinned, but the pass still runs as a
        // dry run (every section is "skipped: budget exhausted") so the
        // amount of hot metadata left unpinned can be reported loudly.
        let disabled = !policy.is_enabled();
        let mut remaining = policy.budget_bytes;
        let mut dense_report = PinReport::default();
        let mut sparse_report = PinReport::default();

        // Priority 1: compact ANN lookup directories (heap-resident; not part
        // of the mmap-backed dry run when pinning is disabled)
        if !disabled {
            for index in self.vector_indexes.values_mut() {
                index.pin_lookup_directory(policy.mode, &mut remaining, &mut dense_report);
            }
        }
        // Priority 2: BMP block-offset tables
        for bmp in self.bmp_indexes.values_mut() {
            bmp.pin_block_starts(policy.mode, &mut remaining, &mut sparse_report);
        }
        // Priority 3: sparse skip sections
        for sparse in self.sparse_indexes.values_mut() {
            sparse.pin_skip_section(policy.mode, &mut remaining, &mut sparse_report);
        }
        // Priority 4: doc-id maps
        for flat in self.flat_vectors.values_mut() {
            flat.pin_doc_ids(policy.mode, &mut remaining, &mut dense_report);
        }
        for bmp in self.bmp_indexes.values_mut() {
            bmp.pin_doc_maps(policy.mode, &mut remaining, &mut sparse_report);
        }
        // Priority 5: BMP E offsets and coarse H
        for bmp in self.bmp_indexes.values_mut() {
            bmp.pin_query_hierarchy(policy.mode, &mut remaining, &mut sparse_report);
        }

        let report = PinReport {
            intended_bytes: dense_report
                .intended_bytes
                .saturating_add(sparse_report.intended_bytes),
            pinned_bytes: dense_report
                .pinned_bytes
                .saturating_add(sparse_report.pinned_bytes),
            skipped_budget_bytes: dense_report
                .skipped_budget_bytes
                .saturating_add(sparse_report.skipped_budget_bytes),
            failed_bytes: dense_report
                .failed_bytes
                .saturating_add(sparse_report.failed_bytes),
            heap_copy_bytes: dense_report
                .heap_copy_bytes
                .saturating_add(sparse_report.heap_copy_bytes),
        };
        if disabled {
            crate::segment::pin::warn_if_pinning_disabled(
                self.schema.index_label(),
                self.meta.id,
                report.intended_bytes,
            );
            return;
        }
        if report.skipped_budget_bytes > 0 || report.failed_bytes > 0 {
            log::warn!(
                "[pin] index={} segment {:016x}: pinned {}/{} (budget skipped {}, mlock failed {}) — \
                 raise HERMES_PIN_METADATA_BUDGET_MB or RLIMIT_MEMLOCK for full coverage",
                self.schema.index_label(),
                self.meta.id,
                crate::format_bytes(report.pinned_bytes),
                crate::format_bytes(report.intended_bytes),
                crate::format_bytes(report.skipped_budget_bytes),
                crate::format_bytes(report.failed_bytes),
            );
        } else if report.pinned_bytes > 0 {
            log::info!(
                "[pin] index={} segment {:016x}: pinned {} of hot metadata ({:?})",
                self.schema.index_label(),
                self.meta.id,
                crate::format_bytes(report.pinned_bytes),
                policy.mode,
            );
        }
        self.dense_pin_report = dense_report;
        self.sparse_pin_report = sparse_report;
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

    /// Virtual-id map of a chunked text field, when the field is chunked and
    /// this segment indexed at least one chunk of it.
    pub fn chunk_map(&self, field: Field) -> Option<&super::chunk_map::ChunkMap> {
        self.chunk_maps.get(&field.0)
    }

    /// All chunk maps of this segment.
    pub fn chunk_maps(&self) -> &FxHashMap<u32, super::chunk_map::ChunkMap> {
        &self.chunk_maps
    }

    /// Whether `field` is declared chunked in the schema (its postings are
    /// keyed by virtual chunk ids, never by document ids).
    pub fn is_chunked_field(&self, field: Field) -> bool {
        self.schema
            .get_field_entry(field)
            .is_some_and(|entry| entry.chunked)
    }

    /// Number of chunks a chunked field holds in this segment (0 when none).
    pub fn num_chunks(&self, field: Field) -> u32 {
        self.chunk_maps
            .get(&field.0)
            .map_or(0, |map| map.num_chunks())
    }

    /// BM25 corpus size for `field`: chunks for a chunked field, documents
    /// otherwise.
    pub fn text_corpus_size(&self, field: Field) -> f32 {
        if self.is_chunked_field(field) {
            self.num_chunks(field) as f32
        } else {
            self.meta.num_docs as f32
        }
    }

    /// Whether this segment carries a `.chunks` file.
    pub fn has_chunks_file(&self) -> bool {
        !self.chunk_maps.is_empty()
    }

    /// Get term dictionary stats for debugging
    pub fn term_dict_stats(&self) -> SSTableStats {
        self.term_dict.stats()
    }

    /// Account for heap, file-backed, and pinned bytes separately.
    pub fn memory_stats(&self) -> SegmentMemoryStats {
        let term_dict_stats = self.term_dict.stats();

        // Report actual decompressed heap retention. Both caches use variable
        // boundary blocks, so multiplying a block count by a guessed size can
        // materially under-report resident memory.
        let term_dict_cache_bytes = self.term_dict.cached_bytes();
        let store_cache_bytes = self.store.cached_bytes();

        // Sparse heap: SoA dimension tables and small reader objects. Posting
        // payloads, BMP grids, and document maps remain file-backed.
        let sparse_heap_bytes: usize = self
            .sparse_indexes
            .values()
            .map(|s| s.estimated_heap_bytes())
            .sum::<usize>()
            + self
                .bmp_indexes
                .values()
                .map(|b| b.estimated_heap_bytes())
                .sum::<usize>();

        // Dense corpus columns are file-backed. Only compact ANN run
        // directories and flat-reader objects count as heap here.
        let dense_heap_bytes: usize = self
            .vector_indexes
            .values()
            .map(|v| v.estimated_heap_bytes())
            .sum::<usize>()
            + self
                .flat_vectors
                .values()
                .map(LazyFlatVectorData::estimated_heap_bytes)
                .sum::<usize>();

        #[cfg(feature = "native")]
        let (sparse_heap_bytes, dense_heap_bytes) = (
            sparse_heap_bytes.saturating_add(
                usize::try_from(self.sparse_pin_report.heap_copy_bytes).unwrap_or(usize::MAX),
            ),
            dense_heap_bytes.saturating_add(
                usize::try_from(self.dense_pin_report.heap_copy_bytes).unwrap_or(usize::MAX),
            ),
        );

        #[cfg(feature = "native")]
        let (
            sparse_pinned_metadata_bytes,
            sparse_pin_intended_bytes,
            dense_pinned_metadata_bytes,
            dense_pin_intended_bytes,
        ) = (
            self.sparse_pin_report.pinned_bytes,
            self.sparse_pin_report.intended_bytes,
            self.dense_pin_report.pinned_bytes,
            self.dense_pin_report.intended_bytes,
        );
        #[cfg(not(feature = "native"))]
        let (
            sparse_pinned_metadata_bytes,
            sparse_pin_intended_bytes,
            dense_pinned_metadata_bytes,
            dense_pin_intended_bytes,
        ) = (0u64, 0u64, 0u64, 0u64);

        let pinned_metadata_bytes =
            sparse_pinned_metadata_bytes.saturating_add(dense_pinned_metadata_bytes);
        let pin_intended_bytes = sparse_pin_intended_bytes.saturating_add(dense_pin_intended_bytes);

        SegmentMemoryStats {
            segment_id: self.meta.id,
            num_docs: self.meta.num_docs,
            term_dict_cache_bytes,
            store_cache_bytes,
            sparse_heap_bytes,
            dense_heap_bytes,
            term_bloom_file_bytes: term_dict_stats.bloom_filter_size as u64,
            sparse_file_backed_bytes: self.sparse_file_backed_bytes,
            dense_file_backed_bytes: self.dense_file_backed_bytes,
            pinned_metadata_bytes,
            pin_intended_bytes,
            sparse_pinned_metadata_bytes,
            sparse_pin_intended_bytes,
            dense_pinned_metadata_bytes,
            dense_pin_intended_bytes,
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
                            log::warn!(
                                "Failed to hydrate binary dense vector field {}: {}",
                                field_id,
                                e
                            );
                        }
                    }
                } else {
                    match lazy_flat.get_vector(flat_idx).await {
                        Ok(vec) => {
                            doc.add_dense_vector(Field(field_id), vec);
                        }
                        Err(e) => {
                            log::warn!("Failed to hydrate dense vector field {}: {}", field_id, e);
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

    /// Previous per-batch preparation path retained as an equivalence oracle.
    #[cfg(test)]
    fn score_quantized_batch_legacy(
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

    /// Search dense vectors through the production IVF-PQ index.
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
        self.search_dense_vector_impl(field, query, k, nprobe, rerank_factor, combiner, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn search_dense_vector_with_probe_cache(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
        plan_cache: &DensePlanCache,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_dense_vector_impl(
            field,
            query,
            k,
            nprobe,
            rerank_factor,
            combiner,
            Some(plan_cache),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn search_dense_vector_impl(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
        plan_cache: Option<&DensePlanCache>,
    ) -> Result<Vec<VectorSearchResult>> {
        let params =
            self.validate_dense_search_request(field, query, nprobe, rerank_factor, combiner)?;
        let fetch_k = checked_dense_fetch_k(k, rerank_factor)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let configured_ann_index = self.vector_indexes.get(&field.0);
        let lazy_flat = self.flat_vectors.get(&field.0);
        // No vectors at all for this field
        if configured_ann_index.is_none() && lazy_flat.is_none() {
            return Ok(Vec::new());
        }

        if configured_ann_index.is_some() && lazy_flat.is_none() {
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

        let needs_document_aggregation = lazy_flat.is_some_and(|flat| {
            flat.num_vectors != flat.num_docs_with_vectors()
                && !matches!(combiner, crate::query::MultiValueCombiner::Max)
        });
        // Keep every configured ANN index active. Multi-value semantics are
        // handled by bounded combiner-aware scans; IVF-TQ accepts only the
        // cosine-normalized generation validated below.
        let ann_index = configured_ann_index;

        // Results are (doc_id, ordinal, score) where score = similarity (higher = better)
        let t0 = std::time::Instant::now();
        let mut flat_results = None;
        let (results, scan_stats): (Vec<(u32, u16, f32)>, DenseAnnScanStats) = if let Some(index) =
            ann_index
        {
            // ANN search through the segment's ANN payload.
            match index {
                VectorIndex::Tq { index: lazy, codec } => {
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    // Estimated similarities feed the shared exact re-rank.
                    search_tq_segment(
                        lazy.get(),
                        codec,
                        query,
                        fetch_k.min(flat.num_docs_with_vectors()),
                        needs_document_aggregation.then_some(combiner),
                        field,
                        params.dim,
                        plan_cache.map(|cache| &cache.tq),
                        ann_keys_are_unique(lazy.get(), flat),
                    )?
                }
                VectorIndex::IvfTq { index: lazy, codec } => {
                    let index = lazy.get();
                    let centroids =
                        self.trained_vectors
                            .centroids
                            .get(&field.0)
                            .ok_or_else(|| {
                                Error::Schema(format!(
                                    "IVF-TQ index requires coarse centroids for field {}",
                                    field.0
                                ))
                            })?;
                    validate_coarse_centroids(centroids, params.dim)?;
                    let routing = self
                        .schema
                        .get_field_entry(field)
                        .and_then(|entry| entry.dense_vector_config.as_ref())
                        .map_or(crate::dsl::IvfRoutingMode::Auto, |config| {
                            config.ivf_routing
                        });
                    validate_ivf_tq_ann(index, centroids, codec, params.dim, routing, field)?;
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    search_ivf_tq_segment(
                        index,
                        centroids,
                        codec,
                        query,
                        fetch_k.min(flat.num_docs_with_vectors()),
                        needs_document_aggregation.then_some(combiner),
                        field,
                        params.nprobe,
                        routing,
                        plan_cache.map(|cache| &cache.ivf_tq),
                        ann_keys_are_unique(index, flat),
                    )?
                }
                VectorIndex::BinaryIvf(_) => {
                    // A float query cannot be served by a Hamming payload; say
                    // so instead of returning an empty result set.
                    return Err(Error::Query(format!(
                        "dense vector field '{}' is served by a binary IVF index; use BinaryDenseVectorQuery",
                        self.schema.get_field_name(field).unwrap_or("?")
                    )));
                }
                VectorIndex::ScannAh(lazy) => {
                    let artifact = self
                        .trained_vectors
                        .scann_artifacts
                        .get(&field.0)
                        .ok_or_else(|| {
                            Error::Schema(format!(
                                "ScaNN field {} has no loaded global artifact",
                                field.0
                            ))
                        })?;
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    search_scann_ah_segment(
                        lazy.get(),
                        artifact,
                        query,
                        fetch_k.min(flat.num_docs_with_vectors()),
                        combiner,
                        field,
                        params.nprobe,
                        plan_cache.map(|cache| &cache.scann),
                    )
                    .map(|candidates| (candidates, DenseAnnScanStats::default()))?
                }
                VectorIndex::ScannBinary(_) => {
                    return Err(Error::Corruption(format!(
                        "binary ScaNN payload was attached to float field {}",
                        field.0
                    )));
                }
            }
        } else if let Some(lazy_flat) = lazy_flat {
            // Batched brute-force from lazy flat vectors (native-precision SIMD scoring).
            // Combine every value of a document before document-level top-k;
            // vector-level top-k loses documents on multi-valued fields.
            log::debug!(
                "[dense_vector_search] index={} field {}: brute-force on {} vectors (dim={}, quant={:?})",
                self.schema.index_label(),
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
            let prepared_query = PreparedDenseScoreQuery::new(query, quant, dim, params.unit_norm)?;
            let mut flat_stats = DenseAnnScanStats {
                posting_count: n,
                ..DenseAnnScanStats::default()
            };
            let mut scratch = DenseScratch::take();
            scratch.prepare(0, batch_len);
            let scores = &mut scratch.scores;

            for batch_start in (0..n).step_by(batch_len) {
                let batch_count = batch_len.min(n - batch_start);
                let batch_bytes = lazy_flat
                    .read_vectors_batch(batch_start, batch_count)
                    .await
                    .map_err(crate::Error::Io)?;
                let raw = batch_bytes.as_slice();

                prepared_query.score_batch(raw, &mut scores[..batch_count])?;
                flat_stats.scored_blocks += 1;

                for (i, &score) in scores.iter().enumerate().take(batch_count) {
                    let (doc_id, ordinal) = lazy_flat.get_doc_id(batch_start + i);
                    collector.push(doc_id, ordinal, score);
                }
            }

            flat_results = Some(collector.into_results());
            (Vec::new(), flat_stats)
        } else {
            return Ok(Vec::new());
        };
        let l1_elapsed = t0.elapsed();
        {
            let kind = dense_ann_kind_label(ann_index);
            let field_name = self.schema.get_field_name(field).unwrap_or("?");
            crate::observe::dense_l1(
                self.schema.index_label(),
                field_name,
                kind,
                l1_elapsed.as_secs_f64(),
                flat_results.as_ref().map_or(results.len(), Vec::len),
            );
            crate::observe::dense_ann_scan(self.schema.index_label(), field_name, kind, scan_stats);
            crate::observe::warn_non_finite_dense_scores(
                self.schema.index_label(),
                field_name,
                kind,
                scan_stats.non_finite_dropped,
            );
        }
        log::debug!(
            "[dense_vector_search] index={} field {}: L1 returned {} candidates in {:.1}ms",
            self.schema.index_label(),
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
            let vbs = lazy_flat.vector_byte_size();
            let (reranked, stats) = exact_score_dense_candidate_documents(
                &results,
                lazy_flat,
                query,
                params.unit_norm,
                combiner,
                k,
            )
            .await?;

            crate::observe::dense_rerank(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                t_rerank.elapsed().as_secs_f64(),
                stats.resolve_elapsed.as_secs_f64(),
                stats.read_elapsed.as_secs_f64(),
                stats.vector_count,
            );
            log::debug!(
                "[dense_vector_search] index={} field {}: rerank {} vectors (dim={}, quant={:?}, bytes_per_vector={}): resolve={:.1}ms read={:.1}ms score={:.1}ms",
                self.schema.index_label(),
                field.0,
                stats.vector_count,
                lazy_flat.dim,
                lazy_flat.quantization,
                vbs,
                stats.resolve_elapsed.as_secs_f64() * 1000.0,
                stats.read_elapsed.as_secs_f64() * 1000.0,
                stats.score_elapsed.as_secs_f64() * 1000.0,
            );

            log::debug!(
                "[dense_vector_search] index={} field {}: rerank total={:.1}ms",
                self.schema.index_label(),
                field.0,
                t_rerank.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(reranked);
        }

        Ok(combine_grouped_ordinal_results(results, combiner, k))
    }

    /// Search binary dense vectors using IVF when available, otherwise
    /// brute-force Hamming distance.
    ///
    /// Returns VectorSearchResult with ordinal tracking.
    async fn search_binary_dense_vector_impl(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
        probe_cache: Option<&std::sync::Mutex<Option<crate::structures::IvfProbePlan>>>,
    ) -> Result<Vec<VectorSearchResult>> {
        let schema_dim = self.validate_binary_search_request(field, query)?;
        combiner.validate().map_err(Error::Query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let t0 = crate::observe::Timer::start();
        if let Some(VectorIndex::ScannBinary(lazy)) = self.vector_indexes.get(&field.0) {
            let artifact = self
                .trained_vectors
                .scann_artifacts
                .get(&field.0)
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "binary ScaNN field {} has no loaded global artifact",
                        field.0
                    ))
                })?;
            lazy.get()
                .validate_scann_generation(
                    artifact.config(),
                    artifact.generation(),
                    artifact.artifact_id(),
                )
                .map_err(|error| {
                    Error::Corruption(format!(
                        "binary ScaNN generation mismatch for field {}: {error}",
                        field.0
                    ))
                })?;
            let config = self
                .schema
                .get_field_entry(field)
                .and_then(|entry| entry.binary_dense_vector_config.as_ref())
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "binary ScaNN field {} has no schema configuration",
                        field.0
                    ))
                })?;
            let model = artifact.binary_model().map_err(Error::Io)?;
            let clusters = binary_scann_probe_clusters(&model, query, config.nprobe, probe_cache)?;
            let flat = self.flat_vectors.get(&field.0).ok_or_else(|| {
                Error::Corruption(format!(
                    "binary ScaNN field {} is missing flat vectors",
                    field.0
                ))
            })?;
            let candidate_limit =
                checked_binary_combined_fetch_k(k)?.min(flat.num_docs_with_vectors());
            let (documents, ordinal_scores) = lazy
                .get()
                .search_binary_combined_documents(candidate_limit, query, &clusters, combiner)
                .map_err(|error| {
                    Error::Corruption(format!(
                        "invalid binary ScaNN payload for field {}: {error}",
                        field.0
                    ))
                })?;
            let results = exact_score_binary_candidate_document_ids(
                documents
                    .into_iter()
                    .map(|candidate| candidate.doc_id)
                    .collect(),
                &ordinal_scores,
                flat,
                query,
                schema_dim,
                combiner,
                k,
            )
            .await?;
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "binary_scann",
                t0.secs(),
                results.len(),
            );
            return Ok(results);
        }
        if let Some(VectorIndex::BinaryIvf(lazy)) = self.vector_indexes.get(&field.0) {
            let ivf = lazy.get();
            let config = self
                .schema
                .get_field_entry(field)
                .and_then(|entry| entry.binary_dense_vector_config.as_ref())
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "binary IVF field {} has no schema configuration",
                        field.0
                    ))
                })?;
            let quantizer = self
                .trained_vectors
                .binary_quantizers
                .get(&field.0)
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "global binary IVF field {} has no loaded quantizer",
                        field.0
                    ))
                })?;
            validate_binary_ann(ivf, quantizer, config, schema_dim, field)?;
            let flat = self.flat_vectors.get(&field.0).ok_or_else(|| {
                Error::Corruption(format!(
                    "global binary IVF field {} is missing flat vector storage",
                    field.0
                ))
            })?;
            let single_valued = flat.num_vectors == flat.num_docs_with_vectors();
            let clusters = binary_probe_clusters(
                quantizer,
                query,
                config.nprobe,
                config.ivf_routing,
                probe_cache,
            )?;
            let results = if !single_valued
                && !matches!(combiner, crate::query::MultiValueCombiner::Max)
            {
                let candidate_limit =
                    checked_binary_combined_fetch_k(k)?.min(flat.num_docs_with_vectors());
                let (candidate_documents, probed_ordinal_scores) = ivf
                    .search_binary_combined_documents(candidate_limit, query, &clusters, combiner)
                    .map_err(|error| {
                        Error::Corruption(format!(
                            "invalid binary IVF payload for field {}: {error}",
                            field.0,
                        ))
                    })?;
                exact_score_binary_candidate_document_ids(
                    candidate_documents
                        .into_iter()
                        .map(|candidate| candidate.doc_id)
                        .collect(),
                    &probed_ordinal_scores,
                    flat,
                    query,
                    schema_dim,
                    combiner,
                    k,
                )
                .await?
            } else {
                let candidate_docs = if single_valued {
                    k
                } else {
                    // Completing the selected documents from flat storage can
                    // reorder a multi-value Max result when another ordinal
                    // lives outside the probed leaves. Keep the same bounded
                    // oversubscription used by combined binary reranking.
                    checked_binary_combined_fetch_k(k)?
                }
                .min(flat.num_docs_with_vectors());
                let ann_results = if single_valued {
                    ivf.search_binary_clusters::<false>(query, candidate_docs, &clusters)
                } else {
                    ivf.search_binary_clusters::<true>(query, candidate_docs, &clusters)
                }
                .map_err(|error| {
                    Error::Corruption(format!(
                        "invalid binary IVF payload for field {}: {error}",
                        field.0,
                    ))
                })?;
                // Binary IVF stores the original packed codes, so its leaf
                // scores are already exact for a single-valued field.
                if single_valued {
                    let ann_results = validate_binary_single_value_ann_results(ann_results, flat)?;
                    combine_ordinal_results(ann_results, combiner, k)
                } else {
                    exact_score_binary_candidate_documents(
                        &ann_results,
                        flat,
                        query,
                        schema_dim,
                        combiner,
                        k,
                    )
                    .await?
                }
            };
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "global_binary_ivf",
                t0.secs(),
                results.len(),
            );
            return Ok(results);
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
        let mut scratch = DenseScratch::take();
        scratch.prepare(0, batch_len);
        let scores = &mut scratch.scores;

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

    pub async fn search_binary_dense_vector(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_binary_dense_vector_impl(field, query, k, combiner, None)
            .await
    }

    pub(crate) async fn search_binary_dense_vector_with_probe_cache(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
        probe_cache: &std::sync::Mutex<Option<crate::structures::IvfProbePlan>>,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_binary_dense_vector_impl(field, query, k, combiner, Some(probe_cache))
            .await
    }

    /// Get coarse centroids for a field.
    pub fn coarse_centroids(&self, field_id: u32) -> Option<&Arc<CoarseCentroids>> {
        self.trained_vectors.centroids.get(&field_id)
    }

    pub fn set_trained_vectors(
        &mut self,
        trained_vectors: Arc<crate::segment::TrainedVectorStructures>,
    ) {
        self.trained_vectors = trained_vectors;
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
        self.search_dense_vector_sync_impl(field, query, k, nprobe, rerank_factor, combiner, None)
    }

    #[cfg(feature = "sync")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_dense_vector_sync_with_probe_cache(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
        plan_cache: &DensePlanCache,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_dense_vector_sync_impl(
            field,
            query,
            k,
            nprobe,
            rerank_factor,
            combiner,
            Some(plan_cache),
        )
    }

    #[cfg(feature = "sync")]
    #[allow(clippy::too_many_arguments)]
    fn search_dense_vector_sync_impl(
        &self,
        field: Field,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: f32,
        combiner: crate::query::MultiValueCombiner,
        plan_cache: Option<&DensePlanCache>,
    ) -> Result<Vec<VectorSearchResult>> {
        let params =
            self.validate_dense_search_request(field, query, nprobe, rerank_factor, combiner)?;
        let fetch_k = checked_dense_fetch_k(k, rerank_factor)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let configured_ann_index = self.vector_indexes.get(&field.0);
        let lazy_flat = self.flat_vectors.get(&field.0);
        if configured_ann_index.is_none() && lazy_flat.is_none() {
            return Ok(Vec::new());
        }

        if configured_ann_index.is_some() && lazy_flat.is_none() {
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

        let needs_document_aggregation = lazy_flat.is_some_and(|flat| {
            flat.num_vectors != flat.num_docs_with_vectors()
                && !matches!(combiner, crate::query::MultiValueCombiner::Max)
        });
        // Sync and async search share the same ANN candidate modes; neither
        // silently substitutes a raw flat scan for an indexed field.
        let ann_index = configured_ann_index;

        let (results, scan_stats): (Vec<(u32, u16, f32)>, DenseAnnScanStats) = if let Some(index) =
            ann_index
        {
            // ANN search (already sync)
            match index {
                VectorIndex::Tq { index: lazy, codec } => {
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    search_tq_segment(
                        lazy.get(),
                        codec,
                        query,
                        fetch_k.min(flat.num_docs_with_vectors()),
                        needs_document_aggregation.then_some(combiner),
                        field,
                        params.dim,
                        plan_cache.map(|cache| &cache.tq),
                        ann_keys_are_unique(lazy.get(), flat),
                    )?
                }
                VectorIndex::IvfTq { index: lazy, codec } => {
                    let index = lazy.get();
                    let centroids =
                        self.trained_vectors
                            .centroids
                            .get(&field.0)
                            .ok_or_else(|| {
                                Error::Schema(format!(
                                    "IVF-TQ index requires coarse centroids for field {}",
                                    field.0
                                ))
                            })?;
                    validate_coarse_centroids(centroids, params.dim)?;
                    let routing = self
                        .schema
                        .get_field_entry(field)
                        .and_then(|entry| entry.dense_vector_config.as_ref())
                        .map_or(crate::dsl::IvfRoutingMode::Auto, |config| {
                            config.ivf_routing
                        });
                    validate_ivf_tq_ann(index, centroids, codec, params.dim, routing, field)?;
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    search_ivf_tq_segment(
                        index,
                        centroids,
                        codec,
                        query,
                        fetch_k.min(flat.num_docs_with_vectors()),
                        needs_document_aggregation.then_some(combiner),
                        field,
                        params.nprobe,
                        routing,
                        plan_cache.map(|cache| &cache.ivf_tq),
                        ann_keys_are_unique(index, flat),
                    )?
                }
                VectorIndex::BinaryIvf(_) => {
                    // A float query cannot be served by a Hamming payload; say
                    // so instead of returning an empty result set.
                    return Err(Error::Query(format!(
                        "dense vector field '{}' is served by a binary IVF index; use BinaryDenseVectorQuery",
                        self.schema.get_field_name(field).unwrap_or("?")
                    )));
                }
                VectorIndex::ScannAh(lazy) => {
                    let artifact = self
                        .trained_vectors
                        .scann_artifacts
                        .get(&field.0)
                        .ok_or_else(|| {
                            Error::Schema(format!(
                                "ScaNN field {} has no loaded global artifact",
                                field.0
                            ))
                        })?;
                    let flat = lazy_flat.expect("ANN/flat pairing validated above");
                    search_scann_ah_segment(
                        lazy.get(),
                        artifact,
                        query,
                        fetch_k.min(flat.num_docs_with_vectors()),
                        combiner,
                        field,
                        params.nprobe,
                        plan_cache.map(|cache| &cache.scann),
                    )
                    .map(|candidates| (candidates, DenseAnnScanStats::default()))?
                }
                VectorIndex::ScannBinary(_) => {
                    return Err(Error::Corruption(format!(
                        "binary ScaNN payload was attached to float field {}",
                        field.0
                    )));
                }
            }
        } else if let Some(lazy_flat) = lazy_flat {
            // Batched brute-force (sync mmap reads), parallel on large segments.
            let (results, flat_stats) = brute_force_flat_scan_sync(
                lazy_flat,
                query,
                params.unit_norm,
                fetch_k.min(lazy_flat.num_vectors),
                combiner,
            )?;
            crate::observe::dense_ann_scan(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "flat",
                flat_stats,
            );
            return Ok(results);
        } else {
            return Ok(Vec::new());
        };
        {
            let kind = dense_ann_kind_label(ann_index);
            let field_name = self.schema.get_field_name(field).unwrap_or("?");
            crate::observe::dense_ann_scan(self.schema.index_label(), field_name, kind, scan_stats);
            crate::observe::warn_non_finite_dense_scores(
                self.schema.index_label(),
                field_name,
                kind,
                scan_stats.non_finite_dropped,
            );
        }

        // Rerank ANN candidates using raw vectors (sync)
        if ann_index.is_some()
            && !results.is_empty()
            && let Some(lazy_flat) = lazy_flat
        {
            return exact_score_dense_candidate_documents_sync(
                &results,
                lazy_flat,
                query,
                params.unit_norm,
                combiner,
                k,
            );
        }

        Ok(combine_grouped_ordinal_results(results, combiner, k))
    }

    /// Synchronous binary dense vector search (mmap/RAM only).
    ///
    /// Mirrors [`Self::search_binary_dense_vector`] for the rayon-parallel
    /// sync scorer path used by multi-threaded runtimes.
    #[cfg(feature = "sync")]
    fn search_binary_dense_vector_sync_impl(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
        probe_cache: Option<&std::sync::Mutex<Option<crate::structures::IvfProbePlan>>>,
    ) -> Result<Vec<VectorSearchResult>> {
        let schema_dim = self.validate_binary_search_request(field, query)?;
        combiner.validate().map_err(Error::Query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let t0 = crate::observe::Timer::start();
        if let Some(VectorIndex::ScannBinary(lazy)) = self.vector_indexes.get(&field.0) {
            let artifact = self
                .trained_vectors
                .scann_artifacts
                .get(&field.0)
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "binary ScaNN field {} has no loaded global artifact",
                        field.0
                    ))
                })?;
            lazy.get()
                .validate_scann_generation(
                    artifact.config(),
                    artifact.generation(),
                    artifact.artifact_id(),
                )
                .map_err(|error| {
                    Error::Corruption(format!(
                        "binary ScaNN generation mismatch for field {}: {error}",
                        field.0
                    ))
                })?;
            let config = self
                .schema
                .get_field_entry(field)
                .and_then(|entry| entry.binary_dense_vector_config.as_ref())
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "binary ScaNN field {} has no schema configuration",
                        field.0
                    ))
                })?;
            let model = artifact.binary_model().map_err(Error::Io)?;
            let clusters = binary_scann_probe_clusters(&model, query, config.nprobe, probe_cache)?;
            let flat = self.flat_vectors.get(&field.0).ok_or_else(|| {
                Error::Corruption(format!(
                    "binary ScaNN field {} is missing flat vectors",
                    field.0
                ))
            })?;
            let candidate_limit =
                checked_binary_combined_fetch_k(k)?.min(flat.num_docs_with_vectors());
            let (documents, ordinal_scores) = lazy
                .get()
                .search_binary_combined_documents(candidate_limit, query, &clusters, combiner)
                .map_err(|error| {
                    Error::Corruption(format!(
                        "invalid binary ScaNN payload for field {}: {error}",
                        field.0
                    ))
                })?;
            let results = exact_score_binary_candidate_document_ids_sync(
                documents
                    .into_iter()
                    .map(|candidate| candidate.doc_id)
                    .collect(),
                &ordinal_scores,
                flat,
                query,
                schema_dim,
                combiner,
                k,
            )?;
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "binary_scann",
                t0.secs(),
                results.len(),
            );
            return Ok(results);
        }
        if let Some(VectorIndex::BinaryIvf(lazy)) = self.vector_indexes.get(&field.0) {
            let ivf = lazy.get();
            let config = self
                .schema
                .get_field_entry(field)
                .and_then(|entry| entry.binary_dense_vector_config.as_ref())
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "binary IVF field {} has no schema configuration",
                        field.0
                    ))
                })?;
            let quantizer = self
                .trained_vectors
                .binary_quantizers
                .get(&field.0)
                .ok_or_else(|| {
                    Error::Schema(format!(
                        "global binary IVF field {} has no loaded quantizer",
                        field.0
                    ))
                })?;
            validate_binary_ann(ivf, quantizer, config, schema_dim, field)?;
            let flat = self.flat_vectors.get(&field.0).ok_or_else(|| {
                Error::Corruption(format!(
                    "global binary IVF field {} is missing flat vector storage",
                    field.0
                ))
            })?;
            let single_valued = flat.num_vectors == flat.num_docs_with_vectors();
            let clusters = binary_probe_clusters(
                quantizer,
                query,
                config.nprobe,
                config.ivf_routing,
                probe_cache,
            )?;
            let results = if !single_valued
                && !matches!(combiner, crate::query::MultiValueCombiner::Max)
            {
                let candidate_limit =
                    checked_binary_combined_fetch_k(k)?.min(flat.num_docs_with_vectors());
                let (candidate_documents, probed_ordinal_scores) = ivf
                    .search_binary_combined_documents(candidate_limit, query, &clusters, combiner)
                    .map_err(|error| {
                        Error::Corruption(format!(
                            "invalid binary IVF payload for field {}: {error}",
                            field.0,
                        ))
                    })?;
                exact_score_binary_candidate_document_ids_sync(
                    candidate_documents
                        .into_iter()
                        .map(|candidate| candidate.doc_id)
                        .collect(),
                    &probed_ordinal_scores,
                    flat,
                    query,
                    schema_dim,
                    combiner,
                    k,
                )?
            } else {
                let candidate_docs = if single_valued {
                    k
                } else {
                    checked_binary_combined_fetch_k(k)?
                }
                .min(flat.num_docs_with_vectors());
                let ann_results = if single_valued {
                    ivf.search_binary_clusters::<false>(query, candidate_docs, &clusters)
                } else {
                    ivf.search_binary_clusters::<true>(query, candidate_docs, &clusters)
                }
                .map_err(|error| {
                    Error::Corruption(format!(
                        "invalid binary IVF payload for field {}: {error}",
                        field.0,
                    ))
                })?;
                if single_valued {
                    let ann_results = validate_binary_single_value_ann_results(ann_results, flat)?;
                    combine_ordinal_results(ann_results, combiner, k)
                } else {
                    exact_score_binary_candidate_documents_sync(
                        &ann_results,
                        flat,
                        query,
                        schema_dim,
                        combiner,
                        k,
                    )?
                }
            };
            crate::observe::dense_l1(
                self.schema.index_label(),
                self.schema.get_field_name(field).unwrap_or("?"),
                "global_binary_ivf",
                t0.secs(),
                results.len(),
            );
            return Ok(results);
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
        let mut scratch = DenseScratch::take();
        scratch.prepare(0, batch_len);
        let scores = &mut scratch.scores;

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

    #[cfg(feature = "sync")]
    pub fn search_binary_dense_vector_sync(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_binary_dense_vector_sync_impl(field, query, k, combiner, None)
    }

    #[cfg(feature = "sync")]
    pub(crate) fn search_binary_dense_vector_sync_with_probe_cache(
        &self,
        field: Field,
        query: &[u8],
        k: usize,
        combiner: crate::query::MultiValueCombiner,
        probe_cache: &std::sync::Mutex<Option<crate::structures::IvfProbePlan>>,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_binary_dense_vector_sync_impl(field, query, k, combiner, Some(probe_cache))
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
            2.01,
            MAX_DENSE_RERANK_FACTOR + 1.0,
        ] {
            assert!(
                checked_dense_fetch_k(10, factor).is_err(),
                "factor={factor}"
            );
        }
    }

    fn values_as_bytes<T>(values: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        }
    }

    fn assert_prepared_dense_scores_match_legacy(
        quantization: DenseVectorQuantization,
        raw: &[u8],
        unit_norm: bool,
    ) {
        const DIM: usize = 4;
        const VECTOR_COUNT: usize = 4;
        let query = [0.25, -0.5, 0.75, 1.0];
        let mut expected = [0.0; VECTOR_COUNT];
        SegmentReader::score_quantized_batch_legacy(
            &query,
            raw,
            quantization,
            DIM,
            &mut expected,
            unit_norm,
        )
        .unwrap();

        let prepared = PreparedDenseScoreQuery::new(&query, quantization, DIM, unit_norm).unwrap();
        let vector_bytes = DIM
            * match quantization {
                DenseVectorQuantization::F32 => std::mem::size_of::<f32>(),
                DenseVectorQuantization::F16 => std::mem::size_of::<u16>(),
                DenseVectorQuantization::UInt8 => 1,
                DenseVectorQuantization::Binary => unreachable!(),
            };
        let split = 2 * vector_bytes;
        let mut actual = [0.0; VECTOR_COUNT];
        prepared
            .score_batch(&raw[..split], &mut actual[..2])
            .unwrap();
        prepared
            .score_batch(&raw[split..], &mut actual[2..])
            .unwrap();

        assert_eq!(
            actual.map(f32::to_bits),
            expected.map(f32::to_bits),
            "quantization={quantization:?}, unit_norm={unit_norm}"
        );
    }

    #[test]
    fn prepared_dense_query_matches_legacy_scoring_across_batches() {
        let vectors_f32 = [
            0.5, -0.25, 0.75, 1.0, -1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0, 0.75, 0.5, -0.5,
            -0.25,
        ];
        let vectors_f16: Vec<u16> = vectors_f32
            .iter()
            .map(|&value| crate::structures::simd::f32_to_f16(value))
            .collect();
        let vectors_u8 = [
            255, 96, 224, 160, 0, 192, 144, 128, 128, 128, 128, 128, 224, 192, 64, 96,
        ];

        for unit_norm in [false, true] {
            assert_prepared_dense_scores_match_legacy(
                DenseVectorQuantization::F32,
                values_as_bytes(&vectors_f32),
                unit_norm,
            );
            assert_prepared_dense_scores_match_legacy(
                DenseVectorQuantization::F16,
                values_as_bytes(&vectors_f16),
                unit_norm,
            );
            assert_prepared_dense_scores_match_legacy(
                DenseVectorQuantization::UInt8,
                &vectors_u8,
                unit_norm,
            );
        }
    }

    #[test]
    fn prepared_dense_query_preserves_scoring_validation_errors() {
        assert!(matches!(
            PreparedDenseScoreQuery::new(&[1.0], DenseVectorQuantization::F32, 2, false).err(),
            Some(Error::Query(_))
        ));
        assert!(matches!(
            PreparedDenseScoreQuery::new(&[1.0], DenseVectorQuantization::Binary, 1, false).err(),
            Some(Error::InvalidFieldType { .. })
        ));

        let query = [1.0, 2.0];
        let prepared =
            PreparedDenseScoreQuery::new(&query, DenseVectorQuantization::F32, 2, false).unwrap();
        let mut scores = [0.0];
        assert!(matches!(
            prepared.score_batch(&[0; 7], &mut scores),
            Err(Error::Corruption(_))
        ));
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
        assert_eq!(checked_dense_fetch_k(10_000, 2.0).unwrap(), 20_000);
        assert!(checked_dense_fetch_k(10_001, 2.0).is_err());
        assert!(checked_dense_fetch_k(usize::MAX, 2.0).is_err());
    }

    #[test]
    fn binary_combined_fetch_count_uses_shared_bounded_oversampling() {
        assert_eq!(checked_binary_combined_fetch_k(3).unwrap(), 6);
        assert_eq!(checked_binary_combined_fetch_k(10_000).unwrap(), 20_000);
        assert_eq!(checked_binary_combined_fetch_k(10_001).unwrap(), 20_000);
        assert_eq!(checked_binary_combined_fetch_k(20_000).unwrap(), 20_000);
        assert!(checked_binary_combined_fetch_k(20_001).is_err());
        assert!(checked_binary_combined_fetch_k(usize::MAX).is_err());
    }

    #[cfg(feature = "native")]
    #[test]
    fn legacy_ivf_tq_generation_is_rejected_while_opening() {
        use crate::directories::OwnedBytes;
        use crate::dsl::IvfRoutingMode;
        use crate::segment::ann_disk::{AnnDiskIndex, AnnKind};

        let centroids = CoarseCentroids {
            num_clusters: 1,
            dim: 2,
            centroids: vec![1.0, 0.0],
            version: 7,
            soar_config: None,
            routing_index: None,
        };
        let mut build_centroids = centroids.clone();
        build_centroids.version =
            crate::structures::mark_ivf_tq_cosine_generation(build_centroids.version);
        let mut bytes = crate::segment::ann_build::build_ivf_tq(
            2,
            IvfRoutingMode::Flat,
            &build_centroids,
            &[(0, 0)],
            &[1.0, 0.0],
        )
        .unwrap();
        // Rewrite only the in-band centroid generation in the header to model
        // a persisted pre-cosine artifact.
        bytes[24..32].copy_from_slice(&centroids.version.to_le_bytes());
        let error = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::IvfTq, 1)
            .err()
            .expect("legacy IVF-TQ payload must fail while opening")
            .to_string();
        assert!(error.contains("unsupported legacy generation"), "{error}");
    }

    #[test]
    fn rerank_batch_is_capped_by_actual_candidate_vectors() {
        assert_eq!(bounded_rerank_batch(3_072, DENSE_SCORE_BATCH, 20), 20);
        assert_eq!(
            bounded_rerank_batch(3_072, DENSE_SCORE_BATCH, 10_000),
            MAX_VECTOR_SCORE_BATCH_BYTES / 3_072
        );
        assert_eq!(bounded_rerank_batch(3_072, DENSE_SCORE_BATCH, 0), 1);
    }

    #[test]
    fn file_ranges_reject_overflow_and_truncation() {
        assert_eq!(checked_file_range(4, 3, 7, "test").unwrap(), 4..7);
        assert!(checked_file_range(u64::MAX, 1, u64::MAX, "test").is_err());
        assert!(checked_file_range(5, 3, 7, "test").is_err());
    }

    #[test]
    fn shared_tq_plan_cache_rebuilds_for_divergent_query_clones() {
        let codec = crate::structures::TqCodec::new(4);
        let cache = std::sync::Mutex::new(None);
        let original_query = vec![1.0, 2.0, 3.0, 4.0];

        let original =
            cached_tq_query_plan(&codec, &original_query, Some(&cache)).expect("build plan");
        let reused =
            cached_tq_query_plan(&codec, &original_query, Some(&cache)).expect("reuse plan");
        assert!(
            std::sync::Arc::ptr_eq(&original, &reused),
            "unchanged queries must share their plan across segments"
        );

        let mut divergent_clone = original_query.clone();
        divergent_clone[0] = -1.0;
        let rebuilt =
            cached_tq_query_plan(&codec, &divergent_clone, Some(&cache)).expect("rebuild plan");
        assert!(
            !std::sync::Arc::ptr_eq(&original, &rebuilt),
            "a clone with a mutated vector must not reuse stale LUTs"
        );
        assert!(rebuilt.matches_query(&divergent_clone));
        assert!(!rebuilt.matches_query(&original_query));
    }

    #[test]
    fn candidate_vector_reads_coalesce_contiguous_values() {
        let mut runs = Vec::new();
        plan_vector_read_runs(&[3, 4, 5, 9, 12, 13], &mut runs).unwrap();
        assert_eq!(runs.len(), 3);
        assert!(matches!(
            runs.as_slice(),
            [
                VectorReadRun {
                    buffer_start: 0,
                    flat_start: 3,
                    count: 3,
                },
                VectorReadRun {
                    buffer_start: 3,
                    flat_start: 9,
                    count: 1,
                },
                VectorReadRun {
                    buffer_start: 4,
                    flat_start: 12,
                    count: 2,
                },
            ]
        ));
        assert!(plan_vector_read_runs(&[3, 3], &mut runs).is_err());
    }

    #[tokio::test]
    async fn binary_single_value_ann_fast_path_validates_and_deduplicates() {
        use crate::directories::{FileHandle, OwnedBytes};
        use crate::segment::FlatVectorData;

        let mut encoded = Vec::new();
        FlatVectorData::serialize_binary_from_bits_streaming(
            8,
            &[0x0f, 0xf0],
            &[(1, 0), (3, 2)],
            &mut encoded,
        )
        .unwrap();
        let flat = LazyFlatVectorData::open_with_doc_limit(
            FileHandle::from_bytes(OwnedBytes::new(encoded)),
            Some(4),
        )
        .await
        .unwrap();
        assert_eq!(flat.num_vectors, flat.num_docs_with_vectors());

        let validated = validate_binary_single_value_ann_results(
            vec![(3, 2, 0.9), (1, 0, 0.8), (3, 2, 0.7)],
            &flat,
        )
        .unwrap();
        assert_eq!(validated, vec![(3, 2, 0.9), (1, 0, 0.8)]);

        assert!(matches!(
            validate_binary_single_value_ann_results(vec![(2, 0, 1.0)], &flat),
            Err(Error::Corruption(_))
        ));
        assert!(matches!(
            validate_binary_single_value_ann_results(vec![(3, 0, 1.0)], &flat),
            Err(Error::Corruption(_))
        ));
    }

    #[tokio::test]
    async fn multivalue_ann_rerank_streams_past_document_candidate_cap() {
        use crate::directories::{FileHandle, OwnedBytes};
        use crate::segment::FlatVectorData;

        const VALUES: usize = MAX_DENSE_CANDIDATES_PER_SEGMENT + 1;
        let mut encoded = Vec::new();
        let vectors = vec![1.0f32; VALUES];
        let doc_ids: Vec<_> = (0..VALUES).map(|ordinal| (0, ordinal as u16)).collect();
        FlatVectorData::serialize_binary_from_flat_streaming(
            1,
            &vectors,
            &doc_ids,
            DenseVectorQuantization::F32,
            &mut encoded,
        )
        .unwrap();
        let flat = LazyFlatVectorData::open_with_doc_limit(
            FileHandle::from_bytes(OwnedBytes::new(encoded)),
            Some(1),
        )
        .await
        .unwrap();

        let (results, stats) = exact_score_dense_candidate_documents(
            &[(0, 0, 0.0)],
            &flat,
            &[1.0],
            false,
            crate::query::MultiValueCombiner::Max,
            1,
        )
        .await
        .unwrap();
        assert_eq!(stats.vector_count, VALUES);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].ordinals.len(), VALUES);
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }
}
