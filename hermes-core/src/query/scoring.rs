//! Shared scoring abstractions for text and sparse vector search
//!
//! Provides common traits and utilities for efficient top-k retrieval:
//! - `ScoringIterator`: Common interface for posting list iteration with scoring
//! - `TopKCollector`: Efficient min-heap for maintaining top-k results
//! - `BlockMaxScoreExecutor`: Unified Block-Max MaxScore with conjunction optimization
//! - `BmpExecutor`: Block-at-a-time executor for learned sparse retrieval (12+ terms)
//! - `SparseTermScorer`: ScoringIterator implementation for sparse vectors

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use log::{debug, trace};

use crate::DocId;
use crate::structures::BlockSparsePostingList;

/// Common interface for scoring iterators (text terms or sparse dimensions)
///
/// Abstracts the common operations needed for WAND-style top-k retrieval.
pub trait ScoringIterator {
    /// Current document ID (u32::MAX if exhausted)
    fn doc(&self) -> DocId;

    /// Current ordinal for multi-valued fields (default 0)
    fn ordinal(&mut self) -> u16 {
        0
    }

    /// Advance to next document, returns new doc ID
    fn advance(&mut self) -> DocId;

    /// Seek to first document >= target, returns new doc ID
    fn seek(&mut self, target: DocId) -> DocId;

    /// Check if iterator is exhausted
    fn is_exhausted(&self) -> bool {
        self.doc() == u32::MAX
    }

    /// Score contribution for current document
    fn score(&self) -> f32;

    /// Maximum possible score for this term/dimension (global upper bound)
    fn max_score(&self) -> f32;

    /// Current block's maximum score upper bound (for block-level pruning)
    fn current_block_max_score(&self) -> f32;

    /// Skip to the next block, returning the first doc_id in the new block.
    /// Used for block-max WAND optimization when current block can't beat threshold.
    /// Default implementation just advances (no block-level skipping).
    fn skip_to_next_block(&mut self) -> DocId {
        self.advance()
    }
}

/// Entry for top-k min-heap
#[derive(Clone, Copy)]
pub struct HeapEntry {
    pub doc_id: DocId,
    pub score: f32,
    pub ordinal: u16,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower scores come first (to be evicted)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Efficient top-k collector using min-heap
///
/// Maintains the k highest-scoring documents using a min-heap where the
/// lowest score is at the top for O(1) threshold lookup and O(log k) eviction.
/// No deduplication - caller must ensure each doc_id is inserted only once.
pub struct ScoreCollector {
    /// Min-heap of top-k entries (lowest score at top for eviction)
    heap: BinaryHeap<HeapEntry>,
    pub k: usize,
    /// Cached threshold: avoids repeated heap.peek() in hot loops.
    /// Updated only when the heap changes (insert/pop).
    cached_threshold: f32,
}

impl ScoreCollector {
    /// Create a new collector for top-k results
    pub fn new(k: usize) -> Self {
        // Cap capacity to avoid allocation overflow for very large k
        let capacity = k.saturating_add(1).min(1_000_000);
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            k,
            cached_threshold: 0.0,
        }
    }

    /// Current score threshold (minimum score to enter top-k)
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.cached_threshold
    }

    /// Recompute cached threshold from heap state
    #[inline]
    fn update_threshold(&mut self) {
        self.cached_threshold = if self.heap.len() >= self.k {
            self.heap.peek().map(|e| e.score).unwrap_or(0.0)
        } else {
            0.0
        };
    }

    /// Insert a document score. Returns true if inserted in top-k.
    /// Caller must ensure each doc_id is inserted only once.
    #[inline]
    pub fn insert(&mut self, doc_id: DocId, score: f32) -> bool {
        self.insert_with_ordinal(doc_id, score, 0)
    }

    /// Insert a document score with ordinal. Returns true if inserted in top-k.
    /// Caller must ensure each doc_id is inserted only once.
    #[inline]
    pub fn insert_with_ordinal(&mut self, doc_id: DocId, score: f32, ordinal: u16) -> bool {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry {
                doc_id,
                score,
                ordinal,
            });
            self.update_threshold();
            true
        } else if score > self.cached_threshold {
            self.heap.push(HeapEntry {
                doc_id,
                score,
                ordinal,
            });
            self.heap.pop(); // Remove lowest
            self.update_threshold();
            true
        } else {
            false
        }
    }

    /// Check if a score could potentially enter top-k
    #[inline]
    pub fn would_enter(&self, score: f32) -> bool {
        self.heap.len() < self.k || score > self.cached_threshold
    }

    /// Get number of documents collected so far
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if collector is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Convert to sorted top-k results (descending by score)
    pub fn into_sorted_results(self) -> Vec<(DocId, f32, u16)> {
        let heap_vec = self.heap.into_vec();
        let mut results: Vec<(DocId, f32, u16)> = Vec::with_capacity(heap_vec.len());
        for e in heap_vec {
            results.push((e.doc_id, e.score, e.ordinal));
        }

        // Sort by score descending, then doc_id ascending
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        results
    }
}

/// Search result from WAND execution
#[derive(Debug, Clone, Copy)]
pub struct ScoredDoc {
    pub doc_id: DocId,
    pub score: f32,
    /// Ordinal for multi-valued fields (which vector in the field matched)
    pub ordinal: u16,
}

/// Unified Block-Max MaxScore executor for top-k retrieval
///
/// Combines three optimizations from the literature into one executor:
/// 1. **MaxScore partitioning** (Turtle & Flood 1995): terms split into essential
///    (must check) and non-essential (only scored if candidate is promising)
/// 2. **Block-max pruning** (Ding & Suel 2011): skip blocks where per-block
///    upper bounds can't beat the current threshold
/// 3. **Conjunction optimization** (Lucene/Grand 2023): progressively intersect
///    essential terms as threshold rises, skipping docs that lack enough terms
///
/// Works with any type implementing `ScoringIterator` (text or sparse).
/// Replaces separate WAND and MaxScore executors with better performance
/// across all query lengths.
pub struct BlockMaxScoreExecutor<S: ScoringIterator> {
    /// Scorers sorted by max_score ascending (non-essential first)
    scorers: Vec<S>,
    /// Cumulative max_score prefix sums: prefix_sums[i] = sum(max_score[0..=i])
    prefix_sums: Vec<f32>,
    /// Top-k collector
    collector: ScoreCollector,
    /// Heap factor for approximate search (SEISMIC-style)
    /// - 1.0 = exact search (default)
    /// - 0.8 = approximate, faster with minor recall loss
    heap_factor: f32,
}

/// Backwards-compatible alias for `BlockMaxScoreExecutor`
pub type WandExecutor<S> = BlockMaxScoreExecutor<S>;

impl<S: ScoringIterator> BlockMaxScoreExecutor<S> {
    /// Create a new executor with exact search (heap_factor = 1.0)
    pub fn new(scorers: Vec<S>, k: usize) -> Self {
        Self::with_heap_factor(scorers, k, 1.0)
    }

    /// Create a new executor with approximate search
    ///
    /// `heap_factor` controls the trade-off between speed and recall:
    /// - 1.0 = exact search
    /// - 0.8 = ~20% faster, minor recall loss
    /// - 0.5 = much faster, noticeable recall loss
    pub fn with_heap_factor(mut scorers: Vec<S>, k: usize, heap_factor: f32) -> Self {
        // Sort scorers by max_score ascending (non-essential terms first)
        scorers.sort_by(|a, b| {
            a.max_score()
                .partial_cmp(&b.max_score())
                .unwrap_or(Ordering::Equal)
        });

        // Compute prefix sums of max_scores
        let mut prefix_sums = Vec::with_capacity(scorers.len());
        let mut cumsum = 0.0f32;
        for s in &scorers {
            cumsum += s.max_score();
            prefix_sums.push(cumsum);
        }

        debug!(
            "Creating BlockMaxScoreExecutor: num_scorers={}, k={}, total_upper={:.4}, heap_factor={:.2}",
            scorers.len(),
            k,
            cumsum,
            heap_factor
        );

        Self {
            scorers,
            prefix_sums,
            collector: ScoreCollector::new(k),
            heap_factor: heap_factor.clamp(0.0, 1.0),
        }
    }

    /// Find partition point: [0..partition) = non-essential, [partition..n) = essential
    /// Non-essential terms have cumulative max_score <= threshold
    #[inline]
    fn find_partition(&self) -> usize {
        let threshold = self.collector.threshold() * self.heap_factor;
        // Binary search: prefix_sums is monotonically increasing
        self.prefix_sums.partition_point(|&sum| sum <= threshold)
    }

    /// Execute Block-Max MaxScore and return top-k results
    ///
    /// Algorithm:
    /// 1. Partition terms into essential/non-essential based on max_score
    /// 2. Find min_doc across essential scorers
    /// 3. Conjunction check: skip if not enough essential terms present
    /// 4. Block-max check: skip if block upper bounds can't beat threshold
    /// 5. Score essential scorers, check if non-essential scoring is needed
    /// 6. Score non-essential scorers, group by ordinal, insert results
    pub fn execute(mut self) -> Vec<ScoredDoc> {
        if self.scorers.is_empty() {
            debug!("BlockMaxScoreExecutor: no scorers, returning empty results");
            return Vec::new();
        }

        let n = self.scorers.len();
        let mut docs_scored = 0u64;
        let mut docs_skipped = 0u64;
        let mut blocks_skipped = 0u64;
        let mut conjunction_skipped = 0u64;

        // Pre-allocate scratch buffers outside the loop
        let mut ordinal_scores: Vec<(u16, f32)> = Vec::with_capacity(n * 2);

        loop {
            let partition = self.find_partition();

            // If all terms are non-essential, we're done
            if partition >= n {
                debug!("BlockMaxScore: all terms non-essential, early termination");
                break;
            }

            // Single fused pass over essential scorers: find min_doc and
            // accumulate conjunction/block-max upper bounds simultaneously.
            // This replaces 3 separate iterations with 1, reducing cache misses.
            let mut min_doc = u32::MAX;
            let mut present_upper = 0.0f32;
            let mut block_max_sum = 0.0f32;
            for i in partition..n {
                let doc = self.scorers[i].doc();
                if doc < min_doc {
                    min_doc = doc;
                    // New min_doc — reset accumulators to this scorer only
                    present_upper = self.scorers[i].max_score();
                    block_max_sum = self.scorers[i].current_block_max_score();
                } else if doc == min_doc {
                    present_upper += self.scorers[i].max_score();
                    block_max_sum += self.scorers[i].current_block_max_score();
                }
            }

            if min_doc == u32::MAX {
                break; // All essential scorers exhausted
            }

            let non_essential_upper = if partition > 0 {
                self.prefix_sums[partition - 1]
            } else {
                0.0
            };
            let adjusted_threshold = self.collector.threshold() * self.heap_factor;

            // --- Conjunction optimization (Lucene-style) ---
            // Check if enough essential terms are present at min_doc.
            if self.collector.len() >= self.collector.k
                && present_upper + non_essential_upper <= adjusted_threshold
            {
                for i in partition..n {
                    if self.scorers[i].doc() == min_doc {
                        self.scorers[i].advance();
                    }
                }
                conjunction_skipped += 1;
                continue;
            }

            // --- Block-max pruning ---
            // If block-max sum + non-essential upper can't beat threshold, skip blocks.
            if self.collector.len() >= self.collector.k
                && block_max_sum + non_essential_upper <= adjusted_threshold
            {
                for i in partition..n {
                    if self.scorers[i].doc() == min_doc {
                        self.scorers[i].skip_to_next_block();
                    }
                }
                blocks_skipped += 1;
                continue;
            }

            // --- Score essential scorers ---
            // Drain all entries for min_doc from each essential scorer
            ordinal_scores.clear();

            for i in partition..n {
                if self.scorers[i].doc() == min_doc {
                    while self.scorers[i].doc() == min_doc {
                        ordinal_scores.push((self.scorers[i].ordinal(), self.scorers[i].score()));
                        self.scorers[i].advance();
                    }
                }
            }

            // Check if essential score + non-essential upper could beat threshold
            let essential_total: f32 = ordinal_scores.iter().map(|(_, s)| *s).sum();

            if self.collector.len() >= self.collector.k
                && essential_total + non_essential_upper <= adjusted_threshold
            {
                docs_skipped += 1;
                continue;
            }

            // --- Score non-essential scorers ---
            for i in 0..partition {
                let doc = self.scorers[i].seek(min_doc);
                if doc == min_doc {
                    while self.scorers[i].doc() == min_doc {
                        ordinal_scores.push((self.scorers[i].ordinal(), self.scorers[i].score()));
                        self.scorers[i].advance();
                    }
                }
            }

            // --- Group by ordinal and insert ---
            ordinal_scores.sort_unstable_by_key(|(ord, _)| *ord);
            let mut j = 0;
            while j < ordinal_scores.len() {
                let current_ord = ordinal_scores[j].0;
                let mut score = 0.0f32;
                while j < ordinal_scores.len() && ordinal_scores[j].0 == current_ord {
                    score += ordinal_scores[j].1;
                    j += 1;
                }

                trace!(
                    "Doc {}: ordinal={}, score={:.4}, threshold={:.4}",
                    min_doc, current_ord, score, adjusted_threshold
                );

                if self
                    .collector
                    .insert_with_ordinal(min_doc, score, current_ord)
                {
                    docs_scored += 1;
                } else {
                    docs_skipped += 1;
                }
            }
        }

        let results: Vec<ScoredDoc> = self
            .collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| ScoredDoc {
                doc_id,
                score,
                ordinal,
            })
            .collect();

        debug!(
            "BlockMaxScoreExecutor completed: scored={}, skipped={}, blocks_skipped={}, conjunction_skipped={}, returned={}, top_score={:.4}",
            docs_scored,
            docs_skipped,
            blocks_skipped,
            conjunction_skipped,
            results.len(),
            results.first().map(|r| r.score).unwrap_or(0.0)
        );

        results
    }
}

/// Scorer for full-text terms using WAND optimization
///
/// Wraps a `BlockPostingList` with BM25 parameters to implement `ScoringIterator`.
/// Enables MaxScore pruning for efficient top-k retrieval in OR queries.
pub struct TextTermScorer {
    /// Iterator over the posting list (owned)
    iter: crate::structures::BlockPostingIterator<'static>,
    /// IDF component for BM25
    idf: f32,
    /// Average field length for BM25 normalization
    avg_field_len: f32,
    /// Pre-computed max score (using max_tf from posting list)
    max_score: f32,
}

impl TextTermScorer {
    /// Create a new text term scorer with BM25 parameters
    pub fn new(
        posting_list: crate::structures::BlockPostingList,
        idf: f32,
        avg_field_len: f32,
    ) -> Self {
        // Compute max score using actual max_tf from posting list
        let max_tf = posting_list.max_tf() as f32;
        let doc_count = posting_list.doc_count();
        let max_score = super::bm25_upper_bound(max_tf.max(1.0), idf);

        debug!(
            "Created TextTermScorer: doc_count={}, max_tf={:.0}, idf={:.4}, avg_field_len={:.2}, max_score={:.4}",
            doc_count, max_tf, idf, avg_field_len, max_score
        );

        Self {
            iter: posting_list.into_iterator(),
            idf,
            avg_field_len,
            max_score,
        }
    }
}

impl ScoringIterator for TextTermScorer {
    #[inline]
    fn doc(&self) -> DocId {
        self.iter.doc()
    }

    #[inline]
    fn advance(&mut self) -> DocId {
        self.iter.advance()
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        self.iter.seek(target)
    }

    #[inline]
    fn score(&self) -> f32 {
        let tf = self.iter.term_freq() as f32;
        // Use tf as proxy for doc length (common approximation when field lengths aren't stored)
        super::bm25_score(tf, self.idf, tf, self.avg_field_len)
    }

    #[inline]
    fn max_score(&self) -> f32 {
        self.max_score
    }

    #[inline]
    fn current_block_max_score(&self) -> f32 {
        // Use per-block max_tf for tighter Block-Max WAND bounds
        let block_max_tf = self.iter.current_block_max_tf() as f32;
        super::bm25_upper_bound(block_max_tf.max(1.0), self.idf)
    }

    #[inline]
    fn skip_to_next_block(&mut self) -> DocId {
        self.iter.skip_to_next_block()
    }
}

/// Scorer for sparse vector dimensions
///
/// Wraps a `BlockSparsePostingList` with a query weight to implement `ScoringIterator`.
pub struct SparseTermScorer<'a> {
    /// Iterator over the posting list
    iter: crate::structures::BlockSparsePostingIterator<'a>,
    /// Query weight for this dimension
    query_weight: f32,
    /// Pre-computed |query_weight| to avoid repeated .abs() in hot paths
    abs_query_weight: f32,
    /// Global max score (|query_weight| * global_max_weight)
    max_score: f32,
}

impl<'a> SparseTermScorer<'a> {
    /// Create a new sparse term scorer
    ///
    /// Note: Assumes positive weights for WAND upper bound calculation.
    /// For negative query weights, uses absolute value to ensure valid upper bound.
    pub fn new(posting_list: &'a BlockSparsePostingList, query_weight: f32) -> Self {
        // Upper bound must account for sign: |query_weight| * max_weight
        // This ensures the bound is valid regardless of weight sign
        let abs_qw = query_weight.abs();
        let max_score = abs_qw * posting_list.global_max_weight();
        Self {
            iter: posting_list.iterator(),
            query_weight,
            abs_query_weight: abs_qw,
            max_score,
        }
    }

    /// Create from Arc reference (for use with shared posting lists)
    pub fn from_arc(posting_list: &'a Arc<BlockSparsePostingList>, query_weight: f32) -> Self {
        Self::new(posting_list.as_ref(), query_weight)
    }
}

impl ScoringIterator for SparseTermScorer<'_> {
    #[inline]
    fn doc(&self) -> DocId {
        self.iter.doc()
    }

    #[inline]
    fn ordinal(&mut self) -> u16 {
        self.iter.ordinal()
    }

    #[inline]
    fn advance(&mut self) -> DocId {
        self.iter.advance()
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        self.iter.seek(target)
    }

    #[inline]
    fn score(&self) -> f32 {
        // Dot product contribution: query_weight * stored_weight
        self.query_weight * self.iter.weight()
    }

    #[inline]
    fn max_score(&self) -> f32 {
        self.max_score
    }

    #[inline]
    fn current_block_max_score(&self) -> f32 {
        self.iter
            .current_block_max_contribution(self.abs_query_weight)
    }

    #[inline]
    fn skip_to_next_block(&mut self) -> DocId {
        self.iter.skip_to_next_block()
    }
}

/// Block-Max Pruning (BMP) executor for learned sparse retrieval
///
/// Processes blocks in score-descending order using a priority queue.
/// Best for queries with many terms (20+), like SPLADE expansions.
/// Uses document accumulators (FxHashMap) instead of per-term iterators.
///
/// **Memory-efficient**: Only skip entries (block metadata) are kept in memory.
/// Actual block data is loaded on-demand via mmap range reads during execution.
///
/// Reference: Mallia et al., "Faster Learned Sparse Retrieval with
/// Block-Max Pruning" (SIGIR 2024)
pub struct BmpExecutor<'a> {
    /// Sparse index for on-demand block loading
    sparse_index: &'a crate::segment::SparseIndex,
    /// Query terms: (dim_id, query_weight) for each matched dimension
    query_terms: Vec<(u32, f32)>,
    /// Number of results to return
    k: usize,
    /// Heap factor for approximate search
    heap_factor: f32,
}

/// Superblock size: group S consecutive blocks into one priority queue entry.
/// Reduces heap operations by S× (e.g. 8× fewer push/pop for S=8).
const BMP_SUPERBLOCK_SIZE: usize = 8;

/// Entry in the BMP priority queue: represents a superblock (group of up to S blocks)
struct BmpBlockEntry {
    /// Upper bound contribution of this superblock (sum of constituent blocks)
    contribution: f32,
    /// Index into query_terms
    term_idx: usize,
    /// First block index in this superblock
    block_start: usize,
    /// Number of blocks in this superblock (1..=BMP_SUPERBLOCK_SIZE)
    block_count: usize,
}

impl PartialEq for BmpBlockEntry {
    fn eq(&self, other: &Self) -> bool {
        self.contribution == other.contribution
    }
}

impl Eq for BmpBlockEntry {}

impl Ord for BmpBlockEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher contributions come first
        self.contribution
            .partial_cmp(&other.contribution)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for BmpBlockEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> BmpExecutor<'a> {
    /// Create a new BMP executor with lazy block loading
    ///
    /// `query_terms` should contain only dimensions that exist in the index.
    /// Block metadata (skip entries) is read from the sparse index directly.
    pub fn new(
        sparse_index: &'a crate::segment::SparseIndex,
        query_terms: Vec<(u32, f32)>,
        k: usize,
        heap_factor: f32,
    ) -> Self {
        Self {
            sparse_index,
            query_terms,
            k,
            heap_factor: heap_factor.clamp(0.0, 1.0),
        }
    }

    /// Execute BMP and return top-k results
    ///
    /// Builds the priority queue from skip entries (already in memory),
    /// then loads blocks on-demand via mmap range reads as they are visited.
    ///
    /// Uses a hybrid accumulator: flat `Vec<f32>` for single-ordinal (ordinal=0)
    /// entries with O(1) insert, FxHashMap fallback for multi-ordinal entries.
    pub async fn execute(self) -> crate::Result<Vec<ScoredDoc>> {
        use rustc_hash::FxHashMap;

        if self.query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let num_terms = self.query_terms.len();
        let si = self.sparse_index;

        // Build priority queue from skip entries, grouped into superblocks
        let mut block_queue: BinaryHeap<BmpBlockEntry> = BinaryHeap::new();
        let mut remaining_max: Vec<f32> = Vec::with_capacity(num_terms);
        // Pre-resolved skip_start per term (avoids repeated dim_id binary search)
        let mut term_skip_starts: Vec<usize> = Vec::with_capacity(num_terms);
        let mut global_min_doc = u32::MAX;
        let mut global_max_doc = 0u32;

        for (term_idx, &(dim_id, qw)) in self.query_terms.iter().enumerate() {
            let mut term_remaining = 0.0f32;
            let mut term_skip_start = 0usize;

            let abs_qw = qw.abs();
            if let Some((skip_start, skip_count, _global_max)) = si.get_skip_range(dim_id) {
                term_skip_start = skip_start;
                // Group blocks into superblocks of BMP_SUPERBLOCK_SIZE
                let mut sb_start = 0;
                while sb_start < skip_count {
                    let sb_count = (skip_count - sb_start).min(BMP_SUPERBLOCK_SIZE);
                    let mut sb_contribution = 0.0f32;
                    for j in 0..sb_count {
                        let skip = si.read_skip_entry(skip_start + sb_start + j);
                        sb_contribution += abs_qw * skip.max_weight;
                        global_min_doc = global_min_doc.min(skip.first_doc);
                        global_max_doc = global_max_doc.max(skip.last_doc);
                    }
                    term_remaining += sb_contribution;
                    block_queue.push(BmpBlockEntry {
                        contribution: sb_contribution,
                        term_idx,
                        block_start: sb_start,
                        block_count: sb_count,
                    });
                    sb_start += sb_count;
                }
            }
            remaining_max.push(term_remaining);
            term_skip_starts.push(term_skip_start);
        }

        // Hybrid accumulator: flat array for ordinal=0, FxHashMap for multi-ordinal
        let doc_range = if global_max_doc >= global_min_doc {
            (global_max_doc - global_min_doc + 1) as usize
        } else {
            0
        };
        // Use flat array if range is reasonable (< 256K docs)
        let use_flat = doc_range > 0 && doc_range <= 256 * 1024;
        let mut flat_scores: Vec<f32> = if use_flat {
            vec![0.0; doc_range]
        } else {
            Vec::new()
        };
        // Dirty list: track which doc offsets were touched (avoids scanning full array)
        let mut dirty: Vec<u32> = if use_flat {
            Vec::with_capacity(4096)
        } else {
            Vec::new()
        };
        // FxHashMap fallback for multi-ordinal entries or when flat array is too large
        let mut multi_ord_accumulators: FxHashMap<u64, f32> = FxHashMap::default();

        let mut total_remaining: f32 = remaining_max.iter().sum();
        let mut blocks_processed = 0u64;
        let mut blocks_skipped = 0u64;

        // Incremental top-k tracker for threshold
        let mut top_k = ScoreCollector::new(self.k);

        // Reusable decode buffers
        let mut doc_ids_buf: Vec<u32> = Vec::with_capacity(128);
        let mut weights_buf: Vec<f32> = Vec::with_capacity(128);
        let mut ordinals_buf: Vec<u16> = Vec::with_capacity(128);

        // Process blocks in contribution-descending order, loading each on-demand
        while let Some(entry) = block_queue.pop() {
            remaining_max[entry.term_idx] -= entry.contribution;
            total_remaining -= entry.contribution;

            let adjusted_threshold = top_k.threshold() * self.heap_factor;
            if top_k.len() >= self.k && total_remaining <= adjusted_threshold {
                blocks_skipped += block_queue.len() as u64;
                debug!(
                    "BMP early termination after {} blocks: remaining={:.4} <= threshold={:.4}",
                    blocks_processed, total_remaining, adjusted_threshold
                );
                break;
            }

            // Coalesced superblock loading: single mmap read for all blocks in superblock
            let dim_id = self.query_terms[entry.term_idx].0;
            let qw = self.query_terms[entry.term_idx].1;
            let abs_qw = qw.abs();

            let sb_blocks = si
                .get_blocks_range(dim_id, entry.block_start, entry.block_count)
                .await?;

            let skip_start = term_skip_starts[entry.term_idx];
            let adjusted_threshold2 = top_k.threshold() * self.heap_factor;

            for (blk_offset, block) in sb_blocks.into_iter().enumerate() {
                let blk_idx = entry.block_start + blk_offset;

                // Per-block pruning within superblock
                if top_k.len() >= self.k {
                    let skip = si.read_skip_entry(skip_start + blk_idx);
                    let blk_contrib = abs_qw * skip.max_weight;
                    if blk_contrib + total_remaining <= adjusted_threshold2 {
                        blocks_skipped += 1;
                        continue;
                    }
                }

                block.decode_doc_ids_into(&mut doc_ids_buf);
                block.decode_scored_weights_into(qw, &mut weights_buf);
                let count = block.header.count as usize;

                // Fast path: skip ordinals decode when all ordinals are 0
                // (ordinal_bits == 0 in header). This is the common single-value case.
                if block.header.ordinal_bits == 0 && use_flat {
                    for i in 0..count {
                        let doc_id = doc_ids_buf[i];
                        let off = (doc_id - global_min_doc) as usize;
                        if flat_scores[off] == 0.0 {
                            dirty.push(doc_id);
                        }
                        flat_scores[off] += weights_buf[i];
                        top_k.insert_with_ordinal(doc_id, flat_scores[off], 0);
                    }
                } else {
                    block.decode_ordinals_into(&mut ordinals_buf);
                    if use_flat {
                        for i in 0..count {
                            let doc_id = doc_ids_buf[i];
                            let ordinal = ordinals_buf[i];
                            let score_contribution = weights_buf[i];

                            if ordinal == 0 {
                                let off = (doc_id - global_min_doc) as usize;
                                if flat_scores[off] == 0.0 {
                                    dirty.push(doc_id);
                                }
                                flat_scores[off] += score_contribution;
                                top_k.insert_with_ordinal(doc_id, flat_scores[off], 0);
                            } else {
                                let key = (doc_id as u64) << 16 | ordinal as u64;
                                let acc = multi_ord_accumulators.entry(key).or_insert(0.0);
                                *acc += score_contribution;
                                top_k.insert_with_ordinal(doc_id, *acc, ordinal);
                            }
                        }
                    } else {
                        for i in 0..count {
                            let key = (doc_ids_buf[i] as u64) << 16 | ordinals_buf[i] as u64;
                            let acc = multi_ord_accumulators.entry(key).or_insert(0.0);
                            *acc += weights_buf[i];
                            top_k.insert_with_ordinal(doc_ids_buf[i], *acc, ordinals_buf[i]);
                        }
                    }
                }

                blocks_processed += 1;
            }
        }

        // Collect results from both accumulators
        let mut scored: Vec<ScoredDoc> = Vec::new();

        let num_accumulators = if use_flat {
            // Flat array entries (ordinal=0)
            scored.reserve(dirty.len() + multi_ord_accumulators.len());
            for &doc_id in &dirty {
                let off = (doc_id - global_min_doc) as usize;
                let score = flat_scores[off];
                if score > 0.0 {
                    scored.push(ScoredDoc {
                        doc_id,
                        score,
                        ordinal: 0,
                    });
                }
            }
            dirty.len() + multi_ord_accumulators.len()
        } else {
            multi_ord_accumulators.len()
        };

        // Multi-ordinal entries
        scored.extend(
            multi_ord_accumulators
                .into_iter()
                .map(|(key, score)| ScoredDoc {
                    doc_id: (key >> 16) as DocId,
                    score,
                    ordinal: (key & 0xFFFF) as u16,
                }),
        );

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scored.truncate(self.k);
        let results = scored;

        debug!(
            "BmpExecutor completed: blocks_processed={}, blocks_skipped={}, accumulators={}, flat={}, returned={}, top_score={:.4}",
            blocks_processed,
            blocks_skipped,
            num_accumulators,
            use_flat,
            results.len(),
            results.first().map(|r| r.score).unwrap_or(0.0)
        );

        Ok(results)
    }
}

/// Lazy Block-Max MaxScore executor for sparse retrieval (1-11 terms)
///
/// Combines BlockMaxScore's cursor-based document-at-a-time traversal with
/// BMP's lazy block loading. Skip entries (already in memory via zero-copy
/// mmap) drive block-level navigation; actual block data is loaded on-demand
/// only when the cursor visits that block.
///
/// For typical 1-11 term queries with MaxScore pruning, many blocks are
/// skipped entirely — lazy loading avoids the I/O and decode cost for those
/// blocks. This hybrid achieves BMP's memory efficiency with BlockMaxScore's
/// superior pruning for few-term queries.
pub struct LazyBlockMaxScoreExecutor<'a> {
    sparse_index: &'a crate::segment::SparseIndex,
    cursors: Vec<LazyTermCursor>,
    prefix_sums: Vec<f32>,
    collector: ScoreCollector,
    heap_factor: f32,
}

/// Per-term cursor state for lazy block loading
struct LazyTermCursor {
    query_weight: f32,
    /// Pre-computed |query_weight| to avoid repeated .abs() in hot paths
    abs_query_weight: f32,
    max_score: f32,
    /// Index of first skip entry in the SparseIndex skip section (zero-alloc)
    skip_start: usize,
    /// Number of skip entries (blocks) for this dimension
    skip_count: usize,
    /// Base byte offset for block data (pre-resolved, avoids dim_id lookup per load)
    block_data_offset: u64,
    /// Current block index (0-based relative to this dimension's blocks)
    block_idx: usize,
    /// Decoded block data (loaded on demand, reused across seeks)
    doc_ids: Vec<u32>,
    ordinals: Vec<u16>,
    weights: Vec<f32>,
    /// Position within current decoded block
    pos: usize,
    /// Whether block at block_idx is decoded into doc_ids/ordinals/weights
    block_loaded: bool,
    exhausted: bool,
}

impl LazyTermCursor {
    fn new(
        query_weight: f32,
        skip_start: usize,
        skip_count: usize,
        global_max_weight: f32,
        block_data_offset: u64,
    ) -> Self {
        let exhausted = skip_count == 0;
        let abs_qw = query_weight.abs();
        Self {
            query_weight,
            abs_query_weight: abs_qw,
            max_score: abs_qw * global_max_weight,
            skip_start,
            skip_count,
            block_data_offset,
            block_idx: 0,
            doc_ids: Vec::with_capacity(128),
            ordinals: Vec::with_capacity(128),
            weights: Vec::with_capacity(128),
            pos: 0,
            block_loaded: false,
            exhausted,
        }
    }

    /// Ensure current block is loaded and decoded
    async fn ensure_block_loaded(
        &mut self,
        sparse_index: &crate::segment::SparseIndex,
    ) -> crate::Result<bool> {
        if self.exhausted || self.block_loaded {
            return Ok(!self.exhausted);
        }
        match sparse_index
            .load_block_direct(self.skip_start, self.block_data_offset, self.block_idx)
            .await?
        {
            Some(block) => {
                block.decode_doc_ids_into(&mut self.doc_ids);
                block.decode_ordinals_into(&mut self.ordinals);
                block.decode_scored_weights_into(self.query_weight, &mut self.weights);
                self.pos = 0;
                self.block_loaded = true;
                Ok(true)
            }
            None => {
                self.exhausted = true;
                Ok(false)
            }
        }
    }

    #[inline]
    fn doc_with_si(&self, si: &crate::segment::SparseIndex) -> DocId {
        if self.exhausted {
            return u32::MAX;
        }
        if !self.block_loaded {
            // Block not yet loaded — return first_doc of current skip entry
            // as a lower bound (actual doc may be higher after decode)
            return si
                .read_skip_entry(self.skip_start + self.block_idx)
                .first_doc;
        }
        self.doc_ids.get(self.pos).copied().unwrap_or(u32::MAX)
    }

    #[inline]
    fn doc(&self) -> DocId {
        if self.exhausted {
            return u32::MAX;
        }
        if self.block_loaded {
            return self.doc_ids.get(self.pos).copied().unwrap_or(u32::MAX);
        }
        // Fallback: should only be called when block is loaded
        u32::MAX
    }

    #[inline]
    fn ordinal(&self) -> u16 {
        if !self.block_loaded {
            return 0;
        }
        self.ordinals.get(self.pos).copied().unwrap_or(0)
    }

    #[inline]
    fn score(&self) -> f32 {
        if !self.block_loaded {
            return 0.0;
        }
        self.weights.get(self.pos).copied().unwrap_or(0.0)
    }

    #[inline]
    fn current_block_max_score(&self, si: &crate::segment::SparseIndex) -> f32 {
        if self.exhausted || self.block_idx >= self.skip_count {
            return 0.0;
        }
        self.abs_query_weight
            * si.read_skip_entry(self.skip_start + self.block_idx)
                .max_weight
    }

    /// Advance to next posting within current block, or move to next block
    async fn advance(
        &mut self,
        sparse_index: &crate::segment::SparseIndex,
    ) -> crate::Result<DocId> {
        if self.exhausted {
            return Ok(u32::MAX);
        }
        self.ensure_block_loaded(sparse_index).await?;
        if self.exhausted {
            return Ok(u32::MAX);
        }
        self.pos += 1;
        if self.pos >= self.doc_ids.len() {
            self.block_idx += 1;
            self.block_loaded = false;
            if self.block_idx >= self.skip_count {
                self.exhausted = true;
                return Ok(u32::MAX);
            }
            // Don't load next block yet — lazy
        }
        Ok(self.doc())
    }

    /// Seek to first doc >= target using skip entries for block navigation
    async fn seek(
        &mut self,
        sparse_index: &crate::segment::SparseIndex,
        target: DocId,
    ) -> crate::Result<DocId> {
        if self.exhausted {
            return Ok(u32::MAX);
        }

        // If block is loaded and target is within current block range
        if self.block_loaded
            && let Some(&last) = self.doc_ids.last()
        {
            if last >= target && self.doc_ids[self.pos] < target {
                // Binary search within current block
                let remaining = &self.doc_ids[self.pos..];
                let offset = crate::structures::simd::find_first_ge_u32(remaining, target);
                self.pos += offset;
                if self.pos >= self.doc_ids.len() {
                    self.block_idx += 1;
                    self.block_loaded = false;
                    if self.block_idx >= self.skip_count {
                        self.exhausted = true;
                        return Ok(u32::MAX);
                    }
                }
                return Ok(self.doc());
            }
            if self.doc_ids[self.pos] >= target {
                return Ok(self.doc());
            }
        }

        // Binary search on skip entries: find first block where last_doc >= target.
        // Start from block_idx (earlier blocks already exhausted).
        let mut lo = self.block_idx;
        let mut hi = self.skip_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if sparse_index.read_skip_entry(self.skip_start + mid).last_doc < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= self.skip_count {
            self.exhausted = true;
            return Ok(u32::MAX);
        }
        if lo != self.block_idx || !self.block_loaded {
            self.block_idx = lo;
            self.block_loaded = false;
        }
        self.ensure_block_loaded(sparse_index).await?;
        if self.exhausted {
            return Ok(u32::MAX);
        }
        self.pos = crate::structures::simd::find_first_ge_u32(&self.doc_ids, target);
        if self.pos >= self.doc_ids.len() {
            self.block_idx += 1;
            self.block_loaded = false;
            if self.block_idx >= self.skip_count {
                self.exhausted = true;
                return Ok(u32::MAX);
            }
            self.ensure_block_loaded(sparse_index).await?;
        }
        Ok(self.doc())
    }

    /// Skip to next block without loading it (for block-max pruning)
    fn skip_to_next_block(&mut self, si: &crate::segment::SparseIndex) -> DocId {
        if self.exhausted {
            return u32::MAX;
        }
        self.block_idx += 1;
        self.block_loaded = false;
        if self.block_idx >= self.skip_count {
            self.exhausted = true;
            return u32::MAX;
        }
        // Return first_doc of next block as lower bound
        si.read_skip_entry(self.skip_start + self.block_idx)
            .first_doc
    }
}

impl<'a> LazyBlockMaxScoreExecutor<'a> {
    /// Create a new lazy executor
    ///
    /// `query_terms` should contain only dimensions present in the index.
    /// Skip entries are read from the zero-copy mmap section (no I/O).
    pub fn new(
        sparse_index: &'a crate::segment::SparseIndex,
        query_terms: Vec<(u32, f32)>,
        k: usize,
        heap_factor: f32,
    ) -> Self {
        let mut cursors: Vec<LazyTermCursor> = query_terms
            .iter()
            .filter_map(|&(dim_id, qw)| {
                let (skip_start, skip_count, global_max, block_data_offset) =
                    sparse_index.get_skip_range_full(dim_id)?;
                Some(LazyTermCursor::new(
                    qw,
                    skip_start,
                    skip_count,
                    global_max,
                    block_data_offset,
                ))
            })
            .collect();

        // Sort by max_score ascending (non-essential first)
        cursors.sort_by(|a, b| {
            a.max_score
                .partial_cmp(&b.max_score)
                .unwrap_or(Ordering::Equal)
        });

        let mut prefix_sums = Vec::with_capacity(cursors.len());
        let mut cumsum = 0.0f32;
        for c in &cursors {
            cumsum += c.max_score;
            prefix_sums.push(cumsum);
        }

        debug!(
            "Creating LazyBlockMaxScoreExecutor: num_terms={}, k={}, total_upper={:.4}, heap_factor={:.2}",
            cursors.len(),
            k,
            cumsum,
            heap_factor
        );

        Self {
            sparse_index,
            cursors,
            prefix_sums,
            collector: ScoreCollector::new(k),
            heap_factor: heap_factor.clamp(0.0, 1.0),
        }
    }

    #[inline]
    fn find_partition(&self) -> usize {
        let threshold = self.collector.threshold() * self.heap_factor;
        // Binary search: prefix_sums is monotonically increasing
        self.prefix_sums.partition_point(|&sum| sum <= threshold)
    }

    /// Execute lazy Block-Max MaxScore and return top-k results
    pub async fn execute(mut self) -> crate::Result<Vec<ScoredDoc>> {
        if self.cursors.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.cursors.len();
        let si = self.sparse_index;

        // Load first block for each cursor (ensures doc() returns real values)
        for cursor in &mut self.cursors {
            cursor.ensure_block_loaded(si).await?;
        }

        let mut docs_scored = 0u64;
        let mut docs_skipped = 0u64;
        let mut blocks_skipped = 0u64;
        let mut blocks_loaded = 0u64;
        let mut conjunction_skipped = 0u64;
        let mut ordinal_scores: Vec<(u16, f32)> = Vec::with_capacity(n * 2);

        loop {
            let partition = self.find_partition();
            if partition >= n {
                break;
            }

            // Find minimum doc_id across essential cursors
            let mut min_doc = u32::MAX;
            for i in partition..n {
                let doc = self.cursors[i].doc_with_si(si);
                if doc < min_doc {
                    min_doc = doc;
                }
            }
            if min_doc == u32::MAX {
                break;
            }

            let non_essential_upper = if partition > 0 {
                self.prefix_sums[partition - 1]
            } else {
                0.0
            };
            let adjusted_threshold = self.collector.threshold() * self.heap_factor;

            // --- Conjunction optimization ---
            if self.collector.len() >= self.collector.k {
                let present_upper: f32 = (partition..n)
                    .filter(|&i| self.cursors[i].doc_with_si(si) == min_doc)
                    .map(|i| self.cursors[i].max_score)
                    .sum();

                if present_upper + non_essential_upper <= adjusted_threshold {
                    for i in partition..n {
                        if self.cursors[i].doc_with_si(si) == min_doc {
                            self.cursors[i].ensure_block_loaded(si).await?;
                            self.cursors[i].advance(si).await?;
                            blocks_loaded += u64::from(self.cursors[i].block_loaded);
                        }
                    }
                    conjunction_skipped += 1;
                    continue;
                }
            }

            // --- Block-max pruning ---
            if self.collector.len() >= self.collector.k {
                let block_max_sum: f32 = (partition..n)
                    .filter(|&i| self.cursors[i].doc_with_si(si) == min_doc)
                    .map(|i| self.cursors[i].current_block_max_score(si))
                    .sum();

                if block_max_sum + non_essential_upper <= adjusted_threshold {
                    for i in partition..n {
                        if self.cursors[i].doc_with_si(si) == min_doc {
                            self.cursors[i].skip_to_next_block(si);
                            // Ensure next block is loaded for doc() to return real value
                            self.cursors[i].ensure_block_loaded(si).await?;
                            blocks_loaded += 1;
                        }
                    }
                    blocks_skipped += 1;
                    continue;
                }
            }

            // --- Score essential cursors ---
            ordinal_scores.clear();
            for i in partition..n {
                if self.cursors[i].doc_with_si(si) == min_doc {
                    self.cursors[i].ensure_block_loaded(si).await?;
                    while self.cursors[i].doc_with_si(si) == min_doc {
                        ordinal_scores.push((self.cursors[i].ordinal(), self.cursors[i].score()));
                        self.cursors[i].advance(si).await?;
                    }
                }
            }

            let essential_total: f32 = ordinal_scores.iter().map(|(_, s)| *s).sum();
            if self.collector.len() >= self.collector.k
                && essential_total + non_essential_upper <= adjusted_threshold
            {
                docs_skipped += 1;
                continue;
            }

            // --- Score non-essential cursors (highest max_score first for early exit) ---
            // Cursors are sorted by max_score ascending, so iterate in reverse to
            // score highest-impact cursors first and exit early when remaining
            // non-essential upper bound can't change the outcome.
            let mut running_total = essential_total;
            for i in (0..partition).rev() {
                // prefix_sums[i] = sum of max_scores for cursors 0..=i
                // If running total + all remaining (including cursor i) can't beat threshold, stop
                if self.collector.len() >= self.collector.k
                    && running_total + self.prefix_sums[i] <= adjusted_threshold
                {
                    break;
                }

                let doc = self.cursors[i].seek(si, min_doc).await?;
                if doc == min_doc {
                    while self.cursors[i].doc_with_si(si) == min_doc {
                        let s = self.cursors[i].score();
                        running_total += s;
                        ordinal_scores.push((self.cursors[i].ordinal(), s));
                        self.cursors[i].advance(si).await?;
                    }
                }
            }

            // --- Group by ordinal and insert ---
            ordinal_scores.sort_unstable_by_key(|(ord, _)| *ord);
            let mut j = 0;
            while j < ordinal_scores.len() {
                let current_ord = ordinal_scores[j].0;
                let mut score = 0.0f32;
                while j < ordinal_scores.len() && ordinal_scores[j].0 == current_ord {
                    score += ordinal_scores[j].1;
                    j += 1;
                }
                if self
                    .collector
                    .insert_with_ordinal(min_doc, score, current_ord)
                {
                    docs_scored += 1;
                } else {
                    docs_skipped += 1;
                }
            }
        }

        let results: Vec<ScoredDoc> = self
            .collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score, ordinal)| ScoredDoc {
                doc_id,
                score,
                ordinal,
            })
            .collect();

        debug!(
            "LazyBlockMaxScoreExecutor completed: scored={}, skipped={}, blocks_skipped={}, blocks_loaded={}, conjunction_skipped={}, returned={}, top_score={:.4}",
            docs_scored,
            docs_skipped,
            blocks_skipped,
            blocks_loaded,
            conjunction_skipped,
            results.len(),
            results.first().map(|r| r.score).unwrap_or(0.0)
        );

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_collector_basic() {
        let mut collector = ScoreCollector::new(3);

        collector.insert(1, 1.0);
        collector.insert(2, 2.0);
        collector.insert(3, 3.0);
        assert_eq!(collector.threshold(), 1.0);

        collector.insert(4, 4.0);
        assert_eq!(collector.threshold(), 2.0);

        let results = collector.into_sorted_results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 4); // Highest score
        assert_eq!(results[1].0, 3);
        assert_eq!(results[2].0, 2);
    }

    #[test]
    fn test_score_collector_threshold() {
        let mut collector = ScoreCollector::new(2);

        collector.insert(1, 5.0);
        collector.insert(2, 3.0);
        assert_eq!(collector.threshold(), 3.0);

        // Should not enter (score too low)
        assert!(!collector.would_enter(2.0));
        assert!(!collector.insert(3, 2.0));

        // Should enter (score high enough)
        assert!(collector.would_enter(4.0));
        assert!(collector.insert(4, 4.0));
        assert_eq!(collector.threshold(), 4.0);
    }

    #[test]
    fn test_heap_entry_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(HeapEntry {
            doc_id: 1,
            score: 3.0,
            ordinal: 0,
        });
        heap.push(HeapEntry {
            doc_id: 2,
            score: 1.0,
            ordinal: 0,
        });
        heap.push(HeapEntry {
            doc_id: 3,
            score: 2.0,
            ordinal: 0,
        });

        // Min-heap: lowest score should come out first
        assert_eq!(heap.pop().unwrap().score, 1.0);
        assert_eq!(heap.pop().unwrap().score, 2.0);
        assert_eq!(heap.pop().unwrap().score, 3.0);
    }
}
