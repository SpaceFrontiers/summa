//! Shared scoring abstractions for text and sparse vector search
//!
//! Provides common traits and utilities for efficient top-k retrieval:
//! - `ScoringIterator`: Common interface for posting list iteration with scoring
//! - `TopKCollector`: Efficient min-heap for maintaining top-k results
//! - `WandExecutor`: Generic MaxScore WAND algorithm
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
}

impl ScoreCollector {
    /// Create a new collector for top-k results
    pub fn new(k: usize) -> Self {
        // Cap capacity to avoid allocation overflow for very large k
        let capacity = k.saturating_add(1).min(1_000_000);
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            k,
        }
    }

    /// Current score threshold (minimum score to enter top-k)
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() >= self.k {
            self.heap.peek().map(|e| e.score).unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Insert a document score. Returns true if inserted in top-k.
    /// Caller must ensure each doc_id is inserted only once.
    #[inline]
    pub fn insert(&mut self, doc_id: DocId, score: f32) -> bool {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry { doc_id, score });
            true
        } else if score > self.threshold() {
            self.heap.push(HeapEntry { doc_id, score });
            self.heap.pop(); // Remove lowest
            true
        } else {
            false
        }
    }

    /// Check if a score could potentially enter top-k
    #[inline]
    pub fn would_enter(&self, score: f32) -> bool {
        self.heap.len() < self.k || score > self.threshold()
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
    pub fn into_sorted_results(self) -> Vec<(DocId, f32)> {
        let mut results: Vec<_> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|e| (e.doc_id, e.score))
            .collect();

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
}

/// Generic MaxScore WAND executor for top-k retrieval
///
/// Works with any type implementing `ScoringIterator`.
/// Implements:
/// - WAND pivot-based pruning: skip documents that can't beat threshold
/// - Block-max WAND: skip blocks that can't beat threshold
/// - Efficient top-k collection
pub struct WandExecutor<S: ScoringIterator> {
    /// Scorers for each query term
    scorers: Vec<S>,
    /// Top-k collector
    collector: ScoreCollector,
    /// Heap factor for approximate search (SEISMIC-style)
    /// A block/document is skipped if max_possible < heap_factor * threshold
    /// - 1.0 = exact search (default)
    /// - 0.8 = approximate, faster with minor recall loss
    heap_factor: f32,
}

impl<S: ScoringIterator> WandExecutor<S> {
    /// Create a new WAND executor with exact search (heap_factor = 1.0)
    pub fn new(scorers: Vec<S>, k: usize) -> Self {
        Self::with_heap_factor(scorers, k, 1.0)
    }

    /// Create a new WAND executor with approximate search
    ///
    /// `heap_factor` controls the trade-off between speed and recall:
    /// - 1.0 = exact search
    /// - 0.8 = ~20% faster, minor recall loss
    /// - 0.5 = much faster, noticeable recall loss
    pub fn with_heap_factor(scorers: Vec<S>, k: usize, heap_factor: f32) -> Self {
        let total_upper: f32 = scorers.iter().map(|s| s.max_score()).sum();

        debug!(
            "Creating WandExecutor: num_scorers={}, k={}, total_upper={:.4}, heap_factor={:.2}",
            scorers.len(),
            k,
            total_upper,
            heap_factor
        );

        Self {
            scorers,
            collector: ScoreCollector::new(k),
            heap_factor: heap_factor.clamp(0.0, 1.0),
        }
    }

    /// Execute WAND and return top-k results
    ///
    /// Implements the WAND (Weak AND) algorithm with pivot-based pruning:
    /// 1. Maintain iterators sorted by current docID (using sorted vector)
    /// 2. Find pivot: first term where cumulative upper bounds > threshold
    /// 3. If all iterators at pivot docID, fully score; otherwise skip to pivot
    /// 4. Insert into collector and advance
    ///
    /// Reference: Broder et al., "Efficient Query Evaluation using a Two-Level
    /// Retrieval Process" (CIKM 2003)
    ///
    /// Note: For small number of terms (typical queries), a sorted vector with
    /// insertion sort is faster than a heap due to better cache locality.
    /// The vector stays mostly sorted, so insertion sort is ~O(n) amortized.
    pub fn execute(mut self) -> Vec<ScoredDoc> {
        if self.scorers.is_empty() {
            debug!("WandExecutor: no scorers, returning empty results");
            return Vec::new();
        }

        let mut docs_scored = 0u64;
        let mut docs_skipped = 0u64;
        let num_scorers = self.scorers.len();

        // Indices sorted by current docID - initial sort O(n log n)
        let mut sorted_indices: Vec<usize> = (0..num_scorers).collect();
        sorted_indices.sort_by_key(|&i| self.scorers[i].doc());

        loop {
            // Find first non-exhausted iterator (they're sorted, so check first)
            let first_active = sorted_indices
                .iter()
                .position(|&i| self.scorers[i].doc() != u32::MAX);

            let first_active = match first_active {
                Some(pos) => pos,
                None => break, // All exhausted
            };

            // Early termination: if total upper bound can't beat (adjusted) threshold
            // heap_factor < 1.0 makes pruning more aggressive (approximate search)
            let total_upper: f32 = sorted_indices[first_active..]
                .iter()
                .map(|&i| self.scorers[i].max_score())
                .sum();

            let adjusted_threshold = self.collector.threshold() * self.heap_factor;
            if self.collector.len() >= self.collector.k && total_upper <= adjusted_threshold {
                debug!(
                    "Early termination: upper_bound={:.4} <= adjusted_threshold={:.4}",
                    total_upper, adjusted_threshold
                );
                break;
            }

            // Find pivot: first term where cumulative upper bounds > adjusted threshold
            let mut cumsum = 0.0f32;
            let mut pivot_pos = first_active;

            for (pos, &idx) in sorted_indices.iter().enumerate().skip(first_active) {
                cumsum += self.scorers[idx].max_score();
                if cumsum > adjusted_threshold || self.collector.len() < self.collector.k {
                    pivot_pos = pos;
                    break;
                }
            }

            let pivot_idx = sorted_indices[pivot_pos];
            let pivot_doc = self.scorers[pivot_idx].doc();

            if pivot_doc == u32::MAX {
                break;
            }

            // Check if all iterators before pivot are at pivot_doc
            let all_at_pivot = sorted_indices[first_active..=pivot_pos]
                .iter()
                .all(|&i| self.scorers[i].doc() == pivot_doc);

            if all_at_pivot {
                // All terms up to pivot are at the same doc - fully score it
                let mut score = 0.0f32;
                let mut matching_terms = 0u32;

                // Score from all iterators that have this document and advance them
                // Collect indices that need re-sorting
                let mut modified_positions: Vec<usize> = Vec::new();

                for (pos, &idx) in sorted_indices.iter().enumerate().skip(first_active) {
                    let doc = self.scorers[idx].doc();
                    if doc == pivot_doc {
                        score += self.scorers[idx].score();
                        matching_terms += 1;
                        self.scorers[idx].advance();
                        modified_positions.push(pos);
                    } else if doc > pivot_doc {
                        break;
                    }
                }

                trace!(
                    "Doc {}: score={:.4}, matching={}/{}, threshold={:.4}",
                    pivot_doc, score, matching_terms, num_scorers, adjusted_threshold
                );

                if self.collector.insert(pivot_doc, score) {
                    docs_scored += 1;
                } else {
                    docs_skipped += 1;
                }

                // Re-sort modified iterators using insertion sort (efficient for nearly-sorted)
                // Move each modified iterator to its correct position
                for &pos in modified_positions.iter().rev() {
                    let idx = sorted_indices[pos];
                    let new_doc = self.scorers[idx].doc();
                    // Bubble up to correct position
                    let mut curr = pos;
                    while curr + 1 < sorted_indices.len()
                        && self.scorers[sorted_indices[curr + 1]].doc() < new_doc
                    {
                        sorted_indices.swap(curr, curr + 1);
                        curr += 1;
                    }
                }
            } else {
                // Not all at pivot - skip the first iterator to pivot_doc
                let first_pos = first_active;
                let first_idx = sorted_indices[first_pos];
                self.scorers[first_idx].seek(pivot_doc);
                docs_skipped += 1;

                // Re-sort the modified iterator
                let new_doc = self.scorers[first_idx].doc();
                let mut curr = first_pos;
                while curr + 1 < sorted_indices.len()
                    && self.scorers[sorted_indices[curr + 1]].doc() < new_doc
                {
                    sorted_indices.swap(curr, curr + 1);
                    curr += 1;
                }
            }
        }

        let results: Vec<ScoredDoc> = self
            .collector
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, score)| ScoredDoc { doc_id, score })
            .collect();

        debug!(
            "WandExecutor completed: scored={}, skipped={}, returned={}, top_score={:.4}",
            docs_scored,
            docs_skipped,
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
    /// Global max score (query_weight * global_max_weight)
    max_score: f32,
}

impl<'a> SparseTermScorer<'a> {
    /// Create a new sparse term scorer
    pub fn new(posting_list: &'a BlockSparsePostingList, query_weight: f32) -> Self {
        let max_score = query_weight * posting_list.global_max_weight();
        Self {
            iter: posting_list.iterator(),
            query_weight,
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
        self.iter.current_block_max_contribution(self.query_weight)
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
        });
        heap.push(HeapEntry {
            doc_id: 2,
            score: 1.0,
        });
        heap.push(HeapEntry {
            doc_id: 3,
            score: 2.0,
        });

        // Min-heap: lowest score should come out first
        assert_eq!(heap.pop().unwrap().score, 1.0);
        assert_eq!(heap.pop().unwrap().score, 2.0);
        assert_eq!(heap.pop().unwrap().score, 3.0);
    }
}
