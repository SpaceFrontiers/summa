//! BlockWAND and MaxScore query optimization with BM25F scoring
//!
//! Implements efficient top-k retrieval using:
//! - MaxScore: skips terms that can't contribute to top-k
//! - BlockWAND: block-level score upper bounds for early termination
//! - BM25F: field-aware scoring with length normalization

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::structures::{BitpackedPostingIterator, BitpackedPostingList};
use crate::{DocId, Score};

/// BM25F parameters for WAND scoring
pub const WAND_K1: f32 = 1.2;
pub const WAND_B: f32 = 0.75;

/// Term scorer with MaxScore info and BM25F support
pub struct TermScorer<'a> {
    /// Iterator over postings
    pub iter: BitpackedPostingIterator<'a>,
    /// Maximum possible score for this term (computed with BM25F upper bound)
    pub max_score: f32,
    /// IDF component
    pub idf: f32,
    /// Term index (for tracking)
    pub term_idx: usize,
    /// Field boost for BM25F
    pub field_boost: f32,
    /// Average field length for BM25F length normalization
    pub avg_field_len: f32,
}

impl<'a> TermScorer<'a> {
    /// Create a new term scorer with default BM25F parameters
    pub fn new(posting_list: &'a BitpackedPostingList, idf: f32, term_idx: usize) -> Self {
        Self {
            iter: posting_list.iterator(),
            max_score: posting_list.max_score,
            idf,
            term_idx,
            field_boost: 1.0,
            avg_field_len: 1.0, // Default: no length normalization effect
        }
    }

    /// Create a term scorer with BM25F parameters
    pub fn with_bm25f(
        posting_list: &'a BitpackedPostingList,
        idf: f32,
        term_idx: usize,
        field_boost: f32,
        avg_field_len: f32,
    ) -> Self {
        // Recompute max_score with field boost
        let max_score = if field_boost != 1.0 {
            // Find max_tf across all blocks and recompute upper bound
            let max_tf = posting_list
                .blocks
                .iter()
                .map(|b| b.max_tf)
                .max()
                .unwrap_or(1);
            BitpackedPostingList::compute_bm25f_upper_bound(max_tf, idf, field_boost)
        } else {
            posting_list.max_score
        };

        Self {
            iter: posting_list.iterator(),
            max_score,
            idf,
            term_idx,
            field_boost,
            avg_field_len,
        }
    }

    /// Current document
    #[inline]
    pub fn doc(&self) -> DocId {
        self.iter.doc()
    }

    /// Compute BM25F score for current document
    #[inline]
    pub fn score(&self) -> Score {
        let tf = self.iter.term_freq() as f32;

        // BM25F scoring with length normalization
        // Since we don't have per-doc field length, we approximate using tf
        // This is a common approximation when field lengths aren't stored per-posting
        let length_norm = 1.0 - WAND_B + WAND_B * (tf / self.avg_field_len.max(1.0));
        let tf_norm = (tf * self.field_boost * (WAND_K1 + 1.0))
            / (tf * self.field_boost + WAND_K1 * length_norm);

        self.idf * tf_norm
    }

    /// Get current block's max score (for block-level pruning)
    #[inline]
    pub fn current_block_max_score(&self) -> f32 {
        if self.field_boost == 1.0 {
            self.iter.current_block_max_score()
        } else {
            // Recompute with field boost
            let block_max_tf = self.iter.current_block_max_tf();
            BitpackedPostingList::compute_bm25f_upper_bound(
                block_max_tf,
                self.idf,
                self.field_boost,
            )
        }
    }

    /// Advance to next document
    #[inline]
    pub fn advance(&mut self) -> DocId {
        self.iter.advance()
    }

    /// Seek to doc >= target
    #[inline]
    pub fn seek(&mut self, target: DocId) -> DocId {
        self.iter.seek(target)
    }

    /// Is this scorer exhausted?
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.doc() == u32::MAX
    }
}

/// Result entry for top-k heap
#[derive(Clone, Copy)]
struct HeapEntry {
    doc_id: DocId,
    score: Score,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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

/// Search result
#[derive(Debug, Clone, Copy)]
pub struct WandResult {
    pub doc_id: DocId,
    pub score: Score,
}

/// MaxScore WAND algorithm for efficient top-k retrieval
///
/// Key optimizations:
/// 1. Terms sorted by max_score descending
/// 2. Threshold tracking: skip terms whose max_score < current threshold
/// 3. Block-level skipping using block max scores
pub struct MaxScoreWand<'a> {
    /// Term scorers sorted by current doc_id
    scorers: Vec<TermScorer<'a>>,
    /// Top-k results heap
    heap: BinaryHeap<HeapEntry>,
    /// Number of results to return
    k: usize,
    /// Current score threshold (min score in top-k)
    threshold: Score,
    /// Sum of max_scores for "essential" terms (reserved for future use)
    #[allow(dead_code)]
    essential_max_sum: Score,
}

impl<'a> MaxScoreWand<'a> {
    /// Create a new MaxScore WAND executor
    pub fn new(mut scorers: Vec<TermScorer<'a>>, k: usize) -> Self {
        // Sort scorers by max_score descending
        scorers.sort_by(|a, b| {
            b.max_score
                .partial_cmp(&a.max_score)
                .unwrap_or(Ordering::Equal)
        });

        let essential_max_sum: Score = scorers.iter().map(|s| s.max_score).sum();

        Self {
            scorers,
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            threshold: 0.0,
            essential_max_sum,
        }
    }

    /// Execute the query and return top-k results
    pub fn execute(mut self) -> Vec<WandResult> {
        if self.scorers.is_empty() {
            return Vec::new();
        }

        // Remove exhausted scorers
        self.scorers.retain(|s| !s.is_exhausted());

        while !self.scorers.is_empty() {
            // Sort by current doc_id
            self.scorers.sort_by_key(|s| s.doc());

            // Find pivot: first position where cumulative max_score >= threshold
            let pivot_idx = self.find_pivot();

            if pivot_idx.is_none() {
                break;
            }
            let pivot_idx = pivot_idx.unwrap();
            let pivot_doc = self.scorers[pivot_idx].doc();

            if pivot_doc == u32::MAX {
                break;
            }

            // Check if all scorers up to pivot are at pivot_doc
            let all_at_pivot = self.scorers[..=pivot_idx]
                .iter()
                .all(|s| s.doc() == pivot_doc);

            if all_at_pivot {
                // Score this document
                let score = self.score_document(pivot_doc);
                self.maybe_insert(pivot_doc, score);

                // Advance all scorers at pivot_doc
                for scorer in &mut self.scorers {
                    if scorer.doc() == pivot_doc {
                        scorer.advance();
                    }
                }
            } else {
                // Advance scorers before pivot to pivot_doc
                for i in 0..pivot_idx {
                    if self.scorers[i].doc() < pivot_doc {
                        self.scorers[i].seek(pivot_doc);
                    }
                }
            }

            // Remove exhausted scorers
            self.scorers.retain(|s| !s.is_exhausted());
        }

        self.into_results()
    }

    /// Find pivot index where cumulative max_score >= threshold
    fn find_pivot(&self) -> Option<usize> {
        let mut cumsum = 0.0f32;

        for (i, scorer) in self.scorers.iter().enumerate() {
            cumsum += scorer.max_score;
            if cumsum >= self.threshold {
                return Some(i);
            }
        }

        // If we can't reach threshold, we're done
        if cumsum < self.threshold {
            None
        } else {
            Some(self.scorers.len() - 1)
        }
    }

    /// Score a document across all matching scorers
    fn score_document(&self, doc_id: DocId) -> Score {
        let mut score = 0.0;
        for scorer in &self.scorers {
            if scorer.doc() == doc_id {
                score += scorer.score();
            }
        }
        score
    }

    /// Insert into top-k heap if score is high enough
    fn maybe_insert(&mut self, doc_id: DocId, score: Score) {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry { doc_id, score });
            if self.heap.len() == self.k {
                self.threshold = self.heap.peek().map(|e| e.score).unwrap_or(0.0);
            }
        } else if score > self.threshold {
            self.heap.pop();
            self.heap.push(HeapEntry { doc_id, score });
            self.threshold = self.heap.peek().map(|e| e.score).unwrap_or(0.0);
        }
    }

    /// Convert heap to sorted results
    fn into_results(self) -> Vec<WandResult> {
        let mut results: Vec<_> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|e| WandResult {
                doc_id: e.doc_id,
                score: e.score,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        results
    }
}

/// BlockWAND: Block-level WAND with early termination
///
/// Uses block-level max scores for more aggressive skipping
pub struct BlockWand<'a> {
    scorers: Vec<TermScorer<'a>>,
    heap: BinaryHeap<HeapEntry>,
    k: usize,
    threshold: Score,
}

impl<'a> BlockWand<'a> {
    pub fn new(scorers: Vec<TermScorer<'a>>, k: usize) -> Self {
        Self {
            scorers,
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            threshold: 0.0,
        }
    }

    /// Execute with block-level skipping
    pub fn execute(mut self) -> Vec<WandResult> {
        if self.scorers.is_empty() {
            return Vec::new();
        }

        self.scorers.retain(|s| !s.is_exhausted());

        while !self.scorers.is_empty() {
            // Sort by current doc
            self.scorers.sort_by_key(|s| s.doc());

            // Find minimum doc across all scorers
            let min_doc = self.scorers[0].doc();
            if min_doc == u32::MAX {
                break;
            }

            // Compute upper bound score for this doc using block max scores (BM25F aware)
            let upper_bound: Score = self
                .scorers
                .iter()
                .filter(|s| s.doc() <= min_doc || s.current_block_max_score() > 0.0)
                .map(|s| {
                    if s.doc() == min_doc {
                        s.score() // Exact BM25F score
                    } else {
                        s.current_block_max_score() // BM25F upper bound
                    }
                })
                .sum();

            if upper_bound >= self.threshold {
                // Need to evaluate this document
                // First, advance all scorers to min_doc
                for scorer in &mut self.scorers {
                    if scorer.doc() < min_doc {
                        scorer.seek(min_doc);
                    }
                }

                // Score document
                let score = self.score_document(min_doc);
                self.maybe_insert(min_doc, score);
            }

            // Advance scorers at min_doc
            for scorer in &mut self.scorers {
                if scorer.doc() == min_doc {
                    scorer.advance();
                }
            }

            self.scorers.retain(|s| !s.is_exhausted());
        }

        self.into_results()
    }

    fn score_document(&self, doc_id: DocId) -> Score {
        self.scorers
            .iter()
            .filter(|s| s.doc() == doc_id)
            .map(|s| s.score())
            .sum()
    }

    fn maybe_insert(&mut self, doc_id: DocId, score: Score) {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry { doc_id, score });
            if self.heap.len() == self.k {
                self.threshold = self.heap.peek().map(|e| e.score).unwrap_or(0.0);
            }
        } else if score > self.threshold {
            self.heap.pop();
            self.heap.push(HeapEntry { doc_id, score });
            self.threshold = self.heap.peek().map(|e| e.score).unwrap_or(0.0);
        }
    }

    fn into_results(self) -> Vec<WandResult> {
        let mut results: Vec<_> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|e| WandResult {
                doc_id: e.doc_id,
                score: e.score,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        results
    }
}

/// Simple DAAT (Document-At-A-Time) scorer for comparison
pub fn daat_or<'a>(scorers: &mut [TermScorer<'a>], k: usize) -> Vec<WandResult> {
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
    let mut threshold = 0.0f32;

    loop {
        // Find minimum doc
        let min_doc = scorers
            .iter()
            .filter(|s| !s.is_exhausted())
            .map(|s| s.doc())
            .min();

        let min_doc = match min_doc {
            Some(d) if d != u32::MAX => d,
            _ => break,
        };

        // Score this document
        let score: Score = scorers
            .iter()
            .filter(|s| s.doc() == min_doc)
            .map(|s| s.score())
            .sum();

        // Insert if good enough
        if heap.len() < k {
            heap.push(HeapEntry {
                doc_id: min_doc,
                score,
            });
            if heap.len() == k {
                threshold = heap.peek().map(|e| e.score).unwrap_or(0.0);
            }
        } else if score > threshold {
            heap.pop();
            heap.push(HeapEntry {
                doc_id: min_doc,
                score,
            });
            threshold = heap.peek().map(|e| e.score).unwrap_or(0.0);
        }

        // Advance scorers at min_doc
        for scorer in scorers.iter_mut() {
            if scorer.doc() == min_doc {
                scorer.advance();
            }
        }
    }

    let mut results: Vec<_> = heap
        .into_vec()
        .into_iter()
        .map(|e| WandResult {
            doc_id: e.doc_id,
            score: e.score,
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_posting_list(
        doc_ids: &[u32],
        term_freqs: &[u32],
        idf: f32,
    ) -> BitpackedPostingList {
        BitpackedPostingList::from_postings(doc_ids, term_freqs, idf)
    }

    #[test]
    fn test_maxscore_wand_basic() {
        // Term 1: docs 1, 3, 5, 7
        let pl1 = create_test_posting_list(&[1, 3, 5, 7], &[2, 1, 3, 1], 1.0);
        // Term 2: docs 2, 3, 6, 7
        let pl2 = create_test_posting_list(&[2, 3, 6, 7], &[1, 2, 1, 2], 1.5);

        let scorers = vec![TermScorer::new(&pl1, 1.0, 0), TermScorer::new(&pl2, 1.5, 1)];

        let results = MaxScoreWand::new(scorers, 3).execute();

        assert!(!results.is_empty());
        // Doc 3 and 7 should have highest scores (match both terms)
        let top_docs: Vec<_> = results.iter().map(|r| r.doc_id).collect();
        assert!(top_docs.contains(&3) || top_docs.contains(&7));
    }

    #[test]
    fn test_block_wand_basic() {
        let pl1 = create_test_posting_list(&[1, 3, 5, 7, 9], &[1, 2, 1, 3, 1], 1.0);
        let pl2 = create_test_posting_list(&[2, 3, 7, 8], &[1, 1, 2, 1], 1.2);

        let scorers = vec![TermScorer::new(&pl1, 1.0, 0), TermScorer::new(&pl2, 1.2, 1)];

        let results = BlockWand::new(scorers, 5).execute();

        assert!(!results.is_empty());
        // Should find documents from both lists
        let doc_ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();
        assert!(doc_ids.iter().any(|&d| d == 3 || d == 7)); // Intersection docs
    }

    #[test]
    fn test_daat_or() {
        let pl1 = create_test_posting_list(&[1, 2, 3], &[1, 1, 1], 1.0);
        let pl2 = create_test_posting_list(&[2, 3, 4], &[1, 1, 1], 1.0);

        let mut scorers = vec![TermScorer::new(&pl1, 1.0, 0), TermScorer::new(&pl2, 1.0, 1)];

        let results = daat_or(&mut scorers, 10);

        assert_eq!(results.len(), 4); // Docs 1, 2, 3, 4

        // Docs 2 and 3 should have higher scores (match both)
        assert!(results[0].doc_id == 2 || results[0].doc_id == 3);
        assert!(results[1].doc_id == 2 || results[1].doc_id == 3);
    }

    #[test]
    fn test_maxscore_threshold_pruning() {
        // Create posting lists where MaxScore can prune effectively
        // High-scoring term
        let pl1 = create_test_posting_list(&[1, 100, 200], &[10, 10, 10], 2.0);
        // Low-scoring term with many docs
        let pl2 = create_test_posting_list(&(0..50).collect::<Vec<_>>(), &[1; 50], 0.1);

        let scorers = vec![TermScorer::new(&pl1, 2.0, 0), TermScorer::new(&pl2, 0.1, 1)];

        let results = MaxScoreWand::new(scorers, 3).execute();

        // Top results should be from pl1 (higher scores)
        assert!(
            results
                .iter()
                .any(|r| r.doc_id == 1 || r.doc_id == 100 || r.doc_id == 200)
        );
    }
}
