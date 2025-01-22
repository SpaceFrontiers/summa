//! Search result collection and response types

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Result, Score};

use super::Query;

/// Unique document address: segment_id (hex) + local doc_id within segment
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DocAddress {
    /// Segment ID as hex string (32 chars)
    pub segment_id: String,
    /// Document ID within the segment
    pub doc_id: DocId,
}

impl DocAddress {
    pub fn new(segment_id: u128, doc_id: DocId) -> Self {
        Self {
            segment_id: format!("{:032x}", segment_id),
            doc_id,
        }
    }

    /// Parse segment_id from hex string
    pub fn segment_id_u128(&self) -> Option<u128> {
        u128::from_str_radix(&self.segment_id, 16).ok()
    }
}

/// Search result with doc_id and score (internal use)
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub doc_id: DocId,
    pub score: Score,
}

/// Search hit with unique document address and score
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchHit {
    /// Unique document address (segment_id + local doc_id)
    pub address: DocAddress,
    pub score: Score,
}

/// Search response with hits (IDs only, no documents)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub total_hits: u32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Collector for top-k results
pub struct TopKCollector {
    heap: BinaryHeap<SearchResult>,
    k: usize,
}

impl TopKCollector {
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    pub fn collect(&mut self, doc_id: DocId, score: Score) {
        if self.heap.len() < self.k {
            self.heap.push(SearchResult { doc_id, score });
        } else if let Some(min) = self.heap.peek()
            && score > min.score
        {
            self.heap.pop();
            self.heap.push(SearchResult { doc_id, score });
        }
    }

    pub fn into_sorted_results(self) -> Vec<SearchResult> {
        let mut results: Vec<_> = self.heap.into_vec();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results
    }
}

/// Execute a search query on a single segment (async)
pub async fn search_segment(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let mut scorer = query.scorer(reader, limit).await?;
    let mut collector = TopKCollector::new(limit);

    let mut doc = scorer.doc();

    while doc != TERMINATED {
        collector.collect(doc, scorer.score());
        doc = scorer.advance();
    }

    Ok(collector.into_sorted_results())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_collector() {
        let mut collector = TopKCollector::new(3);

        collector.collect(0, 1.0);
        collector.collect(1, 3.0);
        collector.collect(2, 2.0);
        collector.collect(3, 4.0);
        collector.collect(4, 0.5);

        let results = collector.into_sorted_results();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, 3); // score 4.0
        assert_eq!(results[1].doc_id, 1); // score 3.0
        assert_eq!(results[2].doc_id, 2); // score 2.0
    }
}
