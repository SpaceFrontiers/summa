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

/// A scored position/ordinal within a field
/// For text fields: position is the token position
/// For vector fields: position is the ordinal (which vector in multi-value)
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ScoredPosition {
    /// Position (text) or ordinal (vector)
    pub position: u32,
    /// Individual score contribution from this position/ordinal
    pub score: f32,
}

impl ScoredPosition {
    pub fn new(position: u32, score: f32) -> Self {
        Self { position, score }
    }
}

/// Search result with doc_id and score (internal use)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub doc_id: DocId,
    pub score: Score,
    /// Matched positions per field: (field_id, scored_positions)
    /// Each position includes its individual score contribution
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub positions: Vec<(u32, Vec<ScoredPosition>)>,
}

/// Matched field info with ordinals (for multi-valued fields)
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MatchedField {
    /// Field ID
    pub field_id: u32,
    /// Matched element ordinals (for multi-valued fields with position tracking)
    /// Empty if position tracking is not enabled for this field
    pub ordinals: Vec<u32>,
}

impl SearchResult {
    /// Extract unique ordinals from positions for each field
    /// For text fields: ordinal = position >> 20 (from encoded position)
    /// For vector fields: position IS the ordinal directly
    pub fn extract_ordinals(&self) -> Vec<MatchedField> {
        use rustc_hash::FxHashSet;

        self.positions
            .iter()
            .map(|(field_id, scored_positions)| {
                let mut ordinals: FxHashSet<u32> = FxHashSet::default();
                for sp in scored_positions {
                    // For text fields with encoded positions, extract ordinal
                    // For vector fields, position IS the ordinal
                    // We use a heuristic: if position > 0xFFFFF (20 bits), it's encoded
                    let ordinal = if sp.position > 0xFFFFF {
                        sp.position >> 20
                    } else {
                        sp.position
                    };
                    ordinals.insert(ordinal);
                }
                let mut ordinals: Vec<u32> = ordinals.into_iter().collect();
                ordinals.sort_unstable();
                MatchedField {
                    field_id: *field_id,
                    ordinals,
                }
            })
            .collect()
    }

    /// Get all scored positions for a specific field
    pub fn field_positions(&self, field_id: u32) -> Option<&[ScoredPosition]> {
        self.positions
            .iter()
            .find(|(fid, _)| *fid == field_id)
            .map(|(_, positions)| positions.as_slice())
    }
}

/// Search hit with unique document address and score
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchHit {
    /// Unique document address (segment_id + local doc_id)
    pub address: DocAddress,
    pub score: Score,
    /// Matched fields with element ordinals (populated when position tracking is enabled)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_fields: Vec<MatchedField>,
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

/// Trait for search result collectors
///
/// Implement this trait to create custom collectors that can be
/// combined and passed to query execution.
pub trait Collector {
    /// Called for each matching document
    /// positions: Vec of (field_id, scored_positions)
    fn collect(&mut self, doc_id: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]);

    /// Whether this collector needs position information
    fn needs_positions(&self) -> bool {
        false
    }
}

/// Collector for top-k results
pub struct TopKCollector {
    heap: BinaryHeap<SearchResult>,
    k: usize,
    collect_positions: bool,
    /// Total documents seen by this collector
    total_seen: u32,
}

impl TopKCollector {
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            collect_positions: false,
            total_seen: 0,
        }
    }

    /// Create a collector that also collects positions
    pub fn with_positions(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            collect_positions: true,
            total_seen: 0,
        }
    }

    /// Get the total number of documents seen (scored) by this collector
    pub fn total_seen(&self) -> u32 {
        self.total_seen
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

    /// Consume collector and return (sorted_results, total_seen)
    pub fn into_results_with_count(self) -> (Vec<SearchResult>, u32) {
        let total = self.total_seen;
        (self.into_sorted_results(), total)
    }
}

impl Collector for TopKCollector {
    fn collect(&mut self, doc_id: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]) {
        self.total_seen += 1;

        let positions = if self.collect_positions {
            positions.to_vec()
        } else {
            Vec::new()
        };

        if self.heap.len() < self.k {
            self.heap.push(SearchResult {
                doc_id,
                score,
                positions,
            });
        } else if let Some(min) = self.heap.peek()
            && score > min.score
        {
            self.heap.pop();
            self.heap.push(SearchResult {
                doc_id,
                score,
                positions,
            });
        }
    }

    fn needs_positions(&self) -> bool {
        self.collect_positions
    }
}

/// Collector that counts all matching documents
#[derive(Default)]
pub struct CountCollector {
    count: u64,
}

impl CountCollector {
    pub fn new() -> Self {
        Self { count: 0 }
    }

    /// Get the total count
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Collector for CountCollector {
    #[inline]
    fn collect(
        &mut self,
        _doc_id: DocId,
        _score: Score,
        _positions: &[(u32, Vec<ScoredPosition>)],
    ) {
        self.count += 1;
    }
}

/// Execute a search query on a single segment (async)
pub async fn search_segment(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let mut collector = TopKCollector::new(limit);
    collect_segment_with_limit(reader, query, &mut collector, limit).await?;
    Ok(collector.into_sorted_results())
}

/// Execute a search query on a single segment and return (results, total_seen) (async)
pub async fn search_segment_with_count(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<(Vec<SearchResult>, u32)> {
    let mut collector = TopKCollector::new(limit);
    collect_segment_with_limit(reader, query, &mut collector, limit).await?;
    Ok(collector.into_results_with_count())
}

/// Execute a search query on a single segment with position collection (async)
pub async fn search_segment_with_positions(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let mut collector = TopKCollector::with_positions(limit);
    collect_segment_with_limit(reader, query, &mut collector, limit).await?;
    Ok(collector.into_sorted_results())
}

/// Execute a search query on a single segment with positions and return (results, total_seen)
pub async fn search_segment_with_positions_and_count(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<(Vec<SearchResult>, u32)> {
    let mut collector = TopKCollector::with_positions(limit);
    collect_segment_with_limit(reader, query, &mut collector, limit).await?;
    Ok(collector.into_results_with_count())
}

/// Count all documents matching a query on a single segment (async)
pub async fn count_segment(reader: &SegmentReader, query: &dyn Query) -> Result<u64> {
    let mut collector = CountCollector::new();
    collect_segment(reader, query, &mut collector).await?;
    Ok(collector.count())
}

// Implement Collector for tuple of 2 collectors
impl<A: Collector, B: Collector> Collector for (&mut A, &mut B) {
    fn collect(&mut self, doc_id: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]) {
        self.0.collect(doc_id, score, positions);
        self.1.collect(doc_id, score, positions);
    }
    fn needs_positions(&self) -> bool {
        self.0.needs_positions() || self.1.needs_positions()
    }
}

// Implement Collector for tuple of 3 collectors
impl<A: Collector, B: Collector, C: Collector> Collector for (&mut A, &mut B, &mut C) {
    fn collect(&mut self, doc_id: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]) {
        self.0.collect(doc_id, score, positions);
        self.1.collect(doc_id, score, positions);
        self.2.collect(doc_id, score, positions);
    }
    fn needs_positions(&self) -> bool {
        self.0.needs_positions() || self.1.needs_positions() || self.2.needs_positions()
    }
}

/// Execute a query with one or more collectors (async)
///
/// Uses a large limit for the scorer to disable WAND pruning.
/// For queries that benefit from WAND pruning (e.g., sparse vector search),
/// use `collect_segment_with_limit` instead.
///
/// # Examples
/// ```ignore
/// // Single collector
/// let mut top_k = TopKCollector::new(10);
/// collect_segment(reader, query, &mut top_k).await?;
///
/// // Multiple collectors (tuple)
/// let mut top_k = TopKCollector::new(10);
/// let mut count = CountCollector::new();
/// collect_segment(reader, query, &mut (&mut top_k, &mut count)).await?;
/// ```
pub async fn collect_segment<C: Collector>(
    reader: &SegmentReader,
    query: &dyn Query,
    collector: &mut C,
) -> Result<()> {
    // Use large limit to disable WAND skipping for exhaustive collection
    collect_segment_with_limit(reader, query, collector, usize::MAX / 2).await
}

/// Execute a query with one or more collectors and a specific limit (async)
///
/// The limit is passed to the scorer to enable WAND pruning for queries
/// that support it (e.g., sparse vector search). This significantly improves
/// performance when only the top-k results are needed.
///
/// Doc IDs are automatically adjusted by the segment's doc_id_offset to produce
/// global doc IDs that can be used across all segments.
pub async fn collect_segment_with_limit<C: Collector>(
    reader: &SegmentReader,
    query: &dyn Query,
    collector: &mut C,
    limit: usize,
) -> Result<()> {
    let needs_positions = collector.needs_positions();
    let doc_id_offset = reader.doc_id_offset();
    let mut scorer = query.scorer(reader, limit).await?;

    let mut doc = scorer.doc();
    while doc != TERMINATED {
        let positions = if needs_positions {
            scorer.matched_positions().unwrap_or_default()
        } else {
            Vec::new()
        };
        // Add doc_id_offset to convert segment-local ID to global ID
        collector.collect(doc + doc_id_offset, scorer.score(), &positions);
        doc = scorer.advance();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_collector() {
        let mut collector = TopKCollector::new(3);

        collector.collect(0, 1.0, &[]);
        collector.collect(1, 3.0, &[]);
        collector.collect(2, 2.0, &[]);
        collector.collect(3, 4.0, &[]);
        collector.collect(4, 0.5, &[]);

        let results = collector.into_sorted_results();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, 3); // score 4.0
        assert_eq!(results[1].doc_id, 1); // score 3.0
        assert_eq!(results[2].doc_id, 2); // score 2.0
    }

    #[test]
    fn test_count_collector() {
        let mut collector = CountCollector::new();

        collector.collect(0, 1.0, &[]);
        collector.collect(1, 2.0, &[]);
        collector.collect(2, 3.0, &[]);

        assert_eq!(collector.count(), 3);
    }

    #[test]
    fn test_multi_collector() {
        let mut top_k = TopKCollector::new(2);
        let mut count = CountCollector::new();

        // Simulate what collect_segment_multi does
        for (doc_id, score) in [(0, 1.0), (1, 3.0), (2, 2.0), (3, 4.0), (4, 0.5)] {
            top_k.collect(doc_id, score, &[]);
            count.collect(doc_id, score, &[]);
        }

        // Count should have all 5 documents
        assert_eq!(count.count(), 5);

        // TopK should only have top 2 results
        let results = top_k.into_sorted_results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, 3); // score 4.0
        assert_eq!(results[1].doc_id, 1); // score 3.0
    }
}
