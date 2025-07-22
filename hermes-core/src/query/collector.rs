//! Search result collection and response types

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::{DocId, Result, Score};

use super::Query;

/// Unique document address: segment_id + local doc_id within segment.
/// Stores segment_id as u128 internally (16 bytes) but serializes as hex string
/// for backward compatibility with JSON/gRPC clients.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DocAddress {
    /// Segment ID as u128 (avoids heap allocation vs String)
    segment_id_raw: u128,
    /// Document ID within the segment
    pub doc_id: DocId,
}

impl DocAddress {
    pub fn new(segment_id: u128, doc_id: DocId) -> Self {
        Self {
            segment_id_raw: segment_id,
            doc_id,
        }
    }

    /// Get segment_id as hex string (for display/API)
    pub fn segment_id(&self) -> String {
        format!("{:032x}", self.segment_id_raw)
    }

    /// Get segment_id as u128 (zero-cost)
    pub fn segment_id_u128(&self) -> Option<u128> {
        Some(self.segment_id_raw)
    }
}

impl serde::Serialize for DocAddress {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("DocAddress", 2)?;
        s.serialize_field("segment_id", &format!("{:032x}", self.segment_id_raw))?;
        s.serialize_field("doc_id", &self.doc_id)?;
        s.end()
    }
}

impl<'de> serde::Deserialize<'de> for DocAddress {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Helper {
            segment_id: String,
            doc_id: DocId,
        }
        let h = Helper::deserialize(deserializer)?;
        let raw = u128::from_str_radix(&h.segment_id, 16).map_err(serde::de::Error::custom)?;
        Ok(DocAddress {
            segment_id_raw: raw,
            doc_id: h.doc_id,
        })
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
    /// Segment ID (set by searcher after collection)
    #[serde(default, skip_serializing_if = "is_zero_u128")]
    pub segment_id: u128,
    /// Matched positions per field: (field_id, scored_positions)
    /// Each position includes its individual score contribution
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub positions: Vec<(u32, Vec<ScoredPosition>)>,
}

fn is_zero_u128(v: &u128) -> bool {
    *v == 0
}

/// Canonical result order used by search, reranking, fusion, and pagination.
pub(crate) fn compare_search_results_desc(a: &SearchResult, b: &SearchResult) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.segment_id.cmp(&b.segment_id))
        .then_with(|| a.doc_id.cmp(&b.doc_id))
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
        self.positions
            .iter()
            .map(|(field_id, scored_positions)| {
                // Position lists are typically short. Collecting into one
                // compact buffer and deduplicating in place avoids both the
                // hash-table allocation and the second allocation needed to
                // turn that table back into a sorted response vector.
                let mut ordinals = Vec::with_capacity(scored_positions.len());
                ordinals.extend(scored_positions.iter().map(|sp| {
                    // For text fields with encoded positions, extract ordinal.
                    // For vector fields, position IS the ordinal.
                    if sp.position > 0xFFFFF {
                        sp.position >> 20
                    } else {
                        sp.position
                    }
                }));
                ordinals.sort_unstable();
                ordinals.dedup();
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
        self.score.to_bits() == other.score.to_bits()
            && self.segment_id == other.segment_id
            && self.doc_id == other.doc_id
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
            .total_cmp(&self.score)
            .then_with(|| self.segment_id.cmp(&other.segment_id))
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

    /// Whether this score can enter the collector's retained result set.
    ///
    /// The scorer still calls `collect` when this returns false so counters and
    /// other side effects remain exact; it only skips materializing positions.
    fn would_collect(&self, _doc_id: DocId, _score: Score) -> bool {
        true
    }

    /// Collect already-owned positions. Position-aware collectors can override
    /// this to move the nested vectors instead of cloning them.
    fn collect_owned(&mut self, doc_id: DocId, score: Score, positions: super::MatchedPositions) {
        self.collect(doc_id, score, &positions);
    }

    /// Whether this collector needs position information
    fn needs_positions(&self) -> bool {
        false
    }
}

/// Compact score-only heap entry.
///
/// A segment-local collector does not know its segment ID yet and ordinary
/// searches do not retain positions. Keeping only these two words while the
/// scorer runs makes the common heap 8 bytes per hit instead of storing a
/// full `SearchResult` (including an empty `Vec` and a zero `u128`).
#[derive(Debug, Clone, Copy)]
struct ScoreOnlyResult {
    doc_id: DocId,
    score: Score,
}

impl PartialEq for ScoreOnlyResult {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.doc_id == other.doc_id
    }
}

impl Eq for ScoreOnlyResult {}

impl PartialOrd for ScoreOnlyResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreOnlyResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Position-aware heap entry. The segment ID is stamped after collection, so
/// omitting it here also keeps this variant smaller than `SearchResult`.
#[derive(Debug, Clone)]
struct PositionedResult {
    doc_id: DocId,
    score: Score,
    positions: super::MatchedPositions,
}

impl PartialEq for PositionedResult {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.doc_id == other.doc_id
    }
}

impl Eq for PositionedResult {}

impl PartialOrd for PositionedResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PositionedResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

enum TopKHeap {
    Scores(BinaryHeap<ScoreOnlyResult>),
    Positions(BinaryHeap<PositionedResult>),
}

#[inline(always)]
fn ranks_ahead(doc_id: DocId, score: Score, worst_doc_id: DocId, worst_score: Score) -> bool {
    let order = score.total_cmp(&worst_score);
    order.is_gt() || (order.is_eq() && doc_id < worst_doc_id)
}

/// Collector for top-k results
pub struct TopKCollector {
    heap: TopKHeap,
    k: usize,
    /// Total documents seen by this collector
    total_seen: u32,
}

// Avoid trusting a caller-controlled `k` as an up-front allocation size. The
// heap still grows to the number of results actually retained, but malformed
// or overly broad requests cannot reserve gigabytes once per segment before
// any document has been scored.
const MAX_INITIAL_TOP_K_CAPACITY: usize = 8 * 1024;

impl TopKCollector {
    pub fn new(k: usize) -> Self {
        Self {
            heap: TopKHeap::Scores(BinaryHeap::with_capacity(k.min(MAX_INITIAL_TOP_K_CAPACITY))),
            k,
            total_seen: 0,
        }
    }

    /// Create a collector that also collects positions
    pub fn with_positions(k: usize) -> Self {
        Self {
            heap: TopKHeap::Positions(BinaryHeap::with_capacity(k.min(MAX_INITIAL_TOP_K_CAPACITY))),
            k,
            total_seen: 0,
        }
    }

    /// Get the total number of documents seen (scored) by this collector
    pub fn total_seen(&self) -> u32 {
        self.total_seen
    }

    pub fn into_sorted_results(self) -> Vec<SearchResult> {
        match self.heap {
            TopKHeap::Scores(heap) => {
                let mut compact = heap.into_vec();
                compact.sort_unstable_by(|a, b| {
                    b.score
                        .total_cmp(&a.score)
                        .then_with(|| a.doc_id.cmp(&b.doc_id))
                });
                compact
                    .into_iter()
                    .map(|result| SearchResult {
                        doc_id: result.doc_id,
                        score: result.score,
                        segment_id: 0,
                        positions: Vec::new(),
                    })
                    .collect()
            }
            TopKHeap::Positions(heap) => {
                let mut positioned = heap.into_vec();
                positioned.sort_unstable_by(|a, b| {
                    b.score
                        .total_cmp(&a.score)
                        .then_with(|| a.doc_id.cmp(&b.doc_id))
                });
                positioned
                    .into_iter()
                    .map(|result| SearchResult {
                        doc_id: result.doc_id,
                        score: result.score,
                        segment_id: 0,
                        positions: result.positions,
                    })
                    .collect()
            }
        }
    }

    /// Consume collector and return (sorted_results, total_seen)
    pub fn into_results_with_count(self) -> (Vec<SearchResult>, u32) {
        let total = self.total_seen;
        (self.into_sorted_results(), total)
    }
}

impl Collector for TopKCollector {
    #[inline]
    fn collect(&mut self, doc_id: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]) {
        self.total_seen = self.total_seen.saturating_add(1);
        if self.k == 0 {
            return;
        }

        match &mut self.heap {
            TopKHeap::Scores(heap) => {
                let result = ScoreOnlyResult { doc_id, score };
                if heap.len() < self.k {
                    heap.push(result);
                } else if heap
                    .peek()
                    .is_some_and(|worst| ranks_ahead(doc_id, score, worst.doc_id, worst.score))
                {
                    *heap.peek_mut().expect("full top-k heap") = result;
                }
            }
            TopKHeap::Positions(heap) => {
                if heap.len() >= self.k
                    && !heap
                        .peek()
                        .is_some_and(|worst| ranks_ahead(doc_id, score, worst.doc_id, worst.score))
                {
                    return;
                }
                let result = PositionedResult {
                    doc_id,
                    score,
                    // Only clone positions after the hit is known to be
                    // competitive. Replacing the root drops its old positions.
                    positions: positions.to_vec(),
                };
                if heap.len() < self.k {
                    heap.push(result);
                } else {
                    *heap.peek_mut().expect("full top-k heap") = result;
                }
            }
        }
    }

    #[inline]
    fn would_collect(&self, doc_id: DocId, score: Score) -> bool {
        if self.k == 0 {
            return false;
        }
        match &self.heap {
            TopKHeap::Scores(heap) => {
                heap.len() < self.k
                    || heap
                        .peek()
                        .is_some_and(|min| ranks_ahead(doc_id, score, min.doc_id, min.score))
            }
            TopKHeap::Positions(heap) => {
                heap.len() < self.k
                    || heap
                        .peek()
                        .is_some_and(|min| ranks_ahead(doc_id, score, min.doc_id, min.score))
            }
        }
    }

    #[inline]
    fn collect_owned(&mut self, doc_id: DocId, score: Score, positions: super::MatchedPositions) {
        self.total_seen = self.total_seen.saturating_add(1);
        if self.k == 0 {
            return;
        }

        match &mut self.heap {
            TopKHeap::Scores(heap) => {
                let result = ScoreOnlyResult { doc_id, score };
                if heap.len() < self.k {
                    heap.push(result);
                } else if heap
                    .peek()
                    .is_some_and(|worst| ranks_ahead(doc_id, score, worst.doc_id, worst.score))
                {
                    *heap.peek_mut().expect("full top-k heap") = result;
                }
            }
            TopKHeap::Positions(heap) => {
                if heap.len() >= self.k
                    && !heap
                        .peek()
                        .is_some_and(|worst| ranks_ahead(doc_id, score, worst.doc_id, worst.score))
                {
                    return;
                }
                let result = PositionedResult {
                    doc_id,
                    score,
                    positions,
                };
                if heap.len() < self.k {
                    heap.push(result);
                } else {
                    *heap.peek_mut().expect("full top-k heap") = result;
                }
            }
        }
    }

    #[inline]
    fn needs_positions(&self) -> bool {
        matches!(&self.heap, TopKHeap::Positions(_))
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

/// Execute a search query on a single segment and return (results, total_seen) (async)
pub async fn search_segment_with_count(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = TopKCollector::new(segment_limit);
    collect_segment_with_limit(reader, query, &mut collector, segment_limit).await?;
    Ok(collector.into_results_with_count())
}

/// Execute a search query on a single segment with positions and return (results, total_seen)
pub async fn search_segment_with_positions_and_count(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = TopKCollector::with_positions(segment_limit);
    collect_segment_with_limit(reader, query, &mut collector, segment_limit).await?;
    Ok(collector.into_results_with_count())
}

/// Return positions for the next collector that can retain them. All but the
/// final consumer receive a clone; the final consumer takes the original
/// allocation. Tuple collectors use this to avoid a deep clone when only one
/// child actually needs positions (the common top-k + count case).
fn positions_for_next_collector(
    positions: &mut Option<super::MatchedPositions>,
    remaining_consumers: &mut usize,
) -> super::MatchedPositions {
    assert!(
        *remaining_consumers > 0,
        "position consumer count underflow"
    );
    *remaining_consumers -= 1;
    if *remaining_consumers == 0 {
        positions
            .take()
            .expect("owned positions must remain for the final collector")
    } else {
        positions
            .as_ref()
            .cloned()
            .expect("owned positions must remain while collectors are pending")
    }
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
    fn would_collect(&self, doc_id: DocId, score: Score) -> bool {
        (self.0.needs_positions() && self.0.would_collect(doc_id, score))
            || (self.1.needs_positions() && self.1.would_collect(doc_id, score))
    }
    fn collect_owned(&mut self, doc_id: DocId, score: Score, positions: super::MatchedPositions) {
        let wants = [
            self.0.needs_positions() && self.0.would_collect(doc_id, score),
            self.1.needs_positions() && self.1.would_collect(doc_id, score),
        ];
        let mut remaining = wants.iter().filter(|&&want| want).count();
        let mut positions = Some(positions);

        if wants[0] {
            self.0.collect_owned(
                doc_id,
                score,
                positions_for_next_collector(&mut positions, &mut remaining),
            );
        } else {
            self.0.collect(doc_id, score, &[]);
        }
        if wants[1] {
            self.1.collect_owned(
                doc_id,
                score,
                positions_for_next_collector(&mut positions, &mut remaining),
            );
        } else {
            self.1.collect(doc_id, score, &[]);
        }
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
    fn would_collect(&self, doc_id: DocId, score: Score) -> bool {
        (self.0.needs_positions() && self.0.would_collect(doc_id, score))
            || (self.1.needs_positions() && self.1.would_collect(doc_id, score))
            || (self.2.needs_positions() && self.2.would_collect(doc_id, score))
    }
    fn collect_owned(&mut self, doc_id: DocId, score: Score, positions: super::MatchedPositions) {
        let wants = [
            self.0.needs_positions() && self.0.would_collect(doc_id, score),
            self.1.needs_positions() && self.1.would_collect(doc_id, score),
            self.2.needs_positions() && self.2.would_collect(doc_id, score),
        ];
        let mut remaining = wants.iter().filter(|&&want| want).count();
        let mut positions = Some(positions);

        if wants[0] {
            self.0.collect_owned(
                doc_id,
                score,
                positions_for_next_collector(&mut positions, &mut remaining),
            );
        } else {
            self.0.collect(doc_id, score, &[]);
        }
        if wants[1] {
            self.1.collect_owned(
                doc_id,
                score,
                positions_for_next_collector(&mut positions, &mut remaining),
            );
        } else {
            self.1.collect(doc_id, score, &[]);
        }
        if wants[2] {
            self.2.collect_owned(
                doc_id,
                score,
                positions_for_next_collector(&mut positions, &mut remaining),
            );
        } else {
            self.2.collect(doc_id, score, &[]);
        }
    }
}

/// Execute a query with one or more collectors (async)
///
/// Uses a large limit for the scorer to disable MaxScore pruning.
/// For queries that benefit from MaxScore pruning (e.g., sparse vector search),
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
    // Use large limit to disable MaxScore skipping for exhaustive collection
    collect_segment_with_limit(reader, query, collector, usize::MAX / 2).await
}

/// Execute a query with one or more collectors and a specific limit (async)
///
/// The limit is passed to the scorer to enable MaxScore pruning for queries
/// that support it (e.g., sparse vector search). This significantly improves
/// performance when only the top-k results are needed.
///
/// Doc IDs in the collector are segment-local. The searcher stamps each result
/// with its segment_id, making (segment_id, doc_id) the unique document key.
pub async fn collect_segment_with_limit<C: Collector>(
    reader: &SegmentReader,
    query: &dyn Query,
    collector: &mut C,
    limit: usize,
) -> Result<()> {
    collect_segment_with_limit_seeded(reader, query, collector, limit, 0.0).await
}

/// Async `collect_segment_with_limit` with a cross-segment threshold seed.
///
/// `initial_threshold` is passed to the scorer so exact MaxScore/BMP paths can
/// start pruning from a nonzero floor carried over from earlier segments.
pub async fn collect_segment_with_limit_seeded<C: Collector>(
    reader: &SegmentReader,
    query: &dyn Query,
    collector: &mut C,
    limit: usize,
    initial_threshold: f32,
) -> Result<()> {
    let options = super::ScorerOptions {
        collect_positions: collector.needs_positions(),
        initial_threshold,
        shared_threshold: None,
        lsp_plan: None,
    };
    let mut scorer = query.scorer_with_options(reader, limit, options).await?;
    drive_scorer(scorer.as_mut(), collector);
    Ok(())
}

/// Drive a scorer through a collector (shared by async and sync paths).
fn drive_scorer<C: Collector>(scorer: &mut dyn super::Scorer, collector: &mut C) {
    let needs_positions = collector.needs_positions();
    let mut doc = scorer.doc();
    while doc != TERMINATED {
        let score = scorer.score();
        if needs_positions && collector.would_collect(doc, score) {
            let positions = scorer.matched_positions().unwrap_or_default();
            collector.collect_owned(doc, score, positions);
        } else {
            collector.collect(doc, score, &[]);
        }
        doc = scorer.advance();
    }
}

// ── Synchronous collector functions (mmap/RAM only) ─────────────────────────

/// Synchronous segment search — returns (results, total_seen).
#[cfg(feature = "sync")]
pub fn search_segment_with_count_sync(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = TopKCollector::new(segment_limit);
    collect_segment_with_limit_sync(reader, query, &mut collector, segment_limit)?;
    Ok(collector.into_results_with_count())
}

/// Synchronous segment search with positions — returns (results, total_seen).
#[cfg(feature = "sync")]
pub fn search_segment_with_positions_and_count_sync(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = TopKCollector::with_positions(segment_limit);
    collect_segment_with_limit_sync(reader, query, &mut collector, segment_limit)?;
    Ok(collector.into_results_with_count())
}

/// Synchronous collect with limit — uses `scorer_sync`.
#[cfg(feature = "sync")]
pub fn collect_segment_with_limit_sync<C: Collector>(
    reader: &SegmentReader,
    query: &dyn Query,
    collector: &mut C,
    limit: usize,
) -> Result<()> {
    collect_segment_with_limit_seeded_sync(reader, query, collector, limit, 0.0)
}

/// Synchronous `collect_segment_with_limit_sync` with a cross-segment threshold
/// seed (see `collect_segment_with_limit_seeded`).
#[cfg(feature = "sync")]
pub fn collect_segment_with_limit_seeded_sync<C: Collector>(
    reader: &SegmentReader,
    query: &dyn Query,
    collector: &mut C,
    limit: usize,
    initial_threshold: f32,
) -> Result<()> {
    let options = super::ScorerOptions {
        collect_positions: collector.needs_positions(),
        initial_threshold,
        shared_threshold: None,
        lsp_plan: None,
    };
    let mut scorer = query.scorer_sync_with_options(reader, limit, options)?;
    drive_scorer(scorer.as_mut(), collector);
    Ok(())
}

/// Per-segment search seeded with a cross-segment top-k floor (sync).
///
/// Behaves like `search_segment_with_count_sync` / its positions variant, but
/// threads `initial_threshold` into the scorer so exact MaxScore/BMP paths
/// prune from the running global k-th score. Used by the multi-segment
/// searcher to propagate the threshold across segments.
#[cfg(feature = "sync")]
pub fn search_segment_seeded_sync(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
    collect_positions: bool,
    initial_threshold: f32,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = if collect_positions {
        TopKCollector::with_positions(segment_limit)
    } else {
        TopKCollector::new(segment_limit)
    };
    collect_segment_with_limit_seeded_sync(
        reader,
        query,
        &mut collector,
        segment_limit,
        initial_threshold,
    )?;
    Ok(collector.into_results_with_count())
}

/// Per-segment search with a live cross-segment top-k floor (sync).
#[cfg(feature = "sync")]
pub fn search_segment_shared_sync(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
    collect_positions: bool,
    shared_threshold: super::SharedThreshold,
) -> Result<(Vec<SearchResult>, u32)> {
    search_segment_shared_sync_planned(
        reader,
        query,
        limit,
        collect_positions,
        shared_threshold,
        None,
    )
}

/// Per-segment search with a live threshold and a query-global LSP/0 plan.
#[cfg(feature = "sync")]
pub(crate) fn search_segment_shared_sync_planned(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
    collect_positions: bool,
    shared_threshold: super::SharedThreshold,
    lsp_plan: Option<std::sync::Arc<super::bmp::LspSegmentPlan>>,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = if collect_positions {
        TopKCollector::with_positions(segment_limit)
    } else {
        TopKCollector::new(segment_limit)
    };
    let options = super::ScorerOptions {
        collect_positions,
        initial_threshold: shared_threshold.get(),
        shared_threshold: Some(shared_threshold),
        lsp_plan,
    };
    let mut scorer = query.scorer_sync_with_options(reader, segment_limit, options)?;
    drive_scorer(scorer.as_mut(), &mut collector);
    Ok(collector.into_results_with_count())
}

/// Per-segment search seeded with a cross-segment top-k floor (async).
pub async fn search_segment_seeded(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
    collect_positions: bool,
    initial_threshold: f32,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = if collect_positions {
        TopKCollector::with_positions(segment_limit)
    } else {
        TopKCollector::new(segment_limit)
    };
    collect_segment_with_limit_seeded(
        reader,
        query,
        &mut collector,
        segment_limit,
        initial_threshold,
    )
    .await?;
    Ok(collector.into_results_with_count())
}

/// Per-segment search with a live cross-segment top-k floor (async).
pub async fn search_segment_shared(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
    collect_positions: bool,
    shared_threshold: super::SharedThreshold,
) -> Result<(Vec<SearchResult>, u32)> {
    search_segment_shared_planned(
        reader,
        query,
        limit,
        collect_positions,
        shared_threshold,
        None,
    )
    .await
}

/// Async per-segment search with a query-global LSP/0 plan.
pub(crate) async fn search_segment_shared_planned(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
    collect_positions: bool,
    shared_threshold: super::SharedThreshold,
    lsp_plan: Option<std::sync::Arc<super::bmp::LspSegmentPlan>>,
) -> Result<(Vec<SearchResult>, u32)> {
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut collector = if collect_positions {
        TopKCollector::with_positions(segment_limit)
    } else {
        TopKCollector::new(segment_limit)
    };
    let options = super::ScorerOptions {
        collect_positions,
        initial_threshold: shared_threshold.get(),
        shared_threshold: Some(shared_threshold),
        lsp_plan,
    };
    let mut scorer = query
        .scorer_with_options(reader, segment_limit, options)
        .await?;
    drive_scorer(scorer.as_mut(), &mut collector);
    Ok(collector.into_results_with_count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[derive(Default)]
    struct OwnedPositionCollector {
        owned_calls: usize,
        borrowed_calls: usize,
        positions: super::super::MatchedPositions,
    }

    impl Collector for OwnedPositionCollector {
        fn collect(
            &mut self,
            _doc_id: DocId,
            _score: Score,
            positions: &[(u32, Vec<ScoredPosition>)],
        ) {
            self.borrowed_calls += 1;
            self.positions = positions.to_vec();
        }

        fn collect_owned(
            &mut self,
            _doc_id: DocId,
            _score: Score,
            positions: super::super::MatchedPositions,
        ) {
            self.owned_calls += 1;
            self.positions = positions;
        }

        fn needs_positions(&self) -> bool {
            true
        }
    }

    struct PositionCountingScorer {
        index: usize,
        position_calls: Arc<AtomicUsize>,
    }

    impl super::super::DocSet for PositionCountingScorer {
        fn doc(&self) -> DocId {
            if self.index < 3 {
                self.index as DocId
            } else {
                TERMINATED
            }
        }

        fn advance(&mut self) -> DocId {
            self.index += 1;
            self.doc()
        }

        fn seek(&mut self, target: DocId) -> DocId {
            self.index = target.min(3) as usize;
            self.doc()
        }

        fn size_hint(&self) -> u32 {
            3u32.saturating_sub(self.index as u32)
        }
    }

    impl super::super::Scorer for PositionCountingScorer {
        fn score(&self) -> Score {
            [10.0, 1.0, 2.0][self.index]
        }

        fn matched_positions(&self) -> Option<super::super::MatchedPositions> {
            self.position_calls.fetch_add(1, AtomicOrdering::Relaxed);
            Some(vec![(7, vec![ScoredPosition::new(self.index as u32, 1.0)])])
        }
    }

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
    fn top_k_zero_retains_no_results() {
        let mut collector = TopKCollector::new(0);
        collector.collect(1, 1.0, &[]);

        assert!(collector.into_sorted_results().is_empty());
    }

    #[test]
    fn huge_top_k_does_not_trigger_a_huge_initial_allocation() {
        let collector = TopKCollector::new(usize::MAX);

        let TopKHeap::Scores(heap) = collector.heap else {
            panic!("score-only constructor selected the position heap");
        };
        assert!(heap.capacity() <= MAX_INITIAL_TOP_K_CAPACITY);
    }

    #[test]
    fn score_only_heap_entry_stays_compact() {
        assert_eq!(std::mem::size_of::<ScoreOnlyResult>(), 8);
        assert!(std::mem::size_of::<SearchResult>() >= 4 * std::mem::size_of::<ScoreOnlyResult>());
    }

    #[test]
    fn top_k_replacement_preserves_score_and_doc_ties() {
        let mut collector = TopKCollector::new(3);
        for (doc_id, score) in [(9, 2.0), (8, 2.0), (7, 2.0), (6, 2.0), (1, 1.0)] {
            collector.collect(doc_id, score, &[]);
        }

        let results = collector.into_sorted_results();
        assert_eq!(
            results
                .iter()
                .map(|result| (result.doc_id, result.score))
                .collect::<Vec<_>>(),
            vec![(6, 2.0), (7, 2.0), (8, 2.0)]
        );
    }

    #[test]
    fn extract_ordinals_sorts_and_deduplicates_without_hashing() {
        let result = SearchResult {
            doc_id: 1,
            score: 1.0,
            segment_id: 0,
            positions: vec![
                (
                    3,
                    vec![
                        ScoredPosition::new(5 << 20, 1.0),
                        ScoredPosition::new(2 << 20, 1.0),
                        ScoredPosition::new(5 << 20, 2.0),
                    ],
                ),
                (
                    7,
                    vec![
                        ScoredPosition::new(4, 1.0),
                        ScoredPosition::new(1, 1.0),
                        ScoredPosition::new(4, 2.0),
                    ],
                ),
            ],
        };

        let fields = result.extract_ordinals();
        assert_eq!(fields[0].ordinals, vec![2, 5]);
        assert_eq!(fields[1].ordinals, vec![1, 4]);
    }

    #[test]
    fn positions_are_only_materialized_for_competitive_hits() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut scorer = PositionCountingScorer {
            index: 0,
            position_calls: Arc::clone(&calls),
        };
        let mut collector = TopKCollector::with_positions(1);

        drive_scorer(&mut scorer, &mut collector);

        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(collector.total_seen(), 3);
        let results = collector.into_sorted_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 0);
        assert_eq!(results[0].positions[0].0, 7);
    }

    #[test]
    fn tuple_moves_owned_positions_to_single_position_collector() {
        let mut positions = OwnedPositionCollector::default();
        let mut count = CountCollector::new();
        let input = vec![(7, vec![ScoredPosition::new(3, 1.0)])];
        let input_ptr = input[0].1.as_ptr();

        (&mut positions, &mut count).collect_owned(11, 2.0, input);

        assert_eq!(positions.owned_calls, 1);
        assert_eq!(positions.borrowed_calls, 0);
        assert_eq!(positions.positions[0].1.as_ptr(), input_ptr);
        assert_eq!(count.count(), 1);
    }

    #[test]
    fn tuple_clones_for_all_but_final_position_collector() {
        let mut first = OwnedPositionCollector::default();
        let mut second = OwnedPositionCollector::default();
        let mut count = CountCollector::new();
        let input = vec![(7, vec![ScoredPosition::new(3, 1.0)])];
        let input_ptr = input[0].1.as_ptr();

        (&mut first, &mut count, &mut second).collect_owned(11, 2.0, input);

        assert_eq!((first.owned_calls, first.borrowed_calls), (1, 0));
        assert_eq!((second.owned_calls, second.borrowed_calls), (1, 0));
        assert_ne!(first.positions[0].1.as_ptr(), input_ptr);
        assert_eq!(second.positions[0].1.as_ptr(), input_ptr);
        assert_eq!(count.count(), 1);
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
