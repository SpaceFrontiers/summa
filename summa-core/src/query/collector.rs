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

    /// Accept a known exact number of matches without visiting documents.
    /// Return false without changing state to request ordinary collection.
    /// This is only offered to score- and position-free collectors.
    fn collect_count(&mut self, _count: u64) -> bool {
        false
    }

    /// Opt into exact counting separately from ranked collection. A finite
    /// limit promises that retaining this many highest ranked matches is enough.
    /// Custom collectors keep exhaustive callbacks by default.
    fn ranked_count_limit(&self) -> Option<usize> {
        None
    }

    /// Account for exact matches omitted by the ranked path. Called only after
    /// ranked_count_limit opts in; must preserve every count/total_seen field.
    fn collect_omitted_count(&mut self, _count: u64) {
        unreachable!("collector did not opt into separate exact counting");
    }

    /// Opt into bounded score blocks when positions and per-hit deadline
    /// checks are unnecessary. Custom collectors retain per-document calls.
    /// Tuple children opting in permit calls to be grouped by collector.
    fn supports_score_blocks(&self) -> bool {
        false
    }

    /// Collect set bits in increasing document order, with `scores[i]` for
    /// document `base + i`. Set bits must address valid document IDs.
    /// The driver calls this only after `supports_score_blocks` returns true.
    fn collect_score_block(&mut self, base: DocId, scores: &[Score; 64], mut bits: u64) {
        while bits != 0 {
            let offset = bits.trailing_zeros() as usize;
            self.collect(base + offset as u32, scores[offset], &[]);
            bits &= bits - 1;
        }
    }

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

    /// Whether this collector consumes scores. Returning false permits the
    /// driver to pass 0.0 without evaluating BM25. Position consumers still
    /// force scoring; custom collectors retain the previous behavior by default.
    fn needs_scores(&self) -> bool {
        true
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ScoreOnlyResult(u64);

impl ScoreOnlyResult {
    #[inline]
    fn new(doc_id: DocId, score: Score) -> Self {
        let bits = score.to_bits();
        // Map IEEE sign/magnitude to unsigned total order, then reverse it
        // so the max-heap root is the worst score / largest document ID.
        let rank = bits ^ (((bits as i32 >> 31) as u32) | 0x8000_0000);
        Self((u64::from(!rank) << 32) | u64::from(doc_id))
    }

    #[inline]
    fn doc_id(self) -> DocId {
        self.0 as u32
    }

    #[inline]
    fn score(self) -> Score {
        let rank = !(self.0 >> 32) as u32;
        let mask = (rank >> 31).wrapping_sub(1) | 0x8000_0000;
        f32::from_bits(rank ^ mask)
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

    /// Whether the heap already holds `k` results.
    pub fn is_full(&self) -> bool {
        match &self.heap {
            TopKHeap::Scores(heap) => heap.len() >= self.k,
            TopKHeap::Positions(heap) => heap.len() >= self.k,
        }
    }

    fn competitive_score(&self) -> Score {
        if !self.is_full() {
            return Score::NEG_INFINITY;
        }
        match &self.heap {
            TopKHeap::Scores(heap) => heap.peek().map_or(Score::NEG_INFINITY, |hit| hit.score()),
            TopKHeap::Positions(heap) => heap.peek().map_or(Score::NEG_INFINITY, |hit| hit.score),
        }
    }

    pub fn into_sorted_results(self) -> Vec<SearchResult> {
        match self.heap {
            TopKHeap::Scores(heap) => {
                let mut compact = heap.into_vec();
                compact.sort_unstable();
                compact
                    .into_iter()
                    .map(|result| SearchResult {
                        doc_id: result.doc_id(),
                        score: result.score(),
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
    fn ranked_count_limit(&self) -> Option<usize> {
        (!self.needs_positions()).then_some(self.k)
    }
    fn collect_omitted_count(&mut self, count: u64) {
        self.total_seen = self
            .total_seen
            .saturating_add(count.min(u64::from(u32::MAX)) as u32);
    }

    fn supports_score_blocks(&self) -> bool {
        matches!(self.heap, TopKHeap::Scores(_))
    }

    fn collect_score_block(&mut self, base: DocId, scores: &[Score; 64], mut bits: u64) {
        let TopKHeap::Scores(heap) = &mut self.heap else {
            while bits != 0 {
                let offset = bits.trailing_zeros() as usize;
                self.collect(base + offset as u32, scores[offset], &[]);
                bits &= bits - 1;
            }
            return;
        };
        self.total_seen = self.total_seen.saturating_add(bits.count_ones());
        if self.k == 0 || bits == 0 {
            return;
        }
        while heap.len() < self.k && bits != 0 {
            let offset = bits.trailing_zeros() as usize;
            heap.push(ScoreOnlyResult::new(base + offset as u32, scores[offset]));
            bits &= bits - 1;
        }
        if bits == 0 {
            return;
        }
        let mut worst = *heap.peek().expect("full top-k heap");
        // Screen independent scores before the serial heap loop. Preserve
        // ties and unordered values for its canonical total-order admission.
        // A rising heap threshold only leaves extra candidates in this mask.
        let mut eligible = 0u64;
        for (i, &score) in scores.iter().enumerate() {
            eligible |= (if score < worst.score() { 0 } else { 1 }) << i;
        }
        bits &= eligible;
        while bits != 0 {
            let offset = bits.trailing_zeros() as usize;
            let doc_id = base + offset as u32;
            let score = scores[offset];
            let result = ScoreOnlyResult::new(doc_id, score);
            if result < worst {
                *heap.peek_mut().expect("full top-k heap") = result;
                worst = *heap.peek().expect("full top-k heap");
            }
            bits &= bits - 1;
        }
    }

    #[inline]
    fn collect(&mut self, doc_id: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]) {
        self.total_seen = self.total_seen.saturating_add(1);
        if self.k == 0 {
            return;
        }

        match &mut self.heap {
            TopKHeap::Scores(heap) => {
                let result = ScoreOnlyResult::new(doc_id, score);
                if heap.len() < self.k {
                    heap.push(result);
                } else if heap.peek().is_some_and(|worst| result < *worst) {
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
                        .is_some_and(|min| ScoreOnlyResult::new(doc_id, score) < *min)
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
                let result = ScoreOnlyResult::new(doc_id, score);
                if heap.len() < self.k {
                    heap.push(result);
                } else if heap.peek().is_some_and(|worst| result < *worst) {
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
    fn ranked_count_limit(&self) -> Option<usize> {
        Some(0)
    }
    fn collect_omitted_count(&mut self, count: u64) {
        self.count += count;
    }

    fn supports_score_blocks(&self) -> bool {
        true
    }

    fn collect_score_block(&mut self, _base: DocId, _scores: &[Score; 64], bits: u64) {
        self.count += u64::from(bits.count_ones());
    }

    fn collect_count(&mut self, count: u64) -> bool {
        self.count += count;
        true
    }

    fn needs_scores(&self) -> bool {
        false
    }

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
    fn ranked_count_limit(&self) -> Option<usize> {
        Some(
            self.0
                .ranked_count_limit()?
                .max(self.1.ranked_count_limit()?),
        )
    }
    fn collect_omitted_count(&mut self, count: u64) {
        self.0.collect_omitted_count(count);
        self.1.collect_omitted_count(count);
    }

    fn supports_score_blocks(&self) -> bool {
        self.0.supports_score_blocks() && self.1.supports_score_blocks()
    }
    fn collect_score_block(&mut self, base: DocId, scores: &[Score; 64], bits: u64) {
        self.0.collect_score_block(base, scores, bits);
        self.1.collect_score_block(base, scores, bits);
    }
    fn needs_scores(&self) -> bool {
        self.0.needs_scores() || self.1.needs_scores()
    }
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
    fn ranked_count_limit(&self) -> Option<usize> {
        Some(
            self.0
                .ranked_count_limit()?
                .max(self.1.ranked_count_limit()?)
                .max(self.2.ranked_count_limit()?),
        )
    }
    fn collect_omitted_count(&mut self, count: u64) {
        self.0.collect_omitted_count(count);
        self.1.collect_omitted_count(count);
        self.2.collect_omitted_count(count);
    }

    fn supports_score_blocks(&self) -> bool {
        self.0.supports_score_blocks()
            && self.1.supports_score_blocks()
            && self.2.supports_score_blocks()
    }
    fn collect_score_block(&mut self, base: DocId, scores: &[Score; 64], bits: u64) {
        self.0.collect_score_block(base, scores, bits);
        self.1.collect_score_block(base, scores, bits);
        self.2.collect_score_block(base, scores, bits);
    }
    fn needs_scores(&self) -> bool {
        self.0.needs_scores() || self.1.needs_scores() || self.2.needs_scores()
    }
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

fn streams_exhaustive_text(query: &dyn Query) -> bool {
    matches!(query.decompose(), super::QueryDecomposition::TextTerm(_))
        || query.should_children().is_some_and(|children| {
            children
                .iter()
                .all(|child| streams_exhaustive_text(child.as_ref()))
        })
}

/// Exact inclusion/exclusion is cheaper when only an OR cardinality is needed.
/// Reuse ordinary conjunction traversal and admit only physical text counts.
async fn two_term_union_count(
    reader: &SegmentReader,
    query: &dyn Query,
    minimum_count: u64,
) -> Result<Option<u64>> {
    if reader.alive_docs().is_some() {
        return Ok(None);
    }
    let Some([left, right]) = query.should_children() else {
        return Ok(None);
    };
    let (Some(a), Some(b)) = (left.count_equivalent_term(), right.count_equivalent_term()) else {
        return Ok(None);
    };
    for info in [&a, &b] {
        if reader.is_chunked_field(info.field)
            || !reader
                .schema()
                .get_field_entry(info.field)
                .is_some_and(|field| field.indexed)
        {
            return Ok(None);
        }
    }
    let mapped = reader.has_text_mapping(a.field) || reader.has_text_mapping(b.field);
    if mapped && a.field != b.field {
        return Ok(None);
    }
    let a = reader.text_doc_freq(a.field, &a.term).await?;
    let b = reader.text_doc_freq(b.field, &b.term).await?;
    // A zero DF also permits the existing fast-column fallback. Do not replace it.
    if a == 0 || b == 0 || u64::from(a.max(b)) < minimum_count {
        return Ok(None);
    }
    let mut intersection = super::BooleanQuery::new();
    intersection.must = vec![left.clone(), right.clone()];
    let mut options = super::ScorerOptions {
        physical_text_field: None,
        complete_text_matches: true,
        skip_scoring_setup: true,
        collect_positions: false,
        ..Default::default()
    };
    if mapped && super::text_mapping::prepare(reader, &intersection, &mut options, true).is_none() {
        return Ok(None);
    }
    let mut scorer = intersection
        .scorer_with_options(reader, usize::MAX / 2, options)
        .await?;
    let mut overlap = CountCollector::new();
    drive_scorer(scorer.as_mut(), &mut overlap);
    reader.check_posting_integrity()?;
    let overlap = overlap.count();
    if overlap > u64::from(a.min(b)) {
        return Err(crate::Error::Corruption(
            "term overlap exceeds dictionary cardinality".into(),
        ));
    }
    Ok(Some(u64::from(a) + u64::from(b) - overlap))
}

/// Restrict split counting to the ordinary exact text ranking plans.
async fn ranked_exact_count(
    reader: &SegmentReader,
    query: &dyn Query,
    limit: usize,
) -> Result<Option<u64>> {
    if reader.alive_docs().is_some() {
        return Ok(None);
    }
    let eligible = |info: super::TermQueryInfo| {
        (info.weight.is_finite()
            && info.weight > 0.0
            && reader
                .schema()
                .get_field_entry(info.field)
                .is_some_and(|entry| entry.indexed)
            && (!reader.has_text_mapping(info.field)
                || reader
                    .chunk_map(info.field)
                    .is_some_and(|map| map.is_document_map()))
            && !reader.is_chunked_field(info.field))
        .then_some(info)
    };
    if let Some(info) = query.ranked_count_equivalent_term()
        && let Some(info) = eligible(info)
    {
        let count = reader.text_doc_freq(info.field, &info.term).await?;
        return Ok(
            (u64::from(count) >= (limit.max(128) as u64).saturating_mul(16))
                .then_some(u64::from(count)),
        );
    }
    if let Some([left, right]) = query.should_children()
        && let (super::QueryDecomposition::TextTerm(a), super::QueryDecomposition::TextTerm(b)) =
            (left.decompose(), right.decompose())
        && let (Some(a), Some(b)) = (eligible(a), eligible(b))
        && a.field == b.field
    {
        return two_term_union_count(reader, query, (limit.max(128) as u64).saturating_mul(128))
            .await;
    }
    Ok(None)
}

/// Execute a query with one or more collectors (async)
///
/// Plain text terms and unions stream matches without an intermediate ranked heap.
/// Other query types retain the large scorer limit used by this API; this does
/// not turn approximate vector retrieval into an exhaustive distance scan.
/// Use `collect_segment_with_limit` when only ranked candidates are needed.
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
    reader.check_posting_integrity()?;
    // Dictionary document frequency is exact only for physical document ids in
    // deletion-free readers. Never substitute count_estimate (chunks/deletes).
    if !collector.needs_scores()
        && !collector.needs_positions()
        && reader.alive_docs().is_none()
        && let Some(info) = query.count_equivalent_term()
        && !reader.is_chunked_field(info.field)
        && reader
            .schema()
            .get_field_entry(info.field)
            .is_some_and(|entry| entry.indexed)
    {
        let count = reader.text_doc_freq(info.field, &info.term).await?;
        // Zero also represents a missing term. Preserve the ordinary scorer's
        // fast-column fallback rather than interpreting it as an exact empty set.
        if count > 0 && collector.collect_count(u64::from(count)) {
            return reader.check_posting_integrity();
        }
    }
    if !collector.needs_scores()
        && !collector.needs_positions()
        && query
            .should_children()
            .is_some_and(|children| children.len() == 2)
        && collector.collect_count(0)
        && let Some(count) = two_term_union_count(reader, query, 0).await?
        && collector.collect_count(count)
    {
        return reader.check_posting_integrity();
    }
    if !collector.needs_positions()
        && let Some(limit) = collector.ranked_count_limit()
        && limit > 0
        && limit < reader.num_docs() as usize
        && let Some(count) = ranked_exact_count(reader, query, limit).await?
        && count > limit as u64
    {
        let mut visited = CountCollector::new();
        collect_segment_with_limit(reader, query, &mut (&mut *collector, &mut visited), limit)
            .await?;
        let omitted = count.checked_sub(visited.count()).ok_or_else(|| {
            crate::Error::Corruption("ranked matches exceed exact cardinality".into())
        })?;
        collector.collect_omitted_count(omitted);
        return reader.check_posting_integrity();
    }
    let ranked_count_limit = collector.ranked_count_limit().filter(|&k| {
        query.supports_ranked_conjunction_count()
            && k > 0
            && k < reader.num_docs() as usize
            && !collector.needs_positions()
            && reader.alive_docs().is_none()
    });
    let mut options = super::ScorerOptions {
        physical_text_field: None,
        // The required-clause scorer deliberately rejects proximity scoring.
        // Select it only for ordinary text terms/unions; preserve the existing
        // execution semantics of opaque, tuned, and vector query types.
        complete_text_matches: streams_exhaustive_text(query),
        ranked_count_limit,
        skip_scoring_setup: !collector.needs_scores() && !collector.needs_positions(),
        eligibility: reader.alive_docs(),
        collect_positions: collector.needs_positions(),
        ..Default::default()
    };
    // Keep the existing non-text limit semantics. Complete text scorers use
    // posting cursors rather than retaining and sorting every matching hit.
    let map = super::text_mapping::prepare(reader, query, &mut options, true);
    let scorer = query
        .scorer_with_options(reader, usize::MAX / 2, options)
        .await?;
    let exact_count = ranked_count_limit.and_then(|_| scorer.exact_ranked_count());
    let mut scorer = super::text_mapping::filtered(scorer, reader.alive_docs(), map);
    if let Some(count) = exact_count {
        let mut visited = CountCollector::new();
        drive_collected(
            scorer.as_mut(),
            &mut (&mut *collector, &mut visited),
            map,
            None,
        );
        let omitted = count.checked_sub(visited.count()).ok_or_else(|| {
            crate::Error::Corruption("ranked matches exceed exact cardinality".into())
        })?;
        collector.collect_omitted_count(omitted);
    } else {
        drive_collected(scorer.as_mut(), collector, map, None);
    }
    reader.check_posting_integrity()
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
    reader.check_posting_integrity()?;
    let mut options = super::ScorerOptions {
        physical_text_field: None,
        complete_text_matches: false,
        ranked_count_limit: None,
        skip_scoring_setup: !collector.needs_scores() && !collector.needs_positions(),
        eligibility: reader.alive_docs(),
        collect_positions: collector.needs_positions(),
        initial_threshold,
        shared_threshold: None,
        lsp_plan: None,
        global_stats: None,
    };
    let map = super::text_mapping::prepare(reader, query, &mut options, false);
    let scorer = query.scorer_with_options(reader, limit, options).await?;
    let mut scorer = super::text_mapping::filtered(scorer, reader.alive_docs(), map);
    drive_collected(scorer.as_mut(), collector, map, None);
    reader.check_posting_integrity()
}

/// Drive a scorer through a collector (shared by async and sync paths).
fn drive_scorer<C: Collector>(scorer: &mut dyn super::Scorer, collector: &mut C) {
    drive_scorer_budgeted(scorer, collector, None);
}

/// Scorers remain sorted in their field-local domain; only the collector sees
/// stable IDs. Heap ties and position admission therefore use logical IDs.
struct MappedCollector<'a, C> {
    inner: &'a mut C,
    map: &'a crate::segment::chunk_map::ChunkMap,
}

impl<C: Collector> Collector for MappedCollector<'_, C> {
    fn collect(&mut self, doc: DocId, score: Score, positions: &[(u32, Vec<ScoredPosition>)]) {
        self.inner.collect(self.map.doc_id(doc), score, positions);
    }
    fn collect_owned(&mut self, doc: DocId, score: Score, positions: super::MatchedPositions) {
        self.inner
            .collect_owned(self.map.doc_id(doc), score, positions);
    }
    fn would_collect(&self, doc: DocId, score: Score) -> bool {
        self.inner.would_collect(self.map.doc_id(doc), score)
    }
    fn collect_count(&mut self, count: u64) -> bool {
        self.inner.collect_count(count)
    }
    fn needs_scores(&self) -> bool {
        self.inner.needs_scores()
    }
    fn needs_positions(&self) -> bool {
        self.inner.needs_positions()
    }
}

fn drive_collected<C: Collector>(
    scorer: &mut dyn super::Scorer,
    collector: &mut C,
    map: Option<&crate::segment::chunk_map::ChunkMap>,
    budget: Option<&super::SharedThreshold>,
) {
    if let Some(map) = map {
        drive_scorer_budgeted(
            scorer,
            &mut MappedCollector {
                inner: collector,
                map,
            },
            budget,
        );
    } else {
        drive_scorer_budgeted(scorer, collector, budget);
    }
}

fn drive_scorer_budgeted<C: Collector>(
    scorer: &mut dyn super::Scorer,
    collector: &mut C,
    budget: Option<&super::SharedThreshold>,
) {
    let needs_positions = collector.needs_positions();
    let needs_scores = collector.needs_scores() || needs_positions;
    let mut doc = scorer.doc();
    if needs_scores && !needs_positions && scorer.supports_score_windows() {
        let mut scores = Box::new([0.0; super::docset::DOC_WINDOW_SIZE as usize]);
        let mut bits = [0; super::docset::DOC_WINDOW_WORDS];
        let collect_blocks = budget.and_then(super::SharedThreshold::deadline).is_none()
            && collector.supports_score_blocks();
        while doc != TERMINATED {
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            scorer.fill_score_window(doc, &mut scores, &mut bits);
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            if collect_blocks {
                for (index, (scores, &word)) in
                    scores.as_chunks::<64>().0.iter().zip(&bits).enumerate()
                {
                    if word != 0 {
                        collector.collect_score_block(doc + index as u32 * 64, scores, word);
                    }
                }
                doc = scorer.doc();
                continue;
            }
            for (index, &word) in bits.iter().enumerate() {
                let mut remaining = word;
                while remaining != 0 {
                    if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                        return;
                    }
                    let offset = index * 64 + remaining.trailing_zeros() as usize;
                    collector.collect(doc + offset as u32, scores[offset], &[]);
                    remaining &= remaining - 1;
                }
            }
            doc = scorer.doc();
        }
        return;
    }
    if needs_scores && !needs_positions && scorer.supports_score_batches() {
        let mut docs = [0; super::docset::DOC_BATCH_SIZE];
        let mut scores = [0.0; super::docset::DOC_BATCH_SIZE];
        while doc != TERMINATED {
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            let count = scorer.fill_score_batch(&mut docs, &mut scores);
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            for i in 0..count {
                if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                    return;
                }
                collector.collect(docs[i], scores[i], &[]);
            }
            doc = scorer.doc();
        }
        return;
    }
    if !needs_scores && scorer.supports_doc_windows() {
        let mut bits = [0; super::docset::DOC_WINDOW_WORDS];
        while doc != TERMINATED {
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            scorer.fill_doc_window(doc, &mut bits);
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            let count = bits.iter().map(|word| u64::from(word.count_ones())).sum();
            if !collector.collect_count(count) {
                for (index, &word) in bits.iter().enumerate() {
                    let mut remaining = word;
                    while remaining != 0 {
                        if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                            return;
                        }
                        let offset = index as u32 * 64 + remaining.trailing_zeros();
                        collector.collect(doc + offset, 0.0, &[]);
                        remaining &= remaining - 1;
                    }
                }
            }
            doc = scorer.doc();
        }
        return;
    }
    if !needs_scores && !needs_positions && scorer.supports_doc_batches() {
        let mut docs = [0; super::docset::DOC_BATCH_SIZE];
        while doc != TERMINATED {
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            let count = scorer.fill_doc_batch(&mut docs);
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            if !collector.collect_count(count as u64) {
                for &doc in &docs[..count] {
                    if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                        return;
                    }
                    collector.collect(doc, 0.0, &[]);
                }
            }
            doc = scorer.doc();
        }
        return;
    }
    while doc != TERMINATED {
        // Check after advance/seek too: an expired negative verifier must
        // never turn an incomplete exclusion check into a collected hit.
        if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
            break;
        }
        let score = if needs_scores { scorer.score() } else { 0.0 };
        if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
            break;
        }
        if needs_positions && collector.would_collect(doc, score) {
            let positions = scorer.matched_positions().unwrap_or_default();
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
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
    reader.check_posting_integrity()?;
    let mut options = super::ScorerOptions {
        physical_text_field: None,
        complete_text_matches: false,
        ranked_count_limit: None,
        skip_scoring_setup: !collector.needs_scores() && !collector.needs_positions(),
        eligibility: reader.alive_docs(),
        collect_positions: collector.needs_positions(),
        initial_threshold,
        shared_threshold: None,
        lsp_plan: None,
        global_stats: None,
    };
    let map = super::text_mapping::prepare(reader, query, &mut options, false);
    let scorer = query.scorer_sync_with_options(reader, limit, options)?;
    let mut scorer = super::text_mapping::filtered(scorer, reader.alive_docs(), map);
    drive_collected(scorer.as_mut(), collector, map, None);
    reader.check_posting_integrity()
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
    global_stats: Option<std::sync::Arc<super::GlobalStats>>,
) -> Result<(Vec<SearchResult>, u32)> {
    reader.check_posting_integrity()?;
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut options = super::ScorerOptions {
        physical_text_field: None,
        complete_text_matches: false,
        ranked_count_limit: None,
        skip_scoring_setup: false,
        eligibility: reader.alive_docs(),
        collect_positions,
        initial_threshold: shared_threshold.get(),
        shared_threshold: Some(shared_threshold.clone()),
        lsp_plan,
        global_stats,
    };
    let map = super::text_mapping::prepare(reader, query, &mut options, false);
    let scorer = query.scorer_sync_with_options(reader, segment_limit, options)?;
    let mut scorer = super::text_mapping::filtered(scorer, reader.alive_docs(), map);
    let results = top_k_from_mapped_scorer(
        scorer.as_mut(),
        segment_limit,
        collect_positions,
        Some(&shared_threshold),
        map,
    );
    reader.check_posting_integrity()?;
    Ok(results)
}

/// Collect a segment's top-k from a freshly built top-level scorer.
///
/// Scorers wrapping an already ranked list (text and vector executors) hand it over
/// through [`Scorer::precomputed_top_k`]; everything else is driven through a
/// `TopKCollector`. Candidate bounds may avoid confirmation of noncompetitive
/// matches. `total_seen` counts confirmed, scored matches, not an exact total.
#[cfg(test)]
fn top_k_from_scorer(
    scorer: &mut dyn super::Scorer,
    segment_limit: usize,
    collect_positions: bool,
    budget: Option<&super::SharedThreshold>,
) -> (Vec<SearchResult>, u32) {
    top_k_from_mapped_scorer(scorer, segment_limit, collect_positions, budget, None)
}

fn top_k_from_mapped_scorer(
    scorer: &mut dyn super::Scorer,
    segment_limit: usize,
    collect_positions: bool,
    budget: Option<&super::SharedThreshold>,
    map: Option<&crate::segment::chunk_map::ChunkMap>,
) -> (Vec<SearchResult>, u32) {
    // A physical-ID cutoff could discard the winner of a stable-ID tie.
    // Take the entire bounded retained list before translating and truncating;
    // the precomputed scorer owns at most its original query limit in hits.
    let handoff_limit = map.map_or(segment_limit, |map| map.num_chunks() as usize);
    if let Some((mut results, seen)) = scorer.precomputed_top_k(handoff_limit, collect_positions) {
        if let Some(map) = map {
            for result in &mut results {
                result.doc_id = map.doc_id(result.doc_id);
            }
            results.sort_unstable_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.doc_id.cmp(&b.doc_id))
            });
            results.truncate(segment_limit);
        }
        return (results, seen);
    }
    let mut collector = if collect_positions {
        TopKCollector::with_positions(segment_limit)
    } else {
        TopKCollector::new(segment_limit)
    };
    if segment_limit > 0 && scorer.supports_candidate_score_bounds() {
        drive_ranked_candidates(scorer, &mut collector, budget, map);
    } else {
        drive_collected(scorer, &mut collector, map, budget);
    }
    collector.into_results_with_count()
}

/// The existing top-k heap owns competitive decisions. Complete and arbitrary
/// collectors never enter this driver, so bounds cannot truncate their counts.
fn drive_ranked_candidates(
    scorer: &mut dyn super::Scorer,
    collector: &mut TopKCollector,
    budget: Option<&super::SharedThreshold>,
    map: Option<&crate::segment::chunk_map::ChunkMap>,
) {
    let mut doc = scorer.doc();
    let mut next_block_check = 0;
    let mut confirmations = 0usize;
    let mut seed_attempted = false;
    let mut seeded_floor = Score::NEG_INFINITY;
    while doc != TERMINATED {
        if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
            break;
        }
        if doc >= next_block_check && (collector.is_full() || seeded_floor.is_finite()) {
            if let Some((last, bound)) = scorer.candidate_block_upper_bound() {
                next_block_check = last.saturating_add(1);
                // ID zero is the best possible stable-ID tie, including mapped
                // physical order. Reject only when no document can enter the heap.
                if bound < seeded_floor || !collector.would_collect(0, bound) {
                    doc = scorer.seek_candidate(next_block_check);
                    continue;
                }
            } else {
                next_block_check = TERMINATED;
            }
        }
        let result_doc = map.map_or(doc, |map| map.doc_id(doc));
        // Skip the bound computation while the heap is still filling.
        let competitive = if collector.is_full() || seeded_floor.is_finite() {
            let bound = scorer.candidate_score_upper_bound();
            bound >= seeded_floor && collector.would_collect(result_doc, bound)
        } else {
            true
        };
        if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
            break;
        }
        confirmations += usize::from(competitive);
        if competitive && scorer.confirm_candidate() {
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            let score = scorer.score();
            if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                break;
            }
            if collector.needs_positions() && collector.would_collect(result_doc, score) {
                let positions = scorer.matched_positions().unwrap_or_default();
                if budget.is_some_and(super::SharedThreshold::stop_if_expired) {
                    break;
                }
                collector.collect_owned(result_doc, score, positions);
            } else {
                collector.collect(result_doc, score, &[]);
            }
        }
        if map.is_some() && confirmations >= 1024 && !seed_attempted {
            seed_attempted = true;
            if let Some(score) = scorer.seed_ranked_score(collector.k)
                && score.is_finite()
            {
                seeded_floor = score;
            }
        }
        // Without a mapping, monotone IDs after this candidate cannot replace
        // an equal-score ID already in the full local heap. Physical RGB order
        // does not establish that relationship, so retain equality there.
        doc = scorer.advance_competitive_candidate(
            collector.competitive_score().max(seeded_floor),
            map.is_some(),
        );
    }
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
    global_stats: Option<std::sync::Arc<super::GlobalStats>>,
) -> Result<(Vec<SearchResult>, u32)> {
    reader.check_posting_integrity()?;
    let segment_limit = limit.min(reader.num_docs() as usize);
    let mut options = super::ScorerOptions {
        physical_text_field: None,
        complete_text_matches: false,
        ranked_count_limit: None,
        skip_scoring_setup: false,
        eligibility: reader.alive_docs(),
        collect_positions,
        initial_threshold: shared_threshold.get(),
        shared_threshold: Some(shared_threshold.clone()),
        lsp_plan,
        global_stats,
    };
    let map = super::text_mapping::prepare(reader, query, &mut options, false);
    let scorer = query
        .scorer_with_options(reader, segment_limit, options)
        .await?;
    let mut scorer = super::text_mapping::filtered(scorer, reader.alive_docs(), map);
    let results = top_k_from_mapped_scorer(
        scorer.as_mut(),
        segment_limit,
        collect_positions,
        Some(&shared_threshold),
        map,
    );
    reader.check_posting_integrity()?;
    Ok(results)
}

// Borrowed candidate lists clone only after their retained-output budget is checked.
impl From<&SearchResult> for SearchResult {
    fn from(value: &SearchResult) -> Self {
        value.clone()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_score_heap_preserves_total_order_and_exact_bits() {
        let mut bits = vec![
            0, 1, 0x7f800000, 0xff800000, 0x80000000, 0x7fc00000, 0xffc00000, 0x7fffffff,
            0xffffffff, 0x7f800001, 0xff800001,
        ];
        let mut state = 317u32;
        for _ in 0..2048 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bits.push(state);
        }
        let mut actual = Vec::new();
        let mut expected = Vec::new();
        for &b in &bits {
            for doc in [0, 1, 713, u32::MAX] {
                let entry = super::ScoreOnlyResult::new(doc, f32::from_bits(b));
                assert_eq!((entry.doc_id(), entry.score().to_bits()), (doc, b));
                actual.push(entry);
                expected.push((doc, f32::from_bits(b)));
            }
        }
        actual.sort_unstable();
        expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        assert_eq!(
            actual
                .iter()
                .map(|x| (x.doc_id(), x.score().to_bits()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|x| (x.0, x.1.to_bits()))
                .collect::<Vec<_>>()
        );
        assert_eq!(std::mem::size_of::<super::ScoreOnlyResult>(), 8);
    }

    use super::*;

    struct BoundedCandidates {
        hits: Vec<(DocId, Score, Score, bool)>,
        at: usize,
        confirmations: std::sync::Arc<std::sync::Mutex<Vec<DocId>>>,
        scores: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        bounds: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        pause_confirmation: bool,
        seed_calls: usize,
    }

    impl BoundedCandidates {
        fn fixture() -> Self {
            Self {
                hits: vec![
                    (0, 5.0, 5.0, true),
                    (1, 0.0, 9.0, false),
                    (2, 1.0, 1.0, true),
                    (3, 7.0, 7.0, true),
                    (4, 7.0, 7.0, true),
                    (5, 9.0, 9.0, true),
                ],
                at: 0,
                confirmations: Default::default(),
                scores: Default::default(),
                bounds: Default::default(),
                pause_confirmation: false,
                seed_calls: 0,
            }
        }
    }

    impl super::super::DocSet for BoundedCandidates {
        fn doc(&self) -> DocId {
            self.hits.get(self.at).map_or(TERMINATED, |hit| hit.0)
        }
        fn advance(&mut self) -> DocId {
            self.at += 1;
            while self.at < self.hits.len() && !super::super::Scorer::confirm_candidate(self) {
                self.at += 1;
            }
            self.doc()
        }
        fn size_hint(&self) -> u32 {
            self.hits.len() as u32
        }
    }

    impl super::super::Scorer for BoundedCandidates {
        fn seed_ranked_score(&mut self, limit: usize) -> Option<Score> {
            self.seed_calls += 1;
            let mut scores: Vec<_> = self.hits[self.at + 1..]
                .iter()
                .filter(|hit| hit.3)
                .map(|hit| hit.1)
                .collect();
            scores.sort_unstable_by(|a, b| b.total_cmp(a));
            scores.get(limit - 1).copied()
        }
        fn supports_candidate_score_bounds(&self) -> bool {
            true
        }
        fn candidate_score_upper_bound(&self) -> Score {
            self.bounds
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.hits[self.at].2
        }
        fn advance_candidate(&mut self) -> DocId {
            self.at += 1;
            super::super::DocSet::doc(self)
        }
        fn confirm_candidate(&mut self) -> bool {
            self.confirmations
                .lock()
                .unwrap()
                .push(self.hits[self.at].0);
            if self.pause_confirmation {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            self.hits[self.at].3
        }
        fn score(&self) -> Score {
            assert!(self.hits[self.at].3);
            self.scores
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.hits[self.at].1
        }
        fn matched_positions(&self) -> Option<super::super::MatchedPositions> {
            Some(vec![(
                7,
                vec![ScoredPosition::new(
                    self.hits[self.at].0,
                    self.hits[self.at].1,
                )],
            )])
        }
    }

    #[test]
    fn mapped_score_seeding_keeps_late_equal_score_winners_and_underfilled_heaps() {
        use crate::directories::OwnedBytes;
        use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
        let mut map = ChunkMapBuilder::default();
        map.set_document_units(true);
        for doc in 0..2048 {
            map.push(2047 - doc, 0, 100).unwrap();
        }
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(0, &map)], &[]).unwrap();
        let maps = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
        for prefix_matches in [false, true] {
            let mut scorer = BoundedCandidates::fixture();
            scorer.hits = (0..2048)
                .map(|doc| {
                    if doc < 2000 {
                        (doc, 1.0, 1.5, prefix_matches)
                    } else {
                        (doc, 2.0, 2.0, true)
                    }
                })
                .collect();
            let (hits, _) =
                top_k_from_mapped_scorer(&mut scorer, 10, true, None, Some(&maps.chunk_maps[&0]));
            assert_eq!(
                hits.iter()
                    .map(|hit| (hit.doc_id, hit.score))
                    .collect::<Vec<_>>(),
                (0..10).map(|doc| (doc, 2.0)).collect::<Vec<_>>()
            );
            assert_eq!(scorer.seed_calls, 1);
            assert_eq!(scorer.confirmations.lock().unwrap().len(), 1072);
        }
        let mut scorer = BoundedCandidates::fixture();
        scorer.hits = (0..2048)
            .map(|doc| (doc, if doc < 2000 { 1.0 } else { 2.0 }, 2.0, true))
            .collect();
        let mut filtered = super::super::PredicatedScorer::new(
            Box::new(scorer),
            vec![Box::new(|doc| doc < 2000)],
            Vec::new(),
            Vec::new(),
        );
        // A child proof may contain documents rejected by its wrapper.
        assert_eq!(
            super::super::Scorer::seed_ranked_score(&mut filtered, 10),
            None
        );
        let (hits, _) =
            top_k_from_mapped_scorer(&mut filtered, 10, false, None, Some(&maps.chunk_maps[&0]));
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.doc_id, hit.score))
                .collect::<Vec<_>>(),
            (48..58).map(|doc| (doc, 1.0)).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn two_term_union_counts_preserve_overlap_duplicates_fields_and_declining_collectors() {
        use crate::query::{BooleanQuery, TermQuery};
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use crate::structures::PostingCodec;
        use std::sync::Arc;
        struct Ids(Vec<u32>);
        impl Collector for Ids {
            fn needs_scores(&self) -> bool {
                false
            }
            fn collect(&mut self, doc: u32, _: f32, _: &[(u32, Vec<ScoredPosition>)]) {
                self.0.push(doc);
            }
        }
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let dir = crate::RamDirectory::new();
            let mut schema = crate::SchemaBuilder::default();
            let a = schema.add_text_field("a", true, false);
            let b = schema.add_text_field("b", true, false);
            let schema = Arc::new(schema.build());
            let mut builder = SegmentBuilder::new(
                schema.clone(),
                SegmentBuilderConfig {
                    posting_codec: codec,
                    ..Default::default()
                },
            )
            .unwrap();
            for i in 0..1701 {
                let mut doc = crate::Document::new();
                doc.add_text(
                    a,
                    match (i % 2 == 0, i % 3 == 0) {
                        (true, true) => "alpha beta alpha",
                        (true, false) => "alpha",
                        (false, true) => "beta",
                        _ => "padding",
                    },
                );
                doc.add_text(b, if i % 5 == 0 { "alpha" } else { "padding" });
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            let mut reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
            let queries = [
                (
                    BooleanQuery::new()
                        .should(TermQuery::text(a, "alpha"))
                        .should(TermQuery::text(a, "beta")),
                    (0..1701)
                        .filter(|i| i % 2 == 0 || i % 3 == 0)
                        .collect::<Vec<u32>>(),
                    true,
                ),
                (
                    BooleanQuery::new()
                        .should(TermQuery::text(a, "alpha"))
                        .should(TermQuery::text(a, "alpha")),
                    (0..1701).filter(|i| i % 2 == 0).collect(),
                    true,
                ),
                (
                    BooleanQuery::new()
                        .should(TermQuery::text(a, "alpha"))
                        .should(TermQuery::text(b, "alpha")),
                    (0..1701).filter(|i| i % 2 == 0 || i % 5 == 0).collect(),
                    true,
                ),
                (
                    BooleanQuery::new()
                        .should(TermQuery::text(a, "alpha"))
                        .should(TermQuery::text(a, "missing")),
                    (0..1701).filter(|i| i % 2 == 0).collect(),
                    false,
                ),
                (
                    BooleanQuery::new()
                        .should(TermQuery::text(a, "missing"))
                        .should(TermQuery::text(b, "missing")),
                    Vec::new(),
                    false,
                ),
            ];
            for (query, expected, admitted) in &queries {
                let known = two_term_union_count(&reader, query, 0).await.unwrap();
                assert_eq!(known, admitted.then_some(expected.len() as u64));
                let mut count = CountCollector::new();
                collect_segment(&reader, query, &mut count).await.unwrap();
                assert_eq!(count.count(), expected.len() as u64);
                let mut ids = Ids(Vec::new());
                collect_segment(&reader, query, &mut ids).await.unwrap();
                assert_eq!(&ids.0, expected);
            }
            let mut alive = crate::query::DocBitset::all(1701);
            alive.clear(0);
            alive.clear(6);
            let deletion = crate::segment::deletion::write(&dir, SegmentId::new(), 1701, &alive)
                .await
                .unwrap();
            reader.load_deletions(&dir, deletion).await.unwrap();
            for (query, expected, _) in &queries {
                assert_eq!(two_term_union_count(&reader, query, 0).await.unwrap(), None);
                let expected = expected.iter().filter(|&&d| d != 0 && d != 6).count() as u64;
                let mut count = CountCollector::new();
                collect_segment(&reader, query, &mut count).await.unwrap();
                assert_eq!(count.count(), expected);
            }
        }
    }

    #[tokio::test]
    async fn union_count_metadata_shortcut_declines_mapped_and_fast_only_fields() {
        use crate::query::{BooleanQuery, TermQuery};
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use std::sync::Arc;
        for chunked in [false, true] {
            let dir = crate::RamDirectory::new();
            let mut schema = crate::SchemaBuilder::default();
            let field = schema.add_text_field("text", chunked, false);
            if chunked {
                schema.set_chunked(field, true);
            } else {
                schema.set_fast(field, true);
            }
            let schema = Arc::new(schema.build());
            let mut builder =
                SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
            for values in [vec!["alpha", "beta", "alpha"], vec!["alpha"], vec!["gamma"]] {
                let mut doc = crate::Document::new();
                if chunked {
                    for value in values {
                        doc.add_text(field, value);
                    }
                } else {
                    doc.add_text(field, values[0]);
                }
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
            let query = BooleanQuery::new()
                .should(TermQuery::text(field, "alpha"))
                .should(TermQuery::text(field, "beta"));
            assert_eq!(
                two_term_union_count(&reader, &query, 0).await.unwrap(),
                None
            );
            let mut count = CountCollector::new();
            collect_segment(&reader, &query, &mut count).await.unwrap();
            assert_eq!(count.count(), 2);
        }
    }

    #[tokio::test]
    async fn fast_only_unions_keep_column_matches_in_count_and_ranked_collection() {
        use crate::query::{BooleanQuery, TermQuery};
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use std::sync::Arc;
        let dir = crate::RamDirectory::new();
        let mut schema = crate::SchemaBuilder::default();
        let field = schema.add_text_field("text", false, false);
        schema.set_fast(field, true);
        let schema = Arc::new(schema.build());
        let mut builder =
            SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
        for text in ["alpha", "alpha", "beta", "gamma"] {
            let mut doc = crate::Document::new();
            doc.add_text(field, text);
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
        let mut single = CountCollector::new();
        collect_segment(&reader, &TermQuery::text(field, "alpha"), &mut single)
            .await
            .unwrap();
        assert_eq!(single.count(), 2);
        let query = BooleanQuery::new()
            .should(TermQuery::text(field, "alpha"))
            .should(TermQuery::text(field, "beta"));
        let mut count = CountCollector::new();
        collect_segment(&reader, &query, &mut count).await.unwrap();
        assert_eq!(count.count(), 3);
        let mut top = TopKCollector::new(2);
        collect_segment_with_limit(&reader, &query, &mut top, 2)
            .await
            .unwrap();
        let hits = top.into_sorted_results();
        assert_eq!(
            hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
            [0, 1]
        );
        assert!(hits.iter().all(|hit| hit.score == 1.0));
        #[cfg(feature = "sync")]
        {
            let mut top = TopKCollector::new(2);
            collect_segment_with_limit_sync(&reader, &query, &mut top, 2).unwrap();
            assert_eq!(top.into_sorted_results(), hits);
        }
    }

    #[tokio::test]
    async fn separate_ranked_counts_preserve_ties_nested_collectors_and_existing_hits() {
        use crate::query::{BooleanQuery, TermQuery};
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use crate::structures::postings::PostingCodec;
        use std::sync::Arc;
        struct Exhaustive<C>(C);
        impl<C: Collector> Collector for Exhaustive<C> {
            fn collect(&mut self, doc: u32, score: f32, positions: &[(u32, Vec<ScoredPosition>)]) {
                self.0.collect(doc, score, positions);
            }
            fn needs_scores(&self) -> bool {
                self.0.needs_scores()
            }
            fn needs_positions(&self) -> bool {
                self.0.needs_positions()
            }
        }
        fn hits(top: TopKCollector) -> Vec<(u32, u32)> {
            top.into_sorted_results()
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect()
        }
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let dir = crate::RamDirectory::new();
            let mut schema = crate::SchemaBuilder::default();
            let field = schema.add_text_field("text", true, false);
            let schema = Arc::new(schema.build());
            let mut builder = SegmentBuilder::new(
                schema.clone(),
                SegmentBuilderConfig {
                    posting_codec: codec,
                    ..Default::default()
                },
            )
            .unwrap();
            for i in 0..24001 {
                let text = match i % 4 {
                    0 => "alpha beta alpha",
                    1 => "alpha",
                    2 => "beta filler filler",
                    _ => "gamma",
                };
                let mut doc = crate::Document::new();
                doc.add_text(field, text);
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
            let queries: Vec<Box<dyn Query>> = vec![
                Box::new(super::super::BoostQuery::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .must(TermQuery::text(field, "beta")),
                    1.0,
                )),
                Box::new(super::super::FilteredQuery::new(
                    Arc::new(
                        BooleanQuery::new()
                            .must(TermQuery::text(field, "alpha"))
                            .must(TermQuery::text(field, "beta")),
                    ),
                    vec![Arc::new(TermQuery::text(field, "beta"))],
                )),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .must(TermQuery::text(field, "beta")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "beta"))
                        .must(TermQuery::text(field, "alpha")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .must(TermQuery::text(field, "alpha")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .must(TermQuery::text(field, "missing")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .must(TermQuery::text(field, "beta"))
                        .must(TermQuery::text(field, "gamma")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(
                            BooleanQuery::new()
                                .must(TermQuery::text(field, "alpha"))
                                .must(TermQuery::text(field, "beta")),
                        )
                        .should(TermQuery::text(field, "gamma")),
                ),
                Box::new(TermQuery::text(field, "alpha")),
                Box::new(
                    BooleanQuery::new()
                        .should(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "beta")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .should(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "alpha")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .should(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "missing")),
                ),
                Box::new(TermQuery::text(field, "missing")),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "beta")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "alpha")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "missing")),
                ),
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(field, "alpha"))
                        .should(TermQuery::text(field, "beta"))
                        .should(TermQuery::text(field, "gamma")),
                ),
            ];
            let required = BooleanQuery::new()
                .must(TermQuery::text(field, "alpha"))
                .should(TermQuery::text(field, "beta"));
            assert_eq!(
                ranked_exact_count(&reader, &required, 17).await.unwrap(),
                Some(12001)
            );
            assert_eq!(
                ranked_exact_count(&reader, &required, 5000).await.unwrap(),
                None
            );
            let conjunction = BooleanQuery::new()
                .must(TermQuery::text(field, "alpha"))
                .must(TermQuery::text(field, "beta"));
            let counted = conjunction
                .scorer_with_options(
                    &reader,
                    usize::MAX / 2,
                    super::super::ScorerOptions {
                        complete_text_matches: true,
                        ranked_count_limit: Some(17),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert_eq!(counted.exact_ranked_count(), Some(6001));
            for deadline in [
                std::time::Instant::now(),
                std::time::Instant::now() + std::time::Duration::from_secs(60),
            ] {
                let bounded = conjunction
                    .scorer_with_options(
                        &reader,
                        usize::MAX / 2,
                        super::super::ScorerOptions {
                            complete_text_matches: true,
                            ranked_count_limit: Some(17),
                            shared_threshold: Some(
                                super::super::SharedThreshold::for_limit(17)
                                    .with_deadline(Some(deadline)),
                            ),
                            ..Default::default()
                        },
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    bounded.exact_ranked_count(),
                    None,
                    "a budgeted stream cannot promise a complete count"
                );
            }

            for query in queries {
                for k in [0, 1, 17, 5000] {
                    let mut actual = TopKCollector::new(k);
                    let mut expected = TopKCollector::new(k);
                    let mut a_count = CountCollector::new();
                    let mut e_count = CountCollector::new();
                    actual.collect(90000, 1e6, &[]);
                    expected.collect(90000, 1e6, &[]);
                    a_count.collect_count(9);
                    e_count.collect_count(9);
                    collect_segment(&reader, query.as_ref(), &mut (&mut actual, &mut a_count))
                        .await
                        .unwrap();
                    collect_segment(
                        &reader,
                        query.as_ref(),
                        &mut Exhaustive((&mut expected, &mut e_count)),
                    )
                    .await
                    .unwrap();
                    assert_eq!(a_count.count(), e_count.count(), "{codec:?} k={k}");
                    assert_eq!(actual.total_seen(), expected.total_seen());
                    assert_eq!(hits(actual), hits(expected), "{codec:?} k={k}");
                }
            }
            let query = BooleanQuery::new()
                .should(TermQuery::text(field, "alpha"))
                .should(TermQuery::text(field, "beta"));
            let mut small = TopKCollector::new(3);
            let mut large = TopKCollector::new(29);
            let mut count = CountCollector::new();
            let mut nested = (&mut small, &mut count);
            let mut tuple = (&mut nested, &mut large);
            assert_eq!(tuple.ranked_count_limit(), Some(29));
            collect_segment(&reader, &query, &mut tuple).await.unwrap();
            let mut expected = TopKCollector::new(29);
            let mut ignored_count = CountCollector::new();
            collect_segment(
                &reader,
                &query,
                &mut Exhaustive((&mut expected, &mut ignored_count)),
            )
            .await
            .unwrap();
            let expected = hits(expected);
            assert_eq!(hits(small), expected[..3]);
            assert_eq!(hits(large), expected);
            assert_eq!(count.count(), 18001);
            assert_eq!(TopKCollector::with_positions(3).ranked_count_limit(), None);
        }
    }

    #[test]
    fn mapped_ranked_handoff_translates_before_truncating_stable_ties() {
        use crate::directories::OwnedBytes;
        use crate::query::docset::DocSet;
        use crate::segment::chunk_map::{ChunkMapBuilder, read_chunk_maps, write_chunk_maps};
        let mut builder = ChunkMapBuilder::default();
        builder.set_document_units(true);
        for doc in [3, 2, 0, 1] {
            builder.push(doc, 0, 1).unwrap();
        }
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
        let maps = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
        let map = &maps.chunk_maps[&0];
        for limit in [0, 1, 2, 4, 8] {
            for positions in [false, true] {
                let ranked = || {
                    super::super::planner::TopKResultScorer::new(
                        (0..4)
                            .map(|doc_id| super::super::ScoredDoc {
                                doc_id,
                                score: 1.0,
                                ordinal: 0,
                            })
                            .collect(),
                    )
                };
                let mut scorer = ranked();
                let actual =
                    top_k_from_mapped_scorer(&mut scorer, limit, positions, None, Some(map));
                let mut expected = TopKCollector::new(limit);
                drive_collected(&mut ranked(), &mut expected, Some(map), None);
                assert_eq!(actual, expected.into_results_with_count());
                // The handoff consumed the retained list; no second heap walk.
                assert_eq!(scorer.doc(), TERMINATED);
            }
        }
    }

    #[tokio::test]
    async fn phrase_block_pruning_preserves_late_winners_score_bits_and_exact_counts() {
        use crate::query::PhraseQuery;
        use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
        use crate::structures::PostingCodec;
        for (codec, posting_ratio_bounds, posting_impact_bounds) in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ]
        .into_iter()
        .flat_map(|codec| {
            [
                (codec, false, false),
                (codec, true, false),
                (codec, true, true),
            ]
        }) {
            let dir = crate::RamDirectory::new();
            let mut schema = crate::SchemaBuilder::default();
            let field = schema.add_text_field("body", true, false);
            schema.set_positions(field, crate::dsl::PositionMode::TokenPosition);
            let schema = std::sync::Arc::new(schema.build());
            let mut builder = SegmentBuilder::new(
                schema.clone(),
                SegmentBuilderConfig {
                    posting_codec: codec,
                    posting_ratio_bounds,
                    posting_impact_bounds,
                    ..Default::default()
                },
            )
            .unwrap();
            for id in 0..2053 {
                // Whole weak blocks lie between short documents and later
                // higher-frequency winners. Include nonmatching intersections.
                let text = match id {
                    0..=3 => "alpha beta".to_owned(),
                    2048.. => "alpha beta alpha beta".to_owned(),
                    _ if id % 7 == 0 => format!("alpha {} beta", "filler ".repeat(80)),
                    _ => format!("alpha beta {}", "filler ".repeat(80)),
                };
                let mut doc = crate::Document::new();
                doc.add_text(field, text);
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
            for slop in [0, 4] {
                let query = PhraseQuery::new(field, vec![b"alpha".to_vec(), b"beta".to_vec()])
                    .with_slop(slop);
                for k in [1, 3, 10, 100] {
                    let mut complete = query.scorer(&reader, 0).await.unwrap();
                    let mut top = TopKCollector::new(k);
                    let mut count = CountCollector::new();
                    drive_scorer(complete.as_mut(), &mut (&mut top, &mut count));
                    let expected = top.into_sorted_results();
                    let (actual, _) = search_segment_with_count(&reader, &query, k).await.unwrap();
                    assert_eq!(actual, expected, "{codec:?}, slop={slop}, k={k}");
                    #[cfg(feature = "sync")]
                    assert_eq!(
                        search_segment_with_count_sync(&reader, &query, k)
                            .unwrap()
                            .0,
                        expected
                    );
                    let mut exact = CountCollector::new();
                    collect_segment(&reader, &query, &mut exact).await.unwrap();
                    assert_eq!(exact.count(), count.count());
                    assert!(exact.count() > 1700);
                }
            }
        }
    }

    #[test]
    fn ranked_bounds_skip_confirmation_but_preserve_late_winners_ties_and_complete_counts() {
        for positions in [false, true] {
            let mut ranked = BoundedCandidates::fixture();
            let (actual, seen) = top_k_from_scorer(&mut ranked, 1, positions, None);
            assert_eq!(actual[0].doc_id, 5);
            assert_eq!(seen, 3);
            assert_eq!(*ranked.confirmations.lock().unwrap(), [0, 1, 3, 5]);
            assert_eq!(ranked.scores.load(std::sync::atomic::Ordering::Relaxed), 3);
            let mut complete = BoundedCandidates::fixture();
            let mut top = if positions {
                TopKCollector::with_positions(1)
            } else {
                TopKCollector::new(1)
            };
            let mut count = CountCollector::new();
            drive_scorer(&mut complete, &mut (&mut top, &mut count));
            assert_eq!(actual, top.into_sorted_results());
            assert_eq!(count.count(), 5);
            assert_eq!(
                complete.bounds.load(std::sync::atomic::Ordering::Relaxed),
                0
            );
            let mut count_only = BoundedCandidates::fixture();
            let mut count = CountCollector::new();
            drive_scorer(&mut count_only, &mut count);
            assert_eq!(count.count(), 5);
            assert_eq!(
                count_only.bounds.load(std::sync::atomic::Ordering::Relaxed),
                0
            );
            assert_eq!(
                count_only.scores.load(std::sync::atomic::Ordering::Relaxed),
                0
            );
        }
    }

    #[test]
    fn deadline_during_candidate_confirmation_prevents_scoring_and_collection() {
        let budget = super::super::SharedThreshold::for_limit(1).with_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_millis(20),
        ));
        let mut scorer = BoundedCandidates::fixture();
        scorer.pause_confirmation = true;
        let (hits, seen) = top_k_from_scorer(&mut scorer, 1, true, Some(&budget));
        assert!(hits.is_empty());
        assert_eq!(seen, 0);
        assert_eq!(scorer.scores.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert!(budget.truncated());
    }

    #[test]
    fn eligibility_wrappers_keep_exact_confirmation_and_do_not_inherit_leaf_bounds() {
        let candidates = BoundedCandidates::fixture();
        let bounds = candidates.bounds.clone();
        let mut alive = super::super::DocBitset::new(6);
        for doc in [2, 3, 4] {
            alive.set(doc);
        }
        let mut filtered = super::super::filtered::filtered(
            Box::new(candidates),
            Some(std::sync::Arc::new(alive)),
        );
        assert!(!filtered.supports_candidate_score_bounds());
        let (hits, seen) = top_k_from_scorer(filtered.as_mut(), 1, false, None);
        assert_eq!(hits[0].doc_id, 3);
        assert_eq!(seen, 3);
        assert_eq!(bounds.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn scored_blocks_preserve_total_float_order_ties_and_exact_counts() {
        let special = [
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::INFINITY,
            f32::from_bits(0xffc00001),
            f32::from_bits(0x7fc00001),
            f32::from_bits(0x7fc00002),
        ];
        for k in [0, 1, 10, 63, 64, 65, 127, 1000] {
            let mut blocks = TopKCollector::new(k);
            let mut scalar = TopKCollector::new(k);
            let mut count = CountCollector::new();
            let mut count2 = CountCollector::new();
            let mut expected = 0;
            for (round, base) in [128, 0, 256, 64, 320, u32::MAX - 64]
                .into_iter()
                .enumerate()
            {
                let scores = std::array::from_fn(|i| special[(i + round * 3) % special.len()]);
                for bits in [0, u64::MAX, 1, 1u64 << 63, 0xaaaaf00f800102a8] {
                    let mut tuple = (&mut blocks, &mut count, &mut count2);
                    assert!(tuple.supports_score_blocks());
                    tuple.collect_score_block(base, &scores, bits);
                    for (i, &score) in scores.iter().enumerate() {
                        if bits & (1u64 << i) != 0 {
                            scalar.collect(base + i as u32, score, &[]);
                            expected += 1;
                        }
                    }
                }
            }
            assert_eq!(count.count(), expected);
            assert_eq!(count2.count(), expected);
            assert_eq!(blocks.total_seen(), expected as u32);
            assert_eq!(blocks.into_sorted_results(), scalar.into_sorted_results());
        }
        let mut top = TopKCollector::new(0);
        top.total_seen = u32::MAX - 5;
        top.collect_score_block(0, &[1.0; 64], u64::MAX);
        assert_eq!(top.total_seen(), u32::MAX);
        assert!(!TopKCollector::with_positions(10).supports_score_blocks());
    }

    #[test]
    fn custom_collectors_keep_interleaved_callbacks_inside_score_windows() {
        use super::super::{DocSet, Scorer};
        use std::cell::RefCell;
        struct Records<'a> {
            child: u8,
            calls: &'a RefCell<Vec<(DocId, u8, u32)>>,
        }
        impl Collector for Records<'_> {
            fn collect(
                &mut self,
                doc: DocId,
                score: Score,
                positions: &[(u32, Vec<ScoredPosition>)],
            ) {
                assert!(positions.is_empty());
                self.calls
                    .borrow_mut()
                    .push((doc, self.child, score.to_bits()));
            }
            fn collect_score_block(&mut self, _: DocId, _: &[Score; 64], _: u64) {
                panic!("custom collector has not opted in");
            }
        }
        struct Window(bool);
        impl DocSet for Window {
            fn doc(&self) -> DocId {
                if self.0 { TERMINATED } else { 0 }
            }
            fn advance(&mut self) -> DocId {
                panic!("use score window")
            }
            fn size_hint(&self) -> u32 {
                4
            }
        }
        impl Scorer for Window {
            fn score(&self) -> Score {
                panic!("use score window")
            }
            fn supports_score_windows(&self) -> bool {
                true
            }
            fn fill_score_window(
                &mut self,
                _: DocId,
                scores: &mut [Score; super::super::docset::DOC_WINDOW_SIZE as usize],
                bits: &mut super::super::docset::DocWindow,
            ) {
                bits.fill(0);
                bits[0] = 3;
                bits[1] = 1 << 63;
                bits[63] = 1 << 63;
                for doc in [0, 1, 127, 4095] {
                    scores[doc] = doc as f32;
                }
                self.0 = true;
            }
        }
        let calls = RefCell::new(Vec::new());
        let mut a = Records {
            child: 0,
            calls: &calls,
        };
        let mut b = Records {
            child: 1,
            calls: &calls,
        };
        let mut top = TopKCollector::new(2);
        let mut collector = (&mut a, &mut b, &mut top);
        assert!(!collector.supports_score_blocks());
        drive_scorer(&mut Window(false), &mut collector);
        let expected: Vec<_> = [0, 1, 127, 4095]
            .into_iter()
            .flat_map(|doc| {
                [
                    (doc, 0, (doc as f32).to_bits()),
                    (doc, 1, (doc as f32).to_bits()),
                ]
            })
            .collect();
        assert_eq!(*calls.borrow(), expected);
        assert_eq!(top.total_seen(), 4);
        assert_eq!(
            top.into_sorted_results()
                .iter()
                .map(|x| x.doc_id)
                .collect::<Vec<_>>(),
            [4095, 127]
        );
        let mut top = TopKCollector::new(2);
        let mut count = CountCollector::new();
        drive_scorer(&mut Window(false), &mut (&mut top, &mut count));
        assert_eq!(count.count(), 4);
        assert_eq!(top.total_seen(), 4);
        assert_eq!(
            top.into_sorted_results()
                .iter()
                .map(|x| x.doc_id)
                .collect::<Vec<_>>(),
            [4095, 127]
        );
    }
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    fn expired_score_batch_is_discarded_before_any_collection() {
        use super::super::{DocSet, Scorer, SharedThreshold};
        struct ExpiresInScoreBatch {
            deadline: std::time::Instant,
            visited: bool,
        }
        impl DocSet for ExpiresInScoreBatch {
            fn doc(&self) -> u32 {
                if self.visited { TERMINATED } else { 0 }
            }
            fn advance(&mut self) -> u32 {
                panic!("must use batch")
            }
            fn size_hint(&self) -> u32 {
                4096
            }
        }
        impl Scorer for ExpiresInScoreBatch {
            fn supports_score_batches(&self) -> bool {
                true
            }
            fn fill_score_batch(
                &mut self,
                docs: &mut super::super::docset::DocBatch,
                scores: &mut super::super::ScoreBatch,
            ) -> usize {
                docs[0] = 0;
                scores[0] = 5.0;
                self.visited = true;
                std::thread::sleep(
                    self.deadline
                        .saturating_duration_since(std::time::Instant::now())
                        + std::time::Duration::from_millis(1),
                );
                1
            }
            fn score(&self) -> f32 {
                panic!("count must not score")
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        let budget = SharedThreshold::new().with_deadline(Some(deadline));
        let mut scorer = ExpiresInScoreBatch {
            deadline,
            visited: false,
        };
        let mut count = CountCollector::new();
        let mut top = TopKCollector::new(10);
        drive_scorer_budgeted(&mut scorer, &mut (&mut top, &mut count), Some(&budget));
        assert!(scorer.visited);
        assert!(budget.truncated());
        assert_eq!(count.count(), 0);
        assert_eq!(top.total_seen(), 0);
        assert!(top.into_sorted_results().is_empty());
    }

    #[test]
    fn expired_document_batch_is_discarded_before_count_collection() {
        use super::super::{DocSet, Scorer, SharedThreshold};
        struct ExpiresInBatch {
            deadline: std::time::Instant,
            visited: bool,
        }
        impl DocSet for ExpiresInBatch {
            fn doc(&self) -> u32 {
                if self.visited { TERMINATED } else { 0 }
            }
            fn advance(&mut self) -> u32 {
                panic!("must use batch")
            }
            fn size_hint(&self) -> u32 {
                4096
            }
            fn supports_doc_batches(&self) -> bool {
                true
            }
            fn fill_doc_batch(&mut self, docs: &mut super::super::docset::DocBatch) -> usize {
                docs[0] = 0;
                self.visited = true;
                std::thread::sleep(
                    self.deadline
                        .saturating_duration_since(std::time::Instant::now())
                        + std::time::Duration::from_millis(1),
                );
                1
            }
        }
        impl Scorer for ExpiresInBatch {
            fn score(&self) -> f32 {
                panic!("count must not score")
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        let budget = SharedThreshold::new().with_deadline(Some(deadline));
        let mut scorer = ExpiresInBatch {
            deadline,
            visited: false,
        };
        let mut count = CountCollector::new();
        drive_scorer_budgeted(&mut scorer, &mut count, Some(&budget));
        assert!(scorer.visited);
        assert!(budget.truncated());
        assert_eq!(count.count(), 0);
    }

    #[test]
    fn expired_document_window_is_discarded_before_count_collection() {
        use super::super::{DocSet, Scorer, SharedThreshold};
        struct ExpiresInWindow {
            deadline: std::time::Instant,
            visited: bool,
        }
        impl DocSet for ExpiresInWindow {
            fn doc(&self) -> u32 {
                if self.visited { TERMINATED } else { 0 }
            }
            fn advance(&mut self) -> u32 {
                panic!("must use window")
            }
            fn size_hint(&self) -> u32 {
                4096
            }
            fn supports_doc_windows(&self) -> bool {
                true
            }
            fn fill_doc_window(&mut self, _base: u32, bits: &mut super::super::docset::DocWindow) {
                bits.fill(u64::MAX);
                self.visited = true;
                std::thread::sleep(
                    self.deadline
                        .saturating_duration_since(std::time::Instant::now())
                        + std::time::Duration::from_millis(1),
                );
            }
        }
        impl Scorer for ExpiresInWindow {
            fn score(&self) -> f32 {
                panic!("count must not score")
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        let budget = SharedThreshold::new().with_deadline(Some(deadline));
        let mut scorer = ExpiresInWindow {
            deadline,
            visited: false,
        };
        let mut count = CountCollector::new();
        drive_scorer_budgeted(&mut scorer, &mut count, Some(&budget));
        assert!(scorer.visited);
        assert!(budget.truncated());
        assert_eq!(count.count(), 0);
    }

    #[test]
    fn expired_score_window_is_discarded_before_rank_and_count_collection() {
        use super::super::{DocSet, Scorer, SharedThreshold};
        struct ExpiresInWindow {
            deadline: std::time::Instant,
            visited: bool,
        }
        impl DocSet for ExpiresInWindow {
            fn doc(&self) -> u32 {
                if self.visited { TERMINATED } else { 0 }
            }
            fn advance(&mut self) -> u32 {
                panic!("must use score window")
            }
            fn size_hint(&self) -> u32 {
                4096
            }
        }
        impl Scorer for ExpiresInWindow {
            fn score(&self) -> f32 {
                panic!("must use score window")
            }
            fn supports_score_windows(&self) -> bool {
                true
            }
            fn fill_score_window(
                &mut self,
                _base: DocId,
                scores: &mut [Score; super::super::docset::DOC_WINDOW_SIZE as usize],
                bits: &mut super::super::docset::DocWindow,
            ) {
                bits.fill(u64::MAX);
                scores.fill(1.0);
                self.visited = true;
                std::thread::sleep(
                    self.deadline
                        .saturating_duration_since(std::time::Instant::now())
                        + std::time::Duration::from_millis(1),
                );
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
        let budget = SharedThreshold::new().with_deadline(Some(deadline));
        let mut scorer = ExpiresInWindow {
            deadline,
            visited: false,
        };
        let mut top = TopKCollector::new(10);
        let mut count = CountCollector::new();
        drive_scorer_budgeted(&mut scorer, &mut (&mut top, &mut count), Some(&budget));
        assert!(scorer.visited);
        assert!(budget.truncated());
        assert_eq!(count.count(), 0);
        assert!(top.into_sorted_results().is_empty());
    }

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

    struct ScoreCountingScorer {
        doc: u32,
        calls: std::sync::Arc<AtomicUsize>,
    }

    impl super::super::DocSet for ScoreCountingScorer {
        fn doc(&self) -> DocId {
            if self.doc < 3 { self.doc } else { TERMINATED }
        }
        fn advance(&mut self) -> DocId {
            self.doc += 1;
            self.doc()
        }
        fn size_hint(&self) -> u32 {
            3u32.saturating_sub(self.doc)
        }
    }

    impl super::super::Scorer for ScoreCountingScorer {
        fn score(&self) -> Score {
            self.calls.fetch_add(1, AtomicOrdering::Relaxed);
            self.doc as f32 + 1.0
        }
    }

    #[test]
    fn count_only_collection_skips_scores_but_ranked_tuples_compute_them() {
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let mut scorer = ScoreCountingScorer {
            doc: 0,
            calls: calls.clone(),
        };
        let mut count = CountCollector::new();
        drive_scorer(&mut scorer, &mut count);
        assert_eq!(count.count(), 3);
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 0);

        let mut scorer = ScoreCountingScorer {
            doc: 0,
            calls: calls.clone(),
        };
        let mut count = CountCollector::new();
        let mut top = TopKCollector::new(1);
        drive_scorer(&mut scorer, &mut (&mut count, &mut top));
        assert_eq!(count.count(), 3);
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 3);
        let hits = top.into_sorted_results();
        assert_eq!((hits[0].doc_id, hits[0].score), (2, 3.0));
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
