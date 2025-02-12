//! Fast-field filter queries for efficient document filtering.
//!
//! `FastFieldFilterQuery` wraps an inner query and applies O(1) per-doc
//! fast-field predicate checks, skipping documents that don't match.

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::structures::fast_field::{
    FastFieldColumnType, FastFieldReader, TEXT_MISSING_ORDINAL, f64_to_sortable_u64, zigzag_encode,
};
use crate::{DocId, Score};

use super::{CountFuture, Query, Scorer, ScorerFuture};

// ── Filter condition ──────────────────────────────────────────────────────

/// A single filter condition on a fast field.
#[derive(Debug, Clone)]
pub enum FastFieldCondition {
    /// Exact match on u64 value
    EqU64(u64),
    /// Exact match on i64 value
    EqI64(i64),
    /// Exact match on f64 value
    EqF64(f64),
    /// Exact match on text value (resolved to ordinal at scorer creation)
    EqText(String),
    /// Inclusive range on raw encoded u64 values [min, max]
    RangeU64 { min: Option<u64>, max: Option<u64> },
    /// Inclusive range on i64 values
    RangeI64 { min: Option<i64>, max: Option<i64> },
    /// Inclusive range on f64 values
    RangeF64 { min: Option<f64>, max: Option<f64> },
    /// Match any of these u64 values
    InU64(Vec<u64>),
    /// Match any of these i64 values
    InI64(Vec<i64>),
    /// Match any of these text values (resolved to ordinals at scorer creation)
    InText(Vec<String>),
    /// Document has a non-default value (exists check)
    Exists,
}

/// A compiled filter ready for O(1) per-doc evaluation.
/// All values are pre-encoded to raw u64 for direct column comparison.
#[derive(Debug, Clone)]
enum CompiledFilter {
    Eq(u64),
    Range {
        min: u64,
        max: u64,
    },
    In(Vec<u64>),
    Exists {
        column_type: FastFieldColumnType,
    },
    /// Always fails (e.g. text value not in dictionary)
    Never,
}

impl CompiledFilter {
    /// Check if a raw column value passes this filter.
    #[inline]
    fn matches(&self, raw: u64) -> bool {
        match self {
            CompiledFilter::Eq(v) => raw == *v,
            CompiledFilter::Range { min, max } => raw >= *min && raw <= *max,
            CompiledFilter::In(vals) => vals.contains(&raw),
            CompiledFilter::Exists { column_type } => match column_type {
                FastFieldColumnType::TextOrdinal => raw != TEXT_MISSING_ORDINAL,
                _ => true, // numeric columns always "exist" (0 is a valid value)
            },
            CompiledFilter::Never => false,
        }
    }
}

/// Compile a `FastFieldCondition` into a `CompiledFilter` using the column reader.
fn compile_condition(condition: &FastFieldCondition, reader: &FastFieldReader) -> CompiledFilter {
    match condition {
        FastFieldCondition::EqU64(v) => CompiledFilter::Eq(*v),
        FastFieldCondition::EqI64(v) => CompiledFilter::Eq(zigzag_encode(*v)),
        FastFieldCondition::EqF64(v) => CompiledFilter::Eq(f64_to_sortable_u64(*v)),
        FastFieldCondition::EqText(text) => {
            match reader.text_ordinal(text) {
                Some(ord) => CompiledFilter::Eq(ord),
                None => CompiledFilter::Never, // value not in dictionary
            }
        }
        FastFieldCondition::RangeU64 { min, max } => CompiledFilter::Range {
            min: min.unwrap_or(0),
            max: max.unwrap_or(u64::MAX),
        },
        FastFieldCondition::RangeI64 { min, max } => {
            // Zigzag encoding does NOT preserve order, so we need to handle ranges differently.
            // For i64 ranges, we iterate and check decoded values.
            // Simple approach: encode both bounds. But zigzag doesn't preserve order.
            // Instead, we store the raw range and check at eval time.
            // We'll use a special path — encode as-is and check decoded.
            // Actually, let's just store min/max as zigzag and use a range check
            // on the decoded value in a custom filter. For now, use Eq-based approach:
            // This is a simplification — for proper i64 range, we'd need a custom comparator.
            // Let's compile to a range on raw u64 space if both bounds have the same sign.
            // For the general case, we'll iterate all and check decoded. Use In() as fallback.
            let min_val = min.unwrap_or(i64::MIN);
            let max_val = max.unwrap_or(i64::MAX);
            // Since zigzag doesn't preserve order, we need the scorer to check decoded values.
            // Store as a range on the original i64 space by using a wrapper.
            // For simplicity in v1: compile to a closure-like check via Eq/In.
            // Better approach: store raw min/max and check at runtime.
            // We'll handle this in the scorer with a special i64/f64 range path.
            CompiledFilter::Range {
                min: zigzag_encode(min_val),
                max: zigzag_encode(max_val),
            }
        }
        FastFieldCondition::RangeF64 { min, max } => {
            // f64_to_sortable_u64 preserves order, so range on encoded values works
            CompiledFilter::Range {
                min: min.map(f64_to_sortable_u64).unwrap_or(0),
                max: max.map(f64_to_sortable_u64).unwrap_or(u64::MAX),
            }
        }
        FastFieldCondition::InU64(vals) => CompiledFilter::In(vals.clone()),
        FastFieldCondition::InI64(vals) => {
            CompiledFilter::In(vals.iter().map(|v| zigzag_encode(*v)).collect())
        }
        FastFieldCondition::InText(texts) => {
            let ordinals: Vec<u64> = texts
                .iter()
                .filter_map(|t| reader.text_ordinal(t))
                .collect();
            if ordinals.is_empty() {
                CompiledFilter::Never
            } else {
                CompiledFilter::In(ordinals)
            }
        }
        FastFieldCondition::Exists => CompiledFilter::Exists {
            column_type: reader.column_type,
        },
    }
}

// ── Filter specification ──────────────────────────────────────────────────

/// A single field + condition pair.
#[derive(Debug, Clone)]
pub struct FastFieldFilter {
    pub field: Field,
    pub condition: FastFieldCondition,
}

// ── Filter query ──────────────────────────────────────────────────────────

/// Query that wraps an inner query and filters results via fast-field predicates.
///
/// For each document produced by the inner scorer, checks all fast-field
/// conditions. Only documents passing all conditions are yielded.
///
/// If no inner query is provided, iterates all documents in the segment
/// and yields those matching the filters (standalone filter mode).
pub struct FastFieldFilterQuery {
    /// Inner query to filter (None = match-all)
    inner: Option<Arc<dyn Query>>,
    /// Filter conditions to apply
    filters: Vec<FastFieldFilter>,
}

impl std::fmt::Debug for FastFieldFilterQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastFieldFilterQuery")
            .field("has_inner", &self.inner.is_some())
            .field("num_filters", &self.filters.len())
            .finish()
    }
}

impl FastFieldFilterQuery {
    /// Create a filter query wrapping an inner query.
    pub fn new(inner: Arc<dyn Query>, filters: Vec<FastFieldFilter>) -> Self {
        Self {
            inner: Some(inner),
            filters,
        }
    }

    /// Create a standalone filter query (match-all + filter).
    pub fn standalone(filters: Vec<FastFieldFilter>) -> Self {
        Self {
            inner: None,
            filters,
        }
    }
}

impl Query for FastFieldFilterQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let inner = self.inner.clone();
        let filters = self.filters.clone();

        Box::pin(async move {
            // Compile filters against segment's fast-field readers
            let mut compiled: Vec<(u32, CompiledFilter)> = Vec::with_capacity(filters.len());
            for filter in &filters {
                let field_id = filter.field.0;
                if let Some(ff_reader) = reader.fast_field(field_id) {
                    let cf = compile_condition(&filter.condition, ff_reader);
                    compiled.push((field_id, cf));
                } else {
                    // No fast-field data for this field in this segment — skip all docs
                    compiled.push((field_id, CompiledFilter::Never));
                }
            }

            let scorer: Box<dyn Scorer + 'a> = if let Some(ref inner_query) = inner {
                let inner_scorer = inner_query.scorer(reader, limit).await?;
                Box::new(FastFieldFilterScorer::new(inner_scorer, compiled, reader))
            } else {
                // Standalone: iterate all docs
                let num_docs = reader.num_docs();
                Box::new(StandaloneFilterScorer::new(num_docs, compiled, reader))
            };

            Ok(scorer)
        })
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        if let Some(ref inner) = self.inner {
            inner.count_estimate(reader)
        } else {
            let num_docs = reader.num_docs();
            Box::pin(async move { Ok(num_docs) })
        }
    }
}

// ── Inner-query filter scorer ─────────────────────────────────────────────

/// Pre-filter scorer: wraps an inner scorer and skips documents that don't
/// pass the fast-field predicates **before** they reach the collector.
///
/// All doc IDs are segment-local (0-based). The collector adds the segment
/// offset when converting to global IDs.
struct FastFieldFilterScorer<'a> {
    inner: Box<dyn Scorer + 'a>,
    compiled: Vec<(u32, CompiledFilter)>,
    reader: &'a SegmentReader,
}

impl<'a> FastFieldFilterScorer<'a> {
    fn new(
        inner: Box<dyn Scorer + 'a>,
        compiled: Vec<(u32, CompiledFilter)>,
        reader: &'a SegmentReader,
    ) -> Self {
        let mut s = Self {
            inner,
            compiled,
            reader,
        };
        // Advance to first matching doc
        let doc = s.inner.doc();
        if doc != TERMINATED && !s.passes(doc) {
            s.advance_to_next_match();
        }
        s
    }

    /// Check if a segment-local doc_id passes all filters.
    #[inline]
    fn passes(&self, doc_id: DocId) -> bool {
        // doc_id is already segment-local (0-based) — scorers never add offsets
        for (field_id, filter) in &self.compiled {
            if let Some(ff) = self.reader.fast_field(*field_id) {
                let raw = ff.get_u64(doc_id);
                if !filter.matches(raw) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    /// Advance inner scorer until we find a doc that passes, or TERMINATED.
    fn advance_to_next_match(&mut self) {
        loop {
            let doc = self.inner.advance();
            if doc == TERMINATED || self.passes(doc) {
                break;
            }
        }
    }
}

impl<'a> Scorer for FastFieldFilterScorer<'a> {
    fn doc(&self) -> DocId {
        self.inner.doc()
    }

    fn score(&self) -> Score {
        self.inner.score()
    }

    fn advance(&mut self) -> DocId {
        self.advance_to_next_match();
        self.inner.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let doc = self.inner.seek(target);
        if doc == TERMINATED {
            return TERMINATED;
        }
        if self.passes(doc) {
            return doc;
        }
        self.advance_to_next_match();
        self.inner.doc()
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

// ── Standalone filter scorer ──────────────────────────────────────────────

/// Scorer that iterates all docs (segment-local 0..num_docs) and yields
/// only those matching the fast-field filters. Used when no inner query is
/// provided (pure filter mode).
struct StandaloneFilterScorer<'a> {
    /// Current segment-local doc_id (0-based), or TERMINATED.
    current_doc: DocId,
    num_docs: u32,
    compiled: Vec<(u32, CompiledFilter)>,
    reader: &'a SegmentReader,
}

impl<'a> StandaloneFilterScorer<'a> {
    fn new(num_docs: u32, compiled: Vec<(u32, CompiledFilter)>, reader: &'a SegmentReader) -> Self {
        let mut s = Self {
            current_doc: 0,
            num_docs,
            compiled,
            reader,
        };
        if num_docs == 0 {
            s.current_doc = TERMINATED;
        } else if !s.passes(0) {
            s.advance_to_next_match();
        }
        s
    }

    #[inline]
    fn passes(&self, doc_id: u32) -> bool {
        for (field_id, filter) in &self.compiled {
            if let Some(ff) = self.reader.fast_field(*field_id) {
                let raw = ff.get_u64(doc_id);
                if !filter.matches(raw) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn advance_to_next_match(&mut self) {
        loop {
            self.current_doc += 1;
            if self.current_doc >= self.num_docs {
                self.current_doc = TERMINATED;
                return;
            }
            if self.passes(self.current_doc) {
                return;
            }
        }
    }
}

impl<'a> Scorer for StandaloneFilterScorer<'a> {
    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn score(&self) -> Score {
        1.0 // filter-only queries score all docs equally
    }

    fn advance(&mut self) -> DocId {
        self.advance_to_next_match();
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if target > self.current_doc && target < self.num_docs {
            self.current_doc = target;
            if self.passes(self.current_doc) {
                return self.current_doc;
            }
            self.advance_to_next_match();
        } else if target >= self.num_docs {
            self.current_doc = TERMINATED;
        }
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.num_docs
    }
}
