//! Fast-field filter queries for efficient document filtering.
//!
//! `FastFieldFilterQuery` wraps an inner query and applies O(1) per-doc
//! fast-field predicate checks, skipping documents that don't match.

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::structures::fast_field::{
    FAST_FIELD_MISSING, FastFieldReader, f64_to_sortable_u64, zigzag_decode, zigzag_encode,
};
use crate::{DocId, Score};

use super::{CountFuture, DocPredicate, Query, Scorer, ScorerFuture};

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
    /// Range on i64 values — checked on decoded (zigzag) values since zigzag
    /// encoding does NOT preserve order.
    RangeI64Decoded {
        min: i64,
        max: i64,
    },
    In(Vec<u64>),
    Exists,
    /// Always fails (e.g. text value not in dictionary)
    Never,
}

impl CompiledFilter {
    /// Check if a raw column value passes this filter.
    ///
    /// Absent values ([`FAST_FIELD_MISSING`]) never match any value filter
    /// (Eq, Range, In). Only [`Exists`](CompiledFilter::Exists) explicitly
    /// tests for presence.
    #[inline]
    fn matches(&self, raw: u64) -> bool {
        match self {
            CompiledFilter::Exists => raw != FAST_FIELD_MISSING,
            CompiledFilter::Never => false,
            // All value-matching filters reject absent docs
            _ if raw == FAST_FIELD_MISSING => false,
            CompiledFilter::Eq(v) => raw == *v,
            CompiledFilter::Range { min, max } => raw >= *min && raw <= *max,
            CompiledFilter::RangeI64Decoded { min, max } => {
                let decoded = zigzag_decode(raw);
                decoded >= *min && decoded <= *max
            }
            CompiledFilter::In(vals) => vals.contains(&raw),
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
            // Zigzag encoding does NOT preserve order, so we check decoded values at runtime.
            CompiledFilter::RangeI64Decoded {
                min: min.unwrap_or(i64::MIN),
                max: max.unwrap_or(i64::MAX),
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
        FastFieldCondition::Exists => CompiledFilter::Exists,
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

/// Check a doc against compiled filters using the segment reader.
#[inline]
fn check_compiled_filters(
    compiled: &[(u32, CompiledFilter)],
    reader: &SegmentReader,
    doc_id: DocId,
) -> bool {
    for (field_id, filter) in compiled {
        if let Some(ff) = reader.fast_field(*field_id) {
            if ff.multi {
                let any_match = ff.for_each_multi_value(doc_id, |val| filter.matches(val));
                if !any_match {
                    return false;
                }
            } else {
                let raw = ff.get_u64(doc_id);
                if !filter.matches(raw) {
                    return false;
                }
            }
        } else {
            return false;
        }
    }
    true
}

/// Build a [`DocPredicate`] from Arc-shared compiled filters.
fn make_filter_predicate<'a>(
    compiled: Arc<Vec<(u32, CompiledFilter)>>,
    reader: &'a SegmentReader,
) -> DocPredicate<'a> {
    Box::new(move |doc_id: DocId| check_compiled_filters(&compiled, reader, doc_id))
}

/// Chain an optional outer predicate with our filter predicate.
fn chain_predicates<'a>(
    outer: Option<DocPredicate<'a>>,
    inner: DocPredicate<'a>,
) -> DocPredicate<'a> {
    match outer {
        Some(outer) => Box::new(move |doc_id| outer(doc_id) && inner(doc_id)),
        None => inner,
    }
}

impl Query for FastFieldFilterQuery {
    fn scorer<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        predicate: Option<DocPredicate<'a>>,
    ) -> ScorerFuture<'a> {
        let inner = self.inner.clone();
        let filters = self.filters.clone();

        Box::pin(async move {
            // Compile filters against segment's fast-field readers
            let mut compiled_vec: Vec<(u32, CompiledFilter)> = Vec::with_capacity(filters.len());
            for filter in &filters {
                let field_id = filter.field.0;
                if let Some(ff_reader) = reader.fast_field(field_id) {
                    let cf = compile_condition(&filter.condition, ff_reader);
                    compiled_vec.push((field_id, cf));
                } else {
                    log::warn!(
                        "[fast-filter] field {} has no fast-field data in segment, filter will reject all docs",
                        field_id
                    );
                    compiled_vec.push((field_id, CompiledFilter::Never));
                }
            }

            // Arc-share compiled filters so we can build two predicates
            let compiled = Arc::new(compiled_vec);

            // Convert incoming predicate to Arc so it can be shared too
            #[cfg(not(target_arch = "wasm32"))]
            type SharedPred<'b> = Arc<dyn Fn(DocId) -> bool + Send + Sync + 'b>;
            #[cfg(target_arch = "wasm32")]
            type SharedPred<'b> = Arc<dyn Fn(DocId) -> bool + 'b>;

            let outer_arc: Option<SharedPred<'a>> = predicate.map(Arc::from);

            if let Some(ref inner_query) = inner {
                // Build two predicates: one for inner query (executor optimization),
                // one for wrapper (safety net for non-executor scorer paths).
                let pred_for_inner = chain_predicates(
                    outer_arc.as_ref().map(|a| {
                        let a = a.clone();
                        Box::new(move |doc_id| a(doc_id)) as DocPredicate<'a>
                    }),
                    make_filter_predicate(compiled.clone(), reader),
                );
                let pred_for_wrapper = chain_predicates(
                    outer_arc.map(|a| Box::new(move |doc_id| a(doc_id)) as DocPredicate<'a>),
                    make_filter_predicate(compiled, reader),
                );

                // Pass predicate to inner query — executors (MaxScore, BMP) check it
                // during scoring for early rejection.
                let inner_scorer = inner_query
                    .scorer(reader, limit, Some(pred_for_inner))
                    .await?;

                // Wrap with PredicateFilterScorer — safety net for paths where the
                // predicate isn't consumed (TermQuery, BooleanScorer MUST, PhraseQuery,
                // vector queries). For executor paths, docs already passed → cheap no-op.
                Ok(
                    Box::new(PredicateFilterScorer::new(inner_scorer, pred_for_wrapper))
                        as Box<dyn Scorer + 'a>,
                )
            } else {
                // Standalone: iterate all docs, filter with predicate
                let pred = chain_predicates(
                    outer_arc.map(|a| Box::new(move |doc_id| a(doc_id)) as DocPredicate<'a>),
                    make_filter_predicate(compiled, reader),
                );
                let num_docs = reader.num_docs();
                let scorer: Box<dyn Scorer + 'a> =
                    Box::new(StandaloneFilterScorer::new(num_docs, pred));
                Ok(scorer)
            }
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

// ── Predicate filter scorer ───────────────────────────────────────────────

/// Lightweight wrapper that checks a [`DocPredicate`] per-doc.
///
/// Safety net for non-executor scorer paths (TermQuery, BooleanScorer with MUST,
/// PhraseQuery, vector queries) where the predicate isn't consumed by an executor.
/// For executor paths (MaxScore, BMP), docs already passed the predicate inside
/// the executor, so the wrapper check is a cheap no-op.
struct PredicateFilterScorer<'a> {
    inner: Box<dyn Scorer + 'a>,
    predicate: DocPredicate<'a>,
}

impl<'a> PredicateFilterScorer<'a> {
    fn new(inner: Box<dyn Scorer + 'a>, predicate: DocPredicate<'a>) -> Self {
        let mut s = Self { inner, predicate };
        let doc = s.inner.doc();
        if doc != TERMINATED && !(s.predicate)(doc) {
            s.advance_to_next_match();
        }
        s
    }

    fn advance_to_next_match(&mut self) {
        loop {
            let doc = self.inner.advance();
            if doc == TERMINATED || (self.predicate)(doc) {
                break;
            }
        }
    }
}

impl<'a> Scorer for PredicateFilterScorer<'a> {
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
        if (self.predicate)(doc) {
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
/// only those matching the predicate. Used when no inner query is
/// provided (pure filter mode).
struct StandaloneFilterScorer<'a> {
    /// Current segment-local doc_id (0-based), or TERMINATED.
    current_doc: DocId,
    num_docs: u32,
    predicate: super::DocPredicate<'a>,
}

impl<'a> StandaloneFilterScorer<'a> {
    fn new(num_docs: u32, predicate: super::DocPredicate<'a>) -> Self {
        let mut s = Self {
            current_doc: 0,
            num_docs,
            predicate,
        };
        if num_docs == 0 {
            s.current_doc = TERMINATED;
        } else if !(s.predicate)(0) {
            s.advance_to_next_match();
        }
        s
    }

    fn advance_to_next_match(&mut self) {
        loop {
            self.current_doc += 1;
            if self.current_doc >= self.num_docs {
                self.current_doc = TERMINATED;
                return;
            }
            if (self.predicate)(self.current_doc) {
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
            if (self.predicate)(self.current_doc) {
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
