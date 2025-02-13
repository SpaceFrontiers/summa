//! Range query for fast-field numeric filtering.
//!
//! `RangeQuery` produces a `RangeScorer` that scans a fast-field column and
//! yields documents whose value falls within the specified bounds. Score is
//! always 1.0 — this is a pure filter query.
//!
//! Supports u64, i64, and f64 fields. For i64/f64, bounds are encoded to the
//! same sortable-u64 representation used by fast fields so that a single raw
//! u64 comparison covers all types.
//!
//! When placed in a `BooleanQuery` MUST clause, the `BooleanScorer`'s
//! seek-based intersection makes this efficient even on large segments.

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::structures::fast_field::{FAST_FIELD_MISSING, f64_to_sortable_u64, zigzag_encode};
use crate::{DocId, Score};

use super::docset::DocSet;
use super::traits::{CountFuture, Query, Scorer, ScorerFuture};

// ── Typed range bounds ───────────────────────────────────────────────────

/// Inclusive range bounds in the user's type domain.
#[derive(Debug, Clone)]
pub enum RangeBound {
    /// u64 range — stored raw
    U64 { min: Option<u64>, max: Option<u64> },
    /// i64 range — will be zigzag-encoded for comparison
    I64 { min: Option<i64>, max: Option<i64> },
    /// f64 range — will be sortable-encoded for comparison
    F64 { min: Option<f64>, max: Option<f64> },
}

impl RangeBound {
    /// Compile to raw u64 inclusive bounds suitable for direct fast-field comparison.
    ///
    /// Returns `(low, high)` where both are inclusive. Missing bounds become
    /// 0 / u64::MAX-1 (reserving u64::MAX for FAST_FIELD_MISSING sentinel).
    fn compile(&self) -> (u64, u64) {
        match self {
            RangeBound::U64 { min, max } => {
                let lo = min.unwrap_or(0);
                let hi = max.unwrap_or(u64::MAX - 1);
                (lo, hi)
            }
            RangeBound::I64 { min, max } => {
                // zigzag encoding preserves magnitude, not order.
                // For correct range comparison on i64, we use sortable encoding
                // (same as f64 but cast through bits). However, fast fields store
                // i64 as zigzag. So we must decode per-doc and compare in i64 domain.
                // We store the raw i64 bounds and handle comparison in the scorer.
                //
                // Sentinel: use a special marker to tell the scorer to use i64 path.
                // We'll handle this in the scorer directly.
                let lo = min.map(zigzag_encode).unwrap_or(0);
                let hi = max.map(zigzag_encode).unwrap_or(u64::MAX - 1);
                (lo, hi)
            }
            RangeBound::F64 { min, max } => {
                let lo = min.map(f64_to_sortable_u64).unwrap_or(0);
                let hi = max.map(f64_to_sortable_u64).unwrap_or(u64::MAX - 1);
                (lo, hi)
            }
        }
    }

    /// Whether this bound requires per-doc i64 decoding (zigzag doesn't preserve order).
    fn is_i64(&self) -> bool {
        matches!(self, RangeBound::I64 { .. })
    }

    /// Get the raw i64 bounds for the i64 path.
    fn i64_bounds(&self) -> (i64, i64) {
        match self {
            RangeBound::I64 { min, max } => (min.unwrap_or(i64::MIN), max.unwrap_or(i64::MAX)),
            _ => (i64::MIN, i64::MAX),
        }
    }
}

// ── RangeQuery ───────────────────────────────────────────────────────────

/// Fast-field range query.
///
/// Scans all documents in a segment and yields those whose fast-field value
/// falls within `[min, max]` (inclusive). Score is always 1.0.
#[derive(Debug, Clone)]
pub struct RangeQuery {
    pub field: Field,
    pub bound: RangeBound,
}

impl std::fmt::Display for RangeQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.bound {
            RangeBound::U64 { min, max } => write!(
                f,
                "Range({}:[{} TO {}])",
                self.field.0,
                min.map_or("*".to_string(), |v| v.to_string()),
                max.map_or("*".to_string(), |v| v.to_string()),
            ),
            RangeBound::I64 { min, max } => write!(
                f,
                "Range({}:[{} TO {}])",
                self.field.0,
                min.map_or("*".to_string(), |v| v.to_string()),
                max.map_or("*".to_string(), |v| v.to_string()),
            ),
            RangeBound::F64 { min, max } => write!(
                f,
                "Range({}:[{} TO {}])",
                self.field.0,
                min.map_or("*".to_string(), |v| v.to_string()),
                max.map_or("*".to_string(), |v| v.to_string()),
            ),
        }
    }
}

impl RangeQuery {
    pub fn new(field: Field, bound: RangeBound) -> Self {
        Self { field, bound }
    }

    /// Convenience: u64 range
    pub fn u64(field: Field, min: Option<u64>, max: Option<u64>) -> Self {
        Self::new(field, RangeBound::U64 { min, max })
    }

    /// Convenience: i64 range
    pub fn i64(field: Field, min: Option<i64>, max: Option<i64>) -> Self {
        Self::new(field, RangeBound::I64 { min, max })
    }

    /// Convenience: f64 range
    pub fn f64(field: Field, min: Option<f64>, max: Option<f64>) -> Self {
        Self::new(field, RangeBound::F64 { min, max })
    }
}

impl Query for RangeQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let bound = self.bound.clone();
        Box::pin(async move {
            match RangeScorer::new(reader, field, &bound) {
                Ok(scorer) => Ok(Box::new(scorer) as Box<dyn Scorer>),
                Err(_) => Ok(Box::new(EmptyRangeScorer) as Box<dyn Scorer>),
            }
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        _limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        match RangeScorer::new(reader, self.field, &self.bound) {
            Ok(scorer) => Ok(Box::new(scorer) as Box<dyn Scorer + 'a>),
            Err(_) => Ok(Box::new(EmptyRangeScorer) as Box<dyn Scorer + 'a>),
        }
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let num_docs = reader.num_docs();
        // Rough estimate: half the segment (we don't know selectivity)
        Box::pin(async move { Ok(num_docs / 2) })
    }

    fn is_filter(&self) -> bool {
        true
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        let fast_field = reader.fast_field(self.field.0)?;
        let (raw_lo, raw_hi) = self.bound.compile();
        let use_i64 = self.bound.is_i64();
        let (i64_lo, i64_hi) = self.bound.i64_bounds();

        Some(Box::new(move |doc_id: DocId| -> bool {
            let raw = fast_field.get_u64(doc_id);
            if raw == FAST_FIELD_MISSING {
                return false;
            }
            if use_i64 {
                let val = crate::structures::fast_field::zigzag_decode(raw);
                val >= i64_lo && val <= i64_hi
            } else {
                raw >= raw_lo && raw <= raw_hi
            }
        }))
    }
}

// ── RangeScorer ──────────────────────────────────────────────────────────

/// Scorer that scans a fast-field column and yields matching docs.
///
/// For u64 and f64 fields, comparison is done in the raw u64 domain (both
/// use order-preserving encodings). For i64 fields, zigzag encoding does NOT
/// preserve order, so we decode each value and compare in i64 domain.
struct RangeScorer<'a> {
    /// Cached fast-field reader — avoids HashMap lookup per doc in matches()
    fast_field: &'a crate::structures::fast_field::FastFieldReader,
    /// For u64/f64: compiled raw bounds. For i64: unused.
    raw_lo: u64,
    raw_hi: u64,
    /// For i64 only: decoded bounds.
    i64_lo: i64,
    i64_hi: i64,
    /// Whether to use i64 comparison path.
    use_i64: bool,
    /// Current document position.
    current: u32,
    num_docs: u32,
}

/// Empty scorer returned when the field has no fast-field data.
struct EmptyRangeScorer;

impl<'a> RangeScorer<'a> {
    fn new(
        reader: &'a SegmentReader,
        field: Field,
        bound: &RangeBound,
    ) -> Result<Self, EmptyRangeScorer> {
        let fast_field = reader.fast_field(field.0).ok_or(EmptyRangeScorer)?;
        let num_docs = reader.num_docs();
        let (raw_lo, raw_hi) = bound.compile();
        let use_i64 = bound.is_i64();
        let (i64_lo, i64_hi) = bound.i64_bounds();

        let mut scorer = Self {
            fast_field,
            raw_lo,
            raw_hi,
            i64_lo,
            i64_hi,
            use_i64,
            current: 0,
            num_docs,
        };

        // Position on first matching doc
        if num_docs > 0 && !scorer.matches(0) {
            scorer.scan_forward();
        }
        Ok(scorer)
    }

    #[inline]
    fn matches(&self, doc_id: DocId) -> bool {
        let raw = self.fast_field.get_u64(doc_id);
        if raw == FAST_FIELD_MISSING {
            return false;
        }

        if self.use_i64 {
            let val = crate::structures::fast_field::zigzag_decode(raw);
            val >= self.i64_lo && val <= self.i64_hi
        } else {
            raw >= self.raw_lo && raw <= self.raw_hi
        }
    }

    /// Advance current past non-matching docs.
    fn scan_forward(&mut self) {
        loop {
            self.current += 1;
            if self.current >= self.num_docs {
                self.current = self.num_docs;
                return;
            }
            if self.matches(self.current) {
                return;
            }
        }
    }
}

impl DocSet for RangeScorer<'_> {
    fn doc(&self) -> DocId {
        if self.current >= self.num_docs {
            TERMINATED
        } else {
            self.current
        }
    }

    fn advance(&mut self) -> DocId {
        self.scan_forward();
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current >= self.num_docs {
            return TERMINATED;
        }
        if target <= self.current {
            return self.current;
        }
        // Position just before target so scan_forward starts at target
        self.current = target - 1;
        self.scan_forward();
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        // Upper bound: remaining docs
        self.num_docs.saturating_sub(self.current)
    }
}

impl Scorer for RangeScorer<'_> {
    fn score(&self) -> Score {
        1.0
    }
}

impl DocSet for EmptyRangeScorer {
    fn doc(&self) -> DocId {
        TERMINATED
    }
    fn advance(&mut self) -> DocId {
        TERMINATED
    }
    fn seek(&mut self, _target: DocId) -> DocId {
        TERMINATED
    }
    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for EmptyRangeScorer {
    fn score(&self) -> Score {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_bound_u64_compile() {
        let b = RangeBound::U64 {
            min: Some(10),
            max: Some(100),
        };
        let (lo, hi) = b.compile();
        assert_eq!(lo, 10);
        assert_eq!(hi, 100);
    }

    #[test]
    fn test_range_bound_f64_compile_preserves_order() {
        let b1 = RangeBound::F64 {
            min: Some(-1.0),
            max: Some(1.0),
        };
        let (lo1, hi1) = b1.compile();
        assert!(lo1 < hi1);

        let b2 = RangeBound::F64 {
            min: Some(0.0),
            max: Some(100.0),
        };
        let (lo2, hi2) = b2.compile();
        assert!(lo2 < hi2);
    }

    #[test]
    fn test_range_bound_open_bounds() {
        let b = RangeBound::U64 {
            min: None,
            max: None,
        };
        let (lo, hi) = b.compile();
        assert_eq!(lo, 0);
        assert_eq!(hi, u64::MAX - 1);
    }

    #[test]
    fn test_range_query_constructors() {
        let q = RangeQuery::u64(Field(0), Some(10), Some(100));
        assert_eq!(q.field, Field(0));
        assert!(matches!(
            q.bound,
            RangeBound::U64 {
                min: Some(10),
                max: Some(100)
            }
        ));

        let q = RangeQuery::i64(Field(1), Some(-50), Some(50));
        assert!(matches!(
            q.bound,
            RangeBound::I64 {
                min: Some(-50),
                max: Some(50)
            }
        ));

        let q = RangeQuery::f64(Field(2), Some(0.5), Some(9.5));
        assert!(matches!(q.bound, RangeBound::F64 { .. }));
    }
}
