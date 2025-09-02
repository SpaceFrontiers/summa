//! Range query for fast-field numeric filtering.
//!
//! `RangeQuery` produces a `RangeScorer` that scans a fast-field column and
//! yields documents whose value falls within the specified bounds. Score is
//! always 1.0 — this is a pure filter query.
//!
//! Supports u64, i64, and f64 fields. Unsigned and sortable-encoded f64 values
//! compare in their stored domain; zigzag-encoded i64 values must be decoded
//! before signed comparison.
//!
//! When placed in a `BooleanQuery` MUST clause, the `BooleanScorer`'s
//! seek-based intersection makes this efficient even on large segments.

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::TERMINATED;
use crate::structures::fast_field::{FAST_FIELD_MISSING, f64_to_sortable_u64, zigzag_decode};
use crate::{DocId, Score};

use super::docset::DocSet;
use super::traits::{CountFuture, Query, Scorer, ScorerFuture};

// ── Typed range bounds ───────────────────────────────────────────────────

/// Inclusive range bounds in the user's type domain.
#[derive(Debug, Clone)]
pub enum RangeBound {
    /// u64 range — stored raw
    U64 { min: Option<u64>, max: Option<u64> },
    /// i64 range — stored values are zigzag-decoded before comparison
    I64 { min: Option<i64>, max: Option<i64> },
    /// f64 range — will be sortable-encoded for comparison
    F64 { min: Option<f64>, max: Option<f64> },
}

/// One compiled comparison shared by scorer, random probes, and batch scans.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CompiledRange {
    Raw { lo: u64, hi: u64 },
    Signed { lo: i64, hi: i64 },
}

impl CompiledRange {
    #[inline]
    fn contains(self, raw: u64) -> bool {
        if raw == FAST_FIELD_MISSING {
            return false;
        }
        match self {
            Self::Raw { lo, hi } => raw >= lo && raw <= hi,
            Self::Signed { lo, hi } => {
                let value = zigzag_decode(raw);
                value >= lo && value <= hi
            }
        }
    }
}

impl RangeBound {
    fn compile(&self) -> CompiledRange {
        match *self {
            Self::U64 { min, max } => CompiledRange::Raw {
                lo: min.unwrap_or(0),
                hi: max.unwrap_or(u64::MAX - 1),
            },
            Self::I64 { min, max } => CompiledRange::Signed {
                lo: min.unwrap_or(i64::MIN),
                hi: max.unwrap_or(i64::MAX),
            },
            Self::F64 { min, max } => CompiledRange::Raw {
                lo: min.map(f64_to_sortable_u64).unwrap_or(0),
                hi: max.map(f64_to_sortable_u64).unwrap_or(u64::MAX - 1),
            },
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
        let bound = self.bound.compile();
        Some(Box::new(move |doc_id| {
            bound.contains(fast_field.get_u64(doc_id))
        }))
    }

    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        let fast_field = reader.fast_field(self.field.0)?;
        if fast_field.multi {
            // Range predicates inspect the first value, not any value. Keep
            // this path until the reader owns a batch API with that contract.
            let pred = self.as_doc_predicate(reader)?;
            return Some(super::DocBitset::from_predicate(reader.num_docs(), &*pred));
        }
        let bound = self.bound.compile();
        let mut bits = super::DocBitset::new(reader.num_docs());
        // The generic callback can inline. Traverse blocks once and dispatch
        // the codec per batch instead of doing random access for every doc.
        fast_field.scan_single_values(|doc_id, raw| {
            if bound.contains(raw) {
                bits.set(doc_id);
            }
        });
        Some(bits)
    }

    fn bitset_cardinality_estimate(&self, reader: &SegmentReader) -> Option<u64> {
        // Sampled: probe ~1k evenly spaced docs with the fast-field predicate.
        // Works for every encoding (incl. i64 zigzag, where min/max
        // interpolation would mis-order). Rounded up so a rare-but-present
        // range never estimates to zero.
        let pred = self.as_doc_predicate(reader)?;
        let n = reader.num_docs();
        if n == 0 {
            return Some(0);
        }
        const SAMPLES: u32 = 1024;
        if n <= SAMPLES {
            return Some((0..n).filter(|&d| pred(d)).count() as u64);
        }
        let step = n / SAMPLES;
        let hits = (0..SAMPLES).filter(|&i| pred(i * step)).count() as u64;
        Some(((hits * n as u64) / SAMPLES as u64).max(1))
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
    bound: CompiledRange,
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
        let mut scorer = Self {
            fast_field,
            bound: bound.compile(),
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
        self.bound.contains(self.fast_field.get_u64(doc_id))
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
        assert_eq!(b.compile(), CompiledRange::Raw { lo: 10, hi: 100 });
    }

    #[test]
    fn test_range_bound_f64_compile_preserves_order() {
        let b1 = RangeBound::F64 {
            min: Some(-1.0),
            max: Some(1.0),
        };
        let CompiledRange::Raw { lo, hi } = b1.compile() else {
            panic!("expected raw bounds")
        };
        assert!(lo < hi);

        let b2 = RangeBound::F64 {
            min: Some(0.0),
            max: Some(100.0),
        };
        let CompiledRange::Raw { lo, hi } = b2.compile() else {
            panic!("expected raw bounds")
        };
        assert!(lo < hi);
    }

    #[test]
    fn test_range_bound_open_bounds() {
        let b = RangeBound::U64 {
            min: None,
            max: None,
        };
        assert_eq!(
            b.compile(),
            CompiledRange::Raw {
                lo: 0,
                hi: u64::MAX - 1
            }
        );
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
