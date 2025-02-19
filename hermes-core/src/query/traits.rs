//! Query and Scorer traits with async support
//!
//! Provides the core abstractions for search queries and document scoring.

use std::future::Future;
use std::pin::Pin;

use crate::segment::SegmentReader;
use crate::{DocId, Result, Score};

/// BM25 parameters
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term frequency saturation parameter (typically 1.2-2.0)
    pub k1: f32,
    /// Length normalization parameter (typically 0.75)
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Future type for scorer creation
#[cfg(not(target_arch = "wasm32"))]
pub type ScorerFuture<'a> = Pin<Box<dyn Future<Output = Result<Box<dyn Scorer + 'a>>> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type ScorerFuture<'a> = Pin<Box<dyn Future<Output = Result<Box<dyn Scorer + 'a>>> + 'a>>;

/// Future type for count estimation
#[cfg(not(target_arch = "wasm32"))]
pub type CountFuture<'a> = Pin<Box<dyn Future<Output = Result<u32>> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
pub type CountFuture<'a> = Pin<Box<dyn Future<Output = Result<u32>> + 'a>>;

/// Per-document predicate closure type (platform-aware Send+Sync bounds)
#[cfg(not(target_arch = "wasm32"))]
pub type DocPredicate<'a> = Box<dyn Fn(DocId) -> bool + Send + Sync + 'a>;
#[cfg(target_arch = "wasm32")]
pub type DocPredicate<'a> = Box<dyn Fn(DocId) -> bool + 'a>;

/// Compact bitset indexed by doc_id. O(1) lookup, ~2.25 MB for 18M docs.
///
/// Built from posting lists or predicate scans. Used by BMP filtered queries
/// for fast per-slot predicate evaluation (~2ns per lookup vs ~30-40ns for
/// a fast-field closure).
pub struct DocBitset {
    pub(crate) bits: Vec<u64>,
}

impl DocBitset {
    /// Create an empty bitset for `num_docs` documents.
    pub fn new(num_docs: u32) -> Self {
        let num_words = (num_docs as usize).div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
        }
    }

    /// Set bit for `doc_id`.
    #[inline]
    pub fn set(&mut self, doc_id: u32) {
        let word = doc_id as usize / 64;
        let bit = doc_id as usize % 64;
        if word < self.bits.len() {
            self.bits[word] |= 1u64 << bit;
        }
    }

    /// Test if `doc_id` is in the bitset.
    #[inline(always)]
    pub fn contains(&self, doc_id: u32) -> bool {
        let word = doc_id as usize / 64;
        let bit = doc_id as usize % 64;
        word < self.bits.len() && self.bits[word] & (1u64 << bit) != 0
    }

    /// Number of set bits (matching docs).
    pub fn count(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Build bitset from a predicate by scanning all docs. O(N).
    pub fn from_predicate(num_docs: u32, pred: &dyn Fn(DocId) -> bool) -> Self {
        let mut bs = Self::new(num_docs);
        for doc_id in 0..num_docs {
            if pred(doc_id) {
                bs.set(doc_id);
            }
        }
        bs
    }
}

/// Info for MaxScore-optimizable term queries
#[derive(Debug, Clone)]
pub struct TermQueryInfo {
    /// Field being searched
    pub field: crate::dsl::Field,
    /// Term bytes (lowercase)
    pub term: Vec<u8>,
}

/// Info for MaxScore-optimizable sparse term queries
#[derive(Debug, Clone, Copy)]
pub struct SparseTermQueryInfo {
    /// Sparse vector field
    pub field: crate::dsl::Field,
    /// Dimension ID in the sparse vector
    pub dim_id: u32,
    /// Query weight for this dimension
    pub weight: f32,
    /// MaxScore heap factor (1.0 = exact, lower = approximate)
    pub heap_factor: f32,
    /// Multi-value combiner for ordinal deduplication
    pub combiner: super::MultiValueCombiner,
    /// Multiplier on executor limit to compensate for ordinal deduplication
    /// (1.0 = exact, 2.0 = fetch 2x then combine down)
    pub over_fetch_factor: f32,
}

/// Decomposition of a query for MaxScore optimization.
///
/// The planner inspects this to decide whether to use text MaxScore,
/// sparse MaxScore, or standard BooleanScorer execution.
#[derive(Debug, Clone)]
pub enum QueryDecomposition {
    /// Single text term — eligible for text MaxScore grouping
    TextTerm(TermQueryInfo),
    /// One or more sparse dimensions — eligible for sparse MaxScore
    SparseTerms(Vec<SparseTermQueryInfo>),
    /// Not decomposable — falls back to standard execution
    Opaque,
}

/// Matched positions for a field (field_id, list of scored positions)
/// Each position includes its individual score contribution
pub type MatchedPositions = Vec<(u32, Vec<super::ScoredPosition>)>;

macro_rules! define_query_traits {
    ($($send_bounds:tt)*) => {
        /// A search query (async)
        ///
        /// Note: `scorer` takes `&self` (not `&'a self`) so that scorers don't borrow the query.
        /// This enables query composition - queries can create sub-queries locally and get their scorers.
        /// Implementations must clone/capture any data they need during scorer creation.
        pub trait Query: std::fmt::Display + $($send_bounds)* {
            /// Create a scorer for this query against a single segment (async)
            ///
            /// The `limit` parameter specifies the maximum number of results to return.
            /// This is passed from the top-level search limit.
            ///
            /// Note: The scorer borrows only the reader, not the query. Implementations
            /// should capture any needed query data (field, terms, etc.) during creation.
            fn scorer<'a>(
                &self,
                reader: &'a SegmentReader,
                limit: usize,
            ) -> ScorerFuture<'a>;

            /// Estimated number of matching documents in a segment (async)
            fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a>;

            /// Create a scorer synchronously (mmap/RAM only).
            ///
            /// Available when the `sync` feature is enabled.
            /// Default implementation returns an error.
            #[cfg(feature = "sync")]
            fn scorer_sync<'a>(
                &self,
                reader: &'a SegmentReader,
                limit: usize,
            ) -> Result<Box<dyn Scorer + 'a>> {
                let _ = (reader, limit);
                Err(crate::error::Error::Query(
                    "sync scorer not supported for this query type".into(),
                ))
            }

            /// Decompose this query for MaxScore optimization.
            ///
            /// Returns `TextTerm` for simple term queries, `SparseTerms` for
            /// sparse vector queries (single or multi-dim), or `Opaque` if
            /// the query cannot be decomposed.
            fn decompose(&self) -> QueryDecomposition {
                QueryDecomposition::Opaque
            }

            /// True if this query is a pure filter (always scores 1.0, no positions).
            /// Used by the planner to convert non-selective MUST filters into predicates.
            fn is_filter(&self) -> bool {
                false
            }

            /// For filter queries: return a cheap per-doc predicate against a segment.
            /// The predicate does O(1) work per doc (e.g., fast-field lookup).
            fn as_doc_predicate<'a>(
                &self,
                _reader: &'a SegmentReader,
            ) -> Option<DocPredicate<'a>> {
                None
            }

            /// Build a compact bitset of matching doc_ids for this query.
            ///
            /// Preferred over `as_doc_predicate` for BMP filtered queries because
            /// bitset lookup is ~2ns vs ~30-40ns for a fast-field closure.
            /// Default returns None; TermQuery overrides this to build from its
            /// posting list in O(M) time.
            fn as_doc_bitset(
                &self,
                _reader: &SegmentReader,
            ) -> Option<DocBitset> {
                None
            }
        }

        /// Scored document stream: a DocSet that also provides scores.
        pub trait Scorer: super::docset::DocSet + $($send_bounds)* {
            /// Score for current document
            fn score(&self) -> Score;

            /// Get matched positions for the current document (if available)
            /// Returns (field_id, positions) pairs where positions are encoded as per PositionMode
            fn matched_positions(&self) -> Option<MatchedPositions> {
                None
            }
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
define_query_traits!(Send + Sync);

#[cfg(target_arch = "wasm32")]
define_query_traits!();

impl Query for Box<dyn Query> {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        (**self).scorer(reader, limit)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        (**self).count_estimate(reader)
    }

    fn decompose(&self) -> QueryDecomposition {
        (**self).decompose()
    }

    fn is_filter(&self) -> bool {
        (**self).is_filter()
    }

    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<DocPredicate<'a>> {
        (**self).as_doc_predicate(reader)
    }

    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<DocBitset> {
        (**self).as_doc_bitset(reader)
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        (**self).scorer_sync(reader, limit)
    }
}

/// Empty scorer for terms that don't exist
pub struct EmptyScorer;

impl super::docset::DocSet for EmptyScorer {
    fn doc(&self) -> DocId {
        crate::structures::TERMINATED
    }

    fn advance(&mut self) -> DocId {
        crate::structures::TERMINATED
    }

    fn seek(&mut self, _target: DocId) -> DocId {
        crate::structures::TERMINATED
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for EmptyScorer {
    fn score(&self) -> Score {
        0.0
    }
}
