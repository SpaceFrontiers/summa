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
        pub trait Query: $($send_bounds)* {
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

            /// Return term info if this is a simple term query eligible for MaxScore optimization
            ///
            /// Returns None for complex queries (boolean, phrase, etc.)
            fn as_term_query_info(&self) -> Option<TermQueryInfo> {
                None
            }

            /// Return sparse term info if this is a single-dimension sparse query
            /// eligible for MaxScore optimization
            fn as_sparse_term_query_info(&self) -> Option<SparseTermQueryInfo> {
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

    fn as_term_query_info(&self) -> Option<TermQueryInfo> {
        (**self).as_term_query_info()
    }

    fn as_sparse_term_query_info(&self) -> Option<SparseTermQueryInfo> {
        (**self).as_sparse_term_query_info()
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
