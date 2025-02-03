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

/// Info for WAND-optimizable term queries
#[derive(Debug, Clone)]
pub struct TermQueryInfo {
    /// Field being searched
    pub field: crate::dsl::Field,
    /// Term bytes (lowercase)
    pub term: Vec<u8>,
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
            fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a>;

            /// Estimated number of matching documents in a segment (async)
            fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a>;

            /// Return term info if this is a simple term query eligible for WAND optimization
            ///
            /// Returns None for complex queries (boolean, phrase, etc.)
            fn as_term_query_info(&self) -> Option<TermQueryInfo> {
                None
            }
        }

        /// Scorer that iterates over matching documents and computes scores
        pub trait Scorer: $($send_bounds)* {
            /// Current document ID, or TERMINATED if exhausted
            fn doc(&self) -> DocId;

            /// Score for current document
            fn score(&self) -> Score;

            /// Advance to next document
            fn advance(&mut self) -> DocId;

            /// Seek to first doc >= target
            fn seek(&mut self, target: DocId) -> DocId;

            /// Size hint for remaining documents
            fn size_hint(&self) -> u32;

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
}

/// Empty scorer for terms that don't exist
pub struct EmptyScorer;

impl Scorer for EmptyScorer {
    fn doc(&self) -> DocId {
        crate::structures::TERMINATED
    }

    fn score(&self) -> Score {
        0.0
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
