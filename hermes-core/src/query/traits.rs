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

/// A search query (async)
#[cfg(not(target_arch = "wasm32"))]
pub trait Query: Send + Sync {
    /// Create a scorer for this query against a single segment (async)
    ///
    /// The `limit` parameter specifies the maximum number of results to return.
    /// This is passed from the top-level search limit.
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a>;

    /// Estimated number of matching documents in a segment (async)
    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a>;

    /// Return term info if this is a simple term query eligible for WAND optimization
    ///
    /// Returns None for complex queries (boolean, phrase, etc.)
    fn as_term_query_info(&self) -> Option<TermQueryInfo> {
        None
    }
}

/// A search query (async) - WASM version without Send bounds
#[cfg(target_arch = "wasm32")]
pub trait Query {
    /// Create a scorer for this query against a single segment (async)
    ///
    /// The `limit` parameter specifies the maximum number of results to return.
    /// This is passed from the top-level search limit.
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a>;

    /// Estimated number of matching documents in a segment (async)
    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a>;

    /// Return term info if this is a simple term query eligible for WAND optimization
    fn as_term_query_info(&self) -> Option<TermQueryInfo> {
        None
    }
}

impl Query for Box<dyn Query> {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        (**self).scorer(reader, limit)
    }

    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a> {
        (**self).count_estimate(reader)
    }

    fn as_term_query_info(&self) -> Option<TermQueryInfo> {
        (**self).as_term_query_info()
    }
}

/// Scorer that iterates over matching documents and computes scores
#[cfg(not(target_arch = "wasm32"))]
pub trait Scorer: Send {
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
}

/// Scorer that iterates over matching documents and computes scores (WASM version)
#[cfg(target_arch = "wasm32")]
pub trait Scorer {
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
