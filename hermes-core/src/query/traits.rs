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

/// A search query (async)
#[cfg(not(target_arch = "wasm32"))]
pub trait Query: Send + Sync {
    /// Create a scorer for this query against a single segment (async)
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a>;

    /// Estimated number of matching documents in a segment (async)
    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a>;
}

/// A search query (async) - WASM version without Send bounds
#[cfg(target_arch = "wasm32")]
pub trait Query {
    /// Create a scorer for this query against a single segment (async)
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a>;

    /// Estimated number of matching documents in a segment (async)
    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a>;
}

impl Query for Box<dyn Query> {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a> {
        (**self).scorer(reader)
    }

    fn count_estimate<'a>(&'a self, reader: &'a SegmentReader) -> CountFuture<'a> {
        (**self).count_estimate(reader)
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
