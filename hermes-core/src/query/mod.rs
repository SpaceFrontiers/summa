//! Query types and search execution

/// Maximum number of query tokens (terms / dimensions) for text and sparse queries.
/// Queries exceeding this limit are trimmed to the top-weighted terms.
pub const MAX_QUERY_TERMS: usize = 64;

/// Maximum candidate depth relative to the result window.
///
/// This is the single query-level oversubscription policy used by fusion,
/// vector reranking, and request-facing adapters. A caller that already asks
/// for an expanded result window may explicitly use that window as its
/// candidate depth; no layer multiplies an explicit depth again.
pub const MAX_CANDIDATE_OVERSUBSCRIPTION: usize = 2;

/// Largest default candidate pool for a requested result window.
pub const fn max_candidate_limit(result_window: usize) -> usize {
    result_window.saturating_mul(MAX_CANDIDATE_OVERSUBSCRIPTION)
}

mod bm25;
pub(crate) mod bmp;
mod boolean;
mod boost;
mod collector;
pub mod docset;
mod fusion;
mod global_stats;
mod phrase;
mod planner;
mod prefix;
mod range;
mod reranker;
mod scoring;
#[cfg(test)]
mod scoring_tests;
mod term;
mod traits;
mod vector;

pub use bm25::*;
pub use boolean::*;
pub use boost::*;
pub use collector::*;
pub use docset::*;
pub use fusion::*;
pub use global_stats::*;
pub use phrase::*;
pub use prefix::*;
pub use range::*;
pub use reranker::*;
pub use scoring::*;
pub use term::*;
pub use traits::*;
pub use vector::*;
