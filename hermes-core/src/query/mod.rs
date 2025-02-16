//! Query types and search execution

/// Maximum number of query tokens (terms / dimensions) for text and sparse queries.
/// Queries exceeding this limit are trimmed to the top-weighted terms.
pub const MAX_QUERY_TOKENS: usize = 16;

mod bm25;
mod boolean;
mod boost;
mod collector;
pub mod docset;
mod global_stats;
mod phrase;
mod planner;
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
pub use global_stats::*;
pub use phrase::*;
pub use range::*;
pub use reranker::*;
pub use scoring::*;
pub use term::*;
pub use traits::*;
pub use vector::*;
