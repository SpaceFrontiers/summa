//! Query types and search execution

/// Maximum number of query tokens (terms / dimensions) for text and sparse queries.
/// Queries exceeding this limit are trimmed to the top-weighted terms.
pub const MAX_QUERY_TERMS: usize = 64;

/// Mark the `cap` rarest terms of a long query from their document
/// frequencies; `None` entries are not ranked and stay unmarked. Absent
/// terms (frequency 0) match nothing, so they rank last; ties keep the
/// earlier term.
pub fn rarest_terms(doc_freqs: &[Option<u64>], cap: usize) -> Vec<bool> {
    let mut ranked: Vec<(bool, u64, usize)> = doc_freqs
        .iter()
        .enumerate()
        .filter_map(|(position, df)| df.map(|df| (df == 0, df, position)))
        .collect();
    ranked.sort_unstable();
    let mut keep = vec![false; doc_freqs.len()];
    for &(_, _, position) in ranked.iter().take(cap) {
        keep[position] = true;
    }
    keep
}

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

mod all;
mod bm25;
pub(crate) mod bmp;
pub(crate) use planner::bmp_executor_limit;
mod boolean;
mod boost;
pub mod candidate_scoring;
mod collector;
pub mod docset;
mod filtered;
mod fusion;
pub use filtered::FilteredQuery;
mod global_stats;
mod phrase;
mod planner;
mod prefix;
mod proximity;
mod range;
mod regex;
mod required_text;
mod reranker;
mod scoring;
#[cfg(test)]
mod scoring_tests;
pub(crate) mod seismic;
mod term;
mod term_pattern;
mod term_union;
mod text_mapping;
mod traits;
mod vector;
mod wildcard;

pub use all::AllQuery;
pub use bm25::*;
pub use boolean::*;
pub use boost::*;
pub use collector::*;
pub use docset::*;
pub use fusion::*;
pub use global_stats::*;
pub use phrase::*;
pub use prefix::*;
pub use proximity::ProximityConfig;
pub use range::*;
pub use regex::RegexQuery;
pub use reranker::*;
pub use scoring::*;
pub use term::*;
pub use traits::*;
pub use vector::*;
pub use wildcard::WildcardQuery;

pub use candidate_scoring::{
    CandidateFeature, CandidateQuery, CandidateScores, CandidateScoringPlan, PassageFeatures,
    RankingModel, ScoreScope, ScoredCandidate,
};
