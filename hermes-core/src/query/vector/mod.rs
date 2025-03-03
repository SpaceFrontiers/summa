//! Vector query types for dense, binary dense, and sparse vector search

mod binary_dense;
mod combiner;
mod dense;
mod sparse;

pub use binary_dense::BinaryDenseVectorQuery;
pub use combiner::MultiValueCombiner;
pub use dense::DenseVectorQuery;
pub use sparse::{SparseTermQuery, SparseVectorQuery};

use crate::segment::VectorSearchResult;
use crate::{DocId, Score, TERMINATED};

use super::ScoredPosition;
use super::traits::{MatchedPositions, Scorer};

/// Shared scorer for vector search results (used by both dense and binary dense queries).
///
/// Wraps sorted VectorSearchResult list and implements DocSet + Scorer.
pub(super) struct VectorResultScorer {
    results: Vec<VectorSearchResult>,
    position: usize,
    field_id: u32,
}

impl VectorResultScorer {
    pub fn new(mut results: Vec<VectorSearchResult>, field_id: u32) -> Self {
        results.sort_unstable_by_key(|r| r.doc_id);
        Self {
            results,
            position: 0,
            field_id,
        }
    }
}

impl super::docset::DocSet for VectorResultScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let remaining = &self.results[self.position..];
        let offset = remaining.partition_point(|r| r.doc_id < target);
        self.position += offset;
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
    }
}

impl Scorer for VectorResultScorer {
    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }

    fn matched_positions(&self) -> Option<MatchedPositions> {
        if self.position >= self.results.len() {
            return None;
        }
        let result = &self.results[self.position];
        let scored_positions: Vec<ScoredPosition> = result
            .ordinals
            .iter()
            .map(|(ordinal, score)| ScoredPosition::new(*ordinal, *score))
            .collect();
        Some(vec![(self.field_id, scored_positions)])
    }
}
