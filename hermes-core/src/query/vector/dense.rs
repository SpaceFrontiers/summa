//! Dense vector query for similarity search (ANN)

use crate::dsl::Field;
use crate::segment::{SegmentReader, VectorSearchResult};
use crate::{DocId, Score, TERMINATED};

use super::combiner::MultiValueCombiner;
use crate::query::ScoredPosition;
use crate::query::traits::{CountFuture, MatchedPositions, Query, Scorer, ScorerFuture};

/// Dense vector query for similarity search
#[derive(Debug, Clone)]
pub struct DenseVectorQuery {
    /// Field containing the dense vectors
    pub field: Field,
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of clusters to probe (for IVF indexes)
    pub nprobe: usize,
    /// Re-ranking factor (multiplied by k for candidate selection, e.g. 3.0)
    pub rerank_factor: f32,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
}

impl std::fmt::Display for DenseVectorQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Dense({}, dim={}, nprobe={}, rerank={})",
            self.field.0,
            self.vector.len(),
            self.nprobe,
            self.rerank_factor
        )
    }
}

impl DenseVectorQuery {
    /// Create a new dense vector query
    pub fn new(field: Field, vector: Vec<f32>) -> Self {
        Self {
            field,
            vector,
            nprobe: 32,
            rerank_factor: 3.0,
            combiner: MultiValueCombiner::Max,
        }
    }

    /// Set the number of clusters to probe (for IVF indexes)
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the re-ranking factor (e.g. 3.0 = fetch 3x candidates for reranking)
    pub fn with_rerank_factor(mut self, factor: f32) -> Self {
        self.rerank_factor = factor;
        self
    }

    /// Set the multi-value score combiner
    pub fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.combiner = combiner;
        self
    }
}

impl Query for DenseVectorQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let vector = self.vector.clone();
        let nprobe = self.nprobe;
        let rerank_factor = self.rerank_factor;
        let combiner = self.combiner;
        Box::pin(async move {
            let results = reader
                .search_dense_vector(field, &vector, limit, nprobe, rerank_factor, combiner)
                .await?;

            Ok(Box::new(DenseVectorScorer::new(results, field.0)) as Box<dyn Scorer>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let results = reader.search_dense_vector_sync(
            self.field,
            &self.vector,
            limit,
            self.nprobe,
            self.rerank_factor,
            self.combiner,
        )?;
        Ok(Box::new(DenseVectorScorer::new(results, self.field.0)) as Box<dyn Scorer>)
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}

/// Scorer for dense vector search results with ordinal tracking
struct DenseVectorScorer {
    results: Vec<VectorSearchResult>,
    position: usize,
    field_id: u32,
}

impl DenseVectorScorer {
    fn new(mut results: Vec<VectorSearchResult>, field_id: u32) -> Self {
        // Sort by doc_id ascending — DocSet contract requires monotonic doc IDs
        results.sort_unstable_by_key(|r| r.doc_id);
        Self {
            results,
            position: 0,
            field_id,
        }
    }
}

impl crate::query::docset::DocSet for DenseVectorScorer {
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
        // Binary search within remaining results for O(log k) seek
        let remaining = &self.results[self.position..];
        let offset = remaining.partition_point(|r| r.doc_id < target);
        self.position += offset;
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
    }
}

impl Scorer for DenseVectorScorer {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_vector_query_builder() {
        let query = DenseVectorQuery::new(Field(0), vec![1.0, 2.0, 3.0])
            .with_nprobe(64)
            .with_rerank_factor(5.0);

        assert_eq!(query.field, Field(0));
        assert_eq!(query.vector.len(), 3);
        assert_eq!(query.nprobe, 64);
        assert_eq!(query.rerank_factor, 5.0);
    }
}
