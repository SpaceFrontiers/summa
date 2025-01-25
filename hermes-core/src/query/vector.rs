//! Vector query types for dense and sparse vector search

use crate::dsl::Field;
use crate::segment::{SegmentReader, VectorSearchResult};
use crate::{DocId, Score, TERMINATED};

use super::ScoredPosition;
use super::traits::{CountFuture, MatchedPositions, Query, Scorer, ScorerFuture};

/// Strategy for combining scores when a document has multiple values for the same field
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MultiValueCombiner {
    /// Sum all scores (default for sparse vectors - accumulates dot product contributions)
    #[default]
    Sum,
    /// Take the maximum score
    Max,
    /// Take the average score
    Avg,
}

/// Dense vector query for similarity search
#[derive(Debug, Clone)]
pub struct DenseVectorQuery {
    /// Field containing the dense vectors
    pub field: Field,
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of clusters to probe (for IVF indexes)
    pub nprobe: usize,
    /// Re-ranking factor (multiplied by k for candidate selection)
    pub rerank_factor: usize,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
}

impl DenseVectorQuery {
    /// Create a new dense vector query
    pub fn new(field: Field, vector: Vec<f32>) -> Self {
        Self {
            field,
            vector,
            nprobe: 32,
            rerank_factor: 3,
            combiner: MultiValueCombiner::Max,
        }
    }

    /// Set the number of clusters to probe (for IVF indexes)
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the re-ranking factor
    pub fn with_rerank_factor(mut self, factor: usize) -> Self {
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
        let rerank_factor = self.rerank_factor;
        let combiner = self.combiner;
        Box::pin(async move {
            let results =
                reader.search_dense_vector(field, &vector, limit, rerank_factor, combiner)?;

            Ok(Box::new(DenseVectorScorer::new(results, field.0)) as Box<dyn Scorer>)
        })
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
    fn new(results: Vec<VectorSearchResult>, field_id: u32) -> Self {
        Self {
            results,
            position: 0,
            field_id,
        }
    }
}

impl Scorer for DenseVectorScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        while self.doc() < target && self.doc() != TERMINATED {
            self.advance();
        }
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
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

/// Sparse vector query for similarity search
#[derive(Debug, Clone)]
pub struct SparseVectorQuery {
    /// Field containing the sparse vectors
    pub field: Field,
    /// Query vector as (dimension_id, weight) pairs
    pub vector: Vec<(u32, f32)>,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
}

impl SparseVectorQuery {
    /// Create a new sparse vector query
    pub fn new(field: Field, vector: Vec<(u32, f32)>) -> Self {
        Self {
            field,
            vector,
            combiner: MultiValueCombiner::Sum,
        }
    }

    /// Set the multi-value score combiner
    pub fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.combiner = combiner;
        self
    }

    /// Create from separate indices and weights vectors
    pub fn from_indices_weights(field: Field, indices: Vec<u32>, weights: Vec<f32>) -> Self {
        let vector: Vec<(u32, f32)> = indices.into_iter().zip(weights).collect();
        Self::new(field, vector)
    }

    /// Create from raw text using a HuggingFace tokenizer (single segment)
    ///
    /// This method tokenizes the text and creates a sparse vector query.
    /// For multi-segment indexes, use `from_text_with_stats` instead.
    ///
    /// # Arguments
    /// * `field` - The sparse vector field to search
    /// * `text` - Raw text to tokenize
    /// * `tokenizer_name` - HuggingFace tokenizer path (e.g., "bert-base-uncased")
    /// * `weighting` - Weighting strategy for tokens
    /// * `sparse_index` - Optional sparse index for IDF lookup (required for IDF weighting)
    #[cfg(feature = "native")]
    pub fn from_text(
        field: Field,
        text: &str,
        tokenizer_name: &str,
        weighting: crate::structures::QueryWeighting,
        sparse_index: Option<&crate::segment::SparseIndex>,
    ) -> crate::Result<Self> {
        use crate::structures::QueryWeighting;
        use crate::tokenizer::tokenizer_cache;

        let tokenizer = tokenizer_cache().get_or_load(tokenizer_name)?;
        let token_ids = tokenizer.tokenize_unique(text)?;

        let weights: Vec<f32> = match weighting {
            QueryWeighting::One => vec![1.0f32; token_ids.len()],
            QueryWeighting::Idf => {
                if let Some(index) = sparse_index {
                    index.idf_weights(&token_ids)
                } else {
                    vec![1.0f32; token_ids.len()]
                }
            }
        };

        let vector: Vec<(u32, f32)> = token_ids.into_iter().zip(weights).collect();
        Ok(Self::new(field, vector))
    }

    /// Create from raw text using global statistics (multi-segment)
    ///
    /// This is the recommended method for multi-segment indexes as it uses
    /// aggregated IDF values across all segments for consistent ranking.
    ///
    /// # Arguments
    /// * `field` - The sparse vector field to search
    /// * `text` - Raw text to tokenize
    /// * `tokenizer` - Pre-loaded HuggingFace tokenizer
    /// * `weighting` - Weighting strategy for tokens
    /// * `global_stats` - Global statistics for IDF computation
    #[cfg(feature = "native")]
    pub fn from_text_with_stats(
        field: Field,
        text: &str,
        tokenizer: &crate::tokenizer::HfTokenizer,
        weighting: crate::structures::QueryWeighting,
        global_stats: Option<&super::GlobalStats>,
    ) -> crate::Result<Self> {
        use crate::structures::QueryWeighting;

        let token_ids = tokenizer.tokenize_unique(text)?;

        let weights: Vec<f32> = match weighting {
            QueryWeighting::One => vec![1.0f32; token_ids.len()],
            QueryWeighting::Idf => {
                if let Some(stats) = global_stats {
                    stats.sparse_idf_weights(field, &token_ids)
                } else {
                    vec![1.0f32; token_ids.len()]
                }
            }
        };

        let vector: Vec<(u32, f32)> = token_ids.into_iter().zip(weights).collect();
        Ok(Self::new(field, vector))
    }

    /// Create from raw text, loading tokenizer from index directory
    ///
    /// This method supports the `index://` prefix for tokenizer paths,
    /// loading tokenizer.json from the index directory.
    ///
    /// # Arguments
    /// * `field` - The sparse vector field to search
    /// * `text` - Raw text to tokenize
    /// * `tokenizer_bytes` - Tokenizer JSON bytes (pre-loaded from directory)
    /// * `weighting` - Weighting strategy for tokens
    /// * `global_stats` - Global statistics for IDF computation
    #[cfg(feature = "native")]
    pub fn from_text_with_tokenizer_bytes(
        field: Field,
        text: &str,
        tokenizer_bytes: &[u8],
        weighting: crate::structures::QueryWeighting,
        global_stats: Option<&super::GlobalStats>,
    ) -> crate::Result<Self> {
        use crate::structures::QueryWeighting;
        use crate::tokenizer::HfTokenizer;

        let tokenizer = HfTokenizer::from_bytes(tokenizer_bytes)?;
        let token_ids = tokenizer.tokenize_unique(text)?;

        let weights: Vec<f32> = match weighting {
            QueryWeighting::One => vec![1.0f32; token_ids.len()],
            QueryWeighting::Idf => {
                if let Some(stats) = global_stats {
                    stats.sparse_idf_weights(field, &token_ids)
                } else {
                    vec![1.0f32; token_ids.len()]
                }
            }
        };

        let vector: Vec<(u32, f32)> = token_ids.into_iter().zip(weights).collect();
        Ok(Self::new(field, vector))
    }
}

impl Query for SparseVectorQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let vector = self.vector.clone();
        let combiner = self.combiner;
        Box::pin(async move {
            let results = reader
                .search_sparse_vector(field, &vector, limit, combiner)
                .await?;

            Ok(Box::new(SparseVectorScorer::new(results, field.0)) as Box<dyn Scorer>)
        })
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}

/// Scorer for sparse vector search results with ordinal tracking
struct SparseVectorScorer {
    results: Vec<VectorSearchResult>,
    position: usize,
    field_id: u32,
}

impl SparseVectorScorer {
    fn new(results: Vec<VectorSearchResult>, field_id: u32) -> Self {
        Self {
            results,
            position: 0,
            field_id,
        }
    }
}

impl Scorer for SparseVectorScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].score
        } else {
            0.0
        }
    }

    fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        while self.doc() < target && self.doc() != TERMINATED {
            self.advance();
        }
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        (self.results.len() - self.position) as u32
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
    use crate::dsl::Field;

    #[test]
    fn test_dense_vector_query_builder() {
        let query = DenseVectorQuery::new(Field(0), vec![1.0, 2.0, 3.0])
            .with_nprobe(64)
            .with_rerank_factor(5);

        assert_eq!(query.field, Field(0));
        assert_eq!(query.vector.len(), 3);
        assert_eq!(query.nprobe, 64);
        assert_eq!(query.rerank_factor, 5);
    }

    #[test]
    fn test_sparse_vector_query_new() {
        let sparse = vec![(1, 0.5), (5, 0.3), (10, 0.2)];
        let query = SparseVectorQuery::new(Field(0), sparse.clone());

        assert_eq!(query.field, Field(0));
        assert_eq!(query.vector, sparse);
    }

    #[test]
    fn test_sparse_vector_query_from_indices_weights() {
        let query =
            SparseVectorQuery::from_indices_weights(Field(0), vec![1, 5, 10], vec![0.5, 0.3, 0.2]);

        assert_eq!(query.vector, vec![(1, 0.5), (5, 0.3), (10, 0.2)]);
    }
}
