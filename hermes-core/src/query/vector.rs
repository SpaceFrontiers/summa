//! Vector query types for dense and sparse vector search

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{DocId, Score, TERMINATED};

use super::traits::{CountFuture, Query, Scorer, ScorerFuture};

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
}

impl DenseVectorQuery {
    /// Create a new dense vector query
    pub fn new(field: Field, vector: Vec<f32>) -> Self {
        Self {
            field,
            vector,
            nprobe: 32,
            rerank_factor: 3,
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
}

impl Query for DenseVectorQuery {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        Box::pin(async move {
            let results =
                reader.search_dense_vector(self.field, &self.vector, limit, self.rerank_factor)?;

            Ok(Box::new(DenseVectorScorer::new(results)) as Box<dyn Scorer>)
        })
    }

    fn count_estimate<'a>(&'a self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}

/// Scorer for dense vector search results
struct DenseVectorScorer {
    results: Vec<(u32, f32)>,
    position: usize,
}

impl DenseVectorScorer {
    fn new(results: Vec<(u32, f32)>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl Scorer for DenseVectorScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].0
        } else {
            TERMINATED
        }
    }

    fn score(&self) -> Score {
        if self.position < self.results.len() {
            // Convert distance to score (smaller distance = higher score)
            let distance = self.results[self.position].1;
            1.0 / (1.0 + distance)
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
}

/// Sparse vector query for similarity search
#[derive(Debug, Clone)]
pub struct SparseVectorQuery {
    /// Field containing the sparse vectors
    pub field: Field,
    /// Query vector as (dimension_id, weight) pairs
    pub vector: Vec<(u32, f32)>,
}

impl SparseVectorQuery {
    /// Create a new sparse vector query
    pub fn new(field: Field, vector: Vec<(u32, f32)>) -> Self {
        Self { field, vector }
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
    fn scorer<'a>(&'a self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        Box::pin(async move {
            let results = reader
                .search_sparse_vector(self.field, &self.vector, limit)
                .await?;

            Ok(Box::new(SparseVectorScorer::new(results)) as Box<dyn Scorer>)
        })
    }

    fn count_estimate<'a>(&'a self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}

/// Scorer for sparse vector search results
struct SparseVectorScorer {
    results: Vec<(u32, f32)>,
    position: usize,
}

impl SparseVectorScorer {
    fn new(results: Vec<(u32, f32)>) -> Self {
        Self {
            results,
            position: 0,
        }
    }
}

impl Scorer for SparseVectorScorer {
    fn doc(&self) -> DocId {
        if self.position < self.results.len() {
            self.results[self.position].0
        } else {
            TERMINATED
        }
    }

    fn score(&self) -> Score {
        if self.position < self.results.len() {
            self.results[self.position].1
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
