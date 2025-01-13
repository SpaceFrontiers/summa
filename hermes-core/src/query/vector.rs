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
    /// Number of results to return
    pub k: usize,
    /// Number of clusters to probe (for IVF indexes)
    pub nprobe: usize,
    /// Re-ranking factor (multiplied by k for candidate selection)
    pub rerank_factor: usize,
}

impl DenseVectorQuery {
    /// Create a new dense vector query
    pub fn new(field: Field, vector: Vec<f32>, k: usize) -> Self {
        Self {
            field,
            vector,
            k,
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
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a> {
        Box::pin(async move {
            let results =
                reader.search_dense_vector(self.field, &self.vector, self.k, self.rerank_factor)?;

            Ok(Box::new(DenseVectorScorer::new(results)) as Box<dyn Scorer>)
        })
    }

    fn count_estimate<'a>(&'a self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        let k = self.k as u32;
        Box::pin(async move { Ok(k) })
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
    /// Query vector as (index, weight) pairs
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
    /// Number of results to return
    pub k: usize,
}

impl SparseVectorQuery {
    /// Create a new sparse vector query
    pub fn new(field: Field, indices: Vec<u32>, weights: Vec<f32>, k: usize) -> Self {
        Self {
            field,
            indices,
            weights,
            k,
        }
    }

    /// Create from a sparse vector map
    pub fn from_map(field: Field, sparse_vec: &[(u32, f32)], k: usize) -> Self {
        let (indices, weights): (Vec<u32>, Vec<f32>) = sparse_vec.iter().copied().unzip();
        Self::new(field, indices, weights, k)
    }
}

impl Query for SparseVectorQuery {
    fn scorer<'a>(&'a self, reader: &'a SegmentReader) -> ScorerFuture<'a> {
        Box::pin(async move {
            let results = reader
                .search_sparse_vector(self.field, &self.indices, &self.weights, self.k)
                .await?;

            Ok(Box::new(SparseVectorScorer::new(results)) as Box<dyn Scorer>)
        })
    }

    fn count_estimate<'a>(&'a self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        let k = self.k as u32;
        Box::pin(async move { Ok(k) })
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
        let query = DenseVectorQuery::new(Field(0), vec![1.0, 2.0, 3.0], 10)
            .with_nprobe(64)
            .with_rerank_factor(5);

        assert_eq!(query.field, Field(0));
        assert_eq!(query.vector.len(), 3);
        assert_eq!(query.k, 10);
        assert_eq!(query.nprobe, 64);
        assert_eq!(query.rerank_factor, 5);
    }

    #[test]
    fn test_sparse_vector_query_from_map() {
        let sparse = vec![(1, 0.5), (5, 0.3), (10, 0.2)];
        let query = SparseVectorQuery::from_map(Field(0), &sparse, 10);

        assert_eq!(query.indices, vec![1, 5, 10]);
        assert_eq!(query.weights, vec![0.5, 0.3, 0.2]);
        assert_eq!(query.k, 10);
    }
}
