//! Binary dense vector query for Hamming distance search

use crate::dsl::Field;
use crate::segment::SegmentReader;

use super::VectorResultScorer;
use super::combiner::MultiValueCombiner;
use crate::query::traits::{CountFuture, Query, Scorer, ScorerFuture};

/// Binary dense vector query for Hamming distance similarity search
///
/// Uses brute-force XOR + popcount scoring. Score = 1.0 - hamming/dim_bits.
#[derive(Debug, Clone)]
pub struct BinaryDenseVectorQuery {
    /// Field containing the binary dense vectors
    pub field: Field,
    /// Query vector (packed bits, ceil(dim/8) bytes)
    pub vector: Vec<u8>,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
}

impl std::fmt::Display for BinaryDenseVectorQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BinaryDense({}, bytes={})",
            self.field.0,
            self.vector.len(),
        )
    }
}

impl BinaryDenseVectorQuery {
    pub fn new(field: Field, vector: Vec<u8>) -> Self {
        Self {
            field,
            vector,
            combiner: MultiValueCombiner::Max,
        }
    }

    pub fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.combiner = combiner;
        self
    }
}

impl Query for BinaryDenseVectorQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let vector = self.vector.clone();
        let combiner = self.combiner;
        Box::pin(async move {
            let results = reader
                .search_binary_dense_vector(field, &vector, limit, combiner)
                .await?;

            Ok(Box::new(VectorResultScorer::new(results, field.0)) as Box<dyn Scorer>)
        })
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}
