//! Binary dense vector query for Hamming distance search

use crate::dsl::Field;
use crate::segment::SegmentReader;
use std::sync::{Arc, Mutex};

use super::VectorResultScorer;
use super::combiner::MultiValueCombiner;
use crate::query::traits::{CountFuture, Query, Scorer, ScorerFuture};

/// Binary dense vector query for Hamming distance similarity search
///
/// Uses global IVF routing when built and a brute-force fallback while the
/// field is accumulating. Leaf scoring remains exact XOR + popcount.
#[derive(Debug, Clone)]
pub struct BinaryDenseVectorQuery {
    /// Field containing the binary dense vectors
    pub field: Field,
    /// Query vector (packed bits, ceil(dim/8) bytes)
    pub vector: Vec<u8>,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
    probe_cache: Arc<Mutex<Option<crate::structures::IvfProbePlan>>>,
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
            probe_cache: Arc::new(Mutex::new(None)),
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
        let probe_cache = Arc::clone(&self.probe_cache);
        Box::pin(async move {
            let results = reader
                .search_binary_dense_vector_with_probe_cache(
                    field,
                    &vector,
                    limit,
                    combiner,
                    &probe_cache,
                )
                .await?;

            Ok(Box::new(VectorResultScorer::new(results, field.0)) as Box<dyn Scorer>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let results = reader.search_binary_dense_vector_sync_with_probe_cache(
            self.field,
            &self.vector,
            limit,
            self.combiner,
            &self.probe_cache,
        )?;
        Ok(Box::new(VectorResultScorer::new(results, self.field.0)) as Box<dyn Scorer>)
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}
