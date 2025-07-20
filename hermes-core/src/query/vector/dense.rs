//! Dense vector query for similarity search (ANN)

use crate::dsl::Field;
use crate::segment::SegmentReader;
use std::sync::{Arc, Mutex};

use super::VectorResultScorer;
use super::combiner::MultiValueCombiner;
use crate::query::traits::{CountFuture, Query, Scorer, ScorerFuture};

/// Maximum number of IVF clusters a single dense query may probe.
///
/// This guard bounds explicit overrides independently of the trained global
/// leaf count. Automatic billion-scale fields normally use far fewer probes.
pub const MAX_DENSE_NPROBE: usize = 65_536;

/// Maximum exact-rerank candidate multiplier accepted by dense search.
pub const MAX_DENSE_RERANK_FACTOR: f32 = crate::query::MAX_CANDIDATE_OVERSUBSCRIPTION as f32;

/// Default exact-rerank candidate multiplier for dense search.
pub const DEFAULT_DENSE_RERANK_FACTOR: f32 = MAX_DENSE_RERANK_FACTOR;

/// Dense vector query for similarity search
#[derive(Debug, Clone)]
pub struct DenseVectorQuery {
    /// Field containing the dense vectors
    pub field: Field,
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of clusters to probe (for IVF indexes)
    pub nprobe: usize,
    /// Re-ranking factor multiplied by k for candidate selection (1x to 2x)
    pub rerank_factor: f32,
    /// How to combine scores for multi-valued documents
    pub combiner: MultiValueCombiner,
    /// One global-quantizer route, shared by all segment scorers spawned for
    /// this query. The cache is versioned, so a query reused after an index
    /// generation change recomputes safely.
    probe_cache: Arc<Mutex<Option<Arc<crate::structures::IvfPqQueryPlan>>>>,
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
            nprobe: 64,
            rerank_factor: DEFAULT_DENSE_RERANK_FACTOR,
            combiner: MultiValueCombiner::Max,
            probe_cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Set the number of clusters to probe (for IVF indexes)
    ///
    /// Values are validated when the query is executed. See
    /// [`MAX_DENSE_NPROBE`].
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the re-ranking factor (e.g. 2.0 = fetch 2x candidates for reranking)
    ///
    /// Values are validated when the query is executed. See
    /// [`MAX_DENSE_RERANK_FACTOR`].
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
        let probe_cache = Arc::clone(&self.probe_cache);
        Box::pin(async move {
            let results = reader
                .search_dense_vector_with_probe_cache(
                    field,
                    &vector,
                    limit,
                    nprobe,
                    rerank_factor,
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
        let results = reader.search_dense_vector_sync_with_probe_cache(
            self.field,
            &self.vector,
            limit,
            self.nprobe,
            self.rerank_factor,
            self.combiner,
            &self.probe_cache,
        )?;
        Ok(Box::new(VectorResultScorer::new(results, self.field.0)) as Box<dyn Scorer>)
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_vector_query_builder() {
        let query = DenseVectorQuery::new(Field(0), vec![1.0, 2.0, 3.0]).with_nprobe(64);

        assert_eq!(query.field, Field(0));
        assert_eq!(query.vector.len(), 3);
        assert_eq!(query.nprobe, 64);
        assert_eq!(query.rerank_factor, 2.0);
    }
}
