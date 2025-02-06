//! Vector query types for dense and sparse vector search

use crate::dsl::Field;
use crate::segment::{SegmentReader, VectorSearchResult};
use crate::{DocId, Score, TERMINATED};

use super::ScoredPosition;
use super::traits::{CountFuture, MatchedPositions, Query, Scorer, ScorerFuture};

/// Strategy for combining scores when a document has multiple values for the same field
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultiValueCombiner {
    /// Sum all scores (accumulates dot product contributions)
    Sum,
    /// Take the maximum score
    Max,
    /// Take the average score
    Avg,
    /// Log-Sum-Exp: smooth maximum approximation (default)
    /// `score = (1/t) * log(Σ exp(t * sᵢ))`
    /// Higher temperature → closer to max; lower → closer to mean
    LogSumExp {
        /// Temperature parameter (default: 1.5)
        temperature: f32,
    },
    /// Weighted Top-K: weight top scores with exponential decay
    /// `score = Σ wᵢ * sorted_scores[i]` where `wᵢ = decay^i`
    WeightedTopK {
        /// Number of top scores to consider (default: 5)
        k: usize,
        /// Decay factor per rank (default: 0.7)
        decay: f32,
    },
}

impl Default for MultiValueCombiner {
    fn default() -> Self {
        // LogSumExp with temperature 1.5 provides good balance between
        // max (best relevance) and sum (saturation from multiple matches)
        MultiValueCombiner::LogSumExp { temperature: 1.5 }
    }
}

impl MultiValueCombiner {
    /// Create LogSumExp combiner with default temperature (1.5)
    pub fn log_sum_exp() -> Self {
        Self::LogSumExp { temperature: 1.5 }
    }

    /// Create LogSumExp combiner with custom temperature
    pub fn log_sum_exp_with_temperature(temperature: f32) -> Self {
        Self::LogSumExp { temperature }
    }

    /// Create WeightedTopK combiner with defaults (k=5, decay=0.7)
    pub fn weighted_top_k() -> Self {
        Self::WeightedTopK { k: 5, decay: 0.7 }
    }

    /// Create WeightedTopK combiner with custom parameters
    pub fn weighted_top_k_with_params(k: usize, decay: f32) -> Self {
        Self::WeightedTopK { k, decay }
    }

    /// Combine multiple scores into a single score
    pub fn combine(&self, scores: &[(u32, f32)]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }

        match self {
            MultiValueCombiner::Sum => scores.iter().map(|(_, s)| s).sum(),
            MultiValueCombiner::Max => scores
                .iter()
                .map(|(_, s)| *s)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0),
            MultiValueCombiner::Avg => {
                let sum: f32 = scores.iter().map(|(_, s)| s).sum();
                sum / scores.len() as f32
            }
            MultiValueCombiner::LogSumExp { temperature } => {
                // Numerically stable log-sum-exp:
                // LSE(x) = max(x) + log(Σ exp(xᵢ - max(x)))
                let t = *temperature;
                let max_score = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                let sum_exp: f32 = scores
                    .iter()
                    .map(|(_, s)| (t * (s - max_score)).exp())
                    .sum();

                max_score + sum_exp.ln() / t
            }
            MultiValueCombiner::WeightedTopK { k, decay } => {
                // Sort scores descending and take top k
                let mut sorted: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                sorted.truncate(*k);

                // Apply exponential decay weights
                let mut weight = 1.0f32;
                let mut weighted_sum = 0.0f32;
                let mut weight_total = 0.0f32;

                for score in sorted {
                    weighted_sum += weight * score;
                    weight_total += weight;
                    weight *= decay;
                }

                if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    0.0
                }
            }
        }
    }
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
    /// Approximate search factor (1.0 = exact, lower values = faster but approximate)
    /// Controls WAND pruning aggressiveness in block-max scoring
    pub heap_factor: f32,
}

impl SparseVectorQuery {
    /// Create a new sparse vector query
    pub fn new(field: Field, vector: Vec<(u32, f32)>) -> Self {
        Self {
            field,
            vector,
            combiner: MultiValueCombiner::Sum,
            heap_factor: 1.0,
        }
    }

    /// Set the multi-value score combiner
    pub fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.combiner = combiner;
        self
    }

    /// Set the heap factor for approximate search
    ///
    /// Controls the trade-off between speed and recall:
    /// - 1.0 = exact search (default)
    /// - 0.8-0.9 = ~20-40% faster with minimal recall loss
    /// - Lower values = more aggressive pruning, faster but lower recall
    pub fn with_heap_factor(mut self, heap_factor: f32) -> Self {
        self.heap_factor = heap_factor.clamp(0.0, 1.0);
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
            QueryWeighting::IdfFile => {
                use crate::tokenizer::idf_weights_cache;
                if let Some(idf) = idf_weights_cache().get_or_load(tokenizer_name) {
                    token_ids.iter().map(|&id| idf.get(id)).collect()
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
                    // Clamp to zero: negative weights don't make sense for IDF
                    stats
                        .sparse_idf_weights(field, &token_ids)
                        .into_iter()
                        .map(|w| w.max(0.0))
                        .collect()
                } else {
                    vec![1.0f32; token_ids.len()]
                }
            }
            QueryWeighting::IdfFile => {
                // IdfFile requires a tokenizer name for HF model lookup;
                // this code path doesn't have one, so fall back to 1.0
                vec![1.0f32; token_ids.len()]
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
                    // Clamp to zero: negative weights don't make sense for IDF
                    stats
                        .sparse_idf_weights(field, &token_ids)
                        .into_iter()
                        .map(|w| w.max(0.0))
                        .collect()
                } else {
                    vec![1.0f32; token_ids.len()]
                }
            }
            QueryWeighting::IdfFile => {
                // IdfFile requires a tokenizer name for HF model lookup;
                // this code path doesn't have one, so fall back to 1.0
                vec![1.0f32; token_ids.len()]
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
        let heap_factor = self.heap_factor;
        Box::pin(async move {
            let results = reader
                .search_sparse_vector(field, &vector, limit, combiner, heap_factor)
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

    #[test]
    fn test_combiner_sum() {
        let scores = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let combiner = MultiValueCombiner::Sum;
        assert!((combiner.combine(&scores) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_combiner_max() {
        let scores = vec![(0, 1.0), (1, 3.0), (2, 2.0)];
        let combiner = MultiValueCombiner::Max;
        assert!((combiner.combine(&scores) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_combiner_avg() {
        let scores = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let combiner = MultiValueCombiner::Avg;
        assert!((combiner.combine(&scores) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_combiner_log_sum_exp() {
        let scores = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
        let combiner = MultiValueCombiner::log_sum_exp();
        let result = combiner.combine(&scores);
        // LogSumExp should be between max (3.0) and max + log(n)/t
        assert!(result >= 3.0);
        assert!(result <= 3.0 + (3.0_f32).ln() / 1.5);
    }

    #[test]
    fn test_combiner_log_sum_exp_approaches_max_with_high_temp() {
        let scores = vec![(0, 1.0), (1, 5.0), (2, 2.0)];
        // High temperature should approach max
        let combiner = MultiValueCombiner::log_sum_exp_with_temperature(10.0);
        let result = combiner.combine(&scores);
        // Should be very close to max (5.0)
        assert!((result - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_combiner_weighted_top_k() {
        let scores = vec![(0, 5.0), (1, 3.0), (2, 1.0), (3, 0.5)];
        let combiner = MultiValueCombiner::weighted_top_k_with_params(3, 0.5);
        let result = combiner.combine(&scores);
        // Top 3: 5.0, 3.0, 1.0 with weights 1.0, 0.5, 0.25
        // weighted_sum = 5*1 + 3*0.5 + 1*0.25 = 6.75
        // weight_total = 1.75
        // result = 6.75 / 1.75 ≈ 3.857
        assert!((result - 3.857).abs() < 0.01);
    }

    #[test]
    fn test_combiner_weighted_top_k_less_than_k() {
        let scores = vec![(0, 2.0), (1, 1.0)];
        let combiner = MultiValueCombiner::weighted_top_k_with_params(5, 0.7);
        let result = combiner.combine(&scores);
        // Only 2 scores, weights 1.0 and 0.7
        // weighted_sum = 2*1 + 1*0.7 = 2.7
        // weight_total = 1.7
        // result = 2.7 / 1.7 ≈ 1.588
        assert!((result - 1.588).abs() < 0.01);
    }

    #[test]
    fn test_combiner_empty_scores() {
        let scores: Vec<(u32, f32)> = vec![];
        assert_eq!(MultiValueCombiner::Sum.combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::Max.combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::Avg.combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::log_sum_exp().combine(&scores), 0.0);
        assert_eq!(MultiValueCombiner::weighted_top_k().combine(&scores), 0.0);
    }

    #[test]
    fn test_combiner_single_score() {
        let scores = vec![(0, 5.0)];
        // All combiners should return 5.0 for a single score
        assert!((MultiValueCombiner::Sum.combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::Max.combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::Avg.combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::log_sum_exp().combine(&scores) - 5.0).abs() < 1e-6);
        assert!((MultiValueCombiner::weighted_top_k().combine(&scores) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_combiner_is_log_sum_exp() {
        let combiner = MultiValueCombiner::default();
        match combiner {
            MultiValueCombiner::LogSumExp { temperature } => {
                assert!((temperature - 1.5).abs() < 1e-6);
            }
            _ => panic!("Default combiner should be LogSumExp"),
        }
    }
}
