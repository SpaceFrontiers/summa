//! Sparse vector queries for similarity search (MaxScore-based)

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{DocId, Score, TERMINATED};

use super::combiner::MultiValueCombiner;
use crate::query::ScoredPosition;
use crate::query::traits::{CountFuture, MatchedPositions, Query, Scorer, ScorerFuture};

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
    /// Controls MaxScore pruning aggressiveness in block-max scoring
    pub heap_factor: f32,
    /// Minimum abs(weight) for query dimensions (0.0 = no filtering)
    /// Dimensions below this threshold are dropped before search.
    pub weight_threshold: f32,
    /// Maximum number of query dimensions to process (None = all)
    /// Keeps only the top-k dimensions by abs(weight).
    pub max_query_dims: Option<usize>,
    /// Fraction of query dimensions to keep (0.0-1.0), same semantics as
    /// indexing-time `pruning`: sort by abs(weight) descending,
    /// keep top fraction. None or 1.0 = no pruning.
    pub pruning: Option<f32>,
    /// Multiplier on executor limit for ordinal deduplication (1.0 = no over-fetch)
    pub over_fetch_factor: f32,
    /// Cached pruned vector; None = use `vector` as-is (no pruning applied)
    pruned: Option<Vec<(u32, f32)>>,
}

impl std::fmt::Display for SparseVectorQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims = self.pruned_dims();
        write!(f, "Sparse({}, dims={}", self.field.0, dims.len())?;
        if self.heap_factor < 1.0 {
            write!(f, ", heap={}", self.heap_factor)?;
        }
        if self.vector.len() != dims.len() {
            write!(f, ", orig={}", self.vector.len())?;
        }
        write!(f, ")")
    }
}

impl SparseVectorQuery {
    /// Create a new sparse vector query
    ///
    /// Default combiner is `LogSumExp { temperature: 0.7 }` which provides
    /// saturation for documents with many sparse vectors (e.g., 100+ ordinals).
    /// This prevents over-weighting from multiple matches while still allowing
    /// additional matches to contribute to the score.
    pub fn new(field: Field, vector: Vec<(u32, f32)>) -> Self {
        let mut q = Self {
            field,
            vector,
            combiner: MultiValueCombiner::LogSumExp { temperature: 0.7 },
            heap_factor: 1.0,
            weight_threshold: 0.0,
            max_query_dims: Some(crate::query::MAX_QUERY_TOKENS),
            pruning: None,
            over_fetch_factor: 2.0,
            pruned: None,
        };
        q.pruned = Some(q.compute_pruned_vector());
        q
    }

    /// Effective query dimensions after pruning. Returns `vector` if no pruning is configured.
    pub(crate) fn pruned_dims(&self) -> &[(u32, f32)] {
        self.pruned.as_deref().unwrap_or(&self.vector)
    }

    /// Set the multi-value score combiner
    pub fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.combiner = combiner;
        self
    }

    /// Set executor over-fetch factor for multi-valued fields.
    /// After MaxScore execution, ordinal combining may reduce result count;
    /// this multiplier compensates by fetching more from the executor.
    /// (1.0 = no over-fetch, 2.0 = fetch 2x then combine down)
    pub fn with_over_fetch_factor(mut self, factor: f32) -> Self {
        self.over_fetch_factor = factor.max(1.0);
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

    /// Set minimum weight threshold for query dimensions
    /// Dimensions with abs(weight) below this are dropped before search.
    pub fn with_weight_threshold(mut self, threshold: f32) -> Self {
        self.weight_threshold = threshold;
        self.pruned = Some(self.compute_pruned_vector());
        self
    }

    /// Set maximum number of query dimensions (top-k by weight)
    pub fn with_max_query_dims(mut self, max_dims: usize) -> Self {
        self.max_query_dims = Some(max_dims);
        self.pruned = Some(self.compute_pruned_vector());
        self
    }

    /// Set pruning fraction (0.0-1.0): keep top fraction of query dims by weight.
    /// Same semantics as indexing-time `pruning`.
    pub fn with_pruning(mut self, fraction: f32) -> Self {
        self.pruning = Some(fraction.clamp(0.0, 1.0));
        self.pruned = Some(self.compute_pruned_vector());
        self
    }

    /// Apply weight_threshold, pruning, and max_query_dims, returning the pruned vector.
    fn compute_pruned_vector(&self) -> Vec<(u32, f32)> {
        let original_len = self.vector.len();

        // Step 1: weight_threshold — drop dimensions below minimum weight
        let mut v: Vec<(u32, f32)> = if self.weight_threshold > 0.0 {
            self.vector
                .iter()
                .copied()
                .filter(|(_, w)| w.abs() >= self.weight_threshold)
                .collect()
        } else {
            self.vector.clone()
        };
        let after_threshold = v.len();

        // Step 2: pruning — keep top fraction by abs(weight), same as indexing
        let mut sorted_by_weight = false;
        if let Some(fraction) = self.pruning
            && fraction < 1.0
            && v.len() > 1
        {
            v.sort_unstable_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted_by_weight = true;
            let keep = ((v.len() as f64 * fraction as f64).ceil() as usize).max(1);
            v.truncate(keep);
        }
        let after_pruning = v.len();

        // Step 3: max_query_dims — absolute cap on dimensions
        if let Some(max_dims) = self.max_query_dims
            && v.len() > max_dims
        {
            if !sorted_by_weight {
                v.sort_unstable_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            v.truncate(max_dims);
        }

        if v.len() < original_len && log::log_enabled!(log::Level::Debug) {
            let src: Vec<_> = self
                .vector
                .iter()
                .map(|(d, w)| format!("({},{:.4})", d, w))
                .collect();
            let pruned_fmt: Vec<_> = v.iter().map(|(d, w)| format!("({},{:.4})", d, w)).collect();
            log::debug!(
                "[sparse query] field={}: pruned {}->{} dims \
                 (threshold: {}->{}, pruning: {}->{}, max_dims: {}->{}), \
                 source=[{}], pruned=[{}]",
                self.field.0,
                original_len,
                v.len(),
                original_len,
                after_threshold,
                after_threshold,
                after_pruning,
                after_pruning,
                v.len(),
                src.join(", "),
                pruned_fmt.join(", "),
            );
        }

        v
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
                if let Some(idf) = idf_weights_cache().get_or_load(tokenizer_name, None) {
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
        global_stats: Option<&crate::query::GlobalStats>,
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
        global_stats: Option<&crate::query::GlobalStats>,
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

impl SparseVectorQuery {
    /// Build SparseTermQueryInfo decomposition for MaxScore execution.
    fn sparse_infos(&self) -> Vec<crate::query::SparseTermQueryInfo> {
        self.pruned_dims()
            .iter()
            .map(|&(dim_id, weight)| crate::query::SparseTermQueryInfo {
                field: self.field,
                dim_id,
                weight,
                heap_factor: self.heap_factor,
                combiner: self.combiner,
                over_fetch_factor: self.over_fetch_factor,
            })
            .collect()
    }
}

impl Query for SparseVectorQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let infos = self.sparse_infos();

        Box::pin(async move {
            if infos.is_empty() {
                return Ok(Box::new(crate::query::EmptyScorer) as Box<dyn Scorer>);
            }

            // MaxScore execution with ordinal combining (handles both single and multi-dim)
            if let Some((executor, info)) =
                crate::query::planner::build_sparse_maxscore_executor(&infos, reader, limit, None)
            {
                let raw = executor.execute().await?;
                return Ok(crate::query::planner::combine_sparse_results(
                    raw,
                    info.combiner,
                    info.field,
                    limit,
                ));
            }

            Ok(Box::new(crate::query::EmptyScorer) as Box<dyn Scorer>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let infos = self.sparse_infos();
        if infos.is_empty() {
            return Ok(Box::new(crate::query::EmptyScorer) as Box<dyn Scorer + 'a>);
        }

        // MaxScore execution with ordinal combining (handles both single and multi-dim)
        if let Some((executor, info)) =
            crate::query::planner::build_sparse_maxscore_executor(&infos, reader, limit, None)
        {
            let raw = executor.execute_sync()?;
            return Ok(crate::query::planner::combine_sparse_results(
                raw,
                info.combiner,
                info.field,
                limit,
            ));
        }

        Ok(Box::new(crate::query::EmptyScorer) as Box<dyn Scorer + 'a>)
    }

    fn count_estimate<'a>(&self, _reader: &'a SegmentReader) -> CountFuture<'a> {
        Box::pin(async move { Ok(u32::MAX) })
    }

    fn decompose(&self) -> crate::query::QueryDecomposition {
        let infos = self.sparse_infos();
        if infos.is_empty() {
            crate::query::QueryDecomposition::Opaque
        } else {
            crate::query::QueryDecomposition::SparseTerms(infos)
        }
    }
}

// ── SparseTermQuery: single sparse dimension query (like TermQuery for text) ──

/// Query for a single sparse vector dimension.
///
/// Analogous to `TermQuery` for text: searches one dimension's posting list
/// with a given weight. Multiple `SparseTermQuery` instances are combined as
/// `BooleanQuery` SHOULD clauses to form a full sparse vector search.
#[derive(Debug, Clone)]
pub struct SparseTermQuery {
    pub field: Field,
    pub dim_id: u32,
    pub weight: f32,
    /// MaxScore heap factor (1.0 = exact, lower = approximate)
    pub heap_factor: f32,
    /// Multi-value combiner for ordinal deduplication
    pub combiner: MultiValueCombiner,
    /// Multiplier on executor limit to compensate for ordinal deduplication
    pub over_fetch_factor: f32,
}

impl std::fmt::Display for SparseTermQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SparseTerm({}, dim={}, w={:.3})",
            self.field.0, self.dim_id, self.weight
        )
    }
}

impl SparseTermQuery {
    pub fn new(field: Field, dim_id: u32, weight: f32) -> Self {
        Self {
            field,
            dim_id,
            weight,
            heap_factor: 1.0,
            combiner: MultiValueCombiner::default(),
            over_fetch_factor: 2.0,
        }
    }

    pub fn with_heap_factor(mut self, heap_factor: f32) -> Self {
        self.heap_factor = heap_factor;
        self
    }

    pub fn with_combiner(mut self, combiner: MultiValueCombiner) -> Self {
        self.combiner = combiner;
        self
    }

    pub fn with_over_fetch_factor(mut self, factor: f32) -> Self {
        self.over_fetch_factor = factor.max(1.0);
        self
    }

    /// Create a SparseTermScorer from this query's config against a segment.
    /// Returns EmptyScorer if the dimension doesn't exist.
    fn make_scorer<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> crate::Result<Option<SparseTermScorer<'a>>> {
        let si = match reader.sparse_index(self.field) {
            Some(si) => si,
            None => return Ok(None),
        };
        let (skip_start, skip_count, global_max, block_data_offset) =
            match si.get_skip_range_full(self.dim_id) {
                Some(v) => v,
                None => return Ok(None),
            };
        let cursor = crate::query::TermCursor::sparse(
            si,
            self.weight,
            skip_start,
            skip_count,
            global_max,
            block_data_offset,
        );
        Ok(Some(SparseTermScorer {
            cursor,
            field_id: self.field.0,
        }))
    }
}

impl Query for SparseTermQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let query = self.clone();
        Box::pin(async move {
            let mut scorer = match query.make_scorer(reader)? {
                Some(s) => s,
                None => return Ok(Box::new(crate::query::EmptyScorer) as Box<dyn Scorer + 'a>),
            };
            scorer.cursor.ensure_block_loaded().await.ok();
            Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        _limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let mut scorer = match self.make_scorer(reader)? {
            Some(s) => s,
            None => return Ok(Box::new(crate::query::EmptyScorer) as Box<dyn Scorer + 'a>),
        };
        scorer.cursor.ensure_block_loaded_sync().ok();
        Ok(Box::new(scorer) as Box<dyn Scorer + 'a>)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let dim_id = self.dim_id;
        Box::pin(async move {
            let si = match reader.sparse_index(field) {
                Some(si) => si,
                None => return Ok(0),
            };
            match si.get_skip_range_full(dim_id) {
                Some((_, skip_count, _, _)) => Ok((skip_count * 256) as u32),
                None => Ok(0),
            }
        })
    }

    fn decompose(&self) -> crate::query::QueryDecomposition {
        crate::query::QueryDecomposition::SparseTerms(vec![crate::query::SparseTermQueryInfo {
            field: self.field,
            dim_id: self.dim_id,
            weight: self.weight,
            heap_factor: self.heap_factor,
            combiner: self.combiner,
            over_fetch_factor: self.over_fetch_factor,
        }])
    }
}

/// Lazy scorer for a single sparse dimension, backed by `TermCursor::Sparse`.
///
/// Iterates through the posting list block-by-block using sync I/O.
/// Score for each doc = `query_weight * quantized_stored_weight`.
struct SparseTermScorer<'a> {
    cursor: crate::query::TermCursor<'a>,
    field_id: u32,
}

impl crate::query::docset::DocSet for SparseTermScorer<'_> {
    fn doc(&self) -> DocId {
        let d = self.cursor.doc();
        if d == u32::MAX { TERMINATED } else { d }
    }

    fn advance(&mut self) -> DocId {
        match self.cursor.advance_sync() {
            Ok(d) if d == u32::MAX => TERMINATED,
            Ok(d) => d,
            Err(_) => TERMINATED,
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        match self.cursor.seek_sync(target) {
            Ok(d) if d == u32::MAX => TERMINATED,
            Ok(d) => d,
            Err(_) => TERMINATED,
        }
    }

    fn size_hint(&self) -> u32 {
        0
    }
}

impl Scorer for SparseTermScorer<'_> {
    fn score(&self) -> Score {
        self.cursor.score()
    }

    fn matched_positions(&self) -> Option<MatchedPositions> {
        let ordinal = self.cursor.ordinal();
        let score = self.cursor.score();
        if score == 0.0 {
            return None;
        }
        Some(vec![(
            self.field_id,
            vec![ScoredPosition::new(ordinal as u32, score)],
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Field;

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
