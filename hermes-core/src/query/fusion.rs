//! Hybrid score fusion: combine ranked lists from independent queries.
//!
//! Unlike the L2 reranker (which re-scores the *first-stage candidates*),
//! fusion takes the *union* of several result lists — a document only found
//! by the dense query can still surface in the fused top-k even if the
//! sparse query missed it entirely, and vice versa.
//!
//! Typical use: run a sparse (BM25/SPLADE) query and a dense vector query,
//! then fuse with Reciprocal Rank Fusion:
//!
//! ```ignore
//! let results = searcher
//!     .search_fused(
//!         &[(&sparse_query, 1.0), (&dense_query, 1.0)],
//!         10,
//!         FusionMethod::default(),
//!     )
//!     .await?;
//! ```

use rustc_hash::FxHashMap;

use super::SearchResult;

/// Default RRF rank constant (from Cormack et al., the standard choice).
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Method for fusing multiple ranked result lists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion: `score(d) = Σ_i w_i / (k + rank_i(d))`.
    ///
    /// Rank-based, so it is insensitive to incompatible score scales
    /// (BM25 vs cosine similarity). `k` dampens the impact of top ranks;
    /// 60 is the standard value.
    Rrf { k: f32 },
    /// Weighted sum of min-max normalized scores:
    /// `score(d) = Σ_i w_i * (s_i(d) - min_i) / (max_i - min_i)`.
    ///
    /// Score-based, preserves score gaps within each list. Sensitive to
    /// outliers; prefer RRF unless the score distributions are known.
    ///
    /// Degenerate lists where every score is identical (including
    /// single-result lists) have no min-max range; every document in such a
    /// list contributes the full `weight`, as if tied at the top. Avoid
    /// feeding filter-like subqueries (many docs, constant score) through
    /// this method — use `Rrf`, which only depends on ranks.
    NormalizedWeightedSum,
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::Rrf { k: DEFAULT_RRF_K }
    }
}

/// Reciprocal Rank Fusion contribution of a single 1-based rank.
/// Shared by list fusion here and the L1/L2 reranker fusion.
#[inline]
pub(crate) fn rrf_contribution(k: f32, rank: usize) -> f32 {
    1.0 / (k + rank as f32)
}

/// Fuse multiple ranked result lists into a single top-`limit` list.
///
/// Each input list must be sorted by descending score (the order produced
/// by `Searcher::search`). `weight` scales that list's contribution.
/// Documents are keyed by `(segment_id, doc_id)`; a document absent from a
/// list contributes nothing for that list. Positions from the first list
/// containing the document are preserved.
pub fn fuse_ranked_lists(
    lists: Vec<(Vec<SearchResult>, f32)>,
    method: FusionMethod,
    limit: usize,
) -> Vec<SearchResult> {
    let capacity = lists.iter().map(|(l, _)| l.len()).sum();
    let mut fused: FxHashMap<(u128, u32), SearchResult> =
        FxHashMap::with_capacity_and_hasher(capacity, Default::default());

    for (list, weight) in lists {
        // Precompute min-max normalization bounds for score-based fusion
        let (min_score, inv_range) = match method {
            FusionMethod::NormalizedWeightedSum if !list.is_empty() => {
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                for r in &list {
                    min = min.min(r.score);
                    max = max.max(r.score);
                }
                let range = max - min;
                (min, if range > 0.0 { 1.0 / range } else { 0.0 })
            }
            _ => (0.0, 0.0),
        };

        for (idx, result) in list.into_iter().enumerate() {
            let contribution = match method {
                FusionMethod::Rrf { k } => weight * rrf_contribution(k, idx + 1),
                FusionMethod::NormalizedWeightedSum => {
                    // Single-result lists normalize to 1.0 (inv_range == 0)
                    if inv_range > 0.0 {
                        weight * (result.score - min_score) * inv_range
                    } else {
                        weight
                    }
                }
            };
            fused
                .entry((result.segment_id, result.doc_id))
                .and_modify(|r| r.score += contribution)
                .or_insert_with(|| SearchResult {
                    score: contribution,
                    ..result
                });
        }
    }

    let mut results: Vec<SearchResult> = fused.into_values().collect();
    if results.len() > limit {
        results.select_nth_unstable_by(limit, |a, b| b.score.total_cmp(&a.score));
        results.truncate(limit);
    }
    results.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(doc_id: u32, score: f32) -> SearchResult {
        SearchResult {
            doc_id,
            score,
            segment_id: 1,
            positions: Vec::new(),
        }
    }

    #[test]
    fn test_rrf_union_includes_single_list_docs() {
        // doc 3 only appears in the dense list — union fusion must keep it
        let sparse = vec![result(1, 10.0), result(2, 5.0)];
        let dense = vec![result(3, 0.9), result(1, 0.8)];

        let fused = fuse_ranked_lists(
            vec![(sparse, 1.0), (dense, 1.0)],
            FusionMethod::Rrf { k: 60.0 },
            10,
        );

        assert_eq!(fused.len(), 3);
        // doc 1 is rank 1 + rank 2 → highest fused score
        assert_eq!(fused[0].doc_id, 1);
        let expected = 1.0 / 61.0 + 1.0 / 62.0;
        assert!((fused[0].score - expected).abs() < 1e-6);
        // docs 2 and 3 both have a single rank contribution
        let ids: Vec<u32> = fused.iter().map(|r| r.doc_id).collect();
        assert!(ids.contains(&2) && ids.contains(&3));
    }

    #[test]
    fn test_rrf_weights_scale_contribution() {
        let a = vec![result(1, 1.0)];
        let b = vec![result(2, 1.0)];

        // Same ranks, but list b weighted 2x → doc 2 wins
        let fused = fuse_ranked_lists(vec![(a, 1.0), (b, 2.0)], FusionMethod::Rrf { k: 60.0 }, 10);
        assert_eq!(fused[0].doc_id, 2);
        assert!((fused[0].score - 2.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_weighted_sum() {
        // Incompatible scales: BM25-ish vs cosine-ish
        let sparse = vec![result(1, 20.0), result(2, 10.0), result(3, 0.0)];
        let dense = vec![result(2, 0.99), result(1, 0.55), result(3, 0.11)];

        let fused = fuse_ranked_lists(
            vec![(sparse, 0.5), (dense, 0.5)],
            FusionMethod::NormalizedWeightedSum,
            10,
        );

        assert_eq!(fused.len(), 3);
        // doc 1: 0.5*1.0 + 0.5*0.5 = 0.75; doc 2: 0.5*0.5 + 0.5*1.0 = 0.75;
        // doc 3: 0. Ties broken by doc_id.
        assert_eq!(fused[0].doc_id, 1);
        assert!((fused[0].score - 0.75).abs() < 1e-6);
        assert!((fused[1].score - 0.75).abs() < 1e-6);
        assert_eq!(fused[2].doc_id, 3);
        assert!(fused[2].score.abs() < 1e-6);
    }

    #[test]
    fn test_limit_truncation() {
        let list: Vec<SearchResult> = (0..100).map(|i| result(i, 100.0 - i as f32)).collect();
        let fused = fuse_ranked_lists(vec![(list, 1.0)], FusionMethod::default(), 5);
        assert_eq!(fused.len(), 5);
        assert_eq!(fused[0].doc_id, 0);
    }

    #[test]
    fn test_duplicate_across_segments_not_merged() {
        // Same doc_id in different segments = different documents
        let mut a = result(1, 1.0);
        a.segment_id = 1;
        let mut b = result(1, 1.0);
        b.segment_id = 2;

        let fused = fuse_ranked_lists(
            vec![(vec![a], 1.0), (vec![b], 1.0)],
            FusionMethod::default(),
            10,
        );
        assert_eq!(fused.len(), 2);
    }
}
