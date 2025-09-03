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

use super::vector::MultiValueCombiner;
use super::{ScoredPosition, SearchResult, compare_search_results_desc};

/// Default RRF rank constant (from Cormack et al., the standard choice).
pub const DEFAULT_RRF_K: f32 = 60.0;
/// Maximum independently executed lists accepted by the Searcher fusion API.
pub const MAX_FUSION_SUB_QUERIES: usize = 16;
/// Maximum aggregate list slots retained before fusion.
pub const MAX_FUSION_CANDIDATE_SLOTS: usize = 200_000;
/// Maximum per-ordinal chunk contributions materialized during fusion.
pub const MAX_FUSION_CHUNK_SLOTS: usize = 500_000;

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
    // Avoid reserving an attacker-controlled sum up front. The map can grow
    // naturally if a trusted embedded caller intentionally fuses more.
    const MAX_INITIAL_FUSION_CAPACITY: usize = 200_000;
    let capacity = lists
        .iter()
        .map(|(list, _)| list.len())
        .fold(0usize, usize::saturating_add)
        .min(MAX_INITIAL_FUSION_CAPACITY);
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
        results.select_nth_unstable_by(limit, compare_search_results_desc);
        results.truncate(limit);
    }
    results.sort_unstable_by(compare_search_results_desc);
    results
}

/// Fuse multiple ranked result lists at **chunk granularity**.
///
/// Sub-query results are exploded into per-chunk entries keyed by
/// `(segment_id, doc_id, ordinal)` — for multi-vector fields the ordinal is
/// the chunk index, and results without per-ordinal scores (e.g. text
/// queries) contribute a single pseudo-chunk with ordinal 0. Chunks are
/// ranked *within each list by chunk score*, fused with `method` per chunk
/// key, then combined into a document score with `combiner`.
///
/// Compared to doc-level [`fuse_ranked_lists`]:
/// - Cross-vertical corroboration on the **same chunk** compounds (both
///   contributions land on one key), while scattered hits on different
///   chunks do not inflate the doc under a `Max`-style combiner — an
///   unreliable vertical's noise cannot outvote a strong single-vertical hit.
/// - Fused results carry per-chunk `positions`, so `ordinal_scores` survive
///   fusion (chunk attribution for snippets / chunk selection).
///
/// `MultiValueCombiner::Max` is the recommended combiner. `LogSumExp` is
/// also safe now that it is a softmax-weighted maximum — at RRF's small
/// score scale it degrades toward a mean rather than growing with chunk
/// count — but `Max` states the intent directly.
pub fn fuse_ranked_lists_chunked(
    lists: Vec<(Vec<SearchResult>, f32)>,
    method: FusionMethod,
    combiner: MultiValueCombiner,
    limit: usize,
) -> Vec<SearchResult> {
    type ChunkKey = (u128, u32, u32); // (segment, doc, ordinal)

    // Every (chunk key, list index, contribution) triple. Sorting this once
    // by (key, list index) replaces two hash maps (per-chunk fusion and
    // per-document grouping) with one sort plus two nested run-length
    // passes; the list index keeps the per-key summation in list order, so
    // fused scores are bit-identical to accumulating into a map.
    let mut contributions: Vec<(ChunkKey, u16, f32)> = Vec::new();
    // Reused scratch: this list's chunks as (key, chunk_score)
    let mut chunks: Vec<(ChunkKey, f32)> = Vec::new();

    for (list_index, (list, weight)) in lists.into_iter().enumerate() {
        chunks.clear();
        for result in &list {
            let mut had_positions = false;
            for (_field_id, scored_positions) in &result.positions {
                for sp in scored_positions {
                    had_positions = true;
                    chunks.push(((result.segment_id, result.doc_id, sp.position), sp.score));
                }
            }
            if !had_positions {
                // No per-chunk detail (text query / positions not collected):
                // the whole doc is one pseudo-chunk at ordinal 0.
                chunks.push(((result.segment_id, result.doc_id, 0), result.score));
            }
        }
        if chunks.is_empty() {
            continue;
        }

        // A branch gets one vote per logical passage, even when a Boolean
        // query returns that ordinal under several fields. Keep its strongest
        // raw score before assigning ranks; duplicates must not shift ranks.
        chunks.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
        chunks.dedup_by_key(|entry| entry.0);

        // Rank chunks within this list by chunk score (desc); deterministic
        // tiebreak on the key.
        chunks.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Min-max bounds for score-based fusion
        let (min_score, inv_range) = match method {
            FusionMethod::NormalizedWeightedSum => {
                let max = chunks.first().map(|c| c.1).unwrap_or(0.0);
                let min = chunks.last().map(|c| c.1).unwrap_or(0.0);
                let range = max - min;
                (min, if range > 0.0 { 1.0 / range } else { 0.0 })
            }
            _ => (0.0, 0.0),
        };

        contributions.reserve(chunks.len());
        let list_index = list_index as u16;
        for (rank, &(key, score)) in chunks.iter().enumerate() {
            let contribution = match method {
                FusionMethod::Rrf { k } => weight * rrf_contribution(k, rank + 1),
                FusionMethod::NormalizedWeightedSum => {
                    if inv_range > 0.0 {
                        weight * (score - min_score) * inv_range
                    } else {
                        weight
                    }
                }
            };
            contributions.push((key, list_index, contribution));
        }
    }

    // Sort by (segment, doc, ordinal, list) so one pass yields documents as
    // runs of ordinals, each ordinal a run of list contributions in list
    // order. Ordinals come out ascending, as before.
    contributions.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut results: Vec<SearchResult> = Vec::new();
    let mut ordinals: Vec<(u32, f32)> = Vec::new();
    let mut index = 0;
    while index < contributions.len() {
        let (segment_id, doc_id, _) = contributions[index].0;
        ordinals.clear();
        while index < contributions.len() {
            let (seg, doc, ordinal) = contributions[index].0;
            if seg != segment_id || doc != doc_id {
                break;
            }
            let mut fused = 0.0f32;
            while index < contributions.len()
                && contributions[index].0 == (segment_id, doc_id, ordinal)
            {
                fused += contributions[index].2;
                index += 1;
            }
            ordinals.push((ordinal, fused));
        }
        let score = combiner.combine(&ordinals);
        let scored_positions: Vec<ScoredPosition> = ordinals
            .iter()
            .map(|&(ord, s)| ScoredPosition::new(ord, s))
            .collect();
        results.push(SearchResult {
            doc_id,
            score,
            segment_id,
            positions: vec![(0, scored_positions)],
        });
    }

    if results.len() > limit {
        results.select_nth_unstable_by(limit, compare_search_results_desc);
        results.truncate(limit);
    }
    results.sort_unstable_by(compare_search_results_desc);
    results
}

/// Validated, bounded entry point for chunk-level fusion used by Searcher and
/// the server. The legacy pure helper remains available for trusted embedded
/// callers, while request-facing paths must account for ordinal expansion
/// before allocating fusion maps.
pub fn try_fuse_ranked_lists_chunked(
    lists: Vec<(Vec<SearchResult>, f32)>,
    method: FusionMethod,
    combiner: MultiValueCombiner,
    limit: usize,
) -> Result<Vec<SearchResult>, String> {
    if lists.is_empty() {
        return Err("fusion requires at least one ranked list".to_string());
    }
    if lists.len() > MAX_FUSION_SUB_QUERIES {
        return Err(format!(
            "fusion supports at most {MAX_FUSION_SUB_QUERIES} ranked lists"
        ));
    }
    if let FusionMethod::Rrf { k } = method
        && (!k.is_finite() || k < 0.0)
    {
        return Err(format!(
            "fusion RRF k must be finite and non-negative, got {k}"
        ));
    }
    combiner.validate()?;

    let mut candidates = 0usize;
    let mut chunks = 0usize;
    for (list_index, (list, weight)) in lists.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "fusion list weight at index {list_index} must be finite and non-negative, \
                 got {weight}"
            ));
        }
        candidates = candidates
            .checked_add(list.len())
            .ok_or_else(|| "fusion candidate count overflow".to_string())?;
        if candidates > MAX_FUSION_CANDIDATE_SLOTS {
            return Err(format!(
                "fusion contains more than {MAX_FUSION_CANDIDATE_SLOTS} candidate slots"
            ));
        }
        for result in list {
            let position_count = result
                .positions
                .iter()
                .try_fold(0usize, |count, (_, positions)| {
                    count.checked_add(positions.len())
                })
                .ok_or_else(|| "fusion chunk count overflow".to_string())?;
            // Results without positions contribute one pseudo-chunk.
            chunks = chunks
                .checked_add(position_count.max(1))
                .ok_or_else(|| "fusion chunk count overflow".to_string())?;
            if chunks > MAX_FUSION_CHUNK_SLOTS {
                return Err(format!(
                    "fusion expands to more than {MAX_FUSION_CHUNK_SLOTS} ordinal chunks"
                ));
            }
        }
    }

    Ok(fuse_ranked_lists_chunked(lists, method, combiner, limit))
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

    fn chunked(doc_id: u32, chunks: &[(u32, f32)]) -> SearchResult {
        let positions = vec![(
            0u32,
            chunks
                .iter()
                .map(|&(ord, s)| ScoredPosition::new(ord, s))
                .collect(),
        )];
        SearchResult {
            doc_id,
            // Doc score = max chunk (mirrors a Max combiner upstream)
            score: chunks.iter().map(|&(_, s)| s).fold(0.0, f32::max),
            segment_id: 1,
            positions,
        }
    }

    #[test]
    fn one_branch_cannot_vote_twice_for_the_same_passage_across_fields() {
        let mut duplicate = chunked(1, &[(0, 10.0)]);
        duplicate
            .positions
            .push((1, vec![ScoredPosition::new(0, 9.0)]));
        let unique = chunked(2, &[(0, 11.0)]);
        let fused = fuse_ranked_lists_chunked(
            vec![(vec![unique, duplicate], 1.0)],
            FusionMethod::Rrf { k: 60.0 },
            MultiValueCombiner::Max,
            10,
        );
        assert_eq!(fused[0].doc_id, 2);
        assert_eq!(fused[1].score, 1.0 / 62.0);
    }

    /// The multilingual/short-query regression: a doc that is rank 1 in the
    /// reliable vertical must not be outvoted by a mediocre doc present in
    /// both lists on DIFFERENT chunks. Under doc-level RRF it was
    /// (2/(60+5) > 1/(60+1)); chunk-level fusion with Max fixes it.
    #[test]
    fn test_chunked_fusion_junk_vertical_does_not_outvote() {
        // Sparse (reliable): doc 1 is the clear best; doc 9 is mediocre.
        let sparse = vec![
            chunked(1, &[(0, 10.0)]),
            chunked(2, &[(0, 5.0)]),
            chunked(3, &[(0, 4.0)]),
            chunked(4, &[(0, 3.0)]),
            chunked(9, &[(2, 2.0)]),
        ];
        // Dense (junk for this query): confident ranks over noise; doc 9
        // appears again but on a DIFFERENT chunk.
        let dense = vec![
            chunked(7, &[(0, 0.31)]),
            chunked(8, &[(1, 0.30)]),
            chunked(6, &[(0, 0.29)]),
            chunked(5, &[(3, 0.28)]),
            chunked(9, &[(5, 0.27)]),
        ];

        let fused = fuse_ranked_lists_chunked(
            vec![(sparse, 1.0), (dense, 1.0)],
            FusionMethod::Rrf { k: 60.0 },
            MultiValueCombiner::Max,
            10,
        );

        assert_eq!(
            fused[0].doc_id, 1,
            "sparse rank-1 doc must win over doc 9 (present in both lists on different chunks)"
        );
    }

    /// Same-chunk corroboration across verticals compounds; different-chunk
    /// hits do not (under Max).
    #[test]
    fn test_chunked_fusion_same_chunk_corroboration_wins() {
        // Doc 1: sparse chunk 3 rank 1 + dense chunk 3 rank 1 (same chunk)
        // Doc 2: sparse chunk 0 rank 2 + dense chunk 7 rank 2 (different chunks)
        let sparse = vec![chunked(1, &[(3, 9.0)]), chunked(2, &[(0, 8.0)])];
        let dense = vec![chunked(1, &[(3, 0.9)]), chunked(2, &[(7, 0.8)])];

        let fused = fuse_ranked_lists_chunked(
            vec![(sparse, 1.0), (dense, 1.0)],
            FusionMethod::Rrf { k: 60.0 },
            MultiValueCombiner::Max,
            10,
        );

        assert_eq!(fused[0].doc_id, 1);
        // Doc 1's fused chunk 3 = 1/61 + 1/61; doc 2's best chunk = 1/62
        let expected_doc1 = 2.0 / 61.0;
        assert!((fused[0].score - expected_doc1).abs() < 1e-6);
        assert!(fused[1].score < expected_doc1 / 1.9);

        // Per-chunk attribution survives fusion
        let (_, positions) = &fused[0].positions[0..1][0];
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].position, 3, "fused chunk ordinal preserved");
    }

    /// Results without per-chunk detail (e.g. text queries) fuse as a single
    /// pseudo-chunk at ordinal 0 and can corroborate vector chunk 0.
    #[test]
    fn test_chunked_fusion_pseudo_chunk_for_docs_without_positions() {
        let text = vec![result(1, 3.0), result(2, 2.0)]; // no positions
        let dense = vec![chunked(1, &[(0, 0.9)])];

        let fused = fuse_ranked_lists_chunked(
            vec![(text, 1.0), (dense, 1.0)],
            FusionMethod::Rrf { k: 60.0 },
            MultiValueCombiner::Max,
            10,
        );

        assert_eq!(fused[0].doc_id, 1);
        assert!((fused[0].score - 2.0 / 61.0).abs() < 1e-6);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_validated_chunked_fusion_rejects_invalid_parameters() {
        assert!(
            try_fuse_ranked_lists_chunked(
                vec![(vec![result(1, 1.0)], -1.0)],
                FusionMethod::default(),
                MultiValueCombiner::Max,
                10,
            )
            .is_err()
        );
        assert!(
            try_fuse_ranked_lists_chunked(
                vec![(vec![result(1, 1.0)], 1.0)],
                FusionMethod::Rrf { k: f32::NAN },
                MultiValueCombiner::Max,
                10,
            )
            .is_err()
        );
    }

    /// The sort-based fusion must reproduce the hash-map formulation exactly:
    /// per-chunk contributions summed in list order (bit-identical floats),
    /// ordinals ascending, documents keyed by (segment, doc).
    #[test]
    fn chunked_fusion_sort_grouping_matches_hash_map_reference() {
        use rustc_hash::FxHashMap;

        fn seg(mut result: SearchResult, segment_id: u128) -> SearchResult {
            result.segment_id = segment_id;
            result
        }
        let lists = vec![
            (
                vec![
                    chunked(1, &[(2, 9.0), (0, 8.5)]),
                    seg(chunked(1, &[(0, 7.0)]), 2),
                    chunked(5, &[(1, 6.0)]),
                    result(8, 5.0),
                ],
                1.0,
            ),
            (
                vec![
                    chunked(5, &[(1, 0.9), (4, 0.8)]),
                    chunked(1, &[(0, 0.7)]),
                    seg(chunked(1, &[(0, 0.6)]), 2),
                    result(9, 0.5),
                ],
                0.7,
            ),
            (vec![chunked(1, &[(2, 3.0)]), chunked(8, &[(0, 2.0)])], 1.3),
        ];

        // Reference: accumulate per (segment, doc, ordinal) in list order.
        let mut reference: FxHashMap<(u128, u32, u32), f32> = FxHashMap::default();
        for (list, weight) in &lists {
            let mut chunks: Vec<((u128, u32, u32), f32)> = Vec::new();
            for r in list {
                let mut had = false;
                for (_, positions) in &r.positions {
                    for p in positions {
                        had = true;
                        chunks.push(((r.segment_id, r.doc_id, p.position), p.score));
                    }
                }
                if !had {
                    chunks.push(((r.segment_id, r.doc_id, 0), r.score));
                }
            }
            chunks.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (rank, &(key, _)) in chunks.iter().enumerate() {
                *reference.entry(key).or_insert(0.0) += weight * rrf_contribution(60.0, rank + 1);
            }
        }

        let fused = fuse_ranked_lists_chunked(
            lists,
            FusionMethod::Rrf { k: 60.0 },
            MultiValueCombiner::Max,
            100,
        );
        let mut seen = 0;
        for result in &fused {
            let (_, positions) = &result.positions[0];
            let ordinals: Vec<u32> = positions.iter().map(|p| p.position).collect();
            let mut sorted = ordinals.clone();
            sorted.sort_unstable();
            assert_eq!(
                ordinals, sorted,
                "ordinals ascending for doc {}",
                result.doc_id
            );
            let mut best = f32::NEG_INFINITY;
            for p in positions {
                let expected = reference[&(result.segment_id, result.doc_id, p.position)];
                assert_eq!(
                    p.score.to_bits(),
                    expected.to_bits(),
                    "chunk ({}, {}, {}) fused score",
                    result.segment_id,
                    result.doc_id,
                    p.position
                );
                best = best.max(expected);
                seen += 1;
            }
            assert_eq!(result.score.to_bits(), best.to_bits());
        }
        assert_eq!(seen, reference.len(), "every fused chunk is reported once");
        assert_eq!(fused.len(), 5, "(1,seg1) (1,seg2) 5 8 9");
        for pair in fused.windows(2) {
            assert!(compare_search_results_desc(&pair[0], &pair[1]).is_le());
        }
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
