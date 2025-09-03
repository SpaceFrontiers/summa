//! Bounded attribution of organic RRF votes, independent of final ranking.
use super::{FusionMethod, MultiValueCombiner, SearchResult, ranked_chunks, rrf_contribution};
use crate::query::ScoreScope;

/// A nomination branch. None scope preserves legacy chunk fusion semantics.
pub struct RrfRankedList<'a> {
    pub query_index: usize,
    pub scope: Option<ScoreScope>,
    pub weight: f32,
    pub hits: &'a [SearchResult],
}

/// One vote from a branch's complete nomination list, before backfill.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RrfContribution {
    pub query_index: usize,
    pub rank: usize,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct RrfScore {
    pub score: f32,
    pub contributions: Vec<RrfContribution>,
}

impl RrfScore {
    pub(crate) fn document_context(&self) -> f32 {
        self.contributions
            .iter()
            .take_while(|vote| vote.ordinal.is_none())
            .map(|vote| vote.score)
            .sum()
    }

    /// Votes are ordered by ordinal/query index. A lookup is O(log votes +
    /// branches), avoiding a full vote scan for each of up to 65,536 passages.
    pub(crate) fn passage_score(&self, ordinal: u32, context: f32) -> f32 {
        let first = self
            .contributions
            .partition_point(|vote| vote.ordinal < Some(ordinal));
        let passage: f32 = self.contributions[first..]
            .iter()
            .take_while(|vote| vote.ordinal == Some(ordinal))
            .map(|vote| vote.score)
            .sum();
        passage + context
    }
}

/// Compute RRF only for the selected hits, using ranks from complete lists.
/// Input/output hit order is preserved. No retrieval, hydration or L1 scoring
/// occurs here. Document context is broadcast to nominated passage scores.
pub fn rrf_scores_for_hits(
    lists: &[RrfRankedList<'_>],
    selected: &[(u128, u32)],
    k: f32,
    combiner: MultiValueCombiner,
) -> Result<Vec<RrfScore>, String> {
    let borrowed: Vec<_> = lists.iter().map(|list| (list.hits, list.weight)).collect();
    super::validate_fusion_lists(&borrowed, FusionMethod::Rrf { k }, combiner)?;
    if selected.len() > super::MAX_FUSION_CANDIDATE_SLOTS {
        return Err("RRF selected hit budget exceeded".into());
    }
    let mut indexes: Vec<_> = selected
        .iter()
        .copied()
        .enumerate()
        .map(|(i, key)| (key, i))
        .collect();
    indexes.sort_unstable_by_key(|row| row.0);
    if indexes.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("duplicate RRF selected address".into());
    }
    if lists.iter().any(|list| {
        list.query_index >= super::MAX_FUSION_SUB_QUERIES
            || list.hits.iter().any(|hit| {
                !hit.score.is_finite()
                    || hit
                        .positions
                        .iter()
                        .any(|(_, positions)| positions.iter().any(|p| !p.score.is_finite()))
            })
    }) {
        return Err("invalid RRF branch identity or nomination score".into());
    }
    let mut branch_indexes: Vec<_> = lists.iter().map(|list| list.query_index).collect();
    branch_indexes.sort_unstable();
    if branch_indexes.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("duplicate RRF branch identity".into());
    }
    let mut output = vec![RrfScore::default(); selected.len()];
    let mut chunks = Vec::new();
    let mut documents = Vec::new();
    for list in lists {
        if list.scope == Some(ScoreScope::Document) {
            documents.clear();
            documents.extend(
                list.hits
                    .iter()
                    .map(|hit| ((hit.segment_id, hit.doc_id), hit.score)),
            );
            // Keep one organic vote even if a trusted caller repeats a doc.
            documents.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.total_cmp(&a.1)));
            documents.dedup_by_key(|row| row.0);
            documents.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for (rank, &(key, _)) in documents.iter().enumerate() {
                if let Ok(index) = indexes.binary_search_by_key(&key, |row| row.0) {
                    output[indexes[index].1]
                        .contributions
                        .push(RrfContribution {
                            query_index: list.query_index,
                            rank: rank + 1,
                            score: list.weight * rrf_contribution(k, rank + 1),
                            ordinal: None,
                        });
                }
            }
        } else {
            ranked_chunks(list.hits, &mut chunks);
            for (rank, &((segment, doc, ordinal), _)) in chunks.iter().enumerate() {
                if let Ok(index) = indexes.binary_search_by_key(&(segment, doc), |row| row.0) {
                    output[indexes[index].1]
                        .contributions
                        .push(RrfContribution {
                            query_index: list.query_index,
                            rank: rank + 1,
                            score: list.weight * rrf_contribution(k, rank + 1),
                            ordinal: Some(ordinal),
                        });
                }
            }
        }
    }
    let mut passages = Vec::new();
    for hit in &mut output {
        hit.contributions
            .sort_unstable_by_key(|vote| (vote.ordinal, vote.query_index));
        if hit.contributions.is_empty() {
            return Err("selected hit is absent from RRF nomination lists".into());
        }
        let context: f32 = hit
            .contributions
            .iter()
            .filter(|vote| vote.ordinal.is_none())
            .map(|vote| vote.score)
            .sum();
        passages.clear();
        for vote in &hit.contributions {
            let Some(ordinal) = vote.ordinal else {
                continue;
            };
            if let Some((last, total)) = passages.last_mut()
                && *last == ordinal
            {
                *total += vote.score;
            } else {
                passages.push((ordinal, vote.score));
            }
        }
        for (_, score) in &mut passages {
            *score += context;
        }
        hit.score = if passages.is_empty() {
            context
        } else {
            combiner.combine(&passages)
        };
        if !hit.score.is_finite() || hit.contributions.iter().any(|vote| !vote.score.is_finite()) {
            return Err("RRF diagnostic score overflow".into());
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::ScoredPosition;

    fn hit(segment: u128, doc_id: u32, score: f32, positions: &[(u32, f32)]) -> SearchResult {
        SearchResult {
            segment_id: segment,
            doc_id,
            score,
            positions: vec![(
                0,
                positions
                    .iter()
                    .map(|&(ordinal, score)| ScoredPosition::new(ordinal, score))
                    .collect(),
            )],
        }
    }

    #[test]
    fn diagnostics_match_legacy_chunk_fusion_bit_for_bit_for_every_combiner() {
        let lists = [
            vec![
                hit(2, 0, 8.0, &[(0, 8.0), (1, 3.0), (0, 7.0)]),
                hit(1, 1, 8.0, &[]),
                hit(1, 0, 7.0, &[(2, 7.0)]),
            ],
            vec![
                hit(1, 0, 5.0, &[(2, 5.0), (0, 0.0)]),
                hit(2, 0, 6.0, &[(1, 6.0)]),
            ],
        ];
        let weights = [0.7, 2.3];
        let borrowed: Vec<_> = lists
            .iter()
            .zip(weights)
            .map(|(hits, weight)| (hits.as_slice(), weight))
            .collect();
        let ranked: Vec<_> = lists
            .iter()
            .zip(weights)
            .enumerate()
            .map(|(query_index, (hits, weight))| RrfRankedList {
                query_index,
                scope: None,
                weight,
                hits,
            })
            .collect();
        for combiner in [
            MultiValueCombiner::Max,
            MultiValueCombiner::Avg,
            MultiValueCombiner::Sum,
            MultiValueCombiner::LogSumExp { temperature: 1.5 },
            MultiValueCombiner::WeightedTopK { k: 5, decay: 0.7 },
        ] {
            for k in [0.0, 60.0] {
                let fused = super::super::try_fuse_ranked_lists_chunked_borrowed(
                    &borrowed,
                    FusionMethod::Rrf { k },
                    combiner,
                    10,
                )
                .unwrap();
                let selected: Vec<_> = fused
                    .iter()
                    .rev()
                    .map(|hit| (hit.segment_id, hit.doc_id))
                    .collect();
                let scores = rrf_scores_for_hits(&ranked, &selected, k, combiner).unwrap();
                for (fused, score) in fused.iter().rev().zip(scores) {
                    assert_eq!(fused.score.to_bits(), score.score.to_bits());
                }
            }
        }
        // Duplicate ordinal 0 did not shift any rank; tied scores use the full address.
        let scores =
            rrf_scores_for_hits(&ranked, &[(2, 0)], 60.0, MultiValueCombiner::Max).unwrap();
        assert_eq!(
            scores[0]
                .contributions
                .iter()
                .map(|v| (v.query_index, v.ordinal, v.rank))
                .collect::<Vec<_>>(),
            vec![(0, Some(0), 2), (0, Some(1), 4), (1, Some(1), 1)]
        );
    }

    #[test]
    fn document_votes_broadcast_to_real_passages_without_creating_ordinal_zero() {
        let document = [hit(1, 0, 10.0, &[(99, 123.0)]), hit(2, 0, 20.0, &[])];
        let chunk = [hit(1, 0, 8.0, &[(3, 8.0), (7, 5.0)])];
        let lists = [
            RrfRankedList {
                query_index: 0,
                scope: Some(ScoreScope::Document),
                weight: 2.0,
                hits: &document,
            },
            RrfRankedList {
                query_index: 2,
                scope: Some(ScoreScope::Chunk),
                weight: 1.0,
                hits: &chunk,
            },
        ];
        let scores =
            rrf_scores_for_hits(&lists, &[(1, 0), (2, 0)], 60.0, MultiValueCombiner::Max).unwrap();
        assert_eq!(scores[0].score, 2.0 / 62.0 + 1.0 / 61.0);
        assert_eq!(
            scores[0]
                .contributions
                .iter()
                .map(|v| (v.query_index, v.ordinal, v.rank))
                .collect::<Vec<_>>(),
            vec![(0, None, 2), (2, Some(3), 1), (2, Some(7), 2)]
        );
        assert_eq!(scores[1].score, 2.0 / 61.0);
        assert_eq!(scores[1].contributions[0].ordinal, None);
    }

    #[test]
    fn diagnostics_reject_incomplete_identity_nonfinite_scores_and_unbounded_inputs() {
        let hits = [hit(1, 0, 1.0, &[])];
        let list = |query_index| RrfRankedList {
            query_index,
            scope: None,
            weight: 1.0,
            hits: &hits,
        };
        let check = |lists: &[RrfRankedList<'_>], selected: &[(u128, u32)]| {
            rrf_scores_for_hits(lists, selected, 60.0, MultiValueCombiner::Max)
        };
        assert!(check(&[list(0)], &[(2, 0)]).is_err());
        assert!(check(&[list(0)], &[(1, 0), (1, 0)]).is_err());
        assert!(check(&[list(0), list(0)], &[(1, 0)]).is_err());
        assert!(check(&[list(16)], &[(1, 0)]).is_err());
        let nonfinite = [hit(1, 0, f32::NAN, &[])];
        assert!(
            check(
                &[RrfRankedList {
                    hits: &nonfinite,
                    ..list(0)
                }],
                &[(1, 0)]
            )
            .is_err()
        );
        let oversized = vec![hit(1, 0, 1.0, &[]); super::super::MAX_FUSION_CANDIDATE_SLOTS + 1];
        assert!(
            check(
                &[RrfRankedList {
                    hits: &oversized,
                    ..list(0)
                }],
                &[(1, 0)]
            )
            .is_err()
        );
        let zero = check(
            &[RrfRankedList {
                weight: 0.0,
                ..list(0)
            }],
            &[(1, 0)],
        )
        .unwrap();
        assert_eq!(zero[0].score, 0.0);
        assert_eq!(zero[0].contributions.len(), 1);
    }
}
