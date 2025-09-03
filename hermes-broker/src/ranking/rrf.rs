//! Validate nomination transport and translate core-owned RRF attribution.
use super::{MAX_TRANSFER_BYTES, MultiValueCombiner, SearchResult, incompatible, key, proto};
use hermes_core::query::{RrfRankedList, ScoreScope, ScoredPosition};
use prost::Message;
use std::collections::BTreeSet;
use tonic::Status;

pub(super) fn take_lists(
    fusion: &proto::FusionQuery,
    responses: &mut [proto::SearchResponse],
) -> Result<Vec<Vec<SearchResult>>, Status> {
    let mut lists = vec![Vec::new(); fusion.queries.len()];
    let expected: BTreeSet<_> = fusion
        .queries
        .iter()
        .enumerate()
        .filter(|(_, branch)| !branch.score_only)
        .map(|(i, _)| i)
        .collect();
    let mut addresses = BTreeSet::new();
    let mut count = 0usize;
    let mut positions = 0usize;
    for response in responses {
        let mut branches = BTreeSet::new();
        for (index, candidates) in nomination_lists(response) {
            if !expected.contains(&index) || !branches.insert(index) {
                return Err(incompatible("invalid RRF nomination branch identity"));
            }
            for candidate in candidates {
                let (segment_id, doc_id) = key(candidate.address.as_ref())?;
                if !addresses.insert((index, segment_id, doc_id)) {
                    return Err(incompatible("duplicate RRF nomination address"));
                }
                if !candidate.score.is_finite()
                    || candidate
                        .ordinal_scores
                        .iter()
                        .any(|s| !s.score.is_finite())
                {
                    return Err(incompatible("non-finite RRF nomination score"));
                }
                count = count.saturating_add(1);
                positions = positions.saturating_add(candidate.ordinal_scores.len().max(1));
                if count > hermes_core::query::MAX_FUSION_CANDIDATE_SLOTS
                    || positions > hermes_core::query::MAX_FUSION_CHUNK_SLOTS
                {
                    return Err(incompatible("combined RRF nomination budget exceeded"));
                }
                lists[index].push(SearchResult {
                    segment_id,
                    doc_id,
                    score: candidate.score,
                    positions: vec![(
                        0,
                        candidate
                            .ordinal_scores
                            .iter()
                            .map(|s| ScoredPosition::new(s.ordinal, s.score))
                            .collect(),
                    )],
                });
            }
        }
        response.fusion_candidates.clear();
        if branches != expected {
            return Err(incompatible(
                "backend omitted RRF nomination lists; complete the Hermes rollout",
            ));
        }
    }
    Ok(lists)
}

pub(super) fn annotate(
    fusion: &proto::FusionQuery,
    scoped: bool,
    combiner: MultiValueCombiner,
    lists: &[Vec<SearchResult>],
    response: &mut proto::SearchResponse,
) -> Result<(), Status> {
    let selected = response
        .hits
        .iter()
        .map(|hit| key(hit.address.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let scores = scores_for_addresses(fusion, scoped, combiner, lists, &selected)?;
    let mut retained = response.encoded_len();
    for (hit, score) in response.hits.iter_mut().zip(scores) {
        attach_score(fusion, hit, score, &mut retained)?;
    }
    response.fusion_candidates.clear();
    if response.encoded_len() > MAX_TRANSFER_BYTES {
        return Err(Status::resource_exhausted(
            "RRF diagnostic response exceeds 64 MiB",
        ));
    }
    Ok(())
}

pub(super) fn nomination_lists(
    response: &proto::SearchResponse,
) -> Vec<(usize, &[proto::FusionCandidate])> {
    if !response.fusion_candidates.is_empty() {
        response
            .fusion_candidates
            .iter()
            .map(|branch| (branch.query_index as usize, branch.candidates.as_slice()))
            .collect()
    } else {
        response
            .trace
            .iter()
            .flat_map(|trace| &trace.shards)
            .flat_map(|shard| &shard.queries)
            .filter(|branch| !branch.score_only)
            .map(|branch| (branch.query_index as usize, branch.candidates.as_slice()))
            .collect()
    }
}

fn scores_for_addresses(
    fusion: &proto::FusionQuery,
    scoped: bool,
    combiner: MultiValueCombiner,
    lists: &[Vec<SearchResult>],
    selected: &[(u128, u32)],
) -> Result<Vec<hermes_core::query::RrfScore>, Status> {
    let lists: Vec<_> = fusion
        .queries
        .iter()
        .enumerate()
        .filter(|(_, branch)| !branch.score_only)
        .map(|(query_index, branch)| RrfRankedList {
            query_index,
            hits: &lists[query_index],
            weight: if branch.weight == 0.0 {
                1.0
            } else {
                branch.weight
            },
            scope: scoped.then_some(if branch.scope == proto::ScoreScope::Document as i32 {
                ScoreScope::Document
            } else {
                ScoreScope::Chunk
            }),
        })
        .collect();
    hermes_core::query::rrf_scores_for_hits(
        &lists,
        selected,
        if fusion.rrf_k == 0.0 {
            hermes_core::query::DEFAULT_RRF_K
        } else {
            fusion.rrf_k
        },
        combiner,
    )
    .map_err(incompatible)
}

/// Global RRF changes passage/document selection, so a truncated raw L1 export
/// is insufficient. Require every organic document before model inference.
pub(super) fn complete_scores(
    fusion: &proto::FusionQuery,
    combiner: MultiValueCombiner,
    lists: &[Vec<SearchResult>],
    responses: &[proto::SearchResponse],
) -> Result<Vec<hermes_core::query::RrfScore>, Status> {
    let expected: BTreeSet<_> = lists
        .iter()
        .flatten()
        .map(|hit| (hit.segment_id, hit.doc_id))
        .collect();
    let selected = responses
        .iter()
        .flat_map(|response| &response.hits)
        .map(|hit| key(hit.address.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    if expected != selected.iter().copied().collect() {
        return Err(incompatible(
            "RRF L1 requires the complete nominated document union",
        ));
    }
    scores_for_addresses(fusion, true, combiner, lists, &selected)
}

fn attach_score(
    fusion: &proto::FusionQuery,
    hit: &mut proto::SearchHit,
    score: hermes_core::query::RrfScore,
    retained: &mut usize,
) -> Result<(), Status> {
    for vote in &score.contributions {
        *retained = retained.saturating_add(
            std::mem::size_of::<proto::RrfContribution>()
                + fusion.queries[vote.query_index].name.len()
                + 32,
        );
    }
    if *retained > MAX_TRANSFER_BYTES {
        return Err(Status::resource_exhausted(
            "RRF diagnostic response exceeds 64 MiB",
        ));
    }
    hit.rrf_score = Some(score.score);
    hit.rrf_contributions = score
        .contributions
        .into_iter()
        .map(|vote| proto::RrfContribution {
            query_index: vote.query_index as u32,
            query_name: fusion.queries[vote.query_index].name.clone(),
            rank: vote.rank as u32,
            ordinal: vote.ordinal,
            score: vote.score,
        })
        .collect();
    Ok(())
}
