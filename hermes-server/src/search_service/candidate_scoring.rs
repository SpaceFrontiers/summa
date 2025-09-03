//! RPC conversion and orchestration for core-owned candidate scoring.
use crate::proto;
use hermes_core::query::{CandidateFeature, CandidateScoringPlan, RankingModel, ScoreScope};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

pub(super) fn rrf_scores<'a, D: hermes_core::Directory + 'static>(
    searcher: &hermes_core::Searcher<D>,
    request: &proto::SearchRequest,
    fusion: &proto::FusionQuery,
    lists: &[(Vec<hermes_core::query::SearchResult>, u32)],
    selected: impl IntoIterator<Item = &'a hermes_core::query::SearchResult>,
) -> Result<Vec<hermes_core::query::RrfScore>, Status> {
    let scoped = request.l1.is_some() || request.score_export.is_some();
    let lists: Vec<_> = fusion
        .queries
        .iter()
        .enumerate()
        .filter(|(_, branch)| !branch.score_only)
        .zip(lists)
        .map(
            |((query_index, branch), (hits, _))| hermes_core::query::RrfRankedList {
                query_index,
                scope: scoped.then_some(if branch.scope == proto::ScoreScope::Document as i32 {
                    ScoreScope::Document
                } else {
                    ScoreScope::Chunk
                }),
                weight: if branch.weight == 0.0 {
                    1.0
                } else {
                    branch.weight
                },
                hits,
            },
        )
        .collect();
    let selected: Vec<_> = selected
        .into_iter()
        .map(|hit| (hit.segment_id, hit.doc_id))
        .collect();
    searcher
        .rrf_scores_for_hits(
            &lists,
            &selected,
            if fusion.rrf_k == 0.0 {
                hermes_core::query::DEFAULT_RRF_K
            } else {
                fusion.rrf_k
            },
            if fusion.combiner == 0 {
                hermes_core::query::MultiValueCombiner::Max
            } else {
                crate::converters::convert_fusion_combiner(fusion.combiner)
            },
        )
        .map_err(crate::error::hermes_error_to_status)
}

pub(super) fn export_rrf_scores(
    scores: Vec<hermes_core::query::RrfScore>,
    fusion: &proto::FusionQuery,
    budget: &mut super::response::SearchResponseBudget,
) -> Result<Vec<(f32, Vec<proto::RrfContribution>)>, Status> {
    let mut output = Vec::with_capacity(scores.len());
    for score in scores {
        let bytes = score
            .contributions
            .iter()
            .try_fold(32usize, |bytes, vote| {
                bytes.checked_add(
                    std::mem::size_of::<proto::RrfContribution>()
                        + fusion.queries[vote.query_index].name.len(),
                )
            })
            .ok_or_else(|| Status::invalid_argument("RRF response size overflow"))?;
        budget.reserve_retained(bytes)?;
        output.push((
            score.score,
            score
                .contributions
                .into_iter()
                .map(|vote| proto::RrfContribution {
                    query_index: vote.query_index as u32,
                    query_name: fusion.queries[vote.query_index].name.clone(),
                    ordinal: vote.ordinal,
                    rank: vote.rank as u32,
                    score: vote.score,
                })
                .collect(),
        ));
    }
    Ok(output)
}

/// Compact branch scores reference one hydrated union, avoiding repeated
/// stored fields when a document was nominated by several branches.
pub(super) fn export_nomination_lists(
    lists: &[(Vec<hermes_core::query::SearchResult>, u32)],
    budget: &mut super::response::SearchResponseBudget,
) -> Result<Vec<proto::FusionCandidateList>, Status> {
    let mut exported = Vec::with_capacity(lists.len());
    for (query_index, (list, _)) in lists.iter().enumerate() {
        let candidates = export_candidates(list, budget)?;
        exported.push(proto::FusionCandidateList {
            query_index: query_index as u32,
            candidates,
        });
    }
    Ok(exported)
}

pub(super) fn export_candidates(
    list: &[hermes_core::query::SearchResult],
    budget: &mut super::response::SearchResponseBudget,
) -> Result<Vec<proto::FusionCandidate>, Status> {
    let ordinals = list.iter().fold(0usize, |sum, hit| {
        sum.saturating_add(
            hit.positions
                .iter()
                .fold(0usize, |count, (_, positions)| {
                    count.saturating_add(positions.len())
                })
                .max(1),
        )
    });
    budget.reserve_candidate_rows(list.len(), ordinals)?;
    let mut candidates = Vec::with_capacity(list.len());
    for hit in list {
        let count = hit
            .positions
            .iter()
            .map(|(_, positions)| positions.len())
            .sum::<usize>();
        budget.reserve_retained(96usize.saturating_add(count.saturating_mul(16)))?;
        let candidate = proto::FusionCandidate {
            address: Some(proto::DocAddress {
                segment_id: format!("{:032x}", hit.segment_id),
                doc_id: hit.doc_id,
            }),
            score: hit.score,
            ordinal_scores: hit
                .positions
                .iter()
                .flat_map(|(_, positions)| {
                    positions.iter().map(|position| proto::OrdinalScore {
                        ordinal: position.position,
                        score: position.score,
                    })
                })
                .collect(),
        };
        budget.reserve_retained(candidate.encoded_len())?;
        candidates.push(candidate);
    }
    Ok(candidates)
}

pub(super) fn export_trace(
    request: &proto::SearchRequest,
    lists: &[(Vec<hermes_core::query::SearchResult>, u32)],
    root_results: &[hermes_core::query::SearchResult],
    total_seen: u32,
    candidate_limit: usize,
    ranking: (&str, bool),
    budget: &mut super::response::SearchResponseBudget,
) -> Result<proto::SearchTrace, Status> {
    let mut queries = Vec::new();
    let mut filters = Vec::new();
    let root = request.query.as_ref().expect("validated query");
    if let Some(proto::query::Query::Fusion(fusion)) = &root.query {
        for filter in &fusion.filters {
            filters.push(filter.clone());
        }
        let depth = if fusion.candidate_depth == 0 {
            candidate_limit
        } else {
            fusion.candidate_depth as usize
        };
        let mut lists = lists.iter();
        for (query_index, branch) in fusion.queries.iter().enumerate() {
            budget.reserve_retained(256 + branch.name.len())?;
            let (seen, candidates) = if branch.score_only {
                (0, Vec::new())
            } else {
                let (hits, seen) = lists
                    .next()
                    .ok_or_else(|| Status::internal("trace nomination branch missing"))?;
                (*seen, export_candidates(hits, budget)?)
            };
            queries.push(proto::QueryTrace {
                query_index: query_index as u32,
                query_name: branch.name.clone(),
                query: branch.query.clone(),
                scope: branch.scope,
                score_only: branch.score_only,
                candidate_depth: if branch.score_only { 0 } else { depth as u32 },
                total_seen: seen,
                candidates,
            });
        }
    } else {
        budget.reserve_retained(256)?;
        queries.push(proto::QueryTrace {
            query: Some(root.clone()),
            candidate_depth: candidate_limit as u32,
            total_seen,
            candidates: export_candidates(root_results, budget)?,
            ..Default::default()
        });
    }
    Ok(proto::SearchTrace {
        shards: vec![proto::ShardSearchTrace {
            index_name: request.index_name.clone(),
            queries,
            filters,
            ranking_method: ranking.0.to_owned(),
            truncated: ranking.1,
            ..Default::default()
        }],
    })
}

pub(super) fn ranking_model(
    model: &proto::L1Ranking,
    names: &[&str],
) -> Result<RankingModel, Status> {
    RankingModel::compile(
        &model.formula,
        names,
        &model
            .missing_values
            .iter()
            .map(|(name, &value)| (name.clone(), value))
            .collect(),
    )
    .map_err(|error| Status::invalid_argument(error.to_string()))
}

pub(super) fn scoring_plan(
    fusion: &proto::FusionQuery,
    queries: &[Arc<dyn hermes_core::query::Query>],
    req: &proto::SearchRequest,
    schema: &hermes_core::Schema,
    model: Option<RankingModel>,
) -> Result<CandidateScoringPlan, Status> {
    let features = fusion
        .queries
        .iter()
        .zip(queries)
        .map(|(branch, query)| {
            let scope = match proto::ScoreScope::try_from(branch.scope) {
                Ok(proto::ScoreScope::Document) => ScoreScope::Document,
                Ok(proto::ScoreScope::Chunk) => ScoreScope::Chunk,
                _ => {
                    return Err(Status::invalid_argument(format!(
                        "query branch '{}' requires explicit document/chunk scope",
                        branch.name
                    )));
                }
            };
            Ok(CandidateFeature {
                name: branch.name.clone(),
                scope,
                query: query
                    .candidate_query()
                    .map_err(crate::error::hermes_error_to_status)?,
            })
        })
        .collect::<Result<Vec<_>, Status>>()?;
    let export_passages = req.score_export.as_ref().map_or(8, |export| {
        if export.passages_per_document == 0 {
            65536
        } else {
            export.passages_per_document as usize
        }
    });
    let plan = CandidateScoringPlan {
        backfill: req.l1.as_ref().and_then(|l1| l1.backfill).unwrap_or(true),
        document_combiner: match fusion.combiner {
            0 => hermes_core::query::MultiValueCombiner::Max,
            value => crate::converters::convert_fusion_combiner(value),
        },
        features,
        model,
        export_passages,
        all_passages: req
            .score_export
            .as_ref()
            .is_some_and(|export| export.all_passages),
    };
    plan.validate(schema)
        .map_err(crate::error::hermes_error_to_status)?;
    Ok(plan)
}

pub(super) fn export_scores(
    scores: hermes_core::query::CandidateScores,
    plan: &CandidateScoringPlan,
) -> proto::CandidateScores {
    let values = |values: Vec<Option<f32>>| -> HashMap<String, f32> {
        plan.features
            .iter()
            .zip(values)
            .filter_map(|(feature, value)| value.map(|v| (feature.name.clone(), v)))
            .collect()
    };
    proto::CandidateScores {
        document: values(scores.document),
        passages: scores
            .passages
            .into_iter()
            .map(|row| proto::PassageScores {
                ordinal: u32::from(row.ordinal),
                scores: values(row.values),
                l1_score: plan.model.as_ref().map(|_| row.score),
            })
            .collect(),
        scored_passages: scores.scored_passages as u32,
    }
}

pub(super) fn retained_raw_score_bytes(
    scores: &hermes_core::query::CandidateScores,
    plan: &CandidateScoringPlan,
) -> Result<usize, Status> {
    let mut bytes = 128usize;
    for values in std::iter::once(&scores.document).chain(scores.passages.iter().map(|p| &p.values))
    {
        bytes = bytes
            .checked_add(128)
            .ok_or_else(|| Status::resource_exhausted("score export size overflow"))?;
        for (feature, value) in plan.features.iter().zip(values) {
            if value.is_some() {
                bytes = bytes
                    .checked_add(feature.name.len() + 96)
                    .ok_or_else(|| Status::resource_exhausted("score export size overflow"))?;
            }
        }
    }
    Ok(bytes)
}

pub(super) fn convert_filters(
    fusion: &proto::FusionQuery,
    schema: &hermes_core::Schema,
    global_stats: Option<&hermes_core::query::LazyGlobalStats>,
    root: Option<&std::path::Path>,
    shape: &super::QueryShapeLimits,
) -> Result<Vec<Arc<dyn hermes_core::query::Query>>, Status> {
    fusion
        .filters
        .iter()
        .map(|filter| {
            crate::converters::convert_query(filter, schema, global_stats, root, shape)
                .map(Arc::from)
                .map_err(|error| {
                    Status::invalid_argument(format!("Invalid fusion filter: {error}"))
                })
        })
        .collect()
}

pub(super) fn with_filters(
    query: Arc<dyn hermes_core::query::Query>,
    filters: &[Arc<dyn hermes_core::query::Query>],
) -> Arc<dyn hermes_core::query::Query> {
    if filters.is_empty() {
        return query;
    }
    Arc::new(hermes_core::query::FilteredQuery::new(
        query,
        filters.to_vec(),
    ))
}
