//! RPC conversion and orchestration for core-owned candidate scoring.
use crate::proto;
use hermes_core::query::{CandidateFeature, CandidateScoringPlan, LinearModel, ScoreScope};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::Status;

pub(super) fn linear_model(model: &proto::L1Ranking) -> LinearModel {
    LinearModel {
        weights: model.weights.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        bias: model.bias,
        transforms: model
            .transforms
            .iter()
            .map(|(k, t)| {
                (
                    k.clone(),
                    hermes_core::query::FeatureTransform {
                        signed_log1p: t.signed_log1p,
                        scale: t.scale.unwrap_or(1.0),
                        offset: t.offset,
                    },
                )
            })
            .collect(),
    }
}

pub(super) fn scoring_plan(
    fusion: &proto::FusionQuery,
    queries: &[Arc<dyn hermes_core::query::Query>],
    req: &proto::SearchRequest,
    schema: &hermes_core::Schema,
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
        features,
        model: req.l1.as_ref().map(linear_model),
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
