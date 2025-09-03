//! Bounded coordinator selection. Scoring and fusion algorithms belong to core;
//! this module validates the wire contract and assembles global candidate lists.
use crate::proto::hermes as proto;
use hermes_core::query::{
    CandidateScores, FeatureTransform, LinearModel, MultiValueCombiner, PassageFeatures,
    ScoredPosition, SearchResult,
};
use prost::Message;
use std::collections::{BTreeMap, BTreeSet};
use tonic::Status;

pub const MAX_TRANSFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_WINDOW: usize = 10_000;
const MAX_ROWS: usize = 2_000_000;

pub struct CoordinatorPlan {
    pub shard_request: proto::SearchRequest,
    request: proto::SearchRequest,
    model: Option<LinearModel>,
    combiner: MultiValueCombiner,
    names: Vec<String>,
    scopes: Vec<i32>,
    output_passages: usize,
    window: usize,
}

pub fn handles(request: &proto::SearchRequest) -> bool {
    request.l1.is_some()
        || (request.reranker.is_none()
            && request.score_export.is_none()
            && matches!(request.query.as_ref().and_then(|query| query.query.as_ref()),
            Some(proto::query::Query::Fusion(fusion)) if fusion.method != proto::FusionMethod::FusionCandidates as i32))
}

pub fn expected_export_method(request: &proto::SearchRequest) -> Option<&'static str> {
    if request.l1.is_some() {
        Some("linear_v1")
    } else if request.score_export.is_some() {
        Some("feature_export_v1")
    } else if matches!(request.query.as_ref().and_then(|q| q.query.as_ref()),
        Some(proto::query::Query::Fusion(fusion)) if fusion.method == proto::FusionMethod::FusionCandidates as i32)
    {
        Some("fusion_candidates_v1")
    } else {
        None
    }
}

fn invalid(message: impl Into<String>) -> Status {
    Status::invalid_argument(message.into())
}
fn incompatible(message: impl Into<String>) -> Status {
    Status::failed_precondition(message.into())
}

fn combiner(value: i32) -> Result<MultiValueCombiner, Status> {
    Ok(
        match proto::MultiValueCombiner::try_from(value)
            .map_err(|_| invalid("unknown fusion combiner"))?
        {
            proto::MultiValueCombiner::CombinerLogSumExp
            | proto::MultiValueCombiner::CombinerMax => MultiValueCombiner::Max,
            proto::MultiValueCombiner::CombinerAvg => MultiValueCombiner::Avg,
            proto::MultiValueCombiner::CombinerSum => MultiValueCombiner::Sum,
            proto::MultiValueCombiner::CombinerWeightedTopK => {
                MultiValueCombiner::WeightedTopK { k: 5, decay: 0.7 }
            }
        },
    )
}

fn key(address: Option<&proto::DocAddress>) -> Result<(u128, u32), Status> {
    let address = address.ok_or_else(|| incompatible("candidate lacks an address"))?;
    let segment = u128::from_str_radix(&address.segment_id, 16)
        .map_err(|_| incompatible("candidate has an invalid segment address"))?;
    Ok((segment, address.doc_id))
}

impl CoordinatorPlan {
    pub fn new(mut request: proto::SearchRequest, shards: usize) -> Result<Self, Status> {
        if request.limit == 0 {
            request.limit = 10;
        }
        let window = request.offset as usize + request.limit as usize;
        if window > MAX_WINDOW || shards == 0 {
            return Err(invalid("coordinator result window exceeds 10000"));
        }
        let Some(proto::query::Query::Fusion(fusion)) = request
            .query
            .as_ref()
            .and_then(|query| query.query.as_ref())
        else {
            return Err(invalid("coordinator ranking requires FusionQuery"));
        };
        if fusion.queries.is_empty()
            || fusion.queries.len() > hermes_core::query::MAX_FUSION_SUB_QUERIES
        {
            return Err(invalid("fusion requires 1..16 branches"));
        }
        if request.reranker.is_some() || request.time_budget_ms != 0 {
            return Err(invalid(
                "coordinator fusion requires complete scoring without a legacy reranker",
            ));
        }
        let method = proto::FusionMethod::try_from(fusion.method)
            .map_err(|_| invalid("unknown fusion method"))?;
        if method == proto::FusionMethod::FusionCandidates {
            return Err(invalid(
                "candidate export does not need ranked coordination",
            ));
        }
        if !fusion.rrf_k.is_finite()
            || fusion.rrf_k < 0.0
            || fusion
                .queries
                .iter()
                .any(|branch| !branch.weight.is_finite() || branch.weight < 0.0)
        {
            return Err(invalid(
                "fusion weights and rank constant must be finite and nonnegative",
            ));
        }
        let combiner = combiner(fusion.combiner)?;
        let names: Vec<_> = fusion
            .queries
            .iter()
            .map(|branch| branch.name.clone())
            .collect();
        let scopes = fusion.queries.iter().map(|branch| branch.scope).collect();
        let model = request.l1.as_ref().map(|model| -> Result<_, Status> {
            if model.weights.len() > 16 || model.transforms.len() > 16 {
                return Err(invalid("linear model exceeds the 16-branch bound"));
            }
            if method != proto::FusionMethod::FusionRrf || fusion.rrf_k != 0.0 || fusion.queries.iter().any(|q| q.weight != 0.0) {
                return Err(invalid("l1 directly determines ranking; legacy fusion weights/method/rrf_k must be unset"));
            }
            if names.iter().collect::<BTreeSet<_>>().len() != names.len()
                || names.iter().any(|name| name.is_empty() || name.len() > 128 || !name.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)))
                || fusion.queries.iter().any(|q| !matches!(proto::ScoreScope::try_from(q.scope), Ok(proto::ScoreScope::Document | proto::ScoreScope::Chunk))) {
                return Err(invalid("linear branches need unique names and explicit document/chunk scope"));
            }
            let model = LinearModel { weights: model.weights.iter().map(|(name, &weight)| (name.clone(), weight)).collect(), bias: model.bias,
                transforms: model.transforms.iter().map(|(name, t)| (name.clone(), FeatureTransform { signed_log1p: t.signed_log1p, scale: t.scale.unwrap_or(1.0), offset: t.offset })).collect() };
            model.validate(&names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|e| invalid(e.to_string()))?;
            Ok(model)
        }).transpose()?;
        let candidate_limit = if request.candidate_limit == 0 {
            window
        } else {
            request.candidate_limit as usize
        };
        if candidate_limit < window
            || candidate_limit > hermes_core::query::max_candidate_limit(window)
        {
            return Err(invalid(
                "candidate_limit must cover the requested window within the engine candidate bound",
            ));
        }
        let depth = if fusion.candidate_depth == 0 {
            candidate_limit
        } else {
            fusion.candidate_depth as usize
        };
        let nominating = fusion
            .queries
            .iter()
            .filter(|branch| !branch.score_only)
            .count();
        if nominating == 0
            || depth > hermes_core::query::max_candidate_limit(window)
            || depth.saturating_mul(nominating).saturating_mul(shards)
                > hermes_core::query::MAX_FUSION_CANDIDATE_SLOTS
        {
            return Err(invalid("combined shard nomination budget exceeded"));
        }
        let output_passages = request.score_export.as_ref().map_or(8, |export| {
            if export.passages_per_document == 0 {
                65536
            } else {
                export.passages_per_document as usize
            }
        });
        if output_passages > 65536 {
            return Err(invalid("score export exceeds 65536 passages per document"));
        }
        let mut shard_request = request.clone();
        shard_request.offset = 0;
        shard_request.limit = window as u32;
        if model.is_some() {
            // Same formula on each shard: local top window is sufficient for
            // exact global top window. Export enough evidence to reapply the
            // formula at the coordinator without transferring discarded docs.
            let minimum = match combiner {
                MultiValueCombiner::Max => 1,
                MultiValueCombiner::WeightedTopK { k, .. } => k,
                _ => 65536,
            };
            shard_request.score_export = Some(proto::ScoreExport {
                passages_per_document: output_passages.max(minimum) as u32,
                all_passages: request
                    .score_export
                    .as_ref()
                    .is_some_and(|export| export.all_passages),
            });
        } else {
            if nominating != fusion.queries.len() {
                return Err(invalid("score_only requires feature backfill"));
            }
            let union_window = depth.saturating_mul(nominating);
            if union_window > MAX_WINDOW {
                return Err(invalid(
                    "per-shard branch union exceeds the 10000-document response window",
                ));
            }
            shard_request.limit = union_window as u32;
            shard_request.candidate_limit = union_window as u32;
            let Some(proto::query::Query::Fusion(fusion)) =
                shard_request.query.as_mut().and_then(|q| q.query.as_mut())
            else {
                unreachable!()
            };
            fusion.method = proto::FusionMethod::FusionCandidates as i32;
            fusion.candidate_depth = depth as u32;
        }
        Ok(Self {
            shard_request,
            request,
            model,
            combiner,
            names,
            scopes,
            output_passages,
            window,
        })
    }

    pub fn finish(
        &self,
        mut responses: Vec<proto::SearchResponse>,
    ) -> Result<proto::SearchResponse, Status> {
        let mut bytes = 0usize;
        for response in &responses {
            bytes = bytes.saturating_add(response.encoded_len());
            if bytes > MAX_TRANSFER_BYTES {
                return Err(Status::resource_exhausted(
                    "combined coordinator response exceeds 64 MiB",
                ));
            }
            let expected = if self.model.is_some() {
                "linear_v1"
            } else {
                "fusion_candidates_v1"
            };
            if response.ranking_method != expected || response.truncated {
                return Err(incompatible(
                    "backend did not return the complete requested candidate contract; complete the Hermes rollout",
                ));
            }
        }
        if let Some(model) = &self.model {
            let names: Vec<_> = self.names.iter().map(String::as_str).collect();
            let mut rows = 0usize;
            let mut addresses = BTreeSet::new();
            for response in &mut responses {
                for hit in &mut response.hits {
                    if !addresses.insert(key(hit.address.as_ref())?) {
                        return Err(incompatible("duplicate candidate address across shards"));
                    }
                    let raw = hit
                        .candidate_scores
                        .as_mut()
                        .ok_or_else(|| incompatible("shard omitted linear candidate features"))?;
                    rows =
                        rows.saturating_add((raw.passages.len() + 1).saturating_mul(names.len()));
                    if rows > MAX_ROWS {
                        return Err(Status::resource_exhausted(
                            "coordinator feature matrix exceeds 2 million values",
                        ));
                    }
                    let values = |map: &std::collections::HashMap<String, f32>,
                                  scope|
                     -> Result<Vec<Option<f32>>, Status> {
                        if map.iter().any(|(name, value)| {
                            !value.is_finite()
                                || !self
                                    .names
                                    .iter()
                                    .zip(&self.scopes)
                                    .any(|(n, &s)| n == name && s == scope)
                        }) {
                            return Err(incompatible(
                                "shard feature keys/scopes do not match the query",
                            ));
                        }
                        Ok(names.iter().map(|name| map.get(*name).copied()).collect())
                    };
                    let mut features = CandidateScores {
                        document: values(&raw.document, proto::ScoreScope::Document as i32)?,
                        passages: raw
                            .passages
                            .iter()
                            .map(|row| {
                                Ok(PassageFeatures {
                                    ordinal: u16::try_from(row.ordinal)
                                        .map_err(|_| incompatible("invalid passage ordinal"))?,
                                    score: 0.0,
                                    values: values(&row.scores, proto::ScoreScope::Chunk as i32)?,
                                })
                            })
                            .collect::<Result<_, Status>>()?,
                        scored_passages: raw.scored_passages as usize,
                    };
                    let score = model
                        .score_candidate(&names, &mut features, self.combiner)
                        .map_err(|e| incompatible(e.to_string()))?;
                    if score.to_bits() != hit.score.to_bits() {
                        return Err(incompatible("shard and broker linear inference disagree"));
                    }
                    hit.score = score;
                    for (raw, scored) in raw.passages.iter_mut().zip(features.passages) {
                        raw.l1_score = Some(scored.score);
                    }
                    raw.passages.sort_by(|a, b| {
                        b.l1_score
                            .unwrap()
                            .total_cmp(&a.l1_score.unwrap())
                            .then(a.ordinal.cmp(&b.ordinal))
                    });
                    raw.passages.truncate(self.output_passages);
                    hit.ordinal_scores = raw
                        .passages
                        .iter()
                        .map(|row| proto::OrdinalScore {
                            ordinal: row.ordinal,
                            score: row.l1_score.unwrap(),
                        })
                        .collect();
                    if self.request.score_export.is_none() {
                        hit.candidate_scores = None;
                    }
                }
            }
            return crate::partition::merge_search_responses(
                responses,
                self.request.offset as usize,
                self.request.limit as usize,
            );
        }
        self.fuse(responses)
    }

    fn fuse(&self, responses: Vec<proto::SearchResponse>) -> Result<proto::SearchResponse, Status> {
        let mut lists: Vec<Vec<SearchResult>> = vec![Vec::new(); self.names.len()];
        let mut hits = BTreeMap::new();
        let mut metadata = Vec::new();
        for mut response in responses {
            for hit in response.hits.drain(..) {
                if hits.insert(key(hit.address.as_ref())?, hit).is_some() {
                    return Err(incompatible("duplicate candidate address across shards"));
                }
            }
            let mut branches = BTreeSet::new();
            for branch in response.fusion_candidates.drain(..) {
                let index = branch.query_index as usize;
                if index >= lists.len() || !branches.insert(index) {
                    return Err(incompatible("invalid exported branch identity"));
                }
                let mut branch_addresses = BTreeSet::new();
                for candidate in branch.candidates {
                    let (segment_id, doc_id) = key(candidate.address.as_ref())?;
                    if !branch_addresses.insert((segment_id, doc_id)) {
                        return Err(incompatible("duplicate candidate address within a branch"));
                    }
                    if !hits.contains_key(&(segment_id, doc_id))
                        || !candidate.score.is_finite()
                        || candidate
                            .ordinal_scores
                            .iter()
                            .any(|s| !s.score.is_finite())
                    {
                        return Err(incompatible(
                            "branch candidate lacks a valid hydrated union entry",
                        ));
                    }
                    lists[index].push(SearchResult {
                        segment_id,
                        doc_id,
                        score: candidate.score,
                        positions: vec![(
                            0,
                            candidate
                                .ordinal_scores
                                .into_iter()
                                .map(|s| ScoredPosition::new(s.ordinal, s.score))
                                .collect(),
                        )],
                    });
                }
            }
            if branches.len() != lists.len() {
                return Err(incompatible("shard omitted a nomination branch"));
            }
            metadata.push(response);
        }
        let Some(proto::query::Query::Fusion(fusion)) =
            self.request.query.as_ref().and_then(|q| q.query.as_ref())
        else {
            unreachable!()
        };
        let method = if fusion.method == proto::FusionMethod::FusionNormalizedWeightedSum as i32 {
            hermes_core::query::FusionMethod::NormalizedWeightedSum
        } else {
            hermes_core::query::FusionMethod::Rrf {
                k: if fusion.rrf_k == 0.0 {
                    hermes_core::query::DEFAULT_RRF_K
                } else {
                    fusion.rrf_k
                },
            }
        };
        let lists = lists
            .into_iter()
            .zip(&fusion.queries)
            .map(|(mut list, branch)| {
                list.sort_by(|a, b| {
                    b.score
                        .total_cmp(&a.score)
                        .then(a.segment_id.cmp(&b.segment_id))
                        .then(a.doc_id.cmp(&b.doc_id))
                });
                (
                    list,
                    if branch.weight == 0.0 {
                        1.0
                    } else {
                        branch.weight
                    },
                )
            })
            .collect();
        let fused = hermes_core::query::try_fuse_ranked_lists_chunked(
            lists,
            method,
            self.combiner,
            self.window,
        )
        .map_err(invalid)?;
        let mut response = crate::partition::merge_search_responses(metadata, 0, self.window)?;
        response.ranking_method = if fusion.method == proto::FusionMethod::FusionRrf as i32 {
            "global_rrf_v1"
        } else {
            "global_weighted_sum_v1"
        }
        .into();
        for result in fused
            .into_iter()
            .skip(self.request.offset as usize)
            .take(self.request.limit as usize)
        {
            let mut hit = hits
                .remove(&(result.segment_id, result.doc_id))
                .ok_or_else(|| incompatible("fused candidate missing from union"))?;
            hit.score = result.score;
            hit.ordinal_scores = result
                .positions
                .into_iter()
                .flat_map(|(_, positions)| {
                    positions.into_iter().map(|p| proto::OrdinalScore {
                        ordinal: p.position,
                        score: p.score,
                    })
                })
                .collect();
            response.hits.push(hit);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn address(segment: u128, doc_id: u32) -> Option<proto::DocAddress> {
        Some(proto::DocAddress {
            segment_id: format!("{segment:032x}"),
            doc_id,
        })
    }
    fn request(linear: bool) -> proto::SearchRequest {
        proto::SearchRequest {
            index_name: "documents".into(),
            limit: 1,
            candidate_limit: 2,
            query: Some(proto::Query {
                query: Some(proto::query::Query::Fusion(proto::FusionQuery {
                    queries: ["x", "y"]
                        .into_iter()
                        .map(|name| proto::WeightedQuery {
                            name: if linear { name.into() } else { String::new() },
                            scope: if linear {
                                proto::ScoreScope::Document as i32
                            } else {
                                0
                            },
                            query: Some(proto::Query {
                                query: Some(proto::query::Query::Term(proto::TermQuery {
                                    field: name.into(),
                                    term: "query".into(),
                                    ..Default::default()
                                })),
                            }),
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                })),
            }),
            l1: linear.then(|| proto::L1Ranking {
                weights: HashMap::from([("x".into(), 1.0), ("y".into(), -0.3)]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn shard_linear_top_window_preserves_global_top_k_and_broker_reapplies_the_same_formula() {
        let mut req = request(true);
        req.offset = 7;
        req.limit = 5;
        req.candidate_limit = 24;
        let plan = CoordinatorPlan::new(req, 3).unwrap();
        assert_eq!(plan.shard_request.limit, 12);
        assert_eq!(plan.shard_request.offset, 0);
        assert!(plan.shard_request.score_export.is_some());
        let mut all = Vec::new();
        let mut shards = vec![Vec::new(); 3];
        for i in 0..150u32 {
            let raw = HashMap::from([
                ("x".into(), (i * 13 % 71) as f32 - 32.0),
                ("y".into(), (i * 7 % 53) as f32),
            ]);
            let score = plan
                .model
                .as_ref()
                .unwrap()
                .score(&["x", "y"], &[Some(raw["x"]), Some(raw["y"])])
                .unwrap();
            let hit = proto::SearchHit {
                address: address(1 + u128::from(i % 3), i),
                score,
                candidate_scores: Some(proto::CandidateScores {
                    document: raw,
                    ..Default::default()
                }),
                ..Default::default()
            };
            all.push(hit.clone());
            shards[(i % 3) as usize].push(hit);
        }
        let sort = |hits: &mut Vec<proto::SearchHit>| {
            hits.sort_by(|a, b| {
                b.score.total_cmp(&a.score).then(
                    key(a.address.as_ref())
                        .unwrap()
                        .cmp(&key(b.address.as_ref()).unwrap()),
                )
            })
        };
        sort(&mut all);
        let responses = shards
            .into_iter()
            .map(|mut hits| {
                sort(&mut hits);
                hits.truncate(12);
                proto::SearchResponse {
                    hits,
                    ranking_method: "linear_v1".into(),
                    ..Default::default()
                }
            })
            .collect();
        let actual = plan.finish(responses).unwrap();
        let actual: Vec<_> = actual
            .hits
            .iter()
            .map(|hit| (key(hit.address.as_ref()).unwrap(), hit.score))
            .collect();
        let expected: Vec<_> = all[7..12]
            .iter()
            .map(|hit| (key(hit.address.as_ref()).unwrap(), hit.score))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn global_rrf_keeps_a_winner_that_shard_local_rrf_would_tie_behind_another_document() {
        let plan = CoordinatorPlan::new(request(false), 2).unwrap();
        assert_eq!(plan.shard_request.limit, 4);
        let response = |segment, documents: &[(u32, f32, f32)]| proto::SearchResponse {
            ranking_method: "fusion_candidates_v1".into(),
            hits: documents
                .iter()
                .map(|&(doc, _, _)| proto::SearchHit {
                    address: address(segment, doc),
                    ..Default::default()
                })
                .collect(),
            fusion_candidates: (0..2)
                .map(|branch| proto::FusionCandidateList {
                    query_index: branch,
                    candidates: documents
                        .iter()
                        .map(|&(doc, x, y)| proto::FusionCandidate {
                            address: address(segment, doc),
                            score: if branch == 0 { x } else { y },
                            ..Default::default()
                        })
                        .collect(),
                })
                .collect(),
            ..Default::default()
        };
        let result = plan
            .finish(vec![
                response(1, &[(1, 100.0, 1.0)]),
                response(2, &[(1, 99.0, 100.0), (2, 98.0, 99.0)]),
            ])
            .unwrap();
        assert_eq!(result.ranking_method, "global_rrf_v1");
        assert_eq!(key(result.hits[0].address.as_ref()).unwrap(), (2, 1));
        assert!((result.hits[0].score - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-7);
    }

    #[test]
    fn linear_mixed_versions_missing_features_and_disagreement_fail_loudly() {
        let plan = CoordinatorPlan::new(request(true), 2).unwrap();
        assert!(plan.finish(vec![proto::SearchResponse::default()]).is_err());
        let mut response = proto::SearchResponse {
            ranking_method: "linear_v1".into(),
            hits: vec![proto::SearchHit {
                address: address(1, 1),
                score: 123.0,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(plan.finish(vec![response.clone()]).is_err());
        response.hits[0].candidate_scores = Some(proto::CandidateScores {
            document: HashMap::from([("x".into(), 2.0)]),
            ..Default::default()
        });
        assert!(
            plan.finish(vec![response])
                .unwrap_err()
                .message()
                .contains("inference disagree")
        );
    }

    #[test]
    fn average_widens_shard_passage_exports_while_max_keeps_bounded_rows() {
        let mut req = request(true);
        req.score_export = Some(proto::ScoreExport {
            passages_per_document: 1,
            all_passages: false,
        });
        let max = CoordinatorPlan::new(req.clone(), 2).unwrap();
        assert_eq!(
            max.shard_request
                .score_export
                .unwrap()
                .passages_per_document,
            1
        );
        let Some(proto::query::Query::Fusion(fusion)) =
            req.query.as_mut().and_then(|q| q.query.as_mut())
        else {
            unreachable!()
        };
        fusion.combiner = proto::MultiValueCombiner::CombinerAvg as i32;
        let avg = CoordinatorPlan::new(req, 2).unwrap();
        assert_eq!(
            avg.shard_request
                .score_export
                .unwrap()
                .passages_per_document,
            65536
        );
    }
}
