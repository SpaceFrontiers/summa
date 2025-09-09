//! Bounded coordinator selection. Scoring and fusion algorithms belong to core;
//! this module validates the wire contract and assembles global candidate lists.
mod rrf;

use crate::proto::hermes as proto;
use hermes_core::query::{
    CandidateScores, MultiValueCombiner, PassageFeatures, RankingModel, ScoredPosition,
    SearchResult,
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
    model: Option<RankingModel>,
    shard_model: Option<RankingModel>,
    combiner: MultiValueCombiner,
    names: Vec<String>,
    scopes: Vec<i32>,
    output_passages: usize,
    window: usize,
    shards: usize,
    passthrough: bool,
}

pub fn handles(request: &proto::SearchRequest) -> bool {
    request.tracing
        || request.include_rrf_scores
        || request.l1.is_some()
        || (request.reranker.is_none()
            && request.score_export.is_none()
            && matches!(request.query.as_ref().and_then(|query| query.query.as_ref()),
            Some(proto::query::Query::Fusion(fusion)) if fusion.method != proto::FusionMethod::FusionCandidates as i32))
}

pub fn expected_export_method(request: &proto::SearchRequest) -> Option<&'static str> {
    if request.l1.is_some() {
        Some("formula_v1")
    } else if request.score_export.is_some() {
        Some("feature_export_v2")
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

pub fn stamp_trace(
    response: &mut proto::SearchResponse,
    shard_id: &str,
    backend_id: &str,
    requested: bool,
) -> Result<(), Status> {
    if !requested {
        return Ok(());
    }
    let trace = response.trace.as_mut().ok_or_else(|| {
        incompatible("backend omitted requested trace; complete the Hermes rollout")
    })?;
    if trace.shards.len() != 1 {
        return Err(incompatible("backend must return exactly one local trace"));
    }
    trace.shards[0].shard_id = shard_id.to_owned();
    trace.shards[0].backend_id = backend_id.to_owned();
    Ok(())
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
            if request.tracing
                && !request.include_rrf_scores
                && request.l1.is_none()
                && request.score_export.is_none()
            {
                if request.candidate_limit != 0
                    && !(window..=hermes_core::query::max_candidate_limit(window))
                        .contains(&(request.candidate_limit as usize))
                {
                    return Err(invalid(
                        "candidate_limit must cover the requested window within the engine candidate bound",
                    ));
                }
                let mut shard_request = request.clone();
                shard_request.offset = 0;
                shard_request.limit = window as u32;
                return Ok(Self {
                    shard_request,
                    request,
                    model: None,
                    shard_model: None,
                    combiner: MultiValueCombiner::Max,
                    names: Vec::new(),
                    scopes: Vec::new(),
                    output_passages: 0,
                    window,
                    shards,
                    passthrough: true,
                });
            }
            return Err(invalid("coordinator ranking requires FusionQuery"));
        };
        if fusion.queries.is_empty()
            || fusion.queries.len() > hermes_core::query::MAX_FUSION_SUB_QUERIES
        {
            return Err(invalid("fusion requires 1..16 branches"));
        }
        if (request.reranker.is_some()
            && (request.l1.is_some() || (!request.include_rrf_scores && !request.tracing)))
            || request.time_budget_ms != 0
        {
            return Err(invalid(
                "coordinator fusion requires complete scoring without a legacy reranker",
            ));
        }
        let method = proto::FusionMethod::try_from(fusion.method)
            .map_err(|_| invalid("unknown fusion method"))?;
        let passthrough = request.l1.is_none()
            && (request.include_rrf_scores || request.tracing)
            && (request.reranker.is_some()
                || request.score_export.is_some()
                || method == proto::FusionMethod::FusionCandidates);
        if method == proto::FusionMethod::FusionCandidates && !passthrough {
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
        if (request.include_rrf_scores || request.tracing)
            && fusion.queries.iter().any(|branch| branch.name.len() > 128)
        {
            return Err(invalid("diagnostic branch names exceed 128 bytes"));
        }
        let combiner = combiner(fusion.combiner)?;
        if request
            .score_export
            .as_ref()
            .is_some_and(|export| export.seed_document_passages)
            && (request
                .l1
                .as_ref()
                .is_some_and(|model| model.backfill == Some(false))
                || !fusion
                    .queries
                    .iter()
                    .any(|branch| branch.scope == proto::ScoreScope::Chunk as i32))
        {
            return Err(invalid(
                "seed_document_passages requires backfill and a chunk-scoped feature",
            ));
        }
        let names: Vec<_> = fusion
            .queries
            .iter()
            .map(|branch| branch.name.clone())
            .collect();
        let scopes = fusion.queries.iter().map(|branch| branch.scope).collect();
        let model = request.l1.as_ref().map(|model| -> Result<_, Status> {
            if model.backfill == Some(false)
                && request.score_export.as_ref().is_some_and(|export| export.all_passages)
            {
                return Err(invalid("all_passages diagnostics require backfill"));
            }
            if model.missing_values.len() > 16 || model.missing_values.keys().any(|name| name.len() > 128) {
                return Err(invalid("L1 missing defaults exceed the 16-branch bound"));
            }
            if method != proto::FusionMethod::FusionRrf || fusion.rrf_k != 0.0 || fusion.queries.iter().any(|q| q.weight != 0.0) {
                return Err(invalid("l1 directly determines ranking; legacy fusion weights/method/rrf_k must be unset"));
            }
            if names.iter().collect::<BTreeSet<_>>().len() != names.len()
                || names.iter().any(|name| name.is_empty() || name.len() > 128 || !name.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)))
                || fusion.queries.iter().any(|q| !matches!(proto::ScoreScope::try_from(q.scope), Ok(proto::ScoreScope::Document | proto::ScoreScope::Chunk))) {
                return Err(invalid("L1 branches need unique names and explicit document/chunk scope"));
            }
            let model = RankingModel::compile(&model.formula, &names.iter().map(String::as_str).collect::<Vec<_>>(), &model.missing_values.iter().map(|(name, &value)| (name.clone(), value)).collect()).map_err(|e| invalid(e.to_string()))?;
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
                seed_document_passages: request
                    .score_export
                    .as_ref()
                    .is_some_and(|export| export.seed_document_passages),
            });
        } else if !passthrough {
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
        let mut shard_model = model.clone();
        if model.as_ref().is_some_and(RankingModel::needs_rrf) {
            let union_window = depth.saturating_mul(nominating);
            if union_window > MAX_WINDOW {
                return Err(invalid(
                    "RRF L1 requires a complete shard union within 10000 documents",
                ));
            }
            shard_request.limit = union_window.max(window) as u32;
            shard_request.candidate_limit = shard_request.limit;
            let Some(proto::query::Query::Fusion(fusion)) =
                shard_request.query.as_mut().and_then(|q| q.query.as_mut())
            else {
                unreachable!()
            };
            fusion.candidate_depth = depth as u32;
            let shard_l1 = shard_request.l1.as_mut().unwrap();
            shard_l1.formula = "0".into();
            shard_l1.missing_values.clear();
            shard_model = Some(
                RankingModel::compile(
                    "0",
                    &names.iter().map(String::as_str).collect::<Vec<_>>(),
                    &BTreeMap::new(),
                )
                .map_err(|e| invalid(e.to_string()))?,
            );
            shard_request
                .score_export
                .as_mut()
                .unwrap()
                .passages_per_document = 65536;
            // The coordinator owns global ranks. A constant shard formula
            // exports every raw row without evaluating global-rank math locally.
            shard_request.include_rrf_scores = true;
        }
        if request.tracing || (model.is_none() && !passthrough) {
            // Trace/Candidates already contains organic lists. Avoid computing
            // and transferring shard-local RRF diagnostics that will be replaced.
            shard_request.include_rrf_scores = false;
        }
        Ok(Self {
            shard_request,
            request,
            model,
            shard_model,
            combiner,
            names,
            scopes,
            output_passages,
            window,
            shards,
            passthrough,
        })
    }

    pub fn finish(
        &self,
        mut responses: Vec<proto::SearchResponse>,
    ) -> Result<proto::SearchResponse, Status> {
        if responses.len() != self.shards {
            return Err(incompatible(
                "coordinator did not receive every shard response",
            ));
        }
        let mut bytes = 0usize;
        let mut trace_candidates = 0usize;
        let mut trace_ordinals = 0usize;
        for response in &responses {
            if self
                .request
                .score_export
                .as_ref()
                .is_some_and(|export| export.seed_document_passages)
                && !response.seeded_document_passages
            {
                return Err(incompatible(
                    "backend did not acknowledge document passage seeding; complete the Hermes rollout",
                ));
            }
            bytes = bytes.saturating_add(response.encoded_len());
            if bytes > MAX_TRANSFER_BYTES {
                return Err(Status::resource_exhausted(
                    "combined coordinator response exceeds 64 MiB",
                ));
            }
            if self.request.tracing {
                let trace = response
                    .trace
                    .as_ref()
                    .ok_or_else(|| incompatible("backend omitted requested trace"))?;
                if trace.shards.len() != 1 {
                    return Err(incompatible("backend must return exactly one local trace"));
                }
                for shard in &trace.shards {
                    for candidate in shard
                        .queries
                        .iter()
                        .flat_map(|query| &query.candidates)
                        .chain(&shard.selected)
                    {
                        trace_candidates = trace_candidates.saturating_add(1);
                        trace_ordinals =
                            trace_ordinals.saturating_add(candidate.ordinal_scores.len().max(1));
                    }
                }
                if trace_candidates > hermes_core::query::MAX_FUSION_CANDIDATE_SLOTS
                    || trace_ordinals > hermes_core::query::MAX_FUSION_CHUNK_SLOTS
                {
                    return Err(invalid(
                        "combined trace candidate or ordinal budget exceeded",
                    ));
                }
            }
            let expected = if self.model.is_some() {
                Some("formula_v1")
            } else if self.passthrough {
                expected_export_method(&self.request)
            } else {
                Some("fusion_candidates_v1")
            };
            if expected.is_some_and(|expected| response.ranking_method != expected)
                || (response.truncated && (!self.passthrough || self.request.include_rrf_scores))
            {
                return Err(incompatible(
                    "backend did not return the complete requested candidate contract; complete the Hermes rollout",
                ));
            }
        }
        let needs_rrf = self.model.as_ref().is_some_and(RankingModel::needs_rrf);
        let diagnostics = if (self.request.include_rrf_scores || needs_rrf)
            && (self.model.is_some() || self.passthrough)
        {
            Some(rrf::take_lists(self.fusion(), &mut responses)?)
        } else {
            None
        };
        if self.passthrough {
            let mut response = crate::partition::merge_search_responses(
                responses,
                self.request.offset as usize,
                self.request.limit as usize,
            )?;
            if let Some(lists) = &diagnostics {
                self.annotate_rrf(lists, &mut response)?;
            }
            if response.encoded_len() > MAX_TRANSFER_BYTES {
                return Err(Status::resource_exhausted("traced response exceeds 64 MiB"));
            }
            return Ok(response);
        }
        if let Some(model) = &self.model {
            let mut global_rrf = if needs_rrf {
                rrf::complete_scores(
                    self.fusion(),
                    self.combiner,
                    diagnostics.as_ref().expect("RRF nominations"),
                    &responses,
                )?
            } else {
                Vec::new()
            }
            .into_iter();
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
                        .ok_or_else(|| incompatible("shard omitted L1 candidate features"))?;
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
                    let score = self
                        .shard_model
                        .as_ref()
                        .expect("shard model")
                        .score_candidate(&names, &mut features, self.combiner, None)
                        .map_err(|e| incompatible(e.to_string()))?;
                    if score.to_bits() != hit.score.to_bits() {
                        return Err(incompatible("shard and broker formula inference disagree"));
                    }
                    hit.score = if needs_rrf {
                        let rrf = global_rrf.next().expect("complete RRF features");
                        hit.rrf_score = None;
                        hit.rrf_contributions.clear();
                        model
                            .score_candidate(&names, &mut features, self.combiner, Some(&rrf))
                            .map_err(|error| invalid(error.to_string()))?
                    } else {
                        score
                    };
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
            if needs_rrf {
                for response in &mut responses {
                    response.ranking_method = "formula_v1".into();
                }
            }
            let mut response = crate::partition::merge_search_responses(
                responses,
                self.request.offset as usize,
                self.request.limit as usize,
            )?;
            if self.request.include_rrf_scores
                && let Some(lists) = diagnostics
            {
                self.annotate_rrf(&lists, &mut response)?;
            }
            if response.encoded_len() > MAX_TRANSFER_BYTES {
                return Err(Status::resource_exhausted("traced response exceeds 64 MiB"));
            }
            return Ok(response);
        }
        self.fuse(responses)
    }

    fn fuse(&self, responses: Vec<proto::SearchResponse>) -> Result<proto::SearchResponse, Status> {
        let mut lists: Vec<Vec<SearchResult>> = vec![Vec::new(); self.names.len()];
        let mut hits = BTreeMap::new();
        let mut metadata = Vec::new();
        for mut response in responses {
            let mut local_addresses = BTreeSet::new();
            for hit in response.hits.drain(..) {
                let address = key(hit.address.as_ref())?;
                local_addresses.insert(address);
                if hits.insert(address, hit).is_some() {
                    return Err(incompatible("duplicate candidate address across shards"));
                }
            }
            let mut branches = BTreeSet::new();
            for (index, candidates) in rrf::nomination_lists(&response) {
                if index >= lists.len() || !branches.insert(index) {
                    return Err(incompatible("invalid exported branch identity"));
                }
                let mut branch_addresses = BTreeSet::new();
                for candidate in candidates {
                    let (segment_id, doc_id) = key(candidate.address.as_ref())?;
                    if !branch_addresses.insert((segment_id, doc_id)) {
                        return Err(incompatible("duplicate candidate address within a branch"));
                    }
                    if !local_addresses.contains(&(segment_id, doc_id))
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
                                .iter()
                                .map(|s| ScoredPosition::new(s.ordinal, s.score))
                                .collect(),
                        )],
                    });
                }
            }
            if branches.len() != lists.len() {
                return Err(incompatible("shard omitted a nomination branch"));
            }
            response.fusion_candidates.clear();
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
        let lists: Vec<_> = lists
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
        let borrowed: Vec<_> = lists
            .iter()
            .map(|(list, weight): &(Vec<SearchResult>, f32)| (list.as_slice(), *weight))
            .collect();
        let fused = hermes_core::query::try_fuse_ranked_lists_chunked_borrowed(
            &borrowed,
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
        if self.request.include_rrf_scores {
            let lists: Vec<_> = lists.into_iter().map(|(list, _)| list).collect();
            self.annotate_rrf(&lists, &mut response)?;
        }
        if response.encoded_len() > MAX_TRANSFER_BYTES {
            return Err(Status::resource_exhausted("traced response exceeds 64 MiB"));
        }
        Ok(response)
    }

    fn fusion(&self) -> &proto::FusionQuery {
        let Some(proto::query::Query::Fusion(fusion)) =
            self.request.query.as_ref().and_then(|q| q.query.as_ref())
        else {
            unreachable!()
        };
        fusion
    }

    fn annotate_rrf(
        &self,
        lists: &[Vec<SearchResult>],
        response: &mut proto::SearchResponse,
    ) -> Result<(), Status> {
        rrf::annotate(
            self.fusion(),
            self.request.l1.is_some() || self.request.score_export.is_some(),
            self.combiner,
            lists,
            response,
        )
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
    fn request(l1: bool) -> proto::SearchRequest {
        proto::SearchRequest {
            index_name: "documents".into(),
            limit: 1,
            candidate_limit: 2,
            query: Some(proto::Query {
                query: Some(proto::query::Query::Fusion(proto::FusionQuery {
                    queries: ["x", "y"]
                        .into_iter()
                        .map(|name| proto::WeightedQuery {
                            name: if l1 { name.into() } else { String::new() },
                            scope: if l1 {
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
            l1: l1.then(|| proto::L1Ranking {
                formula: "x - 0.3 * y".into(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn shard_formula_top_window_preserves_global_top_k_and_broker_reapplies_the_same_formula() {
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
                .score_candidate(
                    &["x", "y"],
                    &mut CandidateScores {
                        document: vec![Some(raw["x"]), Some(raw["y"])],
                        passages: vec![],
                        scored_passages: 0,
                    },
                    MultiValueCombiner::Max,
                    None,
                )
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
                    ranking_method: "formula_v1".into(),
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
        let mut req = request(false);
        req.include_rrf_scores = true;
        let plan = CoordinatorPlan::new(req, 2).unwrap();
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
        assert_eq!(
            result.hits[0].rrf_score.unwrap().to_bits(),
            result.hits[0].score.to_bits()
        );
        assert_eq!(
            result.hits[0]
                .rrf_contributions
                .iter()
                .map(|v| (v.query_index, v.rank))
                .collect::<Vec<_>>(),
            vec![(0, 2), (1, 1)]
        );
        assert_eq!(key(result.hits[0].address.as_ref()).unwrap(), (2, 1));
        assert!((result.hits[0].score - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-7);
    }

    #[test]
    fn formula_mixed_versions_missing_features_and_disagreement_fail_loudly() {
        let plan = CoordinatorPlan::new(request(true), 1).unwrap();
        assert!(plan.finish(vec![proto::SearchResponse::default()]).is_err());
        let mut response = proto::SearchResponse {
            ranking_method: "formula_v1".into(),
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
    fn document_passage_seeding_survives_shard_export_rewrite_and_requires_chunk_backfill() {
        let mut req = request(true);
        req.score_export = Some(proto::ScoreExport {
            seed_document_passages: true,
            ..Default::default()
        });
        assert!(CoordinatorPlan::new(req.clone(), 1).is_err());
        let Some(proto::query::Query::Fusion(fusion)) =
            req.query.as_mut().and_then(|query| query.query.as_mut())
        else {
            panic!("fusion")
        };
        fusion.queries[0].scope = proto::ScoreScope::Chunk as i32;
        let plan = CoordinatorPlan::new(req.clone(), 1).unwrap();
        let error = plan
            .finish(vec![proto::SearchResponse::default()])
            .unwrap_err();
        assert!(
            error
                .message()
                .contains("acknowledge document passage seeding")
        );
        assert!(
            plan.shard_request
                .score_export
                .as_ref()
                .unwrap()
                .seed_document_passages
        );
        req.l1.as_mut().unwrap().backfill = Some(false);
        assert!(CoordinatorPlan::new(req, 1).is_err());
    }

    #[test]
    fn disabled_backfill_rejects_all_passage_expansion_before_dispatch() {
        let mut req = request(true);
        req.l1.as_mut().unwrap().backfill = Some(false);
        req.score_export = Some(proto::ScoreExport {
            all_passages: true,
            seed_document_passages: false,
            ..Default::default()
        });
        let error = CoordinatorPlan::new(req, 1).err().expect("invalid policy");
        assert!(
            error
                .message()
                .contains("all_passages diagnostics require backfill")
        );
    }

    #[test]
    fn average_widens_shard_passage_exports_while_max_keeps_bounded_rows() {
        let mut req = request(true);
        req.score_export = Some(proto::ScoreExport {
            passages_per_document: 1,
            all_passages: false,
            seed_document_passages: false,
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

    #[test]
    fn l1_rrf_uses_discarded_global_nominations_and_traces_survive_pagination() {
        let mut req = request(true);
        req.include_rrf_scores = true;
        req.tracing = true;
        let plan = CoordinatorPlan::new(req, 2).unwrap();
        let response = |segment, x, y, discarded_x| {
            let winner = proto::FusionCandidate {
                address: address(segment, 0),
                score: x,
                ..Default::default()
            };
            let discarded = proto::FusionCandidate {
                address: address(segment, 1),
                score: discarded_x,
                ..Default::default()
            };
            proto::SearchResponse {
                ranking_method: "formula_v1".into(),
                hits: vec![proto::SearchHit {
                    address: address(segment, 0),
                    score: x - 0.3 * y,
                    candidate_scores: Some(proto::CandidateScores {
                        document: HashMap::from([("x".into(), x), ("y".into(), y)]),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                trace: Some(proto::SearchTrace {
                    shards: vec![proto::ShardSearchTrace {
                        shard_id: segment.to_string(),
                        queries: vec![
                            proto::QueryTrace {
                                query_index: 0,
                                candidates: vec![winner.clone(), discarded],
                                ..Default::default()
                            },
                            // y is a backfilled feature for the winner; it contributes no organic RRF vote.
                            proto::QueryTrace {
                                query_index: 1,
                                candidates: Vec::new(),
                                ..Default::default()
                            },
                        ],
                        selected: vec![winner],
                        ..Default::default()
                    }],
                }),
                ..Default::default()
            }
        };
        let responses = vec![response(1, 9.0, 0.0, 20.0), response(2, 10.0, 10.0, 30.0)];
        let result = plan.finish(responses.clone()).unwrap();
        assert_eq!(key(result.hits[0].address.as_ref()).unwrap(), (1, 0));
        assert_eq!(result.hits[0].score, 9.0);
        assert_eq!(result.hits[0].rrf_score, Some(1.0 / 64.0));
        assert_eq!(
            result.hits[0].rrf_contributions,
            vec![proto::RrfContribution {
                query_index: 0,
                query_name: "x".into(),
                rank: 4,
                score: 1.0 / 64.0,
                ordinal: None,
            }]
        );
        assert!(result.hits[0].candidate_scores.is_none());
        assert!(result.fusion_candidates.is_empty());
        assert_eq!(
            result
                .trace
                .unwrap()
                .shards
                .iter()
                .map(|s| s.queries[0].candidates.len())
                .collect::<Vec<_>>(),
            vec![2, 2]
        );
        let mut incomplete = responses;
        incomplete[0].trace = None;
        assert!(
            plan.finish(incomplete)
                .unwrap_err()
                .message()
                .contains("omitted")
        );
    }

    #[test]
    fn tracing_ordinary_search_keeps_all_shard_candidates_after_final_offset() {
        let req = proto::SearchRequest {
            limit: 1,
            offset: 1,
            tracing: true,
            query: Some(proto::Query {
                query: Some(proto::query::Query::All(proto::AllQuery {})),
            }),
            ..Default::default()
        };
        let plan = CoordinatorPlan::new(req, 2).unwrap();
        assert_eq!(plan.shard_request.limit, 2);
        let responses = (1..=2)
            .map(|segment| proto::SearchResponse {
                hits: vec![proto::SearchHit {
                    address: address(segment, 0),
                    score: segment as f32,
                    ..Default::default()
                }],
                trace: Some(proto::SearchTrace {
                    shards: vec![proto::ShardSearchTrace {
                        queries: vec![proto::QueryTrace {
                            candidates: vec![proto::FusionCandidate {
                                address: address(segment, 0),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }],
                        ..Default::default()
                    }],
                }),
                ..Default::default()
            })
            .collect();
        let result = plan.finish(responses).unwrap();
        assert_eq!(key(result.hits[0].address.as_ref()).unwrap(), (1, 0));
        assert_eq!(result.trace.unwrap().shards.len(), 2);
    }

    #[test]
    fn requested_trace_requires_backend_support_and_stamps_routing_identity() {
        let mut response = proto::SearchResponse::default();
        assert!(stamp_trace(&mut response, "s", "b", true).is_err());
        response.trace = Some(proto::SearchTrace {
            shards: vec![proto::ShardSearchTrace::default()],
        });
        stamp_trace(&mut response, "s", "b", true).unwrap();
        let shard = &response.trace.unwrap().shards[0];
        assert_eq!(shard.backend_id, "b");
        assert_eq!(shard.shard_id, "s");
    }

    #[test]
    fn global_rrf_l1_keeps_a_winner_below_each_shards_original_formula_window() {
        let mut req = request(true);
        req.l1.as_mut().unwrap().formula = "0.0001 * x + rrf".into();
        req.include_rrf_scores = true;
        req.tracing = true;
        let plan = CoordinatorPlan::new(req, 2).unwrap();
        assert_eq!(
            plan.shard_request.limit, 4,
            "export every possible union member before global inference"
        );
        assert_eq!(plan.shard_request.l1.as_ref().unwrap().formula, "0");
        let response = |segment, documents: &[(u32, f32, f32)]| proto::SearchResponse {
            ranking_method: "formula_v1".into(),
            hits: documents
                .iter()
                .map(|&(doc, x, y)| proto::SearchHit {
                    address: address(segment, doc),
                    score: 0.0,
                    candidate_scores: Some(proto::CandidateScores {
                        document: HashMap::from([("x".into(), x), ("y".into(), y)]),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            trace: Some(proto::SearchTrace {
                shards: vec![proto::ShardSearchTrace {
                    queries: (0..2)
                        .map(|branch| proto::QueryTrace {
                            query_index: branch,
                            candidates: documents
                                .iter()
                                .map(|&(doc, x, y)| proto::FusionCandidate {
                                    address: address(segment, doc),
                                    score: if branch == 0 { x } else { y },
                                    ..Default::default()
                                })
                                .collect(),
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                }],
            }),
            ..Default::default()
        };
        // Doc (2,1) loses local raw L1 to (2,0), but wins the global formula.
        let responses = vec![
            response(1, &[(0, 98.0, 99.0)]),
            response(2, &[(0, 100.0, 1.0), (1, 99.0, 100.0)]),
        ];
        let result = plan.finish(responses.clone()).unwrap();
        assert_eq!(key(result.hits[0].address.as_ref()).unwrap(), (2, 1));
        assert_eq!(result.ranking_method, "formula_v1");
        let rrf = 1.0 / 62.0 + 1.0 / 61.0;
        assert_eq!(result.hits[0].rrf_score, Some(rrf));
        assert_eq!(
            result.hits[0].score,
            (99.0f64 * 0.0001 + f64::from(rrf)) as f32
        );
        let mut incomplete = responses;
        incomplete[1].hits.pop();
        assert!(
            plan.finish(incomplete)
                .unwrap_err()
                .message()
                .contains("complete")
        );
    }

    #[test]
    fn rrf_formula_retains_passages_that_lose_shard_local_l1_max() {
        let mut req = request(true);
        req.include_rrf_scores = true;
        let model = req.l1.as_mut().unwrap();
        model.formula = "x + 1000 * rrf".into();
        let Some(proto::query::Query::Fusion(fusion)) = req.query.as_mut().unwrap().query.as_mut()
        else {
            unreachable!()
        };
        for query in &mut fusion.queries {
            query.scope = proto::ScoreScope::Chunk as i32;
        }
        fusion.queries[0].score_only = true;
        let plan = CoordinatorPlan::new(req, 1).unwrap();
        assert_eq!(
            plan.shard_request
                .score_export
                .as_ref()
                .unwrap()
                .passages_per_document,
            65536
        );
        let response = proto::SearchResponse {
            ranking_method: "formula_v1".into(),
            hits: vec![proto::SearchHit {
                address: address(1, 0),
                score: 0.0,
                candidate_scores: Some(proto::CandidateScores {
                    document: HashMap::new(),
                    scored_passages: 2,
                    passages: vec![
                        proto::PassageScores {
                            ordinal: 0,
                            scores: HashMap::from([("x".into(), 2.0), ("y".into(), 1.0)]),
                            l1_score: Some(0.0),
                        },
                        proto::PassageScores {
                            ordinal: 1,
                            scores: HashMap::from([("x".into(), 1.9), ("y".into(), 2.0)]),
                            l1_score: Some(0.0),
                        },
                    ],
                }),
                ..Default::default()
            }],
            fusion_candidates: vec![proto::FusionCandidateList {
                query_index: 1,
                candidates: vec![proto::FusionCandidate {
                    address: address(1, 0),
                    score: 2.0,
                    ordinal_scores: vec![
                        proto::OrdinalScore {
                            ordinal: 0,
                            score: 1.0,
                        },
                        proto::OrdinalScore {
                            ordinal: 1,
                            score: 2.0,
                        },
                    ],
                }],
            }],
            ..Default::default()
        };
        let result = plan.finish(vec![response.clone()]).unwrap();
        assert_eq!(result.hits[0].ordinal_scores[0].ordinal, 1);
        assert_eq!(
            result.hits[0].score,
            (f64::from(1.9f32) + 1000.0 * f64::from(1.0f32 / 61.0)) as f32
        );
        let mut incomplete = response;
        incomplete.hits[0]
            .candidate_scores
            .as_mut()
            .unwrap()
            .passages
            .pop();
        assert!(
            plan.finish(vec![incomplete])
                .unwrap_err()
                .message()
                .contains("complete passage")
        );
    }
}
