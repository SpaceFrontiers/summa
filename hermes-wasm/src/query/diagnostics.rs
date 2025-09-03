//! Optional bounded wire diagnostics. Retrieval and RRF remain core-owned.
use super::{HitAddress, JsFusionQuery, JsQuery};
use hermes_core::query::{MAX_FUSION_CANDIDATE_SLOTS, MAX_FUSION_CHUNK_SLOTS, SearchResult};
use serde::Serialize;
use wasm_bindgen::JsValue;

#[derive(Serialize)]
pub(super) struct RrfContribution<'a> {
    pub query_name: &'a str,
    #[serde(flatten)]
    pub vote: hermes_core::query::RrfContribution,
}

#[derive(Serialize)]
pub(super) struct SearchTrace<'a> {
    pub shards: Vec<ShardTrace<'a>>,
}

#[derive(Serialize)]
pub(super) struct ShardTrace<'a> {
    shard_id: &'static str,
    backend_id: &'static str,
    index_name: &'a str,
    queries: Vec<QueryTrace<'a>>,
    pub selected: Vec<Candidate>,
    ranking_method: &'a str,
    truncated: bool,
}

#[derive(Serialize)]
struct QueryTrace<'a> {
    query_index: usize,
    query_name: &'a str,
    query: &'a JsQuery,
    scope: u32,
    score_only: bool,
    candidate_depth: usize,
    total_seen: u32,
    candidates: Vec<Candidate>,
}

#[derive(Serialize)]
pub(super) struct Candidate {
    address: HitAddress,
    score: f32,
    ordinal_scores: Vec<OrdinalScore>,
}

#[derive(Serialize)]
struct OrdinalScore {
    ordinal: u32,
    score: f32,
}

#[derive(Default)]
pub(super) struct Budget {
    rows: usize,
    ordinals: usize,
}

impl Budget {
    pub fn candidates(&mut self, hits: &[SearchResult]) -> Result<Vec<Candidate>, JsValue> {
        self.rows = self.rows.saturating_add(hits.len());
        self.ordinals = hits.iter().fold(self.ordinals, |sum, hit| {
            sum.saturating_add(
                hit.positions
                    .iter()
                    .fold(0usize, |sum, (_, positions)| {
                        sum.saturating_add(positions.len())
                    })
                    .max(1),
            )
        });
        if self.rows > MAX_FUSION_CANDIDATE_SLOTS || self.ordinals > MAX_FUSION_CHUNK_SLOTS {
            return Err(JsValue::from_str("trace candidate/ordinal budget exceeded"));
        }
        Ok(hits
            .iter()
            .map(|hit| Candidate {
                address: HitAddress {
                    segment_id: format!("{:032x}", hit.segment_id),
                    doc_id: hit.doc_id,
                },
                score: hit.score,
                ordinal_scores: hit
                    .positions
                    .iter()
                    .flat_map(|(_, positions)| {
                        positions.iter().map(|position| OrdinalScore {
                            ordinal: position.position,
                            score: position.score,
                        })
                    })
                    .collect(),
            })
            .collect())
    }
}

pub(super) fn fusion_trace<'a>(
    index_name: &'a str,
    fusion: &'a JsFusionQuery,
    lists: &[(Vec<SearchResult>, u32)],
    depth: usize,
    budget: &mut Budget,
) -> Result<SearchTrace<'a>, JsValue> {
    let queries = fusion
        .queries
        .iter()
        .zip(lists)
        .enumerate()
        .map(|(query_index, (branch, (hits, seen)))| {
            Ok(QueryTrace {
                query_index,
                query_name: &branch.name,
                query: &branch.query,
                scope: 0,
                score_only: false,
                candidate_depth: depth,
                total_seen: *seen,
                candidates: budget.candidates(hits)?,
            })
        })
        .collect::<Result<_, JsValue>>()?;
    Ok(trace(
        index_name,
        queries,
        fusion.method.as_deref().unwrap_or("rrf"),
    ))
}

pub(super) fn root_trace<'a>(
    index_name: &'a str,
    query: &'a JsQuery,
    hits: &[SearchResult],
    seen: u32,
    depth: usize,
    budget: &mut Budget,
) -> Result<SearchTrace<'a>, JsValue> {
    Ok(trace(
        index_name,
        vec![QueryTrace {
            query_index: 0,
            query_name: "",
            query,
            scope: 0,
            score_only: false,
            candidate_depth: depth,
            total_seen: seen,
            candidates: budget.candidates(hits)?,
        }],
        "",
    ))
}

fn trace<'a>(
    index_name: &'a str,
    queries: Vec<QueryTrace<'a>>,
    ranking_method: &'a str,
) -> SearchTrace<'a> {
    SearchTrace {
        shards: vec![ShardTrace {
            index_name,
            queries,
            ranking_method,
            shard_id: "",
            backend_id: "",
            truncated: false,
            selected: Vec::new(),
        }],
    }
}

/// Count JSON bytes without materializing another response buffer.
pub(super) fn check_encoded_size(value: &impl Serialize) -> Result<(), JsValue> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            if self.0 > 64 * 1024 * 1024 {
                return Err(std::io::Error::other("diagnostic response exceeds 64 MiB"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Counter(0), value).map_err(|e| JsValue::from_str(&e.to_string()))
}
