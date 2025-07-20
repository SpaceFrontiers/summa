//! Search service gRPC implementation

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use hermes_core::FieldValue as CoreFieldValue;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{Request, Response, Status};

use crate::converters::{convert_field_value, convert_query, convert_reranker, schema_to_sdl};
use crate::proto::search_service_server::SearchService;
use crate::proto::*;
use crate::registry::IndexRegistry;

const DEFAULT_SEARCH_LIMIT: usize = 10;
const MAX_SEARCH_LIMIT: usize = 10_000;
const MAX_SEARCH_WINDOW: usize = 50_000;
const MAX_CANDIDATE_LIMIT: usize = 50_000;
const MAX_FUSION_SUB_QUERIES: usize = hermes_core::query::MAX_FUSION_SUB_QUERIES;
const MAX_FUSION_CANDIDATE_SLOTS: usize = hermes_core::query::MAX_FUSION_CANDIDATE_SLOTS;

// The transport's 4 MiB decode limit bounds wire bytes, not decoded object
// count or downstream expansion. Empty protobuf messages and strings are only
// a few bytes on the wire, so keep independent structural budgets here.
const MAX_QUERY_DEPTH: usize = 32;
const MAX_QUERY_NODES: usize = 256;
const MAX_QUERY_CLAUSES: usize = 512;
const MAX_BOOLEAN_CLAUSES: usize = 128;
const MAX_QUERY_TEXT_BYTES: usize = 64 * 1024;
const MAX_FIELD_NAME_BYTES: usize = 255;
const MAX_INDEX_NAME_BYTES: usize = 255;
const MAX_DENSE_QUERY_DIMS: usize = 65_536;
const MAX_SPARSE_QUERY_DIMS: usize = 4_096;
const MAX_BINARY_QUERY_BYTES: usize = 256 * 1024;
const MAX_TOTAL_QUERY_VECTOR_BYTES: usize = 1024 * 1024;
const MAX_FIELDS_TO_LOAD: usize = 64;
const MAX_FIELDS_TO_LOAD_NAME_BYTES: usize = 16 * 1024;

// Leave headroom below tonic's 64 MiB response limit for compression and
// framing, and also cap estimated retained heap while the response is built.
const MAX_SEARCH_RESPONSE_BYTES: usize = 48 * 1024 * 1024;
const UNKNOWN_INDEX_LABEL: &str = "unknown";

#[derive(Default)]
struct QueryShapeBudget {
    nodes: usize,
    clauses: usize,
    text_bytes: usize,
    vector_bytes: usize,
}

impl QueryShapeBudget {
    fn add_limited(
        current: &mut usize,
        amount: usize,
        maximum: usize,
        description: &str,
    ) -> Result<(), Status> {
        let next = current
            .checked_add(amount)
            .ok_or_else(|| Status::invalid_argument(format!("{description} budget overflows")))?;
        if next > maximum {
            return Err(Status::invalid_argument(format!(
                "{description} must not exceed {maximum} (got {next})"
            )));
        }
        *current = next;
        Ok(())
    }

    fn add_node(&mut self, depth: usize) -> Result<(), Status> {
        if depth > MAX_QUERY_DEPTH {
            return Err(Status::invalid_argument(format!(
                "Query nesting depth must not exceed {MAX_QUERY_DEPTH}"
            )));
        }
        Self::add_limited(&mut self.nodes, 1, MAX_QUERY_NODES, "Query node count")
    }

    fn add_clauses(&mut self, clauses: usize) -> Result<(), Status> {
        Self::add_limited(
            &mut self.clauses,
            clauses,
            MAX_QUERY_CLAUSES,
            "Query clause count",
        )
    }

    fn add_field_name(&mut self, name: &str, description: &str) -> Result<(), Status> {
        if name.len() > MAX_FIELD_NAME_BYTES {
            return Err(Status::invalid_argument(format!(
                "{description} must not exceed {MAX_FIELD_NAME_BYTES} bytes"
            )));
        }
        self.add_text(name.len())
    }

    fn add_text(&mut self, bytes: usize) -> Result<(), Status> {
        Self::add_limited(
            &mut self.text_bytes,
            bytes,
            MAX_QUERY_TEXT_BYTES,
            "Aggregate query text bytes",
        )
    }

    fn add_vector(
        &mut self,
        description: &str,
        elements: usize,
        element_bytes: usize,
        maximum_elements: usize,
    ) -> Result<(), Status> {
        if elements > maximum_elements {
            return Err(Status::invalid_argument(format!(
                "{description} must not exceed {maximum_elements} elements (got {elements})"
            )));
        }
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| Status::invalid_argument(format!("{description} size overflows")))?;
        Self::add_limited(
            &mut self.vector_bytes,
            bytes,
            MAX_TOTAL_QUERY_VECTOR_BYTES,
            "Aggregate query vector bytes",
        )
    }
}

/// Validate decoded request structure before acquiring scarce search capacity
/// or recursively converting the protobuf query tree.
fn validate_search_request_shape(req: &SearchRequest, root: &Query) -> Result<(), Status> {
    if req.index_name.is_empty() || req.index_name.len() > MAX_INDEX_NAME_BYTES {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.index_name must contain 1..={MAX_INDEX_NAME_BYTES} bytes"
        )));
    }
    if req.fields_to_load.len() > MAX_FIELDS_TO_LOAD {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.fields_to_load supports at most {MAX_FIELDS_TO_LOAD} names (got {})",
            req.fields_to_load.len()
        )));
    }
    let mut selected_name_bytes = 0usize;
    for (index, name) in req.fields_to_load.iter().enumerate() {
        if name.len() > MAX_FIELD_NAME_BYTES {
            return Err(Status::invalid_argument(format!(
                "SearchRequest.fields_to_load[{index}] must not exceed {MAX_FIELD_NAME_BYTES} bytes"
            )));
        }
        selected_name_bytes = selected_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| Status::invalid_argument("Field selection byte count overflows"))?;
    }
    if selected_name_bytes > MAX_FIELDS_TO_LOAD_NAME_BYTES {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.fields_to_load names must total at most \
             {MAX_FIELDS_TO_LOAD_NAME_BYTES} bytes (got {selected_name_bytes})"
        )));
    }

    let mut budget = QueryShapeBudget::default();
    let mut stack = vec![(root, 1usize)];
    while let Some((query, depth)) = stack.pop() {
        budget.add_node(depth)?;
        let query = query
            .query
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("Query type is required"))?;
        match query {
            query::Query::Term(term) => {
                budget.add_field_name(&term.field, "TermQuery.field")?;
                budget.add_text(term.term.len())?;
            }
            query::Query::Boolean(boolean) => {
                let clauses = boolean
                    .must
                    .len()
                    .checked_add(boolean.should.len())
                    .and_then(|count| count.checked_add(boolean.must_not.len()))
                    .ok_or_else(|| Status::invalid_argument("Boolean clause count overflows"))?;
                if clauses > MAX_BOOLEAN_CLAUSES {
                    return Err(Status::invalid_argument(format!(
                        "Each BooleanQuery supports at most {MAX_BOOLEAN_CLAUSES} clauses \
                         (got {clauses})"
                    )));
                }
                budget.add_clauses(clauses)?;
                stack.extend(
                    boolean
                        .must
                        .iter()
                        .chain(&boolean.should)
                        .chain(&boolean.must_not)
                        .map(|child| (child, depth + 1)),
                );
            }
            query::Query::Boost(boost) => {
                let child = boost.query.as_deref().ok_or_else(|| {
                    Status::invalid_argument("BoostQuery requires an inner query")
                })?;
                budget.add_clauses(1)?;
                stack.push((child, depth + 1));
            }
            query::Query::All(_) => {}
            query::Query::SparseVector(sparse) => {
                budget.add_field_name(&sparse.field, "SparseVectorQuery.field")?;
                budget.add_text(sparse.text.len())?;
                budget.add_vector(
                    "SparseVectorQuery.indices",
                    sparse.indices.len(),
                    std::mem::size_of::<u32>(),
                    MAX_SPARSE_QUERY_DIMS,
                )?;
                budget.add_vector(
                    "SparseVectorQuery.values",
                    sparse.values.len(),
                    std::mem::size_of::<f32>(),
                    MAX_SPARSE_QUERY_DIMS,
                )?;
            }
            query::Query::DenseVector(dense) => {
                budget.add_field_name(&dense.field, "DenseVectorQuery.field")?;
                budget.add_vector(
                    "DenseVectorQuery.vector",
                    dense.vector.len(),
                    std::mem::size_of::<f32>(),
                    MAX_DENSE_QUERY_DIMS,
                )?;
            }
            query::Query::Match(match_query) => {
                budget.add_field_name(&match_query.field, "MatchQuery.field")?;
                budget.add_text(match_query.text.len())?;
            }
            query::Query::Range(range) => {
                budget.add_field_name(&range.field, "RangeQuery.field")?;
            }
            query::Query::Prefix(prefix) => {
                budget.add_field_name(&prefix.field, "PrefixQuery.field")?;
                budget.add_text(prefix.prefix.len())?;
            }
            query::Query::BinaryDenseVector(binary) => {
                budget.add_field_name(&binary.field, "BinaryDenseVectorQuery.field")?;
                budget.add_vector(
                    "BinaryDenseVectorQuery.vector",
                    binary.vector.len(),
                    1,
                    MAX_BINARY_QUERY_BYTES,
                )?;
            }
            query::Query::Fusion(fusion) => {
                if depth != 1 {
                    return Err(Status::invalid_argument(
                        "FusionQuery is only supported at the top level",
                    ));
                }
                if fusion.queries.len() > MAX_FUSION_SUB_QUERIES {
                    return Err(Status::invalid_argument(format!(
                        "FusionQuery supports at most {MAX_FUSION_SUB_QUERIES} sub-queries \
                         (got {})",
                        fusion.queries.len()
                    )));
                }
                budget.add_clauses(fusion.queries.len())?;
                for weighted in &fusion.queries {
                    let child = weighted
                        .query
                        .as_ref()
                        .ok_or_else(|| Status::invalid_argument("Fusion sub-query is missing"))?;
                    stack.push((child, depth + 1));
                }
            }
        }
    }

    if let Some(reranker) = &req.reranker {
        budget.add_field_name(&reranker.field, "Reranker.field")?;
        budget.add_vector(
            "Reranker.vector",
            reranker.vector.len(),
            std::mem::size_of::<f32>(),
            MAX_DENSE_QUERY_DIMS,
        )?;
        budget.add_vector(
            "Reranker.binary_vector",
            reranker.binary_vector.len(),
            1,
            MAX_BINARY_QUERY_BYTES,
        )?;
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedField {
    id: hermes_core::dsl::Field,
    name: String,
}

/// Resolve and deduplicate field names once. Unknown names retain the existing
/// API behavior (they are ignored), while aliases/duplicates cannot multiply
/// per-hit work or HashMap capacity.
fn resolve_requested_fields(
    schema: &hermes_core::Schema,
    requested: &[String],
) -> Vec<ResolvedField> {
    let mut seen = rustc_hash::FxHashSet::default();
    let mut resolved = Vec::with_capacity(requested.len().min(schema.num_fields()));
    for name in requested {
        let Some(id) = schema.get_field(name) else {
            continue;
        };
        if !seen.insert(id.0) {
            continue;
        }
        let canonical_name = schema.get_field_name(id).unwrap_or(name).to_owned();
        resolved.push(ResolvedField {
            id,
            name: canonical_name,
        });
    }
    resolved
}

#[derive(Debug)]
struct SearchResponseBudget {
    retained_bytes: usize,
    encoded_bytes: usize,
    maximum: usize,
}

impl SearchResponseBudget {
    fn new() -> Self {
        Self::with_maximum(MAX_SEARCH_RESPONSE_BYTES)
    }

    fn with_maximum(maximum: usize) -> Self {
        Self {
            retained_bytes: 0,
            encoded_bytes: 0,
            maximum,
        }
    }

    fn reserve(counter: &mut usize, bytes: usize, maximum: usize) -> Result<(), Status> {
        let next = counter.checked_add(bytes).ok_or_else(|| {
            Status::resource_exhausted("Search response size accounting overflowed")
        })?;
        if next > maximum {
            return Err(Status::resource_exhausted(format!(
                "Search response exceeds the {maximum}-byte hydration budget; \
                 request fewer hits or fields"
            )));
        }
        *counter = next;
        Ok(())
    }

    fn reserve_retained(&mut self, bytes: usize) -> Result<(), Status> {
        Self::reserve(&mut self.retained_bytes, bytes, self.maximum)
    }

    fn reserve_hit(&mut self, hit: &SearchHit) -> Result<(), Status> {
        let payload = prost::Message::encoded_len(hit);
        let framed = payload
            .checked_add(protobuf_varint_len(payload))
            .and_then(|bytes| bytes.checked_add(1))
            .ok_or_else(|| Status::resource_exhausted("Search response encoded size overflowed"))?;
        Self::reserve(&mut self.encoded_bytes, framed, self.maximum)
    }
}

fn protobuf_varint_len(mut value: usize) -> usize {
    let mut bytes = 1;
    while value >= 0x80 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

#[derive(Default)]
struct CountingWriter(usize);

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("serialized JSON size overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Conservative retained-heap estimate charged before cloning a stored value
/// into its protobuf counterpart. Doubling the value object accounts for Vec
/// growth slack when a field has many tiny values.
fn retained_field_value_bytes(value: &CoreFieldValue) -> Result<usize, Status> {
    let payload = match value {
        CoreFieldValue::Text(text) => text.len(),
        CoreFieldValue::U64(_) | CoreFieldValue::I64(_) | CoreFieldValue::F64(_) => 0,
        CoreFieldValue::Bytes(bytes) | CoreFieldValue::BinaryDenseVector(bytes) => bytes.len(),
        CoreFieldValue::SparseVector(entries) => entries
            .len()
            .checked_mul(std::mem::size_of::<u32>() + std::mem::size_of::<f32>())
            .ok_or_else(|| Status::resource_exhausted("Sparse field size overflowed"))?,
        CoreFieldValue::DenseVector(values) => values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Status::resource_exhausted("Dense field size overflowed"))?,
        CoreFieldValue::Json(json) => {
            let mut writer = CountingWriter::default();
            serde_json::to_writer(&mut writer, json).map_err(|error| {
                Status::internal(format!("Failed to size JSON response field: {error}"))
            })?;
            writer.0
        }
    };
    std::mem::size_of::<FieldValue>()
        .saturating_mul(2)
        .checked_add(payload)
        .ok_or_else(|| Status::resource_exhausted("Response field size overflowed"))
}

fn retained_hit_base_bytes(ordinal_count: usize) -> Result<usize, Status> {
    ordinal_count
        .checked_mul(std::mem::size_of::<OrdinalScore>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<SearchHit>()))
        .and_then(|bytes| bytes.checked_add(32)) // segment-id String backing bytes
        .ok_or_else(|| Status::resource_exhausted("Search hit size overflowed"))
}

fn retained_field_entry_bytes(name: &str) -> usize {
    // HashMap growth keeps spare buckets. Charge two entries so many tiny
    // fields cannot evade the payload budget through container overhead.
    (std::mem::size_of::<String>() + std::mem::size_of::<FieldValueList>())
        .saturating_mul(2)
        .saturating_add(name.len())
}

fn canonical_metric_index_label(schema: &hermes_core::Schema) -> &str {
    schema.index_label()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SearchBudget {
    /// Number of results returned to the caller.
    final_limit: usize,
    /// Number of leading results skipped for pagination.
    offset: usize,
    /// Number of ranked results required before applying the offset.
    search_limit: usize,
    rerank_l1_limit: Option<usize>,
    fusion_fetch_limit: Option<usize>,
}

fn bounded_limit(name: &str, value: u32, default: usize, max: usize) -> Result<usize, Status> {
    let value = if value == 0 { default } else { value as usize };
    if value > max {
        return Err(Status::invalid_argument(format!(
            "{name} must not exceed {max} (got {value})"
        )));
    }
    Ok(value)
}

fn try_acquire_search_permit(permits: &Arc<Semaphore>) -> Result<OwnedSemaphorePermit, Status> {
    Arc::clone(permits)
        .try_acquire_owned()
        .map_err(|_| Status::resource_exhausted("Search capacity is full; retry with backoff"))
}

/// Validate all request-controlled result and candidate depths before opening
/// an index, constructing queries, or allocating candidate lists.
fn validate_search_budget(req: &SearchRequest) -> Result<SearchBudget, Status> {
    let query = req
        .query
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("Query is required"))?;
    validate_search_request_shape(req, query)?;
    let final_limit = bounded_limit(
        "SearchRequest.limit",
        req.limit,
        DEFAULT_SEARCH_LIMIT,
        MAX_SEARCH_LIMIT,
    )?;
    let offset = req.offset as usize;
    let search_limit = offset
        .checked_add(final_limit)
        .ok_or_else(|| Status::invalid_argument("Search result window is too large"))?;
    if search_limit > MAX_SEARCH_WINDOW {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.offset + limit must not exceed {MAX_SEARCH_WINDOW} \
             (got {offset} + {final_limit} = {search_limit})"
        )));
    }
    let max_candidate_limit =
        hermes_core::query::max_candidate_limit(search_limit).min(MAX_CANDIDATE_LIMIT);

    let rerank_l1_limit = match &req.reranker {
        Some(reranker) if reranker.limit > 0 => Some(bounded_limit(
            "Reranker.limit",
            reranker.limit,
            search_limit,
            max_candidate_limit,
        )?),
        Some(_) => Some(max_candidate_limit),
        None => None,
    };
    if req
        .reranker
        .as_ref()
        .is_some_and(|reranker| reranker.limit > 0)
        && rerank_l1_limit.is_some_and(|l1_limit| l1_limit < search_limit)
    {
        return Err(Status::invalid_argument(format!(
            "Reranker.limit must be at least offset + limit ({search_limit})"
        )));
    }

    let fusion_fetch_limit = if let Some(query::Query::Fusion(fusion)) = &query.query {
        if fusion.queries.is_empty() {
            return Err(Status::invalid_argument(
                "FusionQuery requires at least one sub-query",
            ));
        }
        if fusion.queries.len() > MAX_FUSION_SUB_QUERIES {
            return Err(Status::invalid_argument(format!(
                "FusionQuery supports at most {MAX_FUSION_SUB_QUERIES} sub-queries (got {})",
                fusion.queries.len()
            )));
        }
        if !fusion.rrf_k.is_finite() || fusion.rrf_k < 0.0 {
            return Err(Status::invalid_argument(format!(
                "FusionQuery.rrf_k must be finite and non-negative (got {})",
                fusion.rrf_k
            )));
        }
        for (index, weighted) in fusion.queries.iter().enumerate() {
            if !weighted.weight.is_finite() || weighted.weight < 0.0 {
                return Err(Status::invalid_argument(format!(
                    "FusionQuery.queries[{index}].weight must be finite and non-negative \
                     (got {})",
                    weighted.weight
                )));
            }
        }

        let fused_limit = rerank_l1_limit.unwrap_or(search_limit);
        let fetch_limit = if fusion.fetch_limit > 0 {
            bounded_limit(
                "FusionQuery.fetch_limit",
                fusion.fetch_limit,
                fused_limit,
                max_candidate_limit,
            )?
        } else if rerank_l1_limit.is_some() {
            // The rerank pool is already the shared candidate budget.
            fused_limit
        } else {
            max_candidate_limit
        };
        if fetch_limit < fused_limit {
            return Err(Status::invalid_argument(format!(
                "FusionQuery.fetch_limit must be at least the fused result window ({fused_limit})"
            )));
        }

        let candidate_slots = fetch_limit
            .checked_mul(fusion.queries.len())
            .ok_or_else(|| Status::invalid_argument("Fusion candidate budget is too large"))?;
        if candidate_slots > MAX_FUSION_CANDIDATE_SLOTS {
            return Err(Status::invalid_argument(format!(
                "FusionQuery candidate budget must not exceed {MAX_FUSION_CANDIDATE_SLOTS} \
                 (fetch_limit {fetch_limit} x {} sub-queries = {candidate_slots})",
                fusion.queries.len()
            )));
        }
        Some(fetch_limit)
    } else {
        None
    };

    Ok(SearchBudget {
        final_limit,
        offset,
        search_limit,
        rerank_l1_limit,
        fusion_fetch_limit,
    })
}

/// Search service implementation
pub struct SearchServiceImpl {
    pub registry: Arc<IndexRegistry>,
    search_permits: Arc<Semaphore>,
}

impl SearchServiceImpl {
    pub fn new(registry: Arc<IndexRegistry>, max_concurrent_searches: usize) -> Self {
        assert!(
            max_concurrent_searches > 0,
            "max_concurrent_searches must be greater than zero"
        );
        Self {
            registry,
            search_permits: Arc::new(Semaphore::new(max_concurrent_searches)),
        }
    }
}

#[tonic::async_trait]
impl SearchService for SearchServiceImpl {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let metric_index = std::sync::OnceLock::new();
        let t = std::time::Instant::now();
        let result = async {
        let budget = validate_search_budget(&req)?;

        // Bound expensive pipelines across all HTTP/2 connections without an
        // unbounded waiter queue retaining decoded requests under overload.
        // Dropping the owned permit on completion/error/cancellation is safe.
        let _search_permit = match try_acquire_search_permit(&self.search_permits) {
            Ok(permit) => permit,
            Err(status) => {
                metrics::counter!(
                    "hermes_search_admission_rejected_total",
                    "index" => UNKNOWN_INDEX_LABEL,
                )
                .increment(1);
                return Err(status);
            }
        };

        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let _ = metric_index.set(canonical_metric_index_label(index.schema()).to_owned());
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        let query = req
            .query
            .ok_or_else(|| Status::invalid_argument("Query is required"))?;

        // Rank enough results to cover the requested page, then apply the
        // offset only after fusion/reranking so pagination preserves ranking.
        let limit = budget.search_limit;

        // Optional L2 reranker config; the L1 pool it consumes is either a
        // single query's results or the fused union of sub-query results.
        let rerank_setup = match &req.reranker {
            Some(reranker) => {
                let config = convert_reranker(reranker, reader.schema())
                    .map_err(|e| Status::invalid_argument(format!("Invalid reranker: {}", e)))?;
                let l1_limit = budget
                    .rerank_l1_limit
                    .ok_or_else(|| Status::internal("Missing validated reranker budget"))?;
                Some((config, l1_limit))
            }
            None => None,
        };

        // ── Phase 1: L1 search ──────────────────────────────────────────────
        let start = Instant::now();
        let t_search = Instant::now();
        let query_desc;
        let (results, total_seen, rerank_config) =
            if let Some(crate::proto::query::Query::Fusion(fusion)) = &query.query {
                // Fusion: run each sub-query independently and fuse the ranked
                // lists (union). Handled here rather than in convert_query
                // because fusion is a searcher-level operation.
                let mut sub_queries = Vec::with_capacity(fusion.queries.len());
                for weighted in &fusion.queries {
                    let sub = weighted
                        .query
                        .as_ref()
                        .ok_or_else(|| Status::invalid_argument("Fusion sub-query is missing"))?;
                    let core = convert_query(
                        sub,
                        reader.schema(),
                        Some(searcher.global_stats()),
                        Some(index.directory().root()),
                    )
                    .map_err(|e| {
                        Status::invalid_argument(format!("Invalid fusion sub-query: {}", e))
                    })?;
                    let weight = if weighted.weight > 0.0 {
                        weighted.weight
                    } else {
                        1.0
                    };
                    sub_queries.push((core, weight));
                }
                let method = match fusion.method() {
                    crate::proto::FusionMethod::FusionRrf => {
                        hermes_core::query::FusionMethod::Rrf {
                            k: if fusion.rrf_k > 0.0 {
                                fusion.rrf_k
                            } else {
                                hermes_core::query::DEFAULT_RRF_K
                            },
                        }
                    }
                    crate::proto::FusionMethod::FusionNormalizedWeightedSum => {
                        hermes_core::query::FusionMethod::NormalizedWeightedSum
                    }
                };

                // With a reranker, the fused list is the L1 candidate pool.
                let fused_limit = rerank_setup.as_ref().map_or(limit, |&(_, l1)| l1);
                let fetch_limit = budget
                    .fusion_fetch_limit
                    .ok_or_else(|| Status::internal("Missing validated fusion budget"))?;

                // Chunk combiner for fused per-ordinal scores. Unset (0) maps
                // to Max — LogSumExp is unsuitable at RRF score magnitudes.
                let combiner = match fusion.combiner {
                    0 => hermes_core::query::MultiValueCombiner::Max,
                    c => crate::converters::convert_fusion_combiner(c),
                };

                query_desc = format!(
                    "fusion of {} sub-queries (method={:?}, fetch={})",
                    sub_queries.len(),
                    method,
                    fetch_limit
                );
                log::info!(
                    "search: index={}, limit={}, query={}",
                    req.index_name,
                    req.limit,
                    query_desc
                );

                let query_refs: Vec<(&dyn hermes_core::query::Query, f32)> = sub_queries
                    .iter()
                    .map(|(query, weight)| (query.as_ref(), *weight))
                    .collect();
                let (fused, seen) = searcher
                    .search_fused_with_count(
                        &query_refs,
                        fetch_limit,
                        fused_limit,
                        method,
                        combiner,
                    )
                    .await
                    .map_err(crate::error::hermes_error_to_status)?;
                let rerank_config = rerank_setup.map(|(config, _)| (config, limit));
                (fused, seen, rerank_config)
            } else {
                let core_query = convert_query(
                    &query,
                    reader.schema(),
                    Some(searcher.global_stats()),
                    Some(index.directory().root()),
                )
                .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

                query_desc = core_query.to_string();
                log::info!(
                    "search: index={}, limit={}, query={}",
                    req.index_name,
                    req.limit,
                    query_desc
                );

                if let Some((config, l1_limit)) = rerank_setup {
                    let (candidates, seen) = searcher
                        .search_with_count(core_query.as_ref(), l1_limit)
                        .await
                        .map_err(crate::error::hermes_error_to_status)?;
                    (candidates, seen, Some((config, limit)))
                } else {
                    let (results, seen) = searcher
                        .search_with_positions(core_query.as_ref(), limit)
                        .await
                        .map_err(crate::error::hermes_error_to_status)?;
                    (results, seen, None)
                }
            };
        let search_us = t_search.elapsed().as_micros() as u64;

        // ── Phase 2: L2 reranking (optional) ────────────────────────────────
        let t_rerank = Instant::now();
        let results = if let Some((config, final_limit)) = rerank_config {
            hermes_core::query::rerank(&searcher, &results, &config, final_limit)
                .await
                .map_err(crate::error::hermes_error_to_status)?
        } else {
            results
        };
        let results: Vec<_> = results
            .into_iter()
            .skip(budget.offset)
            .take(budget.final_limit)
            .collect();
        let rerank_us = t_rerank.elapsed().as_micros() as u64;

        // ── Phase 3: Document field loading ─────────────────────────────────
        let t_load = Instant::now();

        // Resolve names once and iterate only canonical, unique fields per hit.
        // The request-shape validator has already bounded the raw name list.
        let requested_fields =
            resolve_requested_fields(searcher.schema(), &req.fields_to_load);
        let requested_field_ids: Option<rustc_hash::FxHashSet<u32>> =
            (!requested_fields.is_empty()).then(|| {
                requested_fields
                    .iter()
                    .map(|requested| requested.id.0)
                    .collect()
            });

        // Debug: detect duplicate doc_ids across results (only in debug builds)
        #[cfg(debug_assertions)]
        {
            let mut seen: rustc_hash::FxHashMap<(u128, u32), usize> =
                rustc_hash::FxHashMap::default();
            for (i, r) in results.iter().enumerate() {
                if let Some(prev) = seen.insert((r.segment_id, r.doc_id), i) {
                    log::warn!(
                        "Duplicate doc_id in results: seg={:032x} doc={} at positions {} and {}, \
                         scores={:.4}/{:.4}, ordinals={:?}/{:?}",
                        r.segment_id,
                        r.doc_id,
                        prev,
                        i,
                        results[prev].score,
                        r.score,
                        results[prev].positions,
                        r.positions,
                    );
                }
            }
        }

        let mut response_budget = SearchResponseBudget::new();
        let mut hits = Vec::with_capacity(results.len());
        for result in results {
            // Convert ordinal scores before hydration so their retained memory
            // is charged before reading potentially large stored fields.
            let ordinal_scores: Vec<OrdinalScore> = result
                .positions
                .iter()
                .flat_map(|(_, scored_positions)| {
                    scored_positions.iter().map(|sp| OrdinalScore {
                        ordinal: sp.position, // vector position contains the ordinal
                        score: sp.score,
                    })
                })
                .collect();
            response_budget.reserve_retained(retained_hit_base_bytes(ordinal_scores.len())?)?;

            // Allocate map buckets only for fields that actually have values.
            // Pre-sizing every hit from raw request count was an OOM multiplier.
            let mut fields: HashMap<String, FieldValueList> = HashMap::new();

            if !requested_fields.is_empty() {
                let doc = searcher
                    .get_document_with_fields(
                        &hermes_core::query::DocAddress::new(result.segment_id, result.doc_id),
                        requested_field_ids.as_ref(),
                    )
                    .await
                    .map_err(crate::error::hermes_error_to_status)?;

                if let Some(doc) = doc {
                    for requested in &requested_fields {
                        let mut values = Vec::new();
                        for value in doc.get_all(requested.id) {
                            // Charge retained payload before cloning it into
                            // the response, so a single oversized value fails
                            // without first doubling its memory footprint.
                            response_budget
                                .reserve_retained(retained_field_value_bytes(value)?)?;
                            values.push(convert_field_value(value));
                        }
                        if !values.is_empty() {
                            response_budget.reserve_retained(retained_field_entry_bytes(
                                &requested.name,
                            ))?;
                            fields.insert(
                                requested.name.clone(),
                                FieldValueList { values },
                            );
                        }
                    }
                }
            }

            let hit = SearchHit {
                address: Some(DocAddress {
                    segment_id: format!("{:032x}", result.segment_id),
                    doc_id: result.doc_id,
                }),
                score: result.score,
                fields,
                ordinal_scores,
            };
            response_budget.reserve_hit(&hit)?;
            hits.push(hit);
        }
        let load_us = t_load.elapsed().as_micros() as u64;

        let total_us = start.elapsed().as_micros() as u64;
        let took_ms = total_us / 1000;

        if took_ms > 1000 {
            log::warn!(
                "slow query: index={}, took={}ms (search={}us, rerank={}us, load={}us), hits={}, total_seen={}, query={}",
                req.index_name,
                took_ms,
                search_us,
                rerank_us,
                load_us,
                hits.len(),
                total_seen,
                query_desc
            );
        }

        // total_seen = number of documents that were actually scored across all segments
        Ok(Response::new(SearchResponse {
            hits,
            total_hits: total_seen as u64,
            took_ms,
            timings: Some(SearchTimings {
                search_us,
                rerank_us,
                load_us,
                total_us,
            }),
        }))
            }
        .await;
        let status = if result.is_ok() { "ok" } else { "error" };
        let metric_index = metric_index
            .get()
            .cloned()
            .unwrap_or_else(|| UNKNOWN_INDEX_LABEL.to_owned());
        metrics::histogram!(
            "hermes_search_duration_seconds",
            "index" => metric_index.clone(),
            "status" => status,
        )
        .record(t.elapsed().as_secs_f64());
        metrics::counter!(
            "hermes_search_requests_total",
            "index" => metric_index,
            "status" => status,
        )
        .increment(1);
        result
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        let addr = req
            .address
            .ok_or_else(|| Status::invalid_argument("address is required"))?;
        let segment_id = u128::from_str_radix(&addr.segment_id, 16).map_err(|_| {
            Status::invalid_argument(format!("Invalid segment_id: {}", addr.segment_id))
        })?;
        let doc = searcher
            .doc(segment_id, addr.doc_id)
            .await
            .map_err(crate::error::hermes_error_to_status)?
            .ok_or_else(|| Status::not_found("Document not found"))?;

        let mut fields: HashMap<String, FieldValueList> = HashMap::new();
        for (field, value) in doc.field_values() {
            if let Some(entry) = index.schema().get_field_entry(*field) {
                fields
                    .entry(entry.name.clone())
                    .or_insert_with(|| FieldValueList { values: Vec::new() })
                    .values
                    .push(convert_field_value(value));
            }
        }

        Ok(Response::new(GetDocumentResponse { fields }))
    }

    async fn get_index_info(
        &self,
        request: Request<GetIndexInfoRequest>,
    ) -> Result<Response<GetIndexInfoResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        // Convert schema to SDL string
        let schema_str = schema_to_sdl(index.schema());

        // Collect memory stats from segment readers
        let mut total_term_dict_cache = 0u64;
        let mut total_store_cache = 0u64;
        let mut total_sparse_index = 0u64;
        let mut total_dense_index = 0u64;

        for segment in searcher.segment_readers() {
            let stats = segment.memory_stats();
            total_term_dict_cache += stats.term_dict_cache_bytes as u64;
            total_store_cache += stats.store_cache_bytes as u64;
            total_sparse_index += stats.sparse_index_bytes as u64;
            total_dense_index += stats.dense_index_bytes as u64;
        }

        let segment_reader_stats = SegmentReaderStats {
            total_bytes: total_term_dict_cache
                + total_store_cache
                + total_sparse_index
                + total_dense_index,
            term_dict_cache_bytes: total_term_dict_cache,
            store_cache_bytes: total_store_cache,
            sparse_index_bytes: total_sparse_index,
            dense_index_bytes: total_dense_index,
            num_segments_loaded: searcher.segment_readers().len() as u32,
        };

        let memory_stats = MemoryStats {
            total_bytes: segment_reader_stats.total_bytes,
            indexing_buffer: None, // Writer stats not available from reader
            segment_reader: Some(segment_reader_stats),
        };

        // Collect per-field vector statistics across all segments
        let schema = index.schema();
        let mut dense_totals: HashMap<u32, u64> = HashMap::new();
        let mut sparse_totals: HashMap<u32, u64> = HashMap::new();
        let mut sparse_postings: HashMap<u32, u64> = HashMap::new();
        let mut dense_dims: HashMap<u32, u32> = HashMap::new();
        let mut sparse_dims: HashMap<u32, u32> = HashMap::new();

        for segment in searcher.segment_readers() {
            for (&field_id, flat) in segment.flat_vectors() {
                *dense_totals.entry(field_id).or_default() += flat.num_vectors as u64;
                dense_dims.entry(field_id).or_insert(flat.dim as u32);
            }
            for (&field_id, sparse_idx) in segment.sparse_indexes() {
                *sparse_totals.entry(field_id).or_default() += sparse_idx.total_vectors as u64;
                *sparse_postings.entry(field_id).or_default() += sparse_idx.total_postings();
                sparse_dims
                    .entry(field_id)
                    .or_insert(sparse_idx.num_dimensions() as u32);
            }
            for (&field_id, bmp_idx) in segment.bmp_indexes() {
                *sparse_totals.entry(field_id).or_default() += bmp_idx.total_vectors as u64;
                *sparse_postings.entry(field_id).or_default() += bmp_idx.total_postings();
                sparse_dims.entry(field_id).or_insert(bmp_idx.dims());
            }
        }

        let mut vector_stats = Vec::new();
        for (field_id, total) in &dense_totals {
            let name = schema
                .get_field_name(hermes_core::dsl::Field(*field_id))
                .unwrap_or("unknown")
                .to_string();
            vector_stats.push(VectorFieldStats {
                field_name: name,
                vector_type: "dense".to_string(),
                total_vectors: *total,
                dimension: dense_dims.get(field_id).copied().unwrap_or(0),
                avg_terms_per_vector: 0.0,
            });
        }
        for (field_id, total) in &sparse_totals {
            let name = schema
                .get_field_name(hermes_core::dsl::Field(*field_id))
                .unwrap_or("unknown")
                .to_string();
            let postings = sparse_postings.get(field_id).copied().unwrap_or(0);
            let avg_terms_per_vector = if *total > 0 {
                postings as f32 / *total as f32
            } else {
                0.0
            };
            vector_stats.push(VectorFieldStats {
                field_name: name,
                vector_type: "sparse".to_string(),
                total_vectors: *total,
                dimension: sparse_dims.get(field_id).copied().unwrap_or(0),
                avg_terms_per_vector,
            });
        }
        vector_stats.sort_by(|a, b| a.field_name.cmp(&b.field_name));

        Ok(Response::new(GetIndexInfoResponse {
            index_name: req.index_name,
            num_docs: searcher.num_docs(),
            num_segments: searcher.segment_readers().len() as u32,
            schema: schema_str,
            memory_stats: Some(memory_stats),
            vector_stats,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    fn all_query() -> Query {
        Query {
            query: Some(query::Query::All(AllQuery::default())),
        }
    }

    fn ordinary_request() -> SearchRequest {
        SearchRequest {
            index_name: "test-index".to_string(),
            query: Some(all_query()),
            ..Default::default()
        }
    }

    fn fusion_request(sub_queries: usize, fetch_limit: u32) -> SearchRequest {
        SearchRequest {
            index_name: "test-index".to_string(),
            query: Some(Query {
                query: Some(query::Query::Fusion(FusionQuery {
                    queries: (0..sub_queries)
                        .map(|_| WeightedQuery {
                            query: Some(all_query()),
                            weight: 1.0,
                        })
                        .collect(),
                    fetch_limit,
                    ..Default::default()
                })),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn search_budget_applies_defaults() {
        let budget = validate_search_budget(&ordinary_request()).unwrap();

        assert_eq!(budget.final_limit, DEFAULT_SEARCH_LIMIT);
        assert_eq!(budget.offset, 0);
        assert_eq!(budget.search_limit, DEFAULT_SEARCH_LIMIT);
        assert_eq!(budget.rerank_l1_limit, None);
        assert_eq!(budget.fusion_fetch_limit, None);
    }

    #[test]
    fn search_budget_rejects_missing_or_oversized_index_name() {
        let mut req = ordinary_request();
        req.index_name.clear();
        assert_eq!(
            validate_search_budget(&req).unwrap_err().code(),
            Code::InvalidArgument
        );

        req.index_name = "x".repeat(MAX_INDEX_NAME_BYTES + 1);
        assert_eq!(
            validate_search_budget(&req).unwrap_err().code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn search_budget_rejects_excessive_final_limit() {
        let mut req = ordinary_request();
        req.limit = (MAX_SEARCH_LIMIT + 1) as u32;

        let err = validate_search_budget(&req).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[test]
    fn search_budget_accounts_for_offset_and_bounds_the_result_window() {
        let mut req = ordinary_request();
        req.limit = 100;
        req.offset = 400;
        let budget = validate_search_budget(&req).unwrap();
        assert_eq!(budget.final_limit, 100);
        assert_eq!(budget.offset, 400);
        assert_eq!(budget.search_limit, 500);

        req.limit = DEFAULT_SEARCH_LIMIT as u32;
        req.offset = (MAX_SEARCH_WINDOW - DEFAULT_SEARCH_LIMIT) as u32;
        assert!(validate_search_budget(&req).is_ok());

        req.offset += 1;
        let err = validate_search_budget(&req).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[test]
    fn search_budget_caps_rerank_depth_at_two_x_the_result_window() {
        let mut ordinary_default = ordinary_request();
        ordinary_default.limit = 300;
        ordinary_default.reranker = Some(Reranker::default());
        assert_eq!(
            validate_search_budget(&ordinary_default)
                .unwrap()
                .rerank_l1_limit,
            Some(600)
        );

        let mut default_req = ordinary_request();
        default_req.limit = MAX_SEARCH_LIMIT as u32;
        default_req.reranker = Some(Reranker::default());
        assert_eq!(
            validate_search_budget(&default_req)
                .unwrap()
                .rerank_l1_limit,
            Some(MAX_SEARCH_LIMIT * 2)
        );

        let mut excessive_req = ordinary_request();
        excessive_req.limit = 100;
        excessive_req.reranker = Some(Reranker {
            limit: 201,
            ..Default::default()
        });
        let err = validate_search_budget(&excessive_req).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);

        let mut impossible_page = ordinary_request();
        impossible_page.limit = 10;
        impossible_page.offset = 1_000;
        impossible_page.reranker = Some(Reranker {
            limit: 100,
            ..Default::default()
        });
        let err = validate_search_budget(&impossible_page).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[test]
    fn search_admission_rejects_overload_without_queueing() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = try_acquire_search_permit(&permits).unwrap();

        let err = try_acquire_search_permit(&permits).unwrap_err();
        assert_eq!(err.code(), Code::ResourceExhausted);

        drop(permit);
        assert!(try_acquire_search_permit(&permits).is_ok());
    }

    #[test]
    fn fusion_budget_rejects_more_than_two_x_the_result_window() {
        let mut req = fusion_request(2, 201);
        req.limit = 100;

        let err = validate_search_budget(&req).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[test]
    fn fusion_default_does_not_multiply_an_existing_rerank_pool() {
        let mut req = fusion_request(2, 0);
        req.limit = 100;
        req.reranker = Some(Reranker::default());

        let budget = validate_search_budget(&req).unwrap();
        assert_eq!(budget.rerank_l1_limit, Some(200));
        assert_eq!(budget.fusion_fetch_limit, Some(200));
    }

    #[test]
    fn fusion_budget_rejects_too_many_sub_queries_before_conversion() {
        let req = fusion_request(MAX_FUSION_SUB_QUERIES + 1, 50);

        let err = validate_search_budget(&req).unwrap_err();
        assert_eq!(err.code(), Code::InvalidArgument);
    }

    #[test]
    fn fusion_budget_rejects_excessive_fetch_and_aggregate_work() {
        let excessive_fetch = fusion_request(2, (MAX_CANDIDATE_LIMIT + 1) as u32);
        assert_eq!(
            validate_search_budget(&excessive_fetch).unwrap_err().code(),
            Code::InvalidArgument
        );

        let mut excessive_aggregate = fusion_request(5, MAX_CANDIDATE_LIMIT as u32);
        excessive_aggregate.limit = MAX_SEARCH_LIMIT as u32;
        excessive_aggregate.offset = (MAX_SEARCH_WINDOW - MAX_SEARCH_LIMIT) as u32;
        assert_eq!(
            validate_search_budget(&excessive_aggregate)
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn fusion_default_fetch_is_checked_and_capped() {
        let mut req = fusion_request(2, 0);
        req.limit = MAX_SEARCH_LIMIT as u32;
        req.reranker = Some(Reranker::default());

        let budget = validate_search_budget(&req).unwrap();
        assert_eq!(budget.rerank_l1_limit, Some(MAX_SEARCH_LIMIT * 2));
        assert_eq!(budget.fusion_fetch_limit, Some(MAX_SEARCH_LIMIT * 2));
    }

    #[test]
    fn fusion_budget_rejects_non_finite_or_negative_scoring_parameters() {
        for rrf_k in [f32::NAN, f32::INFINITY, -1.0] {
            let mut req = fusion_request(2, 50);
            let Some(query::Query::Fusion(fusion)) =
                req.query.as_mut().and_then(|query| query.query.as_mut())
            else {
                unreachable!();
            };
            fusion.rrf_k = rrf_k;
            assert_eq!(
                validate_search_budget(&req).unwrap_err().code(),
                Code::InvalidArgument
            );
        }

        for weight in [f32::NAN, f32::INFINITY, -1.0] {
            let mut req = fusion_request(2, 50);
            let Some(query::Query::Fusion(fusion)) =
                req.query.as_mut().and_then(|query| query.query.as_mut())
            else {
                unreachable!();
            };
            fusion.queries[0].weight = weight;
            assert_eq!(
                validate_search_budget(&req).unwrap_err().code(),
                Code::InvalidArgument
            );
        }
    }

    #[test]
    fn request_shape_rejects_field_and_boolean_amplification() {
        let mut too_many_fields = ordinary_request();
        too_many_fields.fields_to_load = vec![String::new(); MAX_FIELDS_TO_LOAD + 1];
        assert_eq!(
            validate_search_budget(&too_many_fields).unwrap_err().code(),
            Code::InvalidArgument
        );

        let mut too_many_clauses = ordinary_request();
        too_many_clauses.query = Some(Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                must: (0..=MAX_BOOLEAN_CLAUSES).map(|_| all_query()).collect(),
                ..Default::default()
            })),
        });
        assert_eq!(
            validate_search_budget(&too_many_clauses)
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn request_shape_rejects_depth_node_text_and_vector_expansion() {
        let mut nested = all_query();
        for _ in 0..MAX_QUERY_DEPTH {
            nested = Query {
                query: Some(query::Query::Boost(Box::new(BoostQuery {
                    query: Some(Box::new(nested)),
                    boost: 1.0,
                }))),
            };
        }
        let mut excessive_depth = ordinary_request();
        excessive_depth.query = Some(nested);
        assert_eq!(
            validate_search_budget(&excessive_depth).unwrap_err().code(),
            Code::InvalidArgument
        );

        // 1 root + 128 Boolean children + 128 leaves exceeds the node budget,
        // while each Boolean and the aggregate clause count remain legal.
        let branches = (0..MAX_BOOLEAN_CLAUSES)
            .map(|_| Query {
                query: Some(query::Query::Boolean(BooleanQuery {
                    must: vec![all_query()],
                    ..Default::default()
                })),
            })
            .collect();
        let mut excessive_nodes = ordinary_request();
        excessive_nodes.query = Some(Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                should: branches,
                ..Default::default()
            })),
        });
        assert_eq!(
            validate_search_budget(&excessive_nodes).unwrap_err().code(),
            Code::InvalidArgument
        );

        let mut excessive_text = ordinary_request();
        excessive_text.query = Some(Query {
            query: Some(query::Query::Match(MatchQuery {
                field: "body".to_owned(),
                text: "x".repeat(MAX_QUERY_TEXT_BYTES + 1),
            })),
        });
        assert_eq!(
            validate_search_budget(&excessive_text).unwrap_err().code(),
            Code::InvalidArgument
        );

        let mut excessive_vector = ordinary_request();
        excessive_vector.query = Some(Query {
            query: Some(query::Query::DenseVector(DenseVectorQuery {
                field: "embedding".to_owned(),
                vector: vec![0.0; MAX_DENSE_QUERY_DIMS + 1],
                ..Default::default()
            })),
        });
        assert_eq!(
            validate_search_budget(&excessive_vector)
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn requested_fields_are_resolved_and_deduplicated_once() {
        let mut builder = hermes_core::SchemaBuilder::default();
        let title = builder.add_text_field("title", true, true);
        let body = builder.add_text_field("body", true, true);
        let schema = builder.build();
        let requested = vec![
            "title".to_owned(),
            "missing".to_owned(),
            "title".to_owned(),
            "body".to_owned(),
        ];

        assert_eq!(
            resolve_requested_fields(&schema, &requested),
            vec![
                ResolvedField {
                    id: title,
                    name: "title".to_owned(),
                },
                ResolvedField {
                    id: body,
                    name: "body".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn response_budget_bounds_retained_and_encoded_bytes() {
        let value = CoreFieldValue::Bytes(vec![0; 64]);
        assert!(retained_field_value_bytes(&value).unwrap() > 64);

        let mut retained = SearchResponseBudget::with_maximum(100);
        retained.reserve_retained(80).unwrap();
        let err = retained.reserve_retained(21).unwrap_err();
        assert_eq!(err.code(), Code::ResourceExhausted);

        let hit = SearchHit {
            address: Some(DocAddress {
                segment_id: "0".repeat(32),
                doc_id: 1,
            }),
            score: 1.0,
            fields: HashMap::from([(
                "body".to_owned(),
                FieldValueList {
                    values: vec![FieldValue {
                        value: Some(field_value::Value::Text("x".repeat(128))),
                    }],
                },
            )]),
            ordinal_scores: Vec::new(),
        };
        let mut encoded = SearchResponseBudget::with_maximum(64);
        let err = encoded.reserve_hit(&hit).unwrap_err();
        assert_eq!(err.code(), Code::ResourceExhausted);
    }

    #[test]
    fn metrics_use_only_canonical_schema_labels() {
        let mut builder = hermes_core::SchemaBuilder::default();
        builder.add_text_field("title", true, true);
        let mut schema = builder.build();

        assert_eq!(UNKNOWN_INDEX_LABEL, "unknown");
        assert_eq!(canonical_metric_index_label(&schema), UNKNOWN_INDEX_LABEL);
        schema.set_index_name("known-index");
        assert_eq!(canonical_metric_index_label(&schema), "known-index");
    }
}
