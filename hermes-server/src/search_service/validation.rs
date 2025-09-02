//! Request shape, candidate budgets, and shared search admission.

use crate::proto::*;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::Status;

pub(super) const MAX_FUSION_SUB_QUERIES: usize = hermes_core::query::MAX_FUSION_SUB_QUERIES;
const MAX_FUSION_CANDIDATE_SLOTS: usize = hermes_core::query::MAX_FUSION_CANDIDATE_SLOTS;

/// Structural anti-DoS budgets for one decoded request, set once at startup
/// from the hermes-server CLI (flags of the same names). The transport's
/// decode limit (`--search-max-decode-mb`) bounds wire bytes, not decoded
/// object count or downstream expansion — empty protobuf messages and strings
/// are only a few bytes on the wire — so these are independent budgets.
/// `Default` preserves the historical hard-coded values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryShapeLimits {
    /// Maximum query tree nesting depth.
    pub max_query_depth: usize,
    /// Maximum query tree nodes per request.
    pub max_query_nodes: usize,
    /// Maximum aggregate clauses across the query tree.
    pub max_query_clauses: usize,
    /// Maximum clauses in a single Boolean search query. Statistics-only
    /// containers use the aggregate clause limit because the broker flattens
    /// text leaves before requesting statistics.
    pub max_boolean_clauses: usize,
    /// Maximum aggregate query text bytes (terms, match text, field names).
    pub max_query_text_bytes: usize,
    /// Maximum bytes in one field name.
    pub max_field_name_bytes: usize,
    /// Maximum bytes in the index name.
    pub max_index_name_bytes: usize,
    /// Maximum elements in one dense query/reranker vector.
    pub max_dense_query_dims: usize,
    /// Maximum entries in one sparse query vector.
    pub max_sparse_query_dims: usize,
    /// Maximum bytes in one binary query/reranker vector.
    pub max_binary_query_bytes: usize,
    /// Maximum aggregate query vector bytes per request.
    pub max_total_query_vector_bytes: usize,
    /// Maximum names in SearchRequest.fields_to_load.
    pub max_fields_to_load: usize,
    /// Maximum aggregate bytes across fields_to_load names.
    pub max_fields_to_load_name_bytes: usize,
    /// Maximum tokens a Term/Match query may expand to after tokenization.
    pub max_text_query_tokens: usize,
    /// Maximum dimensions a SparseVectorQuery text may tokenize into.
    pub max_sparse_token_dimensions: usize,
}

impl Default for QueryShapeLimits {
    fn default() -> Self {
        Self {
            max_query_depth: 32,
            max_query_nodes: 256,
            max_query_clauses: 512,
            max_boolean_clauses: 128,
            max_query_text_bytes: 64 * 1024,
            max_field_name_bytes: 255,
            max_index_name_bytes: 255,
            max_dense_query_dims: 65_536,
            max_sparse_query_dims: 4_096,
            max_binary_query_bytes: 256 * 1024,
            max_total_query_vector_bytes: 1024 * 1024,
            max_fields_to_load: 64,
            max_fields_to_load_name_bytes: 16 * 1024,
            max_text_query_tokens: 256,
            max_sparse_token_dimensions: 4_096,
        }
    }
}

/// Request-facing result and hydration limits, set once at startup from the
/// hermes-server CLI (`--default-search-limit`, `--max-search-limit`,
/// `--max-search-window`, `--max-candidate-limit`,
/// `--max-search-response-mb`). `Default` preserves the historical
/// hard-coded values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchLimits {
    /// Results returned when `SearchRequest.limit` is 0.
    pub default_search_limit: usize,
    /// Upper bound on `SearchRequest.limit`.
    pub max_search_limit: usize,
    /// Upper bound on `SearchRequest.offset + limit`.
    pub max_search_window: usize,
    /// Upper bound on the first-stage candidate pool
    /// (`SearchRequest.candidate_limit`).
    pub max_candidate_limit: usize,
    /// Structural anti-DoS budgets for the decoded query tree.
    pub shape: QueryShapeLimits,
    /// Hydration budget for one search response: caps both estimated retained
    /// heap while the response is built and its encoded size. Must leave
    /// headroom below the transport's encode limit (`--search-max-encode-mb`)
    /// for compression and framing — startup validation enforces the ordering.
    /// 2026-08-22: raised 48 → 192 MiB — chunk-level binary reranking over a
    /// book-heavy RAG pool legitimately hydrates >48 MiB (a few dozen books ×
    /// thousands of 320-byte chunk vectors). Downstream caps that must stay
    /// above this: hermes-server `--search-max-encode-mb` (256 MiB),
    /// hermes-broker `--backend-max-decode-mb` / `--search-max-encode-mb`
    /// (256 MiB), and the consumers' hermes-client decode cap (200 MiB).
    pub max_search_response_bytes: usize,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            default_search_limit: 10,
            max_search_limit: 10_000,
            max_search_window: 50_000,
            max_candidate_limit: 50_000,
            shape: QueryShapeLimits::default(),
            max_search_response_bytes: 192 * 1024 * 1024,
        }
    }
}

struct QueryShapeBudget<'a> {
    shape: &'a QueryShapeLimits,
    nodes: usize,
    clauses: usize,
    text_bytes: usize,
    vector_bytes: usize,
}

impl<'a> QueryShapeBudget<'a> {
    fn new(shape: &'a QueryShapeLimits) -> Self {
        Self {
            shape,
            nodes: 0,
            clauses: 0,
            text_bytes: 0,
            vector_bytes: 0,
        }
    }

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
        if depth > self.shape.max_query_depth {
            return Err(Status::invalid_argument(format!(
                "Query nesting depth must not exceed {}",
                self.shape.max_query_depth
            )));
        }
        Self::add_limited(
            &mut self.nodes,
            1,
            self.shape.max_query_nodes,
            "Query node count",
        )
    }

    fn add_clauses(&mut self, clauses: usize) -> Result<(), Status> {
        Self::add_limited(
            &mut self.clauses,
            clauses,
            self.shape.max_query_clauses,
            "Query clause count",
        )
    }

    fn add_field_name(&mut self, name: &str, description: &str) -> Result<(), Status> {
        if name.len() > self.shape.max_field_name_bytes {
            return Err(Status::invalid_argument(format!(
                "{description} must not exceed {} bytes",
                self.shape.max_field_name_bytes
            )));
        }
        self.add_text(name.len())
    }

    fn add_text(&mut self, bytes: usize) -> Result<(), Status> {
        Self::add_limited(
            &mut self.text_bytes,
            bytes,
            self.shape.max_query_text_bytes,
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
            self.shape.max_total_query_vector_bytes,
            "Aggregate query vector bytes",
        )
    }
}

fn validate_index_name_shape(index_name: &str, shape: &QueryShapeLimits) -> Result<(), Status> {
    if index_name.is_empty() || index_name.len() > shape.max_index_name_bytes {
        return Err(Status::invalid_argument(format!(
            "index_name must contain 1..={} bytes",
            shape.max_index_name_bytes
        )));
    }
    Ok(())
}

/// The statistics RPC walks the same query tree as search. Validate it before
/// opening an index or recursively converting a potentially amplified tree.
pub(super) fn validate_text_stats_request(
    req: &GetTextStatsRequest,
    shape: &QueryShapeLimits,
) -> Result<(), Status> {
    validate_index_name_shape(&req.index_name, shape)?;
    let query = req
        .query
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("query is required"))?;
    // No Boolean scorer is built for statistics extraction. The broker can
    // flatten a valid nested search into more than one scoring node's fanout.
    // Keep depth, nodes, aggregate clauses, text, and vector budgets intact.
    let stats_shape = QueryShapeLimits {
        max_boolean_clauses: shape.max_query_clauses,
        ..*shape
    };
    validate_query_shape(query, &stats_shape)?;
    Ok(())
}

fn validate_query_shape<'a>(
    root: &Query,
    shape: &'a QueryShapeLimits,
) -> Result<QueryShapeBudget<'a>, Status> {
    let mut budget = QueryShapeBudget::new(shape);
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
                budget.add_text(term.tokenizer_hint.len())?;
            }
            query::Query::Phrase(phrase) => {
                budget.add_field_name(&phrase.field, "PhraseQuery.field")?;
                budget.add_text(phrase.text.len())?;
                budget.add_text(phrase.tokenizer_hint.len())?;
            }
            query::Query::Boolean(boolean) => {
                let clauses = boolean
                    .must
                    .len()
                    .checked_add(boolean.should.len())
                    .and_then(|count| count.checked_add(boolean.must_not.len()))
                    .ok_or_else(|| Status::invalid_argument("Boolean clause count overflows"))?;
                if clauses > shape.max_boolean_clauses {
                    return Err(Status::invalid_argument(format!(
                        "Each BooleanQuery supports at most {} clauses \
                         (got {clauses})",
                        shape.max_boolean_clauses
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
                    shape.max_sparse_query_dims,
                )?;
                budget.add_vector(
                    "SparseVectorQuery.values",
                    sparse.values.len(),
                    std::mem::size_of::<f32>(),
                    shape.max_sparse_query_dims,
                )?;
            }
            query::Query::DenseVector(dense) => {
                budget.add_field_name(&dense.field, "DenseVectorQuery.field")?;
                budget.add_vector(
                    "DenseVectorQuery.vector",
                    dense.vector.len(),
                    std::mem::size_of::<f32>(),
                    shape.max_dense_query_dims,
                )?;
            }
            query::Query::Match(match_query) => {
                budget.add_field_name(&match_query.field, "MatchQuery.field")?;
                budget.add_text(match_query.text.len())?;
                budget.add_text(match_query.tokenizer_hint.len())?;
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
                    shape.max_binary_query_bytes,
                )?;
            }
            query::Query::Fusion(fusion) => {
                if depth != 1 {
                    return Err(Status::invalid_argument(
                        "FusionQuery is only supported at the top level",
                    ));
                }
                if fusion.filters.len() > 64 {
                    return Err(Status::invalid_argument(
                        "common fusion filters exceed 64 clauses",
                    ));
                }
                if fusion.queries.len() > MAX_FUSION_SUB_QUERIES {
                    return Err(Status::invalid_argument(format!(
                        "FusionQuery supports at most {MAX_FUSION_SUB_QUERIES} sub-queries \
                         (got {})",
                        fusion.queries.len()
                    )));
                }
                budget.add_clauses(fusion.queries.len() + fusion.filters.len())?;
                for filter in &fusion.filters {
                    stack.push((filter, depth + 1));
                }
                for weighted in &fusion.queries {
                    if weighted.name.len() > 128 {
                        return Err(Status::invalid_argument(
                            "fusion branch name exceeds 128 bytes",
                        ));
                    }
                    budget.add_text(weighted.name.len())?;
                    let child = weighted
                        .query
                        .as_ref()
                        .ok_or_else(|| Status::invalid_argument("Fusion sub-query is missing"))?;
                    stack.push((child, depth + 1));
                }
            }
        }
    }

    Ok(budget)
}

/// Validate decoded request structure before acquiring scarce search capacity
/// or recursively converting the protobuf query tree.
fn validate_search_request_shape(
    req: &SearchRequest,
    root: &Query,
    shape: &QueryShapeLimits,
) -> Result<(), Status> {
    validate_index_name_shape(&req.index_name, shape)?;
    if req.fields_to_load.len() > shape.max_fields_to_load {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.fields_to_load supports at most {} names (got {})",
            shape.max_fields_to_load,
            req.fields_to_load.len()
        )));
    }
    let mut selected_name_bytes = 0usize;
    for (index, name) in req.fields_to_load.iter().enumerate() {
        if name.len() > shape.max_field_name_bytes {
            return Err(Status::invalid_argument(format!(
                "SearchRequest.fields_to_load[{index}] must not exceed {} bytes",
                shape.max_field_name_bytes
            )));
        }
        selected_name_bytes = selected_name_bytes
            .checked_add(name.len())
            .ok_or_else(|| Status::invalid_argument("Field selection byte count overflows"))?;
    }
    if selected_name_bytes > shape.max_fields_to_load_name_bytes {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.fields_to_load names must total at most \
             {} bytes (got {selected_name_bytes})",
            shape.max_fields_to_load_name_bytes
        )));
    }

    let scoring = req.l1.is_some() || req.score_export.is_some();
    if scoring {
        let Some(query::Query::Fusion(fusion)) = &root.query else {
            return Err(Status::invalid_argument(
                "l1/score_export require named fusion branches",
            ));
        };
        if fusion.queries.len() > MAX_FUSION_SUB_QUERIES {
            return Err(Status::invalid_argument("too many fusion branches"));
        }
        let names: Vec<&str> = fusion.queries.iter().map(|q| q.name.as_str()).collect();
        let mut seen = std::collections::HashSet::new();
        for branch in &fusion.queries {
            if branch.name.is_empty()
                || branch.name.len() > 128
                || !branch
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
                || !seen.insert(branch.name.as_str())
            {
                return Err(Status::invalid_argument(
                    "l1/score_export require unique, nonempty branch names",
                ));
            }
            if !matches!(
                ScoreScope::try_from(branch.scope),
                Ok(ScoreScope::Document | ScoreScope::Chunk)
            ) {
                return Err(Status::invalid_argument(
                    "l1/score_export require explicit document/chunk scopes",
                ));
            }
        }
        if fusion.queries.iter().all(|branch| branch.score_only) {
            return Err(Status::invalid_argument(
                "at least one branch must nominate candidates",
            ));
        }
        if let Some(model) = &req.l1 {
            if model.weights.len() > MAX_FUSION_SUB_QUERIES
                || model.transforms.len() > MAX_FUSION_SUB_QUERIES
            {
                return Err(Status::invalid_argument(
                    "l1 coefficient count exceeds branch limit",
                ));
            }
            if model
                .weights
                .keys()
                .chain(model.transforms.keys())
                .any(|name| name.len() > 128)
            {
                return Err(Status::invalid_argument(
                    "l1 coefficient name exceeds 128 bytes",
                ));
            }
            super::candidate_scoring::linear_model(model)
                .validate(&names)
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            if fusion.method != FusionMethod::FusionRrf as i32
                || !matches!(fusion.combiner, 0 | 1)
                || fusion.rrf_k != 0.0
                || fusion.queries.iter().any(|q| q.weight != 0.0)
                || req.reranker.as_ref().is_some_and(|r| r.rrf_k != 0.0)
            {
                return Err(Status::invalid_argument(
                    "l1 directly determines ranking; legacy fusion weights/method/rrf_k must be unset",
                ));
            }
        }
        if req.time_budget_ms != 0 {
            return Err(Status::invalid_argument(
                "l1/score_export require complete scoring; time_budget_ms must be unset",
            ));
        }
        if req.l1.is_none() && (req.offset != 0 || req.reranker.is_some()) {
            return Err(Status::invalid_argument(
                "export-only requests require offset=0 and no reranker",
            ));
        }
        if req
            .score_export
            .as_ref()
            .is_some_and(|export| export.passages_per_document > 65536)
        {
            return Err(Status::invalid_argument(
                "score_export passages_per_document exceeds 65536",
            ));
        }
    } else if let Some(query::Query::Fusion(fusion)) = &root.query
        && (fusion.candidate_depth != 0 || fusion.queries.iter().any(|q| q.score_only))
    {
        return Err(Status::invalid_argument(
            "candidate_depth/score_only require l1 or score_export",
        ));
    }

    let mut budget = validate_query_shape(root, shape)?;

    if let Some(reranker) = &req.reranker {
        budget.add_field_name(&reranker.field, "Reranker.field")?;
        budget.add_vector(
            "Reranker.vector",
            reranker.vector.len(),
            std::mem::size_of::<f32>(),
            shape.max_dense_query_dims,
        )?;
        budget.add_vector(
            "Reranker.binary_vector",
            reranker.binary_vector.len(),
            1,
            shape.max_binary_query_bytes,
        )?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SearchBudget {
    /// Number of results returned to the caller.
    pub(super) final_limit: usize,
    /// Number of leading results skipped for pagination.
    pub(super) offset: usize,
    /// Number of ranked results required before applying the offset.
    pub(super) search_limit: usize,
    /// Single first-stage pool used by every collector and retrieval mode.
    pub(super) candidate_limit: usize,
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

pub(super) fn try_acquire_search_permit(
    permits: &Arc<Semaphore>,
) -> Result<OwnedSemaphorePermit, Status> {
    Arc::clone(permits)
        .try_acquire_owned()
        .map_err(|_| Status::resource_exhausted("Search capacity is full; retry with backoff"))
}

/// Validate all request-controlled result and candidate depths before opening
/// an index, constructing queries, or allocating candidate lists.
pub(super) fn validate_search_budget(
    req: &SearchRequest,
    limits: &SearchLimits,
) -> Result<SearchBudget, Status> {
    let query = req
        .query
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("Query is required"))?;
    validate_search_request_shape(req, query, &limits.shape)?;
    let final_limit = bounded_limit(
        "SearchRequest.limit",
        req.limit,
        limits.default_search_limit,
        limits.max_search_limit,
    )?;
    let offset = req.offset as usize;
    let search_limit = offset
        .checked_add(final_limit)
        .ok_or_else(|| Status::invalid_argument("Search result window is too large"))?;
    if search_limit > limits.max_search_window {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.offset + limit must not exceed {} \
             (got {offset} + {final_limit} = {search_limit})",
            limits.max_search_window
        )));
    }
    let max_candidate_limit =
        hermes_core::query::max_candidate_limit(search_limit).min(limits.max_candidate_limit);
    let candidate_limit = bounded_limit(
        "SearchRequest.candidate_limit",
        req.candidate_limit,
        search_limit,
        max_candidate_limit,
    )?;
    if candidate_limit < search_limit {
        return Err(Status::invalid_argument(format!(
            "SearchRequest.candidate_limit must be at least offset + limit ({search_limit})"
        )));
    }

    if let Some(query::Query::Fusion(fusion)) = &query.query {
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

        let depth = if fusion.candidate_depth == 0 {
            candidate_limit
        } else {
            fusion.candidate_depth as usize
        };
        if depth > max_candidate_limit {
            return Err(Status::invalid_argument(format!(
                "fusion candidate_depth exceeds {max_candidate_limit}"
            )));
        }
        let candidate_slots = depth
            .checked_mul(fusion.queries.len())
            .ok_or_else(|| Status::invalid_argument("Fusion candidate budget is too large"))?;
        if candidate_slots > MAX_FUSION_CANDIDATE_SLOTS {
            return Err(Status::invalid_argument(format!(
                "FusionQuery candidate budget must not exceed {MAX_FUSION_CANDIDATE_SLOTS} \
                 (candidate_limit {candidate_limit} x {} sub-queries = {candidate_slots})",
                fusion.queries.len()
            )));
        }
    }

    Ok(SearchBudget {
        final_limit,
        offset,
        search_limit,
        candidate_limit,
    })
}
