//! Search service gRPC implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;
use tonic::{Request, Response, Status};

use crate::converters::{
    convert_field_value, convert_query, convert_reranker, schema_to_sdl, text_stats_from_proto,
    text_stats_to_proto,
};
use crate::proto::search_service_server::SearchService;
use crate::proto::*;
use crate::registry::IndexRegistry;

mod candidate_scoring;
mod response;
#[cfg(test)]
mod tests;
mod validation;

use response::{
    SearchResponseBudget, resolve_requested_fields, retained_field_entry_bytes,
    retained_field_value_bytes, retained_hit_base_bytes,
};
pub use validation::{QueryShapeLimits, SearchLimits};
use validation::{try_acquire_search_permit, validate_search_budget, validate_text_stats_request};

const UNKNOWN_INDEX_LABEL: &str = "unknown";

fn canonical_metric_index_label(schema: &hermes_core::Schema) -> &str {
    schema.index_label()
}

/// Search service implementation
pub struct SearchServiceImpl {
    pub registry: Arc<IndexRegistry>,
    search_permits: Arc<Semaphore>,
    limits: SearchLimits,
}

impl SearchServiceImpl {
    pub fn new(
        registry: Arc<IndexRegistry>,
        max_concurrent_searches: usize,
        limits: SearchLimits,
    ) -> Self {
        assert!(
            max_concurrent_searches > 0,
            "max_concurrent_searches must be greater than zero"
        );
        Self {
            registry,
            search_permits: Arc::new(Semaphore::new(max_concurrent_searches)),
            limits,
        }
    }

    fn acquire_search_permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, Status> {
        try_acquire_search_permit(&self.search_permits).inspect_err(|_| {
            metrics::counter!(
                "hermes_search_admission_rejected_total",
                "index" => UNKNOWN_INDEX_LABEL,
            )
            .increment(1);
        })
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
        let budget = validate_search_budget(&req, &self.limits)?;

        // Bound expensive pipelines across all HTTP/2 connections without an
        // unbounded waiter queue retaining decoded requests under overload.
        // Dropping the owned permit on completion/error/cancellation is safe.
        let _search_permit = self.acquire_search_permit()?;

        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let _ = metric_index.set(canonical_metric_index_label(&index.schema()).to_owned());
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        let query = req
            .query.as_ref()
            .ok_or_else(|| Status::invalid_argument("Query is required"))?;

        // Rank enough results to cover the requested page, then apply the
        // offset only after fusion/reranking so pagination preserves ranking.
        let limit = budget.search_limit;
        let candidate_limit = budget.candidate_limit;

        // Optional L2 reranker config; the L1 pool it consumes is either a
        // single query's results or the fused union of sub-query results.
        let rerank_setup = req
            .reranker
            .as_ref()
            .map(|reranker| {
                convert_reranker(reranker, searcher.schema())
                    .map_err(|e| Status::invalid_argument(format!("Invalid reranker: {}", e)))
            })
            .transpose()?;

        // ── Phase 1: L1 search ──────────────────────────────────────────────
        let start = Instant::now();
        let t_search = Instant::now();
        let query_desc;
        // Anytime mode: text executors stop at the deadline and the response
        // says so; 0 keeps the exact top-k.
        let deadline = (req.time_budget_ms > 0)
            .then(|| Instant::now() + std::time::Duration::from_millis(req.time_budget_ms));
        let mut truncated = false;
        let mut candidate_scoring_us = 0;
        let mut ranking_method = String::new();
        let mut feature_exports = HashMap::new();
        let mut fusion_candidates = Vec::new();
        let mut response_budget = SearchResponseBudget::with_maximum(self.limits.max_search_response_bytes);
        let (results, total_seen, rerank_config) =
            if req.l1.is_some() || req.score_export.is_some() {
                let Some(crate::proto::query::Query::Fusion(fusion)) = &query.query else { unreachable!("validated fusion scoring request") };
                let queries: Vec<Arc<dyn hermes_core::query::Query>> = fusion.queries.iter().map(|branch| {
                    convert_query(branch.query.as_ref().expect("validated branch"), searcher.schema(), Some(searcher.global_stats()), Some(index.directory().root()), &self.limits.shape)
                        .map(Arc::from).map_err(|e| Status::invalid_argument(format!("Invalid scoring branch: {e}")))
                }).collect::<Result<_, _>>()?;
                let plan = candidate_scoring::scoring_plan(fusion, &queries, &req, searcher.schema())?;
                let stats = match req.text_stats.as_ref() {
                    Some(stats) => Arc::new(text_stats_from_proto(stats, searcher.schema())),
                    None => searcher.candidate_text_stats(&plan).await.map_err(crate::error::hermes_error_to_status)?,
                };
                let filters = candidate_scoring::convert_filters(fusion, searcher.schema(), Some(searcher.global_stats()), Some(index.directory().root()), &self.limits.shape)?;
                let nominations: Vec<_> = fusion.queries.iter().zip(&queries).filter(|(branch, _)| !branch.score_only)
                    .map(|(_, query)| candidate_scoring::with_filters(query.clone(), &filters)).collect();
                let depth = if fusion.candidate_depth == 0 { candidate_limit } else { fusion.candidate_depth as usize };
                let (candidates, seen) = searcher.search_candidate_union(&nominations, depth, stats.clone()).await.map_err(crate::error::hermes_error_to_status)?;
                if req.l1.is_none() && candidates.len() > limit {
                    return Err(Status::invalid_argument(format!("feature export needs limit >= complete candidate union ({} documents)", candidates.len())));
                }
                let t_scoring = Instant::now();
                let mut scored = searcher.score_candidates(&candidates, &plan, Some(stats)).await.map_err(crate::error::hermes_error_to_status)?;
                candidate_scoring_us = t_scoring.elapsed().as_micros() as u64;
                let fused_limit = if rerank_setup.is_some() { candidate_limit } else { limit };
                scored.truncate(fused_limit);
                let mut results = Vec::with_capacity(scored.len());
                for candidate in scored {
                    if req.score_export.is_some() {
                        response_budget.reserve_retained(candidate_scoring::retained_raw_score_bytes(&candidate.features, &plan)?)?;
                        feature_exports.insert((candidate.result.segment_id, candidate.result.doc_id), candidate_scoring::export_scores(candidate.features, &plan));
                    }
                    results.push(candidate.result);
                }
                ranking_method = if req.l1.is_some() { "linear_v1" } else { "feature_export_v1" }.into();
                query_desc = format!("{}: {} branches, depth {}, union {}", ranking_method, queries.len(), depth, candidates.len());
                (results, seen, rerank_setup.map(|config| (config, limit)))
            } else if let Some(crate::proto::query::Query::Fusion(fusion)) = &query.query {
                // Fusion: run each sub-query independently and fuse the ranked
                // lists (union). Handled here rather than in convert_query
                // because fusion is a searcher-level operation.
                let filters = candidate_scoring::convert_filters(fusion, searcher.schema(), Some(searcher.global_stats()), Some(index.directory().root()), &self.limits.shape)?;
                let mut sub_queries = Vec::with_capacity(fusion.queries.len());
                for weighted in &fusion.queries {
                    let sub = weighted
                        .query
                        .as_ref()
                        .ok_or_else(|| Status::invalid_argument("Fusion sub-query is missing"))?;
                    let core = convert_query(
                        sub,
                        searcher.schema(),
                        Some(searcher.global_stats()),
                        Some(index.directory().root()),
                        &self.limits.shape,
                    )
                    .map_err(|e| {
                        Status::invalid_argument(format!("Invalid fusion sub-query: {}", e))
                    })?;
                    let weight = if weighted.weight > 0.0 {
                        weighted.weight
                    } else {
                        1.0
                    };
                    sub_queries.push((candidate_scoring::with_filters(Arc::from(core), &filters), weight));
                }
                if fusion.method == crate::proto::FusionMethod::FusionCandidates as i32 {
                    let depth = if fusion.candidate_depth == 0 { candidate_limit } else { fusion.candidate_depth as usize };
                    let queries: Vec<_> = sub_queries.iter().map(|(query, _)| query.clone()).collect();
                    let stats = req.text_stats.as_ref().map(|stats| Arc::new(text_stats_from_proto(stats, searcher.schema())));
                    let lists = searcher.search_candidate_lists(&queries, depth, stats).await.map_err(crate::error::hermes_error_to_status)?;
                    fusion_candidates = candidate_scoring::export_nomination_lists(&lists, &mut response_budget)?;
                    let seen = lists.iter().fold(0u32, |sum, (_, seen)| sum.saturating_add(*seen));
                    let candidates = searcher.merge_candidate_lists(lists.into_iter().map(|(list, _)| list)).map_err(crate::error::hermes_error_to_status)?;
                    if candidates.len() > limit {
                        return Err(Status::invalid_argument(format!("candidate export needs limit >= complete union ({} documents)", candidates.len())));
                    }
                    ranking_method = "fusion_candidates_v1".into();
                    query_desc = format!("candidate export of {} branches, depth {}", sub_queries.len(), depth);
                    (candidates, seen, None)
                } else {
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
                    crate::proto::FusionMethod::FusionCandidates => unreachable!("candidate export handled separately"),
                    crate::proto::FusionMethod::FusionNormalizedWeightedSum => {
                        hermes_core::query::FusionMethod::NormalizedWeightedSum
                    }
                };

                // With a reranker, the fused list is the L1 candidate pool.
                let fused_limit = if rerank_setup.is_some() {
                    candidate_limit
                } else {
                    limit
                };

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
                    candidate_limit
                );
                log::info!(
                    "search: index={}, limit={}, candidates={}, query={}",
                    req.index_name,
                    req.limit,
                    candidate_limit,
                    query_desc
                );

                let query_refs: Vec<(&dyn hermes_core::query::Query, f32)> = sub_queries
                    .iter()
                    .map(|(query, weight)| (query.as_ref(), *weight))
                    .collect();
                let (fused, seen) = searcher
                    .search_fused_with_count(
                        &query_refs,
                        candidate_limit,
                        fused_limit,
                        method,
                        combiner,
                    )
                    .await
                    .map_err(crate::error::hermes_error_to_status)?;
                let rerank_config = rerank_setup.map(|config| (config, limit));
                (fused, seen, rerank_config)
                }
            } else {
                let core_query = convert_query(
                    query,
                    searcher.schema(),
                    Some(searcher.global_stats()),
                    Some(index.directory().root()),
                    &self.limits.shape,
                )
                .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

                query_desc = core_query.to_string();
                log::info!(
                    "search: index={}, limit={}, candidates={}, query={}",
                    req.index_name,
                    req.limit,
                    candidate_limit,
                    query_desc
                );

                // Broker-supplied cross-shard statistics replace this
                // backend's own segment-aggregated IDF.
                let stats_override = req
                    .text_stats
                    .as_ref()
                    .map(|stats| Arc::new(text_stats_from_proto(stats, searcher.schema())));
                if let Some(config) = rerank_setup {
                    let (candidates, seen, hit_budget) = searcher
                        .search_with_count_budgeted_stats(
                            core_query.as_ref(),
                            candidate_limit,
                            deadline,
                            stats_override,
                        )
                        .await
                        .map_err(crate::error::hermes_error_to_status)?;
                    truncated = hit_budget;
                    (candidates, seen, Some((config, limit)))
                } else {
                    let (results, seen, hit_budget) = searcher
                        .search_with_positions_budgeted_stats(
                            core_query.as_ref(),
                            candidate_limit,
                            deadline,
                            stats_override,
                        )
                        .await
                        .map_err(crate::error::hermes_error_to_status)?;
                    truncated = hit_budget;
                    (results, seen, None)
                }
            };
        let search_us = (t_search.elapsed().as_micros() as u64).saturating_sub(candidate_scoring_us);

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

        let mut hits = Vec::with_capacity(results.len());
        for result in results {
            // Convert ordinal scores before hydration so their retained memory
            // is charged before reading potentially large stored fields.
            let mut ordinal_scores: Vec<OrdinalScore> = result
                .positions
                .iter()
                .flat_map(|(_, scored_positions)| {
                    scored_positions.iter().map(|sp| OrdinalScore {
                        ordinal: sp.position, // vector position contains the ordinal
                        score: sp.score,
                    })
                })
                .collect();
            if req.l1.is_some() {
                ordinal_scores.sort_unstable_by_key(|row| row.ordinal);
                ordinal_scores.dedup_by_key(|row| row.ordinal);
            }
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
                candidate_scores: feature_exports.remove(&(result.segment_id, result.doc_id)),
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
                candidate_scoring_us,
            }),
            truncated,
            ranking_method,
            fusion_candidates,
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

    async fn get_text_stats(
        &self,
        request: Request<GetTextStatsRequest>,
    ) -> Result<Response<GetTextStatsResponse>, Status> {
        let req = request.into_inner();
        validate_text_stats_request(&req, &self.limits.shape)?;
        let _search_permit = self.acquire_search_permit()?;
        let query = req
            .query
            .ok_or_else(|| Status::invalid_argument("query is required"))?;
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let core_query = convert_query(
            &query,
            searcher.schema(),
            Some(searcher.global_stats()),
            Some(index.directory().root()),
            &self.limits.shape,
        )
        .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;
        let mut terms = Vec::new();
        core_query.text_terms(&mut terms);
        terms.sort_unstable_by(|a, b| (a.0.0, &a.1).cmp(&(b.0.0, &b.1)));
        terms.dedup_by(|a, b| a.0.0 == b.0.0 && a.1 == b.1);
        let stats = searcher.global_stats().text_stats_for(&terms);
        Ok(Response::new(GetTextStatsResponse {
            stats: Some(text_stats_to_proto(&stats, searcher.schema())),
        }))
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
        let schema_str = schema_to_sdl(&index.schema());

        // Collect memory stats from segment readers
        let mut total_term_dict_cache = 0u64;
        let mut total_store_cache = 0u64;
        let mut total_sparse_index = 0u64;
        let mut total_dense_index = 0u64;

        for segment in searcher.segment_readers() {
            let stats = segment.memory_stats();
            total_term_dict_cache += stats.term_dict_cache_bytes as u64;
            total_store_cache += stats.store_cache_bytes as u64;
            total_sparse_index += stats.sparse_heap_bytes as u64;
            total_dense_index += stats.dense_heap_bytes as u64;
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

        let text_fields = text_field_infos(&index.schema());
        Ok(Response::new(GetIndexInfoResponse {
            index_name: req.index_name,
            num_docs: searcher.num_docs(),
            num_segments: searcher.segment_readers().len() as u32,
            schema: schema_str,
            memory_stats: Some(memory_stats),
            vector_stats,
            text_fields,
            candidate_scoring_version: 1,
            unprepared_candidate_fields: searcher
                .segment_readers()
                .iter()
                .flat_map(|reader| reader.unprepared_candidate_fields())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
        }))
    }
}

/// Tokenization facts of every text field, for clients that pick query
/// forms without parsing the schema SDL.
pub fn text_field_infos(schema: &hermes_core::Schema) -> Vec<TextFieldInfo> {
    schema
        .fields()
        .filter(|(_, entry)| entry.field_type == hermes_core::FieldType::Text)
        .map(|(_, entry)| {
            let spec = entry.tokenizer_spec();
            let lex = spec.as_ref().and_then(|spec| spec.lex());
            TextFieldInfo {
                field: entry.name.clone(),
                lexical: lex.is_some(),
                hint_field: lex
                    .and_then(|options| options.by.clone())
                    .unwrap_or_default(),
                variants: lex.is_some_and(|options| options.variants),
                positions: entry.positions.is_some(),
                chunked: entry.chunked,
            }
        })
        .collect()
}
