//! Search service gRPC implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use crate::converters::{convert_field_value, convert_query, convert_reranker, schema_to_sdl};
use crate::proto::search_service_server::SearchService;
use crate::proto::*;
use crate::registry::IndexRegistry;

/// Search service implementation
pub struct SearchServiceImpl {
    pub registry: Arc<IndexRegistry>,
}

#[tonic::async_trait]
impl SearchService for SearchServiceImpl {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
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

        let query = req
            .query
            .ok_or_else(|| Status::invalid_argument("Query is required"))?;

        let limit = if req.limit == 0 {
            10
        } else {
            req.limit as usize
        };

        // Optional L2 reranker config; the L1 pool it consumes is either a
        // single query's results or the fused union of sub-query results.
        let rerank_setup = match &req.reranker {
            Some(reranker) => {
                let config = convert_reranker(reranker, reader.schema())
                    .map_err(|e| Status::invalid_argument(format!("Invalid reranker: {}", e)))?;
                let l1_limit = if reranker.limit == 0 {
                    limit * 10
                } else {
                    reranker.limit as usize
                };
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
                if sub_queries.is_empty() {
                    return Err(Status::invalid_argument(
                        "FusionQuery requires at least one sub-query",
                    ));
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
                let fetch_limit = if fusion.fetch_limit > 0 {
                    fusion.fetch_limit as usize
                } else {
                    fused_limit * 2
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

                let mut lists = Vec::with_capacity(sub_queries.len());
                let mut seen: u32 = 0;
                for (sub, weight) in &sub_queries {
                    let (sub_results, sub_seen) = searcher
                        .search_with_count(sub.as_ref(), fetch_limit)
                        .await
                        .map_err(crate::error::hermes_error_to_status)?;
                    seen = seen.saturating_add(sub_seen);
                    lists.push((sub_results, *weight));
                }
                let fused = hermes_core::query::fuse_ranked_lists(lists, method, fused_limit);
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
        let rerank_us = t_rerank.elapsed().as_micros() as u64;

        // ── Phase 3: Document field loading ─────────────────────────────────
        let t_load = Instant::now();

        // Resolve requested field names to field IDs once (not per-hit).
        // Only vector fields in this set will be hydrated from flat storage.
        let requested_field_ids: Option<rustc_hash::FxHashSet<u32>> =
            if req.fields_to_load.is_empty() {
                None
            } else {
                Some(
                    req.fields_to_load
                        .iter()
                        .filter_map(|name| searcher.schema().get_field(name))
                        .map(|f| f.0)
                        .collect(),
                )
            };

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

        let num_fields = req.fields_to_load.len();
        let mut hits = Vec::with_capacity(results.len());
        for result in results {
            let mut fields: HashMap<String, FieldValueList> = HashMap::with_capacity(num_fields);

            if !req.fields_to_load.is_empty() {
                let doc = searcher
                    .get_document_with_fields(
                        &hermes_core::query::DocAddress::new(result.segment_id, result.doc_id),
                        requested_field_ids.as_ref(),
                    )
                    .await
                    .map_err(crate::error::hermes_error_to_status)?;

                if let Some(doc) = doc {
                    for field_name in &req.fields_to_load {
                        if let Some(field) = searcher.schema().get_field(field_name) {
                            let values: Vec<_> =
                                doc.get_all(field).map(convert_field_value).collect();
                            if !values.is_empty() {
                                fields.insert(field_name.clone(), FieldValueList { values });
                            }
                        }
                    }
                }
            }

            // Convert ordinal scores from positions
            let ordinal_scores: Vec<OrdinalScore> = result
                .positions
                .iter()
                .flat_map(|(_, scored_positions)| {
                    scored_positions.iter().map(|sp| OrdinalScore {
                        ordinal: sp.position, // position contains the ordinal for vector fields
                        score: sp.score,
                    })
                })
                .collect();

            hits.push(SearchHit {
                address: Some(DocAddress {
                    segment_id: format!("{:032x}", result.segment_id),
                    doc_id: result.doc_id,
                }),
                score: result.score,
                fields,
                ordinal_scores,
            });
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
