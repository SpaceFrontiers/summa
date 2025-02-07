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
        let start = Instant::now();
        let req = request.into_inner();

        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let reader = index
            .reader()
            .await
            .map_err(|e| Status::internal(format!("Failed to get reader: {}", e)))?;
        let searcher = reader
            .searcher()
            .await
            .map_err(|e| Status::internal(format!("Failed to get searcher: {}", e)))?;

        let query = req
            .query
            .ok_or_else(|| Status::invalid_argument("Query is required"))?;

        let core_query = convert_query(&query, reader.schema(), Some(searcher.global_stats()))
            .map_err(|e| Status::invalid_argument(format!("Invalid query: {}", e)))?;

        let limit = if req.limit == 0 {
            10
        } else {
            req.limit as usize
        };

        // Search using Searcher (with count of total docs scored)
        let (results, total_seen) = if let Some(reranker) = &req.reranker {
            let config = convert_reranker(reranker, reader.schema())
                .map_err(|e| Status::invalid_argument(format!("Invalid reranker: {}", e)))?;
            let l1_limit = if reranker.limit == 0 {
                limit * 10
            } else {
                reranker.limit as usize
            };
            searcher
                .search_and_rerank(core_query.as_ref(), l1_limit, limit, &config)
                .await
                .map_err(|e| Status::internal(format!("Search failed: {}", e)))?
        } else {
            searcher
                .search_with_positions(core_query.as_ref(), limit)
                .await
                .map_err(|e| Status::internal(format!("Search failed: {}", e)))?
        };

        // Convert to response with optional field loading
        let mut hits = Vec::new();
        for result in results {
            let mut fields = HashMap::new();

            if !req.fields_to_load.is_empty()
                && let Ok(Some(doc)) = searcher.doc(result.doc_id).await
            {
                for field_name in &req.fields_to_load {
                    if let Some(field) = searcher.schema().get_field(field_name)
                        && let Some(value) = doc.get_first(field)
                    {
                        fields.insert(field_name.clone(), convert_field_value(value));
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
                doc_id: result.doc_id,
                score: result.score,
                fields,
                ordinal_scores,
            });
        }

        let took_ms = start.elapsed().as_millis() as u64;

        // total_seen = number of documents that were actually scored across all segments
        Ok(Response::new(SearchResponse {
            hits,
            total_hits: total_seen,
            took_ms,
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
            .map_err(|e| Status::internal(format!("Failed to get reader: {}", e)))?;
        let searcher = reader
            .searcher()
            .await
            .map_err(|e| Status::internal(format!("Failed to get searcher: {}", e)))?;

        let doc = searcher
            .doc(req.doc_id)
            .await
            .map_err(|e| Status::internal(format!("Failed to get document: {}", e)))?
            .ok_or_else(|| Status::not_found("Document not found"))?;

        let mut fields = HashMap::new();
        for (field, value) in doc.field_values() {
            if let Some(entry) = index.schema().get_field_entry(*field) {
                fields.insert(entry.name.clone(), convert_field_value(value));
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
            .map_err(|e| Status::internal(format!("Failed to get reader: {}", e)))?;
        let searcher = reader
            .searcher()
            .await
            .map_err(|e| Status::internal(format!("Failed to get searcher: {}", e)))?;

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

        Ok(Response::new(GetIndexInfoResponse {
            index_name: req.index_name,
            num_docs: searcher.num_docs(),
            num_segments: searcher.segment_readers().len() as u32,
            schema: schema_str,
            memory_stats: Some(memory_stats),
        }))
    }
}
