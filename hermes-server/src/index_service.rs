//! Index service gRPC implementation

use std::sync::Arc;

use log::{info, warn};
use tonic::{Request, Response, Status};

use hermes_core::parse_schema;

use crate::converters::convert_proto_to_document;
use crate::proto::index_service_server::IndexService;
use crate::proto::*;
use crate::registry::IndexRegistry;

/// Index service implementation
pub struct IndexServiceImpl {
    pub registry: Arc<IndexRegistry>,
}

impl IndexServiceImpl {
    /// Convert a batch of streaming proto messages to Documents off the async
    /// runtime (spawn_blocking) and feed them to the index writer.
    /// Returns (indexed_count, recycled_batch_vec) — the Vec is drained but
    /// retains its capacity so the caller can reuse it.
    async fn flush_stream_batch(
        batch: Vec<IndexDocumentRequest>,
        schema: &Arc<hermes_core::Schema>,
        writer: &Arc<tokio::sync::RwLock<hermes_core::IndexWriter<hermes_core::MmapDirectory>>>,
    ) -> Result<(u32, Vec<IndexDocumentRequest>), Status> {
        let schema = Arc::clone(schema);
        let (docs, recycled) = tokio::task::spawn_blocking(move || {
            let mut docs = Vec::with_capacity(batch.len());
            for req in &batch {
                match convert_proto_to_document(&req.fields, &schema) {
                    Ok(doc) => docs.push(doc),
                    Err(e) => {
                        warn!("Skipping invalid document in stream batch: {}", e);
                    }
                }
            }
            let mut recycled = batch;
            recycled.clear();
            (docs, recycled)
        })
        .await
        .map_err(|e| Status::internal(format!("Conversion task failed: {}", e)))?;

        let count = docs.len() as u32;
        let w = writer.read().await;
        for doc in docs {
            w.add_document(doc)
                .map_err(|e| Status::internal(format!("Failed to index: {}", e)))?;
        }
        Ok((count, recycled))
    }
}

#[tonic::async_trait]
impl IndexService for IndexServiceImpl {
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        let req = request.into_inner();

        if req.schema.is_empty() {
            return Err(Status::invalid_argument("Schema is required"));
        }

        let schema = parse_schema(&req.schema)
            .map_err(|e| Status::invalid_argument(format!("Invalid schema: {}", e)))?;

        self.registry.create_index(&req.index_name, schema).await?;

        info!("Created index: {}", req.index_name);

        Ok(Response::new(CreateIndexResponse { success: true }))
    }

    async fn batch_index_documents(
        &self,
        request: Request<BatchIndexDocumentsRequest>,
    ) -> Result<Response<BatchIndexDocumentsResponse>, Status> {
        let req = request.into_inner();

        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let writer = self.registry.get_writer(&req.index_name).await?;
        let schema = index.schema().clone();

        // Move CPU-bound proto conversion off the async runtime
        let proto_docs = req.documents;
        let (documents, conversion_errors) = tokio::task::spawn_blocking(move || {
            let mut documents = Vec::with_capacity(proto_docs.len());
            let mut conversion_errors = 0u32;
            for named_doc in proto_docs {
                match convert_proto_to_document(&named_doc.fields, &schema) {
                    Ok(doc) => documents.push(doc),
                    Err(_) => conversion_errors += 1,
                }
            }
            (documents, conversion_errors)
        })
        .await
        .map_err(|e| Status::internal(format!("Conversion task failed: {}", e)))?;

        // Read lock allows concurrent document addition from multiple requests
        let doc_count = documents.len() as u32;
        let indexed_count = {
            let w = writer.read().await;
            w.add_documents(documents)
                .map_err(|e| Status::internal(format!("Failed to index documents: {}", e)))?
                as u32
        };
        let index_errors = doc_count - indexed_count;

        let error_count = conversion_errors + index_errors;

        info!(
            "Batch indexed documents: index={}, indexed={}, errors={}",
            req.index_name, indexed_count, error_count
        );

        Ok(Response::new(BatchIndexDocumentsResponse {
            indexed_count,
            error_count,
        }))
    }

    async fn index_documents(
        &self,
        request: Request<tonic::Streaming<IndexDocumentRequest>>,
    ) -> Result<Response<IndexDocumentsResponse>, Status> {
        let mut stream = request.into_inner();
        let mut indexed_count = 0u32;
        let mut current_schema: Option<Arc<hermes_core::Schema>> = None;
        let mut current_writer: Option<
            Arc<tokio::sync::RwLock<hermes_core::IndexWriter<hermes_core::MmapDirectory>>>,
        > = None;
        let mut current_index_name: Option<String> = None;

        // Buffer messages and batch-convert off the async runtime to avoid
        // blocking tokio threads with CPU-bound proto → Document conversion.
        const STREAM_BATCH_SIZE: usize = 64;
        let mut batch: Vec<IndexDocumentRequest> = Vec::with_capacity(STREAM_BATCH_SIZE);

        while let Some(req) = stream.message().await? {
            let needs_switch = current_index_name.as_ref() != Some(&req.index_name);

            // Flush current batch before switching indexes
            if needs_switch && !batch.is_empty() {
                let (count, recycled) = Self::flush_stream_batch(
                    batch,
                    current_schema.as_ref().unwrap(),
                    current_writer.as_ref().unwrap(),
                )
                .await?;
                indexed_count += count;
                batch = recycled;
            }

            if needs_switch {
                let index = self.registry.get_or_open_index(&req.index_name).await?;
                let writer = self.registry.get_writer(&req.index_name).await?;
                current_schema = Some(Arc::clone(index.schema_arc()));
                current_writer = Some(writer);
                current_index_name = Some(req.index_name.clone());
            }

            batch.push(req);

            if batch.len() >= STREAM_BATCH_SIZE {
                let (count, recycled) = Self::flush_stream_batch(
                    batch,
                    current_schema.as_ref().unwrap(),
                    current_writer.as_ref().unwrap(),
                )
                .await?;
                indexed_count += count;
                batch = recycled;
            }
        }

        // Flush remaining batch
        if !batch.is_empty() {
            let (count, _recycled) = Self::flush_stream_batch(
                batch,
                current_schema
                    .as_ref()
                    .ok_or_else(|| Status::internal("No index selected"))?,
                current_writer
                    .as_ref()
                    .ok_or_else(|| Status::internal("No writer selected"))?,
            )
            .await?;
            indexed_count += count;
        }

        Ok(Response::new(IndexDocumentsResponse { indexed_count }))
    }

    async fn commit(
        &self,
        request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let writer = self.registry.get_writer(&req.index_name).await?;

        writer
            .write()
            .await
            .commit()
            .await
            .map_err(|e| Status::internal(format!("Commit failed: {}", e)))?;

        // Get doc count from fresh searcher
        let reader = index
            .reader()
            .await
            .map_err(|e| Status::internal(format!("Failed to get reader: {}", e)))?;
        let searcher = reader
            .searcher()
            .await
            .map_err(|e| Status::internal(format!("Failed to get searcher: {}", e)))?;

        info!("Committed: {}", req.index_name);

        Ok(Response::new(CommitResponse {
            success: true,
            num_docs: searcher.num_docs(),
        }))
    }

    async fn force_merge(
        &self,
        request: Request<ForceMergeRequest>,
    ) -> Result<Response<ForceMergeResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let writer = self.registry.get_writer(&req.index_name).await?;

        writer
            .write()
            .await
            .force_merge()
            .await
            .map_err(|e| Status::internal(format!("Merge failed: {}", e)))?;

        // Get segment count from fresh searcher
        let reader = index
            .reader()
            .await
            .map_err(|e| Status::internal(format!("Failed to get reader: {}", e)))?;
        let searcher = reader
            .searcher()
            .await
            .map_err(|e| Status::internal(format!("Failed to get searcher: {}", e)))?;

        info!("Force merged: {}", req.index_name);

        Ok(Response::new(ForceMergeResponse {
            success: true,
            num_segments: searcher.segment_readers().len() as u32,
        }))
    }

    async fn delete_index(
        &self,
        request: Request<DeleteIndexRequest>,
    ) -> Result<Response<DeleteIndexResponse>, Status> {
        let req = request.into_inner();

        // 1. Evict from registry — atomic, prevents any new operations from
        //    obtaining references to this index.
        let handle = self.registry.evict(&req.index_name);

        // 2. If the index was open, flush and wait for in-flight operations.
        //    The write lock waits for all concurrent read locks (batch_index, etc.)
        //    to finish, then holds exclusive access through file deletion.
        if let Some(handle) = handle {
            let mut w = handle.writer.write().await;
            if let Err(e) = w.commit().await {
                warn!(
                    "Error committing writer during delete: index={}, error={}",
                    req.index_name, e
                );
            }
            w.wait_for_merging_thread().await;
        }

        // 3. Delete directory — safe because:
        //    - No new operations can obtain the index (evicted from registry)
        //    - All in-flight writes finished (write lock acquired above)
        let index_path = self.registry.data_dir.join(&req.index_name);
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)
                .map_err(|e| Status::internal(format!("Failed to delete index: {}", e)))?;
        }

        info!("Deleted index: {}", req.index_name);

        Ok(Response::new(DeleteIndexResponse { success: true }))
    }

    async fn list_indexes(
        &self,
        _request: Request<ListIndexesRequest>,
    ) -> Result<Response<ListIndexesResponse>, Status> {
        let index_names = self.registry.list_indexes();

        info!("Listed indexes: count={}", index_names.len());

        Ok(Response::new(ListIndexesResponse { index_names }))
    }

    async fn retrain_vector_index(
        &self,
        request: Request<RetrainVectorIndexRequest>,
    ) -> Result<Response<RetrainVectorIndexResponse>, Status> {
        let req = request.into_inner();
        let _index = self.registry.get_or_open_index(&req.index_name).await?;
        let writer = self.registry.get_writer(&req.index_name).await?;

        writer
            .write()
            .await
            .rebuild_vector_index()
            .await
            .map_err(|e| Status::internal(format!("Retrain failed: {}", e)))?;

        info!("Retrained vector index: {}", req.index_name);

        Ok(Response::new(RetrainVectorIndexResponse { success: true }))
    }
}
