//! Index service gRPC implementation

use std::sync::Arc;

use log::{debug, info, warn};
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
    /// Returns (indexed_count, errors, recycled_batch_vec).
    async fn flush_stream_batch(
        batch: Vec<IndexDocumentRequest>,
        schema: &Arc<hermes_core::Schema>,
        writer: &Arc<tokio::sync::RwLock<hermes_core::IndexWriter<hermes_core::MmapDirectory>>>,
    ) -> Result<(u32, Vec<DocumentError>, Vec<IndexDocumentRequest>), Status> {
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

        let mut count = 0u32;
        let mut errors = Vec::new();
        let total_docs = docs.len();
        let w = writer.read().await;
        for (i, doc) in docs.into_iter().enumerate() {
            match w.add_document(doc) {
                Ok(()) => count += 1,
                Err(hermes_core::Error::DuplicatePrimaryKey(key)) => {
                    errors.push(DocumentError {
                        index: i as u32,
                        error: format!("Duplicate primary key: {}", key),
                    });
                }
                Err(hermes_core::Error::QueueFull) => {
                    warn!(
                        "QueueFull during stream batch: indexed {}/{} docs before backpressure",
                        count, total_docs
                    );
                    break;
                }
                Err(e) => {
                    errors.push(DocumentError {
                        index: i as u32,
                        error: e.to_string(),
                    });
                }
            }
        }
        Ok((count, errors, recycled))
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

        // Index documents individually to collect per-document errors (e.g. duplicate PK)
        let mut indexed_count = 0u32;
        let mut doc_errors = Vec::new();
        let total_docs = documents.len();
        {
            let w = writer.read().await;
            for (i, doc) in documents.into_iter().enumerate() {
                match w.add_document(doc) {
                    Ok(()) => indexed_count += 1,
                    Err(hermes_core::Error::DuplicatePrimaryKey(key)) => {
                        doc_errors.push(DocumentError {
                            index: i as u32,
                            error: format!("Duplicate primary key: {}", key),
                        });
                    }
                    Err(hermes_core::Error::QueueFull) => {
                        let skipped = total_docs - i;
                        warn!(
                            "QueueFull during batch_index: index={}, indexed {}/{} docs, {} skipped",
                            req.index_name, indexed_count, total_docs, skipped
                        );
                        doc_errors.push(DocumentError {
                            index: i as u32,
                            error: format!("Queue full — {} remaining documents skipped", skipped),
                        });
                        break;
                    }
                    Err(e) => {
                        doc_errors.push(DocumentError {
                            index: i as u32,
                            error: e.to_string(),
                        });
                    }
                }
            }
        }

        let error_count = conversion_errors + doc_errors.len() as u32;

        debug!(
            "Batch indexed documents: index={}, indexed={}, errors={}",
            req.index_name, indexed_count, error_count
        );

        Ok(Response::new(BatchIndexDocumentsResponse {
            indexed_count,
            error_count,
            errors: doc_errors,
        }))
    }

    async fn index_documents(
        &self,
        request: Request<tonic::Streaming<IndexDocumentRequest>>,
    ) -> Result<Response<IndexDocumentsResponse>, Status> {
        let mut stream = request.into_inner();
        let mut indexed_count = 0u32;
        let mut all_errors = Vec::new();
        let mut current_schema: Option<Arc<hermes_core::Schema>> = None;
        let mut current_writer: Option<
            Arc<tokio::sync::RwLock<hermes_core::IndexWriter<hermes_core::MmapDirectory>>>,
        > = None;
        let mut current_index_name: Option<String> = None;

        // Buffer messages and batch-convert off the async runtime to avoid
        // blocking tokio threads with CPU-bound proto → Document conversion.
        const STREAM_BATCH_SIZE: usize = 512;
        let mut batch: Vec<IndexDocumentRequest> = Vec::with_capacity(STREAM_BATCH_SIZE);

        while let Some(req) = stream.message().await? {
            let needs_switch = current_index_name.as_ref() != Some(&req.index_name);

            // Flush current batch before switching indexes
            if needs_switch && !batch.is_empty() {
                let (count, errors, recycled) = Self::flush_stream_batch(
                    batch,
                    current_schema
                        .as_ref()
                        .ok_or_else(|| Status::internal("No schema for current index"))?,
                    current_writer
                        .as_ref()
                        .ok_or_else(|| Status::internal("No writer for current index"))?,
                )
                .await?;
                indexed_count += count;
                all_errors.extend(errors);
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
                let (count, errors, recycled) = Self::flush_stream_batch(
                    batch,
                    current_schema
                        .as_ref()
                        .ok_or_else(|| Status::internal("No schema for current index"))?,
                    current_writer
                        .as_ref()
                        .ok_or_else(|| Status::internal("No writer for current index"))?,
                )
                .await?;
                indexed_count += count;
                all_errors.extend(errors);
                batch = recycled;
            }
        }

        // Flush remaining batch
        if !batch.is_empty() {
            let (count, errors, _recycled) = Self::flush_stream_batch(
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
            all_errors.extend(errors);
        }

        Ok(Response::new(IndexDocumentsResponse {
            indexed_count,
            errors: all_errors,
        }))
    }

    async fn commit(
        &self,
        request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        let req = request.into_inner();
        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let writer = self.registry.get_writer(&req.index_name).await?;

        let changed = writer
            .write()
            .await
            .commit()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        // Force reader reload to pick up newly committed segments.
        // Without this, the 1-second debounce in reader.searcher() would
        // return the stale pre-commit searcher.
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        if changed {
            reader
                .reload()
                .await
                .map_err(crate::error::hermes_error_to_status)?;
        }
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

        info!("Committed: {} (changed={})", req.index_name, changed);

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
            .map_err(crate::error::hermes_error_to_status)?;

        // Force reader reload to pick up merged segments
        let reader = index
            .reader()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        reader
            .reload()
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let searcher = reader
            .searcher()
            .await
            .map_err(crate::error::hermes_error_to_status)?;

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
            tokio::task::spawn_blocking(move || std::fs::remove_dir_all(&index_path))
                .await
                .map_err(|e| Status::internal(format!("Delete task failed: {}", e)))?
                .map_err(|e| Status::internal(format!("Failed to delete index: {}", e)))?;
        }

        info!("Deleted index: {}", req.index_name);

        Ok(Response::new(DeleteIndexResponse { success: true }))
    }

    async fn list_indexes(
        &self,
        _request: Request<ListIndexesRequest>,
    ) -> Result<Response<ListIndexesResponse>, Status> {
        let index_names = self.registry.list_indexes().await?;

        debug!("Listed indexes: count={}", index_names.len());

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
            .map_err(crate::error::hermes_error_to_status)?;

        info!("Retrained vector index: {}", req.index_name);

        Ok(Response::new(RetrainVectorIndexResponse { success: true }))
    }
}
