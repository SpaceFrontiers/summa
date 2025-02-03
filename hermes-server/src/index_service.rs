//! Index service gRPC implementation

use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::info;

use hermes_core::parse_schema;

use crate::converters::convert_proto_to_document;
use crate::proto::index_service_server::IndexService;
use crate::proto::*;
use crate::registry::IndexRegistry;

/// Index service implementation
pub struct IndexServiceImpl {
    pub registry: Arc<IndexRegistry>,
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

        info!(index_name = %req.index_name, "Created index");

        Ok(Response::new(CreateIndexResponse { success: true }))
    }

    async fn batch_index_documents(
        &self,
        request: Request<BatchIndexDocumentsRequest>,
    ) -> Result<Response<BatchIndexDocumentsResponse>, Status> {
        let req = request.into_inner();

        let index = self.registry.get_or_open_index(&req.index_name).await?;
        let writer = self.registry.get_writer(&req.index_name).await?;
        let schema = index.schema();

        // Convert all documents first (this is CPU-bound, could be parallelized further)
        let mut documents = Vec::with_capacity(req.documents.len());
        let mut conversion_errors = 0u32;

        for named_doc in req.documents {
            match convert_proto_to_document(&named_doc.fields, schema) {
                Ok(doc) => documents.push(doc),
                Err(_) => conversion_errors += 1,
            }
        }

        // Use cached writer - lock only for the add_documents call
        let doc_count = documents.len() as u32;
        let indexed_count = {
            let w = writer.lock().await;
            w.add_documents(documents).unwrap_or(0) as u32
        };
        let index_errors = doc_count - indexed_count;

        let error_count = conversion_errors + index_errors;

        info!(
            index_name = %req.index_name,
            indexed = indexed_count,
            errors = error_count,
            "Batch indexed documents"
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
        let mut current_index: Option<Arc<hermes_core::Index<hermes_core::FsDirectory>>> = None;
        let mut current_writer: Option<
            Arc<tokio::sync::Mutex<hermes_core::IndexWriter<hermes_core::FsDirectory>>>,
        > = None;
        let mut current_index_name: Option<String> = None;

        while let Some(req) = stream.message().await? {
            // Get or switch index/writer if index changed
            let needs_switch = current_index_name.as_ref() != Some(&req.index_name);

            if needs_switch {
                let index = self.registry.get_or_open_index(&req.index_name).await?;
                let writer = self.registry.get_writer(&req.index_name).await?;
                current_index = Some(index);
                current_writer = Some(writer);
                current_index_name = Some(req.index_name.clone());
            }

            let writer = current_writer
                .as_ref()
                .ok_or_else(|| Status::internal("No index selected for streaming"))?;
            let index = current_index
                .as_ref()
                .ok_or_else(|| Status::internal("No index selected for streaming"))?;

            let doc = convert_proto_to_document(&req.fields, index.schema())
                .map_err(|e| Status::invalid_argument(format!("Invalid document: {}", e)))?;

            writer
                .lock()
                .await
                .add_document(doc)
                .map_err(|e| Status::internal(format!("Failed to index document: {}", e)))?;

            indexed_count += 1;
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
            .lock()
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

        info!(index_name = %req.index_name, "Committed");

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
            .lock()
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

        info!(index_name = %req.index_name, "Force merged");

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

        // Wait for any pending merges to complete before deleting
        if let Some(writer) = self.registry.get_existing_writer(&req.index_name) {
            writer.lock().await.wait_for_merges().await;
        }

        // Remove writer and index from registry
        self.registry.remove(&req.index_name);

        // Delete directory
        let index_path = self.registry.data_dir.join(&req.index_name);
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)
                .map_err(|e| Status::internal(format!("Failed to delete index: {}", e)))?;
        }

        info!(index_name = %req.index_name, "Deleted index");

        Ok(Response::new(DeleteIndexResponse { success: true }))
    }
}
