//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use std::collections::HashSet;
use std::error::Error;
use std::io::ErrorKind;
use std::sync::Arc;
use std::time::Instant;

use summa_core::components::{IndexHolder, NamedFieldDocument, SummaDocument};
use summa_core::configs::ConfigProxy;
use summa_core::utils::sync::Handler;
use summa_core::validators;
use summa_proto::proto;
use summa_proto::proto::DocumentsResponse;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info, info_span};

use crate::errors::{SummaServerResult, ValidationError};
use crate::services::Index;

#[derive(Clone)]
pub struct IndexApiImpl {
    server_config: Arc<dyn ConfigProxy<crate::configs::server::Config>>,
    index_service: Index,
}

impl IndexApiImpl {
    pub fn new(server_config: &Arc<dyn ConfigProxy<crate::configs::server::Config>>, index_service: &Index) -> SummaServerResult<IndexApiImpl> {
        Ok(IndexApiImpl {
            server_config: server_config.clone(),
            index_service: index_service.clone(),
        })
    }
}

fn match_for_io_error(err_status: &Status) -> Option<&std::io::Error> {
    let mut err: &(dyn Error + 'static) = err_status;

    loop {
        if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
            return Some(io_err);
        }

        err = match err.source() {
            Some(err) => err,
            None => return None,
        };
    }
}

impl IndexApiImpl {
    pub async fn get_index_description(&self, index_holder: &Handler<IndexHolder>) -> proto::IndexDescription {
        proto::IndexDescription {
            index_aliases: self
                .index_service
                .server_config()
                .read()
                .await
                .get()
                .core
                .get_index_aliases_for_index(index_holder.index_name()),
            index_name: index_holder.index_name().to_owned(),
            index_engine: Some(index_holder.index_engine_config().read().await.get().clone()),
            num_docs: index_holder.index_reader().searcher().num_docs(),
            compression: index_holder.compression() as i32,
            index_attributes: index_holder.index_attributes().cloned(),
        }
    }
}

#[tonic::async_trait]
impl proto::index_api_server::IndexApi for IndexApiImpl {
    async fn attach_index(&self, proto_request: Request<proto::AttachIndexRequest>) -> Result<Response<proto::AttachIndexResponse>, Status> {
        let index_holder = self.index_service.attach_index(proto_request.into_inner()).await?;
        let response = proto::AttachIndexResponse {
            index: Some(self.get_index_description(&index_holder).await),
        };
        Ok(Response::new(response))
    }

    async fn commit_index(&self, proto_request: Request<proto::CommitIndexRequest>) -> Result<Response<proto::CommitIndexResponse>, Status> {
        let now = Instant::now();
        let index_holder = self.index_service.get_index_holder(&proto_request.into_inner().index_name).await?;
        self.index_service.commit_and_restart_consumption(&index_holder).await?;
        Ok(Response::new(proto::CommitIndexResponse {
            elapsed_secs: now.elapsed().as_secs_f64(),
        }))
    }

    async fn copy_documents(&self, proto_request: Request<proto::CopyDocumentsRequest>) -> Result<Response<proto::CopyDocumentsResponse>, Status> {
        let now = Instant::now();
        let copied_documents = self
            .index_service
            .copy_documents(proto_request.into_inner())
            .await
            .map_err(crate::errors::Error::from)?;
        let response = proto::CopyDocumentsResponse {
            elapsed_secs: now.elapsed().as_secs_f64(),
            copied_documents,
        };
        Ok(Response::new(response))
    }

    async fn create_index(&self, proto_request: Request<proto::CreateIndexRequest>) -> Result<Response<proto::CreateIndexResponse>, Status> {
        let index_holder = self.index_service.create_index(proto_request.into_inner()).await?;
        let response = proto::CreateIndexResponse {
            index: Some(self.get_index_description(&index_holder).await),
        };
        Ok(Response::new(response))
    }

    async fn copy_index(&self, proto_request: Request<proto::CopyIndexRequest>) -> Result<Response<proto::CopyIndexResponse>, Status> {
        let index_holder = self.index_service.copy_index(proto_request.into_inner()).await?;
        let response = proto::CopyIndexResponse {
            index: Some(self.get_index_description(&index_holder).await),
        };
        Ok(Response::new(response))
    }

    async fn delete_documents(&self, request: Request<proto::DeleteDocumentsRequest>) -> Result<Response<proto::DeleteDocumentsResponse>, Status> {
        let request = request.into_inner();
        let deleted_documents = self
            .index_service
            .get_index_holder(&request.index_name)
            .await?
            .delete_documents(request.query.and_then(|query| query.query).ok_or(ValidationError::MissingQuery)?)
            .await
            .map_err(crate::errors::Error::from)?;
        let response = proto::DeleteDocumentsResponse { deleted_documents };
        Ok(Response::new(response))
    }

    async fn delete_index(&self, proto_request: Request<proto::DeleteIndexRequest>) -> Result<Response<proto::DeleteIndexResponse>, Status> {
        Ok(Response::new(self.index_service.delete_index(proto_request.into_inner()).await?.into()))
    }

    type documentsStream = ReceiverStream<Result<DocumentsResponse, Status>>;

    async fn documents(&self, proto_request: Request<proto::DocumentsRequest>) -> Result<Response<Self::documentsStream>, Status> {
        let proto_request = proto_request.into_inner();
        let span = info_span!("documents", index_name = proto_request.index_name);
        let _ = span.enter();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_name).await?;
        let searcher = index_holder.index_reader().searcher();
        let schema = index_holder.schema().clone();
        let multi_fields = index_holder.multi_fields().clone();

        let query_fields = validators::parse_fields(searcher.schema(), &proto_request.fields).map_err(crate::errors::Error::from)?;
        let query_fields = (!query_fields.is_empty()).then(|| HashSet::from_iter(query_fields.into_iter().map(|x| x.0)));

        let documents_receiver = index_holder
            .documents(&searcher, move |document| {
                let document = NamedFieldDocument::from_document(&schema, &query_fields, &multi_fields, &document).to_json_string();
                Some(Ok(DocumentsResponse { document }))
            })
            .map_err(crate::errors::Error::from)?;
        Ok(Response::new(ReceiverStream::new(documents_receiver)))
    }

    async fn get_indices_aliases(&self, _: Request<proto::GetIndicesAliasesRequest>) -> Result<Response<proto::GetIndicesAliasesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesAliasesResponse {
            indices_aliases: self.server_config.read().await.get().core.aliases.clone(),
        }))
    }

    async fn get_index(&self, proto_request: Request<proto::GetIndexRequest>) -> Result<Response<proto::GetIndexResponse>, Status> {
        let index_holder = self.index_service.get_index_holder(&proto_request.into_inner().index_name).await?;
        Ok(Response::new(proto::GetIndexResponse {
            index: Some(self.get_index_description(&index_holder).await),
        }))
    }

    async fn get_indices(&self, _: Request<proto::GetIndicesRequest>) -> Result<Response<proto::GetIndicesResponse>, Status> {
        let index_holders = self.index_service.index_registry().index_holders().read().await;
        Ok(Response::new(proto::GetIndicesResponse {
            index_names: index_holders.keys().cloned().collect(),
        }))
    }

    async fn index_document_stream(
        &self,
        request: Request<Streaming<proto::IndexDocumentStreamRequest>>,
    ) -> Result<Response<proto::IndexDocumentStreamResponse>, Status> {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        let mut elapsed_secs = 0f64;
        let mut in_stream = request.into_inner();
        let mut last_status_report = Instant::now();
        while let Some(chunk) = in_stream.next().await {
            match chunk {
                Ok(chunk) => {
                    info!(action = "received_chunk", index_name = chunk.index_name, documents = ?chunk.documents.len());
                    let now = Instant::now();
                    let index_holder = self.index_service.get_index_holder(&chunk.index_name).await?;
                    let (success_bulk_docs, failed_bulk_docs) = index_holder
                        .index_bulk(&chunk.documents, chunk.conflict_strategy.and_then(proto::ConflictStrategy::from_i32))
                        .await
                        .map_err(crate::errors::Error::from)?;
                    elapsed_secs += now.elapsed().as_secs_f64();
                    success_docs += success_bulk_docs;
                    failed_docs += failed_bulk_docs;
                    if last_status_report.elapsed().as_secs_f64() > 60f64 {
                        info!(action = "indexed", success_docs = success_docs, failed_docs = failed_docs);
                        last_status_report = Instant::now();
                    }
                    if self.index_service.should_terminate() {
                        break;
                    }
                }
                Err(err) => {
                    if let Some(io_err) = match_for_io_error(&err) {
                        if io_err.kind() == ErrorKind::BrokenPipe {
                            break;
                        }
                    }
                }
            }
        }
        let response = proto::IndexDocumentStreamResponse {
            success_docs,
            failed_docs,
            elapsed_secs,
        };
        Ok(Response::new(response))
    }

    async fn index_document(&self, request: Request<proto::IndexDocumentRequest>) -> Result<Response<proto::IndexDocumentResponse>, Status> {
        let proto_request = request.into_inner();
        self.index_service
            .get_index_holder(&proto_request.index_name)
            .await?
            .index_document(SummaDocument::UnboundJsonBytes(&proto_request.document))
            .await
            .map_err(crate::errors::Error::from)?;
        let response = proto::IndexDocumentResponse {};
        Ok(Response::new(response))
    }

    async fn merge_segments(&self, proto_request: Request<proto::MergeSegmentsRequest>) -> Result<Response<proto::MergeSegmentsResponse>, Status> {
        let segment_id = self.index_service.merge_segments(proto_request.into_inner()).await?;
        let response = proto::MergeSegmentsResponse {
            segment_id: segment_id.map(|segment_id| segment_id.to_string()),
        };
        Ok(Response::new(response))
    }

    async fn set_index_alias(&self, proto_request: Request<proto::SetIndexAliasRequest>) -> Result<Response<proto::SetIndexAliasResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let mut server_config = self.server_config.write().await;
        let old_index_name = server_config
            .get_mut()
            .core
            .set_index_alias(&proto_request.index_alias, &proto_request.index_name)
            .map_err(crate::errors::Error::from)?;
        server_config.commit().await.map_err(crate::errors::Error::from)?;
        let response = proto::SetIndexAliasResponse { old_index_name };
        Ok(Response::new(response))
    }

    async fn vacuum_index(&self, proto_request: Request<proto::VacuumIndexRequest>) -> Result<Response<proto::VacuumIndexResponse>, Status> {
        let vacuum_index_request = proto_request.into_inner();
        let freed_space_bytes = self.index_service.vacuum_index(vacuum_index_request).await?;
        let response = proto::VacuumIndexResponse { freed_space_bytes };
        Ok(Response::new(response))
    }

    async fn warmup_index(&self, proto_request: Request<proto::WarmupIndexRequest>) -> Result<Response<proto::WarmupIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_name).await?;
        let query_parser_config = index_holder
            .index_engine_config()
            .read()
            .await
            .get()
            .query_parser_config
            .as_ref()
            .cloned()
            .unwrap_or_default();
        let now = Instant::now();
        match proto_request.is_full {
            true => index_holder.full_warmup().await.map_err(crate::errors::Error::from)?,
            false => index_holder
                .partial_warmup(true, &query_parser_config.default_fields)
                .await
                .map_err(crate::errors::Error::from)?,
        }
        let elapsed_secs = now.elapsed().as_secs_f64();
        let response = proto::WarmupIndexResponse { elapsed_secs };
        Ok(Response::new(response))
    }
}
