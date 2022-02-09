//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use std::error::Error;
use std::io::ErrorKind;
use std::sync::Arc;
use std::time::Instant;

use summa_core::components::{IndexHolder, SummaDocument};
use summa_core::configs::ConfigProxy;
use summa_core::utils::sync::Handler;
use summa_proto::proto;
use summa_proto::proto::DocumentsResponse;
use tantivy::SegmentId;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{error, info, info_span, warn};
use tracing_futures::Instrument;

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
        let request = proto_request.into_inner();
        match proto::CommitMode::from_i32(request.commit_mode) {
            None | Some(proto::CommitMode::Async) => {
                let index_service = self.index_service.clone();
                let index_holder = index_service.get_index_holder(&request.index_alias).await?;
                tokio::spawn(async move {
                    if let Err(err) = index_service.commit_and_restart_consumption(&index_holder).await {
                        warn!(action = "busy", error = format!("{err:?}"))
                    }
                });
                Ok(Response::new(proto::CommitIndexResponse { elapsed_secs: None }))
            }
            Some(proto::CommitMode::Sync) => {
                let now = Instant::now();
                let index_holder = self.index_service.get_index_holder(&request.index_alias).await?;
                self.index_service.commit_and_restart_consumption(&index_holder).await?;
                Ok(Response::new(proto::CommitIndexResponse {
                    elapsed_secs: Some(now.elapsed().as_secs_f64()),
                }))
            }
        }
    }

    async fn copy_documents(&self, proto_request: Request<proto::CopyDocumentsRequest>) -> Result<Response<proto::CopyDocumentsResponse>, Status> {
        let copy_documents_request = proto_request.into_inner();
        self.index_service
            .copy_documents(copy_documents_request)
            .await
            .map_err(crate::errors::Error::from)?;
        let response = proto::CopyDocumentsResponse {};
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
            .get_index_holder(&request.index_alias)
            .await?
            .delete_documents(request.query.ok_or(ValidationError::MissingQuery)?)
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
        let (sender, receiver) = tokio::sync::mpsc::channel(128);
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        let schema = index_holder.schema().clone();
        let documents_receiver = index_holder.documents().map_err(crate::errors::Error::from)?;
        tokio::task::spawn(async move {
            while let Ok(document) = documents_receiver.as_async().recv().await {
                if sender
                    .send(Ok(DocumentsResponse {
                        document: schema.to_json(&document?),
                    }))
                    .await
                    .is_err()
                {
                    info!(action = "grpc_client_dropped");
                    return Ok::<_, crate::errors::Error>(());
                }
            }
            Ok(())
        });
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    async fn get_indices_aliases(&self, _: Request<proto::GetIndicesAliasesRequest>) -> Result<Response<proto::GetIndicesAliasesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesAliasesResponse {
            indices_aliases: self.server_config.read().await.get().core.aliases.clone(),
        }))
    }

    async fn get_index(&self, proto_request: Request<proto::GetIndexRequest>) -> Result<Response<proto::GetIndexResponse>, Status> {
        let index_holder = self.index_service.get_index_holder(&proto_request.into_inner().index_alias).await?;
        let index_description = self.get_index_description(&index_holder).await;
        Ok(Response::new(proto::GetIndexResponse {
            index: Some(index_description),
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
                    info!(action = "received_chunk", index_alias = chunk.index_alias, documents = ?chunk.documents.len());
                    let now = Instant::now();
                    let index_holder = self.index_service.get_index_holder(&chunk.index_alias).await?;
                    let (success_bulk_docs, failed_bulk_docs) = index_holder.index_bulk(&chunk.documents).await.map_err(crate::errors::Error::from)?;
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
        let request = request.into_inner();
        self.index_service
            .get_index_holder(&request.index_alias)
            .await?
            .index_document(SummaDocument::UnboundJsonBytes(&request.document))
            .await
            .map_err(crate::errors::Error::from)?;
        let response = proto::IndexDocumentResponse {};
        Ok(Response::new(response))
    }

    async fn merge_segments(&self, proto_request: Request<proto::MergeSegmentsRequest>) -> Result<Response<proto::MergeSegmentsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        let mut index_writer_holder = index_holder
            .index_writer_holder()
            .map_err(crate::errors::Error::from)?
            .clone()
            .write_owned()
            .await;
        let index_name = index_holder.index_name().to_string();
        tokio::task::spawn_blocking(move || {
            {
                let segment_ids: Vec<_> = proto_request
                    .segment_ids
                    .iter()
                    .map(|segment_id| SegmentId::from_uuid_string(segment_id))
                    .collect::<Result<Vec<_>, _>>()
                    .expect("wrong uuid");
                let result = index_writer_holder.merge(&segment_ids, None);
                if let Err(error) = result {
                    error!(error = ?error)
                }
            }
            .instrument(info_span!("merge", index_name = ?index_name))
        });
        let response = proto::MergeSegmentsResponse {};
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
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        let mut index_writer_holder = index_holder
            .index_writer_holder()
            .map_err(crate::errors::Error::from)?
            .clone()
            .write_owned()
            .await;
        let index_name = index_holder.index_name().to_string();
        tokio::task::spawn_blocking(move || {
            {
                let result = index_writer_holder.vacuum(None);
                if let Err(error) = result {
                    error!(error = ?error)
                }
            }
            .instrument(info_span!("vacuum", index_name = ?index_name))
        });
        let response = proto::VacuumIndexResponse {};
        Ok(Response::new(response))
    }

    async fn warmup_index(&self, proto_request: Request<proto::WarmupIndexRequest>) -> Result<Response<proto::WarmupIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        let now = Instant::now();
        match proto_request.is_full {
            true => index_holder.full_warmup().await.map_err(crate::errors::Error::from)?,
            false => index_holder.partial_warmup().await.map_err(crate::errors::Error::from)?,
        }
        let elapsed_secs = now.elapsed().as_secs_f64();
        let response = proto::WarmupIndexResponse { elapsed_secs };
        Ok(Response::new(response))
    }
}
