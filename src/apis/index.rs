//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use crate::configs::ApplicationConfigHolder;
use crate::errors::SummaResult;
use crate::proto;
use crate::search_engine::SummaDocument;
use crate::services::IndexService;
use std::error::Error;
use std::io::ErrorKind;
use std::ops::Deref;
use std::time::Instant;
use tantivy::SegmentId;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};
use tracing::{info_span, warn};
use tracing_futures::Instrument;

#[derive(Clone)]
pub struct IndexApiImpl {
    application_config: ApplicationConfigHolder,
    index_service: IndexService,
}

impl IndexApiImpl {
    pub fn new(application_config: &ApplicationConfigHolder, index_service: &IndexService) -> SummaResult<IndexApiImpl> {
        Ok(IndexApiImpl {
            application_config: application_config.clone(),
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

#[tonic::async_trait]
impl proto::index_api_server::IndexApi for IndexApiImpl {
    async fn commit_index(&self, request: Request<proto::CommitIndexRequest>) -> Result<Response<proto::CommitIndexResponse>, Status> {
        let request = request.into_inner();
        let index_holder = self.index_service.get_index_holder(&request.index_alias)?;
        let index_updater = index_holder.index_updater();
        match proto::CommitMode::from_i32(request.commit_mode) {
            None | Some(proto::CommitMode::Async) => {
                tokio::spawn(async move {
                    match index_updater.try_write() {
                        Err(_) => warn!(action = "busy"),
                        Ok(mut index_updater) => match index_updater.commit().await {
                            Ok(_) => (),
                            Err(error) => warn!(error = ?error),
                        },
                    }
                });
                Ok(Response::new(proto::CommitIndexResponse {
                    opstamp: None,
                    elapsed_secs: None,
                }))
            }
            Some(proto::CommitMode::Sync) => match index_updater.try_write() {
                Err(_) => Err(Status::failed_precondition("busy")),
                Ok(mut index_updater) => {
                    let now = Instant::now();
                    let opstamp = index_updater.commit().await?;
                    Ok(Response::new(proto::CommitIndexResponse {
                        opstamp: Some(opstamp),
                        elapsed_secs: Some(now.elapsed().as_secs_f64()),
                    }))
                }
            },
        }
    }

    async fn delete_document(&self, request: Request<proto::DeleteDocumentRequest>) -> Result<Response<proto::DeleteDocumentResponse>, Status> {
        let request = request.into_inner();
        let opstamp = self
            .index_service
            .get_index_holder(&request.index_alias)?
            .index_updater()
            .read()
            .await
            .delete_document(request.primary_key)?;
        let response = proto::DeleteDocumentResponse { opstamp: Some(opstamp) };
        Ok(Response::new(response))
    }

    async fn index_document(&self, request: Request<proto::IndexDocumentRequest>) -> Result<Response<proto::IndexDocumentResponse>, Status> {
        let request = request.into_inner();
        let opstamp = self
            .index_service
            .get_index_holder(&request.index_alias)?
            .index_updater()
            .read()
            .await
            .index_document(SummaDocument::UnboundJsonBytes(&request.document))?;
        let response = proto::IndexDocumentResponse { opstamp };
        Ok(Response::new(response))
    }

    async fn index_document_stream(
        &self,
        request: Request<Streaming<proto::IndexDocumentStreamRequest>>,
    ) -> Result<Response<proto::IndexDocumentStreamResponse>, Status> {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        let mut elapsed_secs = 0f64;
        let mut in_stream = request.into_inner();
        while let Some(chunk) = in_stream.next().await {
            match chunk {
                Ok(chunk) => {
                    let now = Instant::now();
                    let (success_bulk_docs, failed_bulk_docs) = self
                        .index_service
                        .get_index_holder(&chunk.index_alias)?
                        .index_updater()
                        .read()
                        .await
                        .index_bulk(&chunk.documents);
                    elapsed_secs += now.elapsed().as_secs_f64();
                    success_docs += success_bulk_docs;
                    failed_docs += failed_bulk_docs;
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

    async fn alter_index(&self, proto_request: Request<proto::AlterIndexRequest>) -> Result<Response<proto::AlterIndexResponse>, Status> {
        let alter_index_request = proto_request.into_inner().try_into()?;
        let index_holder = self.index_service.alter_index(alter_index_request).await?;
        let response = proto::AlterIndexResponse {
            index: Some(index_holder.deref().into()),
        };
        Ok(Response::new(response))
    }

    async fn create_index(&self, proto_request: Request<proto::CreateIndexRequest>) -> Result<Response<proto::CreateIndexResponse>, Status> {
        let create_index_request = proto_request.into_inner().try_into()?;
        let index_holder = self.index_service.create_index(create_index_request).await?;
        let response = proto::CreateIndexResponse {
            index: Some(index_holder.deref().into()),
        };
        Ok(Response::new(response))
    }

    async fn delete_index(&self, proto_request: Request<proto::DeleteIndexRequest>) -> Result<Response<proto::DeleteIndexResponse>, Status> {
        Ok(Response::new(self.index_service.delete_index(proto_request.into_inner().into()).await?.into()))
    }

    async fn get_index(&self, proto_request: Request<proto::GetIndexRequest>) -> Result<Response<proto::GetIndexResponse>, Status> {
        Ok(Response::new(proto::GetIndexResponse {
            index: Some(self.index_service.get_index_holder(&proto_request.into_inner().index_alias)?.deref().into()),
        }))
    }

    async fn get_indices(&self, _: Request<proto::GetIndicesRequest>) -> Result<Response<proto::GetIndicesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesResponse {
            indices: self
                .index_service
                .index_holders()
                .values()
                .map(|index_holder| index_holder.deref().into())
                .collect(),
        }))
    }

    async fn get_indices_aliases(&self, _: Request<proto::GetIndicesAliasesRequest>) -> Result<Response<proto::GetIndicesAliasesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesAliasesResponse {
            indices_aliases: self.application_config.read().aliases.clone(),
        }))
    }

    async fn merge_segments(&self, proto_request: Request<proto::MergeSegmentsRequest>) -> Result<Response<proto::MergeSegmentsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias)?;
        tokio::spawn(async move {
            let index_name = index_holder.index_name();
            let segment_ids: Vec<_> = proto_request.segment_ids.iter().map(|x| SegmentId::from_uuid_string(x).unwrap()).collect();
            index_holder
                .index_updater()
                .write()
                .await
                .merge(&segment_ids)
                .instrument(info_span!("merge", index_name = ?index_name))
                .await
                .unwrap()
        });
        let response = proto::MergeSegmentsResponse {};
        Ok(Response::new(response))
    }

    async fn set_index_alias(&self, proto_request: Request<proto::SetIndexAliasRequest>) -> Result<Response<proto::SetIndexAliasResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let old_index_name = self
            .application_config
            .write()
            .autosave()
            .set_index_alias(&proto_request.index_alias, &proto_request.index_name)?;
        let response = proto::SetIndexAliasResponse { old_index_name };
        Ok(Response::new(response))
    }

    async fn vacuum_index(&self, proto_request: Request<proto::VacuumIndexRequest>) -> Result<Response<proto::VacuumIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias)?;
        tokio::spawn(async move {
            let index_name = index_holder.index_name();
            index_holder
                .index_updater()
                .write()
                .await
                .vacuum()
                .instrument(info_span!("vacuum", index_name = ?index_name))
                .await
                .unwrap();
        });
        let response = proto::VacuumIndexResponse {};
        Ok(Response::new(response))
    }
}
