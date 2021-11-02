//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use crate::errors::SummaServerResult;
use crate::services::IndexService;
use std::error::Error;
use std::io::ErrorKind;
use std::ops::Deref;
use std::time::Instant;
use summa_core::components::{IndexHolder, SummaDocument};
use summa_core::configs::{ApplicationConfigHolder, Persistable};
use summa_proto::proto;
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
    pub fn new(application_config: &ApplicationConfigHolder, index_service: &IndexService) -> SummaServerResult<IndexApiImpl> {
        Ok(IndexApiImpl {
            application_config: application_config.clone(),
            index_service: index_service.clone(),
        })
    }
}

async fn cast(index_holder: &IndexHolder) -> proto::Index {
    let index_config_proxy = index_holder.index_config_proxy().read().await;
    proto::Index {
        //ToDo
        index_aliases: vec![],
        index_name: index_holder.index_name().to_owned(),
        index_engine: format!("{:?}", index_config_proxy.get().index_engine),
        num_docs: index_holder.index_reader().searcher().num_docs(),
        compression: index_holder.compression() as i32,
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
    async fn alter_index(&self, proto_request: Request<proto::AlterIndexRequest>) -> Result<Response<proto::AlterIndexResponse>, Status> {
        let alter_index_request = proto_request.into_inner().try_into()?;
        let index_holder = self.index_service.alter_index(alter_index_request).await?;
        let response = proto::AlterIndexResponse {
            index: Some(cast(index_holder.deref()).await),
        };
        Ok(Response::new(response))
    }

    async fn attach_index(&self, proto_request: Request<proto::AttachIndexRequest>) -> Result<Response<proto::AttachIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.attach_index(&proto_request.index_name).await?;
        let response = proto::AttachIndexResponse {
            index: Some(cast(index_holder.deref()).await),
        };
        Ok(Response::new(response))
    }

    async fn commit_index(&self, request: Request<proto::CommitIndexRequest>) -> Result<Response<proto::CommitIndexResponse>, Status> {
        let request = request.into_inner();
        let index_holder = self.index_service.get_index_holder(&request.index_alias).await?;
        match proto::CommitMode::from_i32(request.commit_mode) {
            None | Some(proto::CommitMode::Async) => {
                tokio::spawn(async move {
                    match index_holder.index_updater().try_write() {
                        Err(_) => warn!(action = "busy"),
                        Ok(index_updater) => match index_updater.commit(None).await {
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
            Some(proto::CommitMode::Sync) => match index_holder.index_updater().try_write() {
                Err(_) => Err(Status::failed_precondition("busy")),
                Ok(index_updater) => {
                    let now = Instant::now();
                    let opstamp = index_updater.commit(None).await.map_err(crate::errors::Error::from)?;
                    Ok(Response::new(proto::CommitIndexResponse {
                        opstamp: Some(opstamp),
                        elapsed_secs: Some(now.elapsed().as_secs_f64()),
                    }))
                }
            },
        }
    }

    async fn create_index(&self, proto_request: Request<proto::CreateIndexRequest>) -> Result<Response<proto::CreateIndexResponse>, Status> {
        let index_holder = self.index_service.create_index(proto_request.into_inner().try_into()?).await?;
        let response = proto::CreateIndexResponse {
            index: Some(cast(index_holder.deref()).await),
        };
        Ok(Response::new(response))
    }

    async fn delete_document(&self, request: Request<proto::DeleteDocumentRequest>) -> Result<Response<proto::DeleteDocumentResponse>, Status> {
        let request = request.into_inner();
        let opstamp = self
            .index_service
            .get_index_holder(&request.index_alias)
            .await?
            .index_updater()
            .read()
            .await
            .delete_document(request.primary_key)
            .await
            .map_err(crate::errors::Error::from)?;
        let response = proto::DeleteDocumentResponse { opstamp: Some(opstamp) };
        Ok(Response::new(response))
    }

    async fn index_document(&self, request: Request<proto::IndexDocumentRequest>) -> Result<Response<proto::IndexDocumentResponse>, Status> {
        let request = request.into_inner();
        let opstamp = self
            .index_service
            .get_index_holder(&request.index_alias)
            .await?
            .index_updater()
            .read()
            .await
            .index_document(SummaDocument::UnboundJsonBytes(&request.document))
            .await
            .map_err(crate::errors::Error::from)?;
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
                        .get_index_holder(&chunk.index_alias)
                        .await?
                        .index_updater()
                        .read()
                        .await
                        .index_bulk(&chunk.documents)
                        .await;
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

    async fn delete_index(&self, proto_request: Request<proto::DeleteIndexRequest>) -> Result<Response<proto::DeleteIndexResponse>, Status> {
        Ok(Response::new(self.index_service.delete_index(proto_request.into_inner().into()).await?.into()))
    }

    async fn get_index(&self, proto_request: Request<proto::GetIndexRequest>) -> Result<Response<proto::GetIndexResponse>, Status> {
        Ok(Response::new(proto::GetIndexResponse {
            index: Some(cast(self.index_service.get_index_holder(&proto_request.into_inner().index_alias).await?.deref()).await),
        }))
    }

    async fn get_indices(&self, _: Request<proto::GetIndicesRequest>) -> Result<Response<proto::GetIndicesResponse>, Status> {
        let index_holders = self.index_service.index_holders().await;
        let mut indices = Vec::with_capacity(index_holders.len());
        for index_holder in index_holders.values() {
            indices.push(cast(index_holder).await)
        }
        Ok(Response::new(proto::GetIndicesResponse { indices }))
    }

    async fn get_indices_aliases(&self, _: Request<proto::GetIndicesAliasesRequest>) -> Result<Response<proto::GetIndicesAliasesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesAliasesResponse {
            indices_aliases: self.application_config.read().await.aliases.clone(),
        }))
    }

    async fn merge_segments(&self, proto_request: Request<proto::MergeSegmentsRequest>) -> Result<Response<proto::MergeSegmentsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        tokio::spawn(async move {
            let index_name = index_holder.index_name();
            let segment_ids: Vec<_> = proto_request.segment_ids.iter().map(|x| SegmentId::from_uuid_string(x).unwrap()).collect();
            index_holder
                .index_updater()
                .read()
                .await
                .merge_index(&segment_ids, None)
                .instrument(info_span!("merge", index_name = ?index_name))
                .await
                .unwrap()
        });
        let response = proto::MergeSegmentsResponse {};
        Ok(Response::new(response))
    }

    async fn set_index_alias(&self, proto_request: Request<proto::SetIndexAliasRequest>) -> Result<Response<proto::SetIndexAliasResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let mut application_config = self.application_config.write().await;
        let old_index_name = application_config
            .set_index_alias(&proto_request.index_alias, &proto_request.index_name)
            .map_err(crate::errors::Error::from)?;
        application_config.save().map_err(crate::errors::Error::from)?;
        let response = proto::SetIndexAliasResponse { old_index_name };
        Ok(Response::new(response))
    }

    async fn vacuum_index(&self, proto_request: Request<proto::VacuumIndexRequest>) -> Result<Response<proto::VacuumIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        tokio::spawn(async move {
            let index_name = index_holder.index_name();
            index_holder
                .index_updater()
                .read()
                .await
                .vacuum_index(None, None)
                .instrument(info_span!("vacuum", index_name = ?index_name))
                .await
                .unwrap();
        });
        let response = proto::VacuumIndexResponse {};
        Ok(Response::new(response))
    }
}
