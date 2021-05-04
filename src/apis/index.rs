//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use crate::configs::ApplicationConfigHolder;
use crate::errors::SummaResult;
use crate::proto;
use crate::search_engine::SummaDocument;
use crate::services::IndexService;
use std::ops::Deref;
use tantivy::SegmentId;
use tonic::{Request, Response, Status};
use tracing::warn;

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

#[tonic::async_trait]
impl proto::index_api_server::IndexApi for IndexApiImpl {
    async fn commit_index(&self, request: Request<proto::CommitIndexRequest>) -> Result<Response<proto::CommitIndexResponse>, Status> {
        let request = request.into_inner();
        let index_holder = self.index_service.get_index_holder(&request.index_name)?;
        let index_updater = index_holder.index_updater();
        match proto::CommitMode::from_i32(request.commit_mode) {
            None | Some(proto::CommitMode::Async) => {
                tokio::spawn(async move {
                    match index_updater.try_write() {
                        None => warn!(action = "busy"),
                        Some(mut index_updater) => match index_updater.commit().await {
                            Ok(_) => (),
                            Err(error) => warn!(error = ?error),
                        },
                    }
                });
                Ok(Response::new(proto::CommitIndexResponse {
                    opstamp: None,
                    status: "ok".to_owned(),
                }))
            }
            Some(proto::CommitMode::Sync) => match index_updater.try_write() {
                None => Ok(Response::new(proto::CommitIndexResponse {
                    opstamp: None,
                    status: "busy".to_owned(),
                })),
                Some(mut index_updater) => Ok(Response::new(proto::CommitIndexResponse {
                    opstamp: Some(index_updater.commit().await?),
                    status: "ok".to_owned(),
                })),
            },
        }
    }

    async fn index_document(&self, request: Request<proto::IndexDocumentRequest>) -> Result<Response<proto::IndexDocumentResponse>, Status> {
        let request = request.into_inner();
        let opstamp = self
            .index_service
            .get_index_holder(&request.index_name)?
            .index_updater()
            .read()
            .index_document(SummaDocument::UnboundJsonBytes(&request.document))?;
        let response = proto::IndexDocumentResponse { opstamp };
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
            index: Some(self.index_service.get_index_holder(&proto_request.into_inner().index_name)?.deref().into()),
        }))
    }

    async fn get_indices(&self, _: Request<proto::GetIndicesRequest>) -> Result<Response<proto::GetIndicesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesResponse {
            indices: self.index_service.index_holders().values().map(|index_holder| index_holder.deref().into()).collect(),
        }))
    }

    async fn get_indices_aliases(&self, _: Request<proto::GetIndicesAliasesRequest>) -> Result<Response<proto::GetIndicesAliasesResponse>, Status> {
        Ok(Response::new(proto::GetIndicesAliasesResponse {
            indices_aliases: self.application_config.read().aliases.clone(),
        }))
    }

    async fn merge_segments(&self, proto_request: Request<proto::MergeSegmentsRequest>) -> Result<Response<proto::MergeSegmentsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_name)?;
        tokio::spawn(async move {
            let segment_ids: Vec<_> = proto_request.segment_ids.iter().map(|x| SegmentId::from_uuid_string(x).unwrap()).collect();
            index_holder.index_updater().write().merge(&segment_ids).await
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
        let garbage_collection_result = self.index_service.vacuum_index(&proto_request.index_name).await?;
        let response = proto::VacuumIndexResponse {
            deleted_files: garbage_collection_result.deleted_files.iter().map(|x| x.to_string_lossy().to_string()).collect(),
        };
        Ok(Response::new(response))
    }
}
