//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use crate::errors::{SummaResult, ValidationError};
use crate::proto;
use crate::requests::{CreateIndexRequest, DeleteIndexRequest};
use crate::services::IndexService;
use tantivy::SegmentId;

use crate::configs::ApplicationConfigHolder;
use crate::search_engine::SummaDocument;
use tonic::{Request, Response, Status};

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
        match proto::CommitMode::from_i32(request.commit_mode) {
            None | Some(proto::CommitMode::Async) => {
                tokio::spawn(async move { index_holder.try_commit_and_log(true).await });
                Ok(Response::new(proto::CommitIndexResponse { opstamp: None }))
            }
            Some(proto::CommitMode::Sync) => {
                let opstamp = index_holder.commit().await?;
                Ok(Response::new(proto::CommitIndexResponse { opstamp: Some(opstamp) }))
            }
        }
    }

    async fn index_document(&self, request: Request<proto::IndexDocumentRequest>) -> Result<Response<proto::IndexDocumentResponse>, Status> {
        let request = request.into_inner();
        let opstamp = self
            .index_service
            .get_index_holder(&request.index_name)?
            .index_document(SummaDocument::UnboundJsonBytes(&request.document), request.reindex)?;
        let response = proto::IndexDocumentResponse { opstamp };
        Ok(Response::new(response))
    }

    async fn create_index(&self, proto_request: Request<proto::CreateIndexRequest>) -> Result<Response<proto::CreateIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let create_index_request = CreateIndexRequest::from_proto(&proto_request)?;
        let index_name = create_index_request.index_name.clone();
        let index_config = create_index_request.index_config.clone();
        self.index_service.create_index(create_index_request).await?;
        let response = proto::CreateIndexResponse {
            index: Some(proto::Index {
                index_aliases: vec![],
                index_name,
                index_engine: format!("{:?}", &index_config.index_engine),
            }),
        };
        Ok(Response::new(response))
    }

    async fn delete_index(&self, proto_request: Request<proto::DeleteIndexRequest>) -> Result<Response<proto::DeleteIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let delete_index_request = DeleteIndexRequest::from_proto(proto_request)?;
        let delete_index_result = self.index_service.delete_index(delete_index_request).await?;
        let response = proto::DeleteIndexResponse {
            deleted_index_aliases: delete_index_result.deleted_aliases,
        };
        Ok(Response::new(response))
    }

    async fn get_index(&self, proto_request: Request<proto::GetIndexRequest>) -> Result<Response<proto::GetIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let application_config = self.application_config.read();
        let index_config = application_config
            .indices
            .get(&proto_request.index_name)
            .ok_or(ValidationError::MissingIndexError(proto_request.index_name.to_owned()))?;
        let response = proto::GetIndexResponse {
            index: Some(proto::Index {
                index_aliases: self.application_config.read().get_index_aliases_for_index(&proto_request.index_name),
                index_name: proto_request.index_name.clone(),
                index_engine: format!("{:?}", &index_config.index_engine),
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_indices(&self, _: Request<proto::GetIndicesRequest>) -> Result<Response<proto::GetIndicesResponse>, Status> {
        let application_config = self.application_config.read();
        let response = proto::GetIndicesResponse {
            indices: application_config
                .indices
                .iter()
                .map(|(index_name, index_config)| proto::Index {
                    index_aliases: application_config.get_index_aliases_for_index(index_name),
                    index_name: index_name.to_owned(),
                    index_engine: format!("{:?}", &index_config.index_engine),
                })
                .collect(),
        };
        Ok(Response::new(response))
    }

    async fn get_indices_aliases(&self, _: Request<proto::GetIndicesAliasesRequest>) -> Result<Response<proto::GetIndicesAliasesResponse>, Status> {
        let response = proto::GetIndicesAliasesResponse {
            indices_aliases: self.application_config.read().aliases.clone(),
        };
        Ok(Response::new(response))
    }

    async fn merge_segments(&self, proto_request: Request<proto::MergeSegmentsRequest>) -> Result<Response<proto::MergeSegmentsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_name)?;
        tokio::spawn(async move {
            let segment_ids: Vec<_> = proto_request.segment_ids.iter().map(|x| SegmentId::from_uuid_string(x).unwrap()).collect();
            index_holder.merge(&segment_ids).await
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
