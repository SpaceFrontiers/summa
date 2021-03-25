//! Index GRPC API
//!
//! Index GRPC API is using for managing indices

use crate::errors::SummaResult;
use crate::proto;
use crate::requests::CreateIndexRequest;
use crate::services::{AliasService, IndexService};

use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct IndexApiImpl {
    alias_service: AliasService,
    index_service: IndexService,
}

impl IndexApiImpl {
    pub fn new(alias_service: &AliasService, index_service: &IndexService) -> SummaResult<IndexApiImpl> {
        Ok(IndexApiImpl {
            alias_service: alias_service.clone(),
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::index_api_server::IndexApi for IndexApiImpl {
    async fn commit_index(&self, request: Request<proto::CommitIndexRequest>) -> Result<Response<proto::CommitIndexResponse>, Status> {
        let request = request.into_inner();
        let index_updater = self.index_service.get_index_holder(&request.index_name)?.index_updater().clone();
        tokio::spawn(async move {
            let mut index_updater = index_updater.write();
            index_updater.commit(true).await.unwrap();
        });
        let response = proto::CommitIndexResponse {};
        Ok(Response::new(response))
    }

    async fn index_document(&self, request: Request<proto::IndexDocumentRequest>) -> Result<Response<proto::IndexDocumentResponse>, Status> {
        let request = request.into_inner();
        self.index_service
            .get_index_holder(&request.index_name)?
            .index_updater()
            .read()
            .index_document(&request.document, request.reindex)
            .await?;
        let response = proto::IndexDocumentResponse {};
        Ok(Response::new(response))
    }

    async fn create_index(&self, proto_request: Request<proto::CreateIndexRequest>) -> Result<Response<proto::CreateIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let create_index_request = CreateIndexRequest::from_proto(&proto_request)?;
        self.index_service.create_index(create_index_request).await?;
        let response = proto::CreateIndexResponse {};
        Ok(Response::new(response))
    }

    async fn delete_index(&self, proto_request: Request<proto::DeleteIndexRequest>) -> Result<Response<proto::DeleteIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let delete_index_result = self.index_service.delete_index(&proto_request.index_name, proto_request.cascade).await?;
        let response = proto::DeleteIndexResponse {
            deleted_index_aliases: delete_index_result.deleted_aliases,
            deleted_consumer_names: delete_index_result.deleted_consumers,
        };
        Ok(Response::new(response))
    }

    async fn get_index(&self, proto_request: Request<proto::GetIndexRequest>) -> Result<Response<proto::GetIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        self.index_service.get_index_holder(&proto_request.index_name)?;
        let response = proto::GetIndexResponse {
            index: Some(proto::Index {
                index_aliases: self.alias_service.get_index_aliases_for_index(&proto_request.index_name),
                index_name: proto_request.index_name.clone(),
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_indices(&self, _: Request<proto::GetIndicesRequest>) -> Result<Response<proto::GetIndicesResponse>, Status> {
        let response = proto::GetIndicesResponse {
            indices: self
                .index_service
                .index_holders()
                .values()
                .map(|index_holder| proto::Index {
                    index_aliases: self.alias_service.get_index_aliases_for_index(&index_holder.index_name()),
                    index_name: index_holder.index_name().to_string(),
                })
                .collect(),
        };
        Ok(Response::new(response))
    }

    async fn get_index_aliases(&self, _: Request<proto::GetIndexAliasesRequest>) -> Result<Response<proto::GetIndexAliasesResponse>, Status> {
        let response = proto::GetIndexAliasesResponse {
            index_aliases: self.alias_service.index_aliases().clone(),
        };
        Ok(Response::new(response))
    }

    async fn set_index_alias(&self, proto_request: Request<proto::SetIndexAliasRequest>) -> Result<Response<proto::SetIndexAliasResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let old_index_name = self.alias_service.set_index_alias(&proto_request.index_alias, &proto_request.index_name);
        let response = proto::SetIndexAliasResponse { old_index_name };
        Ok(Response::new(response))
    }
}
