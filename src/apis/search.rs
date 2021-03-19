//! Search GRPC API
//!
//! Search GRPC API is using for querying indices

use crate::errors::SummaResult;
use crate::proto;
use crate::services::IndexService;
use tonic::{Request, Response, Status};

#[derive(Debug)]
pub struct SearchApiImpl {
    index_service: IndexService,
}

impl SearchApiImpl {
    pub fn new(index_service: &IndexService) -> SummaResult<SearchApiImpl> {
        Ok(SearchApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::search_api_server::SearchApi for SearchApiImpl {
    async fn search(&self, proto_request: Request<proto::SearchRequest>) -> Result<Response<proto::SearchResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias)?;
        let search_response = index_holder
            .search(&proto_request.query, proto_request.limit.try_into().unwrap(), proto_request.offset.try_into().unwrap())
            .await?;
        Ok(Response::new(search_response))
    }
}
