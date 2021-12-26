//! Search GRPC API
//!
//! Search GRPC API is using for querying indices

use std::time::Instant;

use summa_proto::proto;
use tonic::{Request, Response, Status};
use tracing::{info_span, Instrument};

use crate::errors::SummaServerResult;
use crate::services::Index;

pub struct SearchApiImpl {
    index_service: Index,
}

impl SearchApiImpl {
    pub fn new(index_service: &Index) -> SummaServerResult<SearchApiImpl> {
        Ok(SearchApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::search_api_server::SearchApi for SearchApiImpl {
    async fn search(&self, proto_request: Request<proto::SearchRequest>) -> Result<Response<proto::SearchResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let now = Instant::now();
        let tags = proto_request.tags.clone();
        let collector_outputs = self
            .index_service
            .search(proto_request)
            .instrument(info_span!("search", tags = ?tags))
            .await
            .map_err(crate::errors::Error::from)?;

        let elapsed_secs = now.elapsed().as_secs_f64();
        Ok(Response::new(proto::SearchResponse {
            collector_outputs,
            elapsed_secs,
        }))
    }
}
