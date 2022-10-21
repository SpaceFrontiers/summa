//! Public GRPC API
//!
//! Public GRPC API is using for querying indices but queries are restricted for making it safe to open endpoint in public

use std::time::Instant;

use summa_proto::proto;
use tonic::{Request, Response, Status};
use tracing::{info_span, Instrument};

use crate::errors::SummaServerResult;
use crate::services::Index;

pub struct PublicApiImpl {
    index_service: Index,
}

impl PublicApiImpl {
    pub fn new(index_service: &Index) -> SummaServerResult<PublicApiImpl> {
        Ok(PublicApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::public_api_server::PublicApi for PublicApiImpl {
    async fn search(&self, proto_request: Request<proto::SearchRequest>) -> Result<Response<proto::SearchResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let now = Instant::now();
        let collector_outputs = self
            .index_service
            .constrained_search(proto_request)
            .instrument(info_span!("search"))
            .await
            .map_err(crate::errors::Error::from)?;
        let elapsed_secs = now.elapsed().as_secs_f64();
        Ok(Response::new(proto::SearchResponse {
            collector_outputs,
            elapsed_secs,
        }))
    }
}
