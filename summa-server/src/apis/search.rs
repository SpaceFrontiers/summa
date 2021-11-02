//! Search GRPC API
//!
//! Search GRPC API is using for querying indices

use crate::errors::SummaServerResult;
use crate::services::IndexService;
use std::time::Instant;
use summa_proto::proto;
use tonic::{Request, Response, Status};
use tracing::{info_span, Instrument};

#[derive(Debug)]
pub struct SearchApiImpl {
    index_service: IndexService,
}

impl SearchApiImpl {
    pub fn new(index_service: &IndexService) -> SummaServerResult<SearchApiImpl> {
        Ok(SearchApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::search_api_server::SearchApi for SearchApiImpl {
    async fn search(&self, proto_request: Request<proto::SearchRequest>) -> Result<Response<proto::SearchResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;

        let query = proto_request.query.unwrap_or(proto::Query {
            query: Some(proto::query::Query::All(proto::AllQuery {})),
        });
        let now = Instant::now();

        let collector_outputs = index_holder
            .search(&proto_request.index_alias, &query, proto_request.collectors)
            .instrument(info_span!("search", tags = ?proto_request.tags))
            .await
            .map_err(crate::errors::Error::from)?;
        let elapsed_secs = now.elapsed().as_secs_f64();
        Ok(Response::new(proto::SearchResponse {
            index_name: index_holder.index_name().to_owned(),
            collector_outputs,
            elapsed_secs,
        }))
    }
}
