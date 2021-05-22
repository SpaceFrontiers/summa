//! Search GRPC API
//!
//! Search GRPC API is using for querying indices

use crate::errors::SummaResult;
use crate::proto;
use crate::services::IndexService;

use std::time::Instant;
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

        let query = match proto_request.query {
            None => proto::Query {
                query: Some(proto::query::Query::All(proto::AllQuery {})),
            },
            Some(query) => query,
        };

        let now = Instant::now();
        let collector_results = index_holder.search(&query, proto_request.collectors).await?;
        let elapsed_secs = now.elapsed().as_secs_f64();
        Ok(Response::new(proto::SearchResponse {
            index_name: index_holder.index_name().to_owned(),
            collector_results,
            elapsed_secs,
        }))
    }
}
