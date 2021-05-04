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
        let collector_results = index_holder.search(&proto_request.query.unwrap(), &proto_request.collectors).await?;
        Ok(Response::new(proto::SearchResponse {
            index_name: index_holder.index_name().to_owned(),
            collector_results,
        }))

        /*Ok(match search_result {
            Ok(scored_documents) => {
                let right_bound = std::cmp::min(limit, scored_documents.len());
                let has_next = scored_documents.len() > limit;
                Ok(Response::new(proto::SearchResponse {
                    scored_documents: scored_documents[..right_bound].to_vec(),
                    has_next,
                    index_name: index_holder.index_name().to_owned(),
                }))
            }
            Err(err) => Err(err),
        }?);*/

        //let collector = match proto_request.collector {};

        /*match proto_request.order_by.as_ref() {
            "" => {
                let top_docs_collector = TopDocs::with_limit(limit).and_offset(offset);
            }
            field => {
                let top_docs_collector = TopDocs::with_limit(limit).and_offset(offset).order_by_u64_field(
                    index_holder
                        .schema()
                        .get_field(&proto_request.order_by)
                        .ok_or(FieldDoesNotExistError(proto_request.order_by.to_owned()))?,
                );
                Ok(match index_holder.search_u64(&proto_request.query.unwrap(), top_docs_collector).await {
                    Ok(scored_documents) => {
                        let right_bound = std::cmp::min(limit, scored_documents.len());
                        let has_next = scored_documents.len() > limit;
                        Ok(Response::new(proto::SearchResponse {
                            scored_documents: scored_documents[..right_bound].to_vec(),
                            has_next,
                            index_name: index_holder.index_name().to_owned(),
                        }))
                    }
                    Err(err) => Err(err),
                }?)
            }
        }*/
    }
}
