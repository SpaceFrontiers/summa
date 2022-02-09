//! Reflection GRPC API
//!
//! Reflection GRPC API is using for inspecting indices

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use summa_proto::proto;
use tonic::{Request, Response, Status};

use crate::errors::{SummaServerResult, ValidationError};
use crate::services::Index;

pub struct ReflectionApiImpl {
    index_service: Index,
}

impl ReflectionApiImpl {
    pub fn new(index_service: &Index) -> SummaServerResult<ReflectionApiImpl> {
        Ok(ReflectionApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::reflection_api_server::ReflectionApi for ReflectionApiImpl {
    async fn get_top_terms(&self, proto_request: Request<proto::GetTopTermsRequest>) -> Result<Response<proto::GetTopTermsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_name).await?;
        let field = index_holder.schema().get_field(&proto_request.field_name).map_err(crate::errors::Error::from)?;

        let top_k = proto_request
            .top_k
            .try_into()
            .map_err(|_| ValidationError::InvalidArgument("top_k".to_string()))?;
        let mut per_segment = HashMap::new();

        for segment_reader in index_holder.index_reader().searcher().segment_readers() {
            let inverted_index = segment_reader.inverted_index(field).map_err(crate::errors::Error::from)?;
            let mut top_k_heap = BinaryHeap::new();
            let mut term_stream = inverted_index.terms().stream()?;

            while let Some((term, term_info)) = term_stream.next() {
                if top_k_heap.len() < top_k {
                    top_k_heap.push(Reverse((term_info.doc_freq, term.to_vec())));
                    continue;
                }
                match top_k_heap.peek() {
                    None => {}
                    Some(peek) => {
                        if peek.0 .0 > term_info.doc_freq {
                            continue;
                        }
                        top_k_heap.push(Reverse((term_info.doc_freq, term.to_vec())));
                        top_k_heap.pop();
                    }
                }
            }

            per_segment.insert(
                segment_reader.segment_id().short_uuid_string(),
                proto::SegmentTerms {
                    term_infos: top_k_heap
                        .into_sorted_vec()
                        .iter()
                        .map(|Reverse((doc_freq, key))| proto::TermInfo {
                            key: key.to_vec(),
                            doc_freq: *doc_freq,
                        })
                        .collect(),
                },
            );
        }
        let response = proto::GetTopTermsResponse { per_segment };
        Ok(Response::new(response))
    }
}
