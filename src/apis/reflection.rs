//! Reflection GRPC API
//!
//! Reflection GRPC API is using for inspecting indices

use crate::errors::{Error, SummaResult};
use crate::proto;
use crate::services::IndexService;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use tonic::{Request, Response, Status};
use tracing::log::warn;

pub struct ReflectionApiImpl {
    index_service: IndexService,
}

impl ReflectionApiImpl {
    pub fn new(index_service: &IndexService) -> SummaResult<ReflectionApiImpl> {
        Ok(ReflectionApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::reflection_api_server::ReflectionApi for ReflectionApiImpl {
    async fn get_top_terms(&self, proto_request: Request<proto::GetTopTermsRequest>) -> Result<Response<proto::GetTopTermsResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let top_k = proto_request.top_k.try_into().unwrap();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_name)?;
        let field = index_holder
            .schema()
            .get_field(&proto_request.field_name)
            .ok_or(Error::FieldDoesNotExistError(proto_request.field_name.to_owned()))?;
        let mut per_segment = HashMap::new();
        for segment_reader in index_holder.index_reader().searcher().segment_readers() {
            let inverted_index = segment_reader.inverted_index(field).unwrap();
            let mut top_k_heap = BinaryHeap::new();
            let mut term_stream = inverted_index.terms().stream()?;
            while let Some((term, term_info)) = term_stream.next() {
                if top_k_heap.len() < top_k {
                    top_k_heap.push(Reverse((term_info.doc_freq, term.to_vec())));
                } else if top_k_heap.peek().unwrap().0 .0 < term_info.doc_freq {
                    top_k_heap.push(Reverse((term_info.doc_freq, term.to_vec())));
                    top_k_heap.pop();
                }
            }
            print!("{:?}", top_k_heap);
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
