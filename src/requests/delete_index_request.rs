use crate::errors::SummaResult;
use crate::proto;

pub struct DeleteIndexRequest {
    pub index_name: String,
    pub cascade: bool,
}

impl DeleteIndexRequest {
    pub fn from_proto(proto_request: proto::DeleteIndexRequest) -> SummaResult<DeleteIndexRequest> {
        Ok(DeleteIndexRequest {
            index_name: proto_request.index_name,
            cascade: proto_request.cascade,
        })
    }
}
