use crate::errors::SummaResult;
use crate::proto;

pub struct DeleteConsumerRequest {
    pub index_name: String,
    pub consumer_name: String,
}

impl DeleteConsumerRequest {
    pub fn from_proto(proto_request: proto::DeleteConsumerRequest) -> SummaResult<DeleteConsumerRequest> {
        Ok(DeleteConsumerRequest {
            index_name: proto_request.index_name,
            consumer_name: proto_request.consumer_name,
        })
    }
}
