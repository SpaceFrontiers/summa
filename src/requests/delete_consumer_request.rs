use crate::errors::SummaResult;
use crate::proto;

pub struct DeleteConsumerRequest {
    pub consumer_name: String,
}

impl DeleteConsumerRequest {
    pub fn from_proto(proto_request: proto::DeleteConsumerRequest) -> SummaResult<DeleteConsumerRequest> {
        Ok(DeleteConsumerRequest {
            consumer_name: proto_request.consumer_name,
        })
    }
}
