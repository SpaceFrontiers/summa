use crate::proto;

#[derive(Builder)]
pub struct DeleteConsumerRequest {
    pub index_alias: String,
    pub consumer_name: String,
}

impl From<proto::DeleteConsumerRequest> for DeleteConsumerRequest {
    fn from(proto_request: proto::DeleteConsumerRequest) -> DeleteConsumerRequest {
        DeleteConsumerRequest {
            index_alias: proto_request.index_alias,
            consumer_name: proto_request.consumer_name,
        }
    }
}
