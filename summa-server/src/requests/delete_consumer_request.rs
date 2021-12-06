use summa_proto::proto;
#[derive(Builder)]
pub struct DeleteConsumerRequest {
    pub consumer_name: String,
}

impl From<proto::DeleteConsumerRequest> for DeleteConsumerRequest {
    fn from(proto_request: proto::DeleteConsumerRequest) -> DeleteConsumerRequest {
        DeleteConsumerRequest {
            consumer_name: proto_request.consumer_name,
        }
    }
}
