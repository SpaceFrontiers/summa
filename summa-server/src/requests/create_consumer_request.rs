use summa_proto::proto;

use crate::configs::ConsumerConfig;
use crate::errors::SummaServerResult;

#[derive(Builder)]
pub struct CreateConsumerRequest {
    pub consumer_name: String,
    pub consumer_config: ConsumerConfig,
}

impl CreateConsumerRequest {
    pub fn from_proto(proto_request: &proto::CreateConsumerRequest) -> SummaServerResult<CreateConsumerRequest> {
        let consumer_config = ConsumerConfig::new(
            &proto_request.index_name,
            &proto_request.bootstrap_servers,
            &proto_request.group_id,
            &proto_request.topics,
        )?;
        Ok(CreateConsumerRequest {
            consumer_name: proto_request.consumer_name.clone(),
            consumer_config,
        })
    }
}
