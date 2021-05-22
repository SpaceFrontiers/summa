use crate::configs::ConsumerConfig;
use crate::errors::SummaResult;
use crate::proto;

#[derive(Builder)]
pub struct CreateConsumerRequest {
    pub consumer_name: String,
    pub consumer_config: ConsumerConfig,
    pub index_alias: String,
}

impl CreateConsumerRequest {
    pub fn from_proto(proto_request: &proto::CreateConsumerRequest) -> SummaResult<CreateConsumerRequest> {
        let consumer_config = ConsumerConfig::new(
            &proto_request.bootstrap_servers,
            &proto_request.group_id,
            proto_request.threads,
            &proto_request.topics,
        )?;
        Ok(CreateConsumerRequest {
            consumer_name: proto_request.consumer_name.clone(),
            consumer_config,
            index_alias: proto_request.index_alias.clone(),
        })
    }
}
