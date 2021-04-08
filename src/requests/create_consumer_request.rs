use crate::configs::KafkaConsumerConfig;
use crate::errors::SummaResult;
use crate::proto;

pub struct CreateConsumerRequest {
    pub consumer_name: String,
    pub consumer_config: KafkaConsumerConfig,
    pub index_name: String,
}

impl CreateConsumerRequest {
    pub fn from_proto(proto_request: &proto::CreateConsumerRequest) -> SummaResult<CreateConsumerRequest> {
        let consumer_config = KafkaConsumerConfig::new(&proto_request.bootstrap_servers, &proto_request.group_id, proto_request.threads, &proto_request.topics)?;
        Ok(CreateConsumerRequest {
            consumer_name: proto_request.consumer_name.clone(),
            consumer_config,
            index_name: proto_request.index_name.clone(),
        })
    }
}
