//! Consumer GRPC API
//!
//! Consumer GRPC API is using for managing Kafka consumers

use crate::errors::SummaResult;
use crate::proto;
use crate::requests::{CreateConsumerRequest, DeleteConsumerRequest};
use crate::services::IndexService;
use tonic::{Request, Response, Status};

pub struct ConsumerApiImpl {
    index_service: IndexService,
}

impl ConsumerApiImpl {
    pub fn new(index_service: &IndexService) -> SummaResult<ConsumerApiImpl> {
        Ok(ConsumerApiImpl {
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::consumer_api_server::ConsumerApi for ConsumerApiImpl {
    async fn create_consumer(&self, proto_request: Request<proto::CreateConsumerRequest>) -> Result<Response<proto::CreateConsumerResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let create_consumer_request = CreateConsumerRequest::from_proto(&proto_request)?;
        let index_name = create_consumer_request.index_name.clone();
        let consumer_name = create_consumer_request.consumer_name.clone();
        self.index_service.create_consumer(create_consumer_request).await?;
        let response = proto::CreateConsumerResponse {
            consumer: Some(proto::Consumer { consumer_name, index_name }),
        };
        Ok(Response::new(response))
    }

    async fn get_consumer(&self, proto_request: Request<proto::GetConsumerRequest>) -> Result<Response<proto::GetConsumerResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let consumer_config = self.index_service.consumer_registry().get_consumer_config(&proto_request.consumer_name)?;
        let response = proto::GetConsumerResponse {
            consumer: Some(proto::Consumer {
                consumer_name: proto_request.consumer_name.to_string(),
                index_name: consumer_config.index_name.to_string(),
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_consumers(&self, _: Request<proto::GetConsumersRequest>) -> Result<Response<proto::GetConsumersResponse>, Status> {
        let response = proto::GetConsumersResponse {
            consumers: self
                .index_service
                .consumer_registry()
                .consumer_configs()
                .iter()
                .map(|(consumer_name, consumer_config)| proto::Consumer {
                    consumer_name: consumer_name.to_string(),
                    index_name: consumer_config.index_name.to_string(),
                })
                .collect(),
        };
        Ok(Response::new(response))
    }

    async fn delete_consumer(&self, proto_request: Request<proto::DeleteConsumerRequest>) -> Result<Response<proto::DeleteConsumerResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let delete_consumer_request = DeleteConsumerRequest::from_proto(proto_request)?;
        self.index_service.delete_consumer(delete_consumer_request).await?;
        let response = proto::DeleteConsumerResponse {};
        Ok(Response::new(response))
    }
}
