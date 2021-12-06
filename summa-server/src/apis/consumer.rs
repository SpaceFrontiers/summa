//! Consumer GRPC API
//!
//! Consumer GRPC API is using for managing Kafka consumers

use summa_proto::proto;
use tonic::{Request, Response, Status};

use crate::errors::SummaServerResult;
use crate::errors::ValidationError;
use crate::requests::CreateConsumerRequest;
use crate::services::IndexService;

pub struct ConsumerApiImpl {
    index_service: IndexService,
}

impl ConsumerApiImpl {
    pub fn new(index_service: &IndexService) -> SummaServerResult<ConsumerApiImpl> {
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
        let index_name = self.index_service.create_consumer(&create_consumer_request).await?;
        let response = proto::CreateConsumerResponse {
            consumer: Some(proto::Consumer {
                index_name,
                consumer_name: create_consumer_request.consumer_name.to_owned(),
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_consumer(&self, proto_request: Request<proto::GetConsumerRequest>) -> Result<Response<proto::GetConsumerResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let consumer_config = self
            .index_service
            .get_consumer_config(&proto_request.consumer_name)
            .await
            .ok_or_else(|| ValidationError::MissingConsumer(proto_request.consumer_name.to_string()))?;
        let response = proto::GetConsumerResponse {
            consumer: Some(proto::Consumer {
                consumer_name: proto_request.consumer_name.to_owned(),
                index_name: consumer_config.index_name,
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_consumers(&self, _: Request<proto::GetConsumersRequest>) -> Result<Response<proto::GetConsumersResponse>, Status> {
        let response = proto::GetConsumersResponse {
            consumers: self
                .index_service
                .get_consumers()
                .await
                .iter()
                .map(|(consumer_name, consumer_config)| proto::Consumer {
                    index_name: consumer_config.index_name.to_string(),
                    consumer_name: consumer_name.clone(),
                })
                .collect(),
        };
        Ok(Response::new(response))
    }

    async fn delete_consumer(&self, proto_request: Request<proto::DeleteConsumerRequest>) -> Result<Response<proto::DeleteConsumerResponse>, Status> {
        self.index_service.delete_consumer(proto_request.into_inner().into()).await?;
        Ok(Response::new(proto::DeleteConsumerResponse {}))
    }
}
