//! Consumer GRPC API
//!
//! Consumer GRPC API is using for managing Kafka consumers

use crate::errors::SummaResult;
use crate::proto;
use crate::requests::CreateConsumerRequest;
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
        self.index_service.create_consumer(&create_consumer_request).await?;
        let response = proto::CreateConsumerResponse {
            consumer: Some(proto::Consumer {
                consumer_name: create_consumer_request.consumer_name.to_owned(),
                index_name: create_consumer_request.index_name.to_owned(),
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_consumer(&self, proto_request: Request<proto::GetConsumerRequest>) -> Result<Response<proto::GetConsumerResponse>, Status> {
        let proto_request = proto_request.into_inner();
        self.index_service.get_consumer_config(&proto_request.index_name, &proto_request.consumer_name)?;
        let response = proto::GetConsumerResponse {
            consumer: Some(proto::Consumer {
                consumer_name: proto_request.consumer_name.to_owned(),
                index_name: proto_request.index_name.to_owned(),
            }),
        };
        Ok(Response::new(response))
    }

    async fn get_consumers(&self, _: Request<proto::GetConsumersRequest>) -> Result<Response<proto::GetConsumersResponse>, Status> {
        let response = proto::GetConsumersResponse {
            consumers: self
                .index_service
                .get_consumers()?
                .iter()
                .map(|(index_name, consumer_name)| proto::Consumer {
                    consumer_name: consumer_name.clone(),
                    index_name: index_name.clone(),
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
