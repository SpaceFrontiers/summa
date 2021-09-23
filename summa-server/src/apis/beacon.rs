//! Beacon GRPC API
//!
//! Beacon GRPC API is using for distributing index

use crate::errors::SummaServerResult;
use crate::services::{BeaconService, IndexService};
use summa_proto::proto;
use tonic::{Request, Response, Status};

pub struct BeaconApiImpl {
    beacon_service: BeaconService,
    index_service: IndexService,
}

impl BeaconApiImpl {
    pub fn new(beacon_service: &BeaconService, index_service: &IndexService) -> SummaServerResult<BeaconApiImpl> {
        Ok(BeaconApiImpl {
            beacon_service: beacon_service.clone(),
            index_service: index_service.clone(),
        })
    }
}

#[tonic::async_trait]
impl proto::beacon_api_server::BeaconApi for BeaconApiImpl {
    async fn publish_index(&self, proto_request: Request<proto::PublishIndexRequest>) -> Result<Response<proto::PublishIndexResponse>, Status> {
        let proto_request = proto_request.into_inner();
        let index_holder = self.index_service.get_index_holder(&proto_request.index_alias).await?;
        let key = self.beacon_service.publish_index(index_holder).await?;
        let response = proto::PublishIndexResponse {
            key: Some(proto::publish_index_response::Key {
                name: key.name().to_string(),
                id: key.id().to_string(),
            }),
        };
        Ok(Response::new(response))
    }
}
