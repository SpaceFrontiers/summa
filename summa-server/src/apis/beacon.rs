//! Beacon GRPC API
//!
//! Beacon GRPC API is using for distributing index

use summa_proto::proto;
use tonic::{Request, Response, Status};

use crate::errors::SummaServerResult;
use crate::services::{BeaconService, IndexService};

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
        let hash = self
            .beacon_service
            .publish_index(&format!("/index/{}", index_holder.index_name()), index_holder, proto_request.payload)
            .await?;
        Ok(Response::new(proto::PublishIndexResponse { hash }))
    }
}
