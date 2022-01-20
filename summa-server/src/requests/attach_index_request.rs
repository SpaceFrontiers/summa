use summa_proto::proto;

use crate::errors::Error;

#[derive(Builder)]
pub struct AttachIndexRequest {
    pub index_name: String,
    pub attach_index_request: proto::attach_index_request::IndexEngine,
}

impl TryFrom<proto::AttachIndexRequest> for AttachIndexRequest {
    type Error = Error;

    fn try_from(proto_request: proto::AttachIndexRequest) -> Result<Self, Self::Error> {
        Ok(AttachIndexRequest {
            index_name: proto_request.index_name,
            attach_index_request: proto_request
                .index_engine
                .unwrap_or(proto::attach_index_request::IndexEngine::File(proto::AttachFileEngineRequest {})),
        })
    }
}
