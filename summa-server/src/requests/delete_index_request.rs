use summa_proto::proto;
#[derive(Builder)]
pub struct DeleteIndexRequest {
    pub index_name: String,
}

impl From<proto::DeleteIndexRequest> for DeleteIndexRequest {
    fn from(proto_request: proto::DeleteIndexRequest) -> Self {
        DeleteIndexRequest {
            index_name: proto_request.index_name,
        }
    }
}

impl From<&str> for DeleteIndexRequest {
    fn from(index_name: &str) -> Self {
        DeleteIndexRequest {
            index_name: index_name.to_owned(),
        }
    }
}
