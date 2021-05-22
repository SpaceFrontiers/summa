use crate::errors::{Error, SummaResult};
use crate::proto;
use tantivy::IndexSortByField;

#[derive(Builder)]
pub struct AlterIndexRequest {
    pub index_name: String,
    #[builder(default = "None")]
    pub compression: Option<tantivy::store::Compressor>,
    #[builder(default = "None")]
    pub sort_by_field: Option<IndexSortByField>,
}

impl TryFrom<proto::AlterIndexRequest> for AlterIndexRequest {
    type Error = Error;

    fn try_from(proto_request: proto::AlterIndexRequest) -> SummaResult<AlterIndexRequest> {
        let mut alter_index_request_builder = AlterIndexRequestBuilder::default();
        alter_index_request_builder.index_name(proto_request.index_name.to_owned());
        alter_index_request_builder.compression(
            proto_request
                .compression
                .map(|compression| proto::Compression::from_i32(compression).map(proto::Compression::into))
                .flatten(),
        );
        alter_index_request_builder.sort_by_field(proto_request.sort_by_field.map(proto::SortByField::into));
        Ok(alter_index_request_builder.build().unwrap())
    }
}
