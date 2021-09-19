use crate::errors::{Error, SummaResult};
use crate::proto;
use tantivy::IndexSortByField;

#[derive(Builder)]
pub struct AlterIndexRequest {
    pub index_name: String,
    #[builder(default = "None")]
    pub compression: Option<tantivy::store::Compressor>,
    #[builder(default = "None")]
    pub blocksize: Option<usize>,
    #[builder(default = "None")]
    pub sort_by_field: Option<IndexSortByField>,
    #[builder(default = "None")]
    pub default_fields: Option<Vec<String>>,
    #[builder(default = "None")]
    pub multi_fields: Option<Vec<String>>,
}

impl TryFrom<proto::AlterIndexRequest> for AlterIndexRequest {
    type Error = Error;

    fn try_from(proto_request: proto::AlterIndexRequest) -> SummaResult<AlterIndexRequest> {
        let mut alter_index_request_builder = AlterIndexRequestBuilder::default();
        alter_index_request_builder
            .index_name(proto_request.index_name.to_owned())
            .compression(
                proto_request
                    .compression
                    .and_then(|compression| proto::Compression::from_i32(compression).map(proto::Compression::into)),
            )
            .blocksize(proto_request.blocksize.map(|blocksize| blocksize as usize))
            .sort_by_field(proto_request.sort_by_field.map(proto::SortByField::into))
            .default_fields(proto_request.default_fields.map(|f| f.fields))
            .multi_fields(proto_request.multi_fields.map(|f| f.fields));
        Ok(alter_index_request_builder.build().unwrap())
    }
}
