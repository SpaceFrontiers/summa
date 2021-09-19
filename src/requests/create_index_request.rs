use crate::errors::{Error, SummaResult};
use crate::proto;
use crate::requests::validators;
use tantivy::schema::Schema;
use tantivy::IndexSortByField;

#[derive(Builder)]
pub struct CreateIndexRequest {
    pub index_name: String,
    pub index_engine: proto::IndexEngine,
    pub schema: Schema,
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "tantivy::store::Compressor::None")]
    pub compression: tantivy::store::Compressor,
    #[builder(default = "None")]
    pub blocksize: Option<usize>,
    #[builder(default = "Vec::new()")]
    pub default_fields: Vec<String>,
    #[builder(default = "Vec::new()")]
    pub multi_fields: Vec<String>,
    #[builder(default = "None")]
    pub primary_key: Option<String>,
    #[builder(default = "None")]
    pub sort_by_field: Option<IndexSortByField>,
    #[builder(default = "None")]
    pub writer_threads: Option<u64>,
    #[builder(default = "None")]
    pub writer_heap_size_bytes: Option<u64>,
}

impl TryFrom<proto::CreateIndexRequest> for CreateIndexRequest {
    type Error = Error;

    fn try_from(proto_request: proto::CreateIndexRequest) -> SummaResult<Self> {
        let schema = validators::parse_schema(&proto_request.schema)?;
        let default_fields = validators::parse_default_fields(&schema, &proto_request.default_fields)?;
        let multi_fields = validators::parse_multi_fields(&schema, &proto_request.multi_fields)?;
        let primary_key = validators::parse_primary_key(&schema, &proto_request.primary_key)?;

        let compression = proto::Compression::from_i32(proto_request.compression)
            .map(proto::Compression::into)
            .unwrap_or(tantivy::store::Compressor::None);

        Ok(CreateIndexRequestBuilder::default()
            .index_name(proto_request.index_name)
            .index_engine(proto::IndexEngine::from_i32(proto_request.index_engine).unwrap())
            .schema(schema)
            .primary_key(primary_key)
            .compression(compression)
            .blocksize(proto_request.blocksize.map(|blocksize| blocksize as usize))
            .default_fields(default_fields)
            .multi_fields(multi_fields)
            .sort_by_field(proto_request.sort_by_field.map(proto::SortByField::into))
            .autocommit_interval_ms(proto_request.autocommit_interval_ms)
            .writer_threads(proto_request.writer_threads)
            .writer_heap_size_bytes(proto_request.writer_heap_size_bytes)
            .build()
            .unwrap())
    }
}
