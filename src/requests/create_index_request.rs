use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use tantivy::schema::Schema as Fields;
use tantivy::IndexSortByField;

#[derive(Builder)]
pub struct CreateIndexRequest {
    pub index_name: String,
    pub index_engine: proto::IndexEngine,
    pub fields: Fields,
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "tantivy::store::Compressor::None")]
    pub compression: tantivy::store::Compressor,
    #[builder(default = "Vec::new()")]
    pub default_fields: Vec<String>,
    #[builder(default = "Vec::new()")]
    pub multi_fields: Vec<String>,
    #[builder(default = "None")]
    pub primary_key: Option<String>,
    #[builder(default = "None")]
    pub sort_by_field: Option<IndexSortByField>,
    #[builder(default = "None")]
    pub stop_words: Option<Vec<String>>,
    #[builder(default = "None")]
    pub writer_threads: Option<u64>,
    #[builder(default = "None")]
    pub writer_heap_size_bytes: Option<u64>,
}

impl CreateIndexRequest {
    fn parse_fields(fields: &str) -> SummaResult<Fields> {
        serde_yaml::from_str(fields).map_err(|_| Error::ValidationError(ValidationError::InvalidFieldsError(fields.to_owned())))
    }

    fn parse_default_fields(fields: &Fields, default_fields: &Vec<String>) -> SummaResult<Vec<String>> {
        Ok(default_fields
            .iter()
            .map(|default_field_name| match fields.get_field(default_field_name) {
                Some(_) => Ok(default_field_name.to_owned()),
                None => Err(ValidationError::MissingDefaultField(default_field_name.to_owned())),
            })
            .collect::<Result<_, _>>()?)
    }

    fn parse_primary_key(fields: &Fields, primary_key: &Option<String>) -> SummaResult<Option<String>> {
        Ok(match primary_key {
            Some(primary_key) => Some(match fields.get_field(primary_key) {
                Some(_) => primary_key.to_owned(),
                None => Err(ValidationError::MissingPrimaryKeyError(Some(primary_key.to_owned())))?,
            }),
            None => None,
        })
    }
}

impl TryFrom<proto::CreateIndexRequest> for CreateIndexRequest {
    type Error = Error;

    fn try_from(proto_request: proto::CreateIndexRequest) -> SummaResult<CreateIndexRequest> {
        let fields = CreateIndexRequest::parse_fields(&proto_request.fields)?;
        let default_fields = CreateIndexRequest::parse_default_fields(&fields, &proto_request.default_fields)?;
        let primary_key = CreateIndexRequest::parse_primary_key(&fields, &proto_request.primary_key)?;
        let compression = proto::Compression::from_i32(proto_request.compression)
            .map(proto::Compression::into)
            .unwrap_or(tantivy::store::Compressor::None);
        let multi_fields = proto_request
            .multi_fields
            .iter()
            .map(|multi_field_name| match fields.get_field(multi_field_name) {
                Some(_) => Ok(multi_field_name.to_owned()),
                None => Err(ValidationError::MissingMultiField(multi_field_name.to_owned())),
            })
            .collect::<Vec<Result<_, _>>>()
            .into_iter()
            .collect::<Result<_, _>>()?;
        Ok(CreateIndexRequestBuilder::default()
            .index_name(proto_request.index_name)
            .index_engine(proto::IndexEngine::from_i32(proto_request.index_engine).unwrap())
            .fields(fields.clone())
            .primary_key(primary_key)
            .compression(compression)
            .default_fields(default_fields)
            .multi_fields(multi_fields)
            .sort_by_field(proto_request.sort_by_field.map(proto::SortByField::into))
            .stop_words(if proto_request.stop_words.len() > 0 {
                Some(proto_request.stop_words)
            } else {
                None
            })
            .autocommit_interval_ms(proto_request.autocommit_interval_ms)
            .writer_threads(proto_request.writer_threads)
            .writer_heap_size_bytes(proto_request.writer_heap_size_bytes)
            .build()
            .unwrap())
    }
}
