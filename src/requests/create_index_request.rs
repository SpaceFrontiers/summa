use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use tantivy::schema::Schema;
use tantivy::{IndexSortByField, Order};

#[derive(Builder)]
pub struct CreateIndexRequest {
    pub index_name: String,
    pub index_engine: proto::IndexEngine,
    pub schema: Schema,
    #[builder(default = "None")]
    pub primary_key: Option<String>,
    #[builder(default = "Vec::new()")]
    pub default_fields: Vec<String>,
    #[builder(default = "Vec::new()")]
    pub multi_fields: Vec<String>,
    #[builder(default = "None")]
    pub sort_by_field: Option<IndexSortByField>,
    #[builder(default = "None")]
    pub autocommit_interval_ms: Option<u64>,
    #[builder(default = "None")]
    pub writer_threads: Option<u64>,
    #[builder(default = "None")]
    pub writer_heap_size_bytes: Option<u64>,
}

impl TryFrom<proto::CreateIndexRequest> for CreateIndexRequest {
    type Error = Error;

    fn try_from(proto_request: proto::CreateIndexRequest) -> SummaResult<CreateIndexRequest> {
        let schema: Schema = serde_yaml::from_str(&proto_request.schema)?;
        Ok(CreateIndexRequest {
            index_name: proto_request.index_name.to_owned(),
            index_engine: proto::IndexEngine::from_i32(proto_request.index_engine).unwrap(),
            schema: schema.clone(),
            primary_key: match proto_request.primary_key {
                Some(primary_key) => Some(match schema.get_field(&primary_key) {
                    Some(_) => primary_key.to_owned(),
                    None => Err(ValidationError::MissingPrimaryKeyError(Some(primary_key.to_owned())))?,
                }),
                None => None,
            },
            default_fields: proto_request
                .default_fields
                .iter()
                .map(|default_field_name| match schema.get_field(default_field_name) {
                    Some(_) => Ok(default_field_name.to_owned()),
                    None => Err(ValidationError::MissingDefaultField(default_field_name.to_owned())),
                })
                .collect::<Vec<Result<_, _>>>()
                .into_iter()
                .collect::<Result<_, _>>()?,
            multi_fields: proto_request
                .multi_fields
                .iter()
                .map(|multi_field_name| match schema.get_field(multi_field_name) {
                    Some(_) => Ok(multi_field_name.to_owned()),
                    None => Err(ValidationError::MissingMultiField(multi_field_name.to_owned())),
                })
                .collect::<Vec<Result<_, _>>>()
                .into_iter()
                .collect::<Result<_, _>>()?,
            sort_by_field: match proto_request.sort_by_field {
                Some(ref sort_by_field) => Some(IndexSortByField {
                    field: sort_by_field.field.clone(),
                    order: match proto::Order::from_i32(sort_by_field.order) {
                        None | Some(proto::Order::Asc) => Order::Asc,
                        Some(proto::Order::Desc) => Order::Desc,
                    },
                }),
                None => None,
            },
            autocommit_interval_ms: proto_request.autocommit_interval_ms,
            writer_threads: proto_request.writer_threads,
            writer_heap_size_bytes: proto_request.writer_heap_size_bytes,
        })
    }
}
