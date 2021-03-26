use crate::configs::IndexConfig;

use crate::errors::{SummaResult, ValidationError};
use crate::proto;

use tantivy::schema::{Field, Schema};

pub struct CreateIndexRequest {
    pub index_name: String,
    pub index_config: IndexConfig,
    pub schema: Schema,
}

impl CreateIndexRequest {
    pub fn from_proto(proto_request: &proto::CreateIndexRequest) -> SummaResult<CreateIndexRequest> {
        let schema: Schema = serde_yaml::from_str(&proto_request.schema)?;

        let mut default_fields: Vec<Field> = Vec::new();
        for default_field_name in proto_request.default_fields.iter() {
            match schema.get_field(default_field_name) {
                Some(default_field) => default_fields.push(default_field),
                None => Err(ValidationError::MissingDefaultField(default_field_name.to_string()))?,
            }
        }

        let primary_key = if let Some(ref primary_key) = proto_request.primary_key {
            Some(
                schema
                    .get_field(primary_key)
                    .ok_or_else(|| ValidationError::MissingPrimaryKeyError(Some(primary_key.to_string())))?,
            )
        } else {
            None
        };

        let writer_threads = if let Some(writer_threads) = proto_request.writer_threads {
            if writer_threads > 1 {
                writer_threads
            } else {
                Err(ValidationError::InvalidThreadsNumberError(writer_threads))?
            }
        } else {
            (num_cpus::get() / 2 + 1) as u64
        };

        let writer_heap_size_bytes = if let Some(writer_heap_size_bytes) = proto_request.writer_heap_size_bytes {
            if writer_heap_size_bytes > 1 {
                writer_heap_size_bytes
            } else {
                Err(ValidationError::InvalidThreadsNumberError(writer_threads))?
            }
        } else {
            1024 * 1024 * 1024
        };

        let index_config = IndexConfig {
            autocommit_interval_ms: proto_request.autocommit_interval_ms,
            compression: tantivy::store::Compressor::Brotli,
            default_fields,
            primary_key,
            writer_heap_size_bytes,
            writer_threads,
        };
        Ok(CreateIndexRequest {
            index_name: proto_request.index_name.to_string(),
            index_config,
            schema,
        })
    }
}
