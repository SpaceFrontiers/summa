use tantivy::schema::{Field, Schema};

use crate::errors::{Error, SummaResult, ValidationError};

pub fn parse_schema(schema: &str) -> SummaResult<Schema> {
    serde_yaml::from_str(schema).map_err(|_| Error::Validation(Box::new(ValidationError::InvalidSchema(schema.to_owned()))))
}

pub fn parse_fields(schema: &Schema, fields: &[String]) -> SummaResult<Vec<Field>> {
    Ok(fields.iter().map(|f| schema.get_field(f)).collect::<Result<_, _>>()?)
}
