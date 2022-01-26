use tantivy::schema::Schema;

use crate::errors::{Error, SummaServerResult, ValidationError};

pub fn parse_schema(schema: &str) -> SummaServerResult<Schema> {
    serde_yaml::from_str(schema).map_err(|_| Error::Validation(ValidationError::InvalidSchema(schema.to_owned())))
}

pub fn parse_fields(schema: &Schema, fields: &[String]) -> SummaServerResult<()> {
    for field in fields {
        schema.get_field(field)?;
    }
    Ok(())
}
