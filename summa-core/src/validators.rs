use tantivy::schema::{Field, Schema};

use crate::errors::{Error, SummaResult, ValidationError};

pub fn parse_schema(schema: &str) -> SummaResult<Schema> {
    serde_yaml::from_str(schema).map_err(|_| Error::Validation(Box::new(ValidationError::InvalidSchema(schema.to_owned()))))
}

pub fn parse_fields<'a>(schema: &'a Schema, fields: &'a [String]) -> SummaResult<Vec<(Field, &'a str)>> {
    Ok(fields
        .iter()
        .map(|f| schema.find_field(f).ok_or_else(|| ValidationError::MissingField(f.to_string())))
        .collect::<Result<_, _>>()?)
}
