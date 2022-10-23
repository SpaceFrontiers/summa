use tantivy::schema::{Field, Schema};

use crate::errors::{Error, SummaResult, ValidationError};

pub fn parse_schema(schema: &str) -> SummaResult<Schema> {
    serde_yaml::from_str(schema).map_err(|_| Error::Validation(Box::new(ValidationError::InvalidSchema(schema.to_owned()))))
}

pub fn parse_fields<'a>(schema: &'a Schema, fields: &'a [String], excluded_fields: &'a [String]) -> SummaResult<Vec<(Field, &'a str)>> {
    if excluded_fields.is_empty() {
        Ok(fields
            .iter()
            .map(|f| schema.find_field(f).ok_or_else(|| ValidationError::MissingField(f.to_string())))
            .collect::<Result<_, _>>()?)
    } else if fields.is_empty() {
        Ok(schema
            .fields()
            .filter_map(|(_, field_entry)| {
                if excluded_fields.iter().any(|e| e == field_entry.name()) {
                    None
                } else {
                    Some(
                        schema
                            .find_field(field_entry.name())
                            .ok_or_else(|| ValidationError::MissingField(field_entry.name().to_string())),
                    )
                }
            })
            .collect::<Result<_, _>>()?)
    } else {
        Ok(fields
            .iter()
            .map(|f| schema.find_field(f).ok_or_else(|| ValidationError::MissingField(f.to_string())))
            .collect::<Result<_, _>>()?)
    }
}
