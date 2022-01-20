use tantivy::schema::Schema;

use crate::errors::{Error, SummaServerResult, ValidationError};

pub fn parse_schema(schema: &str) -> SummaServerResult<Schema> {
    serde_yaml::from_str(schema).map_err(|_| Error::Validation(ValidationError::InvalidSchema(schema.to_owned())))
}

pub fn parse_default_fields(schema: &Schema, default_fields: &[String]) -> SummaServerResult<Vec<String>> {
    Ok(default_fields
        .iter()
        .map(|default_field_name| match schema.get_field(default_field_name) {
            Some(_) => Ok(default_field_name.to_owned()),
            None => Err(ValidationError::MissingDefaultField(default_field_name.to_owned())),
        })
        .collect::<Result<_, _>>()?)
}

pub fn parse_multi_fields(schema: &Schema, multi_fields: &[String]) -> SummaServerResult<Vec<String>> {
    Ok(multi_fields
        .iter()
        .map(|multi_field_name| match schema.get_field(multi_field_name) {
            Some(_) => Ok(multi_field_name.to_owned()),
            None => Err(ValidationError::MissingMultiField(multi_field_name.to_owned())),
        })
        .collect::<Vec<Result<_, _>>>()
        .into_iter()
        .collect::<Result<_, _>>()?)
}

pub fn parse_unique_fields(schema: &Schema, unique_fields: &[String]) -> SummaServerResult<Vec<String>> {
    Ok(unique_fields
        .iter()
        .map(|unique_field_name| match schema.get_field(unique_field_name) {
            Some(_) => Ok(unique_field_name.to_owned()),
            None => Err(ValidationError::MissingUniqueField(unique_field_name.to_owned())),
        })
        .collect::<Vec<Result<_, _>>>()
        .into_iter()
        .collect::<Result<_, _>>()?)
}
