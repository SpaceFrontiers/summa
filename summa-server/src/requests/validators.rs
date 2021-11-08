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

pub fn parse_primary_key(schema: &Schema, primary_key: &Option<String>) -> SummaServerResult<Option<String>> {
    Ok(match primary_key {
        Some(primary_key) => Some(match schema.get_field(primary_key) {
            Some(_) => primary_key.to_owned(),
            None => return Err(ValidationError::MissingPrimaryKey(Some(primary_key.to_owned())).into()),
        }),
        None => None,
    })
}
