use std::net::IpAddr;
use std::str::{from_utf8, FromStr};

use base64::Engine;
use serde_json::{json, Value as JsonValue};
use tantivy::schema::{Facet, FieldType, IntoIpv6Addr, Schema, Value};
use tantivy::tokenizer::PreTokenizedString;
use tantivy::{DateTime, Document};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::errors::{Error, SummaResult, ValidationError};
use crate::page_rank::quantize_page_rank;
use crate::utils::current_time;

/// Wrapper for carrying `tantivy::Document` from various sources
pub enum SummaDocument<'a> {
    BoundJsonBytes((&'a Schema, &'a [u8])),
    UnboundJsonBytes(&'a [u8]),
    TantivyDocument(Document),
}

/// Possible error that may occur while parsing a field value
/// At this point the JSON is known to be valid.
#[derive(thiserror::Error, Debug)]
pub enum ValueParsingError {
    #[error("overflow_error: <expected: {expected}, got: {json}>")]
    OverflowError { expected: &'static str, json: JsonValue },
    #[error("type_error: <expected: {expected}, got: {json}")]
    TypeError { expected: &'static str, json: JsonValue },
    #[error("invalid_base64: {base64}")]
    InvalidBase64 { base64: String },
    #[error("null_value_error")]
    NullValueError,
    #[error("parse_error: {json}, error: {error}")]
    ParseError { error: String, json: serde_json::Value },
}

/// Error that may happen when deserializing
/// a document from JSON.
#[derive(thiserror::Error, Debug)]
pub enum DocumentParsingError {
    /// The payload given is not valid JSON.
    #[error("The provided string is not valid JSON")]
    InvalidJson(String),
    /// One of the value node could not be parsed.
    #[error("The field '{0:?}' could not be parsed: {1:?}")]
    ValueError(String, ValueParsingError),
}

pub fn process_dynamic_fields(schema: &Schema, json_object: &mut serde_json::Map<String, JsonValue>) {
    if schema.get_field("page_rank").is_ok() && schema.get_field("quantized_page_rank").is_ok() {
        if let Some(page_rank_value) = json_object.get_mut("page_rank") {
            if let Some(v) = page_rank_value.as_f64() {
                json_object.insert("quantized_page_rank".to_string(), json!(quantize_page_rank(v)));
            }
        }
    }
    if schema.get_field("updated_at").is_ok() {
        json_object.insert("updated_at".to_string(), json!(current_time()));
    }
    if schema.get_field("custom_score").is_ok() && !json_object.contains_key("custom_score") {
        match json_object.get("type") {
            Some(serde_json::value::Value::String(s)) => {
                if s == "book-chapter" {
                    json_object.insert("custom_score".to_string(), json!(0.85))
                } else {
                    json_object.insert("custom_score".to_string(), json!(1.0))
                }
            }
            _ => json_object.insert("custom_score".to_string(), json!(1.0)),
        };
    }
    if schema.get_field("links").is_ok() && schema.get_field("ctr").is_ok() && !json_object.contains_key("ctr") {
        json_object.insert("ctr".to_string(), json!(0.1));
    }
}

impl<'a> SummaDocument<'a> {
    pub fn bound_with(self, schema: &'a Schema) -> SummaDocument {
        match self {
            SummaDocument::UnboundJsonBytes(json_bytes) => SummaDocument::BoundJsonBytes((schema, json_bytes)),
            SummaDocument::BoundJsonBytes((_, json_bytes)) => SummaDocument::BoundJsonBytes((schema, json_bytes)),
            other => other,
        }
    }

    /// Parse single json value
    #[inline]
    pub fn value_from_json(&self, field_type: &FieldType, json: JsonValue) -> Result<Value, ValueParsingError> {
        match json {
            JsonValue::String(field_text) => match *field_type {
                FieldType::Date(_) => {
                    let dt_with_fixed_tz = OffsetDateTime::parse(&field_text, &Rfc3339).map_err(|_err| ValueParsingError::TypeError {
                        expected: "rfc3339 format",
                        json: JsonValue::String(field_text),
                    })?;
                    Ok(DateTime::from_utc(dt_with_fixed_tz).into())
                }
                FieldType::Str(_) => Ok(Value::Str(field_text)),
                FieldType::U64(_) | FieldType::I64(_) | FieldType::F64(_) => Err(ValueParsingError::TypeError {
                    expected: "an integer",
                    json: JsonValue::String(field_text),
                }),
                FieldType::Bool(_) => Err(ValueParsingError::TypeError {
                    expected: "a boolean",
                    json: JsonValue::String(field_text),
                }),
                FieldType::Facet(_) => Ok(Value::Facet(Facet::from(&field_text))),
                FieldType::Bytes(_) => base64::engine::general_purpose::STANDARD
                    .decode(&field_text)
                    .map(Value::Bytes)
                    .map_err(|_| ValueParsingError::InvalidBase64 { base64: field_text }),
                FieldType::JsonObject(_) => Err(ValueParsingError::TypeError {
                    expected: "a json object",
                    json: JsonValue::String(field_text),
                }),
                FieldType::IpAddr(_) => {
                    let ip_addr: IpAddr = IpAddr::from_str(&field_text).map_err(|err| ValueParsingError::ParseError {
                        error: err.to_string(),
                        json: JsonValue::String(field_text),
                    })?;
                    Ok(Value::IpAddr(ip_addr.into_ipv6_addr()))
                }
            },
            JsonValue::Number(field_val_num) => match field_type {
                FieldType::I64(_) | FieldType::Date(_) => {
                    if let Some(field_val_i64) = field_val_num.as_i64() {
                        Ok(Value::I64(field_val_i64))
                    } else {
                        Err(ValueParsingError::OverflowError {
                            expected: "an i64 int",
                            json: JsonValue::Number(field_val_num),
                        })
                    }
                }
                FieldType::U64(_) => {
                    if let Some(field_val_u64) = field_val_num.as_u64() {
                        Ok(Value::U64(field_val_u64))
                    } else {
                        Err(ValueParsingError::OverflowError {
                            expected: "u64",
                            json: JsonValue::Number(field_val_num),
                        })
                    }
                }
                FieldType::F64(_) => {
                    if let Some(field_val_f64) = field_val_num.as_f64() {
                        Ok(Value::F64(field_val_f64))
                    } else {
                        Err(ValueParsingError::OverflowError {
                            expected: "a f64",
                            json: JsonValue::Number(field_val_num),
                        })
                    }
                }
                FieldType::Bool(_) => Err(ValueParsingError::TypeError {
                    expected: "a boolean",
                    json: JsonValue::Number(field_val_num),
                }),
                FieldType::Str(_) | FieldType::Facet(_) | FieldType::Bytes(_) => Err(ValueParsingError::TypeError {
                    expected: "a string",
                    json: JsonValue::Number(field_val_num),
                }),
                FieldType::JsonObject(_) => Err(ValueParsingError::TypeError {
                    expected: "a json object",
                    json: JsonValue::Number(field_val_num),
                }),
                FieldType::IpAddr(_) => Err(ValueParsingError::TypeError {
                    expected: "a string with an ip addr",
                    json: JsonValue::Number(field_val_num),
                }),
            },
            JsonValue::Object(json_map) => match field_type {
                FieldType::Str(_) => {
                    if let Ok(tok_str_val) = serde_json::from_value::<PreTokenizedString>(serde_json::Value::Object(json_map.clone())) {
                        Ok(Value::PreTokStr(tok_str_val))
                    } else {
                        Err(ValueParsingError::TypeError {
                            expected: "a string or an pretokenized string",
                            json: JsonValue::Object(json_map),
                        })
                    }
                }
                FieldType::JsonObject(_) => Ok(Value::JsonObject(json_map)),
                _ => Err(ValueParsingError::TypeError {
                    expected: field_type.value_type().name(),
                    json: JsonValue::Object(json_map),
                }),
            },
            JsonValue::Bool(json_bool_val) => match field_type {
                FieldType::Bool(_) => Ok(Value::Bool(json_bool_val)),
                _ => Err(ValueParsingError::TypeError {
                    expected: field_type.value_type().name(),
                    json: JsonValue::Bool(json_bool_val),
                }),
            },
            JsonValue::Null => Err(ValueParsingError::NullValueError),
            _ => Err(ValueParsingError::TypeError {
                expected: field_type.value_type().name(),
                json: json.clone(),
            }),
        }
    }

    /// Build a document object from a json-object.
    pub fn parse_and_setup_document(&self, schema: &Schema, doc_json: &str) -> SummaResult<Document> {
        let mut json_obj: serde_json::Map<String, JsonValue> =
            serde_json::from_str(doc_json).map_err(|_| DocumentParsingError::InvalidJson(doc_json.to_owned()))?;
        process_dynamic_fields(schema, &mut json_obj);
        self.json_object_to_doc(schema, json_obj)
    }

    /// Build a document object from a json-object.
    pub fn json_object_to_doc(&self, schema: &Schema, json_obj: serde_json::Map<String, JsonValue>) -> SummaResult<Document> {
        let mut doc = Document::default();
        for (field_name, json_value) in json_obj {
            if let Ok(field) = schema.get_field(&field_name) {
                let field_entry = schema.get_field_entry(field);
                let field_type = field_entry.field_type();
                match json_value {
                    JsonValue::Array(json_items) => {
                        for json_item in json_items {
                            match self.value_from_json(field_type, json_item) {
                                Ok(value) => doc.add_field_value(field, value),
                                Err(ValueParsingError::NullValueError) => continue,
                                Err(error) => return Err(DocumentParsingError::ValueError(field_name.to_owned(), error).into()),
                            }
                        }
                    }
                    _ => match self.value_from_json(field_type, json_value) {
                        Ok(value) => doc.add_field_value(field, value),
                        Err(ValueParsingError::NullValueError) => continue,
                        Err(error) => return Err(DocumentParsingError::ValueError(field_name.to_owned(), error).into()),
                    },
                }
            }
        }
        Ok(doc)
    }
}

impl<'a> TryInto<Document> for SummaDocument<'a> {
    type Error = Error;

    fn try_into(self) -> SummaResult<Document> {
        match self {
            SummaDocument::BoundJsonBytes((schema, json_bytes)) => {
                let text_document = from_utf8(json_bytes).map_err(ValidationError::Utf8)?;
                let parsed_document = self.parse_and_setup_document(schema, text_document)?;
                Ok(parsed_document)
            }
            SummaDocument::UnboundJsonBytes(_) => Err(Error::UnboundDocument),
            SummaDocument::TantivyDocument(document) => Ok(document),
        }
    }
}

impl<'a> From<&'a Vec<u8>> for SummaDocument<'a> {
    fn from(v: &'a Vec<u8>) -> Self {
        SummaDocument::UnboundJsonBytes(v)
    }
}
