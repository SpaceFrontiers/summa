use std::str::FromStr;

use base64::Engine;
use prost::encoding::bool;
use tantivy::json_utils::{convert_to_fast_value_and_get_term, JsonTermWriter};
use tantivy::schema::{Field, FieldType};
use tantivy::Term;
use tantivy_common::DateTime;

use crate::errors::SummaResult;
use crate::Error;

pub fn cast_field_to_term(field: &Field, full_path: &str, field_type: &FieldType, value: &str, force_str: bool) -> Term {
    match field_type {
        FieldType::Str(_) => Term::from_field_text(*field, value),
        FieldType::JsonObject(ref json_options) => {
            let mut term = Term::with_capacity(128);
            let mut json_term_writer = JsonTermWriter::from_field_and_json_path(*field, full_path, json_options.is_expand_dots_enabled(), &mut term);
            if force_str {
                json_term_writer.set_str(value);
                json_term_writer.term().clone()
            } else {
                convert_to_fast_value_and_get_term(&mut json_term_writer, value).unwrap_or_else(|| {
                    json_term_writer.set_str(value);
                    json_term_writer.term().clone()
                })
            }
        }
        _ => unreachable!(),
    }
}

pub fn cast_field_to_typed_term(field: &Field, full_path: &str, field_type: &FieldType, value: &str) -> SummaResult<Term> {
    Ok(match field_type {
        FieldType::I64(_) => Term::from_field_i64(
            *field,
            i64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as i64")))?,
        ),
        FieldType::U64(_) => Term::from_field_u64(
            *field,
            u64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as u64")))?,
        ),
        FieldType::F64(_) => Term::from_field_f64(
            *field,
            f64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as f64")))?,
        ),
        FieldType::Bytes(_) => Term::from_field_bytes(
            *field,
            &base64::engine::general_purpose::STANDARD
                .decode(value)
                .map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as bytes")))?,
        ),
        FieldType::Date(_) => Term::from_field_date(
            *field,
            DateTime::from_timestamp_secs(i64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as date")))?),
        ),
        _ => cast_field_to_term(field, full_path, field_type, value, false),
    })
}
