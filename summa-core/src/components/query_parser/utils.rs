use tantivy::json_utils::{convert_to_fast_value_and_get_term, JsonTermWriter};
use tantivy::schema::{Field, FieldType};
use tantivy::Term;

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
