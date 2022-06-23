use std::collections::HashMap;

use regex::Regex;
use tantivy::query::{BooleanQuery, Occur, Query};
use tantivy::schema::{Field, FieldType, Schema};
use tantivy::tokenizer::TokenizerManager;
use tantivy::Term;

use crate::components::query_parser::utils::cast_field_to_term;

pub trait TermFieldMapper {
    fn map(&self, value: &str, fields: &[String]) -> Option<Box<dyn Query>>;
}

fn tokenize_value(schema: &Schema, field: &Field, full_path: &str, value: &str, tokenizer_manager: &TokenizerManager) -> Vec<Term> {
    let field_entry = schema.get_field_entry(*field);
    let mut terms = vec![];
    match field_entry.field_type() {
        FieldType::Str(ref str_options) => {
            let option = str_options.get_indexing_options().expect("no options");
            let mut text_analyzer = tokenizer_manager.get(option.tokenizer()).expect("unknown tokenizer");
            let mut token_stream = text_analyzer.token_stream(value);
            token_stream.process(&mut |token| {
                let term = cast_field_to_term(field, full_path, schema.get_field_entry(*field).field_type(), &token.text, true);
                terms.push(term);
            });
        }
        _ => terms.push(cast_field_to_term(
            field,
            full_path,
            schema.get_field_entry(*field).field_type(),
            &value.replace('-', ""),
            true,
        )),
    };
    terms
}

pub struct DoiMapper {
    schema: Schema,
    tokenizer_manager: TokenizerManager,
}

impl DoiMapper {
    pub fn new(schema: Schema, tokenizer_manager: TokenizerManager) -> Self {
        DoiMapper { schema, tokenizer_manager }
    }
}

impl TermFieldMapper for DoiMapper {
    fn map(&self, value: &str, fields: &[String]) -> Option<Box<dyn Query>> {
        let terms = fields
            .iter()
            .flat_map(|field_name| {
                let (field, full_path) = self.schema.find_field(field_name).expect("inconsistent state");
                tokenize_value(&self.schema, &field, full_path, value, &self.tokenizer_manager)
            })
            .collect();
        Some(Box::new(BooleanQuery::new_multiterms_query(terms)) as Box<dyn Query>)
    }
}

pub struct DoiIsbnMapper {
    schema: Schema,
    tokenizer_manager: TokenizerManager,
}

impl DoiIsbnMapper {
    pub fn new(schema: Schema, tokenizer_manager: TokenizerManager) -> Self {
        DoiIsbnMapper { schema, tokenizer_manager }
    }
}

impl TermFieldMapper for DoiIsbnMapper {
    fn map(&self, value: &str, fields: &[String]) -> Option<Box<dyn Query>> {
        thread_local! {
            static MATCHER: Regex = Regex::new(r"(10.[0-9]+)/((?:cbo)?(?:(?:978[0-9]{10})|(?:978[0-9]{13})|(?:978[-0-9]+)))(.*)").expect("cannot compile regex");
        }
        MATCHER.with(|matcher| {
            let lowercased_doi = value.to_lowercase();
            for cap in matcher.captures_iter(&lowercased_doi) {
                let (_, isbn, _) = (&cap[1], &cap[2], &cap[3]);
                let corrected_isbn = isbn.replace('-', "").replace("cbo", "");
                if corrected_isbn.len() == 10 || corrected_isbn.len() == 13 {
                    let subqueries: Vec<_> = fields
                        .iter()
                        .map(|field_name| {
                            let (field, full_path) = self.schema.find_field(field_name).expect("inconsistent state");
                            let terms = tokenize_value(&self.schema, &field, full_path, &corrected_isbn, &self.tokenizer_manager);
                            let query = Box::new(BooleanQuery::new_multiterms_query(terms)) as Box<dyn Query>;
                            (Occur::Should, query)
                        })
                        .collect();
                    return Some(Box::new(BooleanQuery::new(subqueries)) as Box<dyn Query>);
                }
            }
            None
        })
    }
}

pub struct IsbnMapper {
    schema: Schema,
    tokenizer_manager: TokenizerManager,
}

impl IsbnMapper {
    pub fn new(schema: Schema, tokenizer_manager: TokenizerManager) -> Self {
        IsbnMapper { schema, tokenizer_manager }
    }
}

impl TermFieldMapper for IsbnMapper {
    fn map(&self, value: &str, fields: &[String]) -> Option<Box<dyn Query>> {
        let subqueries: Vec<_> = fields
            .iter()
            .map(|field_name| {
                let (field, full_path) = self.schema.find_field(field_name).expect("inconsistent state");
                let terms = tokenize_value(&self.schema, &field, full_path, &value.replace('-', ""), &self.tokenizer_manager);
                (Occur::Should, Box::new(BooleanQuery::new_multiterms_query(terms)) as Box<dyn Query>)
            })
            .collect();
        Some(Box::new(BooleanQuery::new(subqueries)) as Box<dyn Query>)
    }
}

pub struct TermFieldMappersManager {
    term_field_mappers: HashMap<String, Box<dyn TermFieldMapper>>,
}

impl TermFieldMappersManager {
    pub fn new(schema: &Schema, tokenizer_manager: &TokenizerManager) -> Self {
        let mut term_field_mappers = HashMap::new();
        term_field_mappers.insert(
            "doi".to_string(),
            Box::new(DoiMapper::new(schema.clone(), tokenizer_manager.clone())) as Box<dyn TermFieldMapper>,
        );
        term_field_mappers.insert(
            "doi_isbn".to_string(),
            Box::new(DoiIsbnMapper::new(schema.clone(), tokenizer_manager.clone())) as Box<dyn TermFieldMapper>,
        );
        term_field_mappers.insert(
            "isbn".to_string(),
            Box::new(IsbnMapper::new(schema.clone(), tokenizer_manager.clone())) as Box<dyn TermFieldMapper>,
        );
        TermFieldMappersManager { term_field_mappers }
    }

    pub fn get(&self, name: &str) -> Option<&Box<dyn TermFieldMapper>> {
        self.term_field_mappers.get(name)
    }
}
