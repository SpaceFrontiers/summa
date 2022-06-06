use regex::Regex;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{IndexRecordOption, Schema};

use crate::components::query_parser::summa_ql::cast_field_to_term;

pub trait TermFieldMapper {
    fn map(&self, value: &str) -> Option<Box<dyn Query>>;
    fn is_terminal(&self) -> bool;
}

pub struct DoiMapper {
    schema: Schema,
    fields: Vec<String>,
    is_terminal: bool,
}

impl DoiMapper {
    pub fn new(schema: Schema, fields: Vec<String>) -> Self {
        DoiMapper {
            schema,
            fields,
            is_terminal: true,
        }
    }
}

impl TermFieldMapper for DoiMapper {
    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn map(&self, value: &str) -> Option<Box<dyn Query>> {
        let terms = self
            .fields
            .iter()
            .map(|field_name| {
                let (field, full_path) = self.schema.find_field(field_name).expect("inconsistent state");
                cast_field_to_term(
                    &field,
                    full_path,
                    self.schema.get_field_entry(field).field_type(),
                    &value.replace(' ', ""),
                    true,
                )
            })
            .collect();
        Some(Box::new(BooleanQuery::new_multiterms_query(terms)) as Box<dyn Query>)
    }
}

pub struct DoiIsbnMapper {
    schema: Schema,
    fields: Vec<String>,
    is_terminal: bool,
}

impl DoiIsbnMapper {
    pub fn new(schema: Schema, fields: Vec<String>) -> Self {
        DoiIsbnMapper {
            schema,
            fields,
            is_terminal: true,
        }
    }
}

impl TermFieldMapper for DoiIsbnMapper {
    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn map(&self, value: &str) -> Option<Box<dyn Query>> {
        thread_local! {
            static MATCHER: Regex = Regex::new(r"(10.[0-9]+)/((?:cbo)?(?:(?:978[0-9]{10})|(?:978[0-9]{13})|(?:978[-0-9]+)))(.*)").expect("cannot compile regex");
        }
        MATCHER.with(|matcher| {
            let lowercased_doi = value.to_lowercase();
            for cap in matcher.captures_iter(&lowercased_doi) {
                let (_, isbn, _) = (&cap[1], &cap[2], &cap[3]);
                let corrected_isbn = isbn.replace('-', "").replace("cbo", "");
                if corrected_isbn.len() == 10 || corrected_isbn.len() == 13 {
                    let subqueries: Vec<_> = self
                        .fields
                        .iter()
                        .map(|field_name| {
                            let (field, full_path) = self.schema.find_field(field_name).expect("inconsistent state");
                            let term = cast_field_to_term(&field, full_path, self.schema.get_field_entry(field).field_type(), &corrected_isbn, true);
                            let query = Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>;
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
    fields: Vec<String>,
    is_terminal: bool,
}

impl IsbnMapper {
    pub fn new(schema: Schema, fields: Vec<String>) -> Self {
        IsbnMapper {
            schema,
            fields,
            is_terminal: true,
        }
    }
}

impl TermFieldMapper for IsbnMapper {
    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn map(&self, value: &str) -> Option<Box<dyn Query>> {
        let subqueries: Vec<_> = self
            .fields
            .iter()
            .map(|field_name| {
                let (field, full_path) = self.schema.find_field(field_name).expect("inconsistent state");
                let term = cast_field_to_term(
                    &field,
                    full_path,
                    self.schema.get_field_entry(field).field_type(),
                    &value.replace('-', ""),
                    true,
                );
                let query = Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>;
                (Occur::Should, query)
            })
            .collect();
        Some(Box::new(BooleanQuery::new(subqueries)) as Box<dyn Query>)
    }
}
