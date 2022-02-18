use std::collections::Bound;
use std::ops::Bound::{Included, Unbounded};
use std::str::FromStr;

use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
use rustc_hash::FxHashMap;
use tantivy::query::{BooleanQuery, BoostQuery, PhraseQuery, Query, RangeQuery, TermQuery};
use tantivy::schema::{FacetParseError, Field, FieldEntry, FieldType, IndexRecordOption, Schema, TextFieldIndexing, Type};
use tantivy::tokenizer::{TextAnalyzer, TokenizerManager};
use tantivy::{Score, Term};
use tantivy_query_grammar::Occur;

use crate::errors::SummaResult;
use crate::validators;

#[derive(Parser)]
#[grammar = "src/components/query_parser/summa_ql.pest"] // relative to src
struct SummaQlParser;

pub enum MissingFieldPolicy {
    AsUsualTerms,
    Remove,
}

pub struct QueryParser {
    schema: Schema,
    default_fields: Vec<Field>,
    tokenizer_manager: TokenizerManager,
    boost: FxHashMap<Field, Score>,
    missing_field_policy: MissingFieldPolicy,
}

/// Possible error that may happen when parsing a query.
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum QueryParserError {
    /// Error in the query syntax
    #[error("syntax_error: {0}")]
    Syntax(String),
    /// This query is unsupported.
    #[error("unsupported_query_error: {0}")]
    UnsupportedQuery(String),
    /// The query references a field that is not in the schema
    #[error("field_doest_not_exist_error: '{0}'")]
    FieldDoesNotExist(String),
    /// The query contains a term for a `u64` or `i64`-field, but the value
    /// is neither.
    #[error("expected_int_error: '{0:?}'")]
    ExpectedInt(#[from] std::num::ParseIntError),
    /// The query contains a term for a bytes field, but the value is not valid
    /// base64.
    #[error("expected_base64_error: '{0:?}'")]
    ExpectedBase64(#[from] base64::DecodeError),
    /// The query contains a term for a `f64`-field, but the value
    /// is not a f64.
    #[error("expected_float_error: {0}")]
    ExpectedFloat(#[from] std::num::ParseFloatError),
    /// The query contains a term for a bool field, but the value
    /// is not a bool.
    #[error("exptected_bool: '{0:?}'")]
    ExpectedBool(#[from] std::str::ParseBoolError),
    /// It is forbidden queries that are only "excluding". (e.g. -title:pop)
    #[error("all_but_query_forbidden_error")]
    AllButQueryForbidden,
    /// If no default field is declared, running a query without any
    /// field specified is forbbidden.
    #[error("No default field declared and no field specified in query")]
    NoDefaultFieldDeclared,
    /// The field searched for is not declared
    /// as indexed in the schema.
    #[error("field_not_indexed_error: {0}")]
    FieldNotIndexed(String),
    /// A phrase query was requested for a field that does not
    /// have any positions indexed.
    #[error("field_does_not_have_positions_indexed_error: {0}")]
    FieldDoesNotHavePositionsIndexed(String),
    /// The tokenizer for the given field is unknown
    /// The two argument strings are the name of the field, the name of the tokenizer
    #[error("unknown_tokenizer_error: '{tokenizer:?}' for the field '{field:?}'")]
    UnknownTokenizer {
        /// The name of the tokenizer
        tokenizer: String,
        /// The field name
        field: String,
    },
    /// The query contains a range query with a phrase as one of the bounds.
    /// Only terms can be used as bounds.
    #[error("range_must_not_have_phrase")]
    RangeMustNotHavePhrase,
    /// The format for the date field is not RFC 3339 compliant.
    #[error("date_format_error: {0}")]
    DateFormat(#[from] time::error::Parse),
    /// The format for the facet field is invalid.
    #[error("facet_parse_error: {0}")]
    FacetFormat(#[from] FacetParseError),
    /// The format for the ip field is invalid.
    #[error("ip_format_error: {0}")]
    IpFormat(#[from] std::net::AddrParseError),
    /// Pest parser failed to parse string
    #[error("pest_error: {0}")]
    Pest(#[from] Box<pest::error::Error<Rule>>),
}

pub(crate) fn is_type_valid_for_fastfield_range_query(typ: Type) -> bool {
    match typ {
        Type::U64 | Type::I64 | Type::F64 | Type::Bool | Type::Date => true,
        Type::IpAddr => true,
        Type::Str | Type::Facet | Type::Bytes | Type::Json => false,
    }
}

type Subqueries = Vec<(Occur, Box<dyn Query>)>;

impl QueryParser {
    pub fn new(schema: Schema, default_fields: Vec<String>, tokenizer_manager: &TokenizerManager) -> SummaResult<QueryParser> {
        let default_fields = validators::parse_fields(&schema, &default_fields)?;
        Ok(QueryParser {
            schema,
            default_fields,
            tokenizer_manager: tokenizer_manager.clone(),
            boost: Default::default(),
            missing_field_policy: MissingFieldPolicy::Remove,
        })
    }

    fn get_text_analyzer(&self, field_entry: &FieldEntry, option: &TextFieldIndexing) -> Result<TextAnalyzer, QueryParserError> {
        self.tokenizer_manager
            .get(option.tokenizer())
            .ok_or_else(|| QueryParserError::UnknownTokenizer {
                field: field_entry.name().to_string(),
                tokenizer: option.tokenizer().to_string(),
            })
    }

    fn default_fields_literal(&self, literal: Pair<Rule>) -> Result<Vec<Box<dyn Query>>, QueryParserError> {
        Ok(self
            .default_fields
            .iter()
            .map(|field| self.parse_literal(field, literal.clone()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect())
    }

    fn parse_literal(&self, field: &Field, literal: Pair<Rule>) -> Result<Vec<Box<dyn Query>>, QueryParserError> {
        let literal = literal.into_inner().next().expect("grammar failure");
        let field_entry = self.schema.get_field_entry(*field);
        let field_type = field_entry.field_type();

        if !field_type.is_indexed() {
            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
        }

        return match *field_type {
            FieldType::U64(_) => {
                let val: u64 = u64::from_str(literal.as_str())?;
                Ok(vec![
                    Box::new(TermQuery::new(Term::from_field_u64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>
                ])
            }
            FieldType::I64(_) => {
                let val: i64 = i64::from_str(literal.as_str())?;
                Ok(vec![
                    Box::new(TermQuery::new(Term::from_field_i64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>
                ])
            }
            FieldType::F64(_) => {
                let val: f64 = f64::from_str(literal.as_str())?;
                Ok(vec![
                    Box::new(TermQuery::new(Term::from_field_f64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>
                ])
            }
            FieldType::Bool(_) => {
                let val: bool = bool::from_str(literal.as_str())?;
                Ok(vec![
                    Box::new(TermQuery::new(Term::from_field_bool(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>
                ])
            }
            FieldType::Str(ref str_options) => {
                let option = str_options.get_indexing_options().ok_or_else(|| {
                    // This should have been seen earlier really.
                    QueryParserError::FieldNotIndexed(field_entry.name().to_string())
                })?;
                let text_analyzer = self.get_text_analyzer(field_entry, option)?;
                match literal.as_rule() {
                    Rule::word | Rule::field_name => {
                        let mut token_stream = text_analyzer.token_stream(literal.as_str());
                        let mut term_queries = Vec::new();
                        token_stream.process(&mut |token| {
                            let term_query =
                                Box::new(TermQuery::new(Term::from_field_text(*field, &token.text), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                            term_queries.push(term_query);
                        });
                        Ok(term_queries)
                    }
                    Rule::phrase => {
                        let mut phrase_pairs = literal.into_inner();
                        let words = phrase_pairs.next().expect("grammar failure");
                        let slop = phrase_pairs.next().map(|v| u32::from_str(v.as_str()).expect("grammar failure")).unwrap_or(0);
                        let mut token_stream = text_analyzer.token_stream(words.as_str());
                        let mut terms = Vec::new();
                        token_stream.process(&mut |token| {
                            let term = Term::from_field_text(*field, &token.text);
                            terms.push((token.position, term));
                        });
                        if terms.len() <= 1 {
                            return Ok(terms
                                .into_iter()
                                .map(|(_, term)| Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>)
                                .collect());
                        }
                        if !option.index_option().has_positions() {
                            return Err(QueryParserError::FieldDoesNotHavePositionsIndexed(field_entry.name().to_string()));
                        }
                        return Ok(vec![Box::new(PhraseQuery::new_with_offset_and_slop(terms, slop))]);
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        };
    }

    fn compute_boundary_term(&self, field: Field, phrase: &str) -> Result<Term, QueryParserError> {
        let field_entry = self.schema.get_field_entry(field);
        let field_type = field_entry.field_type();
        let field_supports_ff_range_queries = field_type.is_fast() && is_type_valid_for_fastfield_range_query(field_type.value_type());

        if !field_type.is_indexed() && !field_supports_ff_range_queries {
            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
        }

        match *field_type {
            FieldType::U64(_) => {
                let val: u64 = u64::from_str(phrase)?;
                Ok(Term::from_field_u64(field, val))
            }
            FieldType::I64(_) => {
                let val: i64 = i64::from_str(phrase)?;
                Ok(Term::from_field_i64(field, val))
            }
            FieldType::F64(_) => {
                let val: f64 = f64::from_str(phrase)?;
                Ok(Term::from_field_f64(field, val))
            }
            FieldType::Bool(_) => {
                let val: bool = bool::from_str(phrase)?;
                Ok(Term::from_field_bool(field, val))
            }
            FieldType::Str(ref str_options) => {
                let option = str_options.get_indexing_options().ok_or_else(|| {
                    // This should have been seen earlier really.
                    QueryParserError::FieldNotIndexed(field_entry.name().to_string())
                })?;
                let text_analyzer = self
                    .tokenizer_manager
                    .get(option.tokenizer())
                    .ok_or_else(|| QueryParserError::UnknownTokenizer {
                        field: field_entry.name().to_string(),
                        tokenizer: option.tokenizer().to_string(),
                    })?;
                let mut terms: Vec<Term> = Vec::new();
                let mut token_stream = text_analyzer.token_stream(phrase);
                token_stream.process(&mut |token| {
                    let term = Term::from_field_text(field, &token.text);
                    terms.push(term);
                });
                if terms.len() != 1 {
                    return Err(QueryParserError::UnsupportedQuery(format!(
                        "Range query boundary cannot have multiple tokens: {phrase:?}."
                    )));
                }
                Ok(terms.into_iter().next().expect("grammar failure"))
            }
            _ => unreachable!(),
        }
    }

    fn parse_boundary_word(&self, field: Field, boundary_word: Pair<Rule>) -> Result<Bound<Term>, QueryParserError> {
        Ok(match boundary_word.as_rule() {
            Rule::star => Unbounded,
            Rule::word => Included(self.compute_boundary_term(field, boundary_word.as_str())?),
            _ => unreachable!(),
        })
    }

    fn parse_statement(&self, pair: Pair<Rule>) -> Result<Vec<Box<dyn Query>>, QueryParserError> {
        let mut statement_pairs = pair.into_inner();
        let statement = statement_pairs.next().expect("grammar failure");
        let boost = statement_pairs.next().map(|boost| f32::from_str(boost.as_str()).expect("grammar failure"));
        let mut statement_result = match statement.as_rule() {
            Rule::search_group => {
                let mut search_group = statement.into_inner();
                let field_name = search_group.next().expect("grammar failure");
                let literal_or_range = search_group.next().expect("grammar failure");
                match literal_or_range.as_rule() {
                    Rule::literal => match self.schema.get_field(field_name.as_str()) {
                        Ok(field) => self.parse_literal(&field, literal_or_range),
                        Err(tantivy::TantivyError::FieldNotFound(_)) => match self.missing_field_policy {
                            MissingFieldPolicy::AsUsualTerms => Ok(self
                                .default_fields_literal(field_name)?
                                .into_iter()
                                .chain(self.default_fields_literal(literal_or_range)?.into_iter())
                                .collect()),
                            MissingFieldPolicy::Remove => Ok(vec![]),
                        },
                        _ => unreachable!(),
                    },
                    Rule::range => {
                        let mut range_pairs = literal_or_range.into_inner();
                        let field = self.schema.get_field(field_name.as_str()).expect("grammar failure");
                        let field_entry = self.schema.get_field_entry(field);
                        let left = self.parse_boundary_word(field, range_pairs.next().expect("grammar failure"))?;
                        let right = self.parse_boundary_word(field, range_pairs.next().expect("grammar failure"))?;
                        Ok(vec![Box::new(RangeQuery::new_term_bounds(
                            field_entry.name().to_string(),
                            field_entry.field_type().value_type(),
                            &left,
                            &right,
                        )) as Box<dyn Query>])
                    }
                    _ => unreachable!(),
                }
            }
            Rule::literal => self.default_fields_literal(statement),
            e => panic!("{e:?}"),
        };
        if let Some(boost) = boost {
            statement_result = statement_result.map(|statement_result| {
                statement_result
                    .into_iter()
                    .map(|q| Box::new(BoostQuery::new(q, boost)) as Box<dyn Query>)
                    .collect()
            })
        }
        statement_result
    }

    fn process_pairs(&self, pairs: Pairs<Rule>) -> Result<Subqueries, QueryParserError> {
        let mut subqueries = Subqueries::new();
        for pair in pairs {
            let (occur, statement) = match pair.as_rule() {
                Rule::default_expression => (Occur::Should, pair),
                Rule::positive_expression => (Occur::Must, pair),
                Rule::negative_expression => (Occur::MustNot, pair),
                _ => unreachable!(),
            };
            let terms = self.parse_statement(statement.into_inner().next().expect("grammar failure"))?;
            subqueries.extend(terms.into_iter().map(|term| (occur, term)));
        }
        return Ok(subqueries);
    }

    pub(crate) fn parse_query(&self, query: &str) -> Result<Box<dyn Query>, QueryParserError> {
        let pairs = SummaQlParser::parse(Rule::main, query).map_err(Box::new)?;
        Ok(Box::new(BooleanQuery::new(self.process_pairs(pairs)?)))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use tantivy::schema::{INDEXED, TEXT};

    use super::*;

    #[test]
    pub fn test_parser() -> Result<(), Box<dyn Error>> {
        let tokenizer_manager = TokenizerManager::default();
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("title", TEXT);
        schema_builder.add_text_field("body", TEXT);
        schema_builder.add_i64_field("timestamp", INDEXED);
        let schema = schema_builder.build();

        let default_fields = vec!["title".to_string()];

        let query_parser = QueryParser::new(schema, default_fields, &tokenizer_manager)?;
        let query = query_parser.parse_query("search engine");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engine\")))] })");
        let query = query_parser.parse_query("'search engine'");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(0), phrase_terms: [(0, Term(type=Str, field=0, \"search\")), (1, Term(type=Str, field=0, \"engine\"))], slop: 0 })] })");
        let query = query_parser.parse_query("body:'search engine'~4");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(type=Str, field=1, \"search\")), (1, Term(type=Str, field=1, \"engine\"))], slop: 4 })] })");
        let query = query_parser.parse_query("body:'search engine'");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(type=Str, field=1, \"search\")), (1, Term(type=Str, field=1, \"engine\"))], slop: 0 })] })");
        let query = query_parser.parse_query("title:search engine");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engine\")))] })");
        let query = query_parser.parse_query("Search Engines: The Ultimate, Only Guide!");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engines\"))), (Should, TermQuery(Term(type=Str, field=0, \"the\"))), (Should, TermQuery(Term(type=Str, field=0, \"ultimate\"))), (Should, TermQuery(Term(type=Str, field=0, \"only\"))), (Should, TermQuery(Term(type=Str, field=0, \"guide\")))] })");
        let query = query_parser.parse_query("not_field:search engine");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"engine\")))] })"
        );
        let query = query_parser.parse_query("+body:search -engine");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Must, TermQuery(Term(type=Str, field=1, \"search\"))), (MustNot, TermQuery(Term(type=Str, field=0, \"engine\")))] })"
        );
        let query = query_parser.parse_query("+body:'search engine'");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Must, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(type=Str, field=1, \"search\")), (1, Term(type=Str, field=1, \"engine\"))], slop: 0 })] })"
        );
        let query = query_parser.parse_query("search^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(type=Str, field=0, \"search\")), boost=2))] })"
        );
        let query = query_parser.parse_query("'search engine'~3^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, Boost(query=PhraseQuery { field: Field(0), phrase_terms: [(0, Term(type=Str, field=0, \"search\")), (1, Term(type=Str, field=0, \"engine\"))], slop: 3 }, boost=2))] })"
        );
        let query = query_parser.parse_query("search engine^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, Boost(query=TermQuery(Term(type=Str, field=0, \"engine\")), boost=2))] })"
        );
        let query = query_parser.parse_query("body:title^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(type=Str, field=1, \"title\")), boost=2))] })"
        );
        let query = query_parser.parse_query("body:'title'^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(type=Str, field=1, \"title\")), boost=2))] })"
        );
        let query = query_parser.parse_query("body:[aaa TO ccc]");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, RangeQuery { field: \"body\", value_type: Str, left_bound: Included([97, 97, 97]), right_bound: Included([99, 99, 99]) })] })");
        let query = query_parser.parse_query("body:[ a to  * ]");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, RangeQuery { field: \"body\", value_type: Str, left_bound: Included([97]), right_bound: Unbounded })] })"
        );
        let query = query_parser.parse_query("timestamp:[ 1000 to 2000 ]");
        assert_eq!(format!("{:?}", query), "");
        Ok(())
    }
}
