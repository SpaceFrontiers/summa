use std::collections::Bound;
use std::ops::Bound::{Included, Unbounded};
use std::str::FromStr;

use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
use tantivy::query::{BooleanQuery, BoostQuery, PhraseQuery, Query, RangeQuery, TermQuery};
use tantivy::schema::{FacetParseError, Field, FieldEntry, FieldType, IndexRecordOption, Schema, TextFieldIndexing, Type};
use tantivy::tokenizer::{TextAnalyzer, TokenizerManager};
use tantivy::{Index, Term};
use tantivy_query_grammar::Occur;

use crate::errors::SummaResult;
use crate::validators;

#[derive(Parser)]
#[grammar = "src/components/query_parser/summa_ql.pest"] // relative to src
struct SummaQlParser;

pub enum MissingFieldPolicy {
    AsUsualTerms,
    Remove,
    Fail,
}

pub struct QueryParser {
    schema: Schema,
    default_fields: Vec<Field>,
    tokenizer_manager: TokenizerManager,
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
            missing_field_policy: MissingFieldPolicy::Remove,
        })
    }

    pub fn for_index(index: &Index, default_fields: Vec<String>) -> SummaResult<QueryParser> {
        QueryParser::new(index.schema(), default_fields, index.tokenizers())
    }

    pub fn set_missing_field_policy(&mut self, missing_field_policy: MissingFieldPolicy) {
        self.missing_field_policy = missing_field_policy
    }

    fn get_text_analyzer(&self, field_entry: &FieldEntry, option: &TextFieldIndexing) -> Result<TextAnalyzer, QueryParserError> {
        self.tokenizer_manager
            .get(option.tokenizer())
            .ok_or_else(|| QueryParserError::UnknownTokenizer {
                field: field_entry.name().to_string(),
                tokenizer: option.tokenizer().to_string(),
            })
    }

    fn default_fields_term(&self, term: Pair<Rule>) -> Result<Subqueries, QueryParserError> {
        Ok(self
            .default_fields
            .iter()
            .map(|field| self.parse_term(field, term.clone()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect())
    }

    fn parse_range(&self, pre_term: Pair<Rule>, field: &Field) -> Result<RangeQuery, QueryParserError> {
        let mut range_pairs = pre_term.into_inner();
        let field_entry = self.schema.get_field_entry(*field);
        let left = self.parse_boundary_word(*field, range_pairs.next().expect("grammar failure"))?;
        let right = self.parse_boundary_word(*field, range_pairs.next().expect("grammar failure"))?;
        Ok(RangeQuery::new_term_bounds(
            field_entry.name().to_string(),
            field_entry.field_type().value_type(),
            &left,
            &right,
        ))
    }

    fn parse_term(&self, field: &Field, term: Pair<Rule>) -> Result<Subqueries, QueryParserError> {
        let term = term.into_inner().next().expect("grammar failure");
        let occur = match term.as_rule() {
            Rule::positive_term => Occur::Must,
            Rule::negative_term => Occur::MustNot,
            Rule::default_term => Occur::Should,
            _ => unreachable!(),
        };
        let pre_term = term.into_inner().next().expect("grammar failure");

        let field_entry = self.schema.get_field_entry(*field);
        let field_type = field_entry.field_type();

        if !field_type.is_indexed() {
            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
        }

        return match *field_type {
            FieldType::U64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![(occur, Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>)]),
                Rule::phrase | Rule::word => {
                    let val: u64 = u64::from_str(pre_term.as_str()).unwrap();
                    Ok(vec![(
                        occur,
                        Box::new(TermQuery::new(Term::from_field_u64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                    )])
                }
                _ => unreachable!(),
            },
            FieldType::I64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![(occur, Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>)]),
                Rule::phrase | Rule::word => {
                    let val: i64 = i64::from_str(pre_term.as_str()).unwrap();
                    Ok(vec![(
                        occur,
                        Box::new(TermQuery::new(Term::from_field_i64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                    )])
                }
                _ => unreachable!(),
            },
            FieldType::F64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![(occur, Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>)]),
                Rule::phrase | Rule::word => {
                    let val: f64 = f64::from_str(pre_term.as_str()).unwrap();
                    Ok(vec![(
                        occur,
                        Box::new(TermQuery::new(Term::from_field_f64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                    )])
                }
                _ => unreachable!(),
            },
            FieldType::Bool(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![(occur, Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>)]),
                Rule::phrase | Rule::word => {
                    let val: bool = bool::from_str(pre_term.as_str()).unwrap();
                    Ok(vec![(
                        occur,
                        Box::new(TermQuery::new(Term::from_field_bool(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>,
                    )])
                }
                _ => unreachable!(),
            },
            FieldType::Str(ref str_options) => {
                let option = str_options.get_indexing_options().ok_or_else(|| {
                    // This should have been seen earlier really.
                    QueryParserError::FieldNotIndexed(field_entry.name().to_string())
                })?;
                let text_analyzer = self.get_text_analyzer(field_entry, option)?;
                match pre_term.as_rule() {
                    Rule::word | Rule::field_name => {
                        let mut token_stream = text_analyzer.token_stream(pre_term.as_str());
                        let mut term_queries = Vec::new();
                        token_stream.process(&mut |token| {
                            let term_query =
                                Box::new(TermQuery::new(Term::from_field_text(*field, &token.text), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                            term_queries.push((occur, term_query));
                        });
                        Ok(term_queries)
                    }
                    Rule::phrase => {
                        let mut phrase_pairs = pre_term.into_inner();
                        let words = match phrase_pairs.next() {
                            None => return Ok(vec![]),
                            Some(words) => words,
                        };
                        let slop = phrase_pairs
                            .next()
                            .map(|v| match v.as_str() {
                                "" => 0,
                                _ => u32::from_str(v.as_str()).expect("cannot parse"),
                            })
                            .unwrap_or(0);
                        let mut token_stream = text_analyzer.token_stream(words.as_str());
                        let mut terms = Vec::new();
                        token_stream.process(&mut |token| {
                            let term = Term::from_field_text(*field, &token.text);
                            terms.push((token.position, term));
                        });
                        if terms.len() <= 1 {
                            return Ok(terms
                                .into_iter()
                                .map(|(_, term)| (occur, Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>))
                                .collect());
                        }
                        if !option.index_option().has_positions() {
                            return Err(QueryParserError::FieldDoesNotHavePositionsIndexed(field_entry.name().to_string()));
                        }
                        return Ok(vec![(occur, Box::new(PhraseQuery::new_with_offset_and_slop(terms, slop)))]);
                    }
                    Rule::range => Ok(vec![(occur, Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>)]),
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

    fn parse_statement(&self, pair: Pair<Rule>) -> Result<Subqueries, QueryParserError> {
        let mut statement_pairs = pair.into_inner();
        let search_group_or_term = statement_pairs.next().expect("grammar failure");
        let boost = statement_pairs.next().map(|boost| f32::from_str(boost.as_str()).expect("grammar failure"));
        let statement_result = match search_group_or_term.as_rule() {
            Rule::search_group => {
                let mut search_group = search_group_or_term.into_inner();
                let field_name = search_group.next().expect("grammar failure");
                let grouping_or_term = search_group.next().expect("grammar failure");
                match grouping_or_term.as_rule() {
                    Rule::grouping => {
                        let mut intermediate_results = vec![];
                        match self.schema.get_field(field_name.as_str()) {
                            Ok(field) => {
                                for term in grouping_or_term.into_inner() {
                                    intermediate_results.push(self.parse_term(&field, term)?)
                                }
                            }
                            Err(tantivy::TantivyError::FieldNotFound(_)) => match self.missing_field_policy {
                                MissingFieldPolicy::AsUsualTerms => {
                                    intermediate_results.push(self.default_fields_term(field_name)?);
                                    for term in grouping_or_term.into_inner() {
                                        intermediate_results.push(self.default_fields_term(term)?)
                                    }
                                }
                                MissingFieldPolicy::Remove => return Ok(vec![]),
                                MissingFieldPolicy::Fail => return Err(QueryParserError::FieldDoesNotExist(field_name.as_str().to_string())),
                            },
                            _ => unreachable!(),
                        }
                        Ok(intermediate_results.into_iter().flatten().collect())
                    }
                    Rule::term => match self.schema.get_field(field_name.as_str()) {
                        Ok(field) => self.parse_term(&field, grouping_or_term),
                        Err(tantivy::TantivyError::FieldNotFound(_)) => match self.missing_field_policy {
                            MissingFieldPolicy::AsUsualTerms => Ok(self
                                .default_fields_term(field_name)?
                                .into_iter()
                                .chain(self.default_fields_term(grouping_or_term)?.into_iter())
                                .collect()),
                            MissingFieldPolicy::Remove => Ok(vec![]),
                            MissingFieldPolicy::Fail => Err(QueryParserError::FieldDoesNotExist(field_name.as_str().to_string())),
                        },
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }
            }
            Rule::term => self.default_fields_term(search_group_or_term),
            e => panic!("{e:?}"),
        }?;
        if let Some(boost) = boost {
            Ok(statement_result
                .into_iter()
                .map(|(occur, query)| (occur, Box::new(BoostQuery::new(query, boost)) as Box<dyn Query>))
                .collect())
        } else {
            Ok(statement_result)
        }
    }

    fn parse_statements(&self, pairs: Pairs<Rule>) -> Result<Subqueries, QueryParserError> {
        let mut subqueries = Subqueries::new();
        for pair in pairs {
            let parsed_queries = self.parse_statement(pair)?;
            subqueries.extend(parsed_queries);
        }
        Ok(subqueries)
    }

    pub fn parse_query(&self, query: &str) -> Result<Box<dyn Query>, QueryParserError> {
        let pairs = SummaQlParser::parse(Rule::main, query).map_err(Box::new)?;
        let parsed = self.parse_statements(pairs)?;
        if parsed.len() == 0 {
            Ok(Box::new(tantivy::query::EmptyQuery {}) as Box<dyn Query>)
        } else {
            Ok(Box::new(BooleanQuery::new(parsed)) as Box<dyn Query>)
        }
    }
}

#[cfg(test)]
mod tests {
    use tantivy::schema::{INDEXED, STRING, TEXT};

    use super::*;

    fn create_query_parser() -> QueryParser {
        let tokenizer_manager = TokenizerManager::default();
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("title", TEXT);
        schema_builder.add_text_field("body", TEXT);
        schema_builder.add_i64_field("timestamp", INDEXED);
        schema_builder.add_text_field("doi", STRING);
        let schema = schema_builder.build();
        let default_fields = vec!["title".to_string()];
        QueryParser::new(schema, default_fields, &tokenizer_manager).expect("cannot create parser")
    }

    #[test]
    pub fn test_parser_base() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("search engine");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engine\")))] })");
        let query = query_parser.parse_query("'search engine'");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(0), phrase_terms: [(0, Term(type=Str, field=0, \"search\")), (1, Term(type=Str, field=0, \"engine\"))], slop: 0 })] })");
    }

    #[test]
    pub fn test_parser_slop() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("body:'search engine'~4");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(type=Str, field=1, \"search\")), (1, Term(type=Str, field=1, \"engine\"))], slop: 4 })] })");
    }

    #[test]
    pub fn test_parser_fields() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:'search engine'")),
            "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(type=Str, field=1, \"search\")), (1, Term(type=Str, field=1, \"engine\"))], slop: 0 })] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("timestamp:10")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=I64, field=2, 10)))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("title:search engine")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engine\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("not_field:search engine")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"engine\")))] })"
        );
    }

    #[test]
    pub fn test_free_queries() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("Search Engines: The Ultimate, Only Guide!")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engines\"))), (Should, TermQuery(Term(type=Str, field=0, \"the\"))), (Should, TermQuery(Term(type=Str, field=0, \"ultimate\"))), (Should, TermQuery(Term(type=Str, field=0, \"only\"))), (Should, TermQuery(Term(type=Str, field=0, \"guide\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("!! HI !! (SEARCH! ENGINES!")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"hi\"))), (Should, TermQuery(Term(type=Str, field=0, \"search\"))), (Should, TermQuery(Term(type=Str, field=0, \"engines\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("`non closed")),
            "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(0), phrase_terms: [(0, Term(type=Str, field=0, \"non\")), (1, Term(type=Str, field=0, \"closed\"))], slop: 0 })] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("non closed`")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"non\"))), (Should, TermQuery(Term(type=Str, field=0, \"closed\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("title:(search ")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"title\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("title:(search -")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"title\")))] })"
        );
        assert_eq!(format!("{:?}", query_parser.parse_query("``")), "Ok(EmptyQuery)");
        assert_eq!(format!("{:?}", query_parser.parse_query("```")), "Ok(EmptyQuery)");
        assert_eq!(format!("{:?}", query_parser.parse_query(")(")), "Ok(EmptyQuery)");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("(a)(b)`")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"a\"))), (Should, TermQuery(Term(type=Str, field=0, \"b\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:'10.1182/blood.v53.1.19.bloodjournal53119'")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=3, \"10.1182/blood.v53.1.19.bloodjournal53119\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:10.1182/blood.v53.1.19.bloodjournal53119")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=3, \"10.1182/blood.v53.1.19.bloodjournal53119\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("10.10 10/10")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"10\"))), (Should, TermQuery(Term(type=Str, field=0, \"10\"))), (Should, TermQuery(Term(type=Str, field=0, \"10\"))), (Should, TermQuery(Term(type=Str, field=0, \"10\")))] })"
        );
    }

    #[test]
    pub fn test_non_ascii() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:поисковые системы")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=1, \"поисковые\"))), (Should, TermQuery(Term(type=Str, field=0, \"системы\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("(поисковые системы)")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"поисковые\"))), (Should, TermQuery(Term(type=Str, field=0, \"системы\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("поисковые: системы")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(type=Str, field=0, \"поисковые\"))), (Should, TermQuery(Term(type=Str, field=0, \"системы\")))] })"
        );
    }

    #[test]
    pub fn test_plus_minus() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("body:+search -engine");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Must, TermQuery(Term(type=Str, field=1, \"search\"))), (MustNot, TermQuery(Term(type=Str, field=0, \"engine\")))] })"
        );
        let query = query_parser.parse_query("body:+'search engine'");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Must, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(type=Str, field=1, \"search\")), (1, Term(type=Str, field=1, \"engine\"))], slop: 0 })] })"
        );
    }

    #[test]
    pub fn test_parser_boostings() {
        let query_parser = create_query_parser();
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
    }

    #[test]
    pub fn test_range_queries() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("body:[aaa TO ccc]");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, RangeQuery { field: \"body\", value_type: Str, left_bound: Included([97, 97, 97]), right_bound: Included([99, 99, 99]) })] })");
        let query = query_parser.parse_query("body:[ a to  * ]");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, RangeQuery { field: \"body\", value_type: Str, left_bound: Included([97]), right_bound: Unbounded })] })"
        );
        let query = query_parser.parse_query("timestamp:[ 1000 to 2000 ]");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, RangeQuery { field: \"timestamp\", value_type: I64, left_bound: Included([128, 0, 0, 0, 0, 0, 3, 232]), right_bound: Included([128, 0, 0, 0, 0, 0, 7, 208]) })] })");
        let query = query_parser.parse_query("timestamp:(-[1100 to 1200] [ 1000 to 2000 ] -1500 +3000)");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(MustNot, RangeQuery { field: \"timestamp\", value_type: I64, left_bound: Included([128, 0, 0, 0, 0, 0, 4, 76]), right_bound: Included([128, 0, 0, 0, 0, 0, 4, 176]) }), (Should, RangeQuery { field: \"timestamp\", value_type: I64, left_bound: Included([128, 0, 0, 0, 0, 0, 3, 232]), right_bound: Included([128, 0, 0, 0, 0, 0, 7, 208]) }), (MustNot, TermQuery(Term(type=I64, field=2, 1500))), (Must, TermQuery(Term(type=I64, field=2, 3000)))] })");
    }
}
