use std::collections::{Bound, HashMap};
use std::ops::Bound::{Included, Unbounded};
use std::ops::Deref;
use std::str::FromStr;

use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
use safe_regex::{regex, Matcher3};
use summa_proto::proto;
use tantivy::json_utils::{convert_to_fast_value_and_get_term, JsonTermWriter};
use tantivy::query::{BooleanQuery, BoostQuery, DisjunctionMaxQuery, EmptyQuery, PhraseQuery, Query, QueryClone, RangeQuery, RegexQuery, TermQuery};
use tantivy::schema::{FacetParseError, Field, FieldEntry, FieldType, IndexRecordOption, Schema, TextFieldIndexing, Type};
use tantivy::tokenizer::{TextAnalyzer, TokenizerManager};
use tantivy::{Index, Term};
use tantivy_common::HasLen;
use tantivy_query_grammar::Occur;

use crate::components::query_parser::proto_query_parser::MatchQueryDefaultMode;
use crate::errors::SummaResult;
use crate::utils::transpose;
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
    limit: usize,
    field_aliases: HashMap<String, String>,
    field_boosts: Option<HashMap<String, f32>>,
    default_mode: MatchQueryDefaultMode,
    exact_matches_promoter: Option<proto::ExactMatchesPromoter>,
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
    /// The field is declated as JSON but the query does not contain object path
    #[error("json_field_without_path_error: {0}")]
    JsonFieldWithoutPath(String),
    /// The field is not declated as JSON but the query does contain object path
    #[error("non_json_field_with_path_error: {0}")]
    NonJsonFieldWithPath(String),
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

fn boost_query(query: Box<dyn Query>, boost: Option<f32>) -> Box<dyn Query> {
    if let Some(boost) = boost {
        return Box::new(BoostQuery::new(query, boost)) as Box<dyn Query>;
    }
    query
}

fn multiply_boosts(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a * b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn reduce_should_clause(query: Box<dyn Query>) -> Box<dyn Query> {
    if let Some(boolean_query) = query.deref().as_any().downcast_ref::<BooleanQuery>() {
        let mut subqueries = vec![];
        for (occur, nested_query) in boolean_query.clauses() {
            let nested_query = reduce_should_clause(nested_query.box_clone());
            match occur {
                Occur::Must | Occur::MustNot => subqueries.push((*occur, reduce_should_clause(nested_query.box_clone()))),
                Occur::Should => {
                    if let Some(nested_nested_query) = nested_query.deref().as_any().downcast_ref::<BooleanQuery>() {
                        subqueries.extend(nested_nested_query.clauses().iter().map(|(o, q)| (*o, q.box_clone())))
                    } else {
                        subqueries.push((*occur, reduce_should_clause(nested_query.box_clone())))
                    }
                }
            }
        }
        if subqueries.len() == 1 && subqueries[0].0 == Occur::Should {
            return subqueries[0].1.box_clone();
        }
        return Box::new(BooleanQuery::new(subqueries)) as Box<dyn Query>;
    }
    query
}

fn reduce_empty_queries(query: Box<dyn Query>) -> Box<dyn Query> {
    if let Some(boolean_query) = query.deref().as_any().downcast_ref::<BooleanQuery>() {
        let subqueries: Vec<_> = boolean_query
            .clauses()
            .iter()
            .filter_map(|(occur, nested_query)| {
                if nested_query.deref().as_any().downcast_ref::<EmptyQuery>().is_some() {
                    None
                } else {
                    Some((*occur, reduce_empty_queries(nested_query.box_clone())))
                }
            })
            .collect();
        if subqueries.is_empty() {
            return Box::new(EmptyQuery {}) as Box<dyn Query>;
        }
        return Box::new(BooleanQuery::new(subqueries)) as Box<dyn Query>;
    }
    query
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
            limit: 16,
            field_aliases: HashMap::new(),
            field_boosts: None,
            default_mode: MatchQueryDefaultMode::Boolean,
            exact_matches_promoter: None,
        })
    }

    pub fn for_index(index: &Index, default_fields: Vec<String>) -> SummaResult<QueryParser> {
        QueryParser::new(index.schema(), default_fields, index.tokenizers())
    }

    pub fn set_field_aliases(&mut self, field_aliases: HashMap<String, String>) {
        self.field_aliases = field_aliases;
    }

    pub fn set_field_boosts(&mut self, field_boosts: HashMap<String, f32>) {
        self.field_boosts = Some(field_boosts);
    }

    pub fn set_default_mode(&mut self, default_mode: MatchQueryDefaultMode) {
        self.default_mode = default_mode;
    }

    pub fn set_missing_field_policy(&mut self, missing_field_policy: MissingFieldPolicy) {
        self.missing_field_policy = missing_field_policy;
    }

    pub fn set_exact_match_promoter(&mut self, exact_matches_promoter: proto::ExactMatchesPromoter) {
        self.exact_matches_promoter = Some(exact_matches_promoter);
    }

    pub fn resolve_field_name<'a>(&'a self, field_name: &'a str) -> &str {
        self.field_aliases.get(field_name).map(|s| s.as_str()).unwrap_or(field_name)
    }

    fn get_text_analyzer(&self, field_entry: &FieldEntry, option: &TextFieldIndexing) -> Result<TextAnalyzer, QueryParserError> {
        self.tokenizer_manager
            .get(option.tokenizer())
            .ok_or_else(|| QueryParserError::UnknownTokenizer {
                field: field_entry.name().to_string(),
                tokenizer: option.tokenizer().to_string(),
            })
    }

    fn default_fields_term(&self, term: Pair<Rule>, boost: Option<f32>) -> Result<Box<dyn Query>, QueryParserError> {
        let term = term.into_inner().next().expect("grammar failure");
        let occur = self.parse_occur(&term);
        let pre_term = term.into_inner().next().expect("grammar failure");
        let default_field_queries = self
            .default_fields
            .iter()
            .map(|field| self.parse_pre_term(field, "", pre_term.clone(), boost))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(match occur {
            Occur::Should => {
                let default_field_queries = default_field_queries.into_iter().flatten();
                match self.default_mode {
                    MatchQueryDefaultMode::Boolean => Box::new(BooleanQuery::new(default_field_queries.map(|q| (occur, q)).collect())) as Box<dyn Query>,
                    MatchQueryDefaultMode::DisjuctionMax { tie_breaker } => {
                        Box::new(DisjunctionMaxQuery::with_tie_breaker(default_field_queries.collect(), tie_breaker)) as Box<dyn Query>
                    }
                }
            }
            Occur::MustNot => Box::new(BooleanQuery::new(default_field_queries.into_iter().flatten().map(|q| (occur, q)).collect())) as Box<dyn Query>,
            Occur::Must => {
                if self.default_fields.len() == 1 {
                    Box::new(BooleanQuery::new(
                        default_field_queries.into_iter().flatten().map(|q| (Occur::Must, q)).collect(),
                    )) as Box<dyn Query>
                } else {
                    let transposed_default_field_queries = transpose(default_field_queries);
                    Box::new(BooleanQuery::new(
                        transposed_default_field_queries
                            .into_iter()
                            .map(|queries| {
                                (
                                    Occur::Must,
                                    Box::new(BooleanQuery::new(queries.into_iter().map(|q| (Occur::Should, q)).collect())) as Box<dyn Query>,
                                )
                            })
                            .collect(),
                    )) as Box<dyn Query>
                }
            }
        })
    }

    fn parse_range(&self, pre_term: Pair<Rule>, field: &Field) -> Result<RangeQuery, QueryParserError> {
        let mut range_pairs = pre_term.into_inner();
        let field_entry = self.schema.get_field_entry(*field);
        if !field_entry.field_type().is_indexed() && !field_entry.field_type().is_fast() {
            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
        }
        let left = self.parse_boundary_word(*field, range_pairs.next().expect("grammar failure"))?;
        let right = self.parse_boundary_word(*field, range_pairs.next().expect("grammar failure"))?;

        Ok(RangeQuery::new_term_bounds(
            field_entry.name().to_string(),
            field_entry.field_type().value_type(),
            &left,
            &right,
        ))
    }

    fn parse_words(&self, field: Field, option: &TextFieldIndexing, words: &str) -> Result<Vec<(usize, Term)>, QueryParserError> {
        let field_entry = self.schema.get_field_entry(field);
        let text_analyzer = self.get_text_analyzer(field_entry, option)?;
        let mut token_stream = text_analyzer.token_stream(words);
        let mut terms = Vec::new();
        token_stream.process(&mut |token| {
            let term = Term::from_field_text(field, &token.text);
            terms.push((token.position, term));
        });
        Ok(terms)
    }

    fn parse_pre_term(&self, field: &Field, full_path: &str, pre_term: Pair<Rule>, boost: Option<f32>) -> Result<Vec<Box<dyn Query>>, QueryParserError> {
        let field_entry = self.schema.get_field_entry(*field);
        let field_type = field_entry.field_type();

        if field_type.value_type() == Type::Json && full_path.is_empty() {
            return Err(QueryParserError::JsonFieldWithoutPath(field_entry.name().to_string()));
        }

        if field_type.value_type() != Type::Json && !full_path.is_empty() {
            return Err(QueryParserError::NonJsonFieldWithPath(format!(
                "{}.{}",
                field_entry.name().to_string(),
                full_path
            )));
        }

        let boost = multiply_boosts(self.field_boosts.as_ref().and_then(|boosts| boosts.get(field_entry.name()).cloned()), boost);

        if matches!(pre_term.as_rule(), Rule::range) {
            return Ok(vec![boost_query(Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>, boost)]);
        }

        return match *field_type {
            FieldType::U64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    if !field_type.is_indexed() {
                        return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                    }
                    let val: u64 = u64::from_str(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_u64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::I64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    if !field_type.is_indexed() {
                        return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                    }
                    let val: i64 = i64::from_str(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_i64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::F64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    if !field_type.is_indexed() {
                        return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                    }
                    let val: f64 = f64::from_str(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_f64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::Bool(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    if !field_type.is_indexed() {
                        return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                    }
                    let val: bool = bool::from_str(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_bool(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::Str(_) | FieldType::JsonObject(_) => {
                let indexing = if let FieldType::Str(ref text_options) = field_type {
                    text_options.get_indexing_options().expect("unreachable")
                } else if let FieldType::JsonObject(ref json_options) = field_type {
                    json_options.get_text_indexing_options().expect("unreachable")
                } else {
                    unreachable!()
                };

                match pre_term.as_rule() {
                    Rule::word | Rule::field_name => {
                        if !field_type.is_indexed() {
                            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                        }

                        let mut token_stream = self.get_text_analyzer(field_entry, indexing)?.token_stream(pre_term.as_str());
                        let mut queries = Vec::new();

                        token_stream.process(&mut |token| {
                            let term = match field_type {
                                FieldType::Str(_) => Term::from_field_text(*field, &token.text),
                                FieldType::JsonObject(ref json_options) => {
                                    let mut term = Term::with_capacity(128);
                                    let mut json_term_writer =
                                        JsonTermWriter::from_field_and_json_path(*field, full_path, json_options.is_expand_dots_enabled(), &mut term);
                                    convert_to_fast_value_and_get_term(&mut json_term_writer, &token.text).unwrap_or_else(|| {
                                        json_term_writer.set_str(&token.text);
                                        json_term_writer.term().clone()
                                    })
                                }
                                _ => unreachable!(),
                            };
                            let query = Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                            queries.push(boost_query(query, boost));
                        });

                        Ok(queries)
                    }
                    Rule::phrase => {
                        if !field_type.is_indexed() {
                            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                        }

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
                        let terms = self.parse_words(*field, indexing, words.as_str())?;
                        if terms.len() <= 1 {
                            return Ok(terms
                                .into_iter()
                                .map(|(_, term)| {
                                    let query = Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                                    boost_query(query, boost)
                                })
                                .collect());
                        }
                        if !indexing.index_option().has_positions() {
                            return Err(QueryParserError::FieldDoesNotHavePositionsIndexed(field_entry.name().to_string()));
                        }
                        let query = Box::new(PhraseQuery::new_with_offset_and_slop(terms, slop)) as Box<dyn Query>;
                        return Ok(vec![boost_query(query, boost)]);
                    }
                    Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                    Rule::regex => {
                        if !field_type.is_indexed() {
                            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
                        }
                        let query = Box::new(
                            RegexQuery::from_pattern(pre_term.clone().into_inner().next().expect("grammar failure").as_str(), *field)
                                .map_err(|_| QueryParserError::Syntax(pre_term.as_str().to_string()))?,
                        ) as Box<dyn Query>;
                        return Ok(vec![boost_query(query, boost)]);
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        };
    }

    fn parse_occur(&self, occur: &Pair<Rule>) -> Occur {
        match occur.as_rule() {
            Rule::positive_term => Occur::Must,
            Rule::negative_term => Occur::MustNot,
            Rule::default_term => Occur::Should,
            _ => unreachable!(),
        }
    }

    fn parse_term(&self, term: Pair<Rule>, field: &Field, full_path: &str, boost: Option<f32>) -> Result<Box<dyn Query>, QueryParserError> {
        let term = term.into_inner().next().expect("grammar_failure");
        let occur = self.parse_occur(&term);
        let pre_term = term.into_inner().next().expect("grammar failure");
        Ok(Box::new(BooleanQuery::new(
            self.parse_pre_term(field, full_path, pre_term, boost)?
                .into_iter()
                .map(|q| (occur, q))
                .collect(),
        )))
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

    fn extract_top_level_phrase(&self, pairs: Pairs<Rule>) -> Option<String> {
        let mut terms = vec![];
        for pair in pairs {
            let mut statement_pairs = pair.into_inner();
            let search_group_or_term = statement_pairs.next().expect("grammar failure");
            let boost = statement_pairs.next().map(|boost| f32::from_str(boost.as_str()).expect("grammar failure"));
            match (search_group_or_term.as_rule(), boost) {
                (Rule::term, None) => {
                    let term = search_group_or_term.into_inner().next().expect("grammar_failure");
                    let occur = self.parse_occur(&term);
                    let pre_term = term.into_inner().next().expect("grammar failure");
                    if occur == Occur::Should && matches!(pre_term.as_rule(), Rule::word) {
                        terms.push(pre_term.as_str())
                    }
                }
                _ => return None,
            }
        }
        (!terms.is_empty()).then(|| terms.join(" "))
    }

    fn parse_isbn(&self, isbn: &str) -> Result<Box<dyn Query>, QueryParserError> {
        if let Ok(isbns_fields) = self.schema.get_field("isbns") {
            Ok(Box::new(TermQuery::new(
                Term::from_field_text(isbns_fields, &isbn.replace('-', "")),
                IndexRecordOption::Basic,
            )) as Box<dyn Query>)
        } else {
            Ok(Box::new(EmptyQuery) as Box<dyn Query>)
        }
    }

    fn parse_doi(&self, doi: &str) -> Result<Box<dyn Query>, QueryParserError> {
        if let Ok(doi_field) = self.schema.get_field("doi") {
            // ToDo: use more general approach, i.e. use doi tokenizer
            let mut term_queries = vec![];
            let mut boost_original_match = None;
            let lowercased_doi = doi.to_lowercase();
            // ToDo: compile statically
            let matcher: Matcher3<_> = regex!(br"(10.[0-9]+)/((?:cbo)?(?:(?:978[0-9]{10})|(?:978[0-9]{13})|(?:978[-0-9]+)))(.*)");
            if let Some((prefix, isbn, tail)) = matcher.match_slices(lowercased_doi.as_ref()) {
                let isbn = unsafe { std::str::from_utf8_unchecked(isbn) };
                let corrected_isbn = isbn.replace('-', "").replace("cbo", "");
                if (corrected_isbn.len() == 10 || corrected_isbn.len() == 13) && !prefix.is_empty() {
                    if !tail.is_empty() {
                        term_queries.push((
                            Occur::Should,
                            Box::new(TermQuery::new(
                                Term::from_field_text(doi_field, &format!("{}/{}", unsafe { std::str::from_utf8_unchecked(prefix) }, isbn)),
                                IndexRecordOption::Basic,
                            )) as Box<dyn Query>,
                        ));
                    }
                    if let Ok(isbns_field) = self.schema.get_field("isbns") {
                        term_queries.push((
                            Occur::Should,
                            Box::new(TermQuery::new(Term::from_field_text(isbns_field, &corrected_isbn), IndexRecordOption::Basic)) as Box<dyn Query>,
                        ));
                        boost_original_match = Some(3.0);
                    }
                }
            }
            if let Some(boost) = boost_original_match {
                term_queries.push((
                    Occur::Should,
                    Box::new(BoostQuery::new(
                        Box::new(TermQuery::new(Term::from_field_text(doi_field, &lowercased_doi), IndexRecordOption::Basic)) as Box<dyn Query>,
                        boost,
                    )) as Box<dyn Query>,
                ))
            } else {
                term_queries.push((
                    Occur::Should,
                    Box::new(TermQuery::new(Term::from_field_text(doi_field, &lowercased_doi), IndexRecordOption::Basic)) as Box<dyn Query>,
                ));
            }
            Ok(Box::new(BooleanQuery::new(term_queries)) as Box<dyn Query>)
        } else {
            Ok(Box::new(EmptyQuery) as Box<dyn Query>)
        }
    }

    fn parse_statement(&self, pair: Pair<Rule>) -> Result<Box<dyn Query>, QueryParserError> {
        let mut statement_pairs = pair.into_inner();
        let isbn_doi_or_search_group_or_term = statement_pairs.next().expect("grammar failure");
        let statement_boost = statement_pairs.next().map(|boost| f32::from_str(boost.as_str()).expect("grammar failure"));
        let statement_result = match isbn_doi_or_search_group_or_term.as_rule() {
            Rule::search_group => {
                let mut search_group = isbn_doi_or_search_group_or_term.into_inner();
                let field_name = search_group.next().expect("grammar failure");
                let grouping_or_term = search_group.next().expect("grammar failure");
                match grouping_or_term.as_rule() {
                    Rule::grouping => {
                        let mut intermediate_results = vec![];
                        match self.schema.get_field(self.resolve_field_name(field_name.as_str())) {
                            Ok(field) => {
                                for term in grouping_or_term.into_inner() {
                                    intermediate_results.push(self.parse_term(term, &field, "", statement_boost)?);
                                }
                            }
                            Err(tantivy::TantivyError::FieldNotFound(_)) => match self.missing_field_policy {
                                MissingFieldPolicy::AsUsualTerms => {
                                    intermediate_results.push(self.default_fields_term(field_name, statement_boost)?);
                                    for term in grouping_or_term.into_inner() {
                                        intermediate_results.push(self.default_fields_term(term, statement_boost)?)
                                    }
                                }
                                MissingFieldPolicy::Remove => return Ok(Box::new(EmptyQuery {})),
                                MissingFieldPolicy::Fail => return Err(QueryParserError::FieldDoesNotExist(field_name.as_str().to_string())),
                            },
                            _ => unreachable!(),
                        }
                        Ok(Box::new(BooleanQuery::new(intermediate_results.into_iter().map(|q| (Occur::Should, q)).collect())) as Box<dyn Query>)
                    }
                    Rule::term => match self.schema.find_field(self.resolve_field_name(field_name.as_str())) {
                        Some((field, full_path)) => self.parse_term(grouping_or_term, &field, full_path, statement_boost),
                        None => match self.missing_field_policy {
                            MissingFieldPolicy::AsUsualTerms => Ok(Box::new(BooleanQuery::new(vec![
                                (Occur::Should, self.default_fields_term(field_name, statement_boost)?),
                                (Occur::Should, self.default_fields_term(grouping_or_term, statement_boost)?),
                            ])) as Box<dyn Query>),
                            MissingFieldPolicy::Remove => Ok(Box::new(EmptyQuery {}) as Box<dyn Query>),
                            MissingFieldPolicy::Fail => Err(QueryParserError::FieldDoesNotExist(field_name.as_str().to_string())),
                        },
                    },
                    _ => unreachable!(),
                }
            }
            Rule::doi => self.parse_doi(isbn_doi_or_search_group_or_term.as_str()),
            Rule::isbn => self.parse_isbn(isbn_doi_or_search_group_or_term.as_str()),
            Rule::term => self.default_fields_term(isbn_doi_or_search_group_or_term, statement_boost),
            e => panic!("{e:?}"),
        }?;
        Ok(statement_result)
    }

    fn parse_statements(&self, pairs: Pairs<Rule>) -> Result<Box<dyn Query>, QueryParserError> {
        let mut subqueries = Subqueries::new();

        for pair in pairs.clone() {
            let parsed_queries = self.parse_statement(pair)?;
            subqueries.push((Occur::Should, parsed_queries));
        }

        if let Some(exact_matches_promoter) = &self.exact_matches_promoter {
            if let Some(top_level_phrase) = self.extract_top_level_phrase(pairs) {
                subqueries.extend(
                    self.default_fields
                        .iter()
                        .filter_map(|field| {
                            let field_entry = self.schema.get_field_entry(*field);
                            let field_boost = self.field_boosts.as_ref().and_then(|boosts| boosts.get(field_entry.name()).cloned());
                            if let FieldType::Str(ref str_option) = field_entry.field_type() {
                                let Some(option) = str_option.get_indexing_options() else {
                                    return None
                                };
                                let terms = match self.parse_words(*field, option, &top_level_phrase) {
                                    Ok(terms) => terms,
                                    Err(err) => return Some(Err(err)),
                                };
                                (terms.len() > 1 && option.index_option().has_positions()).then(|| {
                                    let query = Box::new(PhraseQuery::new_with_offset_and_slop(terms, exact_matches_promoter.slop)) as Box<dyn Query>;
                                    Ok(boost_query(query, multiply_boosts(exact_matches_promoter.boost, field_boost)))
                                })
                            } else {
                                None
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .map(|q| (Occur::Should, q)),
                )
            }
        }
        Ok(Box::new(BooleanQuery::new(subqueries.into_iter().take(self.limit).collect())) as Box<dyn Query>)
    }

    pub fn parse_query(&self, query: &str) -> Result<Box<dyn Query>, QueryParserError> {
        let pairs = SummaQlParser::parse(Rule::main, query).map_err(Box::new)?;
        Ok(reduce_empty_queries(reduce_should_clause(self.parse_statements(pairs)?)))
    }
}

#[cfg(test)]
mod tests {
    use tantivy::schema::{TextOptions, INDEXED, STRING, TEXT};
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter};

    use super::*;
    use crate::components::SummaTokenizer;

    fn create_query_parser() -> QueryParser {
        let tokenizer_manager = TokenizerManager::default();
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("title", TEXT);
        schema_builder.add_text_field("body", TEXT);
        schema_builder.add_i64_field("timestamp", INDEXED);
        schema_builder.add_text_field("doi", STRING);
        schema_builder.add_text_field("isbns", STRING);
        let schema = schema_builder.build();
        let default_fields = vec!["title".to_string()];
        QueryParser::new(schema, default_fields, &tokenizer_manager).expect("cannot create parser")
    }

    fn create_complex_query_parser() -> QueryParser {
        let tokenizer_manager = TokenizerManager::default();
        tokenizer_manager.register(
            "summa",
            TextAnalyzer::builder(SummaTokenizer)
                .filter(RemoveLongFilter::limit(100))
                .filter(LowerCaser)
                .build(),
        );
        let mut schema_builder = Schema::builder();
        let text_options = TextOptions::default().set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("summa")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        );
        schema_builder.add_text_field("title", text_options);
        schema_builder.add_text_field("body", TEXT);
        schema_builder.add_text_field("authors", TEXT);
        schema_builder.add_text_field("language", TEXT);
        schema_builder.add_i64_field("timestamp", INDEXED);
        schema_builder.add_text_field("doi", STRING);
        schema_builder.add_text_field("isbns", STRING);
        let schema = schema_builder.build();
        let default_fields = vec!["title".to_string(), "body".to_string()];
        QueryParser::new(schema, default_fields, &tokenizer_manager).expect("cannot create parser")
    }

    #[test]
    pub fn test_parser_base() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("search engine");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=0, type=Str, \"engine\")))] })");
        let query = query_parser.parse_query("'search engine'");
        assert_eq!(
            format!("{:?}", query),
            "Ok(PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"search\")), (1, Term(field=0, type=Str, \"engine\"))], slop: 0 })"
        );
    }

    #[test]
    pub fn test_parser_slop() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("body:'search engine'~4");
        assert_eq!(
            format!("{:?}", query),
            "Ok(PhraseQuery { field: Field(1), phrase_terms: [(0, Term(field=1, type=Str, \"search\")), (1, Term(field=1, type=Str, \"engine\"))], slop: 4 })"
        );
    }

    #[test]
    pub fn test_parser_fields() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:'search engine'")),
            "Ok(PhraseQuery { field: Field(1), phrase_terms: [(0, Term(field=1, type=Str, \"search\")), (1, Term(field=1, type=Str, \"engine\"))], slop: 0 })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("timestamp:10")),
            "Ok(TermQuery(Term(field=2, type=I64, 10)))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("title:search engine")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=0, type=Str, \"engine\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("not_field:search engine")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"engine\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:10.0000/abcd.0123 ")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.0000/abcd.0123\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:https://doi.org/10.0000/abcd.0123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"https://doi.org/10.0000/abcd.0123\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:doi.org/10.0000/abcd.0123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"doi.org/10.0000/abcd.0123\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("10.0000/978123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.0000/978123\")))"
        );
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/9781234567890")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=4, type=Str, \"9781234567890\"))), (Should, Boost(query=TermQuery(Term(field=3, type=Str, \"10.0000/9781234567890\")), boost=3))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/978-12345-6789-0")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=4, type=Str, \"9781234567890\"))), (Should, Boost(query=TermQuery(Term(field=3, type=Str, \"10.0000/978-12345-6789-0\")), boost=3))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/978-12345-6789-0.ch11")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=3, type=Str, \"10.0000/978-12345-6789-0\"))), (Should, TermQuery(Term(field=4, type=Str, \"9781234567890\"))), (Should, Boost(query=TermQuery(Term(field=3, type=Str, \"10.0000/978-12345-6789-0.ch11\")), boost=3))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/cbo978-12345-6789-0.ch11")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=3, type=Str, \"10.0000/cbo978-12345-6789-0\"))), (Should, TermQuery(Term(field=4, type=Str, \"9781234567890\"))), (Should, Boost(query=TermQuery(Term(field=3, type=Str, \"10.0000/cbo978-12345-6789-0.ch11\")), boost=3))] })");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("978-12345-6789-0")),
            "Ok(TermQuery(Term(field=4, type=Str, \"9781234567890\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("9781234567890")),
            "Ok(TermQuery(Term(field=4, type=Str, \"9781234567890\")))"
        );
        assert_eq!(format!("{:?}", query_parser.parse_query("97812-34-5678-909")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"97812\"))), (Should, TermQuery(Term(field=0, type=Str, \"34\"))), (Should, TermQuery(Term(field=0, type=Str, \"5678\"))), (Should, TermQuery(Term(field=0, type=Str, \"909\")))] })");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("isbns:97812-34-5678-90")),
            "Ok(TermQuery(Term(field=4, type=Str, \"97812-34-5678-90\")))"
        );
        assert_eq!(format!("{:?}", query_parser.parse_query("123 97812-34-5678-909")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"123\"))), (Should, TermQuery(Term(field=0, type=Str, \"97812\"))), (Should, TermQuery(Term(field=0, type=Str, \"34\"))), (Should, TermQuery(Term(field=0, type=Str, \"5678\"))), (Should, TermQuery(Term(field=0, type=Str, \"909\")))] })");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("10.0000/cbo123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.0000/cbo123\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi.org/10.0000/abcd.0123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.0000/abcd.0123\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("10.0000/abcd.0123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.0000/abcd.0123\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("https://doi.org/10.0000/abcd.0123")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.0000/abcd.0123\")))"
        );
    }

    #[test]
    pub fn test_free_queries() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("Search Engines: The Ultimate, Only Guide!")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=0, type=Str, \"engines\"))), (Should, TermQuery(Term(field=0, type=Str, \"the\"))), (Should, TermQuery(Term(field=0, type=Str, \"ultimate\"))), (Should, TermQuery(Term(field=0, type=Str, \"only\"))), (Should, TermQuery(Term(field=0, type=Str, \"guide\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("!! HI !! (SEARCH! ENGINES!")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"hi\"))), (Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=0, type=Str, \"engines\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("`non closed")),
            "Ok(PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"non\")), (1, Term(field=0, type=Str, \"closed\"))], slop: 0 })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("\"non closed")),
            "Ok(PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"non\")), (1, Term(field=0, type=Str, \"closed\"))], slop: 0 })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("non closed`")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"non\"))), (Should, TermQuery(Term(field=0, type=Str, \"closed\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("non closed\"")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"non\"))), (Should, TermQuery(Term(field=0, type=Str, \"closed\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("title:(search ")),
            "Ok(TermQuery(Term(field=0, type=Str, \"title\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("title:(search -")),
            "Ok(TermQuery(Term(field=0, type=Str, \"title\")))"
        );
        assert_eq!(format!("{:?}", query_parser.parse_query("``")), "Ok(EmptyQuery)");
        assert_eq!(format!("{:?}", query_parser.parse_query("```")), "Ok(EmptyQuery)");
        assert_eq!(format!("{:?}", query_parser.parse_query(")(")), "Ok(EmptyQuery)");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("(a)(b)`")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"a\"))), (Should, TermQuery(Term(field=0, type=Str, \"b\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:'10.1182/blood.v53.1.19.bloodjournal53119'")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.1182/blood.v53.1.19.bloodjournal53119\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("doi:10.1182/blood.v53.1.19.bloodjournal53119")),
            "Ok(TermQuery(Term(field=3, type=Str, \"10.1182/blood.v53.1.19.bloodjournal53119\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("10.10 10/10")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"10\"))), (Should, TermQuery(Term(field=0, type=Str, \"10\"))), (Should, TermQuery(Term(field=0, type=Str, \"10\"))), (Should, TermQuery(Term(field=0, type=Str, \"10\")))] })"
        );
        let query_parser = create_complex_query_parser();
        assert_eq!(format!("{:?}", query_parser.parse_query("\"search engines\"")), "Ok(BooleanQuery { subqueries: [(Should, PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"search\")), (1, Term(field=0, type=Str, \"engines\"))], slop: 0 }), (Should, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(field=1, type=Str, \"search\")), (1, Term(field=1, type=Str, \"engines\"))], slop: 0 })] })");
    }

    #[test]
    pub fn test_non_ascii() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:поисковые системы")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=1, type=Str, \"поисковые\"))), (Should, TermQuery(Term(field=0, type=Str, \"системы\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("(поисковые системы)")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"поисковые\"))), (Should, TermQuery(Term(field=0, type=Str, \"системы\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("поисковые: системы")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"поисковые\"))), (Should, TermQuery(Term(field=0, type=Str, \"системы\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("healthcare cyber–physical system")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"healthcare\"))), (Should, TermQuery(Term(field=0, type=Str, \"cyber\"))), (Should, TermQuery(Term(field=0, type=Str, \"physical\"))), (Should, TermQuery(Term(field=0, type=Str, \"system\")))] })"
        );
    }

    #[test]
    pub fn test_regex() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:/поисковые/")),
            "Ok(RegexQuery { regex: Regex(\"поисковые\")\n, field: Field(1) })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body://поиск/овые//")),
            "Ok(RegexQuery { regex: Regex(\"поиск/овые\")\n, field: Field(1) })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:/поисковые.*/")),
            "Ok(RegexQuery { regex: Regex(\"поисковые.*\")\n, field: Field(1) })"
        );
    }

    #[test]
    pub fn test_plus_minus() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:+search -engine")),
            "Ok(BooleanQuery { subqueries: [(Must, TermQuery(Term(field=1, type=Str, \"search\"))), (MustNot, TermQuery(Term(field=0, type=Str, \"engine\")))] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:+'search engine'")),
            "Ok(BooleanQuery { subqueries: [(Must, PhraseQuery { field: Field(1), phrase_terms: [(0, Term(field=1, type=Str, \"search\")), (1, Term(field=1, type=Str, \"engine\"))], slop: 0 })] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("+search +engine")),
            "Ok(BooleanQuery { subqueries: [(Must, TermQuery(Term(field=0, type=Str, \"search\"))), (Must, TermQuery(Term(field=0, type=Str, \"engine\")))] })"
        );
        let query_parser = create_complex_query_parser();
        assert_eq!(format!("{:?}", query_parser.parse_query("+search +engine")), "Ok(BooleanQuery { subqueries: [(Must, BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=1, type=Str, \"search\")))] }), (Must, BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"engine\"))), (Should, TermQuery(Term(field=1, type=Str, \"engine\")))] })] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("+search language:+ru")), "Ok(BooleanQuery { subqueries: [(Must, BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=1, type=Str, \"search\")))] }), (Must, TermQuery(Term(field=3, type=Str, \"ru\")))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("+c++ language:+ru")), "Ok(BooleanQuery { subqueries: [(Must, BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"c++\"))), (Should, TermQuery(Term(field=1, type=Str, \"c\")))] }), (Must, TermQuery(Term(field=3, type=Str, \"ru\")))] })");
    }

    #[test]
    pub fn test_parser_boostings() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("search^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(Boost(query=TermQuery(Term(field=0, type=Str, \"search\")), boost=2))"
        );
        let query = query_parser.parse_query("'search engine'~3^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(Boost(query=PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"search\")), (1, Term(field=0, type=Str, \"engine\"))], slop: 3 }, boost=2))"
        );
        let query = query_parser.parse_query("search engine^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, Boost(query=TermQuery(Term(field=0, type=Str, \"engine\")), boost=2))] })"
        );
        let query = query_parser.parse_query("body:title^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(Boost(query=TermQuery(Term(field=1, type=Str, \"title\")), boost=2))"
        );
        let query = query_parser.parse_query("body:'title'^2.0");
        assert_eq!(
            format!("{:?}", query),
            "Ok(Boost(query=TermQuery(Term(field=1, type=Str, \"title\")), boost=2))"
        );
    }

    #[test]
    pub fn test_range_queries() {
        let query_parser = create_query_parser();
        let query = query_parser.parse_query("body:[aaa TO ccc]");
        assert_eq!(
            format!("{:?}", query),
            "Ok(RangeQuery { field: \"body\", value_type: Str, lower_bound: Included([97, 97, 97]), upper_bound: Included([99, 99, 99]), limit: None })"
        );
        let query = query_parser.parse_query("body:[ a to  * ]");
        assert_eq!(
            format!("{:?}", query),
            "Ok(RangeQuery { field: \"body\", value_type: Str, lower_bound: Included([97]), upper_bound: Unbounded, limit: None })"
        );
        let query = query_parser.parse_query("timestamp:[ 1000 to 2000 ]");
        assert_eq!(format!("{:?}", query), "Ok(RangeQuery { field: \"timestamp\", value_type: I64, lower_bound: Included([128, 0, 0, 0, 0, 0, 3, 232]), upper_bound: Included([128, 0, 0, 0, 0, 0, 7, 208]), limit: None })");
        let query = query_parser.parse_query("timestamp:(-[1100 to 1200] [ 1000 to 2000 ] -1500 +3000)");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(MustNot, RangeQuery { field: \"timestamp\", value_type: I64, lower_bound: Included([128, 0, 0, 0, 0, 0, 4, 76]), upper_bound: Included([128, 0, 0, 0, 0, 0, 4, 176]), limit: None }), (Should, RangeQuery { field: \"timestamp\", value_type: I64, lower_bound: Included([128, 0, 0, 0, 0, 0, 3, 232]), upper_bound: Included([128, 0, 0, 0, 0, 0, 7, 208]), limit: None }), (MustNot, TermQuery(Term(field=2, type=I64, 1500))), (Must, TermQuery(Term(field=2, type=I64, 3000)))] })");
    }

    #[test]
    pub fn test_exact_phrase_promoter() {
        let mut query_parser = create_query_parser();
        query_parser.set_exact_match_promoter(proto::ExactMatchesPromoter { slop: 3, boost: Some(2.0) });
        let query = query_parser.parse_query("old school holy-wood");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"old\"))), (Should, TermQuery(Term(field=0, type=Str, \"school\"))), (Should, TermQuery(Term(field=0, type=Str, \"holy\"))), (Should, TermQuery(Term(field=0, type=Str, \"wood\"))), (Should, Boost(query=PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"old\")), (1, Term(field=0, type=Str, \"school\")), (2, Term(field=0, type=Str, \"holy\")), (3, Term(field=0, type=Str, \"wood\"))], slop: 3 }, boost=2))] })");
        let query = query_parser.parse_query("old^2.0 school");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(field=0, type=Str, \"old\")), boost=2)), (Should, TermQuery(Term(field=0, type=Str, \"school\")))] })");
        query_parser.set_field_boosts(HashMap::from_iter(vec![("title".to_string(), 3.0)].into_iter()));
        let query = query_parser.parse_query("old school");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(field=0, type=Str, \"old\")), boost=3)), (Should, Boost(query=TermQuery(Term(field=0, type=Str, \"school\")), boost=3)), (Should, Boost(query=PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"old\")), (1, Term(field=0, type=Str, \"school\"))], slop: 3 }, boost=6))] })");
    }
}
