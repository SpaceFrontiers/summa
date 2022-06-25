use std::collections::Bound;
use std::ops::Bound::{Included, Unbounded};
use std::ops::Deref;
use std::str::FromStr;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
use summa_proto::proto;
use tantivy::query::{BooleanQuery, BoostQuery, DisjunctionMaxQuery, EmptyQuery, PhraseQuery, Query, QueryClone, RangeQuery, RegexQuery, TermQuery};
use tantivy::schema::{Facet, FacetParseError, Field, FieldEntry, FieldType, IndexRecordOption, Schema, TextFieldIndexing, Type};
use tantivy::tokenizer::{TextAnalyzer, TokenizerManager};
use tantivy::{Index, Term};
use tantivy_query_grammar::Occur;

use crate::components::query_parser::morphology::MorphologyManager;
use crate::components::query_parser::proto_query_parser::QueryParserDefaultMode;
use crate::components::query_parser::term_field_mappers::TermFieldMappersManager;
use crate::components::query_parser::utils::cast_field_to_term;
use crate::configs::core::QueryParserConfig;
use crate::errors::SummaResult;
use crate::utils::transpose;
use crate::validators;

#[derive(Parser)]
#[grammar = "src/components/query_parser/summa_ql.pest"] // relative to src
struct SummaQlParser;

pub struct QueryParser {
    schema: Schema,
    tokenizer_manager: TokenizerManager,
    morphology_manager: MorphologyManager,
    term_field_mappers_manager: TermFieldMappersManager,
    query_parser_config: QueryParserConfig,
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
                Occur::Must | Occur::MustNot => subqueries.push((*occur, nested_query)),
                Occur::Should => {
                    if let Some(nested_boolean_query) = nested_query.deref().as_any().downcast_ref::<BooleanQuery>() {
                        subqueries.extend(nested_boolean_query.clauses().iter().map(|(o, q)| (*o, reduce_should_clause(q.box_clone()))))
                    } else {
                        subqueries.push((*occur, nested_query))
                    }
                }
            }
        }
        if subqueries.len() == 1 && subqueries[0].0 == Occur::Should {
            return subqueries.into_iter().next().expect("impossible").1;
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
    pub fn new(
        schema: Schema,
        query_parser_config: QueryParserConfig,
        morphology_manager: &MorphologyManager,
        tokenizer_manager: &TokenizerManager,
    ) -> SummaResult<QueryParser> {
        validators::parse_fields(&schema, &query_parser_config.0.default_fields)?;
        Ok(QueryParser {
            term_field_mappers_manager: TermFieldMappersManager::new(&schema, tokenizer_manager),
            morphology_manager: morphology_manager.clone(),
            tokenizer_manager: tokenizer_manager.clone(),
            query_parser_config,
            schema,
        })
    }

    pub fn for_index(index: &Index, query_parser_config: QueryParserConfig, morphology_manager: &MorphologyManager) -> SummaResult<QueryParser> {
        QueryParser::new(index.schema(), query_parser_config, morphology_manager, index.tokenizers())
    }

    pub fn resolve_field_name<'a>(&'a self, field_name: &'a str) -> &str {
        self.query_parser_config
            .0
            .field_aliases
            .get(field_name)
            .map(|s| s.as_str())
            .unwrap_or(field_name)
    }

    fn get_text_analyzer(&self, field_entry: &FieldEntry, option: &TextFieldIndexing) -> Result<TextAnalyzer, QueryParserError> {
        self.tokenizer_manager
            .get(option.tokenizer())
            .ok_or_else(|| QueryParserError::UnknownTokenizer {
                field: field_entry.name().to_string(),
                tokenizer: option.tokenizer().to_string(),
            })
    }

    fn default_field_queries(&self, term: Pair<Rule>, boost: Option<f32>) -> Result<Box<dyn Query>, QueryParserError> {
        let (occur, term) = match term.as_rule() {
            Rule::field_name => (Occur::Should, term),
            _ => {
                let term = term.into_inner().next().expect("grammar failure");
                let occur = self.parse_occur(&term);
                let pre_term = term.into_inner().next().expect("grammar failure");
                (occur, pre_term.clone())
            }
        };

        let default_field_queries = self
            .query_parser_config
            .0
            .default_fields
            .iter()
            .map(|field| {
                let (field, full_path) = self.schema.find_field(field).expect("inconsistent state");
                self.parse_pre_term(&field, full_path, term.clone(), boost, true)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(match occur {
            Occur::Should => {
                let default_field_queries = default_field_queries.into_iter().flatten();
                match QueryParserDefaultMode::from(self.query_parser_config.0.default_mode.clone()) {
                    QueryParserDefaultMode::Boolean => Box::new(BooleanQuery::new(default_field_queries.map(|q| (occur, q)).collect())) as Box<dyn Query>,
                    QueryParserDefaultMode::DisjuctionMax { tie_breaker } => {
                        Box::new(DisjunctionMaxQuery::with_tie_breaker(default_field_queries.collect(), tie_breaker)) as Box<dyn Query>
                    }
                }
            }
            Occur::MustNot => Box::new(BooleanQuery::new(default_field_queries.into_iter().flatten().map(|q| (occur, q)).collect())) as Box<dyn Query>,
            Occur::Must => {
                if self.query_parser_config.0.default_fields.len() == 1 {
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

    pub fn parse_words(&self, field: Field, full_path: &str, option: &TextFieldIndexing, words: &str) -> Result<Vec<(usize, Term)>, QueryParserError> {
        let field_entry = self.schema.get_field_entry(field);
        let mut text_analyzer = self.get_text_analyzer(field_entry, option)?;
        let mut token_stream = text_analyzer.token_stream(words);
        let mut terms = Vec::new();
        token_stream.process(&mut |token| {
            let term = cast_field_to_term(&field, full_path, field_entry.field_type(), &token.text, true);
            terms.push((token.position, term));
        });
        Ok(terms)
    }

    fn parse_pre_term(
        &self,
        field: &Field,
        full_path: &str,
        pre_term: Pair<Rule>,
        boost: Option<f32>,
        ignore_phrase_for_non_position_field: bool,
    ) -> Result<Vec<Box<dyn Query>>, QueryParserError> {
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

        if !(field_type.is_indexed() || matches!(pre_term.as_rule(), Rule::range) && field_type.is_fast()) {
            return Err(QueryParserError::FieldNotIndexed(field_entry.name().to_string()));
        }

        let boost = multiply_boosts(self.query_parser_config.0.field_boosts.get(field_entry.name()).copied(), boost);

        if matches!(pre_term.as_rule(), Rule::range) {
            return Ok(vec![boost_query(Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>, boost)]);
        }

        return match *field_type {
            FieldType::Bytes(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    let val = &BASE64.decode(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_bytes(*field, val), IndexRecordOption::Basic)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::U64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    let val: u64 = u64::from_str(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_u64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::I64(_) => match pre_term.as_rule() {
                Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                Rule::phrase | Rule::word => {
                    let val: i64 = i64::from_str(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_field_i64(*field, val), IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                    Ok(vec![boost_query(query, boost)])
                }
                _ => unreachable!(),
            },
            FieldType::Facet(_) => match pre_term.as_rule() {
                Rule::phrase | Rule::word => {
                    let val = Facet::from_text(pre_term.as_str())?;
                    let query = Box::new(TermQuery::new(Term::from_facet(*field, &val), IndexRecordOption::Basic)) as Box<dyn Query>;
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
                        let mut text_analyzer = self.get_text_analyzer(field_entry, indexing)?;
                        let mut token_stream = text_analyzer.token_stream(pre_term.as_str());
                        let mut queries = Vec::new();
                        token_stream.process(&mut |token| {
                            let morphology_config = self
                                .query_parser_config
                                .0
                                .morphology_configs
                                .get(field_entry.name())
                                .cloned()
                                .unwrap_or_default();
                            let query = if let Some(morphology) = self.morphology_manager.get(self.query_parser_config.0.query_language()) {
                                // ToDo: Change heuristic
                                if pre_term.as_str().len() < 24 {
                                    morphology.derive_query(morphology_config, field, full_path, field_type, &token.text)
                                } else {
                                    let term = cast_field_to_term(field, full_path, field_type, &token.text, false);
                                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>
                                }
                            } else {
                                let term = cast_field_to_term(field, full_path, field_type, &token.text, false);
                                Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>
                            };
                            queries.push(boost_query(query, boost))
                        });
                        Ok(queries)
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
                        let terms = self.parse_words(*field, full_path, indexing, words.as_str())?;
                        if terms.len() <= 1 {
                            return Ok(terms
                                .into_iter()
                                .map(|(_, term)| {
                                    let query = Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>;
                                    boost_query(query, boost)
                                })
                                .collect());
                        }
                        return if indexing.index_option().has_positions() {
                            let query = Box::new(PhraseQuery::new_with_offset_and_slop(terms, slop)) as Box<dyn Query>;
                            Ok(vec![boost_query(query, boost)])
                        } else if ignore_phrase_for_non_position_field {
                            Ok(vec![])
                        } else {
                            Err(QueryParserError::FieldDoesNotHavePositionsIndexed(field_entry.name().to_string()))
                        };
                    }
                    Rule::range => Ok(vec![Box::new(self.parse_range(pre_term, field)?) as Box<dyn Query>]),
                    Rule::regex => {
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
            Rule::positive_term | Rule::positive_grouping => Occur::Must,
            Rule::negative_term | Rule::negative_grouping => Occur::MustNot,
            Rule::default_term | Rule::default_grouping => Occur::Should,
            _ => unreachable!(),
        }
    }

    fn parse_term(&self, term: Pair<Rule>, field: &Field, full_path: &str, boost: Option<f32>) -> Result<Box<dyn Query>, QueryParserError> {
        let term = term.into_inner().next().expect("grammar failure");
        let occur = self.parse_occur(&term);
        let pre_term = term.into_inner().next().expect("grammar failure");
        Ok(Box::new(BooleanQuery::new(
            self.parse_pre_term(field, full_path, pre_term, boost, false)?
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
            FieldType::Bytes(_) => {
                let val = &BASE64.decode(phrase)?;
                Ok(Term::from_field_bytes(field, val))
            }
            FieldType::Str(ref str_options) => {
                let option = str_options.get_indexing_options().ok_or_else(|| {
                    // This should have been seen earlier really.
                    QueryParserError::FieldNotIndexed(field_entry.name().to_string())
                })?;
                let mut terms: Vec<Term> = Vec::new();
                let mut text_analyzer = self
                    .tokenizer_manager
                    .get(option.tokenizer())
                    .ok_or_else(|| QueryParserError::UnknownTokenizer {
                        field: field_entry.name().to_string(),
                        tokenizer: option.tokenizer().to_string(),
                    })?;
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
                        let grouping = grouping_or_term.into_inner().next().expect("grammar failure");
                        let occur = self.parse_occur(&grouping);
                        let mut intermediate_results = vec![];
                        let resolved_field_name = self.resolve_field_name(field_name.as_str());
                        match self.schema.find_field(resolved_field_name) {
                            Some((field, full_path)) => {
                                for term in grouping.into_inner() {
                                    intermediate_results.push(self.parse_term(term, &field, full_path, statement_boost)?);
                                }
                            }
                            None => {
                                if self.query_parser_config.0.removed_fields.iter().any(|x| x == field_name.as_str()) {
                                    return Ok(Box::new(EmptyQuery {}));
                                }
                                intermediate_results.push(self.default_field_queries(field_name, statement_boost)?);
                                for term in grouping.into_inner() {
                                    intermediate_results.push(self.default_field_queries(term, statement_boost)?)
                                }
                            }
                        }
                        let group_query = Box::new(BooleanQuery::new(intermediate_results.into_iter().map(|q| (Occur::Should, q)).collect())) as Box<dyn Query>;
                        match occur {
                            Occur::Should => Ok(group_query),
                            Occur::Must => Ok(Box::new(BooleanQuery::new(vec![(Occur::Must, group_query)])) as Box<dyn Query>),
                            Occur::MustNot => Ok(Box::new(BooleanQuery::new(vec![(Occur::MustNot, group_query)])) as Box<dyn Query>),
                        }
                    }
                    Rule::term => {
                        let resolved_field_name = self.resolve_field_name(field_name.as_str());
                        match self.schema.find_field(resolved_field_name) {
                            Some((field, full_path)) => self.parse_term(grouping_or_term, &field, full_path, statement_boost),
                            None => {
                                if self.query_parser_config.0.removed_fields.iter().any(|x| x == field_name.as_str()) {
                                    Ok(Box::new(EmptyQuery {}) as Box<dyn Query>)
                                } else {
                                    Ok(Box::new(BooleanQuery::new(vec![
                                        (Occur::Should, self.default_field_queries(field_name, statement_boost)?),
                                        (Occur::Should, self.default_field_queries(grouping_or_term, statement_boost)?),
                                    ])) as Box<dyn Query>)
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            Rule::doi => {
                let mut queries = vec![];

                for term_field_mapper_name in ["doi", "doi_isbn"] {
                    if let Some(term_field_mapper_config) = self.query_parser_config.0.term_field_mapper_configs.get(term_field_mapper_name) {
                        if let Some(term_field_mapper) = self.term_field_mappers_manager.get(term_field_mapper_name) {
                            if let Some(query) = term_field_mapper.map(isbn_doi_or_search_group_or_term.as_str(), &term_field_mapper_config.fields) {
                                queries.push((Occur::Should, query));
                            }
                        }
                    }
                }

                Ok(Box::new(BooleanQuery::new(queries)) as Box<dyn Query>)
            }
            Rule::isbn => {
                let mut queries = vec![];

                for term_field_mapper_name in ["isbn"] {
                    if let Some(term_field_mapper_config) = self.query_parser_config.0.term_field_mapper_configs.get(term_field_mapper_name) {
                        if let Some(term_field_mapper) = self.term_field_mappers_manager.get(term_field_mapper_name) {
                            if let Some(query) = term_field_mapper.map(isbn_doi_or_search_group_or_term.as_str(), &term_field_mapper_config.fields) {
                                queries.push((Occur::Should, query));
                            }
                        }
                    }
                }

                Ok(Box::new(BooleanQuery::new(queries)) as Box<dyn Query>)
            }
            Rule::term => self.default_field_queries(isbn_doi_or_search_group_or_term, statement_boost),
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

        if let Some(top_level_phrase) = self.extract_top_level_phrase(pairs) {
            if let Some(exact_matches_promoter) = &self.query_parser_config.0.exact_matches_promoter {
                let fields = if exact_matches_promoter.fields.is_empty() {
                    self.query_parser_config.0.default_fields.iter()
                } else {
                    exact_matches_promoter.fields.iter()
                };
                subqueries.extend(
                    fields
                        .filter_map(|field| {
                            let (field, full_path) = self.schema.find_field(self.resolve_field_name(field)).expect("no field");
                            let field_entry = self.schema.get_field_entry(field);
                            let field_boost = self.query_parser_config.0.field_boosts.get(field_entry.name()).copied();
                            match field_entry.field_type() {
                                FieldType::Str(ref str_option) => {
                                    let Some(option) = str_option.get_indexing_options() else {
                                        return None
                                    };
                                    let terms = match self.parse_words(field, full_path, option, &top_level_phrase) {
                                        Ok(terms) => terms,
                                        Err(err) => return Some(Err(err)),
                                    };
                                    (terms.len() > 1 && option.index_option().has_positions()).then(|| {
                                        let query = Box::new(PhraseQuery::new_with_offset_and_slop(terms, exact_matches_promoter.slop)) as Box<dyn Query>;
                                        Ok(boost_query(query, multiply_boosts(exact_matches_promoter.boost, field_boost)))
                                    })
                                }
                                FieldType::JsonObject(ref json_option) => {
                                    let Some(option) = json_option.get_text_indexing_options() else {
                                        return None
                                    };
                                    let terms = match self.parse_words(field, full_path, option, &top_level_phrase) {
                                        Ok(terms) => terms,
                                        Err(err) => return Some(Err(err)),
                                    };
                                    (terms.len() > 1 && option.index_option().has_positions()).then(|| {
                                        let query = Box::new(PhraseQuery::new_with_offset_and_slop(terms, exact_matches_promoter.slop)) as Box<dyn Query>;
                                        Ok(boost_query(query, multiply_boosts(exact_matches_promoter.boost, field_boost)))
                                    })
                                }
                                _ => None,
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .map(|q| (Occur::Should, q)),
                )
            }
        }
        Ok(Box::new(BooleanQuery::new(subqueries.into_iter().take(self.query_parser_config.term_limit()).collect())) as Box<dyn Query>)
    }

    pub fn parse_query(&self, query: &str) -> Result<Box<dyn Query>, QueryParserError> {
        let pairs = SummaQlParser::parse(Rule::main, query).map_err(Box::new)?;
        Ok(reduce_empty_queries(reduce_should_clause(self.parse_statements(pairs)?)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tantivy::schema::{TextOptions, INDEXED, STRING, TEXT};
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter};

    use super::*;
    use crate::components::SummaTokenizer;

    fn create_query_parser() -> QueryParser {
        let tokenizer_manager = TokenizerManager::default();
        let morphology_manager = MorphologyManager::default();
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("title", TEXT);
        schema_builder.add_text_field("body", TEXT);
        schema_builder.add_i64_field("timestamp", INDEXED);
        schema_builder.add_text_field("doi", STRING);
        schema_builder.add_text_field("isbns", STRING);
        schema_builder.add_json_field("metadata", TEXT);
        let schema = schema_builder.build();
        let query_parser_config = QueryParserConfig(proto::QueryParserConfig {
            default_fields: vec!["title".to_string()],
            ..Default::default()
        });
        QueryParser::new(schema, query_parser_config, &morphology_manager, &tokenizer_manager).expect("cannot create parser")
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
        let morphology_manager = MorphologyManager::default();
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
        schema_builder.add_json_field("metadata", TEXT);
        let schema = schema_builder.build();
        let query_parser_config = QueryParserConfig(proto::QueryParserConfig {
            default_fields: vec!["title".to_string(), "body".to_string()],
            ..Default::default()
        });
        QueryParser::new(schema, query_parser_config, &morphology_manager, &tokenizer_manager).expect("cannot create parser")
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
        let mut query_parser = create_query_parser();
        query_parser.query_parser_config.0.term_field_mapper_configs.insert(
            "doi".to_string(),
            proto::TermFieldMapperConfig {
                fields: vec!["doi".to_string()],
            },
        );
        query_parser.query_parser_config.0.term_field_mapper_configs.insert(
            "doi_isbn".to_string(),
            proto::TermFieldMapperConfig {
                fields: vec!["metadata.isbns".to_string()],
            },
        );
        query_parser.query_parser_config.0.term_field_mapper_configs.insert(
            "isbn".to_string(),
            proto::TermFieldMapperConfig {
                fields: vec!["metadata.isbns".to_string()],
            },
        );
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
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"not\"))), (Should, TermQuery(Term(field=0, type=Str, \"field\"))), (Should, TermQuery(Term(field=0, type=Str, \"search\"))), (Should, TermQuery(Term(field=0, type=Str, \"engine\")))] })"
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
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/9781234567890")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=3, type=Str, \"10.0000/9781234567890\"))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=Str, \"9781234567890\")))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/978-12345-6789-0")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=3, type=Str, \"10.0000/978-12345-6789-0\"))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=Str, \"9781234567890\")))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/978-12345-6789-0.ch11")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=3, type=Str, \"10.0000/978-12345-6789-0.ch11\"))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=Str, \"9781234567890\")))] })");
        assert_eq!(format!("{:?}", query_parser.parse_query("10.0000/cbo978-12345-6789-0.ch11")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=3, type=Str, \"10.0000/cbo978-12345-6789-0.ch11\"))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=Str, \"9781234567890\")))] })");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("978-12345-6789-0")),
            "Ok(TermQuery(Term(field=5, type=Json, path=isbns, type=Str, \"9781234567890\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("9781234567890")),
            "Ok(TermQuery(Term(field=5, type=Json, path=isbns, type=Str, \"9781234567890\")))"
        );
        assert_eq!(format!("{:?}", query_parser.parse_query("97812-34-5678-909")), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"97812\"))), (Should, TermQuery(Term(field=0, type=Str, \"34\"))), (Should, TermQuery(Term(field=0, type=Str, \"5678\"))), (Should, TermQuery(Term(field=0, type=Str, \"909\")))] })");
        assert_eq!(
            format!("{:?}", query_parser.parse_query("metadata.isbns:97812-34-5678-90")),
            "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=5, type=Json, path=isbns, type=I64, 97812))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=I64, 34))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=I64, 5678))), (Should, TermQuery(Term(field=5, type=Json, path=isbns, type=I64, 90)))] })"
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
    pub fn test_json() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("metadata.a:1")),
            "Ok(TermQuery(Term(field=5, type=Json, path=a, type=I64, 1)))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("metadata.a:\"1\"")),
            "Ok(TermQuery(Term(field=5, type=Json, path=a, type=Str, \"1\")))"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("metadata.a:\"1 2 3\"")),
            "Ok(PhraseQuery { field: Field(5), phrase_terms: [(0, Term(field=5, type=Json, path=a, type=Str, \"1\")), (1, Term(field=5, type=Json, path=a, type=Str, \"2\")), (2, Term(field=5, type=Json, path=a, type=Str, \"3\"))], slop: 0 })"
        );
    }

    #[test]
    pub fn test_grouping() {
        let query_parser = create_query_parser();
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:+(a b)")),
            "Ok(BooleanQuery { subqueries: [(Must, BooleanQuery { subqueries: [(Should, TermQuery(Term(field=1, type=Str, \"a\"))), (Should, TermQuery(Term(field=1, type=Str, \"b\")))] })] })"
        );
        assert_eq!(
            format!("{:?}", query_parser.parse_query("body:-(a b)")),
            "Ok(BooleanQuery { subqueries: [(MustNot, BooleanQuery { subqueries: [(Should, TermQuery(Term(field=1, type=Str, \"a\"))), (Should, TermQuery(Term(field=1, type=Str, \"b\")))] })] })"
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
        query_parser.query_parser_config.0.exact_matches_promoter = Some(proto::ExactMatchesPromoter {
            slop: 3,
            boost: Some(2.0),
            fields: vec![],
        });
        let query = query_parser.parse_query("old school holy-wood");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"old\"))), (Should, TermQuery(Term(field=0, type=Str, \"school\"))), (Should, TermQuery(Term(field=0, type=Str, \"holy\"))), (Should, TermQuery(Term(field=0, type=Str, \"wood\"))), (Should, Boost(query=PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"old\")), (1, Term(field=0, type=Str, \"school\")), (2, Term(field=0, type=Str, \"holy\")), (3, Term(field=0, type=Str, \"wood\"))], slop: 3 }, boost=2))] })");
        let query = query_parser.parse_query("old^2.0 school");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(field=0, type=Str, \"old\")), boost=2)), (Should, TermQuery(Term(field=0, type=Str, \"school\")))] })");
        query_parser.query_parser_config.0.field_boosts = HashMap::from_iter(vec![("title".to_string(), 3.0)]);
        let query = query_parser.parse_query("old school");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, Boost(query=TermQuery(Term(field=0, type=Str, \"old\")), boost=3)), (Should, Boost(query=TermQuery(Term(field=0, type=Str, \"school\")), boost=3)), (Should, Boost(query=PhraseQuery { field: Field(0), phrase_terms: [(0, Term(field=0, type=Str, \"old\")), (1, Term(field=0, type=Str, \"school\"))], slop: 3 }, boost=6))] })");
    }

    #[test]
    pub fn test_inflection() {
        let mut query_parser = create_query_parser();
        let mut morphology_configs = HashMap::new();
        morphology_configs.insert(
            "title".to_string(),
            proto::MorphologyConfig {
                derive_tenses_coefficient: Some(0.3),
            },
        );
        query_parser.query_parser_config.0.morphology_configs = morphology_configs;
        query_parser.query_parser_config.0.query_language = Some("en".to_string());
        let query = query_parser.parse_query("red1 search engine going");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, TermQuery(Term(field=0, type=Str, \"red1\"))), (Should, DisjunctionMaxQuery { disjuncts: [TermQuery(Term(field=0, type=Str, \"search\")), TermQuery(Term(field=0, type=Str, \"searches\"))], tie_breaker: 0.3 }), (Should, DisjunctionMaxQuery { disjuncts: [TermQuery(Term(field=0, type=Str, \"engine\")), TermQuery(Term(field=0, type=Str, \"engines\"))], tie_breaker: 0.3 }), (Should, TermQuery(Term(field=0, type=Str, \"going\")))] })");
        let query = query_parser.parse_query("iso 34-1:2022");
        assert_eq!(format!("{:?}", query), "Ok(BooleanQuery { subqueries: [(Should, DisjunctionMaxQuery { disjuncts: [TermQuery(Term(field=0, type=Str, \"iso\")), TermQuery(Term(field=0, type=Str, \"isos\"))], tie_breaker: 0.3 }), (Should, TermQuery(Term(field=0, type=Str, \"34\"))), (Should, TermQuery(Term(field=0, type=Str, \"1\")))] })");
    }
}
