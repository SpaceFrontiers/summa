use std::collections::HashMap;
use std::ops::Bound;
use std::ops::Bound::Unbounded;
use std::str::FromStr;

use base64::Engine;
#[cfg(feature = "metrics")]
use opentelemetry::metrics::Counter;
#[cfg(feature = "metrics")]
use opentelemetry::Context;
#[cfg(feature = "metrics")]
use opentelemetry::{global, KeyValue};
use summa_proto::proto;
use tantivy::json_utils::{convert_to_fast_value_and_get_term, JsonTermWriter};
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, DisjunctionMaxQuery, EmptyQuery, MoreLikeThisQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery,
};
use tantivy::schema::{Field, FieldEntry, FieldType, IndexRecordOption, Schema};
use tantivy::{DateTime, Index, Score, Term};
use tracing::{info, warn};

use crate::components::queries::ExistsQuery;
use crate::components::query_parser::{QueryParser, QueryParserError};
use crate::errors::{Error, SummaResult, ValidationError};
#[cfg(feature = "metrics")]
use crate::metrics::ToLabel;

/// Responsible for casting `crate::proto::Query` message to `tantivy::query::Query`
pub struct ProtoQueryParser {
    index: Index,
    index_name: String,
    cached_schema: Schema,
    // Counters
    #[cfg(feature = "metrics")]
    query_counter: Counter<u64>,
    #[cfg(feature = "metrics")]
    subquery_counter: Counter<u64>,
    index_default_fields: Vec<String>,
    field_aliases: HashMap<String, String>,
}

pub enum MatchQueryDefaultMode {
    Boolean,
    DisjuctionMax { tie_breaker: Score },
}

impl From<Option<proto::match_query::DefaultMode>> for MatchQueryDefaultMode {
    fn from(value: Option<proto::match_query::DefaultMode>) -> Self {
        match value {
            Some(proto::match_query::DefaultMode::BooleanShouldMode(_)) | None => MatchQueryDefaultMode::Boolean,
            Some(proto::match_query::DefaultMode::DisjuctionMaxMode(proto::MatchQueryDisjuctionMaxMode { tie_breaker })) => {
                MatchQueryDefaultMode::DisjuctionMax { tie_breaker }
            }
        }
    }
}

fn cast_value_to_term(field: Field, full_path: &str, field_type: &FieldType, value: &str) -> SummaResult<Term> {
    Ok(match field_type {
        FieldType::Str(_) => Term::from_field_text(field, value),
        FieldType::JsonObject(json_options) => {
            let mut term = Term::with_capacity(128);
            let mut json_term_writer = JsonTermWriter::from_field_and_json_path(field, full_path, json_options.is_expand_dots_enabled(), &mut term);
            convert_to_fast_value_and_get_term(&mut json_term_writer, value).unwrap_or_else(|| {
                json_term_writer.set_str(value.trim_matches(|c| c == '\'' || c == '\"'));
                json_term_writer.term().clone()
            })
        }
        FieldType::I64(_) => Term::from_field_i64(
            field,
            i64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as i64")))?,
        ),
        FieldType::U64(_) => Term::from_field_u64(
            field,
            u64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as u64")))?,
        ),
        FieldType::F64(_) => Term::from_field_f64(
            field,
            f64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as f64")))?,
        ),
        FieldType::Bytes(_) => Term::from_field_bytes(
            field,
            &base64::engine::general_purpose::STANDARD
                .decode(value)
                .map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as bytes")))?,
        ),
        FieldType::Date(_) => Term::from_field_date(
            field,
            DateTime::from_timestamp_secs(i64::from_str(value).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {value} as date")))?),
        ),
        _ => return Err(Error::InvalidSyntax("invalid range type".to_owned())),
    })
}

fn cast_value_to_bound_term(field: Field, full_path: &str, field_type: &FieldType, value: &str, including: bool) -> SummaResult<Bound<Term>> {
    Ok(match value {
        "*" => Unbounded,
        value => {
            let casted_value = cast_value_to_term(field, full_path, field_type, value)?;
            if including {
                Bound::Included(casted_value)
            } else {
                Bound::Excluded(casted_value)
            }
        }
    })
}

impl ProtoQueryParser {
    pub fn for_index(
        index_name: &str,
        index: &Index,
        index_default_fields: Vec<String>,
        field_aliases: HashMap<String, String>,
    ) -> SummaResult<ProtoQueryParser> {
        #[cfg(feature = "metrics")]
        let query_counter = global::meter("summa").u64_counter("query_counter").with_description("Queries counter").init();
        #[cfg(feature = "metrics")]
        let subquery_counter = global::meter("summa")
            .u64_counter("subquery_counter")
            .with_description("Sub-queries counter")
            .init();

        Ok(ProtoQueryParser {
            index: index.clone(),
            index_name: index_name.to_string(),
            cached_schema: index.schema(),
            #[cfg(feature = "metrics")]
            query_counter,
            #[cfg(feature = "metrics")]
            subquery_counter,
            index_default_fields,
            field_aliases,
        })
    }

    pub fn resolve_field_name<'a>(&'a self, field_name: &'a str) -> &str {
        self.field_aliases.get(field_name).map(|s| s.as_str()).unwrap_or(field_name)
    }

    #[inline]
    pub(crate) fn field_and_field_entry<'a>(&'a self, field_name: &'a str) -> SummaResult<(Field, &str, &FieldEntry)> {
        match self.cached_schema.find_field(self.resolve_field_name(field_name)) {
            Some((field, full_path)) => {
                let field_entry = self.cached_schema.get_field_entry(field);
                Ok((field, full_path, field_entry))
            }
            None => Err(ValidationError::MissingField(field_name.to_string()).into()),
        }
    }

    fn parse_subquery(&self, query: proto::query::Query) -> SummaResult<Box<dyn Query>> {
        #[cfg(feature = "metrics")]
        self.subquery_counter.add(
            &Context::current(),
            1,
            &[
                KeyValue::new("index_name", self.index_name.to_owned()),
                KeyValue::new("query", query.to_label()),
            ],
        );
        Ok(match query {
            proto::query::Query::All(_) => Box::new(AllQuery),
            proto::query::Query::Empty(_) => Box::new(EmptyQuery),
            proto::query::Query::Boolean(boolean_query_proto) => {
                let mut subqueries = vec![];
                for subquery in boolean_query_proto.subqueries {
                    subqueries.push((
                        match subquery.occur() {
                            proto::Occur::Should => Occur::Should,
                            proto::Occur::Must => Occur::Must,
                            proto::Occur::MustNot => Occur::MustNot,
                        },
                        self.parse_subquery(subquery.query.and_then(|query| query.query).ok_or(Error::EmptyQuery)?)?,
                    ))
                }
                Box::new(BooleanQuery::new(subqueries))
            }
            proto::query::Query::DisjunctionMax(disjunction_max_proto) => Box::new(DisjunctionMaxQuery::with_tie_breaker(
                disjunction_max_proto
                    .disjuncts
                    .into_iter()
                    .map(|disjunct| self.parse_subquery(disjunct.query.ok_or(Error::EmptyQuery)?))
                    .collect::<SummaResult<Vec<_>>>()?,
                match disjunction_max_proto.tie_breaker.as_str() {
                    "" => 0.0,
                    s => f32::from_str(s).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {} as f32", disjunction_max_proto.tie_breaker)))?,
                },
            )),
            proto::query::Query::Match(match_query_proto) => {
                let default_fields = if !match_query_proto.default_fields.is_empty() {
                    match_query_proto.default_fields
                } else {
                    self.index_default_fields.clone()
                };
                if default_fields.is_empty() {
                    warn!(
                        action = "missing_default_fields",
                        hint = "Add `default_fields` to match query, otherwise you match nothing"
                    )
                }
                let mut nested_query_parser = QueryParser::for_index(&self.index, default_fields)?;
                nested_query_parser.set_default_mode(match_query_proto.default_mode.into());

                if !match_query_proto.field_boosts.is_empty() {
                    nested_query_parser.set_field_boosts(match_query_proto.field_boosts)
                }

                if let Some(exact_matches_promoter) = match_query_proto.exact_matches_promoter {
                    nested_query_parser.set_exact_match_promoter(exact_matches_promoter)
                }

                if !self.field_aliases.is_empty() {
                    nested_query_parser.set_field_aliases(self.field_aliases.clone())
                }

                match nested_query_parser.parse_query(&match_query_proto.value) {
                    Ok(parsed_query) => {
                        info!(parsed_match_query = ?parsed_query);
                        Ok(parsed_query)
                    }
                    Err(QueryParserError::FieldDoesNotExist(field)) => Err(ValidationError::MissingField(field).into()),
                    Err(e) => Err(Error::InvalidQuerySyntax(Box::new(e), match_query_proto.value.to_owned())),
                }?
            }
            proto::query::Query::Range(range_query_proto) => {
                let (field, full_path, field_entry) = self.field_and_field_entry(&range_query_proto.field)?;
                let value = range_query_proto.value.as_ref().ok_or(ValidationError::MissingRange)?;
                let left = cast_value_to_bound_term(field, full_path, field_entry.field_type(), &value.left, value.including_left)?;
                let right = cast_value_to_bound_term(field, full_path, field_entry.field_type(), &value.right, value.including_right)?;
                Box::new(RangeQuery::new_term_bounds(
                    range_query_proto.field.clone(),
                    field_entry.field_type().value_type(),
                    &left,
                    &right,
                ))
            }
            proto::query::Query::Boost(boost_query_proto) => Box::new(BoostQuery::new(
                self.parse_subquery(boost_query_proto.query.and_then(|query| query.query).ok_or(Error::EmptyQuery)?)?,
                f32::from_str(&boost_query_proto.score).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {} as f32", boost_query_proto.score)))?,
            )),
            proto::query::Query::Regex(regex_query_proto) => {
                let (field, _, _) = self.field_and_field_entry(&regex_query_proto.field)?;
                Box::new(RegexQuery::from_pattern(&regex_query_proto.value, field)?)
            }
            proto::query::Query::Phrase(phrase_query_proto) => {
                let (field, full_path, field_entry) = self.field_and_field_entry(&phrase_query_proto.field)?;
                let tokenizer = self.index.tokenizer_for_field(field)?;

                let mut token_stream = tokenizer.token_stream(&phrase_query_proto.value);
                let mut terms: Vec<(usize, Term)> = vec![];
                while let Some(token) = token_stream.next() {
                    terms.push((token.position, cast_value_to_term(field, full_path, field_entry.field_type(), &token.text)?))
                }
                if terms.is_empty() {
                    Box::new(EmptyQuery)
                } else if terms.len() == 1 {
                    Box::new(TermQuery::new(
                        terms[0].1.clone(),
                        field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                    ))
                } else {
                    Box::new(PhraseQuery::new_with_offset_and_slop(terms, phrase_query_proto.slop))
                }
            }
            proto::query::Query::Term(term_query_proto) => {
                let (field, full_path, field_entry) = self.field_and_field_entry(&term_query_proto.field)?;
                Box::new(TermQuery::new(
                    cast_value_to_term(field, full_path, field_entry.field_type(), &term_query_proto.value)?,
                    field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                ))
            }
            proto::query::Query::MoreLikeThis(more_like_this_query_proto) => {
                let document = self
                    .cached_schema
                    .parse_document(&more_like_this_query_proto.document)
                    .map_err(|_e| Error::InvalidSyntax("bad document".to_owned()))?;
                let field_values = document
                    .get_sorted_field_values()
                    .into_iter()
                    .map(|(field, field_values)| (field, field_values.into_iter().cloned().collect()))
                    .collect();
                let mut query_builder = MoreLikeThisQuery::builder();
                if let Some(min_doc_frequency) = more_like_this_query_proto.min_doc_frequency {
                    query_builder = query_builder.with_min_doc_frequency(min_doc_frequency);
                }
                if let Some(max_doc_frequency) = more_like_this_query_proto.max_doc_frequency {
                    query_builder = query_builder.with_max_doc_frequency(max_doc_frequency);
                }
                if let Some(min_term_frequency) = more_like_this_query_proto.min_term_frequency {
                    query_builder = query_builder.with_min_term_frequency(min_term_frequency as usize);
                }
                if let Some(max_query_terms) = more_like_this_query_proto.max_query_terms {
                    query_builder = query_builder.with_max_query_terms(max_query_terms as usize);
                }
                if let Some(min_word_length) = more_like_this_query_proto.min_word_length {
                    query_builder = query_builder.with_min_word_length(min_word_length as usize);
                }
                if let Some(max_word_length) = more_like_this_query_proto.max_word_length {
                    query_builder = query_builder.with_max_word_length(max_word_length as usize);
                }
                if let Some(ref boost) = more_like_this_query_proto.boost {
                    query_builder =
                        query_builder.with_boost_factor(f32::from_str(boost).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {boost} as f32")))?);
                }
                query_builder = query_builder.with_stop_words(more_like_this_query_proto.stop_words);
                Box::new(query_builder.with_document_fields(field_values))
            }
            proto::query::Query::Exists(exists_query_proto) => {
                let (field, full_path, field_entry) = self.field_and_field_entry(&exists_query_proto.field)?;
                if !field_entry.field_type().is_indexed() {
                    let fni = QueryParserError::FieldNotIndexed(field_entry.name().to_string());
                    return Err(Error::InvalidQuerySyntax(Box::new(fni), exists_query_proto.field.to_string()));
                }
                if full_path == "" {
                    Box::new(ExistsQuery::new(field))
                } else {
                    Box::new(TermQuery::new(
                        cast_value_to_term(field, full_path, field_entry.field_type(), "")?,
                        field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                    ))
                }
            }
        })
    }

    pub fn parse_query(&self, query: proto::query::Query) -> SummaResult<Box<dyn Query>> {
        #[cfg(feature = "metrics")]
        self.query_counter.add(
            &Context::current(),
            1,
            &[
                KeyValue::new("index_name", self.index_name.to_owned()),
                KeyValue::new("query", query.to_label()),
            ],
        );
        self.parse_subquery(query)
    }
}
