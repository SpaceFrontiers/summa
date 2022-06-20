use std::ops::Bound;
use std::ops::Bound::Unbounded;
use std::str::FromStr;

#[cfg(feature = "metrics")]
use opentelemetry::metrics::Counter;
#[cfg(feature = "metrics")]
use opentelemetry::Context;
#[cfg(feature = "metrics")]
use opentelemetry::{global, KeyValue};
use summa_proto::proto;
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, DisjunctionMaxQuery, EmptyQuery, MoreLikeThisQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery,
};
use tantivy::schema::{Field, FieldEntry, FieldType, IndexRecordOption, Schema};
use tantivy::{Index, Score, Term};
use tracing::info;

use crate::components::queries::ExistsQuery;
use crate::components::query_parser::morphology::MorphologyManager;
use crate::components::query_parser::utils::cast_field_to_typed_term;
use crate::components::query_parser::{QueryParser, QueryParserError};
use crate::configs::core::QueryParserConfig;
use crate::errors::{Error, SummaResult, ValidationError};
#[cfg(feature = "metrics")]
use crate::metrics::ToLabel;

/// Responsible for casting `crate::proto::Query` message to `tantivy::query::Query`
#[derive(Clone)]
pub struct ProtoQueryParser {
    index: Index,
    index_name: String,
    cached_schema: Schema,
    // Counters
    #[cfg(feature = "metrics")]
    query_counter: Counter<u64>,
    #[cfg(feature = "metrics")]
    subquery_counter: Counter<u64>,
    query_parser_config: QueryParserConfig,
    morphology_manager: MorphologyManager,
}

pub enum QueryParserDefaultMode {
    Boolean,
    DisjuctionMax { tie_breaker: Score },
}

impl From<Option<proto::query_parser_config::DefaultMode>> for QueryParserDefaultMode {
    fn from(value: Option<proto::query_parser_config::DefaultMode>) -> Self {
        match value {
            Some(proto::query_parser_config::DefaultMode::BooleanShouldMode(_)) | None => QueryParserDefaultMode::Boolean,
            Some(proto::query_parser_config::DefaultMode::DisjuctionMaxMode(proto::MatchQueryDisjuctionMaxMode { tie_breaker })) => {
                QueryParserDefaultMode::DisjuctionMax { tie_breaker }
            }
        }
    }
}

fn cast_value_to_bound_term(field: &Field, full_path: &str, field_type: &FieldType, value: &str, including: bool) -> SummaResult<Bound<Term>> {
    Ok(match value {
        "*" => Unbounded,
        value => {
            let casted_value = cast_field_to_typed_term(field, full_path, field_type, value)?;
            if including {
                Bound::Included(casted_value)
            } else {
                Bound::Excluded(casted_value)
            }
        }
    })
}

impl ProtoQueryParser {
    pub fn for_index(index_name: &str, index: &Index, query_parser_config: proto::QueryParserConfig) -> SummaResult<ProtoQueryParser> {
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
            query_parser_config: QueryParserConfig(query_parser_config),
            morphology_manager: MorphologyManager::default(),
        })
    }

    pub fn resolve_field_name<'a>(&'a self, field_name: &'a str) -> &str {
        self.query_parser_config
            .0
            .field_aliases
            .get(field_name)
            .map(|s| s.as_str())
            .unwrap_or(field_name)
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
                let mut new_query_parser_config = self.query_parser_config.clone();
                if let Some(query_parser_config) = match_query_proto.query_parser_config {
                    new_query_parser_config.merge(QueryParserConfig(query_parser_config));
                }
                let nested_query_parser = QueryParser::for_index(&self.index, new_query_parser_config.clone(), &self.morphology_manager)?;
                match nested_query_parser.parse_query(&match_query_proto.value) {
                    Ok(parsed_query) => {
                        info!(query = ?match_query_proto.value, parsed_match_query = ?parsed_query, query_parser_config = ?new_query_parser_config);
                        Ok(parsed_query)
                    }
                    Err(QueryParserError::FieldDoesNotExist(field)) => Err(ValidationError::MissingField(field).into()),
                    Err(e) => Err(Error::InvalidQuerySyntax(Box::new(e), match_query_proto.value.to_owned())),
                }?
            }
            proto::query::Query::Range(range_query_proto) => {
                let (field, full_path, field_entry) = self.field_and_field_entry(&range_query_proto.field)?;
                let value = range_query_proto.value.as_ref().ok_or(ValidationError::MissingRange)?;
                let left = cast_value_to_bound_term(&field, full_path, field_entry.field_type(), &value.left, value.including_left)?;
                let right = cast_value_to_bound_term(&field, full_path, field_entry.field_type(), &value.right, value.including_right)?;
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
                let mut tokenizer = self.index.tokenizer_for_field(field)?;

                let mut token_stream = tokenizer.token_stream(&phrase_query_proto.value);
                let mut terms: Vec<(usize, Term)> = vec![];
                while let Some(token) = token_stream.next() {
                    terms.push((
                        token.position,
                        cast_field_to_typed_term(&field, full_path, field_entry.field_type(), &token.text)?,
                    ))
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
                    cast_field_to_typed_term(&field, full_path, field_entry.field_type(), &term_query_proto.value)?,
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
                Box::new(ExistsQuery::new(field, full_path))
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
