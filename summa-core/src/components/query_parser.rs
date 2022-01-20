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
use tantivy::query::{
    AllQuery, BooleanQuery, BoostQuery, DisjunctionMaxQuery, EmptyQuery, MoreLikeThisQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery,
};
use tantivy::schema::{Field, FieldEntry, FieldType, IndexRecordOption, Schema};
use tantivy::{DateTime, Index, Term};

use crate::errors::{Error, SummaResult, ValidationError};
#[cfg(feature = "metrics")]
use crate::metrics::ToLabel;

/// Responsible for casting `crate::proto::Query` message to `tantivy::query::Query`
pub struct QueryParser {
    index: Index,
    index_name: String,
    cached_schema: Schema,
    nested_query_parser: tantivy::query::QueryParser,
    // Counters
    #[cfg(feature = "metrics")]
    query_counter: Counter<u64>,
    #[cfg(feature = "metrics")]
    subquery_counter: Counter<u64>,
}

fn cast_value_to_term(field: Field, field_type: &FieldType, value: &str) -> SummaResult<Term> {
    Ok(match field_type {
        FieldType::Str(_) => Term::from_field_text(field, value),
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

fn cast_value_to_bound_term(field: Field, field_type: &FieldType, value: &str, including: bool) -> SummaResult<Bound<Term>> {
    Ok(match value {
        "*" => Unbounded,
        value => {
            let casted_value = cast_value_to_term(field, field_type, value)?;
            if including {
                Bound::Included(casted_value)
            } else {
                Bound::Excluded(casted_value)
            }
        }
    })
}

impl QueryParser {
    pub fn for_index(index_name: &str, index: &Index) -> SummaResult<QueryParser> {
        let index_meta = index.load_metas()?;
        let index_attributes: Option<proto::IndexAttributes> = index_meta.index_attributes()?;
        let default_fields = index_attributes
            .map(|index_attributes| index_attributes.default_fields)
            .unwrap_or_else(Vec::new)
            .iter()
            .map(|field_name| index.schema().get_field(field_name))
            .collect::<Result<_, _>>()?;
        let nested_query_parser = tantivy::query::QueryParser::for_index(index, default_fields);
        #[cfg(feature = "metrics")]
        let query_counter = global::meter("summa").u64_counter("query_counter").with_description("Queries counter").init();
        #[cfg(feature = "metrics")]
        let subquery_counter = global::meter("summa")
            .u64_counter("subquery_counter")
            .with_description("Sub-queries counter")
            .init();

        Ok(QueryParser {
            index: index.clone(),
            index_name: index_name.to_string(),
            cached_schema: index.schema(),
            nested_query_parser,
            #[cfg(feature = "metrics")]
            query_counter,
            #[cfg(feature = "metrics")]
            subquery_counter,
        })
    }

    #[inline]
    pub(crate) fn field_and_field_entry(&self, field_name: &str) -> SummaResult<(Field, &FieldEntry)> {
        let field = self.cached_schema.get_field(field_name)?;
        let field_entry = self.cached_schema.get_field_entry(field);
        Ok((field, field_entry))
    }

    pub fn nested_query_parser(&self) -> &tantivy::query::QueryParser {
        &self.nested_query_parser
    }

    fn parse_subquery(&self, query: &proto::Query) -> SummaResult<Box<dyn Query>> {
        #[cfg(feature = "metrics")]
        self.subquery_counter.add(
            &Context::current(),
            1,
            &[
                KeyValue::new("index_name", self.index_name.to_owned()),
                KeyValue::new("query", query.to_label()),
            ],
        );
        Ok(match &query.query {
            None | Some(proto::query::Query::All(_)) => Box::new(AllQuery),
            Some(proto::query::Query::Empty(_)) => Box::new(EmptyQuery),
            Some(proto::query::Query::Boolean(boolean_query_proto)) => {
                let mut subqueries = vec![];
                for subquery in &boolean_query_proto.subqueries {
                    subqueries.push((
                        match proto::Occur::from_i32(subquery.occur) {
                            None | Some(proto::Occur::Should) => Occur::Should,
                            Some(proto::Occur::Must) => Occur::Must,
                            Some(proto::Occur::MustNot) => Occur::MustNot,
                        },
                        self.parse_subquery(subquery.query.as_ref().ok_or(Error::EmptyQuery)?)?,
                    ))
                }
                Box::new(BooleanQuery::new(subqueries))
            }
            Some(proto::query::Query::DisjunctionMax(disjunction_max_proto)) => Box::new(DisjunctionMaxQuery::with_tie_breaker(
                disjunction_max_proto
                    .disjuncts
                    .iter()
                    .map(|disjunct| self.parse_subquery(disjunct))
                    .collect::<SummaResult<Vec<_>>>()?,
                match disjunction_max_proto.tie_breaker.as_str() {
                    "" => 0.0,
                    s => f32::from_str(s).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {} as f32", disjunction_max_proto.tie_breaker)))?,
                },
            )),
            Some(proto::query::Query::Match(match_query_proto)) => match self.nested_query_parser.parse_query(&match_query_proto.value) {
                Ok(parsed_query) => Ok(parsed_query),
                Err(tantivy::query::QueryParserError::FieldDoesNotExist(field)) => Err(ValidationError::MissingField(field).into()),
                Err(e) => Err(Error::InvalidTantivySyntax(e, match_query_proto.value.to_owned())),
            }?,
            Some(proto::query::Query::Range(range_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&range_query_proto.field)?;
                let value = range_query_proto.value.as_ref().ok_or(ValidationError::MissingRange)?;
                let left = cast_value_to_bound_term(field, field_entry.field_type(), &value.left, value.including_left)?;
                let right = cast_value_to_bound_term(field, field_entry.field_type(), &value.right, value.including_right)?;
                Box::new(RangeQuery::new_term_bounds(
                    range_query_proto.field.clone(),
                    field_entry.field_type().value_type(),
                    &left,
                    &right,
                ))
            }
            Some(proto::query::Query::Boost(boost_query_proto)) => Box::new(BoostQuery::new(
                self.parse_subquery(boost_query_proto.query.as_ref().ok_or(Error::EmptyQuery)?)?,
                f32::from_str(&boost_query_proto.score).map_err(|_e| Error::InvalidSyntax(format!("cannot parse {} as f32", boost_query_proto.score)))?,
            )),
            Some(proto::query::Query::Regex(regex_query_proto)) => {
                let (field, _) = self.field_and_field_entry(&regex_query_proto.field)?;
                Box::new(RegexQuery::from_pattern(&regex_query_proto.value, field)?)
            }
            Some(proto::query::Query::Phrase(phrase_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&phrase_query_proto.field)?;
                let tokenizer = self.index.tokenizer_for_field(field)?;

                let mut token_stream = tokenizer.token_stream(&phrase_query_proto.value);
                let mut terms: Vec<(usize, Term)> = vec![];
                while let Some(token) = token_stream.next() {
                    terms.push((token.position, cast_value_to_term(field, field_entry.field_type(), &token.text)?))
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
            Some(proto::query::Query::Term(term_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&term_query_proto.field)?;
                Box::new(TermQuery::new(
                    cast_value_to_term(field, field_entry.field_type(), &term_query_proto.value)?,
                    field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                ))
            }
            Some(proto::query::Query::MoreLikeThis(more_like_this_query_proto)) => {
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
                query_builder = query_builder.with_stop_words(more_like_this_query_proto.stop_words.clone());
                Box::new(query_builder.with_document_fields(field_values))
            }
        })
    }

    pub fn parse_query(&self, query: &proto::Query) -> SummaResult<Box<dyn Query>> {
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
