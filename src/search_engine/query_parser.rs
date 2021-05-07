use crate::errors::{Error, SummaResult};
use crate::proto;
use std::ops::Bound;
use std::ops::Bound::Unbounded;
use std::str::FromStr;
use tantivy::query::{AllQuery, BooleanQuery, BoostQuery, MoreLikeThisQuery, Occur, PhraseQuery, Query, RangeQuery, RegexQuery, TermQuery};
use tantivy::schema::{Field, FieldEntry, FieldType, IndexRecordOption, Schema};
use tantivy::{Index, Term};

pub struct QueryParser {
    cached_schema: Schema,
    index: Index,
    nested_query_parser: tantivy::query::QueryParser,
}

fn cast_value_to_term(field: Field, field_type: &FieldType, value: &str) -> SummaResult<Term> {
    Ok(match field_type {
        FieldType::Str(_) => Term::from_field_text(field, value),
        FieldType::I64(_) => Term::from_field_i64(
            field,
            i64::from_str(value).map_err(|_e| Error::InvalidSyntaxError(format!("cannot parse {} as i64", value)))?,
        ),
        FieldType::U64(_) => Term::from_field_u64(
            field,
            u64::from_str(value).map_err(|_e| Error::InvalidSyntaxError(format!("cannot parse {} as u64", value)))?,
        ),
        FieldType::F64(_) => Term::from_field_f64(
            field,
            f64::from_str(value).map_err(|_e| Error::InvalidSyntaxError(format!("cannot parse {} as f64", value)))?,
        ),
        FieldType::Bytes(_) => Term::from_field_bytes(
            field,
            &base64::decode(value).map_err(|_e| Error::InvalidSyntaxError(format!("cannot parse {} as bytes", value)))?,
        ),
        FieldType::Date(_) => Term::from_field_f64(
            field,
            f64::from_str(value).map_err(|_e| Error::InvalidSyntaxError(format!("cannot parse {} as date", value)))?,
        ),
        _ => return Err(Error::InvalidSyntaxError("invalid range type".to_owned()))?,
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
    pub fn for_index(index: &Index, default_fields: Vec<Field>) -> QueryParser {
        let nested_query_parser = tantivy::query::QueryParser::for_index(index, default_fields);
        QueryParser {
            cached_schema: index.schema(),
            index: index.clone(),
            nested_query_parser,
        }
    }

    #[inline]
    pub(crate) fn field_and_field_entry(&self, field_name: &str) -> SummaResult<(Field, &FieldEntry)> {
        let field = self.cached_schema.get_field(field_name).ok_or(Error::FieldDoesNotExistError(field_name.to_owned()))?;
        let field_entry = self.cached_schema.get_field_entry(field);
        Ok((field, field_entry))
    }

    pub fn parse_query(&self, query: &proto::Query) -> SummaResult<Box<dyn Query>> {
        Ok(match &query.query {
            None | Some(proto::query::Query::All(_)) => Box::new(AllQuery),
            Some(proto::query::Query::Bool(boolean_query)) => {
                let mut subqueries = vec![];
                for subquery in &boolean_query.subqueries {
                    subqueries.push((
                        match proto::Occur::from_i32(subquery.occur) {
                            None | Some(proto::Occur::Should) => Occur::Should,
                            Some(proto::Occur::Must) => Occur::Must,
                            Some(proto::Occur::MustNot) => Occur::MustNot,
                        },
                        self.parse_query(subquery.query.as_ref().ok_or(Error::EmptyQueryError)?)?,
                    ))
                }
                Box::new(BooleanQuery::new(subqueries))
            }
            Some(proto::query::Query::Match(match_query_proto)) => match self.nested_query_parser.parse_query(&match_query_proto.value) {
                Ok(parsed_query) => Ok(parsed_query),
                Err(tantivy::query::QueryParserError::FieldDoesNotExist(field)) => Err(Error::FieldDoesNotExistError(field)),
                Err(e) => Err(Error::InvalidTantivySyntaxError((e, match_query_proto.value.to_owned()))),
            }?,
            Some(proto::query::Query::Range(range_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&range_query_proto.field)?;
                let value = range_query_proto.value.as_ref().unwrap();
                let left = cast_value_to_bound_term(field, field_entry.field_type(), &value.left, value.including_left)?;
                let right = cast_value_to_bound_term(field, field_entry.field_type(), &value.right, value.including_right)?;
                Box::new(RangeQuery::new_term_bounds(field, field_entry.field_type().value_type(), &left, &right))
            }
            Some(proto::query::Query::Boost(boost_query_proto)) => Box::new(BoostQuery::new(
                self.parse_query(boost_query_proto.query.as_ref().ok_or(Error::EmptyQueryError)?)?,
                f32::from_str(&boost_query_proto.score).map_err(|_e| Error::InvalidSyntaxError(format!("cannot parse {} as f32", boost_query_proto.score)))?,
            )),
            Some(proto::query::Query::Regex(regex_query_proto)) => {
                let (field, _) = self.field_and_field_entry(&regex_query_proto.field)?;
                Box::new(RegexQuery::from_pattern(&regex_query_proto.value, field)?)
            }
            Some(proto::query::Query::Phrase(phrase_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&phrase_query_proto.field)?;
                let tokenizer = self.index.tokenizer_for_field(field)?;
                let mut token_stream = tokenizer.token_stream(&phrase_query_proto.value);
                let mut terms = vec![];
                while let Some(token) = token_stream.next() {
                    terms.push(cast_value_to_term(field, field_entry.field_type(), &token.text)?)
                }
                if terms.len() == 1 {
                    Box::new(TermQuery::new(
                        terms[0].clone(),
                        field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                    ))
                } else {
                    let mut phrase_query = PhraseQuery::new(terms);
                    phrase_query.set_slop(phrase_query_proto.slop);
                    Box::new(phrase_query)
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
                    .map_err(|_e| Error::InvalidSyntaxError("bad document".to_owned()))?;
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
                    query_builder = query_builder.with_min_term_frequency(min_term_frequency.try_into().unwrap());
                }
                if let Some(max_query_terms) = more_like_this_query_proto.max_query_terms {
                    query_builder = query_builder.with_max_query_terms(max_query_terms.try_into().unwrap());
                }
                if let Some(min_word_length) = more_like_this_query_proto.min_word_length {
                    query_builder = query_builder.with_min_word_length(min_word_length.try_into().unwrap());
                }
                if let Some(max_word_length) = more_like_this_query_proto.max_word_length {
                    query_builder = query_builder.with_max_word_length(max_word_length.try_into().unwrap());
                }
                if let Some(boost_factor) = more_like_this_query_proto.boost_factor {
                    query_builder = query_builder.with_boost_factor(boost_factor.try_into().unwrap());
                }
                query_builder = query_builder.with_stop_words(more_like_this_query_proto.stop_words.clone());
                Box::new(query_builder.with_document_fields(field_values))
            }
        })
    }
}
