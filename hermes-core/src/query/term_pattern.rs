//! Shared bounded dictionary matching for whole-term pattern filters.
use super::term_union::{TermUnionScorer, materialize_union, reject_chunked};
use super::traits::{CountFuture, Query, Scorer, ScorerFuture};
use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{Error, Result};
use std::sync::Arc;

const MAX_PATTERN_BYTES: usize = 1024;
const MAX_SCANNED_TERMS: usize = 1_000_000;
const MAX_REGEX_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct TermPatternQuery {
    field: Field,
    pattern: Arc<Pattern>,
    label: &'static str,
}

#[derive(Debug)]
struct Pattern {
    source: String,
    prefix: Vec<u8>,
    regex: regex::Regex,
}

impl TermPatternQuery {
    fn validate_field(reader: &SegmentReader, field: Field, label: &str) -> Result<()> {
        let entry = reader
            .schema()
            .get_field_entry(field)
            .ok_or_else(|| Error::Query(format!("{label}: unknown field {}", field.0)))?;
        if entry.field_type != crate::dsl::FieldType::Text || !entry.indexed {
            return Err(Error::Query(format!(
                "{label} requires an indexed text field, but '{}' is {:?} (indexed={})",
                entry.name, entry.field_type, entry.indexed
            )));
        }
        reject_chunked(reader, field, label)
    }

    pub(super) fn compile(
        field: Field,
        source: &str,
        expression: &str,
        prefix: Vec<u8>,
        label: &'static str,
    ) -> Result<Self> {
        check_length(source, label)?;
        let regex = regex::RegexBuilder::new(&format!("\\A(?:{expression})\\z"))
            .dot_matches_new_line(true)
            .size_limit(MAX_REGEX_BYTES)
            .dfa_size_limit(MAX_REGEX_BYTES)
            .build()
            .map_err(|error| Error::Query(format!("invalid {label} pattern: {error}")))?;
        Ok(Self {
            field,
            pattern: Arc::new(Pattern {
                source: source.to_owned(),
                prefix,
                regex,
            }),
            label,
        })
    }
}

pub(super) fn check_length(source: &str, label: &str) -> Result<()> {
    if source.len() > MAX_PATTERN_BYTES {
        return Err(Error::Query(format!(
            "{label} pattern exceeds {MAX_PATTERN_BYTES} bytes"
        )));
    }
    Ok(())
}

impl Pattern {
    fn matches(&self, term: &[u8]) -> bool {
        std::str::from_utf8(term).is_ok_and(|term| self.regex.is_match(term))
    }
}

impl std::fmt::Display for TermPatternQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({}:{:?})",
            if self.label == "wildcard" {
                "Wildcard"
            } else {
                "Regex"
            },
            self.field.0,
            self.pattern.source
        )
    }
}

impl Query for TermPatternQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let pattern = self.pattern.clone();
        let label = self.label;
        Box::pin(async move {
            Self::validate_field(reader, field, label)?;
            let postings = reader
                .get_matching_postings(field, &pattern.prefix, label, MAX_SCANNED_TERMS, |term| {
                    pattern.matches(term)
                })
                .await?;
            Ok(Box::new(TermUnionScorer::new(materialize_union(
                &postings,
                reader.num_docs(),
                reader.chunk_map(field),
            ))) as Box<dyn Scorer>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        _limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        Self::validate_field(reader, self.field, self.label)?;
        let postings = reader.get_matching_postings_sync(
            self.field,
            &self.pattern.prefix,
            self.label,
            MAX_SCANNED_TERMS,
            |term| self.pattern.matches(term),
        )?;
        Ok(Box::new(TermUnionScorer::new(materialize_union(
            &postings,
            reader.num_docs(),
            reader.chunk_map(self.field),
        ))))
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let pattern = self.pattern.clone();
        let label = self.label;
        Box::pin(async move {
            Self::validate_field(reader, field, label)?;
            let postings = reader
                .get_matching_postings(field, &pattern.prefix, label, MAX_SCANNED_TERMS, |term| {
                    pattern.matches(term)
                })
                .await?;
            Ok(postings
                .iter()
                .fold(0u32, |sum, posting| sum.saturating_add(posting.doc_count()))
                .min(reader.num_docs()))
        })
    }

    fn is_filter(&self) -> bool {
        true
    }
}
