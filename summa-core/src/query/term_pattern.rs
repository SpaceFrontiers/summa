//! Shared bounded dictionary matching for whole-term pattern filters.
use super::term_union::{TermUnionScorer, reject_chunked};
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
    prefixes: Vec<Vec<u8>>,
    matcher: Matcher,
}

#[derive(Debug)]
enum Matcher {
    Regex(regex::Regex),
    SingleStar { prefix: Vec<u8>, suffix: Vec<u8> },
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
        prefixes: Vec<Vec<u8>>,
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
                prefixes,
                matcher: Matcher::Regex(regex),
            }),
            label,
        })
    }

    pub(super) fn single_star(
        field: Field,
        source: &str,
        prefix: Vec<u8>,
        suffix: Vec<u8>,
    ) -> Self {
        Self {
            field,
            pattern: Arc::new(Pattern {
                source: source.to_owned(),
                prefixes: vec![prefix.clone()],
                matcher: Matcher::SingleStar { prefix, suffix },
            }),
            label: "wildcard",
        }
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
        match &self.matcher {
            Matcher::Regex(regex) => {
                std::str::from_utf8(term).is_ok_and(|term| regex.is_match(term))
            }
            Matcher::SingleStar { prefix, suffix } => {
                term.len() >= prefix.len() + suffix.len()
                    && term.starts_with(prefix)
                    && term.ends_with(suffix)
                    && std::str::from_utf8(term).is_ok()
            }
        }
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
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let pattern = self.pattern.clone();
        let label = self.label;
        Box::pin(async move {
            Self::validate_field(reader, field, label)?;
            let postings = reader
                .get_matching_postings(field, &pattern.prefixes, label, MAX_SCANNED_TERMS, |term| {
                    pattern.matches(term)
                })
                .await?;
            Ok(Box::new(TermUnionScorer::from_expanded(
                postings,
                reader.num_docs(),
                reader.chunk_map(field),
                limit,
            )) as Box<dyn Scorer>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        Self::validate_field(reader, self.field, self.label)?;
        let postings = reader.get_matching_postings_sync(
            self.field,
            &self.pattern.prefixes,
            self.label,
            MAX_SCANNED_TERMS,
            |term| self.pattern.matches(term),
        )?;
        Ok(Box::new(TermUnionScorer::from_expanded(
            postings,
            reader.num_docs(),
            reader.chunk_map(self.field),
            limit,
        )))
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let pattern = self.pattern.clone();
        let label = self.label;
        Box::pin(async move {
            Self::validate_field(reader, field, label)?;
            let postings = reader
                .get_matching_postings(field, &pattern.prefixes, label, MAX_SCANNED_TERMS, |term| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_star_matches_regex_for_unicode_overlap_empty_and_invalid_terms() {
        let mut terms = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..4 {
            frontier = frontier
                .iter()
                .flat_map(|prefix| {
                    ['a', 'b', 'é', '🦀', '*', '?', '\n']
                        .into_iter()
                        .map(move |ch| format!("{prefix}{ch}"))
                })
                .collect();
            terms.extend(frontier.iter().cloned());
        }
        for prefix in ["", "a", "ab", "é", "*"] {
            for suffix in ["", "b", "ba", "🦀", "?"] {
                let query = TermPatternQuery::single_star(
                    Field(0),
                    "test",
                    prefix.as_bytes().to_vec(),
                    suffix.as_bytes().to_vec(),
                );
                let expression = regex::RegexBuilder::new(&format!(
                    r"\A{}.*{}\z",
                    regex::escape(prefix),
                    regex::escape(suffix)
                ))
                .dot_matches_new_line(true)
                .build()
                .unwrap();
                for term in &terms {
                    assert_eq!(
                        query.pattern.matches(term.as_bytes()),
                        expression.is_match(term),
                        "{prefix}*{suffix}, {term:?}"
                    );
                }
                let mut invalid = prefix.as_bytes().to_vec();
                invalid.push(0xff);
                invalid.extend_from_slice(suffix.as_bytes());
                assert!(!query.pattern.matches(&invalid));
            }
        }
    }
}
