//! Whole-term Unicode wildcard filters over the existing term dictionary.
use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{Error, Result};

use super::term_union::{TermUnionScorer, materialize_union, reject_chunked};
use super::traits::{CountFuture, Query, Scorer, ScorerFuture};

const MAX_PATTERN_BYTES: usize = 1024;
const MAX_SCANNED_TERMS: usize = 1_000_000;
const MAX_REGEX_BYTES: usize = 2 * 1024 * 1024;

/// Constant-score union of indexed terms matching a whole-term wildcard.
///
/// `*` matches any sequence of Unicode scalar values, `?` exactly one, and `\`
/// escapes the next character. Patterns are not tokenized or stemmed. Expansion
/// uses the same 1,024-term / 5,000,000-posting per-segment limits as prefixes;
/// scans additionally stop with an error after 1,000,000 candidate terms.
#[derive(Debug, Clone)]
pub struct WildcardQuery {
    field: Field,
    pattern: Arc<Pattern>,
}

#[derive(Debug)]
struct Pattern {
    source: String,
    prefix: Vec<u8>,
    regex: regex::Regex,
}

impl WildcardQuery {
    /// Compile a case-sensitive pattern against indexed UTF-8 terms.
    pub fn new(field: Field, pattern: impl AsRef<str>) -> Result<Self> {
        let source = pattern.as_ref();
        if source.len() > MAX_PATTERN_BYTES {
            return Err(Error::Query(format!(
                "wildcard pattern exceeds {MAX_PATTERN_BYTES} bytes"
            )));
        }
        let mut expression = String::from("\\A(?:");
        let mut prefix = String::new();
        let mut literal_prefix = true;
        let mut characters = source.chars();
        while let Some(character) = characters.next() {
            match character {
                '*' => {
                    expression.push_str(".*");
                    literal_prefix = false;
                }
                '?' => {
                    expression.push('.');
                    literal_prefix = false;
                }
                _ => {
                    let literal = if character == '\\' {
                        characters.next().ok_or_else(|| {
                            Error::Query("wildcard pattern ends with an escape".into())
                        })?
                    } else {
                        character
                    };
                    expression.push_str(&regex::escape(literal.encode_utf8(&mut [0; 4])));
                    if literal_prefix {
                        prefix.push(literal);
                    }
                }
            }
        }
        expression.push_str(")\\z");
        let regex = regex::RegexBuilder::new(&expression)
            .dot_matches_new_line(true)
            .size_limit(MAX_REGEX_BYTES)
            .dfa_size_limit(MAX_REGEX_BYTES)
            .build()
            .map_err(|error| Error::Query(format!("invalid wildcard pattern: {error}")))?;
        Ok(Self {
            field,
            pattern: Arc::new(Pattern {
                source: source.to_owned(),
                prefix: prefix.into_bytes(),
                regex,
            }),
        })
    }

    /// Lowercase a pattern to match a lowercase term vocabulary.
    pub fn text(field: Field, pattern: &str) -> Result<Self> {
        if pattern.len() > MAX_PATTERN_BYTES {
            return Err(Error::Query(format!(
                "wildcard pattern exceeds {MAX_PATTERN_BYTES} bytes"
            )));
        }
        Self::new(field, pattern.to_lowercase())
    }
}

impl Pattern {
    fn matches(&self, term: &[u8]) -> bool {
        std::str::from_utf8(term).is_ok_and(|term| self.regex.is_match(term))
    }
}

impl std::fmt::Display for WildcardQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wildcard({}:{:?})", self.field.0, self.pattern.source)
    }
}

impl Query for WildcardQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let pattern = self.pattern.clone();
        Box::pin(async move {
            reject_chunked(reader, field, "WildcardQuery")?;
            let postings = reader
                .get_matching_postings(
                    field,
                    &pattern.prefix,
                    "wildcard",
                    MAX_SCANNED_TERMS,
                    |term| pattern.matches(term),
                )
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
        reject_chunked(reader, self.field, "WildcardQuery")?;
        let postings = reader.get_matching_postings_sync(
            self.field,
            &self.pattern.prefix,
            "wildcard",
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
        Box::pin(async move {
            reject_chunked(reader, field, "WildcardQuery")?;
            let postings = reader
                .get_matching_postings(
                    field,
                    &pattern.prefix,
                    "wildcard",
                    MAX_SCANNED_TERMS,
                    |term| pattern.matches(term),
                )
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
