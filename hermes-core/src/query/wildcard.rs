//! Whole-term Unicode wildcard filters over the existing term dictionary.
use super::term_pattern::{TermPatternQuery, check_length};
#[cfg(feature = "sync")]
use super::traits::Scorer;
use super::traits::{CountFuture, Query, ScorerFuture};
use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{Error, Result};

/// Constant-score union of indexed terms matching a whole-term wildcard.
///
/// `*` matches any sequence of Unicode scalar values, `?` exactly one, and `\`
/// escapes the next character. Patterns are not tokenized or stemmed. Expansion
/// uses the same 1,024-term / 5,000,000-posting per-segment limits as prefixes;
/// scans additionally stop with an error after 1,000,000 candidate terms.
#[derive(Debug, Clone)]
pub struct WildcardQuery(TermPatternQuery);

impl WildcardQuery {
    /// Compile a case-sensitive pattern against indexed UTF-8 terms.
    pub fn new(field: Field, pattern: impl AsRef<str>) -> Result<Self> {
        let source = pattern.as_ref();
        check_length(source, "wildcard")?;
        let mut expression = String::new();
        let mut prefix = String::new();
        let mut suffix = String::new();
        let mut stars = 0usize;
        let mut questions = false;
        let mut literal_prefix = true;
        let mut characters = source.chars();
        while let Some(character) = characters.next() {
            match character {
                '*' => {
                    stars += 1;
                    expression.push_str(".*");
                    literal_prefix = false;
                }
                '?' => {
                    questions = true;
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
                    } else {
                        suffix.push(literal);
                    }
                }
            }
        }
        if stars == 1 && !questions {
            return Ok(Self(TermPatternQuery::single_star(
                field,
                source,
                prefix.into_bytes(),
                suffix.into_bytes(),
            )));
        }
        Ok(Self(TermPatternQuery::compile(
            field,
            source,
            &expression,
            vec![prefix.into_bytes()],
            "wildcard",
        )?))
    }

    /// Lowercase a pattern to match a lowercase term vocabulary.
    pub fn text(field: Field, pattern: &str) -> Result<Self> {
        check_length(pattern, "wildcard")?;
        Self::new(field, pattern.to_lowercase())
    }
}

impl std::fmt::Display for WildcardQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl Query for WildcardQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        self.0.scorer(reader, limit)
    }
    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
    ) -> Result<Box<dyn Scorer + 'a>> {
        self.0.scorer_sync(reader, limit)
    }
    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        self.0.count_estimate(reader)
    }
    fn is_filter(&self) -> bool {
        true
    }
}
