//! Bounded whole-term regular-expression filters.
use super::term_pattern::{TermPatternQuery, check_length};
#[cfg(feature = "sync")]
use super::traits::Scorer;
use super::traits::{CountFuture, Query, ScorerFuture};
use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::{Error, Result};

/// Constant-score union of indexed terms matching a whole regular expression.
///
/// Supports literals, classes/ranges, grouping, alternation, `.`, `?`, `*`, `+`
/// and bounded repetition. Matching is case-sensitive and Unicode-aware; patterns
/// are not analyzed. Extended Lucene operators and regex-engine extensions are
/// rejected. Existing wildcard dictionary/posting limits apply.
#[derive(Debug, Clone)]
pub struct RegexQuery(TermPatternQuery);

impl RegexQuery {
    pub fn new(field: Field, pattern: impl AsRef<str>) -> Result<Self> {
        let source = pattern.as_ref();
        check_length(source, "regex")?;
        validate_syntax(source)?;
        // No guessed prefix: alternation or optional prefixes can make a literal
        // prefix unsafe. Dictionary scans retain the shared hard work limits.
        Ok(Self(TermPatternQuery::compile(
            field,
            source,
            source,
            Vec::new(),
            "regex",
        )?))
    }
}

fn validate_syntax(source: &str) -> Result<()> {
    let unsupported = || {
        Error::Query(
            "unsupported regex syntax; use literals, classes, groups, alternation and repetition"
                .into(),
        )
    };
    let mut chars = source.chars().peekable();
    let mut class = false;
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                let escaped = chars
                    .next()
                    .ok_or_else(|| Error::Query("regex pattern ends with an escape".into()))?;
                if escaped.is_alphanumeric() {
                    return Err(unsupported());
                }
            }
            '[' if class => return Err(unsupported()),
            '[' => class = true,
            ']' => class = false,
            '&' | '~' | '#' | '@' | '<' | '>' | '"' if !class => return Err(unsupported()),
            '^' | '$' if !class => return Err(unsupported()),
            '(' if !class && chars.peek() == Some(&'?') => return Err(unsupported()),
            '&' | '~' | '|' | '-' if class && chars.peek() == Some(&ch) => {
                return Err(unsupported());
            }
            _ => {}
        }
    }
    Ok(())
}

impl std::fmt::Display for RegexQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl Query for RegexQuery {
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
