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
        Ok(Self(TermPatternQuery::compile(
            field,
            source,
            source,
            literal_prefixes(source)?,
            "regex",
        )?))
    }
}

/// Every matching term must start with one extracted literal. Limits make an
/// infinite set fall back to the full field; they never truncate its language.
fn literal_prefixes(source: &str) -> Result<Vec<Vec<u8>>> {
    let hir = regex_syntax::ParserBuilder::new()
        .dot_matches_new_line(true)
        .build()
        .parse(source)
        .map_err(|error| Error::Query(format!("invalid regex pattern: {error}")))?;
    let sequence = regex_syntax::hir::literal::Extractor::new()
        .limit_total(64)
        .limit_literal_len(64)
        .limit_repeat(8)
        .extract(&hir);
    let Some(literals) = sequence.literals() else {
        return Ok(vec![Vec::new()]);
    };
    let mut prefixes: Vec<_> = literals
        .iter()
        .map(|literal| literal.as_bytes().to_vec())
        .collect();
    prefixes.sort_unstable();
    let mut disjoint: Vec<Vec<u8>> = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        if disjoint
            .last()
            .is_none_or(|previous| !prefix.starts_with(previous))
        {
            disjoint.push(prefix);
        }
    }
    Ok(disjoint)
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

#[cfg(test)]
mod tests {
    #[test]
    fn extracted_ranges_keep_alternation_optional_prefixes_and_unicode() {
        // The prior planner always scanned the entire field, even for these
        // selective, syntactically proven ranges.
        let ranges = |source| super::literal_prefixes(source).unwrap();
        assert_eq!(ranges("colou?r"), [b"color".to_vec(), b"colour".to_vec()]);
        assert_eq!(
            ranges("(www|http|https)"),
            [b"http".to_vec(), b"www".to_vec()]
        );
        assert_eq!(ranges("(alpha|alphabet).*"), [b"alpha".to_vec()]);
        assert_eq!(ranges("a?b"), [b"ab".to_vec(), b"b".to_vec()]);
        assert_eq!(ranges(".*tion"), [Vec::<u8>::new()]);
        assert_eq!(
            ranges("(é|🦀)x"),
            ["éx".as_bytes().to_vec(), "🦀x".as_bytes().to_vec()]
        );
        let regex = regex::Regex::new("^(?:[ab]{50})$").unwrap();
        let prefixes = ranges("[ab]{50}");
        assert!(prefixes.len() <= 64);
        for term in ["a".repeat(50), "b".repeat(50), "ab".repeat(25)] {
            assert!(regex.is_match(&term));
            assert!(
                prefixes
                    .iter()
                    .any(|prefix| term.as_bytes().starts_with(prefix))
            );
        }
    }
}
