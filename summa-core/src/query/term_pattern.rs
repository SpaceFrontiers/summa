//! Shared bounded dictionary matching for whole-term pattern filters.
use super::term_union::{
    TermUnionScorer, physical_union_field, union_map, validate_expansion_field,
};
use super::traits::{CountFuture, Query, Scorer, ScorerFuture, ScorerOptions};
use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::segment::reader::{ExpandedPosting, TermDictionaryLookup};
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
    matcher: Matcher,
}

#[derive(Debug)]
enum Matcher {
    Regex {
        prefixes: Vec<Vec<u8>>,
        /// Every match ends with one of these; empty when no finite set does.
        /// Rejects most scanned terms before the automaton runs.
        suffixes: Vec<Literal>,
        /// Unicode-mode byte automaton: every accepted construct matches only
        /// UTF-8 sequences, so a whole-term match proves the term is UTF-8
        /// without validating each scanned term first.
        regex: regex::bytes::Regex,
    },
    Exact {
        terms: Vec<Vec<u8>>,
    },
    SingleStar {
        prefix: Literal,
        suffix: Literal,
    },
}

/// A pattern literal with word-sized forms for the per-term checks. Broad
/// scans test every term of a range, and a `memcmp` call per 1–8-byte literal
/// was about a tenth of broad-wildcard CPU.
#[derive(Debug)]
struct Literal {
    bytes: Vec<u8>,
    /// First `min(len, 8)` bytes, little-endian, zero-padded.
    head: u64,
    head_mask: u64,
    /// Last `min(len, 4)` bytes, little-endian.
    tail: u32,
}

impl Literal {
    fn new(bytes: Vec<u8>) -> Self {
        let n = bytes.len().min(8);
        let mut head = [0; 8];
        head[..n].copy_from_slice(&bytes[..n]);
        let t = bytes.len().min(4);
        let mut tail = [0; 4];
        tail[..t].copy_from_slice(&bytes[bytes.len() - t..]);
        Self {
            head: u64::from_le_bytes(head),
            head_mask: u64::MAX.checked_shr(64 - 8 * n as u32).unwrap_or(0),
            tail: u32::from_le_bytes(tail),
            bytes,
        }
    }

    #[inline(always)]
    fn is_prefix_of(&self, term: &[u8]) -> bool {
        match term.first_chunk::<8>() {
            Some(head) if self.bytes.len() <= 8 => {
                u64::from_le_bytes(*head) & self.head_mask == self.head
            }
            _ => term.starts_with(&self.bytes),
        }
    }

    /// Rejects on the last four bytes first: they are almost always inside
    /// the most recent write of a freshly decoded key, so the load forwards
    /// from one store, and nearly every scanned term fails here.
    #[inline(always)]
    fn is_suffix_of(&self, term: &[u8]) -> bool {
        let n = self.bytes.len();
        match term.last_chunk::<4>() {
            Some(last) if n > 0 => {
                let t = n.min(4) as u32;
                u32::from_le_bytes(*last) >> (32 - 8 * t) == self.tail
                    && (n <= 4 || term.ends_with(&self.bytes))
            }
            _ => term.ends_with(&self.bytes),
        }
    }
}

impl TermPatternQuery {
    async fn expand(&self, reader: &SegmentReader) -> Result<Vec<ExpandedPosting>> {
        validate_expansion_field(reader, self.field, self.label)?;
        reader
            .get_matching_postings(
                self.field,
                self.pattern.lookup(),
                self.label,
                MAX_SCANNED_TERMS,
                |term| self.pattern.matches(term),
            )
            .await
    }

    #[cfg(feature = "sync")]
    fn expand_sync(&self, reader: &SegmentReader) -> Result<Vec<ExpandedPosting>> {
        validate_expansion_field(reader, self.field, self.label)?;
        reader.get_matching_postings_sync(
            self.field,
            self.pattern.lookup(),
            self.label,
            MAX_SCANNED_TERMS,
            |term| self.pattern.matches(term),
        )
    }

    pub(super) fn compile(
        field: Field,
        source: &str,
        expression: &str,
        prefixes: Vec<Vec<u8>>,
        label: &'static str,
    ) -> Result<Self> {
        check_length(source, label)?;
        let invalid = |error: &dyn std::fmt::Display| {
            Error::Query(format!("invalid {label} pattern: {error}"))
        };
        let regex = regex::bytes::RegexBuilder::new(&format!("\\A(?:{expression})\\z"))
            .dot_matches_new_line(true)
            .size_limit(MAX_REGEX_BYTES)
            .dfa_size_limit(MAX_REGEX_BYTES)
            .build()
            .map_err(|error| invalid(&error))?;
        let hir = regex_syntax::ParserBuilder::new()
            .dot_matches_new_line(true)
            .build()
            .parse(expression)
            .map_err(|error| invalid(&error))?;
        let suffixes = regex_syntax::hir::literal::Extractor::new()
            .kind(regex_syntax::hir::literal::ExtractKind::Suffix)
            .limit_total(16)
            .extract(&hir);
        let suffixes = match suffixes.literals() {
            Some(literals) if literals.iter().all(|literal| !literal.is_empty()) => literals
                .iter()
                .map(|literal| Literal::new(literal.as_bytes().to_vec()))
                .collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            field,
            pattern: Arc::new(Pattern {
                source: source.to_owned(),
                matcher: Matcher::Regex {
                    prefixes,
                    suffixes,
                    regex,
                },
            }),
            label,
        })
    }

    pub(super) fn exact(field: Field, source: &str, terms: Vec<Vec<u8>>) -> Self {
        Self {
            field,
            pattern: Arc::new(Pattern {
                source: source.to_owned(),
                matcher: Matcher::Exact { terms },
            }),
            label: "regex",
        }
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
                matcher: Matcher::SingleStar {
                    prefix: Literal::new(prefix),
                    suffix: Literal::new(suffix),
                },
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
    fn lookup(&self) -> TermDictionaryLookup<'_> {
        match &self.matcher {
            Matcher::Exact { terms } => TermDictionaryLookup::Exact(terms),
            Matcher::Regex { prefixes, .. } => TermDictionaryLookup::prefixes(prefixes),
            Matcher::SingleStar { prefix, suffix } => TermDictionaryLookup::Ranges {
                prefixes: std::slice::from_ref(&prefix.bytes),
                ending: suffix.bytes.last_chunk::<4>().copied(),
            },
        }
    }

    fn matches(&self, term: &[u8]) -> bool {
        match &self.matcher {
            Matcher::Exact { terms } => terms
                .binary_search_by(|key| key.as_slice().cmp(term))
                .is_ok(),
            Matcher::Regex {
                suffixes, regex, ..
            } => {
                (suffixes.is_empty() || suffixes.iter().any(|suffix| suffix.is_suffix_of(term)))
                    && regex.is_match(term)
            }
            Matcher::SingleStar { prefix, suffix } => {
                term.len() >= prefix.bytes.len() + suffix.bytes.len()
                    && suffix.is_suffix_of(term)
                    && prefix.is_prefix_of(term)
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
        self.scorer_with_options(reader, limit, ScorerOptions::default())
    }

    fn scorer_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: ScorerOptions,
    ) -> ScorerFuture<'a> {
        let query = self.clone();
        Box::pin(async move {
            let postings = query.expand(reader).await?;
            Ok(Box::new(TermUnionScorer::from_expanded(
                postings,
                reader.num_docs(),
                union_map(reader, query.field, &options),
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
        self.scorer_sync_with_options(reader, limit, ScorerOptions::default())
    }

    #[cfg(feature = "sync")]
    fn scorer_sync_with_options<'a>(
        &self,
        reader: &'a SegmentReader,
        limit: usize,
        options: ScorerOptions,
    ) -> Result<Box<dyn Scorer + 'a>> {
        let postings = self.expand_sync(reader)?;
        Ok(Box::new(TermUnionScorer::from_expanded(
            postings,
            reader.num_docs(),
            union_map(reader, self.field, &options),
            limit,
        )))
    }

    fn physical_text_field(&self, reader: &SegmentReader, _complete: bool) -> Option<Field> {
        physical_union_field(reader, self.field)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let query = self.clone();
        Box::pin(async move {
            let postings = query.expand(reader).await?;
            Ok(postings
                .iter()
                .fold(0u32, |sum, posting| sum.saturating_add(posting.doc_count()))
                .min(reader.num_docs()))
        })
    }

    fn is_filter(&self) -> bool {
        true
    }

    fn exact_count<'a>(&self, reader: &'a SegmentReader) -> Option<CountFuture<'a>> {
        let query = self.clone();
        Some(Box::pin(async move {
            let postings = query.expand(reader).await?;
            Ok(super::term_union::count_expanded(
                postings,
                reader.num_docs(),
            ))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts of term expansions on an RGB-reordered field traverse physical
    /// IDs: the union is built without the document map, so dense postings
    /// fill whole bitset windows. Ranked (incomplete) streams stay logical,
    /// keeping stable-ID tie order.
    #[tokio::test]
    async fn expanded_term_counts_on_rgb_fields_traverse_physical_ids() {
        use crate::query::{
            BooleanQuery, CountCollector, PrefixQuery, RegexQuery, ScorerOptions, TermQuery,
            WildcardQuery, collect_segment,
        };
        use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, SchemaBuilder};
        let mut schema = SchemaBuilder::default();
        let field = schema.add_text_field("text", true, false);
        schema.set_reorder(field, true);
        let dir = RamDirectory::new();
        let config = IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
            .await
            .unwrap();
        let text = |id: u32| match id % 3 {
            0 => "alpha beta gamma",
            1 => "alpine omega",
            _ => "lambda beta",
        };
        for id in 0..3000 {
            let mut doc = Document::new();
            doc.add_text(field, text(id));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.reorder().await.unwrap();
        writer.shutdown().await.unwrap();
        let index = Index::open(dir, config).await.unwrap();
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let segment = &searcher.segment_readers()[0];
        let map = segment.chunk_map(field).unwrap();
        assert!(map.is_document_map());

        let logical: Vec<u32> = (0..3000).filter(|id| id % 3 != 2).collect();
        let mut physical: Vec<u32> = logical
            .iter()
            .map(|&doc| map.document_slot(doc).unwrap())
            .collect();
        physical.sort_unstable();
        assert_ne!(
            physical, logical,
            "the reorder must permute these documents"
        );
        let queries: Vec<Box<dyn Query>> = vec![
            Box::new(PrefixQuery::text(field, "alp")),
            Box::new(WildcardQuery::new(field, "al*").unwrap()),
            Box::new(RegexQuery::new(field, "al(ph|pin)[ae]").unwrap()),
        ];
        for query in &queries {
            assert_eq!(
                query.physical_text_field(segment, true),
                Some(field),
                "{query}"
            );
            // Constant-score: ranked streams go physical too.
            assert_eq!(
                query.physical_text_field(segment, false),
                Some(field),
                "{query}"
            );
            let options = ScorerOptions {
                physical_text_field: Some(field),
                ..Default::default()
            };
            let mut scorer = query
                .scorer_with_options(segment, usize::MAX / 2, options.clone())
                .await
                .unwrap();
            let mut docs = Vec::new();
            while scorer.doc() != crate::structures::TERMINATED {
                docs.push(scorer.doc());
                scorer.advance();
            }
            assert_eq!(docs, physical, "{query}");
            #[cfg(feature = "sync")]
            {
                let mut scorer = query
                    .scorer_sync_with_options(segment, usize::MAX / 2, options)
                    .unwrap();
                let mut docs = Vec::new();
                while scorer.doc() != crate::structures::TERMINATED {
                    docs.push(scorer.doc());
                    scorer.advance();
                }
                assert_eq!(docs, physical, "{query} (sync)");
            }
            let mut count = CountCollector::new();
            collect_segment(segment, query.as_ref(), &mut count)
                .await
                .unwrap();
            assert_eq!(count.count(), 2000, "{query}");
        }
        // A Boolean wrapping only an expansion is constant-score and ranks
        // physically; one mixing in a scored term keeps logical ties, which a
        // ranked physical stream could only honour by visiting every match.
        let wrapped = BooleanQuery::new().should(PrefixQuery::text(field, "alp"));
        assert_eq!(wrapped.physical_text_field(segment, false), Some(field));
        let mixed = BooleanQuery::new()
            .must(PrefixQuery::text(field, "alp"))
            .must(TermQuery::text(field, "beta"));
        assert_eq!(mixed.physical_text_field(segment, false), None);
        assert_eq!(mixed.physical_text_field(segment, true), Some(field));
        let nested = BooleanQuery::new()
            .must(PrefixQuery::text(field, "alp"))
            .must(TermQuery::text(field, "beta"));
        let mut count = CountCollector::new();
        collect_segment(segment, &nested, &mut count).await.unwrap();
        assert_eq!(count.count(), 1000);
    }

    #[test]
    fn compiled_patterns_match_like_utf8_regex_and_reject_invalid_terms() {
        let alphabet = ["a", "b", "c", "e", "s", "é", "🦀", "\n"];
        let mut terms: Vec<Vec<u8>> = vec![Vec::new()];
        let mut frontier = terms.clone();
        for _ in 0..4 {
            frontier = frontier
                .iter()
                .flat_map(|prefix| {
                    alphabet
                        .iter()
                        .map(move |ch| [prefix.as_slice(), ch.as_bytes()].concat())
                })
                .collect();
            terms.extend(frontier.iter().cloned());
        }
        let invalid: Vec<Vec<u8>> = terms
            .iter()
            .step_by(7)
            .flat_map(|term| {
                [
                    [term.as_slice(), &[0xff]].concat(),
                    [&[0xc3], term.as_slice()].concat(),
                    [term.as_slice(), &[0xc3], b"ess"].concat(),
                ]
            })
            .collect();
        terms.extend(invalid);
        for source in [
            "[jkqxz][a-z]*ess",
            "[a-c]*ess",
            "(a|b)*s",
            ".*",
            "[^a]*",
            "ab(c)?",
            "(es|é)+",
            ".*🦀",
            "a.b",
            "(ess|e)",
        ] {
            let query =
                TermPatternQuery::compile(Field(0), source, source, vec![Vec::new()], "regex")
                    .unwrap();
            let reference = regex::RegexBuilder::new(&format!(r"\A(?:{source})\z"))
                .dot_matches_new_line(true)
                .build()
                .unwrap();
            for term in &terms {
                let expected = std::str::from_utf8(term).is_ok_and(|term| reference.is_match(term));
                assert_eq!(query.pattern.matches(term), expected, "{source} {term:?}");
            }
        }
    }

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

    #[test]
    fn word_literal_checks_equal_slice_checks_across_the_eight_byte_boundary() {
        let words = |len: usize| {
            (0..1u32 << len).map(move |bits| {
                (0..len)
                    .map(|i| if bits >> i & 1 == 1 { b'b' } else { b'a' })
                    .collect::<Vec<u8>>()
            })
        };
        for literal_len in 0..=10 {
            let stride = if literal_len <= 6 { 1 } else { 97 };
            for literal in words(literal_len).step_by(stride) {
                let checked = Literal::new(literal.clone());
                for term in (0..=12).flat_map(words) {
                    assert_eq!(checked.is_prefix_of(&term), term.starts_with(&literal));
                    assert_eq!(checked.is_suffix_of(&term), term.ends_with(&literal));
                }
            }
        }
    }
}
