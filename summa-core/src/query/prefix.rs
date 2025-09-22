//! Prefix query — matches all documents containing any term that starts with a
//! given prefix. Ranked callers merge posting cursors lazily; complete/mapped
//! callers materialize bounded document sets. Score is always 1.0
//! (filter-style, like `RangeQuery`).

use crate::dsl::Field;
use crate::segment::SegmentReader;

#[cfg(test)]
use super::term_union::materialize_union;
use super::term_union::{TermUnionScorer, reject_chunked};
use super::traits::{CountFuture, Query, Scorer, ScorerFuture};

/// Prefix query — matches documents containing any term starting with `prefix`.
#[derive(Debug, Clone)]
pub struct PrefixQuery {
    pub field: Field,
    pub prefix: Vec<u8>,
}

impl std::fmt::Display for PrefixQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prefix({}:\"{}*\")",
            self.field.0,
            String::from_utf8_lossy(&self.prefix)
        )
    }
}

impl PrefixQuery {
    /// Create from raw bytes.
    pub fn new(field: Field, prefix: impl Into<Vec<u8>>) -> Self {
        Self {
            field,
            prefix: prefix.into(),
        }
    }

    /// Create from text — lowercased to match default tokenization.
    pub fn text(field: Field, text: &str) -> Self {
        Self {
            field,
            prefix: text.to_lowercase().into_bytes(),
        }
    }
}

impl Query for PrefixQuery {
    fn scorer<'a>(&self, reader: &'a SegmentReader, limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let prefix = self.prefix.clone();
        Box::pin(async move {
            reject_chunked(reader, field, "PrefixQuery")?;
            let postings = reader.get_prefix_expansion(field, &prefix).await?;
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
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        reject_chunked(reader, self.field, "PrefixQuery")?;
        let postings = reader.get_prefix_expansion_sync(self.field, &self.prefix)?;
        Ok(Box::new(TermUnionScorer::from_expanded(
            postings,
            reader.num_docs(),
            reader.chunk_map(self.field),
            limit,
        )))
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let prefix = self.prefix.clone();
        Box::pin(async move {
            let postings = reader.get_prefix_expansion(field, &prefix).await?;
            Ok(postings
                .iter()
                .fold(0u32, |sum, posting| sum.saturating_add(posting.doc_count()))
                .min(reader.num_docs()))
        })
    }

    fn is_filter(&self) -> bool {
        true
    }

    #[cfg(feature = "sync")]
    fn as_doc_predicate<'a>(&self, reader: &'a SegmentReader) -> Option<super::DocPredicate<'a>> {
        let bitset = self.as_doc_bitset(reader)?;
        Some(Box::new(move |doc_id: crate::DocId| {
            bitset.contains(doc_id)
        }))
    }

    #[cfg(feature = "sync")]
    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        if reader.is_chunked_field(self.field) {
            return None;
        }
        let postings = reader
            .get_prefix_postings_sync(self.field, &self.prefix)
            .ok()?;
        let mut bitset = super::DocBitset::new(reader.num_docs());
        for posting in &postings {
            let mut iter = posting.iterator();
            loop {
                let d = iter.doc();
                if d == crate::structures::TERMINATED {
                    break;
                }
                bitset.set(reader.chunk_map(self.field).map_or(d, |map| map.doc_id(d)));
                iter.advance();
            }
        }
        Some(bitset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::DocSet;
    use crate::structures::{BlockPostingList, TERMINATED};

    #[test]
    fn test_materialize_union_empty() {
        let docs = materialize_union(&[], 0, None);
        assert!(docs.is_empty());
    }

    #[test]
    fn test_materialize_union_deduplicates() {
        let mut left = crate::structures::PostingList::new();
        left.push(1, 1);
        left.push(5, 1);
        left.push(9, 1);
        let mut right = crate::structures::PostingList::new();
        right.push(2, 1);
        right.push(5, 1);
        right.push(10, 1);
        let postings = vec![
            BlockPostingList::from_posting_list(&left).unwrap(),
            BlockPostingList::from_posting_list(&right).unwrap(),
        ];

        assert_eq!(materialize_union(&postings, 11, None), vec![1, 2, 5, 9, 10]);
        // A huge segment with a narrow prefix takes the posting-vector path;
        // it must not allocate a num_docs-sized bitset.
        assert_eq!(
            materialize_union(&postings[..1], 1_000_000_000, None),
            vec![1, 5, 9]
        );
    }

    #[test]
    fn test_prefix_scorer_basic() {
        let mut scorer = TermUnionScorer::new(vec![1, 5, 10, 20]);
        assert_eq!(scorer.doc(), 1);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 5);
        assert_eq!(scorer.seek(10), 10);
        assert_eq!(scorer.advance(), 20);
        assert_eq!(scorer.advance(), TERMINATED);
    }

    #[test]
    fn test_prefix_scorer_seek_past() {
        let mut scorer = TermUnionScorer::new(vec![1, 5, 10, 20]);
        assert_eq!(scorer.seek(7), 10);
        assert_eq!(scorer.seek(100), TERMINATED);
    }

    #[test]
    fn test_prefix_query_display() {
        let q = PrefixQuery::text(Field(0), "abc");
        assert_eq!(format!("{}", q), "Prefix(0:\"abc*\")");
    }

    #[test]
    fn test_prefix_query_is_filter() {
        let q = PrefixQuery::text(Field(0), "test");
        assert!(q.is_filter());
    }
}
