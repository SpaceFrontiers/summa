//! Prefix query — matches all documents containing any term that starts with a
//! given prefix. Materializes the union of matching posting lists into a sorted
//! doc ID set bounded by the segment's document count. Score is always 1.0
//! (filter-style, like `RangeQuery`).

use std::sync::Arc;

use crate::dsl::Field;
use crate::segment::SegmentReader;
use crate::structures::{BlockPostingList, TERMINATED};
use crate::{DocId, Score};

use super::docset::{DocSet, SortedVecDocSet};
use super::traits::{CountFuture, EmptyScorer, Query, Scorer, ScorerFuture};

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
    fn scorer<'a>(&self, reader: &'a SegmentReader, _limit: usize) -> ScorerFuture<'a> {
        let field = self.field;
        let prefix = self.prefix.clone();
        Box::pin(async move {
            let postings = reader.get_prefix_postings(field, &prefix).await?;
            if postings.is_empty() {
                return Ok(Box::new(EmptyScorer) as Box<dyn Scorer>);
            }
            let docs = materialize_union(&postings, reader.num_docs());
            if docs.is_empty() {
                return Ok(Box::new(EmptyScorer) as Box<dyn Scorer>);
            }
            Ok(Box::new(PrefixScorer::new(docs)) as Box<dyn Scorer>)
        })
    }

    #[cfg(feature = "sync")]
    fn scorer_sync<'a>(
        &self,
        reader: &'a SegmentReader,
        _limit: usize,
    ) -> crate::Result<Box<dyn Scorer + 'a>> {
        let postings = reader.get_prefix_postings_sync(self.field, &self.prefix)?;
        if postings.is_empty() {
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer>);
        }
        let docs = materialize_union(&postings, reader.num_docs());
        if docs.is_empty() {
            return Ok(Box::new(EmptyScorer) as Box<dyn Scorer>);
        }
        Ok(Box::new(PrefixScorer::new(docs)) as Box<dyn Scorer>)
    }

    fn count_estimate<'a>(&self, reader: &'a SegmentReader) -> CountFuture<'a> {
        let field = self.field;
        let prefix = self.prefix.clone();
        Box::pin(async move {
            let postings = reader.get_prefix_postings(field, &prefix).await?;
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
        Some(Box::new(move |doc_id: DocId| bitset.contains(doc_id)))
    }

    #[cfg(feature = "sync")]
    fn as_doc_bitset(&self, reader: &SegmentReader) -> Option<super::DocBitset> {
        let postings = reader
            .get_prefix_postings_sync(self.field, &self.prefix)
            .ok()?;
        let mut bitset = super::DocBitset::new(reader.num_docs());
        for posting in &postings {
            let mut iter = posting.iterator();
            loop {
                let d = iter.doc();
                if d == TERMINATED {
                    break;
                }
                bitset.set(d);
                iter.advance();
            }
        }
        Some(bitset)
    }
}

// ── PrefixScorer ────────────────────────────────────────────────────────

/// Scorer backed by a pre-materialized sorted doc ID set.
struct PrefixScorer {
    inner: SortedVecDocSet,
}

impl PrefixScorer {
    fn new(docs: Vec<u32>) -> Self {
        Self {
            inner: SortedVecDocSet::new(Arc::new(docs)),
        }
    }
}

impl DocSet for PrefixScorer {
    #[inline]
    fn doc(&self) -> DocId {
        self.inner.doc()
    }

    #[inline]
    fn advance(&mut self) -> DocId {
        self.inner.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.inner.seek(target)
    }

    fn size_hint(&self) -> u32 {
        self.inner.size_hint()
    }
}

impl Scorer for PrefixScorer {
    fn score(&self) -> Score {
        1.0
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Materialize a posting union using the smaller of two bounded scratch forms.
/// Narrow prefixes append/sort doc IDs; broad, overlapping prefixes use a
/// segment-sized bitset so duplicate postings cannot multiply memory.
fn materialize_union(postings: &[BlockPostingList], num_docs: u32) -> Vec<u32> {
    let posting_count = postings.iter().fold(0usize, |sum, posting| {
        sum.saturating_add(posting.doc_count() as usize)
    });
    let posting_bytes = posting_count.saturating_mul(std::mem::size_of::<u32>());
    let bitset_bytes = (num_docs as usize)
        .div_ceil(64)
        .saturating_mul(std::mem::size_of::<u64>());

    if posting_bytes <= bitset_bytes {
        let mut docs = Vec::with_capacity(posting_count);
        for posting in postings {
            let mut iter = posting.iterator();
            loop {
                let d = iter.doc();
                if d == TERMINATED {
                    break;
                }
                docs.push(d);
                iter.advance();
            }
        }
        docs.sort_unstable();
        docs.dedup();
        return docs;
    }

    let mut bitset = super::DocBitset::new(num_docs);
    for posting in postings {
        let mut iter = posting.iterator();
        loop {
            let d = iter.doc();
            if d == TERMINATED {
                break;
            }
            bitset.set(d);
            iter.advance();
        }
    }

    let mut docs = Vec::with_capacity(bitset.count() as usize);
    for (word_idx, &word) in bitset.bits.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            docs.push((word_idx * 64 + bit) as u32);
            remaining &= remaining - 1;
        }
    }
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_materialize_union_empty() {
        let docs = materialize_union(&[], 0);
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

        assert_eq!(materialize_union(&postings, 11), vec![1, 2, 5, 9, 10]);
        // A huge segment with a narrow prefix takes the posting-vector path;
        // it must not allocate a num_docs-sized bitset.
        assert_eq!(
            materialize_union(&postings[..1], 1_000_000_000),
            vec![1, 5, 9]
        );
    }

    #[test]
    fn test_prefix_scorer_basic() {
        let mut scorer = PrefixScorer::new(vec![1, 5, 10, 20]);
        assert_eq!(scorer.doc(), 1);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 5);
        assert_eq!(scorer.seek(10), 10);
        assert_eq!(scorer.advance(), 20);
        assert_eq!(scorer.advance(), TERMINATED);
    }

    #[test]
    fn test_prefix_scorer_seek_past() {
        let mut scorer = PrefixScorer::new(vec![1, 5, 10, 20]);
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
