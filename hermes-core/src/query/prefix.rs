//! Prefix query — matches all documents containing any term that starts with a
//! given prefix. Materializes the union of matching posting lists into a sorted
//! doc ID set, giving O(log N) seek via `SortedVecDocSet`. Score is always 1.0
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
            let docs = materialize_union(&postings);
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
        let docs = materialize_union(&postings);
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
            Ok(postings.iter().map(|p| p.doc_count()).sum())
        })
    }

    fn is_filter(&self) -> bool {
        true
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

/// Iterate all posting lists, collect doc IDs, sort, and deduplicate.
fn materialize_union(postings: &[BlockPostingList]) -> Vec<u32> {
    let total: usize = postings.iter().map(|p| p.doc_count() as usize).sum();
    let mut docs = Vec::with_capacity(total);

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
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_materialize_union_empty() {
        let docs = materialize_union(&[]);
        assert!(docs.is_empty());
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
