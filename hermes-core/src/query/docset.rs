//! DocSet trait and concrete implementations for document iteration.
//!
//! `DocSet` is the base abstraction for forward-only cursors over sorted document IDs.
//! Posting lists, filter results, and scorers all implement this trait.
//! `IntersectionScorer` intersects a Scorer with a DocSet filter, driving from
//! the smaller side by `size_hint`.

use std::sync::Arc;

use crate::DocId;
use crate::structures::TERMINATED;

// ── DocSet trait ─────────────────────────────────────────────────────────

macro_rules! define_docset_trait {
    ($($send_bounds:tt)*) => {
        /// Forward-only cursor over sorted document IDs.
        ///
        /// This is the base iteration abstraction. Posting lists, filter cursors,
        /// and scorers all implement this trait.
        pub trait DocSet: $($send_bounds)* {
            /// Current document ID, or [`TERMINATED`] if exhausted.
            fn doc(&self) -> DocId;

            /// Advance to the next document. Returns the new doc ID or [`TERMINATED`].
            fn advance(&mut self) -> DocId;

            /// Seek to the first document >= `target`. Returns doc ID or [`TERMINATED`].
            fn seek(&mut self, target: DocId) -> DocId {
                let mut doc = self.doc();
                while doc < target {
                    doc = self.advance();
                }
                doc
            }

            /// Estimated number of remaining documents.
            fn size_hint(&self) -> u32;
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
define_docset_trait!(Send + Sync);

#[cfg(target_arch = "wasm32")]
define_docset_trait!();

// ── DocSet for Box<dyn DocSet> ───────────────────────────────────────────

impl DocSet for Box<dyn DocSet + '_> {
    #[inline]
    fn doc(&self) -> DocId {
        (**self).doc()
    }
    #[inline]
    fn advance(&mut self) -> DocId {
        (**self).advance()
    }
    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        (**self).seek(target)
    }
    #[inline]
    fn size_hint(&self) -> u32 {
        (**self).size_hint()
    }
}

// ── SortedVecDocSet ──────────────────────────────────────────────────────

/// DocSet backed by a sorted `Vec<u32>`. Binary search for seek.
pub struct SortedVecDocSet {
    docs: Arc<Vec<u32>>,
    pos: usize,
}

impl SortedVecDocSet {
    pub fn new(docs: Arc<Vec<u32>>) -> Self {
        Self { docs, pos: 0 }
    }
}

impl DocSet for SortedVecDocSet {
    #[inline]
    fn doc(&self) -> DocId {
        self.docs.get(self.pos).copied().unwrap_or(TERMINATED)
    }

    #[inline]
    fn advance(&mut self) -> DocId {
        if self.pos < self.docs.len() {
            self.pos += 1;
        }
        self.doc()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.pos >= self.docs.len() {
            return TERMINATED;
        }
        let remaining = &self.docs[self.pos..];
        match remaining.binary_search(&target) {
            Ok(offset) => {
                self.pos += offset;
                self.docs[self.pos]
            }
            Err(offset) => {
                self.pos += offset;
                self.doc()
            }
        }
    }

    fn size_hint(&self) -> u32 {
        self.docs.len().saturating_sub(self.pos) as u32
    }
}

// ── IntersectionDocSet ───────────────────────────────────────────────────

/// DocSet that yields the intersection of two DocSets.
pub struct IntersectionDocSet<A: DocSet, B: DocSet> {
    a: A,
    b: B,
}

impl<A: DocSet, B: DocSet> IntersectionDocSet<A, B> {
    pub fn new(mut a: A, mut b: B) -> Self {
        // Align both on the first common doc
        let mut da = a.doc();
        let mut db = b.doc();
        loop {
            if da == TERMINATED || db == TERMINATED {
                break;
            }
            if da == db {
                break;
            }
            if da < db {
                da = a.seek(db);
            } else {
                db = b.seek(da);
            }
        }
        Self { a, b }
    }
}

impl<A: DocSet, B: DocSet> DocSet for IntersectionDocSet<A, B> {
    fn doc(&self) -> DocId {
        let da = self.a.doc();
        if da == TERMINATED || self.b.doc() == TERMINATED {
            TERMINATED
        } else {
            da
        }
    }

    fn advance(&mut self) -> DocId {
        let mut da = self.a.advance();
        let mut db = self.b.doc();
        loop {
            if da == TERMINATED || db == TERMINATED {
                return TERMINATED;
            }
            if da == db {
                return da;
            }
            if da < db {
                da = self.a.seek(db);
            } else {
                db = self.b.seek(da);
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let mut da = self.a.seek(target);
        let mut db = self.b.seek(target);
        loop {
            if da == TERMINATED || db == TERMINATED {
                return TERMINATED;
            }
            if da == db {
                return da;
            }
            if da < db {
                da = self.a.seek(db);
            } else {
                db = self.b.seek(da);
            }
        }
    }

    fn size_hint(&self) -> u32 {
        self.a.size_hint().min(self.b.size_hint())
    }
}

// ── AllDocSet ────────────────────────────────────────────────────────────

/// DocSet that yields all documents 0..num_docs.
pub struct AllDocSet {
    current: u32,
    num_docs: u32,
}

impl AllDocSet {
    pub fn new(num_docs: u32) -> Self {
        Self {
            current: 0,
            num_docs,
        }
    }
}

impl DocSet for AllDocSet {
    #[inline]
    fn doc(&self) -> DocId {
        if self.current >= self.num_docs {
            TERMINATED
        } else {
            self.current
        }
    }

    #[inline]
    fn advance(&mut self) -> DocId {
        self.current += 1;
        self.doc()
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        self.current = target;
        self.doc()
    }

    fn size_hint(&self) -> u32 {
        self.num_docs.saturating_sub(self.current)
    }
}

// ── EmptyDocSet ──────────────────────────────────────────────────────────

/// DocSet that is always empty.
pub struct EmptyDocSet;

impl DocSet for EmptyDocSet {
    #[inline]
    fn doc(&self) -> DocId {
        TERMINATED
    }
    #[inline]
    fn advance(&mut self) -> DocId {
        TERMINATED
    }
    #[inline]
    fn seek(&mut self, _target: DocId) -> DocId {
        TERMINATED
    }
    fn size_hint(&self) -> u32 {
        0
    }
}

// ── IntersectionScorer ───────────────────────────────────────────────────

/// Intersects a Scorer with a filter DocSet, driving from the smaller side.
///
/// This is the core composition primitive: filter queries create a filter DocSet,
/// get the inner Scorer, and return `IntersectionScorer(scorer, filter)`.
pub struct IntersectionScorer<'a> {
    scorer: Box<dyn super::Scorer + 'a>,
    filter: Box<dyn DocSet + 'a>,
}

impl<'a> IntersectionScorer<'a> {
    pub fn new(mut scorer: Box<dyn super::Scorer + 'a>, mut filter: Box<dyn DocSet + 'a>) -> Self {
        // Align both on first common doc
        let mut ds = scorer.doc();
        let mut df = filter.doc();
        loop {
            if ds == TERMINATED || df == TERMINATED {
                break;
            }
            if ds == df {
                break;
            }
            if ds < df {
                ds = scorer.seek(df);
            } else {
                df = filter.seek(ds);
            }
        }
        Self { scorer, filter }
    }
}

impl DocSet for IntersectionScorer<'_> {
    fn doc(&self) -> DocId {
        let ds = self.scorer.doc();
        if ds == TERMINATED || self.filter.doc() == TERMINATED {
            TERMINATED
        } else {
            ds
        }
    }

    fn advance(&mut self) -> DocId {
        // Drive from the smaller side
        let filter_smaller = self.filter.size_hint() < self.scorer.size_hint();

        if filter_smaller {
            // Filter drives
            let mut df = self.filter.advance();
            let mut ds = self.scorer.doc();
            loop {
                if df == TERMINATED || ds == TERMINATED {
                    return TERMINATED;
                }
                if df == ds {
                    return df;
                }
                if df < ds {
                    df = self.filter.seek(ds);
                } else {
                    ds = self.scorer.seek(df);
                }
            }
        } else {
            // Scorer drives
            let mut ds = self.scorer.advance();
            let mut df = self.filter.doc();
            loop {
                if ds == TERMINATED || df == TERMINATED {
                    return TERMINATED;
                }
                if ds == df {
                    return ds;
                }
                if ds < df {
                    ds = self.scorer.seek(df);
                } else {
                    df = self.filter.seek(ds);
                }
            }
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        let mut ds = self.scorer.seek(target);
        let mut df = self.filter.seek(target);
        loop {
            if ds == TERMINATED || df == TERMINATED {
                return TERMINATED;
            }
            if ds == df {
                return ds;
            }
            if ds < df {
                ds = self.scorer.seek(df);
            } else {
                df = self.filter.seek(ds);
            }
        }
    }

    fn size_hint(&self) -> u32 {
        self.scorer.size_hint().min(self.filter.size_hint())
    }
}

impl super::Scorer for IntersectionScorer<'_> {
    fn score(&self) -> crate::Score {
        self.scorer.score()
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        self.scorer.matched_positions()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_vec_docset_basic() {
        let docs = Arc::new(vec![1, 3, 5, 7, 9]);
        let mut ds = SortedVecDocSet::new(docs);

        assert_eq!(ds.doc(), 1);
        assert_eq!(ds.advance(), 3);
        assert_eq!(ds.advance(), 5);
        assert_eq!(ds.seek(7), 7);
        assert_eq!(ds.advance(), 9);
        assert_eq!(ds.advance(), TERMINATED);
        assert_eq!(ds.doc(), TERMINATED);
    }

    #[test]
    fn test_sorted_vec_docset_seek_past() {
        let docs = Arc::new(vec![1, 5, 10, 20]);
        let mut ds = SortedVecDocSet::new(docs);

        assert_eq!(ds.seek(3), 5);
        assert_eq!(ds.seek(15), 20);
        assert_eq!(ds.seek(21), TERMINATED);
    }

    #[test]
    fn test_sorted_vec_docset_empty() {
        let docs = Arc::new(vec![]);
        let ds = SortedVecDocSet::new(docs);
        assert_eq!(ds.doc(), TERMINATED);
    }

    #[test]
    fn test_all_docset() {
        let mut ds = AllDocSet::new(3);
        assert_eq!(ds.doc(), 0);
        assert_eq!(ds.advance(), 1);
        assert_eq!(ds.advance(), 2);
        assert_eq!(ds.advance(), TERMINATED);
    }

    #[test]
    fn test_all_docset_seek() {
        let mut ds = AllDocSet::new(10);
        assert_eq!(ds.seek(5), 5);
        assert_eq!(ds.seek(9), 9);
        assert_eq!(ds.seek(10), TERMINATED);
    }

    #[test]
    fn test_empty_docset() {
        let mut ds = EmptyDocSet;
        assert_eq!(ds.doc(), TERMINATED);
        assert_eq!(ds.advance(), TERMINATED);
        assert_eq!(ds.seek(5), TERMINATED);
        assert_eq!(ds.size_hint(), 0);
    }

    #[test]
    fn test_intersection_docset() {
        let a = SortedVecDocSet::new(Arc::new(vec![1, 3, 5, 7, 9]));
        let b = SortedVecDocSet::new(Arc::new(vec![2, 3, 5, 8, 9, 10]));
        let mut isect = IntersectionDocSet::new(a, b);

        assert_eq!(isect.doc(), 3);
        assert_eq!(isect.advance(), 5);
        assert_eq!(isect.advance(), 9);
        assert_eq!(isect.advance(), TERMINATED);
    }

    #[test]
    fn test_intersection_docset_empty() {
        let a = SortedVecDocSet::new(Arc::new(vec![1, 3, 5]));
        let b = SortedVecDocSet::new(Arc::new(vec![2, 4, 6]));
        let isect = IntersectionDocSet::new(a, b);
        assert_eq!(isect.doc(), TERMINATED);
    }

    #[test]
    fn test_intersection_docset_seek() {
        let a = SortedVecDocSet::new(Arc::new(vec![1, 5, 10, 20, 30]));
        let b = SortedVecDocSet::new(Arc::new(vec![5, 10, 15, 20, 25, 30]));
        let mut isect = IntersectionDocSet::new(a, b);

        assert_eq!(isect.doc(), 5);
        assert_eq!(isect.seek(15), 20);
        assert_eq!(isect.advance(), 30);
        assert_eq!(isect.advance(), TERMINATED);
    }

    #[test]
    fn test_size_hint() {
        let docs = Arc::new(vec![1, 2, 3, 4, 5]);
        let mut ds = SortedVecDocSet::new(docs);
        assert_eq!(ds.size_hint(), 5);
        ds.advance();
        assert_eq!(ds.size_hint(), 4);
        ds.seek(4);
        assert_eq!(ds.size_hint(), 2); // pos=3, remaining: [4, 5]
    }
}
