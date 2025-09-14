//! DocSet trait and concrete implementations for document iteration.
//!
//! `DocSet` is the base abstraction for forward-only cursors over sorted document IDs.
//! Posting lists, filter results, and scorers all implement this trait.
//! `PredicatedScorer` wraps a driving Scorer with pushed-down filter predicates.

use std::sync::Arc;

use crate::DocId;
use crate::structures::TERMINATED;

/// Bounded membership scratch for score-free collection, independent of hit count.
pub const DOC_WINDOW_WORDS: usize = 64;
pub const DOC_WINDOW_SIZE: u32 = DOC_WINDOW_WORDS as u32 * 64;
pub type DocWindow = [u64; DOC_WINDOW_WORDS];

/// Compact score-free membership scratch for sparse candidate streams.
pub const DOC_BATCH_SIZE: usize = 128;
pub type DocBatch = [DocId; DOC_BATCH_SIZE];

pub(super) fn fill_batch<D: DocSet + ?Sized>(cursor: &mut D, docs: &mut DocBatch) -> usize {
    let mut count = 0;
    let mut doc = cursor.doc();
    while count < docs.len() && doc != TERMINATED {
        docs[count] = doc;
        count += 1;
        doc = cursor.advance();
    }
    count
}

pub(super) fn retain_batch<D: DocSet + ?Sized>(
    cursor: &mut D,
    docs: &mut DocBatch,
    len: usize,
) -> usize {
    assert!(len <= docs.len());
    let mut kept = 0;
    for i in 0..len {
        let doc = docs[i];
        if cursor.seek(doc) == doc {
            docs[kept] = doc;
            kept += 1;
        }
    }
    kept
}

pub(super) fn fill_window<D: DocSet + ?Sized>(docs: &mut D, base: DocId, bits: &mut DocWindow) {
    bits.fill(0);
    let end = base.saturating_add(DOC_WINDOW_SIZE);
    let mut doc = docs.seek(base);
    while doc < end {
        let offset = (doc - base) as usize;
        bits[offset / 64] |= 1u64 << (offset % 64);
        doc = docs.advance();
    }
}

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

            /// Whether compact membership batches amortize this cursor's work.
            fn supports_doc_batches(&self) -> bool { false }

            /// Consume up to 128 exact sorted matches starting at doc(), leaving
            /// the cursor on the first unconsumed match. Zero means exhausted.
            fn fill_doc_batch(&mut self, docs: &mut DocBatch) -> usize {
                fill_batch(self, docs)
            }

            /// Retain matches from the sorted unique prefix docs[..len]. The
            /// cursor remains at or beyond the last probe; earlier IDs stay consumed.
            fn retain_doc_batch(&mut self, docs: &mut DocBatch, len: usize) -> usize {
                retain_batch(self, docs, len)
            }

            /// Whether this cursor benefits from bounded score-free membership batches.
            fn supports_doc_windows(&self) -> bool { false }

            /// Consume exact matches in `[base, base + DOC_WINDOW_SIZE)`, replacing
            /// `bits`, and leave the cursor on its first match at or after the end.
            /// Calls are forward-only; previously consumed documents stay consumed.
            fn fill_doc_window(&mut self, base: DocId, bits: &mut DocWindow) {
                fill_window(self, base, bits);
            }

        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
define_docset_trait!(Send + Sync);

#[cfg(target_arch = "wasm32")]
define_docset_trait!();

// ── DocSet for Box<dyn DocSet> ───────────────────────────────────────────

impl DocSet for Box<dyn DocSet + '_> {
    fn supports_doc_batches(&self) -> bool {
        (**self).supports_doc_batches()
    }
    fn fill_doc_batch(&mut self, docs: &mut DocBatch) -> usize {
        (**self).fill_doc_batch(docs)
    }
    fn retain_doc_batch(&mut self, docs: &mut DocBatch, len: usize) -> usize {
        (**self).retain_doc_batch(docs, len)
    }
    fn supports_doc_windows(&self) -> bool {
        (**self).supports_doc_windows()
    }
    fn fill_doc_window(&mut self, base: DocId, bits: &mut DocWindow) {
        (**self).fill_doc_window(base, bits);
    }
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
        if self.current < self.num_docs {
            self.current += 1;
        }
        self.doc()
    }

    #[inline]
    fn seek(&mut self, target: DocId) -> DocId {
        self.current = self.current.max(target);
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

// ── PredicatedScorer ─────────────────────────────────────────────────────

/// Wraps a driving Scorer with filter conditions pushed down.
///
/// Used by the query planner to flip iteration order: the SHOULD scorer
/// drives and MUST/MUST_NOT clauses are checked per-doc via:
/// - O(1) predicate closures (e.g., fast-field range checks)
/// - seek()-based verifier scorers (e.g., TermQuery posting list lookups)
///
/// Verifier scorers contribute their actual per-doc score (e.g., BM25).
pub struct PredicatedScorer<'a> {
    /// Driving scorer (typically SHOULD clauses)
    driver: Box<dyn super::Scorer + 'a>,
    /// O(1) predicate checks (from filter queries like RangeQuery, or negated MUST_NOT)
    predicates: Vec<super::DocPredicate<'a>>,
    /// MUST scorers verified via seek() — preserves per-doc scoring
    must_verifiers: Vec<Box<dyn super::Scorer + 'a>>,
    /// MUST_NOT scorers — docs are excluded if these land on them
    must_not_verifiers: Vec<Box<dyn super::Scorer + 'a>>,
}

impl<'a> PredicatedScorer<'a> {
    pub fn new(
        driver: Box<dyn super::Scorer + 'a>,
        predicates: Vec<super::DocPredicate<'a>>,
        must_verifiers: Vec<Box<dyn super::Scorer + 'a>>,
        must_not_verifiers: Vec<Box<dyn super::Scorer + 'a>>,
    ) -> Self {
        let mut s = Self {
            driver,
            predicates,
            must_verifiers,
            must_not_verifiers,
        };
        // Position on first matching doc
        s.skip_non_matching();
        s
    }

    /// Check whether `doc` passes all filter conditions.
    #[inline]
    fn check_filters(&mut self, doc: DocId) -> bool {
        // O(1) predicate checks first (cheapest)
        if !self.predicates.iter().all(|p| p(doc)) {
            return false;
        }
        // MUST verifiers: seek to doc, must land exactly on it
        if !self.must_verifiers.iter_mut().all(|s| s.seek(doc) == doc) {
            return false;
        }
        // MUST_NOT verifiers: seek to doc, must NOT land on it
        self.must_not_verifiers
            .iter_mut()
            .all(|s| s.seek(doc) != doc)
    }

    /// Filter the driver's batch before moving verifiers to the next window.
    /// MUST scores retain the same left-to-right sum as scalar `score()`.
    fn filter_window(
        &mut self,
        base: DocId,
        bits: &mut DocWindow,
        mut scores: Option<&mut [crate::Score; DOC_WINDOW_SIZE as usize]>,
    ) {
        for (word_index, word) in bits.iter_mut().enumerate() {
            let mut remaining = *word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let offset = word_index * 64 + bit;
                if self.check_filters(base + offset as u32) {
                    if let Some(scores) = scores.as_deref_mut() {
                        for verifier in &self.must_verifiers {
                            scores[offset] += verifier.score();
                        }
                    }
                } else {
                    *word &= !(1u64 << bit);
                }
                remaining &= remaining - 1;
            }
        }
        self.skip_non_matching();
    }

    /// Advance driver past non-matching docs.
    fn skip_non_matching(&mut self) -> DocId {
        let mut doc = self.driver.doc();
        while doc != TERMINATED && !self.check_filters(doc) {
            doc = self.driver.advance();
        }
        doc
    }
}

impl DocSet for PredicatedScorer<'_> {
    fn supports_doc_windows(&self) -> bool {
        self.driver.supports_filtered_windows() && self.driver.supports_doc_windows()
    }

    fn fill_doc_window(&mut self, base: DocId, bits: &mut DocWindow) {
        self.driver.fill_doc_window(base, bits);
        self.filter_window(base, bits, None);
    }

    fn doc(&self) -> DocId {
        self.driver.doc()
    }

    fn advance(&mut self) -> DocId {
        self.driver.advance();
        self.skip_non_matching()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.driver.seek(target);
        self.skip_non_matching()
    }

    fn size_hint(&self) -> u32 {
        self.driver.size_hint()
    }
}

impl super::Scorer for PredicatedScorer<'_> {
    fn supports_filtered_windows(&self) -> bool {
        self.driver.supports_filtered_windows()
    }

    fn supports_score_windows(&self) -> bool {
        self.driver.supports_filtered_windows() && self.driver.supports_score_windows()
    }

    fn fill_score_window(
        &mut self,
        base: DocId,
        scores: &mut [crate::Score; DOC_WINDOW_SIZE as usize],
        bits: &mut DocWindow,
    ) {
        self.driver.fill_score_window(base, scores, bits);
        self.filter_window(base, bits, Some(scores));
    }

    fn score(&self) -> crate::Score {
        let mut total = self.driver.score();
        for v in &self.must_verifiers {
            total += v.score();
        }
        total
    }

    fn matched_positions(&self) -> Option<super::MatchedPositions> {
        let mut all: super::MatchedPositions = Vec::new();
        if let Some(p) = self.driver.matched_positions() {
            all.extend(p);
        }
        for v in &self.must_verifiers {
            if let Some(p) = v.matched_positions() {
                all.extend(p);
            }
        }
        if all.is_empty() { None } else { Some(all) }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct WindowScorer(SortedVecDocSet, f32, bool);

    impl DocSet for WindowScorer {
        fn doc(&self) -> DocId {
            self.0.doc()
        }
        fn advance(&mut self) -> DocId {
            self.0.advance()
        }
        fn seek(&mut self, target: DocId) -> DocId {
            self.0.seek(target)
        }
        fn size_hint(&self) -> u32 {
            self.0.size_hint()
        }
        fn supports_doc_windows(&self) -> bool {
            true
        }
    }
    impl crate::query::Scorer for WindowScorer {
        fn supports_filtered_windows(&self) -> bool {
            self.2
        }
        fn score(&self) -> f32 {
            self.1 + (self.doc() % 7) as f32 * 0.125
        }
        fn supports_score_windows(&self) -> bool {
            true
        }
    }

    #[test]
    fn filtering_keeps_leaf_windows_scalar_and_composite_windows_batched() {
        use crate::query::Scorer;
        for beneficial in [false, true] {
            let driver = WindowScorer(
                SortedVecDocSet::new(Arc::new(vec![1, 2, 3])),
                1.0,
                beneficial,
            );
            assert!(driver.supports_doc_windows() && driver.supports_score_windows());
            let wrapped = PredicatedScorer::new(Box::new(driver), vec![], vec![], vec![]);
            let wrapped = PredicatedScorer::new(Box::new(wrapped), vec![], vec![], vec![]);
            assert_eq!(wrapped.supports_doc_windows(), beneficial);
            assert_eq!(wrapped.supports_score_windows(), beneficial);
        }
    }

    #[test]
    fn filtered_windows_preserve_scores_membership_and_forward_cursor() {
        use crate::query::Scorer;
        let docs = Arc::new(vec![
            0,
            1,
            2,
            63,
            64,
            65,
            4095,
            4096,
            4097,
            8192,
            8193,
            u32::MAX - 2,
            u32::MAX - 1,
        ]);
        let make = |required| {
            let scorer = |docs, score| {
                Box::new(WindowScorer(SortedVecDocSet::new(docs), score, true)) as Box<dyn Scorer>
            };
            PredicatedScorer::new(
                scorer(docs.clone(), 0.1),
                vec![Box::new(|doc| doc != 64 && doc != 8192)],
                if required {
                    vec![
                        scorer(
                            Arc::new(docs.iter().copied().filter(|doc| *doc != 4096).collect()),
                            0.3,
                        ),
                        scorer(docs.clone(), 0.7),
                    ]
                } else {
                    vec![]
                },
                vec![scorer(Arc::new(vec![1, 65, 4097]), 0.0)],
            )
        };
        for required in [false, true] {
            let mut scalar = make(required);
            let mut expected = Vec::new();
            while scalar.doc() != TERMINATED {
                expected.push((scalar.doc(), scalar.score().to_bits()));
                scalar.advance();
            }
            for scored in [false, true] {
                let mut batched = make(required);
                assert!(batched.supports_doc_windows());
                assert!(batched.supports_score_windows());
                let mut scores = [f32::NAN; DOC_WINDOW_SIZE as usize];
                let mut bits = [u64::MAX; DOC_WINDOW_WORDS];
                let mut actual = Vec::new();
                for base in [0, 0, 4096, 8192, 12288, u32::MAX - 4096, TERMINATED] {
                    if scored {
                        batched.fill_score_window(base, &mut scores, &mut bits);
                    } else {
                        batched.fill_doc_window(base, &mut bits);
                    }
                    for (word, &mask) in bits.iter().enumerate() {
                        let mut mask = mask;
                        while mask != 0 {
                            let offset = word * 64 + mask.trailing_zeros() as usize;
                            let doc = base + offset as u32;
                            actual.push((doc, if scored { scores[offset].to_bits() } else { 0 }));
                            mask &= mask - 1;
                        }
                    }
                    let next = expected
                        .iter()
                        .find(|(doc, _)| *doc >= base.saturating_add(DOC_WINDOW_SIZE));
                    assert_eq!(batched.doc(), next.map_or(TERMINATED, |(doc, _)| *doc));
                    if let Some((_, score)) = next {
                        assert_eq!(batched.score().to_bits(), *score);
                    }
                }
                let expected: Vec<_> = expected
                    .iter()
                    .map(|&(doc, score)| (doc, if scored { score } else { 0 }))
                    .collect();
                assert_eq!(actual, expected);
            }
        }
    }

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
