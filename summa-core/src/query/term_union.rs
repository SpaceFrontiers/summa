//! Shared constant-score execution for expanded term filters.

use super::Scorer;
use super::docset::{BitsetDocSet, DocSet, SortedVecDocSet};
use crate::segment::reader::ExpandedPosting;
#[cfg(test)]
use crate::structures::BlockPostingList;
use crate::structures::TERMINATED;
use crate::{DocId, Score};
use std::sync::Arc;

/// Complete constant-score membership, with lazy postings for ranked callers.
pub(super) struct TermUnionScorer {
    inner: UnionDocs,
    terminated: bool,
}

enum UnionDocs {
    Materialized(SortedVecDocSet),
    Streaming(PostingUnion),
    Bitmap(BitsetDocSet),
}

struct PostingUnion {
    lists: Vec<UnionCursor>,
    heads: std::collections::BinaryHeap<std::cmp::Reverse<(DocId, usize)>>,
    upper_count: u32,
}

// Most expanded terms never open for a small top-k. Keep their metadata inline
// to avoid an allocation per pending term; only active decode scratch is boxed.
#[allow(clippy::large_enum_variant)]
enum UnionCursor {
    Pending(crate::structures::postings::DeferredPosting),
    Inline(crate::structures::DecodedInlinePostings, usize),
    Active(Box<crate::structures::BlockPostingIterator<'static>>),
    Exhausted,
}

impl UnionCursor {
    fn seek(&mut self, target: DocId) -> DocId {
        if matches!(self, Self::Pending(_)) {
            let Self::Pending(posting) = std::mem::replace(self, Self::Exhausted) else {
                unreachable!()
            };
            let iterator = posting
                .into_list()
                .into_candidate_iterator(target, &mut Default::default());
            *self = Self::Active(Box::new(iterator));
        }
        match self {
            Self::Active(iterator) => iterator.seek(target),
            Self::Inline(postings, pos) => {
                *pos += postings.docs()[*pos..].partition_point(|&doc| doc < target);
                postings.docs().get(*pos).copied().unwrap_or(TERMINATED)
            }
            Self::Exhausted => TERMINATED,
            Self::Pending(_) => unreachable!(),
        }
    }
}

impl PostingUnion {
    fn new(postings: Vec<ExpandedPosting>, num_docs: u32) -> Self {
        let upper_count = postings
            .iter()
            .fold(0u32, |n, list| n.saturating_add(list.doc_count()))
            .min(num_docs);
        let mut lists = Vec::with_capacity(postings.len());
        let mut heads = Vec::with_capacity(postings.len());
        for posting in postings {
            let index = lists.len();
            let (first, cursor) = match posting {
                ExpandedPosting::Inline(postings) => (
                    postings.docs().first().copied(),
                    UnionCursor::Inline(postings, 0),
                ),
                ExpandedPosting::External(posting) => {
                    (posting.first_doc(), UnionCursor::Pending(posting))
                }
            };
            if let Some(first) = first {
                heads.push(std::cmp::Reverse((first, index)));
            }
            lists.push(cursor);
        }
        let mut union = Self {
            lists,
            heads: heads.into(),
            upper_count,
        };
        union.seek(0);
        union
    }

    fn doc(&self) -> DocId {
        self.heads.peek().map_or(TERMINATED, |head| head.0.0)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        while let Some(mut head) = self.heads.peek_mut() {
            let std::cmp::Reverse((doc, index)) = *head;
            // Metadata orders unopened lists, but an actual decoded document
            // must establish the minimum before the collector sees it.
            if doc >= target && !matches!(self.lists[index], UnionCursor::Pending(_)) {
                break;
            }
            let next = self.lists[index].seek(target);
            if next == TERMINATED {
                std::collections::binary_heap::PeekMut::pop(head);
            } else {
                *head = std::cmp::Reverse((next, index));
            }
        }
        self.doc()
    }
}

impl TermUnionScorer {
    #[cfg(test)]
    pub(super) fn new(docs: Vec<u32>) -> Self {
        Self {
            inner: UnionDocs::Materialized(SortedVecDocSet::new(Arc::new(docs))),
            terminated: false,
        }
    }

    #[cfg(test)]
    pub(super) fn from_postings(
        postings: Vec<BlockPostingList>,
        num_docs: u32,
        map: Option<&crate::segment::chunk_map::ChunkMap>,
        limit: usize,
    ) -> Self {
        Self::from_expanded(
            postings
                .into_iter()
                .map(|list| {
                    ExpandedPosting::External(
                        crate::structures::postings::DeferredPosting::from_list(&list),
                    )
                })
                .collect(),
            num_docs,
            map,
            limit,
        )
    }

    pub(super) fn from_expanded(
        postings: Vec<ExpandedPosting>,
        num_docs: u32,
        map: Option<&crate::segment::chunk_map::ChunkMap>,
        limit: usize,
    ) -> Self {
        if map.is_none() && limit > 0 && limit < num_docs as usize {
            Self {
                inner: UnionDocs::Streaming(PostingUnion::new(postings, num_docs)),
                terminated: false,
            }
        } else {
            Self {
                inner: materialize_expanded(postings, num_docs, map),
                terminated: false,
            }
        }
    }
}

impl DocSet for TermUnionScorer {
    fn doc(&self) -> DocId {
        if self.terminated {
            return TERMINATED;
        }
        match &self.inner {
            UnionDocs::Materialized(inner) => inner.doc(),
            UnionDocs::Bitmap(inner) => inner.doc(),
            UnionDocs::Streaming(inner) => inner.doc(),
        }
    }

    fn advance(&mut self) -> DocId {
        if self.terminated {
            return TERMINATED;
        }
        match &mut self.inner {
            UnionDocs::Materialized(inner) => inner.advance(),
            UnionDocs::Bitmap(inner) => inner.advance(),
            UnionDocs::Streaming(inner) => inner.seek(inner.doc().saturating_add(1)),
        }
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.terminated {
            return TERMINATED;
        }
        match &mut self.inner {
            UnionDocs::Materialized(inner) => inner.seek(target),
            UnionDocs::Bitmap(inner) => inner.seek(target),
            UnionDocs::Streaming(inner) => inner.seek(target),
        }
    }

    fn size_hint(&self) -> u32 {
        match &self.inner {
            UnionDocs::Materialized(inner) => inner.size_hint(),
            UnionDocs::Bitmap(inner) => inner.size_hint(),
            UnionDocs::Streaming(inner) => inner.upper_count,
        }
    }

    fn supports_doc_windows(&self) -> bool {
        matches!(self.inner, UnionDocs::Bitmap(_))
    }

    fn fill_doc_window(&mut self, base: DocId, bits: &mut super::docset::DocWindow) {
        if self.terminated {
            bits.fill(0);
            return;
        }
        match &mut self.inner {
            UnionDocs::Bitmap(inner) => inner.fill_doc_window(base, bits),
            _ => super::docset::fill_window(self, base, bits),
        }
    }

    fn supports_doc_batches(&self) -> bool {
        matches!(self.inner, UnionDocs::Materialized(_))
    }

    fn fill_doc_batch(&mut self, docs: &mut super::docset::DocBatch) -> usize {
        if self.terminated {
            return 0;
        }
        match &mut self.inner {
            UnionDocs::Materialized(inner) => inner.fill_doc_batch(docs),
            UnionDocs::Streaming(_) | UnionDocs::Bitmap(_) => super::docset::fill_batch(self, docs),
        }
    }
}

impl Scorer for TermUnionScorer {
    fn supports_filtered_windows(&self) -> bool {
        self.supports_doc_windows()
    }
    fn score(&self) -> Score {
        1.0
    }
    fn supports_candidate_score_bounds(&self) -> bool {
        true
    }
    fn candidate_score_upper_bound(&self) -> Score {
        1.0
    }
    fn advance_competitive_candidate(&mut self, minimum: Score, allow_equal: bool) -> DocId {
        // The ranked collector certifies that all later stable-ID ties lose.
        // Complete and nested callers use ordinary advance/seek instead.
        if minimum > 1.0 || (!allow_equal && minimum == 1.0) {
            self.terminated = true;
            TERMINATED
        } else {
            self.advance()
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Materialize a posting union using the smaller of two bounded scratch forms.
/// Narrow prefixes append/sort doc IDs; broad, overlapping prefixes use a
/// segment-sized bitset so duplicate postings cannot multiply memory.
#[cfg(test)]
pub(super) fn materialize_union(
    postings: &[BlockPostingList],
    num_docs: u32,
    map: Option<&crate::segment::chunk_map::ChunkMap>,
) -> Vec<u32> {
    let inner = materialize_expanded(
        postings
            .iter()
            .map(|list| {
                ExpandedPosting::External(crate::structures::postings::DeferredPosting::from_list(
                    list,
                ))
            })
            .collect::<Vec<_>>(),
        num_docs,
        map,
    );
    collect_union_for_test(inner)
}

#[cfg(test)]
fn collect_union_for_test(inner: UnionDocs) -> Vec<u32> {
    let mut scorer = TermUnionScorer {
        inner,
        terminated: false,
    };
    let mut docs = Vec::new();
    while scorer.doc() != TERMINATED {
        docs.push(scorer.doc());
        scorer.advance();
    }
    docs
}

fn materialize_expanded(
    postings: Vec<ExpandedPosting>,
    num_docs: u32,
    map: Option<&crate::segment::chunk_map::ChunkMap>,
) -> UnionDocs {
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
            let mut append = |batch: &[u32]| {
                if let Some(map) = map {
                    docs.extend(batch.iter().map(|&doc| map.doc_id(doc)));
                } else {
                    docs.extend_from_slice(batch);
                }
                true
            };
            match posting {
                ExpandedPosting::Inline(postings) => {
                    append(postings.docs());
                }
                ExpandedPosting::External(postings) => postings
                    .into_list()
                    .iterator()
                    .visit_doc_ids_until(TERMINATED, append),
            }
        }
        docs.sort_unstable();
        docs.dedup();
        return UnionDocs::Materialized(SortedVecDocSet::new(Arc::new(docs)));
    }

    let mut bitset = super::DocBitset::new(num_docs);
    for posting in postings {
        let mut append = |batch: &[u32]| {
            for &doc in batch {
                bitset.set(map.map_or(doc, |map| map.doc_id(doc)));
            }
            true
        };
        match posting {
            ExpandedPosting::Inline(postings) => {
                append(postings.docs());
            }
            ExpandedPosting::External(postings) => postings
                .into_list()
                .iterator()
                .visit_doc_ids_until(TERMINATED, append),
        }
    }

    UnionDocs::Bitmap(BitsetDocSet::new(bitset))
}

/// Prefix unions materialise document-id sets; postings of a chunked field
/// are keyed by virtual chunk ids, so the union would filter the wrong
/// documents. Fail loudly instead of silently mis-matching.
pub(super) fn reject_chunked(
    reader: &crate::segment::SegmentReader,
    field: crate::Field,
    label: &str,
) -> crate::Result<()> {
    if reader.is_chunked_field(field) {
        return Err(crate::Error::Query(format!(
            "{label} is not supported on chunked text field '{}'; use a MatchQuery or PhraseQuery",
            reader.schema().get_field_name(field).unwrap_or("?")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_inline_and_external_unions_preserve_membership_and_seek() {
        let mut list = crate::structures::PostingList::new();
        for doc in (0..1024).step_by(2) {
            list.push(doc, 1);
        }
        let external = BlockPostingList::from_posting_list(&list).unwrap();
        let make_inputs = || {
            vec![
                ExpandedPosting::Inline(
                    crate::structures::TermInfo::try_inline(&[1, 128, 1025], &[1, 2, 3])
                        .unwrap()
                        .decode_inline_fixed()
                        .unwrap(),
                ),
                ExpandedPosting::External(crate::structures::postings::DeferredPosting::from_list(
                    &external,
                )),
                ExpandedPosting::Inline(
                    crate::structures::TermInfo::try_inline(&[1, 3], &[2, 1])
                        .unwrap()
                        .decode_inline_fixed()
                        .unwrap(),
                ),
            ]
        };
        let mut expected: Vec<u32> = (0..1024).step_by(2).chain([1, 3, 128, 1025]).collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(
            collect_union_for_test(materialize_expanded(make_inputs(), 1026, None)),
            expected
        );
        let mut union = PostingUnion::new(make_inputs(), 1026);
        let mut actual = Vec::new();
        while union.doc() != TERMINATED {
            actual.push(union.doc());
            union.seek(union.doc() + 1);
        }
        assert_eq!(actual, expected);
        let mut union = PostingUnion::new(make_inputs(), 1026);
        for target in [0, 1, 3, 127, 128, 999, 1025, 1026] {
            assert_eq!(
                union.seek(target),
                expected
                    .iter()
                    .copied()
                    .find(|&doc| doc >= target)
                    .unwrap_or(TERMINATED)
            );
        }
    }

    #[test]
    fn materialized_union_batches_preserve_seeks_and_resume_after_partial_tail() {
        let expected: Vec<u32> = (0..1000).map(|doc| doc * 3).collect();
        let mut scorer = TermUnionScorer::new(expected.clone());
        assert!(scorer.supports_doc_batches());
        assert_eq!(scorer.seek(16), 18);
        let mut actual = Vec::new();
        let mut docs = [0; super::super::docset::DOC_BATCH_SIZE];
        loop {
            let count = scorer.fill_doc_batch(&mut docs);
            if count == 0 {
                break;
            }
            actual.extend_from_slice(&docs[..count]);
            assert_eq!(
                scorer.doc(),
                expected
                    .get(6 + actual.len())
                    .copied()
                    .unwrap_or(TERMINATED)
            );
        }
        assert_eq!(actual, expected[6..]);
    }

    #[test]
    fn ranked_union_opens_only_lists_that_can_supply_the_next_document() {
        let mut postings = Vec::new();
        for start in (0..64).rev().map(|i| i * 1024) {
            let mut list = crate::structures::PostingList::new();
            for doc in start..start + 512 {
                list.push(doc, 1);
            }
            postings.push(BlockPostingList::from_posting_list(&list).unwrap());
        }
        let expected = materialize_union(&postings, 65536, None);
        let mut union = PostingUnion::new(
            postings
                .into_iter()
                .map(|list| {
                    ExpandedPosting::External(
                        crate::structures::postings::DeferredPosting::from_list(&list),
                    )
                })
                .collect(),
            65536,
        );
        assert_eq!(union.doc(), 0);
        assert_eq!(
            union
                .lists
                .iter()
                .filter(|list| matches!(list, UnionCursor::Active(_)))
                .count(),
            1
        );
        for target in [1, 100, 511, 512, 2049, 65000, 65536] {
            assert_eq!(
                union.seek(target),
                expected
                    .iter()
                    .copied()
                    .find(|&doc| doc >= target)
                    .unwrap_or(TERMINATED)
            );
        }
    }

    #[test]
    fn lazy_union_keeps_complete_deduplicated_membership_across_seeks_and_block_edges() {
        let postings: Vec<_> = [2, 3, 7]
            .into_iter()
            .map(|stride| {
                let mut list = crate::structures::PostingList::new();
                for doc in (0..2049).step_by(stride) {
                    list.push(doc, 1);
                }
                BlockPostingList::from_posting_list(&list).unwrap()
            })
            .collect();
        let expected = materialize_union(&postings, 2049, None);
        let mut scorer = TermUnionScorer::from_postings(postings.clone(), 2049, None, 10);
        let mut actual = Vec::new();
        while scorer.doc() != TERMINATED {
            actual.push(scorer.doc());
            scorer.advance();
        }
        assert_eq!(actual, expected);
        let mut scorer = TermUnionScorer::from_postings(postings, 2049, None, 10);
        for target in [0, 1, 128, 511, 1200, 2001, 2048, 2050] {
            assert_eq!(
                scorer.seek(target),
                expected
                    .iter()
                    .copied()
                    .find(|&doc| doc >= target)
                    .unwrap_or(TERMINATED)
            );
        }
    }
}
