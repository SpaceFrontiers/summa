//! Shared constant-score execution for expanded term filters.

use super::Scorer;
use super::docset::{BitsetDocSet, DOC_WINDOW_SIZE, DocSet, SortedVecDocSet};
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
    fn count_batch_matches(&mut self, docs: &mut [DocId]) -> usize {
        let Some(&first) = docs.first() else { return 0 };
        self.seek(first);
        match self {
            Self::Active(iterator) => iterator.retain_doc_batch(docs),
            Self::Inline(postings, pos) => docs
                .iter()
                .filter(|doc| postings.docs()[*pos..].binary_search(doc).is_ok())
                .count(),
            Self::Exhausted => 0,
            Self::Pending(_) => unreachable!("nonempty batch opens a pending cursor"),
        }
    }

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

/// Exact physical cardinality. A dominant list contributes its dictionary
/// count; only the deduplicated tail needs membership probes against it.
pub(super) fn count_expanded(mut postings: Vec<ExpandedPosting>, num_docs: u32) -> u32 {
    let Some((largest, dominant_count)) = postings
        .iter()
        .enumerate()
        .map(|(index, posting)| (index, posting.doc_count()))
        .max_by_key(|&(_, count)| count)
    else {
        return 0;
    };
    if postings.len() == 1 {
        return dominant_count;
    }
    let other_count: u64 = postings
        .iter()
        .map(|p| u64::from(p.doc_count()))
        .sum::<u64>()
        - u64::from(dominant_count);
    // Probe only a small tail. A less selective tail touches most dominant
    // blocks, where the ordinary bitmap union is cheaper than random probes.
    if other_count * 64 > u64::from(dominant_count) {
        return match materialize_expanded(postings, num_docs, None) {
            UnionDocs::Materialized(docs) => docs.size_hint(),
            UnionDocs::Bitmap(docs) => docs.size_hint(),
            UnionDocs::Streaming(_) => unreachable!("materialization never streams"),
        };
    }
    let mut dominant = match postings.swap_remove(largest) {
        ExpandedPosting::Inline(postings) => UnionCursor::Inline(postings, 0),
        ExpandedPosting::External(posting) => UnionCursor::Pending(posting),
    };
    let mut tail = TermUnionScorer {
        inner: materialize_expanded(postings, num_docs, None),
        terminated: false,
    };
    let mut count = dominant_count;
    let mut docs = [0; super::docset::DOC_BATCH_SIZE];
    loop {
        let len = tail.fill_doc_batch(&mut docs);
        if len == 0 {
            return count;
        }
        count += (len - dominant.count_batch_matches(&mut docs[..len])) as u32;
    }
}

/// Materialize a posting union in one of two bounded scratch forms. Narrow
/// unions append and sort doc IDs; wider ones use a segment-sized bitset, so
/// duplicate postings cannot multiply memory.
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
    // Sorting costs milliseconds from about a hundred thousand IDs, while a
    // bitset costs one pass over the segment's words plus one bit per
    // posting. Sort only below one posting per 1,024 documents; the bitset
    // is then at most 32 times the ID vector it replaces.
    if posting_count.saturating_mul(1024) <= num_docs as usize {
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
    let mut window = [0u64; super::docset::DOC_WINDOW_WORDS];
    for posting in postings {
        let mut append = |batch: &[u32]| {
            for &doc in batch {
                bitset.set(map.map_or(doc, |map| map.doc_id(doc)));
            }
            true
        };
        let list = match posting {
            ExpandedPosting::Inline(postings) => {
                append(postings.docs());
                continue;
            }
            ExpandedPosting::External(postings) => postings.into_list(),
        };
        let mut iterator = list.iterator();
        if map.is_some() || list.density() == crate::structures::postings::ListDensity::Sparse {
            iterator.visit_doc_ids_until(TERMINATED, append);
            continue;
        }
        // Dense lists fill whole windows without per-document reloads.
        while iterator.doc() != TERMINATED {
            let base = iterator.doc() / DOC_WINDOW_SIZE * DOC_WINDOW_SIZE;
            iterator.fill_doc_window(base, &mut window);
            let words = &mut bitset.bits[(base / 64) as usize..];
            for (word, window) in words.iter_mut().zip(&window) {
                *word |= window;
            }
        }
    }

    UnionDocs::Bitmap(BitsetDocSet::new(bitset))
}

/// Term expansions on an RGB-reordered plain field traverse its physical IDs:
/// a union's size and membership do not depend on the numbering, and a
/// constant-score query ranks its matches in physical order
/// (`docs/physical-tie-order.md`), so a ranked stream stops after `k`.
pub(super) fn physical_union_field(
    reader: &crate::segment::SegmentReader,
    field: crate::Field,
) -> Option<crate::Field> {
    reader
        .chunk_map(field)
        .is_some_and(|map| map.is_document_map())
        .then_some(field)
}

/// The map a union translates its postings through: none when the collector
/// traverses the field's physical IDs.
pub(super) fn union_map<'a>(
    reader: &'a crate::segment::SegmentReader,
    field: crate::Field,
    options: &super::ScorerOptions,
) -> Option<&'a crate::segment::chunk_map::ChunkMap> {
    if options.physical_text_field == Some(field) {
        None
    } else {
        reader.chunk_map(field)
    }
}

/// Admit a text expansion before dictionary lookup. Chunk IDs are not
/// document IDs, so a chunked union cannot use ordinary document collectors.
pub(super) fn validate_expansion_field(
    reader: &crate::segment::SegmentReader,
    field: crate::Field,
    label: &str,
) -> crate::Result<()> {
    let entry = reader
        .schema()
        .get_field_entry(field)
        .ok_or_else(|| crate::Error::Query(format!("{label}: unknown field {}", field.0)))?;
    if entry.field_type != crate::dsl::FieldType::Text || !entry.indexed {
        return Err(crate::Error::Query(format!(
            "{label} requires an indexed text field, but '{}' is {:?} (indexed={})",
            entry.name, entry.field_type, entry.indexed
        )));
    }
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
    fn dominated_counts_deduplicate_tail_overlap_across_codecs_and_union_shapes() {
        use crate::structures::{PostingCodec, PostingList};
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::RoundedBitmap,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            for dense in [false, true] {
                let mut dominant = PostingList::new();
                for doc in (10..8200).step_by(if dense { 1 } else { 2 }) {
                    dominant.push(doc, 1);
                }
                let dominant =
                    BlockPostingList::from_posting_list_with_codec(&dominant, codec).unwrap();
                for tail in [
                    vec![],
                    vec![1, 10, 8000, 8300],
                    (0..8200).step_by(3).collect(),
                ] {
                    let mut small = PostingList::new();
                    for &doc in &tail {
                        small.push(doc, 1);
                    }
                    let small =
                        BlockPostingList::from_posting_list_with_codec(&small, codec).unwrap();
                    let lists = vec![dominant.clone(), small.clone(), small];
                    for num_docs in [8400, 1_000_000_000] {
                        let expected = materialize_union(&lists, num_docs, None).len() as u32;
                        let actual = count_expanded(
                            lists
                                .iter()
                                .map(|list| {
                                    ExpandedPosting::External(
                                        crate::structures::postings::DeferredPosting::from_list(
                                            list,
                                        ),
                                    )
                                })
                                .collect(),
                            num_docs,
                        );
                        assert_eq!(
                            actual, expected,
                            "{codec:?}, dense={dense}, docs={num_docs}"
                        );
                    }
                }
            }
        }
        assert_eq!(count_expanded(vec![], 100), 0);
        let inline = || {
            ExpandedPosting::Inline(
                crate::structures::TermInfo::try_inline(&[1, 3], &[1, 1])
                    .unwrap()
                    .decode_inline_fixed()
                    .unwrap(),
            )
        };
        assert_eq!(count_expanded(vec![inline()], 100), 2);
        assert_eq!(count_expanded(vec![inline(), inline()], 100), 2);
    }

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
    fn unions_sort_only_below_one_posting_per_1024_documents() {
        let num_docs = 1 << 20;
        for (count, sorted) in [(1024u32, true), (1025, false)] {
            let lists: Vec<BlockPostingList> = [0, 1]
                .iter()
                .map(|&shift| {
                    let mut list = crate::structures::PostingList::new();
                    for i in (shift..count).step_by(2) {
                        list.push(i * 1000, 1);
                    }
                    BlockPostingList::from_posting_list(&list).unwrap()
                })
                .collect();
            let inner = materialize_expanded(
                lists
                    .iter()
                    .map(|list| {
                        ExpandedPosting::External(
                            crate::structures::postings::DeferredPosting::from_list(list),
                        )
                    })
                    .collect(),
                num_docs,
                None,
            );
            assert_eq!(matches!(inner, UnionDocs::Materialized(_)), sorted);
            assert_eq!(
                collect_union_for_test(inner),
                (0..count).map(|i| i * 1000).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn dense_window_unions_match_per_document_bits_across_windows() {
        let num_docs = 3 * DOC_WINDOW_SIZE + 1234;
        let shapes: [(u32, u32); 4] = [(3, 1), (5, 4095), (7, 0), (40, 13)];
        let lists: Vec<BlockPostingList> = shapes
            .iter()
            .map(|&(stride, start)| {
                let mut list = crate::structures::PostingList::new();
                for doc in (start..num_docs).step_by(stride as usize) {
                    list.push(doc, 1);
                }
                BlockPostingList::from_posting_list(&list).unwrap()
            })
            .collect();
        assert_eq!(
            lists
                .iter()
                .map(|list| list.density() != crate::structures::postings::ListDensity::Sparse)
                .collect::<Vec<_>>(),
            [true, true, true, false]
        );
        let mut expected: Vec<u32> = shapes
            .iter()
            .flat_map(|&(stride, start)| (start..num_docs).step_by(stride as usize))
            .collect();
        expected.sort_unstable();
        expected.dedup();
        let docs = collect_union_for_test(materialize_expanded(
            lists
                .iter()
                .map(|list| {
                    ExpandedPosting::External(
                        crate::structures::postings::DeferredPosting::from_list(list),
                    )
                })
                .collect(),
            num_docs,
            None,
        ));
        assert_eq!(docs, expected);
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
