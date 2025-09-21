//! Shared constant-score execution for expanded term filters.

use super::Scorer;
use super::docset::{DocSet, SortedVecDocSet};
use crate::structures::{BlockPostingList, TERMINATED};
use crate::{DocId, Score};
use std::sync::Arc;

// ── TermUnionScorer ────────────────────────────────────────────────────────

/// Scorer backed by a pre-materialized sorted doc ID set.
pub(super) struct TermUnionScorer {
    inner: SortedVecDocSet,
}

impl TermUnionScorer {
    pub(super) fn new(docs: Vec<u32>) -> Self {
        Self {
            inner: SortedVecDocSet::new(Arc::new(docs)),
        }
    }
}

impl DocSet for TermUnionScorer {
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

impl Scorer for TermUnionScorer {
    fn score(&self) -> Score {
        1.0
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Materialize a posting union using the smaller of two bounded scratch forms.
/// Narrow prefixes append/sort doc IDs; broad, overlapping prefixes use a
/// segment-sized bitset so duplicate postings cannot multiply memory.
pub(super) fn materialize_union(
    postings: &[BlockPostingList],
    num_docs: u32,
    map: Option<&crate::segment::chunk_map::ChunkMap>,
) -> Vec<u32> {
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
                docs.push(map.map_or(d, |map| map.doc_id(d)));
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
            bitset.set(map.map_or(d, |map| map.doc_id(d)));
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
