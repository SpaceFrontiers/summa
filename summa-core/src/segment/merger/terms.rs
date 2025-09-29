//! Bounded k-way term traversal shared by copy merge and text BP planning.
use crate::segment::reader::SegmentReader;
use crate::structures::{AsyncSSTableIterator, TermInfo};
use crate::{Error, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

/// Bloom sizing for a dictionary rewritten from `segments`: the canonical
/// power-of-two size, built while writing when a safe bound on distinct terms
/// exists (`SSTableWriterConfig::bloom_sizing`).
pub(crate) fn rewrite_bloom_sizing(segments: &[SegmentReader]) -> crate::structures::BloomSizing {
    crate::structures::BloomSizing::PowerOfTwo {
        bound: key_bound_from_counts(
            segments
                .iter()
                .map(|segment| segment.term_dict_stats().num_entries),
        ),
    }
}

/// The sources' total bounds distinct terms from above and the largest source
/// from below. A presized filter costs at most `2 × bits_per_key / 8` bytes
/// per bounded key (power-of-two sizing); buffering costs 16 bytes per
/// distinct key. Presize only when that can never cost more, i.e. when the
/// total is within `64 / bits_per_key` of the largest source — heavily
/// overlapping vocabularies keep exact buffering.
fn key_bound_from_counts(counts: impl IntoIterator<Item = u64>) -> Option<usize> {
    let (mut total, mut largest) = (0u64, 0u64);
    for count in counts {
        total = total.checked_add(count)?;
        largest = largest.max(count);
    }
    let bits_per_key = crate::structures::BLOOM_BITS_PER_KEY as u64;
    (total.saturating_mul(bits_per_key) <= largest.saturating_mul(64))
        .then(|| usize::try_from(total).ok())
        .flatten()
}

/// Entry for k-way merge heap
struct MergeEntry {
    key: Vec<u8>,
    term_info: TermInfo,
    segment_idx: usize,
}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for MergeEntry {}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (BinaryHeap is max-heap by default)
        other.key.cmp(&self.key)
    }
}

pub(crate) struct MergedTerms<'a> {
    iterators: Vec<AsyncSSTableIterator<'a, TermInfo>>,
    heap: BinaryHeap<MergeEntry>,
    cancellation: Option<&'a AtomicBool>,
}

impl<'a> MergedTerms<'a> {
    pub(crate) async fn new(
        segments: &'a [SegmentReader],
        cancellation: Option<&'a AtomicBool>,
    ) -> Result<Self> {
        let mut terms = Self {
            iterators: segments.iter().map(SegmentReader::term_dict_iter).collect(),
            heap: BinaryHeap::with_capacity(segments.len()),
            cancellation,
        };
        for source in 0..segments.len() {
            terms.advance(source).await?;
        }
        Ok(terms)
    }

    async fn advance(&mut self, segment_idx: usize) -> Result<()> {
        if self
            .cancellation
            .is_some_and(|flag| flag.load(AtomicOrdering::Acquire))
        {
            return Err(Error::IndexClosed);
        }
        if let Some((key, term_info)) = self.iterators[segment_idx].next().await? {
            if key.len() < 4 {
                return Err(Error::Corruption("term key is missing its field ID".into()));
            }
            self.heap.push(MergeEntry {
                key,
                term_info,
                segment_idx,
            });
        }
        Ok(())
    }

    /// The final tuple slot is reserved for the caller's scoring-unit offset.
    pub(crate) async fn next(
        &mut self,
        sources: &mut Vec<(usize, TermInfo, u32)>,
    ) -> Result<Option<Vec<u8>>> {
        sources.clear();
        let Some(first) = self.heap.pop() else {
            return Ok(None);
        };
        let key = first.key;
        sources.push((first.segment_idx, first.term_info, 0));
        self.advance(first.segment_idx).await?;
        while self.heap.peek().is_some_and(|entry| entry.key == key) {
            let entry = self.heap.pop().unwrap();
            sources.push((entry.segment_idx, entry.term_info, 0));
            self.advance(entry.segment_idx).await?;
        }
        // Stable source order also fixes the vocabulary pass's edge order.
        sources.sort_unstable_by_key(|source| source.0);
        Ok(Some(key))
    }
}

#[cfg(test)]
mod key_bound_tests {
    use super::key_bound_from_counts;

    #[test]
    fn bloom_presizing_is_used_only_when_it_cannot_exceed_hash_buffering() {
        assert_eq!(key_bound_from_counts([1_000]), Some(1_000));
        // Three similar sources: at most 3,000 distinct keys, at least 1,000.
        assert_eq!(key_bound_from_counts([1_000, 900, 1_100]), Some(3_000));
        // Ten identical vocabularies could be 10× the distinct keys; presizing
        // for them could cost more than buffering the 1,000 real hashes.
        assert_eq!(key_bound_from_counts([1_000; 10]), None);
        assert_eq!(key_bound_from_counts([u64::MAX, 1]), None);
        assert_eq!(key_bound_from_counts([]), Some(0));
    }
}
