//! Decompressed SSTable blocks, their lazily built suffix filters, and the
//! entry offsets of scanned shared-cache blocks. See
//! `docs/dictionary-suffix-filters.md` and `docs/dictionary-entry-offsets.md`.
use super::super::block_cache::RetainedBytes;
use std::sync::OnceLock;

/// One decompressed block. Shared-cache dictionaries may attach a filter of
/// the keys' last four bytes after a complete scan of the block.
pub(crate) struct DecodedBlock {
    data: Box<[u8]>,
    suffixes: OnceLock<Box<SuffixFilter>>,
    /// Offset of every entry in the entry stream, then the stream's end. Only
    /// the copy a shared cache swaps in after a complete scan carries them, so
    /// a cached block's retained bytes never change.
    entry_starts: Option<Box<[u16]>>,
}

impl DecodedBlock {
    pub(crate) fn new(data: Vec<u8>) -> Self {
        Self {
            data: data.into_boxed_slice(),
            suffixes: OnceLock::new(),
            entry_starts: None,
        }
    }

    /// A copy of this block (and its suffix filter, if built) carrying the
    /// offsets recorded by a complete scan.
    pub(crate) fn with_entry_starts(&self, starts: &[u16]) -> Self {
        let suffixes = OnceLock::new();
        if let Some(filter) = self.suffix_filter() {
            let _ = suffixes.set(Box::new(filter.clone()));
        }
        Self {
            data: self.data.clone(),
            suffixes,
            entry_starts: Some(starts.into()),
        }
    }

    pub(crate) fn entry_starts(&self) -> Option<&[u16]> {
        self.entry_starts.as_deref()
    }

    pub(crate) fn suffix_filter(&self) -> Option<&SuffixFilter> {
        self.suffixes.get().map(Box::as_ref)
    }

    /// Publish a filter built from every key of this block. A concurrent
    /// scanner may have published an identical one first.
    pub(crate) fn publish_suffix_filter(&self, filter: Box<SuffixFilter>) {
        let _ = self.suffixes.set(filter);
    }
}

impl std::ops::Deref for DecodedBlock {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.data
    }
}

/// Shared caches charge the filter up front, so the process budget stays exact
/// whether or not a scan ever builds it.
impl RetainedBytes for DecodedBlock {
    fn retained_bytes(&self) -> usize {
        self.data.len()
            + std::mem::size_of::<SuffixFilter>()
            + self
                .entry_starts
                .as_ref()
                .map_or(0, |starts| size_of_val(&**starts))
    }
}

/// Bloom filter over the last four bytes of every key in one block, plus the
/// block's entry count so a skipped block is still charged to scan budgets.
#[derive(Clone)]
pub(crate) struct SuffixFilter {
    bits: [u64; SuffixFilter::WORDS],
    entries: u32,
}

impl SuffixFilter {
    const WORDS: usize = 128;
    const BITS: u64 = (Self::WORDS * 64) as u64;

    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            bits: [0; Self::WORDS],
            entries: 0,
        })
    }

    #[inline(always)]
    fn probes(ending: [u8; 4]) -> [usize; 3] {
        let hash = u64::from(u32::from_le_bytes(ending)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // Three 13-bit probes from the well-mixed high bits.
        [51, 38, 25].map(|shift| ((hash >> shift) % Self::BITS) as usize)
    }

    /// Record one key of the block (keys shorter than four bytes cannot end
    /// with a four-byte suffix and set no bits).
    #[inline(always)]
    pub(crate) fn insert(&mut self, key: &[u8]) {
        self.entries += 1;
        if let Some(ending) = key.last_chunk::<4>() {
            for bit in Self::probes(*ending) {
                self.bits[bit / 64] |= 1 << (bit % 64);
            }
        }
    }

    /// False only if no key of the block ends with `ending`.
    pub(crate) fn may_contain(&self, ending: [u8; 4]) -> bool {
        Self::probes(ending)
            .into_iter()
            .all(|bit| self.bits[bit / 64] >> (bit % 64) & 1 == 1)
    }

    pub(crate) fn entries(&self) -> usize {
        self.entries as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_filter_has_no_false_negatives_and_counts_every_key() {
        let keys: Vec<Vec<u8>> = (0..1500u32)
            .map(|i| format!("\0\0\0\0term{i:05}x").into_bytes())
            .chain([b"ab".to_vec()])
            .collect();
        let mut filter = SuffixFilter::new();
        for key in &keys {
            filter.insert(key);
        }
        assert_eq!(filter.entries(), keys.len());
        for key in keys.iter().filter(|key| key.len() >= 4) {
            assert!(filter.may_contain(*key.last_chunk::<4>().unwrap()));
        }
        let absent = (0..10_000u32)
            .map(|i| i.to_le_bytes())
            .filter(|ending| !filter.may_contain(*ending))
            .count();
        assert!(absent > 9_000, "filter too dense: only {absent} rejections");
    }
}
