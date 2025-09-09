//! Stable survivor mapping: shared visibility words, compact ranks, no row arrays.
use std::{ops::Range, sync::Arc};

use crate::{Error, Result, query::DocBitset};

pub(crate) struct RowMap {
    alive: Option<Arc<DocBitset>>,
    prefix: Vec<u32>,
    physical: u32,
    live: u32,
    owned_bits: usize,
}

impl RowMap {
    #[cfg(test)]
    pub(crate) fn new(
        num_docs: u32,
        max_live_docs: u32,
        keep: impl Fn(u32) -> bool,
        budget: usize,
    ) -> Result<Self> {
        Self::try_new(num_docs, max_live_docs, |doc| Ok(keep(doc)), budget)
    }

    pub(crate) fn try_new(
        num_docs: u32,
        max_live_docs: u32,
        mut keep: impl FnMut(u32) -> Result<bool>,
        budget: usize,
    ) -> Result<Self> {
        if max_live_docs > num_docs {
            return Err(Error::Corruption(
                "live row count exceeds physical rows".into(),
            ));
        }
        let words = (num_docs as usize).div_ceil(64);
        Self::admit(words.saturating_mul(12).saturating_add(4), budget)?;
        let mut alive = DocBitset::new(num_docs);
        let mut count = 0;
        for doc in 0..num_docs {
            if keep(doc)? {
                if count == max_live_docs {
                    return Err(Error::Corruption(
                        "row map exceeds admitted live count".into(),
                    ));
                }
                alive.set(doc);
                count += 1;
            }
        }
        let owned_bits = alive.bits.capacity() * 8;
        let mut map = Self::from_visibility(
            num_docs,
            count,
            Some(Arc::new(alive)),
            budget - owned_bits,
            || Ok(()),
        )?;
        if map.alive.is_some() {
            map.owned_bits = owned_bits;
        }
        Ok(map)
    }

    pub(crate) fn from_visibility(
        physical: u32,
        live: u32,
        alive: Option<Arc<DocBitset>>,
        budget: usize,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        if live > physical {
            return Err(Error::Corruption(
                "live row count exceeds physical rows".into(),
            ));
        }
        let mut map = Self {
            alive: None,
            prefix: Vec::new(),
            physical,
            live,
            owned_bits: 0,
        };
        let Some(alive) = alive else {
            if live != physical {
                return Err(Error::Corruption("row visibility is missing".into()));
            }
            return Ok(map);
        };
        if alive.bits.len() != (physical as usize).div_ceil(64)
            || alive.next_set_bit(physical).is_some()
        {
            return Err(Error::Corruption("invalid row visibility".into()));
        }
        let mixed = live != 0 && live != physical;
        if mixed {
            Self::admit((alive.bits.len() + 1) * 4, budget)?;
            map.prefix = Vec::with_capacity(alive.bits.len() + 1);
        }
        let mut count = 0;
        for (i, &word) in alive.bits.iter().enumerate() {
            if i.is_multiple_of(4096) {
                check()?;
            }
            if mixed {
                map.prefix.push(count);
            }
            count += word.count_ones();
        }
        if count != live {
            return Err(Error::Corruption("invalid row visibility count".into()));
        }
        if mixed {
            map.prefix.push(count);
            map.alive = Some(alive);
        }
        Ok(map)
    }

    fn admit(bytes: usize, budget: usize) -> Result<()> {
        if bytes > budget {
            return Err(Error::Schema(format!(
                "compaction row map requires {bytes} bytes; budget is {budget}"
            )));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn rank(&self, old: u32) -> u32 {
        if old >= self.physical {
            return self.live;
        }
        let Some(alive) = &self.alive else {
            return if self.live == 0 { 0 } else { old };
        };
        let word = old as usize / 64;
        self.prefix[word] + (alive.bits[word] & ((1u64 << (old % 64)) - 1)).count_ones()
    }

    #[inline]
    pub(crate) fn get(&self, old: u32) -> Option<u32> {
        if old >= self.physical || self.live == 0 {
            return None;
        }
        if self.alive.as_ref().is_some_and(|bits| !bits.contains(old)) {
            return None;
        }
        Some(self.rank(old))
    }

    pub(crate) fn count(&self, range: Range<u32>) -> u32 {
        self.rank(range.end) - self.rank(range.start)
    }
    pub(crate) fn len(&self) -> u32 {
        self.live
    }
    pub(crate) fn physical(&self) -> u32 {
        self.physical
    }
    pub(crate) fn memory_bytes(&self) -> usize {
        self.prefix.capacity() * 4 + self.owned_bits
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.iter_range(0..self.physical)
    }

    pub(crate) fn iter_range(&self, range: Range<u32>) -> impl Iterator<Item = u32> + '_ {
        let mut next = range.start;
        let end = range.end.min(self.physical);
        std::iter::from_fn(move || {
            if next >= end || self.live == 0 {
                return None;
            }
            let doc = match &self.alive {
                Some(alive) => alive.next_set_bit(next)?,
                None => next,
            };
            if doc >= end {
                return None;
            }
            next = doc + 1;
            Some(doc)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mostly_deleted_rows_fit_the_live_map_budget_without_growing_it() {
        let rows = RowMap::new(64, 1, |doc| doc == 63, 260).unwrap();
        assert_eq!(rows.memory_bytes(), 16);
        assert_eq!(rows.get(63), Some(0));
        assert_eq!(rows.get(0), None);
        assert_eq!(rows.iter().collect::<Vec<_>>(), [63]);
        assert!(RowMap::new(64, 1, |_| true, 260).is_err());
        assert!(RowMap::new(64, 65, |_| true, 1024).is_err());
    }

    #[test]
    fn sparse_deletions_fit_bitmap_sized_scratch() {
        let rows = RowMap::new(100_000, 90_000, |doc| doc % 10 != 0, 20_000).unwrap();
        assert!(rows.memory_bytes() < 20_000);
        assert_eq!(rows.len(), 90_000);
    }

    #[test]
    fn rank_and_ranges_match_survivors_at_word_boundaries() {
        for n in [0, 1, 63, 64, 65, 127, 128, 513] {
            for modulus in [1, 2, 3, 67, 1024] {
                let kept: Vec<_> = (0..n).filter(|doc| doc % modulus != 0).collect();
                let rows = RowMap::new(n, n, |doc| doc % modulus != 0, 65536).unwrap();
                assert_eq!(rows.iter().collect::<Vec<_>>(), kept);
                for old in 0..=n {
                    let rank = kept.partition_point(|&doc| doc < old) as u32;
                    assert_eq!(rows.rank(old), rank);
                    assert_eq!(
                        rows.get(old),
                        kept.binary_search(&old).ok().map(|i| i as u32)
                    );
                    assert_eq!(rows.count(old..n), rows.len() - rank);
                }
            }
        }
    }

    #[test]
    fn physical_mapping_shares_visibility_and_propagates_cancellation() {
        let mut bits = DocBitset::all(100_000);
        bits.clear(12);
        let bits = Arc::new(bits);
        let map =
            RowMap::from_visibility(100_000, 99_999, Some(bits.clone()), 7_000, || Ok(())).unwrap();
        assert!(Arc::ptr_eq(map.alive.as_ref().unwrap(), &bits));
        assert_eq!(map.memory_bytes(), (100_000usize.div_ceil(64) + 1) * 4);
        let result = RowMap::try_new(
            100_000,
            100_000,
            |doc| {
                if doc == 256 {
                    Err(Error::IndexClosed)
                } else {
                    Ok(true)
                }
            },
            20_000,
        );
        assert!(matches!(result, Err(Error::IndexClosed)));
    }
}
