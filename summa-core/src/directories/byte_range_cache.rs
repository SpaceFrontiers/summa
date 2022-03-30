use std::borrow::{Borrow, Cow};
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use tantivy::directory::OwnedBytes;

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
struct CacheKey<'a, T: ToOwned + ?Sized> {
    tag: Cow<'a, T>,
    range_start: u64,
}

impl<T: ToOwned + ?Sized> CacheKey<'static, T> {
    fn from_owned(tag: T::Owned, range_start: u64) -> Self {
        CacheKey {
            tag: Cow::Owned(tag),
            range_start,
        }
    }
}

impl<'a, T: ToOwned + ?Sized> CacheKey<'a, T> {
    fn from_borrowed(tag: &'a T, range_start: u64) -> Self {
        CacheKey {
            tag: Cow::Borrowed(tag),
            range_start,
        }
    }
}

struct CacheValue {
    range_end: u64,
    bytes: OwnedBytes,
}

/// T is a tag, usually a file path.
struct NeedMutByteRangeCache<T: 'static + ToOwned + ?Sized> {
    cache: BTreeMap<CacheKey<'static, T>, CacheValue>,
    // this is hardly significant as items can get merged if they overlap
    num_items: u64,
    num_bytes: u64,
}

impl<T: 'static + ToOwned + ?Sized + Ord> NeedMutByteRangeCache<T>
where
    T::Owned: std::fmt::Debug,
{
    fn with_infinite_capacity() -> Self {
        NeedMutByteRangeCache {
            cache: BTreeMap::new(),
            num_items: 0,
            num_bytes: 0,
        }
    }

    fn get_slice(&mut self, tag: &T, byte_range: Range<u64>) -> Option<OwnedBytes> {
        if byte_range.start == byte_range.end {
            return Some(OwnedBytes::empty());
        }

        let key = CacheKey::from_borrowed(tag, byte_range.start);
        let (k, v) = if let Some((k, v)) = self.get_block(&key, byte_range.end) {
            (k, v)
        } else {
            return None;
        };

        let start = (byte_range.start - k.range_start) as usize;
        let end = (byte_range.end - k.range_start) as usize;

        Some(v.bytes.slice(start..end))
    }

    fn put_slice(&mut self, tag: T::Owned, byte_range: Range<u64>, bytes: OwnedBytes) {
        let len = (byte_range.end - byte_range.start) as usize;
        assert_eq!(
            len,
            bytes.len(),
            "declared byte_range {:?} length is not equal to data length {} for tag {:?}",
            byte_range,
            bytes.len(),
            tag
        );
        if len == 0 {
            return;
        }

        let start_key = CacheKey::from_borrowed(tag.borrow(), byte_range.start);
        let end_key = CacheKey::from_borrowed(tag.borrow(), byte_range.end);

        let first_matching_block = self.get_block(&start_key, byte_range.start).map(|(k, _v)| k);

        let last_matching_block = self.get_block(&end_key, byte_range.end).map(|(k, _v)| k);

        if first_matching_block.is_some() && first_matching_block == last_matching_block {
            // same start and end: all the range is already covered
            return;
        }

        let first_matching_block = first_matching_block.unwrap_or(&start_key);
        let last_matching_block = last_matching_block.unwrap_or(&end_key);

        let overlapping: Vec<Range<u64>> = self
            .cache
            .range(first_matching_block..=last_matching_block)
            .map(|(k, v)| k.range_start..v.range_end)
            .collect();

        let can_drop_first = overlapping.first().map(|r| byte_range.start <= r.start).unwrap_or(true);

        let can_drop_last = overlapping.last().map(|r| byte_range.end >= r.end).unwrap_or(true);

        let (final_range, final_bytes) = if can_drop_first && can_drop_last {
            (byte_range, bytes)
        } else {
            let start = if can_drop_first {
                byte_range.start
            } else {
                // if no first, can_drop_first is true
                overlapping.first().expect("impossible").start
            };

            let end = if can_drop_last {
                byte_range.end
            } else {
                // if no last, can_drop_last is true
                overlapping.last().expect("impossible").end
            };

            let mut buffer = Vec::with_capacity((end - start) as usize);

            if !can_drop_first {
                let first_range = overlapping.first().expect("impossible");
                let key = CacheKey::from_borrowed(tag.borrow(), first_range.start);
                let block = self.cache.get(&key).expect("impossible");

                let len = (first_range.end.min(byte_range.start) - first_range.start) as usize;
                buffer.extend_from_slice(&block.bytes[..len]);
            }

            buffer.extend_from_slice(&bytes);

            if !can_drop_last {
                let last_range = overlapping.last().expect("impossible");
                let key = CacheKey::from_borrowed(tag.borrow(), last_range.start);
                let block = self.cache.get(&key).expect("impossible");

                let start = (last_range.start.max(byte_range.end) - last_range.start) as usize;
                buffer.extend_from_slice(&block.bytes[start..]);
            }

            debug_assert_eq!((end - start) as usize, buffer.len());

            (start..end, OwnedBytes::new(buffer))
        };

        // not sure why, but the borrow check gets unhappy if I create a borrowed
        // in the loop. It works with .get() instead of .remove() (?).
        let mut key = CacheKey::from_owned(tag, 0);
        for range in overlapping.into_iter() {
            key.range_start = range.start;
            self.cache.remove(&key);
            self.update_counter_drop_item((range.end - range.start) as usize);
        }

        key.range_start = final_range.start;
        let value = CacheValue {
            range_end: final_range.end,
            bytes: final_bytes,
        };
        self.cache.insert(key, value);
        self.update_counter_record_item((final_range.end - final_range.start) as usize);
    }

    // Return a block that contain everything between query.range_start and range_end
    fn get_block<'a, 'b: 'a>(&'a self, query: &CacheKey<'b, T>, range_end: u64) -> Option<(&CacheKey<'_, T>, &CacheValue)> {
        self.cache
            .range(..=query)
            .next_back()
            .filter(|(k, v)| k.tag == query.tag && range_end <= v.range_end)
    }

    fn update_counter_record_item(&mut self, num_bytes: usize) {
        self.num_items += 1;
        self.num_bytes += num_bytes as u64;
    }

    fn update_counter_drop_item(&mut self, num_bytes: usize) {
        self.num_items -= 1;
        self.num_bytes -= num_bytes as u64;
    }
}

/// Cache for ranges of bytes in files.
///
/// This cache is used in the contraption that makes it possible for Quickwit
/// to use tantivy while doing asynchronous io.
/// Quickwit manually populates this cache in an asynchronous "warmup" phase.
/// tantivy then gets its data from this cache without performing any IO.
///
/// Contrary to `MemorySizedCache`, it's able to answer subset of known ranges,
/// does not have any eviction, and assumes an infinite capacity.
///
/// This cache assume immutable data: if you put a new slice and it overlap with
/// cached data, the changes may or may not get recorded.
///
/// At the moment this is hardly a cache as it features no eviction policy.
pub struct ByteRangeCache {
    inner: Mutex<NeedMutByteRangeCache<Path>>,
}

impl ByteRangeCache {
    /// Creates a slice cache that nevers removes any entry.
    pub fn with_infinite_capacity() -> Self {
        ByteRangeCache {
            inner: Mutex::new(NeedMutByteRangeCache::with_infinite_capacity()),
        }
    }

    /// If available, returns the cached view of the slice.
    pub fn get_slice(&self, path: &Path, byte_range: Range<u64>) -> Option<OwnedBytes> {
        self.inner.lock().get_slice(path, byte_range)
    }

    /// Put the given amount of data in the cache.
    pub fn put_slice(&self, path: PathBuf, byte_range: Range<u64>, bytes: OwnedBytes) {
        self.inner.lock().put_slice(path, byte_range, bytes)
    }
}
