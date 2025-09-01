//! Slice-level caching directory with overlap management
//!
//! Caches byte ranges from files, merging overlapping ranges and
//! evicting least-recently-used slices when the cache limit is reached.
//!
//! Concurrency and complexity:
//! - Lazy-handle hits take the state `read()` lock only. Recency is a
//!   per-slice atomic stamp drawn from small thread-local blocks, and hit
//!   accounting is sharded by thread, so readers do not contend on one
//!   global counter. Direct range reads retain the faster serialized path.
//! - Slices of one file never overlap, so a range is either fully contained
//!   in its predecessor slice (`BTreeMap::range(..=start).next_back()`) or it
//!   is a miss; overlap detection on insert walks backwards from the last
//!   slice starting before the new end and stops at the first disjoint one.
//! - Eviction is bounded-approximate LRU through a lazily maintained min-heap
//!   of `(stamp, file, start)` entries: ordering is exact within a thread and
//!   may differ by at most one 64-stamp reservation block across threads. A
//!   popped stale entry is re-pushed with its current stamp. This is
//!   amortized `O(log n)` per operation and never scans all slices (the heap
//!   is rebuilt from live slices only when stale entries outnumber live ones).

use async_trait::async_trait;
use parking_lot::RwLock;
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::{Directory, FileHandle, OwnedBytes, RangeReadFn};

/// File extension for slice cache files
pub const SLICE_CACHE_EXTENSION: &str = "slicecache";

/// Magic bytes for slice cache file format
const SLICE_CACHE_MAGIC: &[u8; 8] = b"HRMSCACH";

/// Current version of the slice cache format
/// v2: Added file size caching
const SLICE_CACHE_VERSION: u32 = 2;

/// Flush in-process hit counts to the `metrics` facade every this many hits.
/// Misses and evictions flush unconditionally (they are already slow paths).
const HIT_METRICS_FLUSH_INTERVAL: u64 = 1024;

/// Reserve recency stamps in small per-thread blocks. This removes the
/// globally contended atomic increment from every cache hit while bounding
/// cross-thread LRU ordering error to at most one block.
const STAMP_BLOCK_SIZE: u64 = 64;
const COUNTER_SHARDS: usize = 64;

static NEXT_STAMP_BLOCK: AtomicU64 = AtomicU64::new(0);
static NEXT_COUNTER_SHARD: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static STAMP_BLOCK: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    static COUNTER_SHARD: usize = NEXT_COUNTER_SHARD.fetch_add(1, Ordering::Relaxed)
        % COUNTER_SHARDS;
}

#[inline]
fn next_stamp() -> u64 {
    STAMP_BLOCK.with(|block| {
        let (next, end) = block.get();
        if next < end {
            block.set((next + 1, end));
            return next;
        }
        let start = NEXT_STAMP_BLOCK.fetch_add(STAMP_BLOCK_SIZE, Ordering::Relaxed) + 1;
        block.set((start + 1, start + STAMP_BLOCK_SIZE));
        start
    })
}

/// Keep frequently updated counters on separate cache lines. Assigning a
/// thread to a shard costs one global increment for the lifetime of that
/// thread, rather than one for every cache hit.
#[repr(align(64))]
struct CounterShard(AtomicU64);

struct ShardedCounter {
    shards: [CounterShard; COUNTER_SHARDS],
}

impl ShardedCounter {
    fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| CounterShard(AtomicU64::new(0))),
        }
    }

    #[inline]
    fn increment(&self) -> u64 {
        COUNTER_SHARD.with(|&shard| self.shards[shard].0.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn load(&self) -> u64 {
        self.shards
            .iter()
            .map(|shard| shard.0.load(Ordering::Relaxed))
            .sum()
    }
}

/// A cached slice of a file
#[derive(Debug)]
struct CachedSlice {
    /// Byte range in the file
    range: Range<u64>,
    /// Arc-backed cached data. Cache hits return cheap sub-slices instead of
    /// allocating and copying the requested range.
    data: OwnedBytes,
    /// Recency stamp for LRU eviction. Updated by hits under the shared
    /// read lock, hence atomic.
    access_count: AtomicU64,
}

impl CachedSlice {
    #[inline]
    fn stamp(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
}

/// Per-file slice cache: non-overlapping slices keyed by start offset.
struct FileSliceCache {
    /// Stable identity used by LRU heap entries (paths can be renamed).
    id: u64,
    /// Slices sorted by start offset for efficient overlap detection
    slices: BTreeMap<u64, CachedSlice>,
    /// Total bytes cached for this file
    total_bytes: usize,
}

impl FileSliceCache {
    fn new(id: u64) -> Self {
        Self {
            id,
            slices: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Serialize this file cache to bytes
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Number of slices
        buf.extend_from_slice(&(self.slices.len() as u32).to_le_bytes());
        for slice in self.slices.values() {
            // Range start and end
            buf.extend_from_slice(&slice.range.start.to_le_bytes());
            buf.extend_from_slice(&slice.range.end.to_le_bytes());
            // Data length and data
            buf.extend_from_slice(&(slice.data.len() as u32).to_le_bytes());
            buf.extend_from_slice(slice.data.as_slice());
        }
        buf
    }

    /// Deserialize from bytes, returns (cache, bytes_consumed)
    fn deserialize(
        data: &[u8],
        id: u64,
        access_counter: u64,
        max_bytes: usize,
    ) -> io::Result<(Self, usize)> {
        let mut pos = 0;
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated slice cache",
            ));
        }
        let num_slices = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut cache = FileSliceCache::new(id);
        for _ in 0..num_slices {
            if pos + 20 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated slice entry",
                ));
            }
            let range_start = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let range_end = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            let data_end = pos.checked_add(data_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "slice data length overflow")
            })?;
            if data_end > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated slice data",
                ));
            }
            if range_end < range_start || range_end - range_start != data_len as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "slice range and data length are inconsistent",
                ));
            }
            let slice_range = range_start..range_end;
            pos = data_end;

            // Do not duplicate an oversized serialized entry just to evict it
            // after the complete cache has been reconstructed. Retain at most
            // one cache budget while parsing each file.
            if data_len <= max_bytes {
                let bytes_to_free = cache
                    .total_bytes
                    .saturating_add(data_len)
                    .saturating_sub(max_bytes);
                cache.evict_lru(bytes_to_free);
                cache.insert(
                    slice_range,
                    OwnedBytes::new(data[data_end - data_len..data_end].to_vec()),
                    access_counter,
                );
                debug_assert!(cache.total_bytes <= max_bytes);
            }
        }
        Ok((cache, pos))
    }

    /// Try to read from cache; `None` if the range is not fully cached.
    ///
    /// Slices never overlap, so only the slice starting at or before
    /// `range.start` can contain the range. Recency is recorded through the
    /// slice's atomic stamp, which is why this takes `&self`.
    fn try_read(&self, range: Range<u64>) -> Option<OwnedBytes> {
        let start = range.start;
        let end = range.end;
        let (&slice_start, slice) = self.slices.range(..=start).next_back()?;
        if slice.range.end < end {
            return None;
        }
        let stamp = next_stamp();
        slice.access_count.store(stamp, Ordering::Relaxed);
        let offset = (start - slice_start) as usize;
        let len = (end - start) as usize;
        Some(slice.data.slice(offset..offset + len))
    }

    /// Insert a slice, merging with overlapping slices.
    ///
    /// Returns the net change in bytes (negative when the merge shrinks the
    /// footprint) and the start offset of the (possibly merged) slice.
    fn insert(&mut self, range: Range<u64>, data: OwnedBytes, access_counter: u64) -> (isize, u64) {
        let start = range.start;
        let end = range.end;
        let data_len = data.len();

        // Overlapping slices all start before `end`; walking backwards from
        // there, the first slice that ends at or before `start` is disjoint
        // and so is everything before it (slices are sorted and disjoint).
        let mut to_remove: Vec<u64> = Vec::new();
        let mut merged_start = start;
        let mut merged_end = end;
        for (&slice_start, slice) in self.slices.range(..end).rev() {
            if slice.range.end <= start {
                break;
            }
            to_remove.push(slice_start);
            merged_start = merged_start.min(slice_start);
            merged_end = merged_end.max(slice.range.end);
        }

        let mut bytes_removed: usize = 0;
        let (final_start, final_data) = if to_remove.is_empty() {
            (start, data)
        } else {
            let merged_len = (merged_end - merged_start) as usize;
            let mut new_data = vec![0u8; merged_len];

            // Copy existing slices, then the new data over any overlap.
            for &slice_start in &to_remove {
                if let Some(slice) = self.slices.remove(&slice_start) {
                    let offset = (slice_start - merged_start) as usize;
                    new_data[offset..offset + slice.data.len()]
                        .copy_from_slice(slice.data.as_slice());
                    bytes_removed += slice.data.len();
                    self.total_bytes -= slice.data.len();
                }
            }
            let offset = (start - merged_start) as usize;
            new_data[offset..offset + data_len].copy_from_slice(data.as_slice());
            (merged_start, OwnedBytes::new(new_data))
        };

        let bytes_added = final_data.len();
        self.total_bytes += bytes_added;
        self.slices.insert(
            final_start,
            CachedSlice {
                range: final_start..final_start + bytes_added as u64,
                data: final_data,
                access_count: AtomicU64::new(access_counter),
            },
        );

        (bytes_added as isize - bytes_removed as isize, final_start)
    }

    /// Evict least recently used slices of this file to free up space.
    /// Used while reconstructing a single file from a serialized cache; the
    /// live cache evicts through the global LRU heap instead.
    fn evict_lru(&mut self, bytes_to_free: usize) -> usize {
        if bytes_to_free == 0 || self.slices.is_empty() {
            return 0;
        }
        let mut order: Vec<(u64, u64)> = self
            .slices
            .iter()
            .map(|(&start, slice)| (slice.stamp(), start))
            .collect();
        order.sort_unstable();

        let mut freed = 0;
        for (_, start) in order {
            if freed >= bytes_to_free {
                break;
            }
            if let Some(slice) = self.slices.remove(&start) {
                freed += slice.data.len();
                self.total_bytes -= slice.data.len();
            }
        }
        freed
    }
}

/// Lazily maintained LRU heap entry. Ordered by stamp first so the heap
/// minimum is the least recently used candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LruEntry {
    stamp: u64,
    file_id: u64,
    start: u64,
}

/// Everything protected by the cache lock.
struct CacheState {
    files: HashMap<Arc<Path>, FileSliceCache>,
    /// file id → path, so heap entries survive renames.
    paths: HashMap<u64, Arc<Path>>,
    lru: BinaryHeap<Reverse<LruEntry>>,
    current_bytes: usize,
    total_slices: usize,
    next_file_id: u64,
}

impl CacheState {
    fn new() -> Self {
        Self {
            files: HashMap::new(),
            paths: HashMap::new(),
            lru: BinaryHeap::new(),
            current_bytes: 0,
            total_slices: 0,
            next_file_id: 0,
        }
    }

    fn file_mut(&mut self, path: &Path) -> &mut FileSliceCache {
        if !self.files.contains_key(path) {
            let id = self.next_file_id;
            self.next_file_id += 1;
            let shared: Arc<Path> = Arc::from(path);
            self.paths.insert(id, Arc::clone(&shared));
            self.files.insert(shared, FileSliceCache::new(id));
        }
        self.files.get_mut(path).expect("file cache just inserted")
    }

    fn remove_file(&mut self, path: &Path) -> Option<FileSliceCache> {
        let file = self.files.remove(path)?;
        self.paths.remove(&file.id);
        self.current_bytes = self.current_bytes.saturating_sub(file.total_bytes);
        self.total_slices = self.total_slices.saturating_sub(file.slices.len());
        // Heap entries of the removed file are discarded lazily on pop.
        Some(file)
    }

    fn rename_file(&mut self, from: &Path, to: &Path) {
        // Any cache already present under the destination is superseded.
        self.remove_file(to);
        if let Some(file) = self.files.remove(from) {
            let id = file.id;
            let shared: Arc<Path> = Arc::from(to);
            self.paths.insert(id, Arc::clone(&shared));
            self.files.insert(shared, file);
        }
    }

    /// Replace (or add) a whole file cache, registering every slice in the
    /// LRU heap. Returns the previous cache, if any.
    fn replace_file(&mut self, path: &Path, mut cache: FileSliceCache) {
        self.remove_file(path);
        let id = self.next_file_id;
        self.next_file_id += 1;
        cache.id = id;
        for (&start, slice) in &cache.slices {
            self.lru.push(Reverse(LruEntry {
                stamp: slice.stamp(),
                file_id: id,
                start,
            }));
        }
        self.current_bytes = self.current_bytes.saturating_add(cache.total_bytes);
        self.total_slices += cache.slices.len();
        let shared: Arc<Path> = Arc::from(path);
        self.paths.insert(id, Arc::clone(&shared));
        self.files.insert(shared, cache);
        self.compact_lru_if_bloated();
    }

    /// Insert one slice into a file cache and account for it globally.
    fn insert_slice(&mut self, path: &Path, range: Range<u64>, data: OwnedBytes, stamp: u64) {
        let file = self.file_mut(path);
        let slices_before = file.slices.len();
        let (net_change, start) = file.insert(range, data, stamp);
        let file_id = file.id;
        let slices_after = file.slices.len();
        self.total_slices = (self.total_slices + slices_after).saturating_sub(slices_before);
        if net_change >= 0 {
            self.current_bytes += net_change as usize;
        } else {
            self.current_bytes = self.current_bytes.saturating_sub((-net_change) as usize);
        }
        self.lru.push(Reverse(LruEntry {
            stamp,
            file_id,
            start,
        }));
        self.compact_lru_if_bloated();
    }

    /// Stale heap entries (merged or evicted slices, superseded stamps) are
    /// discarded lazily; rebuild when they clearly dominate.
    fn compact_lru_if_bloated(&mut self) {
        if self.lru.len() > 2 * self.total_slices + 1024 {
            self.rebuild_lru();
        }
    }

    fn rebuild_lru(&mut self) {
        let mut entries = Vec::with_capacity(self.total_slices);
        for file in self.files.values() {
            for (&start, slice) in &file.slices {
                entries.push(Reverse(LruEntry {
                    stamp: slice.stamp(),
                    file_id: file.id,
                    start,
                }));
            }
        }
        self.lru = BinaryHeap::from(entries);
    }

    /// Evict least recently used slices until `needed` more bytes fit under
    /// `max_bytes`. Returns `(evicted_slices, evicted_bytes)`.
    fn evict_for(&mut self, max_bytes: usize, needed: usize) -> (u64, usize) {
        let target = self
            .current_bytes
            .saturating_add(needed)
            .saturating_sub(max_bytes);
        if target == 0 {
            return (0, 0);
        }
        let mut freed = 0usize;
        let mut evicted = 0u64;
        let mut rebuilt = false;
        while freed < target {
            let Some(Reverse(entry)) = self.lru.pop() else {
                // Every live slice owns at least one heap entry, so an empty
                // heap with live slices means the index is inconsistent.
                // Rebuild once and keep going; give up only when truly empty.
                if self.total_slices == 0 || rebuilt {
                    break;
                }
                self.rebuild_lru();
                rebuilt = true;
                continue;
            };
            let Some(path) = self.paths.get(&entry.file_id) else {
                continue; // file removed
            };
            let Some(file) = self.files.get_mut(path.as_ref()) else {
                continue;
            };
            let Some(slice) = file.slices.get(&entry.start) else {
                continue; // slice merged away or already evicted
            };
            let current = slice.stamp();
            if current != entry.stamp {
                // Touched since this entry was recorded: not the LRU anymore.
                self.lru.push(Reverse(LruEntry {
                    stamp: current,
                    ..entry
                }));
                continue;
            }
            let slice = file
                .slices
                .remove(&entry.start)
                .expect("slice present under lock");
            file.total_bytes -= slice.data.len();
            freed += slice.data.len();
            evicted += 1;
            self.total_slices -= 1;
        }
        self.current_bytes = self.current_bytes.saturating_sub(freed);
        (evicted, freed)
    }

    fn clear(&mut self) {
        self.files.clear();
        self.paths.clear();
        self.lru.clear();
        self.current_bytes = 0;
        self.total_slices = 0;
    }
}

/// Lock-protected cache state plus lock-free counters, shared between the
/// directory and every lazy file handle it hands out.
struct SliceCacheShared {
    state: RwLock<CacheState>,
    /// Maximum total bytes to cache
    max_bytes: usize,
    hits: ShardedCounter,
    misses: AtomicU64,
    evicted_slices: AtomicU64,
    evicted_bytes: AtomicU64,
    /// Index name for Directory-layer metric labels (also forwarded to inner)
    label: super::IndexLabel,
}

impl SliceCacheShared {
    fn new(max_bytes: usize) -> Self {
        Self {
            state: RwLock::new(CacheState::new()),
            max_bytes,
            hits: ShardedCounter::new(),
            misses: AtomicU64::new(0),
            evicted_slices: AtomicU64::new(0),
            evicted_bytes: AtomicU64::new(0),
            label: super::IndexLabel::default(),
        }
    }

    /// Hit path: shared lock, atomic stamp update, and sharded accounting.
    fn try_read(&self, path: &Path, range: Range<u64>) -> Option<OwnedBytes> {
        let hit = {
            let state = self.state.read();
            state.files.get(path).and_then(|file| file.try_read(range))
        };
        self.record_lookup(&hit);
        hit
    }

    /// The direct `Directory::read_range` entry point has no reusable file
    /// handle and its sub-microsecond critical section is faster when
    /// serialized than when many readers bounce the RwLock reader count.
    /// Lazy handles use `try_read` above and remain concurrent.
    fn try_read_direct(&self, path: &Path, range: Range<u64>) -> Option<OwnedBytes> {
        let hit = {
            let state = self.state.write();
            state.files.get(path).and_then(|file| file.try_read(range))
        };
        self.record_lookup(&hit);
        hit
    }

    #[inline]
    fn record_lookup(&self, result: &Option<OwnedBytes>) {
        match result {
            Some(data) => {
                let hits = self.hits.increment();
                if hits.is_multiple_of(HIT_METRICS_FLUSH_INTERVAL) {
                    crate::observe::slice_cache_hits(
                        &self.label.get(),
                        HIT_METRICS_FLUSH_INTERVAL,
                        data.len(),
                    );
                }
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Miss path: exclusive lock, single eviction pass, merge-insert.
    fn insert(&self, path: &Path, range: Range<u64>, data: OwnedBytes) {
        let data_len = data.len();
        crate::observe::slice_cache_miss(&self.label.get(), data_len);
        // An individual entry larger than the entire cache can never fit.
        // Bypass it instead of evicting useful data and exceeding the cap.
        if data_len > self.max_bytes {
            return;
        }
        let stamp = next_stamp();
        let (evicted_slices, evicted_bytes) = {
            let mut state = self.state.write();
            // Free enough space before merging. Besides keeping the retained
            // size bounded, this avoids constructing a large merged
            // allocation only to evict it immediately afterward. Merging
            // never grows the footprint beyond `data_len` (overlap is
            // replaced, not duplicated), so one pass suffices.
            let evicted = state.evict_for(self.max_bytes, data_len);
            state.insert_slice(path, range, data, stamp);
            debug_assert!(state.current_bytes <= self.max_bytes);
            evicted
        };
        if evicted_slices > 0 {
            self.evicted_slices
                .fetch_add(evicted_slices, Ordering::Relaxed);
            self.evicted_bytes
                .fetch_add(evicted_bytes as u64, Ordering::Relaxed);
            crate::observe::slice_cache_evicted(&self.label.get(), evicted_slices, evicted_bytes);
        }
    }
}

/// Slice-caching directory wrapper
///
/// Caches byte ranges from the inner directory, with:
/// - Overlap detection and merging
/// - LRU eviction when cache limit is reached
/// - Bounded total memory usage
/// - File size caching to avoid HEAD requests
pub struct SliceCachingDirectory<D: Directory> {
    inner: Arc<D>,
    shared: Arc<SliceCacheShared>,
    /// Cached file sizes (avoids HEAD requests on lazy open)
    file_sizes: Arc<RwLock<HashMap<PathBuf, u64>>>,
}

impl<D: Directory> SliceCachingDirectory<D> {
    /// Create a new slice-caching directory with the given memory limit
    pub fn new(inner: D, max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(inner),
            shared: Arc::new(SliceCacheShared::new(max_bytes)),
            file_sizes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a reference to the inner directory
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Try to read from cache
    fn try_cache_read(&self, path: &Path, range: Range<u64>) -> Option<OwnedBytes> {
        self.shared.try_read_direct(path, range)
    }

    /// Insert into cache, evicting if necessary
    fn cache_insert(&self, path: &Path, range: Range<u64>, data: OwnedBytes) {
        self.shared.insert(path, range, data)
    }

    fn invalidate(&self, path: &Path) {
        {
            let mut state = self.shared.state.write();
            state.remove_file(path);
        }
        self.file_sizes.write().remove(path);
    }

    /// Get cache statistics
    pub fn stats(&self) -> SliceCacheStats {
        let state = self.shared.state.read();
        let mut total_slices = 0;
        let mut files_cached = 0;

        for fc in state.files.values() {
            if !fc.slices.is_empty() {
                files_cached += 1;
                total_slices += fc.slices.len();
            }
        }

        SliceCacheStats {
            total_bytes: state.current_bytes,
            max_bytes: self.shared.max_bytes,
            total_slices,
            files_cached,
            hits: self.shared.hits.load(),
            misses: self.shared.misses.load(Ordering::Relaxed),
            evicted_slices: self.shared.evicted_slices.load(Ordering::Relaxed),
            evicted_bytes: self.shared.evicted_bytes.load(Ordering::Relaxed),
        }
    }

    /// Serialize the entire cache to a single binary blob
    ///
    /// Format (v2):
    /// - Magic: 8 bytes "HRMSCACH"
    /// - Version: 4 bytes (u32 LE)
    /// - Num files: 4 bytes (u32 LE)
    /// - For each file:
    ///   - Path length: 4 bytes (u32 LE)
    ///   - Path: UTF-8 bytes
    ///   - File cache data (see FileSliceCache::serialize)
    /// - Num file sizes: 4 bytes (u32 LE) [v2+]
    /// - For each file size: [v2+]
    ///   - Path length: 4 bytes (u32 LE)
    ///   - Path: UTF-8 bytes
    ///   - File size: 8 bytes (u64 LE)
    pub fn serialize(&self) -> Vec<u8> {
        let state = self.shared.state.read();
        let file_sizes = self.file_sizes.read();
        let mut buf = Vec::new();

        // Magic and version
        buf.extend_from_slice(SLICE_CACHE_MAGIC);
        buf.extend_from_slice(&SLICE_CACHE_VERSION.to_le_bytes());

        // Count non-empty caches
        let non_empty: Vec<_> = state
            .files
            .iter()
            .filter(|(_, fc)| !fc.slices.is_empty())
            .collect();
        buf.extend_from_slice(&(non_empty.len() as u32).to_le_bytes());

        for (path, file_cache) in non_empty {
            // Path
            let path_str = path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            buf.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(path_bytes);

            // File cache data
            let cache_data = file_cache.serialize();
            buf.extend_from_slice(&cache_data);
        }

        // v2: File sizes section
        buf.extend_from_slice(&(file_sizes.len() as u32).to_le_bytes());
        for (path, &size) in file_sizes.iter() {
            let path_str = path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            buf.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(path_bytes);
            buf.extend_from_slice(&size.to_le_bytes());
        }

        buf
    }

    /// Deserialize and prefill the cache from a binary blob
    ///
    /// This loads cached slices from a previously serialized cache file.
    /// Existing cache entries are preserved; new entries are merged in.
    pub fn deserialize(&self, data: &[u8]) -> io::Result<()> {
        let mut pos = 0;

        // Check magic
        if data.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "slice cache too short",
            ));
        }
        if &data[pos..pos + 8] != SLICE_CACHE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid slice cache magic",
            ));
        }
        pos += 8;

        // Check version (v2 only)
        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if version != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported slice cache version: {} (expected 2)", version),
            ));
        }

        // Number of files
        let num_files = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let max_bytes = self.shared.max_bytes;
        let counter = next_stamp();
        let mut state = self.shared.state.write();

        for _ in 0..num_files {
            // Path length
            if pos + 4 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated path length",
                ));
            }
            let path_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            // Path
            if pos + path_len > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated path"));
            }
            let path_str = std::str::from_utf8(&data[pos..pos + path_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let path = PathBuf::from(path_str);
            pos += path_len;

            // File cache (the id is reassigned on insertion)
            let (file_cache, consumed) =
                FileSliceCache::deserialize(&data[pos..], 0, counter, max_bytes)?;
            pos += consumed;

            state.replace_file(&path, file_cache);
            state.evict_for(max_bytes, 0);
        }

        // Recompute once after loading as a consistency check for serialized
        // caches containing duplicate paths or overlapping ranges.
        state.current_bytes = state.files.values().map(|cache| cache.total_bytes).sum();
        state.total_slices = state.files.values().map(|cache| cache.slices.len()).sum();
        state.evict_for(max_bytes, 0);
        drop(state);

        // Load file sizes
        if pos + 4 <= data.len() {
            let num_sizes = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            let mut file_sizes = self.file_sizes.write();
            for _ in 0..num_sizes {
                if pos + 4 > data.len() {
                    break;
                }
                let path_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;

                if pos + path_len > data.len() {
                    break;
                }
                let path_str = match std::str::from_utf8(&data[pos..pos + path_len]) {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let path = PathBuf::from(path_str);
                pos += path_len;

                if pos + 8 > data.len() {
                    break;
                }
                let size = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                pos += 8;

                file_sizes.insert(path, size);
            }
        }

        Ok(())
    }

    /// Serialize the cache to a writer
    pub fn serialize_to_writer<W: Write>(&self, mut writer: W) -> io::Result<()> {
        let data = self.serialize();
        writer.write_all(&data)
    }

    /// Deserialize the cache from a reader
    pub fn deserialize_from_reader<R: Read>(&self, mut reader: R) -> io::Result<()> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        self.deserialize(&data)
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.shared.state.read().current_bytes == 0
    }

    /// Clear all cached data
    pub fn clear(&self) {
        self.shared.state.write().clear();
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct SliceCacheStats {
    pub total_bytes: usize,
    pub max_bytes: usize,
    pub total_slices: usize,
    pub files_cached: usize,
    /// Range reads served from cache since creation.
    pub hits: u64,
    /// Range reads that went to the inner directory since creation.
    pub misses: u64,
    /// Slices dropped by LRU eviction since creation.
    pub evicted_slices: u64,
    /// Bytes dropped by LRU eviction since creation.
    pub evicted_bytes: u64,
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<D: Directory> Directory for SliceCachingDirectory<D> {
    async fn exists(&self, path: &Path) -> io::Result<bool> {
        self.inner.exists(path).await
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        // Check cache first
        {
            let file_sizes = self.file_sizes.read();
            if let Some(&size) = file_sizes.get(path) {
                return Ok(size);
            }
        }

        // Fetch from inner and cache
        let size = self.inner.file_size(path).await?;
        {
            let mut file_sizes = self.file_sizes.write();
            file_sizes.insert(path.to_path_buf(), size);
        }
        Ok(size)
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileHandle> {
        // Check if we have the full file cached (use our caching file_size)
        let file_size = self.file_size(path).await?;
        let full_range = 0..file_size;

        // Try cache first for full file
        if let Some(data) = self.try_cache_read(path, full_range.clone()) {
            return Ok(FileHandle::from_bytes(data));
        }

        // Read from inner
        let handle = self.inner.open_read(path).await?;
        let bytes = handle.read_bytes().await?;

        // Cache the full file
        self.cache_insert(path, full_range, bytes.clone());

        Ok(FileHandle::from_bytes(bytes))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        // Try cache first
        if let Some(data) = self.try_cache_read(path, range.clone()) {
            return Ok(data);
        }

        // Read from inner
        let data = self.inner.read_range(path, range.clone()).await?;

        // Cache the result
        self.cache_insert(path, range, data.clone());

        Ok(data)
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
        // Get file size (uses cache to avoid HEAD requests)
        let file_size = self.file_size(path).await?;

        // Create a caching wrapper around the inner directory's read_range.
        // The path is shared, not cloned, per read.
        let path: Arc<Path> = Arc::from(path);
        let shared = Arc::clone(&self.shared);
        let inner = Arc::clone(&self.inner);

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let path = Arc::clone(&path);
            let shared = Arc::clone(&shared);
            let inner = Arc::clone(&inner);

            Box::pin(async move {
                // Try cache first
                if let Some(data) = shared.try_read(&path, range.clone()) {
                    return Ok(data);
                }

                // Read from inner
                let data = inner.read_range(&path, range.clone()).await?;

                // Cache the result
                shared.insert(&path, range, data.clone());

                Ok(data)
            })
        });

        Ok(FileHandle::lazy_labeled(
            file_size,
            read_fn,
            self.shared.label.get(),
        ))
    }

    fn local_path(&self, path: &Path) -> Option<PathBuf> {
        self.inner.local_path(path)
    }

    fn set_index_label(&self, label: &str) {
        self.shared.label.set(label);
        self.inner.set_index_label(label);
    }
}

/// DirectoryWriter implementation for SliceCachingDirectory
/// Delegates to inner directory and invalidates cache entries as needed
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<D: super::DirectoryWriter> super::DirectoryWriter for SliceCachingDirectory<D> {
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        // Invalidate cache and file size for this file
        self.invalidate(path);
        // Delegate to inner
        self.inner.write(path, data).await
    }

    async fn delete(&self, path: &Path) -> io::Result<()> {
        self.invalidate(path);
        // Delegate to inner
        self.inner.delete(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        // Move cache entries from old path to new path
        {
            let mut state = self.shared.state.write();
            state.rename_file(from, to);
        }
        // Move file size cache
        {
            let mut file_sizes = self.file_sizes.write();
            if let Some(size) = file_sizes.remove(from) {
                file_sizes.insert(to.to_path_buf(), size);
            }
        }
        // Delegate to inner
        self.inner.rename(from, to).await
    }

    async fn link(&self, from: &Path, to: &Path) -> io::Result<()> {
        // A link creates an immutable alias. Do not copy cache entries: the
        // destination starts cold and is populated under its own path.
        self.inner.link(from, to).await
    }

    async fn sync(&self) -> io::Result<()> {
        self.inner.sync().await
    }

    async fn streaming_writer(&self, path: &Path) -> io::Result<Box<dyn super::StreamingWriter>> {
        // Invalidate cache for this file before writing
        self.invalidate(path);
        self.inner.streaming_writer(path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::{DirectoryWriter, RamDirectory};

    #[tokio::test]
    async fn test_slice_cache_basic() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .await
            .unwrap();

        let cached = SliceCachingDirectory::new(ram, 1024);

        // First read - cache miss
        let data = cached
            .read_range(Path::new("test.bin"), 2..5)
            .await
            .unwrap();
        assert_eq!(data.as_slice(), &[2, 3, 4]);

        // Second read - should be cache hit
        let data = cached
            .read_range(Path::new("test.bin"), 2..5)
            .await
            .unwrap();
        assert_eq!(data.as_slice(), &[2, 3, 4]);

        let stats = cached.stats();
        assert_eq!(stats.total_slices, 1);
        assert_eq!(stats.total_bytes, 3);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn slice_cache_hits_reuse_the_cached_backing_allocation() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[7; 64]).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 64);

        let miss = cached
            .read_range(Path::new("test.bin"), 8..56)
            .await
            .unwrap();
        let hit = cached
            .read_range(Path::new("test.bin"), 8..56)
            .await
            .unwrap();

        assert_eq!(miss.as_slice(), hit.as_slice());
        assert_eq!(miss.as_slice().as_ptr(), hit.as_slice().as_ptr());
    }

    #[tokio::test]
    async fn oversized_slice_bypasses_cache_instead_of_exceeding_limit() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[3; 32]).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 8);

        let data = cached
            .read_range(Path::new("test.bin"), 0..32)
            .await
            .unwrap();
        assert_eq!(data.len(), 32);
        assert_eq!(cached.stats().total_bytes, 0);
    }

    #[tokio::test]
    async fn test_slice_cache_overlap_merge() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .await
            .unwrap();

        let cached = SliceCachingDirectory::new(ram, 1024);

        // Read [2..5]
        cached
            .read_range(Path::new("test.bin"), 2..5)
            .await
            .unwrap();

        // Read [4..7] - overlaps with previous
        cached
            .read_range(Path::new("test.bin"), 4..7)
            .await
            .unwrap();

        let stats = cached.stats();
        // Should be merged into one slice [2..7]
        assert_eq!(stats.total_slices, 1);
        assert_eq!(stats.total_bytes, 5); // bytes 2,3,4,5,6

        // Reading from merged range should work
        let data = cached
            .read_range(Path::new("test.bin"), 3..6)
            .await
            .unwrap();
        assert_eq!(data.as_slice(), &[3, 4, 5]);
    }

    /// Overlap detection walks backwards from the last slice starting before
    /// the new range's end. Pins the edge cases of that walk: a new range
    /// that bridges several slices, one that ends exactly where a slice
    /// starts (adjacent, not overlapping), one that starts exactly where a
    /// slice ends, and one fully inside an existing slice.
    #[tokio::test]
    async fn overlap_merge_bridges_multiple_slices_and_keeps_adjacent_ones_apart() {
        let ram = RamDirectory::new();
        let bytes: Vec<u8> = (0..64).collect();
        ram.write(Path::new("test.bin"), &bytes).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 1024);
        let path = Path::new("test.bin");

        // Three disjoint slices: [4..8), [12..16), [20..24), plus [40..44)
        for range in [4..8, 12..16, 20..24, 40..44] {
            cached.read_range(path, range).await.unwrap();
        }
        assert_eq!(cached.stats().total_slices, 4);

        // Adjacent on both sides but not overlapping: [8..12) stays separate
        // from [4..8) and [12..16).
        cached.read_range(path, 8..12).await.unwrap();
        assert_eq!(cached.stats().total_slices, 5);
        assert_eq!(cached.stats().total_bytes, 20);

        // A range bridging [4..8), [8..12), [12..16) and [20..24) merges all
        // four into one [4..24) slice; [40..44) is untouched.
        cached.read_range(path, 6..22).await.unwrap();
        let stats = cached.stats();
        assert_eq!(stats.total_slices, 2);
        assert_eq!(stats.total_bytes, 20 + 4);

        // Fully contained range is a hit and does not change the layout.
        let misses_before = cached.stats().misses;
        let data = cached.read_range(path, 10..14).await.unwrap();
        assert_eq!(data.as_slice(), &[10, 11, 12, 13]);
        assert_eq!(cached.stats().misses, misses_before);
        assert_eq!(cached.stats().total_slices, 2);

        // Every byte of the merged slice reads back correctly.
        let data = cached.read_range(path, 4..24).await.unwrap();
        assert_eq!(data.as_slice(), &bytes[4..24]);
        let data = cached.read_range(path, 40..44).await.unwrap();
        assert_eq!(data.as_slice(), &bytes[40..44]);
    }

    #[tokio::test]
    async fn test_slice_cache_eviction() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[0; 100]).await.unwrap();

        // Small cache limit
        let cached = SliceCachingDirectory::new(ram, 50);

        // Fill cache
        cached
            .read_range(Path::new("test.bin"), 0..30)
            .await
            .unwrap();

        // This should trigger eviction
        cached
            .read_range(Path::new("test.bin"), 50..80)
            .await
            .unwrap();

        let stats = cached.stats();
        assert!(stats.total_bytes <= 50);
        assert_eq!(stats.evicted_slices, 1);
        assert_eq!(stats.evicted_bytes, 30);
    }

    /// Eviction is exact LRU even though hits only bump an atomic stamp: a
    /// slice touched after its heap entry was recorded must survive an older
    /// untouched slice, across files.
    #[tokio::test]
    async fn eviction_is_lru_across_files_after_hits_refresh_recency() {
        let ram = RamDirectory::new();
        ram.write(Path::new("a.bin"), &[1; 64]).await.unwrap();
        ram.write(Path::new("b.bin"), &[2; 64]).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 32);

        cached.read_range(Path::new("a.bin"), 0..10).await.unwrap(); // oldest
        cached.read_range(Path::new("b.bin"), 0..10).await.unwrap();
        cached.read_range(Path::new("a.bin"), 20..30).await.unwrap();
        // Refresh the oldest slice; b.bin[0..10) is now the LRU.
        cached.read_range(Path::new("a.bin"), 2..8).await.unwrap();

        // Needs 10 more bytes: exactly one slice must go, and it must be b.
        cached.read_range(Path::new("a.bin"), 40..50).await.unwrap();
        let stats = cached.stats();
        assert_eq!(stats.total_bytes, 30);
        assert_eq!(stats.evicted_slices, 1);

        let misses = cached.stats().misses;
        cached.read_range(Path::new("a.bin"), 0..10).await.unwrap();
        cached.read_range(Path::new("a.bin"), 20..30).await.unwrap();
        assert_eq!(cached.stats().misses, misses, "refreshed slices survived");
        cached.read_range(Path::new("b.bin"), 0..10).await.unwrap();
        assert_eq!(cached.stats().misses, misses + 1, "LRU slice was evicted");
    }

    /// Renaming a file keeps its slices (and their LRU entries) usable.
    #[tokio::test]
    async fn rename_keeps_cached_slices_evictable_and_readable() {
        let ram = RamDirectory::new();
        ram.write(Path::new("old.bin"), &[9; 64]).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 16);
        cached.read_range(Path::new("old.bin"), 0..8).await.unwrap();
        cached
            .rename(Path::new("old.bin"), Path::new("new.bin"))
            .await
            .unwrap();

        let misses = cached.stats().misses;
        let data = cached.read_range(Path::new("new.bin"), 0..8).await.unwrap();
        assert_eq!(data.as_slice(), &[9; 8]);
        assert_eq!(cached.stats().misses, misses);

        // Filling the cache must be able to evict the renamed slice.
        cached
            .read_range(Path::new("new.bin"), 16..24)
            .await
            .unwrap();
        cached
            .read_range(Path::new("new.bin"), 32..40)
            .await
            .unwrap();
        let stats = cached.stats();
        assert!(stats.total_bytes <= 16);
        assert_eq!(stats.evicted_slices, 1);
    }

    #[tokio::test]
    async fn test_slice_cache_serialize_deserialize() {
        let ram = RamDirectory::new();
        ram.write(Path::new("file1.bin"), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .await
            .unwrap();
        ram.write(Path::new("file2.bin"), &[10, 11, 12, 13, 14, 15])
            .await
            .unwrap();

        let cached = SliceCachingDirectory::new(ram.clone(), 1024);

        // Read some ranges to populate cache
        cached
            .read_range(Path::new("file1.bin"), 2..6)
            .await
            .unwrap();
        cached
            .read_range(Path::new("file2.bin"), 1..4)
            .await
            .unwrap();

        let stats = cached.stats();
        assert_eq!(stats.files_cached, 2);
        assert_eq!(stats.total_bytes, 7); // 4 + 3

        // Serialize
        let serialized = cached.serialize();
        assert!(!serialized.is_empty());

        // Create new cache and deserialize
        let cached2 = SliceCachingDirectory::new(ram.clone(), 1024);
        assert!(cached2.is_empty());

        cached2.deserialize(&serialized).unwrap();

        let stats2 = cached2.stats();
        assert_eq!(stats2.files_cached, 2);
        assert_eq!(stats2.total_bytes, 7);

        // Verify cached data is correct by reading (should be cache hits)
        let data = cached2
            .read_range(Path::new("file1.bin"), 2..6)
            .await
            .unwrap();
        assert_eq!(data.as_slice(), &[2, 3, 4, 5]);

        let data = cached2
            .read_range(Path::new("file2.bin"), 1..4)
            .await
            .unwrap();
        assert_eq!(data.as_slice(), &[11, 12, 13]);
        assert_eq!(cached2.stats().misses, 0);
    }

    #[tokio::test]
    async fn test_slice_cache_serialize_empty() {
        let ram = RamDirectory::new();
        let cached = SliceCachingDirectory::new(ram, 1024);

        // Serialize empty cache
        let serialized = cached.serialize();
        assert!(!serialized.is_empty()); // Should have header

        // Deserialize into new cache
        let cached2 = SliceCachingDirectory::new(RamDirectory::new(), 1024);
        cached2.deserialize(&serialized).unwrap();
        assert!(cached2.is_empty());
    }

    #[tokio::test]
    async fn deserialization_enforces_the_destination_cache_limit() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[1; 64]).await.unwrap();
        let source = SliceCachingDirectory::new(ram.clone(), 64);
        source
            .read_range(Path::new("test.bin"), 0..64)
            .await
            .unwrap();

        let destination = SliceCachingDirectory::new(ram, 8);
        destination.deserialize(&source.serialize()).unwrap();
        assert!(destination.stats().total_bytes <= 8);
    }

    /// The lazy handle path (segment readers) shares hit/miss accounting and
    /// eviction with `read_range`.
    #[tokio::test]
    async fn lazy_handle_reads_hit_the_shared_cache() {
        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[4; 128]).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 1024);

        cached
            .read_range(Path::new("test.bin"), 0..64)
            .await
            .unwrap();
        let handle = cached.open_lazy(Path::new("test.bin")).await.unwrap();
        let data = handle.read_bytes_range(8..40).await.unwrap();
        assert_eq!(data.len(), 32);
        assert_eq!(cached.stats().hits, 1);
        assert_eq!(cached.stats().misses, 1);

        let data = handle.read_bytes_range(64..128).await.unwrap();
        assert_eq!(data.len(), 64);
        assert_eq!(cached.stats().misses, 2);
        assert_eq!(cached.stats().total_bytes, 128);
    }

    /// Sharding the hit counter must not make the public total approximate:
    /// every hit from every reader thread is included in `stats()`.
    #[tokio::test]
    async fn concurrent_lazy_handle_hits_are_counted_exactly() {
        const THREADS: usize = 8;
        const HITS_PER_THREAD: usize = 250;

        let ram = RamDirectory::new();
        ram.write(Path::new("test.bin"), &[7; 4096]).await.unwrap();
        let cached = SliceCachingDirectory::new(ram, 4096);
        cached
            .read_range(Path::new("test.bin"), 0..4096)
            .await
            .unwrap();
        let handle = cached.open_lazy(Path::new("test.bin")).await.unwrap();

        std::thread::scope(|scope| {
            for thread in 0..THREADS {
                let handle = &handle;
                scope.spawn(move || {
                    for hit in 0..HITS_PER_THREAD {
                        let start = ((thread * 31 + hit * 7) % (4096 - 64)) as u64;
                        let bytes =
                            futures::executor::block_on(handle.read_bytes_range(start..start + 64))
                                .unwrap();
                        assert_eq!(bytes.len(), 64);
                    }
                });
            }
        });

        assert_eq!(cached.stats().hits, (THREADS * HITS_PER_THREAD) as u64);
        assert_eq!(cached.stats().misses, 1);
    }
}
