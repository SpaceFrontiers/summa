//! Slice-level caching directory with overlap management
//!
//! Caches byte ranges from files, merging overlapping ranges and
//! evicting least-recently-used slices when the cache limit is reached.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{Directory, FileSlice, LazyFileHandle, OwnedBytes, RangeReadFn};

/// File extension for slice cache files
pub const SLICE_CACHE_EXTENSION: &str = "slicecache";

/// Magic bytes for slice cache file format
const SLICE_CACHE_MAGIC: &[u8; 8] = b"HRMSCACH";

/// Current version of the slice cache format
/// v2: Added file size caching
const SLICE_CACHE_VERSION: u32 = 2;

/// A cached slice of a file
#[derive(Debug, Clone)]
struct CachedSlice {
    /// Byte range in the file
    range: Range<u64>,
    /// The cached data
    data: Arc<Vec<u8>>,
    /// Access counter for LRU eviction
    access_count: u64,
}

/// Per-file slice cache using interval tree for overlap detection
struct FileSliceCache {
    /// Slices sorted by start offset for efficient overlap detection
    slices: BTreeMap<u64, CachedSlice>,
    /// Total bytes cached for this file
    total_bytes: usize,
}

impl FileSliceCache {
    fn new() -> Self {
        Self {
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
            buf.extend_from_slice(&slice.data);
        }
        buf
    }

    /// Deserialize from bytes, returns (cache, bytes_consumed)
    fn deserialize(data: &[u8], access_counter: u64) -> io::Result<(Self, usize)> {
        let mut pos = 0;
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated slice cache",
            ));
        }
        let num_slices = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut cache = FileSliceCache::new();
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

            if pos + data_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated slice data",
                ));
            }
            let slice_data = data[pos..pos + data_len].to_vec();
            pos += data_len;

            cache.total_bytes += slice_data.len();
            cache.slices.insert(
                range_start,
                CachedSlice {
                    range: range_start..range_end,
                    data: Arc::new(slice_data),
                    access_count: access_counter,
                },
            );
        }
        Ok((cache, pos))
    }

    /// Get iterator over all slices for serialization
    #[allow(dead_code)]
    fn iter_slices(&self) -> impl Iterator<Item = (&u64, &CachedSlice)> {
        self.slices.iter()
    }

    /// Try to read from cache, returns None if not fully cached
    fn try_read(&mut self, range: Range<u64>, access_counter: &mut u64) -> Option<Vec<u8>> {
        // Find slices that might contain our range
        let start = range.start;
        let end = range.end;

        // Look for a slice that contains the entire range
        let mut found_key = None;
        for (&slice_start, slice) in self.slices.range(..=start).rev() {
            if slice_start <= start && slice.range.end >= end {
                found_key = Some((
                    slice_start,
                    (start - slice_start) as usize,
                    (end - start) as usize,
                ));
                break;
            }
        }

        if let Some((key, offset, len)) = found_key {
            // Update access count for LRU
            *access_counter += 1;
            if let Some(s) = self.slices.get_mut(&key) {
                s.access_count = *access_counter;
                return Some(s.data[offset..offset + len].to_vec());
            }
        }

        None
    }

    /// Insert a slice, merging with overlapping slices
    /// Returns the net change in bytes (can be negative if merge reduces size, but typically positive)
    fn insert(&mut self, range: Range<u64>, data: Vec<u8>, access_counter: u64) -> isize {
        let start = range.start;
        let end = range.end;
        let data_len = data.len();

        // Find and remove overlapping slices
        let mut to_remove = Vec::new();
        let mut merged_start = start;
        let mut merged_end = end;
        let mut merged_data: Option<Vec<u8>> = None;
        let mut bytes_removed: usize = 0;

        for (&slice_start, slice) in &self.slices {
            // Check for overlap
            if slice_start < end && slice.range.end > start {
                to_remove.push(slice_start);

                // Extend merged range
                merged_start = merged_start.min(slice_start);
                merged_end = merged_end.max(slice.range.end);
            }
        }

        // If we have overlaps, merge the data
        if !to_remove.is_empty() {
            let merged_len = (merged_end - merged_start) as usize;
            let mut new_data = vec![0u8; merged_len];

            // Copy existing slices
            for &slice_start in &to_remove {
                if let Some(slice) = self.slices.get(&slice_start) {
                    let offset = (slice_start - merged_start) as usize;
                    new_data[offset..offset + slice.data.len()].copy_from_slice(&slice.data);
                    bytes_removed += slice.data.len();
                    self.total_bytes -= slice.data.len();
                }
            }

            // Copy new data (overwrites any overlapping parts)
            let offset = (start - merged_start) as usize;
            new_data[offset..offset + data_len].copy_from_slice(&data);

            // Remove old slices
            for slice_start in to_remove {
                self.slices.remove(&slice_start);
            }

            merged_data = Some(new_data);
        }

        // Insert the (possibly merged) slice
        let (final_start, final_data) = if let Some(md) = merged_data {
            (merged_start, md)
        } else {
            (start, data)
        };

        let bytes_added = final_data.len();
        self.total_bytes += bytes_added;

        self.slices.insert(
            final_start,
            CachedSlice {
                range: final_start..final_start + bytes_added as u64,
                data: Arc::new(final_data),
                access_count: access_counter,
            },
        );

        // Return net change: bytes added minus bytes removed during merge
        bytes_added as isize - bytes_removed as isize
    }

    /// Evict least recently used slices to free up space
    fn evict_lru(&mut self, bytes_to_free: usize) -> usize {
        let mut freed = 0;

        while freed < bytes_to_free && !self.slices.is_empty() {
            // Find the slice with lowest access count
            let lru_key = self
                .slices
                .iter()
                .min_by_key(|(_, s)| s.access_count)
                .map(|(&k, _)| k);

            if let Some(key) = lru_key {
                if let Some(slice) = self.slices.remove(&key) {
                    freed += slice.data.len();
                    self.total_bytes -= slice.data.len();
                }
            } else {
                break;
            }
        }

        freed
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
    /// Per-file slice caches
    caches: Arc<RwLock<std::collections::HashMap<PathBuf, FileSliceCache>>>,
    /// Cached file sizes (avoids HEAD requests on lazy open)
    file_sizes: Arc<RwLock<std::collections::HashMap<PathBuf, u64>>>,
    /// Maximum total bytes to cache
    max_bytes: usize,
    /// Current total bytes cached
    current_bytes: Arc<RwLock<usize>>,
    /// Global access counter for LRU
    access_counter: Arc<RwLock<u64>>,
}

impl<D: Directory> SliceCachingDirectory<D> {
    /// Create a new slice-caching directory with the given memory limit
    pub fn new(inner: D, max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(inner),
            caches: Arc::new(RwLock::new(std::collections::HashMap::new())),
            file_sizes: Arc::new(RwLock::new(std::collections::HashMap::new())),
            max_bytes,
            current_bytes: Arc::new(RwLock::new(0)),
            access_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Get a reference to the inner directory
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Try to read from cache
    fn try_cache_read(&self, path: &Path, range: Range<u64>) -> Option<Vec<u8>> {
        let mut caches = self.caches.write();
        let mut counter = self.access_counter.write();

        if let Some(file_cache) = caches.get_mut(path) {
            file_cache.try_read(range, &mut counter)
        } else {
            None
        }
    }

    /// Insert into cache, evicting if necessary
    fn cache_insert(&self, path: &Path, range: Range<u64>, data: Vec<u8>) {
        let data_len = data.len();

        // Check if we need to evict
        {
            let current = *self.current_bytes.read();
            if current + data_len > self.max_bytes {
                self.evict_to_fit(data_len);
            }
        }

        let mut caches = self.caches.write();
        let counter = *self.access_counter.read();

        let file_cache = caches
            .entry(path.to_path_buf())
            .or_insert_with(FileSliceCache::new);

        let net_change = file_cache.insert(range, data, counter);
        let mut current = self.current_bytes.write();
        if net_change >= 0 {
            *current += net_change as usize;
        } else {
            *current = current.saturating_sub((-net_change) as usize);
        }
    }

    /// Evict slices to make room for new data
    fn evict_to_fit(&self, needed: usize) {
        let mut caches = self.caches.write();
        let mut current = self.current_bytes.write();

        let target = if *current + needed > self.max_bytes {
            (*current + needed) - self.max_bytes
        } else {
            return;
        };

        let mut freed = 0;

        // Evict from all file caches until we have enough space
        while freed < target {
            // Find the file cache with the oldest access
            let oldest_file = caches
                .iter()
                .filter(|(_, fc)| !fc.slices.is_empty())
                .min_by_key(|(_, fc)| {
                    fc.slices
                        .values()
                        .map(|s| s.access_count)
                        .min()
                        .unwrap_or(u64::MAX)
                })
                .map(|(p, _)| p.clone());

            if let Some(path) = oldest_file {
                if let Some(file_cache) = caches.get_mut(&path) {
                    freed += file_cache.evict_lru(target - freed);
                }
            } else {
                break;
            }
        }

        *current = current.saturating_sub(freed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> SliceCacheStats {
        let caches = self.caches.read();
        let mut total_slices = 0;
        let mut files_cached = 0;

        for fc in caches.values() {
            if !fc.slices.is_empty() {
                files_cached += 1;
                total_slices += fc.slices.len();
            }
        }

        SliceCacheStats {
            total_bytes: *self.current_bytes.read(),
            max_bytes: self.max_bytes,
            total_slices,
            files_cached,
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
        let caches = self.caches.read();
        let file_sizes = self.file_sizes.read();
        let mut buf = Vec::new();

        // Magic and version
        buf.extend_from_slice(SLICE_CACHE_MAGIC);
        buf.extend_from_slice(&SLICE_CACHE_VERSION.to_le_bytes());

        // Count non-empty caches
        let non_empty: Vec<_> = caches
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

        // Check version (support v1 and v2)
        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        if version != 1 && version != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported slice cache version: {}", version),
            ));
        }
        pos += 4;

        // Number of files
        let num_files = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut caches = self.caches.write();
        let mut current_bytes = self.current_bytes.write();
        let counter = *self.access_counter.read();

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

            // File cache
            let (file_cache, consumed) = FileSliceCache::deserialize(&data[pos..], counter)?;
            pos += consumed;

            // Merge into existing cache
            *current_bytes += file_cache.total_bytes;
            caches.insert(path, file_cache);
        }

        // v2: Load file sizes if present
        if version >= 2 && pos + 4 <= data.len() {
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
        *self.current_bytes.read() == 0
    }

    /// Clear all cached data
    pub fn clear(&self) {
        let mut caches = self.caches.write();
        let mut current_bytes = self.current_bytes.write();
        caches.clear();
        *current_bytes = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct SliceCacheStats {
    pub total_bytes: usize,
    pub max_bytes: usize,
    pub total_slices: usize,
    pub files_cached: usize,
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

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        // Check if we have the full file cached (use our caching file_size)
        let file_size = self.file_size(path).await?;
        let full_range = 0..file_size;

        // Try cache first for full file
        if let Some(data) = self.try_cache_read(path, full_range.clone()) {
            return Ok(FileSlice::new(OwnedBytes::new(data)));
        }

        // Read from inner
        let slice = self.inner.open_read(path).await?;
        let bytes = slice.read_bytes().await?;

        // Cache the full file
        self.cache_insert(path, full_range, bytes.as_slice().to_vec());

        Ok(FileSlice::new(bytes))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        // Try cache first
        if let Some(data) = self.try_cache_read(path, range.clone()) {
            return Ok(OwnedBytes::new(data));
        }

        // Read from inner
        let data = self.inner.read_range(path, range.clone()).await?;

        // Cache the result
        self.cache_insert(path, range, data.as_slice().to_vec());

        Ok(data)
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle> {
        // Get file size (uses cache to avoid HEAD requests)
        let file_size = self.file_size(path).await?;

        // Create a caching wrapper around the inner directory's read_range
        let path_buf = path.to_path_buf();
        let caches = Arc::clone(&self.caches);
        let current_bytes = Arc::clone(&self.current_bytes);
        let access_counter = Arc::clone(&self.access_counter);
        let max_bytes = self.max_bytes;
        let inner = Arc::clone(&self.inner);

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let path = path_buf.clone();
            let caches = Arc::clone(&caches);
            let current_bytes = Arc::clone(&current_bytes);
            let access_counter = Arc::clone(&access_counter);
            let inner = Arc::clone(&inner);

            Box::pin(async move {
                // Try cache first
                {
                    let mut caches_guard = caches.write();
                    let mut counter = access_counter.write();
                    if let Some(file_cache) = caches_guard.get_mut(&path)
                        && let Some(data) = file_cache.try_read(range.clone(), &mut counter)
                    {
                        log::trace!("Cache HIT: {:?} [{}-{}]", path, range.start, range.end);
                        return Ok(OwnedBytes::new(data));
                    }
                }

                log::trace!("Cache MISS: {:?} [{}-{}]", path, range.start, range.end);

                // Read from inner
                let data = inner.read_range(&path, range.clone()).await?;

                // Cache the result
                let data_len = data.as_slice().len();
                {
                    // Check if we need to evict
                    let current = *current_bytes.read();
                    if current + data_len > max_bytes {
                        // Simple eviction: just skip caching if over limit
                        // Full eviction would require more complex logic here
                    } else {
                        let mut caches_guard = caches.write();
                        let counter = *access_counter.read();

                        let file_cache = caches_guard
                            .entry(path.clone())
                            .or_insert_with(FileSliceCache::new);

                        let net_change =
                            file_cache.insert(range, data.as_slice().to_vec(), counter);
                        let mut current = current_bytes.write();
                        if net_change >= 0 {
                            *current += net_change as usize;
                        } else {
                            *current = current.saturating_sub((-net_change) as usize);
                        }
                    }
                }

                Ok(data)
            })
        });

        Ok(LazyFileHandle::new(file_size, read_fn))
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
}
