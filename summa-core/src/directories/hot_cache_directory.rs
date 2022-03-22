use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fmt, io};

use serde::{Deserialize, Serialize};
use tantivy::directory::error::{LockError, OpenReadError};
use tantivy::directory::{DirectoryLock, FileHandle, FileSlice, Lock, OwnedBytes, WatchCallback, WatchHandle};
use tantivy::error::DataCorruption;
use tantivy::{Directory, HasLen, Index, IndexReader, Opstamp, ReloadPolicy};

use super::debug_proxy_directory::DebugProxyDirectory;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SliceCacheIndexEntry {
    start: u64, //< legacy. We keep this instead of range due to existing indices.
    stop: u64,
    addr: u64,
}

impl SliceCacheIndexEntry {
    pub fn len(&self) -> usize {
        (self.stop - self.start) as usize
    }

    pub fn range(&self) -> Range<u64> {
        self.start..self.stop
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct SliceCacheIndex {
    total_len: u64,
    slices: Vec<SliceCacheIndexEntry>,
}
impl SliceCacheIndex {
    pub fn is_complete(&self) -> bool {
        if self.slices.len() != 1 {
            return false;
        }
        self.slices[0].len() as u64 == self.total_len
    }

    pub fn get(&self, byte_range: Range<u64>) -> Option<usize> {
        let entry_idx = match self.slices.binary_search_by_key(&byte_range.start, |entry| entry.range().start) {
            Ok(idx) => idx,
            Err(0) => {
                return None;
            }
            Err(idx_after) => idx_after - 1,
        };
        let entry = &self.slices[entry_idx];
        if entry.range().start > byte_range.start || entry.range().end < byte_range.end {
            return None;
        }
        Some((entry.addr + byte_range.start - entry.range().start) as usize)
    }
}

#[derive(Default)]
struct StaticDirectoryCacheBuilder {
    file_cache_builder: HashMap<PathBuf, StaticSliceCacheBuilder>,
    file_lengths: HashMap<PathBuf, u64>, // a mapping from file path to file size in bytes
}

impl StaticDirectoryCacheBuilder {
    pub fn add_file(&mut self, path: &Path, file_len: u64) -> &mut StaticSliceCacheBuilder {
        self.file_lengths.insert(path.to_owned(), file_len);
        self.file_cache_builder
            .entry(path.to_owned())
            .or_insert_with(|| StaticSliceCacheBuilder::new(file_len))
    }

    /// Flush needs to be called afterwards.
    pub fn write(self, wrt: &mut dyn io::Write) -> tantivy::Result<()> {
        // Write format version
        wrt.write_all(b"\x00")?;

        let file_lengths_bytes = serde_cbor::to_vec(&self.file_lengths).expect("CBOR failed");
        wrt.write_all(&(file_lengths_bytes.len() as u64).to_le_bytes())?;
        wrt.write_all(&file_lengths_bytes[..])?;

        let mut data_buffer = Vec::new();
        let mut data_idx: Vec<(PathBuf, u64)> = Vec::new();
        let mut offset = 0u64;
        for (path, cache) in self.file_cache_builder {
            let buf = cache.flush()?;
            data_idx.push((path, offset));
            offset += buf.len() as u64;
            data_buffer.extend_from_slice(&buf);
        }
        let idx_bytes = serde_cbor::to_vec(&data_idx).expect("CBOR failed");
        wrt.write_all(&(idx_bytes.len() as u64).to_le_bytes())?;
        wrt.write_all(&idx_bytes[..])?;
        wrt.write_all(&data_buffer[..])?;

        Ok(())
    }
}

pub fn deserialize_cbor<T>(bytes: &mut OwnedBytes) -> serde_cbor::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let len = bytes.read_u64();
    let value = serde_cbor::from_reader(&bytes.as_slice()[..len as usize]);
    bytes.advance(len as usize);
    value
}

pub struct StaticDirectoryCache {
    file_lengths: HashMap<PathBuf, u64>,
    slices: HashMap<PathBuf, Arc<StaticSliceCache>>,
}

impl Debug for StaticDirectoryCache {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticDirectoryCache")
            .field("files", &self.file_lengths.len())
            .field("slices", &self.slices.len())
            .finish()
    }
}

impl StaticDirectoryCache {
    pub fn open(mut bytes: OwnedBytes, opstamp: Opstamp) -> tantivy::Result<StaticDirectoryCache> {
        let format_version = bytes.read_u8();
        let bytes_len = bytes.len();

        if format_version != 0 {
            return Err(tantivy::TantivyError::DataCorruption(DataCorruption::comment_only(format!(
                "Format version not supported: `{format_version}`"
            ))));
        }

        let mut file_lengths: HashMap<PathBuf, u64> = deserialize_cbor(&mut bytes).expect("CBOR failed");
        file_lengths.insert(PathBuf::from(format!("hotcache.{}.bin", opstamp)), bytes_len as u64);

        let mut slice_offsets: Vec<(PathBuf, u64)> = deserialize_cbor(&mut bytes).expect("CBOR failed");
        slice_offsets.push((PathBuf::default(), bytes.len() as u64));

        let slices = slice_offsets
            .windows(2)
            .map(|slice_offsets_window| {
                let path = slice_offsets_window[0].0.clone();
                let start = slice_offsets_window[0].1 as usize;
                let end = slice_offsets_window[1].1 as usize;
                StaticSliceCache::open(bytes.slice(start..end)).map(|s| (path, Arc::new(s)))
            })
            .collect::<tantivy::Result<_>>()?;

        Ok(StaticDirectoryCache { file_lengths, slices })
    }

    pub fn get_slice(&self, path: &Path) -> Arc<StaticSliceCache> {
        self.slices.get(path).cloned().unwrap_or_default()
    }

    pub fn get_file_length(&self, path: &Path) -> Option<u64> {
        self.file_lengths.get(path).cloned()
    }

    pub fn file_lengths(&self) -> &HashMap<PathBuf, u64> {
        &self.file_lengths
    }
}

/// A SliceCache is a static toring
pub struct StaticSliceCache {
    bytes: OwnedBytes,
    index: SliceCacheIndex,
}

impl Default for StaticSliceCache {
    fn default() -> StaticSliceCache {
        StaticSliceCache {
            bytes: OwnedBytes::empty(),
            index: SliceCacheIndex::default(),
        }
    }
}

impl StaticSliceCache {
    pub fn open(owned_bytes: OwnedBytes) -> tantivy::Result<Self> {
        let owned_bytes_len = owned_bytes.len();
        assert!(owned_bytes_len >= 8);
        let (body, len_bytes) = owned_bytes.split(owned_bytes_len - 8);
        let mut body_len_bytes = [0u8; 8];
        body_len_bytes.copy_from_slice(len_bytes.as_slice());
        let body_len = u64::from_le_bytes(body_len_bytes);
        let (body, idx) = body.split(body_len as usize);
        let mut idx_bytes = idx.as_slice();
        let index: SliceCacheIndex =
            serde_cbor::from_reader(&mut idx_bytes).map_err(|err| DataCorruption::comment_only(format!("Failed to deserialize the slice index: {err:?}")))?;
        Ok(StaticSliceCache { bytes: body, index })
    }

    pub fn try_read_all(&self) -> Option<OwnedBytes> {
        if !self.index.is_complete() {
            return None;
        }
        Some(self.bytes.clone())
    }

    pub fn try_read_bytes(&self, byte_range: Range<u64>) -> Option<OwnedBytes> {
        if byte_range.is_empty() {
            return Some(OwnedBytes::empty());
        }
        if let Some(start) = self.index.get(byte_range.clone()) {
            return Some(self.bytes.slice(start..(start + (byte_range.end - byte_range.start) as usize)));
        }
        None
    }
}

struct StaticSliceCacheBuilder {
    wrt: Vec<u8>,
    slices: Vec<SliceCacheIndexEntry>,
    offset: u64,
    total_len: u64,
}

impl StaticSliceCacheBuilder {
    pub fn new(total_len: u64) -> StaticSliceCacheBuilder {
        StaticSliceCacheBuilder {
            wrt: Vec::new(),
            slices: Vec::new(),
            offset: 0u64,
            total_len,
        }
    }

    pub fn add_bytes(&mut self, bytes: &[u8], start: u64) {
        self.wrt.extend_from_slice(bytes);
        let end = start + bytes.len() as u64;
        self.slices.push(SliceCacheIndexEntry {
            start,
            stop: end,
            addr: self.offset,
        });
        self.offset += bytes.len() as u64;
    }

    fn merged_slices(&mut self) -> tantivy::Result<Vec<SliceCacheIndexEntry>> {
        if self.slices.is_empty() {
            return Ok(Vec::new());
        }
        self.slices.sort_unstable_by_key(|e| e.range().start);
        let mut slices = Vec::with_capacity(self.slices.len());
        let mut last = self.slices[0].clone();
        for segment in &self.slices[1..] {
            if segment.range().start < last.range().end {
                return Err(tantivy::TantivyError::InvalidArgument(format!(
                    "Two segments are overlapping on byte {}",
                    segment.range().start
                )));
            }
            if last.stop == segment.range().start && (last.addr + (last.range().end - last.range().start) == segment.addr) {
                // We merge the current segment with the previous one
                last.stop += segment.range().end - segment.range().start;
            } else {
                slices.push(last);
                last = segment.clone();
            }
        }
        slices.push(last);
        Ok(slices)
    }

    pub fn flush(mut self) -> tantivy::Result<Vec<u8>> {
        let merged_slices = self.merged_slices()?;
        let slices_idx = SliceCacheIndex {
            total_len: self.total_len,
            slices: merged_slices,
        };
        serde_cbor::to_writer(&mut self.wrt, &slices_idx).map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
        self.wrt.extend_from_slice(&self.offset.to_le_bytes()[..]);
        Ok(self.wrt)
    }
}

impl Debug for StaticSliceCache {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SliceCache()")
    }
}

/// The hot directory accelerates a given directory,
/// by placing a static cache in front of a directory.
///
/// The `HotDirectory` does not implement write operations. It has been
/// designed for quickwit in order to regroup all of the small random
/// read operations required to open an index.
/// All of these operations are gather into a single file called the
/// hotcache.
#[derive(Clone)]
pub struct HotDirectory {
    inner: Box<InnerHotDirectory>,
}

impl HotDirectory {
    /// Wraps an index, with a static cache serialized into `hot_cache_bytes`.
    pub fn open(underlying: Box<dyn Directory>, static_cache: StaticDirectoryCache) -> tantivy::Result<HotDirectory> {
        Ok(HotDirectory {
            inner: Box::new(InnerHotDirectory {
                underlying,
                cache: Arc::new(static_cache),
            }),
        })
    }
}

struct FileSliceWithCache {
    underlying: FileSlice,
    static_cache: Arc<StaticSliceCache>,
    file_length: u64,
}

#[async_trait]
impl FileHandle for FileSliceWithCache {
    fn read_bytes(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        if let Some(found_bytes) = self.static_cache.try_read_bytes(byte_range.clone()) {
            return Ok(found_bytes);
        }
        self.underlying.read_bytes_slice(byte_range)
    }
    async fn read_bytes_async(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        if let Some(found_bytes) = self.static_cache.try_read_bytes(byte_range.clone()) {
            return Ok(found_bytes);
        }
        self.underlying.read_bytes_slice_async(byte_range).await
    }
}

impl Debug for FileSliceWithCache {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "FileSliceWithCache({:?})", &self.underlying)
    }
}

impl HasLen for FileSliceWithCache {
    fn len(&self) -> u64 {
        self.file_length
    }
}

struct InnerHotDirectory {
    underlying: Box<dyn Directory>,
    cache: Arc<StaticDirectoryCache>,
}

impl Debug for HotDirectory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("HotDirectory").field("underlying", &self.inner.underlying).finish()
    }
}

impl Clone for InnerHotDirectory {
    fn clone(&self) -> Self {
        InnerHotDirectory {
            underlying: self.underlying.box_clone(),
            cache: self.cache.clone(),
        }
    }
}

#[async_trait]
impl Directory for HotDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let file_length = self
            .inner
            .cache
            .get_file_length(path)
            .ok_or_else(|| OpenReadError::FileDoesNotExist(path.to_owned()))?;
        let underlying_filehandle = self.inner.underlying.get_file_handle(path)?;
        let underlying = FileSlice::new_with_num_bytes(underlying_filehandle, file_length);
        let file_slice_with_cache = FileSliceWithCache {
            underlying,
            static_cache: self.inner.cache.get_slice(path),
            file_length,
        };
        Ok(Arc::new(file_slice_with_cache))
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.inner.cache.get_file_length(path).is_some() || self.inner.underlying.exists(path)?)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let slice_cache = self.inner.cache.get_slice(path);
        if let Some(all_bytes) = slice_cache.try_read_all() {
            return Ok(all_bytes.as_slice().to_owned());
        }
        self.inner.underlying.atomic_read(path)
    }

    async fn atomic_read_async(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let slice_cache = self.inner.cache.get_slice(path);
        if let Some(all_bytes) = slice_cache.try_read_all() {
            return Ok(all_bytes.as_slice().to_owned());
        }
        self.inner.underlying.atomic_read_async(path).await
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.inner.underlying.acquire_lock(lock)
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.inner.underlying.watch(watch_callback)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn underlying_directory(&self) -> Option<&dyn Directory> {
        Some(self.inner.underlying.as_ref())
    }

    fn real_directory(&self) -> &dyn Directory {
        self.inner.underlying.real_directory()
    }
}

fn list_index_files(index: &Index) -> tantivy::Result<HashSet<PathBuf>> {
    let index_meta = index.load_metas()?;
    let mut files: HashSet<PathBuf> = index_meta.segments.into_iter().flat_map(|segment_meta| segment_meta.list_files()).collect();
    files.insert(Path::new("meta.json").to_path_buf());
    files.insert(Path::new(".managed.json").to_path_buf());
    Ok(files)
}

/// Given a tantivy directory, automatically identify the parts that should be loaded on startup
/// and writes a static cache file called hotcache in the `output`.
///
/// See [`HotDirectory`] for more information.
pub fn create_hotcache(directory: Box<dyn Directory>) -> tantivy::Result<Vec<u8>> {
    // We use the caching directory here in order to defensively ensure that
    // the content of the directory that will be written in the hotcache is precisely
    // the same that was read on the first pass.
    let debug_proxy_directory = DebugProxyDirectory::wrap(directory);
    let index = Index::open(debug_proxy_directory.clone())?;
    let schema = index.schema();
    let reader: IndexReader = index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
    let searcher = reader.searcher();
    for (field, field_entry) in schema.fields() {
        if !field_entry.is_indexed() {
            continue;
        }
        for reader in searcher.segment_readers() {
            let _inv_idx = reader.inverted_index(field)?;
        }
    }
    let mut cache_builder = StaticDirectoryCacheBuilder::default();
    let read_operations = debug_proxy_directory.drain_read_operations();
    let mut per_file_slices: HashMap<PathBuf, HashSet<Range<u64>>> = HashMap::default();
    for read_operation in read_operations {
        per_file_slices
            .entry(read_operation.path)
            .or_default()
            .insert(read_operation.offset..read_operation.offset + read_operation.num_bytes as u64);
    }
    let index_files = list_index_files(&index)?;
    for file_path in index_files {
        let file_slice_res = debug_proxy_directory.open_read(&file_path);
        if let Err(OpenReadError::FileDoesNotExist(_)) = file_slice_res {
            continue;
        }
        let file_slice = file_slice_res?;
        let file_cache_builder = cache_builder.add_file(&file_path, file_slice.len());
        if let Some(intervals) = per_file_slices.get(&file_path) {
            for byte_range in intervals {
                let len = byte_range.end - byte_range.start;
                // We do not want to store slices that are too large in the hotcache,
                // but on the other hand, the term dictionray index and the docstore
                // index are required for quickwit to work.
                //
                // Warning: we need to work on string here because `Path::ends_with`
                // has very different semantics.
                let file_path_str = file_path.to_string_lossy();
                if file_path_str.ends_with("store") || file_path_str.ends_with("term") || len < 10_000_000 {
                    let bytes = file_slice.read_bytes_slice(byte_range.clone())?;
                    file_cache_builder.add_bytes(bytes.as_slice(), byte_range.start);
                }
            }
        }
    }
    let mut hotcache_bytes = vec![];
    cache_builder.write(&mut hotcache_bytes)?;
    Ok(hotcache_bytes)
}
