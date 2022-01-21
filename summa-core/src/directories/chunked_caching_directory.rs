use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fmt, io};

use parking_lot::lock_api::RwLockWriteGuard;
use parking_lot::{RawRwLock, RwLock, RwLockUpgradableReadGuard};
use tantivy::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use tantivy::directory::{DirectoryLock, FileHandle, FileSlice, Lock, OwnedBytes, WatchCallback, WatchHandle, WritePtr};
use tantivy::{Directory, HasLen};

use crate::directories::chunk_generator::{Chunk, ChunkGenerator};
use crate::directories::requests_composer::{Request, RequestsComposer};
use crate::directories::MemorySizedCache;
use crate::metrics::CacheMetrics;

#[derive(Default, Debug, Clone)]
pub struct FileStat {
    pub file_length: Option<u64>,
    pub generation: u32,
}

#[derive(Default, Debug, Clone)]
pub struct MaterializedFileStat {
    pub file_length: u64,
    pub generation: u32,
}

impl FileStat {
    pub fn inc_gen(&mut self, new_len: Option<u64>) {
        self.file_length = new_len;
        self.generation += 1;
    }
}

#[derive(Default, Clone)]
pub struct FileStats(Arc<RwLock<HashMap<PathBuf, FileStat>>>);

impl FileStats {
    pub fn from_file_lengths(file_lengths: HashMap<PathBuf, u64>) -> Self {
        FileStats(Arc::new(RwLock::new(HashMap::from_iter(file_lengths.into_iter().map(|(k, v)| {
            (
                k,
                FileStat {
                    file_length: Some(v),
                    generation: 0,
                },
            )
        })))))
    }

    pub fn inc_gen(&self, path: &Path, new_len: Option<u64>) -> RwLockWriteGuard<'_, RawRwLock, HashMap<PathBuf, FileStat>> {
        let mut write_lock = self.0.write();
        write_lock.entry(path.to_path_buf()).or_insert_with(Default::default).inc_gen(new_len);
        write_lock
    }

    pub fn get_or_set(&self, path: &Path, f: impl FnOnce() -> usize) -> MaterializedFileStat {
        let read_lock = self.0.upgradable_read();
        let file_stat = read_lock.get(path);
        let file_length = file_stat.and_then(|file_stat| file_stat.file_length);
        let generation = file_stat.map(|file_stat| file_stat.generation).unwrap_or_default();
        match file_length {
            None => {
                let file_length = f() as u64;
                let file_stat = FileStat {
                    file_length: Some(file_length),
                    generation,
                };
                RwLockUpgradableReadGuard::upgrade(read_lock).insert(path.to_path_buf(), file_stat);
                MaterializedFileStat { file_length, generation }
            }
            Some(file_length) => MaterializedFileStat { file_length, generation },
        }
    }
}

/// Caching layer that emits aligned requests to downstream directory.
///
/// Alignment of downstream requests makes possible to cache response chunks and reuse them in further requests
/// For example, for `chunk_size` set to 16 if you call `read_bytes(13..18)` will lead to reading two chunks of data
/// 0..16 and 16..32. Then, both of these chunks will be stored in cache and may be used to serve further request
/// laying inside the cached interval, such as `read_bytes(3..9)` or `read_bytes(19..32)`
pub struct ChunkedCachingDirectory {
    chunk_size: usize,
    cache: Option<Arc<MemorySizedCache>>,
    underlying: Box<dyn Directory>,
    file_stats: FileStats,
}

impl Clone for ChunkedCachingDirectory {
    fn clone(&self) -> Self {
        ChunkedCachingDirectory {
            chunk_size: self.chunk_size,
            cache: self.cache.clone(),
            underlying: self.underlying.box_clone(),
            file_stats: self.file_stats.clone(),
        }
    }
}

impl ChunkedCachingDirectory {
    /// Create chunking layer without caching
    ///
    /// Cache-less chunking may be useful if downstream directory is responsible for caching and works better with
    /// chunked requests. Most obvious example is directory-over-HTTP which Range requests may be cached by browser.
    pub fn new(underlying: Box<dyn Directory>, chunk_size: usize, file_stats: FileStats) -> ChunkedCachingDirectory {
        ChunkedCachingDirectory {
            chunk_size,
            cache: None,
            underlying,
            file_stats,
        }
    }

    /// Bounded cache
    pub fn new_with_capacity_in_bytes(
        underlying: Box<dyn Directory>,
        chunk_size: usize,
        capacity_in_bytes: usize,
        cache_metrics: CacheMetrics,
        file_stats: FileStats,
    ) -> ChunkedCachingDirectory {
        ChunkedCachingDirectory {
            chunk_size,
            cache: Some(Arc::new(MemorySizedCache::with_capacity_in_bytes(capacity_in_bytes, cache_metrics))),
            underlying,
            file_stats,
        }
    }

    /// Unbounded cache
    pub fn new_unbounded(underlying: Box<dyn Directory>, chunk_size: usize, cache_metrics: CacheMetrics, file_stats: FileStats) -> ChunkedCachingDirectory {
        ChunkedCachingDirectory {
            chunk_size,
            cache: Some(Arc::new(MemorySizedCache::with_infinite_capacity(cache_metrics))),
            underlying,
            file_stats,
        }
    }
}

impl Debug for ChunkedCachingDirectory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_struct("ChunkedCachingDirectory")
            .field("chunk_size", &self.chunk_size)
            .field("underlying", &self.underlying)
            .finish()
    }
}

#[async_trait]
impl Directory for ChunkedCachingDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let underlying_filehandle = self.underlying.get_file_handle(path)?;
        let underlying_filehandle_ref = underlying_filehandle.as_ref();
        let file_stat = self.file_stats.get_or_set(path, || underlying_filehandle_ref.len());
        Ok(Arc::new(ChunkedCachingFileHandle {
            path: path.to_path_buf(),
            cache: self.cache.clone(),
            chunk_size: self.chunk_size,
            len: file_stat.file_length as usize,
            generation: file_stat.generation,
            underlying_filehandle,
        }))
    }

    fn open_read(&self, path: &Path) -> Result<FileSlice, OpenReadError> {
        self.underlying.open_read(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        // ToDo: May evict caches
        let _lock = self.file_stats.inc_gen(path, None);
        self.underlying.delete(path)
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.underlying.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        let _lock = self.file_stats.inc_gen(path, None);
        self.underlying.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        let owned_bytes = file_handle
            .read_bytes(0..file_handle.len())
            .map_err(|io_error| OpenReadError::wrap_io_error(io_error, path.to_path_buf()))?;
        Ok(owned_bytes.as_slice().to_vec())
    }

    async fn atomic_read_async(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        let owned_bytes = file_handle
            .read_bytes_async(0..file_handle.len())
            .await
            .map_err(|io_error| OpenReadError::wrap_io_error(io_error, path.to_path_buf()))?;
        Ok(owned_bytes.as_slice().to_vec())
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let _lock = self.file_stats.inc_gen(path, None);
        self.underlying.atomic_write(path, data)
    }

    fn sync_directory(&self) -> io::Result<()> {
        self.underlying.sync_directory()
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.underlying.acquire_lock(lock)
    }

    fn watch(&self, callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.underlying.watch(callback)
    }
}

struct ChunkedCachingFileHandle {
    path: PathBuf,
    cache: Option<Arc<MemorySizedCache>>,
    /// Chunk size
    chunk_size: usize,
    underlying_filehandle: Arc<dyn FileHandle>,
    /// Generation is incremented after each write to file and using for differeing between various versions of file
    generation: u32,
    len: usize,
}

impl Debug for ChunkedCachingFileHandle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "ChunkedCachingFileHandle(path={:?}, underlying={:?}, len={})",
            &self.path,
            self.underlying_filehandle.as_ref(),
            self.len
        )
    }
}

impl ChunkedCachingFileHandle {
    fn try_fill_from_cache(&self, range: Range<usize>, response_buffer: &mut [u8]) -> Vec<Chunk> {
        let chunks = ChunkGenerator::new(range, self.len(), self.chunk_size);
        match &self.cache {
            None => chunks.collect(),
            Some(cache) => {
                let mut missing_chunks = vec![];
                for chunk in chunks {
                    match cache.get_slice(&self.path, self.generation, chunk.index) {
                        Some(item) => response_buffer[chunk.target_ix..][..chunk.len()].clone_from_slice(&item.slice(chunk.data_bounds())),
                        None => missing_chunks.push(chunk),
                    };
                }
                missing_chunks
            }
        }
    }

    fn adopt_response(&self, total_response: &mut [u8], response: OwnedBytes, original_request: &Request) {
        for chunk in original_request.chunks() {
            let item = response.slice(chunk.shifted_chunk_range(original_request.bounds().start));
            total_response[chunk.target_ix..][..chunk.len()].clone_from_slice(&item.slice(chunk.data_bounds()));
            if let Some(cache) = &self.cache {
                cache.put_slice(self.path.to_path_buf(), self.generation, chunk.index, item);
            }
        }
    }
}

#[async_trait]
impl FileHandle for ChunkedCachingFileHandle {
    fn read_bytes(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        let mut response_buffer = vec![0; range.end - range.start];
        let missing_chunks = self.try_fill_from_cache(range, &mut response_buffer);

        for missing_chunks_request in RequestsComposer::for_chunks(missing_chunks).requests() {
            let missing_chunks_response = self.underlying_filehandle.read_bytes(missing_chunks_request.bounds())?;
            self.adopt_response(&mut response_buffer, missing_chunks_response, &missing_chunks_request)
        }
        Ok(OwnedBytes::new(response_buffer))
    }

    async fn read_bytes_async(&self, range: Range<usize>) -> io::Result<OwnedBytes> {
        let mut response_buffer = vec![0; range.end - range.start];
        let missing_chunks = self.try_fill_from_cache(range, &mut response_buffer);

        for missing_chunks_request in RequestsComposer::for_chunks(missing_chunks).requests() {
            let missing_chunks_response = self.underlying_filehandle.read_bytes_async(missing_chunks_request.bounds()).await?;
            self.adopt_response(&mut response_buffer, missing_chunks_response, &missing_chunks_request)
        }

        Ok(OwnedBytes::new(response_buffer))
    }
}

impl HasLen for ChunkedCachingFileHandle {
    fn len(&self) -> usize {
        self.len
    }
}
