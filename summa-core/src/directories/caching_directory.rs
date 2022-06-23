use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{fmt, io};

use async_trait::async_trait;
use parking_lot::lock_api::RwLockWriteGuard;
use parking_lot::{RawRwLock, RwLock, RwLockUpgradableReadGuard};
use tantivy::directory::error::{DeleteError, LockError, OpenReadError, OpenWriteError};
use tantivy::directory::{DirectoryLock, FileHandle, Lock, OwnedBytes, WatchCallback, WatchHandle, WritePtr};
use tantivy::{Directory, HasLen};

use crate::directories::byte_range_cache::ByteRangeCache;

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

    pub fn get_or_set(&self, path: &Path, f: impl FnOnce() -> u64) -> MaterializedFileStat {
        let read_lock = self.0.upgradable_read();
        let file_stat = read_lock.get(path);
        let file_length = file_stat.and_then(|file_stat| file_stat.file_length);
        let generation = file_stat.map(|file_stat| file_stat.generation).unwrap_or_default();
        match file_length {
            None => {
                let file_length = f();
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

/// The caching directory is a simple cache that wraps another directory.
#[derive(Clone)]
pub struct CachingDirectory {
    underlying: Arc<dyn Directory>,
    cache: Arc<ByteRangeCache>,
    file_stats: FileStats,
}

impl CachingDirectory {
    /// Creates a new CachingDirectory.
    ///
    /// Warming: The resulting CacheDirectory will cache all information without ever
    /// removing any item from the cache.
    pub fn bounded(underlying: Arc<dyn Directory>, _capacity_in_bytes: usize, file_stats: FileStats) -> CachingDirectory {
        CachingDirectory {
            underlying,
            cache: Arc::new(ByteRangeCache::with_infinite_capacity()),
            file_stats,
        }
    }

    pub fn unbounded(underlying: Arc<dyn Directory>, file_stats: FileStats) -> CachingDirectory {
        CachingDirectory {
            underlying,
            cache: Arc::new(ByteRangeCache::with_infinite_capacity()),
            file_stats,
        }
    }
}

impl fmt::Debug for CachingDirectory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CachingDirectory({:?})", self.underlying)
    }
}

struct CachingFileHandle {
    path: PathBuf,
    cache: Arc<ByteRangeCache>,
    underlying_filehandle: Arc<dyn FileHandle>,
    generation: u32,
    len: u64,
}

impl CachingFileHandle {
    pub fn get_key(&self) -> PathBuf {
        PathBuf::from(format!("{}@{}", self.path.to_string_lossy(), self.generation))
    }
}

impl fmt::Debug for CachingFileHandle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CachingFileHandle(path={:?}, underlying={:?})",
            &self.path,
            self.underlying_filehandle.as_ref()
        )
    }
}

#[async_trait]
impl FileHandle for CachingFileHandle {
    fn read_bytes(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        if let Some(bytes) = self.cache.get_slice(&self.get_key(), byte_range.clone()) {
            return Ok(bytes);
        }
        let owned_bytes = self.underlying_filehandle.read_bytes(byte_range.clone())?;
        self.cache.put_slice(self.get_key(), byte_range, owned_bytes.clone());
        Ok(owned_bytes)
    }

    async fn read_bytes_async(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        if let Some(owned_bytes) = self.cache.get_slice(&self.get_key(), byte_range.clone()) {
            return Ok(owned_bytes);
        }
        let read_bytes = self.underlying_filehandle.read_bytes_async(byte_range.clone()).await?;
        self.cache.put_slice(self.get_key(), byte_range, read_bytes.clone());
        Ok(read_bytes)
    }
}

impl HasLen for CachingFileHandle {
    fn len(&self) -> u64 {
        self.len
    }
}

#[async_trait]
impl Directory for CachingDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let underlying_filehandle = self.underlying.get_file_handle(path)?;
        let underlying_filehandle_ref = underlying_filehandle.as_ref();
        let file_stat = self.file_stats.get_or_set(path, || underlying_filehandle_ref.len());
        Ok(Arc::new(CachingFileHandle {
            path: path.to_path_buf(),
            cache: self.cache.clone(),
            len: file_stat.file_length,
            generation: file_stat.generation,
            underlying_filehandle,
        }))
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn underlying_directory(&self) -> Option<&dyn Directory> {
        Some(self.underlying.as_ref())
    }

    fn real_directory(&self) -> &dyn Directory {
        self.underlying.real_directory()
    }
}
