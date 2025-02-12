//! Async Directory abstraction for IO operations
//!
//! Supports network, local filesystem, and in-memory storage.
//! All reads are async to minimize blocking on network latency.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Callback type for lazy range reading
#[cfg(not(target_arch = "wasm32"))]
pub type RangeReadFn = Arc<
    dyn Fn(
            Range<u64>,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<OwnedBytes>> + Send>>
        + Send
        + Sync,
>;

#[cfg(target_arch = "wasm32")]
pub type RangeReadFn = Arc<
    dyn Fn(
        Range<u64>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<OwnedBytes>>>>,
>;

/// Unified file handle for both inline (mmap/RAM) and lazy (HTTP/filesystem) access.
///
/// Replaces the previous `FileSlice`, `LazyFileHandle`, and `LazyFileSlice` types.
/// - **Inline**: data is available synchronously (mmap, RAM). Sync reads via `read_bytes_range_sync`.
/// - **Lazy**: data is fetched on-demand via async callback (HTTP, filesystem).
///
/// Use `.slice()` to create sub-range views (zero-copy for Inline, offset-adjusted for Lazy).
#[derive(Clone)]
pub struct FileHandle {
    inner: FileHandleInner,
}

#[derive(Clone)]
enum FileHandleInner {
    /// Data available inline — sync reads possible (mmap, RAM)
    Inline {
        data: OwnedBytes,
        offset: u64,
        len: u64,
    },
    /// Data fetched on-demand via async callback (HTTP, filesystem)
    Lazy {
        read_fn: RangeReadFn,
        offset: u64,
        len: u64,
    },
}

impl std::fmt::Debug for FileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            FileHandleInner::Inline { len, offset, .. } => f
                .debug_struct("FileHandle::Inline")
                .field("offset", offset)
                .field("len", len)
                .finish(),
            FileHandleInner::Lazy { len, offset, .. } => f
                .debug_struct("FileHandle::Lazy")
                .field("offset", offset)
                .field("len", len)
                .finish(),
        }
    }
}

impl FileHandle {
    /// Create an inline file handle from owned bytes (mmap, RAM).
    /// Sync reads are available.
    pub fn from_bytes(data: OwnedBytes) -> Self {
        let len = data.len() as u64;
        Self {
            inner: FileHandleInner::Inline {
                data,
                offset: 0,
                len,
            },
        }
    }

    /// Create an empty file handle.
    pub fn empty() -> Self {
        Self::from_bytes(OwnedBytes::empty())
    }

    /// Create a lazy file handle from an async range-read callback.
    /// Only async reads are available.
    pub fn lazy(len: u64, read_fn: RangeReadFn) -> Self {
        Self {
            inner: FileHandleInner::Lazy {
                read_fn,
                offset: 0,
                len,
            },
        }
    }

    /// Total length in bytes.
    #[inline]
    pub fn len(&self) -> u64 {
        match &self.inner {
            FileHandleInner::Inline { len, .. } => *len,
            FileHandleInner::Lazy { len, .. } => *len,
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether synchronous reads are available (inline/mmap data).
    #[inline]
    pub fn is_sync(&self) -> bool {
        matches!(&self.inner, FileHandleInner::Inline { .. })
    }

    /// Create a sub-range view. Zero-copy for Inline, offset-adjusted for Lazy.
    pub fn slice(&self, range: Range<u64>) -> Self {
        match &self.inner {
            FileHandleInner::Inline { data, offset, len } => {
                let new_offset = offset + range.start;
                let new_len = range.end - range.start;
                debug_assert!(
                    new_offset + new_len <= offset + len,
                    "slice out of bounds: {}+{} > {}+{}",
                    new_offset,
                    new_len,
                    offset,
                    len
                );
                Self {
                    inner: FileHandleInner::Inline {
                        data: data.clone(),
                        offset: new_offset,
                        len: new_len,
                    },
                }
            }
            FileHandleInner::Lazy {
                read_fn,
                offset,
                len,
            } => {
                let new_offset = offset + range.start;
                let new_len = range.end - range.start;
                debug_assert!(
                    new_offset + new_len <= offset + len,
                    "slice out of bounds: {}+{} > {}+{}",
                    new_offset,
                    new_len,
                    offset,
                    len
                );
                Self {
                    inner: FileHandleInner::Lazy {
                        read_fn: Arc::clone(read_fn),
                        offset: new_offset,
                        len: new_len,
                    },
                }
            }
        }
    }

    /// Async range read — works for both Inline and Lazy.
    pub async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        match &self.inner {
            FileHandleInner::Inline { data, offset, len } => {
                if range.end > *len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Range {:?} out of bounds (len: {})", range, len),
                    ));
                }
                let start = (*offset + range.start) as usize;
                let end = (*offset + range.end) as usize;
                Ok(data.slice(start..end))
            }
            FileHandleInner::Lazy {
                read_fn,
                offset,
                len,
            } => {
                if range.end > *len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Range {:?} out of bounds (len: {})", range, len),
                    ));
                }
                let abs_start = offset + range.start;
                let abs_end = offset + range.end;
                (read_fn)(abs_start..abs_end).await
            }
        }
    }

    /// Read all bytes.
    pub async fn read_bytes(&self) -> io::Result<OwnedBytes> {
        self.read_bytes_range(0..self.len()).await
    }

    /// Synchronous range read — only works for Inline handles.
    /// Returns `Err` if the handle is Lazy.
    #[inline]
    pub fn read_bytes_range_sync(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        match &self.inner {
            FileHandleInner::Inline { data, offset, len } => {
                if range.end > *len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Range {:?} out of bounds (len: {})", range, len),
                    ));
                }
                let start = (*offset + range.start) as usize;
                let end = (*offset + range.end) as usize;
                Ok(data.slice(start..end))
            }
            FileHandleInner::Lazy { .. } => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Synchronous read not available on lazy file handle",
            )),
        }
    }

    /// Synchronous read of all bytes — only works for Inline handles.
    #[inline]
    pub fn read_bytes_sync(&self) -> io::Result<OwnedBytes> {
        self.read_bytes_range_sync(0..self.len())
    }
}

/// Backing store for OwnedBytes — supports both heap Vec and mmap.
#[derive(Clone)]
enum SharedBytes {
    Vec(Arc<Vec<u8>>),
    #[cfg(feature = "native")]
    Mmap(Arc<memmap2::Mmap>),
}

impl SharedBytes {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            SharedBytes::Vec(v) => v.as_slice(),
            #[cfg(feature = "native")]
            SharedBytes::Mmap(m) => m.as_ref(),
        }
    }
}

impl std::fmt::Debug for SharedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SharedBytes::Vec(v) => write!(f, "Vec(len={})", v.len()),
            #[cfg(feature = "native")]
            SharedBytes::Mmap(m) => write!(f, "Mmap(len={})", m.len()),
        }
    }
}

/// Owned bytes with cheap cloning (Arc-backed)
///
/// Supports two backing stores:
/// - `Vec<u8>` for owned data (RamDirectory, FsDirectory, decompressed blocks)
/// - `Mmap` for zero-copy memory-mapped files (MmapDirectory, native only)
#[derive(Debug, Clone)]
pub struct OwnedBytes {
    data: SharedBytes,
    range: Range<usize>,
}

impl OwnedBytes {
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            data: SharedBytes::Vec(Arc::new(data)),
            range: 0..len,
        }
    }

    pub fn empty() -> Self {
        Self {
            data: SharedBytes::Vec(Arc::new(Vec::new())),
            range: 0..0,
        }
    }

    /// Create from a pre-existing Arc<Vec<u8>> with a sub-range.
    /// Used by RamDirectory and CachingDirectory to share data without copying.
    pub(crate) fn from_arc_vec(data: Arc<Vec<u8>>, range: Range<usize>) -> Self {
        Self {
            data: SharedBytes::Vec(data),
            range,
        }
    }

    /// Create from a memory-mapped file (zero-copy).
    #[cfg(feature = "native")]
    pub(crate) fn from_mmap(mmap: Arc<memmap2::Mmap>) -> Self {
        let len = mmap.len();
        Self {
            data: SharedBytes::Mmap(mmap),
            range: 0..len,
        }
    }

    /// Create from a memory-mapped file with a sub-range (zero-copy).
    #[cfg(feature = "native")]
    pub(crate) fn from_mmap_range(mmap: Arc<memmap2::Mmap>, range: Range<usize>) -> Self {
        Self {
            data: SharedBytes::Mmap(mmap),
            range,
        }
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    pub fn slice(&self, range: Range<usize>) -> Self {
        let start = self.range.start + range.start;
        let end = self.range.start + range.end;
        Self {
            data: self.data.clone(),
            range: start..end,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data.as_bytes()[self.range.clone()]
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }
}

impl AsRef<[u8]> for OwnedBytes {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::Deref for OwnedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Async directory trait for reading index files
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait Directory: Send + Sync + 'static {
    /// Check if a file exists
    async fn exists(&self, path: &Path) -> io::Result<bool>;

    /// Get file size
    async fn file_size(&self, path: &Path) -> io::Result<u64>;

    /// Open a file for reading (loads entire file into an inline FileHandle)
    async fn open_read(&self, path: &Path) -> io::Result<FileHandle>;

    /// Read a specific byte range from a file (optimized for network)
    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes>;

    /// List files in directory
    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>>;

    /// Open a file handle that fetches ranges on demand.
    /// For mmap directories this returns an Inline handle (sync-capable).
    /// For HTTP/filesystem directories this returns a Lazy handle.
    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle>;
}

/// Async directory trait for reading index files (WASM version - no Send requirement)
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Directory: 'static {
    /// Check if a file exists
    async fn exists(&self, path: &Path) -> io::Result<bool>;

    /// Get file size
    async fn file_size(&self, path: &Path) -> io::Result<u64>;

    /// Open a file for reading (loads entire file into an inline FileHandle)
    async fn open_read(&self, path: &Path) -> io::Result<FileHandle>;

    /// Read a specific byte range from a file (optimized for network)
    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes>;

    /// List files in directory
    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>>;

    /// Open a file handle that fetches ranges on demand.
    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle>;
}

/// A writer for incrementally writing data to a directory file.
///
/// Avoids buffering entire files in memory during merge. File-backed
/// directories write directly to disk; memory directories collect to Vec.
pub trait StreamingWriter: io::Write + Send {
    /// Finalize the write, making data available for reading.
    fn finish(self: Box<Self>) -> io::Result<()>;

    /// Bytes written so far.
    fn bytes_written(&self) -> u64;
}

/// StreamingWriter backed by Vec<u8>, finalized via DirectoryWriter::write.
/// Used as default/fallback and for RamDirectory.
struct BufferedStreamingWriter {
    path: PathBuf,
    buffer: Vec<u8>,
    /// Callback to write the buffer to the directory on finish.
    /// We store the files Arc directly for RamDirectory.
    files: Arc<RwLock<HashMap<PathBuf, Arc<Vec<u8>>>>>,
}

impl io::Write for BufferedStreamingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl StreamingWriter for BufferedStreamingWriter {
    fn finish(self: Box<Self>) -> io::Result<()> {
        self.files.write().insert(self.path, Arc::new(self.buffer));
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.buffer.len() as u64
    }
}

/// Buffer size for FileStreamingWriter (8 MB).
/// Large enough to coalesce millions of tiny writes (e.g. per-vector doc_id writes)
/// into efficient sequential I/O.
#[cfg(feature = "native")]
const FILE_STREAMING_BUF_SIZE: usize = 8 * 1024 * 1024;

/// StreamingWriter backed by a buffered std::fs::File for filesystem directories.
#[cfg(feature = "native")]
pub(crate) struct FileStreamingWriter {
    pub(crate) file: io::BufWriter<std::fs::File>,
    pub(crate) written: u64,
}

#[cfg(feature = "native")]
impl FileStreamingWriter {
    pub(crate) fn new(file: std::fs::File) -> Self {
        Self {
            file: io::BufWriter::with_capacity(FILE_STREAMING_BUF_SIZE, file),
            written: 0,
        }
    }
}

#[cfg(feature = "native")]
impl io::Write for FileStreamingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.file.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(feature = "native")]
impl StreamingWriter for FileStreamingWriter {
    fn finish(self: Box<Self>) -> io::Result<()> {
        let file = self.file.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.written
    }
}

/// Async directory trait for writing index files
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait DirectoryWriter: Directory {
    /// Create/overwrite a file with data
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()>;

    /// Delete a file
    async fn delete(&self, path: &Path) -> io::Result<()>;

    /// Atomic rename
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Sync all pending writes
    async fn sync(&self) -> io::Result<()>;

    /// Create a streaming writer for incremental file writes.
    /// Call finish() on the returned writer to finalize.
    async fn streaming_writer(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>>;
}

/// In-memory directory for testing and small indexes
#[derive(Debug, Default)]
pub struct RamDirectory {
    files: Arc<RwLock<HashMap<PathBuf, Arc<Vec<u8>>>>>,
}

impl Clone for RamDirectory {
    fn clone(&self) -> Self {
        Self {
            files: Arc::clone(&self.files),
        }
    }
}

impl RamDirectory {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Directory for RamDirectory {
    async fn exists(&self, path: &Path) -> io::Result<bool> {
        Ok(self.files.read().contains_key(path))
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        self.files
            .read()
            .get(path)
            .map(|data| data.len() as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileHandle> {
        let files = self.files.read();
        let data = files
            .get(path)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))?;

        Ok(FileHandle::from_bytes(OwnedBytes::from_arc_vec(
            Arc::clone(data),
            0..data.len(),
        )))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        let files = self.files.read();
        let data = files
            .get(path)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))?;

        let start = range.start as usize;
        let end = range.end as usize;

        if end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Range out of bounds",
            ));
        }

        Ok(OwnedBytes::from_arc_vec(Arc::clone(data), start..end))
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        let files = self.files.read();
        Ok(files
            .keys()
            .filter(|p| p.starts_with(prefix))
            .cloned()
            .collect())
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
        // RAM data is always available synchronously — return Inline handle
        self.open_read(path).await
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl DirectoryWriter for RamDirectory {
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.files
            .write()
            .insert(path.to_path_buf(), Arc::new(data.to_vec()));
        Ok(())
    }

    async fn delete(&self, path: &Path) -> io::Result<()> {
        self.files.write().remove(path);
        Ok(())
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut files = self.files.write();
        if let Some(data) = files.remove(from) {
            files.insert(to.to_path_buf(), data);
        }
        Ok(())
    }

    async fn sync(&self) -> io::Result<()> {
        Ok(())
    }

    async fn streaming_writer(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        Ok(Box::new(BufferedStreamingWriter {
            path: path.to_path_buf(),
            buffer: Vec::new(),
            files: Arc::clone(&self.files),
        }))
    }
}

/// Local filesystem directory with async IO via tokio
#[cfg(feature = "native")]
#[derive(Debug, Clone)]
pub struct FsDirectory {
    root: PathBuf,
}

#[cfg(feature = "native")]
impl FsDirectory {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        self.root.join(path)
    }
}

#[cfg(feature = "native")]
#[async_trait]
impl Directory for FsDirectory {
    async fn exists(&self, path: &Path) -> io::Result<bool> {
        let full_path = self.resolve(path);
        Ok(tokio::fs::try_exists(&full_path).await.unwrap_or(false))
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        let full_path = self.resolve(path);
        let metadata = tokio::fs::metadata(&full_path).await?;
        Ok(metadata.len())
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileHandle> {
        let full_path = self.resolve(path);
        let data = tokio::fs::read(&full_path).await?;
        Ok(FileHandle::from_bytes(OwnedBytes::new(data)))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let full_path = self.resolve(path);
        let mut file = tokio::fs::File::open(&full_path).await?;

        file.seek(std::io::SeekFrom::Start(range.start)).await?;

        let len = (range.end - range.start) as usize;
        let mut buffer = vec![0u8; len];
        file.read_exact(&mut buffer).await?;

        Ok(OwnedBytes::new(buffer))
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        let full_path = self.resolve(prefix);
        let mut entries = tokio::fs::read_dir(&full_path).await?;
        let mut files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_file() {
                files.push(entry.path().strip_prefix(&self.root).unwrap().to_path_buf());
            }
        }

        Ok(files)
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
        let full_path = self.resolve(path);
        let metadata = tokio::fs::metadata(&full_path).await?;
        let file_size = metadata.len();

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let full_path = full_path.clone();
            Box::pin(async move {
                use tokio::io::{AsyncReadExt, AsyncSeekExt};

                let mut file = tokio::fs::File::open(&full_path).await?;
                file.seek(std::io::SeekFrom::Start(range.start)).await?;

                let len = (range.end - range.start) as usize;
                let mut buffer = vec![0u8; len];
                file.read_exact(&mut buffer).await?;

                Ok(OwnedBytes::new(buffer))
            })
        });

        Ok(FileHandle::lazy(file_size, read_fn))
    }
}

#[cfg(feature = "native")]
#[async_trait]
impl DirectoryWriter for FsDirectory {
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let full_path = self.resolve(path);

        // Ensure parent directory exists
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&full_path, data).await
    }

    async fn delete(&self, path: &Path) -> io::Result<()> {
        let full_path = self.resolve(path);
        tokio::fs::remove_file(&full_path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let from_path = self.resolve(from);
        let to_path = self.resolve(to);
        tokio::fs::rename(&from_path, &to_path).await
    }

    async fn sync(&self) -> io::Result<()> {
        // fsync the directory
        let dir = std::fs::File::open(&self.root)?;
        dir.sync_all()?;
        Ok(())
    }

    async fn streaming_writer(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        let full_path = self.resolve(path);
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = std::fs::File::create(&full_path)?;
        Ok(Box::new(FileStreamingWriter::new(file)))
    }
}

/// Caching wrapper for any Directory - caches file reads
pub struct CachingDirectory<D: Directory> {
    inner: D,
    cache: RwLock<HashMap<PathBuf, Arc<Vec<u8>>>>,
    max_cached_bytes: usize,
    current_bytes: RwLock<usize>,
}

impl<D: Directory> CachingDirectory<D> {
    pub fn new(inner: D, max_cached_bytes: usize) -> Self {
        Self {
            inner,
            cache: RwLock::new(HashMap::new()),
            max_cached_bytes,
            current_bytes: RwLock::new(0),
        }
    }

    fn try_cache(&self, path: &Path, data: &[u8]) {
        let mut current = self.current_bytes.write();
        if *current + data.len() <= self.max_cached_bytes {
            self.cache
                .write()
                .insert(path.to_path_buf(), Arc::new(data.to_vec()));
            *current += data.len();
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<D: Directory> Directory for CachingDirectory<D> {
    async fn exists(&self, path: &Path) -> io::Result<bool> {
        if self.cache.read().contains_key(path) {
            return Ok(true);
        }
        self.inner.exists(path).await
    }

    async fn file_size(&self, path: &Path) -> io::Result<u64> {
        if let Some(data) = self.cache.read().get(path) {
            return Ok(data.len() as u64);
        }
        self.inner.file_size(path).await
    }

    async fn open_read(&self, path: &Path) -> io::Result<FileHandle> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            return Ok(FileHandle::from_bytes(OwnedBytes::from_arc_vec(
                Arc::clone(data),
                0..data.len(),
            )));
        }

        // Read from inner and potentially cache
        let handle = self.inner.open_read(path).await?;
        let bytes = handle.read_bytes().await?;

        self.try_cache(path, bytes.as_slice());

        Ok(FileHandle::from_bytes(bytes))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            let start = range.start as usize;
            let end = range.end as usize;
            return Ok(OwnedBytes::from_arc_vec(Arc::clone(data), start..end));
        }

        self.inner.read_range(path, range).await
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
        // For caching directory, delegate to inner - caching happens at read_range level
        self.inner.open_lazy(path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ram_directory() {
        let dir = RamDirectory::new();

        // Write file
        dir.write(Path::new("test.txt"), b"hello world")
            .await
            .unwrap();

        // Check exists
        assert!(dir.exists(Path::new("test.txt")).await.unwrap());
        assert!(!dir.exists(Path::new("nonexistent.txt")).await.unwrap());

        // Read file
        let slice = dir.open_read(Path::new("test.txt")).await.unwrap();
        let data = slice.read_bytes().await.unwrap();
        assert_eq!(data.as_slice(), b"hello world");

        // Read range
        let range_data = dir.read_range(Path::new("test.txt"), 0..5).await.unwrap();
        assert_eq!(range_data.as_slice(), b"hello");

        // Delete
        dir.delete(Path::new("test.txt")).await.unwrap();
        assert!(!dir.exists(Path::new("test.txt")).await.unwrap());
    }

    #[tokio::test]
    async fn test_file_handle() {
        let data = OwnedBytes::new(b"hello world".to_vec());
        let handle = FileHandle::from_bytes(data);

        assert_eq!(handle.len(), 11);
        assert!(handle.is_sync());

        let sub = handle.slice(0..5);
        let bytes = sub.read_bytes().await.unwrap();
        assert_eq!(bytes.as_slice(), b"hello");

        let sub2 = handle.slice(6..11);
        let bytes2 = sub2.read_bytes().await.unwrap();
        assert_eq!(bytes2.as_slice(), b"world");

        // Sync reads work on inline handles
        let sync_bytes = handle.read_bytes_range_sync(0..5).unwrap();
        assert_eq!(sync_bytes.as_slice(), b"hello");
    }

    #[tokio::test]
    async fn test_owned_bytes() {
        let bytes = OwnedBytes::new(vec![1, 2, 3, 4, 5]);

        assert_eq!(bytes.len(), 5);
        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4, 5]);

        let sliced = bytes.slice(1..4);
        assert_eq!(sliced.as_slice(), &[2, 3, 4]);

        // Original unchanged
        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4, 5]);
    }
}
