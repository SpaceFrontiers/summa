//! Async Directory abstraction for IO operations
//!
//! Supports network, local filesystem, and in-memory storage.
//! All reads are async to minimize blocking on network latency.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
#[cfg(feature = "native")]
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A slice of bytes that can be read asynchronously
#[derive(Debug, Clone)]
pub struct FileSlice {
    data: OwnedBytes,
    range: Range<u64>,
}

impl FileSlice {
    pub fn new(data: OwnedBytes) -> Self {
        let len = data.len() as u64;
        Self {
            data,
            range: 0..len,
        }
    }

    pub fn empty() -> Self {
        Self {
            data: OwnedBytes::empty(),
            range: 0..0,
        }
    }

    pub fn slice(&self, range: Range<u64>) -> Self {
        let start = self.range.start + range.start;
        let end = self.range.start + range.end;
        Self {
            data: self.data.clone(),
            range: start..end,
        }
    }

    pub fn len(&self) -> u64 {
        self.range.end - self.range.start
    }

    pub fn is_empty(&self) -> bool {
        self.range.start == self.range.end
    }

    /// Read the entire slice (async for network compatibility)
    pub async fn read_bytes(&self) -> io::Result<OwnedBytes> {
        Ok(self
            .data
            .slice(self.range.start as usize..self.range.end as usize))
    }

    /// Read a specific range within this slice
    pub async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        let start = self.range.start + range.start;
        let end = self.range.start + range.end;
        if end > self.range.end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Range out of bounds",
            ));
        }
        Ok(self.data.slice(start as usize..end as usize))
    }
}

/// Trait for async range reading - implemented by both FileSlice and LazyFileHandle
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg(not(target_arch = "wasm32"))]
pub trait AsyncFileRead: Send + Sync {
    /// Get the total length of the file/slice (u64 to support >4GB files on 32-bit platforms)
    fn len(&self) -> u64;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a specific byte range
    async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes>;

    /// Read all bytes
    async fn read_bytes(&self) -> io::Result<OwnedBytes> {
        self.read_bytes_range(0..self.len()).await
    }
}

/// Trait for async range reading - implemented by both FileSlice and LazyFileHandle (WASM version)
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg(target_arch = "wasm32")]
pub trait AsyncFileRead {
    /// Get the total length of the file/slice (u64 to support >4GB files on 32-bit platforms)
    fn len(&self) -> u64;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a specific byte range
    async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes>;

    /// Read all bytes
    async fn read_bytes(&self) -> io::Result<OwnedBytes> {
        self.read_bytes_range(0..self.len()).await
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl AsyncFileRead for FileSlice {
    fn len(&self) -> u64 {
        self.range.end - self.range.start
    }

    async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        let start = self.range.start + range.start;
        let end = self.range.start + range.end;
        if end > self.range.end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Range out of bounds",
            ));
        }
        Ok(self.data.slice(start as usize..end as usize))
    }
}

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

/// Lazy file handle that fetches ranges on demand via HTTP range requests
/// Does NOT load the entire file into memory
pub struct LazyFileHandle {
    /// Total file size (u64 to support >4GB files on 32-bit platforms like WASM)
    file_size: u64,
    /// Callback to read a range from the underlying directory
    read_fn: RangeReadFn,
}

impl std::fmt::Debug for LazyFileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyFileHandle")
            .field("file_size", &self.file_size)
            .finish()
    }
}

impl Clone for LazyFileHandle {
    fn clone(&self) -> Self {
        Self {
            file_size: self.file_size,
            read_fn: Arc::clone(&self.read_fn),
        }
    }
}

impl LazyFileHandle {
    /// Create a new lazy file handle
    pub fn new(file_size: u64, read_fn: RangeReadFn) -> Self {
        Self { file_size, read_fn }
    }

    /// Get file size
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Create a sub-slice view (still lazy)
    pub fn slice(&self, range: Range<u64>) -> LazyFileSlice {
        LazyFileSlice {
            handle: self.clone(),
            offset: range.start,
            len: range.end - range.start,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl AsyncFileRead for LazyFileHandle {
    fn len(&self) -> u64 {
        self.file_size
    }

    async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        if range.end > self.file_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Range {:?} out of bounds (file size: {})",
                    range, self.file_size
                ),
            ));
        }
        (self.read_fn)(range).await
    }
}

/// A slice view into a LazyFileHandle
#[derive(Clone)]
pub struct LazyFileSlice {
    handle: LazyFileHandle,
    offset: u64,
    len: u64,
}

impl std::fmt::Debug for LazyFileSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyFileSlice")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

impl LazyFileSlice {
    /// Create a sub-slice
    pub fn slice(&self, range: Range<u64>) -> Self {
        Self {
            handle: self.handle.clone(),
            offset: self.offset + range.start,
            len: range.end - range.start,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl AsyncFileRead for LazyFileSlice {
    fn len(&self) -> u64 {
        self.len
    }

    async fn read_bytes_range(&self, range: Range<u64>) -> io::Result<OwnedBytes> {
        if range.end > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Range {:?} out of bounds (slice len: {})", range, self.len),
            ));
        }
        let abs_start = self.offset + range.start;
        let abs_end = self.offset + range.end;
        self.handle.read_bytes_range(abs_start..abs_end).await
    }
}

/// Owned bytes with cheap cloning (Arc-backed)
#[derive(Debug, Clone)]
pub struct OwnedBytes {
    data: Arc<Vec<u8>>,
    range: Range<usize>,
}

impl OwnedBytes {
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            data: Arc::new(data),
            range: 0..len,
        }
    }

    pub fn empty() -> Self {
        Self {
            data: Arc::new(Vec::new()),
            range: 0..0,
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
            data: Arc::clone(&self.data),
            range: start..end,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data[self.range.clone()]
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

    /// Open a file for reading, returns a FileSlice (loads entire file)
    async fn open_read(&self, path: &Path) -> io::Result<FileSlice>;

    /// Read a specific byte range from a file (optimized for network)
    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes>;

    /// List files in directory
    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>>;

    /// Open a lazy file handle that fetches ranges on demand
    /// This is more efficient for large files over network
    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle>;
}

/// Async directory trait for reading index files (WASM version - no Send requirement)
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait Directory: 'static {
    /// Check if a file exists
    async fn exists(&self, path: &Path) -> io::Result<bool>;

    /// Get file size
    async fn file_size(&self, path: &Path) -> io::Result<u64>;

    /// Open a file for reading, returns a FileSlice (loads entire file)
    async fn open_read(&self, path: &Path) -> io::Result<FileSlice>;

    /// Read a specific byte range from a file (optimized for network)
    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes>;

    /// List files in directory
    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>>;

    /// Open a lazy file handle that fetches ranges on demand
    /// This is more efficient for large files over network
    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle>;
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

/// StreamingWriter backed by std::fs::File for filesystem directories.
#[cfg(feature = "native")]
pub(crate) struct FileStreamingWriter {
    pub(crate) file: std::fs::File,
    pub(crate) written: u64,
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
    fn finish(mut self: Box<Self>) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
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

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        let files = self.files.read();
        let data = files
            .get(path)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))?;

        Ok(FileSlice::new(OwnedBytes {
            data: Arc::clone(data),
            range: 0..data.len(),
        }))
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

        Ok(OwnedBytes {
            data: Arc::clone(data),
            range: start..end,
        })
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        let files = self.files.read();
        Ok(files
            .keys()
            .filter(|p| p.starts_with(prefix))
            .cloned()
            .collect())
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle> {
        let files = Arc::clone(&self.files);
        let path = path.to_path_buf();

        let file_size = {
            let files_guard = files.read();
            files_guard
                .get(&path)
                .map(|data| data.len() as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))?
        };

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let files = Arc::clone(&files);
            let path = path.clone();
            Box::pin(async move {
                let files_guard = files.read();
                let data = files_guard
                    .get(&path)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))?;

                let start = range.start as usize;
                let end = range.end as usize;
                if end > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Range out of bounds",
                    ));
                }
                Ok(OwnedBytes {
                    data: Arc::clone(data),
                    range: start..end,
                })
            })
        });

        Ok(LazyFileHandle::new(file_size, read_fn))
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

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        let full_path = self.resolve(path);
        let data = tokio::fs::read(&full_path).await?;
        Ok(FileSlice::new(OwnedBytes::new(data)))
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

    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle> {
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

        Ok(LazyFileHandle::new(file_size, read_fn))
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
        Ok(Box::new(FileStreamingWriter { file, written: 0 }))
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

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            return Ok(FileSlice::new(OwnedBytes {
                data: Arc::clone(data),
                range: 0..data.len(),
            }));
        }

        // Read from inner and potentially cache
        let slice = self.inner.open_read(path).await?;
        let bytes = slice.read_bytes().await?;

        self.try_cache(path, bytes.as_slice());

        Ok(FileSlice::new(bytes))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        // Check cache first
        if let Some(data) = self.cache.read().get(path) {
            let start = range.start as usize;
            let end = range.end as usize;
            return Ok(OwnedBytes {
                data: Arc::clone(data),
                range: start..end,
            });
        }

        self.inner.read_range(path, range).await
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<LazyFileHandle> {
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
    async fn test_file_slice() {
        let data = OwnedBytes::new(b"hello world".to_vec());
        let slice = FileSlice::new(data);

        assert_eq!(slice.len(), 11);

        let sub_slice = slice.slice(0..5);
        let bytes = sub_slice.read_bytes().await.unwrap();
        assert_eq!(bytes.as_slice(), b"hello");

        let sub_slice2 = slice.slice(6..11);
        let bytes2 = sub_slice2.read_bytes().await.unwrap();
        assert_eq!(bytes2.as_slice(), b"world");
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
