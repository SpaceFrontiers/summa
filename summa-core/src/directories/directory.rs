//! Async Directory abstraction for IO operations
//!
//! Supports network, local filesystem, and in-memory storage.
//! All reads are async to minimize blocking on network latency.

use super::{FileHandle, OwnedBytes};
#[cfg(feature = "native")]
use super::{IndexLabel, RangeReadFn};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

    /// Open an asynchronously consumed payload, independently of sync-capable metadata.
    /// The default preserves this backend's existing lazy-read policy.
    async fn open_payload(&self, path: &Path) -> io::Result<FileHandle> {
        self.open_lazy(path).await
    }

    /// Optional explicit reads for sparse blocks, separate from mapped metadata.
    /// None preserves the original handle and does not reopen the file.
    async fn open_sparse_payload(&self, _path: &Path) -> io::Result<Option<FileHandle>> {
        Ok(None)
    }

    /// Attach the owning index's name for Directory-layer metric labels
    /// (`summa_directory_read_*`, `summa_cold_write_bytes_total`).
    /// Called by `Index::open`/`create` once the schema is loaded; wrappers
    /// forward to their inner directory. Default: no-op (directories that
    /// emit no Directory-layer metrics, e.g. RamDirectory).
    fn set_index_label(&self, _label: &str) {}

    /// Resolve a directory-relative file to a native filesystem path.
    ///
    /// Local backends expose this so large, short-lived merge scratch files
    /// can live beside the index instead of silently spilling to the
    /// container's root filesystem. Remote and in-memory backends return
    /// `None`.
    fn local_path(&self, _path: &Path) -> Option<PathBuf> {
        None
    }
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

    /// Open an asynchronously consumed payload, independently of sync-capable metadata.
    /// The default preserves this backend's existing lazy-read policy.
    async fn open_payload(&self, path: &Path) -> io::Result<FileHandle> {
        self.open_lazy(path).await
    }

    /// Optional explicit reads for sparse blocks, separate from mapped metadata.
    /// None preserves the original handle and does not reopen the file.
    async fn open_sparse_payload(&self, _path: &Path) -> io::Result<Option<FileHandle>> {
        Ok(None)
    }

    /// Attach the owning index's name for Directory-layer metric labels.
    /// No-op default; metrics are native-only but the label is harmless.
    fn set_index_label(&self, _label: &str) {}

    /// WASM backends do not expose a native filesystem path.
    fn local_path(&self, _path: &Path) -> Option<PathBuf> {
        None
    }
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

    /// Copy one local-file range at the current output position without
    /// routing bytes through a userspace buffer.
    ///
    /// Filesystem writers implement this with Linux `copy_file_range`.
    /// Other backends return `Unsupported`, allowing merge code to fall back
    /// to its portable mmap/read + write path before any bytes are copied.
    #[cfg(feature = "native")]
    fn copy_from_file_range(
        &mut self,
        _source: &std::fs::File,
        _source_offset: &mut u64,
        _len: usize,
    ) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "streaming writer does not support kernel-assisted range copies",
        ))
    }
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

    fn copy_from_file_range(
        &mut self,
        source: &std::fs::File,
        source_offset: &mut u64,
        len: usize,
    ) -> io::Result<usize> {
        io::Write::flush(&mut self.file)?;
        let copied = copy_file_range_once(source, source_offset, self.file.get_ref(), len)?;
        self.written = self
            .written
            .checked_add(copied as u64)
            .ok_or_else(|| io::Error::other("streaming-writer byte count overflow"))?;
        Ok(copied)
    }
}

#[cfg(feature = "native")]
pub(crate) fn copy_file_range_once(
    source: &std::fs::File,
    source_offset: &mut u64,
    destination: &std::fs::File,
    len: usize,
) -> io::Result<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;

        let mut offset = libc::loff_t::try_from(*source_offset).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "source offset exceeds i64")
        })?;
        let copied = unsafe {
            libc::copy_file_range(
                source.as_raw_fd(),
                &mut offset,
                destination.as_raw_fd(),
                std::ptr::null_mut(),
                len,
                0,
            )
        };
        if copied < 0 {
            let error = io::Error::last_os_error();
            let unsupported = error.raw_os_error().is_some_and(|code| {
                code == libc::ENOSYS
                    || code == libc::EXDEV
                    || code == libc::EOPNOTSUPP
                    || code == libc::EINVAL
            });
            return if unsupported {
                Err(io::Error::new(io::ErrorKind::Unsupported, error))
            } else {
                Err(error)
            };
        }
        *source_offset = u64::try_from(offset)
            .map_err(|_| io::Error::other("copy_file_range returned a negative source offset"))?;
        Ok(copied as usize)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (source, source_offset, destination, len);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "kernel-assisted range copies are only available on Linux",
        ))
    }
}

/// Async directory trait for writing index files
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait DirectoryWriter: Directory {
    /// Create/overwrite a file with data
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()>;

    /// Create/overwrite a file with data, durably.
    ///
    /// [`Self::write`] does not guarantee the bytes reach stable storage
    /// before returning (filesystem implementations leave them in the OS
    /// page cache). Any file that durably-published metadata will reference
    /// (e.g. segment `.meta`) must be written through this method instead:
    /// it routes through [`Self::streaming_writer`], whose `finish()` fsyncs
    /// on filesystem implementations.
    async fn write_durable(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        use io::Write as _;
        let mut writer = self.streaming_writer(path).await?;
        writer.write_all(data)?;
        writer.finish()
    }

    /// Delete a file
    async fn delete(&self, path: &Path) -> io::Result<()>;

    /// Atomic rename
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Create another immutable name for an existing file without copying its
    /// contents when the backend supports it. Segment rewrites use this to
    /// retain unchanged multi-gigabyte files while replacing only one index
    /// payload. Backends without link semantics return `Unsupported`; callers
    /// then fall back to a streaming copy.
    async fn link(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "directory backend does not support immutable file links",
        ))
    }

    /// Sync all pending writes
    async fn sync(&self) -> io::Result<()>;

    /// Create a streaming writer for incremental file writes.
    /// Call finish() on the returned writer to finalize.
    async fn streaming_writer(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>>;

    /// Streaming writer for **bulk one-shot data** (merge/reorder outputs).
    ///
    /// Filesystem directories return a page-cache-dropping writer (see
    /// `docs/cold-io.md`) so multi-GB merge writes cannot evict the serving
    /// segments' warm pages. Output is byte-identical to the buffered
    /// writer. Default impl delegates to [`Self::streaming_writer`].
    async fn streaming_writer_cold(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        self.streaming_writer(path).await
    }

    /// Cold writer with a userspace buffering hint for concurrent outputs.
    /// Local files clamp the buffer to 1 byte through 8 MiB. Other backends
    /// may delegate to their usual cold writer; this does not bound owned
    /// output in memory-backed directories or backend-specific caches.
    async fn streaming_writer_cold_with_capacity(
        &self,
        path: &Path,
        _buffer_capacity: usize,
    ) -> io::Result<Box<dyn StreamingWriter>> {
        self.streaming_writer_cold(path).await
    }
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

    /// Synchronous file listing (for serialization).
    pub fn list_files_sync(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        let files = self.files.read();
        Ok(files
            .keys()
            .filter(|p| p.starts_with(prefix))
            .cloned()
            .collect())
    }

    /// Synchronous file read (for serialization).
    pub fn read_file_sync(&self, path: &Path) -> io::Result<Vec<u8>> {
        let files = self.files.read();
        files
            .get(path)
            .map(|data| data.as_ref().clone())
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found"))
    }

    /// Synchronous file write (for deserialization).
    pub fn write_sync(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.files
            .write()
            .insert(path.to_path_buf(), Arc::new(data.to_vec()));
        Ok(())
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

        FileHandle::from_bytes(OwnedBytes::from_arc_vec(Arc::clone(data), 0..data.len()))
            .read_bytes_range_sync(range)
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

    async fn link(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut files = self.files.write();
        let data = files.get(from).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("source file {from:?} does not exist"),
            )
        })?;
        files.insert(to.to_path_buf(), data);
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
    label: IndexLabel,
}

#[cfg(feature = "native")]
impl FsDirectory {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            label: IndexLabel::default(),
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
        // `try_exists` maps NotFound to Ok(false); any other stat failure
        // (EACCES, EIO, ...) must propagate so callers can distinguish a
        // genuinely missing file from a transient IO error — swallowing it
        // as `false` quarantines a healthy segment as "missing mandatory
        // files" instead of retrying.
        tokio::fs::try_exists(&full_path).await
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
        // The outer FileHandle/cache wrapper meters the logical read once.
        self.open_lazy(path)
            .await?
            .read_bytes_range_unmetered(range)
            .await
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        super::local::list_files(&self.root, prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
        // Open once and keep the descriptor in the handle. Each range read is
        // then a single positional read on one blocking thread instead of
        // open + seek + read (three `spawn_blocking` hops) per range.
        let full_path = self.resolve(path);
        let (file, file_size) = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&full_path)?;
            let file_size = file.metadata()?.len();
            Ok::<_, io::Error>((file, file_size))
        })
        .await
        .map_err(io::Error::other)??;
        let file = Arc::new(file);

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let file = Arc::clone(&file);
            Box::pin(async move {
                tokio::task::spawn_blocking(move || {
                    let len = (range.end - range.start) as usize;
                    let mut buffer = vec![0u8; len];
                    super::local::read_exact_at(&file, &mut buffer, range.start)?;
                    Ok(OwnedBytes::new(buffer))
                })
                .await
                .map_err(io::Error::other)?
            })
        });

        Ok(FileHandle::lazy_labeled(
            file_size,
            read_fn,
            self.label.get(),
        ))
    }

    fn set_index_label(&self, label: &str) {
        self.label.set(label);
    }

    fn local_path(&self, path: &Path) -> Option<PathBuf> {
        Some(self.resolve(path))
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
        // Metadata publication is the only rename user. Keep the atomic
        // filesystem operation in a single future poll: tokio::fs::rename is
        // backed by a cancellable await around spawn_blocking, so a dropped
        // commit future could observe neither completion nor failure even
        // though the rename later succeeded. The caller must update its
        // in-memory metadata in the same poll after this returns.
        std::fs::rename(&from_path, &to_path)
    }

    async fn link(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::hard_link(self.resolve(from), self.resolve(to))
    }

    async fn sync(&self) -> io::Result<()> {
        // fsync the directory
        let dir = std::fs::File::open(&self.root)?;
        dir.sync_all()?;
        Ok(())
    }

    async fn streaming_writer(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        super::local::streaming_writer(&self.resolve(path)).await
    }

    async fn streaming_writer_cold(&self, path: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        super::local::streaming_writer_cold(&self.resolve(path), self.label.get(), None).await
    }

    async fn streaming_writer_cold_with_capacity(
        &self,
        path: &Path,
        buffer_capacity: usize,
    ) -> io::Result<Box<dyn StreamingWriter>> {
        super::local::streaming_writer_cold(
            &self.resolve(path),
            self.label.get(),
            Some(buffer_capacity),
        )
        .await
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
            return FileHandle::from_bytes(OwnedBytes::from_arc_vec(
                Arc::clone(data),
                0..data.len(),
            ))
            .read_bytes_range_sync(range);
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

    async fn open_payload(&self, path: &Path) -> io::Result<FileHandle> {
        self.inner.open_payload(path).await
    }

    async fn open_sparse_payload(&self, path: &Path) -> io::Result<Option<FileHandle>> {
        self.inner.open_sparse_payload(path).await
    }

    fn set_index_label(&self, label: &str) {
        self.inner.set_index_label(label);
    }

    fn local_path(&self, path: &Path) -> Option<PathBuf> {
        self.inner.local_path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn local_and_cached_directories_reject_invalid_ranges() {
        async fn check(directory: &dyn Directory) {
            for range in [Range { start: 5, end: 3 }, 0..9, u64::MAX..u64::MAX] {
                assert_eq!(
                    directory
                        .read_range(Path::new("data"), range)
                        .await
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidInput
                );
            }
        }
        let ram = RamDirectory::new();
        ram.write(Path::new("data"), b"12345678").await.unwrap();
        check(&ram).await;
        let cache = CachingDirectory::new(ram.clone(), 1024);
        cache.open_read(Path::new("data")).await.unwrap();
        check(&cache).await;
        let slices = crate::directories::SliceCachingDirectory::new(ram, 1024);
        slices.open_read(Path::new("data")).await.unwrap();
        check(&slices).await;
        let temp = tempfile::tempdir().unwrap();
        let fs = FsDirectory::new(temp.path());
        fs.write(Path::new("data"), b"12345678").await.unwrap();
        check(&fs).await;
        check(&crate::directories::MmapDirectory::new(temp.path())).await;
    }

    #[cfg(all(feature = "native", feature = "metrics"))]
    #[test]
    fn filesystem_cache_wrappers_meter_each_logical_read_once() {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("data"), b"1234").unwrap();
        metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let directory = crate::directories::SliceCachingDirectory::new(
                    FsDirectory::new(root.path()),
                    1024,
                );
                directory.set_index_label("range_metering_test");
                let lazy = directory.open_lazy(Path::new("data")).await.unwrap();
                let payload = directory.open_payload(Path::new("data")).await.unwrap();
                for (handle, range) in [(&lazy, 0..2), (&payload, 2..4)] {
                    for _ in 0..2 {
                        handle.read_bytes_range(range.clone()).await.unwrap();
                    }
                }
            })
        });
        let observations: usize = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter_map(|(key, _, _, value)| match value {
                DebugValue::Histogram(values)
                    if key.key().name() == "summa_directory_read_bytes" =>
                {
                    Some(values.len())
                }
                _ => None,
            })
            .sum();
        assert_eq!(
            observations, 4,
            "each miss/hit must be metered once by its outer handle"
        );
    }

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

    /// A transient stat failure (EACCES here, EIO on flaky storage) must
    /// surface as `Err`, not `Ok(false)`: callers classify a missing
    /// mandatory segment file as deterministic corruption and quarantine
    /// the segment until restart.
    #[cfg(all(unix, feature = "native"))]
    #[tokio::test]
    async fn test_fs_exists_propagates_stat_errors_instead_of_reporting_missing() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let dir = FsDirectory::new(temp_dir.path());
        dir.write(Path::new("locked/seg.meta"), b"data")
            .await
            .unwrap();

        // Removing search permission from the parent makes stat on the child
        // fail with EACCES while the file itself still exists.
        let locked = temp_dir.path().join("locked");
        let original = std::fs::metadata(&locked).unwrap().permissions();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::metadata(locked.join("seg.meta")).is_ok() {
            // Running as root: directory permissions are not enforced, so the
            // stat failure cannot be provoked.
            std::fs::set_permissions(&locked, original).unwrap();
            return;
        }
        let result = dir.exists(Path::new("locked/seg.meta")).await;
        std::fs::set_permissions(&locked, original).unwrap();

        let error =
            result.expect_err("stat failure must propagate as Err, not be misreported as missing");
        assert_ne!(error.kind(), io::ErrorKind::NotFound);
        // Once stat succeeds again the file is reported present.
        assert!(dir.exists(Path::new("locked/seg.meta")).await.unwrap());
    }
}
