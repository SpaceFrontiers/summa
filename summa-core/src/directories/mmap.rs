//! Memory-mapped directory for efficient access to large indices
//!
//! This module is only compiled with the "native" feature.

use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use memmap2::Mmap;

use super::{Directory, DirectoryWriter, FileHandle, OwnedBytes, StreamingWriter};

#[derive(Clone)]
struct PayloadReads {
    service: Arc<super::PayloadReadService>,
    scope: Arc<super::payload::scope::Scope>,
    sparse: bool,
}

/// Memory-mapped directory for efficient access to large index files
///
/// Uses memory-mapped files to avoid loading entire files into memory.
/// The OS manages paging, making this ideal for indices larger than RAM.
///
/// Benefits:
/// - Files are not fully loaded into memory
/// - OS handles caching and paging automatically
/// - Multiple processes can share the same mapped pages
/// - Efficient random access patterns
///
/// Note: Write operations still use regular file I/O.
/// No application-level cache - the OS page cache handles this efficiently.
pub struct MmapDirectory {
    root: PathBuf,
    label: super::IndexLabel,
    payload_reads: Option<PayloadReads>,
}

impl MmapDirectory {
    /// Create a new MmapDirectory rooted at the given path
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            label: super::IndexLabel::default(),
            payload_reads: None,
        }
    }

    /// Attach a shared explicit-read service for asynchronous payloads. Metadata and
    /// synchronous readers remain mapped. Unlink waits for accepted reads; whole
    /// directory removal requires `retire_payload_reads` after lifecycle drain.
    /// Opening a directory never creates an implicit worker pool.
    pub fn with_payload_reads(mut self, service: Arc<super::PayloadReadService>) -> Self {
        let scope = service.directory_scope(&self.root);
        self.payload_reads = Some(PayloadReads {
            service,
            scope,
            sparse: false,
        });
        self
    }

    /// Opt asynchronous MaxScore blocks into the same service as stored documents.
    /// Metadata and explicit synchronous block reads retain their mapping.
    pub fn with_sparse_payload_reads(self, service: Arc<super::PayloadReadService>) -> Self {
        let mut directory = self.with_payload_reads(service);
        directory.payload_reads.as_mut().unwrap().sparse = true;
        directory
    }

    /// Stop this index's payload admission and wait for accepted reads. Other
    /// directories attached to the shared service remain usable. Call after the
    /// existing writer/segment-manager drain, before removing the directory.
    pub async fn retire_payload_reads(&self) -> io::Result<()> {
        if let Some(reads) = &self.payload_reads {
            let mut retired = reads.scope.exclusive().await;
            *retired = true;
            reads.service.ensure_drained_resources()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn payload_scope_for_test(&self) -> &super::payload::scope::Scope {
        &self.payload_reads.as_ref().unwrap().scope
    }

    /// Get the root directory path
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        self.root.join(path)
    }

    /// Memory-map a file (no application cache - OS page cache handles this)
    fn mmap_file(&self, path: &Path) -> io::Result<Arc<Mmap>> {
        let full_path = self.resolve(path);
        let file = std::fs::File::open(&full_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Arc::new(mmap))
    }
}

impl Clone for MmapDirectory {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            // Shared, so a label set on any clone is visible on all
            label: self.label.clone(),
            payload_reads: self.payload_reads.clone(),
        }
    }
}

#[async_trait]
impl Directory for MmapDirectory {
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
        let mmap = self.mmap_file(path)?;
        // Zero-copy: OwnedBytes references the mmap directly
        Ok(FileHandle::from_bytes(OwnedBytes::from_mmap(mmap)))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        self.open_read(path).await?.read_bytes_range(range).await
    }

    async fn list_files(&self, prefix: &Path) -> io::Result<Vec<PathBuf>> {
        super::local::list_files(&self.root, prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> io::Result<FileHandle> {
        // Mmap data is always available synchronously — return Inline handle
        // This eliminates the async callback overhead entirely for mmap paths
        self.open_read(path).await
    }

    async fn open_payload(&self, path: &Path) -> io::Result<FileHandle> {
        match &self.payload_reads {
            Some(reads) => {
                reads
                    .service
                    .open(
                        self.resolve(path),
                        self.label.get(),
                        Some(reads.scope.clone()),
                    )
                    .await
            }
            None => self.open_lazy(path).await,
        }
    }

    async fn open_sparse_payload(&self, path: &Path) -> io::Result<Option<FileHandle>> {
        if self
            .payload_reads
            .as_ref()
            .is_some_and(|reads| reads.sparse)
        {
            self.open_payload(path).await.map(Some)
        } else {
            Ok(None)
        }
    }

    fn set_index_label(&self, label: &str) {
        self.label.set(label);
    }

    fn local_path(&self, path: &Path) -> Option<PathBuf> {
        Some(self.resolve(path))
    }
}

#[async_trait]
impl DirectoryWriter for MmapDirectory {
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
        let lease = if let Some(reads) = &self.payload_reads {
            let lease = reads.scope.exclusive().await;
            reads.service.ensure_drained_resources()?;
            Some(lease)
        } else {
            None
        };
        // The blocking unlink owns its lease if the cleanup future is cancelled.
        tokio::task::spawn_blocking(move || {
            let _lease = lease;
            std::fs::remove_file(full_path)
        })
        .await
        .map_err(io::Error::other)?
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let from_path = self.resolve(from);
        let to_path = self.resolve(to);
        // This is the metadata commit point. A synchronous rename completes
        // in one future poll, preventing cancellation from detaching the
        // filesystem rename from the following in-memory publication.
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_mmap_directory_basic() {
        let temp_dir = TempDir::new().unwrap();
        let dir = MmapDirectory::new(temp_dir.path());

        // Write a file
        let test_data = b"Hello, mmap world!";
        dir.write(Path::new("test.txt"), test_data).await.unwrap();

        // Check exists
        assert!(dir.exists(Path::new("test.txt")).await.unwrap());
        assert!(!dir.exists(Path::new("nonexistent.txt")).await.unwrap());

        // Check file size
        assert_eq!(
            dir.file_size(Path::new("test.txt")).await.unwrap(),
            test_data.len() as u64
        );

        // Read full file
        let slice = dir.open_read(Path::new("test.txt")).await.unwrap();
        let bytes = slice.read_bytes().await.unwrap();
        assert_eq!(bytes.as_slice(), test_data);

        // Read range
        let range_bytes = dir.read_range(Path::new("test.txt"), 7..12).await.unwrap();
        assert_eq!(range_bytes.as_slice(), b"mmap ");
    }

    /// A transient stat failure (EACCES here, EIO on flaky storage) must
    /// surface as `Err`, not `Ok(false)`: callers classify a missing
    /// mandatory segment file as deterministic corruption and quarantine
    /// the segment until restart.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_mmap_exists_propagates_stat_errors_instead_of_reporting_missing() {
        use std::os::unix::fs::PermissionsExt;

        let temp_dir = TempDir::new().unwrap();
        let dir = MmapDirectory::new(temp_dir.path());
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

    #[tokio::test]
    async fn test_mmap_directory_lazy_handle() {
        let temp_dir = TempDir::new().unwrap();
        let dir = MmapDirectory::new(temp_dir.path());

        // Write a larger file
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        dir.write(Path::new("large.bin"), &data).await.unwrap();

        // Open lazy handle — should be Inline (sync-capable) for mmap
        let handle = dir.open_lazy(Path::new("large.bin")).await.unwrap();
        assert_eq!(handle.len(), 1000);
        assert!(handle.is_sync());

        // Async reads
        let range1 = handle.read_bytes_range(0..100).await.unwrap();
        assert_eq!(range1.len(), 100);
        assert_eq!(range1.as_slice(), &data[0..100]);

        // Sync reads
        let range2 = handle.read_bytes_range_sync(500..600).unwrap();
        assert_eq!(range2.as_slice(), &data[500..600]);
    }
}
