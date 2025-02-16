//! Memory-mapped directory for efficient access to large indices
//!
//! This module is only compiled with the "native" feature.

use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use memmap2::Mmap;

use super::{
    Directory, DirectoryWriter, FileHandle, FileStreamingWriter, OwnedBytes, StreamingWriter,
};

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
}

impl MmapDirectory {
    /// Create a new MmapDirectory rooted at the given path
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
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
        }
    }
}

#[async_trait]
impl Directory for MmapDirectory {
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
        let mmap = self.mmap_file(path)?;
        // Zero-copy: OwnedBytes references the mmap directly
        Ok(FileHandle::from_bytes(OwnedBytes::from_mmap(mmap)))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        let mmap = self.mmap_file(path)?;
        let start = range.start as usize;
        let end = range.end as usize;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Range {}..{} exceeds file size {}", start, end, mmap.len()),
            ));
        }

        // Zero-copy: slice references the mmap directly
        Ok(OwnedBytes::from_mmap_range(mmap, start..end))
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
        // Mmap data is always available synchronously — return Inline handle
        // This eliminates the async callback overhead entirely for mmap paths
        self.open_read(path).await
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
