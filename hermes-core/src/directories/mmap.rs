//! Memory-mapped directory for efficient access to large indices
//!
//! This module is only compiled with the "native" feature.

use std::collections::HashMap;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use memmap2::Mmap;
use parking_lot::RwLock;

use super::{Directory, DirectoryWriter, FileSlice, LazyFileHandle, OwnedBytes, RangeReadFn};

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
pub struct MmapDirectory {
    root: PathBuf,
    /// Cache of memory-mapped files
    mmap_cache: Arc<RwLock<HashMap<PathBuf, Arc<Mmap>>>>,
}

impl MmapDirectory {
    /// Create a new MmapDirectory rooted at the given path
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        self.root.join(path)
    }

    /// Get or create a memory-mapped file
    async fn get_mmap(&self, path: &Path) -> io::Result<Arc<Mmap>> {
        let full_path = self.resolve(path);

        // Check cache first
        {
            let cache = self.mmap_cache.read();
            if let Some(mmap) = cache.get(&full_path) {
                return Ok(Arc::clone(mmap));
            }
        }

        // Create new mmap
        let file = std::fs::File::open(&full_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mmap = Arc::new(mmap);

        // Cache it
        {
            let mut cache = self.mmap_cache.write();
            cache.insert(full_path, Arc::clone(&mmap));
        }

        Ok(mmap)
    }

    /// Clear the mmap cache (useful after writes)
    pub fn clear_cache(&self) {
        self.mmap_cache.write().clear();
    }

    /// Remove a specific file from the cache
    fn invalidate_cache(&self, path: &Path) {
        let full_path = self.resolve(path);
        self.mmap_cache.write().remove(&full_path);
    }
}

impl Clone for MmapDirectory {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            mmap_cache: Arc::clone(&self.mmap_cache),
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

    async fn open_read(&self, path: &Path) -> io::Result<FileSlice> {
        let mmap = self.get_mmap(path).await?;
        // Create OwnedBytes that references the mmap data
        // The Arc<Mmap> keeps the mapping alive
        let bytes = mmap.to_vec(); // Copy for now - could optimize with custom type
        Ok(FileSlice::new(OwnedBytes::new(bytes)))
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<OwnedBytes> {
        let mmap = self.get_mmap(path).await?;
        let start = range.start as usize;
        let end = range.end as usize;

        if end > mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Range {}..{} exceeds file size {}", start, end, mmap.len()),
            ));
        }

        Ok(OwnedBytes::new(mmap[start..end].to_vec()))
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
        let mmap = self.get_mmap(path).await?;
        let file_size = mmap.len();

        // Clone the mmap Arc for the closure
        let mmap_clone = Arc::clone(&mmap);

        let read_fn: RangeReadFn = Arc::new(move |range: Range<u64>| {
            let mmap = Arc::clone(&mmap_clone);
            Box::pin(async move {
                let start = range.start as usize;
                let end = range.end as usize;

                if end > mmap.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Range {}..{} exceeds file size {}", start, end, mmap.len()),
                    ));
                }

                Ok(OwnedBytes::new(mmap[start..end].to_vec()))
            })
        });

        Ok(LazyFileHandle::new(file_size, read_fn))
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

        // Invalidate cache before writing
        self.invalidate_cache(path);

        tokio::fs::write(&full_path, data).await
    }

    async fn delete(&self, path: &Path) -> io::Result<()> {
        // Invalidate cache before deleting
        self.invalidate_cache(path);

        let full_path = self.resolve(path);
        tokio::fs::remove_file(&full_path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        // Invalidate both paths
        self.invalidate_cache(from);
        self.invalidate_cache(to);

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
    async fn test_mmap_directory_cache() {
        let temp_dir = TempDir::new().unwrap();
        let dir = MmapDirectory::new(temp_dir.path());

        // Write a file
        dir.write(Path::new("cached.txt"), b"cached content")
            .await
            .unwrap();

        // Read twice - second should use cache
        let _ = dir.open_read(Path::new("cached.txt")).await.unwrap();
        let _ = dir.open_read(Path::new("cached.txt")).await.unwrap();

        // Cache should have one entry
        assert_eq!(dir.mmap_cache.read().len(), 1);

        // Clear cache
        dir.clear_cache();
        assert_eq!(dir.mmap_cache.read().len(), 0);
    }

    #[tokio::test]
    async fn test_mmap_directory_lazy_handle() {
        use crate::directories::AsyncFileRead;

        let temp_dir = TempDir::new().unwrap();
        let dir = MmapDirectory::new(temp_dir.path());

        // Write a larger file
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        dir.write(Path::new("large.bin"), &data).await.unwrap();

        // Open lazy handle
        let handle = dir.open_lazy(Path::new("large.bin")).await.unwrap();
        assert_eq!(handle.len(), 1000);

        // Read ranges
        let range1 = handle.read_bytes_range(0..100).await.unwrap();
        assert_eq!(range1.len(), 100);
        assert_eq!(range1.as_slice(), &data[0..100]);

        let range2 = handle.read_bytes_range(500..600).await.unwrap();
        assert_eq!(range2.as_slice(), &data[500..600]);
    }
}
