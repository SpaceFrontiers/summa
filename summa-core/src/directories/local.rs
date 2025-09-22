//! Shared filesystem operations for heap-backed and mmap directory readers.
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{ColdStreamingWriter, FileStreamingWriter, StreamingWriter};

pub(super) async fn list_files(root: &Path, prefix: &Path) -> io::Result<Vec<PathBuf>> {
    let mut entries = tokio::fs::read_dir(root.join(prefix)).await?;
    let mut files = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_type().await?.is_file() {
            files.push(entry.path().strip_prefix(root).unwrap().to_path_buf());
        }
    }
    Ok(files)
}

async fn create_file(path: &Path) -> io::Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    std::fs::File::create(path)
}

pub(super) async fn streaming_writer(path: &Path) -> io::Result<Box<dyn StreamingWriter>> {
    Ok(Box::new(FileStreamingWriter::new(create_file(path).await?)))
}

pub(super) async fn streaming_writer_cold(
    path: &Path,
    label: Arc<str>,
    capacity: Option<usize>,
) -> io::Result<Box<dyn StreamingWriter>> {
    let file = create_file(path).await?;
    Ok(Box::new(match capacity {
        Some(capacity) => ColdStreamingWriter::with_capacity(file, label, capacity),
        None => ColdStreamingWriter::new(file, label),
    }))
}
