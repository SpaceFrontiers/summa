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

/// Positional exact read that does not move the shared file cursor, so one
/// `File` can serve concurrent range reads.
#[cfg(unix)]
pub(super) fn read_exact_at(
    file: &std::fs::File,
    buffer: &mut [u8],
    offset: u64,
) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buffer, offset)
}

#[cfg(windows)]
pub(super) fn read_exact_at(
    file: &std::fs::File,
    mut buffer: &mut [u8],
    mut offset: u64,
) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buffer.is_empty() {
        match file.seek_read(buffer, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            Ok(read) => {
                buffer = &mut buffer[read..];
                offset += read as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
