//! Regression test: segment `.meta` files must be written through the
//! durable (fsyncing) write path.
//!
//! Every other segment file goes through `streaming_writer`, whose
//! filesystem `finish()` calls `File::sync_all`. The `.meta` file was the
//! one exception (bare `DirectoryWriter::write` = `tokio::fs::write`, page
//! cache only), so a power failure after an acknowledged commit could leave
//! durably-published metadata.json referencing a segment whose `.meta` is
//! torn or empty — committed documents become unreadable, and for merge /
//! reorder outputs (whose fsynced sources are deleted right after publish)
//! permanently lost.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use hermes_core::directories::{
    Directory, DirectoryWriter, FileHandle, OwnedBytes, RamDirectory, StreamingWriter,
};
use hermes_core::{Document, IndexConfig, IndexWriter, SchemaBuilder};

/// RamDirectory wrapper recording which paths were written via the
/// non-durable `write()` vs the fsyncing `streaming_writer()` path.
#[derive(Clone, Default)]
struct RecordingDirectory {
    inner: RamDirectory,
    plain_writes: Arc<Mutex<HashSet<PathBuf>>>,
    streaming_writes: Arc<Mutex<HashSet<PathBuf>>>,
}

#[async_trait]
impl Directory for RecordingDirectory {
    async fn exists(&self, path: &Path) -> std::io::Result<bool> {
        self.inner.exists(path).await
    }

    async fn file_size(&self, path: &Path) -> std::io::Result<u64> {
        self.inner.file_size(path).await
    }

    async fn open_read(&self, path: &Path) -> std::io::Result<FileHandle> {
        self.inner.open_read(path).await
    }

    async fn read_range(
        &self,
        path: &Path,
        range: std::ops::Range<u64>,
    ) -> std::io::Result<OwnedBytes> {
        self.inner.read_range(path, range).await
    }

    async fn list_files(&self, prefix: &Path) -> std::io::Result<Vec<PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> std::io::Result<FileHandle> {
        self.inner.open_lazy(path).await
    }
}

#[async_trait]
impl DirectoryWriter for RecordingDirectory {
    async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        self.plain_writes.lock().unwrap().insert(path.to_path_buf());
        self.inner.write(path, data).await
    }

    async fn delete(&self, path: &Path) -> std::io::Result<()> {
        self.inner.delete(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        self.inner.rename(from, to).await
    }

    async fn sync(&self) -> std::io::Result<()> {
        self.inner.sync().await
    }

    async fn streaming_writer(&self, path: &Path) -> std::io::Result<Box<dyn StreamingWriter>> {
        self.streaming_writes
            .lock()
            .unwrap()
            .insert(path.to_path_buf());
        self.inner.streaming_writer(path).await
    }
}

fn is_segment_meta(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "meta")
}

/// Build (flush), merge, and reorder outputs must all route their `.meta`
/// through the fsyncing streaming-writer path, never bare `write()`.
#[tokio::test(flavor = "multi_thread")]
async fn test_segment_meta_written_through_durable_path() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let dir = RecordingDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();

    // Two commits -> two segments, so force_merge below produces a merged
    // segment (a third .meta) and reorder rewrites it (a fourth).
    for batch in 0..2 {
        for i in 0..50 {
            let mut doc = Document::new();
            doc.add_text(title, format!("hello world batch {batch} doc {i}"));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    writer.force_merge().await.unwrap();
    writer.reorder().await.unwrap();
    writer.wait_for_all_merges().await;

    let streaming = dir.streaming_writes.lock().unwrap();
    let metas_durable: Vec<_> = streaming.iter().filter(|p| is_segment_meta(p)).collect();
    assert!(
        !metas_durable.is_empty(),
        "expected at least one segment .meta to be produced"
    );

    let plain = dir.plain_writes.lock().unwrap();
    let metas_plain: Vec<_> = plain.iter().filter(|p| is_segment_meta(p)).collect();
    assert!(
        metas_plain.is_empty(),
        "segment .meta written via non-durable DirectoryWriter::write \
         (page cache only, lost on power failure): {metas_plain:?}"
    );
}

/// Sanity: the flow above must actually produce segment .meta files (via
/// either path) — guards the main test against becoming vacuous if the
/// commit flow changes.
#[tokio::test(flavor = "multi_thread")]
async fn test_segment_meta_files_are_produced_at_all() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let dir = RecordingDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    let mut doc = Document::new();
    doc.add_text(title, "hello");
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();

    let all: Vec<PathBuf> = {
        let plain = dir.plain_writes.lock().unwrap();
        let streaming = dir.streaming_writes.lock().unwrap();
        plain.iter().chain(streaming.iter()).cloned().collect()
    };
    assert!(
        all.iter().any(|p| is_segment_meta(p)),
        "no segment .meta produced by commit at all; files seen: {all:?}"
    );
}
