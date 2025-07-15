//! Regression tests for the WASM `IndexWriter` (`WasmIndexWriter`).
//!
//! Pins transactional / fail-loud behavior of the wasm writer:
//! - a rejected `add_document` must not poison the segment builder
//!   (doc counts skewed forever, whole buffered batch lost at commit)
//! - a failed metadata save during `commit()` must be retryable
//!   (pending segments must not be drained before the durable save)
//! - schemas declaring a primary key must be rejected loudly, because
//!   primary-key deduplication does not exist on the wasm branch
//!
//! Run with: `cargo test -p hermes-core --no-default-features \
//!   --features wasm,http --test wasm_writer_regressions`
#![cfg(all(feature = "wasm", not(feature = "native")))]

use std::future::Future;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hermes_core::directories::{
    Directory, DirectoryWriter, FileHandle, OwnedBytes, RamDirectory, StreamingWriter,
};
use hermes_core::{Document, IndexConfig, IndexMetadata, Schema, WasmIndexWriter};

fn total_docs(metadata: &IndexMetadata) -> u32 {
    metadata.segment_metas.values().map(|m| m.num_docs).sum()
}

// ---------------------------------------------------------------------------
// Finding: a single failed add_document poisons the builder (doc_id advanced,
// store write skipped), so commit() later fails with "Store doc count
// mismatch" and every buffered document is lost.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_wasm_add_document_rejecting_invalid_vector_does_not_poison_builder() {
    let mut builder = Schema::builder();
    let title = builder.add_text_field("title", true, true);
    let emb = builder.add_dense_vector_field("emb", 3, true, true);
    let spv = builder.add_sparse_vector_field("spv", true, false);
    let bin = builder.add_binary_dense_vector_field("bin", 8, true, false);
    let schema = builder.build();

    let mut writer = WasmIndexWriter::create(RamDirectory::new(), schema, IndexConfig::default())
        .await
        .expect("create");

    let valid_doc = |i: usize| {
        let mut d = Document::new();
        d.add_text(title, format!("hello document {i}"));
        d.add_dense_vector(emb, vec![0.1, 0.2, 0.3]);
        d.add_sparse_vector(spv, vec![(1, 0.5), (7, 1.5)]);
        d.add_binary_dense_vector(bin, vec![0b1010_1010]);
        d
    };

    for i in 0..2 {
        writer.add_document(valid_doc(i)).await.expect("valid doc");
    }
    assert_eq!(writer.pending_docs(), 2);

    // Dense vector with the wrong dimension must be rejected...
    let mut bad_dim = Document::new();
    bad_dim.add_text(title, "bad dim");
    bad_dim.add_dense_vector(emb, vec![0.1, 0.2]);
    let err = writer.add_document(bad_dim).await.unwrap_err();
    assert!(
        err.to_string().contains("dimension mismatch"),
        "unexpected error: {err}"
    );
    // ...WITHOUT advancing builder state (all-or-nothing).
    assert_eq!(
        writer.pending_docs(),
        2,
        "rejected document must not advance the builder doc count"
    );

    // Non-finite dense value.
    let mut nan_dense = Document::new();
    nan_dense.add_text(title, "nan dense");
    nan_dense.add_dense_vector(emb, vec![0.1, f32::NAN, 0.3]);
    writer.add_document(nan_dense).await.unwrap_err();
    assert_eq!(writer.pending_docs(), 2);

    // Non-finite sparse weight.
    let mut nan_sparse = Document::new();
    nan_sparse.add_text(title, "nan sparse");
    nan_sparse.add_sparse_vector(spv, vec![(3, f32::INFINITY)]);
    writer.add_document(nan_sparse).await.unwrap_err();
    assert_eq!(writer.pending_docs(), 2);

    // Binary vector with the wrong byte length.
    let mut bad_bin = Document::new();
    bad_bin.add_text(title, "bad bin");
    bad_bin.add_binary_dense_vector(bin, vec![0xFF, 0xFF]);
    writer.add_document(bad_bin).await.unwrap_err();
    assert_eq!(writer.pending_docs(), 2);

    // The writer keeps accepting valid documents after the rejections...
    for i in 2..4 {
        writer.add_document(valid_doc(i)).await.expect("valid doc");
    }
    assert_eq!(writer.pending_docs(), 4);

    // ...and commit succeeds with exactly the 4 valid documents.
    let committed = writer
        .commit()
        .await
        .expect("commit must not fail after a rejected document");
    assert!(committed);
    assert_eq!(total_docs(writer.metadata()), 4);
}

// ---------------------------------------------------------------------------
// Finding: commit() drains pending_segments into in-memory metadata BEFORE the
// durable save; when save fails, a retried commit() returns Ok(false) and the
// segments are never durably registered — silent data loss on reload.
// ---------------------------------------------------------------------------

type BoxFut<'r, T> = Pin<Box<dyn Future<Output = T> + Send + 'r>>;

/// DirectoryWriter test double: delegates to an inner `RamDirectory` but fails
/// `rename` while `fail_renames` is set. `IndexMetadata::save` publishes
/// metadata.json via write-tmp-then-rename, so this makes exactly the durable
/// metadata save fail while segment file writes stay healthy.
struct RenameFailDirectory {
    inner: RamDirectory,
    fail_renames: Arc<AtomicBool>,
}

impl Directory for RenameFailDirectory {
    fn exists<'a, 'b, 'r>(&'a self, path: &'b Path) -> BoxFut<'r, io::Result<bool>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.exists(path))
    }

    fn file_size<'a, 'b, 'r>(&'a self, path: &'b Path) -> BoxFut<'r, io::Result<u64>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.file_size(path))
    }

    fn open_read<'a, 'b, 'r>(&'a self, path: &'b Path) -> BoxFut<'r, io::Result<FileHandle>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.open_read(path))
    }

    fn read_range<'a, 'b, 'r>(
        &'a self,
        path: &'b Path,
        range: Range<u64>,
    ) -> BoxFut<'r, io::Result<OwnedBytes>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.read_range(path, range))
    }

    fn list_files<'a, 'b, 'r>(&'a self, prefix: &'b Path) -> BoxFut<'r, io::Result<Vec<PathBuf>>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.list_files(prefix))
    }

    fn open_lazy<'a, 'b, 'r>(&'a self, path: &'b Path) -> BoxFut<'r, io::Result<FileHandle>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.open_lazy(path))
    }
}

impl DirectoryWriter for RenameFailDirectory {
    fn write<'a, 'b, 'c, 'r>(&'a self, path: &'b Path, data: &'c [u8]) -> BoxFut<'r, io::Result<()>>
    where
        'a: 'r,
        'b: 'r,
        'c: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.write(path, data))
    }

    fn delete<'a, 'b, 'r>(&'a self, path: &'b Path) -> BoxFut<'r, io::Result<()>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.delete(path))
    }

    fn rename<'a, 'b, 'c, 'r>(&'a self, from: &'b Path, to: &'c Path) -> BoxFut<'r, io::Result<()>>
    where
        'a: 'r,
        'b: 'r,
        'c: 'r,
        Self: 'r,
    {
        Box::pin(async move {
            if self.fail_renames.load(Ordering::SeqCst) {
                return Err(io::Error::other("injected rename failure"));
            }
            self.inner.rename(from, to).await
        })
    }

    fn sync<'a, 'r>(&'a self) -> BoxFut<'r, io::Result<()>>
    where
        'a: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.sync())
    }

    fn streaming_writer<'a, 'b, 'r>(
        &'a self,
        path: &'b Path,
    ) -> BoxFut<'r, io::Result<Box<dyn StreamingWriter>>>
    where
        'a: 'r,
        'b: 'r,
        Self: 'r,
    {
        Box::pin(self.inner.streaming_writer(path))
    }
}

#[tokio::test]
async fn test_wasm_commit_failed_metadata_save_is_retryable() {
    let fail_renames = Arc::new(AtomicBool::new(false));
    let dir = RenameFailDirectory {
        inner: RamDirectory::new(),
        fail_renames: Arc::clone(&fail_renames),
    };

    let mut builder = Schema::builder();
    let title = builder.add_text_field("title", true, true);
    let schema = builder.build();

    let mut writer = WasmIndexWriter::create(dir, schema, IndexConfig::default())
        .await
        .expect("create");

    let mut doc = Document::new();
    doc.add_text(title, "hello world");
    writer.add_document(doc).await.expect("add");

    // First commit: metadata save fails after the segment files are built.
    fail_renames.store(true, Ordering::SeqCst);
    writer
        .commit()
        .await
        .expect_err("commit must surface the failed metadata save");

    // A failed save must leave the pending segments intact so the commit is
    // retryable once the storage recovers.
    fail_renames.store(false, Ordering::SeqCst);
    let committed = writer.commit().await.expect("retried commit");
    assert!(
        committed,
        "retried commit must re-attempt the durable metadata save (pending \
         segments must survive a failed save)"
    );

    // The durable metadata must now reference the committed segment.
    let durable = IndexMetadata::load(writer.directory().as_ref())
        .await
        .expect("load metadata");
    assert_eq!(total_docs(&durable), 1);
    assert_eq!(total_docs(writer.metadata()), 1);
}

// ---------------------------------------------------------------------------
// Finding: schemas with a `[primary]` field are silently accepted by the wasm
// writer even though no primary-key deduplication exists on the wasm branch.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_wasm_writer_create_rejects_primary_key_schema_loudly() {
    let mut builder = Schema::builder();
    let id = builder.add_text_field("id", true, true);
    builder.add_text_field("title", true, true);
    builder.set_primary_key(id);
    let schema = builder.build();

    let err =
        match WasmIndexWriter::create(RamDirectory::new(), schema, IndexConfig::default()).await {
            Ok(_) => panic!("wasm writer must reject primary-key schemas: dedup is not enforced"),
            Err(e) => e,
        };
    let msg = err.to_string();
    assert!(
        msg.contains("primary key") && msg.contains("id"),
        "error must name the primary-key field and the missing capability: {msg}"
    );
}

#[tokio::test]
async fn test_wasm_writer_open_rejects_primary_key_schema_loudly() {
    let mut builder = Schema::builder();
    let id = builder.add_text_field("id", true, true);
    builder.set_primary_key(id);
    let schema = builder.build();

    // Simulate an existing (e.g. natively built) index with a PK schema.
    let dir = RamDirectory::new();
    IndexMetadata::new(schema)
        .save(&dir)
        .await
        .expect("save metadata");

    let err = match WasmIndexWriter::open(dir, IndexConfig::default()).await {
        Ok(_) => panic!("wasm writer must refuse to open an index whose schema declares a PK"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("primary key") && msg.contains("id"),
        "error must name the primary-key field and the missing capability: {msg}"
    );
}
