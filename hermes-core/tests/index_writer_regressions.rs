//! Regression tests for IndexWriter lifecycle safety:
//!
//! 1. `create()` on a directory that already contains an index must fail
//!    loudly instead of silently replacing metadata.json (which orphans every
//!    committed segment for the next orphan sweep to delete).
//! 2. A second writer for the same filesystem index directory must be
//!    rejected via the advisory single-writer lock, not silently allowed to
//!    double-write and sweep the first writer's in-flight outputs.
//! 3. `PreparedCommit::abort()` must not clear the fail-closed primary-key
//!    reservations retained after a failed post-commit PK refresh — wiping
//!    them admits duplicate primary keys.
//! 4. A commit cancelled mid-flush (client disconnect drops the future)
//!    leaves a detached waiter on the flush condvar; the retried commit must
//!    still observe the flush completion instead of stalling to its deadline.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use hermes_core::directories::{
    Directory, DirectoryWriter, FileHandle, OwnedBytes, RamDirectory, StreamingWriter,
};
use hermes_core::{Document, Error, Index, IndexConfig, IndexWriter, MmapDirectory, Schema};

/// RamDirectory wrapper with two test hooks:
/// - `fail_fast_reads`: reads of `*.fast` files fail (injects a transient IO
///   error into the post-commit primary-key refresh).
/// - `gate_open`: while false, `streaming_writer` blocks, holding worker
///   segment flushes open at a deterministic point.
#[derive(Clone)]
struct HookedDirectory {
    inner: RamDirectory,
    fail_fast_reads: Arc<AtomicBool>,
    gate_open: Arc<AtomicBool>,
}

impl HookedDirectory {
    fn new() -> Self {
        Self {
            inner: RamDirectory::new(),
            fail_fast_reads: Arc::new(AtomicBool::new(false)),
            gate_open: Arc::new(AtomicBool::new(true)),
        }
    }

    fn injected_fast_failure(&self, path: &Path) -> Option<std::io::Error> {
        if self.fail_fast_reads.load(Ordering::Acquire)
            && path.extension().is_some_and(|ext| ext == "fast")
        {
            // NOT ErrorKind::NotFound: the fast-field loader treats missing
            // files as "no fast fields" instead of an error.
            return Some(std::io::Error::other("injected .fast read failure"));
        }
        None
    }
}

#[async_trait]
impl Directory for HookedDirectory {
    async fn exists(&self, path: &Path) -> std::io::Result<bool> {
        self.inner.exists(path).await
    }

    async fn file_size(&self, path: &Path) -> std::io::Result<u64> {
        self.inner.file_size(path).await
    }

    async fn open_read(&self, path: &Path) -> std::io::Result<FileHandle> {
        if let Some(error) = self.injected_fast_failure(path) {
            return Err(error);
        }
        self.inner.open_read(path).await
    }

    async fn read_range(
        &self,
        path: &Path,
        range: std::ops::Range<u64>,
    ) -> std::io::Result<OwnedBytes> {
        if let Some(error) = self.injected_fast_failure(path) {
            return Err(error);
        }
        self.inner.read_range(path, range).await
    }

    async fn list_files(&self, prefix: &Path) -> std::io::Result<Vec<PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(&self, path: &Path) -> std::io::Result<FileHandle> {
        if let Some(error) = self.injected_fast_failure(path) {
            return Err(error);
        }
        self.inner.open_lazy(path).await
    }
}

#[async_trait]
impl DirectoryWriter for HookedDirectory {
    async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
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
        while !self.gate_open.load(Ordering::Acquire) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        self.inner.streaming_writer(path).await
    }
}

fn title_schema() -> (Schema, hermes_core::Field) {
    let mut builder = Schema::builder();
    let title = builder.add_text_field("title", true, true);
    (builder.build(), title)
}

fn title_doc(field: hermes_core::Field, text: &str) -> Document {
    let mut doc = Document::new();
    doc.add_text(field, text);
    doc
}

// ---------------------------------------------------------------------------
// 1. create() must refuse to clobber an existing index
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_create_on_existing_index_fails_instead_of_clobbering() {
    let (schema, title) = title_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    writer
        .add_document(title_doc(title, "committed document"))
        .unwrap();
    writer.commit().await.unwrap();
    drop(writer);

    // IndexWriter::create on the populated directory must fail loudly...
    let err = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .err()
        .expect("IndexWriter::create over an existing index must fail");
    assert!(
        err.to_string().contains("already exists"),
        "error must be actionable, got: {err}"
    );

    // ...and so must Index::create.
    let err = Index::create(dir.clone(), schema.clone(), config.clone())
        .await
        .err()
        .expect("Index::create over an existing index must fail");
    assert!(
        err.to_string().contains("already exists"),
        "error must be actionable, got: {err}"
    );

    // The committed data survived the refused creates.
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 1);
}

// ---------------------------------------------------------------------------
// 2. single-writer lock for filesystem-rooted directories
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_second_writer_open_is_rejected_while_writer_lock_held() {
    let (schema, title) = title_schema();
    let tmp = tempfile::tempdir().unwrap();
    let config = IndexConfig::default();

    let mut writer_a = IndexWriter::create(MmapDirectory::new(tmp.path()), schema, config.clone())
        .await
        .unwrap();
    writer_a
        .add_document(title_doc(title, "writer A owns this index"))
        .unwrap();
    writer_a.commit().await.unwrap();

    // A concurrent second writer process would sweep writer A's in-flight
    // outputs and clobber metadata.json — it must be rejected instead.
    let err = IndexWriter::open(MmapDirectory::new(tmp.path()), config.clone())
        .await
        .err()
        .expect("second writer for a locked index directory must be rejected");
    assert!(
        err.to_string().contains("single-writer lock"),
        "error must name the lock, got: {err}"
    );

    // Dropping the first writer releases the OS advisory lock.
    drop(writer_a);
    IndexWriter::open(MmapDirectory::new(tmp.path()), config)
        .await
        .expect("lock must be released when the previous writer is dropped");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_writer_from_index_defers_lock_conflict_to_first_write() {
    let (schema, title) = title_schema();
    let tmp = tempfile::tempdir().unwrap();
    let config = IndexConfig::default();

    let index = Index::create(MmapDirectory::new(tmp.path()), schema, config.clone())
        .await
        .unwrap();
    let writer_a = index.writer();

    // `Index::writer` is infallible, so the conflict surfaces on the first
    // mutating operation instead of silently double-writing.
    let writer_b = index.writer();
    let err = writer_b
        .add_document(title_doc(title, "must be rejected"))
        .expect_err("second writer's mutations must fail while the lock is held");
    assert!(
        err.to_string().contains("single-writer lock"),
        "error must name the lock, got: {err}"
    );

    // The writer holding the lock keeps working.
    writer_a.add_document(title_doc(title, "accepted")).unwrap();

    // An external writer (e.g. hermes-tool against a served index) is
    // rejected at open while the from_index writer is alive.
    let err = IndexWriter::open(MmapDirectory::new(tmp.path()), config)
        .await
        .err()
        .expect("external writer must be rejected while Index::writer is alive");
    assert!(err.to_string().contains("single-writer lock"));
}

// ---------------------------------------------------------------------------
// 3. abort must keep fail-closed PK reservations after a failed refresh
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_abort_after_failed_pk_refresh_keeps_fail_closed_reservations() {
    let mut builder = Schema::builder();
    let id = builder.add_text_field("id", true, true);
    builder.set_primary_key(id);
    let title = builder.add_text_field("title", true, true);
    let schema = builder.build();

    let dir = HookedDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    let make_doc = |key: &str| {
        let mut doc = Document::new();
        doc.add_text(id, key);
        doc.add_text(title, "body");
        doc
    };

    writer.add_document(make_doc("k1")).unwrap();

    // Publication succeeds, but the post-commit PK refresh fails (transient
    // IO error loading the new segment's fast fields). The commit still
    // reports success and deliberately retains k1's reservation as the only
    // record that k1 is committed (fail-closed).
    dir.fail_fast_reads.store(true, Ordering::Release);
    assert!(writer.commit().await.unwrap());
    dir.fail_fast_reads.store(false, Ordering::Release);

    // Sanity: the retained reservation still rejects duplicates.
    assert!(matches!(
        writer.add_document(make_doc("k1")),
        Err(Error::DuplicatePrimaryKey(_))
    ));

    // A later generation is prepared and aborted (e.g. the caller's external
    // WAL write failed). This must clear only that generation's reservations
    // — never the fail-closed ones retained above.
    writer.add_document(make_doc("k2")).unwrap();
    let prepared = writer.prepare_commit().await.unwrap();
    prepared.abort();

    // The committed key must STILL be rejected: k1 exists in a durably
    // committed segment, and admitting it here would durably commit a
    // duplicate primary key.
    let err = writer
        .add_document(make_doc("k1"))
        .expect_err("duplicate of a committed primary key must stay rejected after abort");
    assert!(
        matches!(err, Error::DuplicatePrimaryKey(ref k) if k == "k1"),
        "expected DuplicatePrimaryKey(k1), got: {err:?}"
    );

    // Fresh keys are still accepted — the writer is not wedged.
    writer.add_document(make_doc("k3")).unwrap();
}

// ---------------------------------------------------------------------------
// 4. retried prepare_commit survives a cancelled commit's dead flush waiter
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_retried_prepare_commit_survives_cancelled_commit_waiter() {
    let (schema, title) = title_schema();
    let dir = HookedDirectory::new();
    let config = IndexConfig {
        num_indexing_threads: 1,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema, config)
        .await
        .unwrap();
    writer
        .add_document(title_doc(title, "gated document"))
        .unwrap();

    // Hold the worker's segment flush open so prepare_commit blocks in its
    // flush wait.
    dir.gate_open.store(false, Ordering::Release);

    // First commit attempt is cancelled mid-flush (tonic drops the handler
    // future on client disconnect). Its spawn_blocking waiter stays parked
    // on the flush condvar.
    {
        let fut = writer.prepare_commit();
        tokio::pin!(fut);
        let poll = tokio::time::timeout(std::time::Duration::from_millis(500), &mut fut).await;
        assert!(
            poll.is_err(),
            "prepare_commit must still be waiting on the gated flush"
        );
        // Dropping the future here detaches the parked waiter.
    }

    // Release the flush shortly after the retry has parked its own waiter.
    let gate = Arc::clone(&dir.gate_open);
    let opener = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        gate.store(true, Ordering::Release);
    });

    // The retried commit must observe the flush completion promptly. With a
    // single notify_one, the dead waiter above consumes the wakeup and this
    // stalls for the full 300s flush deadline.
    let prepared =
        tokio::time::timeout(std::time::Duration::from_secs(30), writer.prepare_commit())
            .await
            .expect("retried prepare_commit must not miss the flush wakeup")
            .expect("retried prepare_commit must succeed");
    assert!(prepared.commit().await.unwrap());
    opener.await.unwrap();
}
