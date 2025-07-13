use crate::directories::RamDirectory;
use crate::dsl::{Document, Field, SchemaBuilder};
use crate::error::Error;
use crate::index::{IndexConfig, IndexWriter};

#[derive(Clone)]
struct BlockingMetadataDirectory {
    inner: RamDirectory,
    block_next_rename: std::sync::Arc<std::sync::atomic::AtomicBool>,
    fail_next_rename: std::sync::Arc<std::sync::atomic::AtomicBool>,
    panic_next_bloom_write: std::sync::Arc<std::sync::atomic::AtomicBool>,
    rename_started: std::sync::Arc<tokio::sync::Semaphore>,
    allow_rename: std::sync::Arc<tokio::sync::Semaphore>,
}

impl Default for BlockingMetadataDirectory {
    fn default() -> Self {
        Self {
            inner: RamDirectory::default(),
            block_next_rename: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            fail_next_rename: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            panic_next_bloom_write: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            rename_started: std::sync::Arc::new(tokio::sync::Semaphore::new(0)),
            allow_rename: std::sync::Arc::new(tokio::sync::Semaphore::new(0)),
        }
    }
}

impl BlockingMetadataDirectory {
    fn block_next_metadata_rename(&self) {
        self.block_next_rename
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn fail_next_metadata_rename(&self) {
        self.fail_next_rename
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn panic_next_bloom_write(&self) {
        self.panic_next_bloom_write
            .store(true, std::sync::atomic::Ordering::Release);
    }

    async fn wait_until_rename_started(&self) {
        self.rename_started.acquire().await.unwrap().forget();
    }

    fn release_rename(&self) {
        self.allow_rename.add_permits(1);
    }
}

#[async_trait::async_trait]
impl crate::directories::Directory for BlockingMetadataDirectory {
    async fn exists(&self, path: &std::path::Path) -> std::io::Result<bool> {
        self.inner.exists(path).await
    }

    async fn file_size(&self, path: &std::path::Path) -> std::io::Result<u64> {
        self.inner.file_size(path).await
    }

    async fn open_read(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<crate::directories::FileHandle> {
        self.inner.open_read(path).await
    }

    async fn read_range(
        &self,
        path: &std::path::Path,
        range: std::ops::Range<u64>,
    ) -> std::io::Result<crate::directories::OwnedBytes> {
        self.inner.read_range(path, range).await
    }

    async fn list_files(
        &self,
        prefix: &std::path::Path,
    ) -> std::io::Result<Vec<std::path::PathBuf>> {
        self.inner.list_files(prefix).await
    }

    async fn open_lazy(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<crate::directories::FileHandle> {
        self.inner.open_lazy(path).await
    }
}

#[async_trait::async_trait]
impl crate::directories::DirectoryWriter for BlockingMetadataDirectory {
    async fn write(&self, path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
        if path == std::path::Path::new(crate::index::primary_key::PK_BLOOM_FILE)
            && self
                .panic_next_bloom_write
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            panic!("injected post-publication bloom-write panic");
        }
        self.inner.write(path, data).await
    }

    async fn delete(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.inner.delete(path).await
    }

    async fn rename(&self, from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
        if to == std::path::Path::new(crate::index::INDEX_META_FILENAME)
            && self
                .fail_next_rename
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected metadata rename failure",
            ));
        }
        if to == std::path::Path::new(crate::index::INDEX_META_FILENAME)
            && self
                .block_next_rename
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            self.rename_started.add_permits(1);
            self.allow_rename.acquire().await.unwrap().forget();
        }
        self.inner.rename(from, to).await
    }

    async fn sync(&self) -> std::io::Result<()> {
        self.inner.sync().await
    }

    async fn streaming_writer(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<Box<dyn crate::directories::StreamingWriter>> {
        self.inner.streaming_writer(path).await
    }
}

/// Helper: build a schema with a primary key text field + a regular text field.
fn make_schema() -> (crate::dsl::Schema, Field, Field) {
    let mut builder = SchemaBuilder::default();
    let pk = builder.add_text_field("id", true, true);
    builder.set_fast(pk, true);
    builder.set_primary_key(pk);
    let title = builder.add_text_field("title", true, true);
    let schema = builder.build();
    (schema, pk, title)
}

fn make_doc(pk: Field, title: Field, id: &str, title_val: &str) -> Document {
    let mut doc = Document::new();
    doc.add_text(pk, id);
    doc.add_text(title, title_val);
    doc
}

// ---------------------------------------------------------------------------
// Basic dedup within a single writer session (uncommitted keys)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_rejects_duplicate_uncommitted() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir, schema, config).await.unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    // First insert succeeds
    assert!(
        writer
            .add_document(make_doc(pk, title, "doc1", "Hello"))
            .is_ok()
    );

    // Duplicate is rejected
    let err = writer
        .add_document(make_doc(pk, title, "doc1", "Hello again"))
        .unwrap_err();
    match err {
        Error::DuplicatePrimaryKey(k) => assert_eq!(k, "doc1"),
        other => panic!("Expected DuplicatePrimaryKey, got {:?}", other),
    }

    // Different key still works
    assert!(
        writer
            .add_document(make_doc(pk, title, "doc2", "World"))
            .is_ok()
    );
}

// ---------------------------------------------------------------------------
// Dedup across commit boundary (committed keys checked via fast-field dict)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_across_commit() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    writer
        .add_document(make_doc(pk, title, "key1", "First"))
        .unwrap();
    writer
        .add_document(make_doc(pk, title, "key2", "Second"))
        .unwrap();
    writer.commit().await.unwrap();

    // After commit, the committed segment's fast-field dict has key1 and key2.
    // Duplicates against committed keys should be rejected.
    let err = writer
        .add_document(make_doc(pk, title, "key1", "Duplicate"))
        .unwrap_err();
    match err {
        Error::DuplicatePrimaryKey(k) => assert_eq!(k, "key1"),
        other => panic!("Expected DuplicatePrimaryKey, got {:?}", other),
    }

    let err = writer
        .add_document(make_doc(pk, title, "key2", "Duplicate"))
        .unwrap_err();
    assert!(matches!(err, Error::DuplicatePrimaryKey(_)));

    // New key works fine
    writer
        .add_document(make_doc(pk, title, "key3", "Third"))
        .unwrap();
    writer.commit().await.unwrap();
}

// ---------------------------------------------------------------------------
// Multiple commit cycles
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_multiple_commits() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    // Commit 1
    for i in 0..10 {
        writer
            .add_document(make_doc(pk, title, &format!("k{}", i), "val"))
            .unwrap();
    }
    writer.commit().await.unwrap();

    // Commit 2 — all old keys rejected, new keys accepted
    for i in 0..10 {
        assert!(
            writer
                .add_document(make_doc(pk, title, &format!("k{}", i), "dup"))
                .is_err()
        );
    }
    for i in 10..20 {
        writer
            .add_document(make_doc(pk, title, &format!("k{}", i), "val"))
            .unwrap();
    }
    writer.commit().await.unwrap();

    // Commit 3 — all 0..20 rejected, 20..25 accepted
    for i in 0..20 {
        assert!(
            writer
                .add_document(make_doc(pk, title, &format!("k{}", i), "dup"))
                .is_err(),
            "key k{} should be rejected as duplicate",
            i
        );
    }
    for i in 20..25 {
        writer
            .add_document(make_doc(pk, title, &format!("k{}", i), "val"))
            .unwrap();
    }
    writer.commit().await.unwrap();
}

// ---------------------------------------------------------------------------
// Abort clears uncommitted keys so they can be re-inserted
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_abort_clears_uncommitted() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    writer
        .add_document(make_doc(pk, title, "abort_key", "First"))
        .unwrap();
    assert!(
        writer
            .add_document(make_doc(pk, title, "abort_key", "Dup"))
            .is_err()
    );

    // Prepare commit, then abort
    let prepared = writer.prepare_commit().await.unwrap();
    prepared.abort();

    // After abort, uncommitted keys are cleared; key should be accepted again
    // (bloom may still have it, but committed readers don't, so bloom false-positive
    // path falls through to "key not found" → accepted)
    writer
        .add_document(make_doc(pk, title, "abort_key", "Retry"))
        .unwrap();
    writer.commit().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cancelled_commit_finishes_publication_before_resuming_ingestion() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "cancelled-commit", "original"))
        .unwrap();

    let writer = std::sync::Arc::new(tokio::sync::Mutex::new(writer));
    dir.block_next_metadata_rename();
    let request = {
        let writer = std::sync::Arc::clone(&writer);
        tokio::spawn(async move { writer.lock().await.commit().await })
    };

    dir.wait_until_rename_started().await;
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());

    // The request no longer owns the writer, but the detached finalizer still
    // owns publication and must keep ingestion unavailable until it resolves.
    let error = writer
        .lock()
        .await
        .add_document(make_doc(pk, title, "another", "too early"))
        .unwrap_err();
    assert!(matches!(error, Error::CommitInProgress));

    dir.release_rename();
    let writer_guard = writer.lock().await;
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        writer_guard.wait_for_commit_finalization(),
    )
    .await
    .expect("owned commit finalizer did not finish");

    let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(metadata.segment_metas.len(), 1);

    // Most importantly, cancellation cannot clear this reservation before the
    // newly published fast field enters committed PK state.
    let duplicate = writer_guard
        .add_document(make_doc(pk, title, "cancelled-commit", "duplicate"))
        .unwrap_err();
    assert!(matches!(duplicate, Error::DuplicatePrimaryKey(_)));

    writer_guard
        .add_document(make_doc(pk, title, "after-cancel", "accepted"))
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_failed_publication_stays_paused_and_retries_losslessly() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "retry-me", "original"))
        .unwrap();

    dir.fail_next_metadata_rename();
    assert!(matches!(writer.commit().await, Err(Error::Io(_))));

    // The prepared generation remains intact and workers remain paused; new
    // ingestion gets explicit backpressure instead of mixing generations.
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "too-early", "blocked")),
        Err(Error::CommitInProgress)
    ));

    assert!(writer.commit().await.unwrap());
    let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(metadata.segment_metas.len(), 1);
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "retry-me", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    writer
        .add_document(make_doc(pk, title, "after-retry", "accepted"))
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_post_publication_finalizer_panic_still_reports_committed() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "durable-before-panic", "original"))
        .unwrap();

    dir.panic_next_bloom_write();
    assert!(writer.commit().await.unwrap());

    let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(metadata.segment_metas.len(), 1);
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "durable-before-panic", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    writer
        .add_document(make_doc(pk, title, "after-panic", "accepted"))
        .unwrap();
}

// ---------------------------------------------------------------------------
// No primary key → dedup is a no-op, duplicates are allowed
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_no_primary_key_allows_duplicates() {
    let mut builder = SchemaBuilder::default();
    let title = builder.add_text_field("title", true, true);
    let schema = builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir, schema, config).await.unwrap();
    writer.init_primary_key_dedup().await.unwrap(); // no-op

    let mut doc1 = Document::new();
    doc1.add_text(title, "same");
    let mut doc2 = Document::new();
    doc2.add_text(title, "same");

    assert!(writer.add_document(doc1).is_ok());
    assert!(writer.add_document(doc2).is_ok()); // allowed — no PK
}

// ---------------------------------------------------------------------------
// Concurrent inserts: exactly one wins
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_concurrent_inserts() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir, schema, config).await.unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    // Wrap in Arc for sharing across threads.
    // SAFETY: PrimaryKeyIndex::check_and_insert takes &self (not &mut self).
    // We need &mut for add_document though, so we use a different approach:
    // we test concurrency at the PrimaryKeyIndex level (already covered in
    // primary_key.rs unit tests). Here we test sequential rapid inserts instead.
    for i in 0..100 {
        let key = format!("concurrent_{}", i);
        assert!(
            writer
                .add_document(make_doc(pk, title, &key, "val"))
                .is_ok()
        );
    }

    // All duplicates rejected
    for i in 0..100 {
        let key = format!("concurrent_{}", i);
        assert!(
            writer
                .add_document(make_doc(pk, title, &key, "dup"))
                .is_err()
        );
    }
}

// ---------------------------------------------------------------------------
// Reopen writer on existing index with committed keys
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_reopen_existing_index() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Writer 1: add and commit keys
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    writer
        .add_document(make_doc(pk, title, "existing1", "val"))
        .unwrap();
    writer
        .add_document(make_doc(pk, title, "existing2", "val"))
        .unwrap();
    writer.commit().await.unwrap();
    drop(writer);

    // Writer 2: reopen — should reject committed keys
    let mut writer2 = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer2.init_primary_key_dedup().await.unwrap();

    let err = writer2
        .add_document(make_doc(pk, title, "existing1", "dup"))
        .unwrap_err();
    assert!(matches!(err, Error::DuplicatePrimaryKey(_)));

    let err = writer2
        .add_document(make_doc(pk, title, "existing2", "dup"))
        .unwrap_err();
    assert!(matches!(err, Error::DuplicatePrimaryKey(_)));

    // New key works
    writer2
        .add_document(make_doc(pk, title, "new_key", "val"))
        .unwrap();
    writer2.commit().await.unwrap();
}

// ---------------------------------------------------------------------------
// Batch: some succeed, some fail — partial success
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_partial_batch() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir, schema, config).await.unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    // Pre-insert a key
    writer
        .add_document(make_doc(pk, title, "pre", "existing"))
        .unwrap();

    // Batch with mix of new and duplicate keys
    let keys = ["new1", "pre", "new2", "new1", "new3"];
    let mut ok_count = 0;
    let mut dup_count = 0;
    for key in keys {
        match writer.add_document(make_doc(pk, title, key, "val")) {
            Ok(()) => ok_count += 1,
            Err(Error::DuplicatePrimaryKey(_)) => dup_count += 1,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
    // "new1" succeeds first time, "pre" is dup, "new2" succeeds,
    // "new1" second time is dup, "new3" succeeds
    assert_eq!(ok_count, 3, "3 new keys should succeed");
    assert_eq!(dup_count, 2, "2 duplicates should fail");
}

// ---------------------------------------------------------------------------
// Large batch to exercise bloom filter false-positive path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_dedup_large_batch_bloom_fps() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    // Insert 5000 keys
    for i in 0..5000 {
        writer
            .add_document(make_doc(pk, title, &format!("big_{}", i), "val"))
            .unwrap();
    }
    writer.commit().await.unwrap();

    // All 5000 should be rejected as duplicates (committed check)
    let mut false_accepts = 0;
    for i in 0..5000 {
        if writer
            .add_document(make_doc(pk, title, &format!("big_{}", i), "dup"))
            .is_ok()
        {
            false_accepts += 1;
        }
    }
    assert_eq!(
        false_accepts, 0,
        "No committed keys should be falsely accepted"
    );

    // 5000 new keys should all succeed
    for i in 5000..10000 {
        writer
            .add_document(make_doc(pk, title, &format!("big_{}", i), "val"))
            .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Schema primary_field() returns correct field
// ---------------------------------------------------------------------------

#[test]
fn test_schema_primary_field() {
    let (schema, pk, _) = make_schema();
    assert_eq!(schema.primary_field(), Some(pk));

    // Schema without primary key
    let mut builder = SchemaBuilder::default();
    builder.add_text_field("title", true, true);
    let schema = builder.build();
    assert_eq!(schema.primary_field(), None);
}
