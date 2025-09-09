use crate::directories::RamDirectory;
use crate::dsl::{Document, Field, SchemaBuilder};
use crate::error::Error;
use crate::index::{IndexConfig, IndexWriter};

#[derive(Clone)]
struct BlockingMetadataDirectory {
    inner: RamDirectory,
    block_next_rename: std::sync::Arc<std::sync::atomic::AtomicBool>,
    block_next_row_stats: std::sync::Arc<std::sync::atomic::AtomicBool>,
    fail_next_rename: std::sync::Arc<std::sync::atomic::AtomicBool>,
    fail_pk_refresh_after_rename: std::sync::Arc<std::sync::atomic::AtomicBool>,
    fail_next_fast_read: std::sync::Arc<std::sync::atomic::AtomicBool>,
    panic_next_bloom_write: std::sync::Arc<std::sync::atomic::AtomicBool>,
    rename_started: std::sync::Arc<tokio::sync::Semaphore>,
    allow_rename: std::sync::Arc<tokio::sync::Semaphore>,
}

impl Default for BlockingMetadataDirectory {
    fn default() -> Self {
        Self {
            inner: RamDirectory::default(),
            block_next_rename: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            block_next_row_stats: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            fail_next_rename: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            fail_pk_refresh_after_rename: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                false,
            )),
            fail_next_fast_read: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
        if path
            .extension()
            .is_some_and(|extension| extension == "fast")
            && self
                .fail_next_fast_read
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Err(std::io::Error::other(
                "injected PK visibility refresh failure",
            ));
        }
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
        self.inner.rename(from, to).await?;
        if to == std::path::Path::new(crate::index::INDEX_META_FILENAME)
            && self
                .fail_pk_refresh_after_rename
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            self.fail_next_fast_read
                .store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    async fn sync(&self) -> std::io::Result<()> {
        self.inner.sync().await
    }

    async fn streaming_writer(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<Box<dyn crate::directories::StreamingWriter>> {
        if path
            .extension()
            .is_some_and(|extension| extension == "rowstats")
            && self
                .block_next_row_stats
                .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            self.rename_started.add_permits(1);
            self.allow_rename.acquire().await.unwrap().forget();
        }
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

#[tokio::test]
async fn test_force_merge_refreshes_primary_key_snapshot_per_batch() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema, config)
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    writer
        .add_document(make_doc(pk, title, "first", "one"))
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "second", "two"))
        .unwrap();
    writer.commit().await.unwrap();

    let old_segment_ids = writer.segment_manager().get_segment_ids().await;
    assert_eq!(old_segment_ids.len(), 2);
    let tracker = writer.segment_manager().tracker();
    assert!(
        old_segment_ids
            .iter()
            .all(|segment_id| tracker.ref_count(segment_id) > 0),
        "the primary-key index must hold the pre-merge snapshot"
    );

    writer.force_merge().await.unwrap();
    writer.segment_manager().wait_for_shutdown().await;

    assert_eq!(writer.segment_manager().get_segment_ids().await.len(), 1);
    for old_id in old_segment_ids {
        assert_eq!(
            tracker.ref_count(&old_id),
            0,
            "force merge retained an old primary-key segment snapshot"
        );
        let segment_id = crate::segment::SegmentId::from_hex(&old_id).unwrap();
        let old_paths = crate::segment::SegmentFiles::new(segment_id.0).lifecycle_paths();
        let remaining = dir.list_files_sync(std::path::Path::new("")).unwrap();
        assert!(
            old_paths.iter().all(|path| !remaining.contains(path)),
            "retired source files remain after the primary-key snapshot refresh"
        );
    }

    let error = writer
        .add_document(make_doc(pk, title, "first", "duplicate"))
        .unwrap_err();
    assert!(matches!(error, Error::DuplicatePrimaryKey(_)));
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

#[tokio::test]
async fn deleted_primary_keys_survive_bloom_cache_reopen_without_blocking_reuse() {
    for reused in ["reused", "RETAINED", "ключ/日本語"] {
        let (schema, pk, title) = make_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
            .await
            .unwrap();
        writer.init_primary_key_dedup().await.unwrap();
        writer
            .add_document(make_doc(pk, title, reused, "old"))
            .unwrap();
        writer
            .add_document(make_doc(pk, title, "retained", "stable"))
            .unwrap();
        writer.commit().await.unwrap();
        writer.delete_primary_key(reused).unwrap();
        writer.commit().await.unwrap();
        writer.shutdown().await.unwrap();
        drop(writer);
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.init_primary_key_dedup().await.unwrap();
        // The persisted Bloom still contains this key. Visibility, not a Bloom
        // positive, decides whether the exact dictionary match is a duplicate.
        writer
            .add_document(make_doc(pk, title, reused, "new"))
            .unwrap();
        assert!(matches!(
            writer.add_document(make_doc(pk, title, "retained", "duplicate")),
            Err(Error::DuplicatePrimaryKey(_))
        ));
        assert!(matches!(
            writer.add_document(make_doc(pk, title, reused, "duplicate")),
            Err(Error::DuplicatePrimaryKey(_))
        ));
        writer.commit().await.unwrap();
        writer.force_merge().await.unwrap();
        for key in [reused, "retained"] {
            assert!(matches!(
                writer.add_document(make_doc(pk, title, key, "duplicate")),
                Err(Error::DuplicatePrimaryKey(_))
            ));
        }
        writer.shutdown().await.unwrap();
        drop(writer);
        let mut reopened = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        reopened.init_primary_key_dedup().await.unwrap();
        assert!(matches!(
            reopened.add_document(make_doc(pk, title, reused, "duplicate")),
            Err(Error::DuplicatePrimaryKey(_))
        ));
        let index = crate::index::Index::open(dir, config).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), 2);
    }
}

#[tokio::test]
async fn aborted_and_failed_upserts_preserve_old_primary_key_rows() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "same", "old"))
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .upsert_document(make_doc(pk, title, "same", "aborted"))
        .unwrap();
    writer.prepare_commit().await.unwrap().abort();
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "same", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    writer
        .upsert_document(make_doc(pk, title, "same", "new"))
        .unwrap();
    dir.fail_next_metadata_rename();
    assert!(writer.commit().await.is_err());
    let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
    assert!(
        metadata
            .segment_metas
            .values()
            .all(|info| info.deletions.is_none())
    );
    writer.commit().await.unwrap();
    let index = crate::index::Index::open(dir, IndexConfig::default())
        .await
        .unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 1);
    assert_eq!(
        searcher
            .search(&crate::query::TermQuery::new(title, "new"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        searcher
            .search(&crate::query::TermQuery::new(title, "old"), 10)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn cancelled_upsert_commit_publishes_deletion_and_replacement_together() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "same", "old"))
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .upsert_document(make_doc(pk, title, "same", "new"))
        .unwrap();
    let writer = std::sync::Arc::new(tokio::sync::Mutex::new(writer));
    dir.block_next_metadata_rename();
    let operation = {
        let writer = std::sync::Arc::clone(&writer);
        tokio::spawn(async move { writer.lock().await.commit().await })
    };
    dir.wait_until_rename_started().await;
    operation.abort();
    assert!(operation.await.unwrap_err().is_cancelled());
    dir.release_rename();
    let mut guard = writer.lock().await;
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        guard.wait_for_commit_finalization(),
    )
    .await
    .unwrap();
    assert!(matches!(
        guard.add_document(make_doc(pk, title, "same", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    guard.commit().await.unwrap();
    let index = crate::index::Index::open(dir, IndexConfig::default())
        .await
        .unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 1);
    assert_eq!(
        searcher
            .search(&crate::query::TermQuery::new(title, "new"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn compaction_rejects_stale_visibility_instead_of_resurrecting_concurrent_deletes() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    for key in ["first", "second", "third"] {
        writer
            .add_document(make_doc(pk, title, key, "present"))
            .unwrap();
    }
    writer.commit().await.unwrap();
    writer.delete_primary_key("first").unwrap();
    writer.commit().await.unwrap();
    let manager = std::sync::Arc::clone(writer.segment_manager());
    let id = crate::index::IndexMetadata::load(&dir)
        .await
        .unwrap()
        .segment_ids()[0]
        .clone();
    dir.block_next_row_stats
        .store(true, std::sync::atomic::Ordering::Release);
    let compact = {
        let manager = std::sync::Arc::clone(&manager);
        let id = id.clone();
        tokio::spawn(async move { manager.compact_segment(&id, 16 * 1024 * 1024).await })
    };
    dir.wait_until_rename_started().await;
    writer.delete_primary_key("second").unwrap();
    writer.commit().await.unwrap();
    dir.release_rename();
    let error = compact.await.unwrap().unwrap_err();
    assert!(error.to_string().contains("visibility changed"), "{error}");
    manager
        .compact_segment(&id, 16 * 1024 * 1024)
        .await
        .unwrap();
    let index = crate::index::Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 1);
    assert_eq!(searcher.segment_readers()[0].num_docs(), 1);
    let hits = searcher.search(&crate::query::AllQuery, 10).await.unwrap();
    assert_eq!(
        searcher
            .doc(hits[0].segment_id, hits[0].doc_id)
            .await
            .unwrap()
            .unwrap()
            .get_first(pk)
            .unwrap()
            .as_text(),
        Some("third")
    );
    writer
        .add_document(make_doc(pk, title, "second", "reused"))
        .unwrap();
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "third", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
}

#[tokio::test]
async fn cancelled_compaction_retains_ownership_and_shutdown_drains_its_blocking_writer() {
    use crate::directories::Directory;
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    for key in ["deleted", "live"] {
        writer
            .add_document(make_doc(pk, title, key, "present"))
            .unwrap();
    }
    writer.commit().await.unwrap();
    writer.delete_primary_key("deleted").unwrap();
    writer.commit().await.unwrap();
    let manager = std::sync::Arc::clone(writer.segment_manager());
    let id = crate::index::IndexMetadata::load(&dir)
        .await
        .unwrap()
        .segment_ids()[0]
        .clone();
    // Drain any cleanup from commit before capturing the exact source file set.
    manager.cleanup_orphan_segments().await.unwrap();
    let mut before = dir.list_files(std::path::Path::new("")).await.unwrap();
    before.sort();
    dir.block_next_row_stats
        .store(true, std::sync::atomic::Ordering::Release);
    let compact = {
        let manager = std::sync::Arc::clone(&manager);
        tokio::spawn(async move { manager.compact_segment(&id, 16 * 1024 * 1024).await })
    };
    dir.wait_until_rename_started().await;
    compact.abort();
    assert!(compact.await.unwrap_err().is_cancelled());
    manager.begin_shutdown();
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(25),
            manager.wait_for_shutdown()
        )
        .await
        .is_err()
    );
    dir.release_rename();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        manager.wait_for_shutdown(),
    )
    .await
    .unwrap();
    let mut after = dir.list_files(std::path::Path::new("")).await.unwrap();
    after.sort();
    assert_eq!(before, after, "cancelled compaction left a partial output");
}

#[tokio::test]
async fn primary_key_visibility_uses_global_ordinals_after_stacking_dictionaries() {
    use crate::directories::Directory;
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    for batch in [["z", "b"], ["a", "c"]] {
        for key in batch {
            writer
                .add_document(make_doc(pk, title, key, "present"))
                .unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    let index = crate::index::Index::open(dir.clone(), config)
        .await
        .unwrap();
    let reader = index.reader().await.unwrap();
    let physical = reader.searcher().await.unwrap();
    assert_eq!(
        reader.searcher().await.unwrap().segment_readers()[0]
            .fast_field(pk.0)
            .unwrap()
            .num_blocks(),
        2
    );
    writer.delete_primary_key("b").unwrap();
    writer.commit().await.unwrap();
    // Compare the complete persisted mask with the former scalar text scan,
    // including a source whose local dictionaries have different orderings.
    let source = &physical.segment_readers()[0];
    let mut expected = crate::query::DocBitset::all(source.num_docs());
    for doc in 0..source.num_docs() {
        if source.fast_field(pk.0).unwrap().get_text(doc) == Some("b") {
            expected.clear(doc);
        }
    }
    let scratch = RamDirectory::new();
    let reference = crate::segment::deletion::write(
        &scratch,
        crate::segment::SegmentId::new(),
        source.num_docs(),
        &expected,
    )
    .await
    .unwrap();
    let metadata = crate::index::IndexMetadata::load(&dir).await.unwrap();
    let actual = metadata
        .segment_metas
        .values()
        .next()
        .unwrap()
        .deletions
        .as_ref()
        .unwrap();
    let reference_bytes = scratch
        .open_read(&reference.path().unwrap())
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    let actual_bytes = dir
        .open_read(&actual.path().unwrap())
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap();
    assert_eq!(actual_bytes.as_slice(), reference_bytes.as_slice());
    for key in ["z", "a", "c"] {
        assert!(matches!(
            writer.add_document(make_doc(pk, title, key, "duplicate")),
            Err(Error::DuplicatePrimaryKey(_))
        ));
    }
    writer
        .add_document(make_doc(pk, title, "b", "reused"))
        .unwrap();
    writer.commit().await.unwrap();
}

#[tokio::test]
async fn failed_visibility_refresh_and_abort_do_not_replay_published_upsert_deletions() {
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let index = crate::index::Index::create(
        dir.clone(),
        schema,
        IndexConfig {
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for key in ["updated", "deleted", "untouched"] {
        writer
            .add_document(make_doc(pk, title, key, "old"))
            .unwrap();
    }
    writer.commit().await.unwrap();
    writer
        .upsert_document(make_doc(pk, title, "updated", "replacement"))
        .unwrap();
    writer.delete_primary_key("deleted").unwrap();
    dir.fail_pk_refresh_after_rename
        .store(true, std::sync::atomic::Ordering::Release);
    assert!(
        writer.commit().await.unwrap(),
        "durable publication must still succeed"
    );
    assert!(
        !dir.fail_next_fast_read
            .load(std::sync::atomic::Ordering::Acquire),
        "injected read error was not consumed"
    );
    assert!(matches!(
        writer.delete_primary_key("updated"),
        Err(Error::CommitInProgress)
    ));
    writer.prepare_commit().await.unwrap().abort();
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "updated", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    writer
        .add_document(make_doc(pk, title, "recovery", "new"))
        .unwrap();
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 3);
    assert_eq!(
        searcher
            .search(&crate::query::TermQuery::new(title, "replacement"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
    writer
        .add_document(make_doc(pk, title, "deleted", "reused"))
        .unwrap();
    writer.force_merge_with_compaction(true).await.unwrap();
    reader.reload().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 4);
    assert_eq!(searcher.segment_readers()[0].num_docs(), 4);
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "updated", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
}

#[tokio::test]
async fn ambiguous_or_undeletable_primary_keys_are_rejected_before_mutation() {
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "original", "old"))
        .unwrap();
    writer.commit().await.unwrap();

    let mut ambiguous = make_doc(pk, title, "original", "replacement");
    ambiguous.add_text(pk, "different");
    assert!(matches!(
        writer.upsert_document(ambiguous),
        Err(Error::Document(_))
    ));
    let mut ambiguous = make_doc(pk, title, "fresh", "insertion");
    ambiguous.add_text(pk, "different");
    assert!(matches!(
        writer.add_document(ambiguous),
        Err(Error::Document(_))
    ));
    assert!(matches!(
        writer.add_document(make_doc(pk, title, &"x".repeat(65_537), "long")),
        Err(Error::Document(_))
    ));
    assert!(
        !writer.commit().await.unwrap(),
        "rejected mutations staged changes"
    );
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "original", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    writer
        .add_document(make_doc(pk, title, "fresh", "valid"))
        .unwrap();
    writer.commit().await.unwrap();
    writer.delete_primary_key("fresh").unwrap();
    writer.commit().await.unwrap();
    writer.force_merge_with_compaction(true).await.unwrap();
    let index = crate::index::Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 1);
    let hits = searcher
        .search(&crate::query::TermQuery::new(title, "old"), 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_merge_carries_concurrent_deletes_upserts_and_pending_key_reservations() {
    use std::collections::BTreeMap;
    let (schema, pk, title) = make_schema();
    let dir = BlockingMetadataDirectory::default();
    let index = crate::index::Index::create(
        dir.clone(),
        schema,
        IndexConfig {
            num_indexing_threads: 1,
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for range in [0..65, 65..132] {
        for key in range {
            writer
                .add_document(make_doc(pk, title, &format!("key{key}"), "old"))
                .unwrap();
        }
        writer.commit().await.unwrap();
    }
    let reader = index.reader().await.unwrap();
    let original = reader.searcher().await.unwrap();
    for key in [0, 64] {
        writer.delete_primary_key(&format!("key{key}")).unwrap();
    }
    writer.commit().await.unwrap();
    reader.reload().await.unwrap();
    let before_merge = reader.searcher().await.unwrap();
    dir.block_next_row_stats
        .store(true, std::sync::atomic::Ordering::Release);
    let manager = std::sync::Arc::clone(writer.segment_manager());
    let merge = tokio::spawn(async move { manager.force_merge().await });
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        dir.wait_until_rename_started(),
    )
    .await
    .unwrap();
    writer
        .upsert_document(make_doc(pk, title, "key1", "replacement"))
        .unwrap();
    for key in [65, 131] {
        writer.delete_primary_key(&format!("key{key}")).unwrap();
    }
    writer.commit().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "pending", "unpublished"))
        .unwrap();
    dir.release_rename();
    tokio::time::timeout(std::time::Duration::from_secs(10), merge)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "pending", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    assert!(matches!(
        writer.add_document(make_doc(pk, title, "key1", "duplicate")),
        Err(Error::DuplicatePrimaryKey(_))
    ));
    reader.reload().await.unwrap();
    let merged = reader.searcher().await.unwrap();
    assert_eq!(
        merged
            .segment_readers()
            .iter()
            .map(|segment| segment.num_docs())
            .sum::<u32>(),
        133
    );
    for (snapshot, removed, replacement) in [
        (&original, vec![], false),
        (&before_merge, vec![0, 64], false),
        (&merged, vec![0, 64, 65, 131], true),
    ] {
        let mut actual = BTreeMap::new();
        for hit in snapshot.search(&crate::query::AllQuery, 200).await.unwrap() {
            let doc = snapshot
                .doc(hit.segment_id, hit.doc_id)
                .await
                .unwrap()
                .unwrap();
            let key = doc.get_first(pk).unwrap().as_text().unwrap().to_owned();
            assert!(
                actual
                    .insert(
                        key,
                        doc.get_first(title).unwrap().as_text().unwrap().to_owned()
                    )
                    .is_none(),
                "duplicate live key"
            );
        }
        let expected: BTreeMap<_, _> = (0..132)
            .filter(|key| !removed.contains(key))
            .map(|key| {
                (
                    format!("key{key}"),
                    if replacement && key == 1 {
                        "replacement"
                    } else {
                        "old"
                    }
                    .to_owned(),
                )
            })
            .collect();
        assert_eq!(actual, expected);
    }
    writer.prepare_commit().await.unwrap().abort();
    writer.force_merge_with_compaction(true).await.unwrap();
    reader.reload().await.unwrap();
    let compacted = reader.searcher().await.unwrap();
    assert_eq!(compacted.num_docs(), 128);
    assert_eq!(compacted.segment_readers()[0].num_docs(), 128);
    assert_eq!(
        original
            .search(&crate::query::AllQuery, 200)
            .await
            .unwrap()
            .len(),
        132
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn upsert_commit_finishes_while_background_maintenance_pool_is_occupied() {
    use std::sync::Arc;
    use std::time::Duration;
    let (schema, pk, title) = make_schema();
    let dir = RamDirectory::new();
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap(),
    );
    let config = IndexConfig {
        background_reorder_pool: Some(pool.clone()),
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    writer
        .add_document(make_doc(pk, title, "same", "old"))
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .upsert_document(make_doc(pk, title, "same", "replacement"))
        .unwrap();
    let (release, blocked) = std::sync::mpsc::channel();
    let (started, ready) = tokio::sync::oneshot::channel();
    pool.spawn(move || {
        let _ = started.send(());
        let _ = blocked.recv();
    });
    ready.await.unwrap();
    let committed = tokio::time::timeout(Duration::from_secs(3), writer.commit()).await;
    // Always release the pool before asserting so a failing regression cannot
    // strand the commit finalizer or the test runtime during shutdown.
    release.send(()).unwrap();
    writer.wait_for_commit_finalization().await;
    committed
        .expect("publication queued behind bulk maintenance")
        .unwrap();
    let index = crate::index::Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(searcher.num_docs(), 1);
    assert_eq!(
        searcher
            .search(&crate::query::TermQuery::new(title, "replacement"), 10)
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        searcher
            .search(&crate::query::TermQuery::new(title, "old"), 10)
            .await
            .unwrap()
            .is_empty()
    );
}
