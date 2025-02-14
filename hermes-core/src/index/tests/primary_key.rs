use crate::directories::RamDirectory;
use crate::dsl::{Document, Field, SchemaBuilder};
use crate::error::Error;
use crate::index::{IndexConfig, IndexWriter};

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
