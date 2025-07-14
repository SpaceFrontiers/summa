//! Regression tests: `SchemaBuilder::set_primary_key` must force the same
//! field attributes the SDL path forces (fast + indexed). Without `fast`, the
//! committed-key dedup lookup silently finds no fast-field text dictionary and
//! accepts duplicates of any committed primary key.

use hermes_core::{Document, Error, IndexConfig, IndexWriter, RamDirectory, Schema};

#[tokio::test(flavor = "multi_thread")]
async fn test_builder_primary_key_dedup_across_commit_without_explicit_set_fast() {
    let mut builder = Schema::builder();
    // Deliberately NO builder.set_fast(id, true): set_primary_key must force
    // fast + indexed itself, matching the SDL path.
    let id = builder.add_text_field("id", true, true);
    builder.set_primary_key(id);
    let title = builder.add_text_field("title", true, true);
    let schema = builder.build();

    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir, schema, IndexConfig::default())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();

    let mut doc = Document::new();
    doc.add_text(id, "key1");
    doc.add_text(title, "first");
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();

    // After commit the uncommitted-key set is cleared; the duplicate must be
    // caught by the committed-key lookup (fast-field text dict).
    let mut dup = Document::new();
    dup.add_text(id, "key1");
    dup.add_text(title, "duplicate");
    let err = writer
        .add_document(dup)
        .expect_err("duplicate of a committed primary key must be rejected");
    match err {
        Error::DuplicatePrimaryKey(k) => assert_eq!(k, "key1"),
        other => panic!("expected DuplicatePrimaryKey, got {:?}", other),
    }
}
