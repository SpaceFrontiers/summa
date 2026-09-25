use super::*;
use crate::directories::RamDirectory;
use crate::dsl::SchemaBuilder;

#[tokio::test]
async fn rejected_replacement_keeps_the_previous_staged_row_and_hash() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let hash = schema.add_text_field("hash", false, true);
    schema.set_content_hash(hash);
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(
        dir.clone(),
        schema.build(),
        IndexConfig {
            num_threads: 1,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    let doc = |value: &str| {
        let mut doc = Document::new();
        doc.add_text(id, "key");
        doc.add_text(hash, value);
        doc
    };
    writer.add_document(doc("first")).unwrap();
    // Hold a bounded queue without a consumer to deterministically reject the
    // replacement, whether the original worker already consumed the first row.
    let (blocked, receiver) = async_channel::bounded(1);
    blocked
        .try_send(QueuedDocument {
            doc: Document::new(),
            row: None,
        })
        .unwrap();
    let original = std::mem::replace(&mut *writer.doc_sender.write(), blocked);
    assert!(matches!(
        writer.upsert_document(doc("rejected")).await,
        Err(Error::QueueFull)
    ));
    *writer.doc_sender.write() = original;
    drop(receiver);
    writer.upsert_document(doc("first")).await.unwrap();
    // Budget rejection is also transactional, including before a hash clone.
    let oversized = "x".repeat(64 * 1024 * 1024);
    assert!(
        writer
            .upsert_document(doc(&oversized))
            .await
            .unwrap_err()
            .to_string()
            .contains("64 MiB")
    );
    drop(oversized);
    writer.commit().await.unwrap();
    let index = super::super::Index::open(dir, IndexConfig::default())
        .await
        .unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let hits = searcher.search(&crate::query::AllQuery, 10).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(
        searcher
            .doc(hits[0].segment_id, hits[0].doc_id)
            .await
            .unwrap()
            .unwrap()
            .get_first(hash)
            .unwrap()
            .as_text(),
        Some("first")
    );
    assert_eq!(
        searcher
            .segment_readers()
            .iter()
            .map(|segment| segment.num_docs())
            .sum::<u32>(),
        1
    );
}
