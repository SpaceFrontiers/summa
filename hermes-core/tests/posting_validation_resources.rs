#![cfg(feature = "sync")]

use std::sync::Arc;

use hermes_core::directories::MmapDirectory;
use hermes_core::index::{Index, IndexConfig};
use hermes_core::query::TermQuery;
use hermes_core::{Document, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn mmap_validation_budget_survives_reload_and_old_searcher_stays_valid() {
    let root = tempfile::tempdir().unwrap();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    let budget = 4096;
    let index = Index::create(
        MmapDirectory::new(root.path()),
        schema.build(),
        IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            posting_validation_cache_bytes: budget,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    for _ in 0..5 {
        let mut doc = Document::new();
        doc.add_text(field, "common");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let old = reader.searcher().await.unwrap();
    let query = TermQuery::text(field, "common");
    assert_eq!(
        old.search_with_offset_and_count_sync(&query, 10, 0)
            .unwrap()
            .1,
        5
    );
    let old_segment = Arc::clone(&old.segment_readers()[0]);
    let mut old_bytes = Vec::new();
    old_segment
        .get_prefix_postings(field, b"comm")
        .await
        .unwrap()[0]
        .serialize(&mut old_bytes)
        .unwrap();
    for _ in 0..5 {
        let mut doc = Document::new();
        doc.add_text(field, "common other");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    reader.reload().await.unwrap();
    let new = reader.searcher().await.unwrap();
    assert_eq!(
        new.search_with_offset_and_count(&query, 10, 0)
            .await
            .unwrap()
            .1,
        10
    );
    assert_eq!(
        old.search_with_offset_and_count(&query, 10, 0)
            .await
            .unwrap()
            .1,
        5
    );
    let reused = new
        .segment_readers()
        .iter()
        .find(|s| s.meta().id == old_segment.meta().id)
        .unwrap();
    assert!(Arc::ptr_eq(reused, &old_segment));
    let mut reused_bytes = Vec::new();
    reused.get_prefix_postings_sync(field, b"comm").unwrap()[0]
        .serialize(&mut reused_bytes)
        .unwrap();
    assert_eq!(old_bytes, reused_bytes);
    assert!(new.segment_readers().len() > old.segment_readers().len());
    for segment in new.segment_readers() {
        let heap = segment.memory_stats().posting_validation_cache_bytes;
        assert!(heap > 0 && heap <= budget);
    }
    writer.shutdown().await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_validation_budget_is_rejected_before_creating_index_files() {
    let root = tempfile::tempdir().unwrap();
    let result = Index::create(
        MmapDirectory::new(root.path()),
        SchemaBuilder::default().build(),
        IndexConfig {
            posting_validation_cache_bytes: 64 * 1024 * 1024 + 1,
            ..Default::default()
        },
    )
    .await;
    assert!(
        matches!(result, Err(error) if error.to_string().contains("posting_validation_cache_bytes"))
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}
