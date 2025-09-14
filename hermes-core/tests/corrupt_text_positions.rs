#![cfg(feature = "sync")]

use std::path::Path;

use hermes_core::directories::{Directory, DirectoryWriter};
use hermes_core::dsl::PositionMode;
use hermes_core::index::{Index, IndexConfig, IndexWriter};
use hermes_core::query::PhraseQuery;
use hermes_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn corrupt_position_blocks_error_instead_of_becoming_missing_phrase_hits() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    schema.set_positions(field, PositionMode::TokenPosition);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        posting_validation_cache_bytes: 4096,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for _ in 0..5 {
        let mut doc = Document::new();
        doc.add_text(field, "alpha beta");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);

    let old_index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let old_reader = old_index.reader().await.unwrap();
    let old_searcher = old_reader.searcher().await.unwrap();
    let query = PhraseQuery::text(field, "alpha beta");
    let expected = old_searcher
        .search_with_offset_and_count_sync(&query, 10, 0)
        .unwrap();
    assert_eq!(expected.1, 5);
    let cache_bytes = old_searcher.segment_readers()[0]
        .memory_stats()
        .posting_validation_cache_bytes;
    assert!(cache_bytes > 0 && cache_bytes <= config.posting_validation_cache_bytes);
    let path = dir
        .list_files(Path::new(""))
        .await
        .unwrap()
        .into_iter()
        .find(|p| p.extension().is_some_and(|e| e == "pos"))
        .unwrap();
    let mut bytes = dir
        .open_read(&path)
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap()
        .to_vec();
    bytes[2] = 7; // Invalid width in the first term's first position block.
    dir.write(&path, &bytes).await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let sync = searcher.search_with_offset_and_count_sync(&query, 10, 0);
    let asynchronous = searcher.search_with_offset_and_count(&query, 10, 0).await;
    for result in [sync, asynchronous] {
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("invalid position block"),
            "{error}"
        );
    }
    let old_result = old_searcher
        .search_with_offset_and_count(&query, 10, 0)
        .await
        .unwrap();
    assert_eq!(old_result.1, expected.1);
    assert_eq!(old_result.0.len(), expected.0.len());
}
