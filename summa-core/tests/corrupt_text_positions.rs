#![cfg(feature = "sync")]

use std::path::Path;

use summa_core::directories::{Directory, DirectoryWriter};
use summa_core::dsl::PositionMode;
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::PhraseQuery;
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn query_position_views_trust_payloads_and_old_reader_retains_its_bytes() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    schema.set_positions(field, PositionMode::TokenPosition);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
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
    // Ordinary search opens the writer-owned view without auditing position blocks.
    // Explicit PositionStream::open corruption coverage lives with the format.
    let segment = &searcher.segment_readers()[0];
    assert!(
        segment
            .get_positions(field, b"alpha")
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        segment
            .get_positions_sync(field, b"alpha")
            .unwrap()
            .is_some()
    );
    let old_result = old_searcher
        .search_with_offset_and_count(&query, 10, 0)
        .await
        .unwrap();
    assert_eq!(old_result.1, expected.1);
    assert_eq!(old_result.0.len(), expected.0.len());
}
