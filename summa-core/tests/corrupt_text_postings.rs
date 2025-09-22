#![cfg(feature = "sync")]

use std::path::Path;

use summa_core::directories::{Directory, DirectoryWriter};
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::TermQuery;
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn unsupported_posting_codec_fails_without_poisoning_old_reader() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    // More than three documents forces the sole term into external postings.
    for _ in 0..5 {
        let mut doc = Document::new();
        doc.add_text(field, "common");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let old_index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let old_reader = old_index.reader().await.unwrap();
    let old_searcher = old_reader.searcher().await.unwrap();
    let query = TermQuery::text(field, "common");
    let expected = old_searcher
        .search_with_offset_and_count_sync(&query, 10, 0)
        .unwrap();
    let posts: Vec<_> = dir
        .list_files(Path::new(""))
        .await
        .unwrap()
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "post"))
        .collect();
    assert_eq!(posts.len(), 1);
    let mut bytes = dir
        .open_read(&posts[0])
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap()
        .to_vec();
    bytes[6] = 0xe1;
    dir.write(&posts[0], &bytes).await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    // The old RAM owner is immutable even when the directory publishes new bytes.
    let old_result = old_searcher
        .search_with_offset_and_count(&query, 10, 0)
        .await
        .unwrap();
    assert_eq!(old_result.1, expected.1);
    assert_eq!(old_result.0.len(), expected.0.len());
    let sync_error = searcher
        .search_with_offset_and_count_sync(&query, 10, 0)
        .unwrap_err();
    let async_error = searcher
        .search_with_offset_and_count(&query, 10, 0)
        .await
        .unwrap_err();
    assert!(
        sync_error
            .to_string()
            .contains("posting payload corruption"),
        "{sync_error}"
    );
    assert!(
        async_error
            .to_string()
            .contains("posting payload corruption"),
        "{async_error}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn merge_rejects_corrupt_single_source_term_before_publishing_metadata() {
    use std::sync::Arc;
    use summa_core::segment::{
        SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
    };
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field("text", true, false);
    let schema = Arc::new(schema.build());
    let mut ids = Vec::new();
    for term in ["alpha", "beta"] {
        let mut builder =
            SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
        for _ in 0..5 {
            let mut doc = Document::new();
            doc.add_text(text, term);
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        ids.push(id);
    }
    let posting = dir
        .list_files(Path::new(""))
        .await
        .unwrap()
        .into_iter()
        .find(|p| p.extension().is_some_and(|e| e == "post"))
        .unwrap();
    let mut bytes = dir
        .open_read(&posting)
        .await
        .unwrap()
        .read_bytes()
        .await
        .unwrap()
        .to_vec();
    bytes[6] = 0xe1;
    dir.write(&posting, &bytes).await.unwrap();
    let mut sources = Vec::new();
    for id in ids {
        sources.push(
            SegmentReader::open(&dir, id, schema.clone(), 0)
                .await
                .unwrap(),
        );
    }
    let before: Vec<_> = dir
        .list_files(Path::new(""))
        .await
        .unwrap()
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "meta"))
        .collect();
    let result = SegmentMerger::new(schema)
        .merge(&dir, &sources, SegmentId::new(), None)
        .await;
    assert!(
        matches!(result, Err(summa_core::Error::Corruption(_))),
        "{result:?}"
    );
    let mut after: Vec<_> = dir
        .list_files(Path::new(""))
        .await
        .unwrap()
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "meta"))
        .collect();
    let mut before = before;
    before.sort();
    after.sort();
    assert_eq!(
        before, after,
        "a failed merge must not publish its segment metadata"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unsupported_block_codec_fails_collection_instead_of_returning_partial_hits() {
    use summa_core::query::{CountCollector, TopKCollector, collect_segment};
    for (sync, corrupt_block) in [(false, 0), (false, 1), (true, 0), (true, 1)] {
        for text in ["common", "\"common common\"", "common OR absent"] {
            let dir = RamDirectory::new();
            let mut schema = SchemaBuilder::default();
            let field = schema.add_text_field("text", true, false);
            schema.set_positions(field, summa_core::dsl::PositionMode::TokenPosition);
            let config = IndexConfig {
                num_threads: 1,
                num_indexing_threads: 1,
                ..Default::default()
            };
            let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
                .await
                .unwrap();
            for _ in 0..260 {
                let mut doc = Document::new();
                doc.add_text(field, "common common");
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
            writer.shutdown().await.unwrap();
            drop(writer);
            let old_index = Index::open(dir.clone(), config.clone()).await.unwrap();
            let old_reader = old_index.reader().await.unwrap();
            let old_searcher = old_reader.searcher().await.unwrap();
            let path = dir
                .list_files(Path::new(""))
                .await
                .unwrap()
                .into_iter()
                .find(|p| p.extension().is_some_and(|e| e == "post"))
                .unwrap();
            let mut bytes = dir
                .open_read(&path)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap()
                .to_vec();
            let pristine = summa_core::structures::BlockPostingList::deserialize(&bytes).unwrap();
            let (offset, _, _) = pristine
                .decode_block_doc_ids_only(corrupt_block, &mut Vec::new())
                .unwrap();
            bytes[offset + 6] = 0xe1;
            dir.write(&path, &bytes).await.unwrap();
            let index = Index::open(dir, config).await.unwrap();
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            let query = searcher.query_parser().parse_strict(text).unwrap();
            let result = if sync {
                searcher.search_with_offset_and_count_sync(query.as_ref(), 260, 0)
            } else {
                searcher
                    .search_with_offset_and_count(query.as_ref(), 260, 0)
                    .await
            };
            assert!(
                matches!(result, Err(summa_core::Error::Corruption(_))),
                "{sync} {text}: {result:?}"
            );
            let healthy = old_searcher
                .search_with_offset_and_count(query.as_ref(), 260, 0)
                .await
                .unwrap();
            assert_eq!(
                healthy.0.len(),
                260,
                "replacement must not poison the old immutable reader"
            );
            // A later metadata-count shortcut must not hide known corruption.
            let segment = &searcher.segment_readers()[0];
            let mut count = CountCollector::new();
            assert!(
                collect_segment(segment, &TermQuery::text(field, "common"), &mut count)
                    .await
                    .is_err()
            );
            let mut top = TopKCollector::new(10);
            assert!(
                collect_segment(segment, query.as_ref(), &mut top)
                    .await
                    .is_err()
            );
        }
    }
}
