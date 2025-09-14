#![cfg(feature = "native")]

use hermes_core::index::{Index, IndexConfig};
use hermes_core::query::{CountCollector, TermQuery, collect_segment};
use hermes_core::structures::SSTableBlockSize;
use hermes_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test]
async fn dictionary_block_and_cache_policies_survive_build_merge_compaction_and_reload() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_fast(id, true);
    schema.set_primary_key(id);
    let body = schema.add_text_field("body", true, false);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            term_dict_block_size: SSTableBlockSize::try_from(512).unwrap(),
            term_cache_blocks: 100,
            term_cache_budget_bytes: Some(1024),
            merge_policy: Box::new(hermes_core::merge::NoMergePolicy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for batch in 0..2 {
        for i in 0..2000 {
            let suffix: String = [i / 676, i / 26 % 26, i % 26]
                .into_iter()
                .map(|n| char::from(b'a' + n as u8))
                .collect();
            let mut doc = Document::new();
            doc.add_text(id, format!("id{batch}-{i}"));
            doc.add_text(body, format!("common word{suffix}"));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let reader = index.reader().await.unwrap();
    let before = reader.searcher().await.unwrap();
    assert_eq!(before.num_segments(), 2);
    for segment in before.segment_readers() {
        assert!(
            segment.term_dict_stats().num_blocks > 20,
            "builder lost the 512-byte target"
        );
    }
    writer.force_merge().await.unwrap();
    reader.reload().await.unwrap();
    let merged = reader.searcher().await.unwrap();
    assert_eq!(merged.num_segments(), 1);
    assert!(
        merged.segment_readers()[0].term_dict_stats().num_blocks > 40,
        "merge lost the 512-byte target"
    );
    let query = TermQuery::text(body, "common");
    let mut count = CountCollector::new();
    collect_segment(&merged.segment_readers()[0], &query, &mut count)
        .await
        .unwrap();
    assert_eq!(count.count(), 4000);
    writer.delete_primary_key("id0-0").unwrap();
    assert_eq!(writer.compact(4 * 1024 * 1024).await.unwrap(), 1);
    reader.reload().await.unwrap();
    let compacted = reader.searcher().await.unwrap();
    let segment = &compacted.segment_readers()[0];
    assert!(
        segment.term_dict_stats().num_blocks > 40,
        "compaction lost the 512-byte target"
    );
    let mut count = CountCollector::new();
    collect_segment(segment, &query, &mut count).await.unwrap();
    assert_eq!(count.count(), 3999);
    let mut terms = segment.term_dict_iter();
    while terms.next().await.unwrap().is_some() {
        assert!(segment.memory_stats().term_dict_cache_bytes <= 1024);
    }
    // The held, pre-replacement reader remains an immutable complete view.
    let mut count = CountCollector::new();
    collect_segment(&merged.segment_readers()[0], &query, &mut count)
        .await
        .unwrap();
    assert_eq!(count.count(), 4000);
    writer.shutdown().await.unwrap();
}
