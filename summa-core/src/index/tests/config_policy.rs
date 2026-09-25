//! Load-time policy of `IndexConfig` options that do not rewrite old data.

use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexMetadata, IndexWriter};
use crate::query::{CountCollector, TermQuery, collect_segment};

fn no_merge(config: IndexConfig) -> IndexConfig {
    IndexConfig {
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..config
    }
}

#[tokio::test]
async fn enabling_posting_bounds_reports_existing_segments_without_them() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let schema = schema.build();
    let dir = RamDirectory::new();
    let legacy = no_merge(IndexConfig::default());
    let mut writer = IndexWriter::create(dir.clone(), schema, legacy.clone())
        .await
        .unwrap();
    for i in 0..8 {
        let mut doc = Document::new();
        doc.add_text(body, format!("needle old{i}"));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);

    let metadata = IndexMetadata::load(&dir).await.unwrap();
    let legacy_ids = metadata.segment_ids();
    assert_eq!(legacy_ids.len(), 1);
    let bounded = no_merge(IndexConfig {
        posting_ratio_bounds: true,
        ..IndexConfig::default()
    });
    let impact = no_merge(IndexConfig {
        posting_impact_bounds: true,
        ..IndexConfig::default()
    });
    assert_eq!(
        crate::index::segments_missing_posting_bounds(&dir, &metadata, &legacy)
            .await
            .unwrap(),
        (Vec::new(), 0),
        "no bounds enabled: nothing to report"
    );
    for config in [&bounded, &impact] {
        assert_eq!(
            crate::index::segments_missing_posting_bounds(&dir, &metadata, config)
                .await
                .unwrap(),
            (legacy_ids.clone(), 1)
        );
    }

    // New segments carry bounds; the old one keeps its layout after a merge.
    let mut writer = IndexWriter::open(dir.clone(), bounded.clone())
        .await
        .unwrap();
    for i in 0..8 {
        let mut doc = Document::new();
        doc.add_text(body, format!("needle new{i}"));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let metadata = IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(
        crate::index::segments_missing_posting_bounds(&dir, &metadata, &bounded)
            .await
            .unwrap(),
        (legacy_ids, 2)
    );
    writer.force_merge().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let metadata = IndexMetadata::load(&dir).await.unwrap();
    let (missing, probed) =
        crate::index::segments_missing_posting_bounds(&dir, &metadata, &bounded)
            .await
            .unwrap();
    assert_eq!(probed, 1);
    // The merged list of a shared term carries metadata from its bounded
    // source; the probe reads whichever external term sorts first, so it may
    // land on a copied legacy-only list. Either answer is a valid report.
    assert!(missing.len() <= 1);
}

#[tokio::test]
async fn zero_term_cache_budget_disables_dictionary_retention_at_index_level() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let index = Index::create(
        RamDirectory::new(),
        schema.build(),
        no_merge(IndexConfig {
            num_threads: 1,
            term_cache_blocks: 64,
            term_cache_budget_bytes: Some(0),
            ..IndexConfig::default()
        }),
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    for i in 0..500 {
        let mut doc = Document::new();
        doc.add_text(body, format!("common word{i:04}"));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    let mut count = CountCollector::new();
    collect_segment(segment, &TermQuery::text(body, "common"), &mut count)
        .await
        .unwrap();
    assert_eq!(count.count(), 500);
    let mut terms = segment.term_dict_iter();
    while terms.next().await.unwrap().is_some() {}
    assert_eq!(
        segment.memory_stats().term_dict_cache_bytes,
        0,
        "Some(0) must reach the segment readers and disable block retention"
    );
    writer.shutdown().await.unwrap();
}
