//! Ratio/impact bound metadata across encoded-copy merges.
//!
//! A merged list carries bound metadata when any source does. Promoted inline
//! sources must be encoded with the same metadata, otherwise their zero
//! record drags the minimum of their whole L1 group down and disables
//! bound-based skipping for every block in that group. Legacy blocks cannot
//! gain metadata without a rebuild, so mixing them in must stay observable.

use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexMetadata, IndexWriter};
use crate::segment::{SegmentId, SegmentMerger, SegmentReader};
use std::sync::Arc;

fn bounded_config(impact: bool) -> IndexConfig {
    IndexConfig {
        num_indexing_threads: 1,
        posting_ratio_bounds: !impact,
        posting_impact_bounds: impact,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    }
}

fn needle_docs(field: crate::Field, count: usize, suffix: &str) -> Vec<Document> {
    (0..count)
        .map(|i| {
            let mut doc = Document::new();
            doc.add_text(field, format!("needle {suffix}{i} filler filler"));
            doc
        })
        .collect()
}

async fn open_segments(dir: &RamDirectory, schema: Arc<crate::Schema>) -> Vec<SegmentReader> {
    let metadata = IndexMetadata::load(dir).await.unwrap();
    let mut segments = Vec::new();
    for id in metadata.segment_ids() {
        let id = SegmentId::from_hex(&id).unwrap();
        segments.push(
            SegmentReader::open(dir, id, Arc::clone(&schema), 256)
                .await
                .unwrap(),
        );
    }
    segments
}

async fn inline_promotion_keeps_group_bounds(impact: bool) {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let schema = schema.build();
    let dir = RamDirectory::new();
    let config = bounded_config(impact);
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    // Segment 1: a single posting stays inline in the term dictionary.
    let mut first = Document::new();
    first.add_text(body, "needle");
    writer.add_document(first).unwrap();
    writer.commit().await.unwrap();
    // Segment 2: external bounded blocks for the same term. Impact
    // envelopes exist only on multi-block lists, so this spans three blocks.
    let external_docs = if impact { 300 } else { 8 };
    for doc in needle_docs(body, external_docs, "value") {
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();
    drop(writer);

    let index = Index::open(dir.clone(), config).await.unwrap();
    let segments = index.segment_readers().await.unwrap();
    assert_eq!(segments.len(), 1);
    let postings = segments[0]
        .get_postings(body, b"needle")
        .await
        .unwrap()
        .unwrap();
    let blocks = postings.num_blocks();
    assert_eq!(
        blocks,
        1 + external_docs.div_ceil(crate::structures::postings::POSTING_BLOCK_SIZE),
        "inline promotion appends one tiny block to the copied external blocks"
    );
    assert!(postings.has_ratio_bounds());
    assert_eq!(postings.has_impact_bounds(), impact);
    for block in 0..blocks {
        assert!(
            postings.block_length_ratio(block) > 0.0,
            "block {block} lost its ratio bound"
        );
    }
    assert!(
        postings.group_length_ratio(0) > 0.0,
        "an unbounded promoted inline block zeroes the L1 group ratio"
    );
    if impact {
        // Impact envelopes are only built for multi-block lists, so the
        // promoted single-posting block keeps an unknown record and the group
        // envelope stays unknown; the ratio bounds above still prune it.
        // `MergeStats::posting_blocks_without_bounds` counts that block.
        assert_eq!(postings.block_impact_point_count(0), Some(0));
        assert!(postings.block_impact_point_count(1).unwrap_or(0) > 0);
        assert_eq!(postings.group_impact_point_count(0), Some(0));
    }
}

#[tokio::test]
async fn promoted_inline_source_keeps_ratio_group_bounds() {
    inline_promotion_keeps_group_bounds(false).await;
}

#[tokio::test]
async fn promoted_inline_source_keeps_impact_group_bounds() {
    inline_promotion_keeps_group_bounds(true).await;
}

#[tokio::test]
async fn legacy_blocks_merged_into_bounded_list_are_counted() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let schema = schema.build();
    let dir = RamDirectory::new();

    // Segment 1 predates bounds; segment 2 is written after they are enabled.
    let legacy = IndexConfig {
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), legacy)
        .await
        .unwrap();
    for doc in needle_docs(body, 8, "old") {
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);

    let bounded = bounded_config(false);
    let mut writer = IndexWriter::open(dir.clone(), bounded.clone())
        .await
        .unwrap();
    for doc in needle_docs(body, 8, "new") {
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);

    let schema = Arc::new(schema);
    let segments = open_segments(&dir, Arc::clone(&schema)).await;
    assert_eq!(segments.len(), 2);
    let mut source_bounds = Vec::new();
    for segment in &segments {
        let list = segment
            .get_postings(body, b"needle")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(list.num_blocks(), 1);
        source_bounds.push(list.has_ratio_bounds());
    }
    source_bounds.sort();
    assert_eq!(
        source_bounds,
        [false, true],
        "one legacy and one bounded source"
    );
    let merger = SegmentMerger::new(Arc::clone(&schema))
        .with_posting_config(bounded.optimization, bounded.effective_posting_codec());
    let merged_id = SegmentId::new();
    let (_, stats) = merger
        .merge(&dir, &segments, merged_id, None)
        .await
        .unwrap();
    // `needle` and `filler` are both shared terms with one legacy block each.
    assert_eq!(
        stats.posting_blocks_without_bounds, 2,
        "legacy blocks joining bounded lists without metadata must be counted"
    );

    let merged = SegmentReader::open(&dir, merged_id, schema, 256)
        .await
        .unwrap();
    let postings = merged.get_postings(body, b"needle").await.unwrap().unwrap();
    assert_eq!(postings.num_blocks(), 2);
    assert!(postings.has_ratio_bounds());
    let (legacy_block, bounded_block) = if postings.block_first_doc(0) == Some(0) {
        (0, 1)
    } else {
        (1, 0)
    };
    assert_eq!(postings.block_length_ratio(legacy_block), 0.0);
    assert!(postings.block_length_ratio(bounded_block) > 0.0);
    assert_eq!(
        postings.group_length_ratio(0),
        0.0,
        "legacy blocks keep their unknown bound; only a rebuild adds metadata"
    );
}

#[tokio::test]
async fn merging_only_legacy_segments_counts_nothing() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let schema = schema.build();
    let dir = RamDirectory::new();
    let legacy = IndexConfig {
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), legacy.clone())
        .await
        .unwrap();
    for batch in ["a", "b"] {
        for doc in needle_docs(body, 8, batch) {
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.shutdown().await.unwrap();
    drop(writer);

    let schema = Arc::new(schema);
    let segments = open_segments(&dir, Arc::clone(&schema)).await;
    let merger = SegmentMerger::new(Arc::clone(&schema))
        .with_posting_config(legacy.optimization, legacy.effective_posting_codec());
    let (_, stats) = merger
        .merge(&dir, &segments, SegmentId::new(), None)
        .await
        .unwrap();
    assert_eq!(stats.posting_blocks_without_bounds, 0);
}

#[tokio::test]
async fn compaction_rebuilt_bounds_remain_conservative_when_chunk_length_floor_falls() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let body = schema.add_text_field("body", true, false);
    schema.set_chunked(body, true);
    let index = Index::create(RamDirectory::new(), schema.build(), bounded_config(false))
        .await
        .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for i in 0..130 {
        let mut doc = Document::new();
        doc.add_text(id, i.to_string());
        doc.add_text(
            body,
            if i < 10 {
                "needle".to_owned()
            } else {
                format!("needle {}", "filler ".repeat(99))
            },
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let reader = index.reader().await.unwrap();
    let before = reader.searcher().await.unwrap();
    assert_eq!(
        before.segment_readers()[0]
            .chunk_map(body)
            .unwrap()
            .length_floor(),
        100
    );
    for i in 10..130 {
        writer.delete_primary_key(&i.to_string()).unwrap();
    }
    writer.commit().await.unwrap();
    writer.compact(32 * 1024 * 1024).await.unwrap();
    reader.reload().await.unwrap();
    let after = reader.searcher().await.unwrap();
    let segment = &after.segment_readers()[0];
    assert_eq!(segment.chunk_map(body).unwrap().length_floor(), 1);
    let postings = segment
        .get_postings(body, b"needle")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(postings.doc_count(), 10);
    assert!(
        postings.block_length_ratio(0) <= 1.0,
        "compaction must not encode the deleted source's length floor as a surviving block bound: {}",
        postings.block_length_ratio(0)
    );
    assert_eq!(postings.min_len(), Some(1));
}
