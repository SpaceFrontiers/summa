//! Hot-metadata pinning tests (segment::pin).
//!
//! Uses MmapDirectory so the metadata sections are genuinely mmap-backed
//! (pinning is a no-op for heap-backed RAM directories).

use std::sync::Arc;

use crate::directories::MmapDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::segment::pin::{PinMode, PinPolicy};
use crate::segment::{SegmentId, SegmentReader};

async fn build_test_index(dir: &MmapDirectory) -> (crate::dsl::Schema, u128) {
    let mut sb = SchemaBuilder::default();
    let sparse_cfg = crate::structures::SparseVectorConfig {
        format: crate::structures::SparseFormat::Bmp,
        dims: Some(1024),
        max_weight: Some(5.0),
        ..Default::default()
    };
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, sparse_cfg);
    let dense = sb.add_dense_vector_field("dense", 8, true, true);
    let schema = sb.build();

    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    for i in 0..300u32 {
        let mut doc = Document::new();
        doc.add_sparse_vector(
            sparse,
            vec![(i % 512, 1.0 + (i % 7) as f32), ((i + 7) % 512, 0.5)],
        );
        doc.add_dense_vector(dense, (0..8).map(|d| (i + d) as f32 / 300.0).collect());
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir.clone(), config).await.unwrap();
    let segments = index.segment_readers().await.unwrap();
    let seg_id = segments[0].meta().id;
    (schema, seg_id)
}

/// Copy-mode pinning: metadata sections move to the heap, accounting is
/// reported, and search results are unchanged afterwards.
#[tokio::test]
async fn test_pin_metadata_copy_mode_preserves_search() {
    use crate::query::SparseVectorQuery;

    let tmp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp.path());
    let (schema, seg_id) = build_test_index(&dir).await;

    let mut reader = SegmentReader::open(&dir, SegmentId(seg_id), Arc::new(schema.clone()), 64)
        .await
        .unwrap();

    // Generous budget: everything pinnable gets pinned
    reader.apply_pin_policy(&PinPolicy {
        budget_bytes: 64 * 1024 * 1024,
        mode: PinMode::Copy,
    });
    let stats = reader.memory_stats();
    assert!(
        stats.pinned_metadata_bytes > 0,
        "BMP starts/doc-maps/sb_grid + flat doc_ids should be pinnable"
    );
    assert_eq!(
        stats.pinned_metadata_bytes, stats.pin_intended_bytes,
        "generous budget must pin everything eligible"
    );

    // Search still works on the heap-copied metadata
    let field = schema.get_field("sparse").unwrap();
    let query = SparseVectorQuery::new(field, vec![(3, 1.0), (10, 1.0)]);
    let results = crate::query::search_segment_with_count(&reader, &query, 10)
        .await
        .unwrap()
        .0;
    assert!(!results.is_empty(), "search must survive pinning");
}

/// Budget exhaustion is respected and reported (fail-loud accounting).
#[tokio::test]
async fn test_pin_metadata_budget_exhaustion_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp.path());
    let (schema, seg_id) = build_test_index(&dir).await;

    let mut reader = SegmentReader::open(&dir, SegmentId(seg_id), Arc::new(schema), 64)
        .await
        .unwrap();

    // Tiny budget: something must be skipped
    let budget = 128u64;
    reader.apply_pin_policy(&PinPolicy {
        budget_bytes: budget,
        mode: PinMode::Copy,
    });
    let stats = reader.memory_stats();
    assert!(stats.pinned_metadata_bytes <= budget);
    assert!(
        stats.pin_intended_bytes > stats.pinned_metadata_bytes,
        "tiny budget must leave a visible intended-vs-pinned gap"
    );
}
