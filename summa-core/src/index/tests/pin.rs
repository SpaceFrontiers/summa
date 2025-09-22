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

async fn build_test_index(
    dir: &MmapDirectory,
    format: crate::structures::SparseFormat,
) -> (crate::dsl::Schema, u128) {
    let mut sb = SchemaBuilder::default();
    let sparse_cfg = crate::structures::SparseVectorConfig {
        format,
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
    for format in [
        crate::structures::SparseFormat::Bmp,
        crate::structures::SparseFormat::Seismic,
    ] {
        use crate::query::SparseVectorQuery;

        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp.path());
        let (schema, seg_id) = build_test_index(&dir, format).await;

        let mut reader = SegmentReader::open(&dir, SegmentId(seg_id), Arc::new(schema.clone()), 64)
            .await
            .unwrap();

        let field = schema.get_field("sparse").unwrap();
        let query = SparseVectorQuery::new(field, vec![(3, 1.0), (10, 1.0)]);
        let before = crate::query::search_segment_with_count(&reader, &query, 10)
            .await
            .unwrap();

        // Generous budget: everything pinnable gets pinned
        reader.apply_pin_policy(&PinPolicy {
            budget_bytes: 64 * 1024 * 1024,
            mode: PinMode::Copy,
        });
        let stats = reader.memory_stats();
        assert!(stats.sparse_file_backed_bytes > 0);
        assert!(stats.dense_file_backed_bytes > 0);
        assert!(
            stats.sparse_file_backed_bytes > stats.sparse_heap_bytes as u64,
            "Sparse corpus bytes must be reported as file-backed, not heap"
        );
        assert!(
            stats.pinned_metadata_bytes > 0,
            "Sparse directories and flat doc_ids should be pinnable"
        );
        assert!(stats.sparse_pinned_metadata_bytes > 0);
        assert!(stats.dense_pinned_metadata_bytes > 0);
        assert_eq!(
            stats.pinned_metadata_bytes,
            stats
                .sparse_pinned_metadata_bytes
                .saturating_add(stats.dense_pinned_metadata_bytes)
        );
        assert_eq!(
            stats.pin_intended_bytes,
            stats
                .sparse_pin_intended_bytes
                .saturating_add(stats.dense_pin_intended_bytes)
        );
        assert_eq!(
            stats.pinned_metadata_bytes, stats.pin_intended_bytes,
            "generous budget must pin everything eligible"
        );

        // Search still works on the heap-copied metadata
        let after = crate::query::search_segment_with_count(&reader, &query, 10)
            .await
            .unwrap();
        assert!(!before.0.is_empty(), "fixture must have competitive hits");
        assert_eq!(before.1, after.1, "pinning must preserve scored count");
        let hits = |results: Vec<crate::query::SearchResult>| {
            results
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            hits(before.0),
            hits(after.0),
            "pinning must preserve exact ranking and scores"
        );
    }
}

/// Budget exhaustion is respected and reported (fail-loud accounting).
#[tokio::test]
async fn test_pin_metadata_budget_exhaustion_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp.path());
    let (schema, seg_id) = build_test_index(&dir, crate::structures::SparseFormat::Bmp).await;

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
    assert_eq!(
        stats.pinned_metadata_bytes,
        stats
            .sparse_pinned_metadata_bytes
            .saturating_add(stats.dense_pinned_metadata_bytes)
    );
    assert!(
        stats.pin_intended_bytes > stats.pinned_metadata_bytes,
        "tiny budget must leave a visible intended-vs-pinned gap"
    );
}

#[tokio::test]
async fn disabled_pinning_reports_eligible_sparse_metadata_without_allocating() {
    for format in [
        crate::structures::SparseFormat::Bmp,
        crate::structures::SparseFormat::Seismic,
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = MmapDirectory::new(tmp.path());
        let (schema, seg_id) = build_test_index(&dir, format).await;
        let mut reader = SegmentReader::open(&dir, SegmentId(seg_id), Arc::new(schema), 64)
            .await
            .unwrap();
        let before_heap = reader.memory_stats().sparse_heap_bytes;
        reader.apply_pin_policy(&PinPolicy::disabled());
        let stats = reader.memory_stats();
        assert!(stats.sparse_pin_intended_bytes > 0);
        assert!(stats.dense_pin_intended_bytes > 0);
        assert_eq!(stats.pinned_metadata_bytes, 0);
        assert_eq!(stats.sparse_heap_bytes, before_heap);
    }
}

#[tokio::test]
async fn deletion_views_share_copy_pinned_metadata_until_the_last_reader_drops() {
    let temp = tempfile::tempdir().unwrap();
    let directory = MmapDirectory::new(temp.path());
    let (schema, segment_id) =
        build_test_index(&directory, crate::structures::SparseFormat::Bmp).await;
    let sparse = schema.get_field("sparse").unwrap();
    let mut original = SegmentReader::open(&directory, SegmentId(segment_id), Arc::new(schema), 8)
        .await
        .unwrap();
    original.apply_pin_policy(&PinPolicy {
        budget_bytes: 64 * 1024 * 1024,
        mode: PinMode::Copy,
    });
    assert!(original.memory_stats().pinned_metadata_bytes > 0);
    let mut alive = crate::query::DocBitset::all(original.num_docs());
    alive.clear(0);
    let deletion =
        crate::segment::deletion::write(&directory, SegmentId::new(), original.num_docs(), &alive)
            .await
            .unwrap();
    let changed = original
        .with_deletions(&directory, Some(deletion))
        .await
        .unwrap();
    assert_eq!(
        original
            .bmp_index(sparse)
            .unwrap()
            .doc_map_ids_slice()
            .as_ptr(),
        changed
            .bmp_index(sparse)
            .unwrap()
            .doc_map_ids_slice()
            .as_ptr()
    );
    assert_eq!(
        original.memory_stats().pinned_metadata_bytes,
        changed.memory_stats().pinned_metadata_bytes
    );
    assert!(original.is_alive(0));
    assert!(!changed.is_alive(0));
    drop(original);
    assert_eq!(changed.num_live_docs(), 299);
    let restored = changed.with_deletions(&directory, None).await.unwrap();
    assert!(restored.is_alive(0));
    assert!(!changed.is_alive(0));
    drop(changed);
    assert_eq!(
        crate::query::search_segment_with_count(&restored, &crate::query::AllQuery, 500)
            .await
            .unwrap()
            .0
            .len(),
        300
    );
}
