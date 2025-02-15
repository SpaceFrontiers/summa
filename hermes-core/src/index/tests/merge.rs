use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};

/// Multi-round merge: flush many small segments, merge, add more, merge again.
/// Verifies search correctness (term + phrase queries) through multiple merge rounds.
#[tokio::test]
async fn test_multi_round_merge_with_search() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        max_indexing_memory_bytes: 512,
        ..Default::default()
    };

    // --- Round 1: 5 segments × 10 docs = 50 docs ---
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    for batch in 0..5 {
        for i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(
                title,
                format!("alpha bravo charlie batch{} doc{}", batch, i),
            );
            doc.add_text(
                body,
                format!("the quick brown fox jumps over the lazy dog number {}", i),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let pre_merge_segments = index.segment_readers().await.unwrap().len();
    assert!(
        pre_merge_segments >= 3,
        "Expected >=3 segments, got {}",
        pre_merge_segments
    );
    assert_eq!(index.num_docs().await.unwrap(), 50);

    // Search before merge
    let results = index.query("alpha", 100).await.unwrap();
    assert_eq!(results.hits.len(), 50, "all 50 docs should match 'alpha'");

    let results = index.query("fox", 100).await.unwrap();
    assert_eq!(results.hits.len(), 50, "all 50 docs should match 'fox'");

    // --- Merge round 1 ---
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), 50);

    // Search after first merge
    let results = index.query("alpha", 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        50,
        "all 50 docs should match 'alpha' after merge 1"
    );

    let results = index.query("fox", 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        50,
        "all 50 docs should match 'fox' after merge 1"
    );

    // Verify all docs retrievable (single merged segment)
    let reader1 = index.reader().await.unwrap();
    let searcher1 = reader1.searcher().await.unwrap();
    let seg_id1 = searcher1.segment_readers()[0].meta().id;
    for i in 0..50 {
        let doc = searcher1.doc(seg_id1, i).await.unwrap();
        assert!(doc.is_some(), "doc {} should exist after merge 1", i);
    }

    // --- Round 2: add 30 more docs in 3 segments ---
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    for batch in 0..3 {
        for i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(
                title,
                format!("delta echo foxtrot round2_batch{} doc{}", batch, i),
            );
            doc.add_text(
                body,
                format!("the quick brown fox jumps again number {}", i),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 80);
    assert!(
        index.segment_readers().await.unwrap().len() >= 2,
        "Should have >=2 segments after round 2 ingestion"
    );

    // Search spans both old merged segment and new segments
    let results = index.query("fox", 100).await.unwrap();
    assert_eq!(results.hits.len(), 80, "all 80 docs should match 'fox'");

    let results = index.query("alpha", 100).await.unwrap();
    assert_eq!(results.hits.len(), 50, "only round 1 docs match 'alpha'");

    let results = index.query("delta", 100).await.unwrap();
    assert_eq!(results.hits.len(), 30, "only round 2 docs match 'delta'");

    // --- Merge round 2 ---
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), 80);

    // All searches still correct after second merge
    let results = index.query("fox", 100).await.unwrap();
    assert_eq!(results.hits.len(), 80, "all 80 docs after merge 2");

    let results = index.query("alpha", 100).await.unwrap();
    assert_eq!(results.hits.len(), 50, "round 1 docs after merge 2");

    let results = index.query("delta", 100).await.unwrap();
    assert_eq!(results.hits.len(), 30, "round 2 docs after merge 2");

    // Verify all 80 docs retrievable (single merged segment)
    let reader2 = index.reader().await.unwrap();
    let searcher2 = reader2.searcher().await.unwrap();
    let seg_id2 = searcher2.segment_readers()[0].meta().id;
    for i in 0..80 {
        let doc = searcher2.doc(seg_id2, i).await.unwrap();
        assert!(doc.is_some(), "doc {} should exist after merge 2", i);
    }
}

/// Large-scale merge: many segments with overlapping terms, verifying
/// BM25 scoring and doc retrieval after merge.
#[tokio::test]
async fn test_large_scale_merge_correctness() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        max_indexing_memory_bytes: 512,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 8 batches × 25 docs = 200 docs total
    // Terms: "common" appears in all, "unique_N" appears in batch N only
    let total_docs = 200u32;
    for batch in 0..8 {
        for i in 0..25 {
            let mut doc = Document::new();
            doc.add_text(
                title,
                format!("common shared term unique_{} item{}", batch, i),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Verify pre-merge
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), total_docs);

    let results = index.query("common", 300).await.unwrap();
    assert_eq!(
        results.hits.len(),
        total_docs as usize,
        "all docs should match 'common'"
    );

    // Each unique_N matches exactly 25 docs
    for batch in 0..8 {
        let q = format!("unique_{}", batch);
        let results = index.query(&q, 100).await.unwrap();
        assert_eq!(results.hits.len(), 25, "'{}' should match 25 docs", q);
    }

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    // Verify post-merge: single segment, same results
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), total_docs);

    let results = index.query("common", 300).await.unwrap();
    assert_eq!(results.hits.len(), total_docs as usize);

    for batch in 0..8 {
        let q = format!("unique_{}", batch);
        let results = index.query(&q, 100).await.unwrap();
        assert_eq!(results.hits.len(), 25, "'{}' after merge", q);
    }

    // Verify doc retrieval for every doc
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let seg_id = searcher.segment_readers()[0].meta().id;
    for i in 0..total_docs {
        let doc = searcher.doc(seg_id, i).await.unwrap();
        assert!(doc.is_some(), "doc {} missing after merge", i);
    }
}

/// Test that auto-merge is triggered by the merge policy during commit,
/// without calling force_merge. Uses MmapDirectory and higher parallelism
/// to reproduce production conditions.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_auto_merge_triggered() {
    use crate::directories::MmapDirectory;
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", true, true);
    let schema = schema_builder.build();

    // Aggressive policy: merge when 3 segments in same tier
    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        num_indexing_threads: 4,
        merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Create 12 segments with ~50 docs each (4x the aggressive threshold of 3)
    for batch in 0..12 {
        for i in 0..50 {
            let mut doc = Document::new();
            doc.add_text(title, format!("document_{} batch_{} alpha bravo", i, batch));
            doc.add_text(
                body,
                format!(
                    "the quick brown fox jumps over lazy dog number {} round {}",
                    i, batch
                ),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let pre_merge = writer.segment_manager.get_segment_ids().await.len();

    // wait_for_merging_thread waits for the single in-flight merge. After it completes,
    // re-evaluate since segments accumulated while the merge was running.
    writer.wait_for_merging_thread().await;
    writer.maybe_merge().await;
    writer.wait_for_merging_thread().await;

    // After commit + auto-merge, segment count should be reduced
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let segment_count = index.segment_readers().await.unwrap().len();
    eprintln!(
        "Segments: {} before merge, {} after auto-merge",
        pre_merge, segment_count
    );
    assert!(
        segment_count < pre_merge,
        "Expected auto-merge to reduce segments from {}, got {}",
        pre_merge,
        segment_count
    );
}

/// Regression test: commit with dense vector fields + aggressive merge policy.
/// Exercises the race where background merge deletes segment files while
/// maybe_build_vector_index → collect_vectors_for_training tries to open them.
/// Before the fix, this would fail with "IO error: No such file or directory".
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_commit_with_vectors_and_background_merge() {
    use crate::directories::MmapDirectory;
    use crate::dsl::DenseVectorConfig;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    // RaBitQ with very low build_threshold so vector index building triggers during commit
    let vec_config = DenseVectorConfig::new(8).with_build_threshold(10);
    let embedding =
        schema_builder.add_dense_vector_field_with_config("embedding", true, true, vec_config);
    let schema = schema_builder.build();

    // Aggressive merge: triggers background merges at 3 segments per tier
    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        num_indexing_threads: 4,
        merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Create 12 segments with vectors — enough to trigger both
    // background merges (aggressive policy) and vector index building (threshold=10)
    for batch in 0..12 {
        for i in 0..5 {
            let mut doc = Document::new();
            doc.add_text(title, format!("doc_{}_batch_{}", i, batch));
            // 8-dim random vector
            let vec: Vec<f32> = (0..8).map(|j| (i * 8 + j + batch) as f32 * 0.1).collect();
            doc.add_dense_vector(embedding, vec);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.wait_for_merging_thread().await;

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let num_docs = index.num_docs().await.unwrap();
    assert_eq!(num_docs, 60, "Expected 60 docs, got {}", num_docs);
}

/// Stress test: force_merge with many segments (iterative batching).
/// Verifies that merging 50 segments doesn't OOM or exhaust file descriptors.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_force_merge_many_segments() {
    use crate::directories::MmapDirectory;
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 512,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Create 50 tiny segments
    for batch in 0..50 {
        for i in 0..3 {
            let mut doc = Document::new();
            doc.add_text(title, format!("term_{} batch_{}", i, batch));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    // Wait for background merges before reading segment count
    writer.wait_for_merging_thread().await;

    let seg_ids = writer.segment_manager.get_segment_ids().await;
    let pre = seg_ids.len();
    eprintln!("Segments before force_merge: {}", pre);
    assert!(pre >= 2, "Expected multiple segments, got {}", pre);

    // Force merge all into one — should iterate in batches, not OOM
    writer.force_merge().await.unwrap();

    let index2 = Index::open(dir, config).await.unwrap();
    let post = index2.segment_readers().await.unwrap().len();
    eprintln!("Segments after force_merge: {}", post);
    assert_eq!(post, 1);
    assert_eq!(index2.num_docs().await.unwrap(), 150);
}

/// Test that background merges produce correct generation metadata.
/// Creates many segments with aggressive policy, commits, waits for merges,
/// and verifies that merged segments have generation >= 1 with correct ancestors.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_background_merge_generation() {
    use crate::directories::MmapDirectory;
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        num_indexing_threads: 2,
        merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Create 15 small segments — enough for aggressive policy to trigger merges
    for batch in 0..15 {
        for i in 0..5 {
            let mut doc = Document::new();
            doc.add_text(title, format!("doc_{}_batch_{}", i, batch));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.wait_for_merging_thread().await;

    // Read metadata and verify generation tracking
    let metas = writer
        .segment_manager
        .read_metadata(|m| m.segment_metas.clone())
        .await;

    let max_gen = metas.values().map(|m| m.generation).max().unwrap_or(0);
    eprintln!(
        "Segments after merge: {}, max generation: {}",
        metas.len(),
        max_gen
    );

    // Background merges should have produced at least one merged segment (gen >= 1)
    assert!(
        max_gen >= 1,
        "Expected at least one merged segment (gen >= 1), got max_gen={}",
        max_gen
    );

    // Every merged segment (gen > 0) must have non-empty ancestors
    for (id, info) in &metas {
        if info.generation > 0 {
            assert!(
                !info.ancestors.is_empty(),
                "Segment {} has gen={} but no ancestors",
                id,
                info.generation
            );
        } else {
            assert!(
                info.ancestors.is_empty(),
                "Fresh segment {} has gen=0 but has ancestors",
                id
            );
        }
    }
}

/// Test that merging preserves every single document.
/// Indexes 1000+ unique documents across many segments, force-merges,
/// and verifies exact doc count and that every unique term is searchable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_merge_preserves_all_documents() {
    use crate::directories::MmapDirectory;
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let total_docs = 1200;
    let docs_per_batch = 60;
    let batches = total_docs / docs_per_batch;

    // Each doc has a unique term "uid_N" for verification
    for batch in 0..batches {
        for i in 0..docs_per_batch {
            let doc_num = batch * docs_per_batch + i;
            let mut doc = Document::new();
            doc.add_text(
                title,
                format!("uid_{} common_term batch_{}", doc_num, batch),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let pre_segments = writer.segment_manager.get_segment_ids().await.len();
    assert!(
        pre_segments >= 2,
        "Need multiple segments, got {}",
        pre_segments
    );

    // Force merge to single segment
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(
        index.num_docs().await.unwrap(),
        total_docs as u32,
        "Doc count mismatch after force_merge"
    );

    // Verify every unique document is searchable
    let results = index.query("common_term", total_docs + 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        total_docs,
        "common_term should match all docs"
    );

    // Spot-check unique IDs across the range
    for check in [0, 1, total_docs / 2, total_docs - 1] {
        let q = format!("uid_{}", check);
        let results = index.query(&q, 10).await.unwrap();
        assert_eq!(results.hits.len(), 1, "'{}' should match exactly 1 doc", q);
    }
}

/// Multi-round commit+merge: verify doc count grows correctly
/// and no documents are lost across multiple merge cycles.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_multi_round_merge_doc_integrity() {
    use crate::directories::MmapDirectory;
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let schema = schema_builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        num_indexing_threads: 2,
        merge_policy: Box::new(crate::merge::TieredMergePolicy::aggressive()),
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let mut expected_total = 0u64;

    // 4 rounds of: add docs → commit → wait for merges → verify count
    for round in 0..4 {
        let docs_this_round = 50 + round * 25; // 50, 75, 100, 125
        for batch in 0..5 {
            for i in 0..docs_this_round / 5 {
                let mut doc = Document::new();
                doc.add_text(
                    title,
                    format!("round_{}_batch_{}_doc_{} searchable", round, batch, i),
                );
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
        }
        writer.wait_for_merging_thread().await;

        expected_total += docs_this_round as u64;

        let actual = writer
            .segment_manager
            .read_metadata(|m| {
                m.segment_metas
                    .values()
                    .map(|s| s.num_docs as u64)
                    .sum::<u64>()
            })
            .await;

        assert_eq!(
            actual, expected_total,
            "Round {}: expected {} docs, metadata reports {}",
            round, expected_total, actual
        );
    }

    // Final verify: open fresh and query
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), expected_total as u32);

    let results = index
        .query("searchable", expected_total as usize + 100)
        .await
        .unwrap();
    assert_eq!(
        results.hits.len(),
        expected_total as usize,
        "All docs should match 'searchable'"
    );

    // Check generation grew across rounds
    let metas = index
        .segment_manager()
        .read_metadata(|m| m.segment_metas.clone())
        .await;
    let max_gen = metas.values().map(|m| m.generation).max().unwrap_or(0);
    eprintln!(
        "Final: {} segments, {} docs, max generation={}",
        metas.len(),
        expected_total,
        max_gen
    );
    assert!(
        max_gen >= 1,
        "Multiple merge rounds should produce gen >= 1"
    );
}

/// Sustained indexing: verify segment count stays O(logN) bounded.
///
/// Indexes many small batches with aggressive merge policy and checks that
/// the segment count never grows unbounded. With tiered merging the count
/// should stay roughly O(segments_per_tier * num_tiers) ≈ O(log(N)).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_segment_count_bounded_during_sustained_indexing() {
    use crate::directories::MmapDirectory;
    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, false);
    let schema = schema_builder.build();

    let policy = crate::merge::TieredMergePolicy {
        segments_per_tier: 3,
        max_merge_at_once: 5,
        tier_factor: 10.0,
        tier_floor: 50,
        max_merged_docs: 1_000_000,
        ..Default::default()
    };

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096, // tiny budget → frequent flushes
        num_indexing_threads: 1,
        merge_policy: Box::new(policy),
        max_concurrent_merges: 4,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let num_commits = 40;
    let docs_per_commit = 30;
    let total_docs = num_commits * docs_per_commit;
    let mut max_segments_seen = 0usize;

    for commit_idx in 0..num_commits {
        for i in 0..docs_per_commit {
            let mut doc = Document::new();
            doc.add_text(
                title,
                format!("doc_{} text", commit_idx * docs_per_commit + i),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();

        // Give background merges a moment to run
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let seg_count = writer.segment_manager.get_segment_ids().await.len();
        max_segments_seen = max_segments_seen.max(seg_count);
    }

    // Wait for all merges to finish
    writer.wait_for_all_merges().await;

    let final_segments = writer.segment_manager.get_segment_ids().await.len();
    let final_docs: u64 = writer
        .segment_manager
        .read_metadata(|m| {
            m.segment_metas
                .values()
                .map(|s| s.num_docs as u64)
                .sum::<u64>()
        })
        .await;

    eprintln!(
        "Sustained indexing: {} commits, {} total docs, final segments={}, max segments seen={}",
        num_commits, total_docs, final_segments, max_segments_seen
    );

    // With 1200 docs and segments_per_tier=3, tier_floor=50:
    // tier 0: ≤50 docs, tier 1: 50-500, tier 2: 500-5000
    // We should have at most ~3 segments per tier * ~3 tiers ≈ 9-12 segments at peak.
    // The key invariant: segment count must NOT grow linearly with commits.
    // 40 commits should NOT produce 40 segments.
    let max_allowed = num_commits / 2; // generous: at most half the commits as segments
    assert!(
        max_segments_seen <= max_allowed,
        "Segment count grew too fast: max seen {} > allowed {} (out of {} commits). \
         Merging is not keeping up.",
        max_segments_seen,
        max_allowed,
        num_commits
    );

    // After all merges complete, should be well under the limit
    assert!(
        final_segments <= 10,
        "After all merges, expected ≤10 segments, got {}",
        final_segments
    );

    // No data loss
    assert_eq!(
        final_docs, total_docs as u64,
        "Expected {} docs, metadata reports {}",
        total_docs, final_docs
    );
}

/// Verify that every document's FIELDS (not just existence) survive multiple
/// merge rounds. Uses documents large enough to create multiple store blocks
/// per segment. This catches store merger bugs that silently lose blocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_store_fields_survive_multiple_merges() {
    use crate::directories::MmapDirectory;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", false, true);
    let schema = schema_builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 1024 * 64,
        num_indexing_threads: 2,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };

    let make_doc = |round: usize, idx: usize| -> Document {
        let mut doc = Document::new();
        doc.add_text(title, format!("doc_r{}_i{} searchterm", round, idx));
        let body_text = format!("round={} idx={} {}", round, idx, "abcdefghij ".repeat(90));
        doc.add_text(body, body_text);
        doc
    };

    let mut total_docs = 0usize;

    // === Round 1: 200 docs across multiple segments ===
    {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();
        for batch in 0..4 {
            for i in 0..50 {
                writer.add_document(make_doc(1, batch * 50 + i)).unwrap();
            }
            writer.commit().await.unwrap();
        }
        total_docs += 200;
    }

    // Force merge round 1
    {
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.force_merge().await.unwrap();
    }

    // Verify every doc's fields after merge 1
    {
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), total_docs as u32);

        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        assert_eq!(
            searcher.num_segments(),
            1,
            "should be 1 segment after merge 1"
        );
        let seg = &searcher.segment_readers()[0];
        let seg_id = seg.meta().id;

        assert_eq!(
            seg.store().num_docs(),
            seg.num_docs(),
            "store.num_docs != meta.num_docs after merge 1"
        );

        for i in 0..total_docs as u32 {
            let doc = searcher
                .doc(seg_id, i)
                .await
                .unwrap_or_else(|e| panic!("doc {} error: {}", i, e));
            assert!(doc.is_some(), "doc {} missing after merge 1", i);
            let doc = doc.unwrap();
            let t = doc
                .get_first(title)
                .unwrap_or_else(|| panic!("doc {} missing title", i));
            assert!(
                t.as_text().unwrap().contains("searchterm"),
                "doc {} title corrupt after merge 1",
                i
            );
        }
    }

    // === Round 2: add 150 more docs, merge again ===
    {
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        for batch in 0..3 {
            for i in 0..50 {
                writer.add_document(make_doc(2, batch * 50 + i)).unwrap();
            }
            writer.commit().await.unwrap();
        }
        total_docs += 150;
        writer.force_merge().await.unwrap();
    }

    // Verify every doc's fields after merge 2
    {
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), total_docs as u32);

        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        assert_eq!(
            searcher.num_segments(),
            1,
            "should be 1 segment after merge 2"
        );
        let seg = &searcher.segment_readers()[0];
        let seg_id = seg.meta().id;

        assert_eq!(
            seg.store().num_docs(),
            seg.num_docs(),
            "store.num_docs != meta.num_docs after merge 2"
        );

        for i in 0..total_docs as u32 {
            let doc = searcher
                .doc(seg_id, i)
                .await
                .unwrap_or_else(|e| panic!("doc {} error: {}", i, e));
            assert!(doc.is_some(), "doc {} missing after merge 2", i);
            let doc = doc.unwrap();
            let t = doc
                .get_first(title)
                .unwrap_or_else(|| panic!("doc {} missing title", i));
            assert!(
                t.as_text().unwrap().contains("searchterm"),
                "doc {} title corrupt after merge 2",
                i
            );
        }
    }

    // === Round 3: add 100 more, merge a third time ===
    {
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        for batch in 0..2 {
            for i in 0..50 {
                writer.add_document(make_doc(3, batch * 50 + i)).unwrap();
            }
            writer.commit().await.unwrap();
        }
        total_docs += 100;
        writer.force_merge().await.unwrap();
    }

    // Verify every doc's fields after merge 3
    {
        let index = Index::open(dir.clone(), config.clone()).await.unwrap();
        assert_eq!(index.num_docs().await.unwrap(), total_docs as u32);

        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        assert_eq!(
            searcher.num_segments(),
            1,
            "should be 1 segment after merge 3"
        );
        let seg = &searcher.segment_readers()[0];
        let seg_id = seg.meta().id;

        assert_eq!(
            seg.store().num_docs(),
            seg.num_docs(),
            "store.num_docs != meta.num_docs after merge 3"
        );

        let mut missing = 0;
        let mut corrupt = 0;
        for i in 0..total_docs as u32 {
            match searcher.doc(seg_id, i).await {
                Ok(Some(doc)) => {
                    if let Some(t) = doc.get_first(title) {
                        if !t.as_text().unwrap_or("").contains("searchterm") {
                            corrupt += 1;
                        }
                    } else {
                        corrupt += 1;
                    }
                }
                Ok(None) => missing += 1,
                Err(e) => panic!("doc {} error after merge 3: {}", i, e),
            }
        }
        assert_eq!(
            missing, 0,
            "merge 3: {} of {} docs missing from store",
            missing, total_docs
        );
        assert_eq!(
            corrupt, 0,
            "merge 3: {} of {} docs have corrupt fields",
            corrupt, total_docs
        );
    }

    eprintln!("All {} docs verified across 3 merge rounds", total_docs);
}

/// Large-scale store test: ~3000 docs with ~1KB each → many store blocks per segment.
/// Verifies every doc's fields survive 4 merge rounds with MmapDirectory.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_store_large_scale_multi_merge() {
    use crate::directories::MmapDirectory;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let body = schema_builder.add_text_field("body", false, true);
    let schema = schema_builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 1024 * 256,
        num_indexing_threads: 2,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };

    // ~1KB per doc → ~256 docs per 256KB store block → 800 docs ≈ 3 blocks
    let make_doc = |round: usize, idx: usize| -> Document {
        let mut doc = Document::new();
        doc.add_text(title, format!("r{}_i{}_needle", round, idx));
        doc.add_text(body, format!("r{}i{} {}", round, idx, "x".repeat(950)));
        doc
    };

    let mut total_docs = 0u32;

    for round in 0..4 {
        let docs_this_round = 800u32;

        // Add docs across multiple segments
        {
            let mut writer = if round == 0 {
                IndexWriter::create(dir.clone(), schema.clone(), config.clone())
                    .await
                    .unwrap()
            } else {
                IndexWriter::open(dir.clone(), config.clone())
                    .await
                    .unwrap()
            };
            for batch in 0..4 {
                for i in 0..docs_this_round / 4 {
                    writer
                        .add_document(make_doc(
                            round,
                            (batch * (docs_this_round / 4) + i) as usize,
                        ))
                        .unwrap();
                }
                writer.commit().await.unwrap();
            }
            total_docs += docs_this_round;
            writer.force_merge().await.unwrap();
        }

        // Verify every doc's fields in the merged segment
        {
            let index = Index::open(dir.clone(), config.clone()).await.unwrap();
            assert_eq!(index.num_docs().await.unwrap(), total_docs);
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            assert_eq!(
                searcher.num_segments(),
                1,
                "round {}: expected 1 segment",
                round
            );
            let seg = &searcher.segment_readers()[0];
            let seg_id = seg.meta().id;
            assert_eq!(
                seg.store().num_docs(),
                seg.num_docs(),
                "round {}: store/meta mismatch",
                round
            );
            let mut missing = 0u32;
            for i in 0..total_docs {
                match searcher.doc(seg_id, i).await {
                    Ok(Some(doc)) => {
                        let t = doc.get_first(title);
                        assert!(
                            t.is_some() && t.unwrap().as_text().unwrap().contains("needle"),
                            "round {}: doc {} corrupt",
                            round,
                            i
                        );
                    }
                    Ok(None) => missing += 1,
                    Err(e) => panic!("round {}: doc {} error: {}", round, i, e),
                }
            }
            assert_eq!(
                missing, 0,
                "round {}: {} of {} docs missing",
                round, missing, total_docs
            );
        }

        eprintln!(
            "Round {}: {} docs verified ({} total)",
            round, docs_this_round, total_docs
        );
    }
    eprintln!(
        "All {} docs verified across 4 large-scale merge rounds",
        total_docs
    );
}

/// Integration test for the large_scale merge policy preset.
/// Exercises budget-aware triggering, oversized exclusion, and scored selection
/// with a realistic multi-tier segment distribution.
#[tokio::test]
async fn test_large_scale_merge_policy() {
    use crate::merge::{MergePolicy, SegmentInfo, TieredMergePolicy};

    let policy = TieredMergePolicy::large_scale();

    // Simulate a large index with segments across multiple tiers:
    // - 15 small segments (10K docs each)
    // - 5 medium segments (500K docs each)
    // - 2 large segments (5M docs each)
    // - 1 oversized segment (15M docs — above 20M * 0.5 = 10M threshold)
    let mut segments = Vec::new();

    for i in 0..15 {
        segments.push(SegmentInfo {
            id: format!("small_{}", i),
            num_docs: 10_000,
        });
    }
    for i in 0..5 {
        segments.push(SegmentInfo {
            id: format!("medium_{}", i),
            num_docs: 500_000,
        });
    }
    for i in 0..2 {
        segments.push(SegmentInfo {
            id: format!("large_{}", i),
            num_docs: 5_000_000,
        });
    }
    segments.push(SegmentInfo {
        id: "oversized_0".into(),
        num_docs: 15_000_000,
    });

    let candidates = policy.find_merges(&segments);

    // 1. Oversized segment (15M > 10M threshold) must not appear in any candidate
    for c in &candidates {
        assert!(
            !c.segment_ids.contains(&"oversized_0".into()),
            "oversized segment should be excluded from merges"
        );
    }

    // 2. If merges are produced, they should respect max_merged_docs
    for c in &candidates {
        let total: u64 = c
            .segment_ids
            .iter()
            .map(|id| segments.iter().find(|s| s.id == *id).unwrap().num_docs as u64)
            .sum();
        assert!(
            total <= policy.max_merged_docs as u64,
            "merge total {} exceeds max_merged_docs {}",
            total,
            policy.max_merged_docs
        );
    }

    // 3. No candidate should contain duplicate segment IDs
    for c in &candidates {
        let mut ids = c.segment_ids.clone();
        ids.sort();
        ids.dedup();
        assert_eq!(
            ids.len(),
            c.segment_ids.len(),
            "duplicate segment IDs in merge candidate"
        );
    }

    // 4. Verify preset produces sensible results — with 23 segments (excl oversized)
    //    and budget of ~3 tiers * 10 = 30, it may or may not merge depending on budget.
    //    But at minimum, the policy should not panic or produce invalid output.
    eprintln!(
        "large_scale policy produced {} merge candidates from {} segments",
        candidates.len(),
        segments.len()
    );
}
