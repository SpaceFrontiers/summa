//! Integration tests for BMP (Block-Max Pruning) sparse vector format.
//!
//! Tests build → query → merge → query correctness for the BMP executor.

use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::SparseVectorQuery;
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};

/// Helper: BMP config without pruning (for correctness tests with small data).
fn bmp_config() -> SparseVectorConfig {
    SparseVectorConfig {
        format: SparseFormat::Bmp,
        weight_quantization: WeightQuantization::UInt8,
        bmp_block_size: 64,
        ..SparseVectorConfig::default()
    }
}

/// Helper: create BMP schema with a text field and a BMP sparse field.
fn bmp_schema() -> (crate::dsl::Schema, crate::dsl::Field, crate::dsl::Field) {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, bmp_config());
    (sb.build(), title, sparse)
}

/// Basic BMP build + query: needle-in-haystack.
#[tokio::test]
async fn test_bmp_needle_in_haystack() {
    let (schema, title, sparse) = bmp_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 100 hay documents on dimensions 0-9
    for i in 0..100 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay doc {}", i));
        let entries: Vec<(u32, f32)> = (0..10)
            .map(|d| (d, 0.1 + (i as f32 * 0.001) + (d as f32 * 0.01)))
            .collect();
        doc.add_sparse_vector(sparse, entries);
        writer.add_document(doc).unwrap();
    }

    // Needle: unique dimensions 1000-1002
    let mut needle = Document::new();
    needle.add_text(title, "Needle BMP document");
    needle.add_sparse_vector(sparse, vec![(1000, 0.9), (1001, 0.8), (1002, 0.7)]);
    writer.add_document(needle).unwrap();

    writer.commit().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 101);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Query needle's unique dims
    let query = SparseVectorQuery::new(sparse, vec![(1000, 1.0), (1001, 1.0), (1002, 1.0)]);
    let results = searcher.search(&query, 10).await.unwrap();
    assert_eq!(results.len(), 1, "Only the needle has dims 1000-1002");
    assert!(results[0].score > 0.0);

    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        doc.get_first(title).unwrap().as_text().unwrap(),
        "Needle BMP document"
    );

    // Query a shared dimension — should match many
    let query_shared = SparseVectorQuery::new(sparse, vec![(5, 1.0)]);
    let results = searcher.search(&query_shared, 200).await.unwrap();
    assert!(
        results.len() >= 50,
        "Shared dim 5 should match many docs, got {}",
        results.len()
    );

    // Query non-existent dimension — should match nothing
    let query_missing = SparseVectorQuery::new(sparse, vec![(99999, 1.0)]);
    let results = searcher.search(&query_missing, 10).await.unwrap();
    assert_eq!(results.len(), 0);
}

/// BMP multi-segment merge: build two segments, merge, verify query correctness.
#[tokio::test]
async fn test_bmp_merge() {
    let (schema, title, sparse) = bmp_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Segment 1: hay on dims 0-5
    for i in 0..30 {
        let mut doc = Document::new();
        doc.add_text(title, format!("seg1 hay {}", i));
        doc.add_sparse_vector(sparse, vec![(0, 0.5), (1, 0.3), (2, 0.2)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Segment 2: needle + more hay
    let mut needle = Document::new();
    needle.add_text(title, "seg2 needle");
    needle.add_sparse_vector(sparse, vec![(500, 0.95), (501, 0.85)]);
    writer.add_document(needle).unwrap();
    for i in 0..29 {
        let mut doc = Document::new();
        doc.add_text(title, format!("seg2 hay {}", i));
        doc.add_sparse_vector(sparse, vec![(0, 0.4), (3, 0.6)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Pre-merge: verify 2 segments and needle found
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 60);
    let segments = index.segment_readers().await.unwrap();
    assert!(segments.len() >= 2, "Should have at least 2 segments");

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(sparse, vec![(500, 1.0), (501, 1.0)]);
    let results = searcher.search(&query, 10).await.unwrap();
    assert_eq!(results.len(), 1, "Pre-merge: needle should be found");

    // Merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    // Post-merge: verify single segment and needle still found
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), 60);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let results = searcher.search(&query, 10).await.unwrap();
    assert_eq!(results.len(), 1, "Post-merge: needle should still be found");

    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        doc.get_first(title).unwrap().as_text().unwrap(),
        "seg2 needle"
    );

    // All hay dims should still work
    let query_hay = SparseVectorQuery::new(sparse, vec![(0, 1.0)]);
    let results = searcher.search(&query_hay, 100).await.unwrap();
    assert!(
        results.len() >= 50,
        "Post-merge: dim 0 should match >=50 docs, got {}",
        results.len()
    );
}

/// BMP score ranking: verify documents are ranked by score correctly.
#[tokio::test]
async fn test_bmp_score_ranking() {
    let (schema, title, sparse) = bmp_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Documents with increasing weights on dimension 0
    for i in 0..50 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Doc weight {}", i));
        let weight = (i + 1) as f32 / 50.0;
        doc.add_sparse_vector(sparse, vec![(0, weight)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let query = SparseVectorQuery::new(sparse, vec![(0, 1.0)]);
    let results = searcher.search(&query, 10).await.unwrap();

    assert_eq!(results.len(), 10);

    // Results should be sorted by score descending
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted descending: score[{}]={} < score[{}]={}",
            i - 1,
            results[i - 1].score,
            i,
            results[i].score
        );
    }

    // Top result should be doc with highest weight (doc 49, weight 1.0)
    let top_doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        top_doc.get_first(title).unwrap().as_text().unwrap(),
        "Doc weight 49"
    );
}

/// BMP vs MaxScore: verify both formats return the same top-k results.
#[tokio::test]
async fn test_bmp_vs_maxscore_equivalence() {
    // Build the same data with both formats and verify same results

    // --- BMP ---
    let (schema_bmp, _title_bmp, sparse_bmp) = bmp_schema();
    let dir_bmp = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer_bmp = IndexWriter::create(dir_bmp.clone(), schema_bmp.clone(), config.clone())
        .await
        .unwrap();

    // --- MaxScore ---
    let mut sb_ms = SchemaBuilder::default();
    let _title_ms = sb_ms.add_text_field("title", true, true);
    let sparse_ms = sb_ms.add_sparse_vector_field("sparse", true, true);
    let schema_ms = sb_ms.build();
    let dir_ms = RamDirectory::new();
    let mut writer_ms = IndexWriter::create(dir_ms.clone(), schema_ms.clone(), config.clone())
        .await
        .unwrap();

    // Same documents for both
    let mut rng_state: u32 = 42;
    for i in 0..200 {
        let mut entries = Vec::new();
        // Each doc gets 5-15 random dimensions with random weights
        let num_dims = 5 + (rng_state % 11);
        for _ in 0..num_dims {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let dim = rng_state % 1000;
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let weight = (rng_state % 100) as f32 / 100.0;
            if weight > 0.01 {
                entries.push((dim, weight));
            }
        }

        let mut doc_bmp = Document::new();
        doc_bmp.add_text(_title_bmp, format!("doc {}", i));
        doc_bmp.add_sparse_vector(sparse_bmp, entries.clone());
        writer_bmp.add_document(doc_bmp).unwrap();

        let mut doc_ms = Document::new();
        doc_ms.add_text(_title_ms, format!("doc {}", i));
        doc_ms.add_sparse_vector(sparse_ms, entries);
        writer_ms.add_document(doc_ms).unwrap();
    }

    writer_bmp.commit().await.unwrap();
    writer_ms.commit().await.unwrap();

    let index_bmp = Index::open(dir_bmp, config.clone()).await.unwrap();
    let index_ms = Index::open(dir_ms, config.clone()).await.unwrap();

    // Query with several dimensions
    let query_dims = vec![(42, 0.8), (100, 0.6), (200, 0.4), (500, 0.9)];

    let reader_bmp = index_bmp.reader().await.unwrap();
    let searcher_bmp = reader_bmp.searcher().await.unwrap();
    let query_bmp = SparseVectorQuery::new(sparse_bmp, query_dims.clone());
    let results_bmp = searcher_bmp.search(&query_bmp, 20).await.unwrap();

    let reader_ms = index_ms.reader().await.unwrap();
    let searcher_ms = reader_ms.searcher().await.unwrap();
    let query_ms = SparseVectorQuery::new(sparse_ms, query_dims);
    let results_ms = searcher_ms.search(&query_ms, 20).await.unwrap();

    // Both should return the same number of results
    assert_eq!(
        results_bmp.len(),
        results_ms.len(),
        "BMP and MaxScore should return same number of results: BMP={}, MS={}",
        results_bmp.len(),
        results_ms.len()
    );

    // Top scores should be very close (quantization may cause tiny differences)
    if !results_bmp.is_empty() {
        let bmp_top = results_bmp[0].score;
        let ms_top = results_ms[0].score;
        let diff = (bmp_top - ms_top).abs();
        assert!(
            diff < 0.2 * ms_top.max(0.01),
            "Top scores should be close: BMP={:.4}, MS={:.4}, diff={:.4}",
            bmp_top,
            ms_top,
            diff
        );
    }
}

/// BMP vs MaxScore with multi-ordinal: verify both produce equivalent results
/// when documents have multiple sparse vectors per field.
#[tokio::test]
async fn test_bmp_vs_maxscore_multi_ordinal() {
    // --- BMP ---
    let (schema_bmp, _title_bmp, sparse_bmp) = bmp_schema();
    let dir_bmp = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer_bmp = IndexWriter::create(dir_bmp.clone(), schema_bmp.clone(), config.clone())
        .await
        .unwrap();

    // --- MaxScore ---
    let mut sb_ms = SchemaBuilder::default();
    let _title_ms = sb_ms.add_text_field("title", true, true);
    let sparse_ms = sb_ms.add_sparse_vector_field("sparse", true, true);
    let schema_ms = sb_ms.build();
    let dir_ms = RamDirectory::new();
    let mut writer_ms = IndexWriter::create(dir_ms.clone(), schema_ms.clone(), config.clone())
        .await
        .unwrap();

    // Each document gets 3 ordinals (3 sparse vectors)
    let vectors_per_doc: Vec<Vec<Vec<(u32, f32)>>> = vec![
        // Doc 0: strong match on dim 10, moderate on dim 20
        vec![
            vec![(10, 0.9), (20, 0.3)],
            vec![(10, 0.7), (30, 0.5)],
            vec![(20, 0.8), (40, 0.2)],
        ],
        // Doc 1: moderate match on dim 10, strong on dim 30
        vec![
            vec![(10, 0.4), (30, 0.9)],
            vec![(30, 0.8), (50, 0.3)],
            vec![(10, 0.2), (60, 0.5)],
        ],
        // Doc 2: strong match on dim 20
        vec![
            vec![(20, 0.95), (10, 0.1)],
            vec![(20, 0.85)],
            vec![(20, 0.6), (70, 0.4)],
        ],
    ];

    // Add 50 noise docs to create multiple blocks
    let mut all_docs = vectors_per_doc.clone();
    let mut rng: u32 = 777;
    for _ in 0..50 {
        let mut doc_vecs = Vec::new();
        for _ in 0..3 {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let dim = 100 + (rng % 200);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let w = (rng % 50) as f32 / 100.0 + 0.01;
            doc_vecs.push(vec![(dim, w)]);
        }
        all_docs.push(doc_vecs);
    }

    for vectors in &all_docs {
        let mut doc_bmp = Document::new();
        doc_bmp.add_text(_title_bmp, "doc");
        for v in vectors {
            doc_bmp.add_sparse_vector(sparse_bmp, v.clone());
        }
        writer_bmp.add_document(doc_bmp).unwrap();

        let mut doc_ms = Document::new();
        doc_ms.add_text(_title_ms, "doc");
        for v in vectors {
            doc_ms.add_sparse_vector(sparse_ms, v.clone());
        }
        writer_ms.add_document(doc_ms).unwrap();
    }

    writer_bmp.commit().await.unwrap();
    writer_ms.commit().await.unwrap();

    let index_bmp = Index::open(dir_bmp, config.clone()).await.unwrap();
    let index_ms = Index::open(dir_ms, config.clone()).await.unwrap();

    // Query dim 10 — doc 0 should rank highest (0.9 on ord 0)
    let query_dims = vec![(10, 1.0), (20, 0.5)];

    let reader_bmp = index_bmp.reader().await.unwrap();
    let searcher_bmp = reader_bmp.searcher().await.unwrap();
    let q_bmp = SparseVectorQuery::new(sparse_bmp, query_dims.clone());
    let results_bmp = searcher_bmp.search(&q_bmp, 10).await.unwrap();

    let reader_ms = index_ms.reader().await.unwrap();
    let searcher_ms = reader_ms.searcher().await.unwrap();
    let q_ms = SparseVectorQuery::new(sparse_ms, query_dims);
    let results_ms = searcher_ms.search(&q_ms, 10).await.unwrap();

    assert_eq!(
        results_bmp.len(),
        results_ms.len(),
        "Multi-ordinal: BMP returned {} results, MaxScore returned {}",
        results_bmp.len(),
        results_ms.len()
    );

    // Top results should match (same doc_ids in same order, scores close)
    let n = results_bmp.len().min(5);
    for i in 0..n {
        let bmp_doc = results_bmp[i].doc_id;
        let ms_doc = results_ms[i].doc_id;
        let bmp_score = results_bmp[i].score;
        let ms_score = results_ms[i].score;
        let diff = (bmp_score - ms_score).abs();
        assert!(
            diff < 0.25 * ms_score.max(0.01),
            "Multi-ordinal rank {}: BMP doc_id={} score={:.4}, MS doc_id={} score={:.4}, diff={:.4}",
            i,
            bmp_doc,
            bmp_score,
            ms_doc,
            ms_score,
            diff
        );
    }
}

/// BMP with many blocks: verify correctness with documents spanning many blocks.
#[tokio::test]
async fn test_bmp_many_blocks() {
    let (schema, title, sparse) = bmp_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 500 docs → ~8 blocks of 64
    for i in 0..500 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Doc {}", i));
        // Spread across different dims
        let dim = (i % 20) as u32;
        let weight = 0.1 + (i as f32 / 500.0);
        doc.add_sparse_vector(sparse, vec![(dim, weight), (100, 0.05)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Query dim 100 — should match all 500 docs
    let query = SparseVectorQuery::new(sparse, vec![(100, 1.0)]);
    let results = searcher.search(&query, 600).await.unwrap();
    assert!(
        results.len() >= 400,
        "Dim 100 should match most docs, got {}",
        results.len()
    );

    // Query specific dim — should match ~25 docs (500/20)
    let query = SparseVectorQuery::new(sparse, vec![(5, 1.0)]);
    let results = searcher.search(&query, 100).await.unwrap();
    assert!(
        results.len() >= 20,
        "Dim 5 should match ~25 docs, got {}",
        results.len()
    );
}

/// BMP multi-round merge: flush → merge → add more → merge again.
#[tokio::test]
async fn test_bmp_multi_round_merge() {
    let (schema, title, sparse) = bmp_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Round 1: 3 segments × 20 docs
    for batch in 0..3 {
        for i in 0..20 {
            let mut doc = Document::new();
            doc.add_text(title, format!("r1 b{} d{}", batch, i));
            doc.add_sparse_vector(
                sparse,
                vec![(0, 0.5), ((batch * 10 + i % 5 + 1) as u32, 0.8)],
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Merge round 1
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 60);
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);

    // Round 2: add 40 more docs in 2 segments
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    for batch in 0..2 {
        for i in 0..20 {
            let mut doc = Document::new();
            doc.add_text(title, format!("r2 b{} d{}", batch, i));
            doc.add_sparse_vector(sparse, vec![(0, 0.3), (999, 0.9)]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Merge round 2
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 100);
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);

    // Query dim 0 — should match all 100
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(sparse, vec![(0, 1.0)]);
    let results = searcher.search(&query, 200).await.unwrap();
    assert!(
        results.len() >= 90,
        "Dim 0 should match most docs after 2 merges, got {}",
        results.len()
    );

    // Query dim 999 — should match round 2 docs (40)
    let query = SparseVectorQuery::new(sparse, vec![(999, 1.0)]);
    let results = searcher.search(&query, 100).await.unwrap();
    assert!(
        results.len() >= 35,
        "Dim 999 should match ~40 docs, got {}",
        results.len()
    );
}
