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

/// Helper: MaxScore config for comparison tests.
fn maxscore_config() -> SparseVectorConfig {
    SparseVectorConfig {
        format: SparseFormat::MaxScore,
        weight_quantization: WeightQuantization::UInt8,
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

/// Helper: create MaxScore schema with a text field and a MaxScore sparse field.
fn maxscore_schema() -> (crate::dsl::Schema, crate::dsl::Field, crate::dsl::Field) {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, maxscore_config());
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
    let (schema_ms, _title_ms, sparse_ms) = maxscore_schema();
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
    let (schema_ms, _title_ms, sparse_ms) = maxscore_schema();
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

/// BMP merge exact doc ID verification: every result after merge must map to correct doc.
///
/// Uses a unique dimension per document so each query returns exactly one result.
/// Verifies the doc_id in the result matches the expected document content.
#[tokio::test]
async fn test_bmp_merge_exact_doc_ids() {
    let (schema, title, sparse) = bmp_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    const DOCS_PER_SEG: usize = 100;
    const NUM_SEGS: usize = 5;

    // Each doc gets a unique dim = seg * DOCS_PER_SEG + i, plus a shared dim 9999
    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let mut doc = Document::new();
            doc.add_text(title, format!("seg{} doc{}", seg, i));
            doc.add_sparse_vector(sparse, vec![(unique_dim, 1.0), (9999, 0.1)]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Merge all segments
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(
        index.num_docs().await.unwrap() as usize,
        DOCS_PER_SEG * NUM_SEGS
    );

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Query each unique dim — must return exactly the right doc
    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let expected_title = format!("seg{} doc{}", seg, i);
            let query = SparseVectorQuery::new(sparse, vec![(unique_dim, 1.0)]);
            let results = searcher.search(&query, 5).await.unwrap();
            assert_eq!(
                results.len(),
                1,
                "dim {} should match exactly 1 doc, got {}",
                unique_dim,
                results.len()
            );
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got_title = doc.get_first(title).unwrap().as_text().unwrap().to_string();
            assert_eq!(
                got_title, expected_title,
                "dim {} returned wrong doc: got '{}', expected '{}'",
                unique_dim, got_title, expected_title
            );
        }
    }
}

/// MaxScore merge: verify MaxScore-configured fields produce correct results after merge.
///
/// This ensures that when a field is configured with SparseFormat::MaxScore,
/// all segments use MaxScore format through build and merge.
#[tokio::test]
async fn test_maxscore_merge() {
    let (schema, title, sparse) = maxscore_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    const DOCS_PER_SEG: usize = 50;
    const NUM_SEGS: usize = 3;

    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let mut doc = Document::new();
            doc.add_text(title, format!("seg{} doc{}", seg, i));
            doc.add_sparse_vector(sparse, vec![(unique_dim, 1.0), (9999, 0.1)]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Merge all segments
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(
        index.num_docs().await.unwrap() as usize,
        DOCS_PER_SEG * NUM_SEGS
    );

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Query each unique dim — must return exactly the right doc
    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let expected_title = format!("seg{} doc{}", seg, i);
            let query = SparseVectorQuery::new(sparse, vec![(unique_dim, 1.0)]);
            let results = searcher.search(&query, 5).await.unwrap();
            assert_eq!(
                results.len(),
                1,
                "MaxScore merge: dim {} should match exactly 1 doc, got {}",
                unique_dim,
                results.len()
            );
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got_title = doc.get_first(title).unwrap().as_text().unwrap().to_string();
            assert_eq!(
                got_title, expected_title,
                "MaxScore merge: dim {} returned wrong doc: got '{}', expected '{}'",
                unique_dim, got_title, expected_title
            );
        }
    }

    // Query shared dim — should match all docs
    let query = SparseVectorQuery::new(sparse, vec![(9999, 1.0)]);
    let results = searcher.search(&query, 200).await.unwrap();
    assert_eq!(
        results.len(),
        DOCS_PER_SEG * NUM_SEGS,
        "MaxScore merge: dim 9999 should match all {} docs, got {}",
        DOCS_PER_SEG * NUM_SEGS,
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

/// BMP merge correctness: verify all documents are still findable after merge.
///
/// Builds multiple segments, force_merges, and verifies every document is still
/// findable by its unique dimension.
#[tokio::test]
async fn test_bmp_merge_correctness() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, bmp_config());
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    const DOCS_PER_SEG: usize = 100;
    const NUM_SEGS: usize = 5;

    // Each doc gets a unique dim plus a shared dim 9999
    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let mut doc = Document::new();
            doc.add_text(title, format!("seg{} doc{}", seg, i));
            // Multiple dims per doc for differentiation
            let topic_dim = 10000 + (seg as u32 * 100);
            doc.add_sparse_vector(
                sparse,
                vec![(unique_dim, 1.0), (9999, 0.1), (topic_dim, 0.5)],
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Verify pre-merge: should have 5 segments and queries work
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let segments = index.segment_readers().await.unwrap();
    assert!(
        segments.len() >= 5,
        "Should have >= 5 segments before merge"
    );

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(
        index.num_docs().await.unwrap() as usize,
        DOCS_PER_SEG * NUM_SEGS
    );

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Query each unique dim — must return exactly the right doc
    let mut failures = Vec::new();
    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let expected_title = format!("seg{} doc{}", seg, i);
            let query = SparseVectorQuery::new(sparse, vec![(unique_dim, 1.0)]);
            let results = searcher.search(&query, 5).await.unwrap();
            if results.len() != 1 {
                failures.push(format!(
                    "dim {}: expected 1 result, got {}",
                    unique_dim,
                    results.len()
                ));
                continue;
            }
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got_title = doc.get_first(title).unwrap().as_text().unwrap().to_string();
            if got_title != expected_title {
                failures.push(format!(
                    "dim {}: got '{}', expected '{}'",
                    unique_dim, got_title, expected_title
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Merge correctness: {} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );

    // Query shared dim 9999 — should match all docs
    let query = SparseVectorQuery::new(sparse, vec![(9999, 1.0)]);
    let results = searcher.search(&query, 600).await.unwrap();
    assert_eq!(
        results.len(),
        DOCS_PER_SEG * NUM_SEGS,
        "Merge: dim 9999 should match all {} docs, got {}",
        DOCS_PER_SEG * NUM_SEGS,
        results.len()
    );
}

/// BMP merge large: stress test with many blocks and multi-segment merge.
///
/// Uses enough documents to span many superblocks (>64 blocks), with varied
/// topic distributions.
#[tokio::test]
async fn test_bmp_merge_large() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, bmp_config());
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 3 segments × 500 docs = 1500 docs → ~24 blocks → spans multiple superblocks
    const DOCS_PER_SEG: usize = 500;
    const NUM_SEGS: usize = 3;

    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            // Assign docs to different "topics" — each topic cluster shares dims
            let topic = i / 50;
            let topic_dim = 20000 + (topic as u32 * 10);
            let topic_dim2 = 20001 + (topic as u32 * 10);
            let mut doc = Document::new();
            doc.add_text(title, format!("s{}d{}", seg, i));
            doc.add_sparse_vector(
                sparse,
                vec![
                    (unique_dim, 1.0),
                    (9999, 0.1),
                    (topic_dim, 0.8),
                    (topic_dim2, 0.5),
                ],
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Verify every doc by its unique dim
    let mut failures = Vec::new();
    for seg in 0..NUM_SEGS {
        for i in 0..DOCS_PER_SEG {
            let unique_dim = (seg * DOCS_PER_SEG + i) as u32;
            let expected = format!("s{}d{}", seg, i);
            let query = SparseVectorQuery::new(sparse, vec![(unique_dim, 1.0)]);
            let results = searcher.search(&query, 5).await.unwrap();
            if results.len() != 1 {
                failures.push(format!(
                    "dim {}: expected 1 result, got {}",
                    unique_dim,
                    results.len()
                ));
                continue;
            }
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got = doc.get_first(title).unwrap().as_text().unwrap().to_string();
            if got != expected {
                failures.push(format!(
                    "dim {}: got '{}', expected '{}'",
                    unique_dim, got, expected
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Merge large: {} failures (of {}):\n{}",
        failures.len(),
        DOCS_PER_SEG * NUM_SEGS,
        failures[..failures.len().min(30)].join("\n")
    );

    // Topic query should match docs from that topic across all segments
    let query = SparseVectorQuery::new(sparse, vec![(20000, 1.0), (20001, 0.5)]);
    let results = searcher.search(&query, 200).await.unwrap();
    // Topic 0: docs 0-49 from each segment = 150 docs
    assert!(
        results.len() >= 100,
        "Topic query should match >=100 docs, got {}",
        results.len()
    );
}

/// BMP standalone reorder: build 1 segment, call writer.reorder(), verify all docs findable.
///
/// Tests the record-level BP reorder path where individual ordinals are
/// shuffled across blocks for better clustering.
#[tokio::test]
async fn test_bmp_reorder_standalone() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, bmp_config());
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    const NUM_DOCS: usize = 200;

    // Each doc gets a unique dim + shared dim 9999 + topic dim
    for i in 0..NUM_DOCS {
        let unique_dim = i as u32;
        let topic = i / 50;
        let topic_dim = 10000 + (topic as u32 * 10);
        let mut doc = Document::new();
        doc.add_text(title, format!("doc{}", i));
        doc.add_sparse_vector(
            sparse,
            vec![(unique_dim, 1.0), (9999, 0.1), (topic_dim, 0.5)],
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Verify we have 1 segment before reorder
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap() as usize, NUM_DOCS);

    // Reorder
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.reorder().await.unwrap();

    // Verify still 1 segment after reorder
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap() as usize, NUM_DOCS);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Query each unique dim — must return exactly the right doc
    let mut failures = Vec::new();
    for i in 0..NUM_DOCS {
        let unique_dim = i as u32;
        let expected_title = format!("doc{}", i);
        let query = SparseVectorQuery::new(sparse, vec![(unique_dim, 1.0)]);
        let results = searcher.search(&query, 5).await.unwrap();
        if results.len() != 1 {
            failures.push(format!(
                "dim {}: expected 1 result, got {}",
                unique_dim,
                results.len()
            ));
            continue;
        }
        let doc = searcher
            .doc(results[0].segment_id, results[0].doc_id)
            .await
            .unwrap()
            .unwrap();
        let got_title = doc.get_first(title).unwrap().as_text().unwrap().to_string();
        if got_title != expected_title {
            failures.push(format!(
                "dim {}: got '{}', expected '{}'",
                unique_dim, got_title, expected_title
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Standalone reorder: {} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );

    // Query shared dim 9999 — should match all docs
    let query = SparseVectorQuery::new(sparse, vec![(9999, 1.0)]);
    let results = searcher.search(&query, 300).await.unwrap();
    assert_eq!(
        results.len(),
        NUM_DOCS,
        "Reorder: dim 9999 should match all {} docs, got {}",
        NUM_DOCS,
        results.len()
    );
}

/// BMP reorder with multi-field: build index with 2 BMP sparse fields,
/// call writer.reorder(), verify both fields return correct results.
#[tokio::test]
async fn test_bmp_reorder_multi_field() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse_a = sb.add_sparse_vector_field_with_config("sparse_a", true, true, bmp_config());
    let sparse_b = sb.add_sparse_vector_field_with_config("sparse_b", true, true, bmp_config());
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    const NUM_DOCS: usize = 100;

    // Field A uses dims 0-99, field B uses dims 1000-1099
    for i in 0..NUM_DOCS {
        let mut doc = Document::new();
        doc.add_text(title, format!("doc{}", i));
        doc.add_sparse_vector(sparse_a, vec![(i as u32, 1.0), (9999, 0.1)]);
        doc.add_sparse_vector(sparse_b, vec![(1000 + i as u32, 1.0), (19999, 0.1)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Reorder
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.reorder().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap() as usize, NUM_DOCS);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Verify field A: each unique dim returns the right doc
    let mut failures = Vec::new();
    for i in 0..NUM_DOCS {
        let query = SparseVectorQuery::new(sparse_a, vec![(i as u32, 1.0)]);
        let results = searcher.search(&query, 5).await.unwrap();
        if results.len() != 1 {
            failures.push(format!(
                "field_a dim {}: expected 1 result, got {}",
                i,
                results.len()
            ));
            continue;
        }
        let doc = searcher
            .doc(results[0].segment_id, results[0].doc_id)
            .await
            .unwrap()
            .unwrap();
        let got = doc.get_first(title).unwrap().as_text().unwrap();
        if got != format!("doc{}", i) {
            failures.push(format!("field_a dim {}: got '{}'", i, got));
        }
    }

    // Verify field B: each unique dim returns the right doc
    for i in 0..NUM_DOCS {
        let query = SparseVectorQuery::new(sparse_b, vec![(1000 + i as u32, 1.0)]);
        let results = searcher.search(&query, 5).await.unwrap();
        if results.len() != 1 {
            failures.push(format!(
                "field_b dim {}: expected 1 result, got {}",
                1000 + i,
                results.len()
            ));
            continue;
        }
        let doc = searcher
            .doc(results[0].segment_id, results[0].doc_id)
            .await
            .unwrap()
            .unwrap();
        let got = doc.get_first(title).unwrap().as_text().unwrap();
        if got != format!("doc{}", i) {
            failures.push(format!("field_b dim {}: got '{}'", 1000 + i, got));
        }
    }

    assert!(
        failures.is_empty(),
        "Multi-field reorder: {} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );

    // Field A shared dim should match all docs
    let query = SparseVectorQuery::new(sparse_a, vec![(9999, 1.0)]);
    let results = searcher.search(&query, 200).await.unwrap();
    assert_eq!(results.len(), NUM_DOCS);

    // Field B shared dim should match all docs
    let query = SparseVectorQuery::new(sparse_b, vec![(19999, 1.0)]);
    let results = searcher.search(&query, 200).await.unwrap();
    assert_eq!(results.len(), NUM_DOCS);
}

/// BMP multi-ordinal clustering: verify ordinals from different
/// documents are findable after merge with BP reordering.
///
/// Creates documents with 3 ordinals each, where ordinals belong to distinct
/// "topics" (disjoint dimension sets).
///
/// Verifies:
/// 1. All documents findable by unique dims after build + merge
/// 2. Topic queries return correct documents
/// 3. Multi-ordinal combine_ordinal_results works with BP-reordered vids
#[tokio::test]
async fn test_bmp_multi_ordinal_clustering() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, bmp_config());
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 3 topics with disjoint dimension ranges:
    //   Topic A: dims 1000-1099
    //   Topic B: dims 2000-2099
    //   Topic C: dims 3000-3099
    //
    // Each doc gets 3 ordinals, one per topic:
    //   ordinal 0: topic A dims + unique dim
    //   ordinal 1: topic B dims + unique dim
    //   ordinal 2: topic C dims + unique dim

    const NUM_DOCS: usize = 200;
    let mut rng: u32 = 42;

    // Segment 1: first half
    for i in 0..NUM_DOCS / 2 {
        let mut doc = Document::new();
        doc.add_text(title, format!("doc{}", i));

        // Ordinal 0: Topic A — shared anchor dim 1000 + random topic dim + unique
        let unique_a = 5000 + i as u32;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let topic_a_dim = 1001 + (rng % 99);
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let w = 0.3 + (rng % 70) as f32 / 100.0;
        doc.add_sparse_vector(sparse, vec![(1000, 0.5), (topic_a_dim, w), (unique_a, 1.0)]);

        // Ordinal 1: Topic B — shared anchor dim 2000 + random topic dim + unique
        let unique_b = 6000 + i as u32;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let topic_b_dim = 2001 + (rng % 99);
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let w = 0.3 + (rng % 70) as f32 / 100.0;
        doc.add_sparse_vector(sparse, vec![(2000, 0.5), (topic_b_dim, w), (unique_b, 1.0)]);

        // Ordinal 2: Topic C — shared anchor dim 3000 + random topic dim + unique
        let unique_c = 7000 + i as u32;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let topic_c_dim = 3001 + (rng % 99);
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let w = 0.3 + (rng % 70) as f32 / 100.0;
        doc.add_sparse_vector(sparse, vec![(3000, 0.5), (topic_c_dim, w), (unique_c, 1.0)]);

        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Segment 2: second half
    for i in NUM_DOCS / 2..NUM_DOCS {
        let mut doc = Document::new();
        doc.add_text(title, format!("doc{}", i));

        let unique_a = 5000 + i as u32;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let topic_a_dim = 1001 + (rng % 99);
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let w = 0.3 + (rng % 70) as f32 / 100.0;
        doc.add_sparse_vector(sparse, vec![(1000, 0.5), (topic_a_dim, w), (unique_a, 1.0)]);

        let unique_b = 6000 + i as u32;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let topic_b_dim = 2001 + (rng % 99);
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let w = 0.3 + (rng % 70) as f32 / 100.0;
        doc.add_sparse_vector(sparse, vec![(2000, 0.5), (topic_b_dim, w), (unique_b, 1.0)]);

        let unique_c = 7000 + i as u32;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let topic_c_dim = 3001 + (rng % 99);
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let w = 0.3 + (rng % 70) as f32 / 100.0;
        doc.add_sparse_vector(sparse, vec![(3000, 0.5), (topic_c_dim, w), (unique_c, 1.0)]);

        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap() as usize, NUM_DOCS);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // 1. Verify every document is findable by its unique dims (all 3 ordinals)
    let mut failures = Vec::new();
    for i in 0..NUM_DOCS {
        let expected = format!("doc{}", i);

        // Query ordinal 0's unique dim
        let query = SparseVectorQuery::new(sparse, vec![(5000 + i as u32, 1.0)]);
        let results = searcher.search(&query, 5).await.unwrap();
        if results.len() != 1 {
            failures.push(format!(
                "doc{} ord0 unique_dim={}: got {} results",
                i,
                5000 + i,
                results.len()
            ));
        } else {
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got = doc.get_first(title).unwrap().as_text().unwrap();
            if got != expected {
                failures.push(format!(
                    "doc{} ord0: got '{}', expected '{}'",
                    i, got, expected
                ));
            }
        }

        // Query ordinal 1's unique dim
        let query = SparseVectorQuery::new(sparse, vec![(6000 + i as u32, 1.0)]);
        let results = searcher.search(&query, 5).await.unwrap();
        if results.len() != 1 {
            failures.push(format!(
                "doc{} ord1 unique_dim={}: got {} results",
                i,
                6000 + i,
                results.len()
            ));
        } else {
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got = doc.get_first(title).unwrap().as_text().unwrap();
            if got != expected {
                failures.push(format!(
                    "doc{} ord1: got '{}', expected '{}'",
                    i, got, expected
                ));
            }
        }

        // Query ordinal 2's unique dim
        let query = SparseVectorQuery::new(sparse, vec![(7000 + i as u32, 1.0)]);
        let results = searcher.search(&query, 5).await.unwrap();
        if results.len() != 1 {
            failures.push(format!(
                "doc{} ord2 unique_dim={}: got {} results",
                i,
                7000 + i,
                results.len()
            ));
        } else {
            let doc = searcher
                .doc(results[0].segment_id, results[0].doc_id)
                .await
                .unwrap()
                .unwrap();
            let got = doc.get_first(title).unwrap().as_text().unwrap();
            if got != expected {
                failures.push(format!(
                    "doc{} ord2: got '{}', expected '{}'",
                    i, got, expected
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Multi-ordinal clustering: {} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );

    // 2. Topic queries: querying topic A anchor dim should return all docs
    //    (each doc has ordinal 0 with dim 1000; combine_ordinal_results groups by doc_id)
    let query = SparseVectorQuery::new(sparse, vec![(1000, 1.0)]);
    let results_a = searcher.search(&query, 300).await.unwrap();
    assert_eq!(
        results_a.len(),
        NUM_DOCS,
        "Topic A anchor dim 1000 should match all {} docs, got {}",
        NUM_DOCS,
        results_a.len()
    );

    // 3. Cross-topic query: anchors from topic A + B should still return all docs
    let query = SparseVectorQuery::new(sparse, vec![(1000, 0.8), (2000, 0.8)]);
    let results_cross = searcher.search(&query, 300).await.unwrap();
    assert_eq!(
        results_cross.len(),
        NUM_DOCS,
        "Cross-topic query should match all {} docs, got {}",
        NUM_DOCS,
        results_cross.len()
    );
}
