//! Sparse build, merge, directory and lifecycle behavior through Seismic.
use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::SparseVectorQuery;
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};
fn seismic_config() -> SparseVectorConfig {
    SparseVectorConfig {
        format: crate::structures::SparseFormat::Seismic,
        dims: Some(120_000),
        ..Default::default()
    }
}
fn seismic_schema() -> (crate::dsl::Schema, crate::dsl::Field, crate::dsl::Field) {
    let mut schema = SchemaBuilder::default();
    let title = schema.add_text_field("title", true, true);
    let sparse = schema.add_sparse_vector_field_with_config("sparse", true, true, seismic_config());
    (schema.build(), title, sparse)
}
#[tokio::test]
async fn test_seismic_needle_in_haystack() {
    let (schema, title, sparse) = seismic_schema();
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
    needle.add_text(title, "Needle Seismic document");
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
        "Needle Seismic document"
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

#[tokio::test]
async fn test_seismic_pruned_candidate_query_scores_with_full_query() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let sparse = schema_builder.add_sparse_vector_field_with_config(
        "sparse",
        true,
        true,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            weight_quantization: WeightQuantization::UInt8,
            dims: Some(2),
            ..SparseVectorConfig::default()
        },
    );
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    let mut candidate_winner = Document::new();
    candidate_winner.add_text(title, "candidate-only winner");
    candidate_winner.add_sparse_vector(sparse, vec![(0, 1.0)]);
    writer.add_document(candidate_winner).unwrap();

    let mut full_query_winner = Document::new();
    full_query_winner.add_text(title, "full-query winner");
    full_query_winner.add_sparse_vector(sparse, vec![(0, 0.9), (1, 5.0)]);
    writer.add_document(full_query_winner).unwrap();
    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(sparse, vec![(0, 1.0), (1, 0.9)])
        .with_min_query_dims(0)
        .with_pruning(0.5);
    assert_eq!(query.pruned_dims(), &[(0, 1.0)]);

    let results = searcher.search(&query, 1).await.unwrap();
    assert_eq!(results.len(), 1);
    let document = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        document.get_first(title).unwrap().as_text(),
        Some("full-query winner")
    );
}

#[tokio::test]
async fn test_seismic_merge() {
    let (schema, title, sparse) = seismic_schema();
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

#[tokio::test]
async fn test_seismic_score_ranking() {
    let (schema, title, sparse) = seismic_schema();
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

#[tokio::test]
async fn test_seismic_many_blocks() {
    let (schema, title, sparse) = seismic_schema();
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

#[tokio::test]
async fn test_seismic_merge_exact_doc_ids() {
    let (schema, title, sparse) = seismic_schema();
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

#[tokio::test]
async fn test_seismic_multi_round_merge() {
    let (schema, title, sparse) = seismic_schema();
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

#[tokio::test]
async fn test_seismic_merge_correctness() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, seismic_config());
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
    assert!(
        segments
            .iter()
            .all(|segment| segment.seismic_index(sparse).unwrap().run_count() == 1),
        "each new segment owns one encoded nomination run"
    );

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let merged_segments = index.segment_readers().await.unwrap();
    assert_eq!(merged_segments.len(), 1);
    let merged = merged_segments[0].seismic_index(sparse).unwrap();
    assert_eq!(merged.run_count(), NUM_SEGS);
    assert_eq!(merged.total_vectors(), (NUM_SEGS * DOCS_PER_SEG) as u32);
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

#[tokio::test]
async fn test_seismic_merge_large() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, seismic_config());
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

#[tokio::test]
async fn test_seismic_reorder_accepts_segment_without_sparse_values() {
    let (schema, title, _sparse) = seismic_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    let mut doc = Document::new();
    doc.add_text(title, "text only");
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();

    // A sparse field in the schema does not imply that every segment owns a
    // `.sparse` file. Reorder must preserve this valid optional-file case.
    writer.reorder().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 1);
}

#[tokio::test]
async fn test_reorder_skips_segment_consumed_by_merge() {
    let (schema, _title, sparse) = seismic_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    // One Index instance throughout: searcher snapshot, force_merge, and the
    // stale reorder must all share the same SegmentManager/tracker, as in the
    // server — that is what keeps the merged-away files on disk (deferred
    // deletion), the dangerous variant of the race.
    let index = Index::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    let mut writer = index.writer();
    // Two commits → two segments
    for seg in 0..2 {
        for i in 0..300u32 {
            let mut doc = Document::new();
            doc.add_sparse_vector(sparse, vec![(seg * 1000 + i, 1.0), (50_000 + i % 7, 0.5)]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let segs = index.segment_readers().await.unwrap();
    assert_eq!(segs.len(), 2);
    let victim = crate::segment::SegmentId(segs[0].meta().id).to_hex();
    let total_docs: u32 = segs.iter().map(|s| s.num_docs()).sum();
    drop(segs);

    // Hold a searcher snapshot so the merged-away source files stay on disk.
    let reader = index.reader().await.unwrap();
    let _searcher = reader.searcher().await.unwrap();

    writer.force_merge().await.unwrap();

    // The replaced sources are absent from metadata but still protected by
    // `_searcher`. Orphan cleanup must not bypass that deferred-deletion
    // lifetime (production used to sweep these and log them as failed output).
    let swept = index
        .segment_manager()
        .cleanup_orphan_segments()
        .await
        .unwrap();
    assert_eq!(swept, 0, "snapshot-deferred sources are not orphans");
    use crate::directories::Directory;
    let files = dir.list_files(std::path::Path::new("")).await.unwrap();
    assert!(
        files
            .iter()
            .any(|path| path.to_string_lossy().contains(&victim)),
        "snapshot must keep the retired source files alive"
    );

    // Stale candidate: reorder the segment the merge just consumed.
    let reordered = index
        .segment_manager()
        .reorder_single_segment(&victim, None, crate::segment::BpBudget::full())
        .await
        .unwrap();
    assert!(
        !reordered,
        "stale reorder of a merged-away segment must skip"
    );

    // No duplicate docs may appear.
    let index2 = Index::open(dir.clone(), config.clone()).await.unwrap();
    let segs2 = index2.segment_readers().await.unwrap();
    let total_after: u32 = segs2.iter().map(|s| s.num_docs()).sum();
    assert_eq!(
        total_after, total_docs,
        "reordering a merged-away segment duplicated its documents"
    );
}

#[tokio::test]
async fn test_failed_reorder_leaves_no_orphan_files() {
    use crate::directories::Directory;

    let (schema, _title, sparse) = seismic_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let index = Index::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    let mut writer = index.writer();
    for i in 0..200u32 {
        let mut doc = Document::new();
        doc.add_sparse_vector(sparse, vec![(i, 1.0), (50_000 + i % 5, 0.5)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    async fn seg_files(dir: &RamDirectory) -> Vec<String> {
        let mut ids: Vec<String> = dir
            .list_files(std::path::Path::new(""))
            .await
            .unwrap()
            .into_iter()
            .filter_map(|p| {
                let n = p.to_string_lossy().to_string();
                n.starts_with("seg_").then(|| n[4..36].to_string())
            })
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }

    let victim = {
        let segs = index.segment_readers().await.unwrap();
        crate::segment::SegmentId(segs[0].meta().id).to_hex()
    };

    // Failed pass must not leave output files: delete the source's files
    // (still in metadata — prod's ENOENT state) and reorder it.
    crate::segment::delete_segment(&dir, crate::segment::SegmentId::from_hex(&victim).unwrap())
        .await
        .unwrap();
    let before = seg_files(&dir).await;
    let err = index
        .segment_manager()
        .reorder_single_segment(&victim, None, crate::segment::BpBudget::full())
        .await;
    assert!(err.is_err(), "reorder of a fileless segment must fail");
    assert_eq!(
        seg_files(&dir).await,
        before,
        "failed reorder must delete its partial output files"
    );

    // A deletion handed to the async callback must remain protected until
    // that filesystem attempt completes. Production previously let the
    // sweeper race the callback and report successfully replaced sources as
    // failed-output orphans.
    let orphan_hex = "00000000000000000000000000000abc";
    let orphan_id = crate::segment::SegmentId::from_hex(orphan_hex).unwrap();
    use crate::directories::DirectoryWriter;
    dir.write(
        std::path::Path::new(&format!("seg_{orphan_hex}.store")),
        b"junk",
    )
    .await
    .unwrap();
    dir.write(
        std::path::Path::new(&format!("seg_{orphan_hex}.sparse")),
        b"junk",
    )
    .await
    .unwrap();

    let tracker = index.segment_manager().tracker();
    tracker.register(orphan_hex);
    let scheduled = tracker.mark_for_deletion(&[orphan_hex.to_string()]);
    assert_eq!(scheduled, vec![orphan_id]);
    assert!(tracker.is_deletion_protected(orphan_hex));

    let swept = index
        .segment_manager()
        .cleanup_orphan_segments()
        .await
        .unwrap();
    assert_eq!(
        swept, 0,
        "orphan sweep must not race an already scheduled deletion"
    );
    assert!(
        seg_files(&dir).await.contains(&orphan_hex.to_string()),
        "scheduled files must remain untouched by the sweeper"
    );

    // If the async filesystem attempt failed, completing the handoff allows
    // the next sweep to retry and remove the actual orphan.
    tracker.complete_deletion(&scheduled);
    let swept = index
        .segment_manager()
        .cleanup_orphan_segments()
        .await
        .unwrap();
    assert!(swept >= 1, "orphan sweep must delete stray segment files");
    let after = seg_files(&dir).await;
    assert!(
        !after.contains(&orphan_hex.to_string()),
        "orphan files must be gone"
    );
}

#[tokio::test]
async fn test_seismic_multi_segment_search_returns_all_matching_docs() {
    let (schema, doc_id, emb) = repro_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    // Segment 1 (largest — deterministic pilot): the two top-scoring docs.
    for (label, weight) in [("a1", 2.0f32), ("a2", 1.5f32)] {
        let mut doc = Document::new();
        doc.add_text(doc_id, label);
        doc.add_sparse_vector(emb, vec![(5, weight), (9000, weight)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Segment 2: one doc scoring below segment 1's k-th heap entry.
    let mut doc_b = Document::new();
    doc_b.add_text(doc_id, "b");
    doc_b.add_sparse_vector(emb, vec![(5, 1.0)]);
    writer.add_document(doc_b).unwrap();
    writer.commit().await.unwrap();

    // Segment 3: one doc scoring below both.
    let mut doc_c = Document::new();
    doc_c.add_text(doc_id, "c");
    doc_c.add_sparse_vector(emb, vec![(9000, 0.5)]);
    writer.add_document(doc_c).unwrap();
    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 3);
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let query = SparseVectorQuery::new(emb, vec![(5, 1.0), (9000, 1.0)]);
    let results = searcher.search(&query, 10).await.unwrap();
    let labeled = results_by_label(&searcher, doc_id, &results).await;
    assert_eq!(
        labeled
            .iter()
            .map(|(label, _)| label.as_str())
            .collect::<Vec<_>>(),
        vec!["a1", "a2", "b", "c"],
        "multi-segment Seismic search must return matches from every segment, got {labeled:?}",
    );
    for ((_, score), expected) in labeled.iter().zip([4.0f32, 3.0, 1.0, 0.5]) {
        assert!(
            (score - expected).abs() < 0.05,
            "score {score} diverged from expected {expected}: {labeled:?}",
        );
    }
}

#[tokio::test]
async fn test_seismic_multi_segment_search_one_doc_segments_cli_repro() {
    let (schema, doc_id, emb) = repro_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for (label, entries) in [
        ("a", vec![(5u32, 1.5f32), (9000, 2.0), (130_000, 0.5)]),
        ("b", vec![(5, 0.5), (77, 3.0)]),
        ("c", vec![(9000, 2.5), (130_000, 1.0)]),
    ] {
        let mut doc = Document::new();
        doc.add_text(doc_id, label);
        doc.add_sparse_vector(emb, entries);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 3);
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let query = SparseVectorQuery::new(emb, vec![(5, 1.0), (9000, 1.0)]);
    // Run repeatedly: the failure mode depended on segment completion order.
    for round in 0..20 {
        let results = searcher.search(&query, 5).await.unwrap();
        let labeled = results_by_label(&searcher, doc_id, &results).await;
        assert_eq!(
            labeled
                .iter()
                .map(|(label, _)| label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "c", "b"],
            "round {round}: Seismic dropped a segment's results: {labeled:?}",
        );
        for ((_, score), expected) in labeled.iter().zip([3.51f32, 2.51, 0.51]) {
            assert!(
                (score - expected).abs() < 0.05,
                "round {round}: score {score} diverged from expected {expected}",
            );
        }
    }
}

#[tokio::test]
async fn test_seismic_force_merge_collapses_to_single_segment_and_search_is_correct() {
    let (schema, doc_id, emb) = repro_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for (label, entries) in [
        ("a", vec![(5u32, 1.5f32), (9000, 2.0), (130_000, 0.5)]),
        ("b", vec![(5, 0.5), (77, 3.0)]),
        ("c", vec![(9000, 2.5), (130_000, 1.0)]),
    ] {
        let mut doc = Document::new();
        doc.add_text(doc_id, label);
        doc.add_sparse_vector(emb, entries);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 3);
    drop(index);

    writer.force_merge().await.unwrap();
    drop(writer);

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(
        index.segment_readers().await.unwrap().len(),
        1,
        "force merge must end at exactly one segment",
    );
    assert_eq!(index.num_docs().await.unwrap(), 3);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(emb, vec![(5, 1.0), (9000, 1.0)]);
    let results = searcher.search(&query, 5).await.unwrap();
    let labeled = results_by_label(&searcher, doc_id, &results).await;
    assert_eq!(
        labeled
            .iter()
            .map(|(label, _)| label.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "c", "b"],
        "post-merge Seismic search must return every matching doc: {labeled:?}",
    );
    for ((_, score), expected) in labeled.iter().zip([3.51f32, 2.51, 0.51]) {
        assert!(
            (score - expected).abs() < 0.05,
            "post-merge score {score} diverged from expected {expected}",
        );
    }
}

#[tokio::test]
async fn test_seismic_force_merge_ignores_policy_segment_cap() {
    let (schema, _doc_id, emb) = repro_schema();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        // Policy cap of 2 docs per segment would split 3 one-doc sources
        // into 2 outputs if force merge honored it.
        merge_policy: Box::new(crate::merge::TieredMergePolicy {
            max_segment_docs: 2,
            max_merged_docs: 2,
            ..crate::merge::TieredMergePolicy::default()
        }),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    for weight in [1.0f32, 2.0, 3.0] {
        let mut doc = Document::new();
        doc.add_sparse_vector(emb, vec![(5, weight)]);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    drop(writer);

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(
        index.segment_readers().await.unwrap().len(),
        1,
        "explicit force merge must converge to one segment despite the policy cap",
    );
}

#[tokio::test]
async fn test_seismic_without_dims_config_works_or_fails_loud() {
    let mut sb = SchemaBuilder::default();
    let doc_id = sb.add_text_field("doc_id", true, true);
    let emb = sb.add_sparse_vector_field_with_config(
        "emb",
        true,
        true,
        SparseVectorConfig {
            format: SparseFormat::Seismic,
            weight_quantization: WeightQuantization::UInt8,

            // No dims / max_weight configured.
            ..SparseVectorConfig::default()
        },
    );
    let schema = sb.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for (label, entries) in [
        ("a", vec![(5u32, 1.5f32), (9000, 2.0), (130_000, 0.5)]),
        ("b", vec![(5, 0.5), (77, 3.0)]),
        ("c", vec![(9000, 2.5), (130_000, 1.0)]),
    ] {
        let mut doc = Document::new();
        doc.add_text(doc_id, label);
        doc.add_sparse_vector(emb, entries);
        writer.add_document(doc).unwrap();
    }

    let committed = writer.commit().await;
    match committed {
        Err(error) => {
            // Loud refusal is acceptable; silence is not.
            let message = format!("{error}");
            assert!(
                message.contains("dims"),
                "load-time refusal must explain the dims capability mismatch: {message}",
            );
        }
        Ok(_) => {
            let index = Index::open(dir, config).await.unwrap();
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            let query = SparseVectorQuery::new(emb, vec![(5, 1.0), (9000, 1.0)]);
            let results = searcher.search(&query, 5).await.unwrap();
            assert_eq!(
                results.len(),
                3,
                "commit succeeded, so search must return every matching doc",
            );
        }
    }
}

#[tokio::test]
async fn test_seismic_force_merge_works_on_lazy_fs_directory() {
    let (schema, doc_id, emb) = repro_schema();
    let temp = tempfile::TempDir::new().unwrap();
    let dir = crate::directories::FsDirectory::new(temp.path());
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for (label, entries) in [
        ("a", vec![(5u32, 1.5f32), (9000, 2.0), (130_000, 0.5)]),
        ("b", vec![(5, 0.5), (77, 3.0)]),
        ("c", vec![(9000, 2.5), (130_000, 1.0)]),
    ] {
        let mut doc = Document::new();
        doc.add_text(doc_id, label);
        doc.add_sparse_vector(emb, entries);
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
    }

    // Opening and searching Seismic segments must work on a lazy directory.
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 3);
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(emb, vec![(5, 1.0), (9000, 1.0)]);
    let results = searcher.search(&query, 5).await.unwrap();
    assert_eq!(results.len(), 3, "pre-merge lazy-directory search");
    drop(index);

    writer.force_merge().await.unwrap();
    drop(writer);

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(
        index.segment_readers().await.unwrap().len(),
        1,
        "force merge on a lazy directory must end at one segment",
    );
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let results = searcher.search(&query, 5).await.unwrap();
    let mut labels = Vec::new();
    for result in &results {
        let doc = searcher
            .doc(result.segment_id, result.doc_id)
            .await
            .unwrap()
            .unwrap();
        labels.push(
            doc.get_first(doc_id)
                .unwrap()
                .as_text()
                .unwrap()
                .to_string(),
        );
    }
    assert_eq!(
        labels,
        vec!["a", "c", "b"],
        "post-merge lazy-directory search"
    );
}

fn repro_seismic_config() -> SparseVectorConfig {
    SparseVectorConfig {
        format: SparseFormat::Seismic,
        weight_quantization: WeightQuantization::UInt8,

        dims: Some(131_072),
        ..SparseVectorConfig::default()
    }
}

fn repro_schema() -> (crate::dsl::Schema, crate::dsl::Field, crate::dsl::Field) {
    let mut sb = SchemaBuilder::default();
    let doc_id = sb.add_text_field("doc_id", true, true);
    let emb = sb.add_sparse_vector_field_with_config("emb", true, true, repro_seismic_config());
    (sb.build(), doc_id, emb)
}

async fn results_by_label(
    searcher: &crate::index::Searcher<RamDirectory>,
    doc_id: crate::dsl::Field,
    results: &[crate::query::SearchResult],
) -> Vec<(String, f32)> {
    let mut labeled = Vec::with_capacity(results.len());
    for result in results {
        let doc = searcher
            .doc(result.segment_id, result.doc_id)
            .await
            .unwrap()
            .unwrap();
        labeled.push((
            doc.get_first(doc_id)
                .unwrap()
                .as_text()
                .unwrap()
                .to_string(),
            result.score,
        ));
    }
    labeled
}
