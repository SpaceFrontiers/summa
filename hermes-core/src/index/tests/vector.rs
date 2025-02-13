use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};

#[tokio::test]
async fn test_vector_index_threshold_switch() {
    use crate::dsl::{DenseVectorConfig, DenseVectorQuantization, VectorIndexType};

    // Create schema with dense vector field configured for IVF-RaBitQ
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let embedding = schema_builder.add_dense_vector_field_with_config(
        "embedding",
        true, // indexed
        true, // stored
        DenseVectorConfig {
            dim: 8,
            index_type: VectorIndexType::IvfRaBitQ,
            quantization: DenseVectorQuantization::F32,
            num_clusters: Some(4), // Small for test
            nprobe: 2,
            build_threshold: Some(50), // Build when we have 50+ vectors
            unit_norm: false,
        },
    );
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Phase 1: Add vectors below threshold (should use Flat index)
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Add 30 documents (below threshold of 50)
    for i in 0..30 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Document {}", i));
        // Simple embedding: [i, i, i, i, i, i, i, i] normalized
        let vec: Vec<f32> = (0..8).map(|_| (i as f32) / 30.0).collect();
        doc.add_dense_vector(embedding, vec);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Open index and verify it's using Flat (not built yet)
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert!(
        index.segment_manager.trained().is_none(),
        "Should not have trained centroids below threshold"
    );

    // Search should work with Flat index
    let query_vec: Vec<f32> = vec![0.5; 8];
    let segments = index.segment_readers().await.unwrap();
    assert!(!segments.is_empty());

    let results = segments[0]
        .search_dense_vector(
            embedding,
            &query_vec,
            5,
            0,
            1.0,
            crate::query::MultiValueCombiner::Max,
        )
        .await
        .unwrap();
    assert!(!results.is_empty(), "Flat search should return results");

    // Phase 2: Add more vectors to cross threshold
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();

    // Add 30 more documents (total 60, above threshold of 50)
    for i in 30..60 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Document {}", i));
        let vec: Vec<f32> = (0..8).map(|_| (i as f32) / 60.0).collect();
        doc.add_dense_vector(embedding, vec);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Manually trigger vector index build (no longer auto-triggered by commit)
    writer.build_vector_index().await.unwrap();

    // Reopen index and verify trained structures are loaded
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert!(
        index.segment_manager.trained().is_some(),
        "Should have loaded trained centroids for embedding field"
    );

    // Search should still work
    let segments = index.segment_readers().await.unwrap();
    let results = segments[0]
        .search_dense_vector(
            embedding,
            &query_vec,
            5,
            0,
            1.0,
            crate::query::MultiValueCombiner::Max,
        )
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "Search should return results after build"
    );

    // Phase 3: Verify calling build_vector_index again is a no-op
    let writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.build_vector_index().await.unwrap(); // Should skip training

    // Still built (trained structures present in ArcSwap)
    assert!(writer.segment_manager.trained().is_some());
}

/// Sparse vector needle-in-haystack: one document with unique dimensions.
#[tokio::test]
async fn test_needle_sparse_vector() {
    use crate::query::SparseVectorQuery;

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field("sparse", true, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 100 hay documents with sparse vectors on dimensions 0-9
    for i in 0..100 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay sparse doc {}", i));
        // All hay docs share dimensions 0-9 with varying weights
        let entries: Vec<(u32, f32)> = (0..10)
            .map(|d| (d, 0.1 + (i as f32 * 0.001) + (d as f32 * 0.01)))
            .collect();
        doc.add_sparse_vector(sparse, entries);
        writer.add_document(doc).unwrap();
    }

    // Needle: unique dimensions 1000, 1001, 1002 (no other doc has these)
    let mut needle = Document::new();
    needle.add_text(title, "Needle sparse document");
    needle.add_sparse_vector(
        sparse,
        vec![(1000, 0.9), (1001, 0.8), (1002, 0.7), (5, 0.3)],
    );
    writer.add_document(needle).unwrap();

    // 50 more hay docs
    for i in 100..150 {
        let mut doc = Document::new();
        doc.add_text(title, format!("More hay sparse doc {}", i));
        let entries: Vec<(u32, f32)> = (0..10).map(|d| (d, 0.2 + (d as f32 * 0.02))).collect();
        doc.add_sparse_vector(sparse, entries);
        writer.add_document(doc).unwrap();
    }

    writer.commit().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 151);

    // Query with needle's unique dimensions
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(sparse, vec![(1000, 1.0), (1001, 1.0), (1002, 1.0)]);
    let results = searcher.search(&query, 10).await.unwrap();
    assert_eq!(results.len(), 1, "Only needle has dims 1000-1002");
    assert!(results[0].score > 0.0, "Needle score should be positive");

    // Verify it's the right document
    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    let title_val = doc.get_first(title).unwrap().as_text().unwrap();
    assert_eq!(title_val, "Needle sparse document");

    // Query with shared dimension — should match many
    let query_shared = SparseVectorQuery::new(sparse, vec![(5, 1.0)]);
    let results = searcher.search(&query_shared, 200).await.unwrap();
    assert!(
        results.len() >= 100,
        "Shared dim 5 should match many docs, got {}",
        results.len()
    );

    // Query with non-existent dimension — should match nothing
    let query_missing = SparseVectorQuery::new(sparse, vec![(99999, 1.0)]);
    let results = searcher.search(&query_missing, 10).await.unwrap();
    assert_eq!(
        results.len(),
        0,
        "Non-existent dimension should match nothing"
    );
}

/// Sparse vector needle across multiple segments with merge.
#[tokio::test]
async fn test_needle_sparse_vector_multi_segment_merge() {
    use crate::query::SparseVectorQuery;

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let sparse = sb.add_sparse_vector_field("sparse", true, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Segment 1: hay
    for i in 0..30 {
        let mut doc = Document::new();
        doc.add_text(title, format!("seg1 hay {}", i));
        doc.add_sparse_vector(sparse, vec![(0, 0.5), (1, 0.3)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Segment 2: needle + hay
    let mut needle = Document::new();
    needle.add_text(title, "seg2 needle");
    needle.add_sparse_vector(sparse, vec![(500, 0.95), (501, 0.85)]);
    writer.add_document(needle).unwrap();
    for i in 0..29 {
        let mut doc = Document::new();
        doc.add_text(title, format!("seg2 hay {}", i));
        doc.add_sparse_vector(sparse, vec![(0, 0.4), (2, 0.6)]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Verify pre-merge
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 60);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(sparse, vec![(500, 1.0), (501, 1.0)]);
    let results = searcher.search(&query, 10).await.unwrap();
    assert_eq!(results.len(), 1, "Pre-merge: needle should be found");
    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        doc.get_first(title).unwrap().as_text().unwrap(),
        "seg2 needle"
    );

    // Force merge
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    // Verify post-merge
    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), 60);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = SparseVectorQuery::new(sparse, vec![(500, 1.0), (501, 1.0)]);
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
}

/// Dense vector needle-in-haystack using brute-force (Flat) search.
#[tokio::test]
async fn test_needle_dense_vector_flat() {
    use crate::dsl::{DenseVectorConfig, VectorIndexType};
    use crate::query::DenseVectorQuery;

    let dim = 16;
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let embedding = sb.add_dense_vector_field_with_config(
        "embedding",
        true,
        true,
        DenseVectorConfig {
            dim,
            index_type: VectorIndexType::Flat,
            quantization: crate::dsl::DenseVectorQuantization::F32,
            num_clusters: None,
            nprobe: 0,
            build_threshold: None,
            unit_norm: false,
        },
    );
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 100 hay docs: vectors near origin (small random-ish values)
    for i in 0..100 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay dense doc {}", i));
        // Hay vectors: low-magnitude, varying direction
        let vec: Vec<f32> = (0..dim)
            .map(|d| ((i * 7 + d * 13) % 100) as f32 / 1000.0)
            .collect();
        doc.add_dense_vector(embedding, vec);
        writer.add_document(doc).unwrap();
    }

    // Needle: vector pointing strongly in one direction [1,1,1,...,1]
    let mut needle = Document::new();
    needle.add_text(title, "Needle dense document");
    let needle_vec: Vec<f32> = vec![1.0; dim];
    needle.add_dense_vector(embedding, needle_vec.clone());
    writer.add_document(needle).unwrap();

    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 101);

    // Query with the needle vector — it should be the top result
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = DenseVectorQuery::new(embedding, needle_vec);
    let results = searcher.search(&query, 5).await.unwrap();
    assert!(!results.is_empty(), "Should find at least 1 result");

    // The needle (exact match) should be the top result with highest score
    let top_doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    let top_title = top_doc.get_first(title).unwrap().as_text().unwrap();
    assert_eq!(
        top_title, "Needle dense document",
        "Top result should be the needle (exact vector match)"
    );
    assert!(
        results[0].score > 0.9,
        "Exact match should have very high cosine similarity, got {}",
        results[0].score
    );
}

/// Combined: full-text + sparse + dense in the same index.
/// Verifies all three retrieval paths work independently on the same dataset.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_needle_combined_all_modalities() {
    use crate::directories::MmapDirectory;
    use crate::dsl::{DenseVectorConfig, VectorIndexType};
    use crate::query::{DenseVectorQuery, SparseVectorQuery, TermQuery};

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let dim = 8;
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let body = sb.add_text_field("body", true, true);
    let sparse = sb.add_sparse_vector_field("sparse", true, true);
    let embedding = sb.add_dense_vector_field_with_config(
        "embedding",
        true,
        true,
        DenseVectorConfig {
            dim,
            index_type: VectorIndexType::Flat,
            quantization: crate::dsl::DenseVectorQuantization::F32,
            num_clusters: None,
            nprobe: 0,
            build_threshold: None,
            unit_norm: false,
        },
    );
    let schema = sb.build();

    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 80 hay docs with all three modalities
    for i in 0..80u32 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay doc {}", i));
        doc.add_text(body, "general filler text about nothing special");
        doc.add_sparse_vector(sparse, vec![(0, 0.3), (1, 0.2), ((i % 10) + 10, 0.5)]);
        let vec: Vec<f32> = (0..dim)
            .map(|d| ((i as usize * 3 + d * 7) % 50) as f32 / 100.0)
            .collect();
        doc.add_dense_vector(embedding, vec);
        writer.add_document(doc).unwrap();
    }

    // Needle doc: unique in ALL three modalities
    let mut needle = Document::new();
    needle.add_text(title, "The extraordinary rhinoceros");
    needle.add_text(
        body,
        "This document about rhinoceros is the only one with this word",
    );
    needle.add_sparse_vector(sparse, vec![(9999, 0.99), (9998, 0.88)]);
    let needle_vec = vec![0.9; dim];
    needle.add_dense_vector(embedding, needle_vec.clone());
    writer.add_document(needle).unwrap();

    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 81);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // --- Full-text needle ---
    let tq = TermQuery::text(body, "rhinoceros");
    let results = searcher.search(&tq, 10).await.unwrap();
    assert_eq!(
        results.len(),
        1,
        "Full-text: should find exactly the needle"
    );
    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        doc.get_first(title)
            .unwrap()
            .as_text()
            .unwrap()
            .contains("rhinoceros")
    );

    // --- Sparse vector needle ---
    let sq = SparseVectorQuery::new(sparse, vec![(9999, 1.0), (9998, 1.0)]);
    let results = searcher.search(&sq, 10).await.unwrap();
    assert_eq!(results.len(), 1, "Sparse: should find exactly the needle");
    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert!(
        doc.get_first(title)
            .unwrap()
            .as_text()
            .unwrap()
            .contains("rhinoceros")
    );

    // --- Dense vector needle ---
    let dq = DenseVectorQuery::new(embedding, needle_vec);
    let results = searcher.search(&dq, 1).await.unwrap();
    assert!(!results.is_empty(), "Dense: should find at least 1 result");
    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        doc.get_first(title).unwrap().as_text().unwrap(),
        "The extraordinary rhinoceros",
        "Dense: top-1 should be the needle"
    );

    // Verify all three found the same document
    let ft_doc_id = {
        let tq = TermQuery::text(body, "rhinoceros");
        let r = searcher.search(&tq, 1).await.unwrap();
        r[0].doc_id
    };
    let sp_doc_id = {
        let sq = SparseVectorQuery::new(sparse, vec![(9999, 1.0)]);
        let r = searcher.search(&sq, 1).await.unwrap();
        r[0].doc_id
    };
    let dn_doc_id = {
        let dq = DenseVectorQuery::new(embedding, vec![0.9; dim]);
        let r = searcher.search(&dq, 1).await.unwrap();
        r[0].doc_id
    };

    assert_eq!(
        ft_doc_id, sp_doc_id,
        "Full-text and sparse should find same doc"
    );
    assert_eq!(
        sp_doc_id, dn_doc_id,
        "Sparse and dense should find same doc"
    );
}
