use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};

#[tokio::test]
async fn test_vector_index_threshold_switch() {
    use crate::dsl::{DenseVectorConfig, DenseVectorQuantization, VectorIndexType};

    // Create schema with a dense vector field configured for IVF-PQ.
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field("title", true, true);
    let embedding = schema_builder.add_dense_vector_field_with_config(
        "embedding",
        true, // indexed
        true, // stored
        DenseVectorConfig {
            dim: 8,
            index_type: VectorIndexType::IvfPq,
            quantization: DenseVectorQuantization::F32,
            num_clusters: Some(4), // Small for test
            ivf_routing: crate::dsl::IvfRoutingMode::Auto,
            nprobe: 2,
            unit_norm: false,
            soar: None,
        },
    );
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };

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
    assert_eq!(segments.len(), 2, "test requires two unmerged segments");
    assert!(segments.iter().all(|segment| matches!(
        segment.vector_indexes().get(&embedding.0),
        Some(crate::segment::VectorIndex::IvfPq(_))
    )));
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

#[tokio::test]
async fn test_vector_retrain_atomically_replaces_the_complete_generation() {
    use crate::directories::Directory;
    use crate::dsl::DenseVectorConfig;
    use crate::query::DenseVectorQuery;

    let mut sb = SchemaBuilder::default();
    let embedding = sb.add_dense_vector_field_with_config(
        "embedding",
        true,
        true,
        DenseVectorConfig::with_ivf_pq(8, Some(4), 2),
    );
    let schema = sb.build();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        num_indexing_threads: 1,
        ..Default::default()
    };
    let live_index = Index::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    let mut writer = live_index.writer();

    for batch in 0..2 {
        for i in 0..24 {
            let n = (batch * 24 + i) as f32;
            let mut doc = Document::new();
            doc.add_dense_vector(
                embedding,
                (0..8).map(|dim| (n * (dim + 1) as f32).sin()).collect(),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.build_vector_index().await.unwrap();

    // Hold a searcher for generation 1 across the complete retrain. It must
    // retain both its old segments and their matching old codebook.
    let old_reader = live_index.reader().await.unwrap();
    let old_searcher = old_reader.searcher().await.unwrap();
    let first_version = writer.segment_manager.trained().unwrap().codebooks[&embedding.0].version;

    // A materially different third segment changes the training sample and is
    // initially encoded with generation 1 by normal ingestion.
    for i in 0..24 {
        let mut doc = Document::new();
        doc.add_dense_vector(
            embedding,
            (0..8)
                .map(|dim| 1000.0 + i as f32 * 17.0 + dim as f32 * 31.0)
                .collect(),
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let before_ids = writer.segment_manager.get_segment_ids().await;

    writer.retrain_vector_index().await.unwrap();
    let trained = writer.segment_manager.trained().unwrap();
    let second_version = trained.codebooks[&embedding.0].version;
    assert_ne!(
        first_version, second_version,
        "the expanded corpus must retrain the PQ codebook"
    );

    let after_ids = writer.segment_manager.get_segment_ids().await;
    assert_eq!(after_ids.len(), before_ids.len());
    assert!(
        after_ids.iter().all(|id| !before_ids.contains(id)),
        "every segment using the old codebook must be replaced in one generation"
    );
    let current_index = Index::open(dir.clone(), config).await.unwrap();
    for segment in current_index.segment_readers().await.unwrap() {
        let Some(crate::segment::VectorIndex::IvfPq(index)) = segment.get_vector_index(embedding)
        else {
            panic!("every current segment must contain IVF-PQ");
        };
        let header = index.get().header();
        assert_eq!(header.codebook_version, second_version);
        assert_eq!(
            header.quantizer_version,
            trained.centroids[&embedding.0].version
        );
    }

    let old_results = old_searcher
        .search(&DenseVectorQuery::new(embedding, vec![0.25; 8]), 5)
        .await
        .expect("an old reader must remain paired with generation 1");
    assert!(!old_results.is_empty());

    let artifacts = dir
        .list_files(std::path::Path::new(""))
        .await
        .unwrap()
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("vector_artifact_"))
        })
        .count();
    assert_eq!(
        artifacts, 2,
        "only the published centroid/codebook pair remains"
    );
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
            ivf_routing: crate::dsl::IvfRoutingMode::Auto,
            nprobe: 0,
            unit_norm: false,
            soar: None,
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

/// Binary dense vector needle-in-haystack: L1 search + L2 reranking via Hamming distance.
///
/// Tests: single segment, multi-segment, cross-segment rerank, text L1 → binary L2 rerank,
/// score correctness, ranking order, merge preservation.
#[tokio::test]
async fn test_binary_dense_vector_rerank() {
    use crate::dsl::BinaryDenseVectorConfig;
    use crate::query::{BinaryDenseVectorQuery, RerankerConfig, TermQuery};

    let dim_bits = 64; // 64 bits = 8 bytes per vector
    let byte_len = dim_bits / 8;

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let body = sb.add_text_field("body", true, true);
    let bvec = sb.add_binary_dense_vector_field_with_config(
        "bvec",
        true,
        true,
        BinaryDenseVectorConfig::new(dim_bits),
    );
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // --- Segment 1: needle + hay ---
    // Needle: all 1s
    let needle_vec = vec![0xFF_u8; byte_len];
    let mut needle = Document::new();
    needle.add_text(title, "Needle binary document");
    needle.add_text(body, "searchterm unique content");
    needle.add_binary_dense_vector(bvec, needle_vec.clone());
    writer.add_document(needle).unwrap();

    // 25 hay documents in segment 1
    for i in 0u8..25 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay binary doc {}", i));
        doc.add_text(body, "searchterm common filler");
        let v: Vec<u8> = (0..byte_len)
            .map(|d| i.wrapping_add(d as u8) & 0x55)
            .collect();
        doc.add_binary_dense_vector(bvec, v);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // --- Segment 2: near-needle + more hay ---
    // Near-needle: 63 of 64 bits match (one bit flipped)
    let mut near_vec = vec![0xFF_u8; byte_len];
    near_vec[0] = 0xFE; // flip lowest bit
    let mut near = Document::new();
    near.add_text(title, "Near-needle binary document");
    near.add_text(body, "searchterm close match");
    near.add_binary_dense_vector(bvec, near_vec.clone());
    writer.add_document(near).unwrap();

    // 25 more hay documents in segment 2
    for i in 25u8..50 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay binary doc {}", i));
        doc.add_text(body, "searchterm common filler");
        let v: Vec<u8> = (0..byte_len)
            .map(|d| i.wrapping_add(d as u8) & 0x55)
            .collect();
        doc.add_binary_dense_vector(bvec, v);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 52);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // === Test 1: L1 binary search ===
    let query = BinaryDenseVectorQuery::new(bvec, needle_vec.clone());
    let results = searcher.search(&query, 5).await.unwrap();
    assert!(
        !results.is_empty(),
        "L1 binary search should return results"
    );

    let top_doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        top_doc.get_first(title).unwrap().as_text().unwrap(),
        "Needle binary document",
        "L1: exact match should be top result"
    );
    assert!(
        (results[0].score - 1.0).abs() < 1e-6,
        "Exact match score should be 1.0, got {}",
        results[0].score
    );

    // Near-needle should be second
    assert!(results.len() >= 2);
    let second_doc = searcher
        .doc(results[1].segment_id, results[1].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        second_doc.get_first(title).unwrap().as_text().unwrap(),
        "Near-needle binary document",
        "L1: near-needle should be second"
    );
    let expected_near = 1.0 - 1.0 / dim_bits as f32;
    assert!(
        (results[1].score - expected_near).abs() < 1e-6,
        "Near-needle score should be {}, got {}",
        expected_near,
        results[1].score
    );

    // === Test 2: L1 binary + L2 binary rerank (cross-segment) ===
    let reranker_config = RerankerConfig {
        field: bvec,
        vector: Vec::new(),
        binary_vector: needle_vec.clone(),
        combiner: crate::query::MultiValueCombiner::Max,
        unit_norm: false,
        matryoshka_dims: None,
        rrf_k: 0.0,
    };

    let query = BinaryDenseVectorQuery::new(bvec, needle_vec.clone());
    let (reranked, _total) = searcher
        .search_and_rerank(&query, 52, 5, &reranker_config)
        .await
        .unwrap();
    assert!(!reranked.is_empty(), "Reranker should return results");

    let top_doc = searcher
        .doc(reranked[0].segment_id, reranked[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        top_doc.get_first(title).unwrap().as_text().unwrap(),
        "Needle binary document",
        "Reranked: exact match should be top"
    );
    assert!(
        (reranked[0].score - 1.0).abs() < 1e-6,
        "Reranked exact match score should be 1.0, got {}",
        reranked[0].score
    );
    assert!(reranked.len() >= 2);
    let second_doc = searcher
        .doc(reranked[1].segment_id, reranked[1].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        second_doc.get_first(title).unwrap().as_text().unwrap(),
        "Near-needle binary document",
        "Reranked: near-needle should be second"
    );
    assert!(
        (reranked[1].score - expected_near).abs() < 1e-6,
        "Reranked near-needle score should be {}, got {}",
        expected_near,
        reranked[1].score
    );

    // === Test 3: Text L1 → Binary L2 rerank ===
    // L1 retrieves all "searchterm" docs (BM25 order), L2 reranks by Hamming
    let text_query = TermQuery::text(body, "searchterm");
    let (reranked, _total) = searcher
        .search_and_rerank(&text_query, 52, 5, &reranker_config)
        .await
        .unwrap();
    assert!(!reranked.is_empty(), "Text+rerank should return results");

    // After binary reranking, needle (exact match) should be top
    let top_doc = searcher
        .doc(reranked[0].segment_id, reranked[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        top_doc.get_first(title).unwrap().as_text().unwrap(),
        "Needle binary document",
        "Text L1 + Binary L2: exact match should be top after reranking"
    );
    assert!(
        (reranked[0].score - 1.0).abs() < 1e-6,
        "Text+rerank: needle score should be 1.0, got {}",
        reranked[0].score
    );
    // Near-needle should be second
    let second_doc = searcher
        .doc(reranked[1].segment_id, reranked[1].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        second_doc.get_first(title).unwrap().as_text().unwrap(),
        "Near-needle binary document",
        "Text L1 + Binary L2: near-needle should be second after reranking"
    );

    // All reranked scores should be monotonically non-increasing
    for w in reranked.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "Reranked scores should be non-increasing: {} < {}",
            w[0].score,
            w[1].score
        );
    }

    // === Test 4: After merge — verify reranking still works ===
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.segment_readers().await.unwrap().len(), 1);
    assert_eq!(index.num_docs().await.unwrap(), 52);

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Text L1 → Binary L2 rerank after merge
    let text_query = TermQuery::text(body, "searchterm");
    let (reranked, _total) = searcher
        .search_and_rerank(&text_query, 52, 5, &reranker_config)
        .await
        .unwrap();
    assert!(
        !reranked.is_empty(),
        "Post-merge: reranker should return results"
    );

    let top_doc = searcher
        .doc(reranked[0].segment_id, reranked[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        top_doc.get_first(title).unwrap().as_text().unwrap(),
        "Needle binary document",
        "Post-merge: needle should be top after reranking"
    );
    assert!(
        (reranked[0].score - 1.0).abs() < 1e-6,
        "Post-merge: needle score should be 1.0, got {}",
        reranked[0].score
    );
    let second_doc = searcher
        .doc(reranked[1].segment_id, reranked[1].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        second_doc.get_first(title).unwrap().as_text().unwrap(),
        "Near-needle binary document",
        "Post-merge: near-needle should be second after reranking"
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
            ivf_routing: crate::dsl::IvfRoutingMode::Auto,
            nprobe: 0,
            unit_norm: false,
            soar: None,
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

#[tokio::test]
async fn test_search_fused_hybrid_union() {
    // current_thread runtime → sequential sub-query fallback path
    search_fused_hybrid_union_impl().await;
}

/// Same scenario on a multi-thread runtime: sub-queries fan out on rayon
/// under one block_in_place (the parallel fusion path) — results must be
/// identical to the sequential path, including the query-local MaxScore
/// threshold cell (no cross-sub-query threshold leaking).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_search_fused_hybrid_union_parallel_subqueries() {
    search_fused_hybrid_union_impl().await;
}

async fn search_fused_hybrid_union_impl() {
    use crate::dsl::{DenseVectorConfig, VectorIndexType};
    use crate::query::{DenseVectorQuery, FusionMethod, TermQuery};

    let dim = 8;
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
            ivf_routing: crate::dsl::IvfRoutingMode::Auto,
            nprobe: 0,
            unit_norm: false,
            soar: None,
        },
    );
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Doc 0: matches text only (orthogonal vector)
    let mut d = Document::new();
    d.add_text(title, "zebra quantum");
    d.add_dense_vector(embedding, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    writer.add_document(d).unwrap();

    // Doc 1: matches vector only
    let mut d = Document::new();
    d.add_text(title, "unrelated words here");
    d.add_dense_vector(embedding, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    writer.add_document(d).unwrap();

    // Doc 2: matches both (strongest text match via tf=2, near-exact vector)
    let mut d = Document::new();
    d.add_text(title, "zebra zebra habitat");
    d.add_dense_vector(embedding, vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    writer.add_document(d).unwrap();

    // Hay
    for i in 0..20 {
        let mut d = Document::new();
        d.add_text(title, format!("filler document number {}", i));
        d.add_dense_vector(embedding, vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        writer.add_document(d).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let text_query = TermQuery::text(title, "zebra");
    let dense_query =
        DenseVectorQuery::new(embedding, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let fused = searcher
        .search_fused(
            &[(&text_query, 1.0), (&dense_query, 1.0)],
            10,
            10,
            FusionMethod::default(),
            crate::query::MultiValueCombiner::Max,
        )
        .await
        .unwrap();
    let (counted_fused, total_seen) = searcher
        .search_fused_with_count(
            &[(&text_query, 1.0), (&dense_query, 1.0)],
            10,
            10,
            FusionMethod::default(),
            crate::query::MultiValueCombiner::Max,
        )
        .await
        .unwrap();
    assert!(total_seen > 0);
    assert_eq!(
        fused.iter().map(|result| result.doc_id).collect::<Vec<_>>(),
        counted_fused
            .iter()
            .map(|result| result.doc_id)
            .collect::<Vec<_>>()
    );

    // Union semantics: text-only, dense-only, and both-match docs all present
    assert!(fused.len() >= 3, "expected at least 3 fused results");
    assert_eq!(
        fused[0].doc_id, 2,
        "doc matching both retrievers should rank first"
    );
    let ids: Vec<u32> = fused.iter().map(|r| r.doc_id).collect();
    assert!(
        ids.contains(&0),
        "text-only doc must survive fusion (union)"
    );
    assert!(
        ids.contains(&1),
        "dense-only doc must survive fusion (union)"
    );

    // Sanity: neither single retriever alone surfaces all three
    let text_only = searcher.search(&text_query, 10).await.unwrap();
    assert!(!text_only.iter().any(|r| r.doc_id == 1));
}

/// doc_mass cropping: excessive low-weight tail terms of a sparse vector are
/// dropped at indexing time; head terms covering the mass fraction survive.
#[tokio::test]
async fn test_sparse_doc_mass_cropping() {
    use crate::query::SparseVectorQuery;
    use crate::structures::SparseVectorConfig;

    let mut sb = SchemaBuilder::default();
    let mut config = SparseVectorConfig::default().with_doc_mass(0.5);
    config.min_terms = 0;
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, config);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let idx_config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), idx_config.clone())
        .await
        .unwrap();

    // Head term (dim 1) carries most of the mass; dims 3 and 4 are the
    // excessive tail that doc_mass=0.5 must crop.
    let mut doc = Document::new();
    doc.add_sparse_vector(sparse, vec![(1, 10.0), (2, 5.0), (3, 0.1), (4, 0.05)]);
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();

    let index = Index::open(dir, idx_config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Head dim survives
    let r = searcher
        .search(&SparseVectorQuery::new(sparse, vec![(1, 1.0)]), 10)
        .await
        .unwrap();
    assert_eq!(r.len(), 1, "head term must remain searchable");

    // Tail dims are cropped
    for dim in [3u32, 4u32] {
        let r = searcher
            .search(&SparseVectorQuery::new(sparse, vec![(dim, 1.0)]), 10)
            .await
            .unwrap();
        assert!(
            r.is_empty(),
            "tail dim {} should be cropped by doc_mass",
            dim
        );
    }
}

/// min_terms protects short sparse vectors from doc_mass cropping.
#[tokio::test]
async fn test_sparse_doc_mass_respects_min_terms() {
    use crate::query::SparseVectorQuery;
    use crate::structures::SparseVectorConfig;

    let mut sb = SchemaBuilder::default();
    let mut config = SparseVectorConfig::default().with_doc_mass(0.5);
    config.min_terms = 4; // vector below has exactly 4 entries -> not cropped
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, true, config);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let idx_config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), idx_config.clone())
        .await
        .unwrap();

    let mut doc = Document::new();
    doc.add_sparse_vector(sparse, vec![(1, 10.0), (2, 5.0), (3, 0.1), (4, 0.05)]);
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();

    let index = Index::open(dir, idx_config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let r = searcher
        .search(&SparseVectorQuery::new(sparse, vec![(4, 1.0)]), 10)
        .await
        .unwrap();
    assert_eq!(r.len(), 1, "short vectors must not be cropped");
}

/// Regression: BinaryDenseVectorQuery must work on a multi-threaded runtime,
/// where the searcher routes through the rayon-parallel sync scorer path.
/// Previously this failed with "sync scorer not supported for this query type".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_binary_dense_search_multi_thread_runtime() {
    use crate::query::BinaryDenseVectorQuery;

    let byte_len = 8; // 64-bit binary vectors
    let mut sb = SchemaBuilder::default();
    let bvec = sb.add_binary_dense_vector_field("bvec", byte_len * 8, true, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Needle: all bits set
    let needle_vec = vec![0xFF_u8; byte_len];
    let mut needle = Document::new();
    needle.add_binary_dense_vector(bvec, needle_vec.clone());
    writer.add_document(needle).unwrap();

    // Hay: half the bits set
    for i in 0u8..20 {
        let mut doc = Document::new();
        let v: Vec<u8> = (0..byte_len)
            .map(|d| i.wrapping_add(d as u8) & 0x55)
            .collect();
        doc.add_binary_dense_vector(bvec, v);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let query = BinaryDenseVectorQuery::new(bvec, needle_vec);
    let results = searcher
        .search(&query, 5)
        .await
        .expect("binary search must work on multi-thread runtime (sync scorer path)");
    assert!(!results.is_empty());
    assert!(
        results[0].score >= 0.99,
        "needle should be an exact Hamming match, got {}",
        results[0].score
    );
}

/// Binary IVF: index built at commit (threshold crossed), searched via IVF path
/// on both async and sync (multi-thread) paths, recall vs brute force.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_binary_ivf_end_to_end() {
    use crate::dsl::{BinaryDenseVectorConfig, BinaryIndexType};
    use crate::query::BinaryDenseVectorQuery;

    let dim_bits = 64;
    let byte_len = dim_bits / 8;

    let mut sb = SchemaBuilder::default();
    let cfg = BinaryDenseVectorConfig::new(dim_bits).with_ivf(Some(8), 8);
    let bvec = sb.add_binary_dense_vector_field_with_config("bvec", true, true, cfg.clone());
    assert_eq!(cfg.index_type, BinaryIndexType::Ivf);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Needle + 199 hay docs (crosses the 100-vector build threshold)
    let needle_vec = vec![0xFF_u8; byte_len];
    let mut needle = Document::new();
    needle.add_binary_dense_vector(bvec, needle_vec.clone());
    writer.add_document(needle).unwrap();

    for i in 0u32..199 {
        let mut doc = Document::new();
        let v: Vec<u8> = (0..byte_len)
            .map(|d| ((i as u8).wrapping_add(d as u8)) & 0x55)
            .collect();
        doc.add_binary_dense_vector(bvec, v);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.build_vector_index().await.unwrap();
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let built_segments = index.segment_readers().await.unwrap();
    assert_eq!(built_segments.len(), 1);
    assert!(matches!(
        built_segments[0].get_vector_index(bvec),
        Some(crate::segment::VectorIndex::BinaryIvf(_))
    ));
    drop(index);
    let first_quantizer =
        writer.segment_manager.trained().unwrap().binary_quantizers[&bvec.0].version;

    // Expand the corpus with the complementary bit distribution, then verify
    // that explicit retraining rebuilds every binary segment against the new
    // global quantizer before an ordinary force merge.
    for i in 0..200u16 {
        let mut doc = Document::new();
        let mut vector = vec![0xAA; byte_len];
        vector[(i as usize) % byte_len] ^= (i as u8).rotate_left((i % 8) as u32);
        doc.add_binary_dense_vector(bvec, vector);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.retrain_vector_index().await.unwrap();
    let second_quantizer =
        writer.segment_manager.trained().unwrap().binary_quantizers[&bvec.0].version;
    assert_ne!(first_quantizer, second_quantizer);
    let retrained = Index::open(dir.clone(), config.clone()).await.unwrap();
    for segment in retrained.segment_readers().await.unwrap() {
        let Some(crate::segment::VectorIndex::BinaryIvf(lazy)) = segment.get_vector_index(bvec)
        else {
            panic!("every retrained binary segment must contain IVF");
        };
        assert_eq!(lazy.get().header().quantizer_version, second_quantizer);
    }
    drop(retrained);
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let segments = index.segment_readers().await.unwrap();
    assert!(
        matches!(
            segments[0].get_vector_index(bvec),
            Some(crate::segment::VectorIndex::BinaryIvf(_))
        ),
        "binary IVF index should be built at commit"
    );

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Multi-thread runtime → sync scorer path through the IVF index
    let query = BinaryDenseVectorQuery::new(bvec, needle_vec);
    let results = searcher.search(&query, 5).await.unwrap();
    assert!(!results.is_empty());
    assert!(
        results[0].score >= 0.99,
        "needle must be found through IVF probing, got {}",
        results[0].score
    );
}

/// Multi-valued binary IVF: the IVF probe's exact Hamming scores are reused
/// for the ordinals it returned, remaining ordinals are exact-scored from
/// flat storage, and the document combiner sees every ordinal exactly once —
/// so with a full probe (nprobe = num_clusters) top-k scores must equal
/// exact brute-force Hamming. Regression for the single-scan +
/// probe-score-reuse rewrite of the binary IVF search path, on both the
/// async (current-thread) and sync (multi-thread) scorer paths.
async fn binary_ivf_multi_value_exact_scores() {
    use crate::dsl::BinaryDenseVectorConfig;
    use crate::query::BinaryDenseVectorQuery;

    let dim_bits = 64;
    let byte_len = dim_bits / 8;

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let cfg = BinaryDenseVectorConfig::new(dim_bits).with_ivf(Some(4), 4); // full probe
    let bvec = sb.add_binary_dense_vector_field_with_config("bvec", true, true, cfg);
    sb.set_multi(bvec, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Needle doc: one exact-match vector plus one far ordinal — Max combiner
    // must score the doc by its best ordinal (the probe-scored one).
    let needle_vec = vec![0xFF_u8; byte_len];
    let mut needle = Document::new();
    needle.add_text(title, "needle");
    needle.add_binary_dense_vector(bvec, needle_vec.clone());
    needle.add_binary_dense_vector(bvec, vec![0x01_u8; byte_len]);
    writer.add_document(needle).unwrap();

    // Near doc: single vector, one bit flipped.
    let mut near_vec = vec![0xFF_u8; byte_len];
    near_vec[0] = 0xFE;
    let mut near = Document::new();
    near.add_text(title, "near");
    near.add_binary_dense_vector(bvec, near_vec);
    writer.add_document(near).unwrap();

    // Hay: two vectors per doc, at most half the bits set (score <= ~0.75).
    for i in 0u32..30 {
        let mut doc = Document::new();
        doc.add_text(title, format!("hay {i}"));
        for ordinal in 0u8..2 {
            let v: Vec<u8> = (0..byte_len)
                .map(|d| ((i as u8).wrapping_add(d as u8).wrapping_add(ordinal)) & 0x55)
                .collect();
            doc.add_binary_dense_vector(bvec, v);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.build_vector_index().await.unwrap();
    let mut merge_trigger = Document::new();
    merge_trigger.add_text(title, "merge trigger");
    merge_trigger.add_binary_dense_vector(bvec, vec![0; byte_len]);
    writer.add_document(merge_trigger).unwrap();
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let segments = index.segment_readers().await.unwrap();
    assert!(
        matches!(
            segments[0].get_vector_index(bvec),
            Some(crate::segment::VectorIndex::BinaryIvf(_))
        ),
        "binary IVF index should be built at commit (63 vectors >= threshold 50)"
    );

    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let results = searcher
        .search(&BinaryDenseVectorQuery::new(bvec, needle_vec), 3)
        .await
        .unwrap();
    assert!(results.len() >= 2);

    let top = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(top.get_first(title).unwrap().as_text().unwrap(), "needle");
    assert!(
        (results[0].score - 1.0).abs() < 1e-6,
        "needle doc must score by its exact-match ordinal, got {}",
        results[0].score
    );

    let second = searcher
        .doc(results[1].segment_id, results[1].doc_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(second.get_first(title).unwrap().as_text().unwrap(), "near");
    let expected_near = 1.0 - 1.0 / dim_bits as f32;
    assert!(
        (results[1].score - expected_near).abs() < 1e-6,
        "near doc must keep its exact Hamming score {expected_near}, got {}",
        results[1].score
    );
}

#[tokio::test]
async fn test_binary_ivf_multi_value_exact_scores_async_path() {
    binary_ivf_multi_value_exact_scores().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_binary_ivf_multi_value_exact_scores_sync_path() {
    binary_ivf_multi_value_exact_scores().await;
}
