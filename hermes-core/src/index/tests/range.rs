//! End-to-end tests for RangeQuery on fast fields.

use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::{BooleanQuery, RangeQuery, SparseVectorQuery, TermQuery};

/// Schema: title(text indexed+stored), price(u64 fast), temp(i64 fast), weight(f64 fast)
/// 50 docs across 2 batches, force-merged.
async fn create_range_test_index() -> (
    Index<crate::directories::MmapDirectory>,
    crate::dsl::Field, // title
    crate::dsl::Field, // price (u64)
    crate::dsl::Field, // temp (i64)
    crate::dsl::Field, // weight (f64)
) {
    use crate::directories::MmapDirectory;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let price = sb.add_u64_field("price", false, true);
    sb.set_fast(price, true);
    let temp = sb.add_i64_field("temperature", false, true);
    sb.set_fast(temp, true);
    let weight = sb.add_f64_field("weight", false, true);
    sb.set_fast(weight, true);
    let schema = sb.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    // Batch 1: docs 0..25
    for i in 0u64..25 {
        let mut doc = Document::new();
        doc.add_text(title, format!("product_{} electronics", i));
        doc.add_u64(price, 100 + i * 10); // 100, 110, ..., 340
        doc.add_i64(temp, i as i64 * 10 - 50); // -50, -40, ..., 190
        doc.add_f64(weight, 0.5 + i as f64 * 0.25); // 0.5, 0.75, ..., 6.5
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Batch 2: docs 25..50
    for i in 25u64..50 {
        let mut doc = Document::new();
        doc.add_text(title, format!("product_{} clothing", i));
        doc.add_u64(price, 100 + i * 10); // 350, 360, ..., 590
        doc.add_i64(temp, i as i64 * 10 - 50); // 200, 210, ..., 440
        doc.add_f64(weight, 0.5 + i as f64 * 0.25); // 6.75, 7.0, ..., 12.75
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 50);
    (index, title, price, temp, weight)
}

// ── Standalone RangeQuery tests ──────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_u64_exact() {
    let (index, _, price, _, _) = create_range_test_index().await;

    // price in [200, 300] → i in [10, 20] → 11 docs
    let q = RangeQuery::u64(price, Some(200), Some(300));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        11,
        "U64 range [200,300] should match 11 docs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_u64_open_min() {
    let (index, _, price, _, _) = create_range_test_index().await;

    // price <= 150 → i in [0, 5] → 6 docs (100,110,120,130,140,150)
    let q = RangeQuery::u64(price, None, Some(150));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        6,
        "U64 range [_,150] should match 6 docs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_u64_open_max() {
    let (index, _, price, _, _) = create_range_test_index().await;

    // price >= 550 → i in [45, 49] → 5 docs (550,560,570,580,590)
    let q = RangeQuery::u64(price, Some(550), None);
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        5,
        "U64 range [550,_] should match 5 docs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_i64_crossing_zero() {
    let (index, _, _, temp, _) = create_range_test_index().await;

    // temp in [-20, 20] → i*10-50 in [-20,20] → i in [3,7] → 5 docs
    let q = RangeQuery::i64(temp, Some(-20), Some(20));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        5,
        "I64 range [-20,20] should match 5 docs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_i64_all_negative() {
    let (index, _, _, temp, _) = create_range_test_index().await;

    // temp in [-50, -10] → i*10-50 in [-50,-10] → i in [0,4] → 5 docs
    let q = RangeQuery::i64(temp, Some(-50), Some(-10));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        5,
        "I64 range [-50,-10] should match 5 docs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_f64() {
    let (index, _, _, _, weight) = create_range_test_index().await;

    // weight in [1.0, 3.0] → 0.5+i*0.25 in [1.0,3.0] → i in [2,10] → 9 docs
    let q = RangeQuery::f64(weight, Some(1.0), Some(3.0));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        9,
        "F64 range [1.0,3.0] should match 9 docs"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_no_matches() {
    let (index, _, price, _, _) = create_range_test_index().await;

    // price in [9000, 9999] → no docs
    let q = RangeQuery::u64(price, Some(9000), Some(9999));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(results.hits.len(), 0, "Out-of-range should match 0 docs");
}

// ── BooleanQuery MUST + RangeQuery (intersection via seek) ───────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_in_boolean_must() {
    let (index, title, price, _, _) = create_range_test_index().await;

    // "electronics" matches docs 0..24 (price 100..340)
    // price in [200, 300] → i in [10,20] → intersection: docs 10..20 = 11 docs
    let q = BooleanQuery::new()
        .must(TermQuery::new(title, b"electronics".to_vec()))
        .must(RangeQuery::u64(price, Some(200), Some(300)));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        11,
        "Boolean MUST [text + u64 range] should match 11 docs"
    );
    // All hits should have positive BM25 scores from the text query
    for hit in &results.hits {
        assert!(
            hit.score > 0.0,
            "Hits should have positive scores from text query"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_range_two_ranges_in_boolean() {
    let (index, _, price, temp, _) = create_range_test_index().await;

    // price in [200, 400] → i in [10, 30]
    // temp in [0, 100] → i*10-50 in [0,100] → i in [5, 15]
    // Intersection: i in [10, 15] → 6 docs
    let q = BooleanQuery::new()
        .must(RangeQuery::u64(price, Some(200), Some(400)))
        .must(RangeQuery::i64(temp, Some(0), Some(100)));
    let results = index.search(&q, 100).await.unwrap();
    assert_eq!(
        results.hits.len(),
        6,
        "Two range MUST should match 6 docs (intersection)"
    );
}

// ── MUST RangeQuery (timestamp) + SHOULD SparseVectorQuery ──────────────

/// Realistic scenario: filter by timestamp range, rank by sparse vector relevance.
///
/// Schema: content(text indexed+stored), timestamp(u64 fast), embedding(sparse_vector)
/// 100 docs: timestamp = 1000 + i*100 (1000..10900), sparse dims based on i.
/// "Recent" window: timestamp in [5000, 7000] → i in [40, 60] → 21 docs.
/// Sparse query targets dims {500, 501} — only docs 50 and 51 have these.
///
/// With SHOULD-drives-MUST optimization: only docs matching SHOULD sparse
/// query AND passing the MUST range predicate are returned. MUST-only docs
/// (score=filter_score only) are not returned since they can never outrank
/// SHOULD docs in top-k.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_must_range_should_sparse() {
    use crate::directories::MmapDirectory;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut sb = SchemaBuilder::default();
    let content = sb.add_text_field("content", true, true);
    let timestamp = sb.add_u64_field("timestamp", false, true);
    sb.set_fast(timestamp, true);
    let embedding = sb.add_sparse_vector_field("embedding", true, true);
    let schema = sb.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 8192,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for i in 0u64..100 {
        let mut doc = Document::new();
        doc.add_text(content, format!("document_{}", i));
        doc.add_u64(timestamp, 1000 + i * 100);

        // Every doc gets some shared dims + a unique dim based on i
        let mut entries: Vec<(u32, f32)> = vec![(0, 0.5), (1, 0.3)];
        // Add a unique dimension = i so we can target specific docs
        entries.push((i as u32, 0.8));
        doc.add_sparse_vector(embedding, entries);

        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 100);

    // ── Query: MUST timestamp in [5000, 7000], SHOULD sparse{50: 1.0, 51: 1.0} ──
    let q = BooleanQuery::new()
        .must(RangeQuery::u64(timestamp, Some(5000), Some(7000)))
        .should(SparseVectorQuery::new(
            embedding,
            vec![(50, 1.0), (51, 1.0)],
        ));

    let results = index.search(&q, 100).await.unwrap();

    // SHOULD-drives-MUST: only SHOULD-matching docs filtered by Range
    // SHOULD sparse {50, 51} matches docs 50, 51; both in range [5000,7000]
    assert_eq!(
        results.hits.len(),
        2,
        "Only SHOULD-matching docs in range are returned, got {}",
        results.hits.len()
    );

    let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
    assert!(
        doc_ids.contains(&50) && doc_ids.contains(&51),
        "Results should be docs 50 and 51, got {:?}",
        doc_ids,
    );

    // Scores are pure sparse scores (no filter_score bonus)
    for hit in &results.hits {
        assert!(
            hit.score > 0.0,
            "SHOULD-matching docs should have positive score, got {}",
            hit.score,
        );
    }
}
