//! Comprehensive BooleanQuery tests covering MUST, SHOULD, MUST_NOT combinations
//! with text TermQuery, RangeQuery, SparseVectorQuery, and SparseTermQuery.

use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::{BooleanQuery, RangeQuery, SparseTermQuery, SparseVectorQuery, TermQuery};

/// Shared test index: 100 docs with text, timestamp(u64 fast), sparse embedding.
/// - Doc i: text="document_{i}", timestamp=1000+i*100, sparse dims: {0:0.5, 1:0.3, i:0.8}
async fn create_boolean_test_index() -> (
    Index<crate::directories::MmapDirectory>,
    crate::dsl::Field, // content (text)
    crate::dsl::Field, // timestamp (u64 fast)
    crate::dsl::Field, // embedding (sparse_vector)
) {
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
        doc.add_text(content, format!("doc{}", i));
        doc.add_u64(timestamp, 1000 + i * 100);
        // Unique dim = 100+i to avoid collision with shared dims 0,1
        let entries: Vec<(u32, f32)> = vec![(0, 0.5), (1, 0.3), (100 + i as u32, 0.8)];
        doc.add_sparse_vector(embedding, entries);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 100);
    (index, content, timestamp, embedding)
}

// ── Standalone SparseTermQuery ──────────────────────────────────────────

/// Single SparseTermQuery for one dimension should find only the matching doc.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sparse_term_query_single_dim() {
    let (index, _content, _ts, embedding) = create_boolean_test_index().await;

    // Dim 142 only exists on doc 42 (unique dim = 100+i)
    let q = SparseTermQuery::new(embedding, 142, 1.0);
    let results = index.search(&q, 10).await.unwrap();

    assert_eq!(results.hits.len(), 1, "Only doc 42 has dim 42");
    assert_eq!(results.hits[0].address.doc_id, 42);
    assert!(results.hits[0].score > 0.0);
}

/// SparseTermQuery for shared dim 0 should match all 100 docs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sparse_term_query_shared_dim() {
    let (index, _content, _ts, embedding) = create_boolean_test_index().await;

    let q = SparseTermQuery::new(embedding, 0, 1.0);
    let results = index.search(&q, 200).await.unwrap();

    assert_eq!(results.hits.len(), 100, "Dim 0 is shared across all docs");
}

/// SparseTermQuery for non-existent dim returns empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sparse_term_query_missing_dim() {
    let (index, _content, _ts, embedding) = create_boolean_test_index().await;

    let q = SparseTermQuery::new(embedding, 99999, 1.0);
    let results = index.search(&q, 10).await.unwrap();

    assert_eq!(
        results.hits.len(),
        0,
        "Non-existent dim should match nothing"
    );
}

// ── SHOULD-only (pure OR) paths ──────────────────────────────────────────

/// SparseVectorQuery with multiple dims → BooleanQuery → MaxScore path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sparse_vector_query_maxscore_path() {
    let (index, _content, _ts, embedding) = create_boolean_test_index().await;

    // Query dims 110, 120 — each unique to their doc, plus shared dim 0
    let q = SparseVectorQuery::new(embedding, vec![(110, 1.0), (120, 1.0)]);
    let results = index.search(&q, 10).await.unwrap();

    // Docs 10 and 20 match uniquely (high score), all others match only shared dim 0
    let top2: Vec<u32> = results
        .hits
        .iter()
        .take(2)
        .map(|h| h.address.doc_id)
        .collect();
    assert!(
        top2.contains(&10) && top2.contains(&20),
        "Top 2 should be docs 10 and 20, got {:?}",
        top2
    );
}

// ── MUST + SHOULD ────────────────────────────────────────────────────────

/// MUST RangeQuery + SHOULD SparseVectorQuery: SHOULD-drives-MUST optimization.
/// Only docs matching SHOULD AND passing MUST predicate are returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_must_range_should_sparse_lazy() {
    let (index, _content, timestamp, embedding) = create_boolean_test_index().await;

    // timestamp [5000, 7000] → docs 40..=60 (21 docs)
    // sparse dims {150: 1.0, 151: 1.0} → match docs 50, 51
    let q = BooleanQuery::new()
        .must(RangeQuery::u64_range(timestamp, Some(5000), Some(7000)))
        .should(SparseVectorQuery::new(
            embedding,
            vec![(150, 1.0), (151, 1.0)],
        ));

    let results = index.search(&q, 100).await.unwrap();
    // SHOULD-drives-MUST: only SHOULD-matching docs filtered by Range
    assert_eq!(results.hits.len(), 2);

    let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
    assert!(
        doc_ids.contains(&50) && doc_ids.contains(&51),
        "Docs 50 and 51 should be returned, got {:?}",
        doc_ids
    );
}

/// MUST TermQuery + SHOULD SparseTermQuery: term filter + sparse boost.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_must_term_should_sparse_term() {
    let (index, content, _ts, embedding) = create_boolean_test_index().await;

    // MUST: match doc25 (text "doc25" tokenizes to "doc25")
    // SHOULD: boost doc with dim 125
    let q = BooleanQuery::new()
        .must(TermQuery::text(content, "doc25"))
        .should(SparseTermQuery::new(embedding, 125, 1.0));

    let results = index.search(&q, 10).await.unwrap();

    assert_eq!(results.hits.len(), 1, "Only doc_25 matches the MUST term");
    assert_eq!(results.hits[0].address.doc_id, 25);
    // Score should be > BM25 alone because of sparse boost
    assert!(results.hits[0].score > 0.0);
}

// ── MUST_NOT ─────────────────────────────────────────────────────────────

/// SHOULD sparse + MUST_NOT RangeQuery: exclude by timestamp.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_should_sparse_must_not_range() {
    let (index, _content, timestamp, embedding) = create_boolean_test_index().await;

    // SHOULD: match dim 0 (all 100 docs)
    // MUST_NOT: exclude timestamp >= 5000 (docs 40..99)
    let q = BooleanQuery::new()
        .should(SparseTermQuery::new(embedding, 0, 1.0))
        .must_not(RangeQuery::u64_range(timestamp, Some(5000), None));

    let results = index.search(&q, 200).await.unwrap();

    // Should only have docs 0..39 (timestamp < 5000)
    assert_eq!(
        results.hits.len(),
        40,
        "Should exclude docs with ts >= 5000"
    );
    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 40,
            "Doc {} should have been excluded",
            hit.address.doc_id
        );
    }
}

// ── MUST + SHOULD + MUST_NOT (triple combo) ──────────────────────────────

/// All three clauses: MUST range filter, SHOULD sparse boost, MUST_NOT text exclude.
/// Uses broad SHOULD (dim 0 matches all) so SHOULD-drives-MUST returns all range docs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_must_should_must_not_combined() {
    let (index, content, timestamp, embedding) = create_boolean_test_index().await;

    // MUST: timestamp [2000, 6000] → docs 10..=50 (41 docs)
    // SHOULD: dim 0 (all docs) + dim 130 (boosts doc 30)
    // MUST_NOT: text "doc30" (exclude doc 30)
    let q = BooleanQuery::new()
        .must(RangeQuery::u64_range(timestamp, Some(2000), Some(6000)))
        .should(SparseTermQuery::new(embedding, 0, 0.5))
        .should(SparseTermQuery::new(embedding, 130, 2.0))
        .must_not(TermQuery::text(content, "doc30"));

    let results = index.search(&q, 100).await.unwrap();

    // SHOULD dim 0 matches all 100 docs, filtered by Range → 41, minus doc30 → 40
    assert_eq!(
        results.hits.len(),
        40,
        "Should be 40 docs (41 range - 1 excluded)"
    );

    // Doc 30 must NOT be in results
    let has_30 = results.hits.iter().any(|h| h.address.doc_id == 30);
    assert!(!has_30, "Doc 30 should be excluded by MUST_NOT");
}

// ── Multiple MUST + multiple SHOULD ──────────────────────────────────────

/// Two MUST clauses (range intersection) + two SHOULD SparseTermQuery.
/// SHOULD-drives-MUST: only docs matching SHOULD AND both Range predicates returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_two_must_two_should_sparse() {
    let (index, _content, timestamp, embedding) = create_boolean_test_index().await;

    // MUST: timestamp >= 3000 AND timestamp <= 5000 → docs 20..=40 (21 docs)
    // SHOULD: sparse dim 125 + sparse dim 130 → match docs 25, 30
    let q = BooleanQuery::new()
        .must(RangeQuery::u64_range(timestamp, Some(3000), None))
        .must(RangeQuery::u64_range(timestamp, None, Some(5000)))
        .should(SparseTermQuery::new(embedding, 125, 1.0))
        .should(SparseTermQuery::new(embedding, 130, 1.0));

    let results = index.search(&q, 100).await.unwrap();
    // SHOULD matches docs 25, 30 only; both pass Range [3000,5000]
    assert_eq!(results.hits.len(), 2);

    let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
    assert!(
        doc_ids.contains(&25) && doc_ids.contains(&30),
        "Results should be docs 25 and 30, got {:?}",
        doc_ids
    );
}

// ── Edge cases ───────────────────────────────────────────────────────────

/// Empty SHOULD sparse (no matching dims) returns no results.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_should_only_no_match() {
    let (index, _content, _ts, embedding) = create_boolean_test_index().await;

    let q = BooleanQuery::new().should(SparseTermQuery::new(embedding, 99999, 1.0));

    let results = index.search(&q, 10).await.unwrap();
    assert_eq!(
        results.hits.len(),
        0,
        "Non-existent dim should yield no results"
    );
}

/// SparseVectorQuery with single dim goes through direct SparseTermQuery path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sparse_vector_single_dim_path() {
    let (index, _content, _ts, embedding) = create_boolean_test_index().await;

    // Single dim → SparseVectorQuery creates SparseTermQuery directly
    let q = SparseVectorQuery::new(embedding, vec![(142, 1.0)]);
    let results = index.search(&q, 10).await.unwrap();

    assert_eq!(
        results.hits.len(),
        1,
        "Single dim 142 should match only doc 42"
    );
    assert_eq!(results.hits[0].address.doc_id, 42);
}

// ── Pruning tests ────────────────────────────────────────────────────────
//
// Dedicated index for pruning tests:
// 200 docs. Each doc i has sparse entries:
//   - Shared dim 0: weight 0.1 (low, common to all)
//   - "Strong" group (docs 0..49):   dim 1000 with weight 5.0
//   - "Medium" group (docs 50..99):  dim 2000 with weight 2.0
//   - "Weak" group (docs 100..149):  dim 3000 with weight 0.05
//   - "Unique" group (docs 150..199): dim (4000+i) with weight 1.0
//
// Queries use dims {1000: 10.0, 2000: 5.0, 3000: 0.01, dim_0: 0.001}.
// Pruning should progressively drop the weakest query dims, removing
// "weak" group and shared-only docs from results.

async fn create_pruning_test_index() -> (
    Index<crate::directories::MmapDirectory>,
    crate::dsl::Field, // embedding
) {
    use crate::directories::MmapDirectory;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut sb = SchemaBuilder::default();
    let embedding = sb.add_sparse_vector_field("embedding", true, true);
    let schema = sb.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 8192,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for i in 0u64..200 {
        let mut doc = Document::new();
        let mut entries: Vec<(u32, f32)> = vec![(0, 0.1)]; // shared dim
        match i {
            0..50 => entries.push((1000, 5.0)),               // strong group
            50..100 => entries.push((2000, 2.0)),             // medium group
            100..150 => entries.push((3000, 0.05)),           // weak group
            150..200 => entries.push((4000 + i as u32, 1.0)), // unique per doc
            _ => {}
        }
        doc.add_sparse_vector(embedding, entries);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 200);
    (index, embedding)
}

/// Baseline: no pruning. Query hits all groups (strong, medium, weak, shared).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_baseline_no_pruning() {
    let (index, embedding) = create_pruning_test_index().await;

    // 4 query dims: strong (1000), medium (2000), weak (3000), shared (0)
    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    );

    let results = index.search(&q, 200).await.unwrap();

    // All 200 docs match at least dim 0 (shared)
    assert_eq!(
        results.hits.len(),
        200,
        "Baseline: all docs match via shared dim 0"
    );

    // Strong group should be ranked first
    let top_doc = results.hits[0].address.doc_id;
    assert!(
        top_doc < 50,
        "Top hit should be from strong group, got doc {}",
        top_doc
    );
}

/// weight_threshold=0.005 drops dim 0 (weight=0.001) → docs 150..199 disappear
/// (they only match dim 0; their unique dims aren't in the query).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_weight_threshold_drops_shared_dim() {
    let (index, embedding) = create_pruning_test_index().await;

    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_weight_threshold(0.005);

    let results = index.search(&q, 200).await.unwrap();

    // dim 0 (weight 0.001) is dropped. Only dims 1000, 2000, 3000 remain.
    // Strong group (50 docs) + medium group (50 docs) + weak group (50 docs) = 150
    // Unique group docs 150..199 only had dim 0 → gone
    assert_eq!(
        results.hits.len(),
        150,
        "After threshold 0.005: unique group (50 docs) should be gone, got {}",
        results.hits.len()
    );

    // No doc from unique group should appear
    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 150,
            "Doc {} from unique group should be pruned",
            hit.address.doc_id
        );
    }
}

/// weight_threshold=0.05 drops dims 0 (0.001) and 3000 (0.01) →
/// weak group (docs 100..149) and unique group (docs 150..199) disappear.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_weight_threshold_drops_weak_and_shared() {
    let (index, embedding) = create_pruning_test_index().await;

    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_weight_threshold(0.05);

    let results = index.search(&q, 200).await.unwrap();

    // Only dims 1000 (10.0) and 2000 (5.0) survive.
    // Strong (50) + medium (50) = 100 docs
    assert_eq!(
        results.hits.len(),
        100,
        "After threshold 0.05: only strong+medium (100 docs), got {}",
        results.hits.len()
    );

    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 100,
            "Doc {} should have been pruned (weak/unique group)",
            hit.address.doc_id
        );
    }
}

/// max_query_dims=2 keeps only the 2 strongest dims (1000: 10.0, 2000: 5.0) →
/// weak and unique groups disappear.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_max_query_dims() {
    let (index, embedding) = create_pruning_test_index().await;

    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_max_query_dims(2);

    let results = index.search(&q, 200).await.unwrap();

    // Only 2 strongest dims kept: 1000, 2000
    // Strong (50) + medium (50) = 100
    assert_eq!(
        results.hits.len(),
        100,
        "max_query_dims=2: only strong+medium (100 docs), got {}",
        results.hits.len()
    );

    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 100,
            "Doc {} should be pruned (only 2 dims kept)",
            hit.address.doc_id
        );
    }
}

/// max_query_dims=1 keeps only the single strongest dim (1000: 10.0) →
/// only strong group survives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_max_query_dims_single() {
    let (index, embedding) = create_pruning_test_index().await;

    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_max_query_dims(1);

    let results = index.search(&q, 200).await.unwrap();

    // Only dim 1000 kept → 50 docs
    assert_eq!(
        results.hits.len(),
        50,
        "max_query_dims=1: only strong group (50 docs), got {}",
        results.hits.len()
    );

    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 50,
            "Doc {} should be pruned (only top-1 dim kept)",
            hit.address.doc_id
        );
    }
}

/// pruning=0.5 keeps top 50% of dims (2 out of 4) → same as max_query_dims=2.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_fraction_half() {
    let (index, embedding) = create_pruning_test_index().await;

    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_pruning(0.5);

    let results = index.search(&q, 200).await.unwrap();

    // 50% of 4 dims = 2 dims kept: 1000, 2000
    assert_eq!(
        results.hits.len(),
        100,
        "pruning=0.5: keep top 2 dims → 100 docs, got {}",
        results.hits.len()
    );

    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 100,
            "Doc {} should be pruned (bottom 50% dims dropped)",
            hit.address.doc_id
        );
    }
}

/// pruning=0.25 keeps top 25% of dims (1 out of 4) → only strong group.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_fraction_quarter() {
    let (index, embedding) = create_pruning_test_index().await;

    let q = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_pruning(0.25);

    let results = index.search(&q, 200).await.unwrap();

    // 25% of 4 = 1 dim kept (ceil(1.0) = 1): dim 1000
    assert_eq!(
        results.hits.len(),
        50,
        "pruning=0.25: keep 1 dim → 50 docs, got {}",
        results.hits.len()
    );

    for hit in &results.hits {
        assert!(
            hit.address.doc_id < 50,
            "Doc {} should be pruned (top 25% = 1 dim)",
            hit.address.doc_id
        );
    }
}

/// Score impact: docs that match pruned dims get lower scores.
/// Compare scores with and without pruning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pruning_score_impact() {
    let (index, embedding) = create_pruning_test_index().await;

    // Unpruned query
    let q_full = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    );

    // Pruned: drop dim 0 and dim 3000
    let q_pruned = SparseVectorQuery::new(
        embedding,
        vec![(1000, 10.0), (2000, 5.0), (3000, 0.01), (0, 0.001)],
    )
    .with_weight_threshold(0.05);

    let full = index.search(&q_full, 200).await.unwrap();
    let pruned = index.search(&q_pruned, 200).await.unwrap();

    // Strong group doc 0: in both results
    let full_score = full
        .hits
        .iter()
        .find(|h| h.address.doc_id == 0)
        .unwrap()
        .score;
    let pruned_score = pruned
        .hits
        .iter()
        .find(|h| h.address.doc_id == 0)
        .unwrap()
        .score;

    // Pruned score should be <= full score (missing dim 0's tiny contribution)
    assert!(
        pruned_score <= full_score,
        "Pruned score ({}) should be <= full score ({})",
        pruned_score,
        full_score
    );

    // A weak-group doc (e.g., doc 120) should exist in full but NOT in pruned
    assert!(
        full.hits.iter().any(|h| h.address.doc_id == 120),
        "Doc 120 should be in full results"
    );
    assert!(
        !pruned.hits.iter().any(|h| h.address.doc_id == 120),
        "Doc 120 should NOT be in pruned results (dim 3000 dropped)"
    );
}

// ── Multi-field text search (per-field MaxScore) ──────────────────────

/// Test that multi-field text SHOULD queries return correct results.
/// This exercises the per-field MaxScore grouping optimization: terms on
/// different fields are grouped by field, MaxScore runs per group, and the
/// outer BooleanScorer unions the compact per-field results.
#[tokio::test]
async fn test_multi_field_text_should() {
    use crate::directories::MmapDirectory;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let body = sb.add_text_field("body", true, true);
    let schema = sb.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 8192,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    // Doc 0: "rust" in title, "python" in body — matches both terms
    let mut doc = Document::new();
    doc.add_text(title, "rust programming");
    doc.add_text(body, "python scripting");
    writer.add_document(doc).unwrap();

    // Doc 1: "rust" in both fields
    let mut doc = Document::new();
    doc.add_text(title, "rust language");
    doc.add_text(body, "rust compiler");
    writer.add_document(doc).unwrap();

    // Doc 2: "python" only in body
    let mut doc = Document::new();
    doc.add_text(title, "java enterprise");
    doc.add_text(body, "python machine learning");
    writer.add_document(doc).unwrap();

    // Doc 3: neither term
    let mut doc = Document::new();
    doc.add_text(title, "java enterprise");
    doc.add_text(body, "java spring");
    writer.add_document(doc).unwrap();

    // Doc 4-13: filler docs to have enough data
    for i in 4..14 {
        let mut doc = Document::new();
        doc.add_text(title, format!("filler title {}", i));
        doc.add_text(body, format!("filler body {}", i));
        writer.add_document(doc).unwrap();
    }

    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();

    // Search "rust python" across both fields — SHOULD OR semantics
    let mut query = BooleanQuery::new();
    query = query.should(TermQuery::text(title, "rust"));
    query = query.should(TermQuery::text(body, "rust"));
    query = query.should(TermQuery::text(title, "python"));
    query = query.should(TermQuery::text(body, "python"));

    let results = index.search(&query, 10).await.unwrap();

    // Docs 0, 1, 2 should all match (they contain "rust" or "python" in some field)
    let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
    assert!(
        doc_ids.contains(&0),
        "Doc 0 should match (rust in title, python in body)"
    );
    assert!(
        doc_ids.contains(&1),
        "Doc 1 should match (rust in both fields)"
    );
    assert!(doc_ids.contains(&2), "Doc 2 should match (python in body)");

    // Doc 3 should NOT match (no rust or python)
    assert!(!doc_ids.contains(&3), "Doc 3 should not match (java only)");

    // Doc 1 should score highest (rust in BOTH fields)
    let doc1_score = results
        .hits
        .iter()
        .find(|h| h.address.doc_id == 1)
        .unwrap()
        .score;
    let doc2_score = results
        .hits
        .iter()
        .find(|h| h.address.doc_id == 2)
        .unwrap()
        .score;
    assert!(
        doc1_score > doc2_score,
        "Doc 1 (rust in both fields) should score higher than doc 2 (python in body only): {} vs {}",
        doc1_score,
        doc2_score
    );
}
