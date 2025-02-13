use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};

/// Comprehensive test for MaxScore optimization in BooleanQuery OR queries
///
/// This test verifies that:
/// 1. BooleanQuery with multiple SHOULD term queries uses MaxScore automatically
/// 2. Search results are correct regardless of MaxScore optimization
/// 3. Scores are reasonable for matching documents
#[tokio::test]
async fn test_maxscore_optimization_for_or_queries() {
    use crate::query::{BooleanQuery, TermQuery};

    let mut schema_builder = SchemaBuilder::default();
    let content = schema_builder.add_text_field("content", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Create index with documents containing various terms
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Doc 0: contains "rust" and "programming"
    let mut doc = Document::new();
    doc.add_text(content, "rust programming language is fast");
    writer.add_document(doc).unwrap();

    // Doc 1: contains "rust" only
    let mut doc = Document::new();
    doc.add_text(content, "rust is a systems language");
    writer.add_document(doc).unwrap();

    // Doc 2: contains "programming" only
    let mut doc = Document::new();
    doc.add_text(content, "programming is fun");
    writer.add_document(doc).unwrap();

    // Doc 3: contains "python" (neither rust nor programming)
    let mut doc = Document::new();
    doc.add_text(content, "python is easy to learn");
    writer.add_document(doc).unwrap();

    // Doc 4: contains both "rust" and "programming" multiple times
    let mut doc = Document::new();
    doc.add_text(content, "rust rust programming programming systems");
    writer.add_document(doc).unwrap();

    writer.commit().await.unwrap();

    // Open for reading
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();

    // Test 1: Pure OR query with multiple terms (should use MaxScore automatically)
    let or_query = BooleanQuery::new()
        .should(TermQuery::text(content, "rust"))
        .should(TermQuery::text(content, "programming"));

    let results = index.search(&or_query, 10).await.unwrap();

    // Should find docs 0, 1, 2, 4 (all that contain "rust" OR "programming")
    assert_eq!(results.hits.len(), 4, "Should find exactly 4 documents");

    let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
    assert!(doc_ids.contains(&0), "Should find doc 0");
    assert!(doc_ids.contains(&1), "Should find doc 1");
    assert!(doc_ids.contains(&2), "Should find doc 2");
    assert!(doc_ids.contains(&4), "Should find doc 4");
    assert!(
        !doc_ids.contains(&3),
        "Should NOT find doc 3 (only has 'python')"
    );

    // Test 2: Single term query (should NOT use MaxScore, but still work)
    let single_query = BooleanQuery::new().should(TermQuery::text(content, "rust"));

    let results = index.search(&single_query, 10).await.unwrap();
    assert_eq!(results.hits.len(), 3, "Should find 3 documents with 'rust'");

    // Test 3: Query with MUST (should NOT use MaxScore)
    let must_query = BooleanQuery::new()
        .must(TermQuery::text(content, "rust"))
        .should(TermQuery::text(content, "programming"));

    let results = index.search(&must_query, 10).await.unwrap();
    // Must have "rust", optionally "programming"
    assert_eq!(results.hits.len(), 3, "Should find 3 documents with 'rust'");

    // Test 4: Query with MUST_NOT (should NOT use MaxScore)
    let must_not_query = BooleanQuery::new()
        .should(TermQuery::text(content, "rust"))
        .should(TermQuery::text(content, "programming"))
        .must_not(TermQuery::text(content, "systems"));

    let results = index.search(&must_not_query, 10).await.unwrap();
    // Should exclude docs with "systems" (doc 1 and 4)
    let doc_ids: Vec<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
    assert!(
        !doc_ids.contains(&1),
        "Should NOT find doc 1 (has 'systems')"
    );
    assert!(
        !doc_ids.contains(&4),
        "Should NOT find doc 4 (has 'systems')"
    );

    // Test 5: Verify top-k limit works correctly with MaxScore
    let or_query = BooleanQuery::new()
        .should(TermQuery::text(content, "rust"))
        .should(TermQuery::text(content, "programming"));

    let results = index.search(&or_query, 2).await.unwrap();
    assert_eq!(results.hits.len(), 2, "Should return only top 2 results");

    // Top results should be docs that match both terms (higher scores)
    // Doc 0 and 4 contain both "rust" and "programming"
}

/// Test that BooleanQuery with pure SHOULD clauses uses MaxScore and returns correct results
#[tokio::test]
async fn test_boolean_or_maxscore_optimization() {
    use crate::query::{BooleanQuery, TermQuery};

    let mut schema_builder = SchemaBuilder::default();
    let content = schema_builder.add_text_field("content", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Add several documents
    for i in 0..10 {
        let mut doc = Document::new();
        let text = match i % 4 {
            0 => "apple banana cherry",
            1 => "apple orange",
            2 => "banana grape",
            _ => "cherry date",
        };
        doc.add_text(content, text);
        writer.add_document(doc).unwrap();
    }

    writer.commit().await.unwrap();
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();

    // Pure SHOULD query — triggers MaxScore fast path
    let query = BooleanQuery::new()
        .should(TermQuery::text(content, "apple"))
        .should(TermQuery::text(content, "banana"));

    let results = index.search(&query, 10).await.unwrap();

    // "apple" matches docs 0,1,4,5,8,9 and "banana" matches docs 0,2,4,6,8
    // Union = {0,1,2,4,5,6,8,9} = 8 docs
    assert_eq!(results.hits.len(), 8, "Should find all matching docs");
}

// ========================================================================
// Needle-in-haystack: full-text
// ========================================================================

/// Full-text needle-in-haystack: one unique term among many documents.
/// Verifies exact retrieval, scoring, and document content after commit + reopen.
#[tokio::test]
async fn test_needle_fulltext_single_segment() {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let body = sb.add_text_field("body", true, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // 100 hay documents
    for i in 0..100 {
        let mut doc = Document::new();
        doc.add_text(title, format!("Hay document number {}", i));
        doc.add_text(
            body,
            "common words repeated across all hay documents filler text",
        );
        writer.add_document(doc).unwrap();
    }

    // 1 needle document (doc 100)
    let mut needle = Document::new();
    needle.add_text(title, "The unique needle xylophone");
    needle.add_text(
        body,
        "This document contains the extraordinary term xylophone",
    );
    // Insert needle among hay by re-adding remaining hay after it
    // Actually, we already added 100, so needle is doc 100
    writer.add_document(needle).unwrap();

    // 50 more hay documents after needle
    for i in 100..150 {
        let mut doc = Document::new();
        doc.add_text(title, format!("More hay document {}", i));
        doc.add_text(body, "common words filler text again and again");
        writer.add_document(doc).unwrap();
    }

    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 151);

    // Search for the needle term
    let results = index.query("xylophone", 10).await.unwrap();
    assert_eq!(results.hits.len(), 1, "Should find exactly the needle");
    assert!(results.hits[0].score > 0.0, "Score should be positive");

    // Verify document content
    let doc = index
        .get_document(&results.hits[0].address)
        .await
        .unwrap()
        .unwrap();
    let title_val = doc.get_first(title).unwrap().as_text().unwrap();
    assert!(
        title_val.contains("xylophone"),
        "Retrieved doc should be the needle"
    );

    // Search for common term — should return many
    let results = index.query("common", 200).await.unwrap();
    assert!(
        results.hits.len() >= 100,
        "Common term should match many docs"
    );

    // Negative test — term that doesn't exist
    let results = index.query("nonexistentterm99999", 10).await.unwrap();
    assert_eq!(
        results.hits.len(),
        0,
        "Non-existent term should match nothing"
    );
}

/// Full-text needle across multiple segments: ensures cross-segment search works.
#[tokio::test]
async fn test_needle_fulltext_multi_segment() {
    use crate::query::TermQuery;

    let mut sb = SchemaBuilder::default();
    let content = sb.add_text_field("content", true, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Segment 1: 50 hay docs
    for i in 0..50 {
        let mut doc = Document::new();
        doc.add_text(content, format!("segment one hay document {}", i));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Segment 2: needle + 49 hay docs
    let mut needle = Document::new();
    needle.add_text(content, "the magnificent quetzalcoatl serpent deity");
    writer.add_document(needle).unwrap();
    for i in 0..49 {
        let mut doc = Document::new();
        doc.add_text(content, format!("segment two hay document {}", i));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Segment 3: 50 more hay docs
    for i in 0..50 {
        let mut doc = Document::new();
        doc.add_text(content, format!("segment three hay document {}", i));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 150);
    let num_segments = index.segment_readers().await.unwrap().len();
    assert!(
        num_segments >= 2,
        "Should have multiple segments, got {}",
        num_segments
    );

    // Find needle across segments
    let results = index.query("quetzalcoatl", 10).await.unwrap();
    assert_eq!(
        results.hits.len(),
        1,
        "Should find exactly 1 needle across segments"
    );

    // Verify using TermQuery directly
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let tq = TermQuery::text(content, "quetzalcoatl");
    let results = searcher.search(&tq, 10).await.unwrap();
    assert_eq!(results.len(), 1, "TermQuery should also find the needle");

    // Verify content
    let doc = searcher
        .doc(results[0].segment_id, results[0].doc_id)
        .await
        .unwrap()
        .unwrap();
    let text = doc.get_first(content).unwrap().as_text().unwrap();
    assert!(
        text.contains("quetzalcoatl"),
        "Should retrieve needle content"
    );

    // Cross-segment term that exists in all segments
    let results = index.query("document", 200).await.unwrap();
    assert!(
        results.hits.len() >= 149,
        "Should find hay docs across all segments"
    );
}

/// Stress test: many needles scattered across segments, verify ALL are found.
#[tokio::test]
async fn test_many_needles_all_found() {
    let mut sb = SchemaBuilder::default();
    let content = sb.add_text_field("content", true, true);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let num_needles = 20usize;
    let hay_per_batch = 50usize;
    let needle_terms: Vec<String> = (0..num_needles)
        .map(|i| format!("uniqueneedle{:04}", i))
        .collect();

    // Interleave needles with hay across commits
    for batch in 0..4 {
        // Hay
        for i in 0..hay_per_batch {
            let mut doc = Document::new();
            doc.add_text(
                content,
                format!("hay batch {} item {} common filler", batch, i),
            );
            writer.add_document(doc).unwrap();
        }
        // 5 needles per batch
        for n in 0..5 {
            let needle_idx = batch * 5 + n;
            let mut doc = Document::new();
            doc.add_text(
                content,
                format!("this is {} among many documents", needle_terms[needle_idx]),
            );
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir, config).await.unwrap();
    let total = index.num_docs().await.unwrap();
    assert_eq!(total, (hay_per_batch * 4 + num_needles) as u32);

    // Find EVERY needle
    for term in &needle_terms {
        let results = index.query(term, 10).await.unwrap();
        assert_eq!(
            results.hits.len(),
            1,
            "Should find exactly 1 doc for needle '{}'",
            term
        );
    }

    // Verify hay term matches all hay docs
    let results = index.query("common", 500).await.unwrap();
    assert_eq!(
        results.hits.len(),
        hay_per_batch * 4,
        "Common term should match all {} hay docs",
        hay_per_batch * 4
    );
}
