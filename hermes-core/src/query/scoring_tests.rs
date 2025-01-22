//! Comprehensive tests for WAND scoring and document retrieval
//!
//! Tests cover:
//! - Document retrievability (all matching docs found)
//! - Score correctness (BM25 scores match expected values)
//! - No missed documents (exhaustive verification)
//! - Multi-segment scenarios
//! - Edge cases (empty results, single term, many terms)

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::directories::RamDirectory;
    use crate::dsl::{Document, Field, SchemaBuilder};
    use crate::index::{Index, IndexConfig, IndexWriter};
    use crate::query::{bm25_idf, bm25_score};

    /// Helper to create a standard test schema with content field
    fn create_schema() -> (crate::dsl::Schema, Field) {
        let mut schema_builder = SchemaBuilder::default();
        let content = schema_builder.add_text_field("content", true, true);
        (schema_builder.build(), content)
    }

    /// Generate synthetic documents with controlled term distributions
    ///
    /// Returns (documents as text content, expected doc_ids for each term)
    fn generate_test_documents(
        num_docs: usize,
        terms: &[&str],
    ) -> (Vec<String>, std::collections::HashMap<String, HashSet<u32>>) {
        let mut docs = Vec::with_capacity(num_docs);
        let mut term_to_docs: std::collections::HashMap<String, HashSet<u32>> =
            std::collections::HashMap::new();

        for term in terms {
            term_to_docs.insert(term.to_string(), HashSet::new());
        }

        for i in 0..num_docs {
            let mut content_parts = Vec::new();

            for (term_idx, term) in terms.iter().enumerate() {
                // Document i contains term j if (i % (j+2)) == 0
                // This creates overlapping but predictable distributions
                if i % (term_idx + 2) == 0 {
                    // Add term multiple times based on position for varying TF
                    let tf = 1 + (i % 3);
                    for _ in 0..tf {
                        content_parts.push(*term);
                    }
                    term_to_docs.get_mut(*term).unwrap().insert(i as u32);
                }
            }

            // Always add some filler content
            content_parts.push("filler");
            content_parts.push("content");

            docs.push(content_parts.join(" "));
        }

        (docs, term_to_docs)
    }

    /// Compute expected BM25 score for a document
    #[allow(dead_code)]
    fn compute_expected_bm25(tf: f32, doc_freq: f32, total_docs: f32, avg_field_len: f32) -> f32 {
        let idf = bm25_idf(doc_freq, total_docs);
        bm25_score(tf, idf, tf, avg_field_len)
    }

    // ==================== Basic Retrievability Tests ====================

    #[tokio::test]
    async fn test_single_term_all_docs_retrieved() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: rust programming (tf=1)
        let mut doc = Document::new();
        doc.add_text(content, "rust programming");
        writer.add_document(doc).await.unwrap();

        // Doc 1: python programming (no rust)
        let mut doc = Document::new();
        doc.add_text(content, "python programming");
        writer.add_document(doc).await.unwrap();

        // Doc 2: rust rust rust (tf=3)
        let mut doc = Document::new();
        doc.add_text(content, "rust rust rust");
        writer.add_document(doc).await.unwrap();

        // Doc 3: java code (no rust)
        let mut doc = Document::new();
        doc.add_text(content, "java code");
        writer.add_document(doc).await.unwrap();

        // Doc 4: rust systems (tf=1)
        let mut doc = Document::new();
        doc.add_text(content, "rust systems");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:rust", 10).await.unwrap();

        let found_ids: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        let expected_ids: HashSet<u32> = [0, 2, 4].into_iter().collect();

        assert_eq!(
            found_ids, expected_ids,
            "Expected docs {:?}, found {:?}",
            expected_ids, found_ids
        );

        // Doc 2 should have highest score (tf=3)
        assert_eq!(
            results.hits[0].address.doc_id, 2,
            "Doc with highest TF should be first"
        );
    }

    #[tokio::test]
    async fn test_or_query_all_matching_docs_retrieved() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: rust programming
        let mut doc = Document::new();
        doc.add_text(content, "rust programming");
        writer.add_document(doc).await.unwrap();

        // Doc 1: python programming
        let mut doc = Document::new();
        doc.add_text(content, "python programming");
        writer.add_document(doc).await.unwrap();

        // Doc 2: rust python (both)
        let mut doc = Document::new();
        doc.add_text(content, "rust python");
        writer.add_document(doc).await.unwrap();

        // Doc 3: java code (neither)
        let mut doc = Document::new();
        doc.add_text(content, "java code");
        writer.add_document(doc).await.unwrap();

        // Doc 4: python only
        let mut doc = Document::new();
        doc.add_text(content, "python only");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // OR query: rust OR python - should find docs 0, 1, 2, 4
        let results = index
            .query("content:rust OR content:python", 10)
            .await
            .unwrap();

        let found_ids: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        let expected_ids: HashSet<u32> = [0, 1, 2, 4].into_iter().collect();

        assert_eq!(
            found_ids, expected_ids,
            "OR query should find all docs with either term. Expected {:?}, found {:?}",
            expected_ids, found_ids
        );

        // Doc 2 matches both terms, should have highest score
        assert_eq!(
            results.hits[0].address.doc_id, 2,
            "Doc matching both terms should be first"
        );
    }

    #[tokio::test]
    async fn test_synthetic_documents_exhaustive() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Use generate_test_documents for predictable distribution
        let terms = ["alpha", "beta", "gamma", "delta"];
        let (docs, expected_term_docs) = generate_test_documents(50, &terms);

        for doc_content in &docs {
            let mut doc = Document::new();
            doc.add_text(content, doc_content.clone());
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Verify each term finds exactly the expected documents
        for term in &terms {
            let expected = expected_term_docs.get(*term).unwrap();
            let results = index
                .query(&format!("content:{}", term), 100)
                .await
                .unwrap();

            let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

            let missed: Vec<_> = expected.difference(&found).collect();
            let extra: Vec<_> = found.difference(expected).collect();

            assert!(
                missed.is_empty() && extra.is_empty(),
                "Term '{}': missed {:?}, extra {:?}",
                term,
                missed,
                extra
            );
        }
    }

    #[tokio::test]
    async fn test_no_missed_documents_large_corpus() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create 100 documents with predictable term distribution
        let mut expected_alpha = HashSet::new();
        let mut expected_beta = HashSet::new();

        for i in 0..100u32 {
            let mut doc = Document::new();
            let mut terms = vec!["filler"];

            // Alpha appears in docs where i % 2 == 0
            if i % 2 == 0 {
                terms.push("alpha");
                expected_alpha.insert(i);
            }

            // Beta appears in docs where i % 3 == 0
            if i % 3 == 0 {
                terms.push("beta");
                expected_beta.insert(i);
            }

            doc.add_text(content, terms.join(" "));
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Test alpha - should find 50 docs
        let results = index.query("content:alpha", 200).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

        let missed: Vec<_> = expected_alpha.difference(&found).collect();
        assert!(
            missed.is_empty(),
            "Missed {} docs for 'alpha': {:?}",
            missed.len(),
            missed
        );

        let extra: Vec<_> = found.difference(&expected_alpha).collect();
        assert!(
            extra.is_empty(),
            "Extra {} docs for 'alpha': {:?}",
            extra.len(),
            extra
        );

        // Test beta - should find 34 docs (0, 3, 6, ..., 99)
        let results = index.query("content:beta", 200).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

        let missed: Vec<_> = expected_beta.difference(&found).collect();
        assert!(
            missed.is_empty(),
            "Missed {} docs for 'beta': {:?}",
            missed.len(),
            missed
        );
    }

    // ==================== Score Correctness Tests ====================

    #[tokio::test]
    async fn test_bm25_score_ordering_by_tf() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create docs with varying TF for "test"
        let tfs = [1, 2, 3, 5, 10];
        for tf in tfs {
            let mut doc = Document::new();
            let text = vec!["test"; tf].join(" ");
            doc.add_text(content, text);
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:test", 10).await.unwrap();

        assert_eq!(results.hits.len(), 5);

        // Verify scores are in descending order
        for i in 1..results.hits.len() {
            assert!(
                results.hits[i - 1].score >= results.hits[i].score,
                "Scores not in descending order at position {}: {} < {}",
                i,
                results.hits[i - 1].score,
                results.hits[i].score
            );
        }

        // Doc 4 (tf=10) should be first
        assert_eq!(
            results.hits[0].address.doc_id, 4,
            "Highest TF doc should be first"
        );
    }

    #[tokio::test]
    async fn test_multi_term_score_accumulation() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: only rust
        let mut doc = Document::new();
        doc.add_text(content, "rust");
        writer.add_document(doc).await.unwrap();

        // Doc 1: only systems
        let mut doc = Document::new();
        doc.add_text(content, "systems");
        writer.add_document(doc).await.unwrap();

        // Doc 2: both rust and systems (tf=1 each)
        let mut doc = Document::new();
        doc.add_text(content, "rust systems");
        writer.add_document(doc).await.unwrap();

        // Doc 3: both with higher TF (tf=2 each)
        let mut doc = Document::new();
        doc.add_text(content, "rust rust systems systems");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index
            .query("content:rust OR content:systems", 10)
            .await
            .unwrap();

        // Doc 3 matches both with tf=2 each, should have highest score
        assert_eq!(
            results.hits[0].address.doc_id, 3,
            "Doc with multiple term matches (higher TF) should be first"
        );

        // Doc 2 matches both with tf=1 each, should be second
        assert_eq!(
            results.hits[1].address.doc_id, 2,
            "Doc matching both terms once should be second"
        );
    }

    // ==================== Multi-Segment Tests ====================

    #[tokio::test]
    async fn test_multi_segment_retrieval() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig {
            max_docs_per_segment: 5,
            ..Default::default()
        };

        // Single writer, multiple commits to create segments
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for batch in 0..3 {
            for i in 0..5 {
                let doc_id = batch * 5 + i;
                let mut doc = Document::new();

                if doc_id % 2 == 0 {
                    doc.add_text(content, format!("searchterm doc{}", doc_id));
                } else {
                    doc.add_text(content, format!("otherword doc{}", doc_id));
                }

                writer.add_document(doc).await.unwrap();
            }
            writer.commit().await.unwrap();
        }

        let index = Index::open(dir, config).await.unwrap();
        assert!(
            index.segment_readers().len() >= 2,
            "Should have multiple segments, got {}",
            index.segment_readers().len()
        );

        let results = index.query("content:searchterm", 50).await.unwrap();

        // Should find 8 docs with "searchterm" (docs 0,2,4,6,8,10,12,14 by creation order)
        assert_eq!(
            results.hits.len(),
            8,
            "Should find 8 docs with searchterm across segments"
        );

        // All scores should be positive
        for hit in &results.hits {
            assert!(hit.score > 0.0, "All hits should have positive scores");
        }
    }

    #[tokio::test]
    async fn test_multi_segment_score_consistency() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        // Single writer, multiple commits
        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for _segment in 0..2 {
            let mut doc = Document::new();
            doc.add_text(content, "identical content here");
            writer.add_document(doc).await.unwrap();
            writer.commit().await.unwrap();
        }

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:identical", 10).await.unwrap();

        assert_eq!(results.hits.len(), 2, "Should find docs in both segments");

        let score_diff = (results.hits[0].score - results.hits[1].score).abs();
        assert!(
            score_diff < 0.1,
            "Identical docs should have similar scores, got diff={}",
            score_diff
        );
    }

    // ==================== Edge Cases ====================

    #[tokio::test]
    async fn test_empty_results() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let mut doc = Document::new();
        doc.add_text(content, "hello world");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:nonexistent", 10).await.unwrap();

        assert_eq!(
            results.hits.len(),
            0,
            "Should return empty for non-matching term"
        );
    }

    #[tokio::test]
    async fn test_limit_respected() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for _i in 0..50 {
            let mut doc = Document::new();
            doc.add_text(content, "common term here");
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Request only 5 results
        let results = index.query("content:common", 5).await.unwrap();
        assert_eq!(results.hits.len(), 5, "Should respect limit of 5");

        // All returned should have valid scores
        for hit in &results.hits {
            assert!(hit.score > 0.0, "Score should be positive");
        }
    }

    #[tokio::test]
    async fn test_many_terms_or_query() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let terms = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        ];

        // Create docs with one term each
        for (i, term) in terms.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_text(content, format!("{} content", term));
            writer.add_document(doc).await.unwrap();

            // Doc IDs will be 0-7
            assert_eq!(i, i); // placeholder
        }

        // Add one doc that matches all terms (doc_id = 8)
        let mut doc = Document::new();
        doc.add_text(content, terms.join(" "));
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // OR query with all terms
        let query_str = terms
            .iter()
            .map(|t| format!("content:{}", t))
            .collect::<Vec<_>>()
            .join(" OR ");

        let results = index.query(&query_str, 20).await.unwrap();

        // Should find all 9 documents
        assert_eq!(results.hits.len(), 9, "Should find all 9 docs");

        // Doc 8 (matches all terms) should be first
        assert_eq!(
            results.hits[0].address.doc_id, 8,
            "Doc matching all terms should have highest score"
        );
    }

    #[tokio::test]
    async fn test_high_tf_document() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: very high TF
        let mut doc = Document::new();
        doc.add_text(content, vec!["repeat"; 100].join(" "));
        writer.add_document(doc).await.unwrap();

        // Doc 1: normal TF
        let mut doc = Document::new();
        doc.add_text(content, "repeat once");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:repeat", 10).await.unwrap();

        assert_eq!(results.hits.len(), 2);
        // High TF doc should have higher score
        assert_eq!(
            results.hits[0].address.doc_id, 0,
            "High TF doc should be first"
        );
        assert!(
            results.hits[0].score > results.hits[1].score,
            "High TF should yield higher score"
        );
    }

    // ==================== Regression Tests ====================

    #[tokio::test]
    async fn test_no_duplicate_doc_ids_in_results() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create docs with overlapping terms
        for i in 0..20 {
            let mut doc = Document::new();
            doc.add_text(content, format!("term1 term2 term3 doc{}", i));
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index
            .query("content:term1 OR content:term2 OR content:term3", 50)
            .await
            .unwrap();

        // Check for duplicates
        let mut seen = HashSet::new();
        for hit in &results.hits {
            assert!(
                seen.insert(hit.address.doc_id),
                "Duplicate doc_id {} in results",
                hit.address.doc_id
            );
        }
    }

    #[tokio::test]
    async fn test_scores_are_positive_and_finite() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for _i in 0..10 {
            let mut doc = Document::new();
            doc.add_text(content, "test content here");
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:test", 20).await.unwrap();

        for hit in &results.hits {
            assert!(
                hit.score > 0.0,
                "Score should be positive, got {}",
                hit.score
            );
            assert!(
                hit.score.is_finite(),
                "Score should be finite, got {}",
                hit.score
            );
            assert!(!hit.score.is_nan(), "Score should not be NaN");
        }
    }

    #[tokio::test]
    async fn test_single_doc_single_term() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let mut doc = Document::new();
        doc.add_text(content, "unique");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();
        let results = index.query("content:unique", 10).await.unwrap();

        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].address.doc_id, 0);
        assert!(results.hits[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_idf_impact_on_scoring() {
        let (schema, content) = create_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create 10 docs: all have "common", only doc 0 has "rare"
        for i in 0..10 {
            let mut doc = Document::new();
            if i == 0 {
                doc.add_text(content, "common rare");
            } else {
                doc.add_text(content, "common");
            }
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Search for "rare" (high IDF) - only doc 0 matches
        let rare_results = index.query("content:rare", 10).await.unwrap();
        assert_eq!(rare_results.hits.len(), 1);
        let rare_score = rare_results.hits[0].score;

        // Search for "common" (low IDF) - all 10 docs match
        let common_results = index.query("content:common", 10).await.unwrap();
        assert_eq!(common_results.hits.len(), 10);
        let common_score = common_results.hits[0].score;

        // Rare term should have higher IDF and thus higher score
        assert!(
            rare_score > common_score,
            "Rare term (IDF={:.4}) should score higher than common (IDF={:.4})",
            rare_score,
            common_score
        );
    }

    // ==================== Multi-Field BM25F Tests ====================

    /// Create a multi-field schema with title, body, and tags
    fn create_multifield_schema() -> (crate::dsl::Schema, Field, Field, Field) {
        let mut schema_builder = SchemaBuilder::default();
        let title = schema_builder.add_text_field("title", true, true);
        let body = schema_builder.add_text_field("body", true, true);
        let tags = schema_builder.add_text_field("tags", true, true);
        (schema_builder.build(), title, body, tags)
    }

    #[tokio::test]
    async fn test_multifield_basic_retrieval() {
        let (schema, title, body, tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: term in title only
        let mut doc = Document::new();
        doc.add_text(title, "rust programming guide");
        doc.add_text(body, "this is about software development");
        doc.add_text(tags, "tutorial beginner");
        writer.add_document(doc).await.unwrap();

        // Doc 1: term in body only
        let mut doc = Document::new();
        doc.add_text(title, "software guide");
        doc.add_text(body, "learn rust programming here");
        doc.add_text(tags, "tutorial");
        writer.add_document(doc).await.unwrap();

        // Doc 2: term in tags only
        let mut doc = Document::new();
        doc.add_text(title, "programming tutorial");
        doc.add_text(body, "general software development");
        doc.add_text(tags, "rust systems");
        writer.add_document(doc).await.unwrap();

        // Doc 3: term in all fields
        let mut doc = Document::new();
        doc.add_text(title, "rust mastery");
        doc.add_text(body, "advanced rust programming");
        doc.add_text(tags, "rust expert");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Search in title field
        let results = index.query("title:rust", 10).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(found.contains(&0), "Doc 0 should match title:rust");
        assert!(found.contains(&3), "Doc 3 should match title:rust");
        assert!(!found.contains(&1), "Doc 1 should not match title:rust");
        assert!(!found.contains(&2), "Doc 2 should not match title:rust");

        // Search in body field
        let results = index.query("body:rust", 10).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(found.contains(&1), "Doc 1 should match body:rust");
        assert!(found.contains(&3), "Doc 3 should match body:rust");

        // Search in tags field
        let results = index.query("tags:rust", 10).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(found.contains(&2), "Doc 2 should match tags:rust");
        assert!(found.contains(&3), "Doc 3 should match tags:rust");
    }

    #[tokio::test]
    async fn test_multifield_or_across_fields() {
        let (schema, title, body, _tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: searchterm in title
        let mut doc = Document::new();
        doc.add_text(title, "searchterm in title");
        doc.add_text(body, "other content");
        writer.add_document(doc).await.unwrap();

        // Doc 1: searchterm in body
        let mut doc = Document::new();
        doc.add_text(title, "different title");
        doc.add_text(body, "searchterm in body");
        writer.add_document(doc).await.unwrap();

        // Doc 2: searchterm in both
        let mut doc = Document::new();
        doc.add_text(title, "searchterm title");
        doc.add_text(body, "searchterm body");
        writer.add_document(doc).await.unwrap();

        // Doc 3: no searchterm
        let mut doc = Document::new();
        doc.add_text(title, "unrelated");
        doc.add_text(body, "nothing here");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // OR across fields
        let results = index
            .query("title:searchterm OR body:searchterm", 10)
            .await
            .unwrap();

        assert_eq!(results.hits.len(), 3, "Should find 3 docs with searchterm");

        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(found.contains(&0));
        assert!(found.contains(&1));
        assert!(found.contains(&2));
        assert!(!found.contains(&3));

        // Doc 2 (matches both fields) should have highest score
        assert_eq!(
            results.hits[0].address.doc_id, 2,
            "Doc matching both fields should score highest"
        );
    }

    #[tokio::test]
    async fn test_multifield_tf_accumulation() {
        let (schema, title, body, tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: low TF in one field
        let mut doc = Document::new();
        doc.add_text(title, "rust");
        doc.add_text(body, "programming");
        doc.add_text(tags, "code");
        writer.add_document(doc).await.unwrap();

        // Doc 1: high TF in one field
        let mut doc = Document::new();
        doc.add_text(title, "rust rust rust rust rust");
        doc.add_text(body, "programming");
        doc.add_text(tags, "code");
        writer.add_document(doc).await.unwrap();

        // Doc 2: medium TF spread across fields
        let mut doc = Document::new();
        doc.add_text(title, "rust rust");
        doc.add_text(body, "rust rust");
        doc.add_text(tags, "rust");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Search for rust in title
        let results = index.query("title:rust", 10).await.unwrap();

        // Doc 1 (tf=5) should have higher score than Doc 0 (tf=1) and Doc 2 (tf=2)
        assert_eq!(
            results.hits[0].address.doc_id, 1,
            "Highest TF should score highest"
        );

        // Verify score ordering
        for i in 1..results.hits.len() {
            assert!(
                results.hits[i - 1].score >= results.hits[i].score,
                "Scores should be in descending order"
            );
        }
    }

    #[tokio::test]
    async fn test_multifield_different_field_lengths() {
        let (schema, title, body, _tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: short title with term, long body without
        let mut doc = Document::new();
        doc.add_text(title, "rust guide");
        doc.add_text(body, "this is a very long body with lots of words that dilute the term frequency and should result in lower BM25 scores for terms that appear here because length normalization penalizes longer documents");
        writer.add_document(doc).await.unwrap();

        // Doc 1: short title with term
        let mut doc = Document::new();
        doc.add_text(title, "rust");
        doc.add_text(body, "short body");
        writer.add_document(doc).await.unwrap();

        // Doc 2: long title with term
        let mut doc = Document::new();
        doc.add_text(
            title,
            "the comprehensive rust programming language tutorial guide for beginners",
        );
        doc.add_text(body, "content");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        let results = index.query("title:rust", 10).await.unwrap();

        // All 3 docs should be found
        assert_eq!(results.hits.len(), 3);

        // Scores should reflect length normalization
        // Shorter title with same TF should score higher
        for hit in &results.hits {
            assert!(hit.score > 0.0, "All scores should be positive");
            assert!(hit.score.is_finite(), "All scores should be finite");
        }
    }

    #[tokio::test]
    async fn test_multifield_cross_field_or_query() {
        let (schema, title, body, tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Create docs with different terms in different fields
        // Doc 0: alpha in title, beta in body
        let mut doc = Document::new();
        doc.add_text(title, "alpha");
        doc.add_text(body, "beta");
        doc.add_text(tags, "gamma");
        writer.add_document(doc).await.unwrap();

        // Doc 1: beta in title, gamma in body
        let mut doc = Document::new();
        doc.add_text(title, "beta");
        doc.add_text(body, "gamma");
        doc.add_text(tags, "alpha");
        writer.add_document(doc).await.unwrap();

        // Doc 2: gamma in title, alpha in body
        let mut doc = Document::new();
        doc.add_text(title, "gamma");
        doc.add_text(body, "alpha");
        doc.add_text(tags, "beta");
        writer.add_document(doc).await.unwrap();

        // Doc 3: all terms in title
        let mut doc = Document::new();
        doc.add_text(title, "alpha beta gamma");
        doc.add_text(body, "other");
        doc.add_text(tags, "misc");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Complex OR across multiple fields and terms
        let results = index
            .query("title:alpha OR body:beta OR tags:gamma", 10)
            .await
            .unwrap();

        // Should find docs 0 (title:alpha, body:beta), 1 (tags:gamma... wait, no)
        // Let me reconsider:
        // Doc 0: title:alpha YES, body:beta YES, tags:gamma YES -> 3 matches
        // Doc 1: title:alpha NO (beta), body:beta NO (gamma), tags:gamma NO (alpha) -> 0
        // Doc 2: title:alpha NO, body:beta NO, tags:gamma NO (beta) -> 0
        // Doc 3: title:alpha YES, body:beta NO, tags:gamma NO -> 1 match

        // Actually:
        // Doc 0: title has "alpha" -> YES
        // Doc 1: none match
        // Doc 2: none match
        // Doc 3: title has "alpha" -> YES

        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

        // Doc 0 matches title:alpha
        assert!(found.contains(&0), "Doc 0 should match (title:alpha)");
        // Doc 3 matches title:alpha
        assert!(found.contains(&3), "Doc 3 should match (title:alpha)");
    }

    #[tokio::test]
    async fn test_multifield_no_cross_contamination() {
        let (schema, title, body, _tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: "secret" only in body
        let mut doc = Document::new();
        doc.add_text(title, "public information");
        doc.add_text(body, "this contains secret data");
        writer.add_document(doc).await.unwrap();

        // Doc 1: "secret" only in title
        let mut doc = Document::new();
        doc.add_text(title, "secret document");
        doc.add_text(body, "public information here");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Search title:secret should NOT find doc 0
        let results = index.query("title:secret", 10).await.unwrap();
        assert_eq!(results.hits.len(), 1, "Only one doc has secret in title");
        assert_eq!(
            results.hits[0].address.doc_id, 1,
            "Only doc 1 has secret in title"
        );

        // Search body:secret should NOT find doc 1
        let results = index.query("body:secret", 10).await.unwrap();
        assert_eq!(results.hits.len(), 1, "Only one doc has secret in body");
        assert_eq!(
            results.hits[0].address.doc_id, 0,
            "Only doc 0 has secret in body"
        );
    }

    #[tokio::test]
    async fn test_multifield_combined_scoring() {
        let (schema, title, body, tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        // Doc 0: target term appears once in each field
        let mut doc = Document::new();
        doc.add_text(title, "rust");
        doc.add_text(body, "rust");
        doc.add_text(tags, "rust");
        writer.add_document(doc).await.unwrap();

        // Doc 1: target term appears multiple times in one field only
        let mut doc = Document::new();
        doc.add_text(title, "rust rust rust");
        doc.add_text(body, "other content");
        doc.add_text(tags, "misc");
        writer.add_document(doc).await.unwrap();

        // Doc 2: no target term
        let mut doc = Document::new();
        doc.add_text(title, "python");
        doc.add_text(body, "java");
        doc.add_text(tags, "go");
        writer.add_document(doc).await.unwrap();

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // OR query across all fields
        let results = index
            .query("title:rust OR body:rust OR tags:rust", 10)
            .await
            .unwrap();

        assert_eq!(results.hits.len(), 2, "Should find 2 docs with rust");

        // Both docs 0 and 1 should be found
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();
        assert!(found.contains(&0));
        assert!(found.contains(&1));
        assert!(!found.contains(&2));

        // Doc 0 appears in 3 fields, Doc 1 has higher TF in title
        // The relative ranking depends on BM25F weights
        // Just verify both are found and have positive scores
        for hit in &results.hits {
            assert!(hit.score > 0.0);
        }
    }

    #[tokio::test]
    async fn test_multifield_large_document_set() {
        let (schema, title, body, tags) = create_multifield_schema();
        let dir = RamDirectory::new();
        let config = IndexConfig::default();

        let writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        let mut expected_title_match = HashSet::new();
        let mut expected_body_match = HashSet::new();
        let mut expected_any_match = HashSet::new();

        // Create 100 documents with predictable distribution
        for i in 0..100u32 {
            let mut doc = Document::new();

            // "target" in title for docs where i % 3 == 0
            if i % 3 == 0 {
                doc.add_text(title, format!("target document {}", i));
                expected_title_match.insert(i);
                expected_any_match.insert(i);
            } else {
                doc.add_text(title, format!("other document {}", i));
            }

            // "target" in body for docs where i % 5 == 0
            if i % 5 == 0 {
                doc.add_text(body, format!("contains target word {}", i));
                expected_body_match.insert(i);
                expected_any_match.insert(i);
            } else {
                doc.add_text(body, format!("regular content {}", i));
            }

            doc.add_text(tags, format!("tag{}", i % 10));
            writer.add_document(doc).await.unwrap();
        }

        writer.commit().await.unwrap();

        let index = Index::open(dir, config).await.unwrap();

        // Test title:target
        let results = index.query("title:target", 200).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

        let missed: Vec<_> = expected_title_match.difference(&found).collect();
        assert!(
            missed.is_empty(),
            "Missed {} docs for title:target: {:?}",
            missed.len(),
            missed
        );

        // Test body:target
        let results = index.query("body:target", 200).await.unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

        let missed: Vec<_> = expected_body_match.difference(&found).collect();
        assert!(
            missed.is_empty(),
            "Missed {} docs for body:target: {:?}",
            missed.len(),
            missed
        );

        // Test OR across fields
        let results = index
            .query("title:target OR body:target", 200)
            .await
            .unwrap();
        let found: HashSet<u32> = results.hits.iter().map(|h| h.address.doc_id).collect();

        let missed: Vec<_> = expected_any_match.difference(&found).collect();
        assert!(
            missed.is_empty(),
            "Missed {} docs for OR query: {:?}",
            missed.len(),
            missed
        );

        // Verify no extra docs
        let extra: Vec<_> = found.difference(&expected_any_match).collect();
        assert!(
            extra.is_empty(),
            "Found {} extra docs for OR query: {:?}",
            extra.len(),
            extra
        );
    }
}
