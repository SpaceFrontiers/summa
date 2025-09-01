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

/// Test that Russian stemmer works end-to-end: indexing + search via query string.
/// Regression test for https://github.com/SpaceFrontiers/hermes/issues/9
#[tokio::test]
async fn test_russian_stemmer_search() {
    let mut schema_builder = SchemaBuilder::default();
    let title = schema_builder.add_text_field_with_tokenizer("title", true, true, "ru_stem");
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    let mut doc = Document::new();
    doc.add_text(title, "бегущие собаки");
    writer.add_document(doc).unwrap();

    let mut doc = Document::new();
    doc.add_text(title, "маленькая собака");
    writer.add_document(doc).unwrap();

    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();

    // Exact word should match (stemmer maps "собаки" -> "собак")
    let results = index.query("собаки", 10).await.unwrap();
    assert!(
        !results.hits.is_empty(),
        "Russian stemmer: 'собаки' should match documents"
    );

    // Different inflection of same root should also match
    let results = index.query("собака", 10).await.unwrap();
    assert!(
        !results.hits.is_empty(),
        "Russian stemmer: 'собака' should match (same stem as 'собаки')"
    );

    // Field-qualified search should also work
    let results = index.query("title:бегущие", 10).await.unwrap();
    assert_eq!(
        results.hits.len(),
        1,
        "Russian stemmer: field-qualified search should find 1 doc"
    );
}

/// Cross-segment top-k threshold propagation must not change results.
///
/// When a query runs over many segments, each segment seeds its MaxScore
/// pruning from the running global k-th score (`SharedThreshold`) and raises
/// that floor once it fills its own heap. This is a performance optimization
/// and MUST be exact: the seeded top-k over many segments has to equal the
/// exhaustive (un-pruned) top-k. A regression that seeded too aggressively
/// (e.g. from partial per-field scores) would drop or reorder a valid hit and
/// trip this test.
#[tokio::test]
async fn test_cross_segment_threshold_topk_matches_exhaustive() {
    use crate::query::{BooleanQuery, TermQuery};

    // Single text field so a multi-term OR hits the single-field MaxScore path
    // that consumes the cross-segment threshold seed.
    let mut schema_builder = SchemaBuilder::default();
    let content = schema_builder.add_text_field("content", true, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    // A commit per batch + no merging => many small segments kept separate, so
    // the cross-segment threshold is actually exercised.
    let config = IndexConfig {
        max_indexing_memory_bytes: 1024,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Varied term frequencies so BM25 scores spread out and pruning has teeth.
    let terms = ["alpha", "beta", "gamma", "delta"];
    let mut n_docs = 0u32;
    for batch in 0..12 {
        for i in 0..8 {
            let mut text = String::new();
            let repeats = (i % 4) + 1;
            for _ in 0..repeats {
                text.push_str(terms[(i + batch) % terms.len()]);
                text.push(' ');
            }
            if i % 2 == 0 {
                text.push_str("alpha ");
            }
            if i % 3 == 0 {
                text.push_str("beta beta ");
            }
            let mut doc = Document::new();
            doc.add_text(content, text.trim());
            writer.add_document(doc).unwrap();
            n_docs += 1;
        }
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), n_docs);
    assert!(
        index.segment_readers().await.unwrap().len() >= 3,
        "test needs multiple segments to exercise the cross-segment threshold"
    );

    let query = BooleanQuery::new()
        .should(TermQuery::text(content, "alpha"))
        .should(TermQuery::text(content, "beta"))
        .should(TermQuery::text(content, "gamma"));

    // Ground truth: fetching all matches never fills a per-segment top-k of
    // size n_docs, so no pruning and no cross-segment seeding occur.
    let exhaustive = index.search(&query, n_docs as usize).await.unwrap();
    assert!(
        exhaustive.hits.len() > 5,
        "need enough matches for the comparison to be meaningful"
    );

    // Seeded top-k for several small k must equal the exhaustive prefix exactly.
    for k in [1usize, 3, 5, 10] {
        let topk = index.search(&query, k).await.unwrap();
        let expected = &exhaustive.hits[..k.min(exhaustive.hits.len())];
        assert_eq!(
            topk.hits.len(),
            expected.len(),
            "k={k}: cross-segment pruning changed the result count"
        );
        // Compare the score *sequence*, not doc identity: seeding prunes at the
        // exact k-th score, so which of several docs tied at the boundary is
        // returned may differ from the exhaustive run — that's a valid top-k
        // either way. A dropped/mis-scored hit still changes the sequence.
        for (got, want) in topk.hits.iter().zip(expected.iter()) {
            assert!(
                (got.score - want.score).abs() < 1e-5,
                "k={k}: top-k score sequence diverged from exhaustive ({} vs {}) => \
                 threshold pruning dropped a valid hit",
                got.score,
                want.score
            );
        }
    }
}

/// One `text<stem(by: languages, default: simple)>` field holds documents of
/// several languages: each document is stemmed with the language(s) listed in
/// its own `languages` values, and queries are stemmed by the hint they carry.
#[tokio::test]
async fn dynamic_stemmer_indexes_per_document_language() {
    use crate::query::{BooleanQuery, TermQuery};
    use crate::tokenizer::{DynamicStemmer, Tokenizer};

    let mut schema_builder = SchemaBuilder::default();
    let languages =
        schema_builder.add_text_field_with_tokenizer("languages", false, true, "raw_ci");
    let content = schema_builder.add_text_field_with_tokenizer(
        "content",
        true,
        true,
        "stem(by: languages, default: simple)",
    );
    let schema = schema_builder.build();
    assert_eq!(schema.tokenizer_hint_field(content), Some(languages));

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Doc 0: English
    let mut doc = Document::new();
    doc.add_text(languages, "en");
    doc.add_text(content, "running foxes");
    writer.add_document(doc).unwrap();
    // Doc 1: Russian
    let mut doc = Document::new();
    doc.add_text(languages, "ru");
    doc.add_text(content, "бегущие собаки");
    writer.add_document(doc).unwrap();
    // Doc 2: Russian body with an English abstract, tagged with both languages
    let mut doc = Document::new();
    doc.add_text(languages, "ru");
    doc.add_text(languages, "en");
    doc.add_text(content, "бегущие собаки running foxes");
    writer.add_document(doc).unwrap();
    // Doc 3: untagged → simple tokenizer, exact forms only
    let mut doc = Document::new();
    doc.add_text(content, "running foxes");
    writer.add_document(doc).unwrap();

    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    // Emulate the server: stem the query with the field's tokenizer + hint.
    let stemmer = DynamicStemmer::new(None);
    let query_for = |text: &str, hint: Option<&str>| {
        let mut bq = BooleanQuery::new();
        for token in Tokenizer::tokenize_hinted(&stemmer, text, hint) {
            bq = bq.should(TermQuery::text(content, &token.text));
        }
        bq
    };
    let hits = |response: crate::query::SearchResponse| {
        let mut ids: Vec<u32> = response.hits.iter().map(|h| h.address.doc_id).collect();
        ids.sort_unstable();
        ids
    };

    // English inflection with an English hint: the English doc and the
    // bilingual doc (its Latin tokens were stemmed with en), never the
    // untagged doc (indexed as the exact form "foxes").
    let response = index
        .search(&query_for("fox", Some("en")), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0, 2]);

    // Russian inflection with a Russian hint: both Russian-tagged docs.
    let response = index
        .search(&query_for("собака", Some("ru")), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![1, 2]);

    // No hint: exact tokens only, which is what the untagged doc indexed.
    let response = index.search(&query_for("foxes", None), 10).await.unwrap();
    assert_eq!(hits(response), vec![3]);

    // A Russian hint does not stem Latin tokens, so "foxes" stays exact and
    // still reaches the untagged document.
    let response = index
        .search(&query_for("foxes", Some("ru")), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![3]);
}

/// Plain (non-chunked) text fields persist per-document lengths, so BM25
/// normalises by the real field length instead of `tf`, and MaxScore prunes
/// with block bounds that use each block's minimum length while staying
/// rank-safe against a brute-force evaluation.
#[tokio::test]
async fn plain_text_fields_score_with_persisted_lengths_and_prune_safely() {
    use crate::query::{BooleanQuery, TermQuery, bm25_idf, bm25_score};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let schema = schema_builder.build();

    // Deterministic corpus: term frequencies of three terms plus filler, with
    // field lengths spread between 1 and ~300 tokens.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // 4000 docs: every term spans dozens of blocks, so block and superblock
    // skips both fire against the brute-force ranking.
    let n = 4000usize;
    let mut tfs: Vec<[u32; 3]> = Vec::with_capacity(n);
    let mut lens: Vec<u32> = Vec::with_capacity(n);
    let mut texts: Vec<String> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut counts = [(rng() % 4) as u32, (rng() % 3) as u32, (rng() % 2) as u32];
        let filler = (rng() % 300) as u32;
        if counts.iter().sum::<u32>() + filler == 0 {
            counts[0] = 1;
        }
        let mut words: Vec<&str> = Vec::new();
        words.extend(std::iter::repeat_n("alpha", counts[0] as usize));
        words.extend(std::iter::repeat_n("beta", counts[1] as usize));
        words.extend(std::iter::repeat_n("gamma", counts[2] as usize));
        words.extend(std::iter::repeat_n("zzz", filler as usize));
        tfs.push(counts);
        lens.push(words.len() as u32);
        texts.push(words.join(" "));
    }

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    for text in &texts {
        let mut doc = Document::new();
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    // Brute-force BM25 with real lengths.
    let avg = lens.iter().map(|&l| l as f32).sum::<f32>() / n as f32;
    let terms = ["alpha", "beta", "gamma"];
    let idf: Vec<f32> = (0..3)
        .map(|t| {
            let df = tfs.iter().filter(|c| c[t] > 0).count() as f32;
            bm25_idf(df, n as f32)
        })
        .collect();
    let component = |doc: usize, t: usize| -> f32 {
        let tf = tfs[doc][t] as f32;
        if tf == 0.0 {
            0.0
        } else {
            bm25_score(tf, idf[t], lens[doc] as f32, avg)
        }
    };
    let expected: Vec<f32> = (0..n)
        .map(|d| (0..3).map(|t| component(d, t)).sum())
        .collect();

    let mut query = BooleanQuery::new();
    for term in terms {
        query = query.should(TermQuery::text(body, term));
    }
    let response = index.search(&query, 10).await.unwrap();
    assert_eq!(response.hits.len(), 10);
    let mut best: Vec<f32> = expected.clone();
    best.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for (hit, want) in response.hits.iter().zip(&best) {
        let doc = hit.address.doc_id as usize;
        assert!(
            (hit.score - expected[doc]).abs() < 1e-3,
            "doc {doc}: got {} expected {}",
            hit.score,
            expected[doc]
        );
        assert!(
            (hit.score - want).abs() < 1e-3,
            "rank-safety: got {} expected {want}",
            hit.score
        );
    }

    // A single term goes through TermScorer: same length-normalised scores.
    let response = index
        .search(&TermQuery::text(body, "alpha"), 10)
        .await
        .unwrap();
    for hit in &response.hits {
        let doc = hit.address.doc_id as usize;
        assert!((hit.score - component(doc, 0)).abs() < 1e-3, "doc {doc}");
    }

    // Length matters: at equal tf the shorter field scores higher.
    let short = (0..n)
        .filter(|&d| tfs[d] == [1, 0, 0])
        .min_by_key(|&d| lens[d])
        .unwrap();
    let long = (0..n)
        .filter(|&d| tfs[d] == [1, 0, 0])
        .max_by_key(|&d| lens[d])
        .unwrap();
    assert!(lens[short] < lens[long]);
    assert!(expected[short] > expected[long]);
}

/// Length columns of plain fields are concatenated on merge (with zero fill
/// for segments without the field), so scores after a merge equal the
/// single-segment scores.
#[tokio::test]
async fn plain_field_lengths_survive_merges() {
    use crate::query::{TermQuery, bm25_idf, bm25_score};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let title = schema_builder.add_text_field_with_tokenizer("title", true, false, "simple");
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    // Segment 1: only `title` has values (no `body` length column).
    for text in ["needle", "needle haystack"] {
        let mut doc = Document::new();
        doc.add_text(title, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    // Segment 2: `body` with very different lengths.
    let long_body = format!("needle {}", "word ".repeat(120));
    for text in ["needle", long_body.as_str(), "other"] {
        let mut doc = Document::new();
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(
        searcher.segment_readers().len(),
        1,
        "force_merge must leave one segment"
    );

    // body: two docs with the term, lengths 1 and 121; avg over docs with the field.
    let avg_body = (1.0 + 121.0 + 1.0) / 3.0;
    let idf_body = bm25_idf(2.0, 5.0);
    let response = index
        .search(&TermQuery::text(body, "needle"), 10)
        .await
        .unwrap();
    assert_eq!(response.hits.len(), 2);
    let mut scores: Vec<f32> = response.hits.iter().map(|h| h.score).collect();
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let expected_short = bm25_score(1.0, idf_body, 1.0, avg_body);
    let expected_long = bm25_score(1.0, idf_body, 121.0, avg_body);
    assert!((scores[0] - expected_short).abs() < 1e-3, "{scores:?}");
    assert!((scores[1] - expected_long).abs() < 1e-3, "{scores:?}");
    assert!(expected_short > expected_long);
}

/// Plain fields: a MUST phrase plus a fast-field filter run as a bitset
/// predicate inside text MaxScore, and documents matching only the MUST
/// clauses fill the tail with score 0.
#[tokio::test]
async fn filtered_text_maxscore_keeps_boolean_semantics_on_plain_fields() {
    use crate::dsl::PositionMode;
    use crate::query::{BooleanQuery, PhraseQuery, TermQuery};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    schema_builder.set_positions(body, PositionMode::TokenPosition);
    let kind = schema_builder.add_text_field_with_tokenizer("kind", true, true, "raw_ci");
    schema_builder.set_fast(kind, true);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    for (text, k) in [
        ("solid state physics review", "a"),
        ("state of the solid art", "b"),
        ("solid state devices", "b"),
        ("physics review", "a"),
    ] {
        let mut doc = Document::new();
        doc.add_text(body, text);
        doc.add_text(kind, k);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let phrase = PhraseQuery::new(body, vec![b"solid".to_vec(), b"state".to_vec()]);
    let by_doc = |response: &crate::query::SearchResponse| -> Vec<(u32, f32)> {
        let mut hits: Vec<(u32, f32)> = response
            .hits
            .iter()
            .map(|h| (h.address.doc_id, h.score))
            .collect();
        hits.sort_by_key(|(d, _)| *d);
        hits
    };

    let query = BooleanQuery::new()
        .must(phrase.clone())
        .should(TermQuery::text(body, "physics"))
        .should(TermQuery::text(body, "review"));
    let hits = by_doc(&index.search(&query, 10).await.unwrap());
    assert_eq!(hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(), vec![0, 2]);
    assert!(hits[0].1 > 0.0, "{hits:?}");
    assert_eq!(hits[1].1, 0.0, "doc 2 matches only the phrase: {hits:?}");

    let query = BooleanQuery::new()
        .must(phrase)
        .must(TermQuery::text(kind, "b"))
        .should(TermQuery::text(body, "physics"))
        .should(TermQuery::text(body, "review"));
    let hits = by_doc(&index.search(&query, 10).await.unwrap());
    assert_eq!(hits, vec![(2, 0.0)]);

    // A tight limit keeps only scored documents.
    let query = BooleanQuery::new()
        .must(TermQuery::text(body, "state"))
        .should(TermQuery::text(body, "physics"))
        .should(TermQuery::text(body, "review"));
    let hits = by_doc(&index.search(&query, 1).await.unwrap());
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0);
    assert!(hits[0].1 > 0.0);
}

/// A phrase scores by its own frequency (occurrences of the phrase in the
/// unit) with the summed idf of its terms and the unit's real length.
#[tokio::test]
async fn phrase_scores_by_phrase_frequency() {
    use crate::dsl::PositionMode;
    use crate::query::{PhraseQuery, bm25_idf, bm25_score};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    schema_builder.set_positions(body, PositionMode::TokenPosition);
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    // Same length, same term frequencies: doc 0 has the phrase twice, doc 1
    // once (its other "brown" and "fox" are apart), doc 2 not at all.
    let texts = [
        "brown fox brown fox pad",
        "brown fox pad fox brown",
        "brown pad fox pad pad",
    ];
    for text in texts {
        let mut doc = Document::new();
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    let phrase = PhraseQuery::new(body, vec![b"brown".to_vec(), b"fox".to_vec()]);
    let response = index.search(&phrase, 10).await.unwrap();
    let mut hits: Vec<(u32, f32)> = response
        .hits
        .iter()
        .map(|h| (h.address.doc_id, h.score))
        .collect();
    hits.sort_by_key(|(d, _)| *d);
    assert_eq!(hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(), vec![0, 1]);
    assert!(hits[0].1 > hits[1].1, "{hits:?}");

    let idf = bm25_idf(3.0, 3.0) + bm25_idf(3.0, 3.0);
    let avg = 5.0;
    assert!(
        (hits[0].1 - bm25_score(2.0, idf, 5.0, avg)).abs() < 1e-4,
        "{hits:?}"
    );
    assert!(
        (hits[1].1 - bm25_score(1.0, idf, 5.0, avg)).abs() < 1e-4,
        "{hits:?}"
    );
}

/// Block-Max MaxScore regression: an essential cursor that fails the
/// block-max check at the minimum document must not skip past a document
/// another essential cursor still holds inside that block, or the document
/// loses the skipped cursor's contribution and a true top-k hit is dropped.
///
/// Shape: "cc" is in every document (non-essential). "aa" and "bb" are rare
/// with one tf-30 document each, so both stay essential once the threshold
/// is set by docs 0 and 1 (cc + aa + bb, tf 1 each). At doc 2 only "aa" is
/// at the minimum and its first block (tf 1) cannot beat the threshold; "bb"
/// waits at doc 100, inside that block, with tf 6: the true top hit.
#[tokio::test]
async fn block_max_skip_never_jumps_over_another_essential_cursor() {
    use crate::query::{BooleanQuery, TermQuery, bm25_idf, bm25_score};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    const LEN: usize = 32;
    let make = |aa: usize, bb: usize| -> String {
        let mut words: Vec<&str> = vec!["cc"];
        words.extend(std::iter::repeat_n("aa", aa));
        words.extend(std::iter::repeat_n("bb", bb));
        words.extend(std::iter::repeat_n("ff", LEN - 1 - aa - bb));
        words.join(" ")
    };
    // Segment doc ids follow insertion order: index i below is doc i. Both
    // rare terms have df 129 (equal idf) and one tf-30 document, so both are
    // essential once docs 0 and 1 set the threshold; "aa" fills docs 0..=127
    // (one block) and "bb" waits at doc 100 with tf 6.
    let total = 4000usize;
    let mut tfs: Vec<(u32, u32)> = vec![(0, 0); total];
    tfs[..=127].fill((1, 0));
    tfs[0] = (1, 1);
    tfs[1] = (1, 1);
    tfs[100] = (1, 6);
    tfs[150] = (30, 0);
    tfs[201] = (0, 30);
    tfs[2000..2125].fill((0, 1));
    for &(aa, bb) in &tfs {
        let mut doc = Document::new();
        doc.add_text(body, make(aa as usize, bb as usize));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    let n = tfs.len() as f32;
    let idf_c = bm25_idf(n, n);
    let idf_a = bm25_idf(tfs.iter().filter(|(a, _)| *a > 0).count() as f32, n);
    let idf_b = bm25_idf(tfs.iter().filter(|(_, b)| *b > 0).count() as f32, n);
    let score = |doc: usize| {
        let (a, b) = tfs[doc];
        let mut s = bm25_score(1.0, idf_c, LEN as f32, LEN as f32);
        if a > 0 {
            s += bm25_score(a as f32, idf_a, LEN as f32, LEN as f32);
        }
        if b > 0 {
            s += bm25_score(b as f32, idf_b, LEN as f32, LEN as f32);
        }
        s
    };
    let mut expected: Vec<(u32, f32)> = (0..tfs.len()).map(|d| (d as u32, score(d))).collect();
    expected.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
    assert_eq!(expected[0].0, 100, "{:?}", &expected[..4]);

    let query = BooleanQuery::new()
        .should(TermQuery::text(body, "cc"))
        .should(TermQuery::text(body, "aa"))
        .should(TermQuery::text(body, "bb"));
    let response = index.search(&query, 2).await.unwrap();
    let got: Vec<u32> = response.hits.iter().map(|h| h.address.doc_id).collect();
    assert_eq!(
        got.first(),
        Some(&100),
        "got {got:?}, expected {:?}",
        &expected[..4]
    );
    for (hit, (_, s)) in response.hits.iter().zip(&expected) {
        assert!((hit.score - s).abs() < 1e-4, "{} vs {s}", hit.score);
    }
}

/// Per-field BM25 parameters reach every scoring path: with `b: 0` the
/// field length no longer matters, and `k1` changes the saturation curve.
#[tokio::test]
async fn per_field_bm25_parameters_apply_to_scores() {
    use crate::dsl::sdl::parse_sdl;
    use crate::query::{Bm25Params, BooleanQuery, TermQuery};

    let schema = parse_sdl(
        "index i {\n  field flat: text<simple> [indexed<b: 0.0>]\n  field body: text<simple> [indexed<k1: 0.5, b: 0.75>]\n}",
    )
    .unwrap()[0]
        .to_schema();
    let flat = schema.get_field("flat").unwrap();
    let body = schema.get_field("body").unwrap();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    let long = format!("needle {}", "pad ".repeat(60));
    for text in ["needle", long.as_str()] {
        let mut doc = Document::new();
        doc.add_text(flat, text);
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    // b = 0: both documents score the same despite the length difference,
    // through the single-term scorer and through the MaxScore executor.
    let scores = |response: crate::query::SearchResponse| {
        let mut v: Vec<(u32, f32)> = response
            .hits
            .iter()
            .map(|h| (h.address.doc_id, h.score))
            .collect();
        v.sort_by_key(|(d, _)| *d);
        v
    };
    let single = scores(
        index
            .search(&TermQuery::text(flat, "needle"), 10)
            .await
            .unwrap(),
    );
    assert_eq!(single.len(), 2);
    assert!((single[0].1 - single[1].1).abs() < 1e-6, "{single:?}");
    let query = BooleanQuery::new()
        .should(TermQuery::text(flat, "needle"))
        .should(TermQuery::text(flat, "pad"));
    let both = scores(index.search(&query, 10).await.unwrap());
    let needle_only = single[0].1;
    assert!((both[0].1 - needle_only).abs() < 1e-6, "{both:?}");

    // k1 = 0.5 on `body`: the short document's score equals BM25 with those
    // parameters, not the defaults.
    let params = Bm25Params::for_field(&schema, body);
    assert_eq!((params.k1, params.b), (0.5, 0.75));
    let hits = scores(
        index
            .search(&TermQuery::text(body, "needle"), 10)
            .await
            .unwrap(),
    );
    let idf = crate::query::bm25_idf(2.0, 2.0);
    let avg = (1.0 + 61.0) / 2.0;
    assert!(
        (hits[0].1 - params.score(1.0, idf, 1.0, avg)).abs() < 1e-5,
        "{hits:?}"
    );
    assert!((hits[0].1 - Bm25Params::default().score(1.0, idf, 1.0, avg)).abs() > 1e-3);
}

/// Proximity rescoring: with equal BM25 scores, adjacent query terms
/// (ordered window) outrank terms merely within the window, which outrank
/// distant ones; with the stage off all three tie. Plain and chunked fields.
#[tokio::test]
async fn proximity_rescoring_prefers_adjacent_terms() {
    use crate::dsl::PositionMode;
    use crate::query::{BooleanQuery, ProximityConfig, TermQuery};

    let mut schema_builder = SchemaBuilder::default();
    let languages =
        schema_builder.add_text_field_with_tokenizer("languages", false, true, "raw_ci");
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    schema_builder.set_positions(body, PositionMode::TokenPosition);
    let content = schema_builder.add_text_field_with_tokenizer(
        "content",
        true,
        false,
        "stem(by: languages, default: simple)",
    );
    schema_builder.set_chunked(content, true);
    schema_builder.set_positions(content, PositionMode::TokenPosition);
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    // Same length, same term frequencies; only the distance differs.
    // doc 0: adjacent (ordered), doc 1: three apart (unordered window),
    // doc 2: reversed and adjacent (unordered only), doc 3: far apart.
    let texts = [
        "alpha beta p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12",
        "alpha p1 p2 beta p3 p4 p5 p6 p7 p8 p9 p10 p11 p12",
        "beta alpha p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12",
        "alpha p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 beta",
    ];
    for text in texts {
        let mut doc = Document::new();
        doc.add_text(languages, "en");
        doc.add_text(body, text);
        doc.add_text(content, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let scores = |response: crate::query::SearchResponse| {
        let mut v: Vec<(u32, f32)> = response
            .hits
            .iter()
            .map(|h| (h.address.doc_id, h.score))
            .collect();
        v.sort_by_key(|(d, _)| *d);
        v.into_iter().map(|(_, s)| s).collect::<Vec<f32>>()
    };
    for field in [body, content] {
        let plain = BooleanQuery::new()
            .should(TermQuery::text(field, "alpha"))
            .should(TermQuery::text(field, "beta"));
        let base = scores(index.search(&plain, 10).await.unwrap());
        assert_eq!(base.len(), 4);
        assert!(
            base.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-5),
            "{base:?}"
        );

        let near = plain.clone().with_proximity(ProximityConfig::new(1.0, 8));
        let got = scores(index.search(&near, 10).await.unwrap());
        assert!(got[0] > got[1], "{got:?}");
        assert!(got[1] > got[3], "{got:?}");
        assert!((got[1] - got[2]).abs() < 1e-5, "{got:?}");
        assert!(
            (got[3] - base[3]).abs() < 1e-5,
            "far apart: no bonus {got:?}"
        );
        // The limit is honoured after rescoring.
        let top = index.search(&near, 1).await.unwrap();
        assert_eq!(top.hits.len(), 1);
        assert_eq!(top.hits[0].address.doc_id, 0);

        // A filter combined with the rescored terms keeps both effects.
        let filtered = BooleanQuery::new()
            .must(TermQuery::text(field, "p12"))
            .should(TermQuery::text(field, "alpha"))
            .should(TermQuery::text(field, "beta"))
            .with_proximity(ProximityConfig::new(1.0, 8));
        let got = scores(index.search(&filtered, 10).await.unwrap());
        assert_eq!(got.len(), 4);
        assert!(got[0] > got[3], "{got:?}");
    }
}

/// Long-query cap and approximate mode: `max_terms` keeps the rarest terms
/// (the result equals the query over those terms alone), and a heap factor
/// above one returns a subset of the exact top-k with exact scores.
#[tokio::test]
async fn text_maxscore_honours_max_terms_and_heap_factor() {
    use crate::query::{BooleanQuery, TermQuery};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    let mut seed = 0x7A3B_11C9_55D2_0F01u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // "common" is in almost every document, "rare" in a few, "mid" between.
    for _ in 0..3000 {
        let mut words = vec!["common"; (rng() % 3 + 1) as usize];
        if rng() % 3 == 0 {
            words.push("mid");
        }
        if rng() % 40 == 0 {
            words.push("rare");
        }
        words.extend(std::iter::repeat_n("pad", (rng() % 40) as usize));
        let mut doc = Document::new();
        doc.add_text(body, words.join(" "));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let ids = |response: crate::query::SearchResponse| -> Vec<(u32, i64)> {
        response
            .hits
            .iter()
            .map(|h| (h.address.doc_id, (h.score * 1e4).round() as i64))
            .collect()
    };

    // max_terms 1 keeps "rare" only.
    let capped = BooleanQuery::new()
        .should(TermQuery::text(body, "common"))
        .should(TermQuery::text(body, "mid"))
        .should(TermQuery::text(body, "rare"))
        .with_max_terms(1);
    let only_rare = BooleanQuery::new().should(TermQuery::text(body, "rare"));
    assert_eq!(
        ids(index.search(&capped, 20).await.unwrap()),
        ids(index.search(&only_rare, 20).await.unwrap())
    );

    // Approximate mode: a subset of the exact top-k, scores unchanged.
    let full = BooleanQuery::new()
        .should(TermQuery::text(body, "common"))
        .should(TermQuery::text(body, "mid"))
        .should(TermQuery::text(body, "rare"));
    let exact = ids(index.search(&full, 30).await.unwrap());
    let approx = ids(index
        .search(&full.clone().with_text_heap_factor(1.5), 30)
        .await
        .unwrap());
    assert!(!approx.is_empty());
    for hit in &approx {
        assert!(exact.contains(hit), "{hit:?} not in exact top-30");
    }
}

/// Anytime mode: a deadline that has already passed makes the text
/// executor stop after its first budget check and flag the response
/// truncated; a generous deadline changes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_maxscore_stops_at_the_deadline() {
    use crate::query::{BooleanQuery, TermQuery};
    use std::time::{Duration, Instant};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    // Every document matches both terms: the executor cannot skip, so the
    // loop runs once per document and crosses the 4096-iteration check.
    for i in 0..20_000u32 {
        let make = || {
            let mut doc = Document::new();
            let padding = " pad".repeat((i % 7) as usize);
            doc.add_text(body, format!("alpha beta{padding}"));
            doc
        };
        // The writer queue is bounded; wait for the workers to drain it.
        while let Err(crate::Error::QueueFull) = writer.add_document(make()) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = BooleanQuery::new()
        .should(TermQuery::text(body, "alpha"))
        .should(TermQuery::text(body, "beta"));
    let ids = |results: &[crate::query::SearchResult]| -> Vec<(u32, i64)> {
        results
            .iter()
            .map(|r| (r.doc_id, (r.score * 1e4).round() as i64))
            .collect()
    };

    let (exact, exact_seen) = searcher.search_with_positions(&query, 10).await.unwrap();
    assert_eq!(exact.len(), 10);

    let (unhurried, seen, truncated) = searcher
        .search_with_positions_budgeted(&query, 10, Some(Instant::now() + Duration::from_secs(600)))
        .await
        .unwrap();
    assert!(!truncated);
    assert_eq!(seen, exact_seen);
    assert_eq!(ids(&unhurried), ids(&exact));

    let (partial, _, truncated) = searcher
        .search_with_positions_budgeted(&query, 10, Some(Instant::now() - Duration::from_secs(1)))
        .await
        .unwrap();
    assert!(truncated, "an expired deadline must flag the response");
    assert!(partial.len() <= 10);
    assert!(!partial.is_empty(), "best-so-far results are returned");
    // Best-so-far stays a valid ranking: scores are exact and descending.
    assert!(partial.windows(2).all(|w| w[0].score >= w[1].score));
}

/// BM25 IDF and average length come from the whole searcher, not the
/// segment: the same document scores identically in a small and a large
/// segment, and identically to the force-merged index.
#[tokio::test]
async fn text_scores_use_searcher_wide_statistics_across_segments() {
    use crate::query::{BooleanQuery, TermQuery};
    use std::collections::BTreeMap;

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let n = schema_builder.add_u64_field("n", true, true);
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    // Segment 1: two documents, "needle" is in both (local idf ~ 0).
    for (i, text) in [(1u64, "needle haystack"), (2, "needle")] {
        let mut doc = Document::new();
        doc.add_text(body, text);
        doc.add_u64(n, i);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    // Segment 2: the same first document among 300 long "haystack" documents.
    let mut doc = Document::new();
    doc.add_text(body, "needle haystack");
    doc.add_u64(n, 3);
    writer.add_document(doc).unwrap();
    for i in 0..300u64 {
        let mut doc = Document::new();
        doc.add_text(body, "haystack ".repeat(12));
        doc.add_u64(n, 100 + i);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    drop(writer);

    async fn by_n(
        index: &Index<RamDirectory>,
        query: &dyn crate::query::Query,
        n: crate::Field,
    ) -> BTreeMap<u64, i64> {
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let (results, _) = searcher.search_with_count(query, 400).await.unwrap();
        let mut out = BTreeMap::new();
        for result in results {
            let doc = searcher
                .doc(result.segment_id, result.doc_id)
                .await
                .unwrap()
                .unwrap();
            let key = doc.get_first(n).unwrap().as_u64().unwrap();
            out.insert(key, (result.score * 1e4).round() as i64);
        }
        out
    }

    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    assert_eq!(
        index
            .reader()
            .await
            .unwrap()
            .searcher()
            .await
            .unwrap()
            .segment_readers()
            .len(),
        2
    );
    let query = BooleanQuery::new()
        .should(TermQuery::text(body, "needle"))
        .should(TermQuery::text(body, "haystack"));
    let scores = by_n(&index, &query, n).await;
    assert_eq!(
        scores[&1], scores[&3],
        "identical documents in different segments must score alike: {scores:?}"
    );
    let single = by_n(&index, &TermQuery::text(body, "needle"), n).await;
    assert_eq!(single[&1], single[&3], "{single:?}");

    // The merged index scores exactly the same.
    let mut writer = IndexWriter::open(dir.clone(), config.clone())
        .await
        .unwrap();
    writer.force_merge().await.unwrap();
    drop(writer);
    let merged = Index::open(dir, config).await.unwrap();
    assert_eq!(by_n(&merged, &query, n).await, scores);
    assert_eq!(
        by_n(&merged, &TermQuery::text(body, "needle"), n).await,
        single
    );
}

/// A boosted term clause scores like the same term repeated `boost` times
/// (query term frequency), on the text MaxScore path.
#[tokio::test]
async fn boosted_term_scores_like_a_repeated_term() {
    use crate::query::{BooleanQuery, BoostQuery, TermQuery};

    let mut schema_builder = SchemaBuilder::default();
    let body = schema_builder.add_text_field_with_tokenizer("body", true, false, "simple");
    let schema = schema_builder.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    for i in 0..500u32 {
        let mut doc = Document::new();
        let text = match i % 5 {
            0 => "needle haystack".to_string(),
            1 => format!("needle {}", "pad ".repeat(i as usize % 17)),
            2 => "haystack haystack".to_string(),
            _ => format!("needle needle {}", "haystack ".repeat(i as usize % 3)),
        };
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    // Every document is returned, ordered by id: tie order between equal
    // scores is not part of the contract.
    let ids = |response: crate::query::SearchResponse| -> Vec<(u32, i64)> {
        let mut hits: Vec<(u32, i64)> = response
            .hits
            .iter()
            .map(|h| (h.address.doc_id, (h.score * 1e3).round() as i64))
            .collect();
        hits.sort_unstable();
        hits
    };
    let repeated = BooleanQuery::new()
        .should(TermQuery::text(body, "needle"))
        .should(TermQuery::text(body, "needle"))
        .should(TermQuery::text(body, "needle"))
        .should(TermQuery::text(body, "haystack"));
    let boosted = BooleanQuery::new()
        .should(BoostQuery::new(TermQuery::text(body, "needle"), 3.0))
        .should(TermQuery::text(body, "haystack"));
    let repeated_hits = ids(index.search(&repeated, 500).await.unwrap());
    assert_eq!(repeated_hits.len(), 500);
    assert_eq!(
        repeated_hits,
        ids(index.search(&boosted, 500).await.unwrap())
    );
}

/// Stop words dropped at index time leave their positions behind, so a
/// phrase keeps the original word distances: `"quantum of the art"` is
/// `quantum@0 art@3` on both sides and never matches `quantum art`.
#[tokio::test]
async fn phrase_query_keeps_the_gaps_of_dropped_stop_words() {
    use crate::dsl::PositionMode;
    use crate::query::PhraseQuery;
    use crate::tokenizer::{DynamicStemmer, Tokenizer};

    let mut schema_builder = SchemaBuilder::default();
    let languages =
        schema_builder.add_text_field_with_tokenizer("languages", false, true, "raw_ci");
    let content = schema_builder.add_text_field_with_tokenizer(
        "content",
        true,
        true,
        "stem(by: languages, default: simple, stop_words: true)",
    );
    schema_builder.set_positions(content, PositionMode::TokenPosition);
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();
    for text in ["quantum of the art", "quantum art", "the art of quantum"] {
        let mut doc = Document::new();
        doc.add_text(languages, "en");
        doc.add_text(content, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    let stemmer = DynamicStemmer::new(None).with_stop_words(true);
    let phrase = |text: &str, slop| {
        let terms = Tokenizer::tokenize_hinted(&stemmer, text, Some("en"))
            .into_iter()
            .map(|t| (t.position, t.text.into_bytes()))
            .collect();
        PhraseQuery::with_offsets(content, terms).with_slop(slop)
    };
    let hits = |response: crate::query::SearchResponse| {
        let mut ids: Vec<u32> = response.hits.iter().map(|h| h.address.doc_id).collect();
        ids.sort_unstable();
        ids
    };

    let response = index
        .search(&phrase("quantum of the art", 0), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0]);
    let response = index.search(&phrase("quantum art", 0), 10).await.unwrap();
    assert_eq!(hits(response), vec![1]);
    // "art of quantum" is art@0 quantum@2 and matches art@1 quantum@3.
    let response = index
        .search(&phrase("art of quantum", 0), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![2]);
    let response = index.search(&phrase("art quantum", 0), 10).await.unwrap();
    assert_eq!(hits(response), Vec::<u32>::new());
    // Slop is measured against the gapped expectation.
    let response = index.search(&phrase("quantum art", 2), 10).await.unwrap();
    assert_eq!(hits(response), vec![0, 1]);
    // The removed words are not proven: a different filler still matches.
    let response = index
        .search(&phrase("quantum in an art", 0), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0]);
    // Field lengths count only surviving tokens.
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let avg: f32 = searcher.segment_readers()[0].avg_field_len(content);
    assert!((avg - 2.0).abs() < 1e-3, "avg field length {avg}");
}

/// Wire-level phrase semantics: consecutive stemmed terms on a field with
/// token positions; slop widens the window; a field without positions
/// degrades to a MUST of the terms.
#[tokio::test]
async fn phrase_query_matches_consecutive_stemmed_terms() {
    use crate::dsl::PositionMode;
    use crate::query::PhraseQuery;
    use crate::tokenizer::{DynamicStemmer, Tokenizer};

    let mut schema_builder = SchemaBuilder::default();
    let languages =
        schema_builder.add_text_field_with_tokenizer("languages", false, true, "raw_ci");
    let content = schema_builder.add_text_field_with_tokenizer(
        "content",
        true,
        true,
        "stem(by: languages, default: simple)",
    );
    schema_builder.set_positions(content, PositionMode::TokenPosition);
    let flat = schema_builder.add_text_field_with_tokenizer("flat", true, false, "en_stem");
    let schema = schema_builder.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    for text in [
        "the quick brown fox",
        "the brown quick fox",
        "quick and brown foxes",
    ] {
        let mut doc = Document::new();
        doc.add_text(languages, "en");
        doc.add_text(content, text);
        doc.add_text(flat, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();

    let stemmer = DynamicStemmer::new(None);
    let phrase = |field, text: &str, slop| {
        let terms = Tokenizer::tokenize_hinted(&stemmer, text, Some("en"))
            .into_iter()
            .map(|t| t.text.into_bytes())
            .collect();
        PhraseQuery::new(field, terms).with_slop(slop)
    };
    let hits = |response: crate::query::SearchResponse| {
        let mut ids: Vec<u32> = response.hits.iter().map(|h| h.address.doc_id).collect();
        ids.sort_unstable();
        ids
    };

    // Exact phrase: only the document with the terms in order and adjacent.
    let response = index
        .search(&phrase(content, "quick brown", 0), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0]);

    // Stemmed phrase: "quick brown foxes" → quick brown fox matches doc 0.
    let response = index
        .search(&phrase(content, "Quick Brown Foxes", 0), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0]);

    // Slop 1 allows one intervening token ("quick and brown") but keeps the
    // term order, so "brown quick" still does not match.
    let response = index
        .search(&phrase(content, "quick brown", 1), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0, 2]);

    // Without positions the phrase degrades to an AND of the terms.
    let response = index
        .search(&phrase(flat, "quick brown", 0), 10)
        .await
        .unwrap();
    assert_eq!(hits(response), vec![0, 1, 2]);
}
