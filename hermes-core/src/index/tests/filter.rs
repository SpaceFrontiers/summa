use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};

/// End-to-end test: fast fields through build → merge → search with filters.
///
/// Schema: text (indexed+stored), u64 (fast), text (fast+multi).
/// Indexes 60 docs in 3 batches, commits each, force-merges, then searches
/// with EqU64, RangeU64, standalone InText, and text query + fast filter combo.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_fast_field_filters_through_merge() {
    use crate::directories::MmapDirectory;
    use crate::dsl::SchemaBuilder;
    use crate::query::fast_filter::{FastFieldCondition, FastFieldFilter, FastFieldFilterQuery};
    use std::sync::Arc;

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    // Schema: title(text indexed+stored), price(u64 fast), tags(text fast+multi)
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let price = sb.add_u64_field("price", false, true);
    sb.set_fast(price, true);
    let tags = sb.add_text_field("tags", false, true);
    sb.set_fast(tags, true);
    sb.set_multi(tags, true);
    let schema = sb.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // Batch 1: docs 0..19
    for i in 0u64..20 {
        let mut doc = Document::new();
        doc.add_text(title, format!("product_{} electronics", i));
        doc.add_u64(price, 100 + i * 10); // 100, 110, ..., 290
        doc.add_text(tags, "electronics");
        if i % 2 == 0 {
            doc.add_text(tags, "sale");
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Batch 2: docs 20..39
    for i in 20u64..40 {
        let mut doc = Document::new();
        doc.add_text(title, format!("product_{} clothing", i));
        doc.add_u64(price, 50 + i * 5); // 150, 155, ..., 245
        doc.add_text(tags, "clothing");
        if i % 3 == 0 {
            doc.add_text(tags, "premium");
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Batch 3: docs 40..59
    for i in 40u64..60 {
        let mut doc = Document::new();
        doc.add_text(title, format!("product_{} books", i));
        doc.add_u64(price, 10 + i); // 50, 51, ..., 69
        doc.add_text(tags, "books");
        doc.add_text(tags, "sale");
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Verify multiple segments exist before merge
    let pre_merge_segs = writer.segment_manager.get_segment_ids().await.len();
    assert!(
        pre_merge_segs >= 2,
        "Expected multiple segments, got {}",
        pre_merge_segs
    );

    // Force merge to exercise raw block stacking
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 60);

    // After force merge, should have 1 segment with stacked blocks
    let post_segs = index.segment_readers().await.unwrap();
    assert_eq!(post_segs.len(), 1, "force_merge should produce 1 segment");

    // ── Test 1: EqU64 filter (standalone) ──
    // price == 200 → batch1 i=10 (100+10*10=200) + batch2 i=30 (50+30*5=200) = 2 docs
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: price,
            condition: FastFieldCondition::EqU64(200),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(results.hits.len(), 2, "EqU64(200) should match 2 docs");
    }

    // ── Test 2: RangeU64 filter (standalone) ──
    // price in [50, 69] → docs 40..59 (20 docs with price 50..69)
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: price,
            condition: FastFieldCondition::RangeU64 {
                min: Some(50),
                max: Some(69),
            },
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            20,
            "RangeU64[50,69] should match 20 docs"
        );
    }

    // ── Test 3: InText filter (standalone, multi-value) ──
    // tags IN ["sale"] → batch1 even docs (10) + batch3 all docs (20) = 30
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: tags,
            condition: FastFieldCondition::InText(vec!["sale".to_string()]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(results.hits.len(), 30, "InText(sale) should match 30 docs");
    }

    // ── Test 4: Text query + fast filter combo ──
    // query "electronics" + price range [150, 250]
    // "electronics" matches docs 0..19 (price 100..290)
    // price [150, 250]: 150=doc5, 160=doc6, ..., 250=doc15 → docs 5..15 = 11 docs
    {
        let text_query: Arc<dyn crate::query::Query> = Arc::new(crate::query::TermQuery::new(
            title,
            "electronics".to_string(),
        ));
        let q = FastFieldFilterQuery::new(
            text_query,
            vec![FastFieldFilter {
                field: price,
                condition: FastFieldCondition::RangeU64 {
                    min: Some(150),
                    max: Some(250),
                },
            }],
        );
        let results = index.search(&q, 100).await.unwrap();
        assert!(
            results.hits.len() >= 5,
            "text + RangeU64 filter should match several docs, got {}",
            results.hits.len()
        );
        // All hits should have scores > 0 (from text query)
        for hit in &results.hits {
            assert!(hit.score > 0.0, "filtered hits should have positive scores");
        }
    }

    // ── Test 5: InText with value not in any block dict ──
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: tags,
            condition: FastFieldCondition::InText(vec!["nonexistent_tag".to_string()]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            0,
            "InText(nonexistent) should match 0 docs"
        );
    }

    eprintln!(
        "E2E fast-field filter test passed: 60 docs, 3 batches, force_merge, 5 filter queries"
    );
}

/// Comprehensive fast-field filter tests:
///
/// - **Absent values**: docs that have no value for a fast field
/// - **Multi-value numerics**: InU64 / RangeU64 on multi-value u64
/// - **Multi-value text**: EqText / InText on multi-value text
/// - **AND combination**: two filters on different fields (must both match)
/// - **Missing field**: filter on a field_id that has no fast-field data
/// - **RangeI64**: regression test (zigzag encoding doesn't preserve order)
/// - **Exists**: filter checking existence of text and numeric values
/// - **RangeF64**: floating-point range filter
///
/// Filter semantics: multiple filters are AND-combined; within a multi-value
/// field, any value matching suffices (OR within field).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_fast_field_filter_comprehensive() {
    use crate::directories::MmapDirectory;
    use crate::dsl::SchemaBuilder;
    use crate::query::fast_filter::{FastFieldCondition, FastFieldFilter, FastFieldFilterQuery};

    let tmp_dir = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(tmp_dir.path());

    // Schema with diverse fast-field types
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true); // indexed, stored (for text query)
    let price_u64 = sb.add_u64_field("price", false, true);
    sb.set_fast(price_u64, true);
    let temp_i64 = sb.add_i64_field("temperature", false, true);
    sb.set_fast(temp_i64, true);
    let weight_f64 = sb.add_f64_field("weight", false, true);
    sb.set_fast(weight_f64, true);
    let category = sb.add_text_field("category", false, true); // single-value text fast
    sb.set_fast(category, true);
    let tags = sb.add_text_field("tags", false, true); // multi-value text fast
    sb.set_fast(tags, true);
    sb.set_multi(tags, true);
    let scores = sb.add_u64_field("scores", false, true); // multi-value u64 fast
    sb.set_fast(scores, true);
    sb.set_multi(scores, true);
    let schema = sb.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 4096,
        ..Default::default()
    };

    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
        .await
        .unwrap();

    // ── Index 50 docs in 2 batches ──
    // Batch 1: docs 0..24
    for i in 0u64..25 {
        let mut doc = Document::new();
        doc.add_text(title, format!("item_{}", i));

        // price: set for all docs
        doc.add_u64(price_u64, 100 + i * 10);

        // temperature: set for docs 0..19, ABSENT for docs 20..24
        if i < 20 {
            // Range: -50, -40, ..., 140  (i*10 - 50)
            doc.add_i64(temp_i64, i as i64 * 10 - 50);
        }

        // weight: set for docs 0..15, ABSENT for docs 16..24
        if i < 16 {
            doc.add_f64(weight_f64, 0.5 + i as f64 * 0.25);
        }

        // category: single-value text, ABSENT for doc 24
        if i < 24 {
            let cat = if i < 12 { "electronics" } else { "clothing" };
            doc.add_text(category, cat);
        }

        // tags: multi-value text
        doc.add_text(tags, "all");
        if i % 2 == 0 {
            doc.add_text(tags, "even");
        }
        if i < 10 {
            doc.add_text(tags, "top10");
        }

        // scores: multi-value u64
        doc.add_u64(scores, i);
        doc.add_u64(scores, 1000 - i);
        if i % 5 == 0 {
            doc.add_u64(scores, 9999); // sentinel for multiples of 5
        }

        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Batch 2: docs 25..49
    for i in 25u64..50 {
        let mut doc = Document::new();
        doc.add_text(title, format!("item_{}", i));
        doc.add_u64(price_u64, 100 + i * 10);
        doc.add_i64(temp_i64, i as i64 * 10 - 50);
        doc.add_f64(weight_f64, 0.5 + i as f64 * 0.25);
        doc.add_text(category, if i < 37 { "clothing" } else { "books" });

        doc.add_text(tags, "all");
        if i % 2 == 0 {
            doc.add_text(tags, "even");
        }
        if i >= 40 {
            doc.add_text(tags, "new_arrival");
        }

        doc.add_u64(scores, i);
        doc.add_u64(scores, 1000 - i);
        if i % 5 == 0 {
            doc.add_u64(scores, 9999);
        }

        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    // Force merge so we exercise multi-block stacking
    writer.force_merge().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    assert_eq!(index.num_docs().await.unwrap(), 50);

    // ── Test 1: Absent values — Exists filter on numeric field ──
    // temperature is absent for docs 20..24 (5 docs)
    // Exists should match 45 docs (0..19 + 25..49)
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: temp_i64,
            condition: FastFieldCondition::Exists,
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            45,
            "Exists on numeric temp should match 45 (5 absent), got {}",
            results.hits.len()
        );
    }

    // ── Test 2: Absent values — Exists filter on text field ──
    // category is absent for doc 24 → ordinal = FAST_FIELD_MISSING
    // Exists should match 49 docs
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: category,
            condition: FastFieldCondition::Exists,
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            49,
            "Exists on text category should match 49 (doc 24 is absent), got {}",
            results.hits.len()
        );
    }

    // ── Test 3: EqText on single-value text ──
    // category == "electronics" → docs 0..11 (12 docs)
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: category,
            condition: FastFieldCondition::EqText("electronics".to_string()),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            12,
            "EqText(electronics) should match 12 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 4: Multi-value text — InText on tags ──
    // tags contains "top10" → docs 0..9 (10 docs)
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: tags,
            condition: FastFieldCondition::InText(vec!["top10".to_string()]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            10,
            "InText(top10) should match 10 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 5: Multi-value text — EqText on tags ──
    // tags contains "even" → docs 0,2,4,...,48 = 25 docs
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: tags,
            condition: FastFieldCondition::EqText("even".to_string()),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            25,
            "EqText(even) on multi-value tags should match 25 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 6: Multi-value text — InText with multiple values (OR within field) ──
    // tags IN ["top10", "new_arrival"] → docs 0..9 (top10) ∪ docs 40..49 (new_arrival) = 20
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: tags,
            condition: FastFieldCondition::InText(vec![
                "top10".to_string(),
                "new_arrival".to_string(),
            ]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            20,
            "InText(top10, new_arrival) should match 20 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 7: Multi-value u64 — InU64 ──
    // scores contains 9999 → docs where i%5==0: {0,5,10,15,20,25,30,35,40,45} = 10
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: scores,
            condition: FastFieldCondition::InU64(vec![9999]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            10,
            "InU64(9999) on multi-value scores should match 10 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 8: Multi-value u64 — RangeU64 on multi-value ──
    // scores in [990, 1000] → 1000-i in [990,1000] means i in [0,10]
    //   also i in [990,1000] — no doc has i >= 990, so just docs 0..10 = 11
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: scores,
            condition: FastFieldCondition::RangeU64 {
                min: Some(990),
                max: Some(1000),
            },
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            11,
            "RangeU64[990,1000] on multi-value scores should match 11 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 9: AND combination — price range + tag filter ──
    // price in [200, 300] → i in [10, 20] (price=100+i*10)
    // AND tags contains "even" → i in {10,12,14,16,18,20} = 6
    {
        let q = FastFieldFilterQuery::standalone(vec![
            FastFieldFilter {
                field: price_u64,
                condition: FastFieldCondition::RangeU64 {
                    min: Some(200),
                    max: Some(300),
                },
            },
            FastFieldFilter {
                field: tags,
                condition: FastFieldCondition::EqText("even".to_string()),
            },
        ]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            6,
            "price[200,300] AND tags=even should match 6 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 10: AND combination — three filters ──
    // price in [100, 220] → i in [0, 12]
    // AND category == "electronics" → i in [0, 11]
    // AND tags contains "even" → i in {0,2,4,6,8,10} = 6
    {
        let q = FastFieldFilterQuery::standalone(vec![
            FastFieldFilter {
                field: price_u64,
                condition: FastFieldCondition::RangeU64 {
                    min: Some(100),
                    max: Some(220),
                },
            },
            FastFieldFilter {
                field: category,
                condition: FastFieldCondition::EqText("electronics".to_string()),
            },
            FastFieldFilter {
                field: tags,
                condition: FastFieldCondition::EqText("even".to_string()),
            },
        ]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            6,
            "price[100,220] AND category=electronics AND tags=even should match 6, got {}",
            results.hits.len()
        );
    }

    // ── Test 11: RangeI64 regression ──
    // temperature = i*10 - 50 for docs 0..19 and 25..49
    // Range [-10, 30] → i*10-50 ∈ [-10,30] → i ∈ [4,8] = 5 docs
    // Absent docs (20..24) have FAST_FIELD_MISSING → excluded from all value filters
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: temp_i64,
            condition: FastFieldCondition::RangeI64 {
                min: Some(-10),
                max: Some(30),
            },
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            5,
            "RangeI64[-10,30] should match 5 docs (absent excluded), got {}",
            results.hits.len()
        );
    }

    // ── Test 12: RangeI64 crossing zero (regression: zigzag ordering) ──
    // Range [-50, -1] → i*10-50 ∈ [-50,-1] → i ∈ [0,4]
    //   docs 0,1,2,3,4 from batch1 = 5 docs (temp = -50,-40,-30,-20,-10)
    //   Wait, -10 is NOT in [-50,-1] since -10 >= -1 is false. -10 < -1 ✓
    //   i=0: -50, i=1: -40, i=2: -30, i=3: -20, i=4: -10 → all in [-50,-1] ✓
    //   i=5: 0 → NOT in range
    //   So 5 docs: 0,1,2,3,4
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: temp_i64,
            condition: FastFieldCondition::RangeI64 {
                min: Some(-50),
                max: Some(-1),
            },
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            5,
            "RangeI64[-50,-1] (all negative) should match 5 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 13: RangeF64 ──
    // weight = 0.5 + i*0.25 for docs 0..15 (batch1) and 25..49 (batch2)
    // Range [1.0, 3.0] → 0.5 + i*0.25 ∈ [1.0,3.0] → i ∈ [2,10]
    //   docs 2..10 from batch1 = 9 docs
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: weight_f64,
            condition: FastFieldCondition::RangeF64 {
                min: Some(1.0),
                max: Some(3.0),
            },
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            9,
            "RangeF64[1.0,3.0] should match 9 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 14: InText with nonexistent value → 0 matches ──
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: tags,
            condition: FastFieldCondition::InText(vec!["nonexistent".to_string()]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            0,
            "InText(nonexistent) should match 0, got {}",
            results.hits.len()
        );
    }

    // ── Test 15: EqI64 ──
    // temperature == 0 → only i=5 (real temp=0). Absent docs excluded by sentinel.
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: temp_i64,
            condition: FastFieldCondition::EqI64(0),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            1,
            "EqI64(0) should match 1 doc (absent excluded), got {}",
            results.hits.len()
        );
    }

    // ── Test 16: EqF64 ──
    // weight == 0.5 → i=0 (0.5 + 0*0.25 = 0.5), only doc 0
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: weight_f64,
            condition: FastFieldCondition::EqF64(0.5),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            1,
            "EqF64(0.5) should match 1 doc, got {}",
            results.hits.len()
        );
    }

    // ── Test 17: AND combo with absent field values ──
    // category == "electronics" AND weight in [2.0, 4.0]
    // electronics → i in [0,11], weight = 0.5+i*0.25 ∈ [2.0,4.0] → i ∈ [6,14]
    // Intersection: i ∈ [6,11] = 6 docs
    // But weight absent for docs 16..24, and i in [6,11] are all < 16 → all have weight
    {
        let q = FastFieldFilterQuery::standalone(vec![
            FastFieldFilter {
                field: category,
                condition: FastFieldCondition::EqText("electronics".to_string()),
            },
            FastFieldFilter {
                field: weight_f64,
                condition: FastFieldCondition::RangeF64 {
                    min: Some(2.0),
                    max: Some(4.0),
                },
            },
        ]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            6,
            "electronics AND weight[2.0,4.0] should match 6 docs, got {}",
            results.hits.len()
        );
    }

    // ── Test 18: InI64 ──
    // temperature IN [-50, 0, 100]:
    //   i=0→-50✓, i=5→0✓, i=15→100✓ = 3 docs. Absent excluded by sentinel.
    {
        let q = FastFieldFilterQuery::standalone(vec![FastFieldFilter {
            field: temp_i64,
            condition: FastFieldCondition::InI64(vec![-50, 0, 100]),
        }]);
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            3,
            "InI64(-50,0,100) should match 3 docs (absent excluded), got {}",
            results.hits.len()
        );
    }

    eprintln!("Comprehensive fast-field filter test passed: 50 docs, 18 filter scenarios");
}

// ========================================================================
// DocPredicate integration tests
// ========================================================================

/// Helper: create index with 20 docs.
/// All docs have "alpha beta", odd docs also have "gamma".
/// category = i % 4 (values 0,1,2,3).
async fn create_predicate_test_index() -> (
    Index<crate::directories::RamDirectory>,
    crate::dsl::Field,
    crate::dsl::Field,
) {
    use crate::directories::RamDirectory;

    let dir = RamDirectory::new();

    let mut sb = SchemaBuilder::default();
    let content = sb.add_text_field("content", true, true);
    let category = sb.add_u64_field("category", false, true);
    sb.set_fast(category, true);
    let schema = sb.build();

    let config = IndexConfig::default();
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();

    for i in 0u64..20 {
        let mut doc = Document::new();
        let text = if i % 2 == 1 {
            "alpha beta gamma"
        } else {
            "alpha beta"
        };
        doc.add_text(content, text);
        doc.add_u64(category, i % 4);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    (index, content, category)
}

fn filter_eq(
    field: crate::dsl::Field,
    val: u64,
) -> Vec<crate::query::fast_filter::FastFieldFilter> {
    use crate::query::fast_filter::{FastFieldCondition, FastFieldFilter};
    vec![FastFieldFilter {
        field,
        condition: FastFieldCondition::EqU64(val),
    }]
}

/// DocPredicate filtering across all query paths:
/// TermQuery, BooleanQuery OR (MaxScore), BooleanQuery MUST, standalone filter.
#[tokio::test]
async fn test_predicate_filtering_all_paths() {
    use crate::query::fast_filter::FastFieldFilterQuery;
    use std::sync::Arc;

    let (index, content, category) = create_predicate_test_index().await;

    // ── 1. Filter + single TermQuery (non-executor path) ──
    // "alpha" matches all 20 docs. Filter category==0 → {0,4,8,12,16} = 5 docs.
    {
        let term_query: Arc<dyn crate::query::Query> =
            Arc::new(crate::query::TermQuery::new(content, b"alpha".to_vec()));
        let q = FastFieldFilterQuery::new(term_query, filter_eq(category, 0));
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            5,
            "TermQuery + filter(cat==0) should return 5 docs"
        );
        for hit in &results.hits {
            assert_eq!(hit.address.doc_id % 4, 0);
            assert!(hit.score > 0.0);
        }
    }

    // ── 2. Filter + BooleanQuery OR (MaxScore executor path) ──
    // "alpha OR gamma" matches all 20 docs. Filter category==0 → 5 docs.
    {
        let bool_query: Arc<dyn crate::query::Query> = Arc::new(
            crate::query::BooleanQuery::new()
                .should(crate::query::TermQuery::new(content, b"alpha".to_vec()))
                .should(crate::query::TermQuery::new(content, b"gamma".to_vec())),
        );
        let q = FastFieldFilterQuery::new(bool_query, filter_eq(category, 0));
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            5,
            "OR + filter(cat==0) should return 5 docs"
        );
        for hit in &results.hits {
            assert_eq!(hit.address.doc_id % 4, 0);
        }
    }

    // ── 3. Filter + BooleanQuery MUST — no match ──
    // MUST "alpha" AND "gamma" → odd docs. Odd docs have cat {1,3} — none == 0.
    {
        let bool_query: Arc<dyn crate::query::Query> = Arc::new(
            crate::query::BooleanQuery::new()
                .must(crate::query::TermQuery::new(content, b"alpha".to_vec()))
                .must(crate::query::TermQuery::new(content, b"gamma".to_vec())),
        );
        let q = FastFieldFilterQuery::new(bool_query, filter_eq(category, 0));
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            0,
            "MUST + filter(cat==0) should return 0"
        );
    }

    // ── 4. Filter category==1 + BooleanQuery MUST — with matches ──
    // MUST "alpha" AND "gamma" → odd docs. category==1 among odd: {1,5,9,13,17} = 5.
    {
        let bool_query: Arc<dyn crate::query::Query> = Arc::new(
            crate::query::BooleanQuery::new()
                .must(crate::query::TermQuery::new(content, b"alpha".to_vec()))
                .must(crate::query::TermQuery::new(content, b"gamma".to_vec())),
        );
        let q = FastFieldFilterQuery::new(bool_query, filter_eq(category, 1));
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            5,
            "MUST + filter(cat==1) should return 5 odd docs"
        );
        for hit in &results.hits {
            assert!(hit.address.doc_id % 2 == 1);
            assert_eq!(hit.address.doc_id % 4, 1);
        }
    }

    // ── 5. Standalone filter (no inner query) ──
    {
        let q = FastFieldFilterQuery::standalone(filter_eq(category, 0));
        let results = index.search(&q, 100).await.unwrap();
        assert_eq!(
            results.hits.len(),
            5,
            "Standalone filter(cat==0) should return 5 docs"
        );
    }
}
