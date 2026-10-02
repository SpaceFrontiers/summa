//! Count-only expansion shortcuts must preserve membership and eligibility.
use crate::query::{
    BooleanQuery, Collector, CountCollector, PrefixQuery, Query, RegexQuery, TermQuery,
    WildcardQuery, collect_segment,
};
use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory, SchemaBuilder};

#[tokio::test]
async fn pattern_counts_preserve_overlaps_boolean_filters_rgb_and_deleted_rows() {
    struct Docs(Vec<u32>);
    impl Collector for Docs {
        // Accepting zero is not a promise to accept a nonzero cardinality.
        fn collect_count(&mut self, count: u64) -> bool {
            count == 0
        }
        fn collect(&mut self, doc: u32, _: f32, _: &[(u32, Vec<crate::query::ScoredPosition>)]) {
            self.0.push(doc);
        }
        fn needs_scores(&self) -> bool {
            false
        }
        fn needs_positions(&self) -> bool {
            false
        }
    }
    let matches =
        |i: u32| (i >= 10 && i.is_multiple_of(2)) || i.is_multiple_of(257) || i.is_multiple_of(263);
    // The two rare terms have over 128 distinct IDs together, forcing the
    // dominant cursor to retain membership across more than one probe batch.
    for reorder in [false, true] {
        let mut schema = SchemaBuilder::default();
        let id = schema.add_text_field_with_tokenizer("id", true, true, "raw");
        schema.set_primary_key(id);
        let body = schema.add_text_field("body", true, false);
        schema.set_reorder(body, reorder);
        let config = IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            merge_policy: Box::new(crate::merge::NoMergePolicy),
            ..Default::default()
        };
        let dir = RamDirectory::new();
        let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
            .await
            .unwrap();
        writer.init_primary_key_dedup().await.unwrap();
        for i in 0u32..20000 {
            let mut text = format!("base rare{i} ");
            if i >= 10 && i.is_multiple_of(2) {
                text.push_str("alpha ");
            }
            if i.is_multiple_of(257) {
                text.push_str("alpine ");
            }
            if i.is_multiple_of(263) {
                text.push_str("alps ");
            }
            if i % 3 == 0 {
                text.push_str("keep ");
            }
            let mut doc = Document::new();
            doc.add_text(id, i.to_string());
            doc.add_text(body, text);
            let mut admitted = writer.add_document(doc.clone());
            for _ in 0..1000 {
                if !matches!(admitted, Err(crate::Error::QueueFull)) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                admitted = writer.add_document(doc.clone());
            }
            admitted.unwrap();
        }
        writer.commit().await.unwrap();
        if reorder {
            writer.reorder().await.unwrap();
        }
        for deleted in [false, true] {
            if deleted {
                writer.delete_primary_key("0").unwrap();
                writer.delete_primary_key("514").unwrap();
                writer.commit().await.unwrap();
            }
            let index = Index::open(dir.clone(), config.clone()).await.unwrap();
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            assert_eq!(searcher.num_segments(), 1);
            let segment = &searcher.segment_readers()[0];
            let prefix = || PrefixQuery::text(body, "alp");
            let queries: Vec<(Box<dyn Query>, u8)> = vec![
                (Box::new(prefix()), 0),
                (Box::new(WildcardQuery::new(body, "alp*").unwrap()), 0),
                (Box::new(RegexQuery::new(body, "alp(ha|ine|s)").unwrap()), 0),
                (Box::new(BooleanQuery::new().must(prefix())), 0),
                (Box::new(BooleanQuery::new().should(prefix())), 0),
                (
                    Box::new(
                        BooleanQuery::new()
                            .must(prefix())
                            .must(TermQuery::text(body, "keep")),
                    ),
                    1,
                ),
                (
                    Box::new(
                        BooleanQuery::new()
                            .must(prefix())
                            .must_not(TermQuery::text(body, "keep")),
                    ),
                    2,
                ),
                (Box::new(RegexQuery::new(body, "alpha|keep").unwrap()), 3),
                (Box::new(PrefixQuery::text(body, "missing")), 4),
            ];
            for (query, shape) in queries {
                let expected: Vec<u32> = (0..20000)
                    .filter(|&i| {
                        (!deleted || (i != 0 && i != 514))
                            && match shape {
                                0 => matches(i),
                                1 => matches(i) && i % 3 == 0,
                                2 => matches(i) && i % 3 != 0,
                                3 => (i >= 10 && i.is_multiple_of(2)) || i % 3 == 0,
                                _ => false,
                            }
                    })
                    .collect();
                let mut count = CountCollector::new();
                collect_segment(segment, query.as_ref(), &mut count)
                    .await
                    .unwrap();
                assert_eq!(
                    count.count(),
                    expected.len() as u64,
                    "{query}, rgb={reorder}, deleted={deleted}"
                );
                let mut docs = Docs(Vec::new());
                collect_segment(segment, query.as_ref(), &mut docs)
                    .await
                    .unwrap();
                docs.0.sort_unstable();
                assert_eq!(
                    docs.0, expected,
                    "ID collectors must still receive every match: {query}"
                );
            }
            // Count shortcuts cannot bypass expansion admission errors.
            for query in [
                Box::new(PrefixQuery::text(body, "rare")) as Box<dyn Query>,
                Box::new(RegexQuery::new(body, "rare.*").unwrap()),
            ] {
                assert!(
                    collect_segment(segment, query.as_ref(), &mut CountCollector::new())
                        .await
                        .is_err()
                );
            }
        }
        writer.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn prefix_queries_reject_invalid_fields_in_counts_estimates_and_ranked_search() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    let stored = schema.add_text_field("stored", false, true);
    let number = schema.add_u64_field("number", true, false);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let mut doc = Document::new();
    doc.add_text(body, "alpha");
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    for field in [crate::Field(999), stored, number] {
        let query = PrefixQuery::text(field, "alp");
        #[cfg(feature = "sync")]
        assert!(
            query.as_doc_bitset(segment).is_none(),
            "invalid fields must not become empty planner predicates"
        );
        assert!(
            collect_segment(segment, &query, &mut CountCollector::new())
                .await
                .is_err(),
            "invalid field {field:?} must not become an empty count"
        );
        assert!(query.count_estimate(segment).await.is_err());
        assert!(query.scorer(segment, 10).await.is_err());
        #[cfg(feature = "sync")]
        assert!(query.scorer_sync(segment, 10).is_err());
    }
    writer.shutdown().await.unwrap();
}
