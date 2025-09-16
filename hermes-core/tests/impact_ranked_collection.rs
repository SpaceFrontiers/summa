#![cfg(feature = "sync")]

use hermes_core::index::{Index, IndexConfig, IndexWriter};
use hermes_core::query::{BooleanQuery, TermQuery, TopKCollector, collect_segment};
use hermes_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn impact_pruning_preserves_canonical_score_sums_for_wide_boolean_queries() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field("text", true, false);
    let config = IndexConfig {
        posting_impact_bounds: true,
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(hermes_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let terms: Vec<_> = (b'a'..=b'p')
        .map(|c| format!("term{}", char::from(c)))
        .collect();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for row in 0..4096 {
        let mut value = String::new();
        for (i, term) in terms.iter().enumerate() {
            if i == 0 || !(row + i).is_multiple_of(5) {
                value.push_str(&(term.clone() + " ").repeat(1 + (row * 7 + i) % 11));
            }
        }
        if row >= 128 {
            value.push_str(&"padding ".repeat(256 + row % 73));
        }
        if row.is_multiple_of(101) {
            value.push_str("excluded");
        }
        let mut doc = Document::new();
        doc.add_text(text, value);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    let list = segment
        .get_postings_sync(text, terms[0].as_bytes())
        .unwrap()
        .unwrap();
    assert!(list.has_impact_bounds());
    assert!(list.has_ratio_bounds());
    assert!(list.block_impact_point_count(0).unwrap() > 0);
    for arity in [1, 2, 7, 16, 31, 64] {
        for required in [0, 1, arity] {
            let mut query = BooleanQuery::new();
            // Duplicate clauses exercise the scorer's actual addition order.
            for i in 0..arity {
                let term = TermQuery::text(text, &terms[i % terms.len()]);
                query = if i < required {
                    query.must(term)
                } else {
                    query.should(term)
                };
            }
            query = query.must_not(TermQuery::text(text, "excluded"));
            if let Some(stats) = searcher.query_text_stats(&query, None) {
                query = query.with_global_stats(stats);
            }
            let mut all = TopKCollector::new(4096);
            collect_segment(segment, &query, &mut all).await.unwrap();
            let expected = all.into_sorted_results();
            for (limit, offset) in [(1, 0), (10, 0), (1000, 0), (17, 7)] {
                let (native, _) = searcher
                    .search_with_offset_and_count_sync(&query, limit, offset)
                    .unwrap();
                let (asynchronous, _) = searcher
                    .search_with_offset_and_count(&query, limit, offset)
                    .await
                    .unwrap();
                assert_eq!(native, asynchronous);
                let end = (limit + offset).min(expected.len());
                let wanted = &expected[offset.min(end)..end];
                let values = |hits: &[hermes_core::query::SearchResult]| {
                    hits.iter()
                        .map(|h| (h.doc_id, h.score.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    values(&native),
                    values(wanted),
                    "arity={arity}, required={required}, limit={limit}, offset={offset}"
                );
            }
        }
    }
}
