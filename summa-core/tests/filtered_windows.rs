#![cfg(feature = "native")]

use summa_core::query::{
    BooleanQuery, Collector, CountCollector, PhraseQuery, Query, ScorerOptions, TermQuery,
    TopKCollector, collect_segment,
};
use summa_core::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};

#[tokio::test(flavor = "current_thread")]
async fn deleted_text_collection_matches_scalar_scores_counts_and_positions() {
    let mut schema = Schema::builder();
    let key = schema.add_text_field_with_tokenizer("id", true, false, "raw");
    schema.set_primary_key(key);
    let text = schema.add_text_field("text", true, false);
    schema.set_positions(text, summa_core::dsl::PositionMode::TokenPosition);
    let directory = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    for row in 0..10_017 {
        let mut doc = Document::new();
        doc.add_text(key, row.to_string());
        doc.add_text(
            text,
            match row % 3 {
                0 => "alpha beta beta",
                1 => "alpha gamma",
                _ => "beta gamma gamma",
            },
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    for row in 0..10_017 {
        if row % 7 == 0 || (4096..8192).contains(&row) {
            writer.delete_primary_key(&row.to_string()).unwrap();
        }
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(directory, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment = &searcher.segment_readers()[0];
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(TermQuery::text(text, "alpha")),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(text, "alpha"))
                .should(TermQuery::text(text, "beta")),
        ),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(text, "alpha"))
                .should(TermQuery::text(text, "beta"))
                .must_not(TermQuery::text(text, "gamma")),
        ),
        Box::new(PhraseQuery::text(text, "alpha beta")),
    ];
    for query in queries {
        let mut scalar = query
            .scorer_with_options(
                segment,
                segment.num_docs() as usize,
                ScorerOptions::with_positions(),
            )
            .await
            .unwrap();
        let mut expected = TopKCollector::with_positions(100);
        let mut expected_count = 0;
        while scalar.doc() != summa_core::TERMINATED {
            if segment.is_alive(scalar.doc()) {
                expected.collect(
                    scalar.doc(),
                    scalar.score(),
                    &scalar.matched_positions().unwrap_or_default(),
                );
                expected_count += 1;
            }
            scalar.advance();
        }
        let expected = expected.into_sorted_results();
        for positions in [false, true] {
            let mut top = if positions {
                TopKCollector::with_positions(100)
            } else {
                TopKCollector::new(100)
            };
            let mut count = CountCollector::new();
            collect_segment(segment, query.as_ref(), &mut (&mut top, &mut count))
                .await
                .unwrap();
            assert_eq!(count.count(), expected_count, "{query}");
            let actual = top.into_sorted_results();
            assert_eq!(actual, expected, "{query}, positions={positions}");
            if positions {
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(
                        serde_json::to_value(&a.positions).unwrap(),
                        serde_json::to_value(&b.positions).unwrap()
                    );
                }
            }
        }
        let mut count = CountCollector::new();
        collect_segment(segment, query.as_ref(), &mut count)
            .await
            .unwrap();
        assert_eq!(count.count(), expected_count);
        let (hits, _) = summa_core::query::search_segment_with_count(segment, query.as_ref(), 100)
            .await
            .unwrap();
        assert_eq!(hits, expected);
        #[cfg(feature = "sync")]
        {
            let (hits, _) =
                summa_core::query::search_segment_with_count_sync(segment, query.as_ref(), 100)
                    .unwrap();
            assert_eq!(hits, expected);
        }
    }
}
