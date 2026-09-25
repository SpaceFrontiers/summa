#![cfg(feature = "native")]
use std::sync::Arc;
use summa_core::query::{
    BooleanQuery, CountCollector, GlobalStats, GlobalStatsBuilder, TermQuery, TopKCollector,
    collect_segment,
};
#[cfg(feature = "sync")]
use summa_core::query::{Query, ScorerOptions};
use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn explicit_boolean_statistics_reach_fallback_children_and_preserve_nested_overrides() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("body", true, false);
    schema.set_positions(field, summa_core::dsl::PositionMode::TokenPosition);
    let schema = Arc::new(schema.build());
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for row in 0..257 {
        let mut doc = Document::new();
        doc.add_text(
            field,
            format!(
                "{} {} {}",
                "alpha ".repeat(1 + row % 3),
                "beta ".repeat(1 + row % 7),
                "gamma ".repeat(row % 5)
            ),
        );
        builder.add_document(doc).unwrap();
    }
    let id = SegmentId::new();
    builder.build(&dir, id, None).await.unwrap();
    let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
    let stats = |average, alpha, beta| {
        let mut s = GlobalStatsBuilder::new();
        s.set_text_corpus_size(field, 10000);
        s.set_avg_field_len(field, average);
        s.add_text_df(field, "alpha".into(), alpha);
        s.add_text_df(field, "beta".into(), beta);
        s.add_text_df(field, "gamma".into(), 1200);
        Arc::new(s.build(0))
    };
    let outer = stats(175.0, 9000, 103);
    let inner = stats(3.0, 500, 7000);
    let term = |name| TermQuery::text(field, name);
    let explicit =
        |name, s: &Arc<GlobalStats>| TermQuery::with_global_stats(field, name, s.clone());
    let pairs = [
        (
            BooleanQuery::new()
                .must(term("alpha"))
                .should(term("beta"))
                .with_global_stats(outer.clone()),
            BooleanQuery::new()
                .must(explicit("alpha", &outer))
                .should(explicit("beta", &outer)),
        ),
        (
            BooleanQuery::new()
                .must(term("alpha"))
                .must(term("beta"))
                .with_global_stats(outer.clone()),
            BooleanQuery::new()
                .must(explicit("alpha", &outer))
                .must(explicit("beta", &outer)),
        ),
        (
            BooleanQuery::new()
                .must(term("alpha"))
                .with_global_stats(outer.clone()),
            BooleanQuery::new().must(explicit("alpha", &outer)),
        ),
        (
            BooleanQuery::new()
                .should(term("alpha"))
                .with_global_stats(outer.clone()),
            BooleanQuery::new().should(explicit("alpha", &outer)),
        ),
        (
            BooleanQuery::new()
                .must(explicit("alpha", &inner))
                .should(term("beta"))
                .with_global_stats(outer.clone()),
            BooleanQuery::new()
                .must(explicit("alpha", &inner))
                .should(explicit("beta", &outer)),
        ),
        (
            BooleanQuery::new()
                .should(
                    BooleanQuery::new()
                        .should(term("alpha"))
                        .should(term("beta"))
                        .with_global_stats(inner.clone()),
                )
                .should(term("gamma"))
                .with_global_stats(outer.clone()),
            BooleanQuery::new()
                .should(
                    BooleanQuery::new()
                        .should(explicit("alpha", &inner))
                        .should(explicit("beta", &inner)),
                )
                .should(explicit("gamma", &outer)),
        ),
    ];
    for (query, reference) in pairs {
        for positions in [false, true] {
            let collector = || {
                if positions {
                    TopKCollector::with_positions(257)
                } else {
                    TopKCollector::new(257)
                }
            };
            let mut expected = collector();
            let mut count = CountCollector::new();
            collect_segment(&reader, &reference, &mut (&mut expected, &mut count))
                .await
                .unwrap();
            let expected = expected.into_sorted_results();
            let mut actual = collector();
            let mut actual_count = CountCollector::new();
            collect_segment(&reader, &query, &mut (&mut actual, &mut actual_count))
                .await
                .unwrap();
            assert_eq!(actual_count.count(), count.count());
            assert_eq!(
                actual.into_sorted_results(),
                expected,
                "complete positions={positions} {query}"
            );
            let mut ranked = if positions {
                TopKCollector::with_positions(10)
            } else {
                TopKCollector::new(10)
            };
            summa_core::query::collect_segment_with_limit(&reader, &query, &mut ranked, 10)
                .await
                .unwrap();
            assert_eq!(
                ranked.into_sorted_results(),
                expected[..10],
                "ranked {query}"
            );
            #[cfg(feature = "sync")]
            {
                // Parent explicit stats must also override inherited request stats.
                let mut options = ScorerOptions::default();
                options.global_stats = Some(inner.clone());
                options.collect_positions = positions;
                let mut scorer = query
                    .scorer_sync_with_options(&reader, 257, options)
                    .unwrap();
                let mut got = Vec::new();
                while scorer.doc() != summa_core::structures::TERMINATED {
                    got.push((scorer.doc(), scorer.score().to_bits()));
                    scorer.advance();
                }
                let mut wanted: Vec<_> = expected
                    .iter()
                    .map(|h| (h.doc_id, h.score.to_bits()))
                    .collect();
                wanted.sort_unstable_by_key(|h| h.0);
                assert_eq!(got, wanted, "sync {query}");
            }
        }
    }
}
