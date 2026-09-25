#![cfg(feature = "native")]

use std::sync::Arc;
use summa_core::query::{
    BooleanQuery, CountCollector, GlobalStatsBuilder, Query, TermQuery, TopKCollector,
    collect_segment,
};
use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn boolean_grouping_preserves_explicit_per_term_statistics() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    let schema = Arc::new(schema.build());
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for row in 0..257 {
        let mut doc = Document::new();
        doc.add_text(
            field,
            format!(
                "{} {}",
                "alpha ".repeat(row % 7 + 1),
                "beta ".repeat(row % 11 + 1)
            ),
        );
        builder.add_document(doc).unwrap();
    }
    let id = SegmentId::new();
    builder.build(&dir, id, None).await.unwrap();
    let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
    let term = |name: &str, average, df| {
        let mut stats = GlobalStatsBuilder::new();
        stats.set_text_corpus_size(field, 10000);
        stats.set_avg_field_len(field, average);
        stats.add_text_df(field, name.into(), df);
        TermQuery::with_global_stats(field, name, Arc::new(stats.build(0)))
    };
    let alpha = term("alpha", 600.0, 7);
    for beta in [term("beta", 3.0, 9000), TermQuery::text(field, "beta")] {
        let mut expected = vec![0.0f32; 257];
        for query in [&alpha, &beta] {
            let mut top = TopKCollector::new(257);
            collect_segment(&reader, query, &mut top).await.unwrap();
            for hit in top.into_sorted_results() {
                expected[hit.doc_id as usize] += hit.score;
            }
        }
        for shape in 0..3 {
            let query = match shape {
                0 => BooleanQuery::new()
                    .should(alpha.clone())
                    .should(beta.clone()),
                1 => BooleanQuery::new().must(alpha.clone()).must(beta.clone()),
                _ => BooleanQuery::new().must(alpha.clone()).should(beta.clone()),
            };
            for complete in [false, true] {
                let mut top = TopKCollector::new(10);
                let mut count = CountCollector::new();
                if complete {
                    collect_segment(&reader, &query, &mut (&mut top, &mut count))
                        .await
                        .unwrap();
                    assert_eq!(count.count(), 257);
                } else {
                    collect_segment(&reader, &query, &mut top).await.unwrap();
                }
                let mut wanted: Vec<_> = expected
                    .iter()
                    .enumerate()
                    .map(|(doc, &score)| (doc as u32, score))
                    .collect();
                wanted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                let actual = top.into_sorted_results();
                assert_eq!(actual.len(), 10);
                for (got, &(doc, score)) in actual.iter().zip(&wanted) {
                    assert_eq!(
                        (got.doc_id, got.score.to_bits()),
                        (doc, score.to_bits()),
                        "shape={shape} complete={complete}"
                    );
                }
                #[cfg(feature = "sync")]
                {
                    let mut sync = query.scorer_sync(&reader, 10).unwrap();
                    while sync.doc() != summa_core::structures::TERMINATED {
                        assert_eq!(
                            sync.score().to_bits(),
                            expected[sync.doc() as usize].to_bits()
                        );
                        sync.advance();
                    }
                }
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn ranked_conjunctions_match_exhaustive_ids_scores_counts_and_positioned_membership() {
    use summa_core::query::{ScorerOptions, collect_segment_with_limit};
    use summa_core::structures::{PostingCodec, TERMINATED};
    for (codec, impacts) in [
        (PostingCodec::Rounded, false),
        (PostingCodec::Rounded, true),
        (PostingCodec::Simd4x, true),
    ] {
        let dir = RamDirectory::new();
        let mut schema = SchemaBuilder::default();
        let field = schema.add_text_field("text", true, false);
        schema.set_positions(field, summa_core::dsl::PositionMode::TokenPosition);
        let schema = Arc::new(schema.build());
        let mut builder = SegmentBuilder::new(
            schema.clone(),
            SegmentBuilderConfig {
                posting_codec: codec,
                posting_impact_bounds: impacts,
                ..Default::default()
            },
        )
        .unwrap();
        for row in 0..8192 {
            let mut value = "padding ".repeat(1 + row % 97);
            for (term, divisor) in [("alpha", 2), ("beta", 3), ("gamma", 31)] {
                if row % divisor == 0 || row >= 8100 {
                    value.push_str(&format!("{term} ").repeat(1 + row % 11));
                }
            }
            let mut doc = Document::new();
            doc.add_text(field, value);
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
        for terms in [
            vec!["alpha", "beta"],
            vec!["gamma", "beta", "alpha"],
            vec!["beta", "alpha", "beta"],
            vec!["alpha", "missing"],
        ] {
            let mut query = BooleanQuery::new();
            for &term in &terms {
                query = query.must(TermQuery::text(field, term));
            }
            let mut exhaustive = TopKCollector::new(8192);
            let mut count = CountCollector::new();
            collect_segment(&reader, &query, &mut (&mut exhaustive, &mut count))
                .await
                .unwrap();
            let expected = exhaustive.into_sorted_results();
            assert_eq!(count.count() as usize, expected.len());
            for limit in [1, 10, 100, 1000] {
                let mut ranked = TopKCollector::new(limit);
                collect_segment_with_limit(&reader, &query, &mut ranked, limit)
                    .await
                    .unwrap();
                let actual = ranked.into_sorted_results();
                let pairs = |hits: &[summa_core::query::SearchResult]| {
                    hits.iter()
                        .map(|h| (h.doc_id, h.score.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    pairs(&actual),
                    pairs(&expected[..limit.min(expected.len())]),
                    "terms={terms:?} codec={codec:?} impacts={impacts} limit={limit}"
                );
                #[cfg(feature = "sync")]
                {
                    let mut sync = query
                        .scorer_sync_with_options(&reader, limit, ScorerOptions::default())
                        .unwrap();
                    let mut got = Vec::new();
                    while sync.doc() != TERMINATED {
                        got.push((sync.doc(), sync.score().to_bits()));
                        sync.advance();
                    }
                    let mut wanted = pairs(&actual);
                    wanted.sort_unstable_by_key(|hit| hit.0);
                    assert_eq!(got, wanted);
                }
            }
            let mut positioned = query.scorer(&reader, 1).await.unwrap();
            let mut seen = 0;
            while positioned.doc() != TERMINATED {
                assert!(positioned.matched_positions().is_some());
                seen += 1;
                positioned.advance();
            }
            assert_eq!(seen, expected.len());
            let budget = summa_core::query::SharedThreshold::for_limit(10)
                .with_deadline(Some(std::time::Instant::now()));
            let mut options = ScorerOptions::default();
            options.shared_threshold = Some(budget.clone());
            let expired = query
                .scorer_with_options(&reader, 10, options)
                .await
                .unwrap();
            assert_eq!(expired.doc(), TERMINATED);
            assert!(budget.truncated());
        }
    }
}
