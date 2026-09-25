#![cfg(feature = "native")]
use std::sync::Arc;
use summa_core::query::{
    BooleanQuery, CountCollector, Query, TermQuery, TopKCollector, collect_segment,
};
use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use summa_core::{Document, RamDirectory, SchemaBuilder};
#[tokio::test(flavor = "current_thread")]
async fn ranked_required_optional_windows_preserve_complete_scores_counts_and_positioned_fallbacks()
{
    use summa_core::query::{ScorerOptions, collect_segment_with_limit};
    use summa_core::structures::{PostingCodec, TERMINATED};
    for (codec, impacts) in [
        (PostingCodec::Rounded, false),
        (PostingCodec::Rounded, true),
        (PostingCodec::Packed, true),
        (PostingCodec::Pfor, true),
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
        for (required, optional) in [
            (vec!["alpha"], vec!["beta"]),
            (vec!["beta"], vec!["alpha"]),
            (vec!["alpha"], vec!["gamma", "beta"]),
            (vec!["alpha", "beta"], vec!["gamma"]),
            (vec!["gamma"], vec!["alpha", "beta", "alpha"]),
            (vec!["beta", "alpha"], vec!["beta", "alpha"]),
            (vec!["alpha"; 32], vec!["beta"; 32]),
            (vec!["alpha"], vec!["beta"; 63]),
            (vec!["beta"; 63], vec!["gamma"]),
            (vec!["alpha"], vec!["missing"]),
            (vec!["missing"], vec!["alpha"]),
        ] {
            let mut query = BooleanQuery::new();
            for &term in &required {
                query = query.must(TermQuery::text(field, term));
            }
            for &term in &optional {
                query = query.should(TermQuery::text(field, term));
            }
            let terms = (&required, &optional);
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
                    let mut sync = TopKCollector::new(limit);
                    summa_core::query::collect_segment_with_limit_sync(
                        &reader, &query, &mut sync, limit,
                    )
                    .unwrap();
                    assert_eq!(pairs(&sync.into_sorted_results()), pairs(&actual));
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

#[tokio::test(flavor = "current_thread")]
async fn required_optional_windows_preserve_global_scores_positions_and_chunk_fallbacks() {
    use summa_core::dsl::PositionMode;
    use summa_core::query::{GlobalStatsBuilder, SharedThreshold, search_segment_shared};
    use summa_core::structures::PostingCodec;
    for codec in [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ] {
        for chunked in [false, true] {
            let dir = RamDirectory::new();
            let mut schema = SchemaBuilder::default();
            let field = schema.add_text_field("body", true, false);
            schema.set_positions(field, PositionMode::Full);
            schema.set_chunked(field, chunked);
            let schema = Arc::new(schema.build());
            let mut builder = SegmentBuilder::new(
                schema.clone(),
                SegmentBuilderConfig {
                    posting_codec: codec,
                    ..Default::default()
                },
            )
            .unwrap();
            for row in 0..385 {
                let mut doc = Document::new();
                for ordinal in 0..if chunked { 2 } else { 1 } {
                    let value = if row == 0 {
                        "alpha beta".to_string()
                    } else if row >= 370 {
                        "alpha alpha beta ".repeat(1 + (row + ordinal) % 7)
                    } else {
                        "padding ".repeat(32 + row % 127)
                            + match (row + ordinal) % 4 {
                                0 => "alpha gap beta",
                                1 => "alpha beta",
                                2 => "alpha alpha beta",
                                _ => "beta alpha",
                            }
                    };
                    doc.add_text(field, value);
                }
                builder.add_document(doc).unwrap();
            }
            let id = SegmentId::new();
            builder.build(&dir, id, None).await.unwrap();
            let reader = SegmentReader::open(&dir, id, schema, 16).await.unwrap();
            let mut stats = GlobalStatsBuilder::new();
            stats.set_text_corpus_size(field, 10000);
            stats.set_avg_field_len(field, 175.0);
            stats.add_text_df(field, "alpha".into(), 9000);
            stats.add_text_df(field, "beta".into(), 103);
            let global = Arc::new(stats.build(0));
            for query in [
                BooleanQuery::new()
                    .must(TermQuery::text(field, "alpha"))
                    .should(TermQuery::text(field, "beta")),
                BooleanQuery::new()
                    .must(TermQuery::text(field, "beta"))
                    .should(TermQuery::text(field, "alpha"))
                    .with_global_stats(global.clone()),
                BooleanQuery::new()
                    .must(TermQuery::text(field, "alpha"))
                    .should(TermQuery::text(field, "beta"))
                    .must_not(TermQuery::text(field, "gap")),
                BooleanQuery::new()
                    .must(
                        BooleanQuery::new()
                            .should(TermQuery::text(field, "alpha"))
                            .should(TermQuery::text(field, "beta")),
                    )
                    .should(TermQuery::text(field, "gap")),
            ] {
                for positions in [false, true] {
                    let mut all = if positions {
                        TopKCollector::with_positions(1000)
                    } else {
                        TopKCollector::new(1000)
                    };
                    let mut count = CountCollector::new();
                    collect_segment(&reader, &query, &mut (&mut all, &mut count))
                        .await
                        .unwrap();
                    let expected = all.into_sorted_results();
                    assert_eq!(count.count() as usize, expected.len());
                    for k in [0, 1, 10, 100, 1000] {
                        let (actual, seen) = search_segment_shared(
                            &reader,
                            &query,
                            k,
                            positions,
                            SharedThreshold::for_limit(k),
                        )
                        .await
                        .unwrap();
                        assert_eq!(
                            actual,
                            expected[..k.min(expected.len())],
                            "{codec:?} chunked={chunked} positions={positions} k={k} query={query}"
                        );
                        assert!(u64::from(seen) <= count.count());
                        #[cfg(feature = "sync")]
                        {
                            let sync = summa_core::query::search_segment_shared_sync(
                                &reader,
                                &query,
                                k,
                                positions,
                                SharedThreshold::for_limit(k),
                            )
                            .unwrap();
                            assert_eq!(sync, (actual, seen));
                        }
                    }
                    let expired = SharedThreshold::for_limit(10)
                        .with_deadline(Some(std::time::Instant::now()));
                    assert!(
                        search_segment_shared(&reader, &query, 10, positions, expired.clone())
                            .await
                            .unwrap()
                            .0
                            .is_empty()
                    );
                    assert!(expired.truncated());
                }
            }
        }
    }
}
