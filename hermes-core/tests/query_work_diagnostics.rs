#![cfg(all(feature = "sync", feature = "query-diagnostics"))]

use hermes_core::dsl::PositionMode;
use hermes_core::index::{Index, IndexConfig, IndexWriter};
use hermes_core::query::{
    BooleanQuery, CountCollector, PhraseQuery, TermQuery, TopKCollector, collect_segment,
};
use hermes_core::search_diagnostics::{capture, capture_sync};
use hermes_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn work_counts_distinguish_metadata_count_decoding_scoring_and_cached_positions() {
    for compact in [false, true] {
        for quantized in [false, true] {
            let dir = RamDirectory::new();
            let mut schema = SchemaBuilder::default();
            let text = schema.add_text_field("text", true, false);
            schema.set_positions(text, PositionMode::TokenPosition);
            let config = IndexConfig {
                compact_text: compact,
                quantized_norms: quantized,
                posting_ratio_bounds: true,
                num_threads: 1,
                num_indexing_threads: 1,
                merge_policy: Box::new(hermes_core::merge::NoMergePolicy),
                ..Default::default()
            };
            let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
                .await
                .unwrap();
            for _ in 0..300 {
                let mut doc = Document::new();
                doc.add_text(text, "alpha beta");
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
            writer.force_merge().await.unwrap();
            writer.shutdown().await.unwrap();
            let index = Index::open(dir, config).await.unwrap();
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            let segment = &searcher.segment_readers()[0];
            let query = TermQuery::text(text, "alpha");
            let mut count = CountCollector::new();
            let (result, work) = capture(collect_segment(segment, &query, &mut count)).await;
            result.unwrap();
            assert_eq!(count.count(), 300);
            assert_eq!(work.doc_blocks, 0, "term COUNT uses metadata");
            assert_eq!(
                work.search_pool_installs, 0,
                "direct async collection does not enter Rayon"
            );
            assert_eq!(work.exact_score_units + work.lookup_score_units, 0);

            for query in [
                BooleanQuery::new()
                    .should(TermQuery::text(text, "alpha"))
                    .should(TermQuery::text(text, "beta")),
                BooleanQuery::new()
                    .must(TermQuery::text(text, "alpha"))
                    .must(TermQuery::text(text, "beta")),
            ] {
                let mut count = CountCollector::new();
                let (result, work) = capture(collect_segment(segment, &query, &mut count)).await;
                result.unwrap();
                assert_eq!(count.count(), 300);
                assert_eq!(
                    work.norm_tables, 0,
                    "membership must not initialize scoring tables"
                );
                assert_eq!(work.exact_score_units + work.lookup_score_units, 0);
            }

            let mut count = CountCollector::new();
            let (result, work) = capture_sync(|| {
                hermes_core::query::collect_segment_with_limit_sync(
                    segment, &query, &mut count, 300,
                )
            });
            result.unwrap();
            assert_eq!(count.count(), 300);
            assert_eq!(
                work.norm_tables, 0,
                "sync membership skips scoring setup too"
            );

            // Complete collection guarantees every term-document score is evaluated.
            let mut top = TopKCollector::new(300);
            let (result, work) = capture(collect_segment(segment, &query, &mut top)).await;
            result.unwrap();
            assert_eq!(work.doc_blocks, 3);
            assert_eq!(work.doc_values, 300);
            assert_eq!(work.tf_blocks, 3);
            assert_eq!(work.tf_values, 300);
            assert_eq!(work.exact_score_units, if quantized { 0 } else { 300 });
            assert_eq!(work.lookup_score_units, if quantized { 300 } else { 0 });
            assert_eq!(
                work.norm_tables,
                u64::from(quantized),
                "initialize once for ranked collection"
            );
            assert_eq!(work.position_reads, 0);
            let expected = top.into_sorted_results();
            let (actual, ranked_work) =
                capture_sync(|| searcher.search_with_offset_and_count_sync(&query, 300, 0));
            let (actual, _) = actual.unwrap();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!((a.doc_id, a.score.to_bits()), (b.doc_id, b.score.to_bits()));
            }
            assert_eq!(ranked_work.doc_values, 300);
            assert_eq!(
                ranked_work.search_pool_installs, 1,
                "one owner-controlled pool entry"
            );
            assert_eq!(ranked_work.segment_runs, 1);
            let (invalid, invalid_work) =
                capture_sync(|| searcher.search_with_offset_and_count_sync(&query, usize::MAX, 1));
            assert!(invalid.is_err());
            assert_eq!(
                invalid_work.search_pool_installs, 0,
                "invalid windows fail before pool admission"
            );
            assert_eq!(
                ranked_work.exact_score_units + ranked_work.lookup_score_units,
                300
            );

            let phrase = PhraseQuery::text(text, "alpha beta");
            let mut count = CountCollector::new();
            let (result, work) = capture(collect_segment(segment, &phrase, &mut count)).await;
            result.unwrap();
            assert_eq!(count.count(), 300);
            assert_eq!(work.phrase_confirmations, 300);
            assert_eq!(work.position_reads, 600);
            assert_eq!(work.positions_requested, 600);
            assert_eq!(work.position_blocks, 6, "one decode per cached term block");
            assert_eq!(work.position_values, 600);
            assert_eq!(work.phrase_score_units, 0, "COUNT must not score phrases");
            assert!(work.positions_opened > 0);
            let mut repeated = CountCollector::new();
            let (result, warm) = capture(collect_segment(segment, &phrase, &mut repeated)).await;
            result.unwrap();
            assert_eq!(warm.positions_opened, work.positions_opened);
            assert_eq!(warm.position_blocks, work.position_blocks);
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn plain_ranked_terms_and_conjunctions_skip_proven_losing_blocks() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field_with_tokenizer("text", true, false, "simple");
    let config = IndexConfig {
        posting_ratio_bounds: false,
        posting_impact_bounds: false,
        num_threads: 1,
        num_indexing_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for row in 0..8192 {
        let mut doc = Document::new();
        doc.add_text(
            text,
            if row < 128 {
                "alpha beta ".repeat(16)
            } else {
                "alpha beta padding ".to_owned() + &"padding ".repeat(30)
            },
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    assert!(segment.chunk_map(text).is_none());
    assert!(
        !segment
            .get_postings(text, b"alpha")
            .await
            .unwrap()
            .unwrap()
            .has_ratio_bounds()
    );
    let queries: Vec<Box<dyn hermes_core::query::Query>> = vec![
        Box::new(TermQuery::text(text, "alpha")),
        Box::new(hermes_core::query::PrefixQuery::text(text, "alp")),
        Box::new(hermes_core::query::WildcardQuery::text(text, "al*a").unwrap()),
        Box::new(hermes_core::query::RegexQuery::new(text, "al.*a").unwrap()),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(text, "alpha"))
                .must(TermQuery::text(text, "beta")),
        ),
    ];
    for query in queries {
        let mut exhaustive = TopKCollector::new(8192);
        let mut count = CountCollector::new();
        collect_segment(segment, query.as_ref(), &mut (&mut exhaustive, &mut count))
            .await
            .unwrap();
        assert_eq!(count.count(), 8192);
        let expected = exhaustive.into_sorted_results();
        for k in [1, 10, 100] {
            let (actual, work) =
                capture_sync(|| searcher.search_with_offset_and_count_sync(query.as_ref(), k, 0));
            let (actual, _) = actual.unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|h| (h.doc_id, h.score.to_bits()))
                    .collect::<Vec<_>>(),
                expected[..k]
                    .iter()
                    .map(|h| (h.doc_id, h.score.to_bits()))
                    .collect::<Vec<_>>()
            );
            assert!(
                work.exact_score_units + work.lookup_score_units < 8192 / 2,
                "plain ranked queries must skip proven losers: {work:?}"
            );
            assert!(
                work.doc_blocks < 32,
                "ranked term filters must not decode every block: {work:?}"
            );
        }
    }
}
