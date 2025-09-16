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
