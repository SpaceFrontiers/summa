#![cfg(feature = "native")]

use hermes_core::dsl::PositionMode;
use hermes_core::query::{
    CountCollector, GlobalStatsBuilder, PhraseQuery, SharedThreshold, TopKCollector,
    collect_segment, search_segment_shared,
};
use hermes_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use hermes_core::structures::PostingCodec;
use hermes_core::{Document, RamDirectory, SchemaBuilder};
use std::sync::Arc;

#[tokio::test(flavor = "current_thread")]
async fn ranked_phrases_preserve_exhaustive_scores_counts_offsets_and_chunk_ordinals() {
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
                PhraseQuery::text(field, "alpha beta"),
                PhraseQuery::text(field, "alpha beta").with_slop(1),
                PhraseQuery::text(field, "alpha alpha beta").with_global_stats(global.clone()),
                PhraseQuery::with_offsets(
                    field,
                    vec![(0, b"alpha".to_vec()), (2, b"beta".to_vec())],
                )
                .with_global_stats(global.clone()),
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
                            let sync = hermes_core::query::search_segment_shared_sync(
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
