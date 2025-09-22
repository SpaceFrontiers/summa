#![cfg(feature = "sync")]

use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::{TermQuery, TopKCollector, collect_segment};
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn ranked_terms_match_exhaustive_global_scores_ties_and_offsets() {
    for (ratio, impacts) in [(false, false), (true, false), (true, true)] {
        ranked_terms_match_exhaustive_global_scores_ties_and_offsets_with(ratio, impacts).await;
    }
}

async fn ranked_terms_match_exhaustive_global_scores_ties_and_offsets_with(
    posting_ratio_bounds: bool,
    posting_impact_bounds: bool,
) {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field_with_tokenizer("id", true, false, "raw");
    schema.set_primary_key(id);
    let field = schema.add_text_field("text", true, false);
    schema.set_bm25_params(field, Some(0.9), Some(0.6));
    let config = IndexConfig {
        posting_ratio_bounds,
        posting_impact_bounds,
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    for segment in 0..3 {
        for row in 0..4096 {
            let mut doc = Document::new();
            doc.add_text(id, format!("{segment}-{row}"));
            let text = "common ".repeat(1 + row % 5)
                + &"padding ".repeat(segment * 7 + row % 31)
                + if row % 1000 == 0 { "rare" } else { "" };
            doc.add_text(field, text);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.delete_primary_key("0-124").unwrap();
    writer.delete_primary_key("1-279").unwrap();
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_segments(), 3);
    for term in ["common", "rare", "absent"] {
        let mut query = TermQuery::text(field, term);
        if let Some(stats) = searcher.query_text_stats(&query, None) {
            query.set_global_stats(stats);
        }
        // Optional scoring clauses cannot inflate count metadata, and deleted
        // segments must enumerate live membership instead of using physical DF.
        let count_query = summa_core::query::BooleanQuery::new()
            .must(TermQuery::text(field, term))
            .should(TermQuery::text(field, "padding"));
        let mut exact_count = 0;
        for segment in searcher.segment_readers() {
            let mut count = summa_core::query::CountCollector::new();
            collect_segment(segment, &count_query, &mut count)
                .await
                .unwrap();
            exact_count += count.count();
        }
        assert_eq!(
            exact_count,
            match term {
                "common" => 12_286,
                "rare" => 15,
                _ => 0,
            }
        );
        let mut expected = Vec::new();
        for segment in searcher.segment_readers() {
            let mut top = TopKCollector::new(4096);
            collect_segment(segment, &query, &mut top).await.unwrap();
            for mut hit in top.into_sorted_results() {
                hit.segment_id = segment.meta().id;
                expected.push(hit);
            }
        }
        expected.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.segment_id.cmp(&b.segment_id))
                .then(a.doc_id.cmp(&b.doc_id))
        });
        for (limit, offset) in [(1, 0), (10, 0), (1000, 0), (17, 7)] {
            let end = (limit + offset).min(expected.len());
            let wanted = &expected[offset.min(end)..end];
            let (native, _) = searcher
                .search_with_offset_and_count_sync(&query, limit, offset)
                .unwrap();
            let (asynchronous, _) = searcher
                .search_with_offset_and_count(&query, limit, offset)
                .await
                .unwrap();
            assert_eq!(native, asynchronous, "sync/async term={term}");
            assert_eq!(native.len(), wanted.len());
            for (actual, expected) in native.iter().zip(wanted) {
                assert_eq!(
                    (actual.segment_id, actual.doc_id),
                    (expected.segment_id, expected.doc_id),
                    "term={term} limit={limit} offset={offset}"
                );
                assert_eq!(actual.score, expected.score);
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn ranked_terms_keep_document_order_for_equal_bm25_scores() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field("text", true, false);
    let config = IndexConfig {
        posting_impact_bounds: true,
        num_threads: 1,
        num_indexing_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    // Average length is exactly 3. BM25 gives tf=2/length=3 and
    // tf=1/length=1 equal scores; document id must break that tie.
    for row in 0..384 {
        let mut doc = Document::new();
        doc.add_text(
            text,
            [
                "alpha alpha padding",
                "alpha",
                "alpha padding padding padding padding",
            ][row % 3],
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = TermQuery::text(text, "alpha");
    let mut exact = TopKCollector::new(10);
    collect_segment(&searcher.segment_readers()[0], &query, &mut exact)
        .await
        .unwrap();
    let expected = exact.into_sorted_results();
    assert_eq!(expected[0].doc_id, 0);
    for k in [1, 10] {
        let (ranked, _) = searcher
            .search_with_offset_and_count_sync(&query, k, 0)
            .unwrap();
        assert_eq!(
            ranked.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
            expected[..k].iter().map(|h| h.doc_id).collect::<Vec<_>>()
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn ratio_bounds_use_saturated_scoring_lengths_after_reopen_and_merge() {
    for impacts in [false, true] {
        ratio_bounds_use_saturated_scoring_lengths_after_reopen_and_merge_with(impacts).await;
    }
}

async fn ratio_bounds_use_saturated_scoring_lengths_after_reopen_and_merge_with(
    posting_impact_bounds: bool,
) {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    let config = IndexConfig {
        posting_ratio_bounds: true,
        posting_impact_bounds,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for segment in 0..2 {
        for row in 0..300 {
            let mut doc = Document::new();
            let text = if row == 299 {
                "needle ".repeat(100_000 + segment)
            } else {
                "needle ".repeat(1 + row % 31) + &"padding ".repeat(row % 71)
            };
            doc.add_text(field, text);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let query = TermQuery::text(field, "needle");
    assert_eq!(searcher.num_segments(), 1);
    let segment = &searcher.segment_readers()[0];
    assert!(
        segment
            .get_postings_sync(field, b"needle")
            .unwrap()
            .unwrap()
            .has_ratio_bounds()
    );
    let mut all = TopKCollector::new(600);
    collect_segment(segment, &query, &mut all).await.unwrap();
    let expected = all.into_sorted_results();
    for k in [1, 10, 100, 1000] {
        let (actual, _) = searcher
            .search_with_offset_and_count_sync(&query, k, 0)
            .unwrap();
        assert_eq!(actual.len(), k.min(600));
        for (got, want) in actual.iter().zip(&expected) {
            assert_eq!(
                (got.doc_id, got.score.to_bits()),
                (want.doc_id, want.score.to_bits())
            );
        }
    }
}
