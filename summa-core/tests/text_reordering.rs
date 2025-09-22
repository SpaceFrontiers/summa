#![cfg(feature = "native")]

use summa_core::directories::Directory;
use summa_core::query::{
    BooleanQuery, CountCollector, PhraseQuery, Query, TermQuery, TopKCollector, collect_segment,
};
use summa_core::{Document, Index, IndexConfig, IndexWriter, RamDirectory, Schema};

type Snapshot = (u64, Vec<(u32, u32, serde_json::Value)>);

#[tokio::test]
async fn independent_text_permutations_keep_cross_field_scores_and_positions() {
    let mut schema = Schema::builder();
    let left = schema.add_text_field("left", true, false);
    let right = schema.add_text_field("right", true, false);
    for field in [left, right] {
        schema.set_reorder(field, true);
        schema.set_positions(field, summa_core::dsl::PositionMode::TokenPosition);
    }
    let dir = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for i in 0..1024 {
        let mut doc = Document::new();
        for (field, group) in [(left, i % 2), (right, (i / 2) % 2)] {
            if i % 19 != 0 {
                doc.add_text(
                    field,
                    if group == 0 {
                        "alpha beta gamma delta"
                    } else {
                        "lambda mu nu omicron"
                    },
                );
            }
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(left, "alpha"))
                .must(TermQuery::text(right, "alpha")),
        ),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(left, "lambda"))
                .should(TermQuery::text(right, "alpha")),
        ),
        Box::new(
            BooleanQuery::new()
                .must(PhraseQuery::text(left, "alpha beta"))
                .must_not(TermQuery::text(right, "lambda")),
        ),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(left, "alpha"))
                .should(summa_core::query::BoostQuery::new(
                    TermQuery::text(right, "alpha"),
                    0.7,
                )),
        ),
    ];
    let expected = snapshot(dir.clone(), config.clone(), &queries).await;
    writer.reorder().await.unwrap();
    let index = Index::open(dir.clone(), config.clone()).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    let a = segment.chunk_map(left).unwrap();
    let b = segment.chunk_map(right).unwrap();
    assert!((0..a.num_chunks()).any(|slot| a.doc_id(slot) != b.doc_id(slot)));
    assert_snapshots(&snapshot(dir, config, &queries).await, &expected);
    writer.shutdown().await.unwrap();
}

#[cfg(all(feature = "sync", feature = "query-diagnostics"))]
#[tokio::test(flavor = "current_thread")]
async fn reordered_composition_streams_blocks_without_replaying_logical_ids() {
    use summa_core::query::collect_segment_with_limit;
    use summa_core::search_diagnostics::capture;
    let mut schema = Schema::builder();
    let field = schema.add_text_field("text", true, false);
    schema.set_reorder(field, true);
    schema.set_positions(field, summa_core::dsl::PositionMode::TokenPosition);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        posting_ratio_bounds: true,
        compact_text: true,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for i in 0..2048 {
        let mut doc = Document::new();
        doc.add_text(
            field,
            if i % 2 == 0 {
                "alpha beta gamma delta epsilon"
            } else {
                "lambda mu nu xi omicron"
            },
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.reorder().await.unwrap();
    writer.shutdown().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    let map = segment.chunk_map(field).unwrap();
    assert!((1..map.num_chunks()).any(|slot| map.doc_id(slot - 1) > map.doc_id(slot)));
    for (query, count) in [
        (
            BooleanQuery::new()
                .must(TermQuery::text(field, "alpha"))
                .must(TermQuery::text(field, "beta")),
            1024,
        ),
        (
            BooleanQuery::new()
                .should(TermQuery::text(field, "alpha"))
                .should(TermQuery::text(field, "lambda")),
            2048,
        ),
        (
            BooleanQuery::new()
                .must(TermQuery::text(field, "alpha"))
                .must_not(TermQuery::text(field, "lambda")),
            1024,
        ),
    ] {
        let mut warm = CountCollector::new();
        collect_segment(segment, &query, &mut warm).await.unwrap();
        let mut counts = CountCollector::new();
        let (result, work) = capture(collect_segment(segment, &query, &mut counts)).await;
        result.unwrap();
        assert_eq!(counts.count(), count);
        assert_eq!(
            work.tf_blocks, 0,
            "count must not probe frequencies: {query}"
        );
        assert!(
            work.doc_blocks <= 40,
            "count must stream blocks once: {query}: {work:?}"
        );
        let mut top = TopKCollector::new(10);
        let (result, work) =
            capture(collect_segment_with_limit(segment, &query, &mut top, 10)).await;
        result.unwrap();
        assert!(
            work.doc_blocks <= 40,
            "ranking must not replay matches: {query}: {work:?}"
        );
        let hits = top.into_sorted_results();
        let ids: Vec<_> = hits.iter().map(|hit| hit.doc_id).collect();
        let expected: Vec<_> = (0..10)
            .map(|i| if count == 2048 { i } else { i * 2 })
            .collect();
        assert_eq!(ids, expected, "stable ties: {query}");
        let mut sync_top = TopKCollector::new(10);
        summa_core::query::collect_segment_with_limit_sync(segment, &query, &mut sync_top, 10)
            .unwrap();
        assert_eq!(
            sync_top
                .into_sorted_results()
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            hits.into_iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect::<Vec<_>>()
        );
    }
    let phrase = PhraseQuery::text(field, "alpha beta");
    let mut count = CountCollector::new();
    let (result, work) = capture(collect_segment(segment, &phrase, &mut count)).await;
    result.unwrap();
    assert_eq!(count.count(), 1024);
    assert!(
        work.doc_blocks <= 40,
        "phrase must not replay matches: {work:?}"
    );
    let expired = summa_core::query::SharedThreshold::for_limit(10)
        .with_deadline(Some(std::time::Instant::now()));
    let (result, work) = capture(summa_core::query::search_segment_shared(
        segment,
        &phrase,
        10,
        true,
        expired.clone(),
    ))
    .await;
    assert!(result.unwrap().0.is_empty());
    assert!(expired.truncated());
    assert_eq!(
        work.doc_blocks, 0,
        "expired mapped query must not start traversal"
    );
}

async fn snapshot(
    directory: RamDirectory,
    config: IndexConfig,
    queries: &[Box<dyn Query>],
) -> Vec<Snapshot> {
    let index = Index::open(directory, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment = &searcher.segment_readers()[0];
    let mut output = Vec::new();
    for query in queries {
        let mut top = TopKCollector::with_positions(2000);
        let mut count = CountCollector::new();
        collect_segment(segment, query.as_ref(), &mut (&mut top, &mut count))
            .await
            .unwrap();
        output.push((
            count.count(),
            top.into_sorted_results()
                .into_iter()
                .map(|hit| {
                    (
                        hit.doc_id,
                        hit.score.to_bits(),
                        serde_json::to_value(hit.positions).unwrap(),
                    )
                })
                .collect(),
        ));
    }
    output
}

#[tokio::test]
async fn plain_text_rgb_preserves_scores_positions_and_stable_documents() {
    let schema = |reorder| {
        let mut builder = Schema::builder();
        let text = builder.add_text_field("text", true, true);
        builder.set_multi(text, true);
        builder.set_positions(text, summa_core::dsl::PositionMode::TokenPosition);
        builder.set_reorder(text, reorder);
        let marker = builder.add_text_field("marker", true, true);
        builder.set_fast(marker, true);
        builder.set_primary_key(marker);
        (builder.build(), text, marker)
    };
    let (control_schema, text, marker) = schema(false);
    let (mapped_schema, _, _) = schema(true);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let control = RamDirectory::new();
    let mapped = RamDirectory::new();
    let mut control_writer = IndexWriter::create(control.clone(), control_schema, config.clone())
        .await
        .unwrap();
    let mut mapped_writer = IndexWriter::create(mapped.clone(), mapped_schema, config.clone())
        .await
        .unwrap();
    control_writer.init_primary_key_dedup().await.unwrap();
    mapped_writer.init_primary_key_dedup().await.unwrap();
    for doc_id in 0..600 {
        let mut doc = Document::new();
        doc.add_text(marker, format!("row{doc_id}"));
        if doc_id % 17 != 0 {
            let vocabulary = if doc_id % 2 == 0 {
                "alpha beta gamma delta epsilon"
            } else {
                "lambda mu nu xi omicron"
            };
            doc.add_text(text, format!("{vocabulary} ").repeat(doc_id % 11 + 1));
            if doc_id % 3 == 0 {
                doc.add_text(text, "alpha beta");
            }
        }
        control_writer.add_document(doc.clone()).unwrap();
        mapped_writer.add_document(doc).unwrap();
    }
    control_writer.commit().await.unwrap();
    mapped_writer.commit().await.unwrap();
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(TermQuery::text(text, "alpha")),
        Box::new(PhraseQuery::text(text, "alpha beta")),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(text, "alpha"))
                .should(TermQuery::text(text, "lambda")),
        ),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(text, "alpha"))
                .must(TermQuery::text(text, "beta")),
        ),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(text, "lambda"))
                .must_not(TermQuery::text(text, "alpha")),
        ),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(text, "alpha"))
                .should(TermQuery::text(text, "lambda")),
        ),
        Box::new(
            BooleanQuery::new()
                .must(
                    BooleanQuery::new()
                        .should(TermQuery::text(text, "alpha"))
                        .should(TermQuery::text(text, "lambda")),
                )
                .must(PhraseQuery::text(text, "alpha beta")),
        ),
        Box::new(summa_core::query::BoostQuery::new(
            PhraseQuery::text(text, "alpha beta"),
            -2.0,
        )),
        Box::new(BooleanQuery::new().must_not(TermQuery::text(text, "alpha"))),
    ];
    let expected = snapshot(control.clone(), config.clone(), &queries).await;
    let unchanged = unchanged_payloads(mapped.clone(), config.clone()).await;
    assert_snapshots(
        &snapshot(mapped.clone(), config.clone(), &queries).await,
        &expected,
    );
    mapped_writer.reorder().await.unwrap();
    assert_eq!(
        unchanged_payloads(mapped.clone(), config.clone()).await,
        unchanged
    );
    let index = Index::open(mapped.clone(), config.clone()).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let map = searcher.segment_readers()[0]
        .chunk_map(text)
        .expect("plain text reorder must produce a field-local mapping");
    assert_eq!(map.num_chunks(), 600);
    assert!((1..map.num_chunks()).any(|slot| map.doc_id(slot - 1) > map.doc_id(slot)));
    for (query, (_, expected)) in queries.iter().zip(&expected) {
        for limit in [1, 10, 100, 1000] {
            let mut top = TopKCollector::new(limit);
            summa_core::query::collect_segment_with_limit(
                &searcher.segment_readers()[0],
                query.as_ref(),
                &mut top,
                limit,
            )
            .await
            .unwrap();
            let actual: Vec<_> = top
                .into_sorted_results()
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect();
            let expected: Vec<_> = expected
                .iter()
                .take(limit)
                .map(|hit| (hit.0, hit.1))
                .collect();
            assert_eq!(actual, expected, "ranked {query}, limit {limit}");
        }
    }
    assert_snapshots(
        &snapshot(mapped.clone(), config.clone(), &queries).await,
        &expected,
    );
    for doc_id in 600..620 {
        let mut doc = Document::new();
        doc.add_text(marker, format!("row{doc_id}"));
        doc.add_text(text, "alpha beta lambda");
        control_writer.add_document(doc.clone()).unwrap();
        mapped_writer.add_document(doc).unwrap();
    }
    control_writer.commit().await.unwrap();
    mapped_writer.commit().await.unwrap();
    control_writer.force_merge().await.unwrap();
    mapped_writer.force_merge().await.unwrap();
    let expected = snapshot(control.clone(), config.clone(), &queries).await;
    assert_snapshots(
        &snapshot(mapped.clone(), config.clone(), &queries).await,
        &expected,
    );
    for key in ["row2", "row5", "row17", "row612"] {
        control_writer.delete_primary_key(key).unwrap();
        mapped_writer.delete_primary_key(key).unwrap();
    }
    control_writer.commit().await.unwrap();
    mapped_writer.commit().await.unwrap();
    let expected = snapshot(control.clone(), config.clone(), &queries).await;
    assert_snapshots(
        &snapshot(mapped.clone(), config.clone(), &queries).await,
        &expected,
    );
    assert_eq!(control_writer.compact(64 * 1024 * 1024).await.unwrap(), 1);
    assert_eq!(mapped_writer.compact(64 * 1024 * 1024).await.unwrap(), 1);
    assert_snapshots(
        &snapshot(mapped, config.clone(), &queries).await,
        &snapshot(control, config, &queries).await,
    );
    control_writer.shutdown().await.unwrap();
    mapped_writer.shutdown().await.unwrap();
}

fn assert_snapshots(actual: &[Snapshot], expected: &[Snapshot]) {
    for (query, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.0, expected.0, "query {query} count");
        assert_eq!(actual.1.len(), expected.1.len(), "query {query} hits");
        for (at, (actual, expected)) in actual.1.iter().zip(&expected.1).enumerate() {
            assert_eq!(actual, expected, "query {query}, hit {at}");
        }
    }
}

async fn unchanged_payloads(directory: RamDirectory, config: IndexConfig) -> Vec<Vec<u8>> {
    let index = Index::open(directory.clone(), config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let files = summa_core::segment::SegmentFiles::new(searcher.segment_readers()[0].meta().id);
    let mut bytes = Vec::new();
    for path in [files.store, files.fast] {
        bytes.push(
            directory
                .open_read(&path)
                .await
                .unwrap()
                .read_bytes()
                .await
                .unwrap()
                .as_slice()
                .to_vec(),
        );
    }
    bytes
}

#[tokio::test]
async fn mapped_ranked_text_breaks_score_ties_by_document_id() {
    let mut schema = Schema::builder();
    let field = schema.add_text_field("text", true, true);
    schema.set_reorder(field, true);
    let schema = schema.build();
    let dir = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    for doc in 0..2048 {
        let mut row = Document::new();
        row.add_text(
            field,
            if doc % 2 == 0 {
                "common alpha beta"
            } else {
                "common gamma delta"
            },
        );
        writer.add_document(row).unwrap();
    }
    writer.commit().await.unwrap();
    writer.reorder().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(TermQuery::text(field, "common")),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(field, "alpha"))
                .should(TermQuery::text(field, "gamma")),
        ),
    ];
    for query in queries {
        let mut oracle = TopKCollector::new(2048);
        collect_segment(segment, query.as_ref(), &mut oracle)
            .await
            .unwrap();
        let oracle = oracle.into_sorted_results();
        for limit in [1, 10, 100] {
            let mut top = TopKCollector::new(limit);
            summa_core::query::collect_segment_with_limit(segment, query.as_ref(), &mut top, limit)
                .await
                .unwrap();
            let actual: Vec<_> = top
                .into_sorted_results()
                .into_iter()
                .map(|h| (h.doc_id, h.score.to_bits()))
                .collect();
            let expected: Vec<_> = oracle
                .iter()
                .take(limit)
                .map(|h| (h.doc_id, h.score.to_bits()))
                .collect();
            assert_eq!(actual, expected, "{query}, limit {limit}");
        }
    }
    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn merge_time_text_rgb_publishes_the_completed_physical_order() {
    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", true, false);
    schema.set_reorder(text, true);
    schema.set_reorder_on_merge(true);
    let dir = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config)
        .await
        .unwrap();
    for batch in 0..2 {
        for doc in 0..256 {
            let mut row = Document::new();
            row.add_text(
                text,
                if (doc + batch) % 2 == 0 {
                    "alpha beta"
                } else {
                    "gamma delta"
                },
            );
            writer.add_document(row).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.force_merge().await.unwrap();
    let metadata = summa_core::index::IndexMetadata::load(&dir).await.unwrap();
    assert_eq!(metadata.segment_metas.len(), 1);
    assert!(
        metadata.segment_metas.values().next().unwrap().reordered,
        "merge-time text BP must complete before publication"
    );
    let index = Index::open(dir.clone(), IndexConfig::default())
        .await
        .unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let map = searcher.segment_readers()[0].chunk_map(text).unwrap();
    assert!((1..map.num_chunks()).any(|slot| map.doc_id(slot - 1) > map.doc_id(slot)));
    writer.reorder().await.unwrap();
    let metadata = summa_core::index::IndexMetadata::load(&dir).await.unwrap();
    assert!(metadata.segment_metas.values().next().unwrap().reordered);
    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn merge_time_text_budget_failure_keeps_sources_and_removes_unpublished_output() {
    let mut schema = Schema::builder();
    let text = schema.add_text_field("text", true, true);
    schema.set_reorder(text, true);
    schema.set_reorder_on_merge(true);
    let dir = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        bp_memory_budget_bytes: 1,
        merge_policy: Box::new(summa_core::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for _ in 0..2 {
        for _ in 0..256 {
            let mut doc = Document::new();
            doc.add_text(text, "alpha beta");
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let before = summa_core::index::IndexMetadata::load(&dir).await.unwrap();
    let source_files = dir.list_files(std::path::Path::new("")).await.unwrap();
    let error = writer.force_merge().await.unwrap_err();
    assert!(error.to_string().contains("budget"), "{error}");
    let after = summa_core::index::IndexMetadata::load(&dir).await.unwrap();
    let mut before_ids: Vec<_> = before.segment_metas.keys().collect();
    let mut after_ids: Vec<_> = after.segment_metas.keys().collect();
    before_ids.sort();
    after_ids.sort();
    assert_eq!(before_ids, after_ids);
    let segment_files = |files: Vec<std::path::PathBuf>| {
        files
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("seg_")
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(
        segment_files(source_files),
        segment_files(dir.list_files(std::path::Path::new("")).await.unwrap())
    );
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let mut total = 0;
    for segment in searcher.segment_readers() {
        let mut count = CountCollector::new();
        collect_segment(segment, &TermQuery::text(text, "alpha"), &mut count)
            .await
            .unwrap();
        total += count.count();
    }
    assert_eq!(total, 512);
    writer.shutdown().await.unwrap();
}
