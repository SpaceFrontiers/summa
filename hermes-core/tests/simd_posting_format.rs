#![cfg(feature = "sync")]

use std::collections::BTreeMap;

use hermes_core::dsl::PositionMode;
use hermes_core::index::{Index, IndexConfig, IndexWriter};
use hermes_core::query::{BooleanQuery, PhraseQuery, Query, TermQuery};
use hermes_core::structures::PostingCodec;
use hermes_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn mixed_posting_formats_preserve_scores_positions_and_deleted_rows_across_maintenance() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field_with_tokenizer("id", true, true, "raw");
    schema.set_primary_key(id);
    let body = schema.add_text_field("body", true, false);
    schema.set_positions(body, PositionMode::TokenPosition);
    let chunks = schema.add_text_field("chunks", true, false);
    schema.set_chunked(chunks, true);
    schema.set_positions(chunks, PositionMode::TokenPosition);
    schema.set_reorder(chunks, true);
    let multi = schema.add_text_field("multi", true, false);
    schema.set_multi(multi, true);
    schema.set_positions(multi, PositionMode::TokenPosition);
    let schema = schema.build();
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(TermQuery::text(body, "alpha")),
        Box::new(PhraseQuery::text(body, "alpha beta")),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(body, "alpha"))
                .must(TermQuery::text(body, "rare")),
        ),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(body, "alpha"))
                .should(TermQuery::text(body, "rare")),
        ),
        Box::new(PhraseQuery::text(chunks, "alpha beta")),
        Box::new(PhraseQuery::text(multi, "alpha beta")),
    ];
    let mut reference = BTreeMap::new();
    for (case, (codecs, impacts)) in [
        (
            [PostingCodec::Rounded, PostingCodec::Rounded],
            [false, false],
        ),
        ([PostingCodec::Simd4x, PostingCodec::Simd4x], [false, false]),
        (
            [PostingCodec::Rounded, PostingCodec::Simd4x],
            [false, false],
        ),
        ([PostingCodec::Rounded, PostingCodec::Rounded], [true, true]),
        ([PostingCodec::Simd4x, PostingCodec::Rounded], [false, true]),
        ([PostingCodec::Rounded, PostingCodec::Simd4x], [true, false]),
    ]
    .into_iter()
    .enumerate()
    {
        let dir = RamDirectory::new();
        let config = IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            posting_ratio_bounds: true,
            posting_validation_cache_bytes: 4096,
            merge_policy: Box::new(hermes_core::merge::NoMergePolicy),
            ..Default::default()
        };
        for (part, codec) in codecs.into_iter().enumerate() {
            let config = IndexConfig {
                posting_codec: Some(codec),
                posting_impact_bounds: impacts[part],
                ..config.clone()
            };
            let mut writer = if part == 0 {
                IndexWriter::create(dir.clone(), schema.clone(), config)
                    .await
                    .unwrap()
            } else {
                IndexWriter::open(dir.clone(), config).await.unwrap()
            };
            for i in part * 263..(part + 1) * 263 {
                let mut doc = Document::new();
                doc.add_text(id, i.to_string());
                if i % 13 != 0 {
                    doc.add_text(
                        body,
                        "alpha beta ".repeat(i % 7 + 1)
                            + &"padding ".repeat(i % 17)
                            + if i % 5 == 0 { "rare" } else { "" },
                    );
                    doc.add_text(chunks, "alpha beta ".repeat(i % 9 + 1));
                    doc.add_text(chunks, "beta padding ".repeat(i % 11 + 1));
                    doc.add_text(multi, "alpha");
                    doc.add_text(multi, if i % 3 == 0 { "alpha beta" } else { "beta" });
                }
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
            writer.shutdown().await.unwrap();
        }
        let config = IndexConfig {
            posting_codec: Some(codecs[1]),
            ..config
        };
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.init_primary_key_dedup().await.unwrap();
        for phase in 0..4 {
            match phase {
                1 => writer.force_merge().await.unwrap(),
                2 => {
                    for i in (0..526).step_by(19) {
                        writer.delete_primary_key(&i.to_string()).unwrap();
                    }
                    writer.commit().await.unwrap();
                    assert!(writer.compact(16 * 1024 * 1024).await.unwrap() > 0);
                }
                3 => writer.reorder().await.unwrap(),
                _ => {}
            }
            let index = Index::open(dir.clone(), config.clone()).await.unwrap();
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            for (q, query) in queries.iter().enumerate() {
                let native = searcher
                    .search_with_offset_and_count_sync(query.as_ref(), 1000, 0)
                    .unwrap();
                let asynchronous = searcher
                    .search_with_count(query.as_ref(), 1000)
                    .await
                    .unwrap();
                assert_eq!(native.1, asynchronous.1);
                if q < 4 {
                    for limit in [1, 10, 100] {
                        let (ranked, _) = searcher
                            .search_with_offset_and_count_sync(query.as_ref(), limit, 0)
                            .unwrap();
                        assert_eq!(
                            ranked,
                            native.0[..limit.min(native.0.len())],
                            "case {case}, phase {phase}, query {q}, k={limit}"
                        );
                    }
                }
                assert_eq!(
                    native
                        .0
                        .iter()
                        .map(|h| (h.segment_id, h.doc_id, h.score.to_bits()))
                        .collect::<Vec<_>>(),
                    asynchronous
                        .0
                        .iter()
                        .map(|h| (h.segment_id, h.doc_id, h.score.to_bits()))
                        .collect::<Vec<_>>()
                );
                let (hits, _) = searcher
                    .search_with_positions(query.as_ref(), 1000)
                    .await
                    .unwrap();
                let mut normalized = BTreeMap::new();
                for hit in hits {
                    let doc = searcher
                        .doc(hit.segment_id, hit.doc_id)
                        .await
                        .unwrap()
                        .unwrap();
                    let key = doc.get_first(id).unwrap().as_text().unwrap().to_owned();
                    let positions: Vec<_> = hit
                        .positions
                        .into_iter()
                        .map(|(f, p)| {
                            (
                                f,
                                p.into_iter()
                                    .map(|p| (p.position, p.score.to_bits()))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    normalized.insert(key, (hit.score.to_bits(), positions));
                }
                assert!(!normalized.is_empty());
                let result = (native.1, normalized);
                if case == 0 {
                    reference.insert((phase, q), result);
                } else {
                    assert_eq!(
                        &result,
                        &reference[&(phase, q)],
                        "case {case}, phase {phase}, query {q}"
                    );
                }
            }
        }
        writer.shutdown().await.unwrap();
    }
}
