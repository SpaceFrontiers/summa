#![cfg(feature = "sync")]

use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::{BooleanQuery, Query, TermQuery, TopKCollector, collect_segment};
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn ratio_bounds_preserve_bounded_chunk_nomination_and_ordinals_after_merge() {
    for impacts in [false, true] {
        bounds_preserve_chunk_nomination(impacts).await;
    }
}

async fn bounds_preserve_chunk_nomination(posting_impact_bounds: bool) {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    schema.set_chunked(field, true);
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
    for part in 0..2 {
        for row in 0..256 {
            let mut doc = Document::new();
            for ordinal in 0..4 {
                doc.add_text(
                    field,
                    "common ".repeat(1 + (row + ordinal) % 11)
                        + &"padding ".repeat((row * 7 + ordinal * 19 + part) % 97)
                        + if (row + ordinal) % 3 == 0 {
                            " rare"
                        } else {
                            ""
                        },
                );
            }
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
    assert_eq!(searcher.num_segments(), 1);
    let segment = &searcher.segment_readers()[0];
    assert!(
        segment
            .get_postings_sync(field, b"common")
            .unwrap()
            .unwrap()
            .has_ratio_bounds()
    );
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(TermQuery::text(field, "common")),
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(field, "common"))
                .should(TermQuery::text(field, "rare")),
        ),
    ];
    for query in queries {
        let mut all = TopKCollector::with_positions(512);
        collect_segment(segment, query.as_ref(), &mut all)
            .await
            .unwrap();
        let expected = all.into_sorted_results();
        // Standalone chunked search intentionally nominates the top 2*k
        // chunks before MaxP folding. Exhaustive membership exposes more
        // ordinals, so apply that documented budget to the ordinal oracle.
        let mut chunks: Vec<_> = expected
            .iter()
            .flat_map(|hit| {
                hit.positions.iter().flat_map(move |(field, values)| {
                    values
                        .iter()
                        .map(move |p| (hit.doc_id, *field, p.position, p.score))
                })
            })
            .collect();
        chunks.sort_unstable_by(|a, b| b.3.total_cmp(&a.3).then((a.0, a.2).cmp(&(b.0, b.2))));
        for k in [10, 100, 1000] {
            let candidate_count = 2 * k.min(segment.num_docs() as usize);
            let mut seen = std::collections::HashSet::new();
            let nominated: Vec<_> = chunks
                .iter()
                .take(candidate_count)
                .filter(|chunk| seen.insert(chunk.0))
                .take(k)
                .map(|chunk| (chunk.0, chunk.3.to_bits()))
                .collect();
            let (native, _) = searcher
                .search_with_offset_and_count_sync(query.as_ref(), k, 0)
                .unwrap();
            let (asynchronous, _) = searcher
                .search_with_offset_and_count(query.as_ref(), k, 0)
                .await
                .unwrap();
            assert_eq!(native, asynchronous);
            assert_eq!(native.len(), nominated.len());
            let (positioned, _) = searcher
                .search_with_positions(query.as_ref(), k)
                .await
                .unwrap();
            for (got, want) in positioned.iter().zip(&nominated) {
                assert_eq!((got.doc_id, got.score.to_bits()), *want);
                assert!(!got.positions.is_empty());
                let positions = |hit: &summa_core::query::SearchResult| {
                    hit.positions
                        .iter()
                        .flat_map(|(field, values)| {
                            values
                                .iter()
                                .map(move |p| (*field, p.position, p.score.to_bits()))
                        })
                        .collect::<Vec<_>>()
                };
                let mut wanted_ordinals: Vec<_> = chunks
                    .iter()
                    .take(candidate_count)
                    .filter(|chunk| chunk.0 == got.doc_id)
                    .map(|chunk| (chunk.1, chunk.2, chunk.3.to_bits()))
                    .collect();
                wanted_ordinals.sort_unstable_by_key(|entry| (entry.0, entry.1));
                let mut actual_ordinals = positions(got);
                actual_ordinals.sort_unstable_by_key(|entry| (entry.0, entry.1));
                assert_eq!(actual_ordinals, wanted_ordinals);
            }
            for (got, want) in native.iter().zip(&nominated) {
                assert_eq!((got.doc_id, got.score.to_bits()), *want);
            }
        }
    }
}
