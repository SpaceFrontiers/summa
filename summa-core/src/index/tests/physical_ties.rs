//! Constant-score queries on reordered fields rank ties in physical order
//! (`docs/physical-tie-order.md`).

use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::query::{BooleanQuery, PrefixQuery, Query, RegexQuery, TermQuery, WildcardQuery};
use crate::{Index, IndexConfig, IndexWriter};

#[tokio::test]
async fn constant_score_queries_on_reordered_fields_rank_ties_in_physical_order() {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field_with_tokenizer("id", true, true, "raw");
    schema.set_primary_key(id);
    let body = schema.add_text_field("body", true, false);
    schema.set_reorder(body, true);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    let text = |row: u32| match row % 3 {
        0 => "alpha beta gamma",
        1 => "alpine omega",
        _ => "lambda beta",
    };
    for row in 0..3000u32 {
        let mut doc = Document::new();
        doc.add_text(id, row.to_string());
        doc.add_text(body, text(row));
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.reorder().await.unwrap();
    let reader = Index::open(dir.clone(), config.clone()).await.unwrap();
    let physical_first = {
        let reader = reader.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let map = searcher.segment_readers()[0]
            .chunk_map(body)
            .unwrap()
            .clone();
        (0..3000u32)
            .map(|slot| map.doc_id(slot))
            .collect::<Vec<_>>()
    };
    // Delete the physically first matching document.
    let deleted = *physical_first.iter().find(|&&doc| doc % 3 != 2).unwrap();
    writer.delete_primary_key(&deleted.to_string()).unwrap();
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_segments(), 1);
    let segment = &searcher.segment_readers()[0];
    let map = segment.chunk_map(body).unwrap();
    assert!(map.is_document_map());
    // Document IDs here equal the row numbers: one segment, written in order.
    let matches: Vec<u32> = (0..3000u32)
        .map(|slot| map.doc_id(slot))
        .filter(|&doc| doc % 3 != 2 && doc != deleted)
        .collect();
    assert_ne!(
        matches[..10],
        (0..15).filter(|d| d % 3 != 2).collect::<Vec<_>>()[..10]
    );

    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(PrefixQuery::text(body, "alp")),
        Box::new(WildcardQuery::new(body, "al*").unwrap()),
        Box::new(RegexQuery::new(body, "al(ph|pin)[ae]").unwrap()),
        Box::new(BooleanQuery::new().should(PrefixQuery::text(body, "alp"))),
        Box::new(BooleanQuery::new().must(PrefixQuery::text(body, "alp"))),
    ];
    for query in &queries {
        let docs = |hits: Vec<crate::query::SearchResult>| {
            hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>()
        };
        let first = docs(searcher.search(query.as_ref(), 10).await.unwrap());
        assert_eq!(first, matches[..10], "{query}");
        // Offset pages concatenate to a longer page.
        let second = docs(
            searcher
                .search_with_offset(query.as_ref(), 10, 10)
                .await
                .unwrap(),
        );
        assert_eq!(second, matches[10..20], "{query}");
        let long = docs(searcher.search(query.as_ref(), 20).await.unwrap());
        assert_eq!(long, [first, second].concat(), "{query}");
        #[cfg(feature = "sync")]
        {
            let (hits, _) = searcher
                .search_with_offset_and_count_sync(query.as_ref(), 10, 0)
                .unwrap();
            assert_eq!(docs(hits), matches[..10], "{query} (sync)");
        }
    }

    // A scored clause keeps logical ties: every match of this conjunction
    // has the same text, hence the same score.
    let mixed = BooleanQuery::new()
        .must(PrefixQuery::text(body, "alp"))
        .must(TermQuery::text(body, "beta"));
    let hits = searcher.search(&mixed, 10).await.unwrap();
    let logical: Vec<u32> = (0..3000u32)
        .filter(|&doc| doc % 3 == 0 && doc != deleted)
        .take(10)
        .collect();
    assert_eq!(
        hits.iter().map(|hit| hit.doc_id).collect::<Vec<_>>(),
        logical
    );
}

#[tokio::test]
async fn physical_ties_merge_across_segments_by_segment_then_physical_order() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", true, false);
    schema.set_reorder(body, true);
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(crate::merge::NoMergePolicy),
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for segment in 0..2u32 {
        for row in 0..1500u32 {
            let mut doc = Document::new();
            let text = match (row + segment) % 3 {
                0 => "alpha beta gamma",
                1 => "alpine omega",
                _ => "lambda beta",
            };
            doc.add_text(body, text);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.reorder().await.unwrap();
    writer.shutdown().await.unwrap();

    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    assert_eq!(searcher.num_segments(), 2);
    let query = PrefixQuery::text(body, "alp");
    // Expected: segments by ID, each segment's matches in physical order.
    let mut segments: Vec<_> = searcher.segment_readers().iter().collect();
    segments.sort_by_key(|segment| segment.meta().id);
    let mut expected = Vec::new();
    for segment in segments {
        let map = segment.chunk_map(body).unwrap();
        assert!(map.is_document_map());
        let mut physical = Vec::new();
        let mut scorer = query
            .scorer_with_options(
                segment,
                usize::MAX / 2,
                crate::query::ScorerOptions {
                    physical_text_field: Some(body),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        while scorer.doc() != crate::structures::TERMINATED {
            physical.push((segment.meta().id, map.doc_id(scorer.doc())));
            scorer.advance();
        }
        expected.extend(physical);
    }
    assert!(
        expected
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 > pair[1].1),
        "the reorder must permute the matches"
    );
    let hits = searcher.search(&query, 2500).await.unwrap();
    assert_eq!(
        hits.iter()
            .map(|hit| (hit.segment_id, hit.doc_id))
            .collect::<Vec<_>>(),
        expected
    );
    let page = searcher.search_with_offset(&query, 10, 995).await.unwrap();
    assert_eq!(
        page.iter()
            .map(|hit| (hit.segment_id, hit.doc_id))
            .collect::<Vec<_>>(),
        expected[995..1005]
    );
}
