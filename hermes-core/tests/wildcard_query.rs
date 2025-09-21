//! Whole-term wildcard behavior through the public search APIs.
use hermes_core::query::{
    BooleanQuery, CountCollector, PrefixQuery, Query, WildcardQuery, collect_segment,
};
use hermes_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use hermes_core::{Document, RamDirectory, SchemaBuilder};
use std::sync::Arc;

#[tokio::test]
async fn wildcard_filters_match_whole_unicode_terms_and_deduplicate_documents() {
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let chunks = schema.add_text_field("chunks", true, false);
    schema.set_chunked(chunks, true);
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for terms in [
        vec!["there", "theme"],
        vec!["the"],
        vec!["them"],
        vec!["é"],
        vec!["🦀"],
        vec!["a*b"],
        vec!["a?b"],
        vec!["xtherey"],
    ] {
        let mut doc = Document::new();
        for term in terms {
            doc.add_text(field, term);
        }
        builder.add_document(doc).unwrap();
    }
    let id = SegmentId::new();
    builder.build(&directory, id, None).await.unwrap();
    let reader = SegmentReader::open(&directory, id, schema.clone(), 16)
        .await
        .unwrap();
    for (pattern, expected) in [
        ("th*e", vec![0, 1]),
        ("*ere", vec![0]),
        ("?", vec![3, 4]),
        (r"a\*b", vec![5]),
        (r"a\?b", vec![6]),
        ("th?", vec![1]),
        ("*", (0..8).collect()),
        ("absent*", vec![]),
        ("", vec![]),
    ] {
        let query = WildcardQuery::new(field, pattern).unwrap();
        let mut scorer = query.scorer(&reader, 10).await.unwrap();
        let mut actual = Vec::new();
        while scorer.doc() != hermes_core::structures::TERMINATED {
            actual.push(scorer.doc());
            assert_eq!(scorer.score(), 1.0);
            scorer.advance();
        }
        assert_eq!(actual, expected, "{pattern}");
        let mut count = CountCollector::new();
        collect_segment(&reader, &query, &mut count).await.unwrap();
        assert_eq!(count.count() as usize, expected.len());
        #[cfg(feature = "sync")]
        {
            let mut scorer = query.scorer_sync(&reader, 10).unwrap();
            let mut sync = Vec::new();
            while scorer.doc() != hermes_core::structures::TERMINATED {
                sync.push(scorer.doc());
                scorer.advance();
            }
            assert_eq!(sync, expected);
        }
    }
    let unsupported = WildcardQuery::new(chunks, "*").unwrap();
    assert!(
        unsupported
            .scorer(&reader, 10)
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("chunked")
    );
    #[cfg(feature = "sync")]
    assert!(
        unsupported
            .scorer_sync(&reader, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("chunked")
    );
    let parser = hermes_core::dsl::QueryLanguageParser::new(
        schema,
        vec![field],
        Arc::new(hermes_core::tokenizer::TokenizerRegistry::new()),
    );
    for text in [
        r#"tag:wildcard("th*e")"#,
        r#"wildcard("th*e")"#,
        "tag:th*e",
        "th*e",
    ] {
        let query = parser.parse_strict(text).unwrap();
        let mut count = CountCollector::new();
        collect_segment(&reader, query.as_ref(), &mut count)
            .await
            .unwrap();
        assert_eq!(count.count(), 2);
    }
    let wildcard = WildcardQuery::new(field, "th*").unwrap();
    let prefix = PrefixQuery::text(field, "th");
    let mut left = wildcard.scorer(&reader, 10).await.unwrap();
    let mut right = prefix.scorer(&reader, 10).await.unwrap();
    while left.doc() != hermes_core::structures::TERMINATED {
        assert_eq!(left.doc(), right.doc());
        left.advance();
        right.advance();
    }
    assert_eq!(left.doc(), right.doc());
    let query = BooleanQuery::new()
        .must(wildcard)
        .must(WildcardQuery::new(field, "*ere").unwrap());
    let mut count = CountCollector::new();
    collect_segment(&reader, &query, &mut count).await.unwrap();
    assert_eq!(count.count(), 1);
    assert!(WildcardQuery::new(field, "dangling\\").is_err());
}

#[tokio::test]
async fn expanded_term_filters_preserve_logical_ids_after_rgb_reordering() {
    use hermes_core::{Index, IndexConfig, IndexWriter};
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field("text", true, false);
    schema.set_reorder(field, true);
    let dir = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        merge_policy: Box::new(hermes_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for id in 0..2048 {
        let mut doc = Document::new();
        doc.add_text(
            field,
            if id % 2 == 0 {
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
    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(PrefixQuery::text(field, "alp")),
        Box::new(WildcardQuery::new(field, "a*ha").unwrap()),
        Box::new(hermes_core::RegexQuery::new(field, "a.*ha").unwrap()),
    ];
    let expected: Vec<_> = (0..2048).step_by(2).collect();
    for query in queries {
        let mut scorer = query.scorer(segment, 10).await.unwrap();
        let mut actual = Vec::new();
        while scorer.doc() != hermes_core::structures::TERMINATED {
            actual.push(scorer.doc());
            scorer.advance();
        }
        assert_eq!(actual, expected, "{query}");
        #[cfg(feature = "sync")]
        {
            let mut scorer = query.scorer_sync(segment, 10).unwrap();
            for &doc in &expected {
                assert_eq!(scorer.doc(), doc);
                scorer.advance();
            }
            assert_eq!(scorer.doc(), hermes_core::structures::TERMINATED);
            if let Some(bitset) = query.as_doc_bitset(segment) {
                for doc in 0..2048 {
                    assert_eq!(bitset.contains(doc), doc % 2 == 0);
                }
            }
        }
    }
}

#[tokio::test]
async fn wildcard_expansion_errors_instead_of_returning_partial_hits() {
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    let mut doc = Document::new();
    for term in 0..1025 {
        doc.add_text(field, format!("term{term:04}"));
    }
    builder.add_document(doc).unwrap();
    let id = SegmentId::new();
    builder.build(&directory, id, None).await.unwrap();
    let reader = SegmentReader::open(&directory, id, schema, 16)
        .await
        .unwrap();
    let query = WildcardQuery::new(field, "term*").unwrap();
    assert!(
        query
            .scorer(&reader, 10)
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("more than 1024 terms")
    );
    assert!(
        query
            .count_estimate(&reader)
            .await
            .unwrap_err()
            .to_string()
            .contains("more than 1024 terms")
    );
    #[cfg(feature = "sync")]
    assert!(
        query
            .scorer_sync(&reader, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("more than 1024 terms")
    );
    assert!(WildcardQuery::new(field, "x".repeat(1025)).is_err());
}
