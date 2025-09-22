//! Whole-term regular expressions and punctuation through public query parsing.
use std::sync::Arc;
use summa_core::query::{CountCollector, collect_segment};
use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
use summa_core::{Document, QueryLanguageParser, RamDirectory, SchemaBuilder};

#[tokio::test]
async fn regex_queries_match_whole_terms_and_preserve_boolean_composition() {
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for terms in [
        vec!["1999", "2000"],
        vec!["color"],
        vec!["colour"],
        vec!["x1999"],
        vec!["é"],
        vec!["🦀"],
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
    let parser = QueryLanguageParser::new(
        schema,
        vec![field],
        Arc::new(summa_core::tokenizer::TokenizerRegistry::new()),
    );
    for (expression, expected) in [
        (r#"tag:regex("[0-9]{4}")"#, vec![0]),
        (r#"regex("(19|20)[0-9]{2}")"#, vec![0]),
        (r#"regex("colou?r")"#, vec![1, 2]),
        (r#"+regex("colou?r") -tag:color"#, vec![2]),
        (r#"regex(".")"#, vec![4, 5]),
        (r#"regex("missing.*")"#, vec![]),
    ] {
        let query = parser.parse_strict(expression).unwrap();
        let mut scorer = query.scorer(&reader, 10).await.unwrap();
        let mut actual = Vec::new();
        while scorer.doc() != summa_core::TERMINATED {
            actual.push(scorer.doc());
            scorer.advance();
        }
        assert_eq!(actual, expected, "{expression}");
        let mut count = CountCollector::new();
        collect_segment(&reader, query.as_ref(), &mut count)
            .await
            .unwrap();
        assert_eq!(count.count() as usize, expected.len());
    }
}

#[tokio::test]
async fn punctuated_and_escaped_terms_remain_single_literals_with_boolean_modifiers() {
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for terms in [
        vec!["when", "books.google.com"],
        vec!["a", "user:ed"],
        vec!["hill's"],
        vec!["12.6"],
        vec!["a*b"],
        vec!["books", "google", "com"],
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
    let parser = QueryLanguageParser::new(
        schema,
        vec![field],
        Arc::new(summa_core::tokenizer::TokenizerRegistry::new()),
    );
    for (expression, expected) in [
        ("+when +books.google.com", vec![0]),
        (r"+a +user\:ed", vec![1]),
        ("tag:hill's", vec![2]),
        ("their 12.6", vec![3]),
        (r"tag:a\*b", vec![4]),
    ] {
        let query = parser.parse_strict(expression).unwrap();
        let mut scorer = query.scorer(&reader, 10).await.unwrap();
        let mut actual = Vec::new();
        while scorer.doc() != summa_core::TERMINATED {
            actual.push(scorer.doc());
            scorer.advance();
        }
        assert_eq!(actual, expected, "{expression}");
    }
    assert!(parser.parse_strict("tag:broken\\").is_err());
    assert!(parser.parse("tag:broken\\").is_err());
}

#[tokio::test]
async fn benchmark_regex_language_has_exact_case_sensitive_constant_score_results() {
    use summa_core::RegexQuery;
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let chunks = schema.add_text_field("chunks", true, false);
    schema.set_chunked(chunks, true);
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    let terms = [
        "1999",
        "2000",
        "1990s",
        "1234567",
        "123",
        "www",
        "http",
        "https",
        "monday",
        "wednesday",
        "color",
        "colour",
        "reading",
        "undoing",
        "kindness",
        "deadbeef",
        "operation",
        "vision",
        "10th",
        "é",
        "🦀",
        "x1999",
        "COLOUR",
        "a.b",
    ];
    for term in terms {
        let mut doc = Document::new();
        doc.add_text(field, term);
        builder.add_document(doc).unwrap();
    }
    let id = SegmentId::new();
    builder.build(&directory, id, None).await.unwrap();
    let reader = SegmentReader::open(&directory, id, schema.clone(), 16)
        .await
        .unwrap();
    let cases: &[(&str, &[u32])] = &[
        ("[0-9]{4}", &[0, 1]),
        ("(19|20)[0-9]{2}", &[0, 1]),
        ("[0-9]{4}s", &[2]),
        ("[0-9]{7}", &[3]),
        ("[0-9]{1,3}", &[4]),
        ("(www|http|https)", &[5, 6, 7]),
        ("(mon|tues|wednes|thurs|fri|satur|sun)day", &[8, 9]),
        ("colou?r", &[10, 11]),
        ("(re|un)[a-z]*ing", &[12, 13]),
        ("[jkqxz][a-z]*ess", &[14]),
        ("[a-f0-9]{8,}", &[15]),
        (".*(tion|sion)", &[16, 17]),
        (".*0th", &[18]),
        (".", &[19, 20]),
        ("COLOUR", &[22]),
        (r"a\.b", &[23]),
        ("", &[]),
        ("1999|COLOUR", &[0, 22]),
    ];
    for &(pattern, expected) in cases {
        let query = RegexQuery::new(field, pattern).unwrap();
        use summa_core::Query;
        assert!(query.is_filter());
        let mut scorer = query.scorer(&reader, 100).await.unwrap();
        for &doc in expected {
            assert_eq!(scorer.doc(), doc, "{pattern}");
            assert_eq!(scorer.score(), 1.0);
            scorer.advance();
        }
        assert_eq!(scorer.doc(), summa_core::TERMINATED, "{pattern}");
        let mut count = CountCollector::new();
        collect_segment(&reader, &query, &mut count).await.unwrap();
        assert_eq!(count.count() as usize, expected.len(), "{pattern}");
        #[cfg(feature = "sync")]
        {
            let mut scorer = query.scorer_sync(&reader, 100).unwrap();
            for &doc in expected {
                assert_eq!(scorer.doc(), doc, "{pattern}");
                assert_eq!(scorer.score(), 1.0);
                scorer.advance();
            }
            assert_eq!(scorer.doc(), summa_core::TERMINATED);
        }
    }
    use summa_core::Query;
    let query = RegexQuery::new(chunks, ".*").unwrap();
    assert!(
        query
            .scorer(&reader, 10)
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("chunked")
    );
    assert!(
        query
            .count_estimate(&reader)
            .await
            .unwrap_err()
            .to_string()
            .contains("chunked")
    );
    #[cfg(feature = "sync")]
    assert!(
        query
            .scorer_sync(&reader, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("chunked")
    );
    let parser = QueryLanguageParser::new(
        schema,
        vec![field],
        Arc::new(summa_core::tokenizer::TokenizerRegistry::new()),
    );
    for expression in [r#"regex("[")"#, r#"regex("(?i)foo")"#, r#"regex("foo"#] {
        assert!(parser.parse(expression).is_err(), "{expression}");
    }
    for pattern in [
        "[",
        "(",
        "foo\\",
        "(?i)foo",
        "(?=foo)",
        r"(a)\1",
        r"\d",
        "a&b",
        "~a",
        "@",
        "#",
        "<1-9>",
        "^foo$",
        "[a-z&&x]",
        "a{1000000000}",
    ] {
        assert!(RegexQuery::new(field, pattern).is_err(), "{pattern}");
    }
    assert!(RegexQuery::new(field, "x".repeat(1025)).is_err());
}

#[tokio::test]
async fn regex_expansion_budget_fails_without_partial_results() {
    use summa_core::{Query, RegexQuery};
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
    let query = RegexQuery::new(field, "term[0-9]{4}").unwrap();
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
}

#[tokio::test]
async fn pattern_queries_reject_unindexed_nontext_and_unknown_fields() {
    use summa_core::{Query, RegexQuery, WildcardQuery};
    let mut schema = SchemaBuilder::default();
    let stored = schema.add_text_field_with_tokenizer("stored", false, true, "raw");
    let number = schema.add_u64_field("number", true, true);
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    let mut doc = Document::new();
    doc.add_text(stored, "foo");
    doc.add_u64(number, 123);
    builder.add_document(doc).unwrap();
    let id = SegmentId::new();
    builder.build(&directory, id, None).await.unwrap();
    let reader = SegmentReader::open(&directory, id, schema, 16)
        .await
        .unwrap();
    for field in [stored, number, summa_core::Field(999)] {
        let queries: Vec<Box<dyn Query>> = vec![
            Box::new(RegexQuery::new(field, ".*").unwrap()),
            Box::new(WildcardQuery::new(field, "*").unwrap()),
        ];
        for query in queries {
            assert!(query.scorer(&reader, 10).await.is_err());
            assert!(query.count_estimate(&reader).await.is_err());
            #[cfg(feature = "sync")]
            assert!(query.scorer_sync(&reader, 10).is_err());
        }
    }
}
