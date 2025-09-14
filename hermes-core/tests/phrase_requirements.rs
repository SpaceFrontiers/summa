#![cfg(feature = "native")]

use std::sync::Arc;

use hermes_core::dsl::{PositionMode, QueryLanguageParser};
use hermes_core::index::{Index, IndexConfig, IndexWriter};
use hermes_core::query::{PhraseQuery, Query};
use hermes_core::tokenizer::TokenizerRegistry;
use hermes_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn phrases_require_token_positions_instead_of_accepting_unordered_terms() {
    for mode in [
        None,
        Some(PositionMode::Ordinal),
        Some(PositionMode::TokenPosition),
        Some(PositionMode::Full),
    ] {
        let mut schema = SchemaBuilder::default();
        let body = schema.add_text_field("body", true, false);
        if let Some(mode) = mode {
            schema.set_positions(body, mode);
        }
        schema.set_default_fields(vec!["body".into()]);
        let schema = schema.build();
        let config = IndexConfig {
            num_indexing_threads: 1,
            num_threads: 1,
            ..Default::default()
        };
        let dir = RamDirectory::new();
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();
        for text in ["alpha gap beta", "alpha beta", "beta alpha"] {
            let mut doc = Document::new();
            doc.add_text(body, text);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.shutdown().await.unwrap();
        let index = Index::open(dir, config).await.unwrap();
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let query = PhraseQuery::text(body, "alpha beta");
        let supported = mode.is_some_and(|m| m.tracks_token_position());
        let result = searcher.search(&query, 10).await;
        if supported {
            assert_eq!(
                result
                    .unwrap()
                    .iter()
                    .map(|hit| hit.doc_id)
                    .collect::<Vec<_>>(),
                [1]
            );
        } else {
            let error = result.expect_err("a phrase must not degrade to unordered term matching");
            assert!(error.to_string().contains("token positions"), "{error}");
            assert!(error.to_string().contains("body"), "{error}");
        }
        for parser in [
            searcher.query_parser(),
            QueryLanguageParser::new(
                Arc::new(schema),
                vec![body],
                Arc::new(TokenizerRegistry::default()),
            ),
        ] {
            for text in [
                "\"alpha beta\"",
                "body:\"alpha beta\"",
                "alpha AND \"beta alpha\"",
            ] {
                assert_eq!(parser.parse(text).is_ok(), supported, "{mode:?}: {text}");
                assert_eq!(
                    parser.parse_strict(text).is_ok(),
                    supported,
                    "{mode:?}: {text}"
                );
            }
            assert!(parser.parse("\"alpha\"").is_ok());
            assert!(parser.parse("alpha AND beta").is_ok());
        }
        let segment = &searcher.segment_readers()[0];
        assert_eq!(query.scorer(segment, 10).await.is_ok(), supported);
        #[cfg(feature = "sync")]
        assert_eq!(query.scorer_sync(segment, 10).is_ok(), supported);
        assert_eq!(
            searcher
                .search(&PhraseQuery::text(body, "alpha"), 10)
                .await
                .unwrap()
                .len(),
            3
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn quoted_queries_keep_each_default_fields_analyzed_branch() {
    let mut schema = SchemaBuilder::default();
    let label = schema.add_text_field_with_tokenizer("label", true, false, "raw");
    let body = schema.add_text_field("body", true, false);
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_default_fields(vec!["label".into(), "body".into()]);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for (label_text, body_text) in [
        ("alpha beta", "other"),
        ("other", "alpha beta"),
        ("beta alpha", "alpha gap beta"),
    ] {
        let mut doc = Document::new();
        doc.add_text(label, label_text);
        doc.add_text(body, body_text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    for (input, expected) in [
        ("\"alpha beta\"", vec![0, 1]),
        ("label:\"alpha beta\"", vec![0]),
        ("body:\"alpha beta\"", vec![1]),
    ] {
        let query = searcher.query_parser().parse(input).unwrap();
        let mut ids: Vec<_> = searcher
            .search(query.as_ref(), 10)
            .await
            .unwrap()
            .iter()
            .map(|hit| hit.doc_id)
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, expected, "{input}");
    }
}

#[test]
fn unsupported_phrase_branches_are_errors_before_index_io_and_additional_routing() {
    use hermes_core::dsl::query_field_router::{QueryFieldRouter, QueryRouterRule, RoutingMode};
    let mut schema = SchemaBuilder::default();
    let label = schema.add_text_field_with_tokenizer("label", true, false, "raw");
    let body = schema.add_text_field("body", true, false);
    let schema = Arc::new(schema.build());
    let tokenizers = Arc::new(TokenizerRegistry::default());
    let mut parser = QueryLanguageParser::new(schema, vec![label, body], tokenizers);
    assert!(parser.parse("\"alpha beta\"").is_err());
    assert!(parser.parse("label:\"alpha beta\"").is_ok());
    let router = QueryFieldRouter::from_rules(&[QueryRouterRule {
        pattern: ".*".into(),
        substitution: "alpha".into(),
        target_field: "label".into(),
        mode: RoutingMode::Additional,
    }])
    .unwrap();
    parser.set_router(router);
    assert!(
        parser.parse("\"alpha beta\"").is_err(),
        "additional routing must not discard the unsupported phrase branch"
    );
    assert!(parser.parse("alpha").is_ok());
}
