#![cfg(feature = "native")]

use summa_core::dsl::PositionMode;
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::{Document, RamDirectory, SchemaBuilder};

#[tokio::test(flavor = "current_thread")]
async fn required_phrases_and_terms_keep_boolean_scope_and_exclude_prohibited_matches() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("text", true, false);
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_default_fields(vec!["text".into()]);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for text in [
        "alpha beta gamma",
        "alpha gap beta",
        "beta gamma",
        "gamma delta",
        "delta",
        "alpha",
    ] {
        let mut doc = Document::new();
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let parser = searcher.query_parser();
    for (input, expected) in [
        ("+alpha beta", vec![0, 1, 5]),
        ("alpha +beta", vec![0, 1, 2]),
        ("+alpha +beta", vec![0, 1]),
        ("+alpha -beta", vec![5]),
        ("alpha -beta", vec![5]),
        ("alpha AND -beta", vec![5]),
        ("+\"alpha beta\" +gamma", vec![0]),
        ("+text:\"alpha beta\" +gamma", vec![0]),
        ("+(\"alpha beta\" delta) gamma", vec![0, 3, 4]),
        ("(+alpha) beta", vec![0, 1, 2, 5]),
        ("+(alpha beta) -gamma", vec![1, 5]),
        ("alpha OR NOT beta", vec![0, 1, 3, 4, 5]),
        ("-beta", vec![3, 4, 5]),
        ("+text:alp* -beta", vec![5]),
    ] {
        let query = parser.parse_strict(input).unwrap();
        let hits = searcher.search(query.as_ref(), 10).await.unwrap();
        let mut docs: Vec<_> = hits.iter().map(|h| h.doc_id).collect();
        docs.sort_unstable();
        assert_eq!(docs, expected, "query {input}");
        #[cfg(feature = "sync")]
        {
            let (sync, _) = searcher
                .search_with_offset_and_count_sync(query.as_ref(), 10, 0)
                .unwrap();
            assert_eq!(sync, hits, "sync/async query {input}");
        }
    }
    // A modifier followed by whitespace is ordinary text: strict parsing
    // reports it, free-text parsing keeps the words.
    assert!(parser.parse_strict("alpha - beta").is_err());
    for (input, expected) in [
        ("alpha - beta", vec![0, 1, 2, 5]),
        ("2010 - 2020 alpha", vec![0, 1, 5]),
        ("+ alpha beta", vec![0, 1, 2, 5]),
    ] {
        let query = parser.parse(input).unwrap();
        let hits = searcher.search(query.as_ref(), 10).await.unwrap();
        let mut docs: Vec<_> = hits.iter().map(|h| h.doc_id).collect();
        docs.sort_unstable();
        assert_eq!(docs, expected, "query {input} parsed as {query}");
    }
    for input in [
        "+",
        "+alpha -",
        "++alpha",
        "+-alpha",
        "--alpha",
        "++alpha*",
        "+\"alpha",
        "+()",
        "alpha @ +beta",
        "(+alpha @)",
    ] {
        assert!(
            parser.parse_strict(input).is_err(),
            "strict accepted {input}"
        );
        assert!(
            parser.parse(input).is_err(),
            "fallback lost modifier in {input}"
        );
    }
    assert!(parser.parse("c++").is_ok(), "plain text remains supported");
    assert!(parser.parse_strict("c++").is_err());
    // Exclusions must not introduce the score of a complement query.
    let parsed = parser.parse_strict("alpha -beta").unwrap();
    let direct = summa_core::query::BooleanQuery::new()
        .must(summa_core::query::TermQuery::text(body, "alpha"))
        .must_not(summa_core::query::TermQuery::text(body, "beta"));
    assert_eq!(
        searcher.search(parsed.as_ref(), 10).await.unwrap(),
        searcher.search(&direct, 10).await.unwrap(),
    );
}

#[tokio::test(flavor = "current_thread")]
async fn selective_required_phrases_preserve_exact_counts_and_combined_scores() {
    use summa_core::query::{CountCollector, TopKCollector, collect_segment};
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("text", true, false);
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_default_fields(vec!["text".into()]);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    let mut texts = Vec::new();
    for row in 0..1024 {
        let mut text = if row % 3 == 0 {
            "alpha gap beta"
        } else {
            "alpha beta"
        }
        .to_owned();
        if row % 17 == 0 {
            text.push_str(" selective");
        }
        if row % 11 == 0 {
            text.push_str(" excluded");
        }
        let mut doc = Document::new();
        doc.add_text(body, text.clone());
        texts.push(text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let parser = searcher.query_parser();
    let phrase = parser.parse_strict("\"alpha beta\"").unwrap();
    let term = parser.parse_strict("selective").unwrap();
    let phrase_scores: std::collections::HashMap<_, _> = searcher
        .search(phrase.as_ref(), 1024)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| (hit.doc_id, hit.score))
        .collect();
    let term_scores: std::collections::HashMap<_, _> = searcher
        .search(term.as_ref(), 1024)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| (hit.doc_id, hit.score))
        .collect();
    for input in [
        "+\"alpha beta\" +selective",
        "+\"alpha beta\" +selective -excluded",
    ] {
        let query = parser.parse_strict(input).unwrap();
        let mut expected: Vec<_> = texts
            .iter()
            .enumerate()
            .filter(|(_, text)| {
                text.contains("alpha beta")
                    && text.contains("selective")
                    && (!input.contains("-excluded") || !text.contains("excluded"))
            })
            .map(|(id, _)| {
                (
                    id as u32,
                    phrase_scores[&(id as u32)] + term_scores[&(id as u32)],
                )
            })
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut count = CountCollector::new();
        let mut top = TopKCollector::new(1000);
        collect_segment(
            &searcher.segment_readers()[0],
            query.as_ref(),
            &mut (&mut top, &mut count),
        )
        .await
        .unwrap();
        assert_eq!(count.count() as usize, expected.len());
        let exhaustive: Vec<_> = top
            .into_sorted_results()
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect();
        assert_eq!(exhaustive, expected, "{input}");
        for limit in [10, 100, 1000] {
            let hits = searcher.search(query.as_ref(), limit).await.unwrap();
            #[cfg(feature = "sync")]
            assert_eq!(
                hits,
                searcher
                    .search_with_offset_and_count_sync(query.as_ref(), limit, 0)
                    .unwrap()
                    .0
            );
            let actual: Vec<_> = hits
                .into_iter()
                .map(|hit| (hit.doc_id, hit.score))
                .collect();
            assert_eq!(
                actual,
                expected[..expected.len().min(limit)],
                "{input}, k={limit}"
            );
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn words_starting_with_boolean_keywords_remain_searchable() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field("text", true, false);
    schema.set_default_fields(vec!["text".into()]);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for value in ["nothing android", "orchid", "alpha", "beta"] {
        let mut doc = Document::new();
        doc.add_text(text, value);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let parser = searcher.query_parser();
    for (input, expected) in [
        ("NOTHING", vec![0]),
        ("alpha ANDROID", vec![0, 2]),
        ("alpha ORCHID", vec![1, 2]),
        ("alpha AND android", vec![]),
        ("alpha OR(ANDROID)", vec![0, 2]),
        ("NOT(alpha)", vec![0, 1, 3]),
    ] {
        let query = parser.parse_strict(input).unwrap();
        let hits = searcher.search(query.as_ref(), 10).await.unwrap();
        let mut ids: Vec<_> = hits.iter().map(|hit| hit.doc_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, expected, "{input}");
        #[cfg(feature = "sync")]
        {
            let (hits, _) = searcher
                .search_with_offset_and_count_sync(query.as_ref(), 10, 0)
                .unwrap();
            let mut ids: Vec<_> = hits.iter().map(|hit| hit.doc_id).collect();
            ids.sort_unstable();
            assert_eq!(ids, expected, "sync {input}");
        }
    }
}

/// A nested all-required child under a summed OR parent must contribute its
/// complete score stream: a document outside the child's own top-k can still
/// win the parent once the optional terms are added.
#[tokio::test(flavor = "current_thread")]
async fn nested_required_child_under_summed_or_keeps_complete_scores() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("text", true, false);
    schema.set_default_fields(vec!["text".into()]);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for text in [
        "alpha beta",
        "alpha beta",
        "alpha beta gamma delta gamma delta",
        "gamma",
        "delta",
        "filler",
        "filler",
    ] {
        let mut doc = Document::new();
        doc.add_text(body, text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let parser = searcher.query_parser();
    for input in ["(+alpha +beta) gamma delta", "(alpha AND beta) gamma delta"] {
        let query = parser.parse_strict(input).unwrap();
        // A limit at or above the document count takes the complete path.
        let exhaustive = searcher.search(query.as_ref(), 100).await.unwrap();
        assert_eq!(exhaustive[0].doc_id, 2, "query {input}: {exhaustive:?}");
        let limited = searcher.search(query.as_ref(), 2).await.unwrap();
        let pairs = |hits: &[summa_core::query::SearchResult]| {
            hits.iter()
                .map(|hit| (hit.doc_id, hit.score.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(pairs(&limited), pairs(&exhaustive[..2]), "query {input}");
        #[cfg(feature = "sync")]
        {
            let (sync, _) = searcher
                .search_with_offset_and_count_sync(query.as_ref(), 2, 0)
                .unwrap();
            assert_eq!(pairs(&sync), pairs(&exhaustive[..2]), "sync query {input}");
        }
    }
}
