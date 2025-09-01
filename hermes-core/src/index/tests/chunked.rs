//! Chunked text fields: every value of the field is its own BM25 unit and
//! results carry per-chunk ordinals (`docs/chunked-text-fields.md`).

use crate::directories::RamDirectory;
use crate::dsl::{Document, Field, PositionMode, Schema, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::{
    BooleanQuery, FusionMethod, MultiValueCombiner, PhraseQuery, PrefixQuery, SearchResult,
    SparseVectorQuery, TermQuery,
};

struct Fields {
    schema: Schema,
    content: Field,
    kind: Field,
    sparse: Field,
}

fn chunked_schema() -> Fields {
    let mut sb = SchemaBuilder::default();
    let languages = sb.add_text_field_with_tokenizer("languages", false, true, "raw_ci");
    sb.set_fast(languages, true);
    let kind = sb.add_text_field_with_tokenizer("kind", true, true, "raw_ci");
    sb.set_fast(kind, true);
    let content = sb.add_text_field_with_tokenizer(
        "content",
        true,
        false,
        "stem(by: languages, default: simple)",
    );
    sb.set_chunked(content, true);
    sb.set_positions(content, PositionMode::TokenPosition);
    let sparse = sb.add_sparse_vector_field("sparse", true, false);
    Fields {
        schema: sb.build(),
        content,
        kind,
        sparse,
    }
}

fn doc(fields: &Fields, kind: &str, chunks: &[&str]) -> Document {
    let mut d = Document::new();
    d.add_text(fields.kind, kind);
    for chunk in chunks {
        d.add_text(fields.content, *chunk);
    }
    d
}

/// Ordinals reported for a hit, ascending.
fn ordinals(result: &SearchResult) -> Vec<u32> {
    let mut ordinals: Vec<u32> = result
        .positions
        .iter()
        .flat_map(|(_, scored)| scored.iter().map(|sp| sp.position))
        .collect();
    ordinals.sort_unstable();
    ordinals
}

fn by_doc(results: &[SearchResult], doc_id: u32) -> &SearchResult {
    results
        .iter()
        .find(|r| r.doc_id == doc_id)
        .unwrap_or_else(|| panic!("doc {doc_id} missing from {results:?}"))
}

async fn open(dir: RamDirectory) -> Index<RamDirectory> {
    Index::open(dir, IndexConfig::default()).await.unwrap()
}

#[tokio::test]
async fn chunked_match_scores_chunks_and_reports_ordinals() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    writer
        .add_document(doc(
            &f,
            "article",
            &["alpha beta gamma", "delta epsilon", "zeta eta theta needle"],
        ))
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["needle needle here", "other words"]))
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["nothing relevant", "still nothing"]))
        .unwrap();
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    // Multi-term OR → chunked MaxScore. Every matching chunk is an ordinal.
    let or_query = BooleanQuery::new()
        .should(TermQuery::text(f.content, "needle"))
        .should(TermQuery::text(f.content, "gamma"));
    let (results, _) = searcher.search_with_positions(&or_query, 10).await.unwrap();
    assert_eq!(results.len(), 2, "doc 2 has no matching chunk: {results:?}");
    let doc0 = by_doc(&results, 0);
    assert_eq!(
        ordinals(doc0),
        vec![0, 2],
        "gamma in chunk 0, needle in chunk 2"
    );
    let doc1 = by_doc(&results, 1);
    assert_eq!(ordinals(doc1), vec![0]);
    // Document score is the best chunk, not the sum over chunks.
    let chunk_score = |result: &SearchResult, ordinal: u32| {
        result.positions[0]
            .1
            .iter()
            .find(|sp| sp.position == ordinal)
            .map(|sp| sp.score)
            .unwrap()
    };
    let best_chunk = chunk_score(doc0, 0).max(chunk_score(doc0, 2));
    assert!((doc0.score - best_chunk).abs() < 1e-6, "{doc0:?}");
    // Each chunk is scored on its own: doc 1's tf=2 needle chunk outranks
    // doc 0's single-occurrence needle chunk.
    assert!(
        chunk_score(doc1, 0) > chunk_score(doc0, 2),
        "tf=2 short chunk must outrank a single occurrence: {results:?}"
    );

    // Single term → same ordinal reporting.
    let term = TermQuery::text(f.content, "needle");
    let (results, _) = searcher.search_with_positions(&term, 10).await.unwrap();
    assert_eq!(ordinals(by_doc(&results, 0)), vec![2]);
    assert_eq!(ordinals(by_doc(&results, 1)), vec![0]);

    // The positions-free API yields the same documents and scores.
    let (plain, _) = searcher.search_with_count(&term, 10).await.unwrap();
    assert_eq!(plain.len(), 2);
    assert!(plain.iter().all(|r| r.positions.is_empty()));
    for hit in &plain {
        assert_eq!(hit.score, by_doc(&results, hit.doc_id).score);
    }
}

#[tokio::test]
async fn chunked_phrase_never_crosses_a_chunk_boundary() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    // "brown" ends chunk 0 and "fox" starts chunk 1: adjacent in the document,
    // never adjacent inside one chunk.
    writer
        .add_document(doc(&f, "article", &["quick brown", "fox jumps"]))
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["padding text", "quick brown fox"]))
        .unwrap();
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let phrase = |text: &str| {
        PhraseQuery::new(
            f.content,
            text.split(' ').map(|t| t.as_bytes().to_vec()).collect(),
        )
    };
    let (results, _) = searcher
        .search_with_positions(&phrase("brown fox"), 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!(results[0].doc_id, 1);
    assert_eq!(ordinals(&results[0]), vec![1]);

    let (results, _) = searcher
        .search_with_positions(&phrase("quick brown"), 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(ordinals(by_doc(&results, 0)), vec![0]);
    assert_eq!(ordinals(by_doc(&results, 1)), vec![1]);
}

#[tokio::test]
async fn chunked_bm25_normalises_by_real_chunk_length() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    let long = format!("needle {}", "filler ".repeat(40));
    writer.add_document(doc(&f, "article", &[&long])).unwrap();
    writer
        .add_document(doc(&f, "article", &["needle filler filler"]))
        .unwrap();
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let (results, _) = searcher
        .search_with_positions(&TermQuery::text(f.content, "needle"), 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(
        by_doc(&results, 1).score > by_doc(&results, 0).score,
        "same tf, shorter chunk must score higher: {results:?}"
    );
}

#[tokio::test]
async fn chunked_ordinals_survive_segment_merge() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["first chunk", "second needle"]))
        .unwrap();
    writer.commit().await.unwrap();
    // Second segment: its virtual ids restart at 0 and must be re-based on merge.
    writer
        .add_document(doc(
            &f,
            "article",
            &["another chunk", "more text", "final needle"],
        ))
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["needle first"]))
        .unwrap();
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let (results, _) = searcher
        .search_with_positions(&TermQuery::text(f.content, "needle"), 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 3, "{results:?}");
    let segments: std::collections::HashSet<u128> = results.iter().map(|r| r.segment_id).collect();
    assert_eq!(segments.len(), 1, "force_merge must leave one segment");
    assert_eq!(ordinals(by_doc(&results, 0)), vec![1]);
    assert_eq!(ordinals(by_doc(&results, 1)), vec![2]);
    assert_eq!(ordinals(by_doc(&results, 2)), vec![0]);

    // Phrases still resolve per chunk after the merge.
    let phrase = PhraseQuery::new(f.content, vec![b"final".to_vec(), b"needle".to_vec()]);
    let (results, _) = searcher.search_with_positions(&phrase, 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, 1);
    assert_eq!(ordinals(&results[0]), vec![2]);
}

#[tokio::test]
async fn chunked_text_fuses_with_sparse_vectors_on_shared_ordinals() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    // Doc 0: text and vector agree on chunk 0.
    let mut d = doc(&f, "article", &["needle words", "hay words"]);
    d.add_sparse_vector(f.sparse, vec![(1, 1.0)]);
    d.add_sparse_vector(f.sparse, vec![(2, 1.0)]);
    writer.add_document(d).unwrap();
    // Doc 1: text and vector agree on chunk 1.
    let mut d = doc(&f, "article", &["hay words", "needle words"]);
    d.add_sparse_vector(f.sparse, vec![(2, 1.0)]);
    d.add_sparse_vector(f.sparse, vec![(1, 1.0)]);
    writer.add_document(d).unwrap();
    // Doc 2: text hits chunk 0 but the vector hits chunk 1 — no corroboration.
    let mut d = doc(&f, "article", &["needle words", "hay words"]);
    d.add_sparse_vector(f.sparse, vec![(2, 1.0)]);
    d.add_sparse_vector(f.sparse, vec![(1, 1.0)]);
    writer.add_document(d).unwrap();
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let text = TermQuery::text(f.content, "needle");
    let sparse = SparseVectorQuery::new(f.sparse, vec![(1, 1.0)]);
    let fused = searcher
        .search_fused(
            &[(&text, 1.0), (&sparse, 1.0)],
            10,
            10,
            FusionMethod::default(),
            MultiValueCombiner::Max,
        )
        .await
        .unwrap();
    assert_eq!(fused.len(), 3, "{fused:?}");
    let doc0 = by_doc(&fused, 0);
    let doc1 = by_doc(&fused, 1);
    let doc2 = by_doc(&fused, 2);
    assert_eq!(ordinals(doc0), vec![0], "both verticals land on chunk 0");
    assert_eq!(ordinals(doc1), vec![1], "both verticals land on chunk 1");
    assert_eq!(
        ordinals(doc2),
        vec![0, 1],
        "disagreeing verticals stay separate chunks"
    );
    assert!(
        doc0.score > doc2.score && doc1.score > doc2.score,
        "same-chunk corroboration must compound: {fused:?}"
    );
}

#[tokio::test]
async fn chunked_match_composes_with_document_filters() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["hay", "needle here"]))
        .unwrap();
    writer
        .add_document(doc(&f, "book", &["needle here", "hay"]))
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["hay only"]))
        .unwrap();
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();

    let query = BooleanQuery::new()
        .must(TermQuery::text(f.kind, "book"))
        .should(TermQuery::text(f.content, "needle"))
        .should(TermQuery::text(f.content, "here"));
    let (results, _) = searcher.search_with_positions(&query, 10).await.unwrap();
    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!(results[0].doc_id, 1);
    assert_eq!(ordinals(&results[0]), vec![0]);

    // Terms spread over different chunks all report their own ordinal.
    let query = BooleanQuery::new()
        .should(TermQuery::text(f.content, "needle"))
        .should(TermQuery::text(f.content, "hay"));
    let (results, _) = searcher.search_with_positions(&query, 10).await.unwrap();
    assert_eq!(ordinals(by_doc(&results, 0)), vec![0, 1]);
    assert_eq!(ordinals(by_doc(&results, 1)), vec![0, 1]);
    assert_eq!(ordinals(by_doc(&results, 2)), vec![0]);
}

#[tokio::test]
async fn chunked_field_rejects_prefix_queries_loudly() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    writer
        .add_document(doc(&f, "article", &["needle here"]))
        .unwrap();
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let error = searcher
        .search_with_count(&PrefixQuery::text(f.content, "need"), 10)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("chunked"),
        "prefix on a chunked field must fail with an actionable message: {error}"
    );
}
