//! Chunked text fields: every value of the field is its own BM25 unit and
//! results carry per-chunk ordinals (`docs/chunked-text-fields.md`).

use crate::directories::RamDirectory;
use crate::dsl::{Document, Field, PositionMode, Schema, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::{
    BooleanQuery, FusionMethod, MultiValueCombiner, PhraseQuery, PrefixQuery, RangeQuery,
    SearchResult, SparseVectorQuery, TermQuery,
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
        "lex(by: languages, segmenter: simple, stem: snowball, variants: false)",
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

/// Chunk lengths normalise BM25 above the nominal chunk length (the 90th
/// percentile of the field's chunk lengths) and are floored at it below
/// (`docs/chunked-bm25.md`): a chunk far longer than nominal scores lower
/// than a nominal chunk with the same `tf`; a shorter one scores the same.
#[tokio::test]
async fn chunked_bm25_normalises_by_real_chunk_length() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    // Nine nominal chunks of 10 tokens establish the nominal length.
    for i in 0..9 {
        let nominal = format!(
            "needle {}",
            (0..9)
                .map(|j| format!("n{i}x{j}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        writer
            .add_document(doc(&f, "article", &[&nominal]))
            .unwrap();
    }
    let long = format!("needle {}", "filler ".repeat(200));
    writer.add_document(doc(&f, "article", &[&long])).unwrap(); // doc 9
    writer
        .add_document(doc(&f, "article", &["needle short"]))
        .unwrap(); // doc 10
    writer.commit().await.unwrap();

    let index = open(dir).await;
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let (results, _) = searcher
        .search_with_positions(&TermQuery::text(f.content, "needle"), 20)
        .await
        .unwrap();
    assert_eq!(results.len(), 11);
    let nominal = by_doc(&results, 0).score;
    assert!(
        by_doc(&results, 9).score < nominal,
        "same tf, chunk longer than nominal must score lower: {results:?}"
    );
    assert!(
        (by_doc(&results, 10).score - nominal).abs() < 1e-5,
        "same tf, chunk shorter than nominal scores like a nominal one: {results:?}"
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

/// MUST phrases and filters become one document bitset that the chunked
/// text MaxScore executor applies as a predicate: the scored top-k is exact
/// over the filtered documents, documents matching only the filters fill the
/// tail with score 0, and chunk ordinals are still reported.
#[tokio::test]
async fn filters_and_phrases_push_into_chunked_text_maxscore() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    for (kind, chunks) in [
        ("article", vec!["quick brown fox", "lazy dog"]),
        ("book", vec!["quick brown fox jumps", "over the lazy dog"]),
        ("book", vec!["brown fox", "quick dog"]),
        ("article", vec!["quick brown", "fox"]),
        ("book", vec!["nothing here"]),
    ] {
        writer.add_document(doc(&f, kind, &chunks)).unwrap();
    }
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
    let ids = |results: &[SearchResult]| {
        let mut ids: Vec<u32> = results.iter().map(|r| r.doc_id).collect();
        ids.sort_unstable();
        ids
    };

    // Phrase constraint: doc 3 has "brown" and "fox" in different chunks.
    let query = BooleanQuery::new()
        .must(phrase("brown fox"))
        .should(TermQuery::text(f.content, "quick"))
        .should(TermQuery::text(f.content, "dog"));
    let (results, _) = searcher.search_with_positions(&query, 10).await.unwrap();
    assert_eq!(ids(&results), vec![0, 1, 2], "{results:?}");
    assert!(results.iter().all(|r| r.score > 0.0));
    assert_eq!(ordinals(by_doc(&results, 0)), vec![0, 1]);
    assert_eq!(ordinals(by_doc(&results, 2)), vec![1]);

    // Plus a fast-field filter.
    let query = BooleanQuery::new()
        .must(phrase("brown fox"))
        .must(TermQuery::text(f.kind, "book"))
        .should(TermQuery::text(f.content, "quick"))
        .should(TermQuery::text(f.content, "dog"));
    let (results, _) = searcher.search_with_positions(&query, 10).await.unwrap();
    assert_eq!(ids(&results), vec![1, 2], "{results:?}");

    // An OR of phrases (the shape a client uses to try several fields or
    // hints) is one bitset; a document matching only the phrases and none
    // of the scored terms is still returned, with score 0.
    let either = BooleanQuery::new()
        .should(phrase("brown fox"))
        .should(phrase("nothing here"));
    let query = BooleanQuery::new()
        .must(either)
        .should(TermQuery::text(f.content, "quick"))
        .should(TermQuery::text(f.content, "dog"));
    let (results, _) = searcher.search_with_positions(&query, 10).await.unwrap();
    assert_eq!(ids(&results), vec![0, 1, 2, 4], "{results:?}");
    assert_eq!(by_doc(&results, 4).score, 0.0);
    assert!(by_doc(&results, 1).score > 0.0);

    // With a small limit only scored documents make the cut.
    let (results, _) = searcher.search_with_positions(&query, 2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.score > 0.0));
}

/// Field-level BP reordering of a chunked text field: the pass permutes the
/// field's virtual ids (visible in the chunk map), document ids and every
/// other file stay put, and every query returns the same documents, scores
/// and ordinals before and after, also after a subsequent merge.
#[tokio::test]
async fn chunked_text_field_reorders_through_its_chunk_map() {
    use crate::query::PhraseQuery;

    let mut sb = SchemaBuilder::default();
    let languages = sb.add_text_field_with_tokenizer("languages", false, true, "raw_ci");
    sb.set_fast(languages, true);
    let kind = sb.add_text_field_with_tokenizer("kind", true, true, "raw_ci");
    sb.set_fast(kind, true);
    let content = sb.add_text_field_with_tokenizer(
        "content",
        true,
        false,
        "lex(by: languages, segmenter: simple, stem: snowball, variants: false)",
    );
    sb.set_chunked(content, true);
    sb.set_positions(content, PositionMode::TokenPosition);
    sb.set_reorder(content, true);
    // Original document number: merge output may place segments in any
    // order, so results are compared by this rather than by doc id.
    let number = sb.add_u64_field("n", false, false);
    sb.set_fast(number, true);
    let schema = sb.build();

    // Two interleaved topical clusters: even documents use vocabulary A,
    // odd documents vocabulary B, so BP has an obvious better order.
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    let vocab_a = [
        "quantum",
        "lattice",
        "photon",
        "spin",
        "boson",
        "qubit",
        "decoherence",
    ];
    let vocab_b = [
        "kernel",
        "scheduler",
        "thread",
        "mutex",
        "syscall",
        "paging",
        "latency",
    ];
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for d in 0..600u32 {
        let vocab = if d % 2 == 0 { &vocab_a } else { &vocab_b };
        let mut chunks: Vec<String> = Vec::new();
        for _ in 0..2 {
            let words: Vec<&str> = (0..6).map(|_| vocab[(rng() % 7) as usize]).collect();
            chunks.push(words.join(" "));
        }
        let mut doc = Document::new();
        doc.add_text(languages, "en");
        doc.add_text(kind, if d % 3 == 0 { "book" } else { "article" });
        doc.add_u64(number, u64::from(d));
        for chunk in &chunks {
            doc.add_text(content, chunk);
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();

    let queries: Vec<Box<dyn crate::query::Query>> = vec![
        Box::new(
            BooleanQuery::new()
                .should(TermQuery::text(content, "quantum"))
                .should(TermQuery::text(content, "photon"))
                .should(TermQuery::text(content, "kernel")),
        ),
        Box::new(PhraseQuery::new(
            content,
            vec![b"spin".to_vec(), b"boson".to_vec()],
        )),
        Box::new(
            BooleanQuery::new()
                .must(TermQuery::text(kind, "book"))
                .must(PhraseQuery::new(
                    content,
                    vec![b"thread".to_vec(), b"mutex".to_vec()],
                ))
                .should(TermQuery::text(content, "scheduler"))
                .should(TermQuery::text(content, "latency")),
        ),
    ];
    async fn snapshot(
        index: &Index<RamDirectory>,
        queries: &[Box<dyn crate::query::Query>],
        number: Field,
    ) -> Vec<Vec<(u64, i64, Vec<u32>)>> {
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let mut out = Vec::new();
        for (i, query) in queries.iter().enumerate() {
            let (results, _) = searcher.search_with_positions(&**query, 50).await.unwrap();
            let mut rows: Vec<(u64, i64, Vec<u32>)> = results
                .iter()
                .map(|r| {
                    let segment = searcher
                        .segment_readers()
                        .iter()
                        .find(|s| s.meta().id == r.segment_id)
                        .unwrap();
                    let n = segment.fast_field(number.0).unwrap().get_u64(r.doc_id);
                    // The OR query's ordinal list depends on which of many
                    // equally scored chunks make the over-fetched pool, a
                    // tie broken by virtual-id order that the reorder
                    // legitimately changes; the phrase and filtered queries
                    // return every matching chunk.
                    let ords = if i == 0 { Vec::new() } else { ordinals(r) };
                    (n, (r.score * 1e4).round() as i64, ords)
                })
                .collect();
            rows.sort_by_key(|(n, _, _)| *n);
            out.push(rows);
        }
        out
    }

    let before = snapshot(&open(dir.clone()).await, &queries, number).await;
    assert!(before.iter().all(|r| !r.is_empty()), "{before:?}");

    writer.reorder().await.unwrap();
    let index = open(dir.clone()).await;
    let after = snapshot(&index, &queries, number).await;
    assert_eq!(before, after);

    // The field's virtual ids were permuted: chunk-map doc ids are no longer
    // non-decreasing, while document ids themselves did not move.
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let segments = searcher.segment_readers();
    assert_eq!(segments.len(), 1);
    let map = segments[0].chunk_map(content).unwrap();
    assert_eq!(map.num_chunks(), 1200);
    let doc_ids: Vec<u32> = (0..map.num_chunks()).map(|v| map.doc_id(v)).collect();
    assert!(
        doc_ids.windows(2).any(|w| w[0] > w[1]),
        "chunk map still in indexing order"
    );
    let mut seen: Vec<u32> = doc_ids.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 600);

    // Merging a reordered segment keeps working, and results stay equal.
    for d in 600..640u32 {
        let mut doc = Document::new();
        doc.add_text(languages, "en");
        doc.add_text(kind, "article");
        doc.add_u64(number, u64::from(d));
        doc.add_text(
            content,
            if d % 2 == 0 {
                "quantum photon"
            } else {
                "kernel thread"
            },
        );
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.force_merge().await.unwrap();
    // The new documents change idf and average length, so compare the
    // matched documents (and, for the exact-match queries, their ordinals)
    // rather than scores.
    let merged = snapshot(&open(dir.clone()).await, &queries, number).await;
    for (i, (a, b)) in after.iter().zip(&merged).enumerate() {
        let a: Vec<(u64, Vec<u32>)> = a.iter().map(|(d, _, o)| (*d, o.clone())).collect();
        let b: Vec<(u64, Vec<u32>)> = b
            .iter()
            .filter(|(d, _, _)| *d < 600)
            .map(|(d, _, o)| (*d, o.clone()))
            .collect();
        if i < 2 {
            // Top-50 by score can shift with the changed statistics (both
            // the OR and the phrase query have more than 50 matches); the
            // filtered query is an exact set.
            continue;
        }
        assert_eq!(a, b, "query {i}");
    }
    // The short-tail length floor deliberately removes the old ranking boost
    // for the new two-token chunks, so they need not enter an unfiltered
    // top-50 dominated by repeated terms in the older six-token chunks.
    // Address them through a range filter to verify the merged postings and
    // chunk map contain every newly appended document.
    let new_docs = BooleanQuery::new()
        .must(RangeQuery::u64(number, Some(600), Some(639)))
        .should(TermQuery::text(content, "quantum"))
        .should(TermQuery::text(content, "photon"))
        .should(TermQuery::text(content, "kernel"));
    let response = open(dir).await.search(&new_docs, 50).await.unwrap();
    assert_eq!(response.hits.len(), 40, "{response:?}");
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

/// A short tail chunk is scored with the length floored at the average chunk
/// length (`docs/chunked-bm25.md`): with the same `tf` it must not outrank a
/// full-length chunk, and its score must equal the full chunk's.
#[tokio::test]
async fn chunked_short_tail_chunk_is_not_rewarded() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    let full = |i: usize| {
        let mut words = vec!["needle".to_string()];
        for j in 0..99 {
            words.push(format!("w{i}x{j}"));
        }
        words.join(" ")
    };
    // doc 0: two full chunks, the needle in the second.
    writer
        .add_document(doc(
            &f,
            "a",
            &[&full(0).replace("needle", "blank"), &full(1)],
        ))
        .unwrap();
    // doc 1: one full chunk without the needle and a 4-token tail with it.
    writer
        .add_document(doc(
            &f,
            "b",
            &[&full(2).replace("needle", "blank"), "needle tail end here"],
        ))
        .unwrap();
    writer.commit().await.unwrap();
    let index = open(dir).await;

    let results = index
        .search(&TermQuery::text(f.content, "needle"), 10)
        .await
        .unwrap();
    assert_eq!(results.hits.len(), 2, "{results:?}");
    let a = by_doc_hits(&results.hits, 0);
    let b = by_doc_hits(&results.hits, 1);
    assert!(
        (a - b).abs() < 1e-5,
        "tail chunk (len 4) must score like a full chunk: full={a} tail={b}"
    );
}

/// Boolean exclusion stays document-scoped even though chunked postings use
/// virtual chunk ids: a forbidden term in a different chunk must still remove
/// the whole document.
#[tokio::test]
async fn chunked_must_not_excludes_a_match_from_another_chunk() {
    let f = chunked_schema();
    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), f.schema.clone(), IndexConfig::default())
        .await
        .unwrap();
    writer
        .add_document(doc(&f, "a", &["needle lives here", "forbidden elsewhere"]))
        .unwrap();
    writer
        .add_document(doc(&f, "b", &["needle lives here", "clean elsewhere"]))
        .unwrap();
    writer.commit().await.unwrap();
    let index = open(dir).await;

    let query = BooleanQuery::new()
        .should(TermQuery::text(f.content, "needle"))
        .must_not(TermQuery::text(f.content, "forbidden"));
    let results = index.search(&query, 10).await.unwrap();
    assert_eq!(results.hits.len(), 1, "{results:?}");
    assert_eq!(results.hits[0].address.doc_id, 1, "{results:?}");
}

fn by_doc_hits(hits: &[crate::query::SearchHit], doc_id: u32) -> f32 {
    hits.iter()
        .find(|h| h.address.doc_id == doc_id)
        .unwrap_or_else(|| panic!("doc {doc_id} missing"))
        .score
}
