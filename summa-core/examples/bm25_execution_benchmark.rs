//! Same-fixture warm BM25/phrase benchmark; no production latency claim.
//! cargo run --locked --release -p summa-core --example bm25_execution_benchmark
use std::time::Instant;
use summa_core::directories::RamDirectory;
use summa_core::dsl::PositionMode;
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::{BooleanQuery, BoostQuery, PhraseQuery, Query, TermQuery};
use summa_core::{Document, SchemaBuilder};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field_with_tokenizer("body", true, false, "simple");
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_chunked(body, true);
    let directory = RamDirectory::new();
    let config = IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        max_indexing_memory_bytes: 512 * 1024 * 1024,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(directory.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    for id in 0..20_000 {
        let mut doc = Document::new();
        for chunk in 0..4 {
            let text = match (id + chunk) % 4 {
                0 => "machine learning padding ".repeat(40),
                1 => "machine padding learning ".repeat(40),
                2 => "learning machine ".repeat(60),
                _ => "machine learning".to_string(),
            };
            doc.add_text(body, text);
        }
        let mut attempts = 0;
        loop {
            match writer.add_document(doc.clone()) {
                Ok(()) => break,
                Err(summa_core::Error::QueueFull) if attempts < 10_000 => {
                    attempts += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Err(error) => panic!("fixture ingestion: {error}"),
            }
        }
    }
    writer.commit().await.unwrap();
    drop(writer);
    let index = Index::open(directory, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let plain = || {
        BooleanQuery::new()
            .should(TermQuery::text(body, "machine"))
            .should(TermQuery::text(body, "learning"))
    };
    let phrase = || PhraseQuery::text(body, "machine learning");
    let queries: Vec<(&str, Box<dyn Query>)> = vec![
        ("match", Box::new(plain())),
        ("phrase", Box::new(phrase())),
        (
            "phrase_bonus",
            Box::new(
                BooleanQuery::new()
                    .should(plain())
                    .should(BoostQuery::new(phrase(), 2.0)),
            ),
        ),
    ];
    for (name, query) in queries {
        let mut timings = Vec::new();
        let mut expected = None;
        for iteration in 0..11 {
            let start = Instant::now();
            let (hits, seen) = searcher
                .search_with_positions(query.as_ref(), 40)
                .await
                .unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let signature: Vec<_> = hits
                .iter()
                .map(|hit| {
                    (
                        hit.doc_id,
                        hit.score.to_bits(),
                        hit.positions
                            .iter()
                            .flat_map(|(_, p)| p.iter().map(|p| (p.position, p.score.to_bits())))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            if let Some(expected) = &expected {
                assert_eq!(&signature, expected);
            } else {
                println!("{name} seen={seen} signature={signature:?}");
                expected = Some(signature);
            }
            if iteration > 0 {
                timings.push(elapsed);
            }
        }
        timings.sort_by(f64::total_cmp);
        println!(
            "{name} median_ms={:.3} samples_ms={timings:?}",
            timings[timings.len() / 2]
        );
    }
}
