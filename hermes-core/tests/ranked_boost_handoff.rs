#![cfg(feature = "native")]

//! A ranked single-term handoff pre-truncates candidates by positive BM25
//! order. A non-positive boost reverses or flattens that order, so the boosted
//! query must see the complete term stream.

use hermes_core::index::{Index, IndexConfig, IndexWriter};
use hermes_core::query::{BoostQuery, Query, SearchResult, TermQuery};
use hermes_core::{Document, RamDirectory, SchemaBuilder};

fn pairs(hits: &[SearchResult]) -> Vec<(u32, u32)> {
    hits.iter()
        .map(|hit| (hit.doc_id, hit.score.to_bits()))
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn non_positive_boost_over_single_term_ranks_the_complete_stream() {
    let dir = RamDirectory::new();
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("text", true, false);
    let config = IndexConfig {
        num_indexing_threads: 1,
        num_threads: 1,
        posting_ratio_bounds: true,
        ..Default::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
        .await
        .unwrap();
    // More than one posting block so the ranked handoff is eligible.
    for i in 0..400u32 {
        let mut doc = Document::new();
        let mut text = String::from("alpha");
        for _ in 0..(i % 37) {
            text.push_str(" pad");
        }
        doc.add_text(body, &text);
        writer.add_document(doc).unwrap();
    }
    writer.commit().await.unwrap();
    writer.shutdown().await.unwrap();
    drop(writer);
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    for boost in [-1.0f32, 0.0] {
        let query = BoostQuery::new(TermQuery::text(body, "alpha"), boost);
        let exhaustive = searcher.search(&query as &dyn Query, 1_000).await.unwrap();
        let limited = searcher.search(&query as &dyn Query, 5).await.unwrap();
        assert_eq!(pairs(&limited), pairs(&exhaustive[..5]), "boost {boost}");
        #[cfg(feature = "sync")]
        {
            let (sync, _) = searcher
                .search_with_offset_and_count_sync(&query as &dyn Query, 5, 0)
                .unwrap();
            assert_eq!(pairs(&sync), pairs(&exhaustive[..5]), "sync boost {boost}");
        }
    }
}
