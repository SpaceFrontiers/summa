//! Concurrent gated sparse searches must keep the shared search pool live.
//!
//! A 2.2.3 server once stopped completing searches after four concurrent
//! filtered BMP searches: every pool thread slept on a futex and every
//! admission permit stayed held. This mixes that request shape with
//! candidate-branch (phrase, text, sparse) and fusion searches during commits
//! and searcher reloads, on a deliberately small pool and sparse I/O gate, and
//! fails if any round stops making progress. It has not reproduced that stall;
//! it guards the paths involved against liveness regressions.

use std::sync::Arc;
use std::time::Duration;

use crate::directories::RamDirectory;
use crate::dsl::{Document, SchemaBuilder};
use crate::index::{Index, IndexConfig, IndexWriter};
use crate::query::{
    BooleanQuery, FusionMethod, MultiValueCombiner, Query, RangeQuery, SparseVectorQuery, TermQuery,
};
use crate::structures::{SparseFormat, SparseVectorConfig, WeightQuantization};

const SEGMENTS: usize = 12;
const ROUNDS: usize = 40;
const REQUESTS_PER_ROUND: usize = 12;
const DOCS_PER_SEGMENT: usize = 1_500;
const DIMS: u32 = 2_000;

fn sparse_entries(doc: usize) -> Vec<(u32, f32)> {
    (0..24)
        .map(|i| {
            let dim = ((doc * 7919 + i * 104_729) % DIMS as usize) as u32;
            (dim, 0.05 + ((doc + i) % 97) as f32 / 100.0)
        })
        .collect()
}

fn query_entries(seed: usize) -> Vec<(u32, f32)> {
    (0..12)
        .map(|i| {
            (
                ((seed * 31 + i * 151) % DIMS as usize) as u32,
                1.0 - i as f32 / 20.0,
            )
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_mixed_gated_sparse_searches_keep_search_pool_live() {
    let mut sb = SchemaBuilder::default();
    let kind = sb.add_text_field("kind", true, false);
    let content = sb.add_text_field("content", true, false);
    sb.set_positions(content, crate::dsl::PositionMode::TokenPosition);
    let issued_at = sb.add_i64_field("issued_at", false, false);
    sb.set_fast(issued_at, true);
    let sparse = sb.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::Bmp,
            weight_quantization: WeightQuantization::UInt8,
            bmp_block_size: 32,
            dims: Some(DIMS),
            ..SparseVectorConfig::default()
        },
    );
    let schema = sb.build();
    let config = IndexConfig {
        num_threads: 2,
        sparse_io_concurrency: 1,
        background_reload: true,
        num_indexing_threads: 1,
        num_compression_threads: 1,
        merge_policy: Box::new(crate::NoMergePolicy),
        ..IndexConfig::default()
    };

    let dir = RamDirectory::new();
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    for segment in 0..SEGMENTS {
        for i in 0..DOCS_PER_SEGMENT {
            let doc_index = segment * DOCS_PER_SEGMENT + i;
            let mut doc = Document::new();
            doc.add_text(
                kind,
                if doc_index.is_multiple_of(3) {
                    "message"
                } else {
                    "post"
                },
            );
            doc.add_text(
                content,
                format!("recent news technology breakthrough {}", doc_index % 50),
            );
            doc.add_i64(issued_at, doc_index as i64);
            doc.add_sparse_vector(sparse, sparse_entries(doc_index));
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    let index = Index::open(dir, config).await.unwrap();
    let reader = index.reader().await.unwrap();
    assert!(
        reader.searcher().await.unwrap().num_segments() >= SEGMENTS / 2,
        "need many segments"
    );
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writes = {
        let stop = Arc::clone(&stop);
        tokio::spawn(async move {
            let mut next = SEGMENTS * DOCS_PER_SEGMENT;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                for _ in 0..200 {
                    let mut doc = Document::new();
                    doc.add_text(kind, "message");
                    doc.add_text(content, "recent news technology breakthrough");
                    doc.add_i64(issued_at, next as i64);
                    doc.add_sparse_vector(sparse, sparse_entries(next));
                    writer.add_document(doc).unwrap();
                    next += 1;
                }
                writer.commit().await.unwrap();
            }
        })
    };

    let filtered = |seed: usize| -> Arc<dyn Query> {
        Arc::new(
            BooleanQuery::new()
                .must(TermQuery::text(kind, "message"))
                .must(RangeQuery::i64(issued_at, Some(0), Some(i64::MAX)))
                .should(SparseVectorQuery::new(sparse, query_entries(seed))),
        )
    };

    for round in 0..ROUNDS {
        let mut tasks = Vec::new();
        // Pick up the writer's latest commit, as a serving reader would.
        let searcher = reader.searcher().await.unwrap();
        for request in 0..REQUESTS_PER_ROUND {
            let searcher = Arc::clone(&searcher);
            let seed = round * REQUESTS_PER_ROUND + request;
            let branch_phrase: Arc<dyn Query> = Arc::new(crate::query::PhraseQuery::new(
                content,
                ["recent", "news", "technology", "breakthrough"]
                    .iter()
                    .map(|word| word.as_bytes().to_vec())
                    .collect(),
            ));
            let query = filtered(seed);
            let branch_text: Arc<dyn Query> = Arc::new(TermQuery::text(content, "breakthrough"));
            let branch_sparse: Arc<dyn Query> =
                Arc::new(SparseVectorQuery::new(sparse, query_entries(seed + 1)));
            tasks.push(tokio::spawn(async move {
                match request % 3 {
                    // Plain request path: one pool job per request.
                    0 => searcher
                        .search_budgeted_on_pool(query, 80, false, None, None, ())
                        .await
                        .map(|_| ()),
                    // Formula nomination: candidate branches on the pool.
                    1 => searcher
                        .search_candidate_lists(
                            &[branch_sparse, branch_text, branch_phrase],
                            300,
                            None,
                        )
                        .await
                        .map(|_| ()),
                    // Hybrid fusion.
                    _ => searcher
                        .search_fused_with_count(
                            &[(query.as_ref(), 1.0), (branch_text.as_ref(), 1.0)],
                            80,
                            80,
                            FusionMethod::Rrf { k: 60.0 },
                            MultiValueCombiner::default(),
                        )
                        .await
                        .map(|_| ()),
                }
            }));
        }
        let all = futures::future::join_all(tasks);
        let finished = tokio::time::timeout(Duration::from_secs(60), all)
            .await
            .unwrap_or_else(|_| panic!("round {round}: concurrent searches stopped progressing"));
        for task in finished {
            task.unwrap().unwrap();
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    writes.await.unwrap();
}
