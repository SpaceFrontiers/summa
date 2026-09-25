//! BMP executor hot-path benchmark: wide (60-dimension) queries against one
//! large single-segment index with `bmp_block_size = 256`.
//!
//! This isolates the per-query costs inside `query/bmp.rs` — superblock
//! ordering (`sort_sb_desc_into`), the D-grid pass
//! (`compute_block_ubs_and_presence`), and the per-block posting loop
//! (`score_block_bsearch_int` + docmap resolution) — without the
//! MaxScore comparison, reorder, or multi-ordinal setups of
//! `bmp_vs_maxscore.rs`, so it runs in well under three minutes:
//!
//! ```text
//! cargo bench -p summa-core --bench bmp_hot_path -- --warm-up-time 1 --measurement-time 3 --noplot
//! ```
//!
//! `BMP_BENCH_WIDE_DOCS` overrides the corpus size (default 200_000).

use std::sync::Arc;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use summa_core::directories::RamDirectory;
use summa_core::dsl::{Document, SchemaBuilder};
use summa_core::index::{IndexConfig, IndexWriter};
use summa_core::query::SparseVectorQuery;
use summa_core::structures::SparseVectorConfig;

const VOCAB: u32 = 30_000;
const QUERY_DIMS: usize = 60;
const NUM_QUERIES: usize = 200;

/// Deterministic LCG (same constants as `bmp_vs_maxscore.rs`).
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    /// Skewed dimension draw: low ids are far more frequent, like a SPLADE
    /// vocabulary, so query dimensions collide with many blocks.
    fn dim(&mut self) -> u32 {
        let raw = self.next_f32();
        (((raw * raw) * VOCAB as f32) as u32).min(VOCAB - 1)
    }
}

fn dedup_sorted(mut entries: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
    entries.sort_by_key(|&(d, _)| d);
    entries.dedup_by(|a, b| {
        if a.0 == b.0 {
            b.1 = b.1.max(a.1);
            true
        } else {
            false
        }
    });
    entries
}

/// ~80-200 non-zero dims per doc, weights concentrated at low values.
fn generate_docs(num_docs: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng(seed);
    (0..num_docs)
        .map(|_| {
            let num_dims = 80 + (rng.next_u32() % 120) as usize;
            let entries = (0..num_dims)
                .map(|_| {
                    let w = rng.next_f32();
                    (rng.dim(), w * w * 3.0 + 0.05)
                })
                .collect();
            dedup_sorted(entries)
        })
        .collect()
}

/// Exactly `QUERY_DIMS` distinct dimensions per query, drawn with the corpus
/// skew (`uniform == false`, most terms are dense in every block) or
/// uniformly over the vocabulary (`uniform == true`, most terms are rare, so
/// grid presence, per-block term masks and threshold pruning do real work).
fn generate_queries(num_queries: usize, seed: u64, uniform: bool) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng(seed);
    (0..num_queries)
        .map(|_| {
            let mut entries: Vec<(u32, f32)> = Vec::with_capacity(QUERY_DIMS);
            while entries.len() < QUERY_DIMS {
                let need = QUERY_DIMS - entries.len();
                for _ in 0..need {
                    let dim = if uniform {
                        rng.next_u32() % VOCAB
                    } else {
                        rng.dim()
                    };
                    entries.push((dim, rng.next_f32() + 0.1));
                }
                entries = dedup_sorted(entries);
            }
            entries
        })
        .collect()
}

struct BuiltIndex {
    index: summa_core::index::Index<RamDirectory>,
    field: summa_core::dsl::Field,
}

fn build_single_segment(rt: &tokio::runtime::Runtime, docs: &[Vec<(u32, f32)>]) -> BuiltIndex {
    let config = SparseVectorConfig {
        bmp_block_size: 256,
        ..SparseVectorConfig::splade_bmp()
    };
    let mut sb = SchemaBuilder::default();
    let field = sb.add_sparse_vector_field_with_config("sparse", true, false, config);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let index_config = IndexConfig::default();
    let commit_interval = (docs.len() / 4).max(20_000);

    let start = Instant::now();
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), index_config.clone())
            .await
            .unwrap();
        for (i, entries) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_sparse_vector(field, entries.clone());
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(summa_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        doc.add_sparse_vector(field, entries.clone());
                    }
                    Err(e) => panic!("add_document failed: {e}"),
                }
            }
            if (i + 1) % commit_interval == 0 {
                writer.commit().await.unwrap();
            }
        }
        writer.force_merge().await.unwrap();
    });
    eprintln!(
        "  built {} docs into one segment: {:.1}s",
        docs.len(),
        start.elapsed().as_secs_f64()
    );

    let index = rt
        .block_on(summa_core::index::Index::open(dir, index_config))
        .unwrap();
    BuiltIndex { index, field }
}

fn bench_wide_query_single_segment(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let num_docs = std::env::var("BMP_BENCH_WIDE_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    eprintln!("\n=== BMP hot path: {QUERY_DIMS}-dim queries, {num_docs} docs, block_size=256 ===");
    let docs = generate_docs(num_docs, 0x5eed_1234);
    let queries = generate_queries(NUM_QUERIES, 0x0badcafe, false);
    let uniform_queries = generate_queries(NUM_QUERIES, 0x0badcafe, true);
    let built = build_single_segment(&rt, &docs);

    let reader = rt.block_on(built.index.reader()).unwrap();
    let searcher = Arc::new(rt.block_on(reader.searcher()).unwrap());
    assert_eq!(searcher.num_segments(), 1, "bench requires one segment");
    let bmp = searcher
        .segment_readers()
        .iter()
        .find_map(|segment| segment.bmp_index(built.field))
        .expect("BMP index");
    eprintln!(
        "  blocks={}, superblocks={}, block_size={}, postings={}",
        bmp.num_blocks,
        bmp.num_superblocks,
        bmp.bmp_block_size,
        bmp.total_postings()
    );

    // Sanity: the query set must actually touch the corpus.
    let probe = rt
        .block_on(searcher.search(
            &SparseVectorQuery::new(built.field, queries[0].clone()).with_lsp_gamma(0),
            10,
        ))
        .unwrap();
    assert_eq!(probe.len(), 10, "probe query returned {} hits", probe.len());

    // Exhaustive local traversal (gamma=0): compute_grid_ubs_int +
    // sort_sb_desc_into + every surviving superblock through the D pass.
    for (k, group_name) in [
        (10usize, "wide_q60_exhaustive_top10"),
        (100, "wide_q60_exhaustive_top100"),
    ] {
        let mut group = c.benchmark_group(group_name);
        group.sample_size(30);
        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(built.field, queries[qi % queries.len()].clone())
                        .with_lsp_gamma(0);
                let results = rt.block_on(searcher.search(&query, k)).unwrap();
                qi += 1;
                results
            });
        });
        group.finish();
    }

    // Default gamma schedule (>= num_superblocks here, so still the local
    // path but with a positive visit cap, exercising the same executor
    // branch production single-segment queries take).
    {
        let mut group = c.benchmark_group("wide_q60_default_gamma_top10");
        group.sample_size(30);
        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(built.field, queries[qi % queries.len()].clone());
                let results = rt.block_on(searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });
        group.finish();
    }

    // Uniform-vocabulary terms: most query dimensions are absent from most
    // blocks, so the D-grid presence, per-block term masks, block pruning and
    // the threshold compare dominate instead of dense-row accumulation.
    for (k, group_name) in [
        (10usize, "wide_q60_uniform_top10"),
        (100, "wide_q60_uniform_top100"),
    ] {
        let mut group = c.benchmark_group(group_name);
        group.sample_size(30);
        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    built.field,
                    uniform_queries[qi % uniform_queries.len()].clone(),
                )
                .with_lsp_gamma(0);
                let results = rt.block_on(searcher.search(&query, k)).unwrap();
                qi += 1;
                results
            });
        });
        group.finish();
    }

    // Small gamma below the superblock count: global LSP/0 plan with a
    // `selection`, i.e. the H/E-grid path plus the same per-block scoring.
    {
        let gamma = (bmp.num_superblocks as usize / 4).max(1);
        let mut group = c.benchmark_group("wide_q60_gamma_quarter_top10");
        group.sample_size(30);
        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(built.field, queries[qi % queries.len()].clone())
                        .with_lsp_gamma(gamma);
                let results = rt.block_on(searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_wide_query_single_segment);
criterion_main!(benches);
