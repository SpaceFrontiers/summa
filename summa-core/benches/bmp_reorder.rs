//! Benchmark: BMP search latency before and after record-level SimHash reorder.
//!
//! Tests two scenarios:
//! 1. Random data (no natural clustering) — verifies correctness, modest speedup
//! 2. Shuffled clustered data — documents have topic structure but arrive in
//!    random order. Reorder should recover the clustering and enable block pruning.
//!
//! Run with default 100k docs:
//!   cargo bench --bench bmp_reorder --features native
//!
//! Run with 1M docs:
//!   BMP_REORDER_DOCS=1000000 cargo bench --bench bmp_reorder --features native

use std::sync::Arc;
use std::time::Instant;

use summa_core::directories::RamDirectory;
use summa_core::dsl::{Document, SchemaBuilder};
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::SparseVectorQuery;
use summa_core::structures::SparseVectorConfig;

// ============================================================================
// Data generation (deterministic, no deps)
// ============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() % 10000) as f32 / 10000.0
    }
}

/// Random sparse vectors: power-law dimension distribution, 80-200 dims/doc.
fn generate_random_docs(num_docs: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;

    (0..num_docs)
        .map(|_| {
            let num_dims = 80 + (rng.next_u32() % 120) as usize;
            let mut entries = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() * rng.next_f32() * 2.0 + 0.01;
                entries.push((dim.min(vocab_size - 1), weight));
            }
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
        })
        .collect()
}

/// Generate clustered docs (by topic) then SHUFFLE them randomly.
///
/// This simulates the real-world scenario: documents have natural topic structure
/// but arrive in arbitrary order (e.g., from a web crawl). The reorder command
/// should recover the clustering by grouping similar docs into the same blocks.
type SparseVec = Vec<(u32, f32)>;

#[allow(clippy::type_complexity)]
fn generate_shuffled_clustered_docs(num_docs: usize, seed: u64) -> (Vec<SparseVec>, Vec<Vec<u32>>) {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;
    let num_topics = 100;
    let dims_per_topic = 500;

    // Generate topic-dimension assignments
    let mut topic_dims: Vec<Vec<u32>> = Vec::with_capacity(num_topics);
    for t in 0..num_topics {
        let base = (t as u32 * (vocab_size / num_topics as u32)) % vocab_size;
        let mut dims: Vec<u32> = (0..dims_per_topic)
            .map(|_| (base + rng.next_u32() % (vocab_size / 5)) % vocab_size)
            .collect();
        dims.sort_unstable();
        dims.dedup();
        topic_dims.push(dims);
    }

    let docs_per_topic = num_docs / num_topics;
    let mut all_docs = Vec::with_capacity(num_docs);

    for topic in topic_dims.iter().take(num_topics) {
        for _ in 0..docs_per_topic {
            let num_dims = 80 + (rng.next_u32() % 120) as usize;
            let mut entries = Vec::with_capacity(num_dims);

            // 70% of dims from topic's core set (higher weights)
            let topic_count = (num_dims as f32 * 0.7) as usize;
            for _ in 0..topic_count {
                let dim = topic[rng.next_u32() as usize % topic.len()];
                let weight = rng.next_f32() * 1.5 + 0.3;
                entries.push((dim, weight));
            }

            // 30% random dims (noise / cross-topic terms)
            for _ in topic_count..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() * rng.next_f32() * 0.5 + 0.01;
                entries.push((dim.min(vocab_size - 1), weight));
            }

            entries.sort_by_key(|&(d, _)| d);
            entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1.max(a.1);
                    true
                } else {
                    false
                }
            });
            all_docs.push(entries);
        }
    }

    // Fill remaining
    while all_docs.len() < num_docs {
        let t = rng.next_u32() as usize % num_topics;
        let topic = &topic_dims[t];
        let num_dims = 80 + (rng.next_u32() % 120) as usize;
        let mut entries = Vec::with_capacity(num_dims);
        let topic_count = (num_dims as f32 * 0.7) as usize;
        for _ in 0..topic_count {
            let dim = topic[rng.next_u32() as usize % topic.len()];
            let weight = rng.next_f32() * 1.5 + 0.3;
            entries.push((dim, weight));
        }
        for _ in topic_count..num_dims {
            let raw = rng.next_f32();
            let dim = ((raw * raw) * vocab_size as f32) as u32;
            let weight = rng.next_f32() * rng.next_f32() * 0.5 + 0.01;
            entries.push((dim.min(vocab_size - 1), weight));
        }
        entries.sort_by_key(|&(d, _)| d);
        entries.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = b.1.max(a.1);
                true
            } else {
                false
            }
        });
        all_docs.push(entries);
    }

    // SHUFFLE the docs — destroy the topic-ordered locality
    // Fisher-Yates shuffle with our deterministic RNG
    for i in (1..all_docs.len()).rev() {
        let j = rng.next_u32() as usize % (i + 1);
        all_docs.swap(i, j);
    }

    (all_docs, topic_dims)
}

/// Generate topical queries matching the clustered data.
fn generate_clustered_queries(
    num_queries: usize,
    topic_dims: &[Vec<u32>],
    seed: u64,
) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let num_topics = topic_dims.len();
    let vocab_size = 30000u32;

    (0..num_queries)
        .map(|_| {
            let t = rng.next_u32() as usize % num_topics;
            let topic = &topic_dims[t];
            let num_dims = 15 + (rng.next_u32() as usize % 25);
            let mut entries = Vec::with_capacity(num_dims);

            let topic_count = (num_dims as f32 * 0.8) as usize;
            for _ in 0..topic_count {
                let dim = topic[rng.next_u32() as usize % topic.len()];
                let weight = rng.next_f32() + 0.2;
                entries.push((dim, weight));
            }
            for _ in topic_count..num_dims {
                let dim = rng.next_u32() % vocab_size;
                let weight = rng.next_f32() * 0.3 + 0.05;
                entries.push((dim, weight));
            }

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
        })
        .collect()
}

/// Random queries: 15-40 dims each.
fn generate_random_queries(num_queries: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;

    (0..num_queries)
        .map(|_| {
            let num_dims = 15 + (rng.next_u32() as usize % 25);
            let mut entries = Vec::with_capacity(num_dims);
            for _ in 0..num_dims {
                let raw = rng.next_f32();
                let dim = ((raw * raw) * vocab_size as f32) as u32;
                let weight = rng.next_f32() + 0.1;
                entries.push((dim.min(vocab_size - 1), weight));
            }
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
        })
        .collect()
}

// ============================================================================
// Benchmark helpers
// ============================================================================

struct BenchResult {
    avg_us: f64,
    p50_us: f64,
    p99_us: f64,
}

fn run_queries(
    rt: &tokio::runtime::Runtime,
    searcher: &Arc<summa_core::index::Searcher<RamDirectory>>,
    sparse_field: summa_core::dsl::Field,
    queries: &[Vec<(u32, f32)>],
    k: usize,
    warmup_rounds: usize,
) -> BenchResult {
    // Warmup
    for i in 0..warmup_rounds {
        let q = &queries[i % queries.len()];
        let query = SparseVectorQuery::new(sparse_field, q.clone());
        let _ = rt.block_on(searcher.search(&query, k)).unwrap();
    }

    // Timed runs — 3 passes, take the best
    let mut best_latencies: Option<Vec<f64>> = None;

    for _ in 0..3 {
        let mut latencies_us = Vec::with_capacity(queries.len());
        for q in queries {
            let query = SparseVectorQuery::new(sparse_field, q.clone());
            let start = Instant::now();
            let _ = rt.block_on(searcher.search(&query, k)).unwrap();
            latencies_us.push(start.elapsed().as_micros() as f64);
        }
        let avg: f64 = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
        let best_avg = best_latencies
            .as_ref()
            .map(|l| l.iter().sum::<f64>() / l.len() as f64)
            .unwrap_or(f64::MAX);
        if avg < best_avg {
            best_latencies = Some(latencies_us);
        }
    }

    let mut latencies_us = best_latencies.unwrap();
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_us.len();
    let avg = latencies_us.iter().sum::<f64>() / n as f64;
    let p50 = latencies_us[n / 2];
    let p99 = latencies_us[(n as f64 * 0.99) as usize];

    BenchResult {
        avg_us: avg,
        p50_us: p50,
        p99_us: p99,
    }
}

fn print_results(label: &str, r: &BenchResult) {
    eprintln!(
        "  {:<30} avg={:>8.0}µs  p50={:>8.0}µs  p99={:>8.0}µs",
        label, r.avg_us, r.p50_us, r.p99_us
    );
}

fn build_index(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    dir: &RamDirectory,
    config: &IndexConfig,
    schema: &summa_core::dsl::Schema,
    sparse_field: summa_core::dsl::Field,
) {
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for (i, entries) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_sparse_vector(sparse_field, entries.clone());
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(summa_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        doc.add_sparse_vector(sparse_field, entries.clone());
                    }
                    Err(e) => panic!("add_document failed: {}", e),
                }
            }
            if (i + 1) % 100_000 == 0 {
                eprintln!("  indexed {}k docs...", (i + 1) / 1000);
            }
        }
        writer.force_merge().await.unwrap();
    });
}

fn do_reorder(rt: &tokio::runtime::Runtime, dir: &RamDirectory, config: &IndexConfig) {
    rt.block_on(async {
        let mut writer = IndexWriter::open(dir.clone(), config.clone())
            .await
            .unwrap();
        writer.reorder().await.unwrap();
    });
}

fn run_scenario(
    rt: &tokio::runtime::Runtime,
    label: &str,
    docs: &[Vec<(u32, f32)>],
    queries: &[Vec<(u32, f32)>],
) {
    let num_docs = docs.len();
    eprintln!("\n============================================================");
    eprintln!("  {} ({} docs, {} queries)", label, num_docs, queries.len());
    eprintln!("============================================================");

    let mut sb = SchemaBuilder::default();
    let sparse_config = SparseVectorConfig::splade_bmp();
    let sparse_field = sb.add_sparse_vector_field_with_config("sparse", true, false, sparse_config);
    let schema = sb.build();
    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Build
    eprintln!("\n  Building BMP index...");
    let build_start = Instant::now();
    build_index(rt, docs, &dir, &config, &schema, sparse_field);
    eprintln!("  Built in {:.1}s", build_start.elapsed().as_secs_f64());

    // -- Before reorder --
    eprintln!("\n  --- Before Reorder ---");
    let index = rt
        .block_on(Index::open(dir.clone(), config.clone()))
        .unwrap();
    let reader = rt.block_on(index.reader()).unwrap();
    let searcher = Arc::new(rt.block_on(reader.searcher()).unwrap());

    if let Some(bmp_idx) = searcher
        .segment_readers()
        .first()
        .and_then(|r| r.bmp_index(sparse_field))
    {
        eprintln!(
            "  blocks={}, real_docs={}, total_postings={}",
            bmp_idx.num_blocks,
            bmp_idx.num_real_docs(),
            bmp_idx.total_postings(),
        );
    }

    let before_top10 = run_queries(rt, &searcher, sparse_field, queries, 10, 50);
    let before_top100 = run_queries(rt, &searcher, sparse_field, queries, 100, 50);
    print_results("top-10 (before)", &before_top10);
    print_results("top-100 (before)", &before_top100);

    // Collect exact results for recall
    let before_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            let query = SparseVectorQuery::new(sparse_field, q.clone());
            let results = rt.block_on(searcher.search(&query, 10)).unwrap();
            results.iter().map(|h| h.doc_id).collect()
        })
        .collect();

    drop(searcher);
    drop(index);

    // -- Reorder --
    eprintln!("\n  Reordering...");
    let reorder_start = Instant::now();
    do_reorder(rt, &dir, &config);
    eprintln!(
        "  Reorder completed in {:.1}s",
        reorder_start.elapsed().as_secs_f64()
    );

    // -- After reorder --
    eprintln!("\n  --- After Reorder ---");
    let index = rt
        .block_on(Index::open(dir.clone(), config.clone()))
        .unwrap();
    let reader = rt.block_on(index.reader()).unwrap();
    let searcher = Arc::new(rt.block_on(reader.searcher()).unwrap());

    if let Some(bmp_idx) = searcher
        .segment_readers()
        .first()
        .and_then(|r| r.bmp_index(sparse_field))
    {
        eprintln!(
            "  blocks={}, real_docs={}, total_postings={}",
            bmp_idx.num_blocks,
            bmp_idx.num_real_docs(),
            bmp_idx.total_postings(),
        );
    }

    let after_top10 = run_queries(rt, &searcher, sparse_field, queries, 10, 50);
    let after_top100 = run_queries(rt, &searcher, sparse_field, queries, 100, 50);
    print_results("top-10 (after)", &after_top10);
    print_results("top-100 (after)", &after_top100);

    // Recall
    let after_results: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            let query = SparseVectorQuery::new(sparse_field, q.clone());
            let results = rt.block_on(searcher.search(&query, 10)).unwrap();
            results.iter().map(|h| h.doc_id).collect()
        })
        .collect();

    let mut total_recall = 0.0;
    for (before, after) in before_results.iter().zip(after_results.iter()) {
        let overlap = after.iter().filter(|d| before.contains(d)).count();
        total_recall += overlap as f64 / before.len().max(1) as f64;
    }
    let avg_recall = total_recall / queries.len() as f64;

    // Summary
    let speedup_10 = before_top10.avg_us / after_top10.avg_us;
    let speedup_100 = before_top100.avg_us / after_top100.avg_us;
    eprintln!("\n  --- Summary: {} ---", label);
    eprintln!(
        "  top-10:  {:.0}µs → {:.0}µs  ({:.2}x {})",
        before_top10.avg_us,
        after_top10.avg_us,
        if speedup_10 >= 1.0 {
            speedup_10
        } else {
            1.0 / speedup_10
        },
        if speedup_10 >= 1.0 {
            "faster"
        } else {
            "slower"
        },
    );
    eprintln!(
        "  top-100: {:.0}µs → {:.0}µs  ({:.2}x {})",
        before_top100.avg_us,
        after_top100.avg_us,
        if speedup_100 >= 1.0 {
            speedup_100
        } else {
            1.0 / speedup_100
        },
        if speedup_100 >= 1.0 {
            "faster"
        } else {
            "slower"
        },
    );
    eprintln!("  recall@10 vs before: {:.3}", avg_recall);
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_REORDER_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let num_queries = 200;

    eprintln!("=== BMP Reorder Benchmark ===");
    eprintln!("  docs:    {}", num_docs);
    eprintln!("  queries: {}", num_queries);

    // ── Scenario 1: Shuffled clustered data ─────────────────────────────
    // Documents have topic structure but arrive in random order.
    // Reorder should recover clustering → big speedup.
    eprintln!("\nGenerating {} shuffled-clustered docs...", num_docs);
    let gen_start = Instant::now();
    let (clustered_docs, topic_dims) = generate_shuffled_clustered_docs(num_docs, 54321);
    let clustered_queries = generate_clustered_queries(num_queries, &topic_dims, 67890);
    eprintln!(
        "  Generated in {:.1}ms",
        gen_start.elapsed().as_secs_f64() * 1000.0
    );

    run_scenario(
        &rt,
        "Shuffled clustered (topical queries)",
        &clustered_docs,
        &clustered_queries,
    );

    // ── Scenario 2: Random data ─────────────────────────────────────────
    // No natural clustering — reorder should still improve slightly.
    eprintln!("\n\nGenerating {} random docs...", num_docs);
    let gen_start = Instant::now();
    let random_docs = generate_random_docs(num_docs, 12345);
    let random_queries = generate_random_queries(num_queries, 67890);
    eprintln!(
        "  Generated in {:.1}ms",
        gen_start.elapsed().as_secs_f64() * 1000.0
    );

    run_scenario(
        &rt,
        "Random data (random queries)",
        &random_docs,
        &random_queries,
    );
}
