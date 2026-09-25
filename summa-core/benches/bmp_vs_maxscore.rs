//! Benchmark: BMP vs MaxScore for sparse vector retrieval
//!
//! Compares the two sparse vector index formats on synthetic SPLADE-like data:
//! - Index build time
//! - Query latency (top-10, top-100)
//!
//! Run: cargo bench --bench bmp_vs_maxscore
//!
//! With more docs: BMP_BENCH_DOCS=50000 cargo bench --bench bmp_vs_maxscore

use std::sync::Arc;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use summa_core::directories::RamDirectory;
use summa_core::dsl::{Document, SchemaBuilder};
use summa_core::index::{IndexConfig, IndexWriter};
use summa_core::query::SparseVectorQuery;
use summa_core::structures::SparseVectorConfig;

// ============================================================================
// Data generation
// ============================================================================

/// Simple LCG pseudo-random number generator (deterministic, no deps).
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

/// Generate SPLADE-like sparse vectors: power-law dimension distribution,
/// ~100-200 non-zero dims per doc, weights concentrated at low values.
fn generate_sparse_docs(num_docs: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
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

/// Generate clustered sparse docs that simulate topic locality.
///
/// Documents are grouped by "topic" (100 topics). Each topic has a core set of
/// ~500 dimensions. Documents from the same topic share many dimensions.
/// Documents are ordered by topic, so consecutive doc_ids (which form BMP blocks)
/// have similar dimension distributions — enabling effective block pruning.
///
/// This simulates BP (Bipartite Partitioning) document ordering used in the
/// SIGIR 2024 BMP paper, where similar documents are placed adjacent.
fn generate_clustered_sparse_docs(num_docs: usize, seed: u64) -> ClusteredData {
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

    ClusteredData {
        docs: all_docs,
        topic_dims,
    }
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

struct ClusteredData {
    docs: Vec<Vec<(u32, f32)>>,
    topic_dims: Vec<Vec<u32>>,
}

/// Floating-point inverted baseline over the original, unquantized corpus.
///
/// Ground truth is independent of both Summa sparse formats, index pruning,
/// impact quantization, LSP gamma, and alpha. Only touched documents are reset
/// between queries, so producing Recall@K does not hide an O(corpus) memset.
struct ExactSparseBaseline {
    postings: Vec<Vec<(u32, f32)>>,
    scores: Vec<f32>,
    touched: Vec<u32>,
}

impl ExactSparseBaseline {
    fn new(documents: &[Vec<(u32, f32)>]) -> Self {
        let dimensions = documents
            .iter()
            .flatten()
            .map(|&(dimension, _)| dimension as usize + 1)
            .max()
            .unwrap_or(0);
        let mut postings = vec![Vec::new(); dimensions];
        for (document, vector) in documents.iter().enumerate() {
            for &(dimension, weight) in vector {
                postings[dimension as usize].push((document as u32, weight));
            }
        }
        Self {
            postings,
            scores: vec![0.0; documents.len()],
            touched: Vec::new(),
        }
    }

    fn top_k(&mut self, query: &[(u32, f32)], k: usize) -> Vec<u32> {
        for &(dimension, query_weight) in query {
            let Some(postings) = self.postings.get(dimension as usize) else {
                continue;
            };
            for &(document, document_weight) in postings {
                let score = &mut self.scores[document as usize];
                if *score == 0.0 {
                    self.touched.push(document);
                }
                *score += query_weight * document_weight;
            }
        }
        let mut top = std::collections::BinaryHeap::<
            std::cmp::Reverse<(u32, std::cmp::Reverse<u32>)>,
        >::with_capacity(k);
        for &document in &self.touched {
            let score = self.scores[document as usize];
            if score <= 0.0 {
                continue;
            }
            let candidate = (score.to_bits(), std::cmp::Reverse(document));
            if top.len() < k {
                top.push(std::cmp::Reverse(candidate));
            } else if top.peek().is_some_and(|minimum| candidate > minimum.0) {
                top.pop();
                top.push(std::cmp::Reverse(candidate));
            }
        }
        for document in self.touched.drain(..) {
            self.scores[document as usize] = 0.0;
        }
        let mut ranked: Vec<_> = top.into_iter().map(|entry| entry.0).collect();
        ranked.sort_unstable_by(|left, right| right.cmp(left));
        ranked
            .into_iter()
            .map(|(_, std::cmp::Reverse(document))| document)
            .collect()
    }
}

fn percentile_micros(samples: &mut [std::time::Duration], percentile: f64) -> f64 {
    samples.sort_unstable();
    let index = ((samples.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    samples[index].as_secs_f64() * 1_000_000.0
}

fn recall_at(predicted: &[u32], truth: &[u32], k: usize) -> f64 {
    let truth = &truth[..truth.len().min(k)];
    let predicted = &predicted[..predicted.len().min(k)];
    predicted
        .iter()
        .filter(|document| truth.contains(document))
        .count() as f64
        / truth.len().max(1) as f64
}

fn mean_recall_at(predicted: &[Vec<u32>], truth: &[Vec<u32>], k: usize) -> f64 {
    predicted
        .iter()
        .zip(truth)
        .map(|(predicted, truth)| recall_at(predicted, truth, k))
        .sum::<f64>()
        / truth.len().max(1) as f64
}

#[derive(Clone, Copy)]
struct SparseSearchOptions {
    limit: usize,
    gamma: usize,
    alpha: f32,
    query_pruning: Option<f32>,
    max_query_dims: Option<usize>,
}

impl SparseSearchOptions {
    const fn exhaustive(limit: usize) -> Self {
        Self {
            limit,
            gamma: 0,
            alpha: 1.0,
            query_pruning: None,
            max_query_dims: None,
        }
    }
}

fn sparse_search_batch(
    rt: &tokio::runtime::Runtime,
    searcher: &summa_core::index::Searcher<RamDirectory>,
    field: summa_core::dsl::Field,
    queries: &[Vec<(u32, f32)>],
    options: SparseSearchOptions,
) -> (Vec<Vec<u32>>, Vec<std::time::Duration>) {
    let mut results = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    for vector in queries {
        let mut query = SparseVectorQuery::new(field, vector.clone())
            .with_lsp_gamma(options.gamma)
            .with_heap_factor(options.alpha);
        if let Some(pruning) = options.query_pruning {
            query = query.with_pruning(pruning);
        }
        if let Some(max_dims) = options.max_query_dims {
            query = query.with_max_query_dims(max_dims);
        }
        let start = Instant::now();
        let hits = rt.block_on(searcher.search(&query, options.limit)).unwrap();
        latencies.push(start.elapsed());
        results.push(hits.into_iter().map(|hit| hit.doc_id).collect());
    }
    (results, latencies)
}

fn bmp_posting_count(
    searcher: &summa_core::index::Searcher<RamDirectory>,
    field: summa_core::dsl::Field,
) -> u64 {
    searcher
        .segment_readers()
        .iter()
        .filter_map(|segment| segment.bmp_index(field))
        .map(|index| index.total_postings())
        .sum()
}

/// Generate random queries with 10-50 non-zero dimensions.
fn generate_queries(num_queries: usize, seed: u64) -> Vec<Vec<(u32, f32)>> {
    generate_queries_with_dims(num_queries, seed, 10, 40)
}

/// Generate random queries with a specified dimension range.
///
/// Each query has `min_dims + random(0..range)` non-zero dimensions.
fn generate_queries_with_dims(
    num_queries: usize,
    seed: u64,
    min_dims: usize,
    range: usize,
) -> Vec<Vec<(u32, f32)>> {
    let mut rng = Rng::new(seed);
    let vocab_size = 30000u32;

    (0..num_queries)
        .map(|_| {
            let num_dims = min_dims + (rng.next_u32() as usize % range.max(1));
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
// Index building helpers
// ============================================================================

struct BuiltIndex {
    index: summa_core::index::Index<RamDirectory>,
    sparse_field: summa_core::dsl::Field,
    build_time_ms: f64,
}

/// Build index with one sparse vector per document.
fn build_index(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    sparse_config: SparseVectorConfig,
    label: &str,
) -> BuiltIndex {
    build_index_inner(rt, docs, sparse_config, label, 1)
}

/// Build index by grouping a flat list of sparse vectors into documents.
///
/// Takes `all_vectors` and groups every `vectors_per_doc` consecutive vectors
/// into a single document. For `vectors_per_doc=1` this is equivalent to
/// `build_index`. Use this to compare single- vs multi-ordinal at the same
/// total vector count.
fn build_index_grouped(
    rt: &tokio::runtime::Runtime,
    all_vectors: &[Vec<(u32, f32)>],
    vectors_per_doc: usize,
    sparse_config: SparseVectorConfig,
    label: &str,
) -> BuiltIndex {
    let mut sb = SchemaBuilder::default();
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, false, sparse_config);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    let num_docs = all_vectors.len() / vectors_per_doc;
    let commit_interval = (num_docs / 5).max(20_000);

    let start = Instant::now();
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for (i, chunk) in all_vectors.chunks(vectors_per_doc).enumerate() {
            let mut doc = Document::new();
            for v in chunk {
                doc.add_sparse_vector(sparse, v.clone());
            }
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(summa_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        for v in chunk {
                            doc.add_sparse_vector(sparse, v.clone());
                        }
                    }
                    Err(e) => panic!("add_document failed: {}", e),
                }
            }
            if (i + 1) % commit_interval == 0 {
                writer.commit().await.unwrap();
            }
        }
        writer.force_merge().await.unwrap();
    });
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  [{}] built: {:.1}ms ({} docs, {} vectors/doc, {} total vectors)",
        label,
        build_time_ms,
        num_docs,
        vectors_per_doc,
        all_vectors.len(),
    );

    let index = rt
        .block_on(summa_core::index::Index::open(dir, config))
        .unwrap();

    BuiltIndex {
        index,
        sparse_field: sparse,
        build_time_ms,
    }
}

fn build_index_inner(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    sparse_config: SparseVectorConfig,
    label: &str,
    _vectors_per_doc: usize,
) -> BuiltIndex {
    build_index_reorder(rt, docs, sparse_config, label, false)
}

fn build_index_reorder(
    rt: &tokio::runtime::Runtime,
    docs: &[Vec<(u32, f32)>],
    sparse_config: SparseVectorConfig,
    label: &str,
    reorder: bool,
) -> BuiltIndex {
    let mut sb = SchemaBuilder::default();
    let sparse = sb.add_sparse_vector_field_with_config("sparse", true, false, sparse_config);
    if reorder {
        sb.set_reorder(sparse, true);
    }
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig::default();

    // Commit interval scales with doc count: at most ~5 segments
    let commit_interval = (docs.len() / 5).max(20_000);

    let start = Instant::now();
    rt.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema.clone(), config.clone())
            .await
            .unwrap();

        for (i, entries) in docs.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_sparse_vector(sparse, entries.clone());
            loop {
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(summa_core::Error::QueueFull) => {
                        tokio::task::yield_now().await;
                        doc = Document::new();
                        doc.add_sparse_vector(sparse, entries.clone());
                    }
                    Err(e) => panic!("add_document failed: {}", e),
                }
            }
            if (i + 1) % commit_interval == 0 {
                writer.commit().await.unwrap();
            }
        }
        writer.force_merge().await.unwrap();
    });
    let build_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  [{}] built: {:.1}ms", label, build_time_ms);

    let index = rt
        .block_on(summa_core::index::Index::open(dir, config))
        .unwrap();

    BuiltIndex {
        index,
        sparse_field: sparse,
        build_time_ms,
    }
}

// ============================================================================
// Diagnostics: measure pruning effectiveness
// ============================================================================

/// Run a few queries and print pruning statistics.
fn print_pruning_diagnostics(
    rt: &tokio::runtime::Runtime,
    bmp: &BuiltIndex,
    maxscore: &BuiltIndex,
    queries: &[Vec<(u32, f32)>],
    label: &str,
) {
    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    let sample = &queries[..queries.len().min(20)];

    // BMP: measure latency
    let bmp_start = Instant::now();
    for q in sample {
        let query = SparseVectorQuery::new(bmp.sparse_field, q.clone());
        let _ = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
    }
    let bmp_avg_us = bmp_start.elapsed().as_micros() as f64 / sample.len() as f64;

    // MaxScore: measure latency
    let ms_start = Instant::now();
    for q in sample {
        let query = SparseVectorQuery::new(maxscore.sparse_field, q.clone());
        let _ = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
    }
    let ms_avg_us = ms_start.elapsed().as_micros() as f64 / sample.len() as f64;

    let (winner, speedup) = if bmp_avg_us < ms_avg_us {
        ("BMP", ms_avg_us / bmp_avg_us)
    } else {
        ("MaxScore", bmp_avg_us / ms_avg_us)
    };

    eprintln!(
        "\n  [{label}] Diagnostics (avg over {} queries):",
        sample.len()
    );
    eprintln!("    BMP:      {:.0}µs/query", bmp_avg_us);
    eprintln!("    MaxScore: {:.0}µs/query", ms_avg_us);
    eprintln!("    Winner:   {winner} ({speedup:.2}x faster)");
}

// ============================================================================
// Benchmark functions
// ============================================================================

fn bench_query_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );
    let docs = generate_sparse_docs(num_docs, 12345);
    let queries = generate_queries(num_queries, 67890);

    // Build both indexes
    eprintln!("Building indexes...");
    let bmp = build_index(&rt, &docs, SparseVectorConfig::splade_bmp(), "BMP");
    let maxscore = build_index(&rt, &docs, SparseVectorConfig::splade(), "MaxScore");

    eprintln!(
        "\nBuild times: BMP={:.1}ms, MaxScore={:.1}ms",
        bmp.build_time_ms, maxscore.build_time_ms
    );

    print_pruning_diagnostics(&rt, &bmp, &maxscore, &queries, "random");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    // Top-10 benchmark
    {
        let mut group = c.benchmark_group("sparse_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    // Top-100 benchmark
    {
        let mut group = c.benchmark_group("sparse_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark with clustered (topic-local) data — simulates real SPLADE workloads.
///
/// Documents are ordered by topic so BMP blocks contain topically similar docs.
/// This enables effective block pruning (the key to BMP's speedup).
fn bench_clustered(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== Clustered data benchmark ===");
    eprintln!(
        "Generating {} clustered docs and {} queries...",
        num_docs, num_queries
    );
    let clustered = generate_clustered_sparse_docs(num_docs, 54321);
    let queries = generate_clustered_queries(num_queries, &clustered.topic_dims, 67890);

    eprintln!("Building indexes (clustered data)...");
    let bmp = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-clustered",
    );
    let maxscore = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade(),
        "MaxScore-clustered",
    );

    eprintln!(
        "Build times: BMP={:.1}ms, MaxScore={:.1}ms",
        bmp.build_time_ms, maxscore.build_time_ms
    );

    print_pruning_diagnostics(&rt, &bmp, &maxscore, &queries, "clustered");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    {
        let mut group = c.benchmark_group("clustered_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    {
        let mut group = c.benchmark_group("clustered_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone());
                let results = rt.block_on(bmp_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark single- vs multi-ordinal with the same total sparse vector count.
///
/// Generates N total sparse vectors, then indexes them two ways:
/// - Single-ordinal: N documents with 1 vector each
/// - Multi-ordinal:  N/5 documents with 5 vectors each
///
/// Both indexes contain the same vectors — the only difference is how they're
/// grouped into documents. This isolates multi-ordinal overhead (doc_map
/// indirection, virtual-to-real mapping) from the raw vector count.
fn bench_multi_ordinal(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let total_vectors = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let vectors_per_doc = 5;
    let num_queries = 100;

    // Round down to multiple of vectors_per_doc for clean grouping
    let total_vectors = (total_vectors / vectors_per_doc) * vectors_per_doc;
    let multi_num_docs = total_vectors / vectors_per_doc;

    eprintln!("\n=== Multi-ordinal benchmark (same total vectors) ===");
    eprintln!(
        "Total sparse vectors: {} (single: {} docs × 1, multi: {} docs × {})",
        total_vectors, total_vectors, multi_num_docs, vectors_per_doc
    );

    let all_vectors = generate_sparse_docs(total_vectors, 12345);
    let queries = generate_queries(num_queries, 67890);

    // Single-ordinal: each vector is a separate document
    eprintln!("Building single-ordinal indexes...");
    let bmp_single = build_index(
        &rt,
        &all_vectors,
        SparseVectorConfig::splade_bmp(),
        "BMP-1ord",
    );
    let ms_single = build_index(
        &rt,
        &all_vectors,
        SparseVectorConfig::splade(),
        "MaxScore-1ord",
    );

    // Multi-ordinal: group consecutive vectors into documents
    eprintln!("Building multi-ordinal indexes...");
    let bmp_multi = build_index_grouped(
        &rt,
        &all_vectors,
        vectors_per_doc,
        SparseVectorConfig::splade_bmp(),
        "BMP-5ord",
    );
    let ms_multi = build_index_grouped(
        &rt,
        &all_vectors,
        vectors_per_doc,
        SparseVectorConfig::splade(),
        "MaxScore-5ord",
    );

    eprintln!(
        "\nBuild times: BMP-1ord={:.1}ms, BMP-5ord={:.1}ms, MaxScore-1ord={:.1}ms, MaxScore-5ord={:.1}ms",
        bmp_single.build_time_ms,
        bmp_multi.build_time_ms,
        ms_single.build_time_ms,
        ms_multi.build_time_ms,
    );

    // Print BMP diagnostics for both variants
    for (label, built) in [("BMP-1ord", &bmp_single), ("BMP-5ord", &bmp_multi)] {
        let reader = rt.block_on(built.index.reader()).unwrap();
        let searcher = rt.block_on(reader.searcher()).unwrap();
        if let Some(bmp_idx) = searcher
            .segment_readers()
            .first()
            .and_then(|r| r.bmp_index(built.sparse_field))
        {
            eprintln!("\n  [{} diagnostics]", label);
            eprintln!(
                "    blocks={}, block_size={}, total_postings={}, num_virtual_docs={}",
                bmp_idx.num_blocks,
                bmp_idx.bmp_block_size,
                bmp_idx.total_postings(),
                bmp_idx.num_virtual_docs,
            );
            let avg_postings_per_block =
                bmp_idx.total_postings() as f64 / bmp_idx.num_blocks as f64;
            eprintln!("    avg_postings/block={:.0}", avg_postings_per_block);
        }
    }

    // Warmup diagnostics
    {
        let sample = &queries[..queries.len().min(20)];

        let bmp_s_reader = rt.block_on(bmp_single.index.reader()).unwrap();
        let bmp_s_searcher = Arc::new(rt.block_on(bmp_s_reader.searcher()).unwrap());
        let bmp_m_reader = rt.block_on(bmp_multi.index.reader()).unwrap();
        let bmp_m_searcher = Arc::new(rt.block_on(bmp_m_reader.searcher()).unwrap());

        let start = Instant::now();
        for q in sample {
            let query = SparseVectorQuery::new(bmp_single.sparse_field, q.clone());
            let _ = rt.block_on(bmp_s_searcher.search(&query, 10)).unwrap();
        }
        let bmp_1_us = start.elapsed().as_micros() as f64 / sample.len() as f64;

        let start = Instant::now();
        for q in sample {
            let query = SparseVectorQuery::new(bmp_multi.sparse_field, q.clone());
            let _ = rt.block_on(bmp_m_searcher.search(&query, 10)).unwrap();
        }
        let bmp_5_us = start.elapsed().as_micros() as f64 / sample.len() as f64;

        eprintln!(
            "\n  [multi-ordinal] Diagnostics (avg over {} queries):",
            sample.len()
        );
        eprintln!("    BMP-1ord: {:.0}µs/query", bmp_1_us);
        eprintln!("    BMP-5ord: {:.0}µs/query", bmp_5_us);
        let ratio = bmp_5_us / bmp_1_us;
        eprintln!("    Ratio:    {:.2}x (1.0 = no overhead)", ratio);
    }

    let bmp_s_reader = rt.block_on(bmp_single.index.reader()).unwrap();
    let bmp_s_searcher = Arc::new(rt.block_on(bmp_s_reader.searcher()).unwrap());
    let bmp_m_reader = rt.block_on(bmp_multi.index.reader()).unwrap();
    let bmp_m_searcher = Arc::new(rt.block_on(bmp_m_reader.searcher()).unwrap());
    let ms_s_reader = rt.block_on(ms_single.index.reader()).unwrap();
    let ms_s_searcher = Arc::new(rt.block_on(ms_s_reader.searcher()).unwrap());
    let ms_m_reader = rt.block_on(ms_multi.index.reader()).unwrap();
    let ms_m_searcher = Arc::new(rt.block_on(ms_m_reader.searcher()).unwrap());

    // Top-10 benchmark: single vs multi for both BMP and MaxScore
    {
        let mut group = c.benchmark_group("multi_ord_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-1ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_single.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(bmp_s_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-5ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_multi.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(bmp_m_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore-1ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    ms_single.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_s_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore-5ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    ms_multi.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_m_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    // Top-100 benchmark: single vs multi
    {
        let mut group = c.benchmark_group("multi_ord_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-1ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_single.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(bmp_s_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-5ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_multi.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(bmp_m_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore-1ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    ms_single.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_s_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore-5ord", total_vectors), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    ms_multi.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_m_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark with varying query lengths (50, 100, 200 non-zero dimensions).
///
/// Longer queries are typical for SPLADE v2 and other learned sparse models.
/// BMP's advantage should increase with query length because:
/// - More query terms → higher block UBs → better pruning discrimination
/// - MaxScore processes more posting lists per query
fn bench_long_queries(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== Long query benchmark ===");
    let docs = generate_sparse_docs(num_docs, 12345);

    eprintln!("Building indexes...");
    let bmp = build_index(&rt, &docs, SparseVectorConfig::splade_bmp(), "BMP-long");
    let maxscore = build_index(&rt, &docs, SparseVectorConfig::splade(), "MaxScore-long");

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    // Query lengths to benchmark: 50, 100, 200 non-zero dims
    // BMP gets the full query; MaxScore stays at default 32 (dies with more)
    for target_dims in [50, 100, 200] {
        let queries = generate_queries_with_dims(
            num_queries,
            67890 + target_dims as u64,
            target_dims,
            target_dims / 4,
        );
        let avg_dims: f64 =
            queries.iter().map(|q| q.len() as f64).sum::<f64>() / queries.len() as f64;
        eprintln!(
            "  query_dims={}: avg actual dims = {:.1} (MaxScore capped to 32)",
            target_dims, avg_dims
        );

        let group_name = format!("long_q{}_top10", target_dims);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone())
                        .with_max_query_dims(target_dims);
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        // MaxScore uses default max_query_dims (32) — can't handle 200+ terms
        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark approximate retrieval with varying heap_factor (alpha).
///
/// Tests BMP and MaxScore with alpha = {1.0, 0.85, 0.7} to measure the
/// speed–accuracy trade-off. Alpha < 1.0 means more aggressive pruning:
/// - BMP: prune when `UB * alpha <= threshold`
/// - MaxScore: prune when `UB <= threshold / alpha`
///
/// Also measures Recall@10 and Recall@100 against an independent, unquantized
/// f32 exhaustive dot-product baseline, so the reported loss includes both
/// index quantization and approximate traversal.
fn bench_approximate(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== Approximate retrieval benchmark ===");
    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );
    let clustered = generate_clustered_sparse_docs(num_docs, 54321);
    let queries = generate_clustered_queries(num_queries, &clustered.topic_dims, 67890);

    eprintln!("Building indexes...");
    let bmp_config = SparseVectorConfig::splade_bmp();
    let bmp = build_index(&rt, &clustered.docs, bmp_config.clone(), "BMP-approx");
    let mut quantization_only_config = bmp_config.clone();
    quantization_only_config.weight_threshold = 0.0;
    let bmp_quantization_only = build_index(
        &rt,
        &clustered.docs,
        quantization_only_config,
        "BMP-uint8-only",
    );
    let mut static_pruning_config = bmp_config;
    static_pruning_config.pruning = Some(0.1);
    let bmp_static_pruning = build_index(
        &rt,
        &clustered.docs,
        static_pruning_config,
        "BMP-posting-prune-10pct",
    );
    let mut azeroth_config = SparseVectorConfig::splade_bmp();
    azeroth_config.doc_mass = Some(0.9);
    azeroth_config.weight_threshold = 0.1;
    azeroth_config.pruning = Some(0.7);
    let bmp_azeroth = build_index(
        &rt,
        &clustered.docs,
        azeroth_config,
        "BMP-azeroth-0.9-0.1-0.7",
    );
    let mut threshold_only_config = SparseVectorConfig::splade_bmp();
    threshold_only_config.weight_threshold = 0.1;
    let bmp_threshold_only = build_index(
        &rt,
        &clustered.docs,
        threshold_only_config,
        "BMP-threshold-0.1",
    );
    let mut mass_only_config = SparseVectorConfig::splade_bmp();
    mass_only_config.weight_threshold = 0.0;
    mass_only_config.doc_mass = Some(0.9);
    let bmp_mass_only = build_index(&rt, &clustered.docs, mass_only_config, "BMP-doc-mass-0.9");
    let mut pruning_only_config = SparseVectorConfig::splade_bmp();
    pruning_only_config.weight_threshold = 0.0;
    pruning_only_config.pruning = Some(0.7);
    let bmp_pruning_only = build_index(
        &rt,
        &clustered.docs,
        pruning_only_config,
        "BMP-posting-prune-70pct",
    );
    let maxscore = build_index(
        &rt,
        &clustered.docs,
        SparseVectorConfig::splade(),
        "MaxScore-approx",
    );

    let bmp_reader = rt.block_on(bmp.index.reader()).unwrap();
    let bmp_searcher = Arc::new(rt.block_on(bmp_reader.searcher()).unwrap());

    let bmp_quantization_only_reader = rt.block_on(bmp_quantization_only.index.reader()).unwrap();
    let bmp_quantization_only_searcher = Arc::new(
        rt.block_on(bmp_quantization_only_reader.searcher())
            .unwrap(),
    );

    let bmp_static_pruning_reader = rt.block_on(bmp_static_pruning.index.reader()).unwrap();
    let bmp_static_pruning_searcher =
        Arc::new(rt.block_on(bmp_static_pruning_reader.searcher()).unwrap());

    let bmp_azeroth_reader = rt.block_on(bmp_azeroth.index.reader()).unwrap();
    let bmp_azeroth_searcher = Arc::new(rt.block_on(bmp_azeroth_reader.searcher()).unwrap());

    let bmp_threshold_only_reader = rt.block_on(bmp_threshold_only.index.reader()).unwrap();
    let bmp_threshold_only_searcher =
        Arc::new(rt.block_on(bmp_threshold_only_reader.searcher()).unwrap());

    let bmp_mass_only_reader = rt.block_on(bmp_mass_only.index.reader()).unwrap();
    let bmp_mass_only_searcher = Arc::new(rt.block_on(bmp_mass_only_reader.searcher()).unwrap());

    let bmp_pruning_only_reader = rt.block_on(bmp_pruning_only.index.reader()).unwrap();
    let bmp_pruning_only_searcher =
        Arc::new(rt.block_on(bmp_pruning_only_reader.searcher()).unwrap());

    let ms_reader = rt.block_on(maxscore.index.reader()).unwrap();
    let ms_searcher = Arc::new(rt.block_on(ms_reader.searcher()).unwrap());

    let alphas: &[f32] = &[1.0, 0.85, 0.7];

    // Recall and latency against independent floating-point ground truth.
    {
        let mut baseline = ExactSparseBaseline::new(&clustered.docs);
        let original_truth: Vec<Vec<u32>> = queries
            .iter()
            .map(|query| baseline.top_k(query, 100))
            .collect();
        let (indexed_truth, mut indexed_latencies) = sparse_search_batch(
            &rt,
            &bmp_searcher,
            bmp.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let (quantization_only_results, mut quantization_only_latencies) = sparse_search_batch(
            &rt,
            &bmp_quantization_only_searcher,
            bmp_quantization_only.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let (static_pruning_results, mut static_pruning_latencies) = sparse_search_batch(
            &rt,
            &bmp_static_pruning_searcher,
            bmp_static_pruning.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let (azeroth_exhaustive_results, mut azeroth_exhaustive_latencies) = sparse_search_batch(
            &rt,
            &bmp_azeroth_searcher,
            bmp_azeroth.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let (azeroth_api_results, mut azeroth_api_latencies) = sparse_search_batch(
            &rt,
            &bmp_azeroth_searcher,
            bmp_azeroth.sparse_field,
            &queries,
            SparseSearchOptions {
                limit: 100,
                gamma: 500,
                alpha: 0.85,
                query_pruning: Some(0.8),
                max_query_dims: Some(16),
            },
        );
        let (fixed_storage_api_results, mut fixed_storage_api_latencies) = sparse_search_batch(
            &rt,
            &bmp_threshold_only_searcher,
            bmp_threshold_only.sparse_field,
            &queries,
            SparseSearchOptions {
                limit: 100,
                gamma: 500,
                alpha: 0.85,
                query_pruning: Some(0.8),
                max_query_dims: Some(16),
            },
        );
        let (fixed_profile_results, mut fixed_profile_latencies) = sparse_search_batch(
            &rt,
            &bmp_threshold_only_searcher,
            bmp_threshold_only.sparse_field,
            &queries,
            SparseSearchOptions {
                limit: 100,
                gamma: 500,
                alpha: 0.85,
                query_pruning: None,
                max_query_dims: None,
            },
        );
        let (threshold_only_results, mut threshold_only_latencies) = sparse_search_batch(
            &rt,
            &bmp_threshold_only_searcher,
            bmp_threshold_only.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let (mass_only_results, mut mass_only_latencies) = sparse_search_batch(
            &rt,
            &bmp_mass_only_searcher,
            bmp_mass_only.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let (pruning_only_results, mut pruning_only_latencies) = sparse_search_batch(
            &rt,
            &bmp_pruning_only_searcher,
            bmp_pruning_only.sparse_field,
            &queries,
            SparseSearchOptions::exhaustive(100),
        );
        let baseline_postings = bmp_posting_count(
            &bmp_quantization_only_searcher,
            bmp_quantization_only.sparse_field,
        );
        eprintln!("\n  BMP quality decomposition vs original f32:");
        eprintln!(
            "    {:<24} {:>9} {:>9} {:>10} {:>11} {:>11}",
            "index", "f32 R@10", "f32 R@100", "postings", "p50 us", "p95 us"
        );
        for (label, results, latencies, postings) in [
            (
                "uint8 only",
                &quantization_only_results,
                &mut quantization_only_latencies,
                baseline_postings,
            ),
            (
                "default (+threshold)",
                &indexed_truth,
                &mut indexed_latencies,
                bmp_posting_count(&bmp_searcher, bmp.sparse_field),
            ),
            (
                "threshold 0.1 only",
                &threshold_only_results,
                &mut threshold_only_latencies,
                bmp_posting_count(
                    &bmp_threshold_only_searcher,
                    bmp_threshold_only.sparse_field,
                ),
            ),
            (
                "doc_mass 0.9 only",
                &mass_only_results,
                &mut mass_only_latencies,
                bmp_posting_count(&bmp_mass_only_searcher, bmp_mass_only.sparse_field),
            ),
            (
                "top-70% per list only",
                &pruning_only_results,
                &mut pruning_only_latencies,
                bmp_posting_count(&bmp_pruning_only_searcher, bmp_pruning_only.sparse_field),
            ),
            (
                "top-10% per list",
                &static_pruning_results,
                &mut static_pruning_latencies,
                bmp_posting_count(
                    &bmp_static_pruning_searcher,
                    bmp_static_pruning.sparse_field,
                ),
            ),
            (
                "old azeroth storage",
                &azeroth_exhaustive_results,
                &mut azeroth_exhaustive_latencies,
                bmp_posting_count(&bmp_azeroth_searcher, bmp_azeroth.sparse_field),
            ),
            (
                "old + API defaults",
                &azeroth_api_results,
                &mut azeroth_api_latencies,
                bmp_posting_count(&bmp_azeroth_searcher, bmp_azeroth.sparse_field),
            ),
            (
                "fixed storage + old API",
                &fixed_storage_api_results,
                &mut fixed_storage_api_latencies,
                bmp_posting_count(
                    &bmp_threshold_only_searcher,
                    bmp_threshold_only.sparse_field,
                ),
            ),
            (
                "fixed profile",
                &fixed_profile_results,
                &mut fixed_profile_latencies,
                bmp_posting_count(
                    &bmp_threshold_only_searcher,
                    bmp_threshold_only.sparse_field,
                ),
            ),
        ] {
            let p50 = percentile_micros(latencies, 0.50);
            let p95 = percentile_micros(latencies, 0.95);
            eprintln!(
                "    {:<24} {:>9.4} {:>9.4} {:>9.1}% {:>11.1} {:>11.1}",
                label,
                mean_recall_at(results, &original_truth, 10),
                mean_recall_at(results, &original_truth, 100),
                postings as f64 * 100.0 / baseline_postings.max(1) as f64,
                p50,
                p95,
            );
        }
        let total_superblocks: usize = bmp_searcher
            .segment_readers()
            .iter()
            .filter_map(|segment| segment.bmp_index(bmp.sparse_field))
            .map(|index| index.num_superblocks as usize)
            .sum();
        let gamma_eighth = (total_superblocks / 8).max(1);
        let gamma_quarter = (total_superblocks / 4).max(1);
        let gamma_half = (total_superblocks / 2).max(1);
        let settings = vec![
            ("exhaustive".to_string(), 0usize, 1.0f32, None),
            ("gamma250".to_string(), 250, 1.0, None),
            ("gamma500".to_string(), 500, 1.0, None),
            ("gamma1000".to_string(), 1_000, 1.0, None),
            (
                format!("gamma{gamma_eighth}-12pct"),
                gamma_eighth,
                1.0,
                None,
            ),
            (
                format!("gamma{gamma_quarter}-25pct"),
                gamma_quarter,
                1.0,
                None,
            ),
            (format!("gamma{gamma_half}-50pct"), gamma_half, 1.0, None),
            ("gamma250-alpha85".to_string(), 250, 0.85, None),
            (format!("gamma{gamma_eighth}-a85"), gamma_eighth, 0.85, None),
            (format!("gamma{gamma_eighth}-a70"), gamma_eighth, 0.70, None),
            ("gamma250-beta33".to_string(), 250, 1.0, Some(0.33)),
            (
                format!("gamma{gamma_eighth}-b33"),
                gamma_eighth,
                1.0,
                Some(0.33),
            ),
        ];
        eprintln!("\n  Sparse retrieval quality vs original f32 dot-product ground truth:");
        eprintln!(
            "  f32 = end-to-end (index pruning + quantization + traversal); idx = traversal vs exhaustive BMP on the same stored index"
        );
        eprintln!(
            "    {:<20} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11}",
            "setting", "f32 R@10", "f32 R@100", "idx R@10", "idx R@100", "p50 us", "p95 us"
        );
        for (label, gamma, alpha, query_pruning) in settings {
            let (predicted, mut latencies) = sparse_search_batch(
                &rt,
                &bmp_searcher,
                bmp.sparse_field,
                &queries,
                SparseSearchOptions {
                    limit: 100,
                    gamma,
                    alpha,
                    query_pruning,
                    max_query_dims: None,
                },
            );
            let p50 = percentile_micros(&mut latencies, 0.50);
            let p95 = percentile_micros(&mut latencies, 0.95);
            eprintln!(
                "    {:<20} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>11.1} {:>11.1}",
                label,
                mean_recall_at(&predicted, &original_truth, 10),
                mean_recall_at(&predicted, &original_truth, 100),
                mean_recall_at(&predicted, &indexed_truth, 10),
                mean_recall_at(&predicted, &indexed_truth, 100),
                p50,
                p95,
            );
        }
    }

    // Benchmark latency at each alpha
    for &alpha in alphas {
        let group_name = format!("approx_a{}_top10", (alpha * 100.0).round() as u32);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query =
                    SparseVectorQuery::new(bmp.sparse_field, queries[qi % queries.len()].clone())
                        .with_heap_factor(alpha);
                let results = rt.block_on(bmp_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("MaxScore", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    maxscore.sparse_field,
                    queries[qi % queries.len()].clone(),
                )
                .with_heap_factor(alpha);
                let results = rt.block_on(ms_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark: BMP with Recursive Graph Bisection (BP) reorder vs without.
///
/// BP reorders documents to cluster similar docs into the same BMP blocks,
/// improving block pruning effectiveness. This benchmark measures the
/// query-time benefit on random (unclustered) data — the worst case for
/// unordered BMP and the best-case delta for BP reordering.
fn bench_bp_reorder(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== BP reorder benchmark (random data) ===");
    eprintln!(
        "Generating {} docs and {} queries...",
        num_docs, num_queries
    );
    let docs = generate_sparse_docs(num_docs, 12345);
    let queries = generate_queries(num_queries, 67890);

    // Build BMP without reorder (natural doc_id order)
    eprintln!("Building BMP indexes...");
    let bmp_plain = build_index_reorder(
        &rt,
        &docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-plain",
        false,
    );

    // Build BMP with BP reorder
    let bmp_reorder = build_index_reorder(
        &rt,
        &docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-reorder",
        true,
    );

    eprintln!(
        "Build times: plain={:.1}ms, reorder={:.1}ms (BP overhead: {:.1}ms)",
        bmp_plain.build_time_ms,
        bmp_reorder.build_time_ms,
        bmp_reorder.build_time_ms - bmp_plain.build_time_ms,
    );

    // Warmup diagnostics
    print_reorder_diagnostics(&rt, &bmp_plain, &bmp_reorder, &queries, "random");

    let plain_reader = rt.block_on(bmp_plain.index.reader()).unwrap();
    let plain_searcher = Arc::new(rt.block_on(plain_reader.searcher()).unwrap());
    let reorder_reader = rt.block_on(bmp_reorder.index.reader()).unwrap();
    let reorder_searcher = Arc::new(rt.block_on(reorder_reader.searcher()).unwrap());

    // Top-10 benchmark
    {
        let mut group = c.benchmark_group("bp_reorder_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-plain", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_plain.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(plain_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-reorder", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_reorder.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(reorder_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    // Top-100 benchmark
    {
        let mut group = c.benchmark_group("bp_reorder_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-plain", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_plain.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(plain_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-reorder", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_reorder.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(reorder_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Benchmark: BP reorder on clustered (topic-local) data.
///
/// Documents are generated with topic structure but shuffled to simulate
/// random arrival order. BP should recover the topic clustering, giving
/// a large speedup over the unordered baseline.
fn bench_bp_reorder_clustered(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let num_docs = std::env::var("BMP_BENCH_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_queries = 100;

    eprintln!("\n=== BP reorder benchmark (clustered data, shuffled) ===");
    eprintln!(
        "Generating {} clustered docs and {} queries...",
        num_docs, num_queries
    );
    let clustered = generate_clustered_sparse_docs(num_docs, 54321);
    let queries = generate_clustered_queries(num_queries, &clustered.topic_dims, 67890);

    // Shuffle docs to destroy natural topic ordering (simulates random arrival)
    let mut shuffled_docs = clustered.docs;
    {
        let mut rng = Rng::new(99999);
        // Fisher-Yates shuffle
        for i in (1..shuffled_docs.len()).rev() {
            let j = rng.next_u32() as usize % (i + 1);
            shuffled_docs.swap(i, j);
        }
    }

    // Build BMP without reorder (shuffled = poor block locality)
    eprintln!("Building BMP indexes (shuffled clustered data)...");
    let bmp_plain = build_index_reorder(
        &rt,
        &shuffled_docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-shuffled",
        false,
    );

    // Build BMP with BP reorder (should recover topic clustering)
    let bmp_reorder = build_index_reorder(
        &rt,
        &shuffled_docs,
        SparseVectorConfig::splade_bmp(),
        "BMP-bp-reorder",
        true,
    );

    eprintln!(
        "Build times: shuffled={:.1}ms, reorder={:.1}ms (BP overhead: {:.1}ms)",
        bmp_plain.build_time_ms,
        bmp_reorder.build_time_ms,
        bmp_reorder.build_time_ms - bmp_plain.build_time_ms,
    );

    // Warmup diagnostics
    print_reorder_diagnostics(
        &rt,
        &bmp_plain,
        &bmp_reorder,
        &queries,
        "clustered-shuffled",
    );

    let plain_reader = rt.block_on(bmp_plain.index.reader()).unwrap();
    let plain_searcher = Arc::new(rt.block_on(plain_reader.searcher()).unwrap());
    let reorder_reader = rt.block_on(bmp_reorder.index.reader()).unwrap();
    let reorder_searcher = Arc::new(rt.block_on(reorder_reader.searcher()).unwrap());

    // Top-10 benchmark
    {
        let mut group = c.benchmark_group("bp_reorder_clustered_top10");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-shuffled", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_plain.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(plain_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-reorder", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_reorder.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(reorder_searcher.search(&query, 10)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }

    // Top-100 benchmark
    {
        let mut group = c.benchmark_group("bp_reorder_clustered_top100");
        group.sample_size(50);

        group.bench_function(BenchmarkId::new("BMP-shuffled", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_plain.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(plain_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.bench_function(BenchmarkId::new("BMP-reorder", num_docs), |b| {
            let mut qi = 0;
            b.iter(|| {
                let query = SparseVectorQuery::new(
                    bmp_reorder.sparse_field,
                    queries[qi % queries.len()].clone(),
                );
                let results = rt.block_on(reorder_searcher.search(&query, 100)).unwrap();
                qi += 1;
                results
            });
        });

        group.finish();
    }
}

/// Print warmup diagnostics comparing plain vs reordered BMP.
fn print_reorder_diagnostics(
    rt: &tokio::runtime::Runtime,
    plain: &BuiltIndex,
    reorder: &BuiltIndex,
    queries: &[Vec<(u32, f32)>],
    label: &str,
) {
    let plain_reader = rt.block_on(plain.index.reader()).unwrap();
    let plain_searcher = Arc::new(rt.block_on(plain_reader.searcher()).unwrap());
    let reorder_reader = rt.block_on(reorder.index.reader()).unwrap();
    let reorder_searcher = Arc::new(rt.block_on(reorder_reader.searcher()).unwrap());

    let sample = &queries[..queries.len().min(20)];

    let start = Instant::now();
    for q in sample {
        let query = SparseVectorQuery::new(plain.sparse_field, q.clone());
        let _ = rt.block_on(plain_searcher.search(&query, 10)).unwrap();
    }
    let plain_us = start.elapsed().as_micros() as f64 / sample.len() as f64;

    let start = Instant::now();
    for q in sample {
        let query = SparseVectorQuery::new(reorder.sparse_field, q.clone());
        let _ = rt.block_on(reorder_searcher.search(&query, 10)).unwrap();
    }
    let reorder_us = start.elapsed().as_micros() as f64 / sample.len() as f64;

    let speedup = plain_us / reorder_us;
    eprintln!(
        "\n  [BP reorder — {}] Diagnostics (avg over {} queries):",
        label,
        sample.len()
    );
    eprintln!("    BMP plain:   {:.0}µs/query", plain_us);
    eprintln!("    BMP reorder: {:.0}µs/query", reorder_us);
    eprintln!("    Speedup:     {:.2}x", speedup);
}

criterion_group!(
    benches,
    bench_query_latency,
    bench_clustered,
    bench_multi_ordinal,
    bench_long_queries,
    bench_approximate,
    bench_bp_reorder,
    bench_bp_reorder_clustered,
);
criterion_main!(benches);
