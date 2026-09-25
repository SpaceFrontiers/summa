#![allow(clippy::chunks_exact_to_as_chunks)]

//! Dense ANN hot-path benchmarks: float coarse routing (flat vs HNSW
//! crossover), an end-to-end IVF-TQ search over an on-disk segment, and the
//! batch f32/f16/u8 scoring kernels used by the exact rerank and flat scan.
//!
//! Companion to `vector_indexing.rs` (LUT16 block kernel, plan build). Run
//! filtered: `cargo bench -p summa-core --bench dense_ann -- <filter>`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(feature = "sync")]
use summa_core::directories::MmapDirectory;
use summa_core::dsl::IvfRoutingMode;
#[cfg(feature = "sync")]
use summa_core::dsl::{DenseVectorConfig, Document, SchemaBuilder};
#[cfg(feature = "sync")]
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::structures::CoarseCentroids;
use summa_core::structures::simd;

const ROUTING_DIM: usize = 768;
const ROUTING_NPROBE: usize = 64;

/// Whether the command-line filter selects a benchmark group. Criterion only
/// filters at `bench_function` granularity, so without this every group would
/// still pay its fixture cost (HNSW graph builds, a 200k-vector index) when a
/// different group is being measured.
fn selected(group: &str) -> bool {
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-') && arg.parse::<f64>().is_err())
        .collect();
    filters.is_empty()
        || filters
            .iter()
            .any(|filter| group.contains(filter.as_str()) || filter.contains(group))
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit_vector(dim: usize, state: &mut u64) -> Vec<f32> {
    let mut values: Vec<f32> = (0..dim)
        .map(|_| ((splitmix(state) >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        .collect();
    normalize(&mut values);
    values
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter_mut().for_each(|v| *v /= norm);
    }
}

/// Clustered unit-norm rows: `count` rows spread around `centers` centres with
/// `spread` noise, flattened row-major.
fn clustered_rows(dim: usize, count: usize, centers: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let centers: Vec<Vec<f32>> = (0..centers).map(|_| unit_vector(dim, &mut state)).collect();
    let mut rows = Vec::with_capacity(count * dim);
    for index in 0..count {
        let center = &centers[index % centers.len()];
        let noise = unit_vector(dim, &mut state);
        let mut row: Vec<f32> = center
            .iter()
            .zip(&noise)
            .map(|(c, n)| c + spread * n)
            .collect();
        normalize(&mut row);
        rows.extend_from_slice(&row);
    }
    rows
}

fn recall(reference: &[u32], candidate: &[u32]) -> f64 {
    let hits = candidate
        .iter()
        .filter(|leaf| reference.contains(leaf))
        .count();
    hits as f64 / reference.len().max(1) as f64
}

/// Flat vs HNSW probing over the *same* float centroids, so the crossover is
/// a property of the codebook size (mirrors `binary_routing_crossover`).
fn bench_coarse_routing_crossover(c: &mut Criterion) {
    if !selected("coarse_routing_crossover") {
        return;
    }
    let mut group = c.benchmark_group("coarse_routing_crossover");
    group.sample_size(20);
    let probe_count = 64usize;
    let probes = clustered_rows(ROUTING_DIM, probe_count, 16, 0.5, 99);
    for clusters in [1_024usize, 4_096, 8_192, 16_384, 32_768] {
        let centroids = clustered_rows(ROUTING_DIM, clusters, 64, 0.35, clusters as u64);
        let codebook = CoarseCentroids::from_leaf_centroids(
            ROUTING_DIM,
            centroids,
            IvfRoutingMode::Hnsw,
            7,
            "bench",
        );
        // Report graph recall against the exact flat pass once per size so a
        // latency win is never read without its accuracy cost.
        let mut recall_sum = 0.0;
        for probe in probes.chunks_exact(ROUTING_DIM) {
            let flat = codebook.probe(probe, ROUTING_NPROBE, IvfRoutingMode::Flat);
            let graph = codebook.probe(probe, ROUTING_NPROBE, IvfRoutingMode::Hnsw);
            recall_sum += recall(&flat.cluster_ids, &graph.cluster_ids);
        }
        eprintln!(
            "coarse_routing_crossover: {clusters} leaves × dim {ROUTING_DIM}: HNSW recall@{ROUTING_NPROBE} vs flat = {:.3}",
            recall_sum / probe_count as f64
        );
        for mode in [IvfRoutingMode::Flat, IvfRoutingMode::Hnsw] {
            group.bench_with_input(
                BenchmarkId::new(format!("probe_{mode:?}"), clusters),
                &mode,
                |bencher, &mode| {
                    let mut cursor = 0usize;
                    bencher.iter(|| {
                        let start = cursor * ROUTING_DIM;
                        cursor = (cursor + 1) % probe_count;
                        let plan = codebook.probe(
                            black_box(&probes[start..start + ROUTING_DIM]),
                            ROUTING_NPROBE,
                            mode,
                        );
                        black_box(plan.cluster_ids.len())
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(feature = "sync")]
const SEARCH_DIM: usize = 128;
#[cfg(feature = "sync")]
const SEARCH_DOCS: usize = 200_000;
#[cfg(feature = "sync")]
const SEARCH_CLUSTERS: usize = 256;
#[cfg(feature = "sync")]
const SEARCH_K: usize = 10;
#[cfg(feature = "sync")]
const SEARCH_RERANK_FACTOR: f32 = 2.0;

#[cfg(feature = "sync")]
struct SearchFixture {
    _dir: tempfile::TempDir,
    _index: Index<MmapDirectory>,
    segment: std::sync::Arc<summa_core::segment::SegmentReader>,
    field: summa_core::dsl::Field,
    queries: Vec<Vec<f32>>,
}

/// Build one merged on-disk IVF-TQ segment: clustered corpus, trained coarse
/// router, generation rewrite — exactly what production serves from.
#[cfg(feature = "sync")]
fn build_search_fixture(runtime: &tokio::runtime::Runtime, soar: bool) -> SearchFixture {
    let config = if soar {
        DenseVectorConfig::ivf_tq(SEARCH_DIM, Some(SEARCH_CLUSTERS), 32)
    } else {
        DenseVectorConfig::ivf_tq(SEARCH_DIM, Some(SEARCH_CLUSTERS), 32).without_soar()
    };
    let corpus = clustered_rows(SEARCH_DIM, SEARCH_DOCS, 512, 0.6, 4242);
    let mut state = 777u64;
    let queries: Vec<Vec<f32>> = (0..64)
        .map(|q| {
            let base = &corpus[(q * 977 % SEARCH_DOCS) * SEARCH_DIM..][..SEARCH_DIM];
            let noise = unit_vector(SEARCH_DIM, &mut state);
            let mut row: Vec<f32> = base.iter().zip(&noise).map(|(v, n)| v + 0.3 * n).collect();
            normalize(&mut row);
            row
        })
        .collect();

    let dir_handle = tempfile::tempdir().expect("bench tempdir");
    let dir = MmapDirectory::new(dir_handle.path());
    let mut sb = SchemaBuilder::default();
    let field = sb.add_dense_vector_field_with_config("embedding", true, false, config);
    let schema = sb.build();
    let index_config = IndexConfig::default();

    runtime.block_on(async {
        let mut writer = IndexWriter::create(dir.clone(), schema, index_config.clone())
            .await
            .expect("create writer");
        for row in corpus.chunks_exact(SEARCH_DIM) {
            loop {
                let mut doc = Document::new();
                doc.add_dense_vector(field, row.to_vec());
                match writer.add_document(doc) {
                    Ok(()) => break,
                    Err(summa_core::Error::QueueFull) => {
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                    Err(error) => panic!("add document: {error:?}"),
                }
            }
        }
        writer.commit().await.expect("commit");
        writer.force_merge().await.expect("force merge");
        drop(writer);
        let writer = IndexWriter::open(dir.clone(), index_config.clone())
            .await
            .expect("reopen writer");
        writer
            .build_vector_index()
            .await
            .expect("train coarse centroids");
        drop(writer);
    });

    let index = runtime
        .block_on(Index::open(dir, index_config))
        .expect("open index");
    let segments = runtime.block_on(index.segment_readers()).expect("segments");
    assert_eq!(segments.len(), 1, "fixture must be one merged segment");
    let segment = segments.into_iter().next().expect("one segment");
    assert!(
        matches!(
            segment.vector_indexes().get(&field.0),
            Some(summa_core::segment::VectorIndex::IvfTq { .. })
        ),
        "fixture segment must carry an IVF-TQ payload"
    );
    SearchFixture {
        _dir: dir_handle,
        _index: index,
        segment,
        field,
        queries,
    }
}

/// End-to-end IVF-TQ segment search: plan build, probed lane loop, collector,
/// exact rerank. `nprobe` 32 stays on the serial scan; 128 crosses the
/// parallel fan-out threshold on this corpus (100k probed postings).
#[cfg(feature = "sync")]
fn bench_ivf_tq_search(c: &mut Criterion) {
    if !selected("ivf_tq_search") {
        return;
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let mut group = c.benchmark_group("ivf_tq_search");
    group.sample_size(30);
    for soar in [false, true] {
        let fixture = build_search_fixture(&runtime, soar);
        let label = if soar { "soar" } else { "nosoar" };
        for nprobe in [32usize, 128] {
            group.bench_with_input(
                BenchmarkId::new(label, nprobe),
                &nprobe,
                |bencher, &nprobe| {
                    let mut cursor = 0usize;
                    bencher.iter(|| {
                        let query = &fixture.queries[cursor];
                        cursor = (cursor + 1) % fixture.queries.len();
                        let hits = fixture
                            .segment
                            .search_dense_vector_sync(
                                fixture.field,
                                black_box(query),
                                SEARCH_K,
                                nprobe,
                                SEARCH_RERANK_FACTOR,
                                summa_core::query::MultiValueCombiner::default(),
                            )
                            .expect("search");
                        black_box(hits.len())
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "sync"))]
fn bench_ivf_tq_search(_: &mut Criterion) {}

const BATCH_VECTORS: usize = 4_096;
const BATCH_DIM: usize = 768;

fn bench_batch_kernels(c: &mut Criterion) {
    if !selected("dense_batch_kernels") && !selected("dot_product_f32_tail") {
        return;
    }
    let mut state = 31u64;
    let query = unit_vector(BATCH_DIM, &mut state);
    let vectors = clustered_rows(BATCH_DIM, BATCH_VECTORS, 32, 0.7, 5);
    let vectors_f16: Vec<u16> = vectors.iter().map(|&v| simd::f32_to_f16(v)).collect();
    let vectors_f16_bytes: Vec<u8> = vectors_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let vectors_u8: Vec<u8> = vectors
        .iter()
        .map(|&v| simd::f32_to_u8_saturating(v))
        .collect();
    let query_f16: Vec<u16> = query.iter().map(|&v| simd::f32_to_f16(v)).collect();
    let inv_norm_q = 1.0 / simd::norm_f32(&query);
    let mut scores = vec![0.0f32; BATCH_VECTORS];

    let mut group = c.benchmark_group("dense_batch_kernels");
    group.throughput(Throughput::Elements(BATCH_VECTORS as u64));
    group.bench_function(BenchmarkId::new("cosine_f32", BATCH_DIM), |b| {
        b.iter(|| {
            simd::batch_cosine_scores_precomp(
                black_box(&query),
                black_box(&vectors),
                BATCH_DIM,
                &mut scores,
                inv_norm_q,
            );
            black_box(scores[0])
        })
    });
    group.bench_function(BenchmarkId::new("dot_f32", BATCH_DIM), |b| {
        b.iter(|| {
            simd::batch_dot_scores_precomp(
                black_box(&query),
                black_box(&vectors),
                BATCH_DIM,
                &mut scores,
                inv_norm_q,
            );
            black_box(scores[0])
        })
    });
    group.bench_function(BenchmarkId::new("cosine_f16", BATCH_DIM), |b| {
        b.iter(|| {
            simd::batch_cosine_scores_f16_precomp(
                black_box(&query_f16),
                black_box(&vectors_f16_bytes),
                BATCH_DIM,
                &mut scores,
                inv_norm_q,
            );
            black_box(scores[0])
        })
    });
    group.bench_function(BenchmarkId::new("cosine_u8", BATCH_DIM), |b| {
        b.iter(|| {
            simd::batch_cosine_scores_u8_precomp(
                black_box(&query),
                black_box(&vectors_u8),
                BATCH_DIM,
                &mut scores,
                inv_norm_q,
            );
            black_box(scores[0])
        })
    });
    group.finish();

    // Remainder-tail sensitivity: dims that leave 4/8/12 trailing lanes after
    // the 16-wide NEON (32-wide AVX2) main loop.
    let mut group = c.benchmark_group("dot_product_f32_tail");
    for dim in [100usize, 200, 300, 768] {
        let a = unit_vector(dim, &mut state);
        let rows = clustered_rows(dim, 1_024, 8, 0.5, dim as u64);
        group.throughput(Throughput::Elements(1_024));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for row in rows.chunks_exact(dim) {
                    acc += simd::dot_product_f32(black_box(&a), row, dim);
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_coarse_routing_crossover,
    bench_ivf_tq_search,
    bench_batch_kernels
);
criterion_main!(benches);
