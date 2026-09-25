//! Binary dense-vector microbenchmarks: Hamming kernels, coarse routing and
//! build-time assignment.
//!
//! The binary path had no benchmark coverage at all, which is how a 4x wider
//! construction beam and a per-row SIMD dispatch pattern shipped as "quality
//! improvements" with an unmeasured cost. Each group here pairs the current
//! implementation against the shape it replaced, so the win (or its absence) is
//! reproducible rather than asserted.
//!
//! Prod-shaped defaults: 2,560-bit codes (320 bytes) at ~16k leaves.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use summa_core::dsl::IvfRoutingMode;
use summa_core::structures::simd::{
    HammingKernel, batch_hamming_scores, hamming_distance, scores_from_hamming,
};
use summa_core::structures::{BinaryCoarseQuantizer, BinaryIvfConfig};

/// Prod field width: 2,560-bit codes.
const CODE_BYTES: usize = 320;
/// Rows per scan measured by the kernel group.
const SCAN_ROWS: usize = 1_024;

fn random_codes(rows: usize, byte_len: usize, seed: u64) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut codes = vec![0u8; rows * byte_len];
    rng.fill_bytes(&mut codes);
    codes
}

/// Clustered corpus: `groups` anchors, each with `per_group` near-duplicates.
/// Uniform noise would make every routing budget look equally good.
fn clustered_codes(
    byte_len: usize,
    groups: usize,
    per_group: usize,
    flips: usize,
    seed: u64,
) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut codes = Vec::with_capacity(groups * per_group * byte_len);
    for _ in 0..groups {
        let mut anchor = vec![0u8; byte_len];
        rng.fill_bytes(&mut anchor);
        for _ in 0..per_group {
            let mut code = anchor.clone();
            for _ in 0..flips {
                let bit = rng.random_range(0..byte_len * 8);
                code[bit / 8] ^= 1 << (bit % 8);
            }
            codes.extend_from_slice(&code);
        }
    }
    codes
}

/// Scanning one query against many stored codes.
///
/// `dispatch_per_row` reproduces the previous inner loop: one `hamming_distance`
/// call per row, each repeating runtime feature detection and rebuilding the
/// AVX2 nibble table. `resolved_batch` and `resolved_gather` dispatch once for
/// the whole run and score four rows per kernel invocation.
fn bench_hamming_kernels(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary_hamming_scan");
    let kernel = HammingKernel::resolve();
    for byte_len in [8usize, 32, 128, CODE_BYTES] {
        let query = random_codes(1, byte_len, 1);
        let db = random_codes(SCAN_ROWS, byte_len, 2);
        let ids: Vec<u32> = (0..SCAN_ROWS as u32).rev().collect();
        let mut distances = vec![0u32; SCAN_ROWS];
        let mut scores = vec![0f32; SCAN_ROWS];

        group.throughput(Throughput::Bytes((SCAN_ROWS * byte_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("dispatch_per_row", byte_len),
            &byte_len,
            |bencher, &byte_len| {
                bencher.iter(|| {
                    for (row, slot) in distances.iter_mut().enumerate() {
                        *slot = hamming_distance(&query, &db[row * byte_len..(row + 1) * byte_len]);
                    }
                    black_box(distances[0])
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("resolved_batch", byte_len),
            &byte_len,
            |bencher, &byte_len| {
                bencher.iter(|| {
                    kernel.distances(&query, &db, byte_len, &mut distances);
                    black_box(distances[0])
                });
            },
        );
        // Plain u64 `count_ones` loop: the candidate replacement for short
        // codes, where the SIMD kernel's horizontal reduction dominates.
        group.bench_with_input(
            BenchmarkId::new("scalar_u64_batch", byte_len),
            &byte_len,
            |bencher, &byte_len| {
                bencher.iter(|| {
                    HammingKernel::Scalar.distances(&query, &db, byte_len, &mut distances);
                    black_box(distances[0])
                });
            },
        );
        // Graph routing visits scattered rows; this is the gather path's cost.
        group.bench_with_input(
            BenchmarkId::new("resolved_gather", byte_len),
            &byte_len,
            |bencher, &byte_len| {
                bencher.iter(|| {
                    kernel.gather_distances(&query, &db, byte_len, &ids, &mut distances);
                    black_box(distances[0])
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("scores_resolved", byte_len),
            &byte_len,
            |bencher, &byte_len| {
                bencher.iter(|| {
                    scores_from_hamming(kernel, &query, &db, byte_len, byte_len * 8, &mut scores);
                    black_box(scores[0])
                });
            },
        );
    }
    group.finish();
}

/// Per-leaf versus per-parent scoring for two-level routing.
///
/// The old probe loop called the batch kernel once per leaf on a single-row
/// slice, paying argument validation, a reciprocal divide and feature detection
/// per centroid. Children of a parent are contiguous, so one call covers them
/// all.
fn bench_leaf_scoring_shape(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary_leaf_scoring");
    let leaves = 338; // routing_parent_count(114_309) children per parent
    let query = random_codes(1, CODE_BYTES, 3);
    let centroids = random_codes(leaves, CODE_BYTES, 4);
    let kernel = HammingKernel::resolve();
    let dim_bits = CODE_BYTES * 8;

    group.throughput(Throughput::Elements(leaves as u64));
    group.bench_function("one_call_per_leaf", |bencher| {
        let mut score = [0f32];
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(leaves);
        bencher.iter(|| {
            candidates.clear();
            for leaf in 0..leaves {
                let offset = leaf * CODE_BYTES;
                batch_hamming_scores(
                    &query,
                    &centroids[offset..offset + CODE_BYTES],
                    CODE_BYTES,
                    dim_bits,
                    &mut score,
                );
                candidates.push((leaf as u32, score[0]));
            }
            black_box(candidates.len())
        });
    });
    group.bench_function("one_call_per_parent", |bencher| {
        let mut scores = vec![0f32; leaves];
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(leaves);
        bencher.iter(|| {
            scores_from_hamming(
                kernel,
                &query,
                &centroids,
                CODE_BYTES,
                dim_bits,
                &mut scores,
            );
            candidates.clear();
            candidates.extend((0..leaves as u32).zip(scores.iter().copied()));
            black_box(candidates.len())
        });
    });
    group.finish();
}

fn trained_quantizer(
    clusters: usize,
    routing: IvfRoutingMode,
    codes: &[u8],
    byte_len: usize,
) -> BinaryCoarseQuantizer {
    let points = codes.len() / byte_len;
    let mut config = BinaryIvfConfig::new(byte_len * 8, clusters);
    config.train_iters = 3;
    config.max_train_samples = points;
    config.routing = routing;
    BinaryCoarseQuantizer::train(config, codes, points, "bench").expect("binary coarse training")
}

/// Query-time probing and build-time assignment against a realistic codebook.
///
/// `assign` is the operation every vector of a rebuilt segment pays once; its
/// cost is linear in the construction beam, which is why the beam floor is set
/// from measured recall rather than from "construction can afford it".
fn bench_routing(criterion: &mut Criterion) {
    // One anchor per leaf (plus slack): fewer distinct anchors than leaves
    // produces near-duplicate centroids, and greedy descent then tie-walks
    // through equidistant nodes instead of measuring realistic routing.
    let clusters = 16_384;
    let corpus = clustered_codes(CODE_BYTES, 20_000, 3, 32, 7);
    let probes = clustered_codes(CODE_BYTES, 64, 1, 40, 11);

    let mut group = criterion.benchmark_group("binary_routing");
    group.sample_size(20);
    for routing in [
        IvfRoutingMode::Flat,
        IvfRoutingMode::TwoLevel,
        IvfRoutingMode::Hnsw,
    ] {
        let quantizer = trained_quantizer(clusters, routing, &corpus, CODE_BYTES);
        let label = format!("{routing:?}");

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("probe_nprobe64", &label),
            &routing,
            |bencher, &routing| {
                let mut cursor = 0usize;
                bencher.iter(|| {
                    let start = cursor * CODE_BYTES;
                    cursor = (cursor + 1) % (probes.len() / CODE_BYTES);
                    let plan = quantizer
                        .probe(&probes[start..start + CODE_BYTES], 64, routing)
                        .expect("binary probe");
                    black_box(plan.cluster_ids.len())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("assign", &label),
            &routing,
            |bencher, &routing| {
                let mut cursor = 0usize;
                bencher.iter(|| {
                    let start = cursor * CODE_BYTES;
                    cursor = (cursor + 1) % (probes.len() / CODE_BYTES);
                    black_box(
                        quantizer
                            .assign(&probes[start..start + CODE_BYTES], routing)
                            .expect("binary assign"),
                    )
                });
            },
        );
    }
    group.finish();
}

/// Where graph routing actually starts beating an exact flat scan.
///
/// A base-layer search with beam `ef` and degree `2M` touches roughly
/// `ef * 2M` adjacency slots, so below about that many leaves it visits the
/// whole codebook anyway — with random access and heap traffic instead of one
/// sequential SIMD pass, and approximately instead of exactly. Both modes are
/// probed on the *same* centroids here, so the crossing point is a property of
/// the codebook size alone.
fn bench_routing_crossover(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary_routing_crossover");
    group.sample_size(10);
    for clusters in [4_096usize, 16_384, 32_768, 65_536] {
        let corpus = clustered_codes(CODE_BYTES, clusters + clusters / 4, 3, 32, 17);
        let quantizer = trained_quantizer(clusters, IvfRoutingMode::Hnsw, &corpus, CODE_BYTES);
        let probes = clustered_codes(CODE_BYTES, 32, 1, 40, 19);
        for mode in [IvfRoutingMode::Flat, IvfRoutingMode::Hnsw] {
            group.bench_with_input(
                BenchmarkId::new(format!("probe_{mode:?}"), clusters),
                &mode,
                |bencher, &mode| {
                    let mut cursor = 0usize;
                    bencher.iter(|| {
                        let start = cursor * CODE_BYTES;
                        cursor = (cursor + 1) % (probes.len() / CODE_BYTES);
                        let plan = quantizer
                            .probe(&probes[start..start + CODE_BYTES], 64, mode)
                            .expect("binary probe");
                        black_box(plan.cluster_ids.len())
                    });
                },
            );
        }
    }
    group.finish();
}

/// k-majority training: seeding, Lloyd assignment and the majority update that
/// used to allocate and zero a 20 KiB counter block per centroid per iteration.
fn bench_training(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("binary_coarse_training");
    group.sample_size(10);
    let corpus = clustered_codes(CODE_BYTES, 512, 16, 24, 13);
    let points = corpus.len() / CODE_BYTES;
    group.throughput(Throughput::Elements(points as u64));
    for clusters in [256usize, 1_024] {
        group.bench_with_input(
            BenchmarkId::new("k_majority", clusters),
            &clusters,
            |bencher, &clusters| {
                bencher.iter(|| {
                    let quantizer =
                        trained_quantizer(clusters, IvfRoutingMode::Flat, &corpus, CODE_BYTES);
                    black_box(quantizer.num_clusters)
                });
            },
        );
    }

    // Hierarchical training: sqrt(K) parent cells plus one child codebook per
    // parent. This is what every production-sized codebook uses, and the child
    // codebooks are the level that parallelises.
    for clusters in [1_024usize, 4_096] {
        group.bench_with_input(
            BenchmarkId::new("hierarchical", clusters),
            &clusters,
            |bencher, &clusters| {
                bencher.iter(|| {
                    let quantizer =
                        trained_quantizer(clusters, IvfRoutingMode::TwoLevel, &corpus, CODE_BYTES);
                    black_box(quantizer.num_clusters)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hamming_kernels,
    bench_leaf_scoring_shape,
    bench_routing,
    bench_routing_crossover,
    bench_training
);
criterion_main!(benches);
