#![allow(clippy::chunks_exact_to_as_chunks)]

//! Dense ANN microbenchmarks: coarse training, TQ LUT16 block scoring, and
//! IVF-TQ plan build.
//!
//! The end-to-end method comparison (flat vs tq vs ivf_tq, with recall)
//! lives in the ignored `tq_dense_ann_benchmark` integration test; this file
//! isolates the per-block scoring kernel and the per-query plan cost.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use summa_core::dsl::IvfRoutingMode;
use summa_core::structures::vector::quantization::TqEncodeScratch;
use summa_core::structures::vector::quantization::{
    TQ_BLOCK_LANES, tq_pack_ivf_block, tq_score_ivf_block, tq_shared_codec,
};
use summa_core::structures::{CoarseCentroids, CoarseConfig, TqIvfQueryPlan, TqQueryPlan};

const DIM: usize = 768;
const BLOCKS: usize = 256; // 4,096 vectors per iteration

fn random_unit_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            vector.iter_mut().for_each(|value| *value /= norm);
            vector
        })
        .collect()
}

fn packed_blocks(dim: usize, seed: u64) -> (Vec<u8>, usize) {
    let codec = tq_shared_codec(dim);
    let padded = codec.padded_dim();
    let vectors = random_unit_vectors(BLOCKS * TQ_BLOCK_LANES, dim, seed);
    let mut scratch = TqEncodeScratch::default();
    let mut codes = Vec::new();
    let mut rows = vec![0u8; TQ_BLOCK_LANES * padded];
    let mut gammas = [0.0f32; TQ_BLOCK_LANES];
    let mut scales = [1.0f32; TQ_BLOCK_LANES];
    for block in vectors.chunks_exact(TQ_BLOCK_LANES) {
        for (lane, vector) in block.iter().enumerate() {
            let (scale, gamma) = codec.encode_residual_into(
                vector,
                &mut rows[lane * padded..(lane + 1) * padded],
                &mut scratch,
            );
            scales[lane] = scale;
            gammas[lane] = gamma;
        }
        let row_refs: Vec<&[u8]> = (0..TQ_BLOCK_LANES)
            .map(|lane| &rows[lane * padded..(lane + 1) * padded])
            .collect();
        tq_pack_ivf_block(&row_refs, &scales, &gammas, padded, &mut codes);
    }
    let block_bytes = codes.len() / BLOCKS;
    (codes, block_bytes)
}

fn bench_lut16_scan(c: &mut Criterion) {
    let codec = tq_shared_codec(DIM);
    let query = random_unit_vectors(1, DIM, 7).pop().unwrap();
    let plan = TqQueryPlan::build(&codec, &query);
    let (codes, block_bytes) = packed_blocks(DIM, 42);

    let mut group = c.benchmark_group("tq_lut16_scan");
    group.throughput(criterion::Throughput::Elements(
        (BLOCKS * TQ_BLOCK_LANES) as u64,
    ));
    group.bench_function(BenchmarkId::from_parameter(DIM), |b| {
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        b.iter(|| {
            for block in codes.chunks_exact(block_bytes) {
                tq_score_ivf_block(black_box(&plan), black_box(block), 0.25, &mut scores);
            }
            black_box(scores)
        });
    });
    group.finish();
}

fn bench_ivf_tq_plan(c: &mut Criterion) {
    let vectors = random_unit_vectors(10_000, DIM, 42);
    let mut coarse = CoarseCentroids::train(
        &CoarseConfig::new(DIM, 256).with_routing(IvfRoutingMode::Flat),
        &vectors,
        "bench",
    );
    coarse.version = summa_core::structures::mark_ivf_tq_cosine_generation(coarse.version);
    let codec = tq_shared_codec(DIM);
    let query = random_unit_vectors(1, DIM, 9).pop().unwrap();

    let mut group = c.benchmark_group("ivf_tq_plan");
    for nprobe in [16usize, 64] {
        group.bench_with_input(BenchmarkId::from_parameter(nprobe), &nprobe, |b, &n| {
            b.iter(|| {
                black_box(TqIvfQueryPlan::build(
                    &coarse,
                    &codec,
                    black_box(&query),
                    n,
                    IvfRoutingMode::Flat,
                ))
            });
        });
    }
    group.finish();
}

fn bench_ivf_coarse_training(c: &mut Criterion) {
    const TRAIN_DIM: usize = 16;
    const POINTS_PER_CENTROID: usize = 39;

    let mut group = c.benchmark_group("ivf_coarse_training");
    group.sample_size(10);
    // Exercise both sides of the adaptive initialization boundary. The larger
    // case guards the bounded-total-candidate k-means|| path without making
    // routine benchmark runs prohibitively long.
    for clusters in [64usize, 257] {
        let points = clusters * POINTS_PER_CENTROID;
        let vectors = random_unit_vectors(points, TRAIN_DIM, clusters as u64);
        let config = CoarseConfig::new(TRAIN_DIM, clusters)
            .with_routing(IvfRoutingMode::Flat)
            .with_max_iters(5);
        group.throughput(criterion::Throughput::Elements(points as u64));
        group.bench_with_input(BenchmarkId::new("clusters", clusters), &clusters, |b, _| {
            b.iter(|| {
                black_box(CoarseCentroids::train(
                    black_box(&config),
                    black_box(&vectors),
                    "bench",
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_lut16_scan,
    bench_ivf_tq_plan,
    bench_ivf_coarse_training
);
criterion_main!(benches);
