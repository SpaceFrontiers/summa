//! ScaNN microbenchmarks: quantized float routing, FastScan AH block scoring,
//! and binary ScaNN tree routing.
//!
//! Every group builds its model from a synthetic *persisted* artifact rather
//! than by training, so the measured code is exactly the production
//! `QuantizedFloatScannModelView` / `QuantizedBinaryScannModelView` route and
//! not an owned in-memory shortcut. Sizes follow docs/scann-streaming-index.md:
//! 768-dimensional float embeddings routed through a two-level 256 x 256 tree
//! (65,536 leaves), and 2,560-bit binary codes through 16,384 / 65,536 leaves.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use summa_core::structures::vector::scann::{
    AhCodebook, BinaryScannModel, BinaryScannSearchScratch, CENTERS_PER_BLOCK, FAST_SCAN_LANES,
    FastScanQuery, QuantizedBinaryScannModel, QuantizedFloatScannModel, ScannAhCodebook,
    ScannConfig, ScannEncoding, ScannRoutingLevel, ScannTrainedArtifact, ScannTrainedArtifactView,
    pack_fast_scan_block,
};

const FLOAT_DIM: usize = 768;
const AH_DIMS_PER_BLOCK: u16 = 2;
const BINARY_CODE_BYTES: usize = 320;
/// Trained-vector count large enough to satisfy every geometry floor here.
const TRAINED_VECTORS: u64 = 1 << 40;

fn even_child_offsets(parents: usize, children: usize) -> Vec<u32> {
    (0..=parents)
        .map(|parent| (parent * children / parents) as u32)
        .collect()
}

fn random_unit_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
    let norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
    query.iter_mut().for_each(|value| *value /= norm);
    query
}

fn float_level(count: usize, dim: usize, children: Option<usize>, seed: u64) -> ScannRoutingLevel {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut centroid_codes = vec![0u8; count * dim];
    rng.fill_bytes(&mut centroid_codes);
    ScannRoutingLevel {
        centroid_count: count as u32,
        centroid_codes,
        minimums: vec![-1.0; dim],
        steps: vec![2.0 / 255.0; dim],
        child_offsets: children
            .map_or_else(Vec::new, |children| even_child_offsets(count, children)),
    }
}

fn random_codebook(dim: usize, dims_per_block: u16, seed: u64) -> ScannAhCodebook {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let blocks = dim.div_ceil(usize::from(dims_per_block));
    ScannAhCodebook {
        dimensions_per_block: dims_per_block,
        centers_per_block: CENTERS_PER_BLOCK as u16,
        centers: (0..blocks * CENTERS_PER_BLOCK * usize::from(dims_per_block))
            .map(|_| rng.random::<f32>() * 0.2 - 0.1)
            .collect(),
    }
}

fn float_artifact(level_counts: &[usize], dim: usize) -> Vec<u8> {
    let levels: Vec<ScannRoutingLevel> = level_counts
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            float_level(
                count,
                dim,
                level_counts.get(index + 1).copied(),
                0x5ca1 + index as u64,
            )
        })
        .collect();
    ScannTrainedArtifact::new(
        7,
        TRAINED_VECTORS,
        ScannConfig {
            dimension: dim as u32,
            tree_levels: level_counts.len() as u8,
            num_leaves: *level_counts.last().unwrap() as u32,
            encoding: ScannEncoding::AsymmetricHash {
                dimensions_per_block: AH_DIMS_PER_BLOCK,
                bits_per_code: 4,
            },
        },
        levels,
        Some(random_codebook(dim, AH_DIMS_PER_BLOCK, 99)),
    )
    .expect("synthetic float artifact")
    .to_bytes()
    .expect("float artifact bytes")
}

/// Quantized float routing over the mmap-shaped artifact: this is the
/// production `ScannTrainedArtifactBytes::float_model().prepare_query` path.
fn bench_float_routing(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scann_float_routing");
    group.sample_size(20);
    let queries: Vec<Vec<f32>> = (0..16)
        .map(|seed| random_unit_query(FLOAT_DIM, 1_000 + seed))
        .collect();
    for (label, level_counts) in [
        ("two_level_256x256", vec![256usize, 65_536]),
        ("flat_16384", vec![16_384usize]),
    ] {
        let bytes = float_artifact(&level_counts, FLOAT_DIM);
        let view = ScannTrainedArtifactView::parse(&bytes).expect("artifact view");
        let model = QuantizedFloatScannModel::from_artifact_view(&view).expect("quantized model");
        let model_view = model.view(&bytes).expect("model view");
        for probes in [16usize, 64] {
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(
                BenchmarkId::new(format!("prepare_query_{label}"), probes),
                &probes,
                |bencher, &probes| {
                    let mut cursor = 0usize;
                    bencher.iter(|| {
                        let query = &queries[cursor];
                        cursor = (cursor + 1) % queries.len();
                        let plan = model_view.prepare_query(query, probes).expect("route");
                        black_box(plan.routed_leaves().len())
                    });
                },
            );
        }
    }
    group.finish();
}

/// FastScan AH block scoring: 4,096 rows (128 blocks of 32 lanes) against a
/// query lookup table with 32 / 96 / 384 AH blocks.
fn bench_fast_scan(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scann_fast_scan");
    const ROWS: usize = 4_096;
    const DIMS_PER_BLOCK: usize = 2;
    for blocks in [32usize, 96, 384] {
        let dim = blocks * DIMS_PER_BLOCK;
        let codebook = AhCodebook::from_artifact(
            dim,
            &random_codebook(dim, DIMS_PER_BLOCK as u16, 17 + blocks as u64),
        )
        .expect("codebook");
        let query = codebook
            .query_dot_product(&random_unit_query(dim, 3))
            .expect("AH query");
        let fast_query = FastScanQuery::new(&query);
        let mut rng = rand::rngs::StdRng::seed_from_u64(blocks as u64);
        let rows: Vec<u8> = (0..ROWS * blocks)
            .map(|_| rng.random_range(0..CENTERS_PER_BLOCK as u8))
            .collect();
        let mut packed = Vec::new();
        for group_rows in rows.chunks_exact(FAST_SCAN_LANES * blocks) {
            pack_fast_scan_block(group_rows, blocks, &mut packed).expect("pack");
        }
        let block_bytes = packed.len() / (ROWS / FAST_SCAN_LANES);

        group.throughput(Throughput::Elements(ROWS as u64));
        group.bench_with_input(
            BenchmarkId::new("score_block", blocks),
            &blocks,
            |bencher, _| {
                bencher.iter(|| {
                    let mut total = 0.0f32;
                    for codes in packed.chunks_exact(block_bytes) {
                        let scores = fast_query.score_block(codes, 0.25).expect("score");
                        total += scores[0] + scores[31];
                    }
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

fn binary_level(
    count: usize,
    byte_len: usize,
    children: Option<usize>,
    seed: u64,
) -> ScannRoutingLevel {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut centroid_codes = vec![0u8; count * byte_len];
    rng.fill_bytes(&mut centroid_codes);
    ScannRoutingLevel {
        centroid_count: count as u32,
        centroid_codes,
        minimums: Vec::new(),
        steps: Vec::new(),
        child_offsets: children
            .map_or_else(Vec::new, |children| even_child_offsets(count, children)),
    }
}

fn binary_artifact(level_counts: &[usize], byte_len: usize) -> ScannTrainedArtifact {
    let levels: Vec<ScannRoutingLevel> = level_counts
        .iter()
        .enumerate()
        .map(|(index, &count)| {
            binary_level(
                count,
                byte_len,
                level_counts.get(index + 1).copied(),
                0xb1a5 + index as u64,
            )
        })
        .collect();
    ScannTrainedArtifact::new(
        9,
        TRAINED_VECTORS,
        ScannConfig {
            dimension: (byte_len * 8) as u32,
            tree_levels: level_counts.len() as u8,
            num_leaves: *level_counts.last().unwrap() as u32,
            encoding: ScannEncoding::BinaryHamming,
        },
        levels,
        None,
    )
    .expect("synthetic binary artifact")
}

/// Binary ScaNN routing: owned model and the mmap-shaped quantized view.
fn bench_binary_routing(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("scann_binary_routing");
    group.sample_size(20);
    let mut rng = rand::rngs::StdRng::seed_from_u64(77);
    let mut queries = vec![0u8; 16 * BINARY_CODE_BYTES];
    rng.fill_bytes(&mut queries);
    for (label, level_counts) in [
        ("two_level_128x128", vec![128usize, 16_384]),
        ("two_level_256x256", vec![256usize, 65_536]),
    ] {
        let artifact = binary_artifact(&level_counts, BINARY_CODE_BYTES);
        let owned = BinaryScannModel::from_artifact(&artifact).expect("owned binary model");
        let bytes = artifact.to_bytes().expect("binary artifact bytes");
        let view = ScannTrainedArtifactView::parse(&bytes).expect("artifact view");
        let quantized =
            QuantizedBinaryScannModel::from_artifact_view(&view).expect("quantized binary model");
        let quantized_view = quantized.view(&bytes).expect("binary model view");
        let mut scratch = BinaryScannSearchScratch::default();
        for nprobe in [16usize, 64] {
            group.throughput(Throughput::Elements(1));
            group.bench_with_input(
                BenchmarkId::new(format!("probe_view_{label}"), nprobe),
                &nprobe,
                |bencher, &nprobe| {
                    let mut cursor = 0usize;
                    bencher.iter(|| {
                        let start = cursor * BINARY_CODE_BYTES;
                        cursor = (cursor + 1) % 16;
                        let plan = quantized_view
                            .probe(
                                &queries[start..start + BINARY_CODE_BYTES],
                                nprobe,
                                64,
                                &mut scratch,
                            )
                            .expect("probe");
                        black_box(plan.leaf_ids.len())
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("probe_owned_{label}"), nprobe),
                &nprobe,
                |bencher, &nprobe| {
                    let mut cursor = 0usize;
                    bencher.iter(|| {
                        let start = cursor * BINARY_CODE_BYTES;
                        cursor = (cursor + 1) % 16;
                        let plan = owned
                            .probe(
                                &queries[start..start + BINARY_CODE_BYTES],
                                nprobe,
                                64,
                                &mut scratch,
                            )
                            .expect("probe");
                        black_box(plan.leaf_ids.len())
                    });
                },
            );
        }
        group.bench_function(
            BenchmarkId::new(format!("assign_view_{label}"), 1),
            |bencher| {
                let mut cursor = 0usize;
                bencher.iter(|| {
                    let start = cursor * BINARY_CODE_BYTES;
                    cursor = (cursor + 1) % 16;
                    black_box(
                        quantized_view
                            .assign(&queries[start..start + BINARY_CODE_BYTES], &mut scratch)
                            .expect("assign"),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_float_routing,
    bench_fast_scan,
    bench_binary_routing
);
criterion_main!(benches);
