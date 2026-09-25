//! Collector-only experiments, including conservative threshold feedback.
//! No public search behavior is changed. See docs/collector-benchmark.md.

mod collectors;

use std::hint::black_box;
use std::time::Duration;

use collectors::{Collect, LoserTree, Partition, Results};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use summa_core::query::{Bm25Params, HeapEntry, ScoreCollector};

const BLOCK: usize = 128;
const COUNT: usize = 100_000;
const SEED: u64 = 0xc011_ec70;

struct Fixture {
    name: &'static str,
    entries: Vec<HeapEntry>,
    bounds: Vec<f32>,
}

impl Fixture {
    fn new(name: &'static str, mut scores: Vec<f32>) -> Self {
        match name {
            "improving" => scores.sort_unstable_by(f32::total_cmp),
            "deteriorating" => scores.sort_unstable_by(|a, b| b.total_cmp(a)),
            _ => {}
        }
        let entries: Vec<_> = scores
            .into_iter()
            .enumerate()
            .map(|(i, score)| HeapEntry {
                // Permute stable IDs to exercise ties independent of arrival order.
                // Odd multiplication modulo 2^32 is injective over these indices.
                doc_id: (i as u32).wrapping_mul(2_654_435_761),
                score,
                ordinal: (i % 7) as u16,
            })
            .collect();
        let bounds = entries
            .chunks(BLOCK)
            .map(|block| block.iter().map(|entry| entry.score).fold(0.0, f32::max))
            .collect();
        Self {
            name,
            entries,
            bounds,
        }
    }
}

/// Synthetic streams use the canonical scorer, outside timed collection. Topic
/// blocks have locally similar scores; improving/declining are stress controls.
fn fixtures() -> Vec<Fixture> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let params = Bm25Params::default();
    let scores: Vec<_> = (0..COUNT)
        .map(|_| {
            let length = rng.random_range(50..1500) as f32;
            (0..3)
                .map(|term| {
                    params.score(
                        rng.random_range(1..20) as f32,
                        [0.3, 1.1, 2.7][term],
                        length,
                        500.0,
                    )
                })
                .sum()
        })
        .collect();
    let clustered = (0..COUNT)
        .map(|i| {
            let level = ((i / BLOCK * 37) % 101) as f32 / 100.0;
            0.1 + level * 10.0 + rng.random::<f32>() * 0.05
        })
        .collect();
    vec![
        Fixture::new("bm25", scores.clone()),
        Fixture::new("improving", scores.clone()),
        Fixture::new("deteriorating", scores.clone()),
        Fixture::new("ties", scores.iter().map(|v| v.floor()).collect()),
        Fixture::new("clustered", clustered),
        Fixture::new("short", scores[..128].to_vec()),
    ]
}

/// Match the existing eight-score admission screen. Pruned mode uses exact
/// block maxima to isolate threshold feedback, not production bound tightness.
fn run<C: Collect, const PRUNE: bool>(fixture: &Fixture, k: usize) -> Results {
    let mut collector = C::new(k);
    for (entries, &bound) in fixture.entries.chunks(BLOCK).zip(&fixture.bounds) {
        if PRUNE && bound < collector.threshold() {
            continue;
        }
        for entries in entries.chunks(8) {
            let threshold = collector.threshold();
            let mut mask = 0u8;
            for (i, entry) in entries.iter().enumerate() {
                mask |= u8::from(entry.score >= threshold) << i;
            }
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                collector.add(entries[i]);
                mask &= mask - 1;
            }
        }
    }
    collector.finish()
}

fn signature(results: Results) -> Vec<(u32, u32, u16)> {
    results
        .into_iter()
        .map(|(doc, score, ordinal)| (doc, score.to_bits(), ordinal))
        .collect()
}

fn oracle(entries: &[HeapEntry], k: usize) -> Vec<(u32, u32, u16)> {
    let mut entries = entries.to_vec();
    entries.sort_unstable();
    entries.truncate(k);
    entries
        .iter()
        .map(|entry| (entry.doc_id, entry.score.to_bits(), entry.ordinal))
        .collect()
}

fn verify_collector<C: Collect>() {
    let mut rng = StdRng::seed_from_u64(SEED);
    for n in [0, 1, 7, 31, 128, 1001] {
        let entries: Vec<_> = (0..n)
            .map(|i| HeapEntry {
                doc_id: (n - i) / 3,
                ordinal: (i % 3) as u16,
                score: rng.random_range(-8..8) as f32,
            })
            .collect();
        for k in [0, 1, 2, 3, 7, 10, 31, 100, 1000] {
            let mut collector = C::new(k);
            for &entry in &entries {
                collector.add(entry);
            }
            assert_eq!(
                signature(collector.finish()),
                oracle(&entries, k),
                "{} n={n} k={k}",
                C::NAME
            );
        }
    }
    // Finite ties, infinities, signed zeros, and both signs of NaN exercise the
    // production total comparator through direct admission (no finite screen).
    let entries: Vec<_> = [
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        f32::INFINITY,
        f32::NAN,
        -f32::NAN,
    ]
    .into_iter()
    .cycle()
    .take(36)
    .enumerate()
    .map(|(i, score)| HeapEntry {
        doc_id: 36 - i as u32,
        score,
        ordinal: 0,
    })
    .collect();
    for k in 0..=37 {
        let mut collector = C::new(k);
        for &entry in &entries {
            collector.add(entry);
        }
        assert_eq!(
            signature(collector.finish()),
            oracle(&entries, k),
            "{} float k={k}",
            C::NAME
        );
    }
}

fn evidence<C: Collect>(fixture: &Fixture, k: usize) {
    let mut collector = C::new(k);
    let mut visited = 0;
    let mut skipped = 0;
    let mut admitted = 0;
    for (entries, &bound) in fixture.entries.chunks(BLOCK).zip(&fixture.bounds) {
        if bound < collector.threshold() {
            skipped += 1;
            continue;
        }
        visited += entries.len();
        for entries in entries.chunks(8) {
            let threshold = collector.threshold();
            for &entry in entries {
                if entry.score >= threshold {
                    admitted += 1;
                    collector.add(entry);
                }
            }
        }
    }
    println!(
        "EVIDENCE {} fixture={} k={k} backing_bytes={} object_bytes={} visited={visited} skipped_blocks={skipped} admission_calls={admitted}",
        C::NAME,
        fixture.name,
        C::backing_bytes(k),
        size_of::<C>()
    );
}

fn bench<C: Collect, const PRUNE: bool>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    fixture: &Fixture,
    k: usize,
) {
    assert_eq!(
        signature(run::<C, PRUNE>(fixture, k)),
        oracle(&fixture.entries, k),
        "{} {} k={k}",
        C::NAME,
        fixture.name
    );
    if PRUNE {
        evidence::<C>(fixture, k);
    }
    group.bench_function(BenchmarkId::new(C::NAME, k), |bencher| {
        bencher.iter(|| black_box(run::<C, PRUNE>(black_box(fixture), black_box(k))));
    });
}

fn suite<const PRUNE: bool>(c: &mut Criterion, fixtures: &[Fixture], reverse: bool) {
    for fixture in fixtures {
        if PRUNE && fixture.name != "clustered" {
            continue;
        }
        let mode = if PRUNE { "pruned" } else { "replay" };
        let mut group = c.benchmark_group(format!("collector/{mode}/{}", fixture.name));
        group.sample_size(30);
        group.warm_up_time(Duration::from_millis(300));
        group.measurement_time(Duration::from_secs(1));
        group.throughput(Throughput::Elements(fixture.entries.len() as u64));
        for k in [10, 100, 1000] {
            if reverse {
                bench::<LoserTree, PRUNE>(&mut group, fixture, k);
                bench::<Partition, PRUNE>(&mut group, fixture, k);
                bench::<ScoreCollector, PRUNE>(&mut group, fixture, k);
            } else {
                bench::<ScoreCollector, PRUNE>(&mut group, fixture, k);
                bench::<Partition, PRUNE>(&mut group, fixture, k);
                bench::<LoserTree, PRUNE>(&mut group, fixture, k);
            }
        }
        group.finish();
    }
}

fn collectors(c: &mut Criterion) {
    verify_collector::<ScoreCollector>();
    verify_collector::<Partition>();
    verify_collector::<LoserTree>();
    let fixtures = fixtures();
    let reverse = std::env::var_os("COLLECTOR_REVERSE").is_some();
    suite::<false>(c, &fixtures, reverse);
    suite::<true>(c, &fixtures, reverse);
}

criterion_group!(benches, collectors);
criterion_main!(benches);
