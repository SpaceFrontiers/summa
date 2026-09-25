//! Seek speed benchmarks

use std::hint::black_box;

use criterion::{Criterion, Throughput};
use summa_core::structures::{EliasFanoPostingList, HorizontalBP128PostingList, OptP4DPostingList};

use super::common::{Distribution, generate_postings};

/// Criterion benchmark for seek speed
pub fn bench_seek_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("seek_speed");

    let size = 50000;
    let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);

    // Generate seek targets
    let seek_targets: Vec<u32> = (0..1000).map(|i| doc_ids[i * (size / 1000)]).collect();

    let horiz_bp128 = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
    let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
    let opt_p4d = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

    group.throughput(Throughput::Elements(seek_targets.len() as u64));

    group.bench_function("horiz_bp128_seek", |b| {
        b.iter(|| {
            let mut iter = horiz_bp128.iterator();
            let mut sum = 0u64;
            for &target in &seek_targets {
                sum += iter.seek(target) as u64;
            }
            black_box(sum)
        })
    });

    group.bench_function("elias_fano_seek", |b| {
        b.iter(|| {
            let mut iter = elias_fano.iterator();
            let mut sum = 0u64;
            for &target in &seek_targets {
                sum += iter.seek(target) as u64;
            }
            black_box(sum)
        })
    });

    group.bench_function("opt_p4d_seek", |b| {
        b.iter(|| {
            let mut iter = opt_p4d.iterator();
            let mut sum = 0u64;
            for &target in &seek_targets {
                sum += iter.seek(target) as u64;
            }
            black_box(sum)
        })
    });

    group.finish();
}
