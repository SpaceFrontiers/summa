//! Iteration/decoding speed benchmarks

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::time::Instant;
use summa_core::structures::{
    EliasFanoPostingList, HorizontalBP128PostingList, OptP4DPostingList, PartitionedEFPostingList,
    RoaringPostingList, RoundedBP128PostingList, VerticalBP128PostingList,
};

use super::common::{Distribution, generate_postings};

/// Criterion benchmark for iteration speed
pub fn bench_iteration_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_speed");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        let horiz_bp128 = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);
        let opt_p4d = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        group.bench_with_input(BenchmarkId::new("horiz_bp128", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = horiz_bp128.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = elias_fano.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = roaring.iterator();
                iter.init();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("opt_p4d", size), &size, |b, _| {
            b.iter(|| {
                let mut iter = opt_p4d.iterator();
                let mut sum = 0u64;
                while iter.doc() != u32::MAX {
                    sum += iter.doc() as u64;
                    sum += iter.term_freq() as u64;
                    iter.advance();
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

/// Benchmark all formats iteration (for summary table)
pub fn bench_all_formats_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_formats_iteration");

    for dist in &[
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
    ] {
        for &size in &[1000, 10000, 50000] {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let name = format!("{}_{}", dist.name(), size);
            group.throughput(Throughput::Elements(size as u64));

            let horiz_bp128 = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
            let vert_bp128 = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
            let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
            let partitioned_ef = PartitionedEFPostingList::from_postings(&doc_ids, &term_freqs);
            let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);
            let opt_p4d = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

            group.bench_with_input(BenchmarkId::new("horiz_bp128", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = horiz_bp128.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            });

            group.bench_with_input(BenchmarkId::new("vert_bp128", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = vert_bp128.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            });

            group.bench_with_input(BenchmarkId::new("elias_fano", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = elias_fano.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            });

            group.bench_with_input(BenchmarkId::new("partitioned_ef", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = partitioned_ef.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            });

            group.bench_with_input(BenchmarkId::new("roaring", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = roaring.iterator();
                    iter.init();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            });

            group.bench_with_input(BenchmarkId::new("opt_p4d", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = opt_p4d.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            });
        }
    }

    group.finish();
}

/// Measure decoding/iteration speed (returns elements per microsecond)
/// Returns [HorizBP, RoundedBP, VertBP, EF, PEF, Roaring, OptP4D]
pub fn measure_decoding_speed(doc_ids: &[u32], term_freqs: &[u32], iterations: usize) -> [f64; 7] {
    let n = doc_ids.len();
    let bp = HorizontalBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let rounded = RoundedBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let simd = VerticalBP128PostingList::from_postings(doc_ids, term_freqs, 1.0);
    let ef = EliasFanoPostingList::from_postings(doc_ids, term_freqs);
    let pef = PartitionedEFPostingList::from_postings(doc_ids, term_freqs);
    let roar = RoaringPostingList::from_postings(doc_ids, term_freqs);
    let opt_p4d = OptP4DPostingList::from_postings(doc_ids, term_freqs, 1.0);

    // HorizBP
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = bp.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let bp_time = start.elapsed().as_micros() as f64;
    let bp_rate = (n * iterations) as f64 / bp_time;

    // RoundedBP128
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = rounded.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let rounded_time = start.elapsed().as_micros() as f64;
    let rounded_rate = (n * iterations) as f64 / rounded_time;

    // VertBP128
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = simd.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let simd_time = start.elapsed().as_micros() as f64;
    let simd_rate = (n * iterations) as f64 / simd_time;

    // Elias-Fano
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = ef.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let ef_time = start.elapsed().as_micros() as f64;
    let ef_rate = (n * iterations) as f64 / ef_time;

    // Partitioned EF
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = pef.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let pef_time = start.elapsed().as_micros() as f64;
    let pef_rate = (n * iterations) as f64 / pef_time;

    // Roaring
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = roar.iterator();
        iter.init();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let roar_time = start.elapsed().as_micros() as f64;
    let roar_rate = (n * iterations) as f64 / roar_time;

    // OptP4D
    let start = Instant::now();
    for _ in 0..iterations {
        let mut iter = opt_p4d.iterator();
        let mut sum = 0u64;
        while iter.doc() != u32::MAX {
            sum += iter.doc() as u64;
            iter.advance();
        }
        black_box(sum);
    }
    let opt_p4d_time = start.elapsed().as_micros() as f64;
    let opt_p4d_rate = (n * iterations) as f64 / opt_p4d_time;

    [
        bp_rate,
        rounded_rate,
        simd_rate,
        ef_rate,
        pef_rate,
        roar_rate,
        opt_p4d_rate,
    ]
}
