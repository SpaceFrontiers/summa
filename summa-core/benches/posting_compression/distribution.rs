//! Distribution-based benchmarks

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use summa_core::structures::{
    CompressedPostingList, EliasFanoPostingList, HorizontalBP128PostingList, OptP4DPostingList,
    PartitionedEFPostingList, RoaringPostingList, VerticalBP128PostingList,
};

use super::common::{Distribution, generate_postings};

/// Benchmark unified format selection
pub fn bench_unified_format(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_format");

    for dist in &[
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
    ] {
        for &size in &[1000, 10000, 50000] {
            let (doc_ids, term_freqs) = generate_postings(size, *dist);
            let name = format!("{}_{}", dist.name(), size);
            group.throughput(Throughput::Elements(size as u64));

            let list = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 1_000_000, 1.0);

            group.bench_with_input(BenchmarkId::new("iterate", &name), &size, |b, _| {
                b.iter(|| {
                    let mut iter = list.iterator();
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

/// Benchmark by distribution type
pub fn bench_by_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("by_distribution");

    let size = 10000;

    for dist in &[
        Distribution::Sparse,
        Distribution::Medium,
        Distribution::Dense,
        Distribution::Clustered,
        Distribution::Sequential,
    ] {
        let (doc_ids, term_freqs) = generate_postings(size, *dist);
        group.throughput(Throughput::Elements(size as u64));

        let horiz_bp128 = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let vert_bp128 = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let partitioned_ef = PartitionedEFPostingList::from_postings(&doc_ids, &term_freqs);
        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);
        let opt_p4d = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        group.bench_with_input(
            BenchmarkId::new("horiz_bp128", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = horiz_bp128.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vert_bp128", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = vert_bp128.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("elias_fano", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = elias_fano.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("partitioned_ef", dist.name()),
            &size,
            |b, _| {
                b.iter(|| {
                    let mut iter = partitioned_ef.iterator();
                    let mut sum = 0u64;
                    while iter.doc() != u32::MAX {
                        sum += iter.doc() as u64;
                        iter.advance();
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("roaring", dist.name()), &size, |b, _| {
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

        group.bench_with_input(BenchmarkId::new("opt_p4d", dist.name()), &size, |b, _| {
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

    group.finish();
}
