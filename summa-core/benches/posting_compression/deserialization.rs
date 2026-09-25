//! Deserialization speed benchmarks

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::io::Cursor;
use summa_core::structures::{
    CompressedPostingList, EliasFanoPostingList, HorizontalBP128PostingList, OptP4DPostingList,
    PartitionedEFPostingList, RoaringPostingList, VerticalBP128PostingList,
};

use super::common::{Distribution, generate_postings};

/// Benchmark deserialization speed
pub fn bench_deserialization_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialization_speed");

    for &size in &[1000, 10000, 50000] {
        let (doc_ids, term_freqs) = generate_postings(size, Distribution::Medium);
        group.throughput(Throughput::Elements(size as u64));

        // Serialize each format
        let horiz_bp128 = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut bp_buf = Vec::new();
        horiz_bp128.serialize(&mut bp_buf).unwrap();

        let vert_bp128 = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut simd_buf = Vec::new();
        vert_bp128.serialize(&mut simd_buf).unwrap();

        let elias_fano = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let mut ef_buf = Vec::new();
        elias_fano.serialize(&mut ef_buf).unwrap();

        let partitioned_ef = PartitionedEFPostingList::from_postings(&doc_ids, &term_freqs);
        let mut pef_buf = Vec::new();
        partitioned_ef.serialize(&mut pef_buf).unwrap();

        let roaring = RoaringPostingList::from_postings(&doc_ids, &term_freqs);
        let mut roar_buf = Vec::new();
        roaring.serialize(&mut roar_buf).unwrap();

        let opt_p4d = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut opt_p4d_buf = Vec::new();
        opt_p4d.serialize(&mut opt_p4d_buf).unwrap();

        // Use CompressedPostingList for unified deserialization
        let unified = CompressedPostingList::from_postings(&doc_ids, &term_freqs, 1_000_000, 1.0);
        let mut unified_buf = Vec::new();
        unified.serialize(&mut unified_buf).unwrap();

        group.bench_with_input(BenchmarkId::new("horiz_bp128", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&bp_buf);
                black_box(HorizontalBP128PostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("vert_bp128", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&simd_buf);
                black_box(VerticalBP128PostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("elias_fano", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&ef_buf);
                black_box(EliasFanoPostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("partitioned_ef", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&pef_buf);
                black_box(PartitionedEFPostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("roaring", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&roar_buf);
                black_box(RoaringPostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("opt_p4d", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&opt_p4d_buf);
                black_box(OptP4DPostingList::deserialize(&mut cursor).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("unified", size), &size, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&unified_buf);
                black_box(CompressedPostingList::deserialize(&mut cursor).unwrap())
            })
        });
    }

    group.finish();
}
