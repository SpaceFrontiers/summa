//! Compression benchmarks
//!
//! Run with: cargo bench -p summa-core --bench compression

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use summa_core::compression::{CompressionLevel, compress, decompress};

fn generate_text_data(size: usize) -> Vec<u8> {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "search", "index",
        "document", "query", "term", "field", "score",
    ];
    let mut data = Vec::with_capacity(size);
    let mut i = 0;
    while data.len() < size {
        if !data.is_empty() {
            data.push(b' ');
        }
        data.extend_from_slice(words[i % words.len()].as_bytes());
        i += 1;
    }
    data.truncate(size);
    data
}

fn generate_numeric_data(count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(count * 4);
    for i in 0..count {
        data.extend_from_slice(&(i as u32).to_le_bytes());
    }
    data
}

fn bench_compression(c: &mut Criterion) {
    let sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024];

    let mut group = c.benchmark_group("compression");
    for size in sizes {
        let data = generate_text_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("compress", size), &data, |b, data| {
            b.iter(|| compress(black_box(data), CompressionLevel::default()))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("decompression");
    for size in sizes {
        let data = generate_text_data(size);
        let compressed = compress(&data, CompressionLevel::default()).unwrap();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, compressed| b.iter(|| decompress(black_box(compressed))),
        );
    }
    group.finish();
}

fn bench_compression_levels(c: &mut Criterion) {
    let data = generate_text_data(100 * 1024);
    let levels = [1, 3, 5, 9, 15, 19];

    let mut group = c.benchmark_group("compression_levels");
    group.throughput(Throughput::Bytes(data.len() as u64));
    for level in levels {
        group.bench_with_input(BenchmarkId::new("level", level), &level, |b, &level| {
            b.iter(|| compress(black_box(&data), CompressionLevel(level)))
        });
    }
    group.finish();
}

fn bench_data_types(c: &mut Criterion) {
    let size = 100 * 1024;
    let text_data = generate_text_data(size);
    let numeric_data = generate_numeric_data(size / 4);

    let mut group = c.benchmark_group("data_types");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("text", |b| {
        b.iter(|| compress(black_box(&text_data), CompressionLevel::default()))
    });

    group.bench_function("numeric", |b| {
        b.iter(|| compress(black_box(&numeric_data), CompressionLevel::default()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compression,
    bench_compression_levels,
    bench_data_types
);
criterion_main!(benches);
