//! Focused benchmarks for allocation-sensitive core structures.

use std::hint::black_box;
use std::path::Path;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use summa_core::directories::{
    Directory, DirectoryWriter, FsDirectory, RamDirectory, SliceCachingDirectory,
};
use summa_core::query::{
    Collector, MultiValueCombiner, ScoredPosition, SearchResult, TopKCollector,
};
use summa_core::structures::fast_field::codec::{auto_read_batch, serialize_auto};
use summa_core::structures::{BlockSparsePostingList, SparseBlock, WeightQuantization};

fn sparse_postings(count: usize) -> Vec<(u32, u16, f32)> {
    (0..count)
        .map(|index| {
            (
                index as u32 * 3,
                (index % 7) as u16,
                ((index * 17 % 101) as f32 + 1.0) / 101.0,
            )
        })
        .collect()
}

fn bench_top_k(c: &mut Criterion) {
    const CANDIDATES: usize = 100_000;
    const K: usize = 1_000;

    let mut group = c.benchmark_group("core_structures/top_k");
    group.throughput(Throughput::Elements(CANDIDATES as u64));
    group.bench_function("scores_only", |bencher| {
        bencher.iter(|| {
            let mut collector = TopKCollector::new(K);
            for index in 0..CANDIDATES {
                let score = ((index * 2_654_435_761usize) as u32) as f32;
                collector.collect(index as u32, black_box(score), &[]);
            }
            black_box(collector.into_results_with_count())
        });
    });
    group.finish();
}

fn bench_extract_ordinals(c: &mut Criterion) {
    let positions = (0..8)
        .map(|field| {
            let values = (0..128)
                .map(|index| {
                    let ordinal = (index % 16) as u32;
                    ScoredPosition::new((ordinal << 20) | index as u32, index as f32)
                })
                .collect();
            (field, values)
        })
        .collect();
    let result = SearchResult {
        doc_id: 7,
        score: 1.0,
        segment_id: 11,
        positions,
    };

    c.bench_function("core_structures/extract_ordinals", |bencher| {
        bencher.iter(|| black_box(black_box(&result).extract_ordinals()));
    });
}

fn bench_sparse_blocks(c: &mut Criterion) {
    let one_block = sparse_postings(128);
    let many_blocks = sparse_postings(128 * 64);
    let list =
        BlockSparsePostingList::from_postings(&many_blocks, WeightQuantization::Float16).unwrap();
    let decode_block = SparseBlock::from_postings(&one_block, WeightQuantization::Float16).unwrap();

    let mut group = c.benchmark_group("core_structures/sparse_block");
    for quantization in [
        WeightQuantization::Float32,
        WeightQuantization::Float16,
        WeightQuantization::UInt8,
        WeightQuantization::UInt4,
    ] {
        group.bench_with_input(
            BenchmarkId::new("build_128", format!("{quantization:?}")),
            &quantization,
            |bencher, &quantization| {
                bencher.iter(|| {
                    black_box(
                        SparseBlock::from_postings(black_box(&one_block), quantization).unwrap(),
                    )
                });
            },
        );
    }
    group.throughput(Throughput::Elements(many_blocks.len() as u64));
    group.bench_function("serialize_64_blocks", |bencher| {
        bencher.iter(|| black_box(black_box(&list).serialize().unwrap()));
    });
    group.bench_function("decode_f16_128", |bencher| {
        let mut output = Vec::with_capacity(128);
        bencher.iter(|| {
            decode_block.decode_weights_into(&mut output);
            black_box(&output);
        });
    });
    group.finish();
}

fn bench_slice_cache_hit(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let ram = RamDirectory::new();
    let path = Path::new("cached.bin");
    let bytes = vec![0x5au8; 1024 * 1024];
    runtime.block_on(ram.write(path, &bytes)).unwrap();
    let cached = SliceCachingDirectory::new(ram, bytes.len());
    runtime
        .block_on(cached.read_range(path, 0..bytes.len() as u64))
        .unwrap();

    let mut group = c.benchmark_group("core_structures/slice_cache");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("one_mib_hit", |bencher| {
        bencher.iter(|| {
            black_box(
                runtime
                    .block_on(cached.read_range(path, 0..bytes.len() as u64))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

/// Eight threads hammering cache hits on one `SliceCachingDirectory`. The
/// initial reads come from the filesystem so every cached range has its own
/// backing allocation, matching HTTP range responses rather than
/// `RamDirectory`'s slices of one shared `Arc`. Both the `read_range` entry
/// point and the `open_lazy` handle path are exercised.
fn bench_slice_cache_concurrent_hits(c: &mut Criterion) {
    const THREADS: usize = 8;
    const HITS_PER_THREAD: usize = 2_000;
    const FILE_BYTES: usize = 4 * 1024 * 1024;
    const READ_BYTES: u64 = 4096;
    const SLICES: u64 = FILE_BYTES as u64 / READ_BYTES;

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let path = Path::new("hot.bin");
    let bytes: Vec<u8> = (0..FILE_BYTES).map(|index| index as u8).collect();
    std::fs::write(temp_dir.path().join(path), &bytes).unwrap();
    let cached = SliceCachingDirectory::new(FsDirectory::new(temp_dir.path()), FILE_BYTES);
    // Populate the cache with 1024 independently allocated 4 KiB slices.
    for slice in 0..SLICES {
        let start = slice * READ_BYTES;
        runtime
            .block_on(cached.read_range(path, start..start + READ_BYTES))
            .unwrap();
    }
    let handle = runtime.block_on(cached.open_lazy(path)).unwrap();

    let mut group = c.benchmark_group("core_structures/slice_cache_concurrent");
    group.throughput(Throughput::Elements((THREADS * HITS_PER_THREAD) as u64));
    group.bench_function("read_range_hits_8_threads", |bencher| {
        bencher.iter(|| {
            std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let cached = &cached;
                    scope.spawn(move || {
                        for hit in 0..HITS_PER_THREAD {
                            let slice = ((hit * 7 + thread * 131) as u64) % SLICES;
                            let start = slice * READ_BYTES;
                            let data = futures::executor::block_on(
                                cached.read_range(path, start + 8..start + READ_BYTES - 8),
                            )
                            .unwrap();
                            black_box(data);
                        }
                    });
                }
            });
        });
    });
    group.bench_function("lazy_handle_hits_8_threads", |bencher| {
        bencher.iter(|| {
            std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let handle = &handle;
                    scope.spawn(move || {
                        for hit in 0..HITS_PER_THREAD {
                            let slice = ((hit * 7 + thread * 131) as u64) % SLICES;
                            let start = slice * READ_BYTES;
                            let data = futures::executor::block_on(
                                handle.read_bytes_range(start + 8..start + READ_BYTES - 8),
                            )
                            .unwrap();
                            black_box(data);
                        }
                    });
                }
            });
        });
    });
    group.finish();
}

/// Insert/evict churn: a cache far smaller than the file, so every read is a
/// miss that inserts one slice and evicts the least-recently-used one. The
/// overlapping variant additionally merges each new slice into its
/// predecessor.
fn bench_slice_cache_churn(c: &mut Criterion) {
    const FILE_BYTES: usize = 8 * 1024 * 1024;
    const CACHE_BYTES: usize = 512 * 1024;
    const READ_BYTES: u64 = 4096;
    const READS: usize = 2_048;

    let ram = RamDirectory::new();
    let path = Path::new("churn.bin");
    let bytes = vec![0x5au8; FILE_BYTES];
    futures::executor::block_on(ram.write(path, &bytes)).unwrap();

    let mut group = c.benchmark_group("core_structures/slice_cache_churn");
    group.throughput(Throughput::Elements(READS as u64));
    group.bench_function("insert_evict_disjoint", |bencher| {
        let cached = SliceCachingDirectory::new(ram.clone(), CACHE_BYTES);
        let mut next = 0u64;
        bencher.iter(|| {
            for _ in 0..READS {
                let start = (next * READ_BYTES) % FILE_BYTES as u64;
                next += 1;
                black_box(
                    futures::executor::block_on(cached.read_range(path, start..start + READ_BYTES))
                        .unwrap(),
                );
            }
        });
    });
    group.bench_function("insert_evict_overlapping", |bencher| {
        let cached = SliceCachingDirectory::new(ram.clone(), CACHE_BYTES);
        let mut next = 0u64;
        bencher.iter(|| {
            for _ in 0..READS {
                // Half-stride windows: every miss overlaps the previous slice.
                let start = (next * READ_BYTES / 2) % (FILE_BYTES as u64 - READ_BYTES);
                next += 1;
                black_box(
                    futures::executor::block_on(cached.read_range(path, start..start + READ_BYTES))
                        .unwrap(),
                );
            }
        });
    });
    group.finish();
}

fn bench_combiner(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_structures/combiner");
    for &count in &[5usize, 50] {
        let scores: Vec<(u32, f32)> = (0..count)
            .map(|index| (index as u32, ((index * 37 % 101) as f32) / 101.0))
            .collect();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("log_sum_exp", count),
            &scores,
            |bencher, scores| {
                let combiner = MultiValueCombiner::log_sum_exp();
                bencher.iter(|| black_box(combiner.combine(black_box(scores))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("weighted_top_k", count),
            &scores,
            |bencher, scores| {
                let combiner = MultiValueCombiner::weighted_top_k();
                bencher.iter(|| black_box(combiner.combine(black_box(scores))));
            },
        );
    }
    group.finish();
}

fn bench_fast_field_blockwise(c: &mut Criterion) {
    const VALUES: usize = 64 * 1024;
    const BLOCK: usize = 512;

    let values: Vec<u64> = (0..VALUES)
        .map(|index| {
            let block = index / BLOCK;
            let offset = index % BLOCK;
            block as u64 * 1_000_000 + offset as u64 * (block as u64 % 7 + 1)
        })
        .collect();
    let mut encoded = Vec::new();
    serialize_auto(&values, &mut encoded).unwrap();
    assert_eq!(encoded[0], 3, "benchmark data must select blockwise codec");

    let mut group = c.benchmark_group("core_structures/fast_field_blockwise");
    group.throughput(Throughput::Elements(VALUES as u64));
    group.bench_function("serialize", |bencher| {
        let mut output = Vec::new();
        bencher.iter(|| {
            output.clear();
            serialize_auto(black_box(&values), &mut output).unwrap();
            black_box(&output);
        });
    });
    group.bench_function("read_batch", |bencher| {
        let mut output = vec![0u64; VALUES];
        bencher.iter(|| {
            auto_read_batch(black_box(&encoded), 0, &mut output);
            black_box(&output);
        });
    });
    group.finish();
}

/// Identical encoded values: old header walks versus the reader's bounded directory.
fn bench_fast_field_sparse_lookup(c: &mut Criterion) {
    use summa_core::directories::OwnedBytes;
    use summa_core::structures::fast_field::codec::auto_read;
    use summa_core::structures::fast_field::{
        FastFieldColumnType, FastFieldReader, FastFieldWriter,
    };
    let mut group = c.benchmark_group("core_structures/fast_field_sparse_lookup");
    group.throughput(Throughput::Elements(100));
    for count in [512u32, 65_536, 1_048_576] {
        let mut writer = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        for index in 0..count {
            writer.add_u64(
                index,
                (index / 512) as u64 * 1_000_000
                    + (index % 512) as u64 * ((index / 512) as u64 % 7 + 1)
                    + (index % 11) as u64,
            );
        }
        let mut bytes = Vec::new();
        let (toc, _) = writer.serialize(&mut bytes, 0).unwrap();
        let reader = FastFieldReader::open(&OwnedBytes::new(bytes), &toc).unwrap();
        let encoded = reader.blocks()[0].data.as_slice();
        if count > 512 {
            assert_eq!(encoded[0], 3);
        }
        let probes: Vec<_> = (0..100).map(|i| (i * 7919 + 37) % count).collect();
        for &doc in &probes {
            assert_eq!(reader.get_u64(doc), auto_read(encoded, doc as usize));
        }
        group.bench_with_input(BenchmarkId::new("header_walk", count), &count, |b, _| {
            b.iter(|| {
                for &doc in &probes {
                    black_box(auto_read(black_box(encoded), black_box(doc) as usize));
                }
            });
        });
        group.bench_with_input(
            BenchmarkId::new("reader_checkpoints", count),
            &count,
            |b, _| {
                b.iter(|| {
                    for &doc in &probes {
                        black_box(black_box(&reader).get_u64(black_box(doc)));
                    }
                });
            },
        );
    }
    group.finish();
}

/// Term-dictionary point lookups with restart points every 16 entries.
#[cfg(feature = "sync")]
fn bench_term_dict_lookup(c: &mut Criterion) {
    use summa_core::directories::{FileHandle, OwnedBytes};
    use summa_core::structures::{AsyncSSTableReader, SSTableWriter, TermInfo};

    let mut keys: Vec<Vec<u8>> = (0..200_000u64)
        .map(|i| {
            let mut key = (i % 5).to_le_bytes()[..4].to_vec();
            key.extend_from_slice(
                format!("term{:08x}", i.wrapping_mul(0x9E37_79B9) % 50_000_000).as_bytes(),
            );
            key
        })
        .collect();
    keys.sort();
    keys.dedup();
    let build = || -> AsyncSSTableReader<TermInfo> {
        let mut writer = SSTableWriter::<_, TermInfo>::new(Vec::new());
        for (i, key) in keys.iter().enumerate() {
            writer
                .insert(key, &TermInfo::external(i as u64 * 4096, 4096, 17))
                .unwrap();
        }
        let bytes = writer.finish().unwrap();
        let handle = FileHandle::from_bytes(OwnedBytes::new(bytes));
        let rt = tokio::runtime::Runtime::new().unwrap();
        let reader = rt
            .block_on(AsyncSSTableReader::<TermInfo>::open(handle, 4096))
            .unwrap();
        rt.block_on(reader.preload_all_blocks()).unwrap();
        reader
    };
    let reader = build();
    // Random existing keys.
    let probes: Vec<&[u8]> = (0..4096usize)
        .map(|i| keys[(i * 7919) % keys.len()].as_slice())
        .collect();

    let mut group = c.benchmark_group("core_structures/term_dict_lookup");
    group.throughput(Throughput::Elements(probes.len() as u64));
    group.bench_function("restart_points", |bencher| {
        bencher.iter(|| {
            let mut found = 0usize;
            for key in &probes {
                found += usize::from(black_box(reader.get_sync(key).unwrap()).is_some());
            }
            black_box(found)
        })
    });
    group.finish();
}

#[cfg(not(feature = "sync"))]
fn bench_term_dict_lookup(_: &mut Criterion) {}

criterion_group!(
    benches,
    bench_top_k,
    bench_extract_ordinals,
    bench_sparse_blocks,
    bench_slice_cache_hit,
    bench_slice_cache_concurrent_hits,
    bench_slice_cache_churn,
    bench_combiner,
    bench_fast_field_blockwise,
    bench_fast_field_sparse_lookup,
    bench_term_dict_lookup,
);
criterion_main!(benches);
