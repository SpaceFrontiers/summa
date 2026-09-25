//! Real range materialization plus isolated closure/code-generation probes.
//! Build/open/merge and membership assertions are outside timed loops.

use std::cmp::Ordering;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use summa_core::query::{DocBitset, HeapEntry, Query, RangeQuery};
use summa_core::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use summa_core::structures::fast_field::codec::bitpacked_read_batch;
use summa_core::{Document, Field, RamDirectory, SchemaBuilder};

const DOCUMENTS: u32 = 65_536;

fn value(doc: u32) -> u64 {
    // Permutation of [0, 65536); shuffled selectivity within each block.
    u64::from(doc.wrapping_mul(40_503) & 65_535)
}

async fn range_fixture(blocks: u32) -> (SegmentReader, Field) {
    range_pattern_fixture(DOCUMENTS, blocks, |doc| Some(value(doc))).await
}

async fn range_pattern_fixture(
    documents: u32,
    blocks: u32,
    value: impl Fn(u32) -> Option<u64>,
) -> (SegmentReader, Field) {
    range_pattern_fixture_multi(documents, blocks, value, false).await
}

async fn range_pattern_fixture_multi(
    documents: u32,
    blocks: u32,
    value: impl Fn(u32) -> Option<u64>,
    multi: bool,
) -> (SegmentReader, Field) {
    let dir = RamDirectory::new();
    let mut sb = SchemaBuilder::default();
    let field = sb.add_u64_field("value", false, false);
    sb.set_fast(field, true);
    sb.set_multi(field, multi);
    let schema = Arc::new(sb.build());
    let mut sources = Vec::new();
    for block in 0..blocks {
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();
        for local in 0..documents / blocks {
            let mut doc = Document::new();
            if let Some(value) = value(block * (documents / blocks) + local) {
                doc.add_u64(field, value);
                if multi {
                    doc.add_u64(field, 42);
                }
            }
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        builder.build(&dir, id, None).await.unwrap();
        sources.push(
            SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                .await
                .unwrap(),
        );
    }
    let reader = if blocks == 1 {
        sources.pop().unwrap()
    } else {
        let id = SegmentId::new();
        SegmentMerger::new(Arc::clone(&schema))
            .merge(&dir, &sources, id, None)
            .await
            .unwrap();
        SegmentReader::open(&dir, id, schema, 0).await.unwrap()
    };
    assert_eq!(
        reader.fast_field(field.0).unwrap().num_blocks(),
        blocks as usize
    );
    (reader, field)
}

fn bench_range(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("rust_hot_paths/range_bitset");
    group.sample_size(20);
    group.throughput(Throughput::Elements(u64::from(DOCUMENTS)));
    for blocks in [1, 16] {
        let (reader, field) = runtime.block_on(range_fixture(blocks));
        for upper in [655u64, 32_767] {
            let query = RangeQuery::u64(field, Some(0), Some(upper));
            let actual = query.as_doc_bitset(&reader).unwrap();
            let predicate = query.as_doc_predicate(&reader).unwrap();
            let original = DocBitset::from_predicate(DOCUMENTS, &*predicate);
            for doc in 0..DOCUMENTS {
                let expected = value(doc) <= upper;
                assert_eq!(actual.contains(doc), expected);
                assert_eq!(original.contains(doc), expected);
            }
            let fixture = format!("{blocks}_blocks_max_{upper}");
            group.bench_function(BenchmarkId::new("production", &fixture), |b| {
                b.iter(|| black_box(black_box(&query).as_doc_bitset(black_box(&reader)).unwrap()));
            });
            group.bench_function(BenchmarkId::new("predicate_control", &fixture), |b| {
                b.iter(|| {
                    let predicate = black_box(&query)
                        .as_doc_predicate(black_box(&reader))
                        .unwrap();
                    black_box(DocBitset::from_predicate(DOCUMENTS, &*predicate))
                });
            });
        }
    }
    group.finish();
}

// End-to-end materialization controls for pruning and sequential decoding.
fn bench_range_scan_layouts(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("rust_hot_paths/range_scan_layouts");
    group.sample_size(20);
    for (layout, documents, blocks, upper) in [
        ("bucketed", 65_536, 1, 655u64),
        ("bucketed", 65_536, 16, 655),
        ("ordered", 65_536, 1, 6_550),
        ("ordered", 65_536, 16, 6_550),
        ("missing", 65_536, 16, 327_670),
        ("all_match", 65_536, 16, u64::MAX),
        ("piecewise", 1_024, 1, 1 << 31),
        ("piecewise", 65_536, 1, 1 << 31),
        ("piecewise", 1_048_576, 1, 1 << 31),
    ] {
        let value = |doc: u32| {
            if layout == "missing" && (doc / (documents / blocks)).is_multiple_of(2) {
                None
            } else if layout == "bucketed" {
                Some(u64::from(doc / 4096) * (1 << 20) + u64::from(doc.wrapping_mul(40_503) & 4095))
            } else if layout == "piecewise" {
                let block = u64::from(doc / 512);
                let local = u64::from(doc % 512);
                Some(((block * 40_503) & 65_535) * 65_536 + local * (3 + block % 3) + local % 7)
            } else {
                Some(u64::from(doc) * 10 + u64::from(doc % 7))
            }
        };
        let (reader, field) = runtime.block_on(range_pattern_fixture(documents, blocks, value));
        let column = reader.fast_field(field.0).unwrap();
        if layout == "piecewise" {
            assert!(column.blocks().iter().all(|b| b.data.as_slice()[0] == 3));
        }
        if layout == "bucketed" && blocks == 16 {
            assert!(column.blocks().iter().all(|b| b.data.as_slice()[0] == 1));
        }
        let query = RangeQuery::u64(field, Some(0), Some(upper));
        let result = query.as_doc_bitset(&reader).unwrap();
        let mut matches = 0;
        for doc in 0..documents {
            let expected = value(doc).is_some_and(|v| v <= upper);
            assert_eq!(result.contains(doc), expected, "{layout}, doc {doc}");
            matches += u32::from(expected);
        }
        assert_eq!(result.count(), matches);
        eprintln!(
            "{layout}/{documents}/{blocks}: {} encoded column bytes; {} output bytes",
            column.disk_bytes(),
            u64::from(documents).div_ceil(64) * 8
        );
        group.throughput(Throughput::Elements(u64::from(documents)));
        group.bench_function(format!("{layout}/{documents}/{blocks}_blocks"), |b| {
            b.iter(|| black_box(black_box(&query).as_doc_bitset(black_box(&reader)).unwrap()));
        });
    }
    group.finish();
}

// Lazy scorer controls include creation/destruction and selective seek workloads.
#[cfg(feature = "sync")]
fn bench_lazy_range(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("rust_hot_paths/lazy_range");
    group.sample_size(20);
    for layout in [
        "shuffled",
        "piecewise",
        "constant",
        "missing",
        "multi",
        "no_match",
    ] {
        let values = |doc: u32| {
            if (layout == "missing" || layout == "multi") && doc.is_multiple_of(7) {
                None
            } else if layout == "piecewise" {
                let block = u64::from(doc / 512);
                let local = u64::from(doc % 512);
                Some(((block * 40_503) & 65_535) * 65_536 + local * (3 + block % 3) + local % 7)
            } else if layout == "no_match" {
                Some(1 << 60)
            } else if layout == "constant" {
                Some(7)
            } else {
                Some(value(doc))
            }
        };
        let (reader, field) = runtime.block_on(range_pattern_fixture_multi(
            DOCUMENTS,
            1,
            values,
            layout == "multi",
        ));
        let upper = if layout == "piecewise" {
            1 << 31
        } else {
            32_767
        };
        let query = RangeQuery::u64(field, Some(0), Some(upper));
        eprintln!(
            "scorer/{layout}: {} bytes",
            std::mem::size_of_val(&*query.scorer_sync(&reader, 10).unwrap())
        );
        let expected: Vec<_> = (0..DOCUMENTS)
            .filter(|&d| values(d).is_some_and(|v| v <= upper))
            .collect();
        for mode in ["full", "first", "seek"] {
            let execute = || {
                let mut scorer = query.scorer_sync(&reader, 10).unwrap();
                let mut checksum = 0u64;
                if mode == "first" {
                    return u64::from(scorer.doc());
                }
                if mode == "seek" {
                    for target in (0..DOCUMENTS).step_by(1021) {
                        checksum = checksum.wrapping_add(u64::from(scorer.seek(target)));
                    }
                } else {
                    while scorer.doc() != summa_core::structures::TERMINATED {
                        checksum += u64::from(scorer.doc());
                        scorer.advance();
                    }
                }
                checksum
            };
            let oracle = match mode {
                "first" => u64::from(*expected.first().unwrap_or(&u32::MAX)),
                "seek" => (0..DOCUMENTS)
                    .step_by(1021)
                    .map(|target| {
                        u64::from(
                            *expected
                                .iter()
                                .find(|&&doc| doc >= target)
                                .unwrap_or(&u32::MAX),
                        )
                    })
                    .sum(),
                _ => expected.iter().map(|&d| u64::from(d)).sum(),
            };
            assert_eq!(execute(), oracle, "{layout}/{mode}");
            group.bench_function(format!("{layout}/{mode}"), |b| {
                b.iter(|| black_box(execute()))
            });
        }
    }
    group.finish();
}

#[cfg(not(feature = "sync"))]
fn bench_lazy_range(_: &mut Criterion) {
    eprintln!("Lazy scorer benchmarks require the sync feature.");
}

// These symbols deliberately remain identifiable in optimized assembly. Do not
// add inline(never) to production callbacks just to reproduce these experiments.
#[inline(never)]
fn count_static(values: &[u64], predicate: impl Fn(u64) -> bool) -> usize {
    values.iter().filter(|&&v| predicate(v)).count()
}

#[inline(never)]
fn count_dynamic(values: &[u64], predicate: &dyn Fn(u64) -> bool) -> usize {
    values.iter().filter(|&&v| predicate(v)).count()
}

#[inline(never)]
fn heap_order(left: &HeapEntry, right: &HeapEntry) -> Ordering {
    left.cmp(right)
}

fn bench_dispatch(c: &mut Criterion) {
    let values: Vec<_> = (0..DOCUMENTS).map(value).collect();
    let upper = black_box(32_767);
    let predicate = |v| v <= upper;
    assert_eq!(count_static(&values, predicate), 32_768);
    assert_eq!(count_dynamic(&values, &predicate), 32_768);
    let mut group = c.benchmark_group("rust_hot_paths/decoded_predicate");
    group.sample_size(20);
    group.throughput(Throughput::Elements(u64::from(DOCUMENTS)));
    group.bench_function("generic_closure", |b| {
        b.iter(|| black_box(count_static(black_box(&values), predicate)));
    });
    group.bench_function("opaque_dyn_fn", |b| {
        b.iter(|| black_box(count_dynamic(black_box(&values), black_box(&predicate))));
    });
    group.finish();

    let left = HeapEntry {
        doc_id: 7,
        score: 1.0,
        ordinal: 0,
    };
    let right = HeapEntry { doc_id: 8, ..left };
    assert_eq!(
        heap_order(black_box(&left), black_box(&right)),
        Ordering::Less
    );
    eprintln!(
        "layout bytes: HeapEntry={}, DocBitset={}, SharedThreshold={}, ColumnBlock={}, FastFieldReader={}",
        size_of::<HeapEntry>(),
        size_of::<DocBitset>(),
        size_of::<summa_core::query::SharedThreshold>(),
        size_of::<summa_core::structures::fast_field::ColumnBlock>(),
        size_of::<summa_core::structures::fast_field::FastFieldReader>()
    );
}

fn bench_byte_aligned_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_hot_paths/bitpacked_batch");
    group.sample_size(20);
    group.throughput(Throughput::Elements(256));
    for bits in [8u8, 16, 32, 64] {
        // Valid bitpacked payload, excluding the auto-codec tag: min + bpv + data.
        let mut encoded = 17u64.to_le_bytes().to_vec();
        encoded.push(bits);
        let mut expected = Vec::new();
        let mask = u64::MAX >> (64 - bits);
        for i in 0..256u64 {
            let raw = i.wrapping_mul(0x9e37_79b9_7f4a_7c15) & mask;
            encoded.extend_from_slice(&raw.to_le_bytes()[..usize::from(bits / 8)]);
            expected.push(raw.wrapping_add(17));
        }
        let mut output = [0u64; 256];
        bitpacked_read_batch(&encoded, 0, &mut output);
        assert_eq!(output.as_slice(), expected);
        group.bench_function(BenchmarkId::from_parameter(bits), |b| {
            b.iter(|| {
                bitpacked_read_batch(black_box(&encoded), black_box(0), &mut output);
                black_box(&output);
            });
        });
    }
    group.finish();
}

/// Exercise sparse ID windows separately from dense posting traversal. Setup
/// and the independent exact-score oracle stay outside the timed loop.
fn bench_windowed_text_unions(c: &mut Criterion) {
    bench_windowed_text(c, "windowed_text_unions", 4);
    bench_windowed_text(c, "windowed_text_terms", 1);
}

fn bench_windowed_text(c: &mut Criterion, name: &str, term_count: u32) {
    use summa_core::directories::OwnedBytes;
    use summa_core::query::{Bm25Params, MaxScoreExecutor};
    use summa_core::segment::chunk_map::{DocLengthsColumn, read_chunk_maps, write_chunk_maps};
    use summa_core::structures::{BlockPostingList, PostingList};
    const DOCS: u32 = 262_144;
    let norms = vec![100u16; DOCS as usize];
    let mut encoded_norms = Vec::new();
    write_chunk_maps(
        &mut encoded_norms,
        &[],
        &[DocLengthsColumn {
            field_id: 0,
            lengths: &norms,
            total_tokens: u64::from(DOCS) * 100,
        }],
    )
    .unwrap();
    let lengths = read_chunk_maps(OwnedBytes::new(encoded_norms))
        .unwrap()
        .doc_lengths
        .remove(&0)
        .unwrap();
    let params = Bm25Params::default();
    let mut group = c.benchmark_group(name);
    for (name, stride) in [("dense", 4u32), ("sparse", 1024)] {
        let mut lists = Vec::new();
        let mut expected = Vec::new();
        let mut postings_count = 0u64;
        for term in 0..term_count {
            let mut postings = PostingList::new();
            for doc in 0..DOCS {
                // Rotate terms over repeated window slots to catch stale scores.
                if doc % stride == (doc / 4096 + term) % 4 {
                    let tf = doc % 11 + 1;
                    postings.push(doc, tf);
                    expected.push((doc, params.score(tf as f32, 9.3, 100.0, 100.0)));
                    postings_count += 1;
                }
            }
            let list = BlockPostingList::from_posting_list_with(
                &postings,
                false,
                Some(&|doc| lengths.length(doc)),
            )
            .unwrap();
            lists.push((list, 9.3));
        }
        expected.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        expected.truncate(10);
        let execute = || {
            MaxScoreExecutor::text(lists.clone(), 100.0, 10, Some(&lengths), params, 1.0)
                .execute_sync()
                .unwrap()
        };
        let actual: Vec<_> = execute()
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect();
        assert_eq!(actual, expected, "{name} ranking oracle");
        group.throughput(Throughput::Elements(postings_count));
        group.bench_function(name, |b| b.iter(|| black_box(execute())));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_range,
    bench_range_scan_layouts,
    bench_lazy_range,
    bench_dispatch,
    bench_byte_aligned_decode,
    bench_windowed_text_unions
);
criterion_main!(benches);
