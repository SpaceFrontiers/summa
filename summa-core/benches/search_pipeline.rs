//! End-to-end search plumbing benchmark: a small multi-segment RAM index with
//! a text field, a multi-valued flat dense field and a sparse field, queried
//! through the same `Searcher` entry points the server uses.
//!
//! This keeps `combine_ordinal_results`, `VectorResultScorer`,
//! `TopKCollector`, `matched_positions` and chunk-level fusion hot, so
//! per-hit allocation and per-segment sorting changes in the shared plumbing
//! are measured where they land rather than in isolation.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use summa_core::directories::RamDirectory;
use summa_core::dsl::{DenseVectorConfig, Document, SchemaBuilder, VectorIndexType};
use summa_core::query::{
    DenseVectorQuery, FusionMethod, MultiValueCombiner, Query, SparseVectorQuery, TermQuery,
};
use summa_core::{Field, Index, IndexConfig, IndexWriter, Searcher};

const DIM: usize = 32;
const SEGMENTS: usize = 4;
const DOCS_PER_SEGMENT: usize = 800;
const SPARSE_VOCAB: u32 = 2_000;

struct Fixture {
    searcher: Arc<Searcher<RamDirectory>>,
    title: Field,
    embedding: Field,
    sparse: Field,
    // Keep the index alive for as long as the searcher is used.
    _index: Index<RamDirectory>,
}

fn unit_vector(rng: &mut StdRng) -> Vec<f32> {
    let mut vector: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect();
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|v| *v /= norm);
    vector
}

fn sparse_vector(rng: &mut StdRng, dims: usize) -> Vec<(u32, f32)> {
    let mut entries: Vec<(u32, f32)> = (0..dims)
        .map(|_| {
            (
                rng.random_range(0..SPARSE_VOCAB),
                rng.random_range(0.1f32..2.0),
            )
        })
        .collect();
    entries.sort_unstable_by_key(|&(dim, _)| dim);
    entries.dedup_by_key(|entry| entry.0);
    entries
}

async fn build_fixture() -> Fixture {
    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, false);
    let embedding = sb.add_dense_vector_field_with_config(
        "embedding",
        true,
        false,
        DenseVectorConfig {
            dim: DIM,
            index_type: VectorIndexType::Flat,
            quantization: summa_core::dsl::DenseVectorQuantization::F32,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: summa_core::dsl::IvfRoutingMode::Auto,
            nprobe: 0,
            unit_norm: true,
            soar: None,
        },
    );
    let sparse = sb.add_sparse_vector_field("sparse", true, false);
    let schema = sb.build();

    let dir = RamDirectory::new();
    let config = IndexConfig {
        merge_policy: Box::new(summa_core::NoMergePolicy),
        ..IndexConfig::default()
    };
    let mut writer = IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let words = [
        "quantum", "zebra", "habitat", "lattice", "harbor", "signal", "meadow", "orbit", "cipher",
        "glacier", "tundra", "voltage", "pollen", "basalt", "ember", "ledger",
    ];
    for segment in 0..SEGMENTS {
        for doc in 0..DOCS_PER_SEGMENT {
            let mut document = Document::new();
            let mut text = String::new();
            for _ in 0..6 {
                text.push_str(words[rng.random_range(0..words.len())]);
                text.push(' ');
            }
            text.push_str(&format!("segment{segment} doc{doc}"));
            document.add_text(title, text);
            // Every third document carries three chunks so the multi-value
            // combiner and per-ordinal positions are exercised.
            let chunks = if doc % 3 == 0 { 3 } else { 1 };
            for _ in 0..chunks {
                document.add_dense_vector(embedding, unit_vector(&mut rng));
                document.add_sparse_vector(sparse, sparse_vector(&mut rng, 24));
            }
            writer.add_document(document).unwrap();
        }
        writer.commit().await.unwrap();
    }

    let index = Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    assert_eq!(
        searcher.segment_readers().len(),
        SEGMENTS,
        "fixture must span several segments"
    );
    Fixture {
        searcher,
        title,
        embedding,
        sparse,
        _index: index,
    }
}

fn bench_search_pipeline(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let fixture = runtime.block_on(build_fixture());
    let searcher = &fixture.searcher;

    let mut rng = StdRng::seed_from_u64(7);
    let dense_query = DenseVectorQuery::new(fixture.embedding, unit_vector(&mut rng))
        .with_combiner(MultiValueCombiner::log_sum_exp());
    let sparse_query = SparseVectorQuery::new(fixture.sparse, sparse_vector(&mut rng, 24))
        .with_combiner(MultiValueCombiner::Max);
    let text_query = TermQuery::text(fixture.title, "quantum");
    let hybrid: Vec<(&dyn Query, f32)> = vec![
        (&text_query, 1.0),
        (&dense_query, 1.0),
        (&sparse_query, 1.0),
    ];

    let mut group = c.benchmark_group("search_pipeline");
    group.throughput(Throughput::Elements(1));
    for &limit in &[10usize, 200] {
        group.bench_with_input(
            BenchmarkId::new("fused_hybrid", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(searcher.search_fused_with_count(
                                black_box(&hybrid),
                                limit,
                                limit,
                                FusionMethod::default(),
                                MultiValueCombiner::Max,
                            ))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_top_k", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(searcher.search_with_count(black_box(&dense_query), limit))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_top_k_positions", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(
                                searcher.search_with_positions(black_box(&dense_query), limit),
                            )
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("sparse_top_k", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        runtime
                            .block_on(searcher.search_with_count(black_box(&sparse_query), limit))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();

    // The async/current-thread path (HTTP/WASM shape): no rayon, scorers are
    // built through `Query::scorer` and results flow through the bounded
    // async segment stream.
    let current_thread = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("search_pipeline_current_thread");
    group.throughput(Throughput::Elements(1));
    for &limit in &[10usize, 200] {
        group.bench_with_input(
            BenchmarkId::new("fused_hybrid", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        current_thread
                            .block_on(searcher.search_fused_with_count(
                                black_box(&hybrid),
                                limit,
                                limit,
                                FusionMethod::default(),
                                MultiValueCombiner::Max,
                            ))
                            .unwrap(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("dense_top_k", limit),
            &limit,
            |bencher, &limit| {
                bencher.iter(|| {
                    black_box(
                        current_thread
                            .block_on(searcher.search_with_count(black_box(&dense_query), limit))
                            .unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

// Ranked plain text must be measured through the planner: a cursor-only
// benchmark cannot expose a TermQuery that never admits block-max execution.
#[cfg(feature = "sync")]
fn bench_plain_ranked_text(c: &mut Criterion) {
    use summa_core::query::{BooleanQuery, TopKCollector, collect_segment};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("plain_ranked_text");
    group.sample_size(30);
    for layout in ["early", "late", "mixed"] {
        let mut schema = SchemaBuilder::default();
        let body = schema.add_text_field_with_tokenizer("body", true, false, "simple");
        let dir = RamDirectory::new();
        let config = IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            posting_ratio_bounds: false,
            posting_impact_bounds: false,
            ..Default::default()
        };
        let index = runtime.block_on(async {
            let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
                .await
                .unwrap();
            for row in 0..32768 {
                let winner = match layout {
                    "early" => row < 128,
                    "late" => row >= 32640,
                    _ => row % 257 == 0,
                };
                let mut document = Document::new();
                document.add_text(
                    body,
                    if winner {
                        "alpha beta ".repeat(16)
                    } else {
                        "alpha beta ".to_owned() + &"padding ".repeat(30)
                    },
                );
                for attempt in 0..=1000 {
                    match writer.add_document(document.clone()) {
                        Ok(()) => break,
                        Err(summa_core::Error::QueueFull) if attempt < 1000 => {
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        }
                        Err(error) => panic!("fixture ingestion failed: {error}"),
                    }
                }
            }
            writer.commit().await.unwrap();
            writer.shutdown().await.unwrap();
            Index::open(dir, config).await.unwrap()
        });
        let searcher =
            runtime.block_on(async { index.reader().await.unwrap().searcher().await.unwrap() });
        let queries: [(&str, Box<dyn Query>); 3] = [
            ("term", Box::new(TermQuery::text(body, "alpha"))),
            (
                "filter",
                Box::new(summa_core::query::PrefixQuery::text(body, "alp")),
            ),
            (
                "and",
                Box::new(
                    BooleanQuery::new()
                        .must(TermQuery::text(body, "alpha"))
                        .must(TermQuery::text(body, "beta")),
                ),
            ),
        ];
        for (kind, query) in queries {
            group.bench_function(format!("{layout}/{kind}/count"), |b| {
                b.iter(|| {
                    let mut count = summa_core::query::CountCollector::new();
                    runtime
                        .block_on(collect_segment(
                            &searcher.segment_readers()[0],
                            query.as_ref(),
                            &mut count,
                        ))
                        .unwrap();
                    assert_eq!(count.count(), 32768);
                    black_box(count.count())
                });
            });
            let mut oracle = TopKCollector::new(32768);
            runtime
                .block_on(collect_segment(
                    &searcher.segment_readers()[0],
                    query.as_ref(),
                    &mut oracle,
                ))
                .unwrap();
            let expected = oracle.into_sorted_results();
            for k in [10, 100] {
                let actual = searcher
                    .search_with_offset_and_count_sync(query.as_ref(), k, 0)
                    .unwrap()
                    .0;
                assert_eq!(
                    actual
                        .iter()
                        .map(|h| (h.doc_id, h.score.to_bits()))
                        .collect::<Vec<_>>(),
                    expected[..k]
                        .iter()
                        .map(|h| (h.doc_id, h.score.to_bits()))
                        .collect::<Vec<_>>()
                );
                group.bench_function(format!("{layout}/{kind}/{k}"), |b| {
                    b.iter(|| {
                        black_box(
                            searcher
                                .search_with_offset_and_count_sync(black_box(query.as_ref()), k, 0)
                                .unwrap(),
                        )
                    })
                });
            }
        }
    }
    group.finish();
}

#[cfg(feature = "sync")]
fn bench_pattern_filters(c: &mut Criterion) {
    use summa_core::query::{TopKCollector, collect_segment_with_limit_sync};
    use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let schema = Arc::new(schema.build());
    let dir = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for row in 0..50_000 {
        let mut document = Document::new();
        document.add_text(
            field,
            format!("g{row:05}{}", if row % 127 == 0 { "itar" } else { "other" }),
        );
        builder.add_document(document).unwrap();
    }
    let id = SegmentId::new();
    runtime.block_on(builder.build(&dir, id, None)).unwrap();
    let reader = runtime
        .block_on(SegmentReader::open(&dir, id, schema, 16))
        .unwrap();
    let queries: [(&str, Box<dyn Query>); 3] = [
        (
            "regex",
            Box::new(summa_core::RegexQuery::new(field, "g.*itar").unwrap()),
        ),
        (
            "single_star",
            Box::new(summa_core::WildcardQuery::new(field, "g*itar").unwrap()),
        ),
        (
            "prefix",
            Box::new(summa_core::PrefixQuery::text(field, "g000")),
        ),
    ];
    let mut group = c.benchmark_group("pattern_filters");
    group.sample_size(30);
    for (name, query) in queries {
        let mut oracle = TopKCollector::new(50_000);
        collect_segment_with_limit_sync(&reader, query.as_ref(), &mut oracle, 50_000).unwrap();
        let expected = oracle.into_sorted_results();
        assert_eq!(
            expected.len(),
            if name == "prefix" {
                100
            } else {
                (0..50_000).step_by(127).count()
            }
        );
        for k in [10, 100] {
            let mut top = TopKCollector::new(k);
            collect_segment_with_limit_sync(&reader, query.as_ref(), &mut top, k).unwrap();
            assert_eq!(top.into_sorted_results(), expected[..k]);
            group.bench_function(format!("{name}/{k}"), |b| {
                b.iter(|| {
                    let mut top = TopKCollector::new(k);
                    collect_segment_with_limit_sync(
                        &reader,
                        black_box(query.as_ref()),
                        &mut top,
                        k,
                    )
                    .unwrap();
                    black_box(top.into_sorted_results())
                })
            });
        }
    }
    group.finish();
}

#[cfg(feature = "sync")]
fn bench_union_membership(c: &mut Criterion) {
    use summa_core::query::{CountCollector, PrefixQuery, collect_segment};
    use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_text_field_with_tokenizer("tag", true, false, "raw");
    let schema = Arc::new(schema.build());
    let directory = RamDirectory::new();
    let mut builder = SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
    for row in 0..65_536 {
        let mut document = Document::new();
        document.add_text(field, format!("tag{:03}", row % 128));
        if row % 2 == 0 {
            document.add_text(field, "tagcommon");
        }
        builder.add_document(document).unwrap();
    }
    let id = SegmentId::new();
    runtime
        .block_on(builder.build(&directory, id, None))
        .unwrap();
    let reader = runtime
        .block_on(SegmentReader::open(&directory, id, schema, 16))
        .unwrap();
    let mut group = c.benchmark_group("union_membership");
    group.sample_size(30);
    for (name, prefix, expected) in [
        ("dense", "tag", 65_536),
        ("overlap", "tag0", 51_200),
        ("sparse", "tag000", 512),
    ] {
        let query = PrefixQuery::text(field, prefix);
        let mut oracle = CountCollector::new();
        runtime
            .block_on(collect_segment(&reader, &query, &mut oracle))
            .unwrap();
        assert_eq!(oracle.count(), expected);
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut count = CountCollector::new();
                runtime
                    .block_on(collect_segment(&reader, black_box(&query), &mut count))
                    .unwrap();
                black_box(count.count())
            })
        });
    }
    group.finish();
}

#[cfg(feature = "sync")]
fn bench_membership_density(c: &mut Criterion) {
    use summa_core::query::{BooleanQuery, CountCollector, collect_segment};
    use summa_core::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentReader};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("membership_density");
    group.sample_size(30);
    for stride in [1, 2, 5, 20, 128] {
        let mut schema = SchemaBuilder::default();
        let field = schema.add_text_field_with_tokenizer("body", true, false, "simple");
        let schema = Arc::new(schema.build());
        let dir = RamDirectory::new();
        let mut builder =
            SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
        for row in 0..32768 {
            let mut doc = Document::new();
            doc.add_text(
                field,
                if row % stride == 0 {
                    "alpha beta"
                } else {
                    "padding"
                },
            );
            builder.add_document(doc).unwrap();
        }
        let id = SegmentId::new();
        runtime.block_on(builder.build(&dir, id, None)).unwrap();
        let reader = runtime
            .block_on(SegmentReader::open(&dir, id, schema, 16))
            .unwrap();
        let query = BooleanQuery::new()
            .must(TermQuery::text(field, "alpha"))
            .must(TermQuery::text(field, "beta"));
        let mut oracle = CountCollector::new();
        runtime
            .block_on(collect_segment(&reader, &query, &mut oracle))
            .unwrap();
        assert_eq!(oracle.count(), (32768_u64).div_ceil(stride));
        group.bench_function(BenchmarkId::new("stride", stride), |b| {
            b.iter(|| {
                let mut collector = CountCollector::new();
                runtime
                    .block_on(collect_segment(&reader, &query, &mut collector))
                    .unwrap();
                black_box(collector.count())
            })
        });
    }
    group.finish();
}

#[cfg(feature = "sync")]
criterion_group!(
    benches,
    bench_search_pipeline,
    bench_plain_ranked_text,
    bench_pattern_filters,
    bench_union_membership,
    bench_membership_density
);
#[cfg(not(feature = "sync"))]
criterion_group!(benches, bench_search_pipeline);
criterion_main!(benches);
