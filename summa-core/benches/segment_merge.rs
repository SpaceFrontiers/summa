//! Deterministic two-source merge, with fixture construction outside timing.
//! RAM isolates merge CPU/allocation costs; this does not measure cold disk I/O.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use summa_core::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, SegmentMerger, SegmentReader,
};
use summa_core::{Document, RamDirectory, SchemaBuilder};

fn bench_segment_merge(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("segment_merge");
    group.sample_size(20);
    for count in [4_096u32, 65_536] {
        for missing in [false, true] {
            let dir = RamDirectory::new();
            let mut sb = SchemaBuilder::default();
            let field = sb.add_u64_field("sequence", false, false);
            sb.set_fast(field, true);
            sb.set_multi(field, true);
            let schema = Arc::new(sb.build());
            // Simulate an older source without this optional column. Keep
            // identical field IDs and types while disabling its fast storage.
            let mut old = SchemaBuilder::default();
            assert_eq!(old.add_u64_field("sequence", false, false), field);
            old.set_multi(field, true);
            let old_schema = Arc::new(old.build());
            let sources = runtime.block_on(async {
                let mut readers = Vec::new();
                for source in 0..2 {
                    let source_schema = if missing && source == 0 {
                        Arc::clone(&old_schema)
                    } else {
                        Arc::clone(&schema)
                    };
                    let mut builder =
                        SegmentBuilder::new(source_schema, SegmentBuilderConfig::default())
                            .unwrap();
                    for i in 0..count {
                        let mut doc = Document::new();
                        if !missing || source != 0 {
                            doc.add_u64(field, u64::from(i) * 17);
                            doc.add_u64(field, u64::from(i) * 31);
                        }
                        builder.add_document(doc).unwrap();
                    }
                    let id = SegmentId::new();
                    builder.build(&dir, id, None).await.unwrap();
                    readers.push(
                        SegmentReader::open(&dir, id, Arc::clone(&schema), 0)
                            .await
                            .unwrap(),
                    );
                }
                assert_eq!(readers[0].fast_field(field.0).is_none(), missing);
                readers
            });
            let merger = SegmentMerger::new(schema);
            // Overwrite the same unpublished output so iterations do not
            // retain an unbounded collection of output segments in RAM.
            let output = SegmentId::new();
            let (meta, _) = runtime
                .block_on(merger.merge(&dir, &sources, output, None))
                .unwrap();
            assert_eq!(meta.num_docs, 2 * count);
            group.throughput(Throughput::Elements(u64::from(count) * 2));
            group.bench_with_input(
                BenchmarkId::new(
                    if missing {
                        "missing_fast_column"
                    } else {
                        "copy_fast_columns"
                    },
                    count,
                ),
                &count,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            runtime
                                .block_on(merger.merge(&dir, &sources, output, None))
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

/// Exercise the production compactor; setup, deletion, validation and optional
/// byte captures remain outside timing. The output is overwritten each iteration.
fn bench_row_compaction(c: &mut Criterion) {
    use summa_core::{Index, IndexConfig, NoMergePolicy};
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("row_compaction");
    group.sample_size(20);
    for pattern in [
        "mixed_fast_columns",
        "clustered_mixed_fields",
        "scattered_mixed_fields",
    ] {
        for count in [4_096u32, 65_536] {
            let dir = RamDirectory::new();
            let mut sb = SchemaBuilder::default();
            let id = sb.add_text_field("id", false, false);
            sb.set_primary_key(id);
            let mut columns = Vec::new();
            for n in 0..8 {
                let field = sb.add_u64_field(&format!("value{n}"), false, false);
                sb.set_fast(field, true);
                sb.set_multi(field, n == 7);
                columns.push(field);
            }
            let body =
                (pattern != "mixed_fast_columns").then(|| sb.add_text_field("body", true, true));
            let schema = sb.build();
            let (index, mut writer) = runtime.block_on(async {
                let index = Index::create(
                    dir.clone(),
                    schema,
                    IndexConfig {
                        merge_policy: Box::new(NoMergePolicy),
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
                let mut writer = index.writer();
                writer.init_primary_key_dedup().await.unwrap();
                for i in 0..count {
                    let mut doc = Document::new();
                    doc.add_text(id, format!("key{i:08}"));
                    if let Some(body) = body {
                        doc.add_text(
                            body,
                            format!("common common common group{} retained text", i % 17),
                        );
                    }
                    for (n, &field) in columns.iter().enumerate() {
                        if !(i + n as u32).is_multiple_of(7) {
                            doc.add_u64(field, u64::from(i) * 17 + n as u64);
                            if n == 7 {
                                doc.add_u64(field, u64::from(i) * 31);
                            }
                        }
                    }
                    loop {
                        match writer.add_document(doc.clone()) {
                            Ok(()) => break,
                            Err(summa_core::Error::QueueFull) => {
                                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                            }
                            Err(error) => panic!("fixture admission failed: {error}"),
                        }
                    }
                    if body.is_some() && (i + 1).is_multiple_of(1024) {
                        writer.commit().await.unwrap();
                    }
                }
                writer.commit().await.unwrap();
                if body.is_some() {
                    writer.force_merge().await.unwrap();
                }
                for i in 0..count {
                    let deleted = match pattern {
                        "mixed_fast_columns" => i.is_multiple_of(2),
                        "clustered_mixed_fields" => i >= count / 4 && i < count / 2,
                        _ => i.is_multiple_of(4),
                    };
                    if deleted {
                        writer.delete_primary_key(&format!("key{i:08}")).unwrap();
                    }
                }
                writer.commit().await.unwrap();
                (index, writer)
            });
            let searcher =
                runtime.block_on(async { index.reader().await.unwrap().searcher().await.unwrap() });
            assert_eq!(searcher.segment_readers().len(), 1);
            let source = &searcher.segment_readers()[0];
            let merger = SegmentMerger::new(index.schema().clone());
            let output = SegmentId::new();
            let budget = 32 * 1024 * 1024;
            let (meta, _) = runtime
                .block_on(merger.compact(&dir, source, output, budget))
                .unwrap();
            assert_eq!(
                meta.num_docs,
                if body.is_some() {
                    count * 3 / 4
                } else {
                    count / 2
                }
            );
            if let Ok(prefix) = std::env::var("SUMMA_COMPACTION_BYTES") {
                use summa_core::Directory;
                let path = summa_core::segment::SegmentFiles::new(output.0).fast;
                let bytes = runtime.block_on(async {
                    dir.open_read(&path)
                        .await
                        .unwrap()
                        .read_bytes()
                        .await
                        .unwrap()
                });
                std::fs::write(format!("{prefix}-{pattern}-{count}.fast"), bytes.as_slice())
                    .unwrap();
            }
            group.throughput(Throughput::Elements(u64::from(count)));
            group.bench_with_input(BenchmarkId::new(pattern, count), &count, |b, _| {
                b.iter(|| {
                    black_box(
                        runtime
                            .block_on(merger.compact(&dir, source, output, budget))
                            .unwrap(),
                    )
                });
            });
            runtime.block_on(writer.shutdown()).unwrap();
        }
    }
    group.finish();
}

/// End-to-end deletion publication and PK visibility refresh. Each iteration
/// starts from identical immutable files; copying/opening/staging/validation
/// and worker shutdown stay outside the reported duration.
fn bench_row_deletion(c: &mut Criterion) {
    use summa_core::{Directory, DirectoryWriter, Index, IndexConfig, NoMergePolicy};
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("row_deletion");
    group.sample_size(10);
    for count in [4_096u32, 65_536] {
        let config = IndexConfig {
            num_indexing_threads: 1,
            merge_policy: Box::new(NoMergePolicy),
            ..Default::default()
        };
        let files = runtime.block_on(async {
            let dir = RamDirectory::new();
            let mut schema = SchemaBuilder::default();
            let id = schema.add_text_field("id", false, false);
            schema.set_primary_key(id);
            let index = Index::create(dir.clone(), schema.build(), config.clone())
                .await
                .unwrap();
            let mut writer = index.writer();
            writer.init_primary_key_dedup().await.unwrap();
            for i in 0..count {
                let mut doc = Document::new();
                doc.add_text(id, format!("key{i:08}"));
                loop {
                    match writer.add_document(doc.clone()) {
                        Ok(()) => break,
                        Err(summa_core::Error::QueueFull) => {
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await
                        }
                        Err(error) => panic!("fixture admission failed: {error}"),
                    }
                }
            }
            writer.commit().await.unwrap();
            writer.shutdown().await.unwrap();
            let mut files = Vec::new();
            for path in dir.list_files(std::path::Path::new("")).await.unwrap() {
                let bytes = dir
                    .open_read(&path)
                    .await
                    .unwrap()
                    .read_bytes()
                    .await
                    .unwrap();
                files.push((path, bytes));
            }
            files
        });
        group.throughput(Throughput::Elements(u64::from(count)));
        group.bench_with_input(
            BenchmarkId::new("commit_64_keys", count),
            &count,
            |b, &count| {
                b.iter_custom(|iterations| {
                    runtime.block_on(async {
                        let mut elapsed = std::time::Duration::ZERO;
                        for _ in 0..iterations {
                            let dir = RamDirectory::new();
                            for (path, bytes) in &files {
                                dir.write(path, bytes.as_slice()).await.unwrap();
                            }
                            let index = Index::open(dir, config.clone()).await.unwrap();
                            let mut writer = index.writer();
                            writer.init_primary_key_dedup().await.unwrap();
                            for key in (0..count).step_by(count as usize / 64) {
                                writer.delete_primary_key(&format!("key{key:08}")).unwrap();
                            }
                            let start = std::time::Instant::now();
                            assert!(writer.commit().await.unwrap());
                            elapsed += start.elapsed();
                            let searcher = index.reader().await.unwrap().searcher().await.unwrap();
                            assert_eq!(searcher.num_docs(), count - 64);
                            assert_eq!(searcher.segment_readers()[0].num_docs(), count);
                            writer.shutdown().await.unwrap();
                        }
                        elapsed
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_segment_merge,
    bench_row_compaction,
    bench_row_deletion
);
criterion_main!(benches);
