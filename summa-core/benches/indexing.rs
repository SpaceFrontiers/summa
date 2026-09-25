//! Indexing benchmarks

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use summa_core::{
    Document, Field, FsDirectory, IndexConfig, IndexWriter, RamDirectory, Schema, SchemaBuilder,
};
use tempfile::TempDir;

struct TestSchema {
    schema: Schema,
    title: Field,
    body: Field,
    id: Field,
}

fn create_schema() -> TestSchema {
    let mut builder = SchemaBuilder::default();
    let title = builder.add_text_field("title", true, true);
    let body = builder.add_text_field("body", true, false);
    let id = builder.add_u64_field("id", true, true);
    TestSchema {
        schema: builder.build(),
        title,
        body,
        id,
    }
}

fn generate_documents(test_schema: &TestSchema, count: usize) -> Vec<Document> {
    let words = [
        "search",
        "engine",
        "fast",
        "index",
        "document",
        "query",
        "term",
        "field",
        "score",
        "ranking",
        "relevance",
        "match",
        "filter",
        "sort",
    ];

    (0..count)
        .map(|i| {
            let title_words: Vec<&str> = (0..5).map(|j| words[(i + j) % words.len()]).collect();
            let body_words: Vec<&str> = (0..50).map(|j| words[(i + j * 3) % words.len()]).collect();

            let mut doc = Document::new();
            doc.add_text(test_schema.title, title_words.join(" "));
            doc.add_text(test_schema.body, body_words.join(" "));
            doc.add_u64(test_schema.id, i as u64);
            doc
        })
        .collect()
}

fn bench_document_indexing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let test_schema = create_schema();
    let doc_counts = [100, 1000, 10000];

    let mut group = c.benchmark_group("indexing/documents");

    for count in doc_counts {
        let documents = generate_documents(&test_schema, count);
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("ram", count), &documents, |b, docs| {
            b.iter(|| {
                rt.block_on(async {
                    let dir = RamDirectory::new();
                    let config = IndexConfig::default();
                    let mut writer = IndexWriter::create(dir, test_schema.schema.clone(), config)
                        .await
                        .unwrap();

                    for doc in docs {
                        writer.add_document(black_box(doc.clone())).unwrap();
                    }

                    writer.commit().await.unwrap();
                })
            })
        });
    }

    group.finish();
}

fn bench_segment_build(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let test_schema = create_schema();
    let documents = generate_documents(&test_schema, 1000);

    let mut group = c.benchmark_group("indexing/segment_build");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("build_and_commit", |b| {
        b.iter(|| {
            rt.block_on(async {
                let dir = RamDirectory::new();
                let config = IndexConfig::default();
                let mut writer = IndexWriter::create(dir, test_schema.schema.clone(), config)
                    .await
                    .unwrap();

                for doc in &documents {
                    writer.add_document(black_box(doc.clone())).unwrap();
                }

                writer.commit().await.unwrap();
            })
        })
    });

    group.finish();
}

fn bench_fs_indexing(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let test_schema = create_schema();
    let documents = generate_documents(&test_schema, 1000);

    let mut group = c.benchmark_group("indexing/filesystem");
    group.throughput(Throughput::Elements(1000));
    group.sample_size(10); // Fewer samples for disk I/O

    group.bench_function("fs_directory", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_dir = TempDir::new().unwrap();
                let dir = FsDirectory::new(temp_dir.path());
                let config = IndexConfig::default();
                let mut writer = IndexWriter::create(dir, test_schema.schema.clone(), config)
                    .await
                    .unwrap();

                for doc in &documents {
                    writer.add_document(black_box(doc.clone())).unwrap();
                }

                writer.commit().await.unwrap();
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_document_indexing,
    bench_segment_build,
    bench_fs_indexing
);
criterion_main!(benches);
