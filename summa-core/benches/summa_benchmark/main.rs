//! Unified Summa Benchmark Suite
//!
//! Benchmarks:
//! 1. Dense vector indexes with different MRL dimensions
//! 2. Dense vector indexes with different nprobe values
//! 3. Sparse vector indexes (SPLADE)
//! 4. BM25 text search
//! 5. MS MARCO IR metrics comparison (all methods)
//!
//! Run with:
//!   BENCHMARK_DATA=/absolute/path/to/benchmark_data cargo bench -p summa-core --bench summa_benchmark --locked
//!
//! Generate data first:
//!   python summa-core/benches/generate_benchmark_data.py --use-beir --output-dir /absolute/path/to/benchmark_data

mod config;
mod corpus;
mod data;
mod metrics;
mod output;

use std::time::Instant;

use config::BenchmarkConfig;
use corpus::corpus_ids;
use data::{DenseData, GroundTruth, Qrels, SparseData, TextData};
use metrics::{LatencyStats, mrr_at_10, ndcg_at_k, recall_at_k};
use output::{
    IndexingResult, IrResult, MrlResult, print_header, print_indexing_table, print_ir_table,
    print_mrl_table,
};
use summa_core::directories::RamDirectory;
use summa_core::dsl::{DenseVectorConfig, Document, Schema};
use summa_core::index::{Index, IndexConfig, IndexWriter};
use summa_core::query::{DenseVectorQuery, SparseVectorQuery};
use tokio::runtime::Runtime;

/// MRL dimensions to benchmark (Jina v3 supports: 32, 64, 128, 256, 512, 768, 1024)
const MRL_DIMS: &[usize] = &[64, 128, 256, 512, 1024];

/// nprobe values to benchmark for dense vector search
const NPROBE_VALUES: &[usize] = &[8, 16, 32, 64];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(run_benchmarks())
}

async fn run_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    let bench_config = BenchmarkConfig::from_env();

    if !bench_config.has_data() {
        println!(
            "\n❌ Benchmark data not found at {:?}",
            bench_config.data_dir
        );
        println!("   Generate it first:");
        println!(
            "   python summa-core/benches/generate_benchmark_data.py --use-beir --output-dir /absolute/path/to/benchmark_data\n"
        );
        return Err("benchmark data is missing; see docs/benchmarks.md".into());
    }

    // Load dense data (required)
    let dense_docs = match DenseData::load(&bench_config.dense_embeddings_path()) {
        Some(d) => d,
        None => {
            return Err("failed to load dense embeddings".into());
        }
    };

    let dense_queries = match DenseData::load(&bench_config.dense_queries_path()) {
        Some(d) => d,
        None => {
            return Err("failed to load dense queries".into());
        }
    };

    let ground_truth_full = match GroundTruth::load(&bench_config.ground_truth_dense_full_path()) {
        Some(gt) => gt,
        None => {
            return Err("failed to load ground truth".into());
        }
    };

    // Load optional data
    let qrels = Qrels::load(&bench_config.qrels_path());
    let sparse_docs = SparseData::load(&bench_config.sparse_embeddings_path());
    let sparse_queries = SparseData::load(&bench_config.sparse_queries_path());
    let text_passages = TextData::load(&bench_config.passages_path());
    let text_queries = TextData::load(&bench_config.queries_text_path());

    if dense_docs.dim == 0 || dense_docs.vectors.is_empty() || dense_queries.vectors.is_empty() {
        return Err("dense corpus and queries must be nonempty with positive dimensions".into());
    }
    if dense_docs.dim != dense_queries.dim
        || ground_truth_full.neighbors.len() != dense_queries.vectors.len()
        || ground_truth_full.k == 0
    {
        return Err("dense dimensions or ground-truth query counts do not match".into());
    }
    if ground_truth_full
        .neighbors
        .iter()
        .flatten()
        .any(|&id| id as usize >= dense_docs.vectors.len())
    {
        return Err("ground truth contains a document outside the dense corpus".into());
    }
    for (path, loaded) in [
        (bench_config.sparse_embeddings_path(), sparse_docs.is_some()),
        (bench_config.sparse_queries_path(), sparse_queries.is_some()),
        (bench_config.passages_path(), text_passages.is_some()),
        (bench_config.queries_text_path(), text_queries.is_some()),
        (bench_config.qrels_path(), qrels.is_some()),
    ] {
        if path.exists() && !loaded {
            return Err(
                format!("failed to load optional benchmark data: {}", path.display()).into(),
            );
        }
    }
    if sparse_docs.is_some() != sparse_queries.is_some()
        || text_passages.is_some() != text_queries.is_some()
        || sparse_docs
            .as_ref()
            .is_some_and(|d| d.vectors.len() != dense_docs.vectors.len())
        || sparse_queries
            .as_ref()
            .is_some_and(|d| d.vectors.len() != dense_queries.vectors.len())
        || text_passages
            .as_ref()
            .is_some_and(|d| d.texts.len() != dense_docs.vectors.len())
        || text_queries
            .as_ref()
            .is_some_and(|d| d.texts.len() != dense_queries.vectors.len())
    {
        return Err(
            "optional corpus/query files must be paired and align with the dense data".into(),
        );
    }
    if qrels.as_ref().is_some_and(|q| {
        q.relevance.iter().any(|(&query, docs)| {
            query as usize >= dense_queries.vectors.len()
                || docs.is_empty()
                || docs
                    .iter()
                    .any(|&id| id as usize >= dense_docs.vectors.len())
        })
    }) {
        return Err("qrels must contain nonempty judgments within the corpus/query bounds".into());
    }

    // Limit queries if specified
    let (dense_queries, ground_truth_full) = if let Some(n) = bench_config.num_queries {
        (dense_queries.take(n), ground_truth_full.take(n))
    } else {
        (dense_queries, ground_truth_full)
    };

    let full_dim = dense_docs.dim;
    let num_docs = dense_docs.vectors.len();
    let num_queries = dense_queries.vectors.len();

    // Print header
    print_header(&format!(
        "Dataset: {} docs, {} queries | Dense: {}-dim (Jina v3)",
        num_docs, num_queries, full_dim
    ));

    // =========================================================================
    // 1. MRL Dimension Comparison using high-level Index API
    // =========================================================================
    println!("\nRunning MRL dimension comparison with Summa Index API...");
    let mut mrl_results = Vec::new();

    for &mrl_dim in MRL_DIMS.iter().filter(|&&d| d <= full_dim) {
        println!("  Building index with mrl_dim={}...", mrl_dim);

        // Create schema with dense vector field at truncated dimension
        let mut schema_builder = Schema::builder();
        let dense_config = DenseVectorConfig::new(mrl_dim);
        let embedding_field = schema_builder.add_dense_vector_field_with_config(
            "embedding",
            true,  // indexed
            false, // stored (don't need to store for benchmark)
            dense_config,
        );
        let corpus_id = schema_builder.add_u64_field("corpus_id", false, true);
        let schema = schema_builder.build();

        // Create index in RAM
        let directory = RamDirectory::new();
        let index_config = IndexConfig::default();
        let mut writer = IndexWriter::create(directory.clone(), schema, index_config.clone())
            .await
            .expect("Failed to create index writer");

        // Add documents
        let build_start = Instant::now();
        for (i, vector) in dense_docs.vectors.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_u64(corpus_id, i as u64);
            // Truncate vector to mrl_dim
            let truncated: Vec<f32> = vector.iter().take(mrl_dim).copied().collect();
            doc.add_dense_vector(embedding_field, truncated);
            writer.add_document(doc).expect("Failed to add document");

            if (i + 1) % 10000 == 0 {
                print!("\r    Indexed {}/{} docs", i + 1, num_docs);
            }
        }
        writer.commit().await.expect("Failed to commit");
        writer.wait_for_merging_thread().await;
        let build_time = build_start.elapsed();
        println!(
            "\r    Indexed {} docs in {:.2}s",
            num_docs,
            build_time.as_secs_f64()
        );

        // Open index for searching
        let index = Index::open(directory, index_config)
            .await
            .expect("Failed to open index");

        // Search and measure recall
        let k = 10;
        let mut latencies = Vec::new();
        let mut total_recall = 0.0f32;

        for (i, query) in dense_queries.vectors.iter().enumerate() {
            // Truncate query to mrl_dim
            let truncated_query: Vec<f32> = query.iter().take(mrl_dim).copied().collect();
            let query_obj = DenseVectorQuery::new(embedding_field, truncated_query);

            let start = Instant::now();
            let results = index.search(&query_obj, k).await.expect("Search failed");
            latencies.push(start.elapsed());

            // Compute recall against full-dim ground truth
            let predicted = corpus_ids(&index, &results.hits).await;
            let gt = &ground_truth_full.neighbors[i];
            total_recall += recall_at_k(&predicted, gt, k);
        }

        let stats = LatencyStats::from_durations(&latencies);
        let avg_recall = total_recall / num_queries as f32;

        mrl_results.push(MrlResult {
            dim: mrl_dim,
            recall_at_10: avg_recall,
            latency_us: stats.avg_us,
        });
    }

    print_mrl_table(&mrl_results, full_dim);

    // =========================================================================
    // 2. Dense Index with different nprobe values
    // =========================================================================
    println!("\nRunning dense index nprobe comparison...");
    let target_dim = bench_config.dense_dim.min(full_dim);

    // Build one index and test with different nprobe values
    let mut schema_builder = Schema::builder();
    let dense_config = DenseVectorConfig::new(target_dim);
    let embedding_field =
        schema_builder.add_dense_vector_field_with_config("embedding", true, false, dense_config);
    let corpus_id = schema_builder.add_u64_field("corpus_id", false, true);
    let schema = schema_builder.build();

    let directory = RamDirectory::new();
    let index_config = IndexConfig::default();
    let mut writer = IndexWriter::create(directory.clone(), schema, index_config.clone())
        .await
        .expect("Failed to create index writer");

    println!("  Building dense index (dim={})...", target_dim);
    for (i, vector) in dense_docs.vectors.iter().enumerate() {
        let mut doc = Document::new();
        doc.add_u64(corpus_id, i as u64);
        let truncated: Vec<f32> = vector.iter().take(target_dim).copied().collect();
        doc.add_dense_vector(embedding_field, truncated);
        writer.add_document(doc).expect("Failed to add document");
        if (i + 1) % 10000 == 0 {
            print!("\r    Indexed {}/{} docs", i + 1, num_docs);
        }
    }
    writer.commit().await.expect("Failed to commit");
    writer
        .build_vector_index()
        .await
        .expect("Failed to train and build IVF-TQ index");
    writer.wait_for_merging_thread().await;
    println!("\r    Indexed {} docs", num_docs);

    let dense_index = Index::open(directory, index_config)
        .await
        .expect("Failed to open index");

    // Test different nprobe values
    let mut nprobe_results = Vec::new();
    for &nprobe in NPROBE_VALUES {
        let k = 10;
        let mut latencies = Vec::new();
        let mut total_recall = 0.0f32;

        for (i, query) in dense_queries.vectors.iter().enumerate() {
            let truncated_query: Vec<f32> = query.iter().take(target_dim).copied().collect();
            let query_obj =
                DenseVectorQuery::new(embedding_field, truncated_query).with_nprobe(nprobe);

            let start = Instant::now();
            let results = dense_index
                .search(&query_obj, k)
                .await
                .expect("Search failed");
            latencies.push(start.elapsed());

            let predicted = corpus_ids(&dense_index, &results.hits).await;
            let gt = &ground_truth_full.neighbors[i];
            total_recall += recall_at_k(&predicted, gt, k);
        }

        let stats = LatencyStats::from_durations(&latencies);
        nprobe_results.push(MrlResult {
            dim: nprobe, // Reusing MrlResult for nprobe comparison
            recall_at_10: total_recall / num_queries as f32,
            latency_us: stats.avg_us,
        });
    }

    output::print_nprobe_table(&nprobe_results, target_dim);

    // =========================================================================
    // 3. Sparse Vector Index (SPLADE) - if data available
    // =========================================================================
    let sparse_index = if let (Some(sparse_docs), Some(sparse_queries)) =
        (&sparse_docs, &sparse_queries)
    {
        println!("\nBuilding sparse vector index (SPLADE)...");

        let mut schema_builder = Schema::builder();
        let sparse_field = schema_builder.add_sparse_vector_field("sparse_embedding", true, false);
        let corpus_id = schema_builder.add_u64_field("corpus_id", false, true);
        let schema = schema_builder.build();

        let directory = RamDirectory::new();
        let index_config = IndexConfig::default();
        let mut writer = IndexWriter::create(directory.clone(), schema, index_config.clone())
            .await
            .expect("Failed to create sparse index writer");

        for (i, (indices, values)) in sparse_docs.vectors.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_u64(corpus_id, i as u64);
            let entries: Vec<(u32, f32)> = indices
                .iter()
                .copied()
                .zip(values.iter().copied())
                .collect();
            doc.add_sparse_vector(sparse_field, entries);
            writer.add_document(doc).expect("Failed to add document");
            if (i + 1) % 10000 == 0 {
                print!("\r    Indexed {}/{} docs", i + 1, num_docs);
            }
        }
        writer.commit().await.expect("Failed to commit");
        writer.wait_for_merging_thread().await;
        println!("\r    Indexed {} sparse docs", sparse_docs.vectors.len());

        let index = Index::open(directory, index_config)
            .await
            .expect("Failed to open sparse index");

        Some((index, sparse_field, sparse_queries))
    } else {
        println!("\n⚠️  Sparse data not found, skipping sparse benchmarks");
        None
    };

    // =========================================================================
    // 4. BM25 Text Search - if data available
    // =========================================================================
    let bm25_index = if let (Some(passages), Some(queries)) = (&text_passages, &text_queries) {
        println!("\nBuilding BM25 text index...");

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("content", true, false);
        let corpus_id = schema_builder.add_u64_field("corpus_id", false, true);
        let schema = schema_builder.build();

        let directory = RamDirectory::new();
        let index_config = IndexConfig::default();
        let mut writer = IndexWriter::create(directory.clone(), schema, index_config.clone())
            .await
            .expect("Failed to create BM25 index writer");

        for (i, text) in passages.texts.iter().enumerate() {
            let mut doc = Document::new();
            doc.add_u64(corpus_id, i as u64);
            doc.add_text(text_field, text);
            writer.add_document(doc).expect("Failed to add document");
            if (i + 1) % 10000 == 0 {
                print!("\r    Indexed {}/{} docs", i + 1, passages.texts.len());
            }
        }
        writer.commit().await.expect("Failed to commit");
        writer.wait_for_merging_thread().await;
        println!("\r    Indexed {} text docs", passages.texts.len());

        let index = Index::open(directory, index_config)
            .await
            .expect("Failed to open BM25 index");

        Some((index, text_field, queries))
    } else {
        println!("\n⚠️  Text data not found, skipping BM25 benchmarks");
        None
    };

    // =========================================================================
    // 5. MS MARCO IR Metrics Comparison (all methods)
    // =========================================================================
    if let Some(ref qrels) = qrels {
        println!("\nComputing MS MARCO IR metrics (all methods)...");
        let mut ir_results = Vec::new();
        let k = 100;

        // Dense vector IR metrics
        {
            let mut latencies = Vec::new();
            let mut total_mrr = 0.0f32;
            let mut total_ndcg = 0.0f32;
            let mut total_recall = 0.0f32;
            let mut count = 0usize;

            for (i, query) in dense_queries.vectors.iter().enumerate() {
                if let Some(relevant) = qrels.relevance.get(&(i as u32)) {
                    let truncated_query: Vec<f32> =
                        query.iter().take(target_dim).copied().collect();
                    let query_obj = DenseVectorQuery::new(embedding_field, truncated_query);
                    let start = Instant::now();
                    let results = dense_index
                        .search(&query_obj, k)
                        .await
                        .expect("Search failed");
                    latencies.push(start.elapsed());

                    let predicted = corpus_ids(&dense_index, &results.hits).await;
                    total_mrr += mrr_at_10(&predicted, relevant);
                    total_ndcg += ndcg_at_k(&predicted, relevant, 10);
                    let relevant_found = predicted
                        .iter()
                        .filter(|&idx| relevant.contains(&(*idx as u32)))
                        .count();
                    total_recall += relevant_found as f32 / relevant.len() as f32;
                    count += 1;
                }
            }

            if count > 0 {
                let stats = LatencyStats::from_durations(&latencies);
                ir_results.push(IrResult {
                    name: format!("Dense (dim={})", target_dim),
                    mrr_at_10: total_mrr / count as f32,
                    ndcg_at_10: total_ndcg / count as f32,
                    recall_at_100: total_recall / count as f32,
                    latency_us: stats.avg_us,
                });
            }
        }

        // Sparse vector IR metrics
        if let Some((ref index, sparse_field, sparse_queries)) = sparse_index {
            let mut latencies = Vec::new();
            let mut total_mrr = 0.0f32;
            let mut total_ndcg = 0.0f32;
            let mut total_recall = 0.0f32;
            let mut count = 0usize;

            for (i, (indices, values)) in
                sparse_queries.vectors.iter().take(num_queries).enumerate()
            {
                if let Some(relevant) = qrels.relevance.get(&(i as u32)) {
                    let query_obj = SparseVectorQuery::new(
                        sparse_field,
                        indices
                            .iter()
                            .zip(values.iter())
                            .map(|(&i, &v)| (i, v))
                            .collect(),
                    );
                    let start = Instant::now();
                    let results = index.search(&query_obj, k).await.expect("Search failed");
                    latencies.push(start.elapsed());

                    let predicted = corpus_ids(index, &results.hits).await;
                    total_mrr += mrr_at_10(&predicted, relevant);
                    total_ndcg += ndcg_at_k(&predicted, relevant, 10);
                    let relevant_found = predicted
                        .iter()
                        .filter(|&idx| relevant.contains(&(*idx as u32)))
                        .count();
                    total_recall += relevant_found as f32 / relevant.len() as f32;
                    count += 1;
                }
            }

            if count > 0 {
                let stats = LatencyStats::from_durations(&latencies);
                ir_results.push(IrResult {
                    name: "Sparse (SPLADE)".to_string(),
                    mrr_at_10: total_mrr / count as f32,
                    ndcg_at_10: total_ndcg / count as f32,
                    recall_at_100: total_recall / count as f32,
                    latency_us: stats.avg_us,
                });
            }
        }

        // BM25 IR metrics
        if let Some((ref index, text_field, text_queries)) = bm25_index {
            use summa_core::query::TermQuery;

            let mut latencies = Vec::new();
            let mut total_mrr = 0.0f32;
            let mut total_ndcg = 0.0f32;
            let mut total_recall = 0.0f32;
            let mut count = 0usize;

            for (i, query_text) in text_queries.texts.iter().take(num_queries).enumerate() {
                if let Some(relevant) = qrels.relevance.get(&(i as u32)) {
                    // Simple single-term query for now (first word)
                    let first_term = query_text.split_whitespace().next().unwrap_or("");
                    if first_term.is_empty() {
                        continue;
                    }

                    let query_obj = TermQuery::text(text_field, first_term);
                    let start = Instant::now();
                    let results = index.search(&query_obj, k).await.expect("Search failed");
                    latencies.push(start.elapsed());

                    let predicted = corpus_ids(index, &results.hits).await;
                    total_mrr += mrr_at_10(&predicted, relevant);
                    total_ndcg += ndcg_at_k(&predicted, relevant, 10);
                    let relevant_found = predicted
                        .iter()
                        .filter(|&idx| relevant.contains(&(*idx as u32)))
                        .count();
                    total_recall += relevant_found as f32 / relevant.len() as f32;
                    count += 1;
                }
            }

            if count > 0 {
                let stats = LatencyStats::from_durations(&latencies);
                ir_results.push(IrResult {
                    name: "BM25 (single term)".to_string(),
                    mrr_at_10: total_mrr / count as f32,
                    ndcg_at_10: total_ndcg / count as f32,
                    recall_at_100: total_recall / count as f32,
                    latency_us: stats.avg_us,
                });
            }
        }

        if !ir_results.is_empty() {
            print_ir_table(&ir_results);
        }
    } else {
        println!("\n⚠️  No qrels found, skipping IR metrics comparison");
    }

    // =========================================================================
    // 6. Indexing Performance Summary
    // =========================================================================
    println!("\nMeasuring indexing throughput...");
    let subset_size = 10000.min(num_docs);
    let mut indexing_results = Vec::new();

    // Dense indexing throughput
    {
        let mut schema_builder = Schema::builder();
        let dense_config = DenseVectorConfig::new(target_dim);
        let field = schema_builder.add_dense_vector_field_with_config(
            "embedding",
            true,
            false,
            dense_config,
        );
        let schema = schema_builder.build();

        let directory = RamDirectory::new();
        let mut writer = IndexWriter::create(directory, schema, IndexConfig::default())
            .await
            .expect("Failed to create index writer");

        let start = Instant::now();
        for vector in dense_docs.vectors.iter().take(subset_size) {
            let mut doc = Document::new();
            let truncated: Vec<f32> = vector.iter().take(target_dim).copied().collect();
            doc.add_dense_vector(field, truncated);
            writer.add_document(doc).expect("Failed to add document");
        }
        writer.commit().await.expect("Failed to commit");
        writer.wait_for_merging_thread().await;
        let elapsed = start.elapsed().as_secs_f64();

        indexing_results.push(IndexingResult {
            name: format!("Dense (dim={})", target_dim),
            build_time_secs: elapsed,
            merge_time_secs: None,
            throughput_docs_per_sec: subset_size as f64 / elapsed,
        });
    }

    // Sparse indexing throughput (if data available)
    if let Some(ref sparse_docs) = sparse_docs {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_sparse_vector_field("sparse", true, false);
        let schema = schema_builder.build();

        let directory = RamDirectory::new();
        let mut writer = IndexWriter::create(directory, schema, IndexConfig::default())
            .await
            .expect("Failed to create index writer");

        let start = Instant::now();
        for (indices, values) in sparse_docs.vectors.iter().take(subset_size) {
            let mut doc = Document::new();
            let entries: Vec<(u32, f32)> = indices
                .iter()
                .copied()
                .zip(values.iter().copied())
                .collect();
            doc.add_sparse_vector(field, entries);
            writer.add_document(doc).expect("Failed to add document");
        }
        writer.commit().await.expect("Failed to commit");
        writer.wait_for_merging_thread().await;
        let elapsed = start.elapsed().as_secs_f64();

        indexing_results.push(IndexingResult {
            name: "Sparse (SPLADE)".to_string(),
            build_time_secs: elapsed,
            merge_time_secs: None,
            throughput_docs_per_sec: subset_size as f64 / elapsed,
        });
    }

    // BM25 indexing throughput (if data available)
    if let Some(ref passages) = text_passages {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("content", true, false);
        let schema = schema_builder.build();

        let directory = RamDirectory::new();
        let mut writer = IndexWriter::create(directory, schema, IndexConfig::default())
            .await
            .expect("Failed to create index writer");

        let start = Instant::now();
        for text in passages.texts.iter().take(subset_size) {
            let mut doc = Document::new();
            doc.add_text(field, text);
            writer.add_document(doc).expect("Failed to add document");
        }
        writer.commit().await.expect("Failed to commit");
        writer.wait_for_merging_thread().await;
        let elapsed = start.elapsed().as_secs_f64();

        indexing_results.push(IndexingResult {
            name: "BM25 (text)".to_string(),
            build_time_secs: elapsed,
            merge_time_secs: None,
            throughput_docs_per_sec: subset_size as f64 / elapsed,
        });
    }

    print_indexing_table(&indexing_results);

    println!("\n✅ Benchmark complete!\n");
    Ok(())
}
