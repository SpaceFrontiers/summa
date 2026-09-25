use summa_core::{
    directories::MmapDirectory,
    dsl::{Document, SchemaBuilder},
    index::{Index, IndexConfig, IndexWriter},
    query::SparseVectorQuery,
    structures::{SparseFormat, SparseVectorConfig, WeightQuantization},
};
use std::{fs::File, path::Path, time::Instant};
struct Csr {
    bytes: memmap2::Mmap,
    rows: usize,
    nnz: usize,
}
impl Csr {
    fn open(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let bytes = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let rows = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let nnz = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 24 + (rows + 1) * 8 + nnz * 8);
        Self { bytes, rows, nnz }
    }
    fn row(&self, row: usize) -> Vec<(u32, f32)> {
        let start =
            u64::from_le_bytes(self.bytes[24 + row * 8..32 + row * 8].try_into().unwrap()) as usize;
        let end =
            u64::from_le_bytes(self.bytes[32 + row * 8..40 + row * 8].try_into().unwrap()) as usize;
        let off = 24 + (self.rows + 1) * 8;
        (start..end)
            .map(|i| {
                (
                    u32::from_le_bytes(
                        self.bytes[off + i * 4..off + i * 4 + 4].try_into().unwrap(),
                    ),
                    f32::from_le_bytes(
                        self.bytes[off + (self.nnz + i) * 4..off + (self.nnz + i) * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ),
                )
            })
            .collect()
    }
}
fn config() -> IndexConfig {
    IndexConfig {
        num_threads: 4,
        num_indexing_threads: 1,
        num_compression_threads: 2,
        max_indexing_memory_bytes: 16 * 1024 * 1024 * 1024,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..IndexConfig::default()
    }
}
fn bytes(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            if p.is_dir() {
                bytes(&p)
            } else {
                p.metadata().unwrap().len()
            }
        })
        .sum()
}
fn live_bytes(path: &Path) -> u64 {
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("metadata.json")).unwrap()).unwrap();
    let segments = metadata["segment_metas"].as_object().unwrap();
    std::fs::read_dir(path)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            (name == "metadata.json"
                || segments
                    .keys()
                    .any(|id| name.starts_with(&format!("seg_{id}."))))
            .then(|| entry.metadata().unwrap().len())
        })
        .sum()
}
fn backend(path: &Path) -> String {
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("metadata.json")).unwrap()).unwrap();
    metadata["schema"]["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|field| field["name"] == "vector")
        .unwrap()["sparse_vector_config"]["format"]
        .as_str()
        .unwrap_or("Bmp")
        .to_owned()
}
#[tokio::main]
async fn main() {
    if std::env::var_os("SUMMA_BENCH_LOG").is_some() {
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Debug);
    }
    let a: Vec<_> = std::env::args().collect();
    let path = Path::new(&a[2]);
    match a[1].as_str() {
        "build" => {
            let corpus = Csr::open(&a[3]);
            let n = a[4].parse::<usize>().unwrap().min(corpus.rows);
            let batch = a[5].parse::<usize>().unwrap();
            let mut sb = SchemaBuilder::default();
            let id = sb.add_u64_field("id", false, true);
            let mut sparse = SparseVectorConfig {
                dims: Some(
                    u32::try_from(u64::from_le_bytes(corpus.bytes[8..16].try_into().unwrap()))
                        .unwrap(),
                ),
                weight_quantization: WeightQuantization::Float32,
                ..SparseVectorConfig::default()
            };
            sparse.format = match a
                .get(6)
                .map(String::as_str)
                .expect("build requires engine bmp|maxscore|seismic")
            {
                "bmp" => {
                    sparse.max_weight = Some(5.0);
                    SparseFormat::Bmp
                }
                "maxscore" => SparseFormat::MaxScore,
                "seismic" => SparseFormat::Seismic,
                other => panic!("unknown sparse backend {other}"),
            };
            let field = sb.add_sparse_vector_field_with_config("vector", true, false, sparse);
            let started = Instant::now();
            let mut writer = IndexWriter::create(MmapDirectory::new(path), sb.build(), config())
                .await
                .unwrap();
            let mut backpressure_retries = 0u64;
            for row in 0..n {
                loop {
                    let mut doc = Document::new();
                    doc.add_u64(id, row as u64);
                    doc.add_sparse_vector(field, corpus.row(row));
                    match writer.add_document(doc) {
                        Ok(()) => break,
                        Err(summa_core::Error::QueueFull) => {
                            backpressure_retries += 1;
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        }
                        Err(e) => panic!("{e}"),
                    }
                }
                if (row + 1) % batch == 0 {
                    writer.commit().await.unwrap();
                }
            }
            writer.commit().await.unwrap();
            println!(
                "{}",
                serde_json::json!({"operation":"build","rows":n,"backpressure_retries":backpressure_retries,"seconds":started.elapsed().as_secs_f64(),"directory_bytes":bytes(path),"live_bytes":live_bytes(path),"backend":backend(path)})
            );
        }
        "merge" | "maintain" => {
            let mut writer = IndexWriter::open(MmapDirectory::new(path), config())
                .await
                .unwrap();
            let started = Instant::now();
            if a[1] == "merge" {
                writer.force_merge().await.unwrap();
            } else {
                writer.reorder().await.unwrap();
            }
            println!(
                "{}",
                serde_json::json!({"operation":a[1],"seconds":started.elapsed().as_secs_f64(),"directory_bytes":bytes(path),"live_bytes":live_bytes(path),"backend":backend(path)})
            );
        }
        "search" => {
            let queries = Csr::open(&a[3]);
            let n = a[4].parse::<usize>().unwrap();
            let k = a[5].parse::<usize>().unwrap();
            let exhaustive = a[6] == "exact";
            let opening = Instant::now();
            let index = Index::open(MmapDirectory::new(path), config())
                .await
                .unwrap();
            let reader = index.reader().await.unwrap();
            let searcher = reader.searcher().await.unwrap();
            let open_ms = opening.elapsed().as_secs_f64() * 1000.0;
            let schema = reader.schema();
            let field = schema.get_field("vector").unwrap();
            let id = schema.get_field("id").unwrap();
            let selected: Vec<_> = (0..queries.rows)
                .map(|row| (row, queries.row(row)))
                .filter(|(_, v)| v.len() <= 64)
                .take(n)
                .collect();
            let mut latency = Vec::new();
            let mut first_ms = 0.0;
            let mut hits = Vec::new();
            for pass in 0..3 {
                for (qid, vector) in &selected {
                    let mut query = SparseVectorQuery::new(field, vector.clone());
                    query = query.with_exhaustive(exhaustive);
                    if exhaustive {
                        query = query.with_lsp_gamma(0);
                    }
                    let t = Instant::now();
                    let result = searcher.search(&query, k).await.unwrap();
                    let elapsed = t.elapsed().as_secs_f64() * 1000.0;
                    if pass == 0 && first_ms == 0.0 {
                        first_ms = elapsed;
                    }
                    if pass > 0 {
                        latency.push(elapsed);
                    }
                    if pass == 2 {
                        let mut row = Vec::new();
                        for hit in result {
                            let doc = searcher
                                .doc(hit.segment_id, hit.doc_id)
                                .await
                                .unwrap()
                                .unwrap();
                            row.push((doc.get_first(id).unwrap().as_u64().unwrap(), hit.score));
                        }
                        hits.push((*qid, row));
                    }
                }
            }
            latency.sort_by(f64::total_cmp);
            let mean = latency.iter().sum::<f64>() / latency.len() as f64;
            let percentile = |p: f64| latency[((latency.len() - 1) as f64 * p) as usize];
            println!(
                "{}",
                serde_json::json!({"operation":"search","queries":selected.len(),"k":k,"exact":exhaustive,"open_ms":open_ms,"first_query_ms":first_ms,"mean_ms":mean,"p50_ms":percentile(0.5),"p95_ms":percentile(0.95),"p99_ms":percentile(0.99),"directory_bytes":bytes(path),"live_bytes":live_bytes(path),"backend":backend(path)})
            );
            std::fs::write(&a[7], serde_json::to_vec(&hits).unwrap()).unwrap();
        }
        _ => panic!("build|merge|maintain|search"),
    }
}

struct BenchLogger;
static LOGGER: BenchLogger = BenchLogger;
impl log::Log for BenchLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record<'_>) {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        if self.enabled(record.metadata()) {
            eprintln!(
                "{:.6} {} {}",
                START.get_or_init(Instant::now).elapsed().as_secs_f64(),
                record.target(),
                record.args()
            );
        }
    }
    fn flush(&self) {}
}
