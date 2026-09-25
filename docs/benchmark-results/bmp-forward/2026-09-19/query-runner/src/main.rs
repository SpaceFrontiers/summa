use summa_core::{
    Document, Index, IndexConfig, IndexWriter,
    directories::MmapDirectory,
    dsl::SchemaBuilder,
    query::{
        CandidateFeature, CandidateScoringPlan, MultiValueCombiner, Query, RankingModel,
        ScoreScope, SearchResult, SparseVectorQuery,
    },
    structures::{SparseFormat, SparseVectorConfig, WeightQuantization},
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
fn config() -> IndexConfig {
    IndexConfig {
        num_threads: 4,
        num_indexing_threads: 1,
        num_compression_threads: 2,
        max_indexing_memory_bytes: 24 * 1024 * 1024 * 1024,
        merge_policy: Box::new(summa_core::NoMergePolicy),
        ..Default::default()
    }
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn usage() -> serde_json::Value {
    unsafe {
        let mut u: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut u);
        serde_json::json!({"maxrss_kib":u.ru_maxrss,"minor_faults":u.ru_minflt,"major_faults":u.ru_majflt,"user_s":u.ru_utime.tv_sec as f64+u.ru_utime.tv_usec as f64/1e6,"system_s":u.ru_stime.tv_sec as f64+u.ru_stime.tv_usec as f64/1e6})
    }
}
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let a: Vec<_> = std::env::args().collect();
    let path = Path::new(&a[2]);
    if a[1] == "build" {
        let file = File::open(&a[3]).unwrap();
        let bytes = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let mut sb = SchemaBuilder::default();
        let field = sb.add_sparse_vector_field_with_config(
            "vector",
            true,
            false,
            SparseVectorConfig {
                format: SparseFormat::Bmp,
                dims: Some(105879),
                weight_quantization: WeightQuantization::UInt8,
                weight_threshold: 0.0,
                max_weight: Some(5.0),
                bmp_block_size: 32,
                ..Default::default()
            },
        );
        sb.set_multi(field, true);
        let mut writer = IndexWriter::create(MmapDirectory::new(path), sb.build(), config())
            .await
            .unwrap();
        let start = Instant::now();
        let mut p = 0;
        let mut docs = 0;
        let mut vectors = 0;
        while p < bytes.len() {
            let id = u32_at(&bytes, p);
            let mut document = Document::new();
            while p < bytes.len() && u32_at(&bytes, p) == id {
                let n = u32_at(&bytes, p + 4) as usize;
                p += 8;
                let vector = (0..n)
                    .map(|i| {
                        (
                            u32_at(&bytes, p + i * 5),
                            bytes[p + i * 5 + 4] as f32 * 5.0 / 255.0,
                        )
                    })
                    .collect();
                p += n * 5;
                document.add_sparse_vector(field, vector);
                vectors += 1;
            }
            loop {
                match writer.add_document(document.clone()) {
                    Ok(()) => break,
                    Err(summa_core::Error::QueueFull) => {
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await
                    }
                    Err(error) => panic!("{error}"),
                }
            }
            docs += 1;
            if docs % 2000 == 0 {
                eprintln!(
                    "build docs={docs} vectors={vectors} elapsed={:.2}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
        writer.commit().await.unwrap();
        println!(
            "{}",
            serde_json::json!({"build_s":start.elapsed().as_secs_f64(),"documents":docs,"vectors":vectors,"usage":usage()})
        );
        return;
    }
    let opening = Instant::now();
    let index = Index::open(MmapDirectory::new(path), config())
        .await
        .unwrap();
    let reader = index.reader().await.unwrap();
    let searcher = reader.searcher().await.unwrap();
    let open_ms = opening.elapsed().as_secs_f64() * 1000.0;
    let field = reader.schema().get_field("vector").unwrap();
    let queries: Vec<Vec<(u32, f32)>> =
        serde_json::from_slice(&std::fs::read(&a[3]).unwrap()).unwrap();
    let depth: usize = a[4].parse().unwrap();
    let mode = &a[5];
    let passes: usize = a[6].parse().unwrap();
    let count: usize = a[7].parse().unwrap();
    let segments: Vec<_> = searcher
        .segment_readers()
        .iter()
        .map(|s| (s.meta().id, s.num_docs()))
        .collect();
    assert_eq!(segments.len(), 1);
    let before = usage();
    let mut times = Vec::new();
    let mut hits = BufWriter::new(File::create(&a[8]).unwrap());
    for pass in 0..passes {
        for (qid, vector) in queries.iter().take(count).enumerate() {
            let start = Instant::now();
            let query = SparseVectorQuery::new(field, vector.clone())
                .with_combiner(MultiValueCombiner::Max);
            let candidates = if mode == "scattered" || mode == "local" {
                let n = segments[0].1 as usize;
                let first = (qid * 7919) % n;
                (0..depth)
                    .map(|i| SearchResult {
                        doc_id: ((first + i * if mode == "local" { 1 } else { 104729 }) % n) as u32,
                        segment_id: segments[0].0,
                        score: 0.0,
                        positions: Vec::new(),
                    })
                    .collect()
            } else {
                searcher
                    .search(&query, if mode == "retrieval" { 10 } else { depth })
                    .await
                    .unwrap()
            };
            let retrieval_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut result = if mode == "retrieval" {
                candidates
            } else {
                let plan = CandidateScoringPlan {
                    features: vec![CandidateFeature {
                        name: "sparse".into(),
                        scope: ScoreScope::Document,
                        query: query.candidate_query().unwrap(),
                    }],
                    backfill: true,
                    model: Some(
                        RankingModel::compile("sparse", &["sparse"], &Default::default()).unwrap(),
                    ),
                    export_passages: 1,
                    all_passages: true,
                    seed_document_passages: false,
                    document_combiner: MultiValueCombiner::Max,
                };
                searcher
                    .score_candidates(&candidates, &plan, None)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|s| s.result)
                    .collect()
            };
            result
                .sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            let all: Vec<_> = result
                .iter()
                .map(|r| (r.doc_id, r.score.to_bits()))
                .collect();
            result.truncate(10);
            times.push(serde_json::json!({"pass":pass,"query":qid,"retrieval_ms":retrieval_ms,"backfill_ms":total_ms-retrieval_ms,"total_ms":total_ms,"candidates":all.len()}));
            serde_json::to_writer(&mut hits, &serde_json::json!({"pass":pass,"query":qid,"top10":result.iter().map(|r|(r.doc_id,r.score.to_bits())).collect::<Vec<_>>(),"all":all})).unwrap();
            writeln!(hits).unwrap();
        }
    }
    hits.flush().unwrap();
    println!(
        "{}",
        serde_json::json!({"mode":mode,"depth":depth,"open_ms":open_ms,"usage_before":before,"usage_after":usage(),"times":times})
    );
}
