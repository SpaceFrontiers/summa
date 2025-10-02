//! Replay a persisted sparse fixture without rebuilding it.
//! Usage: bmp_replay INDEX QUERIES_JSON PASSES; emits per-query timing/results.
use std::time::Instant;
use summa_core::{
    Index, IndexConfig,
    directories::MmapDirectory,
    query::{MultiValueCombiner, SparseVectorQuery},
};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "bmp_replay INDEX QUERIES_JSON PASSES");
    let passes: usize = args[3].parse().expect("positive pass count");
    let queries: Vec<Vec<(u32, f32)>> =
        serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let index = Index::open(MmapDirectory::new(&args[1]), IndexConfig {
            num_threads: 1, ..Default::default()
        }).await.unwrap();
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let field = reader.schema().get_field("vector").expect("vector field");
        for k in [10, 100] {
            for gamma in [0, 100] {
                for pass in 0..=passes {
                    for (qid, terms) in queries.iter().enumerate() {
                        let query = SparseVectorQuery::new(field, terms.clone())
                            .with_combiner(MultiValueCombiner::Max).with_lsp_gamma(gamma);
                        let start = Instant::now();
                        let hits = searcher.search(&query, k).await.unwrap();
                        let ns = start.elapsed().as_nanos();
                        if pass != 0 {
                            let values: Vec<_> = hits.iter().map(|hit| (hit.doc_id, hit.score.to_bits())).collect();
                            println!("{}", serde_json::json!({"k":k,"gamma":gamma,"pass":pass,"query":qid,"ns":ns,"hits":values}));
                        }
                    }
                }
            }
        }
    });
}
