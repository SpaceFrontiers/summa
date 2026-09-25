//! Benchmark-only HTTP frontend over the existing core index/search APIs.
//! This does not measure the production gRPC server. See searchbench-comparison.md.

mod audit;
mod corpus;
mod diagnose;
mod dispatch;
mod query;
mod response;
#[cfg(feature = "query-diagnostics")]
mod timing;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use summa_core::directories::MmapDirectory;
use summa_core::query::{CountCollector, collect_segment};
use summa_core::tokenizer::TokenizerRegistry;
use summa_core::{Index, IndexConfig, Searcher};

struct App {
    searcher: Arc<Searcher<MmapDirectory>>,
    parser: summa_core::dsl::QueryLanguageParser,
    tokenizer: summa_core::tokenizer::BoxedTokenizer,
    tokenizer_name: String,
    body: summa_core::Field,
    id: summa_core::Field,
    admission: Arc<tokio::sync::Semaphore>,
    dispatch: dispatch::Dispatch,
    #[cfg(feature = "query-diagnostics")]
    timings: Arc<timing::Totals>,
}

impl App {
    fn execute(
        &self,
        envelope: query::Envelope,
        runtime: tokio::runtime::Handle,
        #[cfg(feature = "query-diagnostics")] timings: &mut timing::Trace,
    ) -> Result<response::SearchResponse<'_>> {
        let request = envelope.parse(&self.parser, self.body, &self.tokenizer)?;
        #[cfg(feature = "query-diagnostics")]
        timings.mark(1);
        let segment = &self.searcher.segment_readers()[0];
        if request.limit == 0 {
            let mut collector = CountCollector::new();
            runtime.block_on(collect_segment(
                segment,
                request.query.as_ref(),
                &mut collector,
            ))?;
            #[cfg(feature = "query-diagnostics")]
            timings.mark(2);
            let response = response::SearchResponse::Count {
                docs: [],
                found: collector.count(),
            };
            #[cfg(feature = "query-diagnostics")]
            timings.mark(3);
            return Ok(response);
        }
        let (hits, _) = self.searcher.search_with_offset_and_count_sync(
            request.query.as_ref(),
            request.limit,
            0,
        )?;
        #[cfg(feature = "query-diagnostics")]
        timings.mark(2);
        let response = self.project(&hits);
        #[cfg(feature = "query-diagnostics")]
        timings.mark(3);
        response
    }

    fn project(
        &self,
        hits: &[summa_core::query::SearchResult],
    ) -> Result<response::SearchResponse<'_>> {
        let column = self.searcher.segment_readers()[0]
            .fast_field(self.id.0)
            .context("missing id fast column")?;
        let mut docs = Vec::with_capacity(hits.len());
        for hit in hits {
            let id = column.get_text(hit.doc_id).context("missing corpus id")?;
            docs.push(response::Hit { id });
        }
        Ok(response::SearchResponse::Ranked { docs })
    }
}

fn failure(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": message.to_string()}))).into_response()
}

async fn search(State(app): State<Arc<App>>, Json(value): Json<Value>) -> Response {
    let envelope = match query::Envelope::from_value(value) {
        Ok(envelope) => envelope,
        Err(error) => return failure(StatusCode::BAD_REQUEST, error),
    };
    let permit = match app.admission.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return failure(StatusCode::SERVICE_UNAVAILABLE, "search admission full"),
    };
    let runtime = tokio::runtime::Handle::current();
    #[cfg(feature = "query-diagnostics")]
    let submitted = std::time::Instant::now();
    #[cfg(feature = "query-diagnostics")]
    let totals = app.timings.clone();
    let result = app
        .dispatch
        .run(move || {
            let _permit = permit;
            #[cfg(feature = "query-diagnostics")]
            let mut timings = timing::Trace::start(submitted);
            #[cfg(feature = "query-diagnostics")]
            let (response, work) = summa_core::search_diagnostics::capture_sync(|| {
                app.execute(envelope, runtime, &mut timings)
            });
            #[cfg(not(feature = "query-diagnostics"))]
            let response = app.execute(envelope, runtime);
            let value = response?;
            let bytes = serde_json::to_vec(&value).map_err(anyhow::Error::from)?;
            drop(value);
            #[cfg(feature = "query-diagnostics")]
            {
                timings.mark(4);
                timings.work(work);
                Ok::<_, anyhow::Error>((bytes, timings))
            }
            #[cfg(not(feature = "query-diagnostics"))]
            Ok::<_, anyhow::Error>(bytes)
        })
        .await;
    match result {
        Ok(Ok(encoded)) => {
            #[cfg(feature = "query-diagnostics")]
            let bytes = {
                totals.complete(encoded.1);
                encoded.0
            };
            #[cfg(not(feature = "query-diagnostics"))]
            let bytes = encoded;
            (
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                bytes,
            )
                .into_response()
        }
        Ok(Err(error)) => {
            #[cfg(feature = "query-diagnostics")]
            totals.error();
            failure(StatusCode::BAD_REQUEST, error)
        }
        Err(error) => {
            #[cfg(feature = "query-diagnostics")]
            totals.error();
            failure(StatusCode::INTERNAL_SERVER_ERROR, error)
        }
    }
}

async fn stats(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({
        "engine": "summa-benchmark-http", "body_tokenizer": app.tokenizer_name,
        "documents": app.searcher.num_docs(),
        "segments": app.searcher.segment_readers().iter().map(|reader| json!({
            "id": reader.meta().id.to_string(), "documents": reader.num_docs(),
            "body_reordered": reader.chunk_map(app.body).is_some(),
        })).collect::<Vec<_>>(),
    }))
}

async fn open_app(path: &Path, config: IndexConfig) -> Result<App> {
    let index = Index::open(MmapDirectory::new(path), config).await?;
    let searcher = index.reader().await?.searcher().await?;
    if searcher.num_segments() != 1
        || searcher.segment_readers()[0].num_docs() != searcher.num_docs()
    {
        bail!("requires one deletion-free segment");
    }
    let body = searcher
        .schema()
        .get_field("body")
        .context("missing body")?;
    let id = searcher.schema().get_field("id").context("missing id")?;
    let tokenizer_name = searcher
        .schema()
        .get_field_entry(body)
        .and_then(|entry| entry.tokenizer.as_deref())
        .unwrap_or("default")
        .to_owned();
    let tokenizer = TokenizerRegistry::new()
        .get(&tokenizer_name)
        .context("invalid body tokenizer")?;
    Ok(App {
        parser: searcher.query_parser(),
        searcher,
        tokenizer,
        tokenizer_name,
        body,
        id,
        admission: Arc::new(tokio::sync::Semaphore::new(64)),
        dispatch: dispatch::Dispatch::default(),
        #[cfg(feature = "query-diagnostics")]
        timings: Arc::new(timing::Totals::default()),
    })
}

async fn serve(
    path: &Path,
    port: u16,
    config: IndexConfig,
    dispatch: dispatch::Dispatch,
) -> Result<()> {
    let mut app = open_app(path, config).await?;
    app.dispatch = dispatch;
    let app = Arc::new(app);
    let router = Router::new();
    #[cfg(feature = "query-diagnostics")]
    let router = router.route(
        "/diagnostics",
        get(|State(app): State<Arc<App>>| async move { Json(app.timings.snapshot()) }),
    );
    let router = router
        .route("/search", post(search))
        .route("/stats", get(stats))
        .route("/health", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await?;
    eprintln!("benchmark HTTP listening on 127.0.0.1:{port}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn default_http_workers(available_cpus: usize) -> usize {
    available_cpus
        .div_ceil(8)
        .clamp(2, 8)
        .min(available_cpus.max(1))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        bail!(
            "usage: searchbench_http index|index-rgb|index-impacts|index-unicode|index-unicode-rgb|index-unicode-impacts INDEX CORPUS.jsonl [WORKERS] | serve INDEX PORT [WORKERS] [HTTP_WORKERS] [DISPATCH] | audit|diagnose INDEX QUERIES.jsonl [WORKERS]"
        );
    }
    let workers: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(6);
    if !(1..=64).contains(&workers) {
        bail!("workers must be 1..=64");
    }
    if args.len() > 7 || (args.len() >= 6 && args[1] != "serve") {
        bail!("HTTP_WORKERS and DISPATCH are accepted only by serve; unexpected extra arguments");
    }
    let available_cpus = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or_else(|error| {
            eprintln!("warning: cannot detect available CPUs ({error}); using one CPU");
            1
        });
    let http_workers: usize = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or_else(|| {
            if args[1] == "serve" {
                default_http_workers(available_cpus)
            } else {
                2
            }
        });
    if !(1..=64).contains(&http_workers) {
        bail!("HTTP_WORKERS must be 1..=64");
    }
    let dispatch: dispatch::Dispatch = args
        .get(6)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or_default();
    eprintln!(
        "available CPUs: {available_cpus}; search workers: {workers}; HTTP/runtime workers: {http_workers}; dispatch: {dispatch:?}"
    );
    let config = IndexConfig {
        posting_impact_bounds: matches!(
            args[1].as_str(),
            "index-impacts" | "index-unicode-impacts"
        ),
        num_threads: workers,
        num_indexing_threads: workers,
        max_indexing_memory_bytes: 2_000_000_000,
        ..Default::default()
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(http_workers)
        .max_blocking_threads(workers.max(4))
        .enable_all()
        .build()?;
    match args[1].as_str() {
        "diagnose" => runtime.block_on(diagnose::run(
            Path::new(&args[2]),
            Path::new(&args[3]),
            config,
        )),
        "audit" => runtime.block_on(audit::run(Path::new(&args[2]), Path::new(&args[3]), config)),
        "index"
        | "index-rgb"
        | "index-impacts"
        | "index-unicode"
        | "index-unicode-rgb"
        | "index-unicode-impacts" => runtime.block_on(corpus::build(
            Path::new(&args[2]),
            Path::new(&args[3]),
            config,
            matches!(args[1].as_str(), "index-rgb" | "index-unicode-rgb"),
            if args[1].starts_with("index-unicode") {
                "unicode_word"
            } else {
                corpus::BODY_TOKENIZER
            },
        )),
        "serve" => runtime.block_on(serve(
            Path::new(&args[2]),
            args[3].parse()?,
            config,
            dispatch,
        )),
        _ => bail!("unknown operation"),
    }
}

#[cfg(test)]
mod tests {
    use super::default_http_workers;

    #[test]
    fn http_default_scales_with_available_cpus_and_bounds_thread_count() {
        for (cpus, expected) in [
            (1, 1),
            (2, 2),
            (8, 2),
            (16, 2),
            (17, 3),
            (30, 4),
            (32, 4),
            (64, 8),
            (128, 8),
        ] {
            assert_eq!(default_http_workers(cpus), expected);
        }
    }
}
