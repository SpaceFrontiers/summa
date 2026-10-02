//! Hybrid residency experiment using canonical readers, scorers and payload service.
#[path = "sparse/load.rs"]
mod load;
#[allow(dead_code)] // This probe uses process counters; the other probes also inspect mappings.
mod measure;
mod payload_io;
mod store;
use payload_io::{Backend, PayloadDirectory, Service};
use std::{io::Write, path::Path, time::Instant};
use summa_core::{
    Document, SchemaBuilder,
    directories::MmapDirectory,
    index::{Index, IndexConfig},
    query::{DocAddress, SearchResult, SparseVectorQuery},
    structures::{SparseFormat, SparseVectorConfig, WeightQuantization},
};
type E = Box<dyn std::error::Error + Send + Sync>;
const MARKER: &[u8] = b"summa-hybrid-io-v1\n";
const DOCS: usize = 262144;
const DIMS: u32 = 4096;
const QUERIES: usize = 32;

fn config() -> IndexConfig {
    IndexConfig {
        num_threads: 4,
        ..store::config(0)
    }
}
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
async fn create(root: &Path) -> Result<(), E> {
    assert!(!root.exists(), "create requires a new private fixture");
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field("body", false, true);
    let fields: Vec<_> = [SparseFormat::MaxScore, SparseFormat::Bmp]
        .into_iter()
        .map(|format| {
            schema.add_sparse_vector_field_with_config(
                if format == SparseFormat::MaxScore {
                    "maxscore"
                } else {
                    "bmp"
                },
                true,
                false,
                SparseVectorConfig {
                    format,
                    dims: Some(DIMS),
                    weight_quantization: WeightQuantization::UInt8,
                    ..Default::default()
                },
            )
        })
        .collect();
    let index = Index::create(MmapDirectory::new(root), schema.build(), config()).await?;
    let mut writer = index.writer();
    for n in 0..DOCS {
        let mut state = (n as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
        let mut terms: Vec<_> = (0..64)
            .map(|_| {
                (
                    (next(&mut state) % u64::from(DIMS)) as u32,
                    (next(&mut state) % 1000 + 1) as f32 / 1000.0,
                )
            })
            .collect();
        terms.sort_unstable_by_key(|term| term.0);
        terms.dedup_by_key(|term| term.0);
        let text: String = (0..2048)
            .map(|_| char::from(b' ' + (next(&mut state) % 95) as u8))
            .collect();
        let mut doc = Document::new();
        doc.add_text(body, text);
        for &field in &fields {
            doc.add_sparse_vector(field, terms.clone());
        }
        let deadline = Instant::now() + std::time::Duration::from_secs(120);
        loop {
            match writer.add_document(doc.clone()) {
                Ok(()) => break,
                Err(summa_core::Error::QueueFull) if Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    writer.commit().await?;
    writer.shutdown().await?;
    std::fs::write(root.join(".summa-hybrid-io-probe"), MARKER)?;
    println!("{}", serde_json::json!({"complete":true,"documents":DOCS}));
    Ok(())
}
fn hits(hits: &[SearchResult]) -> serde_json::Value {
    serde_json::json!(
        hits.iter()
            .map(|h| (h.segment_id.to_string(), h.doc_id, h.score.to_bits()))
            .collect::<Vec<_>>()
    )
}
fn memory(searcher: &summa_core::index::Searcher<PayloadDirectory>) -> serde_json::Value {
    let segments: Vec<_> = searcher.segment_readers().iter().map(|s| {
        let m=s.memory_stats();
        serde_json::json!({"documents":m.num_docs,"heap_bytes":m.estimated_heap_bytes(),"intended":m.pin_intended_bytes,"pinned":m.pinned_metadata_bytes,"sparse_file_bytes":m.sparse_file_backed_bytes})
    }).collect();
    serde_json::json!({"segments":segments,"status":std::fs::read_to_string("/proc/self/status").unwrap(),"smaps_rollup":std::fs::read_to_string("/proc/self/smaps_rollup").unwrap()})
}
struct QueryOutput {
    found: Vec<SearchResult>,
    docs: Vec<Document>,
    search_ns: u64,
    ns: u64,
    response_at: Instant,
}
async fn execute_query(
    searcher: &summa_core::index::Searcher<PayloadDirectory>,
    query: &SparseVectorQuery,
) -> Result<QueryOutput, E> {
    let started = Instant::now();
    let found = searcher.search(query, 32).await?;
    let search_ns = started.elapsed().as_nanos() as u64;
    let addresses: Vec<_> = found
        .iter()
        .map(|h| DocAddress::new(h.segment_id, h.doc_id))
        .collect();
    let docs = store::hydrate(searcher, &addresses, true).await?;
    let response_at = Instant::now();
    Ok(QueryOutput {
        found,
        docs,
        search_ns,
        ns: response_at.duration_since(started).as_nanos() as u64,
        response_at,
    })
}
fn actual(output: &QueryOutput) -> Result<serde_json::Value, E> {
    Ok(
        serde_json::json!({"hits":hits(&output.found),"digest":blake3::hash(&serde_json::to_vec(&output.docs)?).to_hex().to_string()}),
    )
}
struct VerifiedQuery {
    index: usize,
    response_at: Instant,
    search_ns: u64,
    ns: u64,
    documents: usize,
    verification_ns: u64,
}
async fn run(args: &[String]) -> Result<(), E> {
    let concurrency = setting("SUMMA_PROBE_CONCURRENCY", 1, 1, 32)?;
    let passes = setting("SUMMA_PROBE_PASSES", 4, 1, 128)?;
    let command = args
        .get(2)
        .ok_or("expected fixture and create/oracle/run/load command")?;
    if !["create", "oracle", "run", "load"].contains(&command.as_str()) {
        return Err("invalid probe command".into());
    }
    let arrival_rate = if command == "load" {
        setting("SUMMA_PROBE_RATE", 500, 1, 10_000)?
    } else {
        0
    };
    let load_config = if command == "load" {
        Some(load::Config::new(
            arrival_rate,
            setting("SUMMA_PROBE_REQUESTS", 5000, 1, 65_536)?,
            concurrency,
        )?)
    } else {
        None
    };
    if (command == "oracle" && args.len() < 4)
        || (matches!(command.as_str(), "run" | "load") && args.len() < 6)
    {
        return Err("expected oracle path, backend and field for run/load".into());
    }
    let root = Path::new(&args[1]);
    if args[2] == "create" {
        return create(root).await;
    }
    assert_eq!(std::fs::read(root.join(".summa-hybrid-io-probe"))?, MARKER);
    let oracle = Path::new(&args[3]);
    let creating = args[2] == "oracle";
    let method = if creating { "mmap" } else { &args[4] };
    let service = match method {
        "mmap" => None,
        "pool" => Some(Service::new(Backend::Pool)?),
        "uring" => Some(Service::new(Backend::IoUring)?),
        _ => return Err("invalid backend".into()),
    };
    let sparse_reads = match args.get(6).map(String::as_str).unwrap_or("sparse") {
        "sparse" => true,
        "store" => false,
        _ => return Err("read scope must be sparse or store".into()),
    };
    let mut directory = PayloadDirectory::new(root, service.clone(), false);
    if sparse_reads && let Some(service) = &service {
        directory = directory.with_sparse_payload_reads(service.clone());
    }
    let index = Index::open(directory, config()).await?;
    let reader = index.reader().await?;
    let searcher = reader.searcher().await?;
    let before = memory(&searcher);
    let pin = summa_core::segment::pin::pin_policy();
    if pin.is_enabled() {
        let intended: u64 = searcher
            .segment_readers()
            .iter()
            .map(|s| s.memory_stats().pin_intended_bytes)
            .sum();
        let pinned: u64 = searcher
            .segment_readers()
            .iter()
            .map(|s| s.memory_stats().pinned_metadata_bytes)
            .sum();
        assert!(
            intended > 0 && intended == pinned,
            "requested metadata pinning must fully succeed"
        );
        assert!(
            std::fs::read_to_string("/proc/self/status")?
                .lines()
                .any(|line| line.starts_with("VmLck:")
                    && line
                        .split_whitespace()
                        .nth(1)
                        .unwrap()
                        .parse::<u64>()
                        .unwrap()
                        > 0)
        );
    }
    let fields = if creating || args[5] == "mixed" {
        vec!["maxscore", "bmp"]
    } else {
        vec![args[5].as_str()]
    };
    let expected: serde_json::Value = if creating {
        serde_json::json!({})
    } else {
        serde_json::from_slice(&std::fs::read(oracle)?)?
    };
    let expected = std::sync::Arc::new(expected);
    let mut output = serde_json::Map::new();
    let stats0 = service.as_ref().map(|s| s.stats());
    let usage0 = measure::usage();
    let bytes0 = measure::read_bytes();
    let mut cpu = 0.0;
    let mut wall = 0.0;
    let mut samples = Vec::new();
    let mut phases = Vec::new();
    let mut queries = Vec::new();
    for q in 0..QUERIES {
        for &field_name in &fields {
            let field = searcher
                .schema()
                .get_field(field_name)
                .ok_or("missing sparse field")?;
            let mut state = (q as u64 + 1).wrapping_mul(0x517cc1b727220a95);
            let mut terms: Vec<_> = (0..8)
                .map(|_| {
                    (
                        (next(&mut state) % u64::from(DIMS)) as u32,
                        (next(&mut state) % 100 + 1) as f32 / 100.0,
                    )
                })
                .collect();
            terms.sort_unstable_by_key(|term| term.0);
            terms.dedup_by_key(|term| term.0);
            queries.push((
                field_name.to_owned(),
                q,
                SparseVectorQuery::new(field, terms),
            ));
        }
    }
    let queries = std::sync::Arc::new(queries);
    let mut load_report = None;
    if let Some(config) = load_config {
        let cpu0 = measure::cpu(&measure::usage());
        let run = load::run(config, |sequence| {
            let searcher = searcher.clone();
            let queries = queries.clone();
            let expected = expected.clone();
            async move {
                let i = sequence % queries.len();
                let response = execute_query(&searcher, &queries[i].2).await?;
                let verification = Instant::now();
                let (field, q, _) = &queries[i];
                if actual(&response)? != expected[field][*q] {
                    return Err::<_, E>(format!("oracle mismatch for {field}/{q}").into());
                }
                // Only compact observations survive verification, never documents.
                Ok(VerifiedQuery {
                    index: i,
                    response_at: response.response_at,
                    search_ns: response.search_ns,
                    ns: response.ns,
                    documents: response.docs.len(),
                    verification_ns: verification.elapsed().as_nanos() as u64,
                })
            }
        })
        .await?;
        cpu = measure::cpu(&measure::usage()) - cpu0;
        wall = run.elapsed.as_secs_f64();
        for row in run.accepted {
            let VerifiedQuery {
                index: i,
                response_at,
                search_ns,
                ns,
                documents,
                verification_ns,
            } = row.output;
            let (field, q, _) = &queries[i];
            let response_ns = response_at.duration_since(run.origin).as_nanos() as u64;
            samples.push(serde_json::json!({"sequence":row.sequence,"field":field,"query":q,
                "ns":ns,"search_ns":search_ns,"documents":documents,"verification_ns":verification_ns,
                "scheduled_ns":row.scheduled_ns,"admitted_ns":row.admitted_ns,"started_ns":row.started_ns,
                "response_ns":response_ns,"completed_ns":row.completed_ns,
                "latency_ns":response_ns-row.scheduled_ns}));
        }
        let rejected: Vec<_> = run.rejected.into_iter().map(|r| serde_json::json!({"sequence":r.sequence,"scheduled_ns":r.scheduled_ns,"observed_ns":r.observed_ns})).collect();
        load_report = Some(
            serde_json::json!({"rate":arrival_rate,"offered":samples.len()+rejected.len(),"accepted":samples.len(),"rejected":rejected,"max_in_flight":run.max_in_flight,"arrival_window_seconds":run.arrival_window.as_secs_f64(),"cpu_includes_verification":true}),
        );
    } else {
        for pass in 0..if creating { 1 } else { passes } {
            let mut phase_cpu = 0.0;
            let mut phase_wall = 0.0;
            for start in (0..queries.len()).step_by(concurrency) {
                let mut jobs = tokio::task::JoinSet::new();
                let u0 = measure::usage();
                let batch_start = Instant::now();
                for i in start..(start + concurrency).min(queries.len()) {
                    let searcher = searcher.clone();
                    let queries = queries.clone();
                    jobs.spawn(async move {
                        Ok::<_, E>((i, execute_query(&searcher, &queries[i].2).await?))
                    });
                }
                let mut completed = Vec::with_capacity(concurrency);
                while let Some(result) = jobs.join_next().await {
                    completed.push(result??);
                }
                let batch_wall = batch_start.elapsed().as_secs_f64();
                let batch_cpu = measure::cpu(&measure::usage()) - measure::cpu(&u0);
                wall += batch_wall;
                cpu += batch_cpu;
                phase_wall += batch_wall;
                phase_cpu += batch_cpu;
                // No digest/score verification or overlapping CPU intervals in timing.
                completed.sort_unstable_by_key(|row| row.0);
                for (i, response) in completed {
                    let QueryOutput { search_ns, ns, .. } = &response;
                    let (field, q, _) = &queries[i];
                    let actual = actual(&response)?;
                    if creating {
                        output
                            .entry(field.clone())
                            .or_insert_with(|| serde_json::json!([]))
                            .as_array_mut()
                            .unwrap()
                            .push(actual);
                    } else {
                        assert_eq!(actual, expected[field][*q]);
                    }
                    samples.push(serde_json::json!({"pass":pass,"field":field,"query":q,
                    "ns":ns,"search_ns":search_ns,"documents":response.docs.len()}));
                }
            }
            phases.push(serde_json::json!({"pass":pass,"queries":queries.len(),
            "wall_seconds":phase_wall,"cpu_seconds":phase_cpu}));
        }
    }
    if creating {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(oracle)?;
        file.write_all(&serde_json::to_vec(&output)?)?;
        file.sync_all()?;
    }
    let usage1 = measure::usage();
    if let Some(s) = &service {
        s.shutdown().await?;
    }
    println!(
        "{}",
        serde_json::json!({"complete":true,"method":method,"sparse_reads":sparse_reads,"pin_budget_bytes":pin.budget_bytes,"pin_mode":format!("{:?}",pin.mode),"samples":samples,"operation_cpu_seconds":cpu,"phases":phases,"operation_wall_seconds":wall,"concurrency":concurrency,"runtime_threads":setting("SUMMA_PROBE_RUNTIME_THREADS",0,0,32)?,"voluntary_switches":usage1.ru_nvcsw-usage0.ru_nvcsw,"involuntary_switches":usage1.ru_nivcsw-usage0.ru_nivcsw,"kernel_read_bytes":measure::read_bytes()-bytes0,"minor_faults":usage1.ru_minflt-usage0.ru_minflt,"major_faults":usage1.ru_majflt-usage0.ru_majflt,"peak_rss_kib":usage1.ru_maxrss,"memory_before":before,"memory_after":memory(&searcher),"stats_before":stats0,"stats_after":service.as_ref().map(|s|s.stats()),"load":load_report})
    );
    Ok(())
}
fn setting(name: &str, default: usize, min: usize, max: usize) -> Result<usize, E> {
    let value = match std::env::var(name) {
        Ok(value) => value.parse()?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(error.into()),
    };
    if !(min..=max).contains(&value) {
        return Err(format!("{name} must be in {min}..={max}").into());
    }
    Ok(value)
}
fn main() -> Result<(), E> {
    let threads = setting("SUMMA_PROBE_RUNTIME_THREADS", 0, 0, 32)?;
    let mut runtime = if threads == 0 {
        tokio::runtime::Builder::new_current_thread()
    } else {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.worker_threads(threads);
        builder
    };
    runtime
        .enable_all()
        .build()?
        .block_on(run(&std::env::args().collect::<Vec<_>>()))
}
