//! Real-corpus diagnostic, sharing the existing hydration service and core decoder.
use super::{E, config, hydrate};
use crate::{
    measure,
    payload_io::{Backend, PayloadDirectory, Service},
};
use std::{io::Write, path::Path, sync::Arc, time::Instant};
use summa_core::{Document, Query, QueryLanguageParser, index::Index, query::DocAddress};

fn digest(docs: &[Document]) -> Result<String, E> {
    Ok(blake3::hash(&serde_json::to_vec(docs)?)
        .to_hex()
        .to_string())
}

fn memory(searcher: &summa_core::index::Searcher<PayloadDirectory>) -> serde_json::Value {
    serde_json::json!({
        "smaps_rollup": std::fs::read_to_string("/proc/self/smaps_rollup").unwrap(),
        "smaps": std::fs::read_to_string("/proc/self/smaps").unwrap(),
        "segments": searcher.segment_readers().iter().map(|s| {
            let m=s.memory_stats(); let d=s.term_dict_stats();
            serde_json::json!({"documents":m.num_docs,"estimated_heap_bytes":m.estimated_heap_bytes(),"dictionary_cache_bytes":m.term_dict_cache_bytes,"store_cache_bytes":m.store_cache_bytes,"row_stats_heap_bytes":m.row_stats_heap_bytes,"fast_field_metadata_heap_bytes":m.fast_field_metadata_heap_bytes,"pinned_metadata_bytes":m.pinned_metadata_bytes,"pin_intended_bytes":m.pin_intended_bytes,"dictionary_blocks":d.num_blocks,"dictionary_entries":d.num_entries,"dictionary_cache_bypasses":d.cache_insert_bypasses})
        }).collect::<Vec<_>>()
    })
}

pub async fn run(args: &[String]) -> Result<(), E> {
    let root = Path::new(&args[1]);
    assert_eq!(
        std::fs::read(root.join(".summa-real-corpus-probe"))?,
        b"summa-real-corpus-v1\n"
    );
    let oracle = Path::new(&args[3]);
    let creating = args[2] == "real-oracle";
    let method = if creating { "mmap" } else { &args[4] };
    let workload = if creating { "oracle" } else { &args[5] };
    assert!(["mmap", "pool", "uring"].contains(&method));
    let idle_bytes: usize = if creating {
        0
    } else {
        args.get(6).map_or(Ok(0), |s| s.parse())?
    };
    let service = match method {
        "pool" => Some(Service::with_buffer_budget(Backend::Pool, idle_bytes)?),
        "uring" => Some(Service::with_buffer_budget(Backend::IoUring, idle_bytes)?),
        _ => None,
    };
    let index = Index::open(
        PayloadDirectory::new(root, service.clone(), false),
        config(0),
    )
    .await?;
    let reader = index.reader().await?;
    let searcher = reader.searcher().await?;
    let field = searcher
        .schema()
        .get_field("body")
        .ok_or("missing stored body field")?;
    let parser = QueryLanguageParser::new(
        searcher.schema_arc(),
        vec![field],
        Arc::new(summa_core::tokenizer::TokenizerRegistry::new()),
    );
    let mut manifest = if creating {
        let terms: Vec<String> = serde_json::from_slice(&std::fs::read(&args[4])?)?;
        assert!(!terms.is_empty() && terms.len() <= 128);
        let count: usize = args[5].parse()?;
        assert!((32..=65536).contains(&count) && count.is_multiple_of(32));
        assert_eq!(
            searcher.num_segments(),
            1,
            "this diagnostic fixes one immutable segment"
        );
        let segment = &searcher.segment_readers()[0];
        let addresses: Vec<_> = (0..count)
            .map(|i| {
                DocAddress::new(
                    segment.meta().id,
                    ((i as u64 * 15485863) % u64::from(segment.num_docs())) as u32,
                )
            })
            .collect();
        let mut traces = Vec::new();
        for chunk in addresses.chunks(32) {
            let docs = hydrate(&searcher, chunk, true).await?;
            traces.push(serde_json::json!({"addresses":chunk,"digest":digest(&docs)?}));
        }
        serde_json::json!({"version":1,"documents":searcher.num_docs(),"traces":traces,"terms":terms,"queries":[]})
    } else {
        serde_json::from_slice(&std::fs::read(oracle)?)?
    };
    assert_eq!(manifest["version"], 1);
    assert_eq!(manifest["documents"], searcher.num_docs());
    let terms: Vec<String> = serde_json::from_value(manifest["terms"].clone())?;
    assert!(terms.len() <= 128);
    let queries: Vec<Box<dyn Query>> = terms
        .iter()
        .map(|q| parser.parse_strict(q))
        .collect::<Result<_, _>>()?;
    if creating {
        let mut expected = Vec::new();
        for query in &queries {
            let hits = searcher.search(query.as_ref(), 32).await?;
            let addresses: Vec<_> = hits
                .iter()
                .map(|h| DocAddress::new(h.segment_id, h.doc_id))
                .collect();
            assert!(!addresses.is_empty());
            let docs = hydrate(&searcher, &addresses, true).await?;
            expected.push(serde_json::json!({"addresses":addresses,"score_bits":hits.iter().map(|h|h.score.to_bits()).collect::<Vec<_>>(),"digest":digest(&docs)?}));
        }
        manifest["queries"] = serde_json::Value::Array(expected);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(oracle)?;
        file.write_all(&serde_json::to_vec(&manifest)?)?;
        file.sync_all()?;
        println!(
            "{}",
            serde_json::json!({"oracle_complete":true,"traces":manifest["traces"].as_array().unwrap().len(),"queries":queries.len()})
        );
        return Ok(());
    }
    assert!(["trace", "query", "retrieval"].contains(&workload));
    let traces = manifest["traces"].as_array().ok_or("invalid traces")?;
    assert!(traces.len() <= 2048);
    let trace_addresses: Vec<Vec<DocAddress>> = traces
        .iter()
        .map(|v| serde_json::from_value(v["addresses"].clone()))
        .collect::<Result<_, _>>()?;
    let query_addresses: Vec<Vec<DocAddress>> = manifest["queries"]
        .as_array()
        .ok_or("invalid query oracle")?
        .iter()
        .map(|v| serde_json::from_value(v["addresses"].clone()))
        .collect::<Result<_, _>>()?;
    assert_eq!(query_addresses.len(), queries.len());
    let before = memory(&searcher);
    let stats0 = service.as_ref().map(|s| s.stats());
    let usage0 = measure::usage();
    let bytes0 = measure::read_bytes();
    let mut samples = Vec::new();
    let mut waves = Vec::new();
    let mut cpu = 0.0;
    let mut verified = 0usize;
    let repetitions = if workload == "trace" { 2 } else { 20 };
    for pass in 0..repetitions {
        let requests = if workload == "trace" {
            traces.len()
        } else {
            queries.len()
        };
        for wave in (0..requests).collect::<Vec<_>>().chunks(4) {
            let u0 = measure::usage();
            let start = Instant::now();
            let pending = wave.iter().map(|&position| {
                let query = queries.get(position);
                let searcher = &searcher;
                let addresses = if workload == "trace" {
                    &trace_addresses[position]
                } else {
                    &query_addresses[position]
                };
                async move {
                    let start = Instant::now();
                    let hits = if workload == "trace" {
                        None
                    } else {
                        Some(searcher.search(query.unwrap().as_ref(), 32).await?)
                    };
                    let actual = hits.as_ref().map(|hs| {
                        hs.iter()
                            .map(|h| DocAddress::new(h.segment_id, h.doc_id))
                            .collect::<Vec<_>>()
                    });
                    let docs = if workload == "retrieval" {
                        None
                    } else {
                        Some(
                            hydrate(
                                searcher,
                                actual.as_deref().unwrap_or(addresses.as_slice()),
                                true,
                            )
                            .await?,
                        )
                    };
                    let elapsed = start.elapsed().as_nanos() as u64;
                    Ok::<_, E>((position, addresses, actual, hits, docs, elapsed))
                }
            });
            let results = futures::future::join_all(pending).await;
            let elapsed = start.elapsed().as_nanos() as u64;
            cpu += measure::cpu(&measure::usage()) - measure::cpu(&u0);
            waves.push(serde_json::json!({"pass":pass,"ns":elapsed}));
            for result in results {
                let (position, addresses, actual, hits, docs, ns) = result?;
                let expected = if workload == "trace" {
                    &traces[position]
                } else {
                    &manifest["queries"][position]
                };
                if let Some(actual) = actual {
                    assert_eq!(&actual, addresses);
                }
                if let Some(hits) = hits {
                    assert_eq!(
                        serde_json::json!(
                            hits.iter().map(|h| h.score.to_bits()).collect::<Vec<_>>()
                        ),
                        expected["score_bits"]
                    );
                }
                if let Some(docs) = docs {
                    assert_eq!(digest(&docs)?, expected["digest"].as_str().unwrap());
                }
                verified += addresses.len();
                samples.push(serde_json::json!({"pass":pass,"ns":ns}));
            }
        }
    }
    let usage1 = measure::usage();
    println!(
        "{}",
        serde_json::json!({"complete":true,"idle_buffer_budget":idle_bytes,"method":method,"workload":workload,"callers":4,"batch_documents":32,"cache_budget_bytes":0,"verified_documents":verified,"samples":samples,"waves":waves,"operation_cpu_seconds":cpu,"cpu_including_verification":measure::cpu(&usage1)-measure::cpu(&usage0),"kernel_read_bytes":measure::read_bytes()-bytes0,"minor_faults":usage1.ru_minflt-usage0.ru_minflt,"major_faults":usage1.ru_majflt-usage0.ru_majflt,"peak_rss_kib":usage1.ru_maxrss,"memory_before":before,"memory_after":memory(&searcher),"stats_before":stats0,"stats_after":service.as_ref().map(|s|s.stats())})
    );
    if let Some(service) = service {
        service.shutdown().await?;
    }
    Ok(())
}
