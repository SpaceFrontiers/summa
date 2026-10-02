//! Actual-store hydration experiment using the canonical payload service and decoder.
mod measure;
mod payload_io;
mod real;
mod store;
use measure::{cpu, read_bytes, residency, usage};
use payload_io::{Backend, PayloadDirectory, Service};
use std::{collections::HashMap, io, os::fd::AsRawFd, path::Path, time::Instant};
use store::{config, hydrate};
use summa_core::{
    Document, SchemaBuilder, directories::MmapDirectory, dsl::Field, index::Index,
    query::DocAddress,
};
type E = Box<dyn std::error::Error>;
const DOCS: usize = 4096;
fn document(n: usize) -> Document {
    let mut state = (n as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    let mut body = Vec::with_capacity(32768);
    for _ in 0..32768 + n % 5 * 23 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        body.push(b' ' + (state % 95) as u8);
    }
    let mut doc = Document::new();
    doc.add_u64(Field(0), n as u64);
    doc.add_text(Field(1), String::from_utf8(body).unwrap());
    doc.add_text(Field(1), format!("second value {n}"));
    doc.add_dense_vector(Field(2), vec![1.0, 2.0, 3.0, n as f32]);
    doc
}
async fn create(root: &Path) -> Result<(), E> {
    assert!(!root.exists(), "private fixture must not exist");
    std::fs::create_dir_all(root)?;
    let mut schema = SchemaBuilder::default();
    schema.add_u64_field("id", true, true);
    schema.add_text_field("body", false, true);
    schema.add_dense_vector_field("vector", 4, false, true);
    let index = Index::create(MmapDirectory::new(root), schema.build(), config(0)).await?;
    let mut writer = index.writer();
    for n in 0..DOCS {
        writer.add_document(document(n))?;
        if (n + 1) % 1024 == 0 {
            writer.commit().await?;
        }
    }
    writer.shutdown().await?;
    std::fs::write(root.join(".summa-hydration-probe"), b"summa-hydration-v1\n")?;
    Ok(())
}
fn advise(root: &Path, cold: bool) -> io::Result<(u64, usize)> {
    let mut total = 0;
    let mut resident = 0;
    for entry in std::fs::read_dir(root)? {
        let p = entry?.path();
        if p.extension().is_none_or(|e| e != "store") {
            continue;
        }
        let f = std::fs::File::open(&p)?;
        let len = f.metadata()?.len();
        total += len;
        if cold {
            let e = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
            if e != 0 {
                return Err(io::Error::from_raw_os_error(e));
            }
        }
        let map = unsafe { memmap2::Mmap::map(&f)? };
        resident += residency(&map);
    }
    Ok((total, resident))
}
fn verify(docs: &[Document]) {
    for doc in docs {
        let n = doc.get_first(Field(0)).unwrap().as_u64().unwrap() as usize;
        assert_eq!(
            serde_json::to_value(doc).unwrap(),
            serde_json::to_value(document(n)).unwrap()
        );
    }
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), E> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(2).is_some_and(|s| s.starts_with("real-")) {
        return real::run(&args).await;
    }
    let root = Path::new(&args[1]);
    if args[2] == "create" {
        create(root).await?;
        println!(
            "{}",
            serde_json::json!({"created":true,"documents":DOCS,"store_bytes":advise(root,false)?.0})
        );
        return Ok(());
    }
    assert_eq!(
        std::fs::read(root.join(".summa-hydration-probe"))?,
        b"summa-hydration-v1\n",
        "run requires the private fixture created by this probe"
    );
    // Establish address identity outside all timed/cache-advised phases.
    let baseline = Index::open(PayloadDirectory::new(root, None, false), config(0)).await?;
    let baseline_searcher = baseline.reader().await?.searcher().await?;
    let mut expected_ids = HashMap::with_capacity(DOCS);
    for segment in baseline_searcher.segment_readers() {
        for doc_id in 0..segment.num_docs() {
            let address = DocAddress::new(segment.meta().id, doc_id);
            let doc = baseline_searcher.get_document(&address).await?.unwrap();
            let id = doc.get_first(Field(0)).unwrap().as_u64().unwrap();
            verify(std::slice::from_ref(&doc));
            assert!(expected_ids.insert(address, id).is_none());
        }
    }
    assert_eq!(expected_ids.len(), DOCS);
    drop(baseline_searcher);
    drop(baseline);
    let idle_bytes: usize = args.get(4).map_or(Ok(0), |s| s.parse())?;
    let pool = Service::with_buffer_budget(Backend::Pool, idle_bytes)?;
    let ring = Service::with_buffer_budget(Backend::IoUring, idle_bytes)?;
    let callers: usize = args.get(3).map_or(Ok(1), |s| s.parse())?;
    assert!([1, 4].contains(&callers));
    for pass in 0..2 {
        let mut methods = vec!["mmap", "fs-demand", "fs-batch", "pool", "uring"];
        if pass == 1 {
            methods.reverse();
        }
        for cache in [0, 16 * 1024 * 1024] {
            for cold in [false, true] {
                for method in &methods {
                    let (store_bytes, resident_before_open) = advise(root, cold)?;
                    let service = match *method {
                        "pool" => Some(pool.clone()),
                        "uring" => Some(ring.clone()),
                        _ => None,
                    };
                    let dir =
                        PayloadDirectory::new(root, service.clone(), method.starts_with("fs-"));
                    let index = Index::open(dir, config(cache)).await?;
                    let reader = index.reader().await?;
                    let searcher = reader.searcher().await?;
                    assert_eq!(searcher.num_docs(), DOCS as u32);
                    assert_eq!(searcher.num_segments(), 4);
                    let all: Vec<_> = searcher
                        .segment_readers()
                        .iter()
                        .flat_map(|s| {
                            (0..s.num_docs()).map(move |doc| DocAddress::new(s.meta().id, doc))
                        })
                        .collect();
                    let working = if cache == 0 { 1024 } else { 256 };
                    let addresses: Vec<_> = (0..1024)
                        .map(|i| all[((i % working) * 251) % all.len()].clone())
                        .collect();
                    let batch = *method != "fs-demand";
                    if !cold {
                        for chunk in addresses.chunks(32) {
                            verify(&hydrate(&searcher, chunk, batch).await?);
                        }
                    }
                    let residency_at_start = advise(root, false)?.1;
                    let u0 = usage();
                    let bytes0 = read_bytes();
                    let stats0 = service.as_ref().map(|s| s.stats());
                    let mut times = vec![];
                    let mut hydration_cpu = 0.0;
                    let mut waves = vec![];
                    for wave in addresses.chunks(32 * callers) {
                        let before = usage();
                        let start = Instant::now();
                        let pending = wave.chunks(32).map(|chunk| {
                            let searcher = &searcher;
                            async move {
                                let start = Instant::now();
                                let docs = hydrate(searcher, chunk, batch).await;
                                (chunk, docs, start.elapsed().as_nanos() as u64)
                            }
                        });
                        let results = futures::future::join_all(pending).await;
                        waves.push(start.elapsed().as_nanos() as u64);
                        hydration_cpu += cpu(&usage()) - cpu(&before);
                        for (chunk, docs, ns) in results {
                            let docs = docs?;
                            times.push(ns);
                            assert_eq!(docs.len(), chunk.len());
                            for (doc, address) in docs.iter().zip(chunk) {
                                assert_eq!(
                                    doc.get_first(Field(0)).unwrap().as_u64().unwrap(),
                                    expected_ids[address]
                                );
                            }
                            verify(&docs);
                        }
                    }
                    let u1 = usage();
                    let bytes1 = read_bytes();
                    println!(
                        "{}",
                        serde_json::json!({"pass":pass,"idle_buffer_budget":idle_bytes,"callers":callers,"wave_ns":waves,"method":method,"cache_budget_bytes":cache,"cold_advised":cold,"working_documents":working,"requested_documents":addresses.len(),"batch_documents":32,"store_bytes":store_bytes,"resident_before_open":resident_before_open,"resident_at_timing_start":residency_at_start,"samples_ns":times,"hydration_cpu_seconds":hydration_cpu,"phase_cpu_including_verification":cpu(&u1)-cpu(&u0),"kernel_read_bytes":bytes1-bytes0,"minor_faults":u1.ru_minflt-u0.ru_minflt,"major_faults":u1.ru_majflt-u0.ru_majflt,"cumulative_peak_rss_kib":u1.ru_maxrss,"stats_before":stats0,"stats_after":service.as_ref().map(|s|s.stats())})
                    );
                    drop(searcher);
                    drop(index);
                }
            }
        }
    }
    pool.shutdown().await?;
    ring.shutdown().await?;
    println!(
        "{}",
        serde_json::json!({"complete":true,"pool":pool.stats(),"uring":ring.stats()})
    );
    Ok(())
}
