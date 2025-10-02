//! Process and per-index memory gauges, sampled periodically.
//!
//! Process RSS alone cannot say what holds memory: anonymous heap, mapped
//! index files and mlock-pinned metadata share it. These gauges split it so
//! an operator can attribute growth before the kernel OOM killer does:
//!
//! - `summa_memory_allocator_bytes{kind}`: jemalloc `allocated` (live
//!   objects), `active`, `resident`, `retained` and `metadata`.
//! - `summa_memory_process_bytes{kind}`: `/proc/self/status` `rss_anon`,
//!   `rss_file` and `locked` (Linux only).
//! - `summa_index_memory_bytes{index,kind}`: `indexing_buffer` (open
//!   segment builders), `primary_key` (dedup index), `reader_heap`
//!   (estimated segment-reader heap) and `pinned_metadata`.
//!
//! Sampling never takes the writer lock and never opens or reloads a reader.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use crate::registry::IndexRegistry;

const ALLOCATOR: &str = "summa_memory_allocator_bytes";
const PROCESS: &str = "summa_memory_process_bytes";
const INDEX: &str = "summa_index_memory_bytes";
const INDEX_KINDS: [&str; 4] = [
    "indexing_buffer",
    "primary_key",
    "reader_heap",
    "pinned_metadata",
];

pub fn spawn(
    registry: Arc<IndexRegistry>,
    interval: Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut reported: HashSet<String> = HashSet::new();
        loop {
            sample(&registry, &mut reported);
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = shutdown.changed() => return,
            }
        }
    })
}

fn sample(registry: &IndexRegistry, reported: &mut HashSet<String>) {
    allocator();
    process();
    let mut current = HashSet::new();
    for (name, index, probe) in registry.memory_snapshot() {
        let (heap, pinned) = index.loaded_reader().map_or((0, 0), |reader| {
            let searcher = reader.current_searcher();
            searcher
                .segment_readers()
                .iter()
                .map(|segment| segment.memory_stats())
                .fold((0u64, 0u64), |(heap, pinned), stats| {
                    (
                        heap.saturating_add(stats.estimated_heap_bytes() as u64),
                        pinned.saturating_add(stats.pinned_metadata_bytes),
                    )
                })
        });
        let values = [
            probe.indexing_buffer_bytes() as f64,
            probe.primary_key_bytes() as f64,
            heap as f64,
            pinned as f64,
        ];
        for (kind, value) in INDEX_KINDS.iter().zip(values) {
            metrics::gauge!(INDEX, "index" => name.clone(), "kind" => *kind).set(value);
        }
        current.insert(name);
    }
    // A closed or deleted index must not keep reporting its last value.
    for name in reported.difference(&current) {
        for kind in INDEX_KINDS {
            metrics::gauge!(INDEX, "index" => name.clone(), "kind" => kind).set(0.0);
        }
    }
    *reported = current;
}

#[cfg(not(target_env = "msvc"))]
fn allocator() {
    use tikv_jemalloc_ctl::{epoch, stats};
    if epoch::advance().is_err() {
        return;
    }
    let readings = [
        ("allocated", stats::allocated::read()),
        ("active", stats::active::read()),
        ("resident", stats::resident::read()),
        ("retained", stats::retained::read()),
        ("metadata", stats::metadata::read()),
    ];
    for (kind, value) in readings {
        if let Ok(bytes) = value {
            metrics::gauge!(ALLOCATOR, "kind" => kind).set(bytes as f64);
        }
    }
}

#[cfg(target_env = "msvc")]
fn allocator() {}

fn process() {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return;
    };
    for (key, kind) in [
        ("RssAnon:", "rss_anon"),
        ("RssFile:", "rss_file"),
        ("VmLck:", "locked"),
    ] {
        if let Some(kib) = status_kib(&status, key) {
            metrics::gauge!(PROCESS, "kind" => kind).set((kib * 1024) as f64);
        }
    }
}

/// The value of a `/proc/self/status` line such as `RssAnon:  1024 kB`.
fn status_kib(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_status_values_parse_in_kib() {
        let status = "Name:\tsumma-server\nVmLck:\t  20349796 kB\nRssAnon:\t26417600 kB\n";
        assert_eq!(status_kib(status, "VmLck:"), Some(20_349_796));
        assert_eq!(status_kib(status, "RssAnon:"), Some(26_417_600));
        assert_eq!(status_kib(status, "RssFile:"), None);
    }
}
