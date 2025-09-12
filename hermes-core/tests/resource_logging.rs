//! Separate test process so the logger and shared-resource state start fresh.
#![cfg(feature = "native")]

use hermes_core::directories::RamDirectory;
use hermes_core::{Index, IndexConfig, IndexMetadata, SchemaBuilder};
use log::{Level, LevelFilter, Log, Metadata, Record};
use std::sync::Mutex;

struct ResourceLogger(Mutex<Vec<(Level, String)>>);

impl Log for ResourceLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target() == "hermes_core::index"
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            let message = record.args().to_string();
            if message.contains("process-wide") {
                self.0.lock().unwrap().push((record.level(), message));
            }
        }
    }

    fn flush(&self) {}
}

static LOGGER: ResourceLogger = ResourceLogger(Mutex::new(Vec::new()));

#[tokio::test]
async fn reopening_indexes_announces_shared_resources_once_and_keeps_debug_diagnostics() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(LevelFilter::Debug);

    let directory = RamDirectory::new();
    IndexMetadata::new(SchemaBuilder::default().build())
        .save(&directory)
        .await
        .unwrap();
    let config = IndexConfig {
        num_threads: 1,
        store_cache_budget_bytes: 1024,
        bmp_io_concurrency: 2,
        ..Default::default()
    };

    for _ in 0..3 {
        let first = Index::open(directory.clone(), config.clone())
            .await
            .unwrap();
        // An overlapping open reuses the live resources without logging again.
        let second = Index::open(directory.clone(), config.clone())
            .await
            .unwrap();
        assert_eq!(first.num_docs().await.unwrap(), 0);
        assert_eq!(second.num_docs().await.unwrap(), 0);
        drop(first);
        drop(second);
    }

    let records = LOGGER.0.lock().unwrap();
    for prefix in [
        #[cfg(feature = "sync")]
        "[search] process-wide CPU pool:",
        "[store_cache] process-wide budget=",
        "[bmp] process-wide random-I/O concurrency=",
    ] {
        let levels: Vec<_> = records
            .iter()
            .filter(|(_, message)| message.starts_with(prefix))
            .map(|(level, _)| *level)
            .collect();
        assert_eq!(
            levels,
            [Level::Info, Level::Debug, Level::Debug],
            "{prefix}"
        );
    }
}
