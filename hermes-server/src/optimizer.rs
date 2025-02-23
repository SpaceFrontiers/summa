//! Background segment optimizer — reorders unreordered segments via BP.
//!
//! Runs as a set of tokio tasks bounded by a semaphore. Periodically scans
//! all indexes for segments that haven't been reordered and applies Recursive
//! Graph Bisection (BP) to improve BMP block clustering.
//!
//! Only indexes with `reorder` fields in their schema are considered.

use std::sync::Arc;
use std::time::Duration;

use log::{debug, info, warn};
use tokio::sync::Semaphore;

use crate::registry::IndexRegistry;

/// Background optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Number of concurrent optimizer threads (0 = disabled).
    pub threads: usize,
    /// Interval between scans for unreordered segments.
    pub scan_interval: Duration,
}

/// Spawn the background optimizer loop.
///
/// Returns a `JoinHandle` that runs until the server shuts down.
/// When `threads == 0`, no optimizer is started.
pub fn spawn_optimizer(
    registry: Arc<IndexRegistry>,
    config: OptimizerConfig,
) -> Option<tokio::task::JoinHandle<()>> {
    if config.threads == 0 {
        return None;
    }

    info!(
        "Starting background optimizer: {} threads, {:.0}s scan interval",
        config.threads,
        config.scan_interval.as_secs_f64(),
    );

    let semaphore = Arc::new(Semaphore::new(config.threads));

    // Bounded rayon pool for BP computation — prevents optimizer from
    // saturating all CPU cores. Shared across all concurrent reorder tasks.
    let rayon_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .thread_name(|idx| format!("optimizer-bp-{}", idx))
            .build()
            .expect("failed to create optimizer rayon pool"),
    );

    Some(tokio::spawn(async move {
        optimizer_loop(registry, semaphore, rayon_pool, config.scan_interval).await;
    }))
}

/// Main optimizer loop: scan → find unreordered → reorder (bounded concurrency).
async fn optimizer_loop(
    registry: Arc<IndexRegistry>,
    semaphore: Arc<Semaphore>,
    rayon_pool: Arc<rayon::ThreadPool>,
    scan_interval: Duration,
) {
    // Initial delay: let the server finish startup before scanning.
    tokio::time::sleep(Duration::from_secs(5)).await;

    loop {
        if let Err(e) = scan_and_optimize(&registry, &semaphore, &rayon_pool).await {
            warn!("[optimizer] scan failed: {}", e);
        }

        tokio::time::sleep(scan_interval).await;
    }
}

/// One scan cycle: list indexes, find unreordered segments, spawn reorder tasks.
async fn scan_and_optimize(
    registry: &IndexRegistry,
    semaphore: &Arc<Semaphore>,
    rayon_pool: &Arc<rayon::ThreadPool>,
) -> Result<(), tonic::Status> {
    let index_names = registry.list_indexes().await?;

    for name in index_names {
        // Open index (cheap if already cached)
        let index = match registry.get_or_open_index(&name).await {
            Ok(idx) => idx,
            Err(e) => {
                debug!("[optimizer] cannot open index '{}': {}", name, e);
                continue;
            }
        };

        // Skip indexes without reorder fields
        if !index.schema().has_reorder_fields() {
            continue;
        }

        let writer = match registry.get_writer(&name).await {
            Ok(w) => w,
            Err(e) => {
                debug!("[optimizer] cannot get writer for '{}': {}", name, e);
                continue;
            }
        };

        // Get segment manager to check unreordered segments
        let segment_manager = {
            let w = writer.read().await;
            Arc::clone(w.segment_manager())
        };

        let unreordered = segment_manager.unreordered_segment_ids().await;

        if unreordered.is_empty() {
            continue;
        }

        debug!(
            "[optimizer] index '{}': {} unreordered segment(s)",
            name,
            unreordered.len()
        );

        for seg_id in unreordered {
            // Acquire semaphore permit to bound concurrency
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    // Semaphore closed — shutting down
                    return Ok(());
                }
            };

            let sm = Arc::clone(&segment_manager);
            let idx_name = name.clone();
            let sid = seg_id.clone();
            let pool = Arc::clone(rayon_pool);

            tokio::spawn(async move {
                let _permit = permit;
                let start = std::time::Instant::now();

                match sm.reorder_single_segment(&sid, Some(pool)).await {
                    Ok(true) => {
                        info!(
                            "[optimizer] reordered segment {} in index '{}' ({:.1}s)",
                            sid,
                            idx_name,
                            start.elapsed().as_secs_f64(),
                        );
                    }
                    Ok(false) => {
                        debug!(
                            "[optimizer] segment {} in index '{}' skipped (in merge)",
                            sid, idx_name
                        );
                    }
                    Err(e) => {
                        warn!(
                            "[optimizer] failed to reorder segment {} in index '{}': {}",
                            sid, idx_name, e
                        );
                    }
                }
            });
        }
    }

    Ok(())
}
