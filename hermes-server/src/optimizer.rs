//! Background segment optimizer — reorders unreordered segments via BP.
//!
//! Runs as a set of tokio tasks bounded by a semaphore. Periodically scans
//! all indexes for segments that haven't been reordered and applies Recursive
//! Graph Bisection (BP) to improve BMP block clustering.
//!
//! Only indexes with `reorder` fields in their schema are considered.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use hermes_core::segment::BpBudget;
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
    /// Segments with at least this many docs get a *budgeted* BP pass
    /// (depth capped at `partial_min_partition_docs`, wall clock capped at
    /// `time_budget`) instead of a full-depth pass.
    pub large_segment_docs: u32,
    /// Wall-clock budget per BP pass on large segments.
    pub time_budget: Duration,
    /// Depth cap for large segments: stop bisection at partitions of this
    /// many docs. 4096 = superblock granularity (64 blocks × 64 docs), which
    /// keeps most of the superblock-pruning win at ~⅓ less depth.
    pub partial_min_partition_docs: usize,
    /// Minimum wait between follow-up passes on a segment whose previous
    /// pass hit its wall-clock budget (`bp_converged == false`). Each
    /// follow-up warm-starts from the previous order and deepens.
    pub unconverged_cooldown: Duration,
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
        optimizer_loop(registry, semaphore, rayon_pool, config).await;
    }))
}

/// Main optimizer loop: scan → find unreordered → reorder (bounded concurrency).
async fn optimizer_loop(
    registry: Arc<IndexRegistry>,
    semaphore: Arc<Semaphore>,
    rayon_pool: Arc<rayon::ThreadPool>,
    config: OptimizerConfig,
) {
    // Initial delay: let the server finish startup before scanning.
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Unix millis of the last budgeted pass that may need a follow-up —
    // paces warm-started deepening passes (each is a full segment rewrite).
    let last_partial_pass = Arc::new(AtomicU64::new(0));

    loop {
        if let Err(e) = scan_and_optimize(
            &registry,
            &semaphore,
            &rayon_pool,
            &config,
            &last_partial_pass,
        )
        .await
        {
            warn!("[optimizer] scan failed: {}", e);
        }

        tokio::time::sleep(config.scan_interval).await;
    }
}

/// One scan cycle: list indexes, find candidates, spawn reorder tasks.
///
/// Priority: never-reordered segments first (small ones full-depth, large
/// ones budgeted). When none exist, revisit unconverged segments (previous
/// pass hit its wall-clock budget) after a cooldown — those passes
/// warm-start and deepen.
async fn scan_and_optimize(
    registry: &IndexRegistry,
    semaphore: &Arc<Semaphore>,
    rayon_pool: &Arc<rayon::ThreadPool>,
    config: &OptimizerConfig,
    last_partial_pass: &Arc<AtomicU64>,
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

        let mut candidates = segment_manager.unreordered_segments().await;

        if candidates.is_empty() {
            // No fresh work — revisit budget-truncated segments (cooldown-paced:
            // every follow-up pass is a full segment rewrite).
            let unconverged = segment_manager.unconverged_segments().await;
            if !unconverged.is_empty() {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let last = last_partial_pass.load(Ordering::Relaxed);
                if now_ms.saturating_sub(last) >= config.unconverged_cooldown.as_millis() as u64 {
                    last_partial_pass.store(now_ms, Ordering::Relaxed);
                    // One deepening pass per cooldown window.
                    candidates = unconverged.into_iter().take(1).collect();
                    debug!(
                        "[optimizer] index '{}': deepening 1 unconverged segment",
                        name,
                    );
                }
            }
        }

        if candidates.is_empty() {
            continue;
        }

        debug!(
            "[optimizer] index '{}': {} reorder candidate(s)",
            name,
            candidates.len()
        );

        for (seg_id, num_docs) in candidates {
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

            // Size-tiered budget: small segments get full-depth BP (seconds of
            // work); large ones get a depth- and wall-clock-budgeted pass.
            let budget = if num_docs >= config.large_segment_docs {
                info!(
                    "[optimizer] segment {} ({} docs) exceeds {} docs — budgeted BP pass \
                     (min_partition={} docs, time budget {:.0}s)",
                    sid,
                    num_docs,
                    config.large_segment_docs,
                    config.partial_min_partition_docs,
                    config.time_budget.as_secs_f64(),
                );
                BpBudget {
                    min_partition_docs: Some(config.partial_min_partition_docs),
                    time_budget: Some(config.time_budget),
                }
            } else {
                BpBudget::full()
            };

            tokio::spawn(async move {
                let _permit = permit;
                let start = std::time::Instant::now();

                match sm.reorder_single_segment(&sid, Some(pool), budget).await {
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
