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
    //
    // Sized at max(threads, cores/2), NOT just `threads`: `threads` bounds
    // how many segments are rewritten concurrently (the semaphore), while
    // the pool bounds how wide a SINGLE BP pass can fan out (recursive
    // bisection halves via rayon::join + parallel gain computation). A lone
    // deepening pass on a 10M+ doc segment should use the same cores/2 slice
    // the merge-time BP pool gets, not serialize onto 4 threads.
    let bp_pool_threads = std::thread::available_parallelism()
        .map(|c| c.get() / 2)
        .unwrap_or(config.threads)
        .max(config.threads);
    let rayon_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(bp_pool_threads)
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

        // Sweep orphan segment files (outputs of failed/panicked merges and
        // reorders — observed leaking 1.7 TB overnight). In-flight outputs
        // are protected by the merge inventory. Runs for every index, not
        // just reorder-enabled ones: merges can fail anywhere.
        match segment_manager.cleanup_orphan_segments().await {
            Ok(0) => {}
            Ok(n) => warn!(
                "[optimizer] swept {} orphan segment(s) in '{}' (failed merge/reorder outputs)",
                n, name
            ),
            Err(e) => debug!("[optimizer] orphan sweep failed for '{}': {}", name, e),
        }

        // Skip indexes without reorder fields
        if !index.schema().has_reorder_fields() {
            continue;
        }

        // Fresh (never-reordered) segments first — they are typically small
        // memtable flushes that finish in sub-second passes.
        let fresh = segment_manager.unreordered_segments().await;
        let mut candidates: Vec<(String, u32, bool)> = fresh
            .into_iter()
            .map(|(id, docs)| (id, docs, false))
            .collect();

        // Deepening is cooldown-paced but NOT starved by fresh work: under
        // continuous ingestion fresh segments arrive every commit, so a
        // "only when idle" rule would postpone deepening indefinitely. One
        // budget-truncated segment per cooldown window (each follow-up is a
        // full segment rewrite; it warm-starts from the previous order and
        // deepens toward block-granularity).
        let unconverged = segment_manager.unconverged_segments().await;
        if !unconverged.is_empty() {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let last = last_partial_pass.load(Ordering::Relaxed);
            if now_ms.saturating_sub(last) >= config.unconverged_cooldown.as_millis() as u64 {
                last_partial_pass.store(now_ms, Ordering::Relaxed);
                if let Some((id, docs)) = unconverged.into_iter().next() {
                    debug!(
                        "[optimizer] index '{}': deepening unconverged segment {} ({} docs)",
                        name, id, docs,
                    );
                    candidates.push((id, docs, true));
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

        for (seg_id, num_docs, is_deepening) in candidates {
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
            // work); large ones get a depth- and wall-clock-budgeted FIRST
            // pass (recorded unconverged), then full-depth deepening passes
            // (warm-started, wall-clock-bounded) until one beats the clock.
            let budget = if is_deepening {
                info!(
                    "[optimizer] deepening segment {} ({} docs): full depth, time budget {:.0}s",
                    sid,
                    num_docs,
                    config.time_budget.as_secs_f64(),
                );
                BpBudget {
                    min_partition_docs: None,
                    time_budget: Some(config.time_budget),
                }
            } else if num_docs >= config.large_segment_docs {
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
