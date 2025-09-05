//! Background segment optimizer — bounded BP and deleted-row compaction.
//!
//! Runs as a set of tokio tasks bounded by a whole-pass semaphore. Periodically scans
//! all indexes for segments that haven't been reordered and applies Recursive
//! Graph Bisection (BP) to improve BMP block clustering.
//!
//! Compaction also considers indexes without reorder fields.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hermes_core::segment::BpBudget;
use log::{debug, info, warn};
use tokio::sync::{Semaphore, TryAcquireError, watch};
use tokio::task::JoinSet;

use crate::registry::IndexRegistry;

/// Background optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Width of the shared BP Rayon pool (0 = optimizer disabled).
    pub threads: usize,
    /// Number of whole-segment optimizer tasks admitted at once. A second,
    /// application-wide gate shared with merge-time/manual BP enforces the
    /// same limit across every producer.
    pub concurrent_passes: usize,
    /// Interval between scans for unreordered segments.
    pub scan_interval: Duration,
    /// Segments with at least this many docs get a *budgeted* BP pass
    /// (depth capped at `partial_min_partition_docs`, wall clock capped at
    /// `time_budget`) instead of a full-depth pass.
    pub large_segment_docs: u32,
    /// Wall-clock budget per BP pass on large segments.
    pub time_budget: Duration,
    /// Depth cap for large segments: stop bisection at partitions of this
    /// many vectors. 256 is one default LSP superblock (8 × 32 vectors).
    pub partial_min_partition_docs: usize,
    /// Minimum wait between follow-up passes on a segment whose previous
    /// pass hit its wall-clock budget (`bp_converged == false`). Each
    /// follow-up warm-starts from the previous order and deepens.
    pub unconverged_cooldown: Duration,
    /// Optimizer follow-up threshold for budget-exhausted rewrites in one
    /// replacement lineage. Without it, a segment that can never beat
    /// `time_budget` is rewritten forever by the optimizer, continually
    /// consuming all BP workers and disk I/O.
    pub max_unconverged_passes: u32,
    /// Deleted / physical row threshold; zero disables automatic compaction.
    pub compaction_deleted_ratio: f64,
    /// Minimum time between completed automatic compactions across all indexes.
    pub compaction_cooldown: Duration,
    /// Per-compaction scratch cap, independent of the BP cap.
    pub compaction_memory_budget: usize,
}

/// Global completion-based pacing for either BP deepening or compaction.
///
/// Segment IDs change after every successful rewrite, so a per-ID cooldown
/// cannot follow a segment lineage. The gate enforces the documented policy
/// directly: at most one deepening pass is active, followed by a cooldown
/// measured from completion. Measuring from start caused passes longer than
/// the cooldown to requeue their replacement immediately and overlap another
/// lineage, keeping background BP busy continuously.
#[derive(Default)]
struct CooldownGate {
    state: Mutex<CooldownGateState>,
}

#[derive(Default)]
struct CooldownGateState {
    in_flight: bool,
    last_finished: Option<Instant>,
}

impl CooldownGate {
    fn try_acquire(self: &Arc<Self>, cooldown: Duration) -> Option<CooldownPermit> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.in_flight
            || state
                .last_finished
                .is_some_and(|finished| finished.elapsed() < cooldown)
        {
            return None;
        }

        state.in_flight = true;
        Some(CooldownPermit {
            gate: Arc::clone(self),
        })
    }
}

/// Completion-based permit. Drop runs on success, error, cancellation, and
/// panic unwind, so the gate cannot become permanently wedged.
struct CooldownPermit {
    gate: Arc<CooldownGate>,
}

impl Drop for CooldownPermit {
    fn drop(&mut self) {
        let mut state = self.gate.state.lock().unwrap_or_else(|e| e.into_inner());
        state.last_finished = Some(Instant::now());
        state.in_flight = false;
    }
}

/// Spawn the background optimizer loop.
///
/// Returns a `JoinHandle` that runs until the server shuts down.
/// When `threads == 0`, no optimizer is started.
pub fn spawn_optimizer(
    registry: Arc<IndexRegistry>,
    config: OptimizerConfig,
    shutdown: watch::Receiver<bool>,
) -> Option<tokio::task::JoinHandle<()>> {
    if config.threads == 0 {
        return None;
    }

    info!(
        "Starting background optimizer: {} shared CPU threads, {} concurrent pass(es), {:.0}s scan interval, {}-pass unconverged follow-up threshold; compaction deleted ratio={} (0 disables), cooldown={}s, scratch={} MiB",
        config.threads,
        config.concurrent_passes,
        config.scan_interval.as_secs_f64(),
        config.max_unconverged_passes,
        config.compaction_deleted_ratio,
        config.compaction_cooldown.as_secs_f64(),
        config.compaction_memory_budget / (1024 * 1024),
    );

    let semaphore = Arc::new(Semaphore::new(config.concurrent_passes.max(1)));

    Some(tokio::spawn(async move {
        optimizer_loop(registry, semaphore, config, shutdown).await;
    }))
}

/// Main optimizer loop: scan → find unreordered → reorder (bounded concurrency).
async fn optimizer_loop(
    registry: Arc<IndexRegistry>,
    semaphore: Arc<Semaphore>,
    config: OptimizerConfig,
    mut shutdown: watch::Receiver<bool>,
) {
    // Initial delay: let the server finish startup before scanning.
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(5)) => {}
        _ = shutdown.changed() => return,
    }

    // A timed-out pass replaces its source with a new unconverged segment ID.
    // Gate that lineage globally and start its cooldown only when work ends.
    let deepening_gate = Arc::new(CooldownGate::default());
    let compaction_gate = Arc::new(CooldownGate::default());
    let mut next_index = 0usize;
    let mut tasks = JoinSet::new();

    'optimizer: loop {
        let scan = scan_and_optimize(
            &registry,
            &semaphore,
            &config,
            &deepening_gate,
            &compaction_gate,
            &mut next_index,
            &mut tasks,
        );
        let scan_result = tokio::select! {
            result = scan => result,
            _ = shutdown.changed() => break 'optimizer,
        };
        if *shutdown.borrow() {
            break;
        }
        if let Err(e) = scan_result {
            warn!("[optimizer] scan failed: {}", e);
        }

        while let Some(result) = tasks.try_join_next() {
            if let Err(error) = result {
                warn!("[optimizer] reorder task failed: {}", error);
            }
        }

        let sleep = tokio::time::sleep(config.scan_interval);
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                _ = &mut sleep => break,
                _ = shutdown.changed() => break 'optimizer,
                result = tasks.join_next(), if !tasks.is_empty() => {
                    if let Some(Err(error)) = result {
                        warn!("[optimizer] reorder task failed: {}", error);
                    }
                }
            }
        }
    }

    semaphore.close();
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            warn!("[optimizer] reorder task failed during shutdown: {}", error);
        }
    }
    info!("[optimizer] shut down");
}

/// One scan cycle: list indexes, find candidates, spawn reorder tasks.
///
/// Priority: one cooldown-eligible unconverged segment first so continuous
/// ingestion cannot starve deepening, then never-reordered segments ordered
/// small-first. Deepening passes warm-start from the previous layout.
#[allow(clippy::too_many_arguments)]
async fn scan_and_optimize(
    registry: &IndexRegistry,
    semaphore: &Arc<Semaphore>,
    config: &OptimizerConfig,
    deepening_gate: &Arc<CooldownGate>,
    compaction_gate: &Arc<CooldownGate>,
    next_index: &mut usize,
    tasks: &mut JoinSet<()>,
) -> Result<(), tonic::Status> {
    let mut index_names = registry.list_indexes().await?;
    // A busy first index used to consume every slot on every scan. Rotate the
    // starting point so continuously ingesting indexes cannot starve peers.
    if !index_names.is_empty() {
        let start = *next_index % index_names.len();
        index_names.rotate_left(start);
        *next_index = (start + 1) % index_names.len();
    }

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

        // Sweep segment files with no lifecycle owner (metadata, active
        // indexing/merge/reorder operation, or deferred reader deletion).
        // Runs for every index: every producer can be cancelled or fail.
        match segment_manager.cleanup_orphan_segments().await {
            Ok(0) => {}
            Ok(n) => warn!(
                "[optimizer] swept {} unowned orphan segment(s) in '{}'",
                n, name
            ),
            Err(e) => debug!("[optimizer] orphan sweep failed for '{}': {}", name, e),
        }

        let compactions = segment_manager
            .compaction_candidates(config.compaction_deleted_ratio)
            .await
            .map_err(crate::error::hermes_error_to_status)?;
        let compacting_ids: std::collections::HashSet<_> =
            compactions.iter().map(|(id, _, _)| id.clone()).collect();
        let mut candidates: Vec<(String, u32, bool, bool)> = compactions
            .into_iter()
            .map(|(id, docs, _)| (id, docs, false, true))
            .collect();
        if index.schema().has_reorder_fields() {
            let mut fresh = segment_manager.unreordered_segments().await;
            fresh.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            candidates.extend(
                fresh
                    .into_iter()
                    .filter(|(id, _)| !compacting_ids.contains(id))
                    .map(|(id, docs)| (id, docs, false, false)),
            );
        }

        // Deepening is cooldown-paced but NOT starved by fresh work: under
        // continuous ingestion fresh segments arrive every commit, so a
        // "only when idle" rule would postpone deepening indefinitely. One
        // budget-truncated segment per cooldown window (each follow-up is a
        // full segment rewrite; it warm-starts from the previous order and
        // deepens toward block-granularity).
        let mut unconverged = if index.schema().has_reorder_fields() {
            segment_manager
                .unconverged_segments_below(config.max_unconverged_passes)
                .await
        } else {
            Vec::new()
        };
        unconverged.retain(|(id, _, _)| !compacting_ids.contains(id));
        unconverged.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if let Some((id, docs, attempts)) = unconverged.into_iter().next() {
            // Put deepening first so an endless stream of fresh flushes cannot
            // consume every slot. The cooldown gate is acquired only after a
            // scheduler slot exists; merely discovering a candidate must not
            // start its cooldown.
            debug!(
                "[optimizer] segment {} has used {}/{} unconverged pass(es)",
                id, attempts, config.max_unconverged_passes,
            );
            candidates.insert(0, (id, docs, true, false));
        }

        if candidates.is_empty() {
            continue;
        }

        debug!(
            "[optimizer] index '{}': {} maintenance candidate(s)",
            name,
            candidates.len()
        );

        for (seg_id, num_docs, is_deepening, is_compaction) in candidates {
            // Never stall the entire index scan behind long-running tasks.
            // Filled capacity is normal; the next periodic scan retries.
            let permit = match semaphore.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(TryAcquireError::NoPermits) => break,
                Err(TryAcquireError::Closed) => return Ok(()),
            };
            let compaction_permit = if is_compaction {
                match compaction_gate.try_acquire(config.compaction_cooldown) {
                    Some(permit) => Some(permit),
                    None => {
                        drop(permit);
                        continue;
                    }
                }
            } else {
                None
            };
            let deepening_permit = if is_deepening {
                match deepening_gate.try_acquire(config.unconverged_cooldown) {
                    Some(permit) => {
                        debug!(
                            "[optimizer] index '{}': deepening unconverged segment {} ({} docs)",
                            name, seg_id, num_docs,
                        );
                        Some(permit)
                    }
                    None => {
                        drop(permit);
                        continue;
                    }
                }
            } else {
                None
            };

            let sm = Arc::clone(&segment_manager);
            let idx_name = name.clone();
            let sid = seg_id.clone();
            let pool = segment_manager.background_cpu_pool();
            let refresh_index = Arc::clone(&index);

            // Size-tiered budget: small segments get full-depth BP (seconds of
            // work); large ones get a depth- and wall-clock-budgeted FIRST
            // pass (recorded unconverged), then full-depth deepening passes
            // (warm-started, wall-clock-bounded) until one beats the clock.
            let budget = if is_compaction {
                BpBudget::full() // not used by compaction
            } else if is_deepening {
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

            let compaction_threshold = config.compaction_deleted_ratio;
            let compaction_memory_budget = config.compaction_memory_budget;
            tasks.spawn(async move {
                let _permit = permit;
                // Starts cooldown when the task finishes, not when it was
                // queued. Also releases the in-flight gate on panic unwind.
                let _deepening_permit = deepening_permit;
                let _compaction_permit = compaction_permit;
                let start = std::time::Instant::now();

                let action = if is_compaction { "compacted" } else { "reordered" };
                let result = if is_compaction {
                    sm.compact_segment_if_eligible(&sid, compaction_threshold, compaction_memory_budget).await
                } else {
                    sm.reorder_single_segment(&sid, Some(pool), budget).await
                };
                match result {
                    Ok(true) => {
                        match refresh_index.reader().await {
                            Ok(reader) => {
                                if let Err(error) = reader.reload().await {
                                    warn!(
                                        "[optimizer] {action} segment {} in index '{}' but failed to reload reader: {}",
                                        sid, idx_name, error,
                                    );
                                }
                            }
                            Err(error) => warn!(
                                "[optimizer] {action} segment {} in index '{}' but failed to open reader: {}",
                                sid, idx_name, error,
                            ),
                        }
                        info!(
                            "[optimizer] {action} segment {} in index '{}' ({:.1}s)",
                            sid,
                            idx_name,
                            start.elapsed().as_secs_f64(),
                        );
                    }
                    Ok(false) => {
                        debug!(
                            "[optimizer] segment {} in index '{}' skipped (busy or no longer eligible)",
                            sid, idx_name
                        );
                    }
                    Err(hermes_core::Error::IndexClosed) => {
                        debug!(
                            "[optimizer] segment {} in index '{}' cancelled during shutdown",
                            sid, idx_name,
                        );
                    }
                    Err(e) => {
                        warn!(
                            "[optimizer] failed maintenance on segment {} in index '{}': {}",
                            sid, idx_name, e
                        );
                    }
                }
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn optimizer_compacts_without_reorder_fields_and_paces_work_across_indexes() {
        let root = tempfile::tempdir().unwrap();
        let registry = IndexRegistry::new(
            root.path().to_owned(),
            hermes_core::IndexConfig {
                merge_policy: Box::new(hermes_core::NoMergePolicy),
                ..Default::default()
            },
        );
        for name in ["a", "b"] {
            let mut schema = hermes_core::SchemaBuilder::default();
            let id = schema.add_text_field("id", true, true);
            schema.set_primary_key(id);
            registry.create_index(name, schema.build()).await.unwrap();
            let writer = registry.get_writer(name).await.unwrap();
            let mut writer = writer.write().await;
            writer.init_primary_key_dedup().await.unwrap();
            for key in ["dead", "live"] {
                let mut doc = hermes_core::Document::new();
                doc.add_text(id, key);
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
            writer.delete_primary_key("dead").unwrap();
            writer.commit().await.unwrap();
        }
        let mut config = OptimizerConfig {
            threads: 1,
            concurrent_passes: 2,
            scan_interval: Duration::from_secs(60),
            large_segment_docs: 1000,
            time_budget: Duration::from_secs(1),
            partial_min_partition_docs: 256,
            unconverged_cooldown: Duration::from_secs(60),
            max_unconverged_passes: 3,
            compaction_deleted_ratio: 0.5,
            compaction_cooldown: Duration::from_secs(60),
            compaction_memory_budget: 16 * 1024 * 1024,
        };
        let slots = Arc::new(Semaphore::new(2));
        let deepening = Arc::new(CooldownGate::default());
        let compaction = Arc::new(CooldownGate::default());
        let mut next = 0;
        let mut tasks = JoinSet::new();
        scan_and_optimize(
            &registry,
            &slots,
            &config,
            &deepening,
            &compaction,
            &mut next,
            &mut tasks,
        )
        .await
        .unwrap();
        assert_eq!(
            tasks.len(),
            1,
            "only one automatic compaction may start globally"
        );
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
        scan_and_optimize(
            &registry,
            &slots,
            &config,
            &deepening,
            &compaction,
            &mut next,
            &mut tasks,
        )
        .await
        .unwrap();
        assert!(
            tasks.is_empty(),
            "completion cooldown must survive replacement IDs"
        );
        config.compaction_cooldown = Duration::ZERO;
        scan_and_optimize(
            &registry,
            &slots,
            &config,
            &deepening,
            &compaction,
            &mut next,
            &mut tasks,
        )
        .await
        .unwrap();
        assert_eq!(tasks.len(), 1);
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
        for name in ["a", "b"] {
            let index = registry.get_or_open_index(name).await.unwrap();
            let searcher = index.reader().await.unwrap().searcher().await.unwrap();
            assert_eq!(searcher.num_docs(), 1);
            assert_eq!(searcher.segment_readers()[0].num_docs(), 1);
        }
        registry.shutdown().await.unwrap();
    }

    #[test]
    fn deepening_gate_blocks_overlap_and_cools_down_from_completion() {
        let gate = Arc::new(CooldownGate::default());

        let first = gate
            .try_acquire(Duration::from_secs(60))
            .expect("first pass should start");
        assert!(
            gate.try_acquire(Duration::ZERO).is_none(),
            "a second deepening pass must not overlap"
        );

        drop(first);
        assert!(
            gate.try_acquire(Duration::from_secs(60)).is_none(),
            "cooldown must begin when the pass completes"
        );
        assert!(
            gate.try_acquire(Duration::ZERO).is_some(),
            "the gate should reopen after its cooldown"
        );
    }

    #[tokio::test]
    async fn optimizer_supervisor_stops_on_application_shutdown() {
        let registry = Arc::new(IndexRegistry::new(
            std::env::temp_dir().join("hermes_optimizer_shutdown_test"),
            hermes_core::IndexConfig::default(),
        ));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let handle = spawn_optimizer(
            registry,
            OptimizerConfig {
                threads: 1,
                concurrent_passes: 1,
                scan_interval: Duration::from_secs(60),
                large_segment_docs: 1_000_000,
                time_budget: Duration::from_secs(60),
                partial_min_partition_docs: 256,
                unconverged_cooldown: Duration::from_secs(60),
                max_unconverged_passes: 1,
                compaction_deleted_ratio: 0.3,
                compaction_cooldown: Duration::from_secs(60),
                compaction_memory_budget: 256 * 1024 * 1024,
            },
            shutdown_rx,
        )
        .expect("optimizer must be enabled");

        shutdown_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("optimizer ignored shutdown")
            .unwrap();
    }
}
