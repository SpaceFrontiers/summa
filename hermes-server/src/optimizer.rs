//! Background segment optimizer — reorders unreordered segments via BP.
//!
//! Runs as a set of tokio tasks bounded by a whole-pass semaphore. Periodically scans
//! all indexes for segments that haven't been reordered and applies Recursive
//! Graph Bisection (BP) to improve BMP block clustering.
//!
//! Only indexes with `reorder` fields in their schema are considered.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hermes_core::segment::BpBudget;
use log::{debug, info, warn};
use tokio::sync::{Semaphore, TryAcquireError};

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
    /// many docs. 4096 = superblock granularity (64 blocks × 64 docs), which
    /// keeps most of the superblock-pruning win at ~⅓ less depth.
    pub partial_min_partition_docs: usize,
    /// Minimum wait between follow-up passes on a segment whose previous
    /// pass hit its wall-clock budget (`bp_converged == false`). Each
    /// follow-up warm-starts from the previous order and deepens.
    pub unconverged_cooldown: Duration,
}

/// Global gate for expensive full-depth deepening passes.
///
/// Segment IDs change after every successful rewrite, so a per-ID cooldown
/// cannot follow a segment lineage. The gate enforces the documented policy
/// directly: at most one deepening pass is active, followed by a cooldown
/// measured from completion. Measuring from start caused passes longer than
/// the cooldown to requeue their replacement immediately and overlap another
/// lineage, keeping background BP busy continuously.
#[derive(Default)]
struct DeepeningGate {
    state: Mutex<DeepeningGateState>,
}

#[derive(Default)]
struct DeepeningGateState {
    in_flight: bool,
    last_finished: Option<Instant>,
}

impl DeepeningGate {
    fn try_acquire(self: &Arc<Self>, cooldown: Duration) -> Option<DeepeningPermit> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.in_flight
            || state
                .last_finished
                .is_some_and(|finished| finished.elapsed() < cooldown)
        {
            return None;
        }

        state.in_flight = true;
        Some(DeepeningPermit {
            gate: Arc::clone(self),
        })
    }
}

/// Completion-based permit. Drop runs on success, error, cancellation, and
/// panic unwind, so the gate cannot become permanently wedged.
struct DeepeningPermit {
    gate: Arc<DeepeningGate>,
}

impl Drop for DeepeningPermit {
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
) -> Option<tokio::task::JoinHandle<()>> {
    if config.threads == 0 {
        return None;
    }

    info!(
        "Starting background optimizer: {} shared BP threads, {} concurrent pass(es), {:.0}s scan interval",
        config.threads,
        config.concurrent_passes,
        config.scan_interval.as_secs_f64(),
    );

    let semaphore = Arc::new(Semaphore::new(config.concurrent_passes.max(1)));

    Some(tokio::spawn(async move {
        optimizer_loop(registry, semaphore, config).await;
    }))
}

/// Main optimizer loop: scan → find unreordered → reorder (bounded concurrency).
async fn optimizer_loop(
    registry: Arc<IndexRegistry>,
    semaphore: Arc<Semaphore>,
    config: OptimizerConfig,
) {
    // Initial delay: let the server finish startup before scanning.
    tokio::time::sleep(Duration::from_secs(5)).await;

    // A timed-out pass replaces its source with a new unconverged segment ID.
    // Gate that lineage globally and start its cooldown only when work ends.
    let deepening_gate = Arc::new(DeepeningGate::default());
    let mut next_index = 0usize;

    loop {
        if let Err(e) = scan_and_optimize(
            &registry,
            &semaphore,
            &config,
            &deepening_gate,
            &mut next_index,
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
/// Priority: one cooldown-eligible unconverged segment first so continuous
/// ingestion cannot starve deepening, then never-reordered segments ordered
/// small-first. Deepening passes warm-start from the previous layout.
async fn scan_and_optimize(
    registry: &IndexRegistry,
    semaphore: &Arc<Semaphore>,
    config: &OptimizerConfig,
    deepening_gate: &Arc<DeepeningGate>,
    next_index: &mut usize,
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

        // Skip indexes without reorder fields
        if !index.schema().has_reorder_fields() {
            continue;
        }

        // Fresh (never-reordered) segments first — they are typically small
        // memtable flushes that finish in sub-second passes.
        let mut fresh = segment_manager.unreordered_segments().await;
        // Short passes first increase throughput and release memory quickly;
        // ID is a deterministic tie-break for reproducible scheduling.
        fresh.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
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
        let mut unconverged = segment_manager.unconverged_segments().await;
        unconverged.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if let Some((id, docs)) = unconverged.into_iter().next() {
            // Put deepening first so an endless stream of fresh flushes cannot
            // consume every slot. The cooldown gate is acquired only after a
            // scheduler slot exists; merely discovering a candidate must not
            // start its cooldown.
            candidates.insert(0, (id, docs, true));
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
            // Never stall the entire index scan behind long-running tasks.
            // Filled capacity is normal; the next periodic scan retries.
            let permit = match semaphore.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(TryAcquireError::NoPermits) => break,
                Err(TryAcquireError::Closed) => return Ok(()),
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
                // Starts cooldown when the task finishes, not when it was
                // queued. Also releases the in-flight gate on panic unwind.
                let _deepening_permit = deepening_permit;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepening_gate_blocks_overlap_and_cools_down_from_completion() {
        let gate = Arc::new(DeepeningGate::default());

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
}
