//! Segment manager — coordinates segment commit, background merging, and trained structures.
//!
//! Architecture:
//! - **Single mutation queue**: All metadata mutations serialize through `tokio::sync::Mutex<ManagerState>`.
//! - **Active-operation ownership**: Every segment that is being built, merged,
//!   or reordered is registered before its first file is written and remains
//!   registered until it is either published in metadata or abandoned.
//! - **Concurrent merges**: Multiple non-overlapping merges can run in parallel.
//!   New merges are rejected only if they share segments with an active operation.
//! - **Auto-trigger**: Each completed merge re-evaluates the merge policy and spawns
//!   new merges if eligible (cascading merges for higher tiers).
//! - **ArcSwap for trained**: Lock-free reads of trained vector structures.
//!
//! # Segment lifecycle invariant
//!
//! Every on-disk `seg_*` ID must be protected by at least one of these owners:
//!
//! 1. `state.metadata` while the segment is live and searchable;
//! 2. `active_operations` while an indexing/merge/reorder task owns it; or
//! 3. `tracker` while a retired segment is still visible to a reader or its
//!    filesystem deletion is scheduled.
//!
//! An ID with no owner is an orphan and may be swept. Transitions are ordered
//! so the new owner is installed before the old owner is released.
//!
//! # Locking model (deadlock-free by construction)
//!
//! ```text
//! Lock ordering (acquire in this order):
//!   1. state               — tokio::sync::Mutex, held for mutations + disk I/O
//!   2. active_operations   — parking_lot::Mutex (sync), sub-μs hold, RAII guard
//!   3. tracker.inner       — parking_lot::Mutex (sync), sub-μs hold
//!
//! Independent bookkeeping (never held with `state`):
//!   trained                — arc_swap::ArcSwapOption, lock-free
//!   merge_handles          — parking_lot::Mutex, synchronous short hold
//!   lifecycle_handles      — parking_lot::Mutex, synchronous short hold
//!   merge/reorder permits  — tokio semaphores shared by configuration
//! ```
//!
//! **Rule:** Never hold a sync lock while `.await`-ing.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwapOption;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinHandle;

use crate::directories::DirectoryWriter;
use crate::error::{Error, Result};
use crate::index::IndexMetadata;
use crate::segment::{
    SegmentFiles, SegmentId, SegmentMeta, SegmentSnapshot, SegmentTracker, TrainedVectorStructures,
};
#[cfg(feature = "native")]
use crate::segment::{SegmentMerger, SegmentReader};

use super::{MergePolicy, SegmentInfo};

// ============================================================================
// RAII active-operation tracking
// ============================================================================

/// Tracks every segment ID owned by an in-flight lifecycle operation.
///
/// Merge/reorder guards include both sources and output, providing mutual
/// exclusion as well as orphan-sweep protection. Indexing guards contain the
/// new output only and live from before the first write through commit/abort.
struct ActiveOperationState {
    segment_ids: HashSet<String>,
    accepting: bool,
}

struct ActiveSegmentOperations {
    inner: parking_lot::Mutex<ActiveOperationState>,
    idle: Notify,
    shutdown: Notify,
}

impl ActiveSegmentOperations {
    fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(ActiveOperationState {
                segment_ids: HashSet::new(),
                accepting: true,
            }),
            idle: Notify::new(),
            shutdown: Notify::new(),
        }
    }

    /// Try to claim IDs for an operation. Returns a guard on success, `None`
    /// if any requested ID is already owned by another active operation.
    fn try_register(self: &Arc<Self>, segment_ids: Vec<String>) -> Option<SegmentOperationGuard> {
        let mut inner = self.inner.lock();
        if !inner.accepting {
            log::debug!("[segment_lifecycle] rejected operation during shutdown");
            return None;
        }
        // Check for overlap with any active lifecycle operation.
        for id in &segment_ids {
            if inner.segment_ids.contains(id) {
                log::debug!(
                    "[segment_lifecycle] rejected: {} overlaps with an active operation ({} active IDs)",
                    id,
                    inner.segment_ids.len()
                );
                return None;
            }
        }
        log::debug!(
            "[segment_lifecycle] registered {} IDs (total active: {})",
            segment_ids.len(),
            inner.segment_ids.len() + segment_ids.len()
        );
        for id in &segment_ids {
            inner.segment_ids.insert(id.clone());
        }
        Some(SegmentOperationGuard {
            active_operations: Arc::clone(self),
            segment_ids,
        })
    }

    /// Snapshot of all IDs owned by active operations.
    fn snapshot(&self) -> HashSet<String> {
        self.inner.lock().segment_ids.clone()
    }

    /// Atomically prevent new lifecycle work from starting. Existing guards
    /// remain valid and can be drained with [`Self::wait_until_idle`].
    fn stop_accepting(&self) {
        let mut inner = self.inner.lock();
        inner.accepting = false;
        self.shutdown.notify_waiters();
        if inner.segment_ids.is_empty() {
            self.idle.notify_waiters();
        }
    }

    fn is_accepting(&self) -> bool {
        self.inner.lock().accepting
    }

    /// Wait until every operation that started before shutdown has released
    /// its ownership. Register/check and notification are ordered to avoid a
    /// missed wakeup between observing a non-empty set and awaiting.
    async fn wait_until_idle(&self) {
        loop {
            let notified = self.idle.notified();
            if self.inner.lock().segment_ids.is_empty() {
                return;
            }
            notified.await;
        }
    }

    /// Resolve when shutdown starts, without missing a notification between
    /// checking the state and registering the waiter.
    async fn wait_for_shutdown(&self) {
        loop {
            let notified = self.shutdown.notified();
            if !self.inner.lock().accepting {
                return;
            }
            notified.await;
        }
    }
}

/// RAII ownership of segment IDs used by an active lifecycle operation.
/// Dropping on success, error, cancellation, or panic makes abandoned outputs
/// eligible for sweeping automatically.
pub(crate) struct SegmentOperationGuard {
    active_operations: Arc<ActiveSegmentOperations>,
    segment_ids: Vec<String>,
}

impl Drop for SegmentOperationGuard {
    fn drop(&mut self) {
        let mut inner = self.active_operations.inner.lock();
        for id in &self.segment_ids {
            inner.segment_ids.remove(id);
        }
        if inner.segment_ids.is_empty() {
            self.active_operations.idle.notify_waiters();
        }
    }
}

/// Merge-time/manual BP pools are shared by every index in this process.
/// A pool per `SegmentManager` multiplied a 96-core host into two 48-thread
/// merge pools plus the optimizer pool (200+ process threads in production).
static BACKGROUND_CPU_POOL: OnceLock<Arc<rayon::ThreadPool>> = OnceLock::new();

const MERGE_RETRY_BASE_DELAY: std::time::Duration = std::time::Duration::from_secs(30);
const MERGE_RETRY_MAX_DELAY: std::time::Duration = std::time::Duration::from_secs(30 * 60);

#[derive(Default)]
struct MergeRetryState {
    retry_after: Option<std::time::Instant>,
    consecutive_failures: u32,
}

fn merge_retry_delay(consecutive_failures: u32) -> std::time::Duration {
    let shift = consecutive_failures.saturating_sub(1).min(16);
    MERGE_RETRY_BASE_DELAY
        .checked_mul(1u32 << shift)
        .unwrap_or(MERGE_RETRY_MAX_DELAY)
        .min(MERGE_RETRY_MAX_DELAY)
}

/// Spawn and register auxiliary lifecycle work as one synchronous operation.
///
/// Registering *after* `spawn` left a small deletion race: shutdown could
/// observe an empty handle list while the newly spawned filesystem task was
/// already running. Holding the handle-list mutex across `Handle::spawn`
/// makes task creation visible to the drain before either side can proceed.
fn try_spawn_lifecycle<F>(
    handles: &parking_lot::Mutex<Vec<JoinHandle<()>>>,
    runtime: &tokio::runtime::Handle,
    future: F,
) -> bool
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut handles = handles.lock();
        handles.retain(|handle| !handle.is_finished());
        handles.push(runtime.spawn(future));
    }))
    .is_ok()
}

/// Deletes an uncommitted merge/reorder output if its task unwinds.
///
/// Normal `Result::Err` paths delete outputs synchronously so callers observe
/// a clean directory before returning. This guard covers the path those
/// branches cannot: a panic after output files have been created. The cleanup
/// callback re-checks metadata before deleting, so a panic after a successful
/// metadata commit cannot remove a live segment.
struct OutputCleanupGuard {
    segment_id: SegmentId,
    cleanup: Option<Arc<dyn Fn(SegmentId) + Send + Sync>>,
}

impl OutputCleanupGuard {
    fn new(segment_id: SegmentId, cleanup: Arc<dyn Fn(SegmentId) + Send + Sync>) -> Self {
        Self {
            segment_id,
            cleanup: Some(cleanup),
        }
    }

    fn disarm(&mut self) {
        self.cleanup = None;
    }
}

impl Drop for OutputCleanupGuard {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup(self.segment_id);
        }
    }
}

/// All mutable state behind the single async Mutex.
struct ManagerState {
    metadata: IndexMetadata,
    merge_policy: Box<dyn MergePolicy>,
}

#[cfg(feature = "native")]
struct MergeTaskError {
    error: Error,
    unavailable_segments: Vec<String>,
}

#[cfg(feature = "native")]
impl MergeTaskError {
    fn source(segment_id: String, error: Error) -> Self {
        Self {
            error,
            unavailable_segments: vec![segment_id],
        }
    }

    fn sources(segment_ids: Vec<String>, error: Error) -> Self {
        Self {
            error,
            unavailable_segments: segment_ids,
        }
    }
}

#[cfg(feature = "native")]
impl From<Error> for MergeTaskError {
    fn from(error: Error) -> Self {
        Self {
            error,
            unavailable_segments: Vec::new(),
        }
    }
}

#[cfg(feature = "native")]
fn is_deterministic_source_error(error: &Error) -> bool {
    matches!(error, Error::Corruption(_) | Error::Serialization(_))
        || matches!(error, Error::Io(error) if error.kind() == std::io::ErrorKind::NotFound)
}

#[cfg(feature = "native")]
fn classify_source_error(segment_id: String, error: Error) -> MergeTaskError {
    if is_deterministic_source_error(&error) {
        MergeTaskError::source(segment_id, error)
    } else {
        // Timeouts, interrupted reads, permission changes, and other generic
        // I/O failures may be transient. Back them off instead of quarantining
        // a healthy metadata segment for the rest of the process lifetime.
        MergeTaskError::from(error)
    }
}

#[cfg(feature = "native")]
type MergeTaskResult<T> = std::result::Result<T, MergeTaskError>;

/// Segment manager — coordinates segment commit, background merging, and trained structures.
///
/// SOLE owner of `metadata.json`. All metadata mutations go through `state` Mutex.
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Serializes ALL metadata mutations.
    state: Arc<AsyncMutex<ManagerState>>,

    /// RAII ownership for every in-flight segment lifecycle operation.
    active_operations: Arc<ActiveSegmentOperations>,

    /// Metadata-live segments involved in a deterministic source/corruption
    /// failure. They stay searchable (and operator-visible) but are excluded
    /// from merges for this process lifetime, preventing a bad candidate from
    /// consuming full rewrite capacity on every retry.
    quarantined_segments: parking_lot::Mutex<HashSet<String>>,

    /// Generic merge failures pause scheduling briefly. Source-specific open
    /// failures use `quarantined_segments` instead so healthy work can continue.
    merge_retry: parking_lot::Mutex<MergeRetryState>,

    /// Per-source backoff for non-deterministic standalone reorder failures.
    /// Optimizer scans are periodic, but a pass can outlast the scan interval;
    /// without completion-based backoff it would restart almost immediately.
    reorder_retries: parking_lot::Mutex<HashMap<String, MergeRetryState>>,

    /// In-flight merge JoinHandles — supports multiple concurrent merges.
    merge_handles: parking_lot::Mutex<Vec<JoinHandle<()>>>,

    /// At most one task per index waits for application-wide merge capacity.
    /// Without this wakeup, an index denied by another index can remain idle
    /// forever when no later commit happens to re-run merge policy evaluation.
    global_merge_wakeup_pending: AtomicBool,

    /// Auxiliary lifecycle tasks: metadata transactions, deferred deletes,
    /// and capacity wakeups. Handles registered here are drained before index
    /// removal.
    lifecycle_handles: Arc<parking_lot::Mutex<Vec<JoinHandle<()>>>>,

    /// Trained vector structures — lock-free reads via ArcSwap.
    trained: ArcSwapOption<TrainedVectorStructures>,

    /// Reference counting for safe segment deletion (sync Mutex for Drop).
    tracker: Arc<SegmentTracker>,

    /// Cached deletion callback for snapshots (avoids allocation per acquire_snapshot).
    delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync>,

    /// Directory for segment I/O
    directory: Arc<D>,
    /// Schema for segment operations
    schema: Arc<crate::dsl::Schema>,
    /// Term cache blocks for segment readers during merge
    term_cache_blocks: usize,
    /// Hard concurrency limit for background merges. A semaphore permit is
    /// acquired before lifecycle ownership, closing the old handle-count race
    /// where concurrent schedulers could exceed the configured maximum.
    merge_permits: Arc<Semaphore>,
    /// Application-wide merge limit shared across index managers.
    global_merge_permits: Arc<Semaphore>,
    /// Shared across every index opened from the same `IndexConfig`. This
    /// bounds whole BP rewrites (optimizer + merge-time + manual) separately
    /// from Rayon thread width, preventing N × memory-budget amplification.
    reorder_permits: Arc<Semaphore>,
    /// Run BP reordering of `reorder`-attributed BMP fields inside merges.
    /// Persisted index configuration (schema-level `reorder_on_merge: true`
    /// in SDL); merged segments are marked `reordered` and skipped by the
    /// standalone optimizer pass.
    reorder_on_merge: bool,
    /// Wall-clock budget for merge-time BP (from `IndexConfig`); truncated
    /// passes mark the merged segment `bp_converged = false` so the
    /// background optimizer deepens it later (warm-started).
    merge_bp_time_budget: Option<std::time::Duration>,
    /// Memory budget for the BP forward index (merge-time and background
    /// reorder). Over-budget passes drop highest-df dims, logged loudly.
    bp_memory_budget_bytes: usize,
    /// Application-owned shared pool, when configured. This is the server
    /// path and ensures optimizer and merge-time work use the same threads.
    background_reorder_pool: Option<Arc<rayon::ThreadPool>>,
}

impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    /// Create a new segment manager with existing metadata
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        directory: Arc<D>,
        schema: Arc<crate::dsl::Schema>,
        metadata: IndexMetadata,
        merge_policy: Box<dyn MergePolicy>,
        term_cache_blocks: usize,
        max_concurrent_merges: usize,
        global_merge_permits: Arc<Semaphore>,
        merge_bp_time_budget: Option<std::time::Duration>,
        bp_memory_budget_bytes: usize,
        reorder_permits: Arc<Semaphore>,
        background_reorder_pool: Option<Arc<rayon::ThreadPool>>,
    ) -> Self {
        // Persisted index option: set via `reorder_on_merge: true` in the SDL
        // at index creation. Absent = disabled (merges block-copy).
        let reorder_on_merge = schema.reorder_on_merge();
        if reorder_on_merge {
            log::info!("[merge] reorder-on-merge enabled by index schema");
        }

        let tracker = Arc::new(SegmentTracker::new());
        for seg_id in metadata.segment_metas.keys() {
            tracker.register(seg_id);
        }

        let lifecycle_handles: Arc<parking_lot::Mutex<Vec<JoinHandle<()>>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync> = {
            let dir = Arc::clone(&directory);
            let tracker = Arc::clone(&tracker);
            let lifecycle_handles = Arc::clone(&lifecycle_handles);
            Arc::new(move |segment_ids| {
                // Guard: if the tokio runtime is gone (program exit), skip async
                // deletion. Segment files become orphans cleaned up on next startup.
                let Ok(handle) = tokio::runtime::Handle::try_current() else {
                    // Release in-process protection as well: if the process is
                    // still alive, a later sweep must be able to retry.
                    tracker.complete_deletion(&segment_ids);
                    return;
                };
                let dir = Arc::clone(&dir);
                let task_tracker = Arc::clone(&tracker);
                let cleanup_ids = segment_ids.clone();
                let future = async move {
                    for &segment_id in &segment_ids {
                        log::info!(
                            "[segment_cleanup] deleting deferred segment {}",
                            segment_id.to_hex()
                        );
                        if let Err(error) =
                            crate::segment::delete_segment(dir.as_ref(), segment_id).await
                        {
                            log::warn!(
                                "[segment_cleanup] deferred delete failed for {}: {}",
                                segment_id.to_hex(),
                                error,
                            );
                        }
                    }
                    task_tracker.complete_deletion(&segment_ids);
                };
                if !try_spawn_lifecycle(&lifecycle_handles, &handle, future) {
                    // Spawning can fail only during runtime teardown. Release
                    // the scheduled-deletion claim so an in-process sweep can
                    // retry; crash recovery handles a process exit.
                    tracker.complete_deletion(&cleanup_ids);
                    log::warn!(
                        "[segment_cleanup] runtime rejected deferred deletion; files will be swept later"
                    );
                }
            })
        };

        Self {
            state: Arc::new(AsyncMutex::new(ManagerState {
                metadata,
                merge_policy,
            })),
            active_operations: Arc::new(ActiveSegmentOperations::new()),
            quarantined_segments: parking_lot::Mutex::new(HashSet::new()),
            merge_retry: parking_lot::Mutex::new(MergeRetryState::default()),
            reorder_retries: parking_lot::Mutex::new(HashMap::new()),
            merge_handles: parking_lot::Mutex::new(Vec::new()),
            global_merge_wakeup_pending: AtomicBool::new(false),
            lifecycle_handles,
            trained: ArcSwapOption::new(None),
            tracker,
            delete_fn,
            directory,
            schema,
            term_cache_blocks,
            merge_permits: Arc::new(Semaphore::new(max_concurrent_merges.max(1))),
            global_merge_permits,
            reorder_permits,
            reorder_on_merge,
            merge_bp_time_budget,
            bp_memory_budget_bytes,
            background_reorder_pool,
        }
    }

    /// Bounded rayon pool for background CPU (merge-time BP, manual reorder).
    /// Query scoring uses the global rayon pool; keeping background BP off it
    /// prevents a large merge from queueing every search behind gain passes.
    pub fn background_cpu_pool(&self) -> Arc<rayon::ThreadPool> {
        if let Some(pool) = &self.background_reorder_pool {
            return Arc::clone(pool);
        }
        Arc::clone(BACKGROUND_CPU_POOL.get_or_init(|| {
            let threads = (num_cpus::get() / 2).max(1);
            log::info!(
                "[merge] process-wide background CPU pool: {} thread(s)",
                threads
            );
            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("hermes-bg-cpu-{}", i))
                    .build()
                    .expect("failed to build background CPU pool"),
            )
        }))
    }

    /// Stop new indexing/merge/reorder operations from claiming segment IDs.
    /// Used as the first half of index deletion; the writer then joins its
    /// workers before [`Self::wait_for_shutdown`] drains remaining ownership.
    pub fn begin_shutdown(&self) {
        self.active_operations.stop_accepting();
    }

    /// Run a lifecycle mutation independently of its requesting future.
    ///
    /// Metadata writes contain an atomic rename. If an RPC is cancelled while
    /// awaiting that I/O, dropping the request must not abandon the matching
    /// in-memory/tracker transition. The spawned transaction is tracked for
    /// index shutdown; the oneshot only reports its result to a caller that is
    /// still interested.
    async fn run_lifecycle_transaction<T, F>(&self, transaction: F) -> Result<T>
    where
        T: Send + 'static,
        F: std::future::Future<Output = Result<T>> + Send + 'static,
    {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let future = async move {
            let result = transaction.await;
            let _ = result_tx.send(result);
        };
        let runtime = tokio::runtime::Handle::current();
        if !try_spawn_lifecycle(&self.lifecycle_handles, &runtime, future) {
            return Err(Error::Internal(
                "runtime rejected lifecycle metadata transaction".into(),
            ));
        }
        result_rx.await.map_err(|_| {
            Error::Internal("lifecycle metadata transaction terminated unexpectedly".into())
        })?
    }

    /// Arm unwind cleanup for an output that is not visible in metadata yet.
    fn output_cleanup_guard(self: &Arc<Self>, output_id: SegmentId) -> OutputCleanupGuard {
        let manager = Arc::clone(self);
        let cleanup: Arc<dyn Fn(SegmentId) + Send + Sync> = Arc::new(move |segment_id| {
            let Ok(handle) = tokio::runtime::Handle::try_current() else {
                log::warn!(
                    "[segment_cleanup] runtime unavailable; partial output {} will be swept on startup",
                    segment_id.to_hex(),
                );
                return;
            };

            let cleanup_manager = Arc::clone(&manager);
            let future = async move {
                cleanup_manager
                    .delete_output_if_unregistered(segment_id, "task unwind")
                    .await;
            };
            if !try_spawn_lifecycle(&manager.lifecycle_handles, &handle, future) {
                log::warn!(
                    "[segment_cleanup] runtime rejected output cleanup; {} will be swept on startup",
                    segment_id.to_hex(),
                );
            }
        });

        OutputCleanupGuard::new(output_id, cleanup)
    }

    /// Delete an abandoned indexing output while retaining its lifecycle
    /// claim until the last file operation completes. The explicit runtime
    /// handle makes this safe from dedicated indexing OS threads, which are
    /// outside Tokio's entered context.
    pub(crate) fn schedule_unpublished_segment_cleanup(
        self: &Arc<Self>,
        output_id: SegmentId,
        operation: SegmentOperationGuard,
        runtime: tokio::runtime::Handle,
    ) {
        let manager = Arc::clone(self);
        let output_hex = output_id.to_hex();
        let future = async move {
            manager
                .delete_output_if_unregistered(output_id, "indexing abort or failure")
                .await;
            drop(operation);
        };
        if !try_spawn_lifecycle(&self.lifecycle_handles, &runtime, future) {
            // The dropped future releases operation ownership. Startup sweep
            // handles its output if the runtime is already tearing down.
            log::warn!(
                "[segment_cleanup] runtime unavailable; indexing output {} will be swept on startup",
                output_hex,
            );
        }
    }

    /// Claim a newly generated indexing segment before its first file write.
    ///
    /// The returned guard must travel with the built segment until metadata
    /// publication or abort. UUID collisions are treated as corruption rather
    /// than silently sharing lifecycle ownership.
    pub(crate) fn protect_new_segment(&self, segment_id: String) -> Result<SegmentOperationGuard> {
        match self
            .active_operations
            .try_register(vec![segment_id.clone()])
        {
            Some(operation) => Ok(operation),
            None if !self.active_operations.is_accepting() => Err(Error::IndexClosed),
            None => Err(Error::Corruption(format!(
                "new segment ID {} is already owned by an active operation",
                segment_id
            ))),
        }
    }

    /// Validate the small, mandatory core of a completed segment before it can
    /// become metadata-live. Optional vector/sparse/position/fast files are
    /// schema- and data-dependent and are validated by `SegmentReader` when used.
    async fn validate_completed_segment(&self, segment_id: &str, expected_docs: u32) -> Result<()> {
        let id = SegmentId::from_hex(segment_id).ok_or_else(|| {
            Error::Corruption(format!("invalid completed segment ID: {}", segment_id))
        })?;
        let files = SegmentFiles::new(id.0);

        for path in files.mandatory_paths() {
            if !self.directory.exists(path).await.map_err(Error::Io)? {
                return Err(Error::Corruption(format!(
                    "segment {} cannot be published: mandatory file {:?} is missing",
                    segment_id, path
                )));
            }
        }

        let meta_slice = self.directory.open_read(&files.meta).await.map_err(|e| {
            Error::Corruption(format!(
                "segment {} cannot be published: missing/unreadable {:?}: {}",
                segment_id, files.meta, e
            ))
        })?;
        let meta_bytes = meta_slice.read_bytes().await.map_err(|e| {
            Error::Corruption(format!(
                "segment {} cannot be published: failed reading {:?}: {}",
                segment_id, files.meta, e
            ))
        })?;
        let meta = SegmentMeta::deserialize(meta_bytes.as_slice()).map_err(|e| {
            Error::Corruption(format!(
                "segment {} cannot be published: invalid {:?}: {}",
                segment_id, files.meta, e
            ))
        })?;

        if meta.id != id.0 || meta.num_docs != expected_docs {
            return Err(Error::Corruption(format!(
                "segment {} cannot be published: metadata identity/docs mismatch \
                 (id={:032x}, docs={}, expected_docs={})",
                segment_id, meta.id, meta.num_docs, expected_docs
            )));
        }

        Ok(())
    }

    fn quarantine_segment(&self, segment_id: &str, error: &Error) {
        let inserted = self
            .quarantined_segments
            .lock()
            .insert(segment_id.to_string());
        if inserted {
            log::error!(
                "[merge] quarantined metadata-live segment {} after deterministic source/validation failure: {}. \
                 It remains metadata-live for explicit repair but is excluded from merges until restart",
                segment_id,
                error,
            );
        }
    }

    fn pause_merge_retries(&self, error: &Error) -> std::time::Duration {
        let mut retry = self.merge_retry.lock();
        retry.consecutive_failures = retry.consecutive_failures.saturating_add(1);
        let delay = merge_retry_delay(retry.consecutive_failures);
        retry.retry_after = std::time::Instant::now().checked_add(delay);
        log::warn!(
            "[merge] pausing background merge scheduling for {:.0}s after consecutive failure #{}: {}",
            delay.as_secs_f64(),
            retry.consecutive_failures,
            error,
        );
        delay
    }

    fn clear_merge_retry_backoff(&self) {
        *self.merge_retry.lock() = MergeRetryState::default();
    }

    fn merge_retry_is_paused(&self) -> bool {
        let mut retry = self.merge_retry.lock();
        match retry.retry_after {
            Some(deadline) if deadline > std::time::Instant::now() => true,
            Some(_) => {
                retry.retry_after = None;
                false
            }
            None => false,
        }
    }

    fn pause_reorder_retries(&self, segment_id: &str, error: &Error) {
        let mut retries = self.reorder_retries.lock();
        let retry = retries.entry(segment_id.to_string()).or_default();
        retry.consecutive_failures = retry.consecutive_failures.saturating_add(1);
        let delay = merge_retry_delay(retry.consecutive_failures);
        retry.retry_after = std::time::Instant::now().checked_add(delay);
        log::warn!(
            "[reorder] pausing optimizer retries for segment {} for {:.0}s after failure #{}: {}",
            segment_id,
            delay.as_secs_f64(),
            retry.consecutive_failures,
            error,
        );
    }

    fn clear_reorder_retry(&self, segment_id: &str) {
        self.reorder_retries.lock().remove(segment_id);
    }

    fn paused_reorder_segments(&self) -> HashSet<String> {
        let now = std::time::Instant::now();
        let mut retries = self.reorder_retries.lock();
        let mut paused = HashSet::new();
        for (segment_id, retry) in retries.iter_mut() {
            match retry.retry_after {
                Some(deadline) if deadline > now => {
                    paused.insert(segment_id.clone());
                }
                Some(_) => retry.retry_after = None,
                None => {}
            }
        }
        paused
    }

    /// Re-evaluate this index when another index releases application-wide
    /// merge capacity. The atomic flag bounds this to one waiter per index and
    /// the tracked handle makes index shutdown drain it deterministically.
    fn schedule_global_merge_wakeup(self: &Arc<Self>) {
        if self
            .global_merge_wakeup_pending
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let manager = Arc::clone(self);
        let future = async move {
            let capacity = tokio::select! {
                biased;
                () = manager.active_operations.wait_for_shutdown() => None,
                permit = Arc::clone(&manager.global_merge_permits).acquire_owned() => permit.ok(),
            };

            manager
                .global_merge_wakeup_pending
                .store(false, Ordering::Release);
            if let Some(permit) = capacity {
                // This task is only a notification. The normal scheduler must
                // acquire both global and per-index permits atomically enough
                // for its own candidate selection.
                drop(permit);
                manager.maybe_merge().await;
            }
        };
        let runtime = tokio::runtime::Handle::current();
        if !try_spawn_lifecycle(&self.lifecycle_handles, &runtime, future) {
            self.global_merge_wakeup_pending
                .store(false, Ordering::Release);
            log::warn!("[merge] runtime rejected global-capacity wakeup task");
        }
    }

    #[cfg(test)]
    pub(crate) fn is_segment_quarantined(&self, segment_id: &str) -> bool {
        self.quarantined_segments.lock().contains(segment_id)
    }

    /// Delete a failed output only if metadata did not make it live.
    ///
    /// Rechecking under `state` also makes unwind cleanup safe if it races
    /// successful publication of the same output.
    async fn delete_output_if_unregistered(&self, output_id: SegmentId, reason: &str) {
        let output_hex = output_id.to_hex();
        {
            let st = self.state.lock().await;
            if st.metadata.has_segment(&output_hex) {
                return;
            }
        }

        // UUIDs are generated per producer and cannot be adopted by another
        // publisher after this check. Never hold the metadata mutex while a
        // multi-GB filesystem deletion runs.
        log::info!(
            "[segment_cleanup] deleting uncommitted output {} after {}",
            output_hex,
            reason,
        );
        if let Err(error) = crate::segment::delete_segment(self.directory.as_ref(), output_id).await
        {
            log::warn!(
                "[segment_cleanup] failed deleting uncommitted output {}: {}",
                output_hex,
                error,
            );
        }
    }

    // ========================================================================
    // Read path (brief lock or lock-free)
    // ========================================================================

    /// Get the current segment IDs
    pub async fn get_segment_ids(&self) -> Vec<String> {
        self.state.lock().await.metadata.segment_ids()
    }

    /// Get trained vector structures (lock-free via ArcSwap)
    pub fn trained(&self) -> Option<Arc<TrainedVectorStructures>> {
        self.trained.load_full()
    }

    /// Load trained structures from disk and publish to ArcSwap.
    /// Copies metadata under lock, releases lock, then does disk I/O.
    pub async fn load_and_publish_trained(&self) {
        // Copy vector_fields under lock (cheap clone of HashMap<u32, FieldMeta>)
        let vector_fields = {
            let st = self.state.lock().await;
            st.metadata.vector_fields.clone()
        };
        // Disk I/O happens WITHOUT holding the state lock
        let trained =
            IndexMetadata::load_trained_from_fields(&vector_fields, self.directory.as_ref()).await;
        if let Some(t) = trained {
            self.trained.store(Some(Arc::new(t)));
        }
    }

    /// Clear trained structures (sets ArcSwap to None)
    pub(crate) fn clear_trained(&self) {
        self.trained.store(None);
    }

    /// Read metadata with a closure (no persist)
    pub(crate) async fn read_metadata<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&IndexMetadata) -> R,
    {
        let st = self.state.lock().await;
        f(&st.metadata)
    }

    /// Update metadata with a closure and persist atomically
    pub(crate) async fn update_metadata<F>(self: &Arc<Self>, f: F) -> Result<()>
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut st = Arc::clone(&self.state).lock_owned().await;
        let mut next = st.metadata.clone();
        f(&mut next);
        let directory = Arc::clone(&self.directory);
        self.run_lifecycle_transaction(async move {
            next.save(directory.as_ref()).await?;
            st.metadata = next;
            Ok(())
        })
        .await
    }

    /// Acquire a snapshot of current segments for reading.
    /// The snapshot holds references — segments won't be deleted while snapshot exists.
    pub async fn acquire_snapshot(&self) -> SegmentSnapshot {
        let acquired = {
            let st = self.state.lock().await;
            let segment_ids = st.metadata.segment_ids();
            self.tracker.acquire(&segment_ids)
        };

        SegmentSnapshot::with_delete_fn(
            Arc::clone(&self.tracker),
            acquired,
            Arc::clone(&self.delete_fn),
        )
    }

    /// Get the segment tracker
    pub fn tracker(&self) -> Arc<SegmentTracker> {
        Arc::clone(&self.tracker)
    }

    /// Get the directory
    pub fn directory(&self) -> Arc<D> {
        Arc::clone(&self.directory)
    }
}

// ============================================================================
// Native-only: commit, merging, force_merge
// ============================================================================

#[cfg(feature = "native")]
impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    /// Atomic commit: register new segments + persist metadata.
    pub async fn commit(self: &Arc<Self>, new_segments: &[(String, u32)]) -> Result<()> {
        // Indexing guards still own these IDs here, so the orphan sweeper
        // cannot remove files between validation and metadata publication.
        for (segment_id, num_docs) in new_segments {
            self.validate_completed_segment(segment_id, *num_docs)
                .await?;
        }

        let mut st = Arc::clone(&self.state).lock_owned().await;
        let mut next = st.metadata.clone();
        let mut added = Vec::new();
        for (segment_id, num_docs) in new_segments {
            if !next.has_segment(segment_id) {
                next.add_segment(segment_id.clone(), *num_docs);
                added.push(segment_id.clone());
            }
        }

        // Durable-before-visible: a save failure leaves both in-memory metadata
        // and tracker unchanged, so callers can retry the prepared commit.
        // The tracked transaction continues if the requesting RPC is cancelled;
        // unpublished cleanup waits on this owned state guard before deciding
        // whether the files became metadata-live.
        let directory = Arc::clone(&self.directory);
        let tracker = Arc::clone(&self.tracker);
        self.run_lifecycle_transaction(async move {
            next.save(directory.as_ref()).await?;
            for segment_id in &added {
                tracker.register(segment_id);
            }
            st.metadata = next;
            Ok(())
        })
        .await
    }

    /// Evaluate merge policy and spawn background merges for all eligible candidates.
    ///
    /// **Atomicity**: The entire filter → find_merges → spawn_merge sequence runs
    /// under the `state` lock to prevent a TOCTOU race where concurrent callers
    /// both see segments as eligible before either claims operation ownership.
    /// `spawn_merge` is non-blocking (just `try_register` + `tokio::spawn`), so
    /// holding the state lock through it is safe and sub-microsecond.
    ///
    /// The hard merge semaphore is acquired before lifecycle ownership, so
    /// concurrent triggers cannot exceed configured merge capacity.
    pub async fn maybe_merge(self: &Arc<Self>) {
        if !self.active_operations.is_accepting() {
            log::debug!("[maybe_merge] manager is shutting down, skipping");
            return;
        }
        if self.merge_retry_is_paused() {
            log::debug!("[maybe_merge] retry backoff active, skipping");
            return;
        }

        // Finished handles no longer need to be retained. Concurrency itself
        // is enforced by `merge_permits`, not this bookkeeping vector.
        {
            let mut handles = self.merge_handles.lock();
            handles.retain(|h| !h.is_finished());
        }
        let local_slots = self.merge_permits.available_permits();
        let global_slots = self.global_merge_permits.available_permits();
        let slots_available = local_slots.min(global_slots);

        // Hold state lock through spawn_merge to make filter + register atomic.
        // This closes the TOCTOU window where concurrent maybe_merge calls could
        // both see the same segments as eligible before either registers them.
        let new_handles = {
            let st = self.state.lock().await;
            let quarantined = self.quarantined_segments.lock().clone();
            let active_ids = self.active_operations.snapshot();

            // Exclude segments owned by another operation, pending retirement,
            // or quarantined after a persistent open/validation failure.
            let segments: Vec<SegmentInfo> = st
                .metadata
                .segment_metas
                .iter()
                .filter(|(id, _)| {
                    !self.tracker.is_pending_deletion(id)
                        && !active_ids.contains(*id)
                        && !quarantined.contains(*id)
                })
                .map(|(id, info)| SegmentInfo {
                    id: id.clone(),
                    num_docs: info.num_docs,
                })
                .collect();

            log::debug!("[maybe_merge] {} eligible segments", segments.len());

            let candidates = st.merge_policy.find_merges(&segments);

            if candidates.is_empty() {
                return;
            }

            // Register a capacity waiter only for an index that actually has
            // eligible work. Scheduling one waiter for every idle index while
            // the process gate was full caused an avoidable wakeup stampede.
            if slots_available == 0 {
                if local_slots > 0 && global_slots == 0 {
                    self.schedule_global_merge_wakeup();
                }
                log::debug!("[maybe_merge] at max concurrent merges, skipping");
                return;
            }

            log::debug!(
                "[maybe_merge] {} merge candidates, {} slots available",
                candidates.len(),
                slots_available
            );

            let mut handles = Vec::new();
            for c in candidates {
                if handles.len() >= slots_available {
                    break;
                }
                if let Some(h) = self.spawn_merge(c.segment_ids) {
                    handles.push(h);
                }
            }
            handles
            // State lock released after spawn_merge claimed operation ownership.
        };

        if !new_handles.is_empty() {
            // Synchronous insertion is part of spawning: there must be no
            // cancellation point where a live task exists but shutdown and
            // force-merge draining cannot see its JoinHandle.
            self.merge_handles.lock().extend(new_handles);
        }
    }

    /// Spawn a background merge task with RAII tracking.
    ///
    /// Pre-generates the output segment ID. The operation guard registers all segment IDs
    /// (old + output) in `active_operations`. When the task ends (success, failure, or
    /// panic), the guard drops and segments are automatically unregistered.
    ///
    /// On completion, the task auto-triggers `maybe_merge` to evaluate cascading merges.
    /// Returns the JoinHandle if the merge was spawned, None if it was skipped.
    fn spawn_merge(self: &Arc<Self>, segment_ids_to_merge: Vec<String>) -> Option<JoinHandle<()>> {
        let global_merge_permit = match Arc::clone(&self.global_merge_permits).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                log::debug!("[spawn_merge] skipped: global merge capacity is full");
                self.schedule_global_merge_wakeup();
                return None;
            }
        };
        let merge_permit = match Arc::clone(&self.merge_permits).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                log::debug!("[spawn_merge] skipped: no merge permit available");
                return None;
            }
        };
        let output_id = SegmentId::new();
        let output_hex = output_id.to_hex();

        let mut all_ids = segment_ids_to_merge.clone();
        all_ids.push(output_hex);

        let guard = match self.active_operations.try_register(all_ids) {
            Some(g) => g,
            None => {
                log::debug!("[spawn_merge] skipped: segments overlap with an active operation");
                return None;
            }
        };

        let sm = Arc::clone(self);
        let ids = segment_ids_to_merge;

        Some(tokio::spawn(async move {
            let mut output_cleanup = sm.output_cleanup_guard(output_id);
            let mut reevaluate = false;
            let mut retry_delay = None;

            let trained_snap = sm.trained();
            let granularity = sm.merge_granularity(&ids).await;
            let result = Self::do_merge(
                sm.directory.as_ref(),
                &sm.schema,
                &ids,
                output_id,
                sm.term_cache_blocks,
                trained_snap.as_deref(),
                sm.reorder_on_merge,
                granularity,
                sm.merge_bp_time_budget,
                sm.bp_memory_budget_bytes,
                Arc::clone(&sm.reorder_permits),
                Some(sm.background_cpu_pool()),
            )
            .await;

            match result {
                Ok((new_id, doc_count, bp_converged)) => {
                    match sm
                        .replace_segments(
                            &ids,
                            new_id,
                            doc_count,
                            sm.reorder_on_merge,
                            bp_converged,
                        )
                        .await
                    {
                        Ok(()) => {
                            output_cleanup.disarm();
                            sm.clear_merge_retry_backoff();
                            reevaluate = true;
                        }
                        Err(e) => {
                            sm.delete_output_if_unregistered(output_id, "replacement failure")
                                .await;
                            output_cleanup.disarm();
                            retry_delay = Some(sm.pause_merge_retries(&e));
                            log::error!("[merge] failed to publish merged segment: {}", e);
                        }
                    }
                }
                Err(MergeTaskError {
                    error,
                    unavailable_segments,
                }) => {
                    log::error!(
                        "[merge] background merge failed for segments {:?}: {}",
                        ids,
                        error
                    );
                    if !unavailable_segments.is_empty() {
                        for segment_id in &unavailable_segments {
                            sm.quarantine_segment(segment_id, &error);
                        }
                        // Recompute without this known-bad input. This is not a
                        // retry of the same candidate because policy filtering
                        // excludes every quarantined ID.
                        reevaluate = true;
                    } else {
                        retry_delay = Some(sm.pause_merge_retries(&error));
                    }
                    sm.delete_output_if_unregistered(output_id, "merge failure")
                        .await;
                    output_cleanup.disarm();
                }
            }
            // Release source/output ownership before re-evaluating policy, so
            // the completed operation cannot artificially hide candidates.
            drop(guard);
            // A failed merge must not reserve capacity during its retry delay.
            drop(merge_permit);
            drop(global_merge_permit);

            if reevaluate {
                sm.maybe_merge().await;
            } else if let Some(retry_delay) = retry_delay {
                // A backoff without a wakeup can strand eligible segments
                // forever when no later commit happens. Keep this sleep inside
                // the tracked merge task so shutdown can await it safely.
                tokio::select! {
                    () = tokio::time::sleep(retry_delay) => {
                        sm.maybe_merge().await;
                    }
                    () = sm.active_operations.wait_for_shutdown() => {}
                }
            }
        }))
    }

    /// Atomically replace old segments with a new merged segment.
    /// Computes merge generation as max(parent gens) + 1 and records ancestors.
    /// `reordered` marks whether the new segment was BP-reordered.
    async fn replace_segments(
        self: &Arc<Self>,
        old_ids: &[String],
        new_id: String,
        doc_count: u32,
        reordered: bool,
        bp_converged: bool,
    ) -> Result<()> {
        // The operation guard owns the output during validation. Publication
        // below replaces that ownership with metadata + tracker atomically.
        self.validate_completed_segment(&new_id, doc_count).await?;

        let mut st = Arc::clone(&self.state).lock_owned().await;
        // Every source must still be live: callers hold operation ownership,
        // so a missing source means a stale merge/reorder whose input was
        // already replaced. Adding the output would duplicate its documents.
        let missing: Vec<&String> = old_ids
            .iter()
            .filter(|id| !st.metadata.has_segment(id))
            .collect();
        if !missing.is_empty() {
            return Err(Error::Corruption(format!(
                "replace_segments: source segment(s) {:?} not in metadata — \
                 refusing to add output {} (would duplicate documents)",
                missing, new_id
            )));
        }

        let parent_generation = old_ids
            .iter()
            .filter_map(|id| st.metadata.segment_metas.get(id))
            .map(|info| info.generation)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::Corruption("merge generation exceeds u32::MAX".into()))?;
        let retired_ids = old_ids.to_vec();
        let mut next = st.metadata.clone();
        for id in old_ids {
            next.remove_segment(id);
        }
        next.add_merged_segment(
            new_id.clone(),
            doc_count,
            retired_ids.clone(),
            parent_generation,
            reordered,
            bp_converged,
        );

        let directory = Arc::clone(&self.directory);
        let tracker = Arc::clone(&self.tracker);
        self.run_lifecycle_transaction(async move {
            // Durable-before-visible. If persistence fails, old metadata and
            // tracker ownership stay intact and source deletion is never armed.
            next.save(directory.as_ref()).await?;
            tracker.register(&new_id);
            st.metadata = next;

            // Keep state locked until retired sources enter the tracker. The
            // transaction itself also performs deletion, so cancellation of
            // the requesting merge cannot strand pending-deletion ownership.
            let ready_to_delete = tracker.mark_for_deletion(&retired_ids);
            drop(st);
            for &segment_id in &ready_to_delete {
                if let Err(error) =
                    crate::segment::delete_segment(directory.as_ref(), segment_id).await
                {
                    log::warn!(
                        "[segment_cleanup] immediate delete failed for {}: {}",
                        segment_id.to_hex(),
                        error,
                    );
                }
            }
            tracker.complete_deletion(&ready_to_delete);
            Ok(())
        })
        .await
    }

    /// Perform the actual merge operation (pure function — no shared state access).
    /// `output_segment_id` is pre-generated by the caller so active-operation ownership
    /// is installed before any output file is written.
    /// Returns (new_segment_id_hex, total_doc_count).
    #[allow(clippy::too_many_arguments)]
    async fn do_merge(
        directory: &D,
        schema: &Arc<crate::dsl::Schema>,
        segment_ids_to_merge: &[String],
        output_segment_id: SegmentId,
        term_cache_blocks: usize,
        trained: Option<&TrainedVectorStructures>,
        reorder_bmp: bool,
        granularity: crate::segment::reorder::BpGranularity,
        merge_bp_time_budget: Option<std::time::Duration>,
        bp_memory_budget_bytes: usize,
        reorder_permits: Arc<Semaphore>,
        bg_cpu_pool: Option<Arc<rayon::ThreadPool>>,
    ) -> MergeTaskResult<(String, u32, bool)> {
        let output_hex = output_segment_id.to_hex();
        let load_start = std::time::Instant::now();

        let mut segment_ids = Vec::with_capacity(segment_ids_to_merge.len());
        for id_str in segment_ids_to_merge {
            let id = SegmentId::from_hex(id_str).ok_or_else(|| {
                MergeTaskError::source(
                    id_str.clone(),
                    Error::Corruption(format!("Invalid segment ID: {}", id_str)),
                )
            })?;
            segment_ids.push(id);
        }

        // Cheap fail-fast before opening every reader. `join_all` otherwise
        // waits for all healthy multi-GB inputs to load even when one source's
        // `.meta` is already absent, turning a known-corrupt candidate into a
        // large CPU/IO spike before it can be quarantined.
        let mut unavailable_sources = Vec::new();
        let mut missing_files = Vec::new();
        for (id_str, id) in segment_ids_to_merge.iter().zip(&segment_ids) {
            let files = SegmentFiles::new(id.0);
            let mut source_unavailable = false;
            for path in files.mandatory_paths() {
                let exists = directory
                    .exists(path)
                    .await
                    .map_err(|error| MergeTaskError::from(Error::Io(error)))?;
                if !exists {
                    source_unavailable = true;
                    missing_files.push(format!("{}:{:?}", id_str, path));
                }
            }
            if source_unavailable {
                unavailable_sources.push(id_str.clone());
            }
        }
        if !unavailable_sources.is_empty() {
            return Err(MergeTaskError::sources(
                unavailable_sources,
                Error::Corruption(format!(
                    "merge sources are missing mandatory files: {}",
                    missing_files.join(", ")
                )),
            ));
        }

        let schema_arc = Arc::clone(schema);
        let futures: Vec<_> = segment_ids
            .iter()
            .map(|&sid| {
                let sch = Arc::clone(&schema_arc);
                async move { SegmentReader::open(directory, sid, sch, term_cache_blocks).await }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut readers = Vec::with_capacity(results.len());
        let mut total_docs = 0u64;
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(r) => {
                    total_docs += r.meta().num_docs as u64;
                    readers.push(r);
                }
                Err(e) => {
                    log::error!(
                        "[merge] Failed to open segment {}: {:?}",
                        segment_ids_to_merge[i],
                        e
                    );
                    return Err(classify_source_error(segment_ids_to_merge[i].clone(), e));
                }
            }
        }
        if total_docs > u32::MAX as u64 {
            return Err(Error::Internal(format!(
                "Merged segment doc count ({}) exceeds u32::MAX",
                total_docs
            ))
            .into());
        }

        // Pre-merge validation: verify each source segment's store doc count
        // matches its metadata. Catching mismatches early avoids building a
        // corrupted merged segment and leaving orphan files on disk.
        for (i, reader) in readers.iter().enumerate() {
            let meta_docs = reader.meta().num_docs;
            let store_docs = reader.store().num_docs();
            if store_docs != meta_docs {
                return Err(MergeTaskError::source(
                    segment_ids_to_merge[i].clone(),
                    Error::Corruption(format!(
                        "pre-merge validation: segment {} store has {} docs but meta says {}",
                        segment_ids_to_merge[i], store_docs, meta_docs
                    )),
                ));
            }
        }

        log::info!(
            "[merge] loaded {} segment readers in {:.1}s",
            readers.len(),
            load_start.elapsed().as_secs_f64()
        );

        let merger = SegmentMerger::new(Arc::clone(schema))
            .with_bmp_reorder(reorder_bmp)
            .with_granularity(granularity)
            .with_bp_budget(crate::segment::BpBudget {
                min_partition_docs: None,
                time_budget: merge_bp_time_budget,
            })
            .with_bp_memory_budget(bp_memory_budget_bytes)
            .with_reorder_permits(reorder_permits)
            .with_background_pool(bg_cpu_pool);

        log::info!(
            "[merge] {} segments -> {} (trained={})",
            segment_ids_to_merge.len(),
            output_hex,
            trained.map_or(0, |t| t.centroids.len()),
        );

        let (_merged_meta, merge_stats) = merger
            .merge(directory, &readers, output_segment_id, trained)
            .await
            .map_err(|error| {
                if matches!(error, Error::Corruption(_) | Error::Serialization(_)) {
                    // The merge has already opened every input successfully;
                    // a structural/serialization failure is deterministic for
                    // this candidate. Attribute all inputs rather than running
                    // the same multi-GB rewrite forever. This is deliberately
                    // not used for I/O errors, which may be transient/output-side.
                    MergeTaskError::sources(segment_ids_to_merge.to_vec(), error)
                } else {
                    MergeTaskError::from(error)
                }
            })?;
        let bp_converged = merge_stats.bp_converged;
        if !bp_converged {
            log::info!(
                "[merge] merge-time BP hit its wall-clock budget — output marked unconverged; \
                 the background optimizer deepens it later",
            );
        }

        log::info!(
            "[merge] total wall-clock: {:.1}s ({} segments, {} docs)",
            load_start.elapsed().as_secs_f64(),
            readers.len(),
            total_docs,
        );

        Ok((output_hex, total_docs as u32, bp_converged))
    }

    /// Drain all in-flight merge tasks safely.
    ///
    /// Tokio cannot abort a `spawn_blocking` closure once it has started. The
    /// old implementation aborted only the async wrapper and returned while
    /// merge-time BP still owned an `OffsetWriter`, allowing index deletion or
    /// orphan cleanup to race a live writer. Awaiting is the only sound generic
    /// behavior until every merge phase supports cooperative cancellation.
    pub async fn abort_merges(&self) {
        loop {
            let handles: Vec<JoinHandle<()>> = { std::mem::take(&mut *self.merge_handles.lock()) };
            if handles.is_empty() {
                return;
            }
            for handle in handles {
                if let Err(error) = handle.await
                    && error.is_panic()
                {
                    log::error!("[merge] background task panicked while draining: {}", error);
                }
            }
        }
    }

    /// Wait for all current in-flight merges to complete.
    pub async fn wait_for_merging_thread(self: &Arc<Self>) {
        let handles: Vec<JoinHandle<()>> = { std::mem::take(&mut *self.merge_handles.lock()) };
        for h in handles {
            let _ = h.await;
        }
    }

    /// Wait for all eligible merges to complete, including cascading merges.
    ///
    /// Drains current handles, then loops. Each completed merge auto-triggers
    /// `maybe_merge` (which pushes new handles) before its JoinHandle resolves,
    /// so by the time `h.await` returns all cascading handles are registered.
    pub async fn wait_for_all_merges(self: &Arc<Self>) {
        loop {
            let handles: Vec<JoinHandle<()>> = { std::mem::take(&mut *self.merge_handles.lock()) };
            if handles.is_empty() {
                break;
            }
            for h in handles {
                let _ = h.await;
            }
        }
    }

    /// Complete the second half of shutdown after the owning `IndexWriter`
    /// has been dropped. This drains tracked merges and then waits for every
    /// remaining guard, including optimizer reorders that are intentionally
    /// launched outside the writer lock.
    pub async fn wait_for_shutdown(self: &Arc<Self>) {
        self.wait_for_all_merges().await;
        self.active_operations.wait_until_idle().await;
        loop {
            let handles = { std::mem::take(&mut *self.lifecycle_handles.lock()) };
            if handles.is_empty() {
                break;
            }
            for handle in handles {
                if let Err(error) = handle.await
                    && error.is_panic()
                {
                    log::error!("[segment_cleanup] task panicked while draining: {}", error);
                }
            }
        }
    }

    /// Force merge segments into the fewest possible segments, respecting
    /// `max_segment_docs` from the merge policy.
    ///
    /// If the policy defines a max segment size, segments are merged in batches
    /// that stay within that limit. Otherwise, all segments are merged into one.
    ///
    /// Each batch is registered in `active_operations` via an RAII guard to prevent
    /// `maybe_merge` from spawning a conflicting background merge.
    pub async fn force_merge(self: &Arc<Self>) -> Result<()> {
        const FORCE_MERGE_BATCH: usize = 64;

        let max_segment_docs = {
            let st = self.state.lock().await;
            st.merge_policy.max_segment_docs()
        };

        // Wait for all in-flight background merges (including cascading)
        // before starting forced merges to avoid try_register conflicts.
        self.wait_for_all_merges().await;

        loop {
            if !self.active_operations.is_accepting() {
                return Err(Error::IndexClosed);
            }
            // Get segment IDs with their doc counts, sorted ascending by size
            let mut segments: Vec<(String, u32)> = {
                let st = self.state.lock().await;
                st.metadata
                    .segment_metas
                    .iter()
                    .map(|(id, info)| (id.clone(), info.num_docs))
                    .collect()
            };

            if segments.len() < 2 {
                return Ok(());
            }

            segments.sort_by_key(|(_, docs)| *docs);

            // Build a batch respecting max_segment_docs
            let max_docs = max_segment_docs.map(|m| m as u64).unwrap_or(u64::MAX);
            let mut batch = Vec::new();
            let mut batch_docs = 0u64;

            for (id, docs) in &segments {
                if batch.len() >= FORCE_MERGE_BATCH {
                    break;
                }
                let next_total = batch_docs + *docs as u64;
                if next_total > max_docs && !batch.is_empty() {
                    break;
                }
                batch.push(id.clone());
                batch_docs += *docs as u64;
            }

            if batch.len() < 2 {
                return Ok(());
            }

            log::info!(
                "[force_merge] merging batch of {} segments ({} docs)",
                batch.len(),
                batch_docs
            );

            let _global_merge_permit = tokio::select! {
                biased;
                () = self.active_operations.wait_for_shutdown() => {
                    return Err(Error::IndexClosed);
                }
                permit = Arc::clone(&self.global_merge_permits).acquire_owned() => {
                    permit.map_err(|_| {
                        Error::Internal("global background merge scheduler is closed".into())
                    })?
                }
            };

            let output_id = SegmentId::new();
            let output_hex = output_id.to_hex();

            // Register batch + output under `state`, matching orphan cleanup's
            // deletion barrier and preventing a stale batch from starting.
            let mut all_ids = batch.clone();
            all_ids.push(output_hex);
            let guard = {
                let st = self.state.lock().await;
                batch
                    .iter()
                    .all(|id| st.metadata.has_segment(id))
                    .then(|| self.active_operations.try_register(all_ids))
                    .flatten()
            };
            let _guard = match guard {
                Some(g) => g,
                None if !self.active_operations.is_accepting() => {
                    return Err(Error::IndexClosed);
                }
                None => {
                    // A background merge slipped in — wait for it, then retry the loop
                    self.wait_for_merging_thread().await;
                    continue;
                }
            };
            let mut output_cleanup = self.output_cleanup_guard(output_id);

            let trained_snap = self.trained();
            let granularity = self.merge_granularity(&batch).await;
            let merge_result = Self::do_merge(
                self.directory.as_ref(),
                &self.schema,
                &batch,
                output_id,
                self.term_cache_blocks,
                trained_snap.as_deref(),
                self.reorder_on_merge,
                granularity,
                self.merge_bp_time_budget,
                self.bp_memory_budget_bytes,
                Arc::clone(&self.reorder_permits),
                Some(self.background_cpu_pool()),
            )
            .await;
            let (new_segment_id, total_docs, bp_converged) = match merge_result {
                Ok(v) => v,
                Err(MergeTaskError {
                    error,
                    unavailable_segments,
                }) => {
                    for segment_id in &unavailable_segments {
                        self.quarantine_segment(segment_id, &error);
                    }
                    self.delete_output_if_unregistered(output_id, "force-merge failure")
                        .await;
                    output_cleanup.disarm();
                    return Err(error);
                }
            };

            if let Err(e) = self
                .replace_segments(
                    &batch,
                    new_segment_id,
                    total_docs,
                    self.reorder_on_merge,
                    bp_converged,
                )
                .await
            {
                self.delete_output_if_unregistered(output_id, "replacement failure")
                    .await;
                output_cleanup.disarm();
                return Err(e);
            }
            output_cleanup.disarm();

            // _guard drops here, releasing operation ownership.
        }
    }

    /// Reorder all segments via Recursive Graph Bisection (BP) for better BMP pruning.
    ///
    /// Each segment is individually rebuilt with reordered BMP blocks.
    /// Non-BMP fields are copied unchanged via streaming file copy.
    ///
    /// Uses active-operation ownership to prevent concurrent work on the same segment.
    pub async fn reorder_segments(self: &Arc<Self>) -> Result<()> {
        self.wait_for_all_merges().await;
        let segment_ids = self.get_segment_ids().await;

        if segment_ids.is_empty() {
            log::info!("[reorder] no segments to reorder");
            return Ok(());
        }

        log::info!("[reorder] reordering {} segments", segment_ids.len());

        for seg_id in segment_ids {
            match self
                .reorder_single_segment(&seg_id, None, crate::segment::BpBudget::full())
                .await
            {
                Ok(true) => {}
                Ok(false) => log::warn!("[reorder] segment {} skipped (in merge)", seg_id),
                Err(e) => return Err(e),
            }
        }

        log::info!("[reorder] all segments reordered");
        Ok(())
    }

    /// Get segment IDs that have not been reordered yet.
    ///
    /// Excludes segments currently involved in a merge or reorder operation
    /// to avoid wasted work (the optimizer would skip them anyway).
    pub async fn unreordered_segment_ids(&self) -> Vec<String> {
        self.unreordered_segments()
            .await
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Segments never reordered, with doc counts — for the optimizer to pick
    /// a size-appropriate BP budget.
    pub async fn unreordered_segments(&self) -> Vec<(String, u32)> {
        let quarantined = self.quarantined_segments.lock().clone();
        let paused = self.paused_reorder_segments();
        let st = self.state.lock().await;
        let active_ids = self.active_operations.snapshot();
        st.metadata
            .segment_metas
            .iter()
            .filter(|(id, info)| {
                !info.reordered
                    && !active_ids.contains(*id)
                    && !quarantined.contains(*id)
                    && !paused.contains(*id)
            })
            .map(|(id, info)| (id.clone(), info.num_docs))
            .collect()
    }

    /// Segments whose last BP pass hit its wall-clock budget before finishing
    /// (`bp_converged == false`). A warm-started follow-up pass deepens the
    /// ordering; the optimizer revisits these at low priority.
    pub async fn unconverged_segments(&self) -> Vec<(String, u32)> {
        let quarantined = self.quarantined_segments.lock().clone();
        let paused = self.paused_reorder_segments();
        let st = self.state.lock().await;
        let active_ids = self.active_operations.snapshot();
        st.metadata
            .segment_metas
            .iter()
            .filter(|(id, info)| {
                info.reordered
                    && !info.bp_converged
                    && !active_ids.contains(*id)
                    && !quarantined.contains(*id)
                    && !paused.contains(*id)
            })
            .map(|(id, info)| (id.clone(), info.num_docs))
            .collect()
    }

    /// Granularity for a BP pass whose sources are `ids`: `Records` when any
    /// source is an unconverged partial reorder, `Auto` otherwise.
    ///
    /// Alignment with the depth budget (docs/block-level-reorder.md): an
    /// unconverged segment is owed a deepening pass, and the output of this
    /// pass will be marked `bp_converged`. `Auto` would measure the partial
    /// pass's residual coherence, potentially take the blockwise path — which
    /// cannot deepen record clustering — and end the cascade at partial
    /// quality. Only record-level BP discharges the debt.
    async fn merge_granularity(&self, ids: &[String]) -> crate::segment::reorder::BpGranularity {
        let st = self.state.lock().await;
        let deepening = ids.iter().any(|id| {
            st.metadata
                .segment_metas
                .get(id)
                .is_some_and(|info| info.reordered && !info.bp_converged)
        });
        drop(st);
        if deepening {
            log::info!(
                "[reorder] source segment(s) unconverged — forcing record-level BP (deepening pass)",
            );
            crate::segment::reorder::BpGranularity::Records
        } else {
            crate::segment::reorder::BpGranularity::Auto
        }
    }

    /// Reorder a single segment via BP. Returns Ok(true) if reordered, Ok(false) if skipped.
    ///
    /// Non-blocking: operation ownership prevents conflicts with background merges.
    /// Copies unchanged files and rebuilds only the sparse file with reordered BMP data.
    pub async fn reorder_single_segment(
        self: &Arc<Self>,
        seg_id: &str,
        rayon_pool: Option<Arc<rayon::ThreadPool>>,
        bp_budget: crate::segment::BpBudget,
    ) -> Result<bool> {
        let source_id = SegmentId::from_hex(seg_id)
            .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", seg_id)))?;
        if self.quarantined_segments.lock().contains(seg_id) {
            return Err(Error::Corruption(format!(
                "segment {} is quarantined after a deterministic source failure; repair it and restart before reordering",
                seg_id
            )));
        }

        // Whole-pass concurrency is independent from Rayon width. One pass
        // can already use every configured BP worker; this permit bounds the
        // much larger forward-index and rewrite working set across indexes,
        // optimizer tasks, and merge-time BP.
        let _reorder_permit = tokio::select! {
            biased;
            () = self.active_operations.wait_for_shutdown() => {
                return Err(Error::IndexClosed);
            }
            permit = Arc::clone(&self.reorder_permits).acquire_owned() => {
                permit.map_err(|_| {
                    Error::Internal("background reorder scheduler is closed".into())
                })?
            }
        };

        let output_id = SegmentId::new();
        let output_hex = output_id.to_hex();
        let source_ids = [seg_id.to_string()];
        let granularity = self.merge_granularity(&source_ids).await;

        // Register while holding `state`, matching orphan cleanup's deletion
        // barrier. Candidates are scanned ahead of time and can go stale: a
        // merge may have consumed this segment since. Its files may even still
        // be on disk (deferred deletion under a searcher snapshot) — reordering
        // them would re-insert a duplicate copy of docs the merge output holds.
        let all_ids = vec![seg_id.to_string(), output_hex];
        let (_guard, source_docs) = {
            let st = self.state.lock().await;
            let Some(source_meta) = st.metadata.segment_metas.get(seg_id) else {
                log::info!(
                    "[optimizer] segment {} no longer in metadata (merged away), skipping reorder",
                    seg_id
                );
                self.clear_reorder_retry(seg_id);
                return Ok(false);
            };

            match self.active_operations.try_register(all_ids) {
                Some(guard) => (guard, source_meta.num_docs),
                None if !self.active_operations.is_accepting() => {
                    return Err(Error::IndexClosed);
                }
                None => {
                    log::debug!("[optimizer] segment {} in active merge, skipping", seg_id);
                    return Ok(false);
                }
            }
        };

        // Fail before allocating a forward index or creating output files.
        // Missing mandatory files are deterministic and should remove this
        // segment from future optimizer scans, not consume the same CPU every
        // interval. Other I/O failures remain retryable.
        if let Err(error) = self.validate_completed_segment(seg_id, source_docs).await {
            if is_deterministic_source_error(&error) {
                self.quarantine_segment(seg_id, &error);
            } else if !matches!(&error, Error::IndexClosed) {
                self.pause_reorder_retries(seg_id, &error);
            }
            return Err(error);
        }

        let mut output_cleanup = self.output_cleanup_guard(output_id);

        let reorder_result = crate::segment::reorder::reorder_segment(
            self.directory.as_ref(),
            &self.schema,
            source_id,
            output_id,
            self.term_cache_blocks,
            self.bp_memory_budget_bytes,
            bp_budget,
            granularity,
            rayon_pool,
        )
        .await;
        let (new_id, total_docs, bp_converged) = match reorder_result {
            Ok(v) => v,
            Err(e) => {
                // A failed pass may have copied tens of GB before dying;
                // delete the uncommitted output before propagating.
                self.delete_output_if_unregistered(output_id, "reorder failure")
                    .await;
                output_cleanup.disarm();
                if is_deterministic_source_error(&e) {
                    self.quarantine_segment(seg_id, &e);
                } else if !matches!(&e, Error::IndexClosed) {
                    self.pause_reorder_retries(seg_id, &e);
                }
                return Err(e);
            }
        };

        // A pass with a depth floor above block granularity has, by
        // definition, not converged to block-level order — record it as
        // unconverged so the optimizer's deepening ladder revisits it with a
        // full-depth (warm-started) pass. Depth caps are only used by the
        // optimizer's first pass on large segments.
        let ladder_converged = bp_converged && bp_budget.min_partition_docs.is_none();
        if let Err(e) = self
            .replace_segments(
                &[seg_id.to_string()],
                new_id,
                total_docs,
                true,
                ladder_converged,
            )
            .await
        {
            self.delete_output_if_unregistered(output_id, "replacement failure")
                .await;
            output_cleanup.disarm();
            if !matches!(&e, Error::IndexClosed) {
                self.pause_reorder_retries(seg_id, &e);
            }
            return Err(e);
        }
        output_cleanup.disarm();
        self.clear_reorder_retry(seg_id);

        Ok(true)
    }

    /// Clean up orphan segment files not registered in metadata.
    ///
    /// Reads metadata, active-operation ownership, and snapshot-deferred
    /// deletions to determine which segments are legitimate. Filesystem
    /// deletion is asynchronous; in-flight outputs and retired sources still
    /// held by readers are both protected.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        let mut orphan_files: HashMap<String, Vec<std::path::PathBuf>> = HashMap::new();

        if let Ok(entries) = self.directory.list_files(std::path::Path::new("")).await {
            for entry in entries {
                let Some(filename) = entry.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                let Some(rest) = filename.strip_prefix("seg_") else {
                    continue;
                };
                let Some(hex_id) = rest.get(..32) else {
                    continue;
                };
                if !hex_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    continue;
                }
                orphan_files
                    .entry(hex_id.to_ascii_lowercase())
                    .or_default()
                    .push(entry);
            }
        }

        let mut deleted = 0;
        for (hex_id, paths) in &orphan_files {
            // Revalidate and atomically claim deletion under the same
            // state -> active_operations -> tracker order used by publishers.
            // The claim lets us release `state` before filesystem I/O: deleting
            // a multi-GB orphan must not freeze commits and snapshot acquisition.
            let deletion_guard = {
                let st = self.state.lock().await;
                if st.metadata.has_segment(hex_id) {
                    continue;
                }
                let Some(guard) = self
                    .active_operations
                    .try_register(vec![hex_id.to_string()])
                else {
                    continue;
                };
                if self.tracker.is_deletion_protected(hex_id) {
                    drop(guard);
                    continue;
                }
                guard
            };

            // Delete what was actually discovered, not only the currently
            // known SegmentFiles extensions. This also removes partial files
            // left by older formats instead of reporting the same orphan on
            // every startup forever.
            let results =
                futures::future::join_all(paths.iter().map(|path| self.directory.delete(path)))
                    .await;
            let removed = results.into_iter().all(|result| match result {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                Err(error) => {
                    log::warn!(
                        "[segment_cleanup] failed sweeping orphan segment {}: {}",
                        hex_id,
                        error,
                    );
                    false
                }
            });
            // Releasing this claim is the deletion barrier. No producer can
            // adopt the ID while its files are being removed.
            drop(deletion_guard);
            if removed {
                deleted += 1;
                log::info!("[segment_cleanup] swept orphan segment {}", hex_id);
            }
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn lifecycle_test_manager() -> Arc<SegmentManager<crate::directories::RamDirectory>> {
        let schema = crate::dsl::SchemaBuilder::default().build();
        let metadata = IndexMetadata::new(schema.clone());
        Arc::new(SegmentManager::new(
            Arc::new(crate::directories::RamDirectory::new()),
            Arc::new(schema),
            metadata,
            Box::new(crate::merge::NoMergePolicy),
            0,
            1,
            Arc::new(Semaphore::new(1)),
            None,
            1024,
            Arc::new(Semaphore::new(1)),
            None,
        ))
    }

    #[test]
    fn output_cleanup_guard_runs_during_panic_unwind() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_in_callback = Arc::clone(&cleaned);
        let cleanup: Arc<dyn Fn(SegmentId) + Send + Sync> = Arc::new(move |_| {
            cleaned_in_callback.store(true, Ordering::SeqCst);
        });

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = OutputCleanupGuard::new(SegmentId::new(), cleanup);
            panic!("simulated reorder panic");
        }));

        assert!(result.is_err());
        assert!(
            cleaned.load(Ordering::SeqCst),
            "partial output cleanup must run during unwind"
        );
    }

    #[test]
    fn output_cleanup_guard_disarms_after_commit() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_in_callback = Arc::clone(&cleaned);
        let cleanup: Arc<dyn Fn(SegmentId) + Send + Sync> = Arc::new(move |_| {
            cleaned_in_callback.store(true, Ordering::SeqCst);
        });

        {
            let mut guard = OutputCleanupGuard::new(SegmentId::new(), cleanup);
            guard.disarm();
        }

        assert!(!cleaned.load(Ordering::SeqCst));
    }

    #[test]
    fn test_active_operation_guard_releases_ownership() {
        let active = Arc::new(ActiveSegmentOperations::new());
        {
            let _guard = active.try_register(vec!["a".into(), "b".into()]).unwrap();
            let snap = active.snapshot();
            assert!(snap.contains("a"));
            assert!(snap.contains("b"));
        }
        assert!(active.snapshot().is_empty());
    }

    #[test]
    fn test_non_overlapping_operations_can_run_concurrently() {
        let active = Arc::new(ActiveSegmentOperations::new());
        let first = active.try_register(vec!["a".into(), "b".into()]).unwrap();
        let _second = active.try_register(vec!["c".into(), "d".into()]).unwrap();
        let snap = active.snapshot();
        assert_eq!(snap.len(), 4);

        drop(first);
        let snap = active.snapshot();
        assert_eq!(snap.len(), 2);
        assert!(snap.contains("c"));
        assert!(snap.contains("d"));
    }

    #[test]
    fn test_overlapping_operation_is_rejected_until_release() {
        let active = Arc::new(ActiveSegmentOperations::new());
        let first = active.try_register(vec!["a".into(), "b".into()]).unwrap();
        assert!(active.try_register(vec!["b".into(), "c".into()]).is_none());
        drop(first);
        assert!(active.try_register(vec!["b".into(), "c".into()]).is_some());
    }

    #[test]
    fn test_active_operation_snapshot() {
        let active = Arc::new(ActiveSegmentOperations::new());
        let _guard = active.try_register(vec!["x".into(), "y".into()]).unwrap();
        let snap = active.snapshot();
        assert!(snap.contains("x"));
        assert!(snap.contains("y"));
        assert!(!snap.contains("z"));
    }

    #[tokio::test]
    async fn shutdown_rejects_new_work_and_waits_for_existing_guard() {
        let active = Arc::new(ActiveSegmentOperations::new());
        let guard = active.try_register(vec!["live".into()]).unwrap();
        active.stop_accepting();
        assert!(active.try_register(vec!["new".into()]).is_none());

        let waiter = {
            let active = Arc::clone(&active);
            tokio::spawn(async move { active.wait_until_idle().await })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(guard);
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("shutdown waiter missed the final guard notification")
            .unwrap();
    }

    #[tokio::test]
    async fn lifecycle_transaction_survives_request_cancellation_and_is_drained() {
        let manager = lifecycle_test_manager();
        let started = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        let completed = Arc::new(AtomicBool::new(false));

        let request = {
            let manager = Arc::clone(&manager);
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let completed = Arc::clone(&completed);
            tokio::spawn(async move {
                manager
                    .run_lifecycle_transaction(async move {
                        started.add_permits(1);
                        let _permit = release.acquire().await.unwrap();
                        completed.store(true, Ordering::Release);
                        Ok(())
                    })
                    .await
            })
        };

        let _started = started.acquire().await.unwrap();
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        release.add_permits(1);

        manager.begin_shutdown();
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            manager.wait_for_shutdown(),
        )
        .await
        .expect("shutdown did not drain detached lifecycle transaction");
        assert!(completed.load(Ordering::Acquire));
    }

    #[test]
    fn merge_retry_backoff_is_exponential_and_capped() {
        assert_eq!(merge_retry_delay(1), std::time::Duration::from_secs(30));
        assert_eq!(merge_retry_delay(2), std::time::Duration::from_secs(60));
        assert_eq!(merge_retry_delay(3), std::time::Duration::from_secs(120));
        assert_eq!(merge_retry_delay(100), MERGE_RETRY_MAX_DELAY);
    }

    #[test]
    fn only_deterministic_source_errors_are_quarantined() {
        assert!(is_deterministic_source_error(&Error::Corruption(
            "bad footer".into()
        )));
        assert!(is_deterministic_source_error(&Error::Io(
            std::io::Error::from(std::io::ErrorKind::NotFound)
        )));
        assert!(!is_deterministic_source_error(&Error::Io(
            std::io::Error::from(std::io::ErrorKind::TimedOut)
        )));
        assert!(!is_deterministic_source_error(&Error::Io(
            std::io::Error::from(std::io::ErrorKind::PermissionDenied)
        )));
    }

    #[test]
    fn transient_reorder_failure_is_backed_off_until_cleared() {
        let manager = lifecycle_test_manager();
        manager.pause_reorder_retries("source", &Error::Internal("transient".into()));
        assert!(manager.paused_reorder_segments().contains("source"));
        manager.clear_reorder_retry("source");
        assert!(!manager.paused_reorder_segments().contains("source"));
    }
}
