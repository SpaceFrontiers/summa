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
//!   vector_artifact_update — atomic producer gate, held across manual training
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
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

use crate::directories::DirectoryWriter;
use crate::error::{Error, Result};
use crate::index::{IndexMetadata, SegmentMetaInfo};
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
    operation_tokens: HashSet<u64>,
    /// Subset of `operation_tokens` owned by indexing producers. Their guards
    /// travel with built-but-uncommitted `PreparedSegment`s and are released
    /// only by a later commit/abort, so drain barriers must not wait on them:
    /// the commit that would release them can be blocked on the barrier's own
    /// caller (writer write lock / `&mut self`).
    indexing_tokens: HashSet<u64>,
    next_operation_token: u64,
    accepting: bool,
    /// Retraining stages a complete replacement segment generation. Ordinary
    /// merge/reorder work is paused so its source set cannot change midway;
    /// indexing producers remain allowed and deliberately emit flat vectors.
    non_indexing_paused: bool,
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
                operation_tokens: HashSet::new(),
                indexing_tokens: HashSet::new(),
                next_operation_token: 0,
                accepting: true,
                non_indexing_paused: false,
            }),
            idle: Notify::new(),
            shutdown: Notify::new(),
        }
    }

    /// Try to claim IDs for a self-draining lifecycle operation (merge,
    /// reorder, cleanup). Returns a guard on success, `None` if any requested
    /// ID is already owned by another active operation.
    fn try_register(self: &Arc<Self>, segment_ids: Vec<String>) -> Option<SegmentOperationGuard> {
        self.try_register_kind(segment_ids, false, false)
    }

    /// Try to claim IDs for an indexing producer whose guard is held until
    /// metadata publication (commit) rather than task completion.
    fn try_register_indexing(
        self: &Arc<Self>,
        segment_ids: Vec<String>,
    ) -> Option<SegmentOperationGuard> {
        self.try_register_kind(segment_ids, true, false)
    }

    /// Claim source/output IDs for the exclusive vector-generation updater,
    /// which is the only non-indexing producer allowed through its own pause.
    fn try_register_vector_update(
        self: &Arc<Self>,
        segment_ids: Vec<String>,
    ) -> Option<SegmentOperationGuard> {
        self.try_register_kind(segment_ids, false, true)
    }

    fn try_register_kind(
        self: &Arc<Self>,
        segment_ids: Vec<String>,
        indexing: bool,
        vector_update: bool,
    ) -> Option<SegmentOperationGuard> {
        let mut inner = self.inner.lock();
        if !inner.accepting {
            log::debug!("[segment_lifecycle] rejected operation during shutdown");
            return None;
        }
        if !indexing && !vector_update && inner.non_indexing_paused {
            log::debug!("[segment_lifecycle] deferred operation during dense vector retraining");
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
        let operation_token = inner.next_operation_token;
        let next_operation_token = operation_token.checked_add(1)?;
        for id in &segment_ids {
            inner.segment_ids.insert(id.clone());
        }
        inner.next_operation_token = next_operation_token;
        inner.operation_tokens.insert(operation_token);
        if indexing {
            inner.indexing_tokens.insert(operation_token);
        }
        Some(SegmentOperationGuard {
            active_operations: Arc::clone(self),
            segment_ids,
            operation_token,
        })
    }

    /// Snapshot of all IDs owned by active operations.
    fn snapshot(&self) -> HashSet<String> {
        self.inner.lock().segment_ids.clone()
    }

    /// Exact identities of self-draining operations (merge/reorder/cleanup)
    /// active at one instant, plus the number of indexing tokens excluded.
    /// Unlike segment IDs, tokens cannot be reused by a later retry, so an
    /// artifact-update barrier can drain only pre-gate producers without being
    /// starved by new flat producers.
    ///
    /// Indexing tokens are deliberately excluded: their guards are parked in
    /// built-but-uncommitted `PreparedSegment`s and only a later commit — which
    /// may be blocked on the barrier's caller — releases them, so waiting on
    /// them deadlocks (see `begin_vector_artifact_update`).
    fn draining_operation_tokens_snapshot(&self) -> (HashSet<u64>, usize) {
        let inner = self.inner.lock();
        let tokens = inner
            .operation_tokens
            .difference(&inner.indexing_tokens)
            .copied()
            .collect();
        (tokens, inner.indexing_tokens.len())
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

    fn pause_non_indexing(&self) {
        self.inner.lock().non_indexing_paused = true;
    }

    fn resume_non_indexing(&self) {
        self.inner.lock().non_indexing_paused = false;
        self.idle.notify_waiters();
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

    async fn wait_until_operations_finish(&self, operations: &HashSet<u64>) {
        while !operations.is_empty() {
            let notified = self.idle.notified();
            if self.inner.lock().operation_tokens.is_disjoint(operations) {
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
    operation_token: u64,
}

impl Drop for SegmentOperationGuard {
    fn drop(&mut self) {
        let mut inner = self.active_operations.inner.lock();
        for id in &self.segment_ids {
            inner.segment_ids.remove(id);
        }
        inner.operation_tokens.remove(&self.operation_token);
        inner.indexing_tokens.remove(&self.operation_token);
        // Token barriers need notification on every completion, not only the
        // transition to complete global idleness.
        self.active_operations.idle.notify_waiters();
        if inner.segment_ids.is_empty() {
            debug_assert!(inner.operation_tokens.is_empty());
        }
    }
}

/// Exclusive gate for an index-level trained-vector artifact update.
///
/// Segment producers consult this gate before capturing the current trained
/// structures. Once the gate is raised, new producers deliberately emit flat
/// vector data; waiting for already-active producers to drain then guarantees
/// that the committed source set stays stable while replacements are staged.
struct VectorArtifactUpdateLease {
    updating: Arc<AtomicBool>,
    active_operations: Arc<ActiveSegmentOperations>,
}

impl Drop for VectorArtifactUpdateLease {
    fn drop(&mut self) {
        self.updating.store(false, Ordering::Release);
        self.active_operations.resume_non_indexing();
    }
}

#[derive(Clone)]
pub(crate) struct VectorArtifactUpdateGuard {
    _lease: Arc<VectorArtifactUpdateLease>,
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

/// Merge JoinHandles taken out of the shared list for draining.
///
/// Drain futures are awaited inline by RPC handlers (force_merge/reorder) and
/// can be dropped at any await when a client disconnects. Handles are awaited
/// through this guard and removed only after completion, so a cancelled drain
/// returns every un-awaited (and possibly still-running) merge to the shared
/// list instead of silently detaching it from shutdown, abort, and
/// force-merge tracking.
struct DrainedMergeHandles<'a> {
    shared: &'a parking_lot::Mutex<Vec<JoinHandle<()>>>,
    drained: Vec<JoinHandle<()>>,
}

impl<'a> DrainedMergeHandles<'a> {
    fn take(shared: &'a parking_lot::Mutex<Vec<JoinHandle<()>>>) -> Self {
        let drained = std::mem::take(&mut *shared.lock());
        Self { shared, drained }
    }

    fn is_empty(&self) -> bool {
        self.drained.is_empty()
    }

    /// Await the next handle. It stays owned by this guard while being polled
    /// and is discarded only once it has completed, so cancellation at the
    /// await reinserts it via `Drop`.
    async fn join_next(&mut self) -> Option<std::result::Result<(), tokio::task::JoinError>> {
        let handle = self.drained.last_mut()?;
        let result = handle.await;
        self.drained.pop();
        Some(result)
    }
}

impl Drop for DrainedMergeHandles<'_> {
    fn drop(&mut self) {
        if !self.drained.is_empty() {
            self.shared.lock().append(&mut self.drained);
        }
    }
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

#[derive(Clone, Copy)]
enum ReplacementLayout {
    Recomputed {
        reordered: bool,
        bp_converged: bool,
    },
    /// A vector-only rewrite leaves document order and sparse layout exactly
    /// unchanged, so its persisted BP progress must not be reset or advanced.
    PreserveSingleSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VectorSegmentRewriteOutcome {
    Rewritten,
    AlreadyCurrent,
    SourceGone,
    Conflict,
    Deferred,
}

/// Complete but unpublished vector-only replacement. Its lifecycle claim and
/// cleanup guard stay armed until the whole codebook generation commits.
pub(crate) struct StagedVectorSegment {
    source_id: String,
    output_id: SegmentId,
    doc_count: u32,
    _operation: SegmentOperationGuard,
    cleanup: OutputCleanupGuard,
}

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

    /// Times `force_merge` observed a conflicting active operation and retried.
    /// Test-only observability for the conflict-retry backoff.
    #[cfg(test)]
    force_merge_conflict_retries: std::sync::atomic::AtomicU64,

    /// Auxiliary lifecycle tasks: metadata transactions, deferred deletes,
    /// and capacity wakeups. Handles registered here are drained before index
    /// removal.
    lifecycle_handles: Arc<parking_lot::Mutex<Vec<JoinHandle<()>>>>,

    /// Trained vector structures — lock-free reads via ArcSwap.
    /// Wrapped in `Arc` so cancellation-safe metadata transactions can publish
    /// the matching in-memory generation after their durable commit point.
    trained: Arc<ArcSwapOption<TrainedVectorStructures>>,

    /// Raised while index-level trained artifacts and their metadata are being
    /// replaced. Search readers keep using the last valid generation, while
    /// segment producers fall back to flat output until publication completes.
    vector_artifact_update: Arc<AtomicBool>,

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
            #[cfg(test)]
            force_merge_conflict_retries: std::sync::atomic::AtomicU64::new(0),
            lifecycle_handles,
            trained: Arc::new(ArcSwapOption::new(None)),
            vector_artifact_update: Arc::new(AtomicBool::new(false)),
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

    /// Bounded rayon pool for background CPU (merge-time BP, manual reorder,
    /// and global vector-codebook training).
    /// Query scoring uses a dedicated search pool; keeping background work on
    /// this separate bounded pool prevents a merge or retrain from queueing
    /// every search behind its CPU passes.
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
            .try_register_indexing(vec![segment_id.clone()])
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

    /// Capture trained structures for a segment producer.
    ///
    /// The second gate check closes the race where an update begins after the
    /// first check but before the ArcSwap load. Producers have lifecycle guards
    /// before calling this method, so an updater that raised the gate waits for
    /// any producer that successfully captured the previous generation.
    pub(crate) fn trained_for_segment_build(&self) -> Option<Arc<TrainedVectorStructures>> {
        if self.vector_artifact_update.load(Ordering::Acquire) {
            return None;
        }
        let trained = self.trained.load_full();
        if self.vector_artifact_update.load(Ordering::Acquire) {
            None
        } else {
            trained
        }
    }

    /// Start an exclusive trained-artifact update and drain merge/reorder
    /// producers that may already hold the previous generation.
    ///
    /// New segment operations may continue while this waits, but they observe
    /// the gate through `trained_for_segment_build` and therefore emit flat
    /// vector data until the replacement generation is published. The guard
    /// is cancellation-safe: dropping the requesting future reopens ANN
    /// production without leaving the manager wedged.
    ///
    /// Indexing tokens cannot be waited on: their guards are parked inside
    /// built-but-uncommitted `PreparedSegment`s and are released only by a
    /// later commit. That commit typically needs the writer this update's
    /// caller already holds (server write lock / embedded `&mut self`), so
    /// waiting would permanently wedge vector-index finalization. They also
    /// cannot be ignored: such a segment may already contain ANN data bound to
    /// the previous artifact generation and could otherwise be committed after
    /// the new generation is published. Reject promptly and let the caller
    /// commit or abort the pending generation before retrying.
    pub(crate) async fn begin_vector_artifact_update(&self) -> Result<VectorArtifactUpdateGuard> {
        self.vector_artifact_update
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                Error::Internal("a trained-vector artifact update is already in progress".into())
            })?;
        self.active_operations.pause_non_indexing();
        let guard = VectorArtifactUpdateGuard {
            _lease: Arc::new(VectorArtifactUpdateLease {
                updating: Arc::clone(&self.vector_artifact_update),
                active_operations: Arc::clone(&self.active_operations),
            }),
        };
        let (preexisting, parked_indexing) =
            self.active_operations.draining_operation_tokens_snapshot();
        if parked_indexing > 0 {
            return Err(Error::Internal(format!(
                "cannot update trained-vector artifacts while {parked_indexing} indexing \
                 segment(s) are built but uncommitted; commit or abort the pending \
                 generation and retry"
            )));
        }
        self.active_operations
            .wait_until_operations_finish(&preexisting)
            .await;
        Ok(guard)
    }

    /// Load trained structures from disk and publish to ArcSwap.
    /// Copies metadata under lock, releases lock, then does disk I/O.
    pub(crate) async fn try_load_and_publish_trained(&self) -> Result<()> {
        // Copy vector_fields under lock (cheap clone of HashMap<u32, FieldMeta>)
        let vector_fields = {
            let st = self.state.lock().await;
            st.metadata.vector_fields.clone()
        };
        // Disk I/O happens WITHOUT holding the state lock
        let trained = IndexMetadata::try_load_trained_from_fields(
            &vector_fields,
            self.schema.as_ref(),
            self.directory.as_ref(),
        )
        .await?
        .map(Arc::new);
        // Publish exactly the validated snapshot, including None. Retaining a
        // previous map when metadata has no Built fields would let new segments
        // depend on artifacts no longer referenced durably.
        self.trained.store(trained);
        Ok(())
    }

    /// Atomically publish a fully staged vector generation.
    ///
    /// Every replacement segment and its global codebook is complete before
    /// this transaction starts. Metadata, tracker ownership, and the lock-free
    /// trained pointer advance under the same state lock; snapshots therefore
    /// observe either the complete old generation or the complete new one.
    pub(crate) async fn publish_vector_generation(
        self: &Arc<Self>,
        artifact_update: &VectorArtifactUpdateGuard,
        vector_fields: HashMap<u32, crate::index::FieldVectorMeta>,
        next_trained: Arc<TrainedVectorStructures>,
        mut staged: Vec<StagedVectorSegment>,
    ) -> Result<()> {
        if !self.vector_artifact_update.load(Ordering::Acquire) {
            return Err(Error::Internal(
                "vector generation publication lost its exclusive update lease".into(),
            ));
        }

        for replacement in &staged {
            self.validate_completed_segment(&replacement.output_id.to_hex(), replacement.doc_count)
                .await?;
        }

        let mut st = Arc::clone(&self.state).lock_owned().await;
        let mut next = st.metadata.clone();
        next.vector_fields = vector_fields;
        next.refresh_total_vectors();

        for replacement in &staged {
            let source_info = next
                .segment_metas
                .remove(&replacement.source_id)
                .ok_or_else(|| {
                    Error::Corruption(format!(
                        "vector generation source {} disappeared before publication",
                        replacement.source_id,
                    ))
                })?;
            let output_hex = replacement.output_id.to_hex();
            if next.segment_metas.contains_key(&output_hex) {
                return Err(Error::Corruption(format!(
                    "vector generation output {output_hex} is already metadata-live"
                )));
            }
            // A vector-only rewrite changes neither document order nor merge
            // lineage, so preserve the complete lifecycle record verbatim.
            next.add_segment_meta(output_hex, source_info);
        }

        let directory = Arc::clone(&self.directory);
        let trained = Arc::clone(&self.trained);
        let tracker = Arc::clone(&self.tracker);
        // Keep the producer gate raised if the requesting future is cancelled
        // after the metadata transaction has been detached. The last guard
        // clone drops only after durable metadata and ArcSwap state agree.
        let artifact_update = artifact_update.clone();
        self.run_lifecycle_transaction(async move {
            let _artifact_update = artifact_update;
            next.save(directory.as_ref()).await?;

            for replacement in &staged {
                tracker.register(&replacement.output_id.to_hex());
            }
            st.metadata = next;
            trained.store(Some(next_trained));

            // Outputs are now durably live. Disarm unwind cleanup before
            // retiring the old generation and releasing operation ownership.
            for replacement in &mut staged {
                replacement.cleanup.disarm();
            }
            let retired = staged
                .iter()
                .map(|replacement| replacement.source_id.clone())
                .collect::<Vec<_>>();
            let ready_to_delete = tracker.mark_for_deletion(&retired);
            drop(st);
            for &segment_id in &ready_to_delete {
                if let Err(error) =
                    crate::segment::delete_segment(directory.as_ref(), segment_id).await
                {
                    log::warn!(
                        "[segment_cleanup] immediate dense-vector generation delete failed for {}: {}",
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
        let (acquired, trained) = {
            let st = self.state.lock().await;
            let segment_ids = st.metadata.segment_ids();
            (self.tracker.acquire(&segment_ids), self.trained.load_full())
        };

        SegmentSnapshot::with_generation(
            Arc::clone(&self.tracker),
            acquired,
            trained,
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

            let trained_snap = sm.trained_for_segment_build();
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
                            ReplacementLayout::Recomputed {
                                reordered: sm.reorder_on_merge,
                                bp_converged,
                            },
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
                // forever when no later commit happens. The sleep runs as a
                // tracked *lifecycle* task, not inside this merge JoinHandle:
                // wait_for_all_merges/force_merge/reorder drain merge handles,
                // and a pure backoff timer with no work in flight must not
                // stall them for up to MERGE_RETRY_MAX_DELAY.
                sm.schedule_merge_retry_wakeup(retry_delay);
            }
        }))
    }

    /// Re-evaluate merge policy after a failure backoff, outside the tracked
    /// merge JoinHandles that merge waiters drain. Shutdown still drains this
    /// task deterministically (lifecycle handles) and interrupts its sleep.
    fn schedule_merge_retry_wakeup(self: &Arc<Self>, retry_delay: std::time::Duration) {
        let manager = Arc::clone(self);
        let future = async move {
            tokio::select! {
                () = tokio::time::sleep(retry_delay) => {
                    manager.maybe_merge().await;
                }
                () = manager.active_operations.wait_for_shutdown() => {}
            }
        };
        let runtime = tokio::runtime::Handle::current();
        if !try_spawn_lifecycle(&self.lifecycle_handles, &runtime, future) {
            log::warn!(
                "[merge] runtime rejected merge-retry wakeup task; eligible segments may stay \
                 unmerged until the next commit re-runs merge policy evaluation"
            );
        }
    }

    /// Atomically replace old segments with a new merged segment.
    /// Computes merge generation as max(parent gens) + 1 and records ancestors.
    /// `reordered` marks whether the new segment was BP-reordered.
    async fn replace_segments(
        self: &Arc<Self>,
        old_ids: &[String],
        new_id: String,
        doc_count: u32,
        layout: ReplacementLayout,
    ) -> Result<()> {
        // The operation guard owns the output during validation. Publication
        // below replaces that ownership with metadata + tracker atomically.
        self.validate_completed_segment(&new_id, doc_count).await?;
        let output_id = SegmentId::from_hex(&new_id).ok_or_else(|| {
            Error::Corruption(format!("invalid replacement segment ID: {new_id}"))
        })?;
        let output_reader = SegmentReader::open(
            self.directory.as_ref(),
            output_id,
            Arc::clone(&self.schema),
            self.term_cache_blocks,
        )
        .await
        .map_err(|error| match error {
            // Preserve retryable storage failures as I/O. Structural failures
            // are deterministic for this completed output and get explicit
            // corruption context.
            Error::Io(_) | Error::IndexClosed => error,
            error => Error::Corruption(format!(
                "replacement segment {new_id} failed full reader validation: {error}"
            )),
        })?;
        if output_reader.num_docs() != doc_count {
            return Err(Error::Corruption(format!(
                "replacement segment {new_id} opened with {} docs, expected {doc_count}",
                output_reader.num_docs(),
            )));
        }
        drop(output_reader);

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

        let replacement_info = match layout {
            ReplacementLayout::Recomputed {
                reordered,
                bp_converged,
            } => {
                let generation = old_ids
                    .iter()
                    .filter_map(|id| st.metadata.segment_metas.get(id))
                    .map(|info| info.generation)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or_else(|| Error::Corruption("merge generation exceeds u32::MAX".into()))?;
                let parent_unconverged_passes = old_ids
                    .iter()
                    .filter_map(|id| st.metadata.segment_metas.get(id))
                    .map(|info| info.bp_unconverged_passes)
                    .max()
                    .unwrap_or(0);
                let passes = if reordered && !bp_converged {
                    parent_unconverged_passes.saturating_add(1)
                } else {
                    0
                };
                SegmentMetaInfo {
                    num_docs: doc_count,
                    ancestors: old_ids.to_vec(),
                    generation,
                    reordered,
                    bp_converged,
                    bp_unconverged_passes: passes,
                }
            }
            ReplacementLayout::PreserveSingleSource => {
                let [source_id] = old_ids else {
                    return Err(Error::Internal(
                        "layout-preserving replacement requires exactly one source".into(),
                    ));
                };
                let mut source = st
                    .metadata
                    .segment_metas
                    .get(source_id)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Corruption(format!(
                            "layout-preserving replacement source {source_id} disappeared"
                        ))
                    })?;
                source.num_docs = doc_count;
                source
            }
        };
        let retired_ids = old_ids.to_vec();
        let mut next = st.metadata.clone();
        for id in old_ids {
            next.remove_segment(id);
        }
        next.add_segment_meta(new_id.clone(), replacement_info);

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
    ///
    /// Cancellation-safe: dropping this future mid-drain returns un-awaited
    /// handles to `merge_handles` so later drains still see in-flight merges.
    pub async fn abort_merges(&self) {
        loop {
            let mut handles = DrainedMergeHandles::take(&self.merge_handles);
            if handles.is_empty() {
                return;
            }
            while let Some(result) = handles.join_next().await {
                if let Err(error) = result
                    && error.is_panic()
                {
                    log::error!("[merge] background task panicked while draining: {}", error);
                }
            }
        }
    }

    /// Wait for all current in-flight merges to complete.
    ///
    /// Cancellation-safe: dropping this future mid-drain returns un-awaited
    /// handles to `merge_handles` so later drains still see in-flight merges.
    pub async fn wait_for_merging_thread(self: &Arc<Self>) {
        let mut handles = DrainedMergeHandles::take(&self.merge_handles);
        while handles.join_next().await.is_some() {}
    }

    /// Wait for all eligible merges to complete, including cascading merges.
    ///
    /// Drains current handles, then loops. Each completed merge auto-triggers
    /// `maybe_merge` (which pushes new handles) before its JoinHandle resolves,
    /// so by the time `join_next` returns all cascading handles are registered.
    ///
    /// Cancellation-safe: dropping this future mid-drain returns un-awaited
    /// handles to `merge_handles` so later drains still see in-flight merges.
    pub async fn wait_for_all_merges(self: &Arc<Self>) {
        loop {
            let mut handles = DrainedMergeHandles::take(&self.merge_handles);
            if handles.is_empty() {
                break;
            }
            while handles.join_next().await.is_some() {}
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
        // Conflicting owners that never appear in `merge_handles` (background
        // reorders, a concurrent force-merge) can hold a batch segment for
        // minutes to hours; retrying without parking would busy-spin a runtime
        // worker and hammer the state mutex for that whole window.
        const FORCE_MERGE_CONFLICT_BACKOFF: std::time::Duration =
            std::time::Duration::from_millis(100);
        // When every remaining mergeable segment is owned by another
        // operation (e.g. background BP reorders), there is nothing to do but
        // wait for a release; those passes run for minutes, so poll slowly.
        const FORCE_MERGE_HELD_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);

        let max_segment_docs = {
            let st = self.state.lock().await;
            st.merge_policy.max_segment_docs()
        };

        // Wait for all in-flight background merges (including cascading)
        // before starting forced merges to avoid try_register conflicts.
        self.wait_for_all_merges().await;

        // One INFO line per wait episode, not per 1s poll; DEBUG afterwards.
        let mut logged_held_wait = false;

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

            // Route around segments owned by active operations (background
            // reorders, concurrent force-merges) instead of insisting on the
            // deterministic smallest-N batch: retrying a batch that contains a
            // segment mid-BP-pass livelocked here for the whole pass (observed
            // in prod: the optimizer holds exactly the small fresh segments
            // force_merge wants first). Held segments are merged on a later
            // iteration, after their owner releases them.
            let active_ids = self.active_operations.snapshot();
            let held: usize = segments
                .iter()
                .filter(|(id, _)| active_ids.contains(id))
                .count();

            // Build a batch of free segments respecting max_segment_docs
            let max_docs = max_segment_docs.map(|m| m as u64).unwrap_or(u64::MAX);
            let mut batch = Vec::new();
            let mut batch_docs = 0u64;

            for (id, docs) in &segments {
                if active_ids.contains(id) {
                    continue;
                }
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
                if held == 0 {
                    // Nothing left that can merge and nobody will release
                    // more candidates: force merge is complete.
                    return Ok(());
                }
                // All remaining work is behind active owners. Their guards
                // are RAII and their passes are time-budgeted, so this always
                // unblocks; poll slowly rather than spinning.
                if !logged_held_wait {
                    log::info!(
                        "[force_merge] waiting: {} segment(s) held by active \
                         merge/reorder operations, none free to merge",
                        held
                    );
                    logged_held_wait = true;
                } else {
                    log::debug!("[force_merge] still waiting on {} held segment(s)", held);
                }
                #[cfg(test)]
                self.force_merge_conflict_retries
                    .fetch_add(1, Ordering::Relaxed);
                tokio::select! {
                    biased;
                    () = self.active_operations.wait_for_shutdown() => {
                        return Err(Error::IndexClosed);
                    }
                    () = tokio::time::sleep(FORCE_MERGE_HELD_BACKOFF) => {}
                }
                continue;
            }
            logged_held_wait = false;

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
                    #[cfg(test)]
                    self.force_merge_conflict_retries
                        .fetch_add(1, Ordering::Relaxed);
                    // Do not reserve application-wide merge capacity while
                    // parked on a conflict.
                    drop(_global_merge_permit);
                    // The ownership snapshot above is advisory: an operation
                    // can register one of our batch segments between the
                    // snapshot and try_register. A tracked background merge
                    // may also have slipped in — drain those first, then back
                    // off briefly and rebuild the batch from a fresh snapshot.
                    log::debug!("[force_merge] batch lost a registration race, rebuilding");
                    let had_tracked_merges = !self.merge_handles.lock().is_empty();
                    self.wait_for_merging_thread().await;
                    if !had_tracked_merges {
                        tokio::time::sleep(FORCE_MERGE_CONFLICT_BACKOFF).await;
                    }
                    continue;
                }
            };
            // Announce only after ownership is secured: this line used to
            // print before registration, spamming once per 100ms retry while
            // a reorder held a batch segment.
            log::info!(
                "[force_merge] merging batch of {} segments ({} docs)",
                batch.len(),
                batch_docs
            );
            let mut output_cleanup = self.output_cleanup_guard(output_id);

            let trained_snap = self.trained_for_segment_build();
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
                    ReplacementLayout::Recomputed {
                        reordered: self.reorder_on_merge,
                        bp_converged,
                    },
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

    fn segment_needs_vector_rewrite(
        &self,
        reader: &SegmentReader,
        field_ids: &[u32],
        rewrite_existing: bool,
    ) -> Result<bool> {
        for &field_id in field_ids {
            let flat = reader.flat_vectors().get(&field_id);
            let ann = reader.vector_indexes().get(&field_id);
            if ann.is_some() && flat.is_none() {
                return Err(Error::Corruption(format!(
                    "segment {:032x} field {field_id} has ANN data without the required flat vectors",
                    reader.meta().id,
                )));
            }

            let Some(flat) = flat else {
                continue;
            };
            if flat.num_vectors == 0 {
                continue;
            }
            if rewrite_existing {
                return Ok(true);
            }
            let field = crate::dsl::Field(field_id);
            let entry = self.schema.get_field_entry(field).ok_or_else(|| {
                Error::Corruption(format!(
                    "segment {:032x} references unknown vector field {field_id}",
                    reader.meta().id,
                ))
            })?;
            let current = match entry.field_type {
                // TQ payloads carry no trained generation, so an existing one
                // is always current; a tq field must never be staged for a
                // vector-generation rewrite.
                crate::dsl::FieldType::DenseVector
                    if entry.dense_vector_config.as_ref().is_some_and(|config| {
                        config.index_type == crate::dsl::VectorIndexType::Tq
                    }) =>
                {
                    matches!(ann, Some(crate::segment::VectorIndex::Tq { .. }))
                }
                crate::dsl::FieldType::DenseVector
                    if entry.dense_vector_config.as_ref().is_some_and(|config| {
                        config.index_type == crate::dsl::VectorIndexType::IvfTq
                    }) =>
                {
                    matches!(ann, Some(crate::segment::VectorIndex::IvfTq { .. }))
                }
                // Only ivf_tq dense fields are trainable; other dense
                // index types never reach a vector-generation rewrite.
                crate::dsl::FieldType::DenseVector => false,
                crate::dsl::FieldType::BinaryDenseVector => {
                    matches!(ann, Some(crate::segment::VectorIndex::BinaryIvf(_)))
                }
                _ => false,
            };
            if !current {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn acquire_vector_rewrite_capacity(
        &self,
    ) -> Result<(OwnedSemaphorePermit, OwnedSemaphorePermit)> {
        let global = tokio::select! {
            biased;
            () = self.active_operations.wait_for_shutdown() => {
                return Err(Error::IndexClosed);
            }
            permit = Arc::clone(&self.global_merge_permits).acquire_owned() => {
                permit.map_err(|_| Error::Internal(
                    "global background merge scheduler is closed".into()
                ))?
            }
        };
        let local = tokio::select! {
            biased;
            () = self.active_operations.wait_for_shutdown() => {
                return Err(Error::IndexClosed);
            }
            permit = Arc::clone(&self.merge_permits).acquire_owned() => {
                permit.map_err(|_| Error::Internal(
                    "background merge scheduler is closed".into()
                ))?
            }
        };
        Ok((global, local))
    }

    async fn build_vector_replacement(
        self: &Arc<Self>,
        segment_id: &str,
        source_id: SegmentId,
        output_id: SegmentId,
        trained: &TrainedVectorStructures,
        failure_context: &'static str,
    ) -> Result<(String, u32, OutputCleanupGuard)> {
        let mut cleanup = self.output_cleanup_guard(output_id);
        match crate::segment::reorder::rewrite_vector_segment(
            self.directory.as_ref(),
            &self.schema,
            source_id,
            output_id,
            self.term_cache_blocks,
            trained,
            Some(self.background_cpu_pool()),
        )
        .await
        {
            Ok((new_id, doc_count)) => {
                self.validate_completed_segment(&new_id, doc_count).await?;
                Ok((new_id, doc_count, cleanup))
            }
            Err(error) => {
                self.delete_output_if_unregistered(output_id, failure_context)
                    .await;
                cleanup.disarm();
                if is_deterministic_source_error(&error) {
                    self.quarantine_segment(segment_id, &error);
                }
                Err(error)
            }
        }
    }

    /// Build every required replacement segment without exposing any of them.
    /// The returned guards keep both source and output generations alive until
    /// [`Self::publish_vector_generation`] commits the complete set.
    pub(crate) async fn stage_vector_generation(
        self: &Arc<Self>,
        _artifact_update: &VectorArtifactUpdateGuard,
        segment_ids: &[String],
        field_ids: &[u32],
        trained: Arc<TrainedVectorStructures>,
        rewrite_existing: bool,
    ) -> Result<Vec<StagedVectorSegment>> {
        if !self.vector_artifact_update.load(Ordering::Acquire) {
            return Err(Error::Internal(
                "cannot stage a vector generation without an exclusive update lease".into(),
            ));
        }

        let mut staged = Vec::new();
        for segment_id in segment_ids {
            if self.quarantined_segments.lock().contains(segment_id) {
                return Err(Error::Corruption(format!(
                    "segment {segment_id} is quarantined after a deterministic source failure"
                )));
            }
            let source_id = SegmentId::from_hex(segment_id).ok_or_else(|| {
                Error::Corruption(format!("invalid vector rewrite segment ID: {segment_id}"))
            })?;

            // A single rewrite may hold several gigabytes while assigning all
            // vectors. Reuse the ordinary local and process-wide merge bounds.
            let _capacity = self.acquire_vector_rewrite_capacity().await?;

            let output_id = SegmentId::new();
            let output_hex = output_id.to_hex();
            let operation = {
                let st = self.state.lock().await;
                if !st.metadata.has_segment(segment_id) {
                    return Err(Error::Corruption(format!(
                        "vector generation source {segment_id} disappeared while lifecycle work was paused"
                    )));
                }
                self.active_operations
                    .try_register_vector_update(vec![segment_id.clone(), output_hex.clone()])
            }
            .ok_or_else(|| {
                if self.active_operations.is_accepting() {
                    Error::Internal(format!(
                        "vector generation could not claim stable source {segment_id}"
                    ))
                } else {
                    Error::IndexClosed
                }
            })?;

            let reader = SegmentReader::open(
                self.directory.as_ref(),
                source_id,
                Arc::clone(&self.schema),
                self.term_cache_blocks,
            )
            .await?;
            if !self.segment_needs_vector_rewrite(&reader, field_ids, rewrite_existing)? {
                continue;
            }
            drop(reader);

            let (new_id, doc_count, cleanup) = self
                .build_vector_replacement(
                    segment_id,
                    source_id,
                    output_id,
                    trained.as_ref(),
                    "vector generation staging failure",
                )
                .await?;
            debug_assert_eq!(new_id, output_hex);
            let output_reader = SegmentReader::open(
                self.directory.as_ref(),
                output_id,
                Arc::clone(&self.schema),
                self.term_cache_blocks,
            )
            .await?;
            if self.segment_needs_vector_rewrite(&output_reader, field_ids, false)? {
                return Err(Error::Corruption(format!(
                    "staged vector segment {new_id} does not match its candidate codebook generation"
                )));
            }

            staged.push(StagedVectorSegment {
                source_id: segment_id.clone(),
                output_id,
                doc_count,
                _operation: operation,
                cleanup,
            });
        }
        Ok(staged)
    }

    async fn rewrite_vector_segment_once(
        self: &Arc<Self>,
        segment_id: &str,
        field_ids: &[u32],
    ) -> Result<VectorSegmentRewriteOutcome> {
        if self.quarantined_segments.lock().contains(segment_id) {
            return Err(Error::Corruption(format!(
                "segment {segment_id} is quarantined after a deterministic source failure; repair it before ANN finalization"
            )));
        }
        let source_id = SegmentId::from_hex(segment_id).ok_or_else(|| {
            Error::Corruption(format!("invalid vector rewrite segment ID: {segment_id}"))
        })?;

        // Match ordinary merge lock ordering: capacity before lifecycle
        // ownership. A vector rewrite can hold several gigabytes while it
        // assigns vectors, so it participates in both local and process-wide
        // merge limits.
        let _capacity = self.acquire_vector_rewrite_capacity().await?;

        let output_id = SegmentId::new();
        let output_hex = output_id.to_hex();
        let all_ids = vec![segment_id.to_owned(), output_hex];
        let operation = {
            let st = self.state.lock().await;
            if !st.metadata.has_segment(segment_id) {
                return Ok(VectorSegmentRewriteOutcome::SourceGone);
            }
            self.active_operations.try_register(all_ids)
        };
        let _operation = match operation {
            Some(operation) => operation,
            None if !self.active_operations.is_accepting() => return Err(Error::IndexClosed),
            None => return Ok(VectorSegmentRewriteOutcome::Conflict),
        };

        let Some(trained) = self.trained_for_segment_build() else {
            return Ok(VectorSegmentRewriteOutcome::Deferred);
        };

        let reader = SegmentReader::open(
            self.directory.as_ref(),
            source_id,
            Arc::clone(&self.schema),
            self.term_cache_blocks,
        )
        .await?;
        if !self.segment_needs_vector_rewrite(&reader, field_ids, false)? {
            return Ok(VectorSegmentRewriteOutcome::AlreadyCurrent);
        }
        drop(reader);

        let (new_id, doc_count, mut output_cleanup) = self
            .build_vector_replacement(
                segment_id,
                source_id,
                output_id,
                trained.as_ref(),
                "vector rewrite failure",
            )
            .await?;

        if let Err(error) = self
            .replace_segments(
                &[segment_id.to_owned()],
                new_id,
                doc_count,
                ReplacementLayout::PreserveSingleSource,
            )
            .await
        {
            self.delete_output_if_unregistered(output_id, "vector replacement failure")
                .await;
            output_cleanup.disarm();
            return Err(error);
        }
        output_cleanup.disarm();
        Ok(VectorSegmentRewriteOutcome::Rewritten)
    }

    /// Finalize every committed flat vector segment against the published
    /// global ANN generation.
    /// Unlike force-merge this handles one segment and segments already at the
    /// merge policy's maximum size.
    pub(crate) async fn rewrite_vector_segments(
        self: &Arc<Self>,
        field_ids: &[u32],
    ) -> Result<usize> {
        if field_ids.is_empty() {
            return Ok(0);
        }
        let mut rewritten = 0usize;
        loop {
            let segment_ids = self.get_segment_ids().await;
            let mut conflicted = false;
            let mut changed = false;
            for segment_id in segment_ids {
                match self
                    .rewrite_vector_segment_once(&segment_id, field_ids)
                    .await?
                {
                    VectorSegmentRewriteOutcome::Rewritten => {
                        rewritten += 1;
                        changed = true;
                    }
                    VectorSegmentRewriteOutcome::Conflict => conflicted = true,
                    VectorSegmentRewriteOutcome::Deferred => {
                        return Err(Error::Internal(
                            "ANN finalization lost the published trained generation".into(),
                        ));
                    }
                    VectorSegmentRewriteOutcome::AlreadyCurrent
                    | VectorSegmentRewriteOutcome::SourceGone => {}
                }
            }
            if !conflicted && !changed {
                log::info!(
                    "[dense_vector_rewrite] ANN finalization complete ({} segment(s) rewritten)",
                    rewritten,
                );
                return Ok(rewritten);
            }
            tokio::select! {
                biased;
                () = self.active_operations.wait_for_shutdown() => {
                    return Err(Error::IndexClosed);
                }
                () = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
            }
        }
    }

    /// A producer that started in the force-flat phase can commit after the
    /// main finalization snapshot. Upgrade exactly those new segments in a
    /// tracked background task; ordinary producers already using the current
    /// generation are detected and skipped without rewriting.
    pub(crate) fn schedule_vector_segment_upgrades(self: &Arc<Self>, segment_ids: Vec<String>) {
        if segment_ids.is_empty() || self.trained_for_segment_build().is_none() {
            return;
        }
        let manager = Arc::clone(self);
        let future = async move {
            let field_ids = manager
                .read_metadata(|metadata| {
                    metadata
                        .vector_fields
                        .keys()
                        .filter(|field_id| metadata.is_field_built(**field_id))
                        .copied()
                        .collect::<Vec<_>>()
                })
                .await;
            for segment_id in segment_ids {
                loop {
                    match manager
                        .rewrite_vector_segment_once(&segment_id, &field_ids)
                        .await
                    {
                        Ok(VectorSegmentRewriteOutcome::Conflict) => {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                        Ok(VectorSegmentRewriteOutcome::Deferred) => break,
                        Ok(_) => break,
                        Err(error) => {
                            log::error!(
                                "[dense_vector_rewrite] failed to upgrade newly committed segment {}: {}",
                                segment_id,
                                error,
                            );
                            break;
                        }
                    }
                }
            }
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            log::warn!(
                "[dense_vector_rewrite] runtime unavailable; newly committed flat segment upgrade deferred"
            );
            return;
        };
        if !try_spawn_lifecycle(&self.lifecycle_handles, &runtime, future) {
            log::warn!(
                "[dense_vector_rewrite] runtime rejected newly committed flat segment upgrade"
            );
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
        self.unconverged_segments_below(u32::MAX)
            .await
            .into_iter()
            .map(|(id, docs, _)| (id, docs))
            .collect()
    }

    /// Unconverged segments still below a hard replacement-lineage work
    /// bound. Includes the persisted attempt count for scheduler diagnostics.
    pub async fn unconverged_segments_below(
        &self,
        max_unconverged_passes: u32,
    ) -> Vec<(String, u32, u32)> {
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
                    && info.bp_unconverged_passes < max_unconverged_passes
                    && !active_ids.contains(*id)
                    && !quarantined.contains(*id)
                    && !paused.contains(*id)
            })
            .map(|(id, info)| (id.clone(), info.num_docs, info.bp_unconverged_passes))
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
                ReplacementLayout::Recomputed {
                    reordered: true,
                    bp_converged: ladder_converged,
                },
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
    async fn operation_barrier_ignores_producers_started_after_snapshot() {
        let active = Arc::new(ActiveSegmentOperations::new());
        let before_gate = active.try_register(vec!["old".into()]).unwrap();
        let (barrier, parked_indexing) = active.draining_operation_tokens_snapshot();
        assert_eq!(parked_indexing, 0);
        let after_gate = active.try_register(vec!["new-flat".into()]).unwrap();

        let waiter = {
            let active = Arc::clone(&active);
            tokio::spawn(async move { active.wait_until_operations_finish(&barrier).await })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());

        drop(before_gate);
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("pre-gate operation barrier was starved by a post-gate producer")
            .unwrap();
        assert!(active.snapshot().contains("new-flat"));
        drop(after_gate);
    }

    #[tokio::test]
    async fn artifact_update_gate_preserves_search_generation_but_forces_flat_producers() {
        let manager = lifecycle_test_manager();
        manager
            .trained
            .store(Some(Arc::new(TrainedVectorStructures {
                centroids: rustc_hash::FxHashMap::default(),
                binary_quantizers: rustc_hash::FxHashMap::default(),
                ..Default::default()
            })));

        let guard = manager.begin_vector_artifact_update().await.unwrap();
        assert!(
            manager.trained().is_some(),
            "search readers keep the last fully validated generation"
        );
        assert!(
            manager.trained_for_segment_build().is_none(),
            "new segment producers must stay flat during an artifact update"
        );

        let detached_transaction_guard = guard.clone();
        drop(guard);
        assert!(
            manager.trained_for_segment_build().is_none(),
            "a detached lifecycle transaction must retain the producer gate after request cancellation"
        );
        drop(detached_transaction_guard);
        assert!(manager.trained_for_segment_build().is_some());
    }

    #[tokio::test]
    async fn artifact_update_pauses_lifecycle_rewrites_but_allows_flat_indexing() {
        let manager = lifecycle_test_manager();
        let guard = manager.begin_vector_artifact_update().await.unwrap();
        assert!(
            manager
                .active_operations
                .try_register(vec!["merge".into()])
                .is_none(),
            "ordinary merge/reorder work must not change staged sources"
        );
        let indexing = manager
            .active_operations
            .try_register_indexing(vec!["fresh".into()])
            .expect("indexing remains available in flat mode");
        drop(indexing);

        drop(guard);
        assert!(!manager.vector_artifact_update.load(Ordering::Acquire));
        assert!(
            manager
                .active_operations
                .try_register(vec!["merge".into()])
                .is_some()
        );
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

    #[tokio::test]
    async fn unconverged_scheduler_stops_at_the_lineage_limit() {
        let manager = lifecycle_test_manager();
        {
            let mut state = manager.state.lock().await;
            state.metadata.add_segment_meta(
                "eligible".into(),
                SegmentMetaInfo {
                    num_docs: 10,
                    ancestors: Vec::new(),
                    generation: 1,
                    reordered: true,
                    bp_converged: false,
                    bp_unconverged_passes: 2,
                },
            );
            state.metadata.add_segment_meta(
                "at-limit".into(),
                SegmentMetaInfo {
                    num_docs: 20,
                    ancestors: Vec::new(),
                    generation: 1,
                    reordered: true,
                    bp_converged: false,
                    bp_unconverged_passes: 3,
                },
            );
            state.metadata.add_segment_meta(
                "converged".into(),
                SegmentMetaInfo {
                    num_docs: 30,
                    ancestors: Vec::new(),
                    generation: 1,
                    reordered: true,
                    bp_converged: true,
                    bp_unconverged_passes: 0,
                },
            );
            state.metadata.add_segment("fresh".into(), 40);
        }

        assert_eq!(
            manager.unconverged_segments_below(3).await,
            vec![("eligible".into(), 10, 2)]
        );
        assert!(manager.unconverged_segments_below(0).await.is_empty());
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

    /// Fails `exists` with the transient I/O error class that sends a
    /// background merge into its generic retry backoff (not source quarantine).
    #[derive(Default)]
    struct FailingExistsDirectory(crate::directories::RamDirectory);

    #[async_trait::async_trait]
    impl crate::directories::Directory for FailingExistsDirectory {
        async fn exists(&self, _path: &std::path::Path) -> std::io::Result<bool> {
            Err(std::io::Error::from(std::io::ErrorKind::TimedOut))
        }

        async fn file_size(&self, path: &std::path::Path) -> std::io::Result<u64> {
            self.0.file_size(path).await
        }

        async fn open_read(
            &self,
            path: &std::path::Path,
        ) -> std::io::Result<crate::directories::FileHandle> {
            self.0.open_read(path).await
        }

        async fn read_range(
            &self,
            path: &std::path::Path,
            range: std::ops::Range<u64>,
        ) -> std::io::Result<crate::directories::OwnedBytes> {
            self.0.read_range(path, range).await
        }

        async fn list_files(
            &self,
            prefix: &std::path::Path,
        ) -> std::io::Result<Vec<std::path::PathBuf>> {
            self.0.list_files(prefix).await
        }

        async fn open_lazy(
            &self,
            path: &std::path::Path,
        ) -> std::io::Result<crate::directories::FileHandle> {
            self.0.open_lazy(path).await
        }
    }

    #[async_trait::async_trait]
    impl crate::directories::DirectoryWriter for FailingExistsDirectory {
        async fn write(&self, path: &std::path::Path, data: &[u8]) -> std::io::Result<()> {
            self.0.write(path, data).await
        }

        async fn delete(&self, path: &std::path::Path) -> std::io::Result<()> {
            self.0.delete(path).await
        }

        async fn rename(
            &self,
            from: &std::path::Path,
            to: &std::path::Path,
        ) -> std::io::Result<()> {
            self.0.rename(from, to).await
        }

        async fn sync(&self) -> std::io::Result<()> {
            self.0.sync().await
        }

        async fn streaming_writer(
            &self,
            path: &std::path::Path,
        ) -> std::io::Result<Box<dyn crate::directories::StreamingWriter>> {
            self.0.streaming_writer(path).await
        }
    }

    #[derive(Debug, Clone)]
    struct MergeEverythingPolicy;

    impl MergePolicy for MergeEverythingPolicy {
        fn find_merges(&self, segments: &[SegmentInfo]) -> Vec<crate::merge::MergeCandidate> {
            if segments.len() < 2 {
                return Vec::new();
            }
            vec![crate::merge::MergeCandidate {
                segment_ids: segments.iter().map(|s| s.id.clone()).collect(),
            }]
        }

        fn clone_box(&self) -> Box<dyn MergePolicy> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn artifact_update_rejects_built_uncommitted_indexing_segments() {
        let manager = lifecycle_test_manager();
        // Simulates a memory-budget mid-cycle segment build whose guard is
        // parked inside a PreparedSegment: only a later commit releases this
        // token, and that commit can be blocked on the very caller of the
        // artifact update (writer write lock / &mut self).
        let parked_indexing = manager
            .protect_new_segment("00000000000000000000000000000abc".into())
            .unwrap();

        let error = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            manager.begin_vector_artifact_update(),
        )
        .await
        .expect("begin_vector_artifact_update deadlocked on a built-but-uncommitted segment")
        .err()
        .expect("an old-generation prepared segment must block artifact replacement")
        .to_string();
        assert!(error.contains("built but uncommitted"), "{error}");
        assert!(
            !manager.vector_artifact_update.load(Ordering::Acquire),
            "a rejected update must release the producer gate"
        );

        drop(parked_indexing);

        let guard = manager
            .begin_vector_artifact_update()
            .await
            .expect("artifact update should succeed after the pending generation is resolved");
        drop(guard);
    }

    #[tokio::test]
    async fn artifact_update_still_drains_preexisting_lifecycle_operations() {
        let manager = lifecycle_test_manager();
        let merge_like = manager
            .active_operations
            .try_register(vec!["merge-source".into()])
            .unwrap();

        let waiter = {
            let manager = Arc::clone(&manager);
            tokio::spawn(async move { manager.begin_vector_artifact_update().await })
        };
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            !waiter.is_finished(),
            "artifact update must drain merge/reorder producers that may hold the previous generation"
        );

        drop(merge_like);
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("artifact update missed the lifecycle guard release")
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cancelled_merge_drain_returns_unawaited_handles_to_shared_state() {
        let manager = lifecycle_test_manager();
        let release = Arc::new(Semaphore::new(0));
        let merge_task = {
            let release = Arc::clone(&release);
            tokio::spawn(async move {
                let _permit = release.acquire().await.unwrap();
            })
        };
        manager.merge_handles.lock().push(merge_task);

        let waiter = {
            let manager = Arc::clone(&manager);
            tokio::spawn(async move { manager.wait_for_all_merges().await })
        };
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(!waiter.is_finished());
        // Simulates tonic dropping a force_merge/reorder RPC future at the
        // JoinHandle await when the client disconnects.
        waiter.abort();
        let join_error = waiter.await.unwrap_err();
        assert!(join_error.is_cancelled());

        assert!(
            !manager.merge_handles.lock().is_empty(),
            "cancelled drain detached an in-flight merge from shutdown/force-merge tracking"
        );

        // A later drain must still see and await the real in-flight merge.
        release.add_permits(1);
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            manager.wait_for_all_merges(),
        )
        .await
        .expect("subsequent drain missed the reinserted merge handle");
        assert!(manager.merge_handles.lock().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn force_merge_conflict_retry_backs_off_instead_of_busy_spinning() {
        let manager = lifecycle_test_manager();
        {
            let mut state = manager.state.lock().await;
            state
                .metadata
                .add_segment("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(), 10);
            state
                .metadata
                .add_segment("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(), 10);
        }
        // A background reorder (or a concurrent force-merge) owns one segment
        // in the batch but never appears in merge_handles.
        let reorder_like = manager
            .active_operations
            .try_register(vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()])
            .unwrap();

        let force_merge = {
            let manager = Arc::clone(&manager);
            tokio::spawn(async move { manager.force_merge().await })
        };

        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        let retries = manager.force_merge_conflict_retries.load(Ordering::Relaxed);
        assert!(
            retries >= 1,
            "force_merge never observed the conflicting owner (retries={retries})"
        );
        assert!(
            retries < 20,
            "force_merge busy-spun on a conflict that is not a tracked merge (retries={retries})"
        );

        drop(reorder_like);
        // With the conflict gone the loop proceeds; the batch then fails fast
        // in do_merge (the test IDs have no files), proving the loop exited.
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), force_merge)
            .await
            .expect("force_merge kept spinning after the conflicting owner released")
            .unwrap();
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn force_merge_routes_around_segments_held_by_reorder() {
        let manager = lifecycle_test_manager();
        {
            let mut state = manager.state.lock().await;
            state
                .metadata
                .add_segment("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(), 10);
            state
                .metadata
                .add_segment("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(), 10);
            state
                .metadata
                .add_segment("cccccccccccccccccccccccccccccccc".into(), 10);
        }
        // A background reorder owns one segment and holds it for the whole
        // test (in prod: a BP pass runs for minutes while force_merge spins).
        let _reorder_like = manager
            .active_operations
            .try_register(vec!["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()])
            .unwrap();

        // Regression: force_merge used to rebuild the identical smallest-N
        // batch (including the held segment) every 100ms and retry-log
        // forever. It must instead skip the held segment and immediately
        // make progress on the two free ones — reaching do_merge (which
        // fails fast here: the test IDs have no files) proves the batch was
        // built without the held segment while the reorder is STILL active.
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), {
            let manager = Arc::clone(&manager);
            async move { manager.force_merge().await }
        })
        .await
        .expect("force_merge livelocked on a segment held by an active reorder");
        assert!(result.is_err(), "fake segment files must fail the merge");

        assert_eq!(
            manager.force_merge_conflict_retries.load(Ordering::Relaxed),
            0,
            "batch built from the ownership snapshot must not collide with the held segment"
        );
    }

    #[tokio::test]
    async fn merge_failure_retry_backoff_does_not_stall_merge_waiters() {
        let schema = crate::dsl::SchemaBuilder::default().build();
        let mut metadata = IndexMetadata::new(schema.clone());
        metadata.add_segment("00000000000000000000000000000001".into(), 10);
        metadata.add_segment("00000000000000000000000000000002".into(), 10);
        let manager = Arc::new(SegmentManager::new(
            Arc::new(FailingExistsDirectory::default()),
            Arc::new(schema),
            metadata,
            Box::new(MergeEverythingPolicy),
            0,
            1,
            Arc::new(Semaphore::new(1)),
            None,
            1024,
            Arc::new(Semaphore::new(1)),
            None,
        ));

        // Spawns a background merge that fails with a transient I/O error and
        // arms the 30s..30min retry backoff.
        manager.maybe_merge().await;

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            manager.wait_for_all_merges(),
        )
        .await
        .expect("wait_for_all_merges stalled behind a pure retry-backoff timer");
        assert!(
            manager.merge_retry_is_paused(),
            "the failed merge should have armed the retry backoff"
        );

        // Shutdown still drains the pending backoff wakeup deterministically.
        manager.begin_shutdown();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            manager.wait_for_shutdown(),
        )
        .await
        .expect("shutdown did not drain the merge retry wakeup task");
    }
}
