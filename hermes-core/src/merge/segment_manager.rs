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
//! Lock-free state:
//!   trained                — arc_swap::ArcSwapOption, no ordering constraint
//!   merge_handles          — tokio::sync::Mutex, never held with state
//! ```
//!
//! **Rule:** Never hold a sync lock while `.await`-ing.

use std::collections::HashSet;
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use tokio::sync::Mutex as AsyncMutex;
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
struct ActiveSegmentOperations {
    inner: parking_lot::Mutex<HashSet<String>>,
}

impl ActiveSegmentOperations {
    fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(HashSet::new()),
        }
    }

    /// Try to claim IDs for an operation. Returns a guard on success, `None`
    /// if any requested ID is already owned by another active operation.
    fn try_register(self: &Arc<Self>, segment_ids: Vec<String>) -> Option<SegmentOperationGuard> {
        let mut inner = self.inner.lock();
        // Check for overlap with any active lifecycle operation.
        for id in &segment_ids {
            if inner.contains(id) {
                log::debug!(
                    "[segment_lifecycle] rejected: {} overlaps with an active operation ({} active IDs)",
                    id,
                    inner.len()
                );
                return None;
            }
        }
        log::debug!(
            "[segment_lifecycle] registered {} IDs (total active: {})",
            segment_ids.len(),
            inner.len() + segment_ids.len()
        );
        for id in &segment_ids {
            inner.insert(id.clone());
        }
        Some(SegmentOperationGuard {
            active_operations: Arc::clone(self),
            segment_ids,
        })
    }

    /// Snapshot of all IDs owned by active operations.
    fn snapshot(&self) -> HashSet<String> {
        self.inner.lock().clone()
    }

    /// Check if a specific segment is currently involved in a merge.
    fn contains(&self, segment_id: &str) -> bool {
        self.inner.lock().contains(segment_id)
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
            inner.remove(id);
        }
    }
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
    unavailable_segment: Option<String>,
}

#[cfg(feature = "native")]
impl MergeTaskError {
    fn source(segment_id: String, error: Error) -> Self {
        Self {
            error,
            unavailable_segment: Some(segment_id),
        }
    }
}

#[cfg(feature = "native")]
impl From<Error> for MergeTaskError {
    fn from(error: Error) -> Self {
        Self {
            error,
            unavailable_segment: None,
        }
    }
}

#[cfg(feature = "native")]
type MergeTaskResult<T> = std::result::Result<T, MergeTaskError>;

/// Segment manager — coordinates segment commit, background merging, and trained structures.
///
/// SOLE owner of `metadata.json`. All metadata mutations go through `state` Mutex.
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Serializes ALL metadata mutations.
    state: AsyncMutex<ManagerState>,

    /// RAII ownership for every in-flight segment lifecycle operation.
    active_operations: Arc<ActiveSegmentOperations>,

    /// Metadata-live segments that failed to open. They stay searchable (and
    /// operator-visible) but are excluded from merges for this process lifetime,
    /// preventing a corrupt input from creating an immediate retry loop.
    quarantined_segments: parking_lot::Mutex<HashSet<String>>,

    /// Generic merge failures pause scheduling briefly. Source-specific open
    /// failures use `quarantined_segments` instead so healthy work can continue.
    merge_retry_after: parking_lot::Mutex<Option<std::time::Instant>>,

    /// In-flight merge JoinHandles — supports multiple concurrent merges.
    merge_handles: AsyncMutex<Vec<JoinHandle<()>>>,

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
    /// Maximum number of concurrent background merges
    max_concurrent_merges: usize,
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
    /// Bounded rayon pool for background CPU work (merge-time BP, manual
    /// reorder). Built lazily at ~cores/4 so background passes cannot
    /// saturate the global rayon pool that query scoring runs on.
    bg_cpu_pool: std::sync::OnceLock<Arc<rayon::ThreadPool>>,
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
        merge_bp_time_budget: Option<std::time::Duration>,
        bp_memory_budget_bytes: usize,
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

        let delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync> = {
            let dir = Arc::clone(&directory);
            let tracker = Arc::clone(&tracker);
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
                let tracker = Arc::clone(&tracker);
                handle.spawn(async move {
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
                    tracker.complete_deletion(&segment_ids);
                });
            })
        };

        Self {
            state: AsyncMutex::new(ManagerState {
                metadata,
                merge_policy,
            }),
            active_operations: Arc::new(ActiveSegmentOperations::new()),
            quarantined_segments: parking_lot::Mutex::new(HashSet::new()),
            merge_retry_after: parking_lot::Mutex::new(None),
            merge_handles: AsyncMutex::new(Vec::new()),
            trained: ArcSwapOption::new(None),
            tracker,
            delete_fn,
            directory,
            schema,
            term_cache_blocks,
            max_concurrent_merges: max_concurrent_merges.max(1),
            reorder_on_merge,
            merge_bp_time_budget,
            bp_memory_budget_bytes,
            bg_cpu_pool: std::sync::OnceLock::new(),
        }
    }

    /// Bounded rayon pool for background CPU (merge-time BP, manual reorder).
    /// Query scoring uses the global rayon pool; keeping background BP off it
    /// prevents a large merge from queueing every search behind gain passes.
    pub fn background_cpu_pool(&self) -> Arc<rayon::ThreadPool> {
        Arc::clone(self.bg_cpu_pool.get_or_init(|| {
            let threads = (num_cpus::get() / 2).max(1);
            log::info!("[merge] background CPU pool: {} thread(s)", threads);
            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("hermes-bg-cpu-{}", i))
                    .build()
                    .expect("failed to build background CPU pool"),
            )
        }))
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

            let manager = Arc::clone(&manager);
            handle.spawn(async move {
                manager
                    .delete_output_if_unregistered(segment_id, "task unwind")
                    .await;
            });
        });

        OutputCleanupGuard::new(output_id, cleanup)
    }

    /// Claim a newly generated indexing segment before its first file write.
    ///
    /// The returned guard must travel with the built segment until metadata
    /// publication or abort. UUID collisions are treated as corruption rather
    /// than silently sharing lifecycle ownership.
    pub(crate) fn protect_new_segment(&self, segment_id: String) -> Result<SegmentOperationGuard> {
        self.active_operations
            .try_register(vec![segment_id.clone()])
            .ok_or_else(|| {
                Error::Corruption(format!(
                    "new segment ID {} is already owned by an active operation",
                    segment_id
                ))
            })
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
                "[merge] quarantined metadata-live segment {} after open/validation failure: {}. \
                 It remains metadata-live for explicit repair but is excluded from merges until restart",
                segment_id,
                error,
            );
        }
    }

    fn pause_merge_retries(&self, error: &Error) {
        const RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(30);
        *self.merge_retry_after.lock() = Some(std::time::Instant::now() + RETRY_DELAY);
        log::warn!(
            "[merge] pausing background merge scheduling for {:.0}s after failure: {}",
            RETRY_DELAY.as_secs_f64(),
            error,
        );
    }

    fn merge_retry_is_paused(&self) -> bool {
        let mut retry_after = self.merge_retry_after.lock();
        match *retry_after {
            Some(deadline) if deadline > std::time::Instant::now() => true,
            Some(_) => {
                *retry_after = None;
                false
            }
            None => false,
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
        let st = self.state.lock().await;
        if st.metadata.has_segment(&output_hex) {
            return;
        }

        log::warn!(
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
        drop(st);
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
    pub(crate) async fn update_metadata<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut st = self.state.lock().await;
        let mut next = st.metadata.clone();
        f(&mut next);
        next.save(self.directory.as_ref()).await?;
        st.metadata = next;
        Ok(())
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
    pub async fn commit(&self, new_segments: &[(String, u32)]) -> Result<()> {
        // Indexing guards still own these IDs here, so the orphan sweeper
        // cannot remove files between validation and metadata publication.
        for (segment_id, num_docs) in new_segments {
            self.validate_completed_segment(segment_id, *num_docs)
                .await?;
        }

        let mut st = self.state.lock().await;
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
        next.save(self.directory.as_ref()).await?;
        for segment_id in &added {
            self.tracker.register(segment_id);
        }
        st.metadata = next;
        Ok(())
    }

    /// Evaluate merge policy and spawn background merges for all eligible candidates.
    ///
    /// **Atomicity**: The entire filter → find_merges → spawn_merge sequence runs
    /// under the `state` lock to prevent a TOCTOU race where concurrent callers
    /// both see segments as eligible before either claims operation ownership.
    /// `spawn_merge` is non-blocking (just `try_register` + `tokio::spawn`), so
    /// holding the state lock through it is safe and sub-microsecond.
    ///
    /// Note: `max_concurrent_merges` is a soft limit — concurrent auto-triggers
    /// may briefly exceed it by one or two due to TOCTOU between slot counting
    /// and handle registration.
    pub async fn maybe_merge(self: &Arc<Self>) {
        if self.merge_retry_is_paused() {
            log::debug!("[maybe_merge] retry backoff active, skipping");
            return;
        }

        // Drain completed handles and check how many slots are available
        let slots_available = {
            let mut handles = self.merge_handles.lock().await;
            handles.retain(|h| !h.is_finished());
            self.max_concurrent_merges.saturating_sub(handles.len())
        };

        if slots_available == 0 {
            log::debug!("[maybe_merge] at max concurrent merges, skipping");
            return;
        }

        // Hold state lock through spawn_merge to make filter + register atomic.
        // This closes the TOCTOU window where concurrent maybe_merge calls could
        // both see the same segments as eligible before either registers them.
        let new_handles = {
            let st = self.state.lock().await;
            let quarantined = self.quarantined_segments.lock().clone();

            // Exclude segments owned by another operation, pending retirement,
            // or quarantined after a persistent open/validation failure.
            let segments: Vec<SegmentInfo> = st
                .metadata
                .segment_metas
                .iter()
                .filter(|(id, _)| {
                    !self.tracker.is_pending_deletion(id)
                        && !self.active_operations.contains(id)
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
            self.merge_handles.lock().await.extend(new_handles);
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
                            reevaluate = true;
                        }
                        Err(e) => {
                            sm.delete_output_if_unregistered(output_id, "replacement failure")
                                .await;
                            output_cleanup.disarm();
                            sm.pause_merge_retries(&e);
                            log::error!("[merge] failed to publish merged segment: {}", e);
                        }
                    }
                }
                Err(MergeTaskError {
                    error,
                    unavailable_segment,
                }) => {
                    log::error!(
                        "[merge] background merge failed for segments {:?}: {}",
                        ids,
                        error
                    );
                    if let Some(segment_id) = unavailable_segment {
                        sm.quarantine_segment(&segment_id, &error);
                        // Recompute without this known-bad input. This is not a
                        // retry of the same candidate because policy filtering
                        // excludes the quarantined ID.
                        reevaluate = true;
                    } else {
                        sm.pause_merge_retries(&error);
                    }
                    sm.delete_output_if_unregistered(output_id, "merge failure")
                        .await;
                    output_cleanup.disarm();
                }
            }
            // Release source/output ownership before re-evaluating policy, so
            // the completed operation cannot artificially hide candidates.
            drop(guard);

            if reevaluate {
                sm.maybe_merge().await;
            }
        }))
    }

    /// Atomically replace old segments with a new merged segment.
    /// Computes merge generation as max(parent gens) + 1 and records ancestors.
    /// `reordered` marks whether the new segment was BP-reordered.
    async fn replace_segments(
        &self,
        old_ids: &[String],
        new_id: String,
        doc_count: u32,
        reordered: bool,
        bp_converged: bool,
    ) -> Result<()> {
        // The operation guard owns the output during validation. Publication
        // below replaces that ownership with metadata + tracker atomically.
        self.validate_completed_segment(&new_id, doc_count).await?;

        let ready_to_delete = {
            let mut st = self.state.lock().await;
            // Every source must still be live: callers hold operation ownership,
            // guard, so a missing source means a stale merge/reorder whose
            // input was already replaced — adding the output would duplicate
            // its documents. The orphaned output files are swept by
            // cleanup_orphan_segments.
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

            // Compute generation from parents before removing them
            let parent_gen = old_ids
                .iter()
                .filter_map(|id| st.metadata.segment_metas.get(id))
                .map(|info| info.generation)
                .max()
                .unwrap_or(0);
            let ancestors: Vec<String> = old_ids.to_vec();

            let mut next = st.metadata.clone();
            for id in old_ids {
                next.remove_segment(id);
            }
            next.add_merged_segment(
                new_id.clone(),
                doc_count,
                ancestors,
                parent_gen + 1,
                reordered,
                bp_converged,
            );

            // Durable-before-visible. If persistence fails, the old metadata
            // and tracker remain intact and source deletion is never armed.
            next.save(self.directory.as_ref()).await?;

            // Snapshot acquisition uses the same state -> tracker order. Install
            // tracker ownership before publishing the new in-memory metadata.
            self.tracker.register(&new_id);
            st.metadata = next;

            // Keep `state` locked until retired sources enter the tracker.
            // Otherwise orphan cleanup can observe them in the gap where they
            // are absent from metadata but not yet marked as snapshot-deferred.
            self.tracker.mark_for_deletion(old_ids)
        };

        for &segment_id in &ready_to_delete {
            if let Err(error) =
                crate::segment::delete_segment(self.directory.as_ref(), segment_id).await
            {
                log::warn!(
                    "[segment_cleanup] immediate delete failed for {}: {}",
                    segment_id.to_hex(),
                    error,
                );
            }
        }
        self.tracker.complete_deletion(&ready_to_delete);
        Ok(())
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
        for (id_str, id) in segment_ids_to_merge.iter().zip(&segment_ids) {
            let files = SegmentFiles::new(id.0);
            for path in files.mandatory_paths() {
                let exists = directory
                    .exists(path)
                    .await
                    .map_err(|error| MergeTaskError::source(id_str.clone(), Error::Io(error)))?;
                if !exists {
                    return Err(MergeTaskError::source(
                        id_str.clone(),
                        Error::Corruption(format!(
                            "merge source {} is missing mandatory file {:?}",
                            id_str, path
                        )),
                    ));
                }
            }
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
                    return Err(MergeTaskError::source(segment_ids_to_merge[i].clone(), e));
                }
            }
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
            .with_background_pool(bg_cpu_pool);

        log::info!(
            "[merge] {} segments -> {} (trained={})",
            segment_ids_to_merge.len(),
            output_hex,
            trained.map_or(0, |t| t.centroids.len()),
        );

        let (_merged_meta, merge_stats) = merger
            .merge(directory, &readers, output_segment_id, trained)
            .await?;
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

        if total_docs > u32::MAX as u64 {
            return Err(Error::Internal(format!(
                "Merged segment doc count ({}) exceeds u32::MAX",
                total_docs
            ))
            .into());
        }
        Ok((output_hex, total_docs as u32, bp_converged))
    }

    /// Cancel all in-flight merge tasks and wait for their guards to unwind.
    /// Used during index deletion so no writer can outlive directory removal.
    pub async fn abort_merges(&self) {
        let handles: Vec<JoinHandle<()>> =
            { std::mem::take(&mut *self.merge_handles.lock().await) };
        for h in &handles {
            h.abort();
        }
        for h in handles {
            // Await cancellation so the task future is dropped, all file
            // writers stop, and lifecycle/output guards run before callers
            // remove the index directory.
            let _ = h.await;
        }
    }

    /// Wait for all current in-flight merges to complete.
    pub async fn wait_for_merging_thread(self: &Arc<Self>) {
        let handles: Vec<JoinHandle<()>> =
            { std::mem::take(&mut *self.merge_handles.lock().await) };
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
            let handles: Vec<JoinHandle<()>> =
                { std::mem::take(&mut *self.merge_handles.lock().await) };
            if handles.is_empty() {
                break;
            }
            for h in handles {
                let _ = h.await;
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
                Some(self.background_cpu_pool()),
            )
            .await;
            let (new_segment_id, total_docs, bp_converged) = match merge_result {
                Ok(v) => v,
                Err(MergeTaskError {
                    error,
                    unavailable_segment,
                }) => {
                    if let Some(segment_id) = unavailable_segment {
                        self.quarantine_segment(&segment_id, &error);
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
        let st = self.state.lock().await;
        let active_ids = self.active_operations.snapshot();
        st.metadata
            .segment_metas
            .iter()
            .filter(|(id, info)| !info.reordered && !active_ids.contains(*id))
            .map(|(id, info)| (id.clone(), info.num_docs))
            .collect()
    }

    /// Segments whose last BP pass hit its wall-clock budget before finishing
    /// (`bp_converged == false`). A warm-started follow-up pass deepens the
    /// ordering; the optimizer revisits these at low priority.
    pub async fn unconverged_segments(&self) -> Vec<(String, u32)> {
        let st = self.state.lock().await;
        let active_ids = self.active_operations.snapshot();
        st.metadata
            .segment_metas
            .iter()
            .filter(|(id, info)| info.reordered && !info.bp_converged && !active_ids.contains(*id))
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
        let _guard = {
            let st = self.state.lock().await;
            if !st.metadata.has_segment(seg_id) {
                log::info!(
                    "[optimizer] segment {} no longer in metadata (merged away), skipping reorder",
                    seg_id
                );
                return Ok(false);
            }

            match self.active_operations.try_register(all_ids) {
                Some(guard) => guard,
                None => {
                    log::debug!("[optimizer] segment {} in active merge, skipping", seg_id);
                    return Ok(false);
                }
            }
        };

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
            return Err(e);
        }
        output_cleanup.disarm();

        Ok(true)
    }

    /// Clean up orphan segment files not registered in metadata.
    ///
    /// Non-blocking: reads metadata, active-operation ownership, and snapshot-deferred
    /// deletions to determine which segments are legitimate. In-flight outputs
    /// and retired sources still held by readers are both protected.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        let mut orphan_ids: HashSet<String> = HashSet::new();

        if let Ok(entries) = self.directory.list_files(std::path::Path::new("")).await {
            for entry in entries {
                let filename = entry.to_string_lossy();
                if filename.starts_with("seg_") && filename.len() > 37 {
                    let hex_part = &filename[4..36];
                    orphan_ids.insert(hex_part.to_string());
                }
            }
        }

        let mut deleted = 0;
        for hex_id in &orphan_ids {
            // Revalidate immediately before deletion and keep `state` locked
            // through the delete. Merge/reorder registration follows
            // state -> active_operations. Indexing registers its fresh UUID
            // before writing the first file, so an ID already discovered by
            // this scan cannot become a new indexing output after this check.
            let st = self.state.lock().await;
            if st.metadata.has_segment(hex_id)
                || self.active_operations.contains(hex_id)
                || self.tracker.is_deletion_protected(hex_id)
            {
                continue;
            }

            let removed = if let Some(segment_id) = SegmentId::from_hex(hex_id) {
                match crate::segment::delete_segment(self.directory.as_ref(), segment_id).await {
                    Ok(()) => true,
                    Err(error) => {
                        log::warn!(
                            "[segment_cleanup] failed sweeping orphan segment {}: {}",
                            hex_id,
                            error,
                        );
                        false
                    }
                }
            } else {
                false
            };
            // Make the deletion barrier explicit: output registration cannot
            // proceed until the filesystem operation above has completed.
            drop(st);
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
}
