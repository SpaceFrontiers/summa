//! Segment manager — coordinates segment commit, background merging, and trained structures.
//!
//! Architecture:
//! - **Single mutation queue**: All metadata mutations serialize through `tokio::sync::Mutex<ManagerState>`.
//! - **Concurrent merges**: Multiple non-overlapping merges can run in parallel.
//!   Each merge registers its segment IDs in `MergeInventory` via RAII `MergeGuard`.
//!   New merges are rejected only if they share segments with an active merge.
//! - **Auto-trigger**: Each completed merge re-evaluates the merge policy and spawns
//!   new merges if eligible (cascading merges for higher tiers).
//! - **ArcSwap for trained**: Lock-free reads of trained vector structures.
//!
//! # Locking model (deadlock-free by construction)
//!
//! ```text
//! Lock ordering (acquire in this order):
//!   1. state               — tokio::sync::Mutex, held for mutations + disk I/O
//!   2. merge_inventory     — parking_lot::Mutex (sync), sub-μs hold, RAII via MergeGuard
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
use crate::segment::{SegmentId, SegmentSnapshot, SegmentTracker, TrainedVectorStructures};
#[cfg(feature = "native")]
use crate::segment::{SegmentMerger, SegmentReader};

use super::{MergePolicy, SegmentInfo};

// ============================================================================
// RAII merge tracking
// ============================================================================

/// Tracks which segments are involved in active merges.
///
/// Supports multiple concurrent merges. Each merge registers its segment IDs;
/// a new merge is rejected only if any of its segments overlap with an active merge.
/// Uses RAII via `MergeGuard`: when a merge task ends, the guard drops and
/// its segment IDs are automatically unregistered.
struct MergeInventory {
    inner: parking_lot::Mutex<HashSet<String>>,
}

impl MergeInventory {
    fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(HashSet::new()),
        }
    }

    /// Try to register a merge. Returns `MergeGuard` on success, `None` if
    /// any of the requested segments are already in an active merge.
    fn try_register(self: &Arc<Self>, segment_ids: Vec<String>) -> Option<MergeGuard> {
        let mut inner = self.inner.lock();
        // Check for overlap with any active merge
        for id in &segment_ids {
            if inner.contains(id) {
                log::debug!(
                    "[merge_inventory] rejected: {} overlaps with active merge ({} active IDs)",
                    id,
                    inner.len()
                );
                return None;
            }
        }
        log::debug!(
            "[merge_inventory] registered {} IDs (total active: {})",
            segment_ids.len(),
            inner.len() + segment_ids.len()
        );
        for id in &segment_ids {
            inner.insert(id.clone());
        }
        Some(MergeGuard {
            inventory: Arc::clone(self),
            segment_ids,
        })
    }

    /// Snapshot of all in-merge segment IDs (for cleanup_orphan_segments)
    fn snapshot(&self) -> HashSet<String> {
        self.inner.lock().clone()
    }

    /// Check if a specific segment is currently involved in a merge.
    fn contains(&self, segment_id: &str) -> bool {
        self.inner.lock().contains(segment_id)
    }
}

/// RAII guard for a merge operation.
/// Dropped when the merge task completes (success, failure, or panic) —
/// automatically unregisters this merge's segment IDs from the inventory.
struct MergeGuard {
    inventory: Arc<MergeInventory>,
    segment_ids: Vec<String>,
}

impl Drop for MergeGuard {
    fn drop(&mut self) {
        let mut inner = self.inventory.inner.lock();
        for id in &self.segment_ids {
            inner.remove(id);
        }
    }
}

/// All mutable state behind the single async Mutex.
struct ManagerState {
    metadata: IndexMetadata,
    merge_policy: Box<dyn MergePolicy>,
}

/// Segment manager — coordinates segment commit, background merging, and trained structures.
///
/// SOLE owner of `metadata.json`. All metadata mutations go through `state` Mutex.
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Serializes ALL metadata mutations.
    state: AsyncMutex<ManagerState>,

    /// RAII merge tracking: segments are registered on merge start, automatically
    /// unregistered when the merge task ends (via MergeGuard drop).
    merge_inventory: Arc<MergeInventory>,

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
}

impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    /// Create a new segment manager with existing metadata
    pub fn new(
        directory: Arc<D>,
        schema: Arc<crate::dsl::Schema>,
        metadata: IndexMetadata,
        merge_policy: Box<dyn MergePolicy>,
        term_cache_blocks: usize,
        max_concurrent_merges: usize,
    ) -> Self {
        let tracker = Arc::new(SegmentTracker::new());
        for seg_id in metadata.segment_metas.keys() {
            tracker.register(seg_id);
        }

        let delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync> = {
            let dir = Arc::clone(&directory);
            Arc::new(move |segment_ids| {
                // Guard: if the tokio runtime is gone (program exit), skip async
                // deletion. Segment files become orphans cleaned up on next startup.
                let Ok(handle) = tokio::runtime::Handle::try_current() else {
                    return;
                };
                let dir = Arc::clone(&dir);
                handle.spawn(async move {
                    for segment_id in segment_ids {
                        log::info!(
                            "[segment_cleanup] deleting deferred segment {}",
                            segment_id.0
                        );
                        let _ = crate::segment::delete_segment(dir.as_ref(), segment_id).await;
                    }
                });
            })
        };

        Self {
            state: AsyncMutex::new(ManagerState {
                metadata,
                merge_policy,
            }),
            merge_inventory: Arc::new(MergeInventory::new()),
            merge_handles: AsyncMutex::new(Vec::new()),
            trained: ArcSwapOption::new(None),
            tracker,
            delete_fn,
            directory,
            schema,
            term_cache_blocks,
            max_concurrent_merges: max_concurrent_merges.max(1),
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
    pub(crate) async fn update_metadata<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut st = self.state.lock().await;
        f(&mut st.metadata);
        st.metadata.save(self.directory.as_ref()).await
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
    pub async fn commit(&self, new_segments: Vec<(String, u32)>) -> Result<()> {
        let mut st = self.state.lock().await;
        for (segment_id, num_docs) in new_segments {
            if !st.metadata.has_segment(&segment_id) {
                st.metadata.add_segment(segment_id.clone(), num_docs);
                self.tracker.register(&segment_id);
            }
        }
        st.metadata.save(self.directory.as_ref()).await
    }

    /// Evaluate merge policy and spawn background merges for all eligible candidates.
    ///
    /// **Atomicity**: The entire filter → find_merges → spawn_merge sequence runs
    /// under the `state` lock to prevent a TOCTOU race where concurrent callers
    /// both see segments as eligible before either registers them in the inventory.
    /// `spawn_merge` is non-blocking (just `try_register` + `tokio::spawn`), so
    /// holding the state lock through it is safe and sub-microsecond.
    ///
    /// Note: `max_concurrent_merges` is a soft limit — concurrent auto-triggers
    /// may briefly exceed it by one or two due to TOCTOU between slot counting
    /// and handle registration.
    pub async fn maybe_merge(self: &Arc<Self>) {
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

            // Exclude segments that are pending deletion OR already in an active merge.
            let segments: Vec<SegmentInfo> = st
                .metadata
                .segment_metas
                .iter()
                .filter(|(id, _)| {
                    !self.tracker.is_pending_deletion(id) && !self.merge_inventory.contains(id)
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
            // state lock released here — after spawn_merge registered IDs in inventory
        };

        if !new_handles.is_empty() {
            self.merge_handles.lock().await.extend(new_handles);
        }
    }

    /// Spawn a background merge task with RAII tracking.
    ///
    /// Pre-generates the output segment ID. `MergeGuard` registers all segment IDs
    /// (old + output) in `merge_inventory`. When the task ends (success, failure, or
    /// panic), the guard drops and segments are automatically unregistered.
    ///
    /// On completion, the task auto-triggers `maybe_merge` to evaluate cascading merges.
    /// Returns the JoinHandle if the merge was spawned, None if it was skipped.
    fn spawn_merge(self: &Arc<Self>, segment_ids_to_merge: Vec<String>) -> Option<JoinHandle<()>> {
        let output_id = SegmentId::new();
        let output_hex = output_id.to_hex();

        let mut all_ids = segment_ids_to_merge.clone();
        all_ids.push(output_hex);

        let guard = match self.merge_inventory.try_register(all_ids) {
            Some(g) => g,
            None => {
                log::debug!("[spawn_merge] skipped: segments overlap with active merge");
                return None;
            }
        };

        let sm = Arc::clone(self);
        let ids = segment_ids_to_merge;

        Some(tokio::spawn(async move {
            let _guard = guard;

            let trained_snap = sm.trained();
            let result = Self::do_merge(
                sm.directory.as_ref(),
                &sm.schema,
                &ids,
                output_id,
                sm.term_cache_blocks,
                trained_snap.as_deref(),
                false,
            )
            .await;

            match result {
                Ok((new_id, doc_count)) => {
                    if let Err(e) = sm.replace_segments(&ids, new_id, doc_count, false).await {
                        log::error!("[merge] Failed to replace segments after merge: {:?}", e);
                    }
                }
                Err(e) => {
                    log::error!(
                        "[merge] Background merge failed for segments {:?}: {:?}",
                        ids,
                        e
                    );
                }
            }
            // _guard drops here → segment IDs unregistered from inventory

            // Auto-trigger: re-evaluate merge policy after this merge completes.
            // The merged output may now be eligible for a higher-tier merge.
            sm.maybe_merge().await;
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
    ) -> Result<()> {
        self.tracker.register(&new_id);

        {
            let mut st = self.state.lock().await;
            // Compute generation from parents before removing them
            let parent_gen = old_ids
                .iter()
                .filter_map(|id| st.metadata.segment_metas.get(id))
                .map(|info| info.generation)
                .max()
                .unwrap_or(0);
            let ancestors: Vec<String> = old_ids.to_vec();

            for id in old_ids {
                st.metadata.remove_segment(id);
            }
            st.metadata
                .add_merged_segment(new_id, doc_count, ancestors, parent_gen + 1, reordered);
            // Mutation + persist must be atomic — keep under lock
            st.metadata.save(self.directory.as_ref()).await?;
        }

        let ready_to_delete = self.tracker.mark_for_deletion(old_ids);
        for segment_id in ready_to_delete {
            let _ = crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
        }
        Ok(())
    }

    /// Perform the actual merge operation (pure function — no shared state access).
    /// `output_segment_id` is pre-generated by the caller so it can be tracked in `merge_inventory`.
    /// When `force_reorder` is true, all BMP fields get record-level BP reordering.
    /// Returns (new_segment_id_hex, total_doc_count).
    pub(crate) async fn do_merge(
        directory: &D,
        schema: &Arc<crate::dsl::Schema>,
        segment_ids_to_merge: &[String],
        output_segment_id: SegmentId,
        term_cache_blocks: usize,
        trained: Option<&TrainedVectorStructures>,
        force_reorder: bool,
    ) -> Result<(String, u32)> {
        let output_hex = output_segment_id.to_hex();
        let load_start = std::time::Instant::now();

        let segment_ids: Vec<SegmentId> = segment_ids_to_merge
            .iter()
            .map(|id_str| {
                SegmentId::from_hex(id_str)
                    .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))
            })
            .collect::<Result<Vec<_>>>()?;

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
                    return Err(e);
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
                return Err(Error::Corruption(format!(
                    "pre-merge validation: segment {} store has {} docs but meta says {}",
                    segment_ids_to_merge[i], store_docs, meta_docs
                )));
            }
        }

        log::info!(
            "[merge] loaded {} segment readers in {:.1}s",
            readers.len(),
            load_start.elapsed().as_secs_f64()
        );

        let merger = SegmentMerger::new(Arc::clone(schema)).with_force_reorder(force_reorder);

        log::info!(
            "[merge] {} segments -> {} (trained={}, force_reorder={})",
            segment_ids_to_merge.len(),
            output_hex,
            trained.map_or(0, |t| t.centroids.len()),
            force_reorder,
        );

        merger
            .merge(directory, &readers, output_segment_id, trained)
            .await?;

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
            )));
        }
        Ok((output_hex, total_docs as u32))
    }

    /// Abort all in-flight merge tasks without waiting for completion.
    /// Used during index deletion to stop background work immediately.
    pub async fn abort_merges(&self) {
        let handles: Vec<JoinHandle<()>> =
            { std::mem::take(&mut *self.merge_handles.lock().await) };
        for h in handles {
            h.abort();
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

    /// Force merge all segments into one. Iterates in batches until ≤1 segment remains.
    ///
    /// Each batch is registered in `merge_inventory` via `MergeGuard` to prevent
    /// `maybe_merge` from spawning a conflicting background merge.
    pub async fn force_merge(self: &Arc<Self>) -> Result<()> {
        const FORCE_MERGE_BATCH: usize = 64;

        // Wait for all in-flight background merges (including cascading)
        // before starting forced merges to avoid try_register conflicts.
        self.wait_for_all_merges().await;

        loop {
            let ids_to_merge = self.get_segment_ids().await;
            if ids_to_merge.len() < 2 {
                return Ok(());
            }

            let batch: Vec<String> = ids_to_merge.into_iter().take(FORCE_MERGE_BATCH).collect();

            log::info!("[force_merge] merging batch of {} segments", batch.len());

            let output_id = SegmentId::new();
            let output_hex = output_id.to_hex();

            // Register batch + output in inventory so maybe_merge skips them.
            let mut all_ids = batch.clone();
            all_ids.push(output_hex);
            let _guard = match self.merge_inventory.try_register(all_ids) {
                Some(g) => g,
                None => {
                    // A background merge slipped in — wait for it, then retry the loop
                    self.wait_for_merging_thread().await;
                    continue;
                }
            };

            let trained_snap = self.trained();
            let (new_segment_id, total_docs) = Self::do_merge(
                self.directory.as_ref(),
                &self.schema,
                &batch,
                output_id,
                self.term_cache_blocks,
                trained_snap.as_deref(),
                false,
            )
            .await?;

            self.replace_segments(&batch, new_segment_id, total_docs, false)
                .await?;

            // _guard drops here → segments unregistered from inventory
        }
    }

    /// Reorder all segments via Recursive Graph Bisection (BP) for better BMP pruning.
    ///
    /// Each segment is individually rebuilt with record-level BP reordering:
    /// ordinals are shuffled across blocks so that similar content clusters tightly.
    /// Non-BMP fields pass through unchanged (identity-copied via merge).
    ///
    /// Uses the same locking infrastructure as merge: `MergeInventory` prevents
    /// concurrent operations on the same segment, `replace_segments()` does atomic
    /// metadata swap.
    pub async fn reorder_segments(self: &Arc<Self>) -> Result<()> {
        self.wait_for_all_merges().await;
        let segment_ids = self.get_segment_ids().await;

        if segment_ids.is_empty() {
            log::info!("[reorder] no segments to reorder");
            return Ok(());
        }

        log::info!("[reorder] reordering {} segments", segment_ids.len());

        for seg_id in segment_ids {
            let output_id = SegmentId::new();
            let output_hex = output_id.to_hex();

            let all_ids = vec![seg_id.clone(), output_hex];
            let _guard = match self.merge_inventory.try_register(all_ids) {
                Some(g) => g,
                None => {
                    log::warn!("[reorder] segment {} in active merge, skipping", seg_id);
                    continue;
                }
            };

            let trained_snap = self.trained();
            let (new_id, total_docs) = Self::do_merge(
                self.directory.as_ref(),
                &self.schema,
                std::slice::from_ref(&seg_id),
                output_id,
                self.term_cache_blocks,
                trained_snap.as_deref(),
                true, // force_reorder
            )
            .await?;

            self.replace_segments(&[seg_id], new_id, total_docs, true)
                .await?;
        }

        log::info!("[reorder] all segments reordered");
        Ok(())
    }

    /// Get segment IDs that have not been reordered yet.
    ///
    /// Used by background optimizer to find segments that need BP reordering.
    pub async fn unreordered_segment_ids(&self) -> Vec<String> {
        let st = self.state.lock().await;
        st.metadata
            .segment_metas
            .iter()
            .filter(|(_, info)| !info.reordered)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Reorder a single segment via BP. Returns Ok(true) if reordered, Ok(false) if skipped.
    ///
    /// Non-blocking: uses merge inventory to prevent conflicts with background merges.
    /// Does NOT wait for in-flight merges — cooperates with them instead.
    pub async fn reorder_single_segment(self: &Arc<Self>, seg_id: &str) -> Result<bool> {
        let output_id = SegmentId::new();
        let output_hex = output_id.to_hex();

        let all_ids = vec![seg_id.to_string(), output_hex];
        let _guard = match self.merge_inventory.try_register(all_ids) {
            Some(g) => g,
            None => {
                log::debug!("[optimizer] segment {} in active merge, skipping", seg_id);
                return Ok(false);
            }
        };

        let trained_snap = self.trained();
        let (new_id, total_docs) = Self::do_merge(
            self.directory.as_ref(),
            &self.schema,
            std::slice::from_ref(&seg_id.to_string()),
            output_id,
            self.term_cache_blocks,
            trained_snap.as_deref(),
            true, // force_reorder
        )
        .await?;

        self.replace_segments(&[seg_id.to_string()], new_id, total_docs, true)
            .await?;

        Ok(true)
    }

    /// Clean up orphan segment files not registered in metadata.
    ///
    /// Non-blocking: reads both metadata and `merge_inventory` to determine which
    /// segments are legitimate. In-flight merge outputs are protected by the inventory.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        // Read BOTH sets under the same state lock to prevent TOCTOU:
        // without this, a merge completing between the two reads could make
        // its output segment invisible to both sets → wrongly deleted.
        let (registered_set, in_merge_set) = {
            let st = self.state.lock().await;
            let registered = st
                .metadata
                .segment_metas
                .keys()
                .cloned()
                .collect::<HashSet<String>>();
            let in_merge = self.merge_inventory.snapshot();
            (registered, in_merge)
        };

        let mut orphan_ids: HashSet<String> = HashSet::new();

        if let Ok(entries) = self.directory.list_files(std::path::Path::new("")).await {
            for entry in entries {
                let filename = entry.to_string_lossy();
                if filename.starts_with("seg_") && filename.len() > 37 {
                    let hex_part = &filename[4..36];
                    if !registered_set.contains(hex_part) && !in_merge_set.contains(hex_part) {
                        orphan_ids.insert(hex_part.to_string());
                    }
                }
            }
        }

        let mut deleted = 0;
        for hex_id in &orphan_ids {
            if let Some(segment_id) = SegmentId::from_hex(hex_id)
                && crate::segment::delete_segment(self.directory.as_ref(), segment_id)
                    .await
                    .is_ok()
            {
                deleted += 1;
            }
        }

        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_guard_drop_unregisters() {
        let inv = Arc::new(MergeInventory::new());
        {
            let _guard = inv.try_register(vec!["a".into(), "b".into()]).unwrap();
            let snap = inv.snapshot();
            assert!(snap.contains("a"));
            assert!(snap.contains("b"));
        }
        // Guard dropped → segments unregistered
        assert!(inv.snapshot().is_empty());
    }

    #[test]
    fn test_inventory_concurrent_non_overlapping_merges() {
        let inv = Arc::new(MergeInventory::new());
        let _g1 = inv.try_register(vec!["a".into(), "b".into()]).unwrap();
        // Non-overlapping merge succeeds concurrently
        let _g2 = inv.try_register(vec!["c".into(), "d".into()]).unwrap();
        let snap = inv.snapshot();
        assert_eq!(snap.len(), 4);

        // Drop first guard — only its segments are removed
        drop(_g1);
        let snap = inv.snapshot();
        assert_eq!(snap.len(), 2);
        assert!(snap.contains("c"));
        assert!(snap.contains("d"));
    }

    #[test]
    fn test_inventory_overlapping_merge_rejected() {
        let inv = Arc::new(MergeInventory::new());
        let _g1 = inv.try_register(vec!["a".into(), "b".into()]).unwrap();
        // Overlapping merge rejected (shares "b")
        assert!(inv.try_register(vec!["b".into(), "c".into()]).is_none());
        // After drop, the overlapping merge succeeds
        drop(_g1);
        assert!(inv.try_register(vec!["b".into(), "c".into()]).is_some());
    }

    #[test]
    fn test_inventory_snapshot() {
        let inv = Arc::new(MergeInventory::new());
        let _g = inv.try_register(vec!["x".into(), "y".into()]).unwrap();
        let snap = inv.snapshot();
        assert!(snap.contains("x"));
        assert!(snap.contains("y"));
        assert!(!snap.contains("z"));
    }
}
