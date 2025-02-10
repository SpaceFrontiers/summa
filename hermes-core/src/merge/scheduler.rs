//! Segment manager — coordinates segment commit, background merging, and trained structures.
//!
//! Architecture (tantivy-inspired):
//! - **Single mutation queue**: All metadata mutations serialize through `tokio::sync::Mutex<ManagerState>`.
//!   This is the async equivalent of tantivy's single-threaded rayon pool.
//! - **JoinHandle merge tracking**: Background merges tracked via `Vec<JoinHandle>`, not `AtomicUsize + Notify`.
//! - **Explicit `end_merge`**: No Drop guards — merge cleanup always runs after merge completes.
//! - **ArcSwap for trained**: Lock-free reads of trained vector structures.
//!
//! # Locking model (deadlock-free by construction)
//!
//! ```text
//! Lock ordering (acquire in this order):
//!   1. state           — tokio::sync::Mutex, held for mutations + disk I/O
//!   2. merging         — parking_lot::Mutex (sync), sub-μs hold
//!   3. tracker.inner   — parking_lot::Mutex (sync), sub-μs hold
//!
//! Lock-free state:
//!   trained            — arc_swap::ArcSwapOption, no ordering constraint
//!   merge_handles      — tokio::sync::Mutex, never held with state
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

use super::consts::MAX_CONCURRENT_MERGES;
use super::{MergePolicy, SegmentInfo};

/// All mutable state behind the single async Mutex.
struct ManagerState {
    metadata: IndexMetadata,
    merge_policy: Box<dyn MergePolicy>,
}

/// Segment manager — coordinates segment commit, background merging, and trained structures.
///
/// SOLE owner of `metadata.json`. All metadata mutations go through `state` Mutex.
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Serializes ALL metadata mutations (tantivy's single-threaded pool equivalent).
    state: AsyncMutex<ManagerState>,

    /// Segments currently being merged. `parking_lot::Mutex` because `end_merge`
    /// needs sync access from the spawned task after merge completes.
    merging: parking_lot::Mutex<HashSet<String>>,

    /// In-flight merge JoinHandles. Replaces `AtomicUsize + Notify`.
    merge_handles: AsyncMutex<Vec<JoinHandle<()>>>,

    /// Trained vector structures — lock-free reads via ArcSwap.
    trained: ArcSwapOption<TrainedVectorStructures>,

    /// Reference counting for safe segment deletion (sync Mutex for Drop).
    tracker: Arc<SegmentTracker>,

    /// Directory for segment I/O
    directory: Arc<D>,
    /// Schema for segment operations
    schema: Arc<crate::dsl::Schema>,
    /// Term cache blocks for segment readers during merge
    term_cache_blocks: usize,
}

impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    /// Create a new segment manager with existing metadata
    pub fn new(
        directory: Arc<D>,
        schema: Arc<crate::dsl::Schema>,
        metadata: IndexMetadata,
        merge_policy: Box<dyn MergePolicy>,
        term_cache_blocks: usize,
    ) -> Self {
        let tracker = Arc::new(SegmentTracker::new());
        for seg_id in metadata.segment_metas.keys() {
            tracker.register(seg_id);
        }

        Self {
            state: AsyncMutex::new(ManagerState {
                metadata,
                merge_policy,
            }),
            merging: parking_lot::Mutex::new(HashSet::new()),
            merge_handles: AsyncMutex::new(Vec::new()),
            trained: ArcSwapOption::new(None),
            tracker,
            directory,
            schema,
            term_cache_blocks,
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

    /// Publish trained vector structures (lock-free via ArcSwap)
    pub fn publish_trained(&self, trained: TrainedVectorStructures) {
        self.trained.store(Some(Arc::new(trained)));
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
    pub fn clear_trained(&self) {
        self.trained.store(None);
    }

    /// Read metadata with a closure (no persist)
    pub async fn read_metadata<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&IndexMetadata) -> R,
    {
        let st = self.state.lock().await;
        f(&st.metadata)
    }

    /// Update metadata with a closure and persist atomically
    pub async fn update_metadata<F>(&self, f: F) -> Result<()>
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

        let dir = Arc::clone(&self.directory);
        let delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync> = Arc::new(move |segment_ids| {
            let dir = Arc::clone(&dir);
            tokio::spawn(async move {
                for segment_id in segment_ids {
                    log::info!(
                        "[segment_cleanup] deleting deferred segment {}",
                        segment_id.0
                    );
                    let _ = crate::segment::delete_segment(dir.as_ref(), segment_id).await;
                }
            });
        });

        SegmentSnapshot::with_delete_fn(Arc::clone(&self.tracker), acquired, delete_fn)
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
    /// Like tantivy's `segment_manager.commit()` + `save_metas()`.
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

    /// Evaluate merge policy and spawn background merges if needed.
    /// Like tantivy's `consider_merge_options()`.
    ///
    /// Single lock scope: builds segment list AND calls find_merges atomically
    /// to prevent stale-list races with concurrent end_merge.
    pub async fn maybe_merge(self: &Arc<Self>) {
        let (candidates, num_eligible, num_merging) = {
            let st = self.state.lock().await;
            let merging = self.merging.lock();

            let segments: Vec<SegmentInfo> = st
                .metadata
                .segment_metas
                .iter()
                .filter(|(id, _)| !merging.contains(*id) && !self.tracker.is_pending_deletion(id))
                .map(|(id, info)| SegmentInfo {
                    id: id.clone(),
                    num_docs: info.num_docs,
                    size_bytes: None,
                })
                .collect();

            let num_merging = merging.len();
            let num_eligible = segments.len();
            let candidates = st.merge_policy.find_merges(&segments);
            (candidates, num_eligible, num_merging)
        };

        log::debug!(
            "[maybe_merge] {} eligible segments, {} merging -> {} merge candidates",
            num_eligible,
            num_merging,
            candidates.len(),
        );

        for candidate in candidates {
            if candidate.segment_ids.len() >= 2 {
                self.spawn_merge(candidate.segment_ids).await;
            }
        }
    }

    /// Spawn a background merge task. Tracks via JoinHandle (no AtomicUsize/Notify).
    async fn spawn_merge(self: &Arc<Self>, segment_ids_to_merge: Vec<String>) {
        // Limit concurrent merges
        {
            let handles = self.merge_handles.lock().await;
            if handles.len() >= MAX_CONCURRENT_MERGES {
                log::debug!(
                    "[spawn_merge] skipped: {} active merges >= {}",
                    handles.len(),
                    MAX_CONCURRENT_MERGES
                );
                return;
            }
        }

        // Atomically check and mark segments as merging
        {
            let mut merging = self.merging.lock();
            if segment_ids_to_merge.iter().any(|id| merging.contains(id)) {
                return;
            }
            for id in &segment_ids_to_merge {
                merging.insert(id.clone());
            }
        }

        let sm = Arc::clone(self);
        let ids = segment_ids_to_merge;

        let handle = tokio::spawn(async move {
            let trained_snap = sm.trained();
            let result = Self::do_merge(
                sm.directory.as_ref(),
                &sm.schema,
                &ids,
                sm.term_cache_blocks,
                trained_snap.as_deref(),
            )
            .await;

            // Explicit end_merge — ALWAYS runs (tantivy pattern, no Drop guard)
            sm.end_merge(&ids, result).await;
        });

        self.merge_handles.lock().await.push(handle);
    }

    /// Complete a merge: update metadata, clean up merging set, delete old segments.
    /// Always called after merge (success or failure). Like tantivy's `end_merge`.
    async fn end_merge(&self, old_ids: &[String], result: Result<(String, u32)>) {
        // 1. Remove from merging set (sync, sub-μs)
        {
            let mut merging = self.merging.lock();
            for id in old_ids {
                merging.remove(id);
            }
        }

        // 2. On success: replace segments in metadata + delete old files
        match result {
            Ok((new_id, doc_count)) => {
                if let Err(e) = self.replace_segments(old_ids, new_id, doc_count).await {
                    log::error!("[merge] Failed to replace segments after merge: {:?}", e);
                }
            }
            Err(e) => {
                log::error!(
                    "[merge] Background merge failed for segments {:?}: {:?}",
                    old_ids,
                    e
                );
            }
        }
    }

    /// Atomically replace old segments with a new merged segment.
    /// Shared by end_merge and force_merge.
    async fn replace_segments(
        &self,
        old_ids: &[String],
        new_id: String,
        doc_count: u32,
    ) -> Result<()> {
        self.tracker.register(&new_id);

        {
            let mut st = self.state.lock().await;
            for id in old_ids {
                st.metadata.remove_segment(id);
            }
            st.metadata.add_segment(new_id, doc_count);
            st.metadata.save(self.directory.as_ref()).await?;
        }

        let ready_to_delete = self.tracker.mark_for_deletion(old_ids);
        for segment_id in ready_to_delete {
            let _ = crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
        }
        Ok(())
    }

    /// Perform the actual merge operation (pure function — no shared state access).
    /// Returns (new_segment_id_hex, total_doc_count).
    pub async fn do_merge(
        directory: &D,
        schema: &crate::dsl::Schema,
        segment_ids_to_merge: &[String],
        term_cache_blocks: usize,
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<(String, u32)> {
        let load_start = std::time::Instant::now();

        let segment_ids: Vec<SegmentId> = segment_ids_to_merge
            .iter()
            .map(|id_str| {
                SegmentId::from_hex(id_str)
                    .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))
            })
            .collect::<Result<Vec<_>>>()?;

        let schema_arc = Arc::new(schema.clone());
        let futures: Vec<_> = segment_ids
            .iter()
            .map(|&sid| {
                let sch = Arc::clone(&schema_arc);
                async move { SegmentReader::open(directory, sid, sch, 0, term_cache_blocks).await }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut readers = Vec::with_capacity(results.len());
        let mut total_docs = 0u32;
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(r) => {
                    total_docs += r.meta().num_docs;
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

        log::info!(
            "[merge] loaded {} segment readers in {:.1}s",
            readers.len(),
            load_start.elapsed().as_secs_f64()
        );

        let merger = SegmentMerger::new(schema_arc);
        let new_segment_id = SegmentId::new();

        log::info!(
            "[merge] {} segments -> {} (trained={})",
            segment_ids_to_merge.len(),
            new_segment_id.to_hex(),
            trained.map_or(0, |t| t.centroids.len())
        );

        let merge_result = merger
            .merge(directory, &readers, new_segment_id, trained)
            .await;

        if let Err(e) = merge_result {
            log::error!(
                "[merge] Merge failed for segments {:?} -> {}: {:?}",
                segment_ids_to_merge,
                new_segment_id.to_hex(),
                e
            );
            return Err(e);
        }

        log::info!(
            "[merge] total wall-clock: {:.1}s ({} segments, {} docs)",
            load_start.elapsed().as_secs_f64(),
            readers.len(),
            total_docs,
        );

        Ok((new_segment_id.to_hex(), total_docs))
    }

    /// Wait for all in-flight background merges to complete.
    /// Uses JoinHandle — can't lose notifications (fixes the hang bug).
    pub async fn wait_for_merges(&self) {
        let handles: Vec<JoinHandle<()>> = std::mem::take(&mut *self.merge_handles.lock().await);
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Force merge all segments into one. Iterates in batches until ≤1 segment remains.
    /// Moved from IndexWriter to centralize all merge logic in SegmentManager.
    pub async fn force_merge(self: &Arc<Self>) -> Result<()> {
        use super::consts::FORCE_MERGE_BATCH;

        // First wait for any in-flight background merges
        self.wait_for_merges().await;

        loop {
            let ids_to_merge = self.get_segment_ids().await;
            if ids_to_merge.len() < 2 {
                return Ok(());
            }

            let batch: Vec<String> = ids_to_merge.into_iter().take(FORCE_MERGE_BATCH).collect();

            log::info!("[force_merge] merging batch of {} segments", batch.len());

            let trained_snap = self.trained();
            let (new_segment_id, total_docs) = Self::do_merge(
                self.directory.as_ref(),
                &self.schema,
                &batch,
                self.term_cache_blocks,
                trained_snap.as_deref(),
            )
            .await?;

            self.replace_segments(&batch, new_segment_id, total_docs)
                .await?;
        }
    }

    /// Clean up orphan segment files not registered in metadata.
    ///
    /// Waits for in-flight merges first — a merge creates segment files before
    /// updating metadata, so without waiting we'd delete freshly merged segments.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        self.wait_for_merges().await;

        let registered_set: HashSet<String> = {
            let st = self.state.lock().await;
            st.metadata.segment_metas.keys().cloned().collect()
        };

        let mut orphan_ids: HashSet<String> = HashSet::new();

        if let Ok(entries) = self.directory.list_files(std::path::Path::new("")).await {
            for entry in entries {
                let filename = entry.to_string_lossy();
                if filename.starts_with("seg_") && filename.len() > 37 {
                    let hex_part = &filename[4..36];
                    if !registered_set.contains(hex_part) {
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
