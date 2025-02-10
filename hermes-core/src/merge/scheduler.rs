//! Segment manager - coordinates segment registration and background merging
//!
//! This module is only compiled with the "native" feature.
//!
//! The SegmentManager is the SOLE owner of `metadata.json`. All writes to metadata
//! go through this manager, ensuring linearized access and consistency between
//! the in-memory segment list and persisted state.
//!
//! **State separation:**
//! - Building segments: Managed by IndexWriter (pending_builds)
//! - Committed segments: Managed by SegmentManager (metadata.segments)
//! - Merging segments: Subset of committed, tracked here (merging_segments)
//!
//! **Commit workflow:**
//! 1. IndexWriter flushes builders, waits for builds to complete
//! 2. Calls `register_segment()` for each completed segment
//! 3. SegmentManager updates metadata atomically, triggers merge check (non-blocking)
//!
//! **Merge workflow (background):**
//! 1. Acquires segments to merge (marks as merging)
//! 2. Merges into new segment
//! 3. Calls internal `complete_merge()` which atomically updates metadata

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::RwLock as SyncRwLock;
use tokio::sync::{Notify, RwLock};

use crate::directories::DirectoryWriter;
use crate::error::{Error, Result};
use crate::index::IndexMetadata;
use crate::segment::{SegmentId, SegmentSnapshot, SegmentTracker};
#[cfg(feature = "native")]
use crate::segment::{SegmentMerger, SegmentReader};

use super::{MergePolicy, SegmentInfo};

/// Maximum number of concurrent merge operations
const MAX_CONCURRENT_MERGES: usize = 2;

/// Segment manager - coordinates segment registration and background merging
///
/// This is the SOLE owner of `metadata.json` ensuring linearized access.
/// All segment list modifications and metadata updates go through here.
///
/// Uses RwLock for metadata to allow concurrent reads (search) while
/// writes (indexing/merge) get exclusive access.
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Directory for segment operations
    directory: Arc<D>,
    /// Schema for segment operations
    schema: Arc<crate::dsl::Schema>,
    /// Unified metadata (segments + vector index state) - SOLE owner
    /// RwLock allows concurrent reads, exclusive writes
    metadata: Arc<RwLock<IndexMetadata>>,
    /// The merge policy to use
    merge_policy: Box<dyn MergePolicy>,
    /// Count of in-flight background merges
    pending_merges: Arc<AtomicUsize>,
    /// Segments currently being merged (to avoid double-merging)
    merging_segments: Arc<SyncRwLock<HashSet<String>>>,
    /// Term cache blocks for segment readers during merge
    term_cache_blocks: usize,
    /// Notifier for merge completion (avoids busy-waiting)
    merge_complete: Arc<Notify>,
    /// Segment lifecycle tracker for reference counting
    tracker: Arc<SegmentTracker>,
    /// Pause flag to prevent new merges during ANN rebuild
    merge_paused: Arc<AtomicBool>,
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
        // Initialize tracker and register existing segments
        let tracker = Arc::new(SegmentTracker::new());
        for seg_id in metadata.segment_metas.keys() {
            tracker.register(seg_id);
        }

        Self {
            directory,
            schema,
            metadata: Arc::new(RwLock::new(metadata)),
            merge_policy,
            pending_merges: Arc::new(AtomicUsize::new(0)),
            merging_segments: Arc::new(SyncRwLock::new(HashSet::new())),
            term_cache_blocks,
            merge_complete: Arc::new(Notify::new()),
            tracker,
            merge_paused: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the current segment IDs (snapshot)
    pub async fn get_segment_ids(&self) -> Vec<String> {
        self.metadata.read().await.segment_ids()
    }

    /// Get the number of pending background merges
    pub fn pending_merge_count(&self) -> usize {
        self.pending_merges.load(Ordering::SeqCst)
    }

    /// Get a clone of the metadata Arc for read access
    pub fn metadata(&self) -> Arc<RwLock<IndexMetadata>> {
        Arc::clone(&self.metadata)
    }

    /// Update metadata with a closure and persist atomically
    pub async fn update_metadata<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut meta = self.metadata.write().await;
        f(&mut meta);
        meta.save(self.directory.as_ref()).await
    }

    /// Acquire a snapshot of current segments for reading
    /// The snapshot holds references - segments won't be deleted while snapshot exists
    pub async fn acquire_snapshot(&self) -> SegmentSnapshot<D> {
        let acquired = {
            let meta = self.metadata.read().await;
            let segment_ids = meta.segment_ids();
            self.tracker.acquire(&segment_ids)
        };

        // Provide a deletion callback so deferred segment cleanup happens
        // when this snapshot is dropped (after in-flight searches finish)
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

/// Native-only methods for SegmentManager (merging, segment registration)
#[cfg(feature = "native")]
impl<D: DirectoryWriter + 'static> SegmentManager<D> {
    /// Register a new segment with its doc count, persist metadata, and trigger merge check
    ///
    /// This is the main entry point for adding segments after builds complete.
    pub async fn register_segment(
        self: &Arc<Self>,
        segment_id: String,
        num_docs: u32,
    ) -> Result<()> {
        {
            let mut meta = self.metadata.write().await;
            if !meta.has_segment(&segment_id) {
                meta.add_segment(segment_id.clone(), num_docs);
                self.tracker.register(&segment_id);
            }
            meta.save(self.directory.as_ref()).await?;
        }

        // Check if we should trigger a merge (non-blocking)
        self.maybe_merge().await;
        Ok(())
    }

    /// Check merge policy and spawn background merges if needed
    /// Uses doc counts from metadata - no segment loading required
    pub async fn maybe_merge(self: &Arc<Self>) {
        // Get current segment info from metadata (no segment loading!)
        let segments: Vec<SegmentInfo> = {
            let meta = self.metadata.read().await;
            let merging = self.merging_segments.read();

            log::debug!(
                "[maybe_merge] meta has {} segments, {} merging, paused={}, pending_merges={}",
                meta.segment_metas.len(),
                merging.len(),
                self.merge_paused.load(Ordering::SeqCst),
                self.pending_merges.load(Ordering::SeqCst),
            );

            // Filter out segments currently being merged or pending deletion
            meta.segment_metas
                .iter()
                .filter(|(id, _)| !merging.contains(*id) && !self.tracker.is_pending_deletion(id))
                .map(|(id, info)| SegmentInfo {
                    id: id.clone(),
                    num_docs: info.num_docs,
                    size_bytes: None,
                })
                .collect()
        };

        // Ask merge policy for candidates
        let candidates = self.merge_policy.find_merges(&segments);

        log::debug!(
            "[maybe_merge] {} eligible segments -> {} merge candidates",
            segments.len(),
            candidates.len(),
        );

        for candidate in candidates {
            if candidate.segment_ids.len() >= 2 {
                self.spawn_merge(candidate.segment_ids);
            }
        }
    }

    /// Pause background merges (used during ANN rebuild to prevent races)
    pub fn pause_merges(&self) {
        self.merge_paused.store(true, Ordering::SeqCst);
    }

    /// Resume background merges
    pub fn resume_merges(&self) {
        self.merge_paused.store(false, Ordering::SeqCst);
    }

    /// Spawn a background merge task
    fn spawn_merge(self: &Arc<Self>, segment_ids_to_merge: Vec<String>) {
        // Skip if merges are paused (during ANN rebuild)
        if self.merge_paused.load(Ordering::SeqCst) {
            log::debug!("[spawn_merge] skipped: merges paused");
            return;
        }
        // Limit concurrent merges to avoid overwhelming the system during heavy indexing
        if self.pending_merges.load(Ordering::SeqCst) >= MAX_CONCURRENT_MERGES {
            log::debug!(
                "[spawn_merge] skipped: pending_merges >= {}",
                MAX_CONCURRENT_MERGES
            );
            return;
        }

        // Atomically check and mark segments as being merged
        // This prevents race conditions where multiple maybe_merge calls
        // could pick the same segments before they're marked
        {
            let mut merging = self.merging_segments.write();
            // Check if any segment is already being merged
            if segment_ids_to_merge.iter().any(|id| merging.contains(id)) {
                // Some segment already being merged, skip this merge
                return;
            }
            // Mark all segments as being merged
            for id in &segment_ids_to_merge {
                merging.insert(id.clone());
            }
        }

        let directory = Arc::clone(&self.directory);
        let schema = Arc::clone(&self.schema);
        let metadata = Arc::clone(&self.metadata);
        let merging_segments = Arc::clone(&self.merging_segments);
        let pending_merges = Arc::clone(&self.pending_merges);
        let merge_complete = Arc::clone(&self.merge_complete);
        let tracker = Arc::clone(&self.tracker);
        let term_cache_blocks = self.term_cache_blocks;
        let self_clone = Arc::clone(self);

        pending_merges.fetch_add(1, Ordering::SeqCst);

        tokio::spawn(async move {
            let result = Self::do_merge(
                directory.as_ref(),
                &schema,
                &segment_ids_to_merge,
                term_cache_blocks,
                &metadata,
            )
            .await;

            match result {
                Ok((new_segment_id, merged_doc_count)) => {
                    // Register new segment with tracker
                    tracker.register(&new_segment_id);

                    // Atomically update metadata: remove merged segments, add new one with doc count
                    {
                        let mut meta = metadata.write().await;
                        for id in &segment_ids_to_merge {
                            meta.remove_segment(id);
                        }
                        meta.add_segment(new_segment_id, merged_doc_count);
                        if let Err(e) = meta.save(directory.as_ref()).await {
                            log::error!("[merge] Failed to save metadata after merge: {:?}", e);
                        }
                    }

                    // Mark old segments for deletion via tracker (deferred if refs exist)
                    let ready_to_delete = tracker.mark_for_deletion(&segment_ids_to_merge);
                    for segment_id in ready_to_delete {
                        let _ =
                            crate::segment::delete_segment(directory.as_ref(), segment_id).await;
                    }
                }
                Err(e) => {
                    log::error!(
                        "[merge] Background merge failed for segments {:?}: {:?}",
                        segment_ids_to_merge,
                        e
                    );
                }
            }

            // Remove from merging set
            {
                let mut merging = merging_segments.write();
                for id in &segment_ids_to_merge {
                    merging.remove(id);
                }
            }

            // Decrement pending merges counter and notify waiters
            pending_merges.fetch_sub(1, Ordering::SeqCst);
            merge_complete.notify_waiters();

            // Re-evaluate merge policy — there may be more segments to merge
            // that were blocked by MAX_CONCURRENT_MERGES during commit
            self_clone.maybe_merge().await;
        });
    }

    /// Perform the actual merge operation.
    /// Returns (new_segment_id_hex, total_doc_count).
    /// Used by both background merges and force_merge.
    pub async fn do_merge(
        directory: &D,
        schema: &crate::dsl::Schema,
        segment_ids_to_merge: &[String],
        term_cache_blocks: usize,
        metadata: &RwLock<IndexMetadata>,
    ) -> Result<(String, u32)> {
        let load_start = std::time::Instant::now();

        // Parse segment IDs upfront
        let segment_ids: Vec<SegmentId> = segment_ids_to_merge
            .iter()
            .map(|id_str| {
                SegmentId::from_hex(id_str)
                    .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))
            })
            .collect::<Result<Vec<_>>>()?;

        // Load segment readers in parallel (doc_offset=0; merger computes its own)
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

        // Load trained structures (if any) for ANN-aware merging
        let trained = {
            let meta = metadata.read().await;
            meta.load_trained_structures(directory).await
        };

        let merger = SegmentMerger::new(schema_arc);
        let new_segment_id = SegmentId::new();

        log::info!(
            "[merge] {} segments -> {} (trained={})",
            segment_ids_to_merge.len(),
            new_segment_id.to_hex(),
            trained.as_ref().map_or(0, |t| t.centroids.len())
        );

        let merge_result = merger
            .merge(directory, &readers, new_segment_id, trained.as_ref())
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

        // Note: Segment deletion is handled by the caller via tracker
        Ok((new_segment_id.to_hex(), total_docs))
    }

    /// Wait for all pending merges to complete
    pub async fn wait_for_merges(&self) {
        while self.pending_merges.load(Ordering::SeqCst) > 0 {
            self.merge_complete.notified().await;
        }
    }

    /// Replace specific old segments with new ones atomically.
    ///
    /// Only removes `old_to_delete` from metadata (not all segments), then adds
    /// `new_segments`. This is safe against concurrent ingestion: segments committed
    /// between `get_segment_ids()` and this call are preserved.
    pub async fn replace_segments(
        &self,
        new_segments: Vec<(String, u32)>,
        old_to_delete: Vec<String>,
    ) -> Result<()> {
        // Register new segments with tracker
        for (seg_id, _) in &new_segments {
            self.tracker.register(seg_id);
        }

        {
            let mut meta = self.metadata.write().await;
            for id in &old_to_delete {
                meta.remove_segment(id);
            }
            for (seg_id, num_docs) in new_segments {
                meta.add_segment(seg_id, num_docs);
            }
            meta.save(self.directory.as_ref()).await?;
        }

        // Mark old segments for deletion via tracker (deferred if refs exist)
        let ready_to_delete = self.tracker.mark_for_deletion(&old_to_delete);
        for segment_id in ready_to_delete {
            let _ = crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
        }
        Ok(())
    }

    /// Clean up orphan segment files that are not registered
    ///
    /// This can happen if the process halts after segment files are written
    /// but before they are registered in metadata.json. Call this on startup
    /// to reclaim disk space from incomplete operations.
    ///
    /// Returns the number of orphan segments deleted.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        let registered_set: HashSet<String> = {
            let meta = self.metadata.read().await;
            meta.segment_metas.keys().cloned().collect()
        };

        // Find all segment files in directory
        let mut orphan_ids: HashSet<String> = HashSet::new();

        // List directory and find segment files
        if let Ok(entries) = self.directory.list_files(std::path::Path::new("")).await {
            for entry in entries {
                let filename = entry.to_string_lossy();
                // Match pattern: seg_{32 hex chars}.{ext}
                if filename.starts_with("seg_") && filename.len() > 37 {
                    // Extract the hex ID (32 chars after "seg_")
                    let hex_part = &filename[4..36];
                    if !registered_set.contains(hex_part) {
                        orphan_ids.insert(hex_part.to_string());
                    }
                }
            }
        }

        // Delete orphan segments
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
