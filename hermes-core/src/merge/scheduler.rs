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
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Mutex as AsyncMutex, Notify};

use crate::directories::DirectoryWriter;
use crate::error::{Error, Result};
use crate::index::IndexMetadata;
use crate::segment::{SegmentId, SegmentMerger, SegmentReader};

use super::{MergePolicy, SegmentInfo};

/// Segment manager - coordinates segment registration and background merging
///
/// This is the SOLE owner of `metadata.json` ensuring linearized access.
/// All segment list modifications and metadata updates go through here.
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Directory for segment operations
    directory: Arc<D>,
    /// Schema for segment operations
    schema: Arc<crate::dsl::Schema>,
    /// Unified metadata (segments + vector index state) - SOLE owner
    metadata: Arc<AsyncMutex<IndexMetadata>>,
    /// The merge policy to use
    merge_policy: Box<dyn MergePolicy>,
    /// Count of in-flight background merges
    pending_merges: Arc<AtomicUsize>,
    /// Segments currently being merged (to avoid double-merging)
    merging_segments: Arc<AsyncMutex<HashSet<String>>>,
    /// Term cache blocks for segment readers during merge
    term_cache_blocks: usize,
    /// Notifier for merge completion (avoids busy-waiting)
    merge_complete: Arc<Notify>,
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
        Self {
            directory,
            schema,
            metadata: Arc::new(AsyncMutex::new(metadata)),
            merge_policy,
            pending_merges: Arc::new(AtomicUsize::new(0)),
            merging_segments: Arc::new(AsyncMutex::new(HashSet::new())),
            term_cache_blocks,
            merge_complete: Arc::new(Notify::new()),
        }
    }

    /// Get the current segment IDs (snapshot)
    pub async fn get_segment_ids(&self) -> Vec<String> {
        self.metadata.lock().await.segments.clone()
    }

    /// Register a new segment, persist metadata, and trigger merge check
    ///
    /// This is the main entry point for adding segments after builds complete.
    /// It atomically:
    /// 1. Adds the segment ID to the list
    /// 2. Persists metadata to disk
    /// 3. Triggers merge check (spawns background merges if needed)
    pub async fn register_segment(&self, segment_id: String) -> Result<()> {
        {
            let mut meta = self.metadata.lock().await;
            if !meta.segments.contains(&segment_id) {
                meta.segments.push(segment_id);
            }
            meta.save(self.directory.as_ref()).await?;
        }

        // Check if we should trigger a merge (non-blocking)
        self.maybe_merge().await;
        Ok(())
    }

    /// Get the number of pending background merges
    pub fn pending_merge_count(&self) -> usize {
        self.pending_merges.load(Ordering::SeqCst)
    }

    /// Check merge policy and spawn background merges if needed
    pub async fn maybe_merge(&self) {
        // Get current segment info (excluding segments being merged)
        let meta = self.metadata.lock().await;
        let merging = self.merging_segments.lock().await;

        // Filter out segments currently being merged
        let available_segments: Vec<String> = meta
            .segments
            .iter()
            .filter(|id| !merging.contains(*id))
            .cloned()
            .collect();

        drop(merging);
        drop(meta);

        // Build segment info - we estimate doc count based on segment age (newer = smaller)
        let segments: Vec<SegmentInfo> = available_segments
            .iter()
            .enumerate()
            .map(|(i, id)| SegmentInfo {
                id: id.clone(),
                num_docs: ((i + 1) * 1000) as u32,
                size_bytes: None,
            })
            .collect();

        // Ask merge policy for candidates
        let candidates = self.merge_policy.find_merges(&segments);

        for candidate in candidates {
            if candidate.segment_ids.len() >= 2 {
                self.spawn_merge(candidate.segment_ids).await;
            }
        }
    }

    /// Spawn a background merge task
    async fn spawn_merge(&self, segment_ids_to_merge: Vec<String>) {
        // Mark segments as being merged
        {
            let mut merging = self.merging_segments.lock().await;
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
        let term_cache_blocks = self.term_cache_blocks;

        pending_merges.fetch_add(1, Ordering::SeqCst);

        tokio::spawn(async move {
            let result = Self::do_merge(
                directory.as_ref(),
                &schema,
                &segment_ids_to_merge,
                term_cache_blocks,
            )
            .await;

            match result {
                Ok(new_segment_id) => {
                    // Atomically update metadata: remove merged segments, add new one, persist
                    let mut meta = metadata.lock().await;
                    meta.segments
                        .retain(|id| !segment_ids_to_merge.contains(id));
                    meta.segments.push(new_segment_id);
                    if let Err(e) = meta.save(directory.as_ref()).await {
                        eprintln!("[merge] Failed to save metadata after merge: {:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Background merge failed for segments {:?}: {:?}",
                        segment_ids_to_merge, e
                    );
                }
            }

            // Remove from merging set
            let mut merging = merging_segments.lock().await;
            for id in &segment_ids_to_merge {
                merging.remove(id);
            }

            // Decrement pending merges counter and notify waiters
            pending_merges.fetch_sub(1, Ordering::SeqCst);
            merge_complete.notify_waiters();
        });
    }

    /// Perform the actual merge operation (runs in background task)
    async fn do_merge(
        directory: &D,
        schema: &crate::dsl::Schema,
        segment_ids_to_merge: &[String],
        term_cache_blocks: usize,
    ) -> Result<String> {
        // Load segment readers
        let mut readers = Vec::new();
        let mut doc_offset = 0u32;

        for id_str in segment_ids_to_merge {
            let segment_id = SegmentId::from_hex(id_str)
                .ok_or_else(|| Error::Corruption(format!("Invalid segment ID: {}", id_str)))?;
            let reader = match SegmentReader::open(
                directory,
                segment_id,
                Arc::new(schema.clone()),
                doc_offset,
                term_cache_blocks,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[merge] Failed to open segment {}: {:?}", id_str, e);
                    return Err(e);
                }
            };
            doc_offset += reader.meta().num_docs;
            readers.push(reader);
        }

        // Merge into new segment
        let merger = SegmentMerger::new(Arc::new(schema.clone()));
        let new_segment_id = SegmentId::new();
        if let Err(e) = merger.merge(directory, &readers, new_segment_id).await {
            eprintln!(
                "[merge] Merge failed for segments {:?} -> {}: {:?}",
                segment_ids_to_merge,
                new_segment_id.to_hex(),
                e
            );
            return Err(e);
        }

        // Delete old segments
        for id_str in segment_ids_to_merge {
            if let Some(segment_id) = SegmentId::from_hex(id_str) {
                let _ = crate::segment::delete_segment(directory, segment_id).await;
            }
        }

        Ok(new_segment_id.to_hex())
    }

    /// Wait for all pending merges to complete
    pub async fn wait_for_merges(&self) {
        while self.pending_merges.load(Ordering::SeqCst) > 0 {
            self.merge_complete.notified().await;
        }
    }

    /// Get a clone of the metadata Arc for read access
    pub fn metadata(&self) -> Arc<AsyncMutex<IndexMetadata>> {
        Arc::clone(&self.metadata)
    }

    /// Update metadata with a closure and persist atomically
    pub async fn update_metadata<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut meta = self.metadata.lock().await;
        f(&mut meta);
        meta.save(self.directory.as_ref()).await
    }

    /// Replace segment list atomically (for force merge / rebuild)
    pub async fn replace_segments(
        &self,
        new_segments: Vec<String>,
        old_to_delete: Vec<String>,
    ) -> Result<()> {
        {
            let mut meta = self.metadata.lock().await;
            meta.segments = new_segments;
            meta.save(self.directory.as_ref()).await?;
        }

        // Delete old segment files
        for id_str in old_to_delete {
            if let Some(segment_id) = SegmentId::from_hex(&id_str) {
                let _ = crate::segment::delete_segment(self.directory.as_ref(), segment_id).await;
            }
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
            let meta = self.metadata.lock().await;
            meta.segments.iter().cloned().collect()
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
