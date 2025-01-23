//! Segment manager - coordinates segment registration and background merging
//!
//! This module is only compiled with the "native" feature.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Mutex as AsyncMutex, Notify};

use crate::directories::DirectoryWriter;
use crate::error::{Error, Result};
use crate::segment::{SegmentId, SegmentMerger, SegmentReader};

use super::{MergePolicy, SegmentInfo};

/// Segment manager - coordinates segment registration and background merging
///
/// This is the central point for:
/// - Tracking all segment IDs
/// - Registering new segments (from builds or merges)
/// - Triggering merge checks when segments are added
/// - Coordinating background merge tasks
pub struct SegmentManager<D: DirectoryWriter + 'static> {
    /// Directory for segment operations
    directory: Arc<D>,
    /// Schema for segment operations
    schema: Arc<crate::dsl::Schema>,
    /// List of committed segment IDs (hex strings)
    segment_ids: Arc<AsyncMutex<Vec<String>>>,
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
    /// Create a new segment manager
    pub fn new(
        directory: Arc<D>,
        schema: Arc<crate::dsl::Schema>,
        segment_ids: Vec<String>,
        merge_policy: Box<dyn MergePolicy>,
        term_cache_blocks: usize,
    ) -> Self {
        Self {
            directory,
            schema,
            segment_ids: Arc::new(AsyncMutex::new(segment_ids)),
            merge_policy,
            pending_merges: Arc::new(AtomicUsize::new(0)),
            merging_segments: Arc::new(AsyncMutex::new(HashSet::new())),
            term_cache_blocks,
            merge_complete: Arc::new(Notify::new()),
        }
    }

    /// Get a clone of the segment_ids Arc for sharing with background tasks
    pub fn segment_ids(&self) -> Arc<AsyncMutex<Vec<String>>> {
        Arc::clone(&self.segment_ids)
    }

    /// Get the current segment IDs
    pub async fn get_segment_ids(&self) -> Vec<String> {
        self.segment_ids.lock().await.clone()
    }

    /// Register a new segment and trigger merge check
    ///
    /// This is the main entry point for adding segments. It:
    /// 1. Adds the segment ID to the list
    /// 2. Checks the merge policy and spawns background merges if needed
    pub async fn register_segment(&self, segment_id: String) {
        {
            let mut ids = self.segment_ids.lock().await;
            ids.push(segment_id);
        }

        // Check if we should trigger a merge
        self.maybe_merge().await;
    }

    /// Get the number of pending background merges
    pub fn pending_merge_count(&self) -> usize {
        self.pending_merges.load(Ordering::SeqCst)
    }

    /// Check merge policy and spawn background merges if needed
    pub async fn maybe_merge(&self) {
        // Get current segment info (excluding segments being merged)
        let ids = self.segment_ids.lock().await;
        let merging = self.merging_segments.lock().await;

        // Filter out segments currently being merged
        let available_segments: Vec<String> = ids
            .iter()
            .filter(|id| !merging.contains(*id))
            .cloned()
            .collect();

        drop(merging);
        drop(ids);

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
        let segment_ids = Arc::clone(&self.segment_ids);
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
                    // Update segment list: remove merged segments, add new one
                    let mut ids = segment_ids.lock().await;
                    ids.retain(|id| !segment_ids_to_merge.contains(id));
                    ids.push(new_segment_id);
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
            let reader = SegmentReader::open(
                directory,
                segment_id,
                Arc::new(schema.clone()),
                doc_offset,
                term_cache_blocks,
            )
            .await?;
            doc_offset += reader.meta().num_docs;
            readers.push(reader);
        }

        // Merge into new segment
        let merger = SegmentMerger::new(Arc::new(schema.clone()));
        let new_segment_id = SegmentId::new();
        merger.merge(directory, &readers, new_segment_id).await?;

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

    /// Clean up orphan segment files that are not registered
    ///
    /// This can happen if the process halts after segment files are written
    /// but before they are registered in segments.json. Call this on startup
    /// to reclaim disk space from incomplete operations.
    ///
    /// Returns the number of orphan segments deleted.
    pub async fn cleanup_orphan_segments(&self) -> Result<usize> {
        let registered_set: HashSet<String> = {
            let registered_ids = self.segment_ids.lock().await;
            registered_ids.iter().cloned().collect()
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
