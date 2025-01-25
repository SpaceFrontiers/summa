//! Segment lifecycle tracker with reference counting
//!
//! This module provides safe segment deletion by tracking references:
//! - Readers acquire segment snapshots (incrementing ref counts)
//! - When snapshot is dropped, ref counts are decremented
//! - Segments marked for deletion are only deleted when ref count reaches 0
//!
//! This prevents "file not found" errors when mergers delete segments
//! that are still being used by active readers.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::directories::DirectoryWriter;
use crate::segment::SegmentId;

/// Tracks segment references and pending deletions
pub struct SegmentTracker {
    /// Segment ID -> reference count
    ref_counts: RwLock<HashMap<String, usize>>,
    /// Segments marked for deletion (will be deleted when ref count reaches 0)
    pending_deletions: RwLock<HashMap<String, PendingDeletion>>,
}

/// Info about a segment pending deletion
struct PendingDeletion {
    segment_id: SegmentId,
}

impl SegmentTracker {
    /// Create a new segment tracker
    pub fn new() -> Self {
        Self {
            ref_counts: RwLock::new(HashMap::new()),
            pending_deletions: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new segment (called when segment is committed)
    pub fn register(&self, segment_id: &str) {
        let mut refs = self.ref_counts.write();
        refs.entry(segment_id.to_string()).or_insert(0);
    }

    /// Acquire references to a set of segments (called when taking a snapshot)
    /// Returns the segment IDs that were successfully acquired
    pub fn acquire(&self, segment_ids: &[String]) -> Vec<String> {
        let mut refs = self.ref_counts.write();
        let pending = self.pending_deletions.read();

        let mut acquired = Vec::with_capacity(segment_ids.len());
        for id in segment_ids {
            // Don't acquire segments that are pending deletion
            if pending.contains_key(id) {
                continue;
            }
            *refs.entry(id.clone()).or_insert(0) += 1;
            acquired.push(id.clone());
        }
        acquired
    }

    /// Release references to a set of segments (called when snapshot is dropped)
    /// Returns segment IDs that are now ready for deletion (ref count hit 0 and marked for deletion)
    pub fn release(&self, segment_ids: &[String]) -> Vec<SegmentId> {
        let mut refs = self.ref_counts.write();
        let mut pending = self.pending_deletions.write();

        let mut ready_for_deletion = Vec::new();

        for id in segment_ids {
            if let Some(count) = refs.get_mut(id) {
                *count = count.saturating_sub(1);

                // If ref count is 0 and segment is pending deletion, it can be deleted
                if *count == 0
                    && let Some(deletion) = pending.remove(id)
                {
                    refs.remove(id);
                    ready_for_deletion.push(deletion.segment_id);
                }
            }
        }

        ready_for_deletion
    }

    /// Mark segments for deletion (called after merge completes)
    /// Segments with ref count 0 are returned immediately for deletion
    /// Segments with refs > 0 are queued for deletion when refs are released
    pub fn mark_for_deletion(&self, segment_ids: &[String]) -> Vec<SegmentId> {
        let mut refs = self.ref_counts.write();
        let mut pending = self.pending_deletions.write();

        let mut ready_for_deletion = Vec::new();

        for id_str in segment_ids {
            let Some(segment_id) = SegmentId::from_hex(id_str) else {
                continue;
            };

            let ref_count = refs.get(id_str).copied().unwrap_or(0);

            if ref_count == 0 {
                // No refs, can delete immediately
                refs.remove(id_str);
                ready_for_deletion.push(segment_id);
            } else {
                // Has refs, queue for deletion when refs are released
                pending.insert(id_str.clone(), PendingDeletion { segment_id });
            }
        }

        ready_for_deletion
    }

    /// Get current segment IDs (excluding those pending deletion)
    pub fn get_active_segments(&self) -> Vec<String> {
        let refs = self.ref_counts.read();
        let pending = self.pending_deletions.read();

        refs.keys()
            .filter(|id| !pending.contains_key(*id))
            .cloned()
            .collect()
    }

    /// Get the number of active references for a segment
    pub fn ref_count(&self, segment_id: &str) -> usize {
        self.ref_counts.read().get(segment_id).copied().unwrap_or(0)
    }

    /// Check if a segment is pending deletion
    pub fn is_pending_deletion(&self, segment_id: &str) -> bool {
        self.pending_deletions.read().contains_key(segment_id)
    }
}

impl Default for SegmentTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard that holds references to a snapshot of segments
/// When dropped, releases all segment references
pub struct SegmentSnapshot<D: DirectoryWriter + 'static> {
    tracker: Arc<SegmentTracker>,
    segment_ids: Vec<String>,
    directory: Arc<D>,
}

impl<D: DirectoryWriter + 'static> SegmentSnapshot<D> {
    /// Create a new snapshot holding references to the given segments
    pub fn new(tracker: Arc<SegmentTracker>, directory: Arc<D>, segment_ids: Vec<String>) -> Self {
        // Acquire is already done by caller
        Self {
            tracker,
            segment_ids,
            directory,
        }
    }

    /// Get the segment IDs in this snapshot
    pub fn segment_ids(&self) -> &[String] {
        &self.segment_ids
    }

    /// Check if this snapshot is empty
    pub fn is_empty(&self) -> bool {
        self.segment_ids.is_empty()
    }

    /// Get the number of segments in this snapshot
    pub fn len(&self) -> usize {
        self.segment_ids.len()
    }
}

impl<D: DirectoryWriter + 'static> Drop for SegmentSnapshot<D> {
    fn drop(&mut self) {
        let to_delete = self.tracker.release(&self.segment_ids);

        // Delete segments that are now ready
        if !to_delete.is_empty() {
            let dir = Arc::clone(&self.directory);
            tokio::spawn(async move {
                for segment_id in to_delete {
                    let _ = crate::segment::delete_segment(dir.as_ref(), segment_id).await;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Valid 32-char hex segment IDs for testing
    const SEG1: &str = "00000000000000000000000000000001";
    const SEG2: &str = "00000000000000000000000000000002";
    const SEG3: &str = "00000000000000000000000000000003";

    #[test]
    fn test_tracker_register_and_acquire() {
        let tracker = SegmentTracker::new();

        tracker.register(SEG1);
        tracker.register(SEG2);

        let acquired = tracker.acquire(&[SEG1.to_string(), SEG2.to_string()]);
        assert_eq!(acquired.len(), 2);

        assert_eq!(tracker.ref_count(SEG1), 1);
        assert_eq!(tracker.ref_count(SEG2), 1);
    }

    #[test]
    fn test_tracker_release() {
        let tracker = SegmentTracker::new();

        tracker.register(SEG1);
        tracker.acquire(&[SEG1.to_string()]);
        tracker.acquire(&[SEG1.to_string()]);

        assert_eq!(tracker.ref_count(SEG1), 2);

        tracker.release(&[SEG1.to_string()]);
        assert_eq!(tracker.ref_count(SEG1), 1);

        tracker.release(&[SEG1.to_string()]);
        assert_eq!(tracker.ref_count(SEG1), 0);
    }

    #[test]
    fn test_tracker_mark_for_deletion_no_refs() {
        let tracker = SegmentTracker::new();

        tracker.register(SEG1);

        let ready = tracker.mark_for_deletion(&[SEG1.to_string()]);
        assert_eq!(ready.len(), 1);
        assert!(!tracker.is_pending_deletion(SEG1));
    }

    #[test]
    fn test_tracker_mark_for_deletion_with_refs() {
        let tracker = SegmentTracker::new();

        tracker.register(SEG1);
        tracker.acquire(&[SEG1.to_string()]);

        let ready = tracker.mark_for_deletion(&[SEG1.to_string()]);
        assert!(ready.is_empty());
        assert!(tracker.is_pending_deletion(SEG1));

        // Release should now return segment for deletion
        let deleted = tracker.release(&[SEG1.to_string()]);
        assert_eq!(deleted.len(), 1);
        assert!(!tracker.is_pending_deletion(SEG1));
    }

    #[test]
    fn test_tracker_active_segments() {
        let tracker = SegmentTracker::new();

        tracker.register(SEG1);
        tracker.register(SEG2);
        tracker.register(SEG3);

        tracker.acquire(&[SEG2.to_string()]);
        tracker.mark_for_deletion(&[SEG2.to_string()]);

        let active = tracker.get_active_segments();
        assert!(active.contains(&SEG1.to_string()));
        assert!(!active.contains(&SEG2.to_string())); // pending deletion
        assert!(active.contains(&SEG3.to_string()));
    }
}
