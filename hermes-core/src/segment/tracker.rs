//! Segment lifecycle tracker with reference counting
//!
//! Provides safe segment deletion by tracking references:
//! - Readers acquire segment snapshots (incrementing ref counts)
//! - When snapshot is dropped, ref counts are decremented
//! - Segments marked for deletion are only deleted when ref count reaches 0
//!
//! Uses a single `parking_lot::Mutex` for all state (sub-μs holds, needed for sync Drop).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::segment::SegmentId;

/// Internal state protected by single Mutex
struct TrackerInner {
    ref_counts: HashMap<String, usize>,
    pending_deletions: HashMap<String, SegmentId>,
}

/// Tracks segment references and pending deletions
pub struct SegmentTracker {
    inner: Mutex<TrackerInner>,
}

impl SegmentTracker {
    /// Create a new segment tracker
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(TrackerInner {
                ref_counts: HashMap::new(),
                pending_deletions: HashMap::new(),
            }),
        }
    }

    /// Register a new segment (called when segment is committed)
    pub fn register(&self, segment_id: &str) {
        let mut inner = self.inner.lock();
        inner.ref_counts.entry(segment_id.to_string()).or_insert(0);
    }

    /// Acquire references to a set of segments (called when taking a snapshot)
    /// Returns the segment IDs that were successfully acquired
    pub fn acquire(&self, segment_ids: &[String]) -> Vec<String> {
        let mut inner = self.inner.lock();
        let mut acquired = Vec::with_capacity(segment_ids.len());
        for id in segment_ids {
            if inner.pending_deletions.contains_key(id) {
                continue;
            }
            *inner.ref_counts.entry(id.clone()).or_insert(0) += 1;
            acquired.push(id.clone());
        }
        acquired
    }

    /// Release references to a set of segments (called when snapshot is dropped)
    /// Returns segment IDs that are now ready for deletion
    pub fn release(&self, segment_ids: &[String]) -> Vec<SegmentId> {
        let mut inner = self.inner.lock();
        let mut ready_for_deletion = Vec::new();

        for id in segment_ids {
            if let Some(count) = inner.ref_counts.get_mut(id) {
                *count = count.saturating_sub(1);

                if *count == 0
                    && let Some(segment_id) = inner.pending_deletions.remove(id)
                {
                    inner.ref_counts.remove(id);
                    ready_for_deletion.push(segment_id);
                }
            }
        }

        ready_for_deletion
    }

    /// Mark segments for deletion (called after merge completes)
    /// Segments with ref count 0 are returned immediately for deletion
    /// Segments with refs > 0 are queued for deletion when refs are released
    pub fn mark_for_deletion(&self, segment_ids: &[String]) -> Vec<SegmentId> {
        let mut inner = self.inner.lock();
        let mut ready_for_deletion = Vec::new();

        for id_str in segment_ids {
            let Some(segment_id) = SegmentId::from_hex(id_str) else {
                continue;
            };

            // Skip segments not tracked (already deleted or never registered)
            let Some(&ref_count) = inner.ref_counts.get(id_str) else {
                continue;
            };

            if ref_count == 0 {
                inner.ref_counts.remove(id_str);
                ready_for_deletion.push(segment_id);
            } else {
                inner.pending_deletions.insert(id_str.clone(), segment_id);
            }
        }

        ready_for_deletion
    }

    /// Check if a segment is pending deletion
    pub fn is_pending_deletion(&self, segment_id: &str) -> bool {
        self.inner.lock().pending_deletions.contains_key(segment_id)
    }

    /// Get the number of active references for a segment
    pub fn ref_count(&self, segment_id: &str) -> usize {
        self.inner
            .lock()
            .ref_counts
            .get(segment_id)
            .copied()
            .unwrap_or(0)
    }
}

impl Default for SegmentTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard that holds references to a snapshot of segments.
/// When dropped, releases all segment references and triggers deferred deletion.
///
/// Not generic over Directory — the delete callback abstracts away directory access.
pub struct SegmentSnapshot {
    tracker: Arc<SegmentTracker>,
    segment_ids: Vec<String>,
    /// Callback to delete segment files when they become ready for deletion.
    delete_fn: Option<Arc<dyn Fn(Vec<SegmentId>) + Send + Sync>>,
}

impl SegmentSnapshot {
    /// Create a new snapshot holding references to the given segments
    pub fn new(tracker: Arc<SegmentTracker>, segment_ids: Vec<String>) -> Self {
        Self {
            tracker,
            segment_ids,
            delete_fn: None,
        }
    }

    /// Create a snapshot with a deletion callback for deferred segment cleanup
    pub fn with_delete_fn(
        tracker: Arc<SegmentTracker>,
        segment_ids: Vec<String>,
        delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync>,
    ) -> Self {
        Self {
            tracker,
            segment_ids,
            delete_fn: Some(delete_fn),
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

impl Drop for SegmentSnapshot {
    fn drop(&mut self) {
        let to_delete = self.tracker.release(&self.segment_ids);
        if !to_delete.is_empty() {
            if let Some(delete_fn) = &self.delete_fn {
                log::info!(
                    "[segment_snapshot] dropping snapshot, deleting {} deferred segments",
                    to_delete.len()
                );
                delete_fn(to_delete);
            } else {
                log::warn!(
                    "[segment_snapshot] {} segments ready for deletion but no delete_fn provided",
                    to_delete.len()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEG1: &str = "00000000000000000000000000000001";
    const SEG2: &str = "00000000000000000000000000000002";

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

        let deleted = tracker.release(&[SEG1.to_string()]);
        assert_eq!(deleted.len(), 1);
        assert!(!tracker.is_pending_deletion(SEG1));
    }

    #[test]
    fn test_tracker_double_mark_for_deletion() {
        let tracker = SegmentTracker::new();
        tracker.register(SEG1);

        let ready1 = tracker.mark_for_deletion(&[SEG1.to_string()]);
        assert_eq!(ready1.len(), 1);

        // Second mark: segment already removed from ref_counts, should return empty
        let ready2 = tracker.mark_for_deletion(&[SEG1.to_string()]);
        assert!(ready2.is_empty());
    }

    #[test]
    fn test_tracker_acquire_unregistered() {
        let tracker = SegmentTracker::new();

        // Acquire a segment that was never registered — or_insert(0) makes it ref_count=1
        let acquired = tracker.acquire(&[SEG1.to_string()]);
        assert_eq!(acquired.len(), 1);
        assert_eq!(tracker.ref_count(SEG1), 1);
    }

    #[test]
    fn test_tracker_release_without_acquire() {
        let tracker = SegmentTracker::new();
        tracker.register(SEG1);

        // Release without acquire — should not panic, ref stays at 0 (saturating_sub)
        let deleted = tracker.release(&[SEG1.to_string()]);
        assert!(deleted.is_empty());
        assert_eq!(tracker.ref_count(SEG1), 0);
    }

    #[test]
    fn test_snapshot_drop_triggers_deferred_delete() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let tracker = Arc::new(SegmentTracker::new());
        tracker.register(SEG1);
        tracker.register(SEG2);

        let delete_count = Arc::new(AtomicUsize::new(0));
        let dc = Arc::clone(&delete_count);
        let delete_fn: Arc<dyn Fn(Vec<SegmentId>) + Send + Sync> = Arc::new(move |ids| {
            dc.fetch_add(ids.len(), Ordering::SeqCst);
        });

        // Take a snapshot holding refs to both segments
        let acquired = tracker.acquire(&[SEG1.to_string(), SEG2.to_string()]);
        let snapshot =
            SegmentSnapshot::with_delete_fn(Arc::clone(&tracker), acquired, Arc::clone(&delete_fn));

        // Mark both for deletion — should be deferred (refs > 0)
        let ready = tracker.mark_for_deletion(&[SEG1.to_string(), SEG2.to_string()]);
        assert!(ready.is_empty());
        assert!(tracker.is_pending_deletion(SEG1));
        assert!(tracker.is_pending_deletion(SEG2));

        // Drop snapshot → refs go to 0 → delete_fn called
        drop(snapshot);
        assert_eq!(delete_count.load(Ordering::SeqCst), 2);
        assert!(!tracker.is_pending_deletion(SEG1));
        assert!(!tracker.is_pending_deletion(SEG2));
    }
}
