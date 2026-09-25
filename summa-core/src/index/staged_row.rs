//! Unpublished row visibility. Lock order is row handle, then segment bitmap.
use std::sync::Arc;

use crate::query::DocBitset;

#[derive(Default)]
pub(crate) struct StagedSegment {
    dead: parking_lot::Mutex<Vec<u32>>,
}

impl StagedSegment {
    fn cancel(&self, row: u32) {
        self.dead.lock().push(row);
    }

    #[cfg(feature = "native")]
    pub fn is_dirty(&self) -> bool {
        !self.dead.lock().is_empty()
    }

    /// Called only after ingestion has stopped for the commit generation.
    pub fn live_rows(&self, num_docs: u32) -> Option<DocBitset> {
        let dead = self.dead.lock();
        if dead.is_empty() {
            return None;
        }
        let mut live = DocBitset::all(num_docs);
        for row in dead.iter().copied() {
            live.clear(row);
        }
        Some(live)
    }
}

#[derive(Default)]
pub(super) struct StagedRow {
    state: parking_lot::Mutex<RowState>,
}

#[derive(Default)]
struct RowState {
    cancelled: bool,
    location: Option<(Arc<StagedSegment>, u32)>,
}

impl StagedRow {
    /// Claim the physical builder row before indexing. A queued cancellation
    /// skips indexing; a later cancellation clears this exact physical row.
    pub fn attach(&self, segment: &Arc<StagedSegment>, row: u32) -> bool {
        let mut state = self.state.lock();
        if state.cancelled {
            return false;
        }
        debug_assert!(state.location.is_none());
        state.location = Some((Arc::clone(segment), row));
        true
    }

    pub fn cancel(&self) {
        let mut state = self.state.lock();
        if !state.cancelled {
            state.cancelled = true;
            if let Some((segment, row)) = state.location.take() {
                segment.cancel(row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_before_or_after_row_attachment_hides_the_same_document() {
        let segment = Arc::new(StagedSegment::default());
        let queued = StagedRow::default();
        queued.cancel();
        assert!(!queued.attach(&segment, 0));
        assert!(segment.live_rows(0).is_none());
        let encoded = StagedRow::default();
        assert!(encoded.attach(&segment, 64));
        encoded.cancel();
        encoded.cancel();
        let live = segment.live_rows(65).unwrap();
        assert!(live.contains(63));
        assert!(!live.contains(64));
    }
}
