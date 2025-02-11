//! Document store merge.
//!
//! Merges store blocks from multiple segments, handling both raw block
//! copying (when no dictionary compression) and recompression paths.

use super::OffsetWriter;
use super::SegmentMerger;
use crate::Result;
use crate::segment::reader::SegmentReader;
use crate::segment::store::StoreMerger;

impl SegmentMerger {
    /// Merge document stores from all source segments into a single output.
    ///
    /// Uses raw block copying when possible (no dictionary compression),
    /// falling back to recompression when segments use dictionaries.
    pub(super) async fn merge_store(
        &self,
        segments: &[SegmentReader],
        store_writer: &mut OffsetWriter,
    ) -> Result<()> {
        let mut store_merger = StoreMerger::new(store_writer);
        for segment in segments {
            if segment.store_has_dict() {
                store_merger
                    .append_store_recompressing(segment.store())
                    .await
                    .map_err(crate::Error::Io)?;
            } else {
                let raw_blocks = segment.store_raw_blocks();
                let data_slice = segment.store_data_slice();
                store_merger.append_store(data_slice, &raw_blocks).await?;
            }
        }
        store_merger.finish()?;
        Ok(())
    }
}
