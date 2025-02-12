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
    /// Returns (store_num_docs) so the caller can verify against metadata.
    pub(super) async fn merge_store(
        &self,
        segments: &[SegmentReader],
        store_writer: &mut OffsetWriter,
    ) -> Result<u32> {
        // Pre-check: verify each source segment's store num_docs matches its metadata.
        // A mismatch means the source segment is corrupt (e.g. store blocks were lost
        // during initial build). Proceeding would desynchronize postings and store
        // doc_id spaces in the merged output, causing progressive document loss.
        for segment in segments {
            let meta_docs = segment.num_docs();
            let store_docs = segment.store().num_docs();
            if meta_docs != store_docs {
                log::error!(
                    "[merge_store] SOURCE MISMATCH segment {:016x}: meta.num_docs={}, store.num_docs={}",
                    segment.meta().id,
                    meta_docs,
                    store_docs
                );
                return Err(crate::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Source segment {:016x} has store/meta mismatch: store={}, meta={}",
                        segment.meta().id,
                        store_docs,
                        meta_docs
                    ),
                )));
            }
        }

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
        let store_num_docs = store_merger.finish()?;
        Ok(store_num_docs)
    }
}
