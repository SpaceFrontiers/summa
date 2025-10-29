use super::{SegmentReader, VectorIndex};
use crate::segment::PrefixCacheBudget;
use crate::segment::ann_disk::prefix_cache::Admission;
use std::sync::Arc;

impl SegmentReader {
    pub(crate) async fn prepare_prefix_cache(
        &self,
        budget: &Arc<PrefixCacheBudget>,
    ) -> crate::Result<()> {
        let mut fields: Vec<_> = self.vector_indexes.keys().copied().collect();
        fields.sort_unstable();
        for field in fields {
            let Some(VectorIndex::BinaryIvf(index)) = self.vector_indexes.get(&field) else {
                continue;
            };
            let flat = self.flat_vectors.get(&field).ok_or_else(|| {
                crate::Error::Corruption("binary prefix cache requires exact vector lookup".into())
            })?;
            match index.get().prepare_prefix_cache(flat, budget).await? {
                Admission::Loaded(bytes) => log::info!(
                    "[binary_prefix_cache] index={} field={} segment={:032x} admitted_bytes={bytes}",
                    self.schema.index_label(),
                    field,
                    self.meta.id
                ),
                Admission::Refused { bytes, reason } => log::warn!(
                    "[binary_prefix_cache] index={} field={} segment={:032x} mapped_fallback reason={reason} requested_bytes={bytes} budget_bytes={}",
                    self.schema.index_label(),
                    field,
                    self.meta.id,
                    budget.stats().budget_bytes
                ),
                Admission::Disabled | Admission::Existing => {}
            }
        }
        Ok(())
    }

    /// Requested prefix/reference/cache-directory heap bytes. Subset of dense_heap_bytes;
    /// shared by visibility clones and not an OS pinning or RSS measurement.
    pub fn resident_binary_prefix_bytes(&self) -> usize {
        self.vector_indexes
            .values()
            .filter_map(|index| match index {
                VectorIndex::BinaryIvf(index) => Some(index.get().resident_prefix_bytes()),
                _ => None,
            })
            .sum()
    }
}
