//! Segment merger for combining multiple segments

mod dense;
#[cfg(feature = "diagnostics")]
mod diagnostics;
mod fast_fields;
mod postings;
mod sparse;
mod store;

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::reader::SegmentReader;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use super::{OffsetWriter, format_bytes};
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::Schema;

/// Compute per-segment doc ID offsets (each segment's docs start after the previous).
///
/// Returns an error if the total document count across segments exceeds `u32::MAX`.
fn doc_offsets(segments: &[SegmentReader]) -> Result<Vec<u32>> {
    let mut offsets = Vec::with_capacity(segments.len());
    let mut acc = 0u32;
    for seg in segments {
        offsets.push(acc);
        acc = acc.checked_add(seg.num_docs()).ok_or_else(|| {
            crate::Error::Internal(format!(
                "Total document count across segments exceeds u32::MAX ({})",
                u32::MAX
            ))
        })?;
    }
    Ok(offsets)
}

/// Statistics for merge operations
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of terms processed
    pub terms_processed: usize,
    /// Term dictionary output size
    pub term_dict_bytes: usize,
    /// Postings output size
    pub postings_bytes: usize,
    /// Store output size
    pub store_bytes: usize,
    /// Vector index output size
    pub vectors_bytes: usize,
    /// Sparse vector index output size
    pub sparse_bytes: usize,
    /// Whether merge-time BP reorder ran to full depth on every BMP field
    /// (false = a pass hit its wall-clock budget; the segment is valid and
    /// better-ordered, and the background optimizer deepens it later).
    /// True when no BP ran (block-copy merges have nothing to deepen... they
    /// are simply not reordered and tracked by the `reordered` flag instead).
    pub bp_converged: bool,
    /// Fast-field output size
    pub fast_bytes: usize,
}

impl std::fmt::Display for MergeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "terms={}, term_dict={}, postings={}, store={}, vectors={}, sparse={}, fast={}",
            self.terms_processed,
            format_bytes(self.term_dict_bytes),
            format_bytes(self.postings_bytes),
            format_bytes(self.store_bytes),
            format_bytes(self.vectors_bytes),
            format_bytes(self.sparse_bytes),
            format_bytes(self.fast_bytes),
        )
    }
}

// TrainedVectorStructures is defined in super::types (available on all platforms)
pub use super::types::TrainedVectorStructures;

/// Run a CPU/IO-heavy synchronous section, telling tokio to migrate this
/// worker's task queue first (multi-thread runtimes only — `block_in_place`
/// panics on current_thread, where we just run inline).
pub(crate) fn block_in_place_if_multithread<R>(f: impl FnOnce() -> R) -> R {
    if tokio::runtime::Handle::try_current()
        .map(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
        .unwrap_or(false)
    {
        tokio::task::block_in_place(f)
    } else {
        f()
    }
}

/// Segment merger - merges multiple segments into one
pub struct SegmentMerger {
    schema: Arc<Schema>,
    /// Run BP reordering on BMP sparse fields while writing the merged blob
    /// (instead of byte-level block stacking). The output segment is then
    /// already ordered, so the standalone reorder pass is unnecessary.
    reorder_bmp: bool,
    /// Bounded rayon pool for merge-time BP. `None` = global pool (tests);
    /// the SegmentManager always passes its background pool so BP cannot
    /// starve query scoring.
    background_pool: Option<Arc<rayon::ThreadPool>>,
    /// Granularity for merge-time BP. `Auto` by default; the SegmentManager
    /// forces `Records` when any merge source is an unconverged partial
    /// reorder.
    granularity: crate::segment::reorder::BpGranularity,
    /// Budget for merge-time BP. Default unbudgeted; the SegmentManager
    /// passes the index's `merge_bp_time_budget` so huge merges stop holding
    /// a merge slot for the full BP depth — a truncated pass is marked
    /// `bp_converged = false` and the background optimizer deepens it.
    bp_budget: crate::segment::BpBudget,
    /// Memory budget for the BP forward index during merge-time reorder.
    bp_memory_budget: usize,
}

impl SegmentMerger {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            reorder_bmp: false,
            background_pool: None,
            granularity: crate::segment::reorder::BpGranularity::Auto,
            bp_budget: crate::segment::BpBudget::full(),
            bp_memory_budget: crate::segment::reorder::DEFAULT_MEMORY_BUDGET,
        }
    }

    /// Enable BP reordering of BMP fields during the merge (see `reorder_bmp`).
    pub fn with_bmp_reorder(mut self, reorder: bool) -> Self {
        self.reorder_bmp = reorder;
        self
    }

    /// Run merge-time BP on this bounded pool instead of the global one.
    pub fn with_background_pool(mut self, pool: Option<Arc<rayon::ThreadPool>>) -> Self {
        self.background_pool = pool;
        self
    }

    /// Set merge-time BP granularity (see `granularity`).
    pub fn with_granularity(mut self, granularity: crate::segment::reorder::BpGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Bound merge-time BP wall clock (see `bp_budget`).
    pub fn with_bp_budget(mut self, budget: crate::segment::BpBudget) -> Self {
        self.bp_budget = budget;
        self
    }

    /// Memory budget for the BP forward index (see `bp_memory_budget`).
    pub fn with_bp_memory_budget(mut self, bytes: usize) -> Self {
        self.bp_memory_budget = bytes;
        self
    }

    /// Merge segments into one, streaming postings/positions/store directly to files.
    ///
    /// If `trained` is provided, dense vectors use O(1) cluster merge when possible
    /// (homogeneous IVF/ScaNN), otherwise rebuilds ANN from trained structures.
    /// Without trained structures, only flat vectors are merged.
    ///
    /// Uses streaming writers so postings, positions, and store data flow directly
    /// to files instead of buffering everything in memory. Only the term dictionary
    /// (compact key+TermInfo entries) is buffered.
    pub async fn merge<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<(SegmentMeta, MergeStats)> {
        let mut stats = MergeStats::default();
        let files = SegmentFiles::new(new_segment_id.0);

        // === Two-stage merge to bound page cache pressure ===
        //
        // Stage 1: postings + store + fast_fields (concurrent)
        //   Touches .term_dict, .postings, .positions, .store, .fast files.
        //
        // Stage 2: sparse + dense vectors (concurrent)
        //   Touches .sparse, .vectors files.
        //
        // Running all phases concurrently caused OOM on large merges because
        // mmap'd source files from all 16+ segments compete for page cache
        // simultaneously (200+ GB of mmap'd data for BMP grids alone).
        // Two stages halve the concurrent working set.
        let merge_start = std::time::Instant::now();

        // ── Stage 1: text + store + fast fields ─────────────────────────
        let postings_fut = async {
            let mut postings_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.postings).await?);
            let mut positions_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.positions).await?);
            let mut term_dict_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.term_dict).await?);

            let terms_processed = self
                .merge_postings(
                    segments,
                    &mut term_dict_writer,
                    &mut postings_writer,
                    &mut positions_writer,
                )
                .await?;

            let postings_bytes = postings_writer.offset() as usize;
            let term_dict_bytes = term_dict_writer.offset() as usize;
            let positions_bytes = positions_writer.offset();

            postings_writer.finish()?;
            term_dict_writer.finish()?;
            if positions_bytes > 0 {
                positions_writer.finish()?;
            } else {
                drop(positions_writer);
                let _ = dir.delete(&files.positions).await;
            }
            log::info!(
                "[merge] postings done: {} terms, term_dict={}, postings={}, positions={}",
                terms_processed,
                format_bytes(term_dict_bytes),
                format_bytes(postings_bytes),
                format_bytes(positions_bytes as usize),
            );
            Ok::<(usize, usize, usize), crate::Error>((
                terms_processed,
                term_dict_bytes,
                postings_bytes,
            ))
        };

        let store_fut = async {
            let mut store_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.store).await?);
            let store_num_docs = self.merge_store(segments, &mut store_writer).await?;
            let bytes = store_writer.offset() as usize;
            store_writer.finish()?;
            Ok::<(usize, u32), crate::Error>((bytes, store_num_docs))
        };

        let fast_fut = async { self.merge_fast_fields(dir, segments, &files).await };

        let (postings_result, store_result, fast_bytes) =
            tokio::try_join!(postings_fut, store_fut, fast_fut)?;

        log::info!(
            "[merge] stage 1 done in {:.1}s (postings + store + fast)",
            merge_start.elapsed().as_secs_f64()
        );

        // ── Stage 2: sparse + dense vectors ─────────────────────────────
        // Page cache from stage 1 files can now be evicted by the kernel
        // as stage 2 accesses different mmap regions (.sparse, .vectors).
        let sparse_fut = async { self.merge_sparse_vectors(dir, segments, &files).await };

        let dense_fut = async {
            self.merge_dense_vectors(dir, segments, &files, trained)
                .await
        };

        let ((sparse_bytes, bp_converged), vectors_bytes) =
            tokio::try_join!(sparse_fut, dense_fut)?;
        let (store_bytes, store_num_docs) = store_result;
        stats.terms_processed = postings_result.0;
        stats.term_dict_bytes = postings_result.1;
        stats.postings_bytes = postings_result.2;
        stats.store_bytes = store_bytes;
        stats.vectors_bytes = vectors_bytes;
        stats.sparse_bytes = sparse_bytes;
        stats.bp_converged = bp_converged;
        stats.fast_bytes = fast_bytes;
        log::info!(
            "[merge] all phases done in {:.1}s: {}",
            merge_start.elapsed().as_secs_f64(),
            stats
        );

        // === Mandatory: merge field stats + write meta ===
        let mut merged_field_stats: FxHashMap<u32, FieldStats> = FxHashMap::default();
        for segment in segments {
            for (&field_id, field_stats) in &segment.meta().field_stats {
                let entry = merged_field_stats.entry(field_id).or_default();
                entry.total_tokens += field_stats.total_tokens;
                entry.doc_count += field_stats.doc_count;
            }
        }

        let total_docs: u32 = segments
            .iter()
            .try_fold(0u32, |acc, s| acc.checked_add(s.num_docs()))
            .ok_or_else(|| {
                crate::Error::Internal(format!(
                    "Total document count exceeds u32::MAX ({})",
                    u32::MAX
                ))
            })?;

        // Verify store doc count matches metadata — a mismatch here means
        // some store blocks were lost (e.g., compression thread panic) or
        // source segment metadata disagrees with its store.
        if store_num_docs != total_docs {
            log::error!(
                "[merge] STORE/META MISMATCH: store has {} docs but metadata expects {}. \
                 Per-segment: {:?}",
                store_num_docs,
                total_docs,
                segments
                    .iter()
                    .map(|s| (
                        format!("{:016x}", s.meta().id),
                        s.num_docs(),
                        s.store().num_docs()
                    ))
                    .collect::<Vec<_>>()
            );
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Store/meta doc count mismatch: store={}, meta={}",
                    store_num_docs, total_docs
                ),
            )));
        }

        let meta = SegmentMeta {
            id: new_segment_id.0,
            num_docs: total_docs,
            field_stats: merged_field_stats,
        };

        dir.write(&files.meta, &meta.serialize()?).await?;

        let label = if trained.is_some() {
            "ANN merge"
        } else {
            "Merge"
        };
        log::info!("{} complete: {} docs, {}", label, total_docs, stats);

        Ok((meta, stats))
    }
}

/// Delete segment files from directory (all deletions run concurrently).
pub async fn delete_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    segment_id: SegmentId,
) -> Result<()> {
    let files = SegmentFiles::new(segment_id.0);
    let _ = tokio::join!(
        dir.delete(&files.term_dict),
        dir.delete(&files.postings),
        dir.delete(&files.store),
        dir.delete(&files.meta),
        dir.delete(&files.vectors),
        dir.delete(&files.sparse),
        dir.delete(&files.positions),
        dir.delete(&files.fast),
    );
    Ok(())
}
