//! Segment merger for combining multiple segments

mod chunk_maps;
mod compact;
mod compact_vectors;
mod copy;
pub(crate) use copy::append_and_delete_temp;
pub(super) use copy::copy_local_range_or_bytes;
mod dense;
mod fast_fields;
mod postings;
pub(crate) use postings::PostingMergeStats;
mod sparse;
mod store;
mod terms;
pub(crate) use terms::MergedTerms;

pub(crate) use dense::AnnWriteMode;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rustc_hash::FxHashMap;

use super::OffsetWriter;
use super::reader::SegmentReader;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{FieldType, Schema};
use crate::index::{ReorderConcurrencyGate, ReorderPriority};
use crate::structures::SparseFormat;

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

/// Additive count stored in a `u32` field of the merged segment format.
///
/// Source segments are individually valid, so exceeding the limit is a
/// property of this merge plan rather than source corruption.
#[derive(Clone, Copy, Debug, Default)]
struct MergeCapacity(u64);

impl MergeCapacity {
    #[inline]
    fn add(&mut self, count: u64) -> Option<u64> {
        self.0 = self.0.saturating_add(count);
        (self.0 > u64::from(u32::MAX)).then_some(self.0)
    }
}

fn field_capacity_error(
    field_id: u32,
    field_name: &str,
    value_kind: &str,
    count: u64,
) -> crate::Error {
    crate::Error::Schema(format!(
        "merge would produce {count} {value_kind} for field {field_id} ('{field_name}'), \
         exceeding the segment format limit {}; lower max_segment_docs for this \
         multi-valued field",
        u32::MAX,
    ))
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
    /// Whether merge-time BP reorder ran to full depth on every text/BMP field
    /// (false = a pass hit its wall-clock budget; the segment is valid and
    /// better-ordered, and the background optimizer deepens it later).
    /// True when no BP ran (block-copy merges have nothing to deepen... they
    /// are simply not reordered and tracked by the `reordered` flag instead).
    pub bp_converged: bool,
    /// Fast-field output size
    pub fast_bytes: usize,
    /// Posting blocks written into a ratio/impact-bounded list with an
    /// unknown record: legacy external blocks copied next to bounded sources,
    /// and promoted inline blocks joining an impact list (envelopes exist only
    /// for multi-block lists). Their L1 group keeps an unknown (zero) bound;
    /// only a rebuild adds metadata to legacy blocks (`docs/posting-codecs.md`).
    pub posting_blocks_without_bounds: usize,
}

impl std::fmt::Display for MergeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "terms={}, term_dict={}, postings={}, store={}, dense_vectors={}, sparse_vectors={}, fast_fields={}, posting_blocks_without_bounds={}",
            self.terms_processed,
            crate::format_bytes(self.term_dict_bytes as u64),
            crate::format_bytes(self.postings_bytes as u64),
            crate::format_bytes(self.store_bytes as u64),
            crate::format_bytes(self.vectors_bytes as u64),
            crate::format_bytes(self.sparse_bytes as u64),
            crate::format_bytes(self.fast_bytes as u64),
            self.posting_blocks_without_bounds,
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
    /// Term-dictionary compression settings used for the merged segment.
    optimization: crate::structures::IndexOptimization,
    /// Codec used whenever a merge must decode and re-encode postings.
    /// External blocks still take the zero-copy concatenation path and retain
    /// their per-block codecs.
    posting_codec: crate::structures::PostingCodec,
    term_dict_block_size: crate::structures::SSTableBlockSize,
    /// Run BP on opted-in text and BMP fields while writing the merged
    /// generation. Other fields retain the encoded-copy path.
    reorder_fields: bool,
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
    /// Process-shutdown cancellation, kept separate from the public BP budget
    /// so low-level callers retain the existing budget API.
    cancellation: Option<Arc<AtomicBool>>,
    /// Memory budget for the BP forward index during merge-time reorder.
    bp_memory_budget: usize,
    /// Shared whole-pass concurrency limit. Tests and low-level callers may
    /// omit it; SegmentManager always supplies the application-wide gate.
    reorder_permits: Option<Arc<ReorderConcurrencyGate>>,
    /// Automatic merges are background work. An explicit force merge holds a
    /// foreground guard and bypasses the background pause for its BP fields.
    reorder_priority: ReorderPriority,
}

impl SegmentMerger {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            optimization: crate::structures::IndexOptimization::default(),
            posting_codec: crate::structures::PostingCodec::default(),
            term_dict_block_size: crate::structures::SSTableBlockSize::default(),
            reorder_fields: false,
            background_pool: None,
            granularity: crate::segment::reorder::BpGranularity::Auto,
            bp_budget: crate::segment::BpBudget::full(),
            cancellation: None,
            bp_memory_budget: crate::segment::reorder::DEFAULT_MEMORY_BUDGET,
            reorder_permits: None,
            reorder_priority: ReorderPriority::AutomaticMerge,
        }
    }

    /// Set the validated flush target for newly written term dictionaries.
    pub fn with_term_dict_block_size(mut self, size: crate::structures::SSTableBlockSize) -> Self {
        self.term_dict_block_size = size;
        self
    }

    /// Configure posting compression for newly encoded output.
    pub fn with_posting_config(
        mut self,
        optimization: crate::structures::IndexOptimization,
        posting_codec: crate::structures::PostingCodec,
    ) -> Self {
        self.optimization = optimization;
        self.posting_codec = posting_codec;
        self
    }

    /// Enable field-local BP during merge (historically BMP-only).
    /// Text and BMP fields still opt in through their schema `reorder` flag.
    pub fn with_reorder_fields(mut self, reorder: bool) -> Self {
        self.reorder_fields = reorder;
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

    pub(crate) fn with_cancellation(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Memory budget for the BP forward index (see `bp_memory_budget`).
    pub fn with_bp_memory_budget(mut self, bytes: usize) -> Self {
        self.bp_memory_budget = bytes;
        self
    }

    /// Share the application-wide whole-segment reorder gate.
    pub fn with_reorder_permits(mut self, permits: Arc<ReorderConcurrencyGate>) -> Self {
        self.reorder_permits = Some(permits);
        self
    }

    pub(crate) fn with_reorder_priority(mut self, priority: ReorderPriority) -> Self {
        self.reorder_priority = priority;
        self
    }

    async fn acquire_reorder_permit(&self) -> Result<Option<crate::index::ReorderPermit>> {
        self.ensure_not_cancelled()?;
        match &self.reorder_permits {
            Some(gate) => Ok(Some(gate.acquire(self.reorder_priority).await.map_err(
                |_| crate::Error::Internal("background reorder scheduler is closed".into()),
            )?)),
            None => Ok(None),
        }
    }

    pub(super) fn ensure_not_cancelled(&self) -> Result<()> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(|cancelled| cancelled.load(Ordering::Acquire))
        {
            Err(crate::Error::IndexClosed)
        } else {
            Ok(())
        }
    }

    /// Reject additive per-field counts that the on-disk formats cannot
    /// represent. All inputs are already-open metadata views; no vector,
    /// posting, or document payload is read here.
    fn validate_merge_capacities(&self, segments: &[SegmentReader]) -> Result<()> {
        // MaxScore skip entries share one u32-addressed section across fields.
        let mut maxscore_skip_entries = MergeCapacity::default();

        for (field, entry) in self.schema.fields() {
            match entry.field_type {
                FieldType::DenseVector | FieldType::BinaryDenseVector => {
                    let mut vectors = MergeCapacity::default();
                    for segment in segments {
                        let Some(flat) = segment.flat_vectors().get(&field.0) else {
                            continue;
                        };
                        if let Some(total) = vectors.add(flat.num_vectors as u64) {
                            let value_kind = if entry.field_type == FieldType::BinaryDenseVector {
                                "binary vectors"
                            } else {
                                "dense vectors"
                            };
                            return Err(field_capacity_error(
                                field.0,
                                &entry.name,
                                value_kind,
                                total,
                            ));
                        }
                    }
                }
                FieldType::SparseVector => {
                    let format = entry
                        .sparse_vector_config
                        .as_ref()
                        .map(|config| config.format)
                        .unwrap_or_default();
                    match format {
                        SparseFormat::Seismic => {
                            let mut vectors = MergeCapacity::default();
                            for segment in segments {
                                if let Some(index) = segment.seismic_index(field)
                                    && let Some(total) =
                                        vectors.add(u64::from(index.total_vectors()))
                                {
                                    return Err(field_capacity_error(
                                        field.0,
                                        &entry.name,
                                        "Seismic vectors",
                                        total,
                                    ));
                                }
                            }
                        }

                        SparseFormat::Bmp => {
                            let mut vectors = MergeCapacity::default();
                            let mut blocks = MergeCapacity::default();
                            let mut real_slots = MergeCapacity::default();
                            let mut virtual_slots = MergeCapacity::default();

                            for segment in segments {
                                let Some(index) = segment.bmp_indexes().get(&field.0) else {
                                    continue;
                                };
                                for (capacity, count, value_kind) in [
                                    (&mut vectors, u64::from(index.total_vectors), "BMP vectors"),
                                    (&mut blocks, u64::from(index.num_blocks), "BMP blocks"),
                                    (
                                        &mut real_slots,
                                        u64::from(index.num_real_docs()),
                                        "BMP real vector slots",
                                    ),
                                    (
                                        &mut virtual_slots,
                                        u64::from(index.num_virtual_docs),
                                        "BMP padded virtual slots",
                                    ),
                                ] {
                                    if let Some(total) = capacity.add(count) {
                                        return Err(field_capacity_error(
                                            field.0,
                                            &entry.name,
                                            value_kind,
                                            total,
                                        ));
                                    }
                                }
                            }
                        }
                        SparseFormat::MaxScore => {
                            let mut vectors = MergeCapacity::default();
                            let mut dimensions: FxHashMap<u32, (MergeCapacity, MergeCapacity)> =
                                FxHashMap::default();

                            for segment in segments {
                                let Some(index) = segment.sparse_indexes().get(&field.0) else {
                                    continue;
                                };
                                if let Some(total) = vectors.add(u64::from(index.total_vectors)) {
                                    return Err(field_capacity_error(
                                        field.0,
                                        &entry.name,
                                        "MaxScore vectors",
                                        total,
                                    ));
                                }

                                for (dimension, doc_count, block_count) in index.dimension_counts()
                                {
                                    let (docs, blocks) = dimensions.entry(dimension).or_default();
                                    if let Some(total) = docs.add(u64::from(doc_count)) {
                                        return Err(field_capacity_error(
                                            field.0,
                                            &entry.name,
                                            &format!("MaxScore postings for dimension {dimension}"),
                                            total,
                                        ));
                                    }
                                    if let Some(total) = blocks.add(u64::from(block_count)) {
                                        return Err(field_capacity_error(
                                            field.0,
                                            &entry.name,
                                            &format!("MaxScore blocks for dimension {dimension}"),
                                            total,
                                        ));
                                    }
                                    if let Some(total) =
                                        maxscore_skip_entries.add(u64::from(block_count))
                                    {
                                        return Err(crate::Error::Schema(format!(
                                            "merge would produce {total} MaxScore skip entries \
                                             across sparse fields, exceeding the segment format \
                                             limit {}; lower max_segment_docs for multi-valued \
                                             sparse fields",
                                            u32::MAX,
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Merge segments into one, streaming postings/positions/store directly to files.
    ///
    /// If `trained` is provided, dense vectors use O(1) cluster merge when possible
    /// (compatible IVF-PQ), otherwise rebuilds ANN from global artifacts.
    /// Without trained structures, only flat vectors are merged.
    ///
    /// Uses streaming writers so postings, positions, and store data flow directly
    /// to files instead of buffering everything in memory. Only the term dictionary
    /// (compact key+TermInfo entries) is buffered.
    ///
    /// This is the physical encoded-copy primitive. Readers with deletion masks
    /// must use `compact`, or the index writer's merge operation, which owns
    /// visibility capture and atomic publication across multiple sources.
    pub async fn merge<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        new_segment_id: SegmentId,
        trained: Option<&TrainedVectorStructures>,
    ) -> Result<(SegmentMeta, MergeStats)> {
        self.ensure_not_cancelled()?;
        if segments
            .iter()
            .any(|segment| segment.deletion_meta().is_some())
        {
            return Err(crate::Error::Schema(
                "encoded-copy merge cannot discard deletion masks; use IndexWriter::force_merge or SegmentMerger::compact".into(),
            ));
        }
        // Reject an unrepresentable merge before creating any output files.
        // The previous late check left a complete orphan output behind after
        // doing all expensive phases.
        let total_docs: u32 = segments
            .iter()
            .try_fold(0u32, |acc, segment| acc.checked_add(segment.num_docs()))
            .ok_or_else(|| {
                crate::Error::Internal(format!(
                    "Total document count exceeds u32::MAX ({})",
                    u32::MAX
                ))
            })?;

        self.validate_merge_capacities(segments)?;

        let mut stats = MergeStats::default();
        let files = SegmentFiles::new(new_segment_id.0);

        // === Two-stage merge to bound page cache pressure ===
        //
        // Stage 1: postings + store + fast_fields (concurrent)
        //   Touches .term_dict, .postings, .positions, .store, .fast files.
        //
        // Stage 2: sparse + dense vectors. Block-copy sparse work runs with
        // dense vectors; BP sparse work runs first to bound peak memory.
        //   Touches .sparse, .vectors files.
        //
        // Running all phases concurrently caused OOM on large merges because
        // mmap'd source files from all 16+ segments compete for page cache
        // simultaneously (200+ GB of mmap'd data for BMP grids alone).
        // Two stages halve the concurrent working set.
        let merge_start = std::time::Instant::now();

        // ── Stage 1: text + store + fast fields ─────────────────────────
        let reorder_text = self.reorder_fields
            && self.schema.fields().any(|(_, entry)| {
                entry.indexed && entry.reorder && entry.field_type == FieldType::Text
            });
        let postings_fut = async {
            let _permit = if reorder_text {
                self.acquire_reorder_permit().await?
            } else {
                None
            };
            let plans = if reorder_text {
                crate::segment::text_reorder::plan_text_reorders_from_sources(
                    segments,
                    &self.schema,
                    self.bp_memory_budget,
                    self.bp_budget,
                    self.cancellation.as_deref(),
                    self.background_pool.clone(),
                    true,
                )
                .await?
            } else {
                Vec::new()
            };
            let text_converged = plans.iter().all(|plan| plan.converged);
            let mut postings_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.postings).await?);
            let mut positions_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.positions).await?);
            let mut term_dict_writer =
                OffsetWriter::new(dir.streaming_writer_cold(&files.term_dict).await?);

            let posting_stats = self
                .merge_postings(
                    segments,
                    &mut term_dict_writer,
                    &mut postings_writer,
                    &mut positions_writer,
                    &plans,
                )
                .await?;
            let terms_processed = posting_stats.terms_processed;

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
                "[merge] index={} postings done: {} terms, term_dict={}, postings={}, positions={}",
                self.schema.index_label(),
                terms_processed,
                crate::format_bytes(term_dict_bytes as u64),
                crate::format_bytes(postings_bytes as u64),
                crate::format_bytes(positions_bytes),
            );
            // Reuse the retained plan after term scratch is released. Do not
            // overlap legacy map migration with budget-sized term buffers.
            if reorder_text {
                self.merge_chunk_maps(dir, segments, &files, &plans).await?;
            }
            Ok::<(usize, usize, usize, usize, bool), crate::Error>((
                terms_processed,
                term_dict_bytes,
                postings_bytes,
                posting_stats.blocks_without_bounds,
                text_converged,
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

        let chunks_fut = async {
            if reorder_text {
                Ok(0)
            } else {
                self.merge_chunk_maps(dir, segments, &files, &[]).await
            }
        };
        let (postings_result, store_result, fast_bytes, _) =
            tokio::try_join!(postings_fut, store_fut, fast_fut, chunks_fut)?;
        self.ensure_not_cancelled()?;

        log::info!(
            "[merge] index={} stage 1 done in {:.1}s (postings + store + fast)",
            self.schema.index_label(),
            merge_start.elapsed().as_secs_f64()
        );

        // ── Stage 2: sparse + dense vectors ─────────────────────────────
        // Page cache from stage 1 files can now be evicted by the kernel
        // as stage 2 accesses different mmap regions (.sparse, .vectors).
        let sparse_fut = async { self.merge_sparse_vectors(dir, segments, &files).await };

        let dense_fut = async {
            self.merge_dense_vectors(dir, segments, &files, trained, AnnWriteMode::Copy)
                .await
        };

        // Merge-time BP constructs a potentially budget-sized forward index.
        // Do not overlap that allocation and its heavy source-file scan with
        // an ANN rebuild. Block-copy sparse merges remain concurrent with ANN.
        let ((sparse_bytes, bp_converged), vectors_bytes) = if self.reorder_fields {
            let sparse = sparse_fut.await?;
            let dense = dense_fut.await?;
            (sparse, dense)
        } else {
            tokio::try_join!(sparse_fut, dense_fut)?
        };
        self.ensure_not_cancelled()?;
        let (store_bytes, store_num_docs) = store_result;
        stats.terms_processed = postings_result.0;
        stats.term_dict_bytes = postings_result.1;
        stats.postings_bytes = postings_result.2;
        stats.posting_blocks_without_bounds = postings_result.3;
        stats.store_bytes = store_bytes;
        stats.vectors_bytes = vectors_bytes;
        stats.sparse_bytes = sparse_bytes;
        stats.bp_converged = bp_converged && postings_result.4;
        stats.fast_bytes = fast_bytes;
        log::info!(
            "[merge] index={} all phases done in {:.1}s: {}",
            self.schema.index_label(),
            merge_start.elapsed().as_secs_f64(),
            stats
        );

        self.merge_row_stats(dir, segments, &files).await?;

        // === Mandatory: merge field stats + write meta ===
        self.ensure_not_cancelled()?;
        let mut merged_field_stats: FxHashMap<u32, FieldStats> = FxHashMap::default();
        for segment in segments {
            for (&field_id, field_stats) in &segment.meta().field_stats {
                let entry = merged_field_stats.entry(field_id).or_default();
                entry.total_tokens = entry
                    .total_tokens
                    .checked_add(field_stats.total_tokens)
                    .ok_or_else(|| {
                        crate::Error::Corruption(format!(
                            "field {} total-token count overflow while merging",
                            field_id
                        ))
                    })?;
                entry.doc_count = entry
                    .doc_count
                    .checked_add(field_stats.doc_count)
                    .ok_or_else(|| {
                        crate::Error::Corruption(format!(
                            "field {} document count overflow while merging",
                            field_id
                        ))
                    })?;
            }
        }

        // Verify store doc count matches metadata — a mismatch here means
        // some store blocks were lost (e.g., compression thread panic) or
        // source segment metadata disagrees with its store.
        if store_num_docs != total_docs {
            log::error!(
                "[merge] index={} STORE/META MISMATCH: store has {} docs but metadata expects {}. \
                 Per-segment: {:?}",
                self.schema.index_label(),
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

        self.ensure_not_cancelled()?;

        // Durable: replace_segments deletes the fsynced source segments right
        // after publishing this output, so a non-durable .meta could be the
        // only copy of the merged documents across a power failure.
        dir.write_durable(&files.meta, &meta.serialize()?).await?;

        // Dense ANN payloads are byte-copied during an ordinary merge; the
        // wall-clock timer also includes postings, store, sparse BP and any BP
        // scheduler wait. Calling this an "ANN merge" made long sparse waits
        // look like ANN construction in production logs.
        log::info!(
            "[merge] index={} complete: {} docs, {}",
            self.schema.index_label(),
            total_docs,
            stats
        );

        Ok((meta, stats))
    }
}

/// Delete segment files from directory (all deletions run concurrently).
pub async fn delete_segment<D: Directory + DirectoryWriter>(
    dir: &D,
    segment_id: SegmentId,
) -> Result<()> {
    let files = SegmentFiles::new(segment_id.0);
    let paths = files.lifecycle_paths();
    let results = futures::future::join_all(paths.iter().map(|path| dir.delete(path))).await;

    // Missing files are expected for optional components and idempotent
    // retries. Any other failure must be surfaced so cleanup is not falsely
    // reported as successful; a later orphan sweep can retry remaining files.
    for result in results {
        if let Err(error) = result
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(crate::Error::Io(error));
        }
    }
    Ok(())
}

#[cfg(test)]
mod capacity_tests {
    use super::{MergeCapacity, field_capacity_error};

    #[test]
    fn merge_capacity_accepts_the_exact_u32_boundary() {
        let mut capacity = MergeCapacity::default();
        assert_eq!(capacity.add(u64::from(u32::MAX) - 7), None);
        assert_eq!(capacity.add(7), None);
    }

    #[test]
    fn merge_capacity_rejects_the_first_value_beyond_u32() {
        let mut capacity = MergeCapacity::default();
        assert_eq!(capacity.add(u64::from(u32::MAX)), None);
        assert_eq!(capacity.add(1), Some(u64::from(u32::MAX) + 1));
    }

    #[test]
    fn merge_capacity_failure_is_not_source_corruption() {
        let error = field_capacity_error(
            7,
            "body_embedding",
            "dense vectors",
            u64::from(u32::MAX) + 1,
        );
        assert!(matches!(error, crate::Error::Schema(_)));
        assert!(error.to_string().contains("lower max_segment_docs"));
    }
}
