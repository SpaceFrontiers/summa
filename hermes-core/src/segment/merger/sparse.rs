//! Sparse vector merge via byte-level block stacking (V3 format).
//!
//! V3 file layout (footer-based, data-first):
//! ```text
//! [block data for all dims across all fields]
//! [skip section: SparseSkipEntry × total (24B each), contiguous]
//! [TOC: per-field header(13B) + per-dim entries(28B each)]
//! [footer: skip_offset(8) + toc_offset(8) + num_fields(4) + magic(4) = 24B]
//! ```
//!
//! Each dimension is merged by stacking raw block bytes from source segments.
//! No deserialization or re-serialization of block data — only the small
//! skip entries (24 bytes per block) are written fresh in a separate section.
//! The raw block bytes are copied directly from mmap.

use std::io::Write;
use std::sync::Arc;

use super::OffsetWriter;
use super::SegmentMerger;
use super::doc_offsets;
use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::FieldType;
use crate::segment::BmpIndex;
use crate::segment::SparseIndex;
use crate::segment::format::{SparseDimTocEntry, SparseFieldToc, write_sparse_toc_and_footer};
use crate::segment::reader::{DimRawData, SegmentReader};
use crate::segment::types::SegmentFiles;
use crate::structures::{SparseFormat, SparseSkipEntry};

impl SegmentMerger {
    /// Merge sparse vector indexes via byte-level block stacking (V3 format).
    ///
    /// V3 separates block data from skip entries:
    ///   Phase 1: Write raw block data for all dims (copied from mmap)
    ///   Phase 2: Write skip section (all skip entries contiguous)
    ///   Phase 3: Write TOC (per-field header + per-dim entries with embedded metadata)
    ///   Phase 4: Write 24-byte footer
    pub(super) async fn merge_sparse_vectors<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segments: &[SegmentReader],
        files: &SegmentFiles,
    ) -> Result<(usize, bool)> {
        let doc_offs = doc_offsets(segments)?;
        // False iff any merge-time BP pass hit its wall-clock budget.
        let mut all_converged = true;

        // Collect all sparse vector fields from schema
        let sparse_fields: Vec<_> = self
            .schema
            .fields()
            .filter(|(_, entry)| matches!(entry.field_type, FieldType::SparseVector))
            .map(|(field, entry)| {
                (
                    field,
                    entry.sparse_vector_config.clone(),
                    entry.reorder,
                    entry.name.clone(),
                )
            })
            .collect();

        if sparse_fields.is_empty() {
            return Ok((0, true));
        }

        let mut writer = OffsetWriter::new(dir.streaming_writer_cold(&files.sparse).await?);
        let scratch_path = dir.local_path(&files.sparse).unwrap_or_else(|| {
            std::env::temp_dir().join(
                files
                    .sparse
                    .file_name()
                    .unwrap_or_else(|| std::ffi::OsStr::new("hermes.sparse")),
            )
        });

        // Accumulated per-field data for TOC
        let mut field_tocs: Vec<SparseFieldToc> = Vec::new();
        // Skip entries written to a temp file to avoid unbounded memory usage.
        // For large indexes (200M+ docs) the skip section can exceed 3 GB.
        let skip_tmp = files.sparse_skip_temp();
        let mut skip_writer = dir.streaming_writer_cold(&skip_tmp).await?;
        let mut skip_count: u32 = 0;
        let mut skip_entry_buf = Vec::with_capacity(SparseSkipEntry::SIZE);

        for (field, sparse_config, field_reorder, field_name) in &sparse_fields {
            let format = sparse_config.as_ref().map(|c| c.format).unwrap_or_default();
            let quantization = sparse_config
                .as_ref()
                .map(|c| c.weight_quantization)
                .unwrap_or(crate::structures::WeightQuantization::Float32);

            // BMP format: merge BMP indexes if any source segments have them
            if format == SparseFormat::Bmp {
                let bmp_indexes: Vec<Option<&BmpIndex>> = segments
                    .iter()
                    .map(|seg| seg.bmp_indexes().get(&field.0))
                    .collect();
                let has_bmp_data = bmp_indexes.iter().any(|bi| bi.is_some());
                if has_bmp_data {
                    let total_vectors_bmp: u32 = bmp_indexes
                        .iter()
                        .filter_map(|bi| bi.map(|idx| idx.total_vectors))
                        .try_fold(0u32, |total, vectors| total.checked_add(vectors))
                        .ok_or_else(|| {
                            crate::Error::Internal(
                                "merged BMP vector count exceeds the V18 u32 format limit".into(),
                            )
                        })?;
                    let bmp_block_size = sparse_config
                        .as_ref()
                        .map(|c| c.bmp_block_size)
                        .unwrap_or(crate::structures::SparseVectorConfig::DEFAULT_BMP_BLOCK_SIZE);
                    let grid_bits = sparse_config
                        .as_ref()
                        .map(|c| c.bmp_grid_bits)
                        .unwrap_or(crate::structures::SparseVectorConfig::DEFAULT_BMP_GRID_BITS);
                    let dims = sparse_config
                        .as_ref()
                        .and_then(|c| c.dims)
                        .unwrap_or_else(|| {
                            // Derive from first available source
                            bmp_indexes
                                .iter()
                                .find_map(|bi| bi.map(|idx| idx.dims()))
                                .unwrap_or(105879)
                        });
                    let max_weight_scale = sparse_config
                        .as_ref()
                        .and_then(|c| c.max_weight)
                        .unwrap_or_else(|| {
                            bmp_indexes
                                .iter()
                                .find_map(|bi| bi.map(|idx| idx.max_weight_scale))
                                .unwrap_or(5.0)
                        });
                    if self.reorder_bmp && !*field_reorder {
                        // Reorder-on-merge is on, but this field opted out via
                        // its schema — fall through to block-copy. Loud so an
                        // operator can see why the merged field stays unordered.
                        log::info!(
                            "[merge_bmp] field {}: `reorder` schema attribute not set — block-copy merge",
                            field.0,
                        );
                    }
                    if self.reorder_bmp && *field_reorder {
                        // Merge-time BP reorder: write the merged blob in
                        // permuted order instead of block stacking. The output
                        // segment needs no standalone reorder pass afterwards.
                        let sources: Vec<(BmpIndex, u32)> = bmp_indexes
                            .iter()
                            .zip(doc_offs.iter())
                            .filter_map(|(opt, &off)| opt.map(|b| (b.clone(), off)))
                            .collect();
                        let fid = field.0;
                        let fname = field_name.clone();
                        let ilabel = self.schema.index_label().to_owned();
                        let effective_block_size = bmp_block_size.clamp(1, 256) as usize;
                        let pool = self.background_pool.clone();
                        let granularity = self.granularity;
                        log::info!(
                            "[merge_bmp] field {}: reorder-on-merge enabled — running BP over {} source(s)",
                            fid,
                            sources.len(),
                        );
                        let bp_budget = self.bp_budget;
                        let bp_memory_budget = self.bp_memory_budget;
                        let out_grid_bits = grid_bits;
                        let scratch_path = scratch_path.clone();
                        let _reorder_permit = match &self.reorder_permits {
                            Some(permits) => {
                                let gate = Arc::clone(permits);
                                Some(gate.acquire(self.reorder_priority).await.map_err(|_| {
                                    crate::Error::Internal(
                                        "background reorder scheduler is closed".into(),
                                    )
                                })?)
                            }
                            None => None,
                        };
                        // `spawn_blocking` detaches its closure when the
                        // awaiting RPC is cancelled. That used to release the
                        // segment operation guard while this writer was still
                        // active, allowing cleanup to race the output. A
                        // synchronous block-in-place section cannot be
                        // cancelled halfway through the owned writer.
                        let (w, ft, converged) = super::block_in_place_if_multithread(move || {
                            crate::segment::reorder::reorder_bmp_field(
                                &sources,
                                fid,
                                &ilabel,
                                &fname,
                                dims,
                                effective_block_size,
                                out_grid_bits,
                                max_weight_scale,
                                total_vectors_bmp,
                                bp_memory_budget,
                                // Wall-clock-bounded so huge merges don't hold
                                // a merge slot for full BP depth; a truncated
                                // pass is marked unconverged and the background
                                // optimizer deepens it (warm-started).
                                bp_budget,
                                // Auto unless a source is an unconverged
                                // partial reorder (then Records, see
                                // SegmentMerger::granularity).
                                granularity,
                                scratch_path,
                                writer,
                                field_tocs,
                                pool,
                            )
                        })?;
                        writer = w;
                        field_tocs = ft;
                        all_converged &= converged;
                        continue;
                    }

                    // GB-scale sync copy loops — migrate this tokio worker's
                    // queue so the merge doesn't pin a runtime thread.
                    super::block_in_place_if_multithread(|| {
                        merge_bmp_field(
                            &bmp_indexes,
                            &doc_offs,
                            field.0,
                            dims,
                            bmp_block_size,
                            grid_bits,
                            max_weight_scale,
                            total_vectors_bmp,
                            &scratch_path,
                            &mut writer,
                            &mut field_tocs,
                        )
                    })?;
                }
                continue;
            }

            // MaxScore format: byte-level block stacking
            // Collect all unique dimension IDs across segments (sorted for determinism)
            let all_dims: Vec<u32> = {
                let mut set = rustc_hash::FxHashSet::default();
                for segment in segments {
                    if let Some(si) = segment.sparse_indexes().get(&field.0) {
                        for dim_id in si.active_dimensions() {
                            set.insert(dim_id);
                        }
                    }
                }
                let mut v: Vec<u32> = set.into_iter().collect();
                v.sort_unstable();
                v
            };

            if all_dims.is_empty() {
                continue;
            }

            let sparse_indexes: Vec<Option<&SparseIndex>> = segments
                .iter()
                .map(|seg| seg.sparse_indexes().get(&field.0))
                .collect();

            // Sum total_vectors across segments for this field
            let total_vectors: u32 = sparse_indexes
                .iter()
                .filter_map(|si| si.map(|idx| idx.total_vectors))
                .try_fold(0u32, |total, vectors| total.checked_add(vectors))
                .ok_or_else(|| {
                    crate::Error::Internal(
                        "merged sparse vector count exceeds the V3 u32 format limit".into(),
                    )
                })?;

            log::debug!(
                "[sparse_vector_merge] field {}: {} unique dims across {} segments, total_vectors={}",
                field.0,
                all_dims.len(),
                segments.len(),
                total_vectors,
            );

            let mut dim_toc_entries: Vec<SparseDimTocEntry> = Vec::with_capacity(all_dims.len());

            for &dim_id in &all_dims {
                // Collect raw data from each segment (skip entries + raw block bytes)
                let mut sources = Vec::with_capacity(segments.len());
                for (seg_idx, sparse_idx) in sparse_indexes.iter().enumerate() {
                    if let Some(idx) = sparse_idx
                        && let Some(raw) = idx.read_dim_raw(dim_id).await?
                    {
                        sources.push((raw, doc_offs[seg_idx]));
                    }
                }

                if sources.is_empty() {
                    continue;
                }

                // Compute merged metadata
                let total_docs: u32 = sources
                    .iter()
                    .map(|(raw, _)| raw.doc_count)
                    .try_fold(0u32, |total, docs| total.checked_add(docs))
                    .ok_or_else(|| {
                        crate::Error::Internal(
                            "merged sparse dimension doc count exceeds u32::MAX".into(),
                        )
                    })?;
                let global_max: f32 = sources
                    .iter()
                    .map(|(r, _)| r.global_max_weight)
                    .fold(f32::NEG_INFINITY, f32::max);
                let total_blocks: u32 = sources
                    .iter()
                    .map(|(raw, _)| u32::try_from(raw.skip_entries.len()))
                    .try_fold(0u32, |total, blocks| total.checked_add(blocks.ok()?))
                    .ok_or_else(|| {
                        crate::Error::Internal(
                            "merged sparse dimension block count exceeds u32::MAX".into(),
                        )
                    })?;

                // Validate every source before writing any bytes for this
                // dimension. These checks are cheap relative to copying the
                // blocks and turn corrupt skip tables into an ordinary merge
                // error instead of an indexing panic or a partially patched
                // output. This used to live behind the diagnostics feature
                // and its contiguity calculation was never enforced.
                for (source_index, (raw, _)) in sources.iter().enumerate() {
                    validate_sparse_merge_source(dim_id, source_index, raw)?;
                }

                // Phase 1: Write block data only (no header, no skip entries)
                let block_data_offset = writer.offset();

                // Serialize adjusted skip entries directly to byte buffer
                let skip_start = skip_count;
                let mut cumulative_block_offset = 0u64;

                for (raw, doc_offset) in &sources {
                    let data = raw.raw_block_data.as_slice();

                    // Serialize adjusted skip entries to temp file (batched write)
                    for entry in &raw.skip_entries {
                        skip_entry_buf.clear();
                        let first_doc =
                            entry.first_doc.checked_add(*doc_offset).ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "sparse first-doc offset overflow: {} + {}",
                                    entry.first_doc, doc_offset
                                ))
                            })?;
                        let last_doc =
                            entry.last_doc.checked_add(*doc_offset).ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "sparse last-doc offset overflow: {} + {}",
                                    entry.last_doc, doc_offset
                                ))
                            })?;
                        let block_offset = cumulative_block_offset
                            .checked_add(entry.offset)
                            .ok_or_else(|| {
                                crate::Error::Internal(
                                    "merged sparse block-data offset exceeds u64::MAX".into(),
                                )
                            })?;
                        SparseSkipEntry::new(
                            first_doc,
                            last_doc,
                            block_offset,
                            entry.length,
                            entry.max_weight,
                        )
                        .write_to_vec(&mut skip_entry_buf);
                        skip_writer
                            .write_all(&skip_entry_buf)
                            .map_err(crate::Error::Io)?;
                        skip_count = skip_count.checked_add(1).ok_or_else(|| {
                            crate::Error::Internal(
                                "merged sparse skip-entry count exceeds u32::MAX".into(),
                            )
                        })?;
                    }
                    // Advance cumulative offset by this source's total block data size
                    if let Some(last) = raw.skip_entries.last() {
                        cumulative_block_offset = cumulative_block_offset
                            .checked_add(last.offset)
                            .and_then(|offset| offset.checked_add(last.length as u64))
                            .ok_or_else(|| {
                                crate::Error::Internal(
                                    "merged sparse block-data size exceeds u64::MAX".into(),
                                )
                            })?;
                    }

                    // Write raw block data, patching first_doc_id when doc_offset > 0
                    if *doc_offset == 0 {
                        writer.write_all(data)?;
                    } else {
                        for (i, entry) in raw.skip_entries.iter().enumerate() {
                            let start = entry.offset as usize;
                            let end = if i + 1 < raw.skip_entries.len() {
                                raw.skip_entries[i + 1].offset as usize
                            } else {
                                data.len()
                            };
                            let block = &data[start..end];
                            writer.write_all(&block[..SPARSE_FIRST_DOC_ID_OFFSET])?;
                            let old = u32::from_le_bytes(
                                block[SPARSE_FIRST_DOC_ID_OFFSET..SPARSE_FIRST_DOC_ID_OFFSET + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            let adjusted = old.checked_add(*doc_offset).ok_or_else(|| {
                                crate::Error::Corruption(format!(
                                    "sparse block doc-id offset overflow: {} + {}",
                                    old, doc_offset
                                ))
                            })?;
                            writer.write_all(&adjusted.to_le_bytes())?;
                            writer.write_all(&block[SPARSE_FIRST_DOC_ID_OFFSET + 4..])?;
                        }
                    }
                }

                dim_toc_entries.push(SparseDimTocEntry {
                    dim_id,
                    block_data_offset,
                    skip_start,
                    num_blocks: total_blocks,
                    doc_count: total_docs,
                    max_weight: global_max,
                });
            }

            if !dim_toc_entries.is_empty() {
                field_tocs.push(SparseFieldToc {
                    field_id: field.0,
                    quantization: quantization as u8,
                    total_vectors,
                    dims: dim_toc_entries,
                });
            }
        }

        // Finalize the skip temp file so it can be read back
        skip_writer.finish().map_err(crate::Error::Io)?;

        if field_tocs.is_empty() {
            drop(writer);
            let _ = dir.delete(&files.sparse).await;
            let _ = dir.delete(&skip_tmp).await;
            return Ok((0, all_converged));
        }

        // Phase 2: Stream-copy skip entries from temp file to main writer.
        let skip_offset = writer.offset();
        let skip_size = skip_count as u64 * SparseSkipEntry::SIZE as u64;
        super::append_and_delete_temp(dir, &skip_tmp, skip_size, &mut writer).await?;

        // Phase 3 + 4: Write TOC + footer
        let toc_offset = writer.offset();
        write_sparse_toc_and_footer(&mut writer, skip_offset, toc_offset, &field_tocs)
            .map_err(crate::Error::Io)?;

        let output_size = writer.offset() as usize;
        writer.finish().map_err(crate::Error::Io)?;

        let total_dims: usize = field_tocs.iter().map(|f| f.dims.len()).sum();
        log::info!(
            "[sparse_vector_merge] file written: {} ({} fields, {} dims, {} skip entries)",
            crate::format_bytes(output_size as u64),
            field_tocs.len(),
            total_dims,
            skip_count,
        );

        Ok((output_size, all_converged))
    }
}

const SPARSE_BLOCK_HEADER_SIZE: usize = 16;
const SPARSE_FIRST_DOC_ID_OFFSET: usize = 8;

/// Validate the raw/skip relationship required by byte-level block stacking.
/// Reader construction validates the outer sparse-file layout; this verifies
/// per-dimension offsets before any unchecked slices or header patches.
fn validate_sparse_merge_source(dim_id: u32, source_index: usize, raw: &DimRawData) -> Result<()> {
    let data = raw.raw_block_data.as_slice();
    let data_len = data.len() as u64;
    let mut expected_offset = 0u64;
    let mut previous_last_doc = None;

    for (block_index, entry) in raw.skip_entries.iter().enumerate() {
        if entry.offset != expected_offset {
            return Err(crate::Error::Corruption(format!(
                "sparse dim {} source {} block {} is non-contiguous: offset={}, expected={}",
                dim_id, source_index, block_index, entry.offset, expected_offset,
            )));
        }
        if entry.length < SPARSE_BLOCK_HEADER_SIZE as u32 {
            return Err(crate::Error::Corruption(format!(
                "sparse dim {} source {} block {} is shorter than its header: {} bytes",
                dim_id, source_index, block_index, entry.length,
            )));
        }
        let end = entry
            .offset
            .checked_add(u64::from(entry.length))
            .ok_or_else(|| {
                crate::Error::Corruption(format!(
                    "sparse dim {} source {} block {} range overflows u64",
                    dim_id, source_index, block_index,
                ))
            })?;
        if end > data_len {
            return Err(crate::Error::Corruption(format!(
                "sparse dim {} source {} block {} ends at {}, beyond {}-byte data",
                dim_id, source_index, block_index, end, data_len,
            )));
        }
        if entry.first_doc > entry.last_doc
            || previous_last_doc.is_some_and(|previous| entry.first_doc < previous)
        {
            return Err(crate::Error::Corruption(format!(
                "sparse dim {} source {} block {} has non-monotonic doc range {}..={}",
                dim_id, source_index, block_index, entry.first_doc, entry.last_doc,
            )));
        }

        let start = usize::try_from(entry.offset).map_err(|_| {
            crate::Error::Corruption(format!(
                "sparse dim {} source {} block {} offset exceeds usize",
                dim_id, source_index, block_index,
            ))
        })?;
        let count = u16::from_le_bytes(data[start..start + 2].try_into().unwrap());
        let header_first_doc = u32::from_le_bytes(
            data[start + SPARSE_FIRST_DOC_ID_OFFSET..start + SPARSE_FIRST_DOC_ID_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        if count == 0 || header_first_doc != entry.first_doc {
            return Err(crate::Error::Corruption(format!(
                "sparse dim {} source {} block {} header mismatch: count={}, header_first={}, skip_first={}",
                dim_id, source_index, block_index, count, header_first_doc, entry.first_doc,
            )));
        }

        expected_offset = end;
        previous_last_doc = Some(entry.last_doc);
    }

    if expected_offset != data_len {
        return Err(crate::Error::Corruption(format!(
            "sparse dim {} source {} skip table covers {} of {} data bytes",
            dim_id, source_index, expected_offset, data_len,
        )));
    }
    Ok(())
}

/// Merge BMP fields with the **streaming block-copy V18 format**.
///
/// Block-copy merge: all segments share the same `dims`, `bmp_block_size`,
/// and `max_weight_scale`, so blocks are self-contained and can be copied directly.
///
/// Phases:
/// 1. Stream Section B (block data) — sequential chunked copy
/// 2. Write padding + Section A (block_data_starts) — recomputed on-the-fly
/// 3. Stream Section D — payload groups already aligned in output coordinates
///    are copied byte-for-byte; shifted groups use a bounded decode/repack
/// 4. Stream Sections E+H — ceil-u4 values decoded/repacked into the output
///    superblock and coarse-superblock coordinate systems
/// 5. Stream Section F+G (doc_map) — bulk copy with offset patching
/// 6. Write the V18 footer
///
/// Peak memory is one row's group descriptors/selectors, one superblock row,
/// and fixed-size copy/decode buffers; it does not scale with postings.
#[allow(clippy::too_many_arguments)]
fn merge_bmp_field(
    bmp_indexes: &[Option<&BmpIndex>],
    doc_offs: &[u32],
    field_id: u32,
    dims: u32,
    bmp_block_size: u32,
    grid_bits: u8,
    max_weight_scale: f32,
    total_vectors: u32,
    scratch_path: &std::path::Path,
    writer: &mut OffsetWriter,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    use crate::segment::builder::bmp::write_bmp_footer;
    let effective_block_size = bmp_block_size.clamp(1, 256);

    // Pre-build source list: (&BmpIndex, doc_offset). Filters out None segments.
    let sources: Vec<(&BmpIndex, u32)> = bmp_indexes
        .iter()
        .copied()
        .zip(doc_offs.iter().copied())
        .filter_map(|(opt, doc_off)| opt.map(|bmp| (bmp, doc_off)))
        .collect();

    if sources.is_empty() {
        return Ok(());
    }
    // Validation and every following phase are exhaustive sequential scans.
    // Apply scan advice before touching multi-GB doc maps, not only before
    // block-data copying.
    let _source_pages =
        crate::segment::reader::bmp::BmpScanPageGuard::new(sources.iter().map(|(bmp, _)| *bmp));

    // ── Phase 0: Validate all sources share dims, block_size, max_weight_scale ──
    let mut total_source_blocks: u32 = 0;
    let mut num_real_docs_total: u32 = 0;

    for &(bmp, _) in &sources {
        bmp.validate_rewrite_layout(
            "BMP merge",
            dims,
            effective_block_size,
            grid_bits,
            max_weight_scale,
        )?;
        bmp.visit_real_slots_for_rewrite(|_| {})?;
        total_source_blocks = total_source_blocks
            .checked_add(bmp.num_blocks)
            .ok_or_else(|| {
                crate::Error::Internal(
                    "merged BMP block count exceeds the V18 u32 format limit".into(),
                )
            })?;
        num_real_docs_total = num_real_docs_total
            .checked_add(bmp.num_real_docs())
            .ok_or_else(|| {
                crate::Error::Internal(
                    "merged BMP real-document count exceeds the V18 u32 format limit".into(),
                )
            })?;
    }

    if total_source_blocks == 0 {
        return Ok(());
    }

    let num_blocks = total_source_blocks as usize;
    let num_virtual_docs = num_blocks
        .checked_mul(effective_block_size as usize)
        .filter(|&count| count <= u32::MAX as usize)
        .ok_or_else(|| {
            crate::Error::Internal(
                "merged BMP virtual-document count exceeds the V18 u32 format limit".into(),
            )
        })?;
    // Pre-compute aggregate stats (needed for footer, independent of block order)
    let mut total_terms: u64 = 0;
    let mut total_postings: u64 = 0;
    for &(bmp, _) in &sources {
        total_terms = total_terms.saturating_add(bmp.total_terms());
        total_postings = total_postings.saturating_add(bmp.total_postings());
    }
    log::debug!(
        "[merge_bmp_v18] field {}: dims={}, {} sources, {} total_blocks, \
         block_size={}, max_weight_scale={:.4}",
        field_id,
        dims,
        sources.len(),
        num_blocks,
        effective_block_size,
        max_weight_scale,
    );

    let blob_start = writer.offset();

    const CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

    // ═══ Sequential block-copy merge ════════════════════════════════════
    // Blocks are self-contained — copy directly from sources in order.
    // BP reorder is a separate post-merge step (see segment::reorder).

    // ── Phase 1: Stream Section B (block data) — chunked copy ───
    let mut source_byte_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    let mut cumulative_bytes: u64 = 0;

    for &(bmp, _) in &sources {
        source_byte_offsets.push(cumulative_bytes);
        let sentinel = bmp.block_data_sentinel() as usize;
        let src_data = &bmp.block_data_slice()[..sentinel];
        for chunk in src_data.chunks(CHUNK_SIZE) {
            writer.write_all(chunk).map_err(crate::Error::Io)?;
        }
        cumulative_bytes += sentinel as u64;
    }

    // Release block data pages
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_dontneed_block_data();
    }

    // ── Phase 2: Write padding + Section A (block_data_starts) ──
    let block_data_len = writer.offset() - blob_start;
    let padding = (8 - (block_data_len % 8) as usize) % 8;
    if padding > 0 {
        writer
            .write_all(&[0u8; 8][..padding])
            .map_err(crate::Error::Io)?;
    }

    for (src_idx, &(bmp, _)) in sources.iter().enumerate() {
        let base = source_byte_offsets[src_idx];
        for b in 0..bmp.num_blocks {
            let val = base + bmp.block_data_start(b);
            writer
                .write_all(&val.to_le_bytes())
                .map_err(crate::Error::Io)?;
        }
    }
    // Sentinel
    writer
        .write_all(&cumulative_bytes.to_le_bytes())
        .map_err(crate::Error::Io)?;

    // ── Phase 3: Stream Section D. Groups already aligned in output
    // coordinates are copied byte-for-byte; shifted output groups use a
    // bounded 256-cell decode/repack.
    let grid_offset = writer.offset() - blob_start;
    let block_grid_bytes =
        write_merged_block_grid(&sources, dims as usize, num_blocks, grid_bits, writer)?;

    // ── Phase 4: Stream Sections E+H together. Source superblock coordinates
    // restart at every segment, so E is remapped by source block offsets.
    // H is derived from that same E row instead of decoding/filling it again.
    let sb_grid_offset = writer.offset() - blob_start;
    debug_assert_eq!(sb_grid_offset, grid_offset + block_grid_bytes);
    let coarse_spool = scratch_path.with_file_name(format!(
        "{}.merge_coarse_{}.tmp",
        scratch_path
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_else(|| "hermes.sparse".into()),
        field_id,
    ));
    let (sb_grid_bytes, coarse_grid_bytes) =
        write_merged_superblock_grids(&sources, dims as usize, num_blocks, &coarse_spool, writer)?;
    let coarse_grid_offset = sb_grid_offset
        .checked_add(sb_grid_bytes)
        .ok_or_else(|| crate::Error::Internal("merged BMP grid offset exceeds u64".into()))?;
    debug_assert_eq!(
        writer.offset() - blob_start,
        coarse_grid_offset + coarse_grid_bytes
    );

    // Release grid pages
    #[cfg(feature = "native")]
    for &(bmp, _) in &sources {
        bmp.madvise_dontneed_grids();
    }

    // ── Phase 5: Stream Section F+G (doc_map) — bulk copy ───────
    let doc_map_offset = writer.offset() - blob_start;

    const DOC_MAP_CHUNK: usize = 64 * 1024;
    let mut id_buf = vec![0u8; DOC_MAP_CHUNK * 4];

    for &(bmp, doc_offset) in &sources {
        let src_ids = bmp.doc_map_ids_slice();
        let n = bmp.num_virtual_docs as usize;

        if doc_offset == 0 {
            for chunk in src_ids[..n * 4].chunks(CHUNK_SIZE) {
                writer.write_all(chunk).map_err(crate::Error::Io)?;
            }
        } else {
            for chunk_start in (0..n).step_by(DOC_MAP_CHUNK) {
                let chunk_end = (chunk_start + DOC_MAP_CHUNK).min(n);
                let chunk_len = chunk_end - chunk_start;
                let src = &src_ids[chunk_start * 4..chunk_end * 4];
                let dst = &mut id_buf[..chunk_len * 4];
                dst.copy_from_slice(src);

                for i in 0..chunk_len {
                    let off = i * 4;
                    let doc_id = u32::from_le_bytes(dst[off..off + 4].try_into().unwrap());
                    if doc_id != u32::MAX {
                        let adjusted = doc_id.checked_add(doc_offset).ok_or_else(|| {
                            crate::Error::Corruption(format!(
                                "BMP doc-id offset overflow: {} + {}",
                                doc_id, doc_offset
                            ))
                        })?;
                        dst[off..off + 4].copy_from_slice(&adjusted.to_le_bytes());
                    }
                }
                writer.write_all(dst).map_err(crate::Error::Io)?;
            }
        }
    }
    drop(id_buf);

    for &(bmp, _) in &sources {
        let src_ords = bmp.doc_map_ordinals_slice();
        let n = bmp.num_virtual_docs as usize;
        for chunk in src_ords[..n * 2].chunks(CHUNK_SIZE) {
            writer.write_all(chunk).map_err(crate::Error::Io)?;
        }
    }

    // ── Phase 6: Write V18 footer ───────────────────────────────────────
    write_bmp_footer(
        writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
        coarse_grid_offset,
        num_blocks as u32,
        dims,
        effective_block_size,
        num_virtual_docs as u32,
        max_weight_scale,
        doc_map_offset,
        num_real_docs_total,
        grid_bits,
    )
    .map_err(crate::Error::Io)?;

    let blob_len = writer.offset() - blob_start;
    field_tocs.push(SparseFieldToc::bmp(
        field_id,
        total_vectors,
        blob_start,
        blob_len,
    ));

    Ok(())
}

enum MergedBlockGroup<'a> {
    Borrowed(crate::segment::bmp_grid::PackedGridGroup<'a>),
    Repacked { width: u8 },
}

#[allow(clippy::too_many_arguments)]
fn prepare_merged_block_group<'a>(
    output_group: usize,
    num_blocks: usize,
    source_bases: &[usize],
    source_group_bases: &[usize],
    row_groups: &[crate::segment::bmp_grid::PackedGridGroup<'a>],
    source_cursor: &mut usize,
    values: &mut [u8; crate::segment::bmp_grid::GRID_GROUP_CELLS],
) -> Result<MergedBlockGroup<'a>> {
    use crate::segment::bmp_grid::{GRID_GROUP_CELLS, bit_width};

    let output_start = output_group * GRID_GROUP_CELLS;
    let output_end = (output_start + GRID_GROUP_CELLS).min(num_blocks);
    while source_bases
        .get(*source_cursor + 1)
        .is_some_and(|&end| end <= output_start)
    {
        *source_cursor += 1;
    }
    let source_end = source_bases.get(*source_cursor + 1).ok_or_else(|| {
        crate::Error::Corruption("BMP merge block-grid source range is incomplete".into())
    })?;
    let local_start = output_start
        .checked_sub(source_bases[*source_cursor])
        .ok_or_else(|| {
            crate::Error::Corruption("BMP merge block-grid source ranges overlap".into())
        })?;

    // The common large-segment case: this output group is exactly one
    // complete source group. Preserve its payload byte-for-byte.
    if output_end - output_start == GRID_GROUP_CELLS
        && output_end <= *source_end
        && local_start.is_multiple_of(GRID_GROUP_CELLS)
    {
        let group_index = source_group_bases[*source_cursor] + local_start / GRID_GROUP_CELLS;
        let group = row_groups.get(group_index).copied().ok_or_else(|| {
            crate::Error::Corruption("BMP merge block-grid group is missing".into())
        })?;
        return Ok(MergedBlockGroup::Borrowed(group));
    }

    values.fill(0);
    let mut global = output_start;
    let mut current_source = *source_cursor;
    while global < output_end {
        while source_bases
            .get(current_source + 1)
            .is_some_and(|&end| end <= global)
        {
            current_source += 1;
        }
        let current_end = *source_bases.get(current_source + 1).ok_or_else(|| {
            crate::Error::Corruption("BMP merge block-grid source range is incomplete".into())
        })?;
        let local = global
            .checked_sub(source_bases[current_source])
            .ok_or_else(|| {
                crate::Error::Corruption("BMP merge block-grid source ranges overlap".into())
            })?;
        let source_group = local / GRID_GROUP_CELLS;
        let within = local % GRID_GROUP_CELLS;
        let count = (GRID_GROUP_CELLS - within)
            .min(output_end - global)
            .min(current_end - global);
        let destination = global - output_start;
        let group_index = source_group_bases[current_source] + source_group;
        row_groups
            .get(group_index)
            .copied()
            .ok_or_else(|| {
                crate::Error::Corruption("BMP merge block-grid group is missing".into())
            })?
            .decode(within, count, &mut values[destination..destination + count]);
        global += count;
    }
    Ok(MergedBlockGroup::Repacked {
        width: bit_width(values.iter().copied().max().unwrap_or(0)),
    })
}

fn write_merged_block_grid<'a>(
    sources: &[(&'a BmpIndex, u32)],
    dims: usize,
    num_blocks: usize,
    grid_bits: u8,
    writer: &mut dyn Write,
) -> Result<u64> {
    use crate::segment::bmp_grid::{CompressedGridLayout, GRID_GROUP_CELLS, pack_group};

    let layout = CompressedGridLayout::new(dims, num_blocks);
    let mut source_bases = Vec::with_capacity(sources.len() + 1);
    let mut source_group_bases = Vec::with_capacity(sources.len() + 1);
    source_bases.push(0usize);
    source_group_bases.push(0usize);
    for &(bmp, _) in sources {
        let next = source_bases
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(bmp.num_blocks as usize)
            .ok_or_else(|| {
                crate::Error::Internal("merged BMP block-grid offset overflows usize".into())
            })?;
        source_bases.push(next);
        source_group_bases.push(
            source_group_bases
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(bmp.block_grid().groups())
                .ok_or_else(|| {
                    crate::Error::Internal(
                        "merged BMP block-grid group offset overflows usize".into(),
                    )
                })?,
        );
    }
    if source_bases.last().copied() != Some(num_blocks) {
        return Err(crate::Error::Internal(
            "merged BMP block-grid source lengths disagree with output".into(),
        ));
    }

    let mut widths = vec![0u8; layout.groups()];
    let mut row_sizes = Vec::with_capacity(dims);
    let mut row_groups: Vec<crate::segment::bmp_grid::PackedGridGroup<'a>> =
        Vec::with_capacity(*source_group_bases.last().unwrap_or(&0));

    let load_row = |dimension: usize,
                    groups: &mut Vec<crate::segment::bmp_grid::PackedGridGroup<'a>>|
     -> Result<()> {
        groups.clear();
        for &(bmp, _) in sources {
            let grid = bmp.block_grid();
            if grid.dims() != dims || grid.max_width() != grid_bits {
                return Err(crate::Error::Corruption(
                    "BMP merge block-grid metadata disagrees with the field schema".into(),
                ));
            }
            grid.append_row_groups(dimension, groups)?;
        }
        if groups.len() != *source_group_bases.last().unwrap_or(&0) {
            return Err(crate::Error::Corruption(
                "BMP merge block-grid row has an inconsistent group count".into(),
            ));
        }
        Ok(())
    };
    let mut group_values = [0u8; GRID_GROUP_CELLS];
    let collect_widths = |widths: &mut [u8],
                          groups: &[crate::segment::bmp_grid::PackedGridGroup<'a>],
                          values: &mut [u8; GRID_GROUP_CELLS]|
     -> Result<()> {
        let mut source = 0usize;
        for (output_group, output_width) in widths.iter_mut().enumerate() {
            *output_width = match prepare_merged_block_group(
                output_group,
                num_blocks,
                &source_bases,
                &source_group_bases,
                groups,
                &mut source,
                values,
            )? {
                MergedBlockGroup::Borrowed(group) => group.width(),
                MergedBlockGroup::Repacked { width } => width,
            };
        }
        Ok(())
    };

    for dimension in 0..dims {
        load_row(dimension, &mut row_groups)?;
        collect_widths(&mut widths, &row_groups, &mut group_values)?;
        row_sizes.push(layout.row_bytes(&widths)?);
    }
    let table_bytes = layout
        .write_row_offsets(&row_sizes, writer)
        .map_err(crate::Error::Io)?;

    let largest_payload = row_sizes
        .iter()
        .copied()
        .max()
        .unwrap_or(layout.row_header_bytes() as u64)
        .saturating_sub(layout.row_header_bytes() as u64);
    let mut row_payload = Vec::with_capacity(usize::try_from(largest_payload).map_err(|_| {
        crate::Error::Internal("merged BMP block-grid row exceeds addressable memory".into())
    })?);
    let mut packed = [0u8; GRID_GROUP_CELLS];
    for (dimension, &row_size) in row_sizes.iter().enumerate() {
        load_row(dimension, &mut row_groups)?;
        row_payload.clear();
        let mut source = 0usize;
        for (output_group, output_width) in widths.iter_mut().enumerate() {
            match prepare_merged_block_group(
                output_group,
                num_blocks,
                &source_bases,
                &source_group_bases,
                &row_groups,
                &mut source,
                &mut group_values,
            )? {
                MergedBlockGroup::Borrowed(group) => {
                    *output_width = group.width();
                    row_payload.extend_from_slice(group.bytes());
                }
                MergedBlockGroup::Repacked { width } => {
                    *output_width = width;
                    let payload_len = pack_group(&group_values, width, &mut packed)?;
                    row_payload.extend_from_slice(&packed[..payload_len]);
                }
            }
        }
        let expected_payload = usize::try_from(row_size)
            .map_err(|_| {
                crate::Error::Internal(
                    "merged BMP block-grid row exceeds addressable memory".into(),
                )
            })?
            .checked_sub(layout.row_header_bytes())
            .ok_or_else(|| {
                crate::Error::Corruption(
                    "merged BMP block-grid row is shorter than its header".into(),
                )
            })?;
        if row_payload.len() != expected_payload {
            return Err(crate::Error::Corruption(
                "BMP merge block-grid sizing changed during encoding".into(),
            ));
        }
        layout
            .write_row_header(&widths, grid_bits, writer)
            .map_err(crate::Error::Io)?;
        writer.write_all(&row_payload).map_err(crate::Error::Io)?;
    }
    Ok(table_bytes + row_sizes.into_iter().sum::<u64>())
}

struct MergedSuperblockRows {
    superblock: Vec<u8>,
    coarse: Vec<u8>,
    superblock_cells_touched: Vec<usize>,
    coarse_cells_touched: Vec<usize>,
    superblock_widths: Vec<u8>,
    coarse_widths: Vec<u8>,
    superblock_groups_touched: Vec<usize>,
    coarse_groups_touched: Vec<usize>,
    decoded: [u8; crate::segment::bmp_grid::GRID_GROUP_CELLS],
}

impl MergedSuperblockRows {
    fn new(superblock_cells: usize, coarse_cells: usize) -> Self {
        use crate::segment::bmp_grid::{CompressedGridLayout, GRID_GROUP_CELLS};

        Self {
            superblock: vec![0; superblock_cells],
            coarse: vec![0; coarse_cells],
            superblock_cells_touched: Vec::new(),
            coarse_cells_touched: Vec::new(),
            superblock_widths: vec![0; CompressedGridLayout::new(0, superblock_cells).groups()],
            coarse_widths: vec![0; CompressedGridLayout::new(0, coarse_cells).groups()],
            superblock_groups_touched: Vec::new(),
            coarse_groups_touched: Vec::new(),
            decoded: [0; GRID_GROUP_CELLS],
        }
    }

    fn clear_projection(
        row: &mut [u8],
        cells_touched: &mut Vec<usize>,
        widths: &mut [u8],
        groups_touched: &mut Vec<usize>,
    ) {
        for cell in cells_touched.drain(..) {
            row[cell] = 0;
        }
        for group in groups_touched.drain(..) {
            widths[group] = 0;
        }
    }

    fn prepare<'a>(
        &mut self,
        sources: &[(&'a BmpIndex, u32)],
        source_block_bases: &[usize],
        dims: usize,
        dimension: usize,
        source_groups: &mut Vec<crate::segment::bmp_grid::PackedGridGroup<'a>>,
    ) -> Result<()> {
        use crate::segment::bmp_grid::{GRID_GROUP_CELLS, bit_width};

        const SB: usize = crate::segment::BMP_SUPERBLOCK_SIZE as usize;
        Self::clear_projection(
            &mut self.superblock,
            &mut self.superblock_cells_touched,
            &mut self.superblock_widths,
            &mut self.superblock_groups_touched,
        );
        Self::clear_projection(
            &mut self.coarse,
            &mut self.coarse_cells_touched,
            &mut self.coarse_widths,
            &mut self.coarse_groups_touched,
        );

        for (source, &(bmp, _)) in sources.iter().enumerate() {
            let grid = bmp.superblock_grid();
            if grid.dims() != dims
                || grid.max_width() != crate::segment::bmp_grid::LSP_SUPERBLOCK_GRID_BITS
            {
                return Err(crate::Error::Corruption(
                    "BMP merge superblock-grid metadata disagrees with the field schema".into(),
                ));
            }
            let source_blocks = bmp.num_blocks as usize;
            source_groups.clear();
            grid.append_row_groups(dimension, source_groups)?;
            for (source_group, group) in source_groups.iter().copied().enumerate() {
                let source_cell_start = source_group * GRID_GROUP_CELLS;
                let count = GRID_GROUP_CELLS.min(grid.cells() - source_cell_start);
                if group.width() == 0 {
                    continue;
                }
                group.decode(0, count, &mut self.decoded);
                for (within, &value) in self.decoded[..count].iter().enumerate() {
                    if value == 0 {
                        continue;
                    }
                    let source_sb = source_cell_start + within;
                    let local_block_start = source_sb * SB;
                    if local_block_start >= source_blocks {
                        break;
                    }
                    let local_block_end = (local_block_start + SB).min(source_blocks);
                    let global_block_start = source_block_bases[source] + local_block_start;
                    let global_block_end = source_block_bases[source] + local_block_end;
                    let first_output_sb = global_block_start / SB;
                    let last_output_sb = (global_block_end - 1) / SB;
                    for output_sb in first_output_sb..=last_output_sb {
                        let slot = &mut self.superblock[output_sb];
                        if *slot == 0 {
                            self.superblock_cells_touched.push(output_sb);
                        }
                        *slot = (*slot).max(value);
                    }
                }
            }
        }

        for &cell in &self.superblock_cells_touched {
            let value = self.superblock[cell];
            let group = cell / GRID_GROUP_CELLS;
            let width = bit_width(value);
            if self.superblock_widths[group] == 0 {
                self.superblock_groups_touched.push(group);
            }
            self.superblock_widths[group] = self.superblock_widths[group].max(width);

            let coarse_cell = cell / GRID_GROUP_CELLS;
            let coarse_slot = &mut self.coarse[coarse_cell];
            if *coarse_slot == 0 {
                self.coarse_cells_touched.push(coarse_cell);
            }
            *coarse_slot = (*coarse_slot).max(value);
        }
        for &cell in &self.coarse_cells_touched {
            let group = cell / GRID_GROUP_CELLS;
            let width = bit_width(self.coarse[cell]);
            if self.coarse_widths[group] == 0 {
                self.coarse_groups_touched.push(group);
            }
            self.coarse_widths[group] = self.coarse_widths[group].max(width);
        }
        debug_assert!(
            self.superblock_groups_touched
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        debug_assert!(
            self.coarse_groups_touched
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        Ok(())
    }
}

/// Rebuild E and H together from each source-E row.
///
/// Segment boundaries need not align to output superblocks, so E cannot
/// always be copied byte-for-byte. Both projections share two source scans:
/// one for row sizes and one for encoding. Only touched cells/groups are
/// cleared between dimensions; a sparse vocabulary therefore does not
/// repeatedly zero multi-megabyte dense rows. H rows are spooled until the
/// complete E section has been written.
fn write_merged_superblock_grids<'a>(
    sources: &[(&'a BmpIndex, u32)],
    dims: usize,
    num_blocks: usize,
    coarse_spool: &std::path::Path,
    writer: &mut dyn Write,
) -> Result<(u64, u64)> {
    use crate::segment::bmp_grid::{CompressedGridLayout, GRID_GROUP_CELLS, pack_group};

    let mut source_block_bases = Vec::with_capacity(sources.len() + 1);
    source_block_bases.push(0usize);
    for &(bmp, _) in sources {
        let next = source_block_bases
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(bmp.num_blocks as usize)
            .ok_or_else(|| {
                crate::Error::Internal("merged BMP superblock block offset overflows usize".into())
            })?;
        source_block_bases.push(next);
    }
    if source_block_bases.last().copied() != Some(num_blocks) {
        return Err(crate::Error::Internal(
            "merged BMP superblock source lengths disagree with output".into(),
        ));
    }
    let superblock_cells = num_blocks.div_ceil(crate::segment::BMP_SUPERBLOCK_SIZE as usize);
    let coarse_cells =
        superblock_cells.div_ceil(crate::segment::reader::bmp::BMP_COARSE_SUPERBLOCKS as usize);
    let superblock_layout = CompressedGridLayout::new(dims, superblock_cells);
    let coarse_layout = CompressedGridLayout::new(dims, coarse_cells);

    let mut rows = MergedSuperblockRows::new(superblock_cells, coarse_cells);
    let mut superblock_row_sizes = Vec::with_capacity(dims);
    let mut coarse_row_sizes = Vec::with_capacity(dims);
    let mut source_groups: Vec<crate::segment::bmp_grid::PackedGridGroup<'a>> = Vec::new();

    let row_size =
        |layout: CompressedGridLayout, widths: &[u8], groups_touched: &[usize]| -> Result<u64> {
            let payload_units = groups_touched.iter().try_fold(0u64, |total, &group| {
                total.checked_add(u64::from(widths[group])).ok_or_else(|| {
                    crate::Error::Internal("merged BMP projected-grid row size exceeds u64".into())
                })
            })?;
            let payload_bytes = payload_units
                .checked_mul((GRID_GROUP_CELLS / 8) as u64)
                .ok_or_else(|| {
                    crate::Error::Internal("merged BMP projected-grid row size exceeds u64".into())
                })?;
            (layout.row_header_bytes() as u64)
                .checked_add(payload_bytes)
                .ok_or_else(|| {
                    crate::Error::Internal("merged BMP projected-grid row size exceeds u64".into())
                })
        };

    for dimension in 0..dims {
        rows.prepare(
            sources,
            &source_block_bases,
            dims,
            dimension,
            &mut source_groups,
        )?;
        superblock_row_sizes.push(row_size(
            superblock_layout,
            &rows.superblock_widths,
            &rows.superblock_groups_touched,
        )?);
        coarse_row_sizes.push(row_size(
            coarse_layout,
            &rows.coarse_widths,
            &rows.coarse_groups_touched,
        )?);
    }

    let superblock_table_bytes = superblock_layout
        .write_row_offsets(&superblock_row_sizes, writer)
        .map_err(crate::Error::Io)?;
    let coarse_file = std::fs::File::create(coarse_spool).map_err(crate::Error::Io)?;
    struct ScratchCleanup<'a>(&'a std::path::Path);
    impl Drop for ScratchCleanup<'_> {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_file(self.0)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                log::warn!(
                    "[merge_bmp] failed to remove coarse-grid spool {:?}: {}",
                    self.0,
                    error
                );
            }
        }
    }
    let _coarse_cleanup = ScratchCleanup(coarse_spool);
    let mut coarse_writer = std::io::BufWriter::with_capacity(1024 * 1024, coarse_file);

    let mut values = [0u8; GRID_GROUP_CELLS];
    let mut packed = [0u8; GRID_GROUP_CELLS];
    let mut write_row = |layout: CompressedGridLayout,
                         row: &[u8],
                         widths: &[u8],
                         groups_touched: &[usize],
                         output: &mut dyn Write|
     -> Result<u64> {
        layout
            .write_row_header(
                widths,
                crate::segment::bmp_grid::LSP_SUPERBLOCK_GRID_BITS,
                output,
            )
            .map_err(crate::Error::Io)?;
        let mut bytes = layout.row_header_bytes() as u64;
        for &output_group in groups_touched {
            values.fill(0);
            let start = output_group * GRID_GROUP_CELLS;
            let count = GRID_GROUP_CELLS.min(layout.cells() - start);
            values[..count].copy_from_slice(&row[start..start + count]);
            let payload_len = pack_group(&values, widths[output_group], &mut packed)?;
            output
                .write_all(&packed[..payload_len])
                .map_err(crate::Error::Io)?;
            bytes = bytes.checked_add(payload_len as u64).ok_or_else(|| {
                crate::Error::Internal("merged BMP projected-grid row size exceeds u64".into())
            })?;
        }
        Ok(bytes)
    };

    for dimension in 0..dims {
        rows.prepare(
            sources,
            &source_block_bases,
            dims,
            dimension,
            &mut source_groups,
        )?;
        let actual_superblock = write_row(
            superblock_layout,
            &rows.superblock,
            &rows.superblock_widths,
            &rows.superblock_groups_touched,
            writer,
        )?;
        let actual_coarse = write_row(
            coarse_layout,
            &rows.coarse,
            &rows.coarse_widths,
            &rows.coarse_groups_touched,
            &mut coarse_writer,
        )?;
        if actual_superblock != superblock_row_sizes[dimension]
            || actual_coarse != coarse_row_sizes[dimension]
        {
            return Err(crate::Error::Corruption(
                "BMP projected-grid sizing changed during encoding".into(),
            ));
        }
    }
    coarse_writer.flush().map_err(crate::Error::Io)?;
    drop(coarse_writer);

    let coarse_rows_bytes = coarse_row_sizes.iter().sum::<u64>();
    let actual_coarse_bytes = std::fs::metadata(coarse_spool)
        .map_err(crate::Error::Io)?
        .len();
    if actual_coarse_bytes != coarse_rows_bytes {
        return Err(crate::Error::Corruption(format!(
            "BMP coarse-grid spool has {actual_coarse_bytes} bytes, expected {coarse_rows_bytes}"
        )));
    }
    let coarse_table_bytes = coarse_layout
        .write_row_offsets(&coarse_row_sizes, writer)
        .map_err(crate::Error::Io)?;
    let coarse_file = std::fs::File::open(coarse_spool).map_err(crate::Error::Io)?;
    let mut coarse_reader = std::io::BufReader::with_capacity(1024 * 1024, coarse_file);
    let copied = std::io::copy(&mut coarse_reader, writer).map_err(crate::Error::Io)?;
    if copied != coarse_rows_bytes {
        return Err(crate::Error::Corruption(format!(
            "short BMP coarse-grid spool copy: copied {copied}, expected {coarse_rows_bytes}"
        )));
    }

    let superblock_bytes = superblock_table_bytes
        .checked_add(superblock_row_sizes.into_iter().sum::<u64>())
        .ok_or_else(|| crate::Error::Internal("merged BMP superblock grid exceeds u64".into()))?;
    let coarse_bytes = coarse_table_bytes
        .checked_add(coarse_rows_bytes)
        .ok_or_else(|| crate::Error::Internal("merged BMP coarse grid exceeds u64".into()))?;
    Ok((superblock_bytes, coarse_bytes))
}

#[cfg(test)]
mod sparse_source_validation_tests {
    use super::*;
    use crate::directories::OwnedBytes;

    fn raw_source(offset: u64) -> DimRawData {
        let mut block = vec![0u8; SPARSE_BLOCK_HEADER_SIZE];
        block[..2].copy_from_slice(&1u16.to_le_bytes());
        block[SPARSE_FIRST_DOC_ID_OFFSET..SPARSE_FIRST_DOC_ID_OFFSET + 4]
            .copy_from_slice(&7u32.to_le_bytes());
        DimRawData {
            skip_entries: vec![SparseSkipEntry::new(7, 7, offset, 16, 1.0)],
            doc_count: 1,
            global_max_weight: 1.0,
            raw_block_data: OwnedBytes::new(block),
        }
    }

    #[test]
    fn accepts_contiguous_block() {
        validate_sparse_merge_source(3, 0, &raw_source(0)).unwrap();
    }

    #[test]
    fn rejects_bad_offset_before_slicing() {
        let error = validate_sparse_merge_source(3, 0, &raw_source(1)).unwrap_err();
        assert!(matches!(error, crate::Error::Corruption(_)));
    }
}
