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
                if segments
                    .iter()
                    .any(|segment| segment.sparse_indexes().contains_key(&field.0))
                {
                    return Err(crate::Error::Corruption(format!(
                        "field {} is configured as BMP but a source segment contains MaxScore data; rebuild the index",
                        field.0
                    )));
                }
                if has_bmp_data {
                    let total_vectors_bmp: u32 = bmp_indexes
                        .iter()
                        .filter_map(|bi| bi.map(|idx| idx.total_vectors))
                        .try_fold(0u32, |total, vectors| total.checked_add(vectors))
                        .ok_or_else(|| {
                            crate::Error::Internal(
                                "merged BMP vector count exceeds the V17 u32 format limit".into(),
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
                        validate_bmp_sources(
                            &sources,
                            dims,
                            bmp_block_size.clamp(1, 256),
                            grid_bits,
                            max_weight_scale,
                        )?;

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
                        let _reorder_permit = match &self.reorder_permits {
                            Some(permits) => {
                                Some(Arc::clone(permits).acquire_owned().await.map_err(|_| {
                                    crate::Error::Internal(
                                        "background reorder scheduler is closed".into(),
                                    )
                                })?)
                            }
                            None => None,
                        };
                        let (w, ft, converged) = tokio::task::spawn_blocking(move || {
                            crate::segment::reorder::reorder_bmp_field(
                                &sources,
                                fid,
                                &ilabel,
                                &fname,
                                quantization,
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
                                writer,
                                field_tocs,
                                pool,
                            )
                        })
                        .await
                        .map_err(|e| {
                            crate::Error::Internal(format!("merge-time reorder panicked: {}", e))
                        })??;
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
                            quantization,
                            dims,
                            bmp_block_size,
                            grid_bits,
                            max_weight_scale,
                            total_vectors_bmp,
                            &mut writer,
                            &mut field_tocs,
                        )
                    })?;
                }
                continue;
            }

            if segments
                .iter()
                .any(|segment| segment.bmp_indexes().contains_key(&field.0))
            {
                return Err(crate::Error::Corruption(format!(
                    "field {} is configured as MaxScore but a source segment contains BMP data; rebuild the index",
                    field.0
                )));
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

        // Phase 2: Stream-copy skip entries from temp file to main writer (4 MB chunks)
        let skip_offset = writer.offset();
        let skip_size = skip_count as u64 * SparseSkipEntry::SIZE as u64;
        const SKIP_COPY_CHUNK: u64 = 4 * 1024 * 1024;
        {
            let mut pos = 0u64;
            while pos < skip_size {
                let end = (pos + SKIP_COPY_CHUNK).min(skip_size);
                let chunk = dir.read_range(&skip_tmp, pos..end).await?;
                writer
                    .write_all(chunk.as_slice())
                    .map_err(crate::Error::Io)?;
                pos = end;
            }
        }
        dir.delete(&skip_tmp).await.map_err(crate::Error::Io)?;

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

/// Merge BMP fields with **streaming block-copy V17 format**.
///
/// V17 block-copy merge: all segments share the same `dims`, `bmp_block_size`,
/// and `max_weight_scale`, so blocks are self-contained and can be copied directly.
///
/// Phases:
/// 1. Stream Section B (block data) — sequential chunked copy
/// 2. Write padding + Section A (block_data_starts) — recomputed on-the-fly
/// 3. Stream Section D — payload groups already aligned in output coordinates
///    are copied byte-for-byte; shifted groups use a bounded decode/repack
/// 4. Stream Section E — ceil-u4 values decoded/repacked into the output
///    superblock coordinate system
/// 5. Stream Section F+G (doc_map) — bulk copy with offset patching
/// 6. Write the V17 footer
///
/// Peak memory is one row's group descriptors/selectors, one superblock row,
/// and fixed-size copy/decode buffers; it does not scale with postings.
#[allow(clippy::too_many_arguments)]
fn merge_bmp_field(
    bmp_indexes: &[Option<&BmpIndex>],
    doc_offs: &[u32],
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    dims: u32,
    bmp_block_size: u32,
    grid_bits: u8,
    max_weight_scale: f32,
    total_vectors: u32,
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

    // ── Phase 0: Validate all sources share dims, block_size, max_weight_scale ──
    let mut total_source_blocks: u32 = 0;
    let mut num_real_docs_total: u32 = 0;

    for &(bmp, _) in &sources {
        if bmp.dims() != dims {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source dims={} != expected dims={}",
                bmp.dims(),
                dims
            )));
        }
        if bmp.bmp_block_size != effective_block_size {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source block_size={} != expected {}",
                bmp.bmp_block_size, effective_block_size
            )));
        }
        if bmp.grid_bits() != grid_bits {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source grid_bits={} != expected {}",
                bmp.grid_bits(),
                grid_bits
            )));
        }
        if (bmp.max_weight_scale - max_weight_scale).abs() > f32::EPSILON {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source max_weight_scale={:.4} != expected {:.4}",
                bmp.max_weight_scale, max_weight_scale
            )));
        }
        total_source_blocks = total_source_blocks
            .checked_add(bmp.num_blocks)
            .ok_or_else(|| {
                crate::Error::Internal(
                    "merged BMP block count exceeds the V17 u32 format limit".into(),
                )
            })?;
        num_real_docs_total = num_real_docs_total
            .checked_add(bmp.num_real_docs())
            .ok_or_else(|| {
                crate::Error::Internal(
                    "merged BMP real-document count exceeds the V17 u32 format limit".into(),
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
                "merged BMP virtual-document count exceeds the V17 u32 format limit".into(),
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
        "[merge_bmp_v17] field {}: dims={}, {} sources, {} total_blocks, \
         block_size={}, max_weight_scale={:.4}",
        field_id,
        dims,
        sources.len(),
        num_blocks,
        effective_block_size,
        max_weight_scale,
    );

    // Restores query advice and drops scan-faulted mmap pages on every exit.
    let _source_pages =
        crate::segment::reader::bmp::BmpScanPageGuard::new(sources.iter().map(|(bmp, _)| *bmp));

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

    // ── Phase 4: Stream Section E. Source superblock coordinates restart at
    // every segment, so this ceil-u4 section is remapped by
    // source block offsets and repacked.
    let sb_grid_offset = writer.offset() - blob_start;
    debug_assert_eq!(sb_grid_offset, grid_offset + block_grid_bytes);
    write_merged_superblock_grid(&sources, dims as usize, num_blocks, writer)?;

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

    // ── Phase 6: Write V17 footer ───────────────────────────────────────
    write_bmp_footer(
        writer,
        total_terms,
        total_postings,
        grid_offset,
        sb_grid_offset,
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
    push_bmp_field_toc(
        field_tocs,
        field_id,
        quantization,
        total_vectors,
        blob_start,
        blob_len,
    );

    Ok(())
}

/// Validate that all BMP merge sources share `dims`, `block_size`, and
/// `max_weight_scale` (same invariants as the block-copy path's Phase 0).
fn validate_bmp_sources(
    sources: &[(BmpIndex, u32)],
    dims: u32,
    effective_block_size: u32,
    expected_grid_bits: u8,
    max_weight_scale: f32,
) -> Result<()> {
    for (bmp, _) in sources {
        if bmp.dims() != dims {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source dims={} != expected dims={}",
                bmp.dims(),
                dims
            )));
        }
        if bmp.bmp_block_size != effective_block_size {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source block_size={} != expected {}",
                bmp.bmp_block_size, effective_block_size
            )));
        }
        if bmp.grid_bits() != expected_grid_bits {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source grid_bits={} != expected {}",
                bmp.grid_bits(),
                expected_grid_bits
            )));
        }
        if (bmp.max_weight_scale - max_weight_scale).abs() > f32::EPSILON {
            return Err(crate::Error::Corruption(format!(
                "BMP merge: source max_weight_scale={:.4} != expected {:.4}",
                bmp.max_weight_scale, max_weight_scale
            )));
        }
    }
    Ok(())
}

/// Push a BMP sentinel field TOC entry after writing a blob.
fn push_bmp_field_toc(
    field_tocs: &mut Vec<SparseFieldToc>,
    field_id: u32,
    quantization: crate::structures::WeightQuantization,
    total_vectors: u32,
    blob_start: u64,
    blob_len: u64,
) {
    if blob_len == 0 {
        return;
    }
    let mut config_for_byte =
        crate::structures::SparseVectorConfig::from_byte(quantization as u8).unwrap_or_default();
    config_for_byte.format = SparseFormat::Bmp;
    config_for_byte.weight_quantization = quantization;

    field_tocs.push(SparseFieldToc {
        field_id,
        quantization: config_for_byte.to_byte(),
        total_vectors,
        dims: vec![SparseDimTocEntry {
            dim_id: 0xFFFFFFFF, // sentinel for BMP
            block_data_offset: blob_start,
            skip_start: (blob_len & 0xFFFFFFFF) as u32,
            num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
            doc_count: 0,
            max_weight: 0.0,
        }],
    });
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
        widths.fill(0);
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

    for dimension in 0..dims {
        load_row(dimension, &mut row_groups)?;
        collect_widths(&mut widths, &row_groups, &mut group_values)?;
        layout
            .write_row_header(&widths, grid_bits, writer)
            .map_err(crate::Error::Io)?;

        let mut packed = [0u8; GRID_GROUP_CELLS];
        let mut source = 0usize;
        for (output_group, &width) in widths.iter().enumerate() {
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
                    if group.width() != width {
                        return Err(crate::Error::Corruption(
                            "BMP merge block-grid sizing changed during encoding".into(),
                        ));
                    }
                    writer.write_all(group.bytes()).map_err(crate::Error::Io)?;
                }
                MergedBlockGroup::Repacked {
                    width: actual_width,
                } => {
                    if actual_width != width {
                        return Err(crate::Error::Corruption(
                            "BMP merge block-grid sizing changed during encoding".into(),
                        ));
                    }
                    let payload_len = pack_group(&group_values, width, &mut packed)?;
                    writer
                        .write_all(&packed[..payload_len])
                        .map_err(crate::Error::Io)?;
                }
            }
        }
    }
    Ok(table_bytes + row_sizes.into_iter().sum::<u64>())
}

fn write_merged_superblock_grid<'a>(
    sources: &[(&'a BmpIndex, u32)],
    dims: usize,
    num_blocks: usize,
    writer: &mut dyn Write,
) -> Result<u64> {
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
    let cells = num_blocks.div_ceil(crate::segment::BMP_SUPERBLOCK_SIZE as usize);
    let layout = CompressedGridLayout::new(dims, cells);
    let mut widths = vec![0u8; layout.groups()];
    let mut row_sizes = Vec::with_capacity(dims);
    let mut row = vec![0u8; cells];
    let mut decoded = [0u8; GRID_GROUP_CELLS];
    let mut source_groups: Vec<crate::segment::bmp_grid::PackedGridGroup<'a>> = Vec::new();

    let fill_row = |dimension: usize,
                    row: &mut [u8],
                    decoded: &mut [u8; GRID_GROUP_CELLS],
                    source_groups: &mut Vec<crate::segment::bmp_grid::PackedGridGroup<'a>>|
     -> Result<()> {
        const SB: usize = crate::segment::BMP_SUPERBLOCK_SIZE as usize;
        row.fill(0);
        for (source, &(bmp, _)) in sources.iter().enumerate() {
            let grid = bmp.superblock_grid();
            let source_blocks = bmp.num_blocks as usize;
            source_groups.clear();
            grid.append_row_groups(dimension, source_groups)?;
            for (source_group, group) in source_groups.iter().copied().enumerate() {
                let source_cell_start = source_group * GRID_GROUP_CELLS;
                let count = GRID_GROUP_CELLS.min(grid.cells() - source_cell_start);
                if group.width() == 0 {
                    continue;
                }
                group.decode(0, count, decoded);
                for (within, &value) in decoded[..count].iter().enumerate() {
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
                    for slot in &mut row[first_output_sb..=last_output_sb] {
                        *slot = (*slot).max(value);
                    }
                }
            }
        }
        Ok(())
    };
    let collect_widths = |row: &[u8], widths: &mut [u8]| {
        widths.fill(0);
        for (group, values) in row.chunks(GRID_GROUP_CELLS).enumerate() {
            widths[group] =
                crate::segment::bmp_grid::bit_width(values.iter().copied().max().unwrap_or(0));
        }
    };

    for dimension in 0..dims {
        fill_row(dimension, &mut row, &mut decoded, &mut source_groups)?;
        collect_widths(&row, &mut widths);
        row_sizes.push(layout.row_bytes(&widths)?);
    }
    let table_bytes = layout
        .write_row_offsets(&row_sizes, writer)
        .map_err(crate::Error::Io)?;

    let mut values = [0u8; GRID_GROUP_CELLS];
    let mut packed = [0u8; GRID_GROUP_CELLS];
    for dimension in 0..dims {
        fill_row(dimension, &mut row, &mut decoded, &mut source_groups)?;
        collect_widths(&row, &mut widths);
        layout
            .write_row_header(
                &widths,
                crate::segment::bmp_grid::LSP_SUPERBLOCK_GRID_BITS,
                writer,
            )
            .map_err(crate::Error::Io)?;
        for (output_group, &width) in widths.iter().enumerate() {
            values.fill(0);
            let start = output_group * GRID_GROUP_CELLS;
            let count = GRID_GROUP_CELLS.min(cells - start);
            values[..count].copy_from_slice(&row[start..start + count]);
            let payload_len = pack_group(&values, width, &mut packed)?;
            writer
                .write_all(&packed[..payload_len])
                .map_err(crate::Error::Io)?;
        }
    }
    Ok(table_bytes + row_sizes.into_iter().sum::<u64>())
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
