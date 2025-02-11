//! Sparse vector streaming build (V3 footer-based format).
//!
//! Data is written first (one dim at a time), then the TOC and footer
//! are appended. Parallel sort + prune + serialize per dimension.

use std::io::Write;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::Result;
use crate::dsl::{Field, Schema};
use crate::segment::format::{SparseDimTocEntry, SparseFieldToc, write_sparse_toc_and_footer};
use crate::structures::{BlockSparsePostingList, SparseSkipEntry, WeightQuantization};

use crate::DocId;

/// Builder for sparse vector index using BlockSparsePostingList
///
/// Collects (doc_id, ordinal, weight) postings per dimension, then builds
/// BlockSparsePostingList with proper quantization during commit.
pub(super) struct SparseVectorBuilder {
    /// Postings per dimension: dim_id -> Vec<(doc_id, ordinal, weight)>
    pub postings: FxHashMap<u32, Vec<(DocId, u16, f32)>>,
}

impl SparseVectorBuilder {
    pub fn new() -> Self {
        Self {
            postings: FxHashMap::default(),
        }
    }

    /// Add a sparse vector entry with ordinal tracking
    #[inline]
    pub fn add(&mut self, dim_id: u32, doc_id: DocId, ordinal: u16, weight: f32) {
        self.postings
            .entry(dim_id)
            .or_default()
            .push((doc_id, ordinal, weight));
    }

    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
}

/// Stream sparse vectors directly to disk (footer-based format).
///
/// Data is written first (one dim at a time), then the TOC and footer
/// are appended. This matches the dense vectors format pattern.
pub(super) fn build_sparse_streaming(
    sparse_vectors: &mut FxHashMap<u32, SparseVectorBuilder>,
    schema: &Schema,
    writer: &mut dyn Write,
) -> Result<()> {
    if sparse_vectors.is_empty() {
        return Ok(());
    }

    // Collect and sort fields
    let mut field_ids: Vec<u32> = sparse_vectors.keys().copied().collect();
    field_ids.sort_unstable();

    let mut field_tocs: Vec<SparseFieldToc> = Vec::new();
    let mut all_skip_entries: Vec<SparseSkipEntry> = Vec::new();
    let mut current_offset = 0u64;

    for &field_id in &field_ids {
        let builder = sparse_vectors.get_mut(&field_id).unwrap();
        if builder.is_empty() {
            continue;
        }

        let field = Field(field_id);
        let sparse_config = schema
            .get_field_entry(field)
            .and_then(|e| e.sparse_vector_config.as_ref());

        let quantization = sparse_config
            .map(|c| c.weight_quantization)
            .unwrap_or(WeightQuantization::Float32);

        let block_size = sparse_config.map(|c| c.block_size).unwrap_or(128);
        let pruning_fraction = sparse_config.and_then(|c| c.posting_list_pruning);

        // Parallel: sort + prune + serialize_v3 each dimension independently
        let mut dims: Vec<_> = std::mem::take(&mut builder.postings).into_iter().collect();
        dims.sort_unstable_by_key(|(id, _)| *id);

        let serialized_dims: Vec<(u32, u32, Vec<u8>, Vec<SparseSkipEntry>)> = dims
            .into_par_iter()
            .map(|(dim_id, mut postings)| {
                postings.sort_unstable_by_key(|(doc_id, ordinal, _)| (*doc_id, *ordinal));

                if let Some(fraction) = pruning_fraction
                    && postings.len() > 1
                    && fraction < 1.0
                {
                    let original_len = postings.len();
                    postings.sort_by(|a, b| {
                        b.2.abs()
                            .partial_cmp(&a.2.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let keep = ((original_len as f64 * fraction as f64).ceil() as usize).max(1);
                    postings.truncate(keep);
                    postings.sort_unstable_by_key(|(d, o, _)| (*d, *o));
                }

                let block_list = BlockSparsePostingList::from_postings_with_block_size(
                    &postings,
                    quantization,
                    block_size,
                )
                .map_err(crate::Error::Io)?;

                let doc_count = block_list.doc_count;
                let (block_data, skip_entries) =
                    block_list.serialize().map_err(crate::Error::Io)?;

                #[cfg(feature = "diagnostics")]
                super::diagnostics::validate_serialized_blocks(dim_id, &block_data, &skip_entries)?;

                Ok((dim_id, doc_count, block_data, skip_entries))
            })
            .collect::<Result<Vec<_>>>()?;

        // Phase 1: Write block data sequentially, accumulate skip entries
        let mut dim_toc_entries: Vec<SparseDimTocEntry> = Vec::with_capacity(serialized_dims.len());
        for (dim_id, doc_count, block_data, skip_entries) in &serialized_dims {
            let block_data_offset = current_offset;
            let skip_start = all_skip_entries.len() as u32;
            let num_blocks = skip_entries.len() as u32;
            let max_weight = skip_entries
                .iter()
                .map(|e| e.max_weight)
                .fold(0.0f32, f32::max);

            writer.write_all(block_data)?;
            current_offset += block_data.len() as u64;

            all_skip_entries.extend_from_slice(skip_entries);

            dim_toc_entries.push(SparseDimTocEntry {
                dim_id: *dim_id,
                block_data_offset,
                skip_start,
                num_blocks,
                doc_count: *doc_count,
                max_weight,
            });
        }

        if !dim_toc_entries.is_empty() {
            field_tocs.push(SparseFieldToc {
                field_id,
                quantization: quantization as u8,
                dims: dim_toc_entries,
            });
        }
    }

    if field_tocs.is_empty() {
        return Ok(());
    }

    // Phase 2: Write skip section
    let skip_offset = current_offset;
    for entry in &all_skip_entries {
        entry.write(writer).map_err(crate::Error::Io)?;
    }
    current_offset += (all_skip_entries.len() * SparseSkipEntry::SIZE) as u64;

    // Phase 3+4: Write TOC + footer
    let toc_offset = current_offset;
    write_sparse_toc_and_footer(writer, skip_offset, toc_offset, &field_tocs)
        .map_err(crate::Error::Io)?;

    Ok(())
}
