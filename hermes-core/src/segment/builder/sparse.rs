//! Sparse vector streaming build (V3 footer-based format).
//!
//! Data is written first (one dim at a time), then the TOC and footer
//! are appended. Parallel sort + prune + serialize per dimension.
//!
//! Supports two formats:
//! - **MaxScore** (default): Per-dimension variable-size blocks with skip entries
//! - **BMP**: Fixed doc_id range blocks with block-max grid

use std::io::Write;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::Result;
use crate::dsl::{Field, Schema};
use crate::segment::format::{SparseDimTocEntry, SparseFieldToc, write_sparse_toc_and_footer};
use crate::structures::{
    BlockSparsePostingList, SparseFormat, SparseSkipEntry, WeightQuantization, optimal_partition,
};

use crate::DocId;

/// Builder for sparse vector index using BlockSparsePostingList
///
/// Collects (doc_id, ordinal, weight) postings per dimension, then builds
/// BlockSparsePostingList with proper quantization during commit.
pub(super) struct SparseVectorBuilder {
    /// Postings per dimension: dim_id -> Vec<(doc_id, ordinal, weight)>
    pub postings: FxHashMap<u32, Vec<(DocId, u16, f32)>>,
    /// Total number of sparse vectors added (one per index_sparse_vector_field call)
    pub total_vectors: u32,
}

impl SparseVectorBuilder {
    pub fn new() -> Self {
        Self {
            postings: FxHashMap::default(),
            total_vectors: 0,
        }
    }

    /// Record that a new sparse vector is being indexed (call once per vector, before add())
    #[inline]
    pub fn inc_vector_count(&mut self) {
        self.total_vectors += 1;
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
///
/// For BMP-format fields, a self-contained BMP blob is written and a special
/// TOC entry (num_dims=0, single BmpDescriptor) is used.
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

        let format = sparse_config.map(|c| c.format).unwrap_or_default();

        let quantization = sparse_config
            .map(|c| c.weight_quantization)
            .unwrap_or(WeightQuantization::Float32);

        let pruning_fraction = sparse_config.and_then(|c| c.pruning);
        let weight_threshold = sparse_config.map(|c| c.weight_threshold).unwrap_or(0.0);
        let min_terms = sparse_config.map(|c| c.min_terms).unwrap_or(4);
        let total_vectors = builder.total_vectors;

        match format {
            SparseFormat::Bmp => {
                let bmp_block_size = sparse_config.map(|c| c.bmp_block_size).unwrap_or(64);
                let dims = sparse_config.and_then(|c| c.dims).unwrap_or(105879); // default SPLADE vocab
                let max_weight = sparse_config.and_then(|c| c.max_weight).unwrap_or(5.0); // default SPLADE max

                let blob_offset = current_offset;
                let blob_len = super::bmp::build_bmp_blob(
                    &mut builder.postings,
                    bmp_block_size,
                    weight_threshold,
                    pruning_fraction,
                    dims,
                    max_weight,
                    min_terms,
                    writer,
                )
                .map_err(crate::Error::Io)?;

                if blob_len > 0 {
                    current_offset += blob_len;

                    // For BMP, we encode the format bit in the quant byte and use
                    // num_dims=0 with a single pseudo-dim entry containing blob location.
                    let mut config_for_byte =
                        crate::structures::SparseVectorConfig::from_byte(quantization as u8)
                            .unwrap_or_default();
                    config_for_byte.format = SparseFormat::Bmp;
                    config_for_byte.weight_quantization = quantization;

                    field_tocs.push(SparseFieldToc {
                        field_id,
                        quantization: config_for_byte.to_byte(),
                        total_vectors,
                        dims: vec![SparseDimTocEntry {
                            dim_id: 0xFFFFFFFF, // sentinel for BMP
                            block_data_offset: blob_offset,
                            skip_start: (blob_len & 0xFFFFFFFF) as u32,
                            num_blocks: ((blob_len >> 32) & 0xFFFFFFFF) as u32,
                            doc_count: 0,
                            max_weight: 0.0,
                        }],
                    });
                }
            }
            SparseFormat::MaxScore => {
                build_maxscore_field(
                    builder,
                    field_id,
                    quantization,
                    pruning_fraction,
                    weight_threshold,
                    min_terms,
                    total_vectors,
                    writer,
                    &mut current_offset,
                    &mut all_skip_entries,
                    &mut field_tocs,
                )?;
            }
        }
    }

    if field_tocs.is_empty() {
        return Ok(());
    }

    // Phase 2: Write skip section (only MaxScore fields have skip entries)
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

/// Build a MaxScore-format field (existing logic extracted for clarity).
#[allow(clippy::too_many_arguments)]
fn build_maxscore_field(
    builder: &mut SparseVectorBuilder,
    field_id: u32,
    quantization: WeightQuantization,
    pruning_fraction: Option<f32>,
    weight_threshold: f32,
    min_terms: usize,
    total_vectors: u32,
    writer: &mut dyn Write,
    current_offset: &mut u64,
    all_skip_entries: &mut Vec<SparseSkipEntry>,
    field_tocs: &mut Vec<SparseFieldToc>,
) -> Result<()> {
    // Parallel: sort + prune + serialize each dimension independently
    let mut dims: Vec<_> = std::mem::take(&mut builder.postings).into_iter().collect();
    dims.sort_unstable_by_key(|(id, _)| *id);

    let serialized_dims: Vec<(u32, u32, Vec<u8>, Vec<SparseSkipEntry>)> = dims
        .into_par_iter()
        .map(|(dim_id, mut postings)| {
            // Filter by weight threshold (same as BMP path)
            // Skip filtering when the dimension has fewer than min_terms postings
            if weight_threshold > 0.0 && postings.len() >= min_terms {
                postings.retain(|(_, _, w)| w.abs() >= weight_threshold);
            }
            if postings.is_empty() {
                return Ok((dim_id, 0, Vec::new(), Vec::new()));
            }

            let pruned = if let Some(fraction) = pruning_fraction
                && postings.len() >= min_terms
                && fraction < 1.0
            {
                let original_len = postings.len();
                postings.sort_unstable_by(|a, b| {
                    b.2.abs()
                        .partial_cmp(&a.2.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let keep = ((original_len as f64 * fraction as f64).ceil() as usize).max(1);
                postings.truncate(keep);
                true
            } else {
                false
            };
            // Postings arrive in (doc_id, ordinal) order from sequential indexing;
            // only re-sort if pruning destroyed that order.
            if pruned {
                postings.sort_unstable_by_key(|(doc_id, ordinal, _)| (*doc_id, *ordinal));
            }

            let weights: Vec<f32> = postings.iter().map(|(_, _, w)| w.abs()).collect();
            let partition = optimal_partition(&weights);
            let block_list = BlockSparsePostingList::from_postings_with_partition(
                &postings,
                quantization,
                &partition,
            )
            .map_err(crate::Error::Io)?;

            let doc_count = block_list.doc_count;
            let (block_data, skip_entries) = block_list.serialize().map_err(crate::Error::Io)?;

            #[cfg(feature = "diagnostics")]
            super::diagnostics::validate_serialized_blocks(dim_id, &block_data, &skip_entries)?;

            Ok((dim_id, doc_count, block_data, skip_entries))
        })
        .collect::<Result<Vec<_>>>()?;

    // Phase 1: Write block data sequentially, accumulate skip entries
    let mut dim_toc_entries: Vec<SparseDimTocEntry> = Vec::with_capacity(serialized_dims.len());
    for (dim_id, doc_count, block_data, skip_entries) in &serialized_dims {
        if block_data.is_empty() {
            continue; // dim eliminated by weight_threshold / pruning
        }
        let block_data_offset = *current_offset;
        let skip_start = all_skip_entries.len() as u32;
        let num_blocks = skip_entries.len() as u32;
        let max_weight = skip_entries
            .iter()
            .map(|e| e.max_weight)
            .fold(0.0f32, f32::max);

        writer.write_all(block_data)?;
        *current_offset += block_data.len() as u64;

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
            total_vectors,
            dims: dim_toc_entries,
        });
    }

    Ok(())
}
