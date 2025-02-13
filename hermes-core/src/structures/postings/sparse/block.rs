//! Block-based sparse posting list with 3 sub-blocks
//!
//! Format per block (128 entries for SIMD alignment):
//! - Doc IDs: delta-encoded, bit-packed
//! - Ordinals: bit-packed small integers (lazy decode)
//! - Weights: quantized (f32/f16/u8/u4)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor, Read, Write};

use super::config::WeightQuantization;
use crate::DocId;
use crate::directories::OwnedBytes;
use crate::structures::postings::TERMINATED;
use crate::structures::simd;

pub const BLOCK_SIZE: usize = 128;
pub const MAX_BLOCK_SIZE: usize = 256;

#[derive(Debug, Clone, Copy)]
pub struct BlockHeader {
    pub count: u16,
    pub doc_id_bits: u8,
    pub ordinal_bits: u8,
    pub weight_quant: WeightQuantization,
    pub first_doc_id: DocId,
    pub max_weight: f32,
}

impl BlockHeader {
    pub const SIZE: usize = 16;

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u16::<LittleEndian>(self.count)?;
        w.write_u8(self.doc_id_bits)?;
        w.write_u8(self.ordinal_bits)?;
        w.write_u8(self.weight_quant as u8)?;
        w.write_u8(0)?;
        w.write_u16::<LittleEndian>(0)?;
        w.write_u32::<LittleEndian>(self.first_doc_id)?;
        w.write_f32::<LittleEndian>(self.max_weight)?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let count = r.read_u16::<LittleEndian>()?;
        let doc_id_bits = r.read_u8()?;
        let ordinal_bits = r.read_u8()?;
        let weight_quant_byte = r.read_u8()?;
        let _ = r.read_u8()?;
        let _ = r.read_u16::<LittleEndian>()?;
        let first_doc_id = r.read_u32::<LittleEndian>()?;
        let max_weight = r.read_f32::<LittleEndian>()?;

        let weight_quant = WeightQuantization::from_u8(weight_quant_byte)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid weight quant"))?;

        Ok(Self {
            count,
            doc_id_bits,
            ordinal_bits,
            weight_quant,
            first_doc_id,
            max_weight,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SparseBlock {
    pub header: BlockHeader,
    /// Delta-encoded, bit-packed doc IDs (zero-copy from mmap when loaded lazily)
    pub doc_ids_data: OwnedBytes,
    /// Bit-packed ordinals (zero-copy from mmap when loaded lazily)
    pub ordinals_data: OwnedBytes,
    /// Quantized weights (zero-copy from mmap when loaded lazily)
    pub weights_data: OwnedBytes,
}

impl SparseBlock {
    pub fn from_postings(
        postings: &[(DocId, u16, f32)],
        weight_quant: WeightQuantization,
    ) -> io::Result<Self> {
        assert!(!postings.is_empty() && postings.len() <= MAX_BLOCK_SIZE);

        let count = postings.len();
        let first_doc_id = postings[0].0;

        // Delta encode doc IDs
        let mut deltas = Vec::with_capacity(count);
        let mut prev = first_doc_id;
        for &(doc_id, _, _) in postings {
            deltas.push(doc_id.saturating_sub(prev));
            prev = doc_id;
        }
        deltas[0] = 0;

        let doc_id_bits = simd::round_bit_width(find_optimal_bit_width(&deltas[1..]));
        let ordinals: Vec<u16> = postings.iter().map(|(_, o, _)| *o).collect();
        let max_ordinal = ordinals.iter().copied().max().unwrap_or(0);
        let ordinal_bits = if max_ordinal == 0 {
            0
        } else {
            simd::round_bit_width(bits_needed_u16(max_ordinal))
        };

        let weights: Vec<f32> = postings.iter().map(|(_, _, w)| *w).collect();
        let max_weight = weights
            .iter()
            .copied()
            .fold(0.0f32, |acc, w| acc.max(w.abs()));

        let doc_ids_data = OwnedBytes::new({
            let rounded = simd::RoundedBitWidth::from_u8(doc_id_bits);
            let num_deltas = count - 1;
            let byte_count = num_deltas * rounded.bytes_per_value();
            let mut data = vec![0u8; byte_count];
            simd::pack_rounded(&deltas[1..], rounded, &mut data);
            data
        });
        let ordinals_data = OwnedBytes::new(if ordinal_bits > 0 {
            let rounded = simd::RoundedBitWidth::from_u8(ordinal_bits);
            let byte_count = count * rounded.bytes_per_value();
            let mut data = vec![0u8; byte_count];
            let ord_u32: Vec<u32> = ordinals.iter().map(|&o| o as u32).collect();
            simd::pack_rounded(&ord_u32, rounded, &mut data);
            data
        } else {
            Vec::new()
        });
        let weights_data = OwnedBytes::new(encode_weights(&weights, weight_quant)?);

        Ok(Self {
            header: BlockHeader {
                count: count as u16,
                doc_id_bits,
                ordinal_bits,
                weight_quant,
                first_doc_id,
                max_weight,
            },
            doc_ids_data,
            ordinals_data,
            weights_data,
        })
    }

    pub fn decode_doc_ids(&self) -> Vec<DocId> {
        let mut out = Vec::with_capacity(self.header.count as usize);
        self.decode_doc_ids_into(&mut out);
        out
    }

    /// Decode doc IDs into an existing Vec (avoids allocation on reuse).
    ///
    /// Uses SIMD-accelerated unpacking for rounded bit widths (0, 8, 16, 32).
    pub fn decode_doc_ids_into(&self, out: &mut Vec<DocId>) {
        let count = self.header.count as usize;
        out.clear();
        out.resize(count, 0);
        out[0] = self.header.first_doc_id;

        if count > 1 {
            let bits = self.header.doc_id_bits;
            if bits == 0 {
                // All deltas are 0 (multi-value same doc_id repeats)
                out[1..].fill(self.header.first_doc_id);
            } else {
                // SIMD-accelerated unpack (bits is always 8, 16, or 32)
                simd::unpack_rounded(
                    &self.doc_ids_data,
                    simd::RoundedBitWidth::from_u8(bits),
                    &mut out[1..],
                    count - 1,
                );
                // In-place prefix sum (pure delta, NOT gap-1)
                for i in 1..count {
                    out[i] += out[i - 1];
                }
            }
        }
    }

    pub fn decode_ordinals(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(self.header.count as usize);
        self.decode_ordinals_into(&mut out);
        out
    }

    /// Decode ordinals into an existing Vec (avoids allocation on reuse).
    ///
    /// Uses SIMD-accelerated unpacking for rounded bit widths (0, 8, 16, 32).
    pub fn decode_ordinals_into(&self, out: &mut Vec<u16>) {
        let count = self.header.count as usize;
        out.clear();
        if self.header.ordinal_bits == 0 {
            out.resize(count, 0u16);
        } else {
            // SIMD-accelerated unpack (bits is always 8, 16, or 32)
            let mut temp = [0u32; BLOCK_SIZE];
            simd::unpack_rounded(
                &self.ordinals_data,
                simd::RoundedBitWidth::from_u8(self.header.ordinal_bits),
                &mut temp[..count],
                count,
            );
            out.reserve(count);
            for &v in &temp[..count] {
                out.push(v as u16);
            }
        }
    }

    pub fn decode_weights(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.header.count as usize);
        self.decode_weights_into(&mut out);
        out
    }

    /// Decode weights into an existing Vec (avoids allocation on reuse).
    pub fn decode_weights_into(&self, out: &mut Vec<f32>) {
        out.clear();
        decode_weights_into(
            &self.weights_data,
            self.header.weight_quant,
            self.header.count as usize,
            out,
        );
    }

    /// Decode weights pre-multiplied by `query_weight` directly from quantized data.
    ///
    /// For UInt8: computes `(qw * scale) * q + (qw * min)` via SIMD — avoids
    /// allocating an intermediate f32 dequantized buffer. The effective_scale and
    /// effective_bias are computed once per block (not per element).
    ///
    /// For F32/F16/UInt4: falls back to decode + scalar multiply.
    pub fn decode_scored_weights_into(&self, query_weight: f32, out: &mut Vec<f32>) {
        out.clear();
        let count = self.header.count as usize;
        match self.header.weight_quant {
            WeightQuantization::UInt8 if self.weights_data.len() >= 8 => {
                // UInt8 layout: [scale: f32][min: f32][q0, q1, ..., q_{n-1}]
                let scale = f32::from_le_bytes([
                    self.weights_data[0],
                    self.weights_data[1],
                    self.weights_data[2],
                    self.weights_data[3],
                ]);
                let min_val = f32::from_le_bytes([
                    self.weights_data[4],
                    self.weights_data[5],
                    self.weights_data[6],
                    self.weights_data[7],
                ]);
                // Fused: qw * (q * scale + min) = q * (qw * scale) + (qw * min)
                let eff_scale = query_weight * scale;
                let eff_bias = query_weight * min_val;
                out.resize(count, 0.0);
                simd::dequantize_uint8(&self.weights_data[8..], out, eff_scale, eff_bias, count);
            }
            _ => {
                // Fallback: decode to f32, then multiply
                decode_weights_into(&self.weights_data, self.header.weight_quant, count, out);
                for w in out.iter_mut() {
                    *w *= query_weight;
                }
            }
        }
    }

    /// Fused decode + multiply + scatter-accumulate into flat_scores array.
    ///
    /// Equivalent to:
    ///   decode_scored_weights_into(qw, &mut weights_buf);
    ///   for i in 0..count { flat_scores[doc_ids[i] - base] += weights_buf[i]; }
    ///
    /// But avoids allocating/filling weights_buf — decodes directly into flat_scores.
    /// Tracks dirty entries (first touch) for efficient collection.
    ///
    /// `doc_ids` must already be decoded via `decode_doc_ids_into`.
    /// Returns the number of postings accumulated.
    #[inline]
    pub fn accumulate_scored_weights(
        &self,
        query_weight: f32,
        doc_ids: &[u32],
        flat_scores: &mut [f32],
        base_doc: u32,
        dirty: &mut Vec<u32>,
    ) -> usize {
        let count = self.header.count as usize;
        match self.header.weight_quant {
            WeightQuantization::UInt8 if self.weights_data.len() >= 8 => {
                // UInt8 layout: [scale: f32][min: f32][q0, q1, ..., q_{n-1}]
                let scale = f32::from_le_bytes([
                    self.weights_data[0],
                    self.weights_data[1],
                    self.weights_data[2],
                    self.weights_data[3],
                ]);
                let min_val = f32::from_le_bytes([
                    self.weights_data[4],
                    self.weights_data[5],
                    self.weights_data[6],
                    self.weights_data[7],
                ]);
                let eff_scale = query_weight * scale;
                let eff_bias = query_weight * min_val;
                let quant_data = &self.weights_data[8..];

                for i in 0..count.min(quant_data.len()) {
                    let w = quant_data[i] as f32 * eff_scale + eff_bias;
                    let off = (doc_ids[i] - base_doc) as usize;
                    if flat_scores[off] == 0.0 {
                        dirty.push(doc_ids[i]);
                    }
                    flat_scores[off] += w;
                }
                count
            }
            _ => {
                // Fallback: decode to temp buffer, then scatter
                let mut weights_buf = Vec::with_capacity(count);
                decode_weights_into(
                    &self.weights_data,
                    self.header.weight_quant,
                    count,
                    &mut weights_buf,
                );
                for i in 0..count {
                    let w = weights_buf[i] * query_weight;
                    let off = (doc_ids[i] - base_doc) as usize;
                    if flat_scores[off] == 0.0 {
                        dirty.push(doc_ids[i]);
                    }
                    flat_scores[off] += w;
                }
                count
            }
        }
    }

    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.header.write(w)?;
        w.write_u16::<LittleEndian>(self.doc_ids_data.len() as u16)?;
        w.write_u16::<LittleEndian>(self.ordinals_data.len() as u16)?;
        w.write_u16::<LittleEndian>(self.weights_data.len() as u16)?;
        w.write_u16::<LittleEndian>(0)?;
        w.write_all(&self.doc_ids_data)?;
        w.write_all(&self.ordinals_data)?;
        w.write_all(&self.weights_data)?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let header = BlockHeader::read(r)?;
        let doc_ids_len = r.read_u16::<LittleEndian>()? as usize;
        let ordinals_len = r.read_u16::<LittleEndian>()? as usize;
        let weights_len = r.read_u16::<LittleEndian>()? as usize;
        let _ = r.read_u16::<LittleEndian>()?;

        let mut doc_ids_vec = vec![0u8; doc_ids_len];
        r.read_exact(&mut doc_ids_vec)?;
        let mut ordinals_vec = vec![0u8; ordinals_len];
        r.read_exact(&mut ordinals_vec)?;
        let mut weights_vec = vec![0u8; weights_len];
        r.read_exact(&mut weights_vec)?;

        Ok(Self {
            header,
            doc_ids_data: OwnedBytes::new(doc_ids_vec),
            ordinals_data: OwnedBytes::new(ordinals_vec),
            weights_data: OwnedBytes::new(weights_vec),
        })
    }

    /// Zero-copy constructor from OwnedBytes (mmap-backed).
    ///
    /// Parses the block header and sub-block length prefix, then slices the
    /// OwnedBytes into doc_ids/ordinals/weights without any heap allocation.
    /// Sub-slices share the underlying mmap Arc — no data is copied.
    pub fn from_owned_bytes(data: crate::directories::OwnedBytes) -> crate::Result<Self> {
        let b = data.as_slice();
        if b.len() < BlockHeader::SIZE + 8 {
            return Err(crate::Error::Corruption(
                "sparse block too small".to_string(),
            ));
        }
        let mut cursor = Cursor::new(&b[..BlockHeader::SIZE]);
        let header =
            BlockHeader::read(&mut cursor).map_err(|e| crate::Error::Corruption(e.to_string()))?;

        if header.count == 0 {
            let hex: String = b
                .iter()
                .take(32)
                .map(|x| format!("{x:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            return Err(crate::Error::Corruption(format!(
                "sparse block has count=0 (data_len={}, first_32_bytes=[{}])",
                b.len(),
                hex
            )));
        }

        let p = BlockHeader::SIZE;
        let doc_ids_len = u16::from_le_bytes([b[p], b[p + 1]]) as usize;
        let ordinals_len = u16::from_le_bytes([b[p + 2], b[p + 3]]) as usize;
        let weights_len = u16::from_le_bytes([b[p + 4], b[p + 5]]) as usize;
        // p+6..p+8 is padding

        let data_start = p + 8;
        let ord_start = data_start + doc_ids_len;
        let wt_start = ord_start + ordinals_len;
        let expected_end = wt_start + weights_len;

        if expected_end > b.len() {
            let hex: String = b
                .iter()
                .take(32)
                .map(|x| format!("{x:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            return Err(crate::Error::Corruption(format!(
                "sparse block sub-block overflow: count={} doc_ids={}B ords={}B wts={}B need={}B have={}B (first_32=[{}])",
                header.count,
                doc_ids_len,
                ordinals_len,
                weights_len,
                expected_end,
                b.len(),
                hex
            )));
        }

        Ok(Self {
            header,
            doc_ids_data: data.slice(data_start..ord_start),
            ordinals_data: data.slice(ord_start..wt_start),
            weights_data: data.slice(wt_start..wt_start + weights_len),
        })
    }

    /// Create a copy of this block with first_doc_id adjusted by offset.
    ///
    /// This is used during merge to remap doc_ids from different segments.
    /// Only the first_doc_id needs adjustment - deltas within the block
    /// remain unchanged since they're relative to the previous doc.
    pub fn with_doc_offset(&self, doc_offset: u32) -> Self {
        Self {
            header: BlockHeader {
                first_doc_id: self.header.first_doc_id + doc_offset,
                ..self.header
            },
            doc_ids_data: self.doc_ids_data.clone(),
            ordinals_data: self.ordinals_data.clone(),
            weights_data: self.weights_data.clone(),
        }
    }
}

// ============================================================================
// BlockSparsePostingList
// ============================================================================

#[derive(Debug, Clone)]
pub struct BlockSparsePostingList {
    pub doc_count: u32,
    pub blocks: Vec<SparseBlock>,
}

impl BlockSparsePostingList {
    /// Create from postings with configurable block size
    pub fn from_postings_with_block_size(
        postings: &[(DocId, u16, f32)],
        weight_quant: WeightQuantization,
        block_size: usize,
    ) -> io::Result<Self> {
        if postings.is_empty() {
            return Ok(Self {
                doc_count: 0,
                blocks: Vec::new(),
            });
        }

        let block_size = block_size.max(16); // minimum 16 for sanity
        let mut blocks = Vec::new();
        for chunk in postings.chunks(block_size) {
            blocks.push(SparseBlock::from_postings(chunk, weight_quant)?);
        }

        // Count unique document IDs (not total postings).
        // For multi-value fields, the same doc_id appears multiple times
        // with different ordinals. Postings are sorted by (doc_id, ordinal),
        // so we count transitions.
        let mut unique_docs = 1u32;
        for i in 1..postings.len() {
            if postings[i].0 != postings[i - 1].0 {
                unique_docs += 1;
            }
        }

        Ok(Self {
            doc_count: unique_docs,
            blocks,
        })
    }

    /// Create from postings with default block size (128)
    pub fn from_postings(
        postings: &[(DocId, u16, f32)],
        weight_quant: WeightQuantization,
    ) -> io::Result<Self> {
        Self::from_postings_with_block_size(postings, weight_quant, BLOCK_SIZE)
    }

    /// Create from postings using a pre-computed variable-size partition plan.
    ///
    /// `partition` is a slice of block sizes (e.g., [64, 128, 32, ...]) whose
    /// sum must equal `postings.len()`. Each block size must be ≤ MAX_BLOCK_SIZE.
    /// Produced by `optimal_partition()`.
    pub fn from_postings_with_partition(
        postings: &[(DocId, u16, f32)],
        weight_quant: WeightQuantization,
        partition: &[usize],
    ) -> io::Result<Self> {
        if postings.is_empty() {
            return Ok(Self {
                doc_count: 0,
                blocks: Vec::new(),
            });
        }

        let mut blocks = Vec::with_capacity(partition.len());
        let mut offset = 0;
        for &block_size in partition {
            let end = (offset + block_size).min(postings.len());
            blocks.push(SparseBlock::from_postings(
                &postings[offset..end],
                weight_quant,
            )?);
            offset = end;
        }

        let mut unique_docs = 1u32;
        for i in 1..postings.len() {
            if postings[i].0 != postings[i - 1].0 {
                unique_docs += 1;
            }
        }

        Ok(Self {
            doc_count: unique_docs,
            blocks,
        })
    }

    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn global_max_weight(&self) -> f32 {
        self.blocks
            .iter()
            .map(|b| b.header.max_weight)
            .fold(0.0f32, f32::max)
    }

    pub fn block_max_weight(&self, block_idx: usize) -> Option<f32> {
        self.blocks.get(block_idx).map(|b| b.header.max_weight)
    }

    /// Approximate memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        use std::mem::size_of;

        let header_size = size_of::<u32>() * 2; // doc_count + num_blocks
        let blocks_size: usize = self
            .blocks
            .iter()
            .map(|b| {
                size_of::<BlockHeader>()
                    + b.doc_ids_data.len()
                    + b.ordinals_data.len()
                    + b.weights_data.len()
            })
            .sum();
        header_size + blocks_size
    }

    pub fn iterator(&self) -> BlockSparsePostingIterator<'_> {
        BlockSparsePostingIterator::new(self)
    }

    /// Serialize: returns (block_data, skip_entries) separately.
    ///
    /// Block data and skip entries are written to different file sections.
    /// The caller writes block data first, accumulates skip entries, then
    /// writes all skip entries in a contiguous section at the file tail.
    pub fn serialize(&self) -> io::Result<(Vec<u8>, Vec<super::SparseSkipEntry>)> {
        // Serialize all blocks to get their sizes
        let mut block_data = Vec::new();
        let mut skip_entries = Vec::with_capacity(self.blocks.len());
        let mut offset = 0u64;

        for block in &self.blocks {
            let mut buf = Vec::new();
            block.write(&mut buf)?;
            let length = buf.len() as u32;

            let first_doc = block.header.first_doc_id;
            let doc_ids = block.decode_doc_ids();
            let last_doc = doc_ids.last().copied().unwrap_or(first_doc);

            skip_entries.push(super::SparseSkipEntry::new(
                first_doc,
                last_doc,
                offset,
                length,
                block.header.max_weight,
            ));

            block_data.extend_from_slice(&buf);
            offset += length as u64;
        }

        Ok((block_data, skip_entries))
    }

    /// Reconstruct from V3 serialized parts (block_data + skip_entries).
    ///
    /// Parses each block from the raw data using skip entry offsets.
    /// Used for testing roundtrips; production uses lazy block loading.
    #[cfg(test)]
    pub fn from_parts(
        doc_count: u32,
        block_data: &[u8],
        skip_entries: &[super::SparseSkipEntry],
    ) -> io::Result<Self> {
        let mut blocks = Vec::with_capacity(skip_entries.len());
        for entry in skip_entries {
            let start = entry.offset as usize;
            let end = start + entry.length as usize;
            blocks.push(SparseBlock::read(&mut std::io::Cursor::new(
                &block_data[start..end],
            ))?);
        }
        Ok(Self { doc_count, blocks })
    }

    pub fn decode_all(&self) -> Vec<(DocId, u16, f32)> {
        let total_postings: usize = self.blocks.iter().map(|b| b.header.count as usize).sum();
        let mut result = Vec::with_capacity(total_postings);
        for block in &self.blocks {
            let doc_ids = block.decode_doc_ids();
            let ordinals = block.decode_ordinals();
            let weights = block.decode_weights();
            for i in 0..block.header.count as usize {
                result.push((doc_ids[i], ordinals[i], weights[i]));
            }
        }
        result
    }

    /// Merge multiple posting lists from different segments with doc_id offsets.
    ///
    /// This is an optimized O(1) merge that stacks blocks without decode/re-encode.
    /// Each posting list's blocks have their first_doc_id adjusted by the corresponding offset.
    ///
    /// # Arguments
    /// * `lists` - Slice of (posting_list, doc_offset) pairs from each segment
    ///
    /// # Returns
    /// A new posting list with all blocks concatenated and doc_ids remapped
    pub fn merge_with_offsets(lists: &[(&BlockSparsePostingList, u32)]) -> Self {
        if lists.is_empty() {
            return Self {
                doc_count: 0,
                blocks: Vec::new(),
            };
        }

        // Pre-calculate total capacity
        let total_blocks: usize = lists.iter().map(|(pl, _)| pl.blocks.len()).sum();
        let total_docs: u32 = lists.iter().map(|(pl, _)| pl.doc_count).sum();

        let mut merged_blocks = Vec::with_capacity(total_blocks);

        // Stack blocks from each segment with doc_id offset adjustment
        for (posting_list, doc_offset) in lists {
            for block in &posting_list.blocks {
                merged_blocks.push(block.with_doc_offset(*doc_offset));
            }
        }

        Self {
            doc_count: total_docs,
            blocks: merged_blocks,
        }
    }

    fn find_block(&self, target: DocId) -> Option<usize> {
        if self.blocks.is_empty() {
            return None;
        }
        // Binary search on first_doc_id: find the last block whose first_doc_id <= target.
        // O(log N) header comparisons — no block decode needed.
        let idx = self
            .blocks
            .partition_point(|b| b.header.first_doc_id <= target);
        if idx == 0 {
            // target < first_doc_id of block 0 — return block 0 so caller can check
            Some(0)
        } else {
            Some(idx - 1)
        }
    }
}

// ============================================================================
// Iterator
// ============================================================================

pub struct BlockSparsePostingIterator<'a> {
    posting_list: &'a BlockSparsePostingList,
    block_idx: usize,
    in_block_idx: usize,
    current_doc_ids: Vec<DocId>,
    current_ordinals: Vec<u16>,
    current_weights: Vec<f32>,
    /// Whether ordinals have been decoded for current block (lazy decode)
    ordinals_decoded: bool,
    exhausted: bool,
}

impl<'a> BlockSparsePostingIterator<'a> {
    fn new(posting_list: &'a BlockSparsePostingList) -> Self {
        let mut iter = Self {
            posting_list,
            block_idx: 0,
            in_block_idx: 0,
            current_doc_ids: Vec::with_capacity(128),
            current_ordinals: Vec::with_capacity(128),
            current_weights: Vec::with_capacity(128),
            ordinals_decoded: false,
            exhausted: posting_list.blocks.is_empty(),
        };
        if !iter.exhausted {
            iter.load_block(0);
        }
        iter
    }

    fn load_block(&mut self, block_idx: usize) {
        if let Some(block) = self.posting_list.blocks.get(block_idx) {
            block.decode_doc_ids_into(&mut self.current_doc_ids);
            block.decode_weights_into(&mut self.current_weights);
            // Defer ordinal decode until ordinal() is called (lazy)
            self.ordinals_decoded = false;
            self.block_idx = block_idx;
            self.in_block_idx = 0;
        }
    }

    /// Ensure ordinals are decoded for the current block (lazy decode)
    #[inline]
    fn ensure_ordinals_decoded(&mut self) {
        if !self.ordinals_decoded {
            if let Some(block) = self.posting_list.blocks.get(self.block_idx) {
                block.decode_ordinals_into(&mut self.current_ordinals);
            }
            self.ordinals_decoded = true;
        }
    }

    #[inline]
    pub fn doc(&self) -> DocId {
        if self.exhausted {
            TERMINATED
        } else {
            // Safety: load_block guarantees in_block_idx < current_doc_ids.len()
            self.current_doc_ids[self.in_block_idx]
        }
    }

    #[inline]
    pub fn weight(&self) -> f32 {
        if self.exhausted {
            return 0.0;
        }
        // Safety: load_block guarantees in_block_idx < current_weights.len()
        self.current_weights[self.in_block_idx]
    }

    #[inline]
    pub fn ordinal(&mut self) -> u16 {
        if self.exhausted {
            return 0;
        }
        self.ensure_ordinals_decoded();
        self.current_ordinals[self.in_block_idx]
    }

    pub fn advance(&mut self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        self.in_block_idx += 1;
        if self.in_block_idx >= self.current_doc_ids.len() {
            self.block_idx += 1;
            if self.block_idx >= self.posting_list.blocks.len() {
                self.exhausted = true;
            } else {
                self.load_block(self.block_idx);
            }
        }
        self.doc()
    }

    pub fn seek(&mut self, target: DocId) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        if self.doc() >= target {
            return self.doc();
        }

        // Check current block — binary search within decoded doc_ids
        if let Some(&last_doc) = self.current_doc_ids.last()
            && last_doc >= target
        {
            let remaining = &self.current_doc_ids[self.in_block_idx..];
            let pos = crate::structures::simd::find_first_ge_u32(remaining, target);
            self.in_block_idx += pos;
            if self.in_block_idx >= self.current_doc_ids.len() {
                self.block_idx += 1;
                if self.block_idx >= self.posting_list.blocks.len() {
                    self.exhausted = true;
                } else {
                    self.load_block(self.block_idx);
                }
            }
            return self.doc();
        }

        // Find correct block
        if let Some(block_idx) = self.posting_list.find_block(target) {
            self.load_block(block_idx);
            let pos = crate::structures::simd::find_first_ge_u32(&self.current_doc_ids, target);
            self.in_block_idx = pos;
            if self.in_block_idx >= self.current_doc_ids.len() {
                self.block_idx += 1;
                if self.block_idx >= self.posting_list.blocks.len() {
                    self.exhausted = true;
                } else {
                    self.load_block(self.block_idx);
                }
            }
        } else {
            self.exhausted = true;
        }
        self.doc()
    }

    /// Skip to the start of the next block, returning its first doc_id.
    /// Used by block-max pruning to skip entire blocks that can't beat threshold.
    pub fn skip_to_next_block(&mut self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        let next = self.block_idx + 1;
        if next >= self.posting_list.blocks.len() {
            self.exhausted = true;
            return TERMINATED;
        }
        self.load_block(next);
        self.doc()
    }

    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    pub fn current_block_max_weight(&self) -> f32 {
        self.posting_list
            .blocks
            .get(self.block_idx)
            .map(|b| b.header.max_weight)
            .unwrap_or(0.0)
    }

    pub fn current_block_max_contribution(&self, query_weight: f32) -> f32 {
        query_weight * self.current_block_max_weight()
    }
}

// ============================================================================
// Bit-packing utilities
// ============================================================================

fn find_optimal_bit_width(values: &[u32]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let max_val = values.iter().copied().max().unwrap_or(0);
    simd::bits_needed(max_val)
}

fn bits_needed_u16(val: u16) -> u8 {
    if val == 0 {
        0
    } else {
        16 - val.leading_zeros() as u8
    }
}

// ============================================================================
// Weight encoding/decoding
// ============================================================================

fn encode_weights(weights: &[f32], quant: WeightQuantization) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    match quant {
        WeightQuantization::Float32 => {
            for &w in weights {
                data.write_f32::<LittleEndian>(w)?;
            }
        }
        WeightQuantization::Float16 => {
            use half::f16;
            for &w in weights {
                data.write_u16::<LittleEndian>(f16::from_f32(w).to_bits())?;
            }
        }
        WeightQuantization::UInt8 => {
            let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            let scale = if range < f32::EPSILON {
                1.0
            } else {
                range / 255.0
            };
            data.write_f32::<LittleEndian>(scale)?;
            data.write_f32::<LittleEndian>(min)?;
            for &w in weights {
                data.write_u8(((w - min) / scale).round() as u8)?;
            }
        }
        WeightQuantization::UInt4 => {
            let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
            let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            let scale = if range < f32::EPSILON {
                1.0
            } else {
                range / 15.0
            };
            data.write_f32::<LittleEndian>(scale)?;
            data.write_f32::<LittleEndian>(min)?;
            let mut i = 0;
            while i < weights.len() {
                let q1 = ((weights[i] - min) / scale).round() as u8 & 0x0F;
                let q2 = if i + 1 < weights.len() {
                    ((weights[i + 1] - min) / scale).round() as u8 & 0x0F
                } else {
                    0
                };
                data.write_u8((q2 << 4) | q1)?;
                i += 2;
            }
        }
    }
    Ok(data)
}

fn decode_weights_into(data: &[u8], quant: WeightQuantization, count: usize, out: &mut Vec<f32>) {
    let mut cursor = Cursor::new(data);
    match quant {
        WeightQuantization::Float32 => {
            for _ in 0..count {
                out.push(cursor.read_f32::<LittleEndian>().unwrap_or(0.0));
            }
        }
        WeightQuantization::Float16 => {
            use half::f16;
            for _ in 0..count {
                let bits = cursor.read_u16::<LittleEndian>().unwrap_or(0);
                out.push(f16::from_bits(bits).to_f32());
            }
        }
        WeightQuantization::UInt8 => {
            let scale = cursor.read_f32::<LittleEndian>().unwrap_or(1.0);
            let min_val = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            let offset = cursor.position() as usize;
            out.resize(count, 0.0);
            simd::dequantize_uint8(&data[offset..], out, scale, min_val, count);
        }
        WeightQuantization::UInt4 => {
            let scale = cursor.read_f32::<LittleEndian>().unwrap_or(1.0);
            let min = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            let mut i = 0;
            while i < count {
                let byte = cursor.read_u8().unwrap_or(0);
                out.push((byte & 0x0F) as f32 * scale + min);
                i += 1;
                if i < count {
                    out.push((byte >> 4) as f32 * scale + min);
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_roundtrip() {
        let postings = vec![
            (10u32, 0u16, 1.5f32),
            (15, 0, 2.0),
            (20, 1, 0.5),
            (100, 0, 3.0),
        ];
        let block = SparseBlock::from_postings(&postings, WeightQuantization::Float32).unwrap();

        assert_eq!(block.decode_doc_ids(), vec![10, 15, 20, 100]);
        assert_eq!(block.decode_ordinals(), vec![0, 0, 1, 0]);
        let weights = block.decode_weights();
        assert!((weights[0] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_posting_list() {
        let postings: Vec<(DocId, u16, f32)> =
            (0..300).map(|i| (i * 2, 0, i as f32 * 0.1)).collect();
        let list =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        assert_eq!(list.doc_count(), 300);
        assert_eq!(list.num_blocks(), 3);

        let mut iter = list.iterator();
        assert_eq!(iter.doc(), 0);
        iter.advance();
        assert_eq!(iter.doc(), 2);
    }

    #[test]
    fn test_serialization() {
        let postings = vec![(1u32, 0u16, 0.5f32), (10, 1, 1.5), (100, 0, 2.5)];
        let list =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::UInt8).unwrap();

        let (block_data, skip_entries) = list.serialize().unwrap();
        let list2 =
            BlockSparsePostingList::from_parts(list.doc_count(), &block_data, &skip_entries)
                .unwrap();

        assert_eq!(list.doc_count(), list2.doc_count());
    }

    #[test]
    fn test_seek() {
        let postings: Vec<(DocId, u16, f32)> = (0..500).map(|i| (i * 3, 0, i as f32)).collect();
        let list =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        let mut iter = list.iterator();
        assert_eq!(iter.seek(300), 300);
        assert_eq!(iter.seek(301), 303);
        assert_eq!(iter.seek(2000), TERMINATED);
    }

    #[test]
    fn test_merge_with_offsets() {
        // Segment 1: docs 0, 5, 10 with weights
        let postings1: Vec<(DocId, u16, f32)> = vec![(0, 0, 1.0), (5, 0, 2.0), (10, 1, 3.0)];
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();

        // Segment 2: docs 0, 3, 7 with weights (will become 100, 103, 107 after merge)
        let postings2: Vec<(DocId, u16, f32)> = vec![(0, 0, 4.0), (3, 1, 5.0), (7, 0, 6.0)];
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with offsets: segment 1 at offset 0, segment 2 at offset 100
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 100)]);

        assert_eq!(merged.doc_count(), 6);

        // Verify all doc_ids are correct after merge
        let decoded = merged.decode_all();
        assert_eq!(decoded.len(), 6);

        // Segment 1 docs (offset 0)
        assert_eq!(decoded[0].0, 0);
        assert_eq!(decoded[1].0, 5);
        assert_eq!(decoded[2].0, 10);

        // Segment 2 docs (offset 100)
        assert_eq!(decoded[3].0, 100); // 0 + 100
        assert_eq!(decoded[4].0, 103); // 3 + 100
        assert_eq!(decoded[5].0, 107); // 7 + 100

        // Verify weights preserved
        assert!((decoded[0].2 - 1.0).abs() < 0.01);
        assert!((decoded[3].2 - 4.0).abs() < 0.01);

        // Verify ordinals preserved
        assert_eq!(decoded[2].1, 1); // ordinal from segment 1
        assert_eq!(decoded[4].1, 1); // ordinal from segment 2
    }

    #[test]
    fn test_merge_with_offsets_multi_block() {
        // Create posting lists that span multiple blocks
        let postings1: Vec<(DocId, u16, f32)> = (0..200).map(|i| (i * 2, 0, i as f32)).collect();
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();
        assert!(list1.num_blocks() > 1, "Should have multiple blocks");

        let postings2: Vec<(DocId, u16, f32)> = (0..150).map(|i| (i * 3, 1, i as f32)).collect();
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with offset 1000 for segment 2
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 1000)]);

        assert_eq!(merged.doc_count(), 350);
        assert_eq!(merged.num_blocks(), list1.num_blocks() + list2.num_blocks());

        // Verify via iterator
        let mut iter = merged.iterator();

        // First segment docs start at 0
        assert_eq!(iter.doc(), 0);

        // Seek to segment 2 (should be at offset 1000)
        let doc = iter.seek(1000);
        assert_eq!(doc, 1000); // First doc of segment 2: 0 + 1000 = 1000

        // Next doc in segment 2
        iter.advance();
        assert_eq!(iter.doc(), 1003); // 3 + 1000 = 1003
    }

    #[test]
    fn test_merge_with_offsets_serialize_roundtrip() {
        // Verify that serialization preserves adjusted doc_ids
        let postings1: Vec<(DocId, u16, f32)> = vec![(0, 0, 1.0), (5, 0, 2.0), (10, 1, 3.0)];
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();

        let postings2: Vec<(DocId, u16, f32)> = vec![(0, 0, 4.0), (3, 1, 5.0), (7, 0, 6.0)];
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with offset 100 for segment 2
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 100)]);

        // Serialize + reconstruct
        let (block_data, skip_entries) = merged.serialize().unwrap();
        let loaded =
            BlockSparsePostingList::from_parts(merged.doc_count(), &block_data, &skip_entries)
                .unwrap();

        // Verify doc_ids are preserved after round-trip
        let decoded = loaded.decode_all();
        assert_eq!(decoded.len(), 6);

        // Segment 1 docs (offset 0)
        assert_eq!(decoded[0].0, 0);
        assert_eq!(decoded[1].0, 5);
        assert_eq!(decoded[2].0, 10);

        // Segment 2 docs (offset 100) - CRITICAL: these must be offset-adjusted
        assert_eq!(decoded[3].0, 100, "First doc of seg2 should be 0+100=100");
        assert_eq!(decoded[4].0, 103, "Second doc of seg2 should be 3+100=103");
        assert_eq!(decoded[5].0, 107, "Third doc of seg2 should be 7+100=107");

        // Verify iterator also works correctly
        let mut iter = loaded.iterator();
        assert_eq!(iter.doc(), 0);
        iter.advance();
        assert_eq!(iter.doc(), 5);
        iter.advance();
        assert_eq!(iter.doc(), 10);
        iter.advance();
        assert_eq!(iter.doc(), 100);
        iter.advance();
        assert_eq!(iter.doc(), 103);
        iter.advance();
        assert_eq!(iter.doc(), 107);
    }

    #[test]
    fn test_merge_seek_after_roundtrip() {
        // Create posting lists that span multiple blocks to test seek after merge
        let postings1: Vec<(DocId, u16, f32)> = (0..200).map(|i| (i * 2, 0, 1.0)).collect();
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();

        let postings2: Vec<(DocId, u16, f32)> = (0..150).map(|i| (i * 3, 0, 2.0)).collect();
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with offset 1000 for segment 2
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 1000)]);

        // Serialize + reconstruct
        let (block_data, skip_entries) = merged.serialize().unwrap();
        let loaded =
            BlockSparsePostingList::from_parts(merged.doc_count(), &block_data, &skip_entries)
                .unwrap();

        // Test seeking to various positions
        let mut iter = loaded.iterator();

        // Seek to doc in segment 1
        let doc = iter.seek(100);
        assert_eq!(doc, 100, "Seek to 100 in segment 1");

        // Seek to doc in segment 2 (1000 + offset)
        let doc = iter.seek(1000);
        assert_eq!(doc, 1000, "Seek to 1000 (first doc of segment 2)");

        // Seek to middle of segment 2
        let doc = iter.seek(1050);
        assert!(
            doc >= 1050,
            "Seek to 1050 should find doc >= 1050, got {}",
            doc
        );

        // Seek backwards should stay at current position (seek only goes forward)
        let doc = iter.seek(500);
        assert!(
            doc >= 1050,
            "Seek backwards should not go back, got {}",
            doc
        );

        // Fresh iterator - verify block boundaries work
        let mut iter2 = loaded.iterator();

        // Verify we can iterate through all docs
        let mut count = 0;
        let mut prev_doc = 0;
        while iter2.doc() != super::TERMINATED {
            let current = iter2.doc();
            if count > 0 {
                assert!(
                    current > prev_doc,
                    "Docs should be monotonically increasing: {} vs {}",
                    prev_doc,
                    current
                );
            }
            prev_doc = current;
            iter2.advance();
            count += 1;
        }
        assert_eq!(count, 350, "Should have 350 total docs");
    }

    #[test]
    fn test_doc_count_multi_value() {
        // Multi-value: same doc_id with different ordinals
        // doc 0 has 3 ordinals, doc 5 has 2, doc 10 has 1 = 3 unique docs
        let postings: Vec<(DocId, u16, f32)> = vec![
            (0, 0, 1.0),
            (0, 1, 1.5),
            (0, 2, 2.0),
            (5, 0, 3.0),
            (5, 1, 3.5),
            (10, 0, 4.0),
        ];
        let list =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        // doc_count should be 3 (unique docs), not 6 (total postings)
        assert_eq!(list.doc_count(), 3);

        // But we should still have all 6 postings accessible
        let decoded = list.decode_all();
        assert_eq!(decoded.len(), 6);
    }

    /// Test the zero-copy merge path used by the actual sparse merger:
    /// serialize → get raw skip entries + block data → patch first_doc_id → reassemble.
    /// This mirrors the code path in `segment/merger/sparse.rs`.
    #[test]
    fn test_zero_copy_merge_patches_first_doc_id() {
        use crate::structures::SparseSkipEntry;

        // Build two multi-block posting lists
        let postings1: Vec<(DocId, u16, f32)> = (0..200).map(|i| (i * 2, 0, i as f32)).collect();
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();
        assert!(list1.num_blocks() > 1);

        let postings2: Vec<(DocId, u16, f32)> = (0..150).map(|i| (i * 3, 1, i as f32)).collect();
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Serialize both using V3 format (block_data + skip_entries)
        let (raw1, skip1) = list1.serialize().unwrap();
        let (raw2, skip2) = list2.serialize().unwrap();

        // --- Simulate the merger's zero-copy reassembly ---
        let doc_offset: u32 = 1000; // segment 2 starts at doc 1000
        let total_docs = list1.doc_count() + list2.doc_count();

        // Accumulate adjusted skip entries
        let mut merged_skip = Vec::new();
        let mut cumulative_offset = 0u64;
        for entry in &skip1 {
            merged_skip.push(SparseSkipEntry::new(
                entry.first_doc,
                entry.last_doc,
                cumulative_offset + entry.offset,
                entry.length,
                entry.max_weight,
            ));
        }
        if let Some(last) = skip1.last() {
            cumulative_offset += last.offset + last.length as u64;
        }
        for entry in &skip2 {
            merged_skip.push(SparseSkipEntry::new(
                entry.first_doc + doc_offset,
                entry.last_doc + doc_offset,
                cumulative_offset + entry.offset,
                entry.length,
                entry.max_weight,
            ));
        }

        // Concatenate raw block data: source 1 verbatim, source 2 with first_doc_id patched
        let mut merged_block_data = Vec::new();
        merged_block_data.extend_from_slice(&raw1);

        const FIRST_DOC_ID_OFFSET: usize = 8;
        let mut buf2 = raw2.to_vec();
        for entry in &skip2 {
            let off = entry.offset as usize + FIRST_DOC_ID_OFFSET;
            if off + 4 <= buf2.len() {
                let old = u32::from_le_bytes(buf2[off..off + 4].try_into().unwrap());
                let patched = (old + doc_offset).to_le_bytes();
                buf2[off..off + 4].copy_from_slice(&patched);
            }
        }
        merged_block_data.extend_from_slice(&buf2);

        // --- Reconstruct and verify ---
        let loaded =
            BlockSparsePostingList::from_parts(total_docs, &merged_block_data, &merged_skip)
                .unwrap();
        assert_eq!(loaded.doc_count(), 350);

        let mut iter = loaded.iterator();

        // Segment 1: docs 0, 2, 4, ..., 398
        assert_eq!(iter.doc(), 0);
        let doc = iter.seek(100);
        assert_eq!(doc, 100);
        let doc = iter.seek(398);
        assert_eq!(doc, 398);

        // Segment 2: docs 1000, 1003, 1006, ..., 1000 + 149*3 = 1447
        let doc = iter.seek(1000);
        assert_eq!(doc, 1000, "First doc of segment 2 should be 1000");
        iter.advance();
        assert_eq!(iter.doc(), 1003, "Second doc of segment 2 should be 1003");
        let doc = iter.seek(1447);
        assert_eq!(doc, 1447, "Last doc of segment 2 should be 1447");

        // Exhausted
        iter.advance();
        assert_eq!(iter.doc(), super::TERMINATED);

        // Also verify with merge_with_offsets to confirm identical results
        let reference =
            BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, doc_offset)]);
        let mut ref_iter = reference.iterator();
        let mut zc_iter = loaded.iterator();
        while ref_iter.doc() != super::TERMINATED {
            assert_eq!(
                ref_iter.doc(),
                zc_iter.doc(),
                "Zero-copy and reference merge should produce identical doc_ids"
            );
            assert!(
                (ref_iter.weight() - zc_iter.weight()).abs() < 0.01,
                "Weights should match: {} vs {}",
                ref_iter.weight(),
                zc_iter.weight()
            );
            ref_iter.advance();
            zc_iter.advance();
        }
        assert_eq!(zc_iter.doc(), super::TERMINATED);
    }

    #[test]
    fn test_doc_count_single_value() {
        // Single-value: each doc_id appears once (ordinal always 0)
        let postings: Vec<(DocId, u16, f32)> =
            vec![(0, 0, 1.0), (5, 0, 2.0), (10, 0, 3.0), (15, 0, 4.0)];
        let list =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        // doc_count == total postings for single-value
        assert_eq!(list.doc_count(), 4);
    }

    #[test]
    fn test_doc_count_multi_value_serialization_roundtrip() {
        // Verify doc_count survives serialization
        let postings: Vec<(DocId, u16, f32)> =
            vec![(0, 0, 1.0), (0, 1, 1.5), (5, 0, 2.0), (5, 1, 2.5)];
        let list =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();
        assert_eq!(list.doc_count(), 2);

        let (block_data, skip_entries) = list.serialize().unwrap();
        let loaded =
            BlockSparsePostingList::from_parts(list.doc_count(), &block_data, &skip_entries)
                .unwrap();
        assert_eq!(loaded.doc_count(), 2);
    }

    #[test]
    fn test_merge_preserves_weights_and_ordinals() {
        // Test that weights and ordinals are preserved after merge + roundtrip
        let postings1: Vec<(DocId, u16, f32)> = vec![(0, 0, 1.5), (5, 1, 2.5), (10, 2, 3.5)];
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();

        let postings2: Vec<(DocId, u16, f32)> = vec![(0, 0, 4.5), (3, 1, 5.5), (7, 3, 6.5)];
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with offset 100 for segment 2
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 100)]);

        // Serialize + reconstruct
        let (block_data, skip_entries) = merged.serialize().unwrap();
        let loaded =
            BlockSparsePostingList::from_parts(merged.doc_count(), &block_data, &skip_entries)
                .unwrap();

        // Verify all postings via iterator
        let mut iter = loaded.iterator();

        // Segment 1 postings
        assert_eq!(iter.doc(), 0);
        assert!(
            (iter.weight() - 1.5).abs() < 0.01,
            "Weight should be 1.5, got {}",
            iter.weight()
        );
        assert_eq!(iter.ordinal(), 0);

        iter.advance();
        assert_eq!(iter.doc(), 5);
        assert!(
            (iter.weight() - 2.5).abs() < 0.01,
            "Weight should be 2.5, got {}",
            iter.weight()
        );
        assert_eq!(iter.ordinal(), 1);

        iter.advance();
        assert_eq!(iter.doc(), 10);
        assert!(
            (iter.weight() - 3.5).abs() < 0.01,
            "Weight should be 3.5, got {}",
            iter.weight()
        );
        assert_eq!(iter.ordinal(), 2);

        // Segment 2 postings (with offset 100)
        iter.advance();
        assert_eq!(iter.doc(), 100);
        assert!(
            (iter.weight() - 4.5).abs() < 0.01,
            "Weight should be 4.5, got {}",
            iter.weight()
        );
        assert_eq!(iter.ordinal(), 0);

        iter.advance();
        assert_eq!(iter.doc(), 103);
        assert!(
            (iter.weight() - 5.5).abs() < 0.01,
            "Weight should be 5.5, got {}",
            iter.weight()
        );
        assert_eq!(iter.ordinal(), 1);

        iter.advance();
        assert_eq!(iter.doc(), 107);
        assert!(
            (iter.weight() - 6.5).abs() < 0.01,
            "Weight should be 6.5, got {}",
            iter.weight()
        );
        assert_eq!(iter.ordinal(), 3);

        // Verify exhausted
        iter.advance();
        assert_eq!(iter.doc(), super::TERMINATED);
    }

    #[test]
    fn test_merge_global_max_weight() {
        // Verify global_max_weight is correct after merge
        let postings1: Vec<(DocId, u16, f32)> = vec![
            (0, 0, 3.0),
            (1, 0, 7.0), // max in segment 1
            (2, 0, 2.0),
        ];
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();

        let postings2: Vec<(DocId, u16, f32)> = vec![
            (0, 0, 5.0),
            (1, 0, 4.0),
            (2, 0, 6.0), // max in segment 2
        ];
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Verify original global max weights
        assert!((list1.global_max_weight() - 7.0).abs() < 0.01);
        assert!((list2.global_max_weight() - 6.0).abs() < 0.01);

        // Merge
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 100)]);

        // Global max should be 7.0 (from segment 1)
        assert!(
            (merged.global_max_weight() - 7.0).abs() < 0.01,
            "Global max should be 7.0, got {}",
            merged.global_max_weight()
        );

        // Roundtrip
        let (block_data, skip_entries) = merged.serialize().unwrap();
        let loaded =
            BlockSparsePostingList::from_parts(merged.doc_count(), &block_data, &skip_entries)
                .unwrap();

        assert!(
            (loaded.global_max_weight() - 7.0).abs() < 0.01,
            "After roundtrip, global max should still be 7.0, got {}",
            loaded.global_max_weight()
        );
    }

    #[test]
    fn test_scoring_simulation_after_merge() {
        // Simulate scoring: compute query_weight * stored_weight
        let postings1: Vec<(DocId, u16, f32)> = vec![
            (0, 0, 0.5), // doc 0, weight 0.5
            (5, 0, 0.8), // doc 5, weight 0.8
        ];
        let list1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();

        let postings2: Vec<(DocId, u16, f32)> = vec![
            (0, 0, 0.6), // doc 100 after offset, weight 0.6
            (3, 0, 0.9), // doc 103 after offset, weight 0.9
        ];
        let list2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with offset 100
        let merged = BlockSparsePostingList::merge_with_offsets(&[(&list1, 0), (&list2, 100)]);

        // Roundtrip
        let (block_data, skip_entries) = merged.serialize().unwrap();
        let loaded =
            BlockSparsePostingList::from_parts(merged.doc_count(), &block_data, &skip_entries)
                .unwrap();

        // Simulate scoring with query_weight = 2.0
        let query_weight = 2.0f32;
        let mut iter = loaded.iterator();

        // Expected scores: query_weight * stored_weight
        // Doc 0: 2.0 * 0.5 = 1.0
        assert_eq!(iter.doc(), 0);
        let score = query_weight * iter.weight();
        assert!(
            (score - 1.0).abs() < 0.01,
            "Doc 0 score should be 1.0, got {}",
            score
        );

        iter.advance();
        // Doc 5: 2.0 * 0.8 = 1.6
        assert_eq!(iter.doc(), 5);
        let score = query_weight * iter.weight();
        assert!(
            (score - 1.6).abs() < 0.01,
            "Doc 5 score should be 1.6, got {}",
            score
        );

        iter.advance();
        // Doc 100: 2.0 * 0.6 = 1.2
        assert_eq!(iter.doc(), 100);
        let score = query_weight * iter.weight();
        assert!(
            (score - 1.2).abs() < 0.01,
            "Doc 100 score should be 1.2, got {}",
            score
        );

        iter.advance();
        // Doc 103: 2.0 * 0.9 = 1.8
        assert_eq!(iter.doc(), 103);
        let score = query_weight * iter.weight();
        assert!(
            (score - 1.8).abs() < 0.01,
            "Doc 103 score should be 1.8, got {}",
            score
        );
    }
}
