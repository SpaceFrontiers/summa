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
use crate::structures::postings::TERMINATED;
use crate::structures::simd;

pub const BLOCK_SIZE: usize = 128;

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
    pub doc_ids_data: Vec<u8>,
    pub ordinals_data: Vec<u8>,
    pub weights_data: Vec<u8>,
}

impl SparseBlock {
    pub fn from_postings(
        postings: &[(DocId, u16, f32)],
        weight_quant: WeightQuantization,
    ) -> io::Result<Self> {
        assert!(!postings.is_empty() && postings.len() <= BLOCK_SIZE);

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

        let doc_id_bits = find_optimal_bit_width(&deltas[1..]);
        let ordinals: Vec<u16> = postings.iter().map(|(_, o, _)| *o).collect();
        let max_ordinal = ordinals.iter().copied().max().unwrap_or(0);
        let ordinal_bits = if max_ordinal == 0 {
            0
        } else {
            bits_needed_u16(max_ordinal)
        };

        let weights: Vec<f32> = postings.iter().map(|(_, _, w)| *w).collect();
        let max_weight = weights.iter().copied().fold(0.0f32, f32::max);

        let doc_ids_data = pack_bit_array(&deltas[1..], doc_id_bits);
        let ordinals_data = if ordinal_bits > 0 {
            pack_bit_array_u16(&ordinals, ordinal_bits)
        } else {
            Vec::new()
        };
        let weights_data = encode_weights(&weights, weight_quant)?;

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
    pub fn decode_doc_ids_into(&self, out: &mut Vec<DocId>) {
        let count = self.header.count as usize;
        out.clear();
        out.push(self.header.first_doc_id);

        if count > 1 {
            let deltas = unpack_bit_array(&self.doc_ids_data, self.header.doc_id_bits, count - 1);
            let mut prev = self.header.first_doc_id;
            for delta in deltas {
                prev += delta;
                out.push(prev);
            }
        }
    }

    pub fn decode_ordinals(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(self.header.count as usize);
        self.decode_ordinals_into(&mut out);
        out
    }

    /// Decode ordinals into an existing Vec (avoids allocation on reuse).
    pub fn decode_ordinals_into(&self, out: &mut Vec<u16>) {
        let count = self.header.count as usize;
        out.clear();
        if self.header.ordinal_bits == 0 {
            out.resize(count, 0u16);
        } else {
            unpack_bit_array_u16_into(&self.ordinals_data, self.header.ordinal_bits, count, out);
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

        let mut doc_ids_data = vec![0u8; doc_ids_len];
        r.read_exact(&mut doc_ids_data)?;
        let mut ordinals_data = vec![0u8; ordinals_len];
        r.read_exact(&mut ordinals_data)?;
        let mut weights_data = vec![0u8; weights_len];
        r.read_exact(&mut weights_data)?;

        Ok(Self {
            header,
            doc_ids_data,
            ordinals_data,
            weights_data,
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

    /// Serialize with skip list header for lazy loading
    ///
    /// Format:
    /// - doc_count: u32
    /// - global_max_weight: f32
    /// - num_blocks: u32
    /// - skip_list: [SparseSkipEntry] × num_blocks (first_doc, last_doc, offset, length, max_weight)
    /// - block_data: concatenated SparseBlock data
    pub fn serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        use super::SparseSkipEntry;

        w.write_u32::<LittleEndian>(self.doc_count)?;
        w.write_f32::<LittleEndian>(self.global_max_weight())?;
        w.write_u32::<LittleEndian>(self.blocks.len() as u32)?;

        // First pass: serialize blocks to get their sizes
        let mut block_bytes: Vec<Vec<u8>> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let mut buf = Vec::new();
            block.write(&mut buf)?;
            block_bytes.push(buf);
        }

        // Write skip list entries
        let mut offset = 0u32;
        for (block, bytes) in self.blocks.iter().zip(block_bytes.iter()) {
            let first_doc = block.header.first_doc_id;
            let doc_ids = block.decode_doc_ids();
            let last_doc = doc_ids.last().copied().unwrap_or(first_doc);
            let length = bytes.len() as u32;

            let entry =
                SparseSkipEntry::new(first_doc, last_doc, offset, length, block.header.max_weight);
            entry.write(w)?;
            offset += length;
        }

        // Write block data
        for bytes in block_bytes {
            w.write_all(&bytes)?;
        }

        Ok(())
    }

    /// Deserialize fully (loads all blocks into memory)
    /// For lazy loading, use deserialize_header() + load_block()
    pub fn deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        use super::SparseSkipEntry;

        let doc_count = r.read_u32::<LittleEndian>()?;
        let _global_max_weight = r.read_f32::<LittleEndian>()?;
        let num_blocks = r.read_u32::<LittleEndian>()? as usize;

        // Skip the skip list entries
        for _ in 0..num_blocks {
            let _ = SparseSkipEntry::read(r)?;
        }

        // Read all blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(SparseBlock::read(r)?);
        }
        Ok(Self { doc_count, blocks })
    }

    /// Deserialize only the skip list header (for lazy loading)
    /// Returns (doc_count, global_max_weight, skip_entries, header_size)
    pub fn deserialize_header<R: Read>(
        r: &mut R,
    ) -> io::Result<(u32, f32, Vec<super::SparseSkipEntry>, usize)> {
        use super::SparseSkipEntry;

        let doc_count = r.read_u32::<LittleEndian>()?;
        let global_max_weight = r.read_f32::<LittleEndian>()?;
        let num_blocks = r.read_u32::<LittleEndian>()? as usize;

        let mut entries = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            entries.push(SparseSkipEntry::read(r)?);
        }

        // Header size: 4 + 4 + 4 + num_blocks * SparseSkipEntry::SIZE
        let header_size = 4 + 4 + 4 + num_blocks * SparseSkipEntry::SIZE;

        Ok((doc_count, global_max_weight, entries, header_size))
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
    exhausted: bool,
}

impl<'a> BlockSparsePostingIterator<'a> {
    fn new(posting_list: &'a BlockSparsePostingList) -> Self {
        let mut iter = Self {
            posting_list,
            block_idx: 0,
            in_block_idx: 0,
            current_doc_ids: Vec::new(),
            current_ordinals: Vec::new(),
            current_weights: Vec::new(),
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
            block.decode_ordinals_into(&mut self.current_ordinals);
            block.decode_weights_into(&mut self.current_weights);
            self.block_idx = block_idx;
            self.in_block_idx = 0;
        }
    }

    pub fn doc(&self) -> DocId {
        if self.exhausted {
            TERMINATED
        } else {
            self.current_doc_ids
                .get(self.in_block_idx)
                .copied()
                .unwrap_or(TERMINATED)
        }
    }

    pub fn weight(&self) -> f32 {
        self.current_weights
            .get(self.in_block_idx)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn ordinal(&self) -> u16 {
        self.current_ordinals
            .get(self.in_block_idx)
            .copied()
            .unwrap_or(0)
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
    /// Used by block-max WAND to skip entire blocks that can't beat threshold.
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

fn pack_bit_array(values: &[u32], bits: u8) -> Vec<u8> {
    if bits == 0 || values.is_empty() {
        return Vec::new();
    }
    let total_bytes = (values.len() * bits as usize).div_ceil(8);
    let mut result = vec![0u8; total_bytes];
    let mut bit_pos = 0usize;
    for &val in values {
        pack_value(&mut result, bit_pos, val & ((1u32 << bits) - 1), bits);
        bit_pos += bits as usize;
    }
    result
}

fn pack_bit_array_u16(values: &[u16], bits: u8) -> Vec<u8> {
    if bits == 0 || values.is_empty() {
        return Vec::new();
    }
    let total_bytes = (values.len() * bits as usize).div_ceil(8);
    let mut result = vec![0u8; total_bytes];
    let mut bit_pos = 0usize;
    for &val in values {
        pack_value(
            &mut result,
            bit_pos,
            (val as u32) & ((1u32 << bits) - 1),
            bits,
        );
        bit_pos += bits as usize;
    }
    result
}

#[inline]
fn pack_value(data: &mut [u8], bit_pos: usize, val: u32, bits: u8) {
    let mut remaining = bits as usize;
    let mut val = val;
    let mut byte = bit_pos / 8;
    let mut offset = bit_pos % 8;
    while remaining > 0 {
        let space = 8 - offset;
        let to_write = remaining.min(space);
        let mask = (1u32 << to_write) - 1;
        data[byte] |= ((val & mask) as u8) << offset;
        val >>= to_write;
        remaining -= to_write;
        byte += 1;
        offset = 0;
    }
}

fn unpack_bit_array(data: &[u8], bits: u8, count: usize) -> Vec<u32> {
    if bits == 0 || count == 0 {
        return vec![0; count];
    }
    let mut result = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        result.push(unpack_value(data, bit_pos, bits));
        bit_pos += bits as usize;
    }
    result
}

fn unpack_bit_array_u16_into(data: &[u8], bits: u8, count: usize, out: &mut Vec<u16>) {
    if bits == 0 || count == 0 {
        out.resize(count, 0u16);
        return;
    }
    let mut bit_pos = 0usize;
    for _ in 0..count {
        out.push(unpack_value(data, bit_pos, bits) as u16);
        bit_pos += bits as usize;
    }
}

#[inline]
fn unpack_value(data: &[u8], bit_pos: usize, bits: u8) -> u32 {
    let mut val = 0u32;
    let mut remaining = bits as usize;
    let mut byte = bit_pos / 8;
    let mut offset = bit_pos % 8;
    let mut shift = 0;
    while remaining > 0 {
        let space = 8 - offset;
        let to_read = remaining.min(space);
        let mask = (1u8 << to_read) - 1;
        val |= (((data.get(byte).copied().unwrap_or(0) >> offset) & mask) as u32) << shift;
        remaining -= to_read;
        shift += to_read;
        byte += 1;
        offset = 0;
    }
    val
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
            let min = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            for _ in 0..count {
                let q = cursor.read_u8().unwrap_or(0);
                out.push(q as f32 * scale + min);
            }
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

        let mut buf = Vec::new();
        list.serialize(&mut buf).unwrap();
        let list2 = BlockSparsePostingList::deserialize(&mut Cursor::new(&buf)).unwrap();

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

        // Serialize
        let mut bytes = Vec::new();
        merged.serialize(&mut bytes).unwrap();

        // Deserialize
        let mut cursor = std::io::Cursor::new(&bytes);
        let loaded = BlockSparsePostingList::deserialize(&mut cursor).unwrap();

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

        // Serialize and deserialize (simulating what happens after merge file is written)
        let mut bytes = Vec::new();
        merged.serialize(&mut bytes).unwrap();
        let loaded =
            BlockSparsePostingList::deserialize(&mut std::io::Cursor::new(&bytes)).unwrap();

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
    /// serialize → parse raw skip entries + block data → patch first_doc_id → reassemble.
    /// This is the exact code path in `segment/merger/sparse_vectors.rs`.
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

        // Serialize both (this is what the builder writes to disk)
        let mut bytes1 = Vec::new();
        list1.serialize(&mut bytes1).unwrap();
        let mut bytes2 = Vec::new();
        list2.serialize(&mut bytes2).unwrap();

        // --- Simulate read_dim_raw: parse header + skip entries, extract raw block data ---
        fn parse_raw(data: &[u8]) -> (u32, f32, Vec<SparseSkipEntry>, &[u8]) {
            let doc_count = u32::from_le_bytes(data[0..4].try_into().unwrap());
            let global_max = f32::from_le_bytes(data[4..8].try_into().unwrap());
            let num_blocks = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
            let mut pos = 12;
            let mut skip = Vec::new();
            for _ in 0..num_blocks {
                let first_doc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                let last_doc = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
                let offset = u32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap());
                let length = u32::from_le_bytes(data[pos + 12..pos + 16].try_into().unwrap());
                let max_w = f32::from_le_bytes(data[pos + 16..pos + 20].try_into().unwrap());
                skip.push(SparseSkipEntry::new(
                    first_doc, last_doc, offset, length, max_w,
                ));
                pos += 20;
            }
            (doc_count, global_max, skip, &data[pos..])
        }

        let (dc1, gm1, skip1, raw1) = parse_raw(&bytes1);
        let (dc2, gm2, skip2, raw2) = parse_raw(&bytes2);

        // --- Simulate the merger's zero-copy reassembly ---
        let doc_offset: u32 = 1000; // segment 2 starts at doc 1000
        let total_docs = dc1 + dc2;
        let global_max = gm1.max(gm2);
        let total_blocks = (skip1.len() + skip2.len()) as u32;

        let mut output = Vec::new();
        // Write header
        output.extend_from_slice(&total_docs.to_le_bytes());
        output.extend_from_slice(&global_max.to_le_bytes());
        output.extend_from_slice(&total_blocks.to_le_bytes());

        // Write adjusted skip entries
        let mut block_data_offset = 0u32;
        for entry in &skip1 {
            let adjusted = SparseSkipEntry::new(
                entry.first_doc,
                entry.last_doc,
                block_data_offset + entry.offset,
                entry.length,
                entry.max_weight,
            );
            adjusted.write(&mut output).unwrap();
        }
        if let Some(last) = skip1.last() {
            block_data_offset += last.offset + last.length;
        }
        for entry in &skip2 {
            let adjusted = SparseSkipEntry::new(
                entry.first_doc + doc_offset,
                entry.last_doc + doc_offset,
                block_data_offset + entry.offset,
                entry.length,
                entry.max_weight,
            );
            adjusted.write(&mut output).unwrap();
        }

        // Write raw block data: source 1 verbatim, source 2 with first_doc_id patched
        output.extend_from_slice(raw1);

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
        output.extend_from_slice(&buf2);

        // --- Deserialize the reassembled posting list and verify ---
        let loaded = BlockSparsePostingList::deserialize(&mut Cursor::new(&output)).unwrap();
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

        let mut buf = Vec::new();
        list.serialize(&mut buf).unwrap();
        let loaded = BlockSparsePostingList::deserialize(&mut Cursor::new(&buf)).unwrap();
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

        // Serialize and deserialize
        let mut bytes = Vec::new();
        merged.serialize(&mut bytes).unwrap();
        let loaded =
            BlockSparsePostingList::deserialize(&mut std::io::Cursor::new(&bytes)).unwrap();

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
        let mut bytes = Vec::new();
        merged.serialize(&mut bytes).unwrap();
        let loaded =
            BlockSparsePostingList::deserialize(&mut std::io::Cursor::new(&bytes)).unwrap();

        assert!(
            (loaded.global_max_weight() - 7.0).abs() < 0.01,
            "After roundtrip, global max should still be 7.0, got {}",
            loaded.global_max_weight()
        );
    }

    #[test]
    fn test_scoring_simulation_after_merge() {
        // Simulate what SparseTermScorer does - compute query_weight * stored_weight
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
        let mut bytes = Vec::new();
        merged.serialize(&mut bytes).unwrap();
        let loaded =
            BlockSparsePostingList::deserialize(&mut std::io::Cursor::new(&bytes)).unwrap();

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
