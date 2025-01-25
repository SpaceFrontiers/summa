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
        let count = self.header.count as usize;
        let mut doc_ids = Vec::with_capacity(count);
        doc_ids.push(self.header.first_doc_id);

        if count > 1 {
            let deltas = unpack_bit_array(&self.doc_ids_data, self.header.doc_id_bits, count - 1);
            let mut prev = self.header.first_doc_id;
            for delta in deltas {
                prev += delta;
                doc_ids.push(prev);
            }
        }
        doc_ids
    }

    pub fn decode_ordinals(&self) -> Vec<u16> {
        let count = self.header.count as usize;
        if self.header.ordinal_bits == 0 {
            vec![0u16; count]
        } else {
            unpack_bit_array_u16(&self.ordinals_data, self.header.ordinal_bits, count)
        }
    }

    pub fn decode_weights(&self) -> Vec<f32> {
        decode_weights(
            &self.weights_data,
            self.header.weight_quant,
            self.header.count as usize,
        )
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

        Ok(Self {
            doc_count: postings.len() as u32,
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

    pub fn iterator(&self) -> BlockSparsePostingIterator<'_> {
        BlockSparsePostingIterator::new(self)
    }

    pub fn serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.doc_count)?;
        w.write_u32::<LittleEndian>(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.write(w)?;
        }
        Ok(())
    }

    pub fn deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let doc_count = r.read_u32::<LittleEndian>()?;
        let num_blocks = r.read_u32::<LittleEndian>()? as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(SparseBlock::read(r)?);
        }
        Ok(Self { doc_count, blocks })
    }

    pub fn decode_all(&self) -> Vec<(DocId, u16, f32)> {
        let mut result = Vec::with_capacity(self.doc_count as usize);
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

    fn find_block(&self, target: DocId) -> Option<usize> {
        let mut lo = 0;
        let mut hi = self.blocks.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let block = &self.blocks[mid];
            let doc_ids = block.decode_doc_ids();
            let last_doc = doc_ids.last().copied().unwrap_or(block.header.first_doc_id);
            if last_doc < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo < self.blocks.len() {
            Some(lo)
        } else {
            None
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
            self.current_doc_ids = block.decode_doc_ids();
            self.current_weights = block.decode_weights();
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
        if let Some(block) = self.posting_list.blocks.get(self.block_idx) {
            let ordinals = block.decode_ordinals();
            ordinals.get(self.in_block_idx).copied().unwrap_or(0)
        } else {
            0
        }
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

        // Check current block
        if let Some(&last_doc) = self.current_doc_ids.last()
            && last_doc >= target
        {
            while !self.exhausted && self.doc() < target {
                self.in_block_idx += 1;
                if self.in_block_idx >= self.current_doc_ids.len() {
                    self.block_idx += 1;
                    if self.block_idx >= self.posting_list.blocks.len() {
                        self.exhausted = true;
                    } else {
                        self.load_block(self.block_idx);
                    }
                }
            }
            return self.doc();
        }

        // Find correct block
        if let Some(block_idx) = self.posting_list.find_block(target) {
            self.load_block(block_idx);
            while self.in_block_idx < self.current_doc_ids.len()
                && self.current_doc_ids[self.in_block_idx] < target
            {
                self.in_block_idx += 1;
            }
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

fn unpack_bit_array_u16(data: &[u8], bits: u8, count: usize) -> Vec<u16> {
    if bits == 0 || count == 0 {
        return vec![0; count];
    }
    let mut result = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        result.push(unpack_value(data, bit_pos, bits) as u16);
        bit_pos += bits as usize;
    }
    result
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

fn decode_weights(data: &[u8], quant: WeightQuantization, count: usize) -> Vec<f32> {
    let mut cursor = Cursor::new(data);
    let mut weights = Vec::with_capacity(count);
    match quant {
        WeightQuantization::Float32 => {
            for _ in 0..count {
                weights.push(cursor.read_f32::<LittleEndian>().unwrap_or(0.0));
            }
        }
        WeightQuantization::Float16 => {
            use half::f16;
            for _ in 0..count {
                let bits = cursor.read_u16::<LittleEndian>().unwrap_or(0);
                weights.push(f16::from_bits(bits).to_f32());
            }
        }
        WeightQuantization::UInt8 => {
            let scale = cursor.read_f32::<LittleEndian>().unwrap_or(1.0);
            let min = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            for _ in 0..count {
                let q = cursor.read_u8().unwrap_or(0);
                weights.push(q as f32 * scale + min);
            }
        }
        WeightQuantization::UInt4 => {
            let scale = cursor.read_f32::<LittleEndian>().unwrap_or(1.0);
            let min = cursor.read_f32::<LittleEndian>().unwrap_or(0.0);
            let mut i = 0;
            while i < count {
                let byte = cursor.read_u8().unwrap_or(0);
                weights.push((byte & 0x0F) as f32 * scale + min);
                i += 1;
                if i < count {
                    weights.push((byte >> 4) as f32 * scale + min);
                    i += 1;
                }
            }
        }
    }
    weights
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
}
