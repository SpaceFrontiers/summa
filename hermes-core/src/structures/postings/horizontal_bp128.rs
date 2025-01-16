//! Bitpacking utilities for compact integer encoding
//!
//! Implements SIMD-friendly bitpacking for posting list compression.
//! Uses PForDelta-style encoding with exceptions for outliers.
//!
//! Optimizations:
//! - SIMD-accelerated unpacking (when available)
//! - Hillis-Steele parallel prefix sum for delta decoding
//! - Binary search within decoded blocks
//! - Variable block sizes based on posting list length

use crate::structures::simd;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Block size for bitpacking (128 integers per block for SIMD alignment)
pub const HORIZONTAL_BP128_BLOCK_SIZE: usize = 128;

/// Small block size for short posting lists (better cache locality)
pub const SMALL_BLOCK_SIZE: usize = 32;

/// Threshold for using small blocks (posting lists shorter than this use small blocks)
pub const SMALL_BLOCK_THRESHOLD: usize = 256;

/// Pack a block of 128 u32 values using the specified bit width
pub fn pack_block(
    values: &[u32; HORIZONTAL_BP128_BLOCK_SIZE],
    bit_width: u8,
    output: &mut Vec<u8>,
) {
    if bit_width == 0 {
        return;
    }

    let bytes_needed = (HORIZONTAL_BP128_BLOCK_SIZE * bit_width as usize).div_ceil(8);
    let start = output.len();
    output.resize(start + bytes_needed, 0);

    let mut bit_pos = 0usize;
    for &value in values {
        let byte_idx = start + bit_pos / 8;
        let bit_offset = bit_pos % 8;

        // Write value across potentially multiple bytes
        let mut remaining_bits = bit_width as usize;
        let mut val = value;
        let mut current_byte_idx = byte_idx;
        let mut current_bit_offset = bit_offset;

        while remaining_bits > 0 {
            let bits_in_byte = (8 - current_bit_offset).min(remaining_bits);
            let mask = ((1u32 << bits_in_byte) - 1) as u8;
            output[current_byte_idx] |= ((val as u8) & mask) << current_bit_offset;
            val >>= bits_in_byte;
            remaining_bits -= bits_in_byte;
            current_byte_idx += 1;
            current_bit_offset = 0;
        }

        bit_pos += bit_width as usize;
    }
}

/// Unpack a block of 128 u32 values
/// Uses SIMD-optimized unpacking for common bit widths on supported architectures
pub fn unpack_block(input: &[u8], bit_width: u8, output: &mut [u32; HORIZONTAL_BP128_BLOCK_SIZE]) {
    if bit_width == 0 {
        output.fill(0);
        return;
    }

    // Fast path for byte-aligned bit widths with SIMD
    match bit_width {
        8 => simd::unpack_8bit(input, output, HORIZONTAL_BP128_BLOCK_SIZE),
        16 => simd::unpack_16bit(input, output, HORIZONTAL_BP128_BLOCK_SIZE),
        32 => simd::unpack_32bit(input, output, HORIZONTAL_BP128_BLOCK_SIZE),
        _ => unpack_block_generic(input, bit_width, output),
    }
}

/// Generic unpacking for arbitrary bit widths
/// Optimized: reads 64 bits at a time using unaligned pointer read
#[inline]
fn unpack_block_generic(
    input: &[u8],
    bit_width: u8,
    output: &mut [u32; HORIZONTAL_BP128_BLOCK_SIZE],
) {
    let mask = (1u64 << bit_width) - 1;
    let bit_width_usize = bit_width as usize;
    let mut bit_pos = 0usize;

    // Ensure we have enough padding for the last read
    // Max bytes needed: (127 * 32 + 32 + 7) / 8 = 516 bytes for 32-bit width
    // For typical widths (1-20 bits), we need much less
    let input_ptr = input.as_ptr();

    for out in output.iter_mut() {
        let byte_idx = bit_pos >> 3; // bit_pos / 8
        let bit_offset = bit_pos & 7; // bit_pos % 8

        // SAFETY: We read up to 8 bytes. The caller guarantees input has enough data.
        // For 128 values at max 32 bits = 512 bytes, plus up to 7 bits offset = 513 bytes max.
        let word = unsafe { (input_ptr.add(byte_idx) as *const u64).read_unaligned() };

        *out = ((word >> bit_offset) & mask) as u32;
        bit_pos += bit_width_usize;
    }
}

/// Unpack a smaller block (for variable block sizes)
/// Optimized: reads 64 bits at a time using unaligned pointer read
#[inline]
pub fn unpack_block_n(input: &[u8], bit_width: u8, output: &mut [u32], n: usize) {
    if bit_width == 0 {
        output[..n].fill(0);
        return;
    }

    let mask = (1u64 << bit_width) - 1;
    let bit_width_usize = bit_width as usize;
    let mut bit_pos = 0usize;
    let input_ptr = input.as_ptr();

    for out in output[..n].iter_mut() {
        let byte_idx = bit_pos >> 3;
        let bit_offset = bit_pos & 7;

        // SAFETY: Caller guarantees input has enough data for n values at bit_width bits each
        let word = unsafe { (input_ptr.add(byte_idx) as *const u64).read_unaligned() };

        *out = ((word >> bit_offset) & mask) as u32;
        bit_pos += bit_width_usize;
    }
}

/// Binary search within a decoded block to find first element >= target
/// Returns the index within the block, or block.len() if not found
#[inline]
pub fn binary_search_block(block: &[u32], target: u32) -> usize {
    match block.binary_search(&target) {
        Ok(idx) => idx,
        Err(idx) => idx,
    }
}

/// Hillis-Steele inclusive prefix sum for 8 elements
/// Computes: out[i] = sum(input[0..=i])
/// This is the scalar fallback; SIMD version uses AVX2 intrinsics
#[allow(dead_code)]
#[inline]
fn prefix_sum_8(deltas: &mut [u32; 8]) {
    // Step 1: shift by 1
    for i in (1..8).rev() {
        deltas[i] = deltas[i].wrapping_add(deltas[i - 1]);
    }
    // Step 2: shift by 2
    for i in (2..8).rev() {
        deltas[i] = deltas[i].wrapping_add(deltas[i - 2]);
    }
    // Step 4: shift by 4
    for i in (4..8).rev() {
        deltas[i] = deltas[i].wrapping_add(deltas[i - 4]);
    }
}

/// Bitpacked block with skip info for BlockWAND
#[derive(Debug, Clone)]
pub struct HorizontalBP128Block {
    /// Delta-encoded doc_ids (bitpacked)
    pub doc_deltas: Vec<u8>,
    /// Bit width for doc deltas
    pub doc_bit_width: u8,
    /// Term frequencies (bitpacked)
    pub term_freqs: Vec<u8>,
    /// Bit width for term frequencies
    pub tf_bit_width: u8,
    /// First doc_id in this block (absolute)
    pub first_doc_id: u32,
    /// Last doc_id in this block (absolute)
    pub last_doc_id: u32,
    /// Number of docs in this block
    pub num_docs: u16,
    /// Maximum term frequency in this block (for BM25F upper bound calculation)
    pub max_tf: u32,
    /// Maximum impact score in this block (for MaxScore/WAND)
    /// This is computed using BM25F with conservative length normalization
    pub max_block_score: f32,
}

impl HorizontalBP128Block {
    /// Serialize the block
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.first_doc_id)?;
        writer.write_u32::<LittleEndian>(self.last_doc_id)?;
        writer.write_u16::<LittleEndian>(self.num_docs)?;
        writer.write_u8(self.doc_bit_width)?;
        writer.write_u8(self.tf_bit_width)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_block_score)?;

        // Write doc deltas
        writer.write_u16::<LittleEndian>(self.doc_deltas.len() as u16)?;
        writer.write_all(&self.doc_deltas)?;

        // Write term freqs
        writer.write_u16::<LittleEndian>(self.term_freqs.len() as u16)?;
        writer.write_all(&self.term_freqs)?;

        Ok(())
    }

    /// Deserialize a block
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let first_doc_id = reader.read_u32::<LittleEndian>()?;
        let last_doc_id = reader.read_u32::<LittleEndian>()?;
        let num_docs = reader.read_u16::<LittleEndian>()?;
        let doc_bit_width = reader.read_u8()?;
        let tf_bit_width = reader.read_u8()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;
        let max_block_score = reader.read_f32::<LittleEndian>()?;

        let doc_deltas_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut doc_deltas = vec![0u8; doc_deltas_len];
        reader.read_exact(&mut doc_deltas)?;

        let term_freqs_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut term_freqs = vec![0u8; term_freqs_len];
        reader.read_exact(&mut term_freqs)?;

        Ok(Self {
            doc_deltas,
            doc_bit_width,
            term_freqs,
            tf_bit_width,
            first_doc_id,
            last_doc_id,
            num_docs,
            max_tf,
            max_block_score,
        })
    }

    /// Decode doc_ids from this block
    pub fn decode_doc_ids(&self) -> Vec<u32> {
        let mut output = vec![0u32; self.num_docs as usize];
        self.decode_doc_ids_into(&mut output);
        output
    }

    /// Decode doc_ids into a pre-allocated buffer (avoids allocation)
    #[inline]
    pub fn decode_doc_ids_into(&self, output: &mut [u32]) -> usize {
        let count = self.num_docs as usize;
        if count == 0 {
            return 0;
        }

        // Fused unpack + delta decode - no intermediate buffer needed
        simd::unpack_delta_decode(
            &self.doc_deltas,
            self.doc_bit_width,
            output,
            self.first_doc_id,
            count,
        );

        count
    }

    /// Decode term frequencies from this block
    pub fn decode_term_freqs(&self) -> Vec<u32> {
        let mut output = vec![0u32; self.num_docs as usize];
        self.decode_term_freqs_into(&mut output);
        output
    }

    /// Decode term frequencies into a pre-allocated buffer (avoids allocation)
    #[inline]
    pub fn decode_term_freqs_into(&self, output: &mut [u32]) -> usize {
        let count = self.num_docs as usize;
        if count == 0 {
            return 0;
        }

        // Use slice-based unpack to avoid temp buffer copy
        unpack_block_n(&self.term_freqs, self.tf_bit_width, output, count);

        // TF is stored as tf-1, so add 1 back
        simd::add_one(output, count);

        count
    }
}

/// Bitpacked posting list with block-level skip info
#[derive(Debug, Clone)]
pub struct HorizontalBP128PostingList {
    /// Blocks of postings
    pub blocks: Vec<HorizontalBP128Block>,
    /// Total document count
    pub doc_count: u32,
    /// Maximum score across all blocks (for MaxScore pruning)
    pub max_score: f32,
}

impl HorizontalBP128PostingList {
    /// Create from raw doc_ids and term frequencies
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> Self {
        assert_eq!(doc_ids.len(), term_freqs.len());

        if doc_ids.is_empty() {
            return Self {
                blocks: Vec::new(),
                doc_count: 0,
                max_score: 0.0,
            };
        }

        let mut blocks = Vec::new();
        let mut max_score = 0.0f32;
        let mut i = 0;

        while i < doc_ids.len() {
            let block_end = (i + HORIZONTAL_BP128_BLOCK_SIZE).min(doc_ids.len());
            let block_docs = &doc_ids[i..block_end];
            let block_tfs = &term_freqs[i..block_end];

            let block = Self::create_block(block_docs, block_tfs, idf);
            max_score = max_score.max(block.max_block_score);
            blocks.push(block);

            i = block_end;
        }

        Self {
            blocks,
            doc_count: doc_ids.len() as u32,
            max_score,
        }
    }

    /// BM25F parameters for block-max score calculation
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    /// Compute BM25F upper bound score for a given max_tf and IDF
    /// Uses conservative length normalization (assumes shortest possible document)
    #[inline]
    pub fn compute_bm25f_upper_bound(max_tf: u32, idf: f32, field_boost: f32) -> f32 {
        let tf = max_tf as f32;
        // Conservative upper bound: assume dl=0, so length_norm = 1 - b = 0.25
        // This gives the maximum possible score for this tf
        let min_length_norm = 1.0 - Self::B;
        let tf_norm =
            (tf * field_boost * (Self::K1 + 1.0)) / (tf * field_boost + Self::K1 * min_length_norm);
        idf * tf_norm
    }

    fn create_block(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> HorizontalBP128Block {
        let num_docs = doc_ids.len();
        let first_doc_id = doc_ids[0];
        let last_doc_id = *doc_ids.last().unwrap();

        // Compute deltas (delta - 1 to save one bit since deltas are always >= 1)
        let mut deltas = [0u32; HORIZONTAL_BP128_BLOCK_SIZE];
        let mut max_delta = 0u32;
        for j in 1..num_docs {
            let delta = doc_ids[j] - doc_ids[j - 1] - 1;
            deltas[j - 1] = delta;
            max_delta = max_delta.max(delta);
        }

        // Compute max TF and prepare TF array (store tf-1)
        let mut tfs = [0u32; HORIZONTAL_BP128_BLOCK_SIZE];
        let mut max_tf = 0u32;

        for (j, &tf) in term_freqs.iter().enumerate() {
            tfs[j] = tf - 1; // Store tf-1
            max_tf = max_tf.max(tf);
        }

        // BM25F upper bound score using conservative length normalization
        // field_boost defaults to 1.0 at index time; can be adjusted at query time
        let max_block_score = Self::compute_bm25f_upper_bound(max_tf, idf, 1.0);

        let doc_bit_width = simd::bits_needed(max_delta);
        let tf_bit_width = simd::bits_needed(max_tf.saturating_sub(1)); // Store tf-1

        let mut doc_deltas = Vec::new();
        pack_block(&deltas, doc_bit_width, &mut doc_deltas);

        let mut term_freqs_packed = Vec::new();
        pack_block(&tfs, tf_bit_width, &mut term_freqs_packed);

        HorizontalBP128Block {
            doc_deltas,
            doc_bit_width,
            term_freqs: term_freqs_packed,
            tf_bit_width,
            first_doc_id,
            last_doc_id,
            num_docs: num_docs as u16,
            max_tf,
            max_block_score,
        }
    }

    /// Serialize the posting list
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_f32::<LittleEndian>(self.max_score)?;
        writer.write_u32::<LittleEndian>(self.blocks.len() as u32)?;

        for block in &self.blocks {
            block.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize a posting list
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_count = reader.read_u32::<LittleEndian>()?;
        let max_score = reader.read_f32::<LittleEndian>()?;
        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(HorizontalBP128Block::deserialize(reader)?);
        }

        Ok(Self {
            blocks,
            doc_count,
            max_score,
        })
    }

    /// Create an iterator
    pub fn iterator(&self) -> HorizontalBP128Iterator<'_> {
        HorizontalBP128Iterator::new(self)
    }
}

/// Iterator over bitpacked posting list with block skipping support
pub struct HorizontalBP128Iterator<'a> {
    posting_list: &'a HorizontalBP128PostingList,
    /// Current block index
    current_block: usize,
    /// Number of valid elements in current block
    current_block_len: usize,
    /// Pre-allocated buffer for decoded doc_ids (avoids allocation per block)
    block_doc_ids: Vec<u32>,
    /// Pre-allocated buffer for decoded term freqs
    block_term_freqs: Vec<u32>,
    /// Position within current block
    pos_in_block: usize,
    /// Whether we've exhausted all postings
    exhausted: bool,
}

impl<'a> HorizontalBP128Iterator<'a> {
    pub fn new(posting_list: &'a HorizontalBP128PostingList) -> Self {
        // Pre-allocate buffers to block size to avoid allocations during iteration
        let mut iter = Self {
            posting_list,
            current_block: 0,
            current_block_len: 0,
            block_doc_ids: vec![0u32; HORIZONTAL_BP128_BLOCK_SIZE],
            block_term_freqs: vec![0u32; HORIZONTAL_BP128_BLOCK_SIZE],
            pos_in_block: 0,
            exhausted: posting_list.blocks.is_empty(),
        };

        if !iter.exhausted {
            iter.decode_current_block();
        }

        iter
    }

    #[inline]
    fn decode_current_block(&mut self) {
        let block = &self.posting_list.blocks[self.current_block];
        // Decode into pre-allocated buffers (no allocation!)
        self.current_block_len = block.decode_doc_ids_into(&mut self.block_doc_ids);
        block.decode_term_freqs_into(&mut self.block_term_freqs);
        self.pos_in_block = 0;
    }

    /// Current document ID
    #[inline]
    pub fn doc(&self) -> u32 {
        if self.exhausted {
            u32::MAX
        } else {
            self.block_doc_ids[self.pos_in_block]
        }
    }

    /// Current term frequency
    #[inline]
    pub fn term_freq(&self) -> u32 {
        if self.exhausted {
            0
        } else {
            self.block_term_freqs[self.pos_in_block]
        }
    }

    /// Advance to next document
    #[inline]
    pub fn advance(&mut self) -> u32 {
        if self.exhausted {
            return u32::MAX;
        }

        self.pos_in_block += 1;

        if self.pos_in_block >= self.current_block_len {
            self.current_block += 1;
            if self.current_block >= self.posting_list.blocks.len() {
                self.exhausted = true;
                return u32::MAX;
            }
            self.decode_current_block();
        }

        self.doc()
    }

    /// Seek to first doc >= target (with block skipping and binary search)
    pub fn seek(&mut self, target: u32) -> u32 {
        if self.exhausted {
            return u32::MAX;
        }

        // Binary search to find the right block
        let block_idx = self.posting_list.blocks[self.current_block..].binary_search_by(|block| {
            if block.last_doc_id < target {
                std::cmp::Ordering::Less
            } else if block.first_doc_id > target {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        let target_block = match block_idx {
            Ok(idx) => self.current_block + idx,
            Err(idx) => {
                if self.current_block + idx >= self.posting_list.blocks.len() {
                    self.exhausted = true;
                    return u32::MAX;
                }
                self.current_block + idx
            }
        };

        // Move to target block if different
        if target_block != self.current_block {
            self.current_block = target_block;
            self.decode_current_block();
        } else if self.current_block_len == 0 {
            self.decode_current_block();
        }

        // Binary search within the block
        let pos = binary_search_block(
            &self.block_doc_ids[self.pos_in_block..self.current_block_len],
            target,
        );
        self.pos_in_block += pos;

        if self.pos_in_block >= self.current_block_len {
            // Target not in this block, move to next
            self.current_block += 1;
            if self.current_block >= self.posting_list.blocks.len() {
                self.exhausted = true;
                return u32::MAX;
            }
            self.decode_current_block();
        }

        self.doc()
    }

    /// Get max score for remaining blocks (for MaxScore optimization)
    pub fn max_remaining_score(&self) -> f32 {
        if self.exhausted {
            return 0.0;
        }

        self.posting_list.blocks[self.current_block..]
            .iter()
            .map(|b| b.max_block_score)
            .fold(0.0f32, |a, b| a.max(b))
    }

    /// Skip to next block (for BlockWAND)
    pub fn skip_to_block_with_doc(&mut self, target: u32) -> Option<(u32, f32)> {
        while self.current_block < self.posting_list.blocks.len() {
            let block = &self.posting_list.blocks[self.current_block];
            if block.last_doc_id >= target {
                return Some((block.first_doc_id, block.max_block_score));
            }
            self.current_block += 1;
        }
        self.exhausted = true;
        None
    }

    /// Get current block's max score
    pub fn current_block_max_score(&self) -> f32 {
        if self.exhausted {
            0.0
        } else {
            self.posting_list.blocks[self.current_block].max_block_score
        }
    }

    /// Get current block's max term frequency (for BM25F upper bound recalculation)
    pub fn current_block_max_tf(&self) -> u32 {
        if self.exhausted {
            0
        } else {
            self.posting_list.blocks[self.current_block].max_tf
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_needed() {
        assert_eq!(simd::bits_needed(0), 0);
        assert_eq!(simd::bits_needed(1), 1);
        assert_eq!(simd::bits_needed(2), 2);
        assert_eq!(simd::bits_needed(3), 2);
        assert_eq!(simd::bits_needed(255), 8);
        assert_eq!(simd::bits_needed(256), 9);
    }

    #[test]
    fn test_pack_unpack() {
        let mut values = [0u32; HORIZONTAL_BP128_BLOCK_SIZE];
        for (i, value) in values.iter_mut().enumerate() {
            *value = (i * 3) as u32;
        }

        let max_val = values.iter().max().copied().unwrap();
        let bit_width = simd::bits_needed(max_val);

        let mut packed = Vec::new();
        pack_block(&values, bit_width, &mut packed);

        let mut unpacked = [0u32; HORIZONTAL_BP128_BLOCK_SIZE];
        unpack_block(&packed, bit_width, &mut unpacked);

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_bitpacked_posting_list() {
        let doc_ids: Vec<u32> = (0..200).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = (0..200).map(|i| (i % 10) + 1).collect();

        let posting_list = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        assert_eq!(posting_list.doc_count, 200);
        assert_eq!(posting_list.blocks.len(), 2); // 128 + 72

        // Test iteration
        let mut iter = posting_list.iterator();
        for (i, &expected_doc) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected_doc, "Mismatch at position {}", i);
            assert_eq!(iter.term_freq(), term_freqs[i]);
            if i < doc_ids.len() - 1 {
                iter.advance();
            }
        }
    }

    #[test]
    fn test_bitpacked_seek() {
        let doc_ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 1000, 2000];
        let term_freqs: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let posting_list = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut iter = posting_list.iterator();

        assert_eq!(iter.seek(25), 30);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(500), 1000);
        assert_eq!(iter.seek(3000), u32::MAX);
    }

    #[test]
    fn test_serialization() {
        let doc_ids: Vec<u32> = (0..50).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..50).map(|_| 1).collect();

        let posting_list = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.5);

        let mut buffer = Vec::new();
        posting_list.serialize(&mut buffer).unwrap();

        let restored = HorizontalBP128PostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.doc_count, posting_list.doc_count);
        assert_eq!(restored.blocks.len(), posting_list.blocks.len());

        // Verify iteration produces same results
        let mut iter1 = posting_list.iterator();
        let mut iter2 = restored.iterator();

        while iter1.doc() != u32::MAX {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
    }

    #[test]
    fn test_hillis_steele_prefix_sum() {
        // Test the prefix_sum_8 function directly
        let mut deltas = [1u32, 2, 3, 4, 5, 6, 7, 8];
        prefix_sum_8(&mut deltas);
        // Expected: [1, 1+2, 1+2+3, 1+2+3+4, ...]
        assert_eq!(deltas, [1, 3, 6, 10, 15, 21, 28, 36]);

        // Test simd::delta_decode
        let deltas2 = [0u32; 16]; // gaps of 1 (stored as 0)
        let mut output2 = [0u32; 16];
        simd::delta_decode(&mut output2, &deltas2, 100, 8);
        // first_doc_id=100, then +1 each
        assert_eq!(&output2[..8], &[100, 101, 102, 103, 104, 105, 106, 107]);

        // Test with varying deltas (stored as gap-1)
        // gaps: 2, 1, 3, 1, 5, 1, 1 → stored as: 1, 0, 2, 0, 4, 0, 0
        let deltas3 = [1u32, 0, 2, 0, 4, 0, 0, 0];
        let mut output3 = [0u32; 8];
        simd::delta_decode(&mut output3, &deltas3, 10, 8);
        // 10, 10+2=12, 12+1=13, 13+3=16, 16+1=17, 17+5=22, 22+1=23, 23+1=24
        assert_eq!(&output3[..8], &[10, 12, 13, 16, 17, 22, 23, 24]);
    }

    #[test]
    fn test_delta_decode_large_block() {
        // Test with a full 128-element block
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 5 + 100).collect();
        let term_freqs: Vec<u32> = vec![1; 128];

        let posting_list = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let decoded = posting_list.blocks[0].decode_doc_ids();

        assert_eq!(decoded.len(), 128);
        for (i, (&expected, &actual)) in doc_ids.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(expected, actual, "Mismatch at position {}", i);
        }
    }
}
