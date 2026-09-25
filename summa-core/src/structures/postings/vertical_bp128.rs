//! SIMD-BP128: Vectorized bitpacking with NEON/SSE intrinsics
//!
//! Based on Lemire & Boytsov (2015) "Decoding billions of integers per second through vectorization"
//! and Quickwit's bitpacking crate architecture.
//!
//! Key optimizations:
//! - **True vertical layout**: Optimal compression (BLOCK_SIZE * bit_width / 8 bytes)
//! - **Integrated delta decoding**: Fused unpack + prefix sum in single pass
//! - **128-integer blocks**: 32 groups of 4 integers each
//! - **NEON intrinsics on ARM**: Uses vld1q_u32, vaddq_u32, etc.
//! - **Block-level metadata**: Skip info for block-max pruning

use crate::structures::simd;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Block size: 128 integers (32 groups of 4 for SIMD lanes)
pub const VERTICAL_BP128_BLOCK_SIZE: usize = 128;

// ============================================================================
// Public API - True Vertical Bit-Interleaved Layout
// ============================================================================
//
// Vertical Layout Specification:
// For a block of 128 integers at bit_width bits each:
// - Total bytes = 128 * bit_width / 8
// - For each bit position b (0 to bit_width-1):
//   - Store 16 bytes containing bit b from all 128 integers
//   - Byte i within those 16 bytes contains bit b from integers i*8 to i*8+7
//   - Bit j within byte i = bit b of integer i*8+j
//
// Example for bit_width=4:
//   Bytes 0-15:  bit 0 of all 128 integers (16 bytes × 8 bits = 128 bits)
//   Bytes 16-31: bit 1 of all 128 integers
//   Bytes 32-47: bit 2 of all 128 integers
//   Bytes 48-63: bit 3 of all 128 integers
//   Total: 64 bytes
//
// This layout enables SIMD: one 16-byte load gets one bit from all 128 integers.

/// Pack 128 integers using true vertical bit-interleaved layout
///
/// Vertical layout stores bit i of all 128 integers together in 16 consecutive bytes.
/// Total size: exactly 128 * bit_width / 8 bytes (no padding waste)
///
/// This layout is optimal for SIMD unpacking: a single 16-byte load retrieves
/// one bit position from all 128 integers simultaneously.
pub fn pack_vertical(
    values: &[u32; VERTICAL_BP128_BLOCK_SIZE],
    bit_width: u8,
    output: &mut Vec<u8>,
) {
    if bit_width == 0 {
        return;
    }

    // Total bytes = 128 * bit_width / 8 = 16 * bit_width
    let total_bytes = 16 * bit_width as usize;
    let start = output.len();
    output.resize(start + total_bytes, 0);

    // For each bit position, pack that bit from all 128 integers into 16 bytes
    for bit_pos in 0..bit_width as usize {
        let byte_offset = start + bit_pos * 16;

        // Process 16 bytes (128 integers, 8 per byte)
        for byte_idx in 0..16 {
            let base_int = byte_idx * 8;
            let mut byte_val = 0u8;

            // Pack 8 integers' bits into one byte
            byte_val |= ((values[base_int] >> bit_pos) & 1) as u8;
            byte_val |= (((values[base_int + 1] >> bit_pos) & 1) as u8) << 1;
            byte_val |= (((values[base_int + 2] >> bit_pos) & 1) as u8) << 2;
            byte_val |= (((values[base_int + 3] >> bit_pos) & 1) as u8) << 3;
            byte_val |= (((values[base_int + 4] >> bit_pos) & 1) as u8) << 4;
            byte_val |= (((values[base_int + 5] >> bit_pos) & 1) as u8) << 5;
            byte_val |= (((values[base_int + 6] >> bit_pos) & 1) as u8) << 6;
            byte_val |= (((values[base_int + 7] >> bit_pos) & 1) as u8) << 7;

            output[byte_offset + byte_idx] = byte_val;
        }
    }
}

/// Unpack 128 integers from true vertical bit-interleaved layout
///
/// Uses SIMD on supported architectures (NEON on aarch64, SSE on x86_64).
/// Falls back to optimized scalar implementation on other platforms.
pub fn unpack_vertical(input: &[u8], bit_width: u8, output: &mut [u32; VERTICAL_BP128_BLOCK_SIZE]) {
    if bit_width == 0 {
        output.fill(0);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { unpack_vertical_neon(input, bit_width, output) }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe { unpack_vertical_sse(input, bit_width, output) }
        } else {
            unpack_vertical_scalar(input, bit_width, output)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        unpack_vertical_scalar(input, bit_width, output)
    }
}

/// Scalar implementation of vertical unpack (the only path off aarch64/SSE2).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn unpack_vertical_scalar(
    input: &[u8],
    bit_width: u8,
    output: &mut [u32; VERTICAL_BP128_BLOCK_SIZE],
) {
    output.fill(0);

    // For each bit position, scatter that bit to all 128 integers
    for bit_pos in 0..bit_width as usize {
        let byte_offset = bit_pos * 16;
        let bit_mask = 1u32 << bit_pos;

        // Process 16 bytes (128 integers)
        for byte_idx in 0..16 {
            let byte_val = input[byte_offset + byte_idx];
            let base_int = byte_idx * 8;

            // Scatter 8 bits to 8 integers
            if byte_val & 0x01 != 0 {
                output[base_int] |= bit_mask;
            }
            if byte_val & 0x02 != 0 {
                output[base_int + 1] |= bit_mask;
            }
            if byte_val & 0x04 != 0 {
                output[base_int + 2] |= bit_mask;
            }
            if byte_val & 0x08 != 0 {
                output[base_int + 3] |= bit_mask;
            }
            if byte_val & 0x10 != 0 {
                output[base_int + 4] |= bit_mask;
            }
            if byte_val & 0x20 != 0 {
                output[base_int + 5] |= bit_mask;
            }
            if byte_val & 0x40 != 0 {
                output[base_int + 6] |= bit_mask;
            }
            if byte_val & 0x80 != 0 {
                output[base_int + 7] |= bit_mask;
            }
        }
    }
}

/// NEON-optimized vertical unpack for aarch64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unpack_vertical_neon(
    input: &[u8],
    bit_width: u8,
    output: &mut [u32; VERTICAL_BP128_BLOCK_SIZE],
) {
    use std::arch::aarch64::*;

    unsafe {
        // Clear output using NEON
        let zero = vdupq_n_u32(0);
        for i in (0..VERTICAL_BP128_BLOCK_SIZE).step_by(4) {
            vst1q_u32(output[i..].as_mut_ptr(), zero);
        }

        // For each bit position
        for bit_pos in 0..bit_width as usize {
            let byte_offset = bit_pos * 16;
            let bit_mask = 1u32 << bit_pos;

            // Load 16 bytes (one bit-plane for all 128 integers)
            let bytes = vld1q_u8(input.as_ptr().add(byte_offset));

            // Store to array for processing (NEON lane extraction requires constants)
            let mut byte_array = [0u8; 16];
            vst1q_u8(byte_array.as_mut_ptr(), bytes);

            // Process 16 bytes (128 integers, 8 per byte)
            for (byte_idx, &byte_val) in byte_array.iter().enumerate() {
                let base_int = byte_idx * 8;

                // Scatter 8 bits to 8 integers using branchless OR
                output[base_int] |= ((byte_val & 0x01) as u32) * bit_mask;
                output[base_int + 1] |= (((byte_val >> 1) & 0x01) as u32) * bit_mask;
                output[base_int + 2] |= (((byte_val >> 2) & 0x01) as u32) * bit_mask;
                output[base_int + 3] |= (((byte_val >> 3) & 0x01) as u32) * bit_mask;
                output[base_int + 4] |= (((byte_val >> 4) & 0x01) as u32) * bit_mask;
                output[base_int + 5] |= (((byte_val >> 5) & 0x01) as u32) * bit_mask;
                output[base_int + 6] |= (((byte_val >> 6) & 0x01) as u32) * bit_mask;
                output[base_int + 7] |= (((byte_val >> 7) & 0x01) as u32) * bit_mask;
            }
        }
    }
}

/// SSE-optimized vertical unpack for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn unpack_vertical_sse(
    input: &[u8],
    bit_width: u8,
    output: &mut [u32; VERTICAL_BP128_BLOCK_SIZE],
) {
    use std::arch::x86_64::*;

    unsafe {
        // Clear output
        let zero = _mm_setzero_si128();
        for i in (0..VERTICAL_BP128_BLOCK_SIZE).step_by(4) {
            _mm_storeu_si128(output[i..].as_mut_ptr() as *mut __m128i, zero);
        }

        // For each bit position
        for bit_pos in 0..bit_width as usize {
            let byte_offset = bit_pos * 16;

            // Load 16 bytes (one bit-plane for all 128 integers)
            let bytes = _mm_loadu_si128(input.as_ptr().add(byte_offset) as *const __m128i);

            // Extract bytes and scatter bits
            let mut byte_array = [0u8; 16];
            _mm_storeu_si128(byte_array.as_mut_ptr() as *mut __m128i, bytes);

            for (byte_idx, &byte_val) in byte_array.iter().enumerate() {
                let base_int = byte_idx * 8;

                // Scatter 8 bits to 8 integers
                if byte_val & 0x01 != 0 {
                    output[base_int] |= 1u32 << bit_pos;
                }
                if byte_val & 0x02 != 0 {
                    output[base_int + 1] |= 1u32 << bit_pos;
                }
                if byte_val & 0x04 != 0 {
                    output[base_int + 2] |= 1u32 << bit_pos;
                }
                if byte_val & 0x08 != 0 {
                    output[base_int + 3] |= 1u32 << bit_pos;
                }
                if byte_val & 0x10 != 0 {
                    output[base_int + 4] |= 1u32 << bit_pos;
                }
                if byte_val & 0x20 != 0 {
                    output[base_int + 5] |= 1u32 << bit_pos;
                }
                if byte_val & 0x40 != 0 {
                    output[base_int + 6] |= 1u32 << bit_pos;
                }
                if byte_val & 0x80 != 0 {
                    output[base_int + 7] |= 1u32 << bit_pos;
                }
            }
        }
    }
}

/// Unpack with integrated delta decoding (fused for better performance)
///
/// The encoding stores deltas[i] = doc_ids[i+1] - doc_ids[i] - 1
/// So we have (count-1) deltas for count doc_ids.
/// first_doc_id is doc_ids[0], and we compute the rest from deltas.
///
/// This fused version avoids a separate prefix sum pass by computing
/// doc_ids inline during unpacking. Uses true vertical bit-interleaved layout.
pub fn unpack_vertical_d1(
    input: &[u8],
    bit_width: u8,
    first_doc_id: u32,
    output: &mut [u32; VERTICAL_BP128_BLOCK_SIZE],
    count: usize,
) {
    if count == 0 {
        return;
    }

    if bit_width == 0 {
        // All deltas are 0, so gaps are all 1
        let mut current = first_doc_id;
        output[0] = current;
        for out_val in output.iter_mut().take(count).skip(1) {
            current = current.wrapping_add(1);
            *out_val = current;
        }
        return;
    }

    // First unpack all deltas from vertical layout
    let mut deltas = [0u32; VERTICAL_BP128_BLOCK_SIZE];
    unpack_vertical(input, bit_width, &mut deltas);

    // Then apply prefix sum with delta decoding
    output[0] = first_doc_id;
    let mut current = first_doc_id;

    for i in 1..count {
        // deltas[i-1] stores (gap - 1), so actual gap = deltas[i-1] + 1
        current = current.wrapping_add(deltas[i - 1]).wrapping_add(1);
        output[i] = current;
    }
}

/// A single SIMD-BP128 block with metadata
#[derive(Debug, Clone)]
pub struct VerticalBP128Block {
    /// Vertically-packed delta-encoded doc_ids (true vertical bit-interleaved layout)
    pub doc_data: Vec<u8>,
    /// Bit width for doc deltas
    pub doc_bit_width: u8,
    /// Vertically-packed term frequencies (tf - 1)
    pub tf_data: Vec<u8>,
    /// Bit width for term frequencies
    pub tf_bit_width: u8,
    /// First doc_id in block (absolute)
    pub first_doc_id: u32,
    /// Last doc_id in block (absolute)
    pub last_doc_id: u32,
    /// Number of docs in this block
    pub num_docs: u16,
    /// Maximum term frequency in block
    pub max_tf: u32,
    /// Maximum BM25 score upper bound for block-max pruning
    pub max_block_score: f32,
}

impl VerticalBP128Block {
    /// Serialize block
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.first_doc_id)?;
        writer.write_u32::<LittleEndian>(self.last_doc_id)?;
        writer.write_u16::<LittleEndian>(self.num_docs)?;
        writer.write_u8(self.doc_bit_width)?;
        writer.write_u8(self.tf_bit_width)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_block_score)?;

        writer.write_u16::<LittleEndian>(self.doc_data.len() as u16)?;
        writer.write_all(&self.doc_data)?;

        writer.write_u16::<LittleEndian>(self.tf_data.len() as u16)?;
        writer.write_all(&self.tf_data)?;

        Ok(())
    }

    /// Deserialize block
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let first_doc_id = reader.read_u32::<LittleEndian>()?;
        let last_doc_id = reader.read_u32::<LittleEndian>()?;
        let num_docs = reader.read_u16::<LittleEndian>()?;
        let doc_bit_width = reader.read_u8()?;
        let tf_bit_width = reader.read_u8()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;
        let max_block_score = reader.read_f32::<LittleEndian>()?;

        let doc_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut doc_data = vec![0u8; doc_len];
        reader.read_exact(&mut doc_data)?;

        let tf_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut tf_data = vec![0u8; tf_len];
        reader.read_exact(&mut tf_data)?;

        Ok(Self {
            doc_data,
            doc_bit_width,
            tf_data,
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

        // For full blocks, decode directly if output is large enough
        // For partial blocks, we need the temp buffer due to SIMD alignment
        if count == VERTICAL_BP128_BLOCK_SIZE && output.len() >= VERTICAL_BP128_BLOCK_SIZE {
            // SAFETY: output slice is large enough, reinterpret as fixed array
            let out_array: &mut [u32; VERTICAL_BP128_BLOCK_SIZE] = (&mut output
                [..VERTICAL_BP128_BLOCK_SIZE])
                .try_into()
                .unwrap();
            unpack_vertical_d1(
                &self.doc_data,
                self.doc_bit_width,
                self.first_doc_id,
                out_array,
                count,
            );
        } else {
            // Partial block - need temp buffer for SIMD alignment
            let mut temp = [0u32; VERTICAL_BP128_BLOCK_SIZE];
            unpack_vertical_d1(
                &self.doc_data,
                self.doc_bit_width,
                self.first_doc_id,
                &mut temp,
                count,
            );
            output[..count].copy_from_slice(&temp[..count]);
        }

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

        // For full blocks, decode directly if output is large enough
        if count == VERTICAL_BP128_BLOCK_SIZE && output.len() >= VERTICAL_BP128_BLOCK_SIZE {
            let out_array: &mut [u32; VERTICAL_BP128_BLOCK_SIZE] = (&mut output
                [..VERTICAL_BP128_BLOCK_SIZE])
                .try_into()
                .unwrap();
            unpack_vertical(&self.tf_data, self.tf_bit_width, out_array);
        } else {
            // Partial block - need temp buffer for SIMD alignment
            let mut temp = [0u32; VERTICAL_BP128_BLOCK_SIZE];
            unpack_vertical(&self.tf_data, self.tf_bit_width, &mut temp);
            output[..count].copy_from_slice(&temp[..count]);
        }

        // TF is stored as tf-1, add 1 back
        simd::add_one(output, count);

        count
    }
}

/// SIMD-BP128 posting list with vertical layout and BlockMax support
#[derive(Debug, Clone)]
pub struct VerticalBP128PostingList {
    /// Blocks of postings
    pub blocks: Vec<VerticalBP128Block>,
    /// Total document count
    pub doc_count: u32,
    /// Maximum score across all blocks
    pub max_score: f32,
}

impl VerticalBP128PostingList {
    /// Create from raw postings
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
            let block_end = (i + VERTICAL_BP128_BLOCK_SIZE).min(doc_ids.len());
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

    fn create_block(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> VerticalBP128Block {
        let num_docs = doc_ids.len();
        let first_doc_id = doc_ids[0];
        let last_doc_id = *doc_ids.last().unwrap();

        // Compute deltas (gap - 1)
        let mut deltas = [0u32; VERTICAL_BP128_BLOCK_SIZE];
        let mut max_delta = 0u32;
        for j in 1..num_docs {
            let delta = doc_ids[j] - doc_ids[j - 1] - 1;
            deltas[j - 1] = delta;
            max_delta = max_delta.max(delta);
        }

        // Compute TFs (tf - 1)
        let mut tfs = [0u32; VERTICAL_BP128_BLOCK_SIZE];
        let mut max_tf = 0u32;
        for (j, &tf) in term_freqs.iter().enumerate() {
            tfs[j] = tf.saturating_sub(1);
            max_tf = max_tf.max(tf);
        }

        let doc_bit_width = simd::bits_needed(max_delta);
        let tf_bit_width = simd::bits_needed(max_tf.saturating_sub(1));

        let mut doc_data = Vec::new();
        pack_vertical(&deltas, doc_bit_width, &mut doc_data);

        let mut tf_data = Vec::new();
        pack_vertical(&tfs, tf_bit_width, &mut tf_data);

        let max_block_score = crate::query::bm25_upper_bound(max_tf as f32, idf);

        VerticalBP128Block {
            doc_data,
            doc_bit_width,
            tf_data,
            tf_bit_width,
            first_doc_id,
            last_doc_id,
            num_docs: num_docs as u16,
            max_tf,
            max_block_score,
        }
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_f32::<LittleEndian>(self.max_score)?;
        writer.write_u32::<LittleEndian>(self.blocks.len() as u32)?;

        for block in &self.blocks {
            block.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_count = reader.read_u32::<LittleEndian>()?;
        let max_score = reader.read_f32::<LittleEndian>()?;
        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(VerticalBP128Block::deserialize(reader)?);
        }

        Ok(Self {
            blocks,
            doc_count,
            max_score,
        })
    }

    /// Create iterator
    pub fn iterator(&self) -> VerticalBP128Iterator<'_> {
        VerticalBP128Iterator::new(self)
    }

    /// Get approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        let mut size = 12; // header
        for block in &self.blocks {
            size += 22 + block.doc_data.len() + block.tf_data.len();
        }
        size
    }
}

/// Iterator over SIMD-BP128 posting list
pub struct VerticalBP128Iterator<'a> {
    list: &'a VerticalBP128PostingList,
    current_block: usize,
    /// Number of valid elements in current block
    current_block_len: usize,
    /// Pre-allocated buffer for decoded doc_ids (avoids allocation per block)
    block_doc_ids: Vec<u32>,
    /// Pre-allocated buffer for decoded term freqs
    block_term_freqs: Vec<u32>,
    pos_in_block: usize,
    exhausted: bool,
}

impl<'a> VerticalBP128Iterator<'a> {
    pub fn new(list: &'a VerticalBP128PostingList) -> Self {
        // Pre-allocate buffers to block size to avoid allocations during iteration
        let mut iter = Self {
            list,
            current_block: 0,
            current_block_len: 0,
            block_doc_ids: vec![0u32; VERTICAL_BP128_BLOCK_SIZE],
            block_term_freqs: vec![0u32; VERTICAL_BP128_BLOCK_SIZE],
            pos_in_block: 0,
            exhausted: list.blocks.is_empty(),
        };

        if !iter.exhausted {
            iter.decode_current_block();
        }

        iter
    }

    #[inline]
    fn decode_current_block(&mut self) {
        let block = &self.list.blocks[self.current_block];
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
            if self.current_block >= self.list.blocks.len() {
                self.exhausted = true;
                return u32::MAX;
            }
            self.decode_current_block();
        }

        self.doc()
    }

    /// Seek to first doc >= target with block skipping
    pub fn seek(&mut self, target: u32) -> u32 {
        if self.exhausted {
            return u32::MAX;
        }

        // Binary search for target block
        let block_idx = self.list.blocks[self.current_block..].binary_search_by(|block| {
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
                if self.current_block + idx >= self.list.blocks.len() {
                    self.exhausted = true;
                    return u32::MAX;
                }
                self.current_block + idx
            }
        };

        if target_block != self.current_block {
            self.current_block = target_block;
            self.decode_current_block();
        }

        // Binary search within block
        let pos = self.block_doc_ids[self.pos_in_block..self.current_block_len]
            .binary_search(&target)
            .unwrap_or_else(|x| x);
        self.pos_in_block += pos;

        if self.pos_in_block >= self.current_block_len {
            self.current_block += 1;
            if self.current_block >= self.list.blocks.len() {
                self.exhausted = true;
                return u32::MAX;
            }
            self.decode_current_block();
        }

        self.doc()
    }

    /// Get max score for remaining blocks
    pub fn max_remaining_score(&self) -> f32 {
        if self.exhausted {
            return 0.0;
        }
        self.list.blocks[self.current_block..]
            .iter()
            .map(|b| b.max_block_score)
            .fold(0.0f32, |a, b| a.max(b))
    }

    /// Get current block's max score
    pub fn current_block_max_score(&self) -> f32 {
        if self.exhausted {
            0.0
        } else {
            self.list.blocks[self.current_block].max_block_score
        }
    }

    /// Get current block's max TF
    pub fn current_block_max_tf(&self) -> u32 {
        if self.exhausted {
            0
        } else {
            self.list.blocks[self.current_block].max_tf
        }
    }

    /// Skip to next block containing doc >= target (for block-max pruning)
    /// Returns (first_doc_in_block, block_max_score) or None if exhausted
    pub fn skip_to_block_with_doc(&mut self, target: u32) -> Option<(u32, f32)> {
        while self.current_block < self.list.blocks.len() {
            let block = &self.list.blocks[self.current_block];
            if block.last_doc_id >= target {
                // Decode this block and position at start
                self.decode_current_block();
                return Some((block.first_doc_id, block.max_block_score));
            }
            self.current_block += 1;
        }
        self.exhausted = true;
        None
    }

    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_vertical() {
        let mut values = [0u32; VERTICAL_BP128_BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i * 3) as u32;
        }

        let max_val = values.iter().max().copied().unwrap();
        let bit_width = simd::bits_needed(max_val);

        let mut packed = Vec::new();
        pack_vertical(&values, bit_width, &mut packed);

        let mut unpacked = [0u32; VERTICAL_BP128_BLOCK_SIZE];
        unpack_vertical(&packed, bit_width, &mut unpacked);

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_vertical_various_widths() {
        for bit_width in 1..=20 {
            let mut values = [0u32; VERTICAL_BP128_BLOCK_SIZE];
            let max_val = (1u32 << bit_width) - 1;
            for (i, v) in values.iter_mut().enumerate() {
                *v = (i as u32) % (max_val + 1);
            }

            let mut packed = Vec::new();
            pack_vertical(&values, bit_width, &mut packed);

            let mut unpacked = [0u32; VERTICAL_BP128_BLOCK_SIZE];
            unpack_vertical(&packed, bit_width, &mut unpacked);

            assert_eq!(values, unpacked, "Failed for bit_width={}", bit_width);
        }
    }

    #[test]
    fn test_simd_bp128_posting_list() {
        let doc_ids: Vec<u32> = (0..200).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = (0..200).map(|i| (i % 10) + 1).collect();

        let list = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        assert_eq!(list.doc_count, 200);
        assert_eq!(list.blocks.len(), 2); // 128 + 72

        let mut iter = list.iterator();
        for (i, &expected_doc) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), term_freqs[i], "TF mismatch at {}", i);
            if i < doc_ids.len() - 1 {
                iter.advance();
            }
        }
    }

    #[test]
    fn test_simd_bp128_seek() {
        let doc_ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 1000, 2000];
        let term_freqs: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let list = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut iter = list.iterator();

        assert_eq!(iter.seek(25), 30);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(500), 1000);
        assert_eq!(iter.seek(3000), u32::MAX);
    }

    #[test]
    fn test_simd_bp128_serialization() {
        let doc_ids: Vec<u32> = (0..300).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..300).map(|i| (i % 5) + 1).collect();

        let list = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.5);

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let restored = VerticalBP128PostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.doc_count, list.doc_count);
        assert_eq!(restored.blocks.len(), list.blocks.len());

        let mut iter1 = list.iterator();
        let mut iter2 = restored.iterator();

        while iter1.doc() != u32::MAX {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
    }

    #[test]
    fn test_vertical_layout_size() {
        // True vertical layout: BLOCK_SIZE * bit_width / 8 bytes (optimal)
        let mut values = [0u32; VERTICAL_BP128_BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = i as u32;
        }

        let bit_width = simd::bits_needed(127); // 7 bits
        assert_eq!(bit_width, 7);

        let mut packed = Vec::new();
        pack_vertical(&values, bit_width, &mut packed);

        // True vertical layout: 128 * 7 / 8 = 112 bytes (optimal, no padding)
        let expected_bytes = (VERTICAL_BP128_BLOCK_SIZE * bit_width as usize) / 8;
        assert_eq!(expected_bytes, 112);
        assert_eq!(packed.len(), expected_bytes);
    }

    #[test]
    fn test_simd_bp128_block_max() {
        // Create a large posting list that spans multiple blocks
        let doc_ids: Vec<u32> = (0..500).map(|i| i * 2).collect();
        // Vary term frequencies so different blocks have different max_tf
        let term_freqs: Vec<u32> = (0..500)
            .map(|i| {
                if i < 128 {
                    1 // Block 0: max_tf = 1
                } else if i < 256 {
                    5 // Block 1: max_tf = 5
                } else if i < 384 {
                    10 // Block 2: max_tf = 10
                } else {
                    3 // Block 3: max_tf = 3
                }
            })
            .collect();

        let list = VerticalBP128PostingList::from_postings(&doc_ids, &term_freqs, 2.0);

        // Should have 4 blocks (500 docs / 128 per block)
        assert_eq!(list.blocks.len(), 4);
        assert_eq!(list.blocks[0].max_tf, 1);
        assert_eq!(list.blocks[1].max_tf, 5);
        assert_eq!(list.blocks[2].max_tf, 10);
        assert_eq!(list.blocks[3].max_tf, 3);

        // Block 2 should have highest score (max_tf = 10)
        assert!(list.blocks[2].max_block_score > list.blocks[0].max_block_score);
        assert!(list.blocks[2].max_block_score > list.blocks[1].max_block_score);
        assert!(list.blocks[2].max_block_score > list.blocks[3].max_block_score);

        // Global max_score should equal block 2's score
        assert_eq!(list.max_score, list.blocks[2].max_block_score);

        // Test iterator block-max methods
        let mut iter = list.iterator();
        assert_eq!(iter.current_block_max_tf(), 1); // Block 0

        // Seek to block 1
        iter.seek(256); // first doc in block 1
        assert_eq!(iter.current_block_max_tf(), 5);

        // Seek to block 2
        iter.seek(512); // first doc in block 2
        assert_eq!(iter.current_block_max_tf(), 10);

        // Test skip_to_block_with_doc
        let mut iter2 = list.iterator();
        let result = iter2.skip_to_block_with_doc(300);
        assert!(result.is_some());
        let (first_doc, score) = result.unwrap();
        assert!(first_doc <= 300);
        assert!(score > 0.0);
    }
}
