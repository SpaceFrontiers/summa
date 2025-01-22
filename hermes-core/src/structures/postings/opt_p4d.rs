//! OptP4D (Optimized Patched Frame-of-Reference Delta) posting list compression
//!
//! OptP4D is an improvement over PForDelta that finds the optimal bit width for each block
//! by trying all possible bit widths and selecting the one that minimizes total storage.
//!
//! Key features:
//! - Block-based compression (128 integers per block for SIMD alignment)
//! - Delta encoding for doc IDs
//! - Optimal bit-width selection per block
//! - Patched coding: exceptions (values that don't fit) stored separately
//! - Fast SIMD-friendly decoding with NEON (ARM) and SSE (x86) support
//!
//! Format per block:
//! - Header: bit_width (5 bits) + num_exceptions (7 bits) + first_doc_id (32 bits)
//! - Main array: 128 values packed at `bit_width` bits each
//! - Exceptions: [position (7 bits), high_bits (32 - bit_width bits)] for each exception

use crate::structures::simd;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Block size for OptP4D (128 integers for SIMD alignment)
pub const OPT_P4D_BLOCK_SIZE: usize = 128;

/// Maximum number of exceptions before we increase bit width
/// (keeping exceptions under ~10% of block for good compression)
const MAX_EXCEPTIONS_RATIO: f32 = 0.10;

/// Find the optimal bit width for a block of values
/// Returns (bit_width, exception_count, total_bits)
fn find_optimal_bit_width(values: &[u32]) -> (u8, usize, usize) {
    if values.is_empty() {
        return (0, 0, 0);
    }

    let n = values.len();
    let max_exceptions = ((n as f32) * MAX_EXCEPTIONS_RATIO).ceil() as usize;

    // Count how many values need each bit width
    let mut bit_counts = [0usize; 33]; // bit_counts[b] = count of values needing exactly b bits
    for &v in values {
        let bits = simd::bits_needed(v) as usize;
        bit_counts[bits] += 1;
    }

    // Compute cumulative counts: values that fit in b bits or less
    let mut cumulative = [0usize; 33];
    cumulative[0] = bit_counts[0];
    for b in 1..=32 {
        cumulative[b] = cumulative[b - 1] + bit_counts[b];
    }

    let mut best_bits = 32u8;
    let mut best_total = usize::MAX;
    let mut best_exceptions = 0usize;

    // Try each bit width and compute total storage
    for b in 0..=32u8 {
        let fitting = if b == 0 {
            bit_counts[0]
        } else {
            cumulative[b as usize]
        };
        let exceptions = n - fitting;

        // Skip if too many exceptions
        if exceptions > max_exceptions && b < 32 {
            continue;
        }

        // Calculate total bits:
        // - Main array: n * b bits
        // - Exceptions: exceptions * (7 bits position + (32 - b) bits high value)
        let main_bits = n * (b as usize);
        let exception_bits = if b < 32 {
            exceptions * (7 + (32 - b as usize))
        } else {
            0
        };
        let total = main_bits + exception_bits;

        if total < best_total {
            best_total = total;
            best_bits = b;
            best_exceptions = exceptions;
        }
    }

    (best_bits, best_exceptions, best_total)
}

/// Pack values into a bitpacked array with the given bit width (NewPFD/OptPFD style)
///
/// Following the paper "Decoding billions of integers per second through vectorization":
/// - Store the first b bits (low bits) of ALL values in the main array
/// - For exceptions (values >= 2^b), store only the HIGH (32-b) bits separately with positions
///
/// Returns the packed bytes and a list of exceptions (position, high_bits)
fn pack_with_exceptions(values: &[u32], bit_width: u8) -> (Vec<u8>, Vec<(u8, u32)>) {
    if bit_width == 0 {
        // All values must be 0, exceptions store full value
        let exceptions: Vec<(u8, u32)> = values
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, &v)| (i as u8, v)) // For b=0, high bits = full value
            .collect();
        return (Vec::new(), exceptions);
    }

    if bit_width >= 32 {
        // No exceptions possible, just pack all 32 bits
        let bytes_needed = values.len() * 4;
        let mut packed = vec![0u8; bytes_needed];
        for (i, &value) in values.iter().enumerate() {
            let bytes = value.to_le_bytes();
            packed[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        return (packed, Vec::new());
    }

    let mask = (1u64 << bit_width) - 1;
    let bytes_needed = (values.len() * bit_width as usize).div_ceil(8);
    let mut packed = vec![0u8; bytes_needed];
    let mut exceptions = Vec::new();

    let mut bit_pos = 0usize;
    for (i, &value) in values.iter().enumerate() {
        // Store lower b bits in main array (for ALL values, including exceptions)
        let low_bits = (value as u64) & mask;

        // Write low bits to packed array
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;

        let mut remaining_bits = bit_width as usize;
        let mut val = low_bits;
        let mut current_byte_idx = byte_idx;
        let mut current_bit_offset = bit_offset;

        while remaining_bits > 0 {
            let bits_in_byte = (8 - current_bit_offset).min(remaining_bits);
            let byte_mask = ((1u64 << bits_in_byte) - 1) as u8;
            packed[current_byte_idx] |= ((val as u8) & byte_mask) << current_bit_offset;
            val >>= bits_in_byte;
            remaining_bits -= bits_in_byte;
            current_byte_idx += 1;
            current_bit_offset = 0;
        }

        bit_pos += bit_width as usize;

        // Record exception: store only the HIGH (32-b) bits
        let fits = value <= mask as u32;
        if !fits {
            let high_bits = value >> bit_width;
            exceptions.push((i as u8, high_bits));
        }
    }

    (packed, exceptions)
}

/// Unpack values from a bitpacked array and apply exceptions (NewPFD/OptPFD style)
///
/// Following the paper "Decoding billions of integers per second through vectorization":
/// - Low b bits are stored in the main array for ALL values
/// - Exceptions store only the HIGH (32-b) bits
/// - Reconstruct: value = (high_bits << b) | low_bits
///
/// Uses SIMD acceleration for common bit widths (8, 16, 32)
fn unpack_with_exceptions(
    packed: &[u8],
    bit_width: u8,
    exceptions: &[(u8, u32)],
    count: usize,
    output: &mut [u32],
) {
    if bit_width == 0 {
        output[..count].fill(0);
    } else if bit_width == 8 {
        // SIMD-accelerated 8-bit unpacking
        simd::unpack_8bit(packed, output, count);
    } else if bit_width == 16 {
        // SIMD-accelerated 16-bit unpacking
        simd::unpack_16bit(packed, output, count);
    } else if bit_width >= 32 {
        // SIMD-accelerated 32-bit unpacking
        simd::unpack_32bit(packed, output, count);
        return; // No exceptions for 32-bit
    } else {
        // Generic bit unpacking for other bit widths
        let mask = (1u64 << bit_width) - 1;
        let mut bit_pos = 0usize;
        let input_ptr = packed.as_ptr();

        for out in output[..count].iter_mut() {
            let byte_idx = bit_pos >> 3;
            let bit_offset = bit_pos & 7;

            // Read 8 bytes at once for efficiency
            let word = if byte_idx + 8 <= packed.len() {
                unsafe { (input_ptr.add(byte_idx) as *const u64).read_unaligned() }
            } else {
                // Handle edge case near end of buffer
                let mut word = 0u64;
                for (i, &b) in packed[byte_idx..].iter().enumerate() {
                    word |= (b as u64) << (i * 8);
                }
                word
            };

            *out = ((word >> bit_offset) & mask) as u32;
            bit_pos += bit_width as usize;
        }
    }

    // Apply exceptions: combine high bits with low bits already in output
    // value = (high_bits << bit_width) | low_bits
    for &(pos, high_bits) in exceptions {
        if (pos as usize) < count {
            let low_bits = output[pos as usize];
            output[pos as usize] = (high_bits << bit_width) | low_bits;
        }
    }
}

/// Fused unpack + exceptions + delta decode for doc_ids
///
/// Combines unpacking, exception application, and prefix sum in a single pass.
/// Avoids intermediate buffer allocation.
#[inline]
fn unpack_exceptions_delta_decode(
    packed: &[u8],
    bit_width: u8,
    exceptions: &[(u8, u32)],
    output: &mut [u32],
    first_doc_id: u32,
    count: usize,
) {
    if count == 0 {
        return;
    }

    output[0] = first_doc_id;
    if count == 1 {
        return;
    }

    // Build exception lookup for O(1) access
    // Since exceptions are sparse (typically <5%), a simple linear scan is fine
    // But for very large blocks, we could use a small hashmap

    let mask = if bit_width < 32 {
        (1u64 << bit_width) - 1
    } else {
        u64::MAX
    };

    let mut carry = first_doc_id;

    // Fast path for SIMD-friendly bit widths
    match bit_width {
        0 => {
            // All zeros = consecutive doc IDs (gap of 1)
            for item in output.iter_mut().take(count).skip(1) {
                carry = carry.wrapping_add(1);
                *item = carry;
            }
        }
        8 => {
            // Unpack 8-bit, apply exceptions, delta decode in one pass
            for i in 0..count - 1 {
                let mut delta = packed[i] as u32;
                // Check for exception at this position
                for &(pos, high_bits) in exceptions {
                    if pos as usize == i {
                        delta |= high_bits << bit_width;
                        break;
                    }
                }
                carry = carry.wrapping_add(delta).wrapping_add(1);
                output[i + 1] = carry;
            }
        }
        16 => {
            // Unpack 16-bit, apply exceptions, delta decode in one pass
            for i in 0..count - 1 {
                let idx = i * 2;
                let mut delta = u16::from_le_bytes([packed[idx], packed[idx + 1]]) as u32;
                for &(pos, high_bits) in exceptions {
                    if pos as usize == i {
                        delta |= high_bits << bit_width;
                        break;
                    }
                }
                carry = carry.wrapping_add(delta).wrapping_add(1);
                output[i + 1] = carry;
            }
        }
        32 => {
            // 32-bit has no exceptions
            for i in 0..count - 1 {
                let idx = i * 4;
                let delta = u32::from_le_bytes([
                    packed[idx],
                    packed[idx + 1],
                    packed[idx + 2],
                    packed[idx + 3],
                ]);
                carry = carry.wrapping_add(delta).wrapping_add(1);
                output[i + 1] = carry;
            }
        }
        _ => {
            // Generic bit width
            let input_ptr = packed.as_ptr();
            let mut bit_pos = 0usize;

            for i in 0..count - 1 {
                let byte_idx = bit_pos >> 3;
                let bit_offset = bit_pos & 7;

                let word = unsafe { (input_ptr.add(byte_idx) as *const u64).read_unaligned() };
                let mut delta = ((word >> bit_offset) & mask) as u32;

                // Check for exception
                for &(pos, high_bits) in exceptions {
                    if pos as usize == i {
                        delta |= high_bits << bit_width;
                        break;
                    }
                }

                carry = carry.wrapping_add(delta).wrapping_add(1);
                output[i + 1] = carry;
                bit_pos += bit_width as usize;
            }
        }
    }
}

/// A single OptP4D block
#[derive(Debug, Clone)]
pub struct OptP4DBlock {
    /// First doc_id in this block (absolute)
    pub first_doc_id: u32,
    /// Last doc_id in this block (absolute)
    pub last_doc_id: u32,
    /// Number of documents in this block
    pub num_docs: u16,
    /// Bit width for delta encoding
    pub doc_bit_width: u8,
    /// Bit width for term frequencies
    pub tf_bit_width: u8,
    /// Maximum term frequency in this block
    pub max_tf: u32,
    /// Maximum block score for WAND/MaxScore
    pub max_block_score: f32,
    /// Packed doc deltas
    pub doc_deltas: Vec<u8>,
    /// Doc delta exceptions: (position, full_delta)
    pub doc_exceptions: Vec<(u8, u32)>,
    /// Packed term frequencies
    pub term_freqs: Vec<u8>,
    /// TF exceptions: (position, full_tf)
    pub tf_exceptions: Vec<(u8, u32)>,
}

impl OptP4DBlock {
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

        // Write doc exceptions
        writer.write_u8(self.doc_exceptions.len() as u8)?;
        for &(pos, val) in &self.doc_exceptions {
            writer.write_u8(pos)?;
            writer.write_u32::<LittleEndian>(val)?;
        }

        // Write term freqs
        writer.write_u16::<LittleEndian>(self.term_freqs.len() as u16)?;
        writer.write_all(&self.term_freqs)?;

        // Write tf exceptions
        writer.write_u8(self.tf_exceptions.len() as u8)?;
        for &(pos, val) in &self.tf_exceptions {
            writer.write_u8(pos)?;
            writer.write_u32::<LittleEndian>(val)?;
        }

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

        // Read doc deltas
        let doc_deltas_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut doc_deltas = vec![0u8; doc_deltas_len];
        reader.read_exact(&mut doc_deltas)?;

        // Read doc exceptions
        let num_doc_exceptions = reader.read_u8()? as usize;
        let mut doc_exceptions = Vec::with_capacity(num_doc_exceptions);
        for _ in 0..num_doc_exceptions {
            let pos = reader.read_u8()?;
            let val = reader.read_u32::<LittleEndian>()?;
            doc_exceptions.push((pos, val));
        }

        // Read term freqs
        let term_freqs_len = reader.read_u16::<LittleEndian>()? as usize;
        let mut term_freqs = vec![0u8; term_freqs_len];
        reader.read_exact(&mut term_freqs)?;

        // Read tf exceptions
        let num_tf_exceptions = reader.read_u8()? as usize;
        let mut tf_exceptions = Vec::with_capacity(num_tf_exceptions);
        for _ in 0..num_tf_exceptions {
            let pos = reader.read_u8()?;
            let val = reader.read_u32::<LittleEndian>()?;
            tf_exceptions.push((pos, val));
        }

        Ok(Self {
            first_doc_id,
            last_doc_id,
            num_docs,
            doc_bit_width,
            tf_bit_width,
            max_tf,
            max_block_score,
            doc_deltas,
            doc_exceptions,
            term_freqs,
            tf_exceptions,
        })
    }

    /// Decode doc_ids from this block using SIMD-accelerated delta decoding
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

        // Fused unpack + exceptions + delta decode - no intermediate buffer
        unpack_exceptions_delta_decode(
            &self.doc_deltas,
            self.doc_bit_width,
            &self.doc_exceptions,
            output,
            self.first_doc_id,
            count,
        );

        count
    }

    /// Decode term frequencies from this block using SIMD acceleration
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

        // Unpack TFs with exceptions (SIMD-accelerated for 8/16/32-bit)
        unpack_with_exceptions(
            &self.term_freqs,
            self.tf_bit_width,
            &self.tf_exceptions,
            count,
            output,
        );

        // TF is stored as tf-1, so add 1 back using SIMD
        simd::add_one(output, count);

        count
    }
}

/// OptP4D posting list
#[derive(Debug, Clone)]
pub struct OptP4DPostingList {
    /// Blocks of postings
    pub blocks: Vec<OptP4DBlock>,
    /// Total document count
    pub doc_count: u32,
    /// Maximum score across all blocks
    pub max_score: f32,
}

impl OptP4DPostingList {
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
            let block_end = (i + OPT_P4D_BLOCK_SIZE).min(doc_ids.len());
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

    fn create_block(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> OptP4DBlock {
        let num_docs = doc_ids.len();
        let first_doc_id = doc_ids[0];
        let last_doc_id = *doc_ids.last().unwrap();

        // Compute deltas using stack array (delta - 1 to save one bit)
        let mut deltas = [0u32; OPT_P4D_BLOCK_SIZE];
        for j in 1..num_docs {
            deltas[j - 1] = doc_ids[j] - doc_ids[j - 1] - 1;
        }

        // Find optimal bit width for deltas
        let (doc_bit_width, _, _) = find_optimal_bit_width(&deltas[..num_docs.saturating_sub(1)]);
        let (doc_deltas, doc_exceptions) =
            pack_with_exceptions(&deltas[..num_docs.saturating_sub(1)], doc_bit_width);

        // Compute max TF and prepare TF array using stack array (store tf-1)
        let mut tfs = [0u32; OPT_P4D_BLOCK_SIZE];
        let mut max_tf = 0u32;

        for (j, &tf) in term_freqs.iter().enumerate() {
            tfs[j] = tf - 1; // Store tf-1
            max_tf = max_tf.max(tf);
        }

        // Find optimal bit width for TFs
        let (tf_bit_width, _, _) = find_optimal_bit_width(&tfs[..num_docs]);
        let (term_freqs_packed, tf_exceptions) =
            pack_with_exceptions(&tfs[..num_docs], tf_bit_width);

        // BM25F upper bound score
        let max_block_score = crate::query::bm25_upper_bound(max_tf as f32, idf);

        OptP4DBlock {
            first_doc_id,
            last_doc_id,
            num_docs: num_docs as u16,
            doc_bit_width,
            tf_bit_width,
            max_tf,
            max_block_score,
            doc_deltas,
            doc_exceptions,
            term_freqs: term_freqs_packed,
            tf_exceptions,
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
            blocks.push(OptP4DBlock::deserialize(reader)?);
        }

        Ok(Self {
            blocks,
            doc_count,
            max_score,
        })
    }

    /// Get document count
    pub fn len(&self) -> u32 {
        self.doc_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Create an iterator
    pub fn iterator(&self) -> OptP4DIterator<'_> {
        OptP4DIterator::new(self)
    }
}

/// Iterator over OptP4D posting list
pub struct OptP4DIterator<'a> {
    posting_list: &'a OptP4DPostingList,
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

impl<'a> OptP4DIterator<'a> {
    pub fn new(posting_list: &'a OptP4DPostingList) -> Self {
        // Pre-allocate buffers to block size to avoid allocations during iteration
        let mut iter = Self {
            posting_list,
            current_block: 0,
            current_block_len: 0,
            block_doc_ids: vec![0u32; OPT_P4D_BLOCK_SIZE],
            block_term_freqs: vec![0u32; OPT_P4D_BLOCK_SIZE],
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

    /// Seek to first doc >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        if self.exhausted {
            return u32::MAX;
        }

        // Skip blocks where last_doc_id < target
        while self.current_block < self.posting_list.blocks.len() {
            let block = &self.posting_list.blocks[self.current_block];
            if block.last_doc_id >= target {
                break;
            }
            self.current_block += 1;
        }

        if self.current_block >= self.posting_list.blocks.len() {
            self.exhausted = true;
            return u32::MAX;
        }

        // Decode block if needed
        if self.current_block_len == 0 || self.current_block != self.posting_list.blocks.len() - 1 {
            self.decode_current_block();
        }

        // Binary search within block
        match self.block_doc_ids[self.pos_in_block..self.current_block_len].binary_search(&target) {
            Ok(idx) => {
                self.pos_in_block += idx;
            }
            Err(idx) => {
                self.pos_in_block += idx;
                if self.pos_in_block >= self.current_block_len {
                    // Move to next block
                    self.current_block += 1;
                    if self.current_block >= self.posting_list.blocks.len() {
                        self.exhausted = true;
                        return u32::MAX;
                    }
                    self.decode_current_block();
                }
            }
        }

        self.doc()
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
        assert_eq!(simd::bits_needed(4), 3);
        assert_eq!(simd::bits_needed(255), 8);
        assert_eq!(simd::bits_needed(256), 9);
        assert_eq!(simd::bits_needed(u32::MAX), 32);
    }

    #[test]
    fn test_find_optimal_bit_width() {
        // All zeros
        let values = vec![0u32; 100];
        let (bits, exceptions, _) = find_optimal_bit_width(&values);
        assert_eq!(bits, 0);
        assert_eq!(exceptions, 0);

        // All small values
        let values: Vec<u32> = (0..100).map(|i| i % 16).collect();
        let (bits, _, _) = find_optimal_bit_width(&values);
        assert!(bits <= 4);

        // Mix with outliers
        let mut values: Vec<u32> = (0..100).map(|i| i % 16).collect();
        values[50] = 1_000_000; // outlier
        let (bits, exceptions, _) = find_optimal_bit_width(&values);
        assert!(bits < 20); // Should use small bit width with exception
        assert!(exceptions >= 1);
    }

    #[test]
    fn test_pack_unpack_with_exceptions() {
        let values = vec![1, 2, 3, 255, 4, 5, 1000, 6, 7, 8];
        let (packed, exceptions) = pack_with_exceptions(&values, 4);

        let mut output = vec![0u32; values.len()];
        unpack_with_exceptions(&packed, 4, &exceptions, values.len(), &mut output);

        assert_eq!(output, values);
    }

    #[test]
    fn test_opt_p4d_posting_list_small() {
        let doc_ids: Vec<u32> = (0..100).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = vec![1; 100];

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        assert_eq!(list.len(), 100);
        assert_eq!(list.blocks.len(), 1);

        // Verify iteration
        let mut iter = list.iterator();
        for (i, &expected) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected, "Mismatch at {}", i);
            assert_eq!(iter.term_freq(), 1);
            iter.advance();
        }
        assert_eq!(iter.doc(), u32::MAX);
    }

    #[test]
    fn test_opt_p4d_posting_list_large() {
        let doc_ids: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..500).map(|i| (i % 10) + 1).collect();

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        assert_eq!(list.len(), 500);
        assert_eq!(list.blocks.len(), 4); // 500 / 128 = 3.9 -> 4 blocks

        // Verify iteration
        let mut iter = list.iterator();
        for (i, &expected) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected, "Mismatch at {}", i);
            assert_eq!(iter.term_freq(), term_freqs[i]);
            iter.advance();
        }
    }

    #[test]
    fn test_opt_p4d_seek() {
        let doc_ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 1000, 2000];
        let term_freqs: Vec<u32> = vec![1; 8];

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut iter = list.iterator();

        assert_eq!(iter.seek(25), 30);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(500), 1000);
        assert_eq!(iter.seek(3000), u32::MAX);
    }

    #[test]
    fn test_opt_p4d_serialization() {
        let doc_ids: Vec<u32> = (0..200).map(|i| i * 5).collect();
        let term_freqs: Vec<u32> = (0..200).map(|i| (i % 5) + 1).collect();

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let restored = OptP4DPostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.len(), list.len());
        assert_eq!(restored.blocks.len(), list.blocks.len());

        // Verify iteration matches
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
    fn test_opt_p4d_with_outliers() {
        // Create data with some outliers to test exception handling
        let mut doc_ids: Vec<u32> = (0..128).map(|i| i * 2).collect();
        doc_ids[64] = 1_000_000; // Large outlier

        // Fix: ensure doc_ids are sorted
        doc_ids.sort();

        let term_freqs: Vec<u32> = vec![1; 128];

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        // Verify the outlier is handled correctly
        let mut iter = list.iterator();
        let mut found_outlier = false;
        while iter.doc() != u32::MAX {
            if iter.doc() == 1_000_000 {
                found_outlier = true;
            }
            iter.advance();
        }
        assert!(found_outlier, "Outlier value should be preserved");
    }

    #[test]
    fn test_opt_p4d_simd_full_blocks() {
        // Test with multiple full 128-integer blocks to exercise SIMD paths
        let doc_ids: Vec<u32> = (0..1024).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = (0..1024).map(|i| (i % 20) + 1).collect();

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        assert_eq!(list.len(), 1024);
        assert_eq!(list.blocks.len(), 8); // 1024 / 128 = 8 full blocks

        // Verify all values are decoded correctly
        let mut iter = list.iterator();
        for (i, &expected_doc) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), term_freqs[i], "TF mismatch at {}", i);
            iter.advance();
        }
        assert_eq!(iter.doc(), u32::MAX);
    }

    #[test]
    fn test_opt_p4d_simd_8bit_values() {
        // Test with values that fit in 8 bits to exercise SIMD 8-bit unpack
        let doc_ids: Vec<u32> = (0..256).collect();
        let term_freqs: Vec<u32> = (0..256).map(|i| (i % 100) + 1).collect();

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        // Verify all values
        let mut iter = list.iterator();
        for (i, &expected_doc) in doc_ids.iter().enumerate() {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), term_freqs[i], "TF mismatch at {}", i);
            iter.advance();
        }
    }

    #[test]
    fn test_opt_p4d_simd_delta_decode() {
        // Test SIMD delta decoding with various gap sizes
        let mut doc_ids = Vec::with_capacity(512);
        let mut current = 0u32;
        for i in 0..512 {
            current += (i % 10) + 1; // Variable gaps
            doc_ids.push(current);
        }
        let term_freqs: Vec<u32> = vec![1; 512];

        let list = OptP4DPostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        // Verify delta decoding is correct
        let mut iter = list.iterator();
        for (i, &expected_doc) in doc_ids.iter().enumerate() {
            assert_eq!(
                iter.doc(),
                expected_doc,
                "Doc mismatch at {} (expected {}, got {})",
                i,
                expected_doc,
                iter.doc()
            );
            iter.advance();
        }
    }
}
