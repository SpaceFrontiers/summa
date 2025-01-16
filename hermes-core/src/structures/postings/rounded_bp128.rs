//! Rounded BP128 posting list format with SIMD-friendly bit widths
//!
//! This format rounds bit widths to 8, 16, or 32 bits for faster SIMD decoding
//! at the cost of ~10-100% more space compared to exact bitpacking.
//!
//! Use this format when:
//! - Query latency is more important than index size
//! - You have sufficient storage/memory
//! - Your workload is read-heavy
//!
//! The tradeoff: ~2-4x faster decoding for ~20-60% larger posting lists.

use crate::structures::simd::{self, RoundedBitWidth};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Block size for rounded bitpacking (128 integers per block for SIMD alignment)
pub const ROUNDED_BP128_BLOCK_SIZE: usize = 128;

/// Rounded bitpacked block with skip info for BlockWAND
#[derive(Debug, Clone)]
pub struct RoundedBP128Block {
    /// Delta-encoded doc_ids (rounded bitpacked: 8/16/32 bits)
    pub doc_deltas: Vec<u8>,
    /// Bit width for doc deltas (always 0, 8, 16, or 32)
    pub doc_bit_width: u8,
    /// Term frequencies (rounded bitpacked: 8/16/32 bits)
    pub term_freqs: Vec<u8>,
    /// Bit width for term frequencies (always 0, 8, 16, or 32)
    pub tf_bit_width: u8,
    /// First doc_id in this block (absolute)
    pub first_doc_id: u32,
    /// Last doc_id in this block (absolute)
    pub last_doc_id: u32,
    /// Number of docs in this block
    pub num_docs: u16,
    /// Maximum term frequency in this block
    pub max_tf: u32,
    /// Maximum impact score in this block (for MaxScore/WAND)
    pub max_block_score: f32,
}

impl RoundedBP128Block {
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

    /// Decode doc_ids from this block using SIMD-friendly rounded unpacking
    pub fn decode_doc_ids(&self) -> Vec<u32> {
        let mut doc_ids = vec![0u32; self.num_docs as usize];
        self.decode_doc_ids_into(&mut doc_ids);
        doc_ids
    }

    /// Decode doc_ids into a pre-allocated buffer (avoids allocation)
    #[inline]
    pub fn decode_doc_ids_into(&self, output: &mut [u32]) -> usize {
        let n = self.num_docs as usize;

        if n == 0 {
            return 0;
        }

        output[0] = self.first_doc_id;

        if n == 1 {
            return 1;
        }

        // Use fused unpack + delta decode for best performance
        let rounded_width = RoundedBitWidth::from_u8(self.doc_bit_width);
        simd::unpack_rounded_delta_decode(
            &self.doc_deltas,
            rounded_width,
            output,
            self.first_doc_id,
            n,
        );

        n
    }

    /// Decode term frequencies using SIMD-friendly rounded unpacking
    pub fn decode_term_freqs(&self) -> Vec<u32> {
        let mut tfs = vec![0u32; self.num_docs as usize];
        self.decode_term_freqs_into(&mut tfs);
        tfs
    }

    /// Decode term frequencies into a pre-allocated buffer (avoids allocation)
    #[inline]
    pub fn decode_term_freqs_into(&self, output: &mut [u32]) -> usize {
        let n = self.num_docs as usize;

        if n == 0 {
            return 0;
        }

        // Unpack using rounded bit width (fast SIMD path)
        let rounded_width = RoundedBitWidth::from_u8(self.tf_bit_width);
        simd::unpack_rounded(&self.term_freqs, rounded_width, output, n);

        // Add 1 back (we stored tf-1)
        simd::add_one(output, n);

        n
    }
}

/// Rounded BP128 posting list with block-level skip info
///
/// Uses rounded bit widths (0, 8, 16, 32) for faster SIMD decoding
/// at the cost of larger index size.
#[derive(Debug, Clone)]
pub struct RoundedBP128PostingList {
    /// Blocks of postings
    pub blocks: Vec<RoundedBP128Block>,
    /// Total document count
    pub doc_count: u32,
    /// Maximum score across all blocks (for MaxScore pruning)
    pub max_score: f32,
}

impl RoundedBP128PostingList {
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
            let block_end = (i + ROUNDED_BP128_BLOCK_SIZE).min(doc_ids.len());
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
    #[inline]
    pub fn compute_bm25f_upper_bound(max_tf: u32, idf: f32, field_boost: f32) -> f32 {
        let tf = max_tf as f32;
        let min_length_norm = 1.0 - Self::B;
        let tf_norm =
            (tf * field_boost * (Self::K1 + 1.0)) / (tf * field_boost + Self::K1 * min_length_norm);
        idf * tf_norm
    }

    fn create_block(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> RoundedBP128Block {
        let num_docs = doc_ids.len();
        let first_doc_id = doc_ids[0];
        let last_doc_id = *doc_ids.last().unwrap();

        // Compute deltas (delta - 1 to save one bit since deltas are always >= 1)
        let mut deltas = [0u32; ROUNDED_BP128_BLOCK_SIZE];
        let mut max_delta = 0u32;
        for j in 1..num_docs {
            let delta = doc_ids[j] - doc_ids[j - 1] - 1;
            deltas[j - 1] = delta;
            max_delta = max_delta.max(delta);
        }

        // Compute max TF and prepare TF array (store tf-1)
        let mut tfs = [0u32; ROUNDED_BP128_BLOCK_SIZE];
        let mut max_tf = 0u32;

        for (j, &tf) in term_freqs.iter().enumerate() {
            tfs[j] = tf - 1; // Store tf-1
            max_tf = max_tf.max(tf);
        }

        let max_block_score = Self::compute_bm25f_upper_bound(max_tf, idf, 1.0);

        // Use rounded bit widths for SIMD-friendly decoding
        let exact_doc_bits = simd::bits_needed(max_delta);
        let exact_tf_bits = simd::bits_needed(max_tf.saturating_sub(1));

        let doc_rounded = RoundedBitWidth::from_exact(exact_doc_bits);
        let tf_rounded = RoundedBitWidth::from_exact(exact_tf_bits);

        // Pack with rounded bit widths
        let mut doc_deltas = vec![0u8; num_docs.saturating_sub(1) * doc_rounded.bytes_per_value()];
        if num_docs > 1 {
            simd::pack_rounded(&deltas[..num_docs - 1], doc_rounded, &mut doc_deltas);
        }

        let mut term_freqs_packed = vec![0u8; num_docs * tf_rounded.bytes_per_value()];
        simd::pack_rounded(&tfs[..num_docs], tf_rounded, &mut term_freqs_packed);

        RoundedBP128Block {
            doc_deltas,
            doc_bit_width: doc_rounded.as_u8(),
            term_freqs: term_freqs_packed,
            tf_bit_width: tf_rounded.as_u8(),
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
            blocks.push(RoundedBP128Block::deserialize(reader)?);
        }

        Ok(Self {
            blocks,
            doc_count,
            max_score,
        })
    }

    /// Create an iterator
    pub fn iterator(&self) -> RoundedBP128Iterator<'_> {
        RoundedBP128Iterator::new(self)
    }

    /// Get number of documents
    pub fn len(&self) -> u32 {
        self.doc_count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
}

/// Iterator over rounded BP128 posting list with block skipping support
pub struct RoundedBP128Iterator<'a> {
    posting_list: &'a RoundedBP128PostingList,
    current_block: usize,
    position_in_block: usize,
    /// Number of valid elements in current block
    current_block_len: usize,
    /// Pre-allocated buffer for decoded doc_ids (avoids allocation per block)
    decoded_doc_ids: Vec<u32>,
    /// Pre-allocated buffer for decoded term frequencies
    decoded_tfs: Vec<u32>,
}

impl<'a> RoundedBP128Iterator<'a> {
    pub fn new(posting_list: &'a RoundedBP128PostingList) -> Self {
        // Pre-allocate buffers to block size to avoid allocations during iteration
        let mut iter = Self {
            posting_list,
            current_block: 0,
            position_in_block: 0,
            current_block_len: 0,
            decoded_doc_ids: vec![0u32; ROUNDED_BP128_BLOCK_SIZE],
            decoded_tfs: vec![0u32; ROUNDED_BP128_BLOCK_SIZE],
        };

        if !posting_list.blocks.is_empty() {
            iter.decode_current_block();
        }

        iter
    }

    #[inline]
    fn decode_current_block(&mut self) {
        if self.current_block < self.posting_list.blocks.len() {
            let block = &self.posting_list.blocks[self.current_block];
            // Decode into pre-allocated buffers (no allocation!)
            self.current_block_len = block.decode_doc_ids_into(&mut self.decoded_doc_ids);
            block.decode_term_freqs_into(&mut self.decoded_tfs);
        } else {
            self.current_block_len = 0;
        }
    }

    /// Current document ID
    #[inline]
    pub fn doc(&self) -> u32 {
        if self.current_block >= self.posting_list.blocks.len() {
            return u32::MAX;
        }
        if self.position_in_block >= self.current_block_len {
            return u32::MAX;
        }
        self.decoded_doc_ids[self.position_in_block]
    }

    /// Current term frequency
    #[inline]
    pub fn term_freq(&self) -> u32 {
        if self.current_block >= self.posting_list.blocks.len() {
            return 0;
        }
        if self.position_in_block >= self.current_block_len {
            return 0;
        }
        self.decoded_tfs[self.position_in_block]
    }

    /// Advance to next posting
    #[inline]
    pub fn advance(&mut self) -> u32 {
        self.position_in_block += 1;

        if self.position_in_block >= self.current_block_len {
            self.current_block += 1;
            self.position_in_block = 0;

            if self.current_block < self.posting_list.blocks.len() {
                self.decode_current_block();
            }
        }

        self.doc()
    }

    /// Seek to first doc_id >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        // Skip blocks where last_doc_id < target
        while self.current_block < self.posting_list.blocks.len() {
            let block = &self.posting_list.blocks[self.current_block];
            if block.last_doc_id >= target {
                break;
            }
            self.current_block += 1;
            self.position_in_block = 0;
        }

        if self.current_block >= self.posting_list.blocks.len() {
            return u32::MAX;
        }

        // Decode block if needed (check if we're on the right block)
        let block = &self.posting_list.blocks[self.current_block];
        if self.current_block_len == 0
            || self.position_in_block >= self.current_block_len
            || (self.position_in_block == 0 && self.decoded_doc_ids[0] != block.first_doc_id)
        {
            self.decode_current_block();
            self.position_in_block = 0;
        }

        // Binary search within block
        let start = self.position_in_block;
        let slice = &self.decoded_doc_ids[start..self.current_block_len];
        match slice.binary_search(&target) {
            Ok(pos) => {
                self.position_in_block = start + pos;
            }
            Err(pos) => {
                if pos < slice.len() {
                    self.position_in_block = start + pos;
                } else {
                    // Move to next block
                    self.current_block += 1;
                    self.position_in_block = 0;
                    if self.current_block < self.posting_list.blocks.len() {
                        self.decode_current_block();
                        return self.seek(target);
                    }
                    return u32::MAX;
                }
            }
        }

        self.doc()
    }

    /// Get block max score for current block (for WAND/MaxScore)
    #[inline]
    pub fn block_max_score(&self) -> f32 {
        if self.current_block < self.posting_list.blocks.len() {
            self.posting_list.blocks[self.current_block].max_block_score
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_bp128_basic() {
        let doc_ids: Vec<u32> = vec![1, 5, 10, 15, 20];
        let term_freqs: Vec<u32> = vec![1, 2, 3, 4, 5];

        let posting_list = RoundedBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        assert_eq!(posting_list.doc_count, 5);

        let mut iter = posting_list.iterator();
        for (i, (&expected_doc, &expected_tf)) in doc_ids.iter().zip(term_freqs.iter()).enumerate()
        {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), expected_tf, "TF mismatch at {}", i);
            iter.advance();
        }
        assert_eq!(iter.doc(), u32::MAX);
    }

    #[test]
    fn test_rounded_bp128_large_block() {
        // Test with a full 128-element block
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 5 + 100).collect();
        let term_freqs: Vec<u32> = vec![1; 128];

        let posting_list = RoundedBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let decoded = posting_list.blocks[0].decode_doc_ids();

        assert_eq!(decoded.len(), 128);
        for (i, (&expected, &actual)) in doc_ids.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(expected, actual, "Mismatch at position {}", i);
        }
    }

    #[test]
    fn test_rounded_bp128_serialization() {
        let doc_ids: Vec<u32> = (0..200).map(|i| i * 7 + 100).collect();
        let term_freqs: Vec<u32> = (0..200).map(|i| (i % 5) as u32 + 1).collect();

        let posting_list = RoundedBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        let mut buffer = Vec::new();
        posting_list.serialize(&mut buffer).unwrap();

        let restored = RoundedBP128PostingList::deserialize(&mut &buffer[..]).unwrap();
        assert_eq!(restored.doc_count, posting_list.doc_count);

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
    fn test_rounded_bp128_seek() {
        let doc_ids: Vec<u32> = vec![10, 20, 30, 100, 200, 300, 1000, 2000];
        let term_freqs: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let posting_list = RoundedBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let mut iter = posting_list.iterator();

        assert_eq!(iter.seek(25), 30);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(500), 1000);
        assert_eq!(iter.seek(3000), u32::MAX);
    }

    #[test]
    fn test_rounded_bit_widths() {
        // Test that bit widths are actually rounded
        let doc_ids: Vec<u32> = (0..128).map(|i| i * 100).collect(); // Large gaps -> needs >8 bits
        let term_freqs: Vec<u32> = vec![1; 128]; // Small TFs -> 0 bits (all zeros after -1)

        let posting_list = RoundedBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let block = &posting_list.blocks[0];

        // Doc bit width should be rounded to 8, 16, or 32
        assert!(
            block.doc_bit_width == 0
                || block.doc_bit_width == 8
                || block.doc_bit_width == 16
                || block.doc_bit_width == 32,
            "Doc bit width {} is not rounded",
            block.doc_bit_width
        );

        // TF bit width should be rounded
        assert!(
            block.tf_bit_width == 0
                || block.tf_bit_width == 8
                || block.tf_bit_width == 16
                || block.tf_bit_width == 32,
            "TF bit width {} is not rounded",
            block.tf_bit_width
        );
    }

    #[test]
    fn test_rounded_vs_exact_correctness() {
        // Verify rounded produces same decoded values as exact
        use super::super::horizontal_bp128::HorizontalBP128PostingList;

        let doc_ids: Vec<u32> = (0..200).map(|i| i * 7 + 100).collect();
        let term_freqs: Vec<u32> = (0..200).map(|i| (i % 5) as u32 + 1).collect();

        let exact = HorizontalBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);
        let rounded = RoundedBP128PostingList::from_postings(&doc_ids, &term_freqs, 1.0);

        // Rounded should be larger (worse compression)
        let mut exact_buf = Vec::new();
        exact.serialize(&mut exact_buf).unwrap();
        let mut rounded_buf = Vec::new();
        rounded.serialize(&mut rounded_buf).unwrap();

        assert!(
            rounded_buf.len() >= exact_buf.len(),
            "Rounded ({}) should be >= exact ({})",
            rounded_buf.len(),
            exact_buf.len()
        );

        // But both should decode to the same values
        let mut exact_iter = exact.iterator();
        let mut rounded_iter = rounded.iterator();

        while exact_iter.doc() != u32::MAX {
            assert_eq!(exact_iter.doc(), rounded_iter.doc());
            assert_eq!(exact_iter.term_freq(), rounded_iter.term_freq());
            exact_iter.advance();
            rounded_iter.advance();
        }
        assert_eq!(rounded_iter.doc(), u32::MAX);
    }
}
