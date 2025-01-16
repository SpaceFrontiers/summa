//! Elias-Fano encoding for compressed sorted integer sequences
//!
//! Elias-Fano is a quasi-succinct encoding that achieves near-optimal space
//! while supporting O(1) access and fast NextGEQ operations.
//!
//! Space: 2n + n⌈log(m/n)⌉ bits where n = count, m = universe size
//!
//! Used by Google, Facebook, and Lucene for posting list compression.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Elias-Fano encoded monotone sequence
#[derive(Debug, Clone)]
pub struct EliasFano {
    /// Lower bits array (dense, l bits per element)
    lower_bits: Vec<u64>,
    /// Upper bits array (sparse, unary encoded)
    upper_bits: Vec<u64>,
    /// Number of elements
    len: u32,
    /// Universe size (max value + 1)
    universe: u64,
    /// Number of lower bits per element
    lower_bit_width: u8,
}

impl EliasFano {
    /// Create Elias-Fano encoding from a sorted sequence
    pub fn from_sorted_slice(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                lower_bits: Vec::new(),
                upper_bits: Vec::new(),
                len: 0,
                universe: 0,
                lower_bit_width: 0,
            };
        }

        let n = values.len() as u64;
        let max_val = *values.last().unwrap() as u64;
        let universe = max_val + 1;

        // Calculate lower bit width: l = max(0, ⌊log2(m/n)⌋)
        let lower_bit_width = if n == 0 {
            0
        } else {
            let ratio = universe.max(1) / n.max(1);
            if ratio <= 1 {
                0
            } else {
                (64 - ratio.leading_zeros() - 1) as u8
            }
        };

        // Allocate lower bits array
        let lower_bits_total = (n as usize) * (lower_bit_width as usize);
        let lower_words = lower_bits_total.div_ceil(64);
        let mut lower_bits = vec![0u64; lower_words];

        // Allocate upper bits array
        // Upper bits use unary encoding: n ones + (max_val >> l) zeros
        let upper_bound = n + (max_val >> lower_bit_width) + 1;
        let upper_words = (upper_bound as usize).div_ceil(64);
        let mut upper_bits = vec![0u64; upper_words];

        // Encode each value
        let lower_mask = if lower_bit_width == 0 {
            0
        } else {
            (1u64 << lower_bit_width) - 1
        };

        for (i, &val) in values.iter().enumerate() {
            let val = val as u64;

            // Store lower bits
            if lower_bit_width > 0 {
                let lower = val & lower_mask;
                let bit_pos = i * (lower_bit_width as usize);
                let word_idx = bit_pos / 64;
                let bit_offset = bit_pos % 64;

                lower_bits[word_idx] |= lower << bit_offset;
                if bit_offset + (lower_bit_width as usize) > 64 && word_idx + 1 < lower_bits.len() {
                    lower_bits[word_idx + 1] |= lower >> (64 - bit_offset);
                }
            }

            // Store upper bits (unary: position = i + (val >> l))
            let upper = val >> lower_bit_width;
            let upper_pos = (i as u64) + upper;
            let word_idx = (upper_pos / 64) as usize;
            let bit_offset = upper_pos % 64;
            if word_idx < upper_bits.len() {
                upper_bits[word_idx] |= 1u64 << bit_offset;
            }
        }

        Self {
            lower_bits,
            upper_bits,
            len: values.len() as u32,
            universe,
            lower_bit_width,
        }
    }

    /// Number of elements
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access element at position i (0-indexed)
    #[inline]
    pub fn get(&self, i: u32) -> Option<u32> {
        if i >= self.len {
            return None;
        }

        let i = i as usize;

        // Get lower bits
        let lower = if self.lower_bit_width == 0 {
            0u64
        } else {
            let bit_pos = i * (self.lower_bit_width as usize);
            let word_idx = bit_pos / 64;
            let bit_offset = bit_pos % 64;
            let lower_mask = (1u64 << self.lower_bit_width) - 1;

            let mut val = (self.lower_bits[word_idx] >> bit_offset) & lower_mask;
            if bit_offset + (self.lower_bit_width as usize) > 64
                && word_idx + 1 < self.lower_bits.len()
            {
                val |= (self.lower_bits[word_idx + 1] << (64 - bit_offset)) & lower_mask;
            }
            val
        };

        // Get upper bits via select1(i) - i
        let select_pos = self.select1(i as u32)?;
        let upper = (select_pos as u64) - (i as u64);

        Some(((upper << self.lower_bit_width) | lower) as u32)
    }

    /// Find position of i-th set bit (0-indexed)
    /// Uses POPCNT on x86_64 and efficient intrinsics on aarch64
    fn select1(&self, i: u32) -> Option<u32> {
        if i >= self.len {
            return None;
        }

        let mut remaining = i + 1;
        let mut pos = 0u32;

        // Process words using architecture-specific popcount
        for &word in self.upper_bits.iter() {
            let popcount = Self::popcount64(word);
            if popcount >= remaining {
                // Target is in this word - find the remaining-th set bit
                let bit_pos = Self::select_in_word(word, remaining);
                return Some(pos + bit_pos);
            }
            remaining -= popcount;
            pos += 64;
        }

        None
    }

    /// Fast popcount using architecture-specific intrinsics
    #[inline]
    fn popcount64(word: u64) -> u32 {
        // count_ones() compiles to POPCNT on x86_64 and efficient code on aarch64
        word.count_ones()
    }

    /// Find position of k-th set bit within a word (1-indexed k)
    #[inline]
    fn select_in_word(word: u64, k: u32) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            // Use PDEP instruction for O(1) select on BMI2-capable CPUs
            use std::arch::x86_64::_pdep_u64;
            let mask = 1u64 << (k - 1);
            let selected = unsafe { _pdep_u64(mask, word) };
            selected.trailing_zeros()
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            // Fallback: clear lowest set bits one by one
            let mut w = word;
            for _ in 0..k - 1 {
                w &= w - 1; // Clear lowest set bit
            }
            w.trailing_zeros()
        }
    }

    /// Find first element >= target (NextGEQ operation)
    /// Returns (position, value) or None if no such element exists
    pub fn next_geq(&self, target: u32) -> Option<(u32, u32)> {
        if self.len == 0 {
            return None;
        }

        let target = target as u64;

        // Get the bucket (upper bits of target)
        let target_upper = target >> self.lower_bit_width;
        let _target_lower = if self.lower_bit_width == 0 {
            0
        } else {
            target & ((1u64 << self.lower_bit_width) - 1)
        };

        // Find position via select0(target_upper)
        let bucket_start = self.select0(target_upper as u32);

        // Scan from bucket_start
        for pos in bucket_start..self.len {
            if let Some(val) = self.get(pos)
                && val as u64 >= target
            {
                return Some((pos, val));
            }
        }

        None
    }

    /// Find position after i-th zero bit (0-indexed)
    /// Returns the index in the original sequence where the i-th bucket starts
    fn select0(&self, i: u32) -> u32 {
        if i == 0 {
            return 0;
        }

        let mut zeros_seen = 0u32;
        let mut ones_seen = 0u32;

        for &word in &self.upper_bits {
            let zeros_in_word = 64 - word.count_ones();
            let ones_in_word = word.count_ones();

            if zeros_seen + zeros_in_word >= i {
                // Target zero is in this word
                let mut w = word;
                let mut bit_idx = 0u32;
                while bit_idx < 64 {
                    if (w & 1) == 0 {
                        zeros_seen += 1;
                        if zeros_seen == i {
                            // Found the i-th zero, return count of 1s seen so far
                            return ones_seen;
                        }
                    } else {
                        ones_seen += 1;
                    }
                    w >>= 1;
                    bit_idx += 1;
                }
            }
            zeros_seen += zeros_in_word;
            ones_seen += ones_in_word;
        }

        self.len
    }

    /// Serialize to bytes
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.len)?;
        writer.write_u64::<LittleEndian>(self.universe)?;
        writer.write_u8(self.lower_bit_width)?;

        // Write lower bits
        writer.write_u32::<LittleEndian>(self.lower_bits.len() as u32)?;
        for &word in &self.lower_bits {
            writer.write_u64::<LittleEndian>(word)?;
        }

        // Write upper bits
        writer.write_u32::<LittleEndian>(self.upper_bits.len() as u32)?;
        for &word in &self.upper_bits {
            writer.write_u64::<LittleEndian>(word)?;
        }

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let len = reader.read_u32::<LittleEndian>()?;
        let universe = reader.read_u64::<LittleEndian>()?;
        let lower_bit_width = reader.read_u8()?;

        let lower_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut lower_bits = Vec::with_capacity(lower_len);
        for _ in 0..lower_len {
            lower_bits.push(reader.read_u64::<LittleEndian>()?);
        }

        let upper_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut upper_bits = Vec::with_capacity(upper_len);
        for _ in 0..upper_len {
            upper_bits.push(reader.read_u64::<LittleEndian>()?);
        }

        Ok(Self {
            lower_bits,
            upper_bits,
            len,
            universe,
            lower_bit_width,
        })
    }

    /// Get approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        self.lower_bits.len() * 8 + self.upper_bits.len() * 8 + 16
    }

    /// Create an iterator
    pub fn iter(&self) -> EliasFanoIterator<'_> {
        EliasFanoIterator { ef: self, pos: 0 }
    }
}

/// Iterator over Elias-Fano encoded sequence
pub struct EliasFanoIterator<'a> {
    ef: &'a EliasFano,
    pos: u32,
}

impl<'a> Iterator for EliasFanoIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.ef.len {
            return None;
        }

        // Fast path: use cached upper position
        let val = self.ef.get(self.pos)?;
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.ef.len - self.pos) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EliasFanoIterator<'a> {}

/// Block size for Elias-Fano BlockMax (matches bitpacking for consistency)
pub const EF_BLOCK_SIZE: usize = 128;

/// Block metadata for BlockMax WAND optimization
#[derive(Debug, Clone)]
pub struct EFBlockInfo {
    /// First document ID in this block
    pub first_doc_id: u32,
    /// Last document ID in this block
    pub last_doc_id: u32,
    /// Maximum term frequency in this block
    pub max_tf: u32,
    /// Maximum BM25 score upper bound for this block
    pub max_block_score: f32,
    /// Starting position (index) in the posting list
    pub start_pos: u32,
    /// Number of documents in this block
    pub num_docs: u16,
}

impl EFBlockInfo {
    /// Serialize block info
    pub fn serialize<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        use byteorder::WriteBytesExt;
        writer.write_u32::<LittleEndian>(self.first_doc_id)?;
        writer.write_u32::<LittleEndian>(self.last_doc_id)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_block_score)?;
        writer.write_u32::<LittleEndian>(self.start_pos)?;
        writer.write_u16::<LittleEndian>(self.num_docs)?;
        Ok(())
    }

    /// Deserialize block info
    pub fn deserialize<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        use byteorder::ReadBytesExt;
        Ok(Self {
            first_doc_id: reader.read_u32::<LittleEndian>()?,
            last_doc_id: reader.read_u32::<LittleEndian>()?,
            max_tf: reader.read_u32::<LittleEndian>()?,
            max_block_score: reader.read_f32::<LittleEndian>()?,
            start_pos: reader.read_u32::<LittleEndian>()?,
            num_docs: reader.read_u16::<LittleEndian>()?,
        })
    }
}

/// Elias-Fano encoded posting list with term frequencies and BlockMax support
#[derive(Debug, Clone)]
pub struct EliasFanoPostingList {
    /// Document IDs (Elias-Fano encoded)
    pub doc_ids: EliasFano,
    /// Term frequencies (packed, using minimal bits)
    pub term_freqs: Vec<u8>,
    /// Bits per term frequency
    pub tf_bits: u8,
    /// Maximum term frequency (for BM25 upper bound)
    pub max_tf: u32,
    /// Block metadata for BlockMax WAND
    pub blocks: Vec<EFBlockInfo>,
    /// Global maximum score across all blocks
    pub max_score: f32,
}

impl EliasFanoPostingList {
    /// BM25 parameters for block-max score calculation
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    /// Compute BM25 upper bound score for a given max_tf and IDF
    #[inline]
    pub fn compute_bm25_upper_bound(max_tf: u32, idf: f32) -> f32 {
        let tf = max_tf as f32;
        // Conservative upper bound: assume dl=0, so length_norm = 1 - b = 0.25
        let min_length_norm = 1.0 - Self::B;
        let tf_norm = (tf * (Self::K1 + 1.0)) / (tf + Self::K1 * min_length_norm);
        idf * tf_norm
    }

    /// Create from doc_ids and term frequencies (without IDF - for backwards compatibility)
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32]) -> Self {
        Self::from_postings_with_idf(doc_ids, term_freqs, 1.0)
    }

    /// Create from doc_ids and term frequencies with IDF for block-max scores
    pub fn from_postings_with_idf(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> Self {
        assert_eq!(doc_ids.len(), term_freqs.len());

        if doc_ids.is_empty() {
            return Self {
                doc_ids: EliasFano::from_sorted_slice(&[]),
                term_freqs: Vec::new(),
                tf_bits: 0,
                max_tf: 0,
                blocks: Vec::new(),
                max_score: 0.0,
            };
        }

        let ef_doc_ids = EliasFano::from_sorted_slice(doc_ids);

        // Find max TF to determine bit width
        let max_tf = *term_freqs.iter().max().unwrap();
        let tf_bits = if max_tf == 0 {
            0
        } else {
            (32 - max_tf.leading_zeros()) as u8
        };

        // Pack term frequencies
        let total_bits = doc_ids.len() * (tf_bits as usize);
        let total_bytes = total_bits.div_ceil(8);
        let mut packed_tfs = vec![0u8; total_bytes];

        if tf_bits > 0 {
            for (i, &tf) in term_freqs.iter().enumerate() {
                let bit_pos = i * (tf_bits as usize);
                let byte_idx = bit_pos / 8;
                let bit_offset = bit_pos % 8;

                // Store tf - 1 to save one bit (tf is always >= 1)
                let val = tf.saturating_sub(1);

                packed_tfs[byte_idx] |= (val as u8) << bit_offset;
                if bit_offset + (tf_bits as usize) > 8 && byte_idx + 1 < packed_tfs.len() {
                    packed_tfs[byte_idx + 1] |= (val >> (8 - bit_offset)) as u8;
                }
                if bit_offset + (tf_bits as usize) > 16 && byte_idx + 2 < packed_tfs.len() {
                    packed_tfs[byte_idx + 2] |= (val >> (16 - bit_offset)) as u8;
                }
            }
        }

        // Build block metadata for BlockMax WAND
        let mut blocks = Vec::new();
        let mut max_score = 0.0f32;
        let mut i = 0;

        while i < doc_ids.len() {
            let block_end = (i + EF_BLOCK_SIZE).min(doc_ids.len());
            let block_doc_ids = &doc_ids[i..block_end];
            let block_tfs = &term_freqs[i..block_end];

            let block_max_tf = *block_tfs.iter().max().unwrap_or(&1);
            let block_score = Self::compute_bm25_upper_bound(block_max_tf, idf);
            max_score = max_score.max(block_score);

            blocks.push(EFBlockInfo {
                first_doc_id: block_doc_ids[0],
                last_doc_id: *block_doc_ids.last().unwrap(),
                max_tf: block_max_tf,
                max_block_score: block_score,
                start_pos: i as u32,
                num_docs: (block_end - i) as u16,
            });

            i = block_end;
        }

        Self {
            doc_ids: ef_doc_ids,
            term_freqs: packed_tfs,
            tf_bits,
            max_tf,
            blocks,
            max_score,
        }
    }

    /// Get term frequency at position
    pub fn get_tf(&self, pos: u32) -> u32 {
        if self.tf_bits == 0 || pos >= self.doc_ids.len() {
            return 1;
        }

        let bit_pos = (pos as usize) * (self.tf_bits as usize);
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mask = (1u32 << self.tf_bits) - 1;

        let mut val = (self.term_freqs[byte_idx] >> bit_offset) as u32;
        if bit_offset + (self.tf_bits as usize) > 8 && byte_idx + 1 < self.term_freqs.len() {
            val |= (self.term_freqs[byte_idx + 1] as u32) << (8 - bit_offset);
        }
        if bit_offset + (self.tf_bits as usize) > 16 && byte_idx + 2 < self.term_freqs.len() {
            val |= (self.term_freqs[byte_idx + 2] as u32) << (16 - bit_offset);
        }

        (val & mask) + 1 // Add 1 back since we stored tf-1
    }

    /// Number of documents
    pub fn len(&self) -> u32 {
        self.doc_ids.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.doc_ids.serialize(writer)?;
        writer.write_u8(self.tf_bits)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_score)?;
        writer.write_u32::<LittleEndian>(self.term_freqs.len() as u32)?;
        writer.write_all(&self.term_freqs)?;

        // Write block metadata
        writer.write_u32::<LittleEndian>(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_ids = EliasFano::deserialize(reader)?;
        let tf_bits = reader.read_u8()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;
        let max_score = reader.read_f32::<LittleEndian>()?;
        let tf_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut term_freqs = vec![0u8; tf_len];
        reader.read_exact(&mut term_freqs)?;

        // Read block metadata
        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(EFBlockInfo::deserialize(reader)?);
        }

        Ok(Self {
            doc_ids,
            term_freqs,
            tf_bits,
            max_tf,
            blocks,
            max_score,
        })
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get block containing position
    pub fn block_for_pos(&self, pos: u32) -> usize {
        (pos as usize) / EF_BLOCK_SIZE
    }

    /// Create iterator
    pub fn iterator(&self) -> EliasFanoPostingIterator<'_> {
        EliasFanoPostingIterator {
            list: self,
            pos: 0,
            current_block: 0,
        }
    }
}

/// Iterator over Elias-Fano posting list with BlockMax support
pub struct EliasFanoPostingIterator<'a> {
    list: &'a EliasFanoPostingList,
    pos: u32,
    current_block: usize,
}

impl<'a> EliasFanoPostingIterator<'a> {
    /// Current document ID
    pub fn doc(&self) -> u32 {
        self.list.doc_ids.get(self.pos).unwrap_or(u32::MAX)
    }

    /// Current term frequency
    pub fn term_freq(&self) -> u32 {
        self.list.get_tf(self.pos)
    }

    /// Advance to next document
    pub fn advance(&mut self) -> u32 {
        self.pos += 1;
        // Update current block if we've moved past it
        if !self.list.blocks.is_empty() {
            let new_block = self.list.block_for_pos(self.pos);
            if new_block < self.list.blocks.len() {
                self.current_block = new_block;
            }
        }
        self.doc()
    }

    /// Seek to first doc >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        // Use block skip list for faster seeking
        if !self.list.blocks.is_empty() {
            // Binary search to find the right block
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
                    let abs_idx = self.current_block + idx;
                    if abs_idx >= self.list.blocks.len() {
                        self.pos = self.list.len();
                        return u32::MAX;
                    }
                    abs_idx
                }
            };

            // Jump to block start if it's ahead
            if target_block > self.current_block {
                self.current_block = target_block;
                self.pos = self.list.blocks[target_block].start_pos;
            }
        }

        // Use Elias-Fano's next_geq for efficient seeking within block
        if let Some((pos, val)) = self.list.doc_ids.next_geq(target)
            && pos >= self.pos
        {
            self.pos = pos;
            if !self.list.blocks.is_empty() {
                self.current_block = self.list.block_for_pos(pos);
            }
            return val;
        }

        // Linear scan from current position (fallback)
        while self.pos < self.list.len() {
            let doc = self.doc();
            if doc >= target {
                return doc;
            }
            self.pos += 1;
        }

        u32::MAX
    }

    /// Check if exhausted
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.list.len()
    }

    /// Get current block's max score (for BlockMax WAND)
    pub fn current_block_max_score(&self) -> f32 {
        if self.is_exhausted() || self.list.blocks.is_empty() {
            return 0.0;
        }
        if self.current_block < self.list.blocks.len() {
            self.list.blocks[self.current_block].max_block_score
        } else {
            0.0
        }
    }

    /// Get current block's max term frequency (for BM25F recalculation)
    pub fn current_block_max_tf(&self) -> u32 {
        if self.is_exhausted() || self.list.blocks.is_empty() {
            return 0;
        }
        if self.current_block < self.list.blocks.len() {
            self.list.blocks[self.current_block].max_tf
        } else {
            0
        }
    }

    /// Get max score for remaining blocks (for MaxScore optimization)
    pub fn max_remaining_score(&self) -> f32 {
        if self.is_exhausted() || self.list.blocks.is_empty() {
            return 0.0;
        }
        self.list.blocks[self.current_block..]
            .iter()
            .map(|b| b.max_block_score)
            .fold(0.0f32, |a, b| a.max(b))
    }

    /// Skip to next block containing doc >= target (for BlockWAND)
    /// Returns (first_doc_in_block, block_max_score) or None if exhausted
    pub fn skip_to_block_with_doc(&mut self, target: u32) -> Option<(u32, f32)> {
        if self.list.blocks.is_empty() {
            return None;
        }

        while self.current_block < self.list.blocks.len() {
            let block = &self.list.blocks[self.current_block];
            if block.last_doc_id >= target {
                self.pos = block.start_pos;
                return Some((block.first_doc_id, block.max_block_score));
            }
            self.current_block += 1;
        }

        self.pos = self.list.len();
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elias_fano_basic() {
        let values = vec![2, 3, 5, 7, 11, 13, 24];
        let ef = EliasFano::from_sorted_slice(&values);

        assert_eq!(ef.len(), 7);

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(ef.get(i as u32), Some(expected), "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_elias_fano_next_geq() {
        let values = vec![10, 20, 30, 100, 200, 300, 1000];
        let ef = EliasFano::from_sorted_slice(&values);

        assert_eq!(ef.next_geq(5), Some((0, 10)));
        assert_eq!(ef.next_geq(10), Some((0, 10)));
        assert_eq!(ef.next_geq(15), Some((1, 20)));
        assert_eq!(ef.next_geq(100), Some((3, 100)));
        assert_eq!(ef.next_geq(500), Some((6, 1000)));
        assert_eq!(ef.next_geq(2000), None);
    }

    #[test]
    fn test_elias_fano_serialization() {
        let values: Vec<u32> = (0..1000).map(|i| i * 3).collect();
        let ef = EliasFano::from_sorted_slice(&values);

        let mut buffer = Vec::new();
        ef.serialize(&mut buffer).unwrap();

        let restored = EliasFano::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.len(), ef.len());
        for i in 0..ef.len() {
            assert_eq!(restored.get(i), ef.get(i));
        }
    }

    #[test]
    fn test_elias_fano_posting_list() {
        let doc_ids: Vec<u32> = vec![1, 5, 10, 50, 100, 500, 1000];
        let term_freqs: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];

        let list = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);

        assert_eq!(list.len(), 7);
        assert_eq!(list.max_tf, 7);

        let mut iter = list.iterator();
        for (i, (&expected_doc, &expected_tf)) in doc_ids.iter().zip(term_freqs.iter()).enumerate()
        {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), expected_tf, "TF mismatch at {}", i);
            iter.advance();
        }
    }

    #[test]
    fn test_elias_fano_iterator_seek() {
        let doc_ids: Vec<u32> = (0..100).map(|i| i * 10).collect();
        let term_freqs: Vec<u32> = vec![1; 100];

        let list = EliasFanoPostingList::from_postings(&doc_ids, &term_freqs);
        let mut iter = list.iterator();

        assert_eq!(iter.seek(55), 60);
        assert_eq!(iter.seek(100), 100);
        assert_eq!(iter.seek(999), u32::MAX);
    }

    #[test]
    fn test_elias_fano_block_max() {
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

        let list = EliasFanoPostingList::from_postings_with_idf(&doc_ids, &term_freqs, 2.0);

        // Should have 4 blocks (500 docs / 128 per block)
        assert_eq!(list.num_blocks(), 4);
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

        // Test raw Elias-Fano next_geq first
        let (pos, val) = list.doc_ids.next_geq(256).unwrap();
        assert_eq!(val, 256, "next_geq(256) should return 256, got {}", val);
        assert_eq!(pos, 128, "position of 256 should be 128, got {}", pos);

        // Test iterator block-max methods
        let mut iter = list.iterator();
        assert_eq!(iter.current_block_max_tf(), 1); // Block 0

        // Verify block boundaries
        // Block 0: positions 0-127, doc_ids 0-254 (i*2 where i=0..127)
        // Block 1: positions 128-255, doc_ids 256-510 (i*2 where i=128..255)
        // Block 2: positions 256-383, doc_ids 512-766 (i*2 where i=256..383)
        // Block 3: positions 384-499, doc_ids 768-998 (i*2 where i=384..499)
        assert_eq!(list.blocks[0].first_doc_id, 0);
        assert_eq!(list.blocks[0].last_doc_id, 254);
        assert_eq!(list.blocks[1].first_doc_id, 256);
        assert_eq!(list.blocks[1].last_doc_id, 510);

        // Seek to block 1 - use a doc_id clearly in block 1's range
        let doc = iter.seek(256); // first doc in block 1
        assert_eq!(doc, 256, "seek(256) should return 256, got {}", doc);
        // After seek, current_block should be updated
        let block_tf = iter.current_block_max_tf();
        assert_eq!(block_tf, 5, "block 1 max_tf should be 5, got {}", block_tf);

        // Seek to block 2
        let doc = iter.seek(512); // first doc in block 2
        assert_eq!(doc, 512, "seek(512) should return 512, got {}", doc);
        assert_eq!(iter.current_block_max_tf(), 10);
    }

    #[test]
    fn test_elias_fano_block_max_serialization() {
        let doc_ids: Vec<u32> = (0..300).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = (0..300).map(|i| (i % 10) + 1).collect();

        let list = EliasFanoPostingList::from_postings_with_idf(&doc_ids, &term_freqs, 1.5);

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let restored = EliasFanoPostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.len(), list.len());
        assert_eq!(restored.max_tf, list.max_tf);
        assert_eq!(restored.max_score, list.max_score);
        assert_eq!(restored.num_blocks(), list.num_blocks());

        // Verify block metadata
        for (orig, rest) in list.blocks.iter().zip(restored.blocks.iter()) {
            assert_eq!(orig.first_doc_id, rest.first_doc_id);
            assert_eq!(orig.last_doc_id, rest.last_doc_id);
            assert_eq!(orig.max_tf, rest.max_tf);
            assert_eq!(orig.max_block_score, rest.max_block_score);
        }

        // Verify iteration produces same results
        let mut iter1 = list.iterator();
        let mut iter2 = restored.iterator();

        while !iter1.is_exhausted() {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
        assert!(iter2.is_exhausted());
    }
}
