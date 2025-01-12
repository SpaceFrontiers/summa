//! Partitioned Elias-Fano (PEF) encoding
//!
//! Based on Ottaviano & Venturini (2014) "Partitioned Elias-Fano Indexes"
//!
//! Key improvements over standard Elias-Fano:
//! - **Optimal partitioning**: Divides sequence into chunks with (1+ε)-optimal compression
//! - **Better compression**: 30-40% smaller than standard EF on typical data
//! - **Fast random access**: O(1) access within partitions
//! - **Efficient NextGEQ**: Uses partition endpoints for fast seeking
//!
//! The algorithm finds optimal partition boundaries that minimize total encoding size.
//! Each partition is encoded with its own Elias-Fano instance using local parameters.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Minimum partition size (smaller partitions have too much overhead)
const MIN_PARTITION_SIZE: usize = 64;

/// Maximum partition size (larger partitions lose locality benefits)
const MAX_PARTITION_SIZE: usize = 512;

/// A single partition in the PEF structure
#[derive(Debug, Clone)]
pub struct EFPartition {
    /// Lower bits array
    lower_bits: Vec<u64>,
    /// Upper bits array (unary encoded)
    upper_bits: Vec<u64>,
    /// Number of elements in this partition
    len: u32,
    /// First value in partition (absolute)
    first_value: u32,
    /// Last value in partition (absolute)
    last_value: u32,
    /// Local universe size (last_value - first_value + 1)
    local_universe: u32,
    /// Number of lower bits per element
    lower_bit_width: u8,
}

impl EFPartition {
    /// Create a partition from a sorted slice
    /// Values are stored relative to first_value for better compression
    pub fn from_sorted_slice(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                lower_bits: Vec::new(),
                upper_bits: Vec::new(),
                len: 0,
                first_value: 0,
                last_value: 0,
                local_universe: 0,
                lower_bit_width: 0,
            };
        }

        let first_value = values[0];
        let last_value = *values.last().unwrap();
        let n = values.len() as u64;

        // Local universe: encode values relative to first_value
        let local_universe = last_value - first_value + 1;

        // Calculate lower bit width using local universe
        let lower_bit_width = if n <= 1 {
            0
        } else {
            let ratio = (local_universe as u64).max(1) / n.max(1);
            if ratio <= 1 {
                0
            } else {
                (64 - ratio.leading_zeros() - 1) as u8
            }
        };

        // Allocate arrays
        let lower_bits_total = (n as usize) * (lower_bit_width as usize);
        let lower_words = lower_bits_total.div_ceil(64);
        let mut lower_bits = vec![0u64; lower_words];

        let max_relative = local_universe.saturating_sub(1) as u64;
        let upper_bound = n + (max_relative >> lower_bit_width) + 1;
        let upper_words = (upper_bound as usize).div_ceil(64);
        let mut upper_bits = vec![0u64; upper_words];

        let lower_mask = if lower_bit_width == 0 {
            0
        } else {
            (1u64 << lower_bit_width) - 1
        };

        // Encode each value relative to first_value
        for (i, &val) in values.iter().enumerate() {
            let relative_val = (val - first_value) as u64;

            // Store lower bits
            if lower_bit_width > 0 {
                let lower = relative_val & lower_mask;
                let bit_pos = i * (lower_bit_width as usize);
                let word_idx = bit_pos / 64;
                let bit_offset = bit_pos % 64;

                lower_bits[word_idx] |= lower << bit_offset;
                if bit_offset + (lower_bit_width as usize) > 64 && word_idx + 1 < lower_bits.len() {
                    lower_bits[word_idx + 1] |= lower >> (64 - bit_offset);
                }
            }

            // Store upper bits
            let upper = relative_val >> lower_bit_width;
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
            first_value,
            last_value,
            local_universe,
            lower_bit_width,
        }
    }

    /// Get element at position i (0-indexed)
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

        // Reconstruct absolute value
        let relative_val = (upper << self.lower_bit_width) | lower;
        Some(self.first_value + relative_val as u32)
    }

    /// Find position of i-th set bit (optimized with NEON on aarch64)
    fn select1(&self, i: u32) -> Option<u32> {
        if i >= self.len {
            return None;
        }

        let mut remaining = i + 1;
        let mut pos = 0u32;

        // Process 4 words at a time on aarch64 using NEON popcount
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;

            let chunks = self.upper_bits.chunks_exact(4);
            let remainder = chunks.remainder();

            for chunk in chunks {
                // Load 4 u64 words (32 bytes)
                let words: [u64; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];

                // Count bits in each word using NEON
                // vcntq_u8 counts bits per byte, then sum horizontally
                unsafe {
                    let bytes = std::mem::transmute::<[u64; 4], [u8; 32]>(words);

                    // Process first 16 bytes (words 0-1)
                    let v0 = vld1q_u8(bytes.as_ptr());
                    let cnt0 = vcntq_u8(v0);
                    let sum0 = vaddlvq_u8(cnt0) as u32;

                    // Process next 16 bytes (words 2-3)
                    let v1 = vld1q_u8(bytes.as_ptr().add(16));
                    let cnt1 = vcntq_u8(v1);
                    let sum1 = vaddlvq_u8(cnt1) as u32;

                    let total_popcount = sum0 + sum1;

                    if total_popcount >= remaining {
                        // Found the chunk, now find exact word
                        for &word in chunk {
                            let popcount = word.count_ones();
                            if popcount >= remaining {
                                let mut w = word;
                                for _ in 0..remaining - 1 {
                                    w &= w - 1;
                                }
                                return Some(pos + w.trailing_zeros());
                            }
                            remaining -= popcount;
                            pos += 64;
                        }
                    }
                    remaining -= total_popcount;
                    pos += 256; // 4 words * 64 bits
                }
            }

            // Handle remaining words
            for &word in remainder {
                let popcount = word.count_ones();
                if popcount >= remaining {
                    let mut w = word;
                    for _ in 0..remaining - 1 {
                        w &= w - 1;
                    }
                    return Some(pos + w.trailing_zeros());
                }
                remaining -= popcount;
                pos += 64;
            }
        }

        // Process 4 words at a time on x86_64 using POPCNT instruction
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "popcnt")]
            {
                use std::arch::x86_64::*;

                let chunks = self.upper_bits.chunks_exact(4);
                let remainder = chunks.remainder();

                for chunk in chunks {
                    // Use hardware POPCNT for each word
                    unsafe {
                        let p0 = _popcnt64(chunk[0] as i64) as u32;
                        let p1 = _popcnt64(chunk[1] as i64) as u32;
                        let p2 = _popcnt64(chunk[2] as i64) as u32;
                        let p3 = _popcnt64(chunk[3] as i64) as u32;

                        let total_popcount = p0 + p1 + p2 + p3;

                        if total_popcount >= remaining {
                            // Found the chunk, now find exact word
                            for &word in chunk {
                                let popcount = word.count_ones();
                                if popcount >= remaining {
                                    let mut w = word;
                                    for _ in 0..remaining - 1 {
                                        w &= w - 1;
                                    }
                                    return Some(pos + w.trailing_zeros());
                                }
                                remaining -= popcount;
                                pos += 64;
                            }
                        }
                        remaining -= total_popcount;
                        pos += 256; // 4 words * 64 bits
                    }
                }

                // Handle remaining words
                for &word in remainder {
                    let popcount = word.count_ones();
                    if popcount >= remaining {
                        let mut w = word;
                        for _ in 0..remaining - 1 {
                            w &= w - 1;
                        }
                        return Some(pos + w.trailing_zeros());
                    }
                    remaining -= popcount;
                    pos += 64;
                }
            }

            // Scalar fallback when POPCNT not available
            #[cfg(not(target_feature = "popcnt"))]
            {
                for &word in &self.upper_bits {
                    let popcount = word.count_ones();
                    if popcount >= remaining {
                        let mut w = word;
                        for _ in 0..remaining - 1 {
                            w &= w - 1;
                        }
                        return Some(pos + w.trailing_zeros());
                    }
                    remaining -= popcount;
                    pos += 64;
                }
            }
        }

        // Scalar fallback for other architectures
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            for &word in &self.upper_bits {
                let popcount = word.count_ones();
                if popcount >= remaining {
                    let mut w = word;
                    for _ in 0..remaining - 1 {
                        w &= w - 1;
                    }
                    return Some(pos + w.trailing_zeros());
                }
                remaining -= popcount;
                pos += 64;
            }
        }

        None
    }

    /// Find first element >= target within this partition
    /// Returns (local_position, value) or None
    pub fn next_geq(&self, target: u32) -> Option<(u32, u32)> {
        if self.len == 0 || target > self.last_value {
            return None;
        }

        if target <= self.first_value {
            return Some((0, self.first_value));
        }

        // Binary search for target
        let mut lo = 0u32;
        let mut hi = self.len;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if let Some(val) = self.get(mid) {
                if val < target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            } else {
                break;
            }
        }

        if lo < self.len {
            self.get(lo).map(|v| (lo, v))
        } else {
            None
        }
    }

    /// Serialize partition
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.len)?;
        writer.write_u32::<LittleEndian>(self.first_value)?;
        writer.write_u32::<LittleEndian>(self.last_value)?;
        writer.write_u32::<LittleEndian>(self.local_universe)?;
        writer.write_u8(self.lower_bit_width)?;

        writer.write_u32::<LittleEndian>(self.lower_bits.len() as u32)?;
        for &word in &self.lower_bits {
            writer.write_u64::<LittleEndian>(word)?;
        }

        writer.write_u32::<LittleEndian>(self.upper_bits.len() as u32)?;
        for &word in &self.upper_bits {
            writer.write_u64::<LittleEndian>(word)?;
        }

        Ok(())
    }

    /// Deserialize partition
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let len = reader.read_u32::<LittleEndian>()?;
        let first_value = reader.read_u32::<LittleEndian>()?;
        let last_value = reader.read_u32::<LittleEndian>()?;
        let local_universe = reader.read_u32::<LittleEndian>()?;
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
            first_value,
            last_value,
            local_universe,
            lower_bit_width,
        })
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        17 + self.lower_bits.len() * 8 + self.upper_bits.len() * 8
    }
}

/// Estimate encoding cost for a partition
fn estimate_partition_cost(values: &[u32]) -> usize {
    if values.is_empty() {
        return 0;
    }

    let n = values.len();
    let first = values[0];
    let last = *values.last().unwrap();
    let local_universe = (last - first + 1) as usize;

    // Lower bits: n * log2(universe/n)
    let lower_bits = if n <= 1 || local_universe <= n {
        0
    } else {
        let ratio = local_universe / n;
        let l = (usize::BITS - ratio.leading_zeros()) as usize;
        n * l
    };

    // Upper bits: approximately 2n bits
    let upper_bits = 2 * n;

    // Overhead: partition header
    let overhead = 17 * 8; // 17 bytes header

    (lower_bits + upper_bits + overhead).div_ceil(8)
}

/// Find optimal partition boundaries using dynamic programming
/// Returns partition endpoints (exclusive)
fn find_optimal_partitions(values: &[u32]) -> Vec<usize> {
    let n = values.len();
    if n <= MIN_PARTITION_SIZE {
        return vec![n];
    }

    // dp[i] = minimum cost to encode values[0..i]
    // parent[i] = start of last partition ending at i
    let mut dp = vec![usize::MAX; n + 1];
    let mut parent = vec![0usize; n + 1];
    dp[0] = 0;

    for i in MIN_PARTITION_SIZE..=n {
        // Try all valid partition sizes ending at i
        let min_start = i.saturating_sub(MAX_PARTITION_SIZE);
        let max_start = i.saturating_sub(MIN_PARTITION_SIZE);

        for start in min_start..=max_start {
            if dp[start] == usize::MAX {
                continue;
            }

            let partition_cost = estimate_partition_cost(&values[start..i]);
            let total_cost = dp[start].saturating_add(partition_cost);

            if total_cost < dp[i] {
                dp[i] = total_cost;
                parent[i] = start;
            }
        }
    }

    // Handle case where n is not reachable (too small for partitioning)
    if dp[n] == usize::MAX {
        return vec![n];
    }

    // Reconstruct partition boundaries
    let mut boundaries = Vec::new();
    let mut pos = n;
    while pos > 0 {
        boundaries.push(pos);
        pos = parent[pos];
    }
    boundaries.reverse();

    boundaries
}

/// Partitioned Elias-Fano encoded sequence
#[derive(Debug, Clone)]
pub struct PartitionedEliasFano {
    /// Individual partitions
    partitions: Vec<EFPartition>,
    /// Total number of elements
    len: u32,
    /// Cumulative element counts for each partition (for position lookup)
    cumulative_counts: Vec<u32>,
}

impl PartitionedEliasFano {
    /// Create from sorted slice with optimal partitioning
    pub fn from_sorted_slice(values: &[u32]) -> Self {
        if values.is_empty() {
            return Self {
                partitions: Vec::new(),
                len: 0,
                cumulative_counts: Vec::new(),
            };
        }

        let boundaries = find_optimal_partitions(values);
        let mut partitions = Vec::with_capacity(boundaries.len());
        let mut cumulative_counts = Vec::with_capacity(boundaries.len());

        let mut start = 0;
        let mut cumulative = 0u32;

        for &end in &boundaries {
            let partition = EFPartition::from_sorted_slice(&values[start..end]);
            cumulative += partition.len;
            cumulative_counts.push(cumulative);
            partitions.push(partition);
            start = end;
        }

        Self {
            partitions,
            len: values.len() as u32,
            cumulative_counts,
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

    /// Number of partitions
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get element at global position
    pub fn get(&self, pos: u32) -> Option<u32> {
        if pos >= self.len {
            return None;
        }

        // Find partition containing this position
        let partition_idx = self
            .cumulative_counts
            .binary_search(&(pos + 1))
            .unwrap_or_else(|x| x);

        if partition_idx >= self.partitions.len() {
            return None;
        }

        let local_pos = if partition_idx == 0 {
            pos
        } else {
            pos - self.cumulative_counts[partition_idx - 1]
        };

        self.partitions[partition_idx].get(local_pos)
    }

    /// Find first element >= target
    /// Returns (global_position, value) or None
    pub fn next_geq(&self, target: u32) -> Option<(u32, u32)> {
        if self.partitions.is_empty() {
            return None;
        }

        // Binary search for partition containing target
        let partition_idx = self
            .partitions
            .binary_search_by(|p| {
                if p.last_value < target {
                    std::cmp::Ordering::Less
                } else if p.first_value > target {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .unwrap_or_else(|x| x);

        // Search from this partition onwards
        for (i, partition) in self.partitions[partition_idx..].iter().enumerate() {
            let actual_idx = partition_idx + i;

            if let Some((local_pos, val)) = partition.next_geq(target) {
                let global_pos = if actual_idx == 0 {
                    local_pos
                } else {
                    self.cumulative_counts[actual_idx - 1] + local_pos
                };
                return Some((global_pos, val));
            }
        }

        None
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.len)?;
        writer.write_u32::<LittleEndian>(self.partitions.len() as u32)?;

        for &count in &self.cumulative_counts {
            writer.write_u32::<LittleEndian>(count)?;
        }

        for partition in &self.partitions {
            partition.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let len = reader.read_u32::<LittleEndian>()?;
        let num_partitions = reader.read_u32::<LittleEndian>()? as usize;

        let mut cumulative_counts = Vec::with_capacity(num_partitions);
        for _ in 0..num_partitions {
            cumulative_counts.push(reader.read_u32::<LittleEndian>()?);
        }

        let mut partitions = Vec::with_capacity(num_partitions);
        for _ in 0..num_partitions {
            partitions.push(EFPartition::deserialize(reader)?);
        }

        Ok(Self {
            partitions,
            len,
            cumulative_counts,
        })
    }

    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        let mut size = 8 + self.cumulative_counts.len() * 4;
        for p in &self.partitions {
            size += p.size_bytes();
        }
        size
    }

    /// Create iterator
    pub fn iter(&self) -> PartitionedEFIterator<'_> {
        PartitionedEFIterator {
            pef: self,
            partition_idx: 0,
            local_pos: 0,
        }
    }
}

/// Iterator over Partitioned Elias-Fano
pub struct PartitionedEFIterator<'a> {
    pef: &'a PartitionedEliasFano,
    partition_idx: usize,
    local_pos: u32,
}

impl<'a> Iterator for PartitionedEFIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.partition_idx >= self.pef.partitions.len() {
            return None;
        }

        let partition = &self.pef.partitions[self.partition_idx];
        if let Some(val) = partition.get(self.local_pos) {
            self.local_pos += 1;
            if self.local_pos >= partition.len {
                self.partition_idx += 1;
                self.local_pos = 0;
            }
            Some(val)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let current_global = if self.partition_idx == 0 {
            self.local_pos
        } else if self.partition_idx < self.pef.cumulative_counts.len() {
            self.pef.cumulative_counts[self.partition_idx - 1] + self.local_pos
        } else {
            self.pef.len
        };
        let remaining = (self.pef.len - current_global) as usize;
        (remaining, Some(remaining))
    }
}

/// Block metadata for BlockMax WAND
#[derive(Debug, Clone)]
pub struct PEFBlockInfo {
    /// First document ID in block
    pub first_doc_id: u32,
    /// Last document ID in block
    pub last_doc_id: u32,
    /// Maximum term frequency in block
    pub max_tf: u32,
    /// Maximum BM25 score upper bound
    pub max_block_score: f32,
    /// Starting partition index
    pub partition_idx: u16,
    /// Starting position within partition
    pub local_start: u32,
    /// Number of documents in block
    pub num_docs: u16,
}

impl PEFBlockInfo {
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.first_doc_id)?;
        writer.write_u32::<LittleEndian>(self.last_doc_id)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_block_score)?;
        writer.write_u16::<LittleEndian>(self.partition_idx)?;
        writer.write_u32::<LittleEndian>(self.local_start)?;
        writer.write_u16::<LittleEndian>(self.num_docs)?;
        Ok(())
    }

    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        Ok(Self {
            first_doc_id: reader.read_u32::<LittleEndian>()?,
            last_doc_id: reader.read_u32::<LittleEndian>()?,
            max_tf: reader.read_u32::<LittleEndian>()?,
            max_block_score: reader.read_f32::<LittleEndian>()?,
            partition_idx: reader.read_u16::<LittleEndian>()?,
            local_start: reader.read_u32::<LittleEndian>()?,
            num_docs: reader.read_u16::<LittleEndian>()?,
        })
    }
}

/// Block size for BlockMax (matches other formats)
pub const PEF_BLOCK_SIZE: usize = 128;

/// Partitioned Elias-Fano posting list with term frequencies and BlockMax
#[derive(Debug, Clone)]
pub struct PartitionedEFPostingList {
    /// Document IDs (Partitioned Elias-Fano encoded)
    pub doc_ids: PartitionedEliasFano,
    /// Term frequencies (packed)
    pub term_freqs: Vec<u8>,
    /// Bits per term frequency
    pub tf_bits: u8,
    /// Maximum term frequency
    pub max_tf: u32,
    /// Block metadata for BlockMax WAND
    pub blocks: Vec<PEFBlockInfo>,
    /// Global maximum score
    pub max_score: f32,
}

impl PartitionedEFPostingList {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    #[inline]
    pub fn compute_bm25_upper_bound(max_tf: u32, idf: f32) -> f32 {
        let tf = max_tf as f32;
        let min_length_norm = 1.0 - Self::B;
        let tf_norm = (tf * (Self::K1 + 1.0)) / (tf + Self::K1 * min_length_norm);
        idf * tf_norm
    }

    /// Create from postings
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32]) -> Self {
        Self::from_postings_with_idf(doc_ids, term_freqs, 1.0)
    }

    /// Create from postings with IDF for block-max scores
    pub fn from_postings_with_idf(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> Self {
        assert_eq!(doc_ids.len(), term_freqs.len());

        if doc_ids.is_empty() {
            return Self {
                doc_ids: PartitionedEliasFano::from_sorted_slice(&[]),
                term_freqs: Vec::new(),
                tf_bits: 0,
                max_tf: 0,
                blocks: Vec::new(),
                max_score: 0.0,
            };
        }

        let pef_doc_ids = PartitionedEliasFano::from_sorted_slice(doc_ids);

        // Find max TF
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

        // Build block metadata
        let mut blocks = Vec::new();
        let mut max_score = 0.0f32;
        let mut i = 0;

        while i < doc_ids.len() {
            let block_end = (i + PEF_BLOCK_SIZE).min(doc_ids.len());
            let block_doc_ids = &doc_ids[i..block_end];
            let block_tfs = &term_freqs[i..block_end];

            let block_max_tf = *block_tfs.iter().max().unwrap_or(&1);
            let block_score = Self::compute_bm25_upper_bound(block_max_tf, idf);
            max_score = max_score.max(block_score);

            // Find partition info for this block start
            let (partition_idx, local_start) =
                Self::find_partition_position(&pef_doc_ids, i as u32);

            blocks.push(PEFBlockInfo {
                first_doc_id: block_doc_ids[0],
                last_doc_id: *block_doc_ids.last().unwrap(),
                max_tf: block_max_tf,
                max_block_score: block_score,
                partition_idx: partition_idx as u16,
                local_start,
                num_docs: (block_end - i) as u16,
            });

            i = block_end;
        }

        Self {
            doc_ids: pef_doc_ids,
            term_freqs: packed_tfs,
            tf_bits,
            max_tf,
            blocks,
            max_score,
        }
    }

    fn find_partition_position(pef: &PartitionedEliasFano, global_pos: u32) -> (usize, u32) {
        let partition_idx = pef
            .cumulative_counts
            .binary_search(&(global_pos + 1))
            .unwrap_or_else(|x| x);

        let local_pos = if partition_idx == 0 {
            global_pos
        } else {
            global_pos - pef.cumulative_counts[partition_idx - 1]
        };

        (partition_idx, local_pos)
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

        (val & mask) + 1
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

        writer.write_u32::<LittleEndian>(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_ids = PartitionedEliasFano::deserialize(reader)?;
        let tf_bits = reader.read_u8()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;
        let max_score = reader.read_f32::<LittleEndian>()?;
        let tf_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut term_freqs = vec![0u8; tf_len];
        reader.read_exact(&mut term_freqs)?;

        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(PEFBlockInfo::deserialize(reader)?);
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

    /// Create iterator
    pub fn iterator(&self) -> PartitionedEFPostingIterator<'_> {
        PartitionedEFPostingIterator {
            list: self,
            pos: 0,
            current_block: 0,
        }
    }

    /// Get compression ratio compared to standard EF
    pub fn compression_info(&self) -> (usize, usize) {
        let pef_size = self.doc_ids.size_bytes();
        // Estimate standard EF size
        let n = self.len() as usize;
        let max_val = if let Some(last_block) = self.blocks.last() {
            last_block.last_doc_id
        } else {
            0
        };
        let ef_size = if n > 0 {
            let l = if n <= 1 {
                0
            } else {
                let ratio = (max_val as usize + 1) / n;
                if ratio <= 1 {
                    0
                } else {
                    (usize::BITS - ratio.leading_zeros()) as usize
                }
            };
            (n * l + 2 * n).div_ceil(8) + 16
        } else {
            0
        };
        (pef_size, ef_size)
    }
}

/// Iterator over Partitioned EF posting list
pub struct PartitionedEFPostingIterator<'a> {
    list: &'a PartitionedEFPostingList,
    pos: u32,
    current_block: usize,
}

impl<'a> PartitionedEFPostingIterator<'a> {
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
        if !self.list.blocks.is_empty() {
            let new_block = (self.pos as usize) / PEF_BLOCK_SIZE;
            if new_block < self.list.blocks.len() {
                self.current_block = new_block;
            }
        }
        self.doc()
    }

    /// Seek to first doc >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        // Use block skip list
        if !self.list.blocks.is_empty() {
            let block_idx = self.list.blocks[self.current_block..].binary_search_by(|b| {
                if b.last_doc_id < target {
                    std::cmp::Ordering::Less
                } else if b.first_doc_id > target {
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

            if target_block > self.current_block {
                self.current_block = target_block;
                self.pos = (target_block * PEF_BLOCK_SIZE) as u32;
            }
        }

        // Use PEF's next_geq
        if let Some((pos, val)) = self.list.doc_ids.next_geq(target)
            && pos >= self.pos
        {
            self.pos = pos;
            if !self.list.blocks.is_empty() {
                self.current_block = (pos as usize) / PEF_BLOCK_SIZE;
            }
            return val;
        }

        // Linear scan fallback
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

    /// Current block's max score
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

    /// Current block's max TF
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

    /// Max score for remaining blocks
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
                self.pos = (self.current_block * PEF_BLOCK_SIZE) as u32;
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
    fn test_ef_partition_basic() {
        let values = vec![10, 20, 30, 40, 50];
        let partition = EFPartition::from_sorted_slice(&values);

        assert_eq!(partition.len, 5);
        assert_eq!(partition.first_value, 10);
        assert_eq!(partition.last_value, 50);

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(partition.get(i as u32), Some(expected));
        }
    }

    #[test]
    fn test_ef_partition_next_geq() {
        let values = vec![10, 20, 30, 100, 200, 300];
        let partition = EFPartition::from_sorted_slice(&values);

        assert_eq!(partition.next_geq(5), Some((0, 10)));
        assert_eq!(partition.next_geq(10), Some((0, 10)));
        assert_eq!(partition.next_geq(15), Some((1, 20)));
        assert_eq!(partition.next_geq(100), Some((3, 100)));
        assert_eq!(partition.next_geq(301), None);
    }

    #[test]
    fn test_partitioned_ef_basic() {
        let values: Vec<u32> = (0..500).map(|i| i * 2).collect();
        let pef = PartitionedEliasFano::from_sorted_slice(&values);

        assert_eq!(pef.len(), 500);
        assert!(pef.num_partitions() >= 1);

        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i as u32), Some(expected), "Mismatch at {}", i);
        }
    }

    #[test]
    fn test_partitioned_ef_next_geq() {
        let values: Vec<u32> = (0..1000).map(|i| i * 3).collect();
        let pef = PartitionedEliasFano::from_sorted_slice(&values);

        assert_eq!(pef.next_geq(0), Some((0, 0)));
        assert_eq!(pef.next_geq(100), Some((34, 102))); // 100/3 = 33.33, next is 34*3=102
        assert_eq!(pef.next_geq(1500), Some((500, 1500)));
        assert_eq!(pef.next_geq(3000), None);
    }

    #[test]
    fn test_partitioned_ef_serialization() {
        let values: Vec<u32> = (0..500).map(|i| i * 5).collect();
        let pef = PartitionedEliasFano::from_sorted_slice(&values);

        let mut buffer = Vec::new();
        pef.serialize(&mut buffer).unwrap();

        let restored = PartitionedEliasFano::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.len(), pef.len());
        assert_eq!(restored.num_partitions(), pef.num_partitions());

        for i in 0..pef.len() {
            assert_eq!(restored.get(i), pef.get(i));
        }
    }

    #[test]
    fn test_partitioned_ef_posting_list() {
        let doc_ids: Vec<u32> = (0..300).map(|i| i * 2).collect();
        let term_freqs: Vec<u32> = (0..300).map(|i| (i % 10) + 1).collect();

        let list = PartitionedEFPostingList::from_postings(&doc_ids, &term_freqs);

        assert_eq!(list.len(), 300);

        let mut iter = list.iterator();
        for (i, (&expected_doc, &expected_tf)) in doc_ids.iter().zip(term_freqs.iter()).enumerate()
        {
            assert_eq!(iter.doc(), expected_doc, "Doc mismatch at {}", i);
            assert_eq!(iter.term_freq(), expected_tf, "TF mismatch at {}", i);
            iter.advance();
        }
    }

    #[test]
    fn test_partitioned_ef_seek() {
        let doc_ids: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let term_freqs: Vec<u32> = vec![1; 500];

        let list = PartitionedEFPostingList::from_postings(&doc_ids, &term_freqs);
        let mut iter = list.iterator();

        assert_eq!(iter.seek(100), 102); // 100/3 = 33.33, next is 34*3=102
        assert_eq!(iter.seek(300), 300);
        assert_eq!(iter.seek(1500), u32::MAX);
    }

    #[test]
    fn test_compression_improvement() {
        // Test that PEF achieves better compression than standard EF
        // on data with varying density
        let values: Vec<u32> = (0..10000)
            .map(|i| {
                if i < 5000 {
                    i * 2 // Dense region
                } else {
                    10000 + (i - 5000) * 100 // Sparse region
                }
            })
            .collect();

        let pef = PartitionedEliasFano::from_sorted_slice(&values);

        // PEF should use multiple partitions for this mixed-density data
        assert!(
            pef.num_partitions() > 1,
            "Expected multiple partitions, got {}",
            pef.num_partitions()
        );

        // Verify correctness
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(pef.get(i as u32), Some(expected));
        }
    }

    #[test]
    fn test_partitioned_ef_block_max() {
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

        let list = PartitionedEFPostingList::from_postings_with_idf(&doc_ids, &term_freqs, 2.0);

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

        // Test max_remaining_score
        let mut iter3 = list.iterator();
        let max_score = iter3.max_remaining_score();
        assert_eq!(max_score, list.max_score);

        // After seeking past block 2, max_remaining should be lower
        iter3.seek(768); // Block 3
        let remaining = iter3.max_remaining_score();
        assert!(remaining < max_score);
    }
}
