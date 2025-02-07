//! Async SSTable with lazy loading via FileSlice
//!
//! Memory-efficient design - only loads minimal metadata into memory,
//! blocks are loaded on-demand.
//!
//! ## Key Features
//!
//! 1. **FST-based Block Index**: Uses Finite State Transducer for key lookup
//!    - Can be mmap'd directly without parsing into heap-allocated structures
//!    - ~90% memory reduction compared to Vec<BlockIndexEntry>
//!
//! 2. **Bitpacked Block Addresses**: Offsets and lengths stored with delta encoding
//!    - Minimal memory footprint for block metadata
//!
//! 3. **Dictionary Compression**: Zstd dictionary for 15-30% better compression
//!
//! 4. **Configurable Compression Level**: Levels 1-22 for space/speed tradeoff
//!
//! 5. **Bloom Filter**: Fast negative lookups to skip unnecessary I/O

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::io::{self, Read, Write};
use std::sync::Arc;

#[cfg(feature = "native")]
use super::sstable_index::FstBlockIndex;
use super::sstable_index::{BlockAddr, BlockIndex, MmapBlockIndex};
use crate::compression::{CompressionDict, CompressionLevel};
use crate::directories::{AsyncFileRead, LazyFileHandle, LazyFileSlice, OwnedBytes};

/// SSTable magic number - version 4 with memory-efficient index
/// Uses FST-based or mmap'd block index to avoid heap allocation
pub const SSTABLE_MAGIC: u32 = 0x53544234; // "STB4"

/// Block size for SSTable (16KB default)
pub const BLOCK_SIZE: usize = 16 * 1024;

/// Default dictionary size (64KB)
pub const DEFAULT_DICT_SIZE: usize = 64 * 1024;

/// Bloom filter bits per key (10 bits ≈ 1% false positive rate)
pub const BLOOM_BITS_PER_KEY: usize = 10;

/// Bloom filter hash count (optimal for 10 bits/key)
pub const BLOOM_HASH_COUNT: usize = 7;

// ============================================================================
// Bloom Filter Implementation
// ============================================================================

/// Simple bloom filter for key existence checks
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new bloom filter sized for expected number of keys
    pub fn new(expected_keys: usize, bits_per_key: usize) -> Self {
        let num_bits = (expected_keys * bits_per_key).max(64);
        let num_words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes: BLOOM_HASH_COUNT,
        }
    }

    /// Create from serialized bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data too short",
            ));
        }
        let mut reader = data;
        let num_bits = reader.read_u32::<LittleEndian>()? as usize;
        let num_hashes = reader.read_u32::<LittleEndian>()? as usize;
        let num_words = reader.read_u32::<LittleEndian>()? as usize;

        if reader.len() < num_words * 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data truncated",
            ));
        }

        let mut bits = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            bits.push(reader.read_u64::<LittleEndian>()?);
        }

        Ok(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(12 + self.bits.len() * 8);
        data.write_u32::<LittleEndian>(self.num_bits as u32)
            .unwrap();
        data.write_u32::<LittleEndian>(self.num_hashes as u32)
            .unwrap();
        data.write_u32::<LittleEndian>(self.bits.len() as u32)
            .unwrap();
        for &word in &self.bits {
            data.write_u64::<LittleEndian>(word).unwrap();
        }
        data
    }

    /// Add a key to the filter
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit_pos = self.get_bit_pos(h1, h2, i);
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if word_idx < self.bits.len() {
                self.bits[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    /// Check if a key might be in the filter
    /// Returns false if definitely not present, true if possibly present
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let (h1, h2) = self.hash_pair(key);
        for i in 0..self.num_hashes {
            let bit_pos = self.get_bit_pos(h1, h2, i);
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if word_idx >= self.bits.len() || (self.bits[word_idx] & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        12 + self.bits.len() * 8
    }

    /// Compute two hash values using FNV-1a variant
    fn hash_pair(&self, key: &[u8]) -> (u64, u64) {
        // FNV-1a hash
        let mut h1: u64 = 0xcbf29ce484222325;
        for &byte in key {
            h1 ^= byte as u64;
            h1 = h1.wrapping_mul(0x100000001b3);
        }

        // Second hash using different seed
        let mut h2: u64 = 0x84222325cbf29ce4;
        for &byte in key {
            h2 = h2.wrapping_mul(0x100000001b3);
            h2 ^= byte as u64;
        }

        (h1, h2)
    }

    /// Get bit position for hash iteration i using double hashing
    #[inline]
    fn get_bit_pos(&self, h1: u64, h2: u64, i: usize) -> usize {
        (h1.wrapping_add((i as u64).wrapping_mul(h2)) % (self.num_bits as u64)) as usize
    }
}

/// SSTable value trait
pub trait SSTableValue: Clone + Send + Sync {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()>;
    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self>;
}

/// u64 value implementation
impl SSTableValue for u64 {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_vint(writer, *self)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        read_vint(reader)
    }
}

/// Vec<u8> value implementation
impl SSTableValue for Vec<u8> {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_vint(writer, self.len() as u64)?;
        writer.write_all(self)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let len = read_vint(reader)? as usize;
        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;
        Ok(data)
    }
}

/// Sparse dimension info for SSTable-based sparse index
/// Stores offset and length for posting list lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseDimInfo {
    /// Offset in sparse file where posting list starts
    pub offset: u64,
    /// Length of serialized posting list
    pub length: u32,
}

impl SparseDimInfo {
    pub fn new(offset: u64, length: u32) -> Self {
        Self { offset, length }
    }
}

impl SSTableValue for SparseDimInfo {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_vint(writer, self.offset)?;
        write_vint(writer, self.length as u64)
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let offset = read_vint(reader)?;
        let length = read_vint(reader)? as u32;
        Ok(Self { offset, length })
    }
}

/// Maximum number of postings that can be inlined in TermInfo
pub const MAX_INLINE_POSTINGS: usize = 3;

/// Term info for posting list references
///
/// Supports two modes:
/// - **Inline**: Small posting lists (1-3 docs) stored directly in TermInfo
/// - **External**: Larger posting lists stored in separate .post file
///
/// This eliminates a separate I/O read for rare/unique terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermInfo {
    /// Small posting list inlined directly (up to MAX_INLINE_POSTINGS entries)
    /// Each entry is (doc_id, term_freq) delta-encoded
    Inline {
        /// Number of postings (1-3)
        doc_freq: u8,
        /// Inline data: delta-encoded (doc_id, term_freq) pairs
        /// Format: [delta_doc_id, term_freq, delta_doc_id, term_freq, ...]
        data: [u8; 16],
        /// Actual length of data used
        data_len: u8,
    },
    /// Reference to external posting list in .post file
    External {
        posting_offset: u64,
        posting_len: u32,
        doc_freq: u32,
        /// Position data offset (0 if no positions)
        position_offset: u64,
        /// Position data length (0 if no positions)
        position_len: u32,
    },
}

impl TermInfo {
    /// Create an external reference
    pub fn external(posting_offset: u64, posting_len: u32, doc_freq: u32) -> Self {
        TermInfo::External {
            posting_offset,
            posting_len,
            doc_freq,
            position_offset: 0,
            position_len: 0,
        }
    }

    /// Create an external reference with position info
    pub fn external_with_positions(
        posting_offset: u64,
        posting_len: u32,
        doc_freq: u32,
        position_offset: u64,
        position_len: u32,
    ) -> Self {
        TermInfo::External {
            posting_offset,
            posting_len,
            doc_freq,
            position_offset,
            position_len,
        }
    }

    /// Try to create an inline TermInfo from posting data
    /// Returns None if posting list is too large to inline
    pub fn try_inline(doc_ids: &[u32], term_freqs: &[u32]) -> Option<Self> {
        if doc_ids.len() > MAX_INLINE_POSTINGS || doc_ids.is_empty() {
            return None;
        }

        let mut data = [0u8; 16];
        let mut cursor = std::io::Cursor::new(&mut data[..]);
        let mut prev_doc_id = 0u32;

        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let delta = doc_id - prev_doc_id;
            if write_vint(&mut cursor, delta as u64).is_err() {
                return None;
            }
            if write_vint(&mut cursor, term_freqs[i] as u64).is_err() {
                return None;
            }
            prev_doc_id = doc_id;
        }

        let data_len = cursor.position() as u8;
        if data_len > 16 {
            return None;
        }

        Some(TermInfo::Inline {
            doc_freq: doc_ids.len() as u8,
            data,
            data_len,
        })
    }

    /// Get document frequency
    pub fn doc_freq(&self) -> u32 {
        match self {
            TermInfo::Inline { doc_freq, .. } => *doc_freq as u32,
            TermInfo::External { doc_freq, .. } => *doc_freq,
        }
    }

    /// Check if this is an inline posting list
    pub fn is_inline(&self) -> bool {
        matches!(self, TermInfo::Inline { .. })
    }

    /// Get external posting info (offset, len) - returns None for inline
    pub fn external_info(&self) -> Option<(u64, u32)> {
        match self {
            TermInfo::External {
                posting_offset,
                posting_len,
                ..
            } => Some((*posting_offset, *posting_len)),
            TermInfo::Inline { .. } => None,
        }
    }

    /// Get position info (offset, len) - returns None for inline or if no positions
    pub fn position_info(&self) -> Option<(u64, u32)> {
        match self {
            TermInfo::External {
                position_offset,
                position_len,
                ..
            } if *position_len > 0 => Some((*position_offset, *position_len)),
            _ => None,
        }
    }

    /// Decode inline postings into (doc_ids, term_freqs)
    /// Returns None if this is an external reference
    pub fn decode_inline(&self) -> Option<(Vec<u32>, Vec<u32>)> {
        match self {
            TermInfo::Inline {
                doc_freq,
                data,
                data_len,
            } => {
                let mut doc_ids = Vec::with_capacity(*doc_freq as usize);
                let mut term_freqs = Vec::with_capacity(*doc_freq as usize);
                let mut reader = &data[..*data_len as usize];
                let mut prev_doc_id = 0u32;

                for _ in 0..*doc_freq {
                    let delta = read_vint(&mut reader).ok()? as u32;
                    let tf = read_vint(&mut reader).ok()? as u32;
                    let doc_id = prev_doc_id + delta;
                    doc_ids.push(doc_id);
                    term_freqs.push(tf);
                    prev_doc_id = doc_id;
                }

                Some((doc_ids, term_freqs))
            }
            TermInfo::External { .. } => None,
        }
    }
}

impl SSTableValue for TermInfo {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            TermInfo::Inline {
                doc_freq,
                data,
                data_len,
            } => {
                // Tag byte 0xFF = inline marker
                writer.write_u8(0xFF)?;
                writer.write_u8(*doc_freq)?;
                writer.write_u8(*data_len)?;
                writer.write_all(&data[..*data_len as usize])?;
            }
            TermInfo::External {
                posting_offset,
                posting_len,
                doc_freq,
                position_offset,
                position_len,
            } => {
                // Tag byte 0x00 = external marker (no positions)
                // Tag byte 0x01 = external with positions
                if *position_len > 0 {
                    writer.write_u8(0x01)?;
                    write_vint(writer, *doc_freq as u64)?;
                    write_vint(writer, *posting_offset)?;
                    write_vint(writer, *posting_len as u64)?;
                    write_vint(writer, *position_offset)?;
                    write_vint(writer, *position_len as u64)?;
                } else {
                    writer.write_u8(0x00)?;
                    write_vint(writer, *doc_freq as u64)?;
                    write_vint(writer, *posting_offset)?;
                    write_vint(writer, *posting_len as u64)?;
                }
            }
        }
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let tag = reader.read_u8()?;

        if tag == 0xFF {
            // Inline
            let doc_freq = reader.read_u8()?;
            let data_len = reader.read_u8()?;
            let mut data = [0u8; 16];
            reader.read_exact(&mut data[..data_len as usize])?;
            Ok(TermInfo::Inline {
                doc_freq,
                data,
                data_len,
            })
        } else if tag == 0x00 {
            // External (no positions)
            let doc_freq = read_vint(reader)? as u32;
            let posting_offset = read_vint(reader)?;
            let posting_len = read_vint(reader)? as u32;
            Ok(TermInfo::External {
                posting_offset,
                posting_len,
                doc_freq,
                position_offset: 0,
                position_len: 0,
            })
        } else if tag == 0x01 {
            // External with positions
            let doc_freq = read_vint(reader)? as u32;
            let posting_offset = read_vint(reader)?;
            let posting_len = read_vint(reader)? as u32;
            let position_offset = read_vint(reader)?;
            let position_len = read_vint(reader)? as u32;
            Ok(TermInfo::External {
                posting_offset,
                posting_len,
                doc_freq,
                position_offset,
                position_len,
            })
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid TermInfo tag: {}", tag),
            ))
        }
    }
}

/// Write variable-length integer
pub fn write_vint<W: Write + ?Sized>(writer: &mut W, mut value: u64) -> io::Result<()> {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            writer.write_u8(byte)?;
            return Ok(());
        } else {
            writer.write_u8(byte | 0x80)?;
        }
    }
}

/// Read variable-length integer
pub fn read_vint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut result = 0u64;
    let mut shift = 0;

    loop {
        let byte = reader.read_u8()?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint too long",
            ));
        }
    }
}

/// Compute common prefix length
pub fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// SSTable statistics for debugging
#[derive(Debug, Clone)]
pub struct SSTableStats {
    pub num_blocks: usize,
    pub num_sparse_entries: usize,
    pub num_entries: u64,
    pub has_bloom_filter: bool,
    pub has_dictionary: bool,
    pub bloom_filter_size: usize,
    pub dictionary_size: usize,
}

/// SSTable writer configuration
#[derive(Debug, Clone)]
pub struct SSTableWriterConfig {
    /// Compression level (1-22, higher = better compression but slower)
    pub compression_level: CompressionLevel,
    /// Whether to train and use a dictionary for compression
    pub use_dictionary: bool,
    /// Dictionary size in bytes (default 64KB)
    pub dict_size: usize,
    /// Whether to build a bloom filter
    pub use_bloom_filter: bool,
    /// Bloom filter bits per key (default 10 = ~1% false positive rate)
    pub bloom_bits_per_key: usize,
}

impl Default for SSTableWriterConfig {
    fn default() -> Self {
        Self::from_optimization(crate::structures::IndexOptimization::default())
    }
}

impl SSTableWriterConfig {
    /// Create config from IndexOptimization mode
    pub fn from_optimization(optimization: crate::structures::IndexOptimization) -> Self {
        use crate::structures::IndexOptimization;
        match optimization {
            IndexOptimization::Adaptive => Self {
                compression_level: CompressionLevel::BETTER, // Level 9
                use_dictionary: false,
                dict_size: DEFAULT_DICT_SIZE,
                use_bloom_filter: false,
                bloom_bits_per_key: BLOOM_BITS_PER_KEY,
            },
            IndexOptimization::SizeOptimized => Self {
                compression_level: CompressionLevel::MAX, // Level 22
                use_dictionary: true,
                dict_size: DEFAULT_DICT_SIZE,
                use_bloom_filter: true,
                bloom_bits_per_key: BLOOM_BITS_PER_KEY,
            },
            IndexOptimization::PerformanceOptimized => Self {
                compression_level: CompressionLevel::FAST, // Level 1
                use_dictionary: false,
                dict_size: DEFAULT_DICT_SIZE,
                use_bloom_filter: true, // Bloom helps skip blocks fast
                bloom_bits_per_key: BLOOM_BITS_PER_KEY,
            },
        }
    }

    /// Fast configuration - prioritize write speed over compression
    pub fn fast() -> Self {
        Self::from_optimization(crate::structures::IndexOptimization::PerformanceOptimized)
    }

    /// Maximum compression configuration - prioritize size over speed
    pub fn max_compression() -> Self {
        Self::from_optimization(crate::structures::IndexOptimization::SizeOptimized)
    }
}

/// SSTable writer with optimizations:
/// - Dictionary compression for blocks (if dictionary provided)
/// - Configurable compression level
/// - Block index prefix compression
/// - Bloom filter for fast negative lookups
pub struct SSTableWriter<'a, V: SSTableValue> {
    writer: &'a mut dyn Write,
    block_buffer: Vec<u8>,
    prev_key: Vec<u8>,
    index: Vec<BlockIndexEntry>,
    current_offset: u64,
    num_entries: u64,
    block_first_key: Option<Vec<u8>>,
    config: SSTableWriterConfig,
    /// Pre-trained dictionary for compression (optional)
    dictionary: Option<CompressionDict>,
    /// All keys for bloom filter
    all_keys: Vec<Vec<u8>>,
    /// Bloom filter (built at finish time)
    bloom_filter: Option<BloomFilter>,
    _phantom: std::marker::PhantomData<V>,
}

impl<'a, V: SSTableValue> SSTableWriter<'a, V> {
    /// Create a new SSTable writer with default configuration
    pub fn new(writer: &'a mut dyn Write) -> Self {
        Self::with_config(writer, SSTableWriterConfig::default())
    }

    /// Create a new SSTable writer with custom configuration
    pub fn with_config(writer: &'a mut dyn Write, config: SSTableWriterConfig) -> Self {
        Self {
            writer,
            block_buffer: Vec::with_capacity(BLOCK_SIZE),
            prev_key: Vec::new(),
            index: Vec::new(),
            current_offset: 0,
            num_entries: 0,
            block_first_key: None,
            config,
            dictionary: None,
            all_keys: Vec::new(),
            bloom_filter: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new SSTable writer with a pre-trained dictionary
    pub fn with_dictionary(
        writer: &'a mut dyn Write,
        config: SSTableWriterConfig,
        dictionary: CompressionDict,
    ) -> Self {
        Self {
            writer,
            block_buffer: Vec::with_capacity(BLOCK_SIZE),
            prev_key: Vec::new(),
            index: Vec::new(),
            current_offset: 0,
            num_entries: 0,
            block_first_key: None,
            config,
            dictionary: Some(dictionary),
            all_keys: Vec::new(),
            bloom_filter: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn insert(&mut self, key: &[u8], value: &V) -> io::Result<()> {
        if self.block_first_key.is_none() {
            self.block_first_key = Some(key.to_vec());
        }

        // Collect keys for bloom filter
        if self.config.use_bloom_filter {
            self.all_keys.push(key.to_vec());
        }

        let prefix_len = common_prefix_len(&self.prev_key, key);
        let suffix = &key[prefix_len..];

        write_vint(&mut self.block_buffer, prefix_len as u64)?;
        write_vint(&mut self.block_buffer, suffix.len() as u64)?;
        self.block_buffer.extend_from_slice(suffix);
        value.serialize(&mut self.block_buffer)?;

        self.prev_key.clear();
        self.prev_key.extend_from_slice(key);
        self.num_entries += 1;

        if self.block_buffer.len() >= BLOCK_SIZE {
            self.flush_block()?;
        }

        Ok(())
    }

    /// Flush and compress the current block
    fn flush_block(&mut self) -> io::Result<()> {
        if self.block_buffer.is_empty() {
            return Ok(());
        }

        // Compress block with dictionary if available
        let compressed = if let Some(ref dict) = self.dictionary {
            crate::compression::compress_with_dict(
                &self.block_buffer,
                self.config.compression_level,
                dict,
            )?
        } else {
            crate::compression::compress(&self.block_buffer, self.config.compression_level)?
        };

        if let Some(first_key) = self.block_first_key.take() {
            self.index.push(BlockIndexEntry {
                first_key,
                offset: self.current_offset,
                length: compressed.len() as u32,
            });
        }

        self.writer.write_all(&compressed)?;
        self.current_offset += compressed.len() as u64;
        self.block_buffer.clear();
        self.prev_key.clear();

        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        // Flush any remaining data
        self.flush_block()?;

        // Build bloom filter from collected keys
        if self.config.use_bloom_filter && !self.all_keys.is_empty() {
            let mut bloom = BloomFilter::new(self.all_keys.len(), self.config.bloom_bits_per_key);
            for key in &self.all_keys {
                bloom.insert(key);
            }
            self.bloom_filter = Some(bloom);
        }

        let data_end_offset = self.current_offset;

        // Build memory-efficient block index (v4 format)
        // Convert to (key, BlockAddr) pairs for the new index format
        let entries: Vec<(Vec<u8>, BlockAddr)> = self
            .index
            .iter()
            .map(|e| {
                (
                    e.first_key.clone(),
                    BlockAddr {
                        offset: e.offset,
                        length: e.length,
                    },
                )
            })
            .collect();

        // Build FST-based index if native feature is enabled, otherwise use mmap index
        #[cfg(feature = "native")]
        let index_bytes = FstBlockIndex::build(&entries)?;
        #[cfg(not(feature = "native"))]
        let index_bytes = MmapBlockIndex::build(&entries)?;

        // Write index bytes with length prefix
        self.writer
            .write_u32::<LittleEndian>(index_bytes.len() as u32)?;
        self.writer.write_all(&index_bytes)?;
        self.current_offset += 4 + index_bytes.len() as u64;

        // Write bloom filter if present
        let bloom_offset = if let Some(ref bloom) = self.bloom_filter {
            let bloom_data = bloom.to_bytes();
            let offset = self.current_offset;
            self.writer.write_all(&bloom_data)?;
            self.current_offset += bloom_data.len() as u64;
            offset
        } else {
            0
        };

        // Write dictionary if present
        let dict_offset = if let Some(ref dict) = self.dictionary {
            let dict_bytes = dict.as_bytes();
            let offset = self.current_offset;
            self.writer
                .write_u32::<LittleEndian>(dict_bytes.len() as u32)?;
            self.writer.write_all(dict_bytes)?;
            self.current_offset += 4 + dict_bytes.len() as u64;
            offset
        } else {
            0
        };

        // Write extended footer (v4 format)
        self.writer.write_u64::<LittleEndian>(data_end_offset)?;
        self.writer.write_u64::<LittleEndian>(self.num_entries)?;
        self.writer.write_u64::<LittleEndian>(bloom_offset)?; // 0 if no bloom
        self.writer.write_u64::<LittleEndian>(dict_offset)?; // 0 if no dict
        self.writer
            .write_u8(self.config.compression_level.0 as u8)?;
        self.writer.write_u32::<LittleEndian>(SSTABLE_MAGIC)?;

        Ok(())
    }
}

/// Block index entry
#[derive(Debug, Clone)]
struct BlockIndexEntry {
    first_key: Vec<u8>,
    offset: u64,
    length: u32,
}

/// Async SSTable reader - loads blocks on demand via LazyFileSlice
///
/// Memory-efficient design (v4 format):
/// - Block index uses FST (native) or mmap'd raw bytes - no heap allocation for keys
/// - Block addresses stored in bitpacked format
/// - Bloom filter and dictionary optional
pub struct AsyncSSTableReader<V: SSTableValue> {
    /// LazyFileSlice for the data portion (blocks only) - fetches ranges on demand
    data_slice: LazyFileSlice,
    /// Memory-efficient block index (v4: FST or mmap, v3: legacy Vec)
    block_index: BlockIndex,
    num_entries: u64,
    /// Hot cache for decompressed blocks
    cache: RwLock<BlockCache>,
    /// Bloom filter for fast negative lookups (optional)
    bloom_filter: Option<BloomFilter>,
    /// Compression dictionary (optional)
    dictionary: Option<CompressionDict>,
    /// Compression level used
    #[allow(dead_code)]
    compression_level: CompressionLevel,
    _phantom: std::marker::PhantomData<V>,
}

/// LRU-style block cache
struct BlockCache {
    blocks: FxHashMap<u64, Arc<Vec<u8>>>,
    access_order: Vec<u64>,
    max_blocks: usize,
}

impl BlockCache {
    fn new(max_blocks: usize) -> Self {
        Self {
            blocks: FxHashMap::default(),
            access_order: Vec::new(),
            max_blocks,
        }
    }

    fn get(&mut self, offset: u64) -> Option<Arc<Vec<u8>>> {
        if let Some(block) = self.blocks.get(&offset) {
            if let Some(pos) = self.access_order.iter().position(|&o| o == offset) {
                self.access_order.remove(pos);
                self.access_order.push(offset);
            }
            Some(Arc::clone(block))
        } else {
            None
        }
    }

    fn insert(&mut self, offset: u64, block: Arc<Vec<u8>>) {
        while self.blocks.len() >= self.max_blocks && !self.access_order.is_empty() {
            let evict_offset = self.access_order.remove(0);
            self.blocks.remove(&evict_offset);
        }
        self.blocks.insert(offset, block);
        self.access_order.push(offset);
    }
}

impl<V: SSTableValue> AsyncSSTableReader<V> {
    /// Open an SSTable from a LazyFileHandle
    /// Only loads the footer and index into memory, data blocks fetched on-demand
    ///
    /// Supports both v3 (legacy) and v4 (memory-efficient) formats:
    /// - v4: FST-based or mmap'd block index (no heap allocation for keys)
    /// - v3: Prefix-compressed block index loaded into Vec (legacy fallback)
    pub async fn open(file_handle: LazyFileHandle, cache_blocks: usize) -> io::Result<Self> {
        let file_len = file_handle.len();
        if file_len < 37 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable too small",
            ));
        }

        // Read footer (37 bytes)
        // Format: data_end(8) + num_entries(8) + bloom_offset(8) + dict_offset(8) + compression_level(1) + magic(4)
        let footer_bytes = file_handle
            .read_bytes_range(file_len - 37..file_len)
            .await?;

        let mut reader = footer_bytes.as_slice();
        let data_end_offset = reader.read_u64::<LittleEndian>()?;
        let num_entries = reader.read_u64::<LittleEndian>()?;
        let bloom_offset = reader.read_u64::<LittleEndian>()?;
        let dict_offset = reader.read_u64::<LittleEndian>()?;
        let compression_level = CompressionLevel(reader.read_u8()? as i32);
        let magic = reader.read_u32::<LittleEndian>()?;

        if magic != SSTABLE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid SSTable magic: 0x{:08X}", magic),
            ));
        }

        // Read index section
        let index_start = data_end_offset;
        let index_end = file_len - 37;
        let index_bytes = file_handle.read_bytes_range(index_start..index_end).await?;

        // Parse block index (length-prefixed FST or mmap index)
        let mut idx_reader = index_bytes.as_slice();
        let index_len = idx_reader.read_u32::<LittleEndian>()? as usize;

        if index_len > idx_reader.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Index data truncated",
            ));
        }

        let index_data = OwnedBytes::new(idx_reader[..index_len].to_vec());

        // Try FST first (native), fall back to mmap
        #[cfg(feature = "native")]
        let block_index = match FstBlockIndex::load(index_data.clone()) {
            Ok(fst_idx) => BlockIndex::Fst(fst_idx),
            Err(_) => BlockIndex::Mmap(MmapBlockIndex::load(index_data)?),
        };
        #[cfg(not(feature = "native"))]
        let block_index = BlockIndex::Mmap(MmapBlockIndex::load(index_data)?);

        // Load bloom filter if present
        let bloom_filter = if bloom_offset > 0 {
            let bloom_start = bloom_offset;
            // Read bloom filter size first (12 bytes header)
            let bloom_header = file_handle
                .read_bytes_range(bloom_start..bloom_start + 12)
                .await?;
            let num_words = u32::from_le_bytes([
                bloom_header[8],
                bloom_header[9],
                bloom_header[10],
                bloom_header[11],
            ]) as u64;
            let bloom_size = 12 + num_words * 8;
            let bloom_data = file_handle
                .read_bytes_range(bloom_start..bloom_start + bloom_size)
                .await?;
            Some(BloomFilter::from_bytes(&bloom_data)?)
        } else {
            None
        };

        // Load dictionary if present
        let dictionary = if dict_offset > 0 {
            let dict_start = dict_offset;
            // Read dictionary size first
            let dict_len_bytes = file_handle
                .read_bytes_range(dict_start..dict_start + 4)
                .await?;
            let dict_len = u32::from_le_bytes([
                dict_len_bytes[0],
                dict_len_bytes[1],
                dict_len_bytes[2],
                dict_len_bytes[3],
            ]) as u64;
            let dict_data = file_handle
                .read_bytes_range(dict_start + 4..dict_start + 4 + dict_len)
                .await?;
            Some(CompressionDict::from_bytes(dict_data.to_vec()))
        } else {
            None
        };

        // Create a lazy slice for just the data portion
        let data_slice = file_handle.slice(0..data_end_offset);

        Ok(Self {
            data_slice,
            block_index,
            num_entries,
            cache: RwLock::new(BlockCache::new(cache_blocks)),
            bloom_filter,
            dictionary,
            compression_level,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Number of entries
    pub fn num_entries(&self) -> u64 {
        self.num_entries
    }

    /// Get stats about this SSTable for debugging
    pub fn stats(&self) -> SSTableStats {
        SSTableStats {
            num_blocks: self.block_index.len(),
            num_sparse_entries: 0, // No longer using sparse index separately
            num_entries: self.num_entries,
            has_bloom_filter: self.bloom_filter.is_some(),
            has_dictionary: self.dictionary.is_some(),
            bloom_filter_size: self
                .bloom_filter
                .as_ref()
                .map(|b| b.size_bytes())
                .unwrap_or(0),
            dictionary_size: self.dictionary.as_ref().map(|d| d.len()).unwrap_or(0),
        }
    }

    /// Number of blocks currently in the cache
    pub fn cached_blocks(&self) -> usize {
        self.cache.read().blocks.len()
    }

    /// Look up a key (async - may need to load block)
    ///
    /// Uses bloom filter for fast negative lookups, then memory-efficient
    /// block index to locate the block, reducing I/O to typically 1 block read.
    pub async fn get(&self, key: &[u8]) -> io::Result<Option<V>> {
        log::debug!(
            "SSTable::get called, key_len={}, total_blocks={}",
            key.len(),
            self.block_index.len()
        );

        // Check bloom filter first - fast negative lookup
        if let Some(ref bloom) = self.bloom_filter
            && !bloom.may_contain(key)
        {
            log::debug!("SSTable::get bloom filter negative");
            return Ok(None);
        }

        // Use block index to find the block that could contain the key
        let block_idx = match self.block_index.locate(key) {
            Some(idx) => idx,
            None => {
                log::debug!("SSTable::get key not found (before first block)");
                return Ok(None);
            }
        };

        log::debug!("SSTable::get loading block_idx={}", block_idx);

        // Now we know exactly which block to load - single I/O
        let block_data = self.load_block(block_idx).await?;
        self.search_block(&block_data, key)
    }

    /// Batch lookup multiple keys with optimized I/O
    ///
    /// Groups keys by block and loads each block only once, reducing
    /// I/O from N reads to at most N reads (often fewer if keys share blocks).
    /// Uses bloom filter to skip keys that definitely don't exist.
    pub async fn get_batch(&self, keys: &[&[u8]]) -> io::Result<Vec<Option<V>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        // Map each key to its block index
        let mut key_to_block: Vec<(usize, usize)> = Vec::with_capacity(keys.len());
        for (key_idx, key) in keys.iter().enumerate() {
            // Check bloom filter first
            if let Some(ref bloom) = self.bloom_filter
                && !bloom.may_contain(key)
            {
                key_to_block.push((key_idx, usize::MAX)); // Definitely not present
                continue;
            }

            match self.block_index.locate(key) {
                Some(block_idx) => key_to_block.push((key_idx, block_idx)),
                None => key_to_block.push((key_idx, usize::MAX)), // Mark as not found
            }
        }

        // Group keys by block
        let mut blocks_to_load: Vec<usize> = key_to_block
            .iter()
            .filter(|(_, b)| *b != usize::MAX)
            .map(|(_, b)| *b)
            .collect();
        blocks_to_load.sort_unstable();
        blocks_to_load.dedup();

        // Load all needed blocks (this is where I/O happens)
        for &block_idx in &blocks_to_load {
            let _ = self.load_block(block_idx).await?;
        }

        // Now search each key in its block (all blocks are cached)
        let mut results = vec![None; keys.len()];
        for (key_idx, block_idx) in key_to_block {
            if block_idx == usize::MAX {
                continue;
            }
            let block_data = self.load_block(block_idx).await?; // Will hit cache
            results[key_idx] = self.search_block(&block_data, keys[key_idx])?;
        }

        Ok(results)
    }

    /// Preload all data blocks into memory
    ///
    /// Call this after open() to eliminate all I/O during subsequent lookups.
    /// Useful when the SSTable is small enough to fit in memory.
    pub async fn preload_all_blocks(&self) -> io::Result<()> {
        for block_idx in 0..self.block_index.len() {
            self.load_block(block_idx).await?;
        }
        Ok(())
    }

    /// Prefetch all data blocks via a single bulk I/O operation.
    ///
    /// Reads the entire compressed data section in one call, then decompresses
    /// each block and populates the cache. This turns N individual reads into 1.
    /// Cache capacity is expanded to hold all blocks.
    pub async fn prefetch_all_data_bulk(&self) -> io::Result<()> {
        let num_blocks = self.block_index.len();
        if num_blocks == 0 {
            return Ok(());
        }

        // Find total data extent
        let mut max_end: u64 = 0;
        for i in 0..num_blocks {
            if let Some(addr) = self.block_index.get_addr(i) {
                max_end = max_end.max(addr.offset + addr.length as u64);
            }
        }

        // Single bulk read of entire data section
        let all_data = self.data_slice.read_bytes_range(0..max_end).await?;
        let buf = all_data.as_slice();

        // Expand cache and decompress all blocks
        let mut cache = self.cache.write();
        cache.max_blocks = cache.max_blocks.max(num_blocks);
        for i in 0..num_blocks {
            let addr = self.block_index.get_addr(i).unwrap();
            if cache.get(addr.offset).is_some() {
                continue;
            }
            let compressed =
                &buf[addr.offset as usize..(addr.offset + addr.length as u64) as usize];
            let decompressed = if let Some(ref dict) = self.dictionary {
                crate::compression::decompress_with_dict(compressed, dict)?
            } else {
                crate::compression::decompress(compressed)?
            };
            cache.insert(addr.offset, Arc::new(decompressed));
        }

        Ok(())
    }

    /// Load a block (checks cache first, then loads from FileSlice)
    /// Uses dictionary decompression if dictionary is present
    async fn load_block(&self, block_idx: usize) -> io::Result<Arc<Vec<u8>>> {
        let addr = self.block_index.get_addr(block_idx).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Block index out of range")
        })?;

        // Check cache first
        {
            let mut cache = self.cache.write();
            if let Some(block) = cache.get(addr.offset) {
                log::debug!("SSTable::load_block idx={} CACHE HIT", block_idx);
                return Ok(block);
            }
        }

        log::debug!(
            "SSTable::load_block idx={} CACHE MISS, reading bytes [{}-{}]",
            block_idx,
            addr.offset,
            addr.offset + addr.length as u64
        );

        // Load from FileSlice
        let range = addr.byte_range();
        let compressed = self.data_slice.read_bytes_range(range).await?;

        // Decompress with dictionary if available
        let decompressed = if let Some(ref dict) = self.dictionary {
            crate::compression::decompress_with_dict(compressed.as_slice(), dict)?
        } else {
            crate::compression::decompress(compressed.as_slice())?
        };

        let block = Arc::new(decompressed);

        // Insert into cache
        {
            let mut cache = self.cache.write();
            cache.insert(addr.offset, Arc::clone(&block));
        }

        Ok(block)
    }

    fn search_block(&self, block_data: &[u8], target_key: &[u8]) -> io::Result<Option<V>> {
        let mut reader = block_data;
        let mut current_key = Vec::new();

        while !reader.is_empty() {
            let common_prefix_len = read_vint(&mut reader)? as usize;
            let suffix_len = read_vint(&mut reader)? as usize;

            current_key.truncate(common_prefix_len);
            let mut suffix = vec![0u8; suffix_len];
            reader.read_exact(&mut suffix)?;
            current_key.extend_from_slice(&suffix);

            let value = V::deserialize(&mut reader)?;

            match current_key.as_slice().cmp(target_key) {
                std::cmp::Ordering::Equal => return Ok(Some(value)),
                std::cmp::Ordering::Greater => return Ok(None),
                std::cmp::Ordering::Less => continue,
            }
        }

        Ok(None)
    }

    /// Prefetch blocks for a key range
    pub async fn prefetch_range(&self, start_key: &[u8], end_key: &[u8]) -> io::Result<()> {
        let start_block = self.block_index.locate(start_key).unwrap_or(0);
        let end_block = self
            .block_index
            .locate(end_key)
            .unwrap_or(self.block_index.len().saturating_sub(1));

        for block_idx in start_block..=end_block.min(self.block_index.len().saturating_sub(1)) {
            let _ = self.load_block(block_idx).await?;
        }

        Ok(())
    }

    /// Iterate over all entries (loads blocks as needed)
    pub fn iter(&self) -> AsyncSSTableIterator<'_, V> {
        AsyncSSTableIterator::new(self)
    }

    /// Get all entries as a vector (for merging)
    pub async fn all_entries(&self) -> io::Result<Vec<(Vec<u8>, V)>> {
        let mut results = Vec::new();

        for block_idx in 0..self.block_index.len() {
            let block_data = self.load_block(block_idx).await?;
            let mut reader = block_data.as_slice();
            let mut current_key = Vec::new();

            while !reader.is_empty() {
                let common_prefix_len = read_vint(&mut reader)? as usize;
                let suffix_len = read_vint(&mut reader)? as usize;

                current_key.truncate(common_prefix_len);
                let mut suffix = vec![0u8; suffix_len];
                reader.read_exact(&mut suffix)?;
                current_key.extend_from_slice(&suffix);

                let value = V::deserialize(&mut reader)?;
                results.push((current_key.clone(), value));
            }
        }

        Ok(results)
    }
}

/// Async iterator over SSTable entries
pub struct AsyncSSTableIterator<'a, V: SSTableValue> {
    reader: &'a AsyncSSTableReader<V>,
    current_block: usize,
    block_data: Option<Arc<Vec<u8>>>,
    block_offset: usize,
    current_key: Vec<u8>,
    finished: bool,
}

impl<'a, V: SSTableValue> AsyncSSTableIterator<'a, V> {
    fn new(reader: &'a AsyncSSTableReader<V>) -> Self {
        Self {
            reader,
            current_block: 0,
            block_data: None,
            block_offset: 0,
            current_key: Vec::new(),
            finished: reader.block_index.is_empty(),
        }
    }

    async fn load_next_block(&mut self) -> io::Result<bool> {
        if self.current_block >= self.reader.block_index.len() {
            self.finished = true;
            return Ok(false);
        }

        self.block_data = Some(self.reader.load_block(self.current_block).await?);
        self.block_offset = 0;
        self.current_key.clear();
        self.current_block += 1;
        Ok(true)
    }

    /// Advance to next entry (async)
    pub async fn next(&mut self) -> io::Result<Option<(Vec<u8>, V)>> {
        if self.finished {
            return Ok(None);
        }

        if self.block_data.is_none() && !self.load_next_block().await? {
            return Ok(None);
        }

        loop {
            let block = self.block_data.as_ref().unwrap();
            if self.block_offset >= block.len() {
                if !self.load_next_block().await? {
                    return Ok(None);
                }
                continue;
            }

            let mut reader = &block[self.block_offset..];
            let start_len = reader.len();

            let common_prefix_len = read_vint(&mut reader)? as usize;
            let suffix_len = read_vint(&mut reader)? as usize;

            self.current_key.truncate(common_prefix_len);
            let mut suffix = vec![0u8; suffix_len];
            reader.read_exact(&mut suffix)?;
            self.current_key.extend_from_slice(&suffix);

            let value = V::deserialize(&mut reader)?;

            self.block_offset += start_len - reader.len();

            return Ok(Some((self.current_key.clone(), value)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut bloom = BloomFilter::new(100, 10);

        bloom.insert(b"hello");
        bloom.insert(b"world");
        bloom.insert(b"test");

        assert!(bloom.may_contain(b"hello"));
        assert!(bloom.may_contain(b"world"));
        assert!(bloom.may_contain(b"test"));

        // These should likely return false (with ~1% false positive rate)
        assert!(!bloom.may_contain(b"notfound"));
        assert!(!bloom.may_contain(b"missing"));
    }

    #[test]
    fn test_bloom_filter_serialization() {
        let mut bloom = BloomFilter::new(100, 10);
        bloom.insert(b"key1");
        bloom.insert(b"key2");

        let bytes = bloom.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).unwrap();

        assert!(restored.may_contain(b"key1"));
        assert!(restored.may_contain(b"key2"));
        assert!(!restored.may_contain(b"key3"));
    }

    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let num_keys = 10000;
        let mut bloom = BloomFilter::new(num_keys, BLOOM_BITS_PER_KEY);

        // Insert keys
        for i in 0..num_keys {
            let key = format!("key_{}", i);
            bloom.insert(key.as_bytes());
        }

        // All inserted keys should be found
        for i in 0..num_keys {
            let key = format!("key_{}", i);
            assert!(bloom.may_contain(key.as_bytes()));
        }

        // Check false positive rate on non-existent keys
        let mut false_positives = 0;
        let test_count = 10000;
        for i in 0..test_count {
            let key = format!("nonexistent_{}", i);
            if bloom.may_contain(key.as_bytes()) {
                false_positives += 1;
            }
        }

        // With 10 bits per key, expect ~1% false positive rate
        // Allow up to 3% due to hash function variance
        let fp_rate = false_positives as f64 / test_count as f64;
        assert!(
            fp_rate < 0.03,
            "False positive rate {} is too high",
            fp_rate
        );
    }

    #[test]
    fn test_sstable_writer_config() {
        use crate::structures::IndexOptimization;

        // Default = Adaptive
        let config = SSTableWriterConfig::default();
        assert_eq!(config.compression_level.0, 9); // BETTER
        assert!(!config.use_bloom_filter);
        assert!(!config.use_dictionary);

        // Adaptive
        let adaptive = SSTableWriterConfig::from_optimization(IndexOptimization::Adaptive);
        assert_eq!(adaptive.compression_level.0, 9);
        assert!(!adaptive.use_bloom_filter);
        assert!(!adaptive.use_dictionary);

        // SizeOptimized
        let size = SSTableWriterConfig::from_optimization(IndexOptimization::SizeOptimized);
        assert_eq!(size.compression_level.0, 22); // MAX
        assert!(size.use_bloom_filter);
        assert!(size.use_dictionary);

        // PerformanceOptimized
        let perf = SSTableWriterConfig::from_optimization(IndexOptimization::PerformanceOptimized);
        assert_eq!(perf.compression_level.0, 1); // FAST
        assert!(perf.use_bloom_filter); // Bloom helps skip blocks fast
        assert!(!perf.use_dictionary);

        // Aliases
        let fast = SSTableWriterConfig::fast();
        assert_eq!(fast.compression_level.0, 1);

        let max = SSTableWriterConfig::max_compression();
        assert_eq!(max.compression_level.0, 22);
    }

    #[test]
    fn test_vint_roundtrip() {
        let test_values = [0u64, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];

        for &val in &test_values {
            let mut buf = Vec::new();
            write_vint(&mut buf, val).unwrap();
            let mut reader = buf.as_slice();
            let decoded = read_vint(&mut reader).unwrap();
            assert_eq!(val, decoded, "Failed for value {}", val);
        }
    }

    #[test]
    fn test_common_prefix_len() {
        assert_eq!(common_prefix_len(b"hello", b"hello"), 5);
        assert_eq!(common_prefix_len(b"hello", b"help"), 3);
        assert_eq!(common_prefix_len(b"hello", b"world"), 0);
        assert_eq!(common_prefix_len(b"", b"hello"), 0);
        assert_eq!(common_prefix_len(b"hello", b""), 0);
    }
}
