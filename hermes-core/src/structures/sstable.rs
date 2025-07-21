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

#[cfg(feature = "fst-index")]
use super::sstable_index::FstBlockIndex;
use super::sstable_index::{BlockAddr, BlockIndex, MmapBlockIndex};
use crate::compression::{CompressionDict, CompressionLevel};
use crate::directories::{FileHandle, OwnedBytes};

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
const BLOOM_FILTER_HEADER_SIZE: usize = 16;

const MAX_SSTABLE_BLOCK_BYTES: usize = 64 * 1024 * 1024;
const MAX_SSTABLE_DICTIONARY_BYTES: u64 = 16 * 1024 * 1024;

/// Results and truncation flag returned by a budgeted prefix scan.
pub type PrefixScanResult<V> = (Vec<(Vec<u8>, V)>, bool);

// ============================================================================
// Bloom Filter Implementation
// ============================================================================
/// Simple bloom filter for key existence checks
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: BloomBits,
    num_bits: usize,
    num_hashes: usize,
}

/// Bloom filter storage — Vec for write path, OwnedBytes for zero-copy read path.
#[derive(Debug, Clone)]
enum BloomBits {
    /// Mutable storage for building (SSTable writer)
    Vec(Vec<u64>),
    /// Zero-copy mmap reference for reading (raw LE u64 words, no header)
    Bytes(OwnedBytes),
}

impl BloomBits {
    #[inline]
    fn len(&self) -> usize {
        match self {
            BloomBits::Vec(v) => v.len(),
            BloomBits::Bytes(b) => b.len() / 8,
        }
    }

    #[inline]
    fn get(&self, word_idx: usize) -> u64 {
        match self {
            BloomBits::Vec(v) => v[word_idx],
            BloomBits::Bytes(b) => {
                let off = word_idx * 8;
                u64::from_le_bytes([
                    b[off],
                    b[off + 1],
                    b[off + 2],
                    b[off + 3],
                    b[off + 4],
                    b[off + 5],
                    b[off + 6],
                    b[off + 7],
                ])
            }
        }
    }

    #[inline]
    fn set_bit(&mut self, word_idx: usize, bit_idx: usize) {
        match self {
            BloomBits::Vec(v) => v[word_idx] |= 1u64 << bit_idx,
            BloomBits::Bytes(_) => panic!("cannot mutate read-only bloom filter"),
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            BloomBits::Vec(v) => v.len() * 8,
            BloomBits::Bytes(b) => b.len(),
        }
    }

    fn write_to(&self, writer: &mut (impl Write + ?Sized)) -> io::Result<()> {
        match self {
            BloomBits::Vec(words) => {
                #[cfg(target_endian = "little")]
                {
                    // SAFETY: u64 has no padding and the native byte order is
                    // the on-disk little-endian order.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            words.as_ptr().cast::<u8>(),
                            words.len().saturating_mul(8),
                        )
                    };
                    writer.write_all(bytes)
                }
                #[cfg(target_endian = "big")]
                {
                    for &word in words {
                        writer.write_u64::<LittleEndian>(word)?;
                    }
                    Ok(())
                }
            }
            BloomBits::Bytes(bytes) => writer.write_all(bytes.as_slice()),
        }
    }
}

impl BloomFilter {
    pub(crate) const SERIALIZED_HEADER_SIZE: usize = BLOOM_FILTER_HEADER_SIZE;

    /// Create a new bloom filter sized for expected number of keys
    pub fn new(expected_keys: usize, bits_per_key: usize) -> Self {
        let num_bits = expected_keys.saturating_mul(bits_per_key).max(64);
        let num_words = num_bits.div_ceil(64);
        Self {
            bits: BloomBits::Vec(vec![0u64; num_words]),
            num_bits,
            num_hashes: BLOOM_HASH_COUNT,
        }
    }

    /// Create from serialized bytes into a mutable Vec (for building/mutation).
    /// Unlike `from_owned_bytes`, this copies data into a `Vec<u64>` so that
    /// `insert()` works. Used by the primary-key bloom cache.
    pub fn from_bytes_mutable(data: &[u8]) -> io::Result<Self> {
        if data.len() < BLOOM_FILTER_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data too short",
            ));
        }
        let num_bits = usize::try_from(u64::from_le_bytes(data[0..8].try_into().unwrap()))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Bloom filter bit count exceeds addressable memory",
                )
            })?;
        let num_hashes = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let num_words = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

        validate_bloom_header(data.len(), num_bits, num_hashes, num_words)?;
        let expected_len = BLOOM_FILTER_HEADER_SIZE + num_words * 8;
        if data.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data truncated",
            ));
        }

        let mut vec = vec![0u64; num_words];
        for (i, v) in vec.iter_mut().enumerate() {
            let off = BLOOM_FILTER_HEADER_SIZE + i * 8;
            *v = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        }

        Ok(Self {
            bits: BloomBits::Vec(vec),
            num_bits,
            num_hashes,
        })
    }

    /// Create from serialized OwnedBytes (zero-copy for mmap)
    pub fn from_owned_bytes(data: OwnedBytes) -> io::Result<Self> {
        if data.len() < BLOOM_FILTER_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data too short",
            ));
        }
        let d = data.as_slice();
        let num_bits =
            usize::try_from(u64::from_le_bytes(d[0..8].try_into().unwrap())).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Bloom filter bit count exceeds addressable memory",
                )
            })?;
        let num_hashes = u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize;
        let num_words = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;

        validate_bloom_header(d.len(), num_bits, num_hashes, num_words)?;
        let expected_len = BLOOM_FILTER_HEADER_SIZE + num_words * 8;
        if d.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data truncated",
            ));
        }

        // Slice past the header to get raw u64 LE words (zero-copy).
        let bits_bytes =
            data.slice(BLOOM_FILTER_HEADER_SIZE..BLOOM_FILTER_HEADER_SIZE + num_words * 8);

        Ok(Self {
            bits: BloomBits::Bytes(bits_bytes),
            num_bits,
            num_hashes,
        })
    }

    /// Serialized header + word bytes.
    pub fn serialized_len(&self) -> usize {
        BLOOM_FILTER_HEADER_SIZE + self.bits.len() * 8
    }

    /// Stream the serialized representation without an intermediate buffer.
    pub fn write_to(&self, writer: &mut (impl Write + ?Sized)) -> io::Result<()> {
        let num_words = self.bits.len();
        write_bloom_header(writer, self.num_bits, self.num_hashes, num_words)?;
        self.bits.write_to(writer)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.serialized_len());
        self.write_to(&mut data)
            .expect("writing a bloom filter to Vec cannot fail");
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
                self.bits.set_bit(word_idx, bit_idx);
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
            if word_idx >= self.bits.len() || (self.bits.get(word_idx) & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        BLOOM_FILTER_HEADER_SIZE + self.bits.size_bytes()
    }

    /// Insert a pre-computed hash pair into the filter
    pub fn insert_hashed(&mut self, h1: u64, h2: u64) {
        for i in 0..self.num_hashes {
            let bit_pos = self.get_bit_pos(h1, h2, i);
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if word_idx < self.bits.len() {
                self.bits.set_bit(word_idx, bit_idx);
            }
        }
    }

    /// Compute two hash values using FNV-1a variant (single pass over key bytes)
    #[inline]
    fn hash_pair(&self, key: &[u8]) -> (u64, u64) {
        let mut h1: u64 = 0xcbf29ce484222325;
        let mut h2: u64 = 0x84222325cbf29ce4;
        for &byte in key {
            h1 ^= byte as u64;
            h1 = h1.wrapping_mul(0x100000001b3);
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

fn write_bloom_header(
    writer: &mut (impl Write + ?Sized),
    num_bits: usize,
    num_hashes: usize,
    num_words: usize,
) -> io::Result<()> {
    writer.write_u64::<LittleEndian>(u64::try_from(num_bits).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Bloom filter bit count exceeds u64",
        )
    })?)?;
    writer.write_u32::<LittleEndian>(u32::try_from(num_hashes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Bloom filter hash count exceeds u32",
        )
    })?)?;
    writer.write_u32::<LittleEndian>(u32::try_from(num_words).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Bloom filter word count exceeds u32",
        )
    })?)?;
    Ok(())
}

fn validate_bloom_header(
    data_len: usize,
    num_bits: usize,
    num_hashes: usize,
    num_words: usize,
) -> io::Result<()> {
    if num_bits == 0 || num_hashes == 0 || num_hashes > 32 || num_words == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid bloom filter parameters",
        ));
    }
    let word_bytes = num_words
        .checked_mul(8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bloom filter size overflow"))?;
    let expected_len = BLOOM_FILTER_HEADER_SIZE
        .checked_add(word_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bloom filter size overflow"))?;
    let capacity_bits = num_words
        .checked_mul(64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bloom filter size overflow"))?;
    if expected_len > data_len
        || num_bits > capacity_bits
        || num_bits <= capacity_bits.saturating_sub(64)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "inconsistent bloom filter dimensions",
        ));
    }
    Ok(())
}

/// Compute bloom filter hash pair for a key (standalone, no BloomFilter needed).
/// Uses the same FNV-1a double-hashing as BloomFilter::hash_pair (single pass).
#[inline]
fn bloom_hash_pair(key: &[u8]) -> (u64, u64) {
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x84222325cbf29ce4;
    for &byte in key {
        h1 ^= byte as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 = h2.wrapping_mul(0x100000001b3);
        h2 ^= byte as u64;
    }
    (h1, h2)
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
        let len = usize::try_from(read_vint(reader)?).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "SSTable value length too large")
        })?;
        if len > MAX_SSTABLE_BLOCK_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable value exceeds block safety limit",
            ));
        }
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
        let length = u32::try_from(read_vint(reader)?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "sparse posting length exceeds u32",
            )
        })?;
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
        posting_len: u64,
        doc_freq: u32,
        /// Position data offset (0 if no positions)
        position_offset: u64,
        /// Position data length (0 if no positions)
        position_len: u64,
    },
}

impl TermInfo {
    /// Create an external reference
    pub fn external(posting_offset: u64, posting_len: u64, doc_freq: u32) -> Self {
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
        posting_len: u64,
        doc_freq: u32,
        position_offset: u64,
        position_len: u64,
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
        if doc_ids.len() > MAX_INLINE_POSTINGS
            || doc_ids.is_empty()
            || doc_ids.len() != term_freqs.len()
        {
            return None;
        }

        let mut data = [0u8; 16];
        let mut cursor = std::io::Cursor::new(&mut data[..]);
        let mut prev_doc_id = 0u32;

        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let delta = doc_id.checked_sub(prev_doc_id)?;
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

    /// Try to create an inline TermInfo from an iterator of (doc_id, term_freq) pairs.
    /// Zero-allocation alternative to `try_inline` — avoids collecting into Vec<u32>.
    /// `count` is the number of postings (must match iterator length).
    pub fn try_inline_iter(count: usize, iter: impl Iterator<Item = (u32, u32)>) -> Option<Self> {
        if count > MAX_INLINE_POSTINGS || count == 0 {
            return None;
        }

        let mut data = [0u8; 16];
        let mut cursor = std::io::Cursor::new(&mut data[..]);
        let mut prev_doc_id = 0u32;

        let mut actual_count = 0usize;
        for (doc_id, tf) in iter {
            if actual_count >= count {
                return None;
            }
            let delta = doc_id.checked_sub(prev_doc_id)?;
            if write_vint(&mut cursor, delta as u64).is_err() {
                return None;
            }
            if write_vint(&mut cursor, tf as u64).is_err() {
                return None;
            }
            prev_doc_id = doc_id;
            actual_count += 1;
        }

        if actual_count != count {
            return None;
        }

        let data_len = cursor.position() as u8;

        Some(TermInfo::Inline {
            doc_freq: count as u8,
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
    pub fn external_info(&self) -> Option<(u64, u64)> {
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
    pub fn position_info(&self) -> Option<(u64, u64)> {
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
                if *doc_freq == 0
                    || *doc_freq as usize > MAX_INLINE_POSTINGS
                    || *data_len as usize > data.len()
                {
                    return None;
                }
                let mut doc_ids = Vec::with_capacity(*doc_freq as usize);
                let mut term_freqs = Vec::with_capacity(*doc_freq as usize);
                let mut reader = &data[..*data_len as usize];
                let mut prev_doc_id = 0u32;

                for _ in 0..*doc_freq {
                    let delta = read_vint(&mut reader).ok()? as u32;
                    let tf = read_vint(&mut reader).ok()? as u32;
                    let doc_id = prev_doc_id.checked_add(delta)?;
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
                if *doc_freq == 0
                    || *doc_freq as usize > MAX_INLINE_POSTINGS
                    || *data_len as usize > data.len()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid inline TermInfo",
                    ));
                }
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
                    write_vint(writer, *posting_len)?;
                    write_vint(writer, *position_offset)?;
                    write_vint(writer, *position_len)?;
                } else {
                    writer.write_u8(0x00)?;
                    write_vint(writer, *doc_freq as u64)?;
                    write_vint(writer, *posting_offset)?;
                    write_vint(writer, *posting_len)?;
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
            if doc_freq == 0 || doc_freq as usize > MAX_INLINE_POSTINGS || data_len as usize > 16 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid inline TermInfo lengths",
                ));
            }
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
            let posting_len = read_vint(reader)?;
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
            let posting_len = read_vint(reader)?;
            let position_offset = read_vint(reader)?;
            let position_len = read_vint(reader)?;
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
                use_bloom_filter: true, // Bloom is cheap (~1.25 B/key) and avoids needless block reads
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
pub struct SSTableWriter<W: Write, V: SSTableValue> {
    writer: W,
    block_buffer: Vec<u8>,
    prev_key: Vec<u8>,
    index: Vec<BlockIndexEntry>,
    current_offset: u64,
    num_entries: u64,
    block_first_key: Option<Vec<u8>>,
    config: SSTableWriterConfig,
    /// Pre-trained dictionary for compression (optional)
    dictionary: Option<CompressionDict>,
    /// Bloom filter key hashes — compact (u64, u64) pairs instead of full keys.
    /// Filter is built at finish() time with correct sizing.
    bloom_hashes: Vec<(u64, u64)>,
    _phantom: std::marker::PhantomData<V>,
}

impl<W: Write, V: SSTableValue> SSTableWriter<W, V> {
    /// Create a new SSTable writer with default configuration
    pub fn new(writer: W) -> Self {
        Self::with_config(writer, SSTableWriterConfig::default())
    }

    /// Create a new SSTable writer with custom configuration
    pub fn with_config(writer: W, config: SSTableWriterConfig) -> Self {
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
            bloom_hashes: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new SSTable writer with a pre-trained dictionary
    pub fn with_dictionary(
        writer: W,
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
            bloom_hashes: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn insert(&mut self, key: &[u8], value: &V) -> io::Result<()> {
        if self.block_first_key.is_none() {
            self.block_first_key = Some(key.to_vec());
        }

        // Store compact hash pair for bloom filter (16 bytes vs ~48+ per key)
        if self.config.use_bloom_filter {
            self.bloom_hashes.push(bloom_hash_pair(key));
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

    pub fn finish(mut self) -> io::Result<W> {
        // Flush any remaining data
        self.flush_block()?;

        // Build bloom filter from collected hashes (properly sized)
        let bloom_filter = if self.config.use_bloom_filter && !self.bloom_hashes.is_empty() {
            let mut bloom =
                BloomFilter::new(self.bloom_hashes.len(), self.config.bloom_bits_per_key);
            for (h1, h2) in &self.bloom_hashes {
                bloom.insert_hashed(*h1, *h2);
            }
            Some(bloom)
        } else {
            None
        };

        let data_end_offset = self.current_offset;

        // Build memory-efficient block index
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
        let bloom_offset = if let Some(ref bloom) = bloom_filter {
            let offset = self.current_offset;
            bloom.write_to(&mut self.writer)?;
            self.current_offset += bloom.serialized_len() as u64;
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

        // Write extended footer
        self.writer.write_u64::<LittleEndian>(data_end_offset)?;
        self.writer.write_u64::<LittleEndian>(self.num_entries)?;
        self.writer.write_u64::<LittleEndian>(bloom_offset)?; // 0 if no bloom
        self.writer.write_u64::<LittleEndian>(dict_offset)?; // 0 if no dict
        self.writer
            .write_u8(self.config.compression_level.0 as u8)?;
        self.writer.write_u32::<LittleEndian>(SSTABLE_MAGIC)?;

        Ok(self.writer)
    }
}

/// Block index entry
#[derive(Debug, Clone)]
struct BlockIndexEntry {
    first_key: Vec<u8>,
    offset: u64,
    length: u32,
}

/// Async SSTable reader - loads blocks on demand via FileHandle
///
/// Memory-efficient design:
/// - Block index uses FST (native) or mmap'd raw bytes - no heap allocation for keys
/// - Block addresses stored in bitpacked format
/// - Bloom filter and dictionary optional
pub struct AsyncSSTableReader<V: SSTableValue> {
    /// FileHandle for the data portion (blocks only) - fetches ranges on demand
    data_slice: FileHandle,
    /// Memory-efficient block index (FST or mmap)
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

/// Bounded block cache with a contention-free read path.
///
/// Normal reads use [`BlockCache::peek`] under a shared lock and deliberately
/// do not promote hits, so normal eviction order is insertion order. `get()`
/// still promotes entries for exclusive warm-up and race-resolution paths.
struct BlockCache {
    blocks: FxHashMap<u64, Arc<[u8]>>,
    lru_order: std::collections::VecDeque<u64>,
    max_blocks: usize,
}

impl BlockCache {
    fn new(max_blocks: usize) -> Self {
        Self {
            blocks: FxHashMap::default(),
            lru_order: std::collections::VecDeque::with_capacity(max_blocks),
            max_blocks,
        }
    }

    fn get(&mut self, offset: u64) -> Option<Arc<[u8]>> {
        if self.blocks.contains_key(&offset) {
            self.promote(offset);
            self.blocks.get(&offset).map(Arc::clone)
        } else {
            None
        }
    }

    /// Read-only cache probe — no LRU promotion, safe behind a read lock.
    fn peek(&self, offset: u64) -> Option<Arc<[u8]>> {
        self.blocks.get(&offset).map(Arc::clone)
    }

    fn insert(&mut self, offset: u64, block: Arc<[u8]>) {
        if self.max_blocks == 0 {
            return;
        }
        if self.blocks.contains_key(&offset) {
            self.promote(offset);
            return;
        }
        while self.blocks.len() >= self.max_blocks {
            if let Some(evict_offset) = self.lru_order.pop_front() {
                self.blocks.remove(&evict_offset);
            } else {
                break;
            }
        }
        self.blocks.insert(offset, block);
        self.lru_order.push_back(offset);
    }

    /// Move entry to MRU position (back of deque)
    fn promote(&mut self, offset: u64) {
        if let Some(pos) = self.lru_order.iter().position(|&k| k == offset) {
            self.lru_order.remove(pos);
            self.lru_order.push_back(offset);
        }
    }
}

impl<V: SSTableValue> AsyncSSTableReader<V> {
    /// Open an SSTable from a FileHandle
    /// Only loads the footer and index into memory, data blocks fetched on-demand
    ///
    /// Uses FST-based (native) or mmap'd block index (no heap allocation for keys)
    pub async fn open(file_handle: FileHandle, cache_blocks: usize) -> io::Result<Self> {
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

        let footer_start = file_len - 37;
        if data_end_offset > footer_start {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable data section extends past its footer",
            ));
        }
        if bloom_offset != 0 && (bloom_offset < data_end_offset || bloom_offset >= footer_start) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable bloom filter offset is out of bounds",
            ));
        }
        if dict_offset != 0
            && (dict_offset < data_end_offset
                || dict_offset >= footer_start
                || (bloom_offset != 0 && dict_offset <= bloom_offset))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable dictionary offset is out of bounds",
            ));
        }

        // Read index section
        let index_start = data_end_offset;
        let index_end = if bloom_offset != 0 {
            bloom_offset
        } else if dict_offset != 0 {
            dict_offset
        } else {
            footer_start
        };
        let index_bytes = file_handle.read_bytes_range(index_start..index_end).await?;

        // Parse block index (length-prefixed FST or mmap index)
        let mut idx_reader = index_bytes.as_slice();
        let index_len = idx_reader.read_u32::<LittleEndian>()? as usize;

        if index_len != idx_reader.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Index data truncated",
            ));
        }

        let index_data = index_bytes.slice(4..4 + index_len);

        // Try FST first (when fst-index feature available), fall back to mmap
        #[cfg(feature = "fst-index")]
        let block_index = match FstBlockIndex::load(index_data.clone()) {
            Ok(fst_idx) => BlockIndex::Fst(fst_idx),
            Err(_) => BlockIndex::Mmap(MmapBlockIndex::load(index_data)?),
        };
        #[cfg(not(feature = "fst-index"))]
        let block_index = BlockIndex::Mmap(MmapBlockIndex::load(index_data)?);

        let mut expected_offset = 0u64;
        for addr in block_index.all_addrs() {
            let end = addr.offset.checked_add(addr.length as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "SSTable block range overflow")
            })?;
            if addr.length == 0 || addr.offset != expected_offset || end > data_end_offset {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "SSTable block addresses are inconsistent",
                ));
            }
            expected_offset = end;
        }
        if expected_offset != data_end_offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SSTable block addresses do not cover the data section",
            ));
        }

        // Load bloom filter if present
        let bloom_filter = if bloom_offset > 0 {
            let bloom_start = bloom_offset;
            let bloom_end = if dict_offset != 0 {
                dict_offset
            } else {
                footer_start
            };
            // Read the canonical header first to determine the payload size.
            let header_size = BloomFilter::SERIALIZED_HEADER_SIZE as u64;
            let bloom_header_end = bloom_start.checked_add(header_size).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "bloom filter range overflow")
            })?;
            if bloom_header_end > bloom_end {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "bloom filter header is truncated",
                ));
            }
            let bloom_header = file_handle
                .read_bytes_range(bloom_start..bloom_header_end)
                .await?;
            let num_words = u32::from_le_bytes([
                bloom_header[12],
                bloom_header[13],
                bloom_header[14],
                bloom_header[15],
            ]) as u64;
            let bloom_size = num_words
                .checked_mul(8)
                .and_then(|bytes| bytes.checked_add(header_size))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bloom filter size overflow")
                })?;
            let actual_bloom_size = bloom_end - bloom_start;
            if bloom_size != actual_bloom_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bloom filter length is inconsistent",
                ));
            }
            let bloom_data = file_handle.read_bytes_range(bloom_start..bloom_end).await?;
            Some(BloomFilter::from_owned_bytes(bloom_data)?)
        } else {
            None
        };

        // Load dictionary if present
        let dictionary = if dict_offset > 0 {
            let dict_start = dict_offset;
            // Read dictionary size first
            let dict_header_end = dict_start.checked_add(4).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "dictionary range overflow")
            })?;
            if dict_header_end > footer_start {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "dictionary header is truncated",
                ));
            }
            let dict_len_bytes = file_handle
                .read_bytes_range(dict_start..dict_header_end)
                .await?;
            let dict_len = u32::from_le_bytes([
                dict_len_bytes[0],
                dict_len_bytes[1],
                dict_len_bytes[2],
                dict_len_bytes[3],
            ]) as u64;
            if dict_len > MAX_SSTABLE_DICTIONARY_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "SSTable dictionary exceeds safety limit",
                ));
            }
            let dict_end = dict_header_end.checked_add(dict_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "dictionary range overflow")
            })?;
            if dict_end != footer_start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "SSTable dictionary length is inconsistent",
                ));
            }
            let dict_data = file_handle
                .read_bytes_range(dict_header_end..dict_end)
                .await?;
            Some(CompressionDict::from_owned_bytes(dict_data))
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

    /// Heap bytes retained by decompressed cached blocks.
    ///
    /// This deliberately reports the actual block lengths instead of assuming
    /// the configured writer block size: compression dictionaries and boundary
    /// blocks make the retained size variable.
    pub fn cached_bytes(&self) -> usize {
        self.cache
            .read()
            .blocks
            .values()
            .map(|block| block.len())
            .sum()
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
                let end = addr.offset.checked_add(addr.length as u64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "SSTable block range overflow")
                })?;
                max_end = max_end.max(end);
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
            let start = usize::try_from(addr.offset).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "SSTable block offset too large")
            })?;
            let end = addr
                .offset
                .checked_add(addr.length as u64)
                .and_then(|end| usize::try_from(end).ok())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "SSTable block range overflow")
                })?;
            let compressed = buf.get(start..end).ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "SSTable block is truncated")
            })?;
            let decompressed = if let Some(ref dict) = self.dictionary {
                crate::compression::decompress_with_dict_limited(
                    compressed,
                    dict,
                    MAX_SSTABLE_BLOCK_BYTES,
                )?
            } else {
                crate::compression::decompress_limited(compressed, MAX_SSTABLE_BLOCK_BYTES)?
            };
            cache.insert(addr.offset, Arc::from(decompressed));
        }

        Ok(())
    }

    /// Load a block (checks cache first, then loads from FileSlice)
    /// Uses dictionary decompression if dictionary is present
    async fn load_block(&self, block_idx: usize) -> io::Result<Arc<[u8]>> {
        let addr = self.block_index.get_addr(block_idx).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Block index out of range")
        })?;

        // Fast path: read-lock peek (no LRU promotion, zero writer contention)
        {
            if let Some(block) = self.cache.read().peek(addr.offset) {
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
            crate::compression::decompress_with_dict_limited(
                compressed.as_slice(),
                dict,
                MAX_SSTABLE_BLOCK_BYTES,
            )?
        } else {
            crate::compression::decompress_limited(compressed.as_slice(), MAX_SSTABLE_BLOCK_BYTES)?
        };

        let block: Arc<[u8]> = Arc::from(decompressed);

        // Insert into cache under the write lock.
        {
            let mut cache = self.cache.write();
            cache.insert(addr.offset, Arc::clone(&block));
        }

        Ok(block)
    }

    /// Synchronous block load — only works for Inline (mmap/RAM) file handles.
    #[cfg(feature = "sync")]
    fn load_block_sync(&self, block_idx: usize) -> io::Result<Arc<[u8]>> {
        let addr = self.block_index.get_addr(block_idx).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Block index out of range")
        })?;

        // Fast path: read-lock peek (no LRU promotion, zero writer contention)
        {
            if let Some(block) = self.cache.read().peek(addr.offset) {
                return Ok(block);
            }
        }

        // Load from FileSlice (sync — requires Inline handle)
        let range = addr.byte_range();
        let compressed = self.data_slice.read_bytes_range_sync(range)?;

        // Decompress with dictionary if available
        let decompressed = if let Some(ref dict) = self.dictionary {
            crate::compression::decompress_with_dict_limited(
                compressed.as_slice(),
                dict,
                MAX_SSTABLE_BLOCK_BYTES,
            )?
        } else {
            crate::compression::decompress_limited(compressed.as_slice(), MAX_SSTABLE_BLOCK_BYTES)?
        };

        let block: Arc<[u8]> = Arc::from(decompressed);

        // Insert into cache under the write lock.
        {
            let mut cache = self.cache.write();
            cache.insert(addr.offset, Arc::clone(&block));
        }

        Ok(block)
    }

    /// Synchronous key lookup — only works for Inline (mmap/RAM) file handles.
    #[cfg(feature = "sync")]
    pub fn get_sync(&self, key: &[u8]) -> io::Result<Option<V>> {
        // Check bloom filter first — fast negative lookup
        if let Some(ref bloom) = self.bloom_filter
            && !bloom.may_contain(key)
        {
            return Ok(None);
        }

        // Use block index to find the block that could contain the key
        let block_idx = match self.block_index.locate(key) {
            Some(idx) => idx,
            None => {
                return Ok(None);
            }
        };

        let block_data = self.load_block_sync(block_idx)?;
        self.search_block(&block_data, key)
    }

    fn search_block(&self, block_data: &[u8], target_key: &[u8]) -> io::Result<Option<V>> {
        let mut reader = block_data;
        let mut current_key = Vec::new();

        while !reader.is_empty() {
            let common_prefix_len = read_vint(&mut reader)? as usize;
            let suffix_len = read_vint(&mut reader)? as usize;

            if suffix_len > reader.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "SSTable block suffix truncated",
                ));
            }
            current_key.truncate(common_prefix_len);
            current_key.extend_from_slice(&reader[..suffix_len]);
            reader = &reader[suffix_len..];

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
            let mut reader = &block_data[..];
            let mut current_key = Vec::new();

            while !reader.is_empty() {
                let common_prefix_len = read_vint(&mut reader)? as usize;
                let suffix_len = read_vint(&mut reader)? as usize;

                if suffix_len > reader.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "SSTable block suffix truncated",
                    ));
                }
                current_key.truncate(common_prefix_len);
                current_key.extend_from_slice(&reader[..suffix_len]);
                reader = &reader[suffix_len..];

                let value = V::deserialize(&mut reader)?;
                results.push((current_key.clone(), value));
            }
        }

        Ok(results)
    }

    /// Scan all entries whose key starts with `prefix`.
    ///
    /// Uses the block index to locate the starting block, then iterates
    /// forward collecting matching entries. Early-terminates once keys
    /// exceed the prefix range (keys are sorted).
    pub async fn prefix_scan(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, V)>> {
        let (results, _) = self.prefix_scan_limited(prefix, usize::MAX).await?;
        Ok(results)
    }

    /// Prefix scan with an explicit result budget. The boolean indicates that
    /// at least one additional matching entry existed beyond the budget.
    pub async fn prefix_scan_limited(
        &self,
        prefix: &[u8],
        max_results: usize,
    ) -> io::Result<PrefixScanResult<V>> {
        if self.block_index.is_empty() || prefix.is_empty() {
            return Ok((Vec::new(), false));
        }

        let start_block = match self.block_index.locate(prefix) {
            Some(idx) => idx,
            None => return Ok((Vec::new(), false)),
        };

        let mut results = Vec::new();

        for block_idx in start_block..self.block_index.len() {
            let block_data = self.load_block(block_idx).await?;
            let mut reader = &block_data[..];
            let mut current_key = Vec::new();

            while !reader.is_empty() {
                let common_prefix_len = read_vint(&mut reader)? as usize;
                let suffix_len = read_vint(&mut reader)? as usize;

                if suffix_len > reader.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "SSTable block suffix truncated",
                    ));
                }
                current_key.truncate(common_prefix_len);
                current_key.extend_from_slice(&reader[..suffix_len]);
                reader = &reader[suffix_len..];

                let value = V::deserialize(&mut reader)?;

                if current_key.starts_with(prefix) {
                    if results.len() >= max_results {
                        return Ok((results, true));
                    }
                    results.push((current_key.clone(), value));
                } else if current_key.as_slice() > prefix {
                    // Keys are sorted — past the prefix range, done
                    return Ok((results, false));
                }
            }
        }

        Ok((results, false))
    }

    /// Synchronous prefix scan — requires Inline (mmap/RAM) file handles.
    #[cfg(feature = "sync")]
    pub fn prefix_scan_sync(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, V)>> {
        let (results, _) = self.prefix_scan_limited_sync(prefix, usize::MAX)?;
        Ok(results)
    }

    /// Synchronous prefix scan with an explicit result budget.
    #[cfg(feature = "sync")]
    pub fn prefix_scan_limited_sync(
        &self,
        prefix: &[u8],
        max_results: usize,
    ) -> io::Result<PrefixScanResult<V>> {
        if self.block_index.is_empty() || prefix.is_empty() {
            return Ok((Vec::new(), false));
        }

        let start_block = match self.block_index.locate(prefix) {
            Some(idx) => idx,
            None => return Ok((Vec::new(), false)),
        };

        let mut results = Vec::new();

        for block_idx in start_block..self.block_index.len() {
            let block_data = self.load_block_sync(block_idx)?;
            let mut reader = &block_data[..];
            let mut current_key = Vec::new();

            while !reader.is_empty() {
                let common_prefix_len = read_vint(&mut reader)? as usize;
                let suffix_len = read_vint(&mut reader)? as usize;

                if suffix_len > reader.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "SSTable block suffix truncated",
                    ));
                }
                current_key.truncate(common_prefix_len);
                current_key.extend_from_slice(&reader[..suffix_len]);
                reader = &reader[suffix_len..];

                let value = V::deserialize(&mut reader)?;

                if current_key.starts_with(prefix) {
                    if results.len() >= max_results {
                        return Ok((results, true));
                    }
                    results.push((current_key.clone(), value));
                } else if current_key.as_slice() > prefix {
                    return Ok((results, false));
                }
            }
        }

        Ok((results, false))
    }
}

/// Async iterator over SSTable entries
pub struct AsyncSSTableIterator<'a, V: SSTableValue> {
    reader: &'a AsyncSSTableReader<V>,
    current_block: usize,
    block_data: Option<Arc<[u8]>>,
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

            if suffix_len > reader.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "SSTable block suffix truncated",
                ));
            }
            self.current_key.truncate(common_prefix_len);
            self.current_key.extend_from_slice(&reader[..suffix_len]);
            reader = &reader[suffix_len..];

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
        let restored = BloomFilter::from_owned_bytes(OwnedBytes::new(bytes)).unwrap();

        assert!(restored.may_contain(b"key1"));
        assert!(restored.may_contain(b"key2"));
        assert!(!restored.may_contain(b"key3"));
    }

    #[test]
    fn bloom_header_preserves_bit_counts_above_u32() {
        let num_bits = u32::MAX as usize + 1;
        let mut header = Vec::new();
        write_bloom_header(&mut header, num_bits, BLOOM_HASH_COUNT, 1).unwrap();
        assert_eq!(header.len(), BLOOM_FILTER_HEADER_SIZE);
        assert_eq!(
            u64::from_le_bytes(header[0..8].try_into().unwrap()),
            num_bits as u64
        );
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
        assert!(config.use_bloom_filter); // Bloom always on — cheap and fast
        assert!(!config.use_dictionary);

        // Adaptive
        let adaptive = SSTableWriterConfig::from_optimization(IndexOptimization::Adaptive);
        assert_eq!(adaptive.compression_level.0, 9);
        assert!(adaptive.use_bloom_filter);
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
