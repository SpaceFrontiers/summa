//! Sparse vector posting list with quantized weights
//!
//! Sparse vectors are stored as inverted index posting lists where:
//! - Each dimension ID is a "term"
//! - Each document has a weight for that dimension
//!
//! ## Configurable Components
//!
//! **Index (term/dimension ID) size:**
//! - `IndexSize::U16`: 16-bit indices (0-65535), ideal for SPLADE (~30K vocab)
//! - `IndexSize::U32`: 32-bit indices (0-4B), for large vocabularies
//!
//! **Weight quantization:**
//! - `Float32`: Full precision (4 bytes per weight)
//! - `Float16`: Half precision (2 bytes per weight)
//! - `UInt8`: 8-bit quantization with scale factor (1 byte per weight)
//! - `UInt4`: 4-bit quantization with scale factor (0.5 bytes per weight)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

use super::posting_common::{
    RoundedBitWidth, pack_deltas_fixed, read_vint, unpack_deltas_fixed, write_vint,
};
use crate::DocId;

/// Size of the index (term/dimension ID) in sparse vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum IndexSize {
    /// 16-bit index (0-65535), ideal for SPLADE vocabularies
    U16 = 0,
    /// 32-bit index (0-4B), for large vocabularies
    #[default]
    U32 = 1,
}

impl IndexSize {
    /// Bytes per index
    pub fn bytes(&self) -> usize {
        match self {
            IndexSize::U16 => 2,
            IndexSize::U32 => 4,
        }
    }

    /// Maximum value representable
    pub fn max_value(&self) -> u32 {
        match self {
            IndexSize::U16 => u16::MAX as u32,
            IndexSize::U32 => u32::MAX,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(IndexSize::U16),
            1 => Some(IndexSize::U32),
            _ => None,
        }
    }
}

/// Quantization format for sparse vector weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum WeightQuantization {
    /// Full 32-bit float precision
    #[default]
    Float32 = 0,
    /// 16-bit float (half precision)
    Float16 = 1,
    /// 8-bit unsigned integer with scale factor
    UInt8 = 2,
    /// 4-bit unsigned integer with scale factor (packed, 2 per byte)
    UInt4 = 3,
}

impl WeightQuantization {
    /// Bytes per weight (approximate for UInt4)
    pub fn bytes_per_weight(&self) -> f32 {
        match self {
            WeightQuantization::Float32 => 4.0,
            WeightQuantization::Float16 => 2.0,
            WeightQuantization::UInt8 => 1.0,
            WeightQuantization::UInt4 => 0.5,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(WeightQuantization::Float32),
            1 => Some(WeightQuantization::Float16),
            2 => Some(WeightQuantization::UInt8),
            3 => Some(WeightQuantization::UInt4),
            _ => None,
        }
    }
}

/// Configuration for sparse vector storage
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SparseVectorConfig {
    /// Size of dimension/term indices
    pub index_size: IndexSize,
    /// Quantization for weights
    pub weight_quantization: WeightQuantization,
    /// Minimum weight threshold - weights below this value are not indexed
    /// This reduces index size and can improve query speed at the cost of recall
    #[serde(default)]
    pub weight_threshold: f32,
}

impl Default for SparseVectorConfig {
    fn default() -> Self {
        Self {
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float32,
            weight_threshold: 0.0,
        }
    }
}

impl SparseVectorConfig {
    /// SPLADE-optimized config: u16 indices, int8 weights
    pub fn splade() -> Self {
        Self {
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt8,
            weight_threshold: 0.0,
        }
    }

    /// Compact config: u16 indices, 4-bit weights
    pub fn compact() -> Self {
        Self {
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt4,
            weight_threshold: 0.0,
        }
    }

    /// Full precision config
    pub fn full_precision() -> Self {
        Self {
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float32,
            weight_threshold: 0.0,
        }
    }

    /// Set weight threshold (builder pattern)
    pub fn with_weight_threshold(mut self, threshold: f32) -> Self {
        self.weight_threshold = threshold;
        self
    }

    /// Bytes per entry (index + weight)
    pub fn bytes_per_entry(&self) -> f32 {
        self.index_size.bytes() as f32 + self.weight_quantization.bytes_per_weight()
    }

    /// Serialize config to a single byte
    pub fn to_byte(&self) -> u8 {
        ((self.index_size as u8) << 4) | (self.weight_quantization as u8)
    }

    /// Deserialize config from a single byte
    /// Note: weight_threshold is not serialized in the byte, defaults to 0.0
    pub fn from_byte(b: u8) -> Option<Self> {
        let index_size = IndexSize::from_u8(b >> 4)?;
        let weight_quantization = WeightQuantization::from_u8(b & 0x0F)?;
        Some(Self {
            index_size,
            weight_quantization,
            weight_threshold: 0.0,
        })
    }
}

/// A sparse vector entry: (dimension_id, weight)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseEntry {
    pub dim_id: u32,
    pub weight: f32,
}

/// Sparse vector representation
#[derive(Debug, Clone, Default)]
pub struct SparseVector {
    entries: Vec<SparseEntry>,
}

impl SparseVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Create from dimension IDs and weights
    pub fn from_entries(dim_ids: &[u32], weights: &[f32]) -> Self {
        assert_eq!(dim_ids.len(), weights.len());
        let mut entries: Vec<SparseEntry> = dim_ids
            .iter()
            .zip(weights.iter())
            .map(|(&dim_id, &weight)| SparseEntry { dim_id, weight })
            .collect();
        // Sort by dimension ID for efficient intersection
        entries.sort_by_key(|e| e.dim_id);
        Self { entries }
    }

    /// Add an entry (must maintain sorted order by dim_id)
    pub fn push(&mut self, dim_id: u32, weight: f32) {
        debug_assert!(
            self.entries.is_empty() || self.entries.last().unwrap().dim_id < dim_id,
            "Entries must be added in sorted order by dim_id"
        );
        self.entries.push(SparseEntry { dim_id, weight });
    }

    /// Number of non-zero dimensions
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SparseEntry> {
        self.entries.iter()
    }

    /// Compute dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.entries.len() && j < other.entries.len() {
            let a = &self.entries[i];
            let b = &other.entries[j];

            match a.dim_id.cmp(&b.dim_id) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += a.weight * b.weight;
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    /// L2 norm squared
    pub fn norm_squared(&self) -> f32 {
        self.entries.iter().map(|e| e.weight * e.weight).sum()
    }

    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }
}

/// A sparse posting entry: doc_id with quantized weight
#[derive(Debug, Clone, Copy)]
pub struct SparsePosting {
    pub doc_id: DocId,
    pub weight: f32,
}

/// Block size for sparse posting lists (matches OptP4D for SIMD alignment)
pub const SPARSE_BLOCK_SIZE: usize = 128;

/// Skip entry for sparse posting lists with block-max support
///
/// Extends the basic skip entry with `max_weight` for Block-Max WAND optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseSkipEntry {
    /// First doc_id in the block (absolute)
    pub first_doc: DocId,
    /// Last doc_id in the block
    pub last_doc: DocId,
    /// Byte offset to block data
    pub offset: u32,
    /// Maximum weight in this block (for Block-Max optimization)
    pub max_weight: f32,
}

impl SparseSkipEntry {
    pub fn new(first_doc: DocId, last_doc: DocId, offset: u32, max_weight: f32) -> Self {
        Self {
            first_doc,
            last_doc,
            offset,
            max_weight,
        }
    }

    /// Compute the maximum possible contribution of this block to a dot product
    ///
    /// For a query dimension with weight `query_weight`, the maximum contribution
    /// from this block is `query_weight * max_weight`.
    #[inline]
    pub fn block_max_contribution(&self, query_weight: f32) -> f32 {
        query_weight * self.max_weight
    }

    /// Write skip entry to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.first_doc)?;
        writer.write_u32::<LittleEndian>(self.last_doc)?;
        writer.write_u32::<LittleEndian>(self.offset)?;
        writer.write_f32::<LittleEndian>(self.max_weight)?;
        Ok(())
    }

    /// Read skip entry from reader
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let first_doc = reader.read_u32::<LittleEndian>()?;
        let last_doc = reader.read_u32::<LittleEndian>()?;
        let offset = reader.read_u32::<LittleEndian>()?;
        let max_weight = reader.read_f32::<LittleEndian>()?;
        Ok(Self {
            first_doc,
            last_doc,
            offset,
            max_weight,
        })
    }
}

/// Skip list for sparse posting lists with block-max support
#[derive(Debug, Clone, Default)]
pub struct SparseSkipList {
    entries: Vec<SparseSkipEntry>,
    /// Global maximum weight across all blocks (for MaxScore pruning)
    global_max_weight: f32,
}

impl SparseSkipList {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a skip entry
    pub fn push(&mut self, first_doc: DocId, last_doc: DocId, offset: u32, max_weight: f32) {
        self.global_max_weight = self.global_max_weight.max(max_weight);
        self.entries.push(SparseSkipEntry::new(
            first_doc, last_doc, offset, max_weight,
        ));
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entry by index
    pub fn get(&self, index: usize) -> Option<&SparseSkipEntry> {
        self.entries.get(index)
    }

    /// Global maximum weight across all blocks
    pub fn global_max_weight(&self) -> f32 {
        self.global_max_weight
    }

    /// Find block index containing doc_id >= target
    pub fn find_block(&self, target: DocId) -> Option<usize> {
        self.entries.iter().position(|e| e.last_doc >= target)
    }

    /// Iterate over entries
    pub fn iter(&self) -> impl Iterator<Item = &SparseSkipEntry> {
        self.entries.iter()
    }

    /// Write skip list to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.entries.len() as u32)?;
        writer.write_f32::<LittleEndian>(self.global_max_weight)?;
        for entry in &self.entries {
            entry.write(writer)?;
        }
        Ok(())
    }

    /// Read skip list from reader
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let count = reader.read_u32::<LittleEndian>()? as usize;
        let global_max_weight = reader.read_f32::<LittleEndian>()?;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(SparseSkipEntry::read(reader)?);
        }
        Ok(Self {
            entries,
            global_max_weight,
        })
    }
}

/// Sparse posting list for a single dimension
///
/// Stores (doc_id, weight) pairs for all documents that have a non-zero
/// weight for this dimension. Weights are quantized according to the
/// specified quantization format.
#[derive(Debug, Clone)]
pub struct SparsePostingList {
    /// Quantization format
    quantization: WeightQuantization,
    /// Scale factor for UInt8/UInt4 quantization (weight = quantized * scale)
    scale: f32,
    /// Minimum value for UInt8/UInt4 quantization (weight = quantized * scale + min)
    min_val: f32,
    /// Number of postings
    doc_count: u32,
    /// Compressed data: [doc_ids...][weights...]
    data: Vec<u8>,
}

impl SparsePostingList {
    /// Create from postings with specified quantization
    pub fn from_postings(
        postings: &[(DocId, f32)],
        quantization: WeightQuantization,
    ) -> io::Result<Self> {
        if postings.is_empty() {
            return Ok(Self {
                quantization,
                scale: 1.0,
                min_val: 0.0,
                doc_count: 0,
                data: Vec::new(),
            });
        }

        // Compute min/max for quantization
        let weights: Vec<f32> = postings.iter().map(|(_, w)| *w).collect();
        let min_val = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let (scale, adjusted_min) = match quantization {
            WeightQuantization::Float32 | WeightQuantization::Float16 => (1.0, 0.0),
            WeightQuantization::UInt8 => {
                let range = max_val - min_val;
                if range < f32::EPSILON {
                    (1.0, min_val)
                } else {
                    (range / 255.0, min_val)
                }
            }
            WeightQuantization::UInt4 => {
                let range = max_val - min_val;
                if range < f32::EPSILON {
                    (1.0, min_val)
                } else {
                    (range / 15.0, min_val)
                }
            }
        };

        let mut data = Vec::new();

        // Write doc IDs with delta encoding
        let mut prev_doc_id = 0u32;
        for (doc_id, _) in postings {
            let delta = doc_id - prev_doc_id;
            write_vint(&mut data, delta as u64)?;
            prev_doc_id = *doc_id;
        }

        // Write weights based on quantization
        match quantization {
            WeightQuantization::Float32 => {
                for (_, weight) in postings {
                    data.write_f32::<LittleEndian>(*weight)?;
                }
            }
            WeightQuantization::Float16 => {
                // Use SIMD-accelerated batch conversion via half::slice
                use half::slice::HalfFloatSliceExt;
                let weights: Vec<f32> = postings.iter().map(|(_, w)| *w).collect();
                let mut f16_slice: Vec<half::f16> = vec![half::f16::ZERO; weights.len()];
                f16_slice.convert_from_f32_slice(&weights);
                for h in f16_slice {
                    data.write_u16::<LittleEndian>(h.to_bits())?;
                }
            }
            WeightQuantization::UInt8 => {
                for (_, weight) in postings {
                    let quantized = ((*weight - adjusted_min) / scale).round() as u8;
                    data.write_u8(quantized)?;
                }
            }
            WeightQuantization::UInt4 => {
                // Pack two 4-bit values per byte
                let mut i = 0;
                while i < postings.len() {
                    let q1 = ((postings[i].1 - adjusted_min) / scale).round() as u8 & 0x0F;
                    let q2 = if i + 1 < postings.len() {
                        ((postings[i + 1].1 - adjusted_min) / scale).round() as u8 & 0x0F
                    } else {
                        0
                    };
                    data.write_u8((q2 << 4) | q1)?;
                    i += 2;
                }
            }
        }

        Ok(Self {
            quantization,
            scale,
            min_val: adjusted_min,
            doc_count: postings.len() as u32,
            data,
        })
    }

    /// Serialize to bytes
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u8(self.quantization as u8)?;
        writer.write_f32::<LittleEndian>(self.scale)?;
        writer.write_f32::<LittleEndian>(self.min_val)?;
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_u32::<LittleEndian>(self.data.len() as u32)?;
        writer.write_all(&self.data)?;
        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let quant_byte = reader.read_u8()?;
        let quantization = WeightQuantization::from_u8(quant_byte).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Invalid quantization type")
        })?;
        let scale = reader.read_f32::<LittleEndian>()?;
        let min_val = reader.read_f32::<LittleEndian>()?;
        let doc_count = reader.read_u32::<LittleEndian>()?;
        let data_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        Ok(Self {
            quantization,
            scale,
            min_val,
            doc_count,
            data,
        })
    }

    /// Number of documents in this posting list
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Get quantization format
    pub fn quantization(&self) -> WeightQuantization {
        self.quantization
    }

    /// Create an iterator
    pub fn iterator(&self) -> SparsePostingIterator<'_> {
        SparsePostingIterator::new(self)
    }

    /// Decode all postings (for merge operations)
    pub fn decode_all(&self) -> io::Result<Vec<(DocId, f32)>> {
        let mut result = Vec::with_capacity(self.doc_count as usize);
        let mut iter = self.iterator();

        while !iter.exhausted {
            result.push((iter.doc_id, iter.weight));
            iter.advance();
        }

        Ok(result)
    }
}

/// Iterator over sparse posting list
pub struct SparsePostingIterator<'a> {
    posting_list: &'a SparsePostingList,
    /// Current position in doc_id stream
    doc_id_offset: usize,
    /// Current position in weight stream
    weight_offset: usize,
    /// Current index
    index: usize,
    /// Current doc_id
    doc_id: DocId,
    /// Current weight
    weight: f32,
    /// Whether iterator is exhausted
    exhausted: bool,
}

impl<'a> SparsePostingIterator<'a> {
    fn new(posting_list: &'a SparsePostingList) -> Self {
        let mut iter = Self {
            posting_list,
            doc_id_offset: 0,
            weight_offset: 0,
            index: 0,
            doc_id: 0,
            weight: 0.0,
            exhausted: posting_list.doc_count == 0,
        };

        if !iter.exhausted {
            // Calculate weight offset (after all doc_id deltas)
            iter.weight_offset = iter.calculate_weight_offset();
            iter.load_current();
        }

        iter
    }

    fn calculate_weight_offset(&self) -> usize {
        // Read through all doc_id deltas to find where weights start
        let mut offset = 0;
        let mut reader = &self.posting_list.data[..];

        for _ in 0..self.posting_list.doc_count {
            if read_vint(&mut reader).is_ok() {
                offset = self.posting_list.data.len() - reader.len();
            }
        }

        offset
    }

    fn load_current(&mut self) {
        if self.index >= self.posting_list.doc_count as usize {
            self.exhausted = true;
            return;
        }

        // Read doc_id delta
        let mut reader = &self.posting_list.data[self.doc_id_offset..];
        if let Ok(delta) = read_vint(&mut reader) {
            self.doc_id = self.doc_id.wrapping_add(delta as u32);
            self.doc_id_offset = self.posting_list.data.len() - reader.len();
        }

        // Read weight based on quantization
        let weight_idx = self.index;
        let pl = self.posting_list;

        self.weight = match pl.quantization {
            WeightQuantization::Float32 => {
                let offset = self.weight_offset + weight_idx * 4;
                if offset + 4 <= pl.data.len() {
                    let bytes = &pl.data[offset..offset + 4];
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    0.0
                }
            }
            WeightQuantization::Float16 => {
                let offset = self.weight_offset + weight_idx * 2;
                if offset + 2 <= pl.data.len() {
                    let bits = u16::from_le_bytes([pl.data[offset], pl.data[offset + 1]]);
                    half::f16::from_bits(bits).to_f32()
                } else {
                    0.0
                }
            }
            WeightQuantization::UInt8 => {
                let offset = self.weight_offset + weight_idx;
                if offset < pl.data.len() {
                    let quantized = pl.data[offset];
                    quantized as f32 * pl.scale + pl.min_val
                } else {
                    0.0
                }
            }
            WeightQuantization::UInt4 => {
                let byte_offset = self.weight_offset + weight_idx / 2;
                if byte_offset < pl.data.len() {
                    let byte = pl.data[byte_offset];
                    let quantized = if weight_idx.is_multiple_of(2) {
                        byte & 0x0F
                    } else {
                        (byte >> 4) & 0x0F
                    };
                    quantized as f32 * pl.scale + pl.min_val
                } else {
                    0.0
                }
            }
        };
    }

    /// Current document ID
    pub fn doc(&self) -> DocId {
        if self.exhausted {
            super::TERMINATED
        } else {
            self.doc_id
        }
    }

    /// Current weight
    pub fn weight(&self) -> f32 {
        if self.exhausted { 0.0 } else { self.weight }
    }

    /// Advance to next posting
    pub fn advance(&mut self) -> DocId {
        if self.exhausted {
            return super::TERMINATED;
        }

        self.index += 1;
        if self.index >= self.posting_list.doc_count as usize {
            self.exhausted = true;
            return super::TERMINATED;
        }

        self.load_current();
        self.doc_id
    }

    /// Seek to first doc_id >= target
    pub fn seek(&mut self, target: DocId) -> DocId {
        while !self.exhausted && self.doc_id < target {
            self.advance();
        }
        self.doc()
    }
}

/// Block-based sparse posting list for skip-list style access
///
/// Similar to BlockPostingList but stores quantized weights.
/// Includes block-max metadata for Block-Max WAND optimization.
#[derive(Debug, Clone)]
pub struct BlockSparsePostingList {
    /// Quantization format
    quantization: WeightQuantization,
    /// Global scale factor for UInt8/UInt4
    scale: f32,
    /// Global minimum value for UInt8/UInt4
    min_val: f32,
    /// Skip list with block-max support
    skip_list: SparseSkipList,
    /// Compressed block data
    data: Vec<u8>,
    /// Total number of postings
    doc_count: u32,
}

impl BlockSparsePostingList {
    /// Build from postings with specified quantization
    pub fn from_postings(
        postings: &[(DocId, f32)],
        quantization: WeightQuantization,
    ) -> io::Result<Self> {
        if postings.is_empty() {
            return Ok(Self {
                quantization,
                scale: 1.0,
                min_val: 0.0,
                skip_list: SparseSkipList::new(),
                data: Vec::new(),
                doc_count: 0,
            });
        }

        // Compute global min/max for quantization
        let weights: Vec<f32> = postings.iter().map(|(_, w)| *w).collect();
        let min_val = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let (scale, adjusted_min) = match quantization {
            WeightQuantization::Float32 | WeightQuantization::Float16 => (1.0, 0.0),
            WeightQuantization::UInt8 => {
                let range = max_val - min_val;
                if range < f32::EPSILON {
                    (1.0, min_val)
                } else {
                    (range / 255.0, min_val)
                }
            }
            WeightQuantization::UInt4 => {
                let range = max_val - min_val;
                if range < f32::EPSILON {
                    (1.0, min_val)
                } else {
                    (range / 15.0, min_val)
                }
            }
        };

        let mut skip_list = SparseSkipList::new();
        let mut data = Vec::new();

        let mut i = 0;
        while i < postings.len() {
            let block_end = (i + SPARSE_BLOCK_SIZE).min(postings.len());
            let block = &postings[i..block_end];

            let first_doc_id = block.first().unwrap().0;
            let last_doc_id = block.last().unwrap().0;

            // Compute max weight in this block for Block-Max optimization
            let block_max_weight = block
                .iter()
                .map(|(_, w)| *w)
                .fold(f32::NEG_INFINITY, f32::max);

            // Pack doc IDs with fixed-width delta encoding (SIMD-friendly)
            let block_doc_ids: Vec<DocId> = block.iter().map(|(d, _)| *d).collect();
            let (doc_bit_width, packed_doc_ids) = pack_deltas_fixed(&block_doc_ids);

            // Block header: [count: u16][doc_bit_width: u8][packed_doc_ids...][weights...]
            let block_start = data.len() as u32;
            skip_list.push(first_doc_id, last_doc_id, block_start, block_max_weight);

            data.write_u16::<LittleEndian>(block.len() as u16)?;
            data.write_u8(doc_bit_width as u8)?;
            data.extend_from_slice(&packed_doc_ids);

            // Write weights based on quantization
            match quantization {
                WeightQuantization::Float32 => {
                    for (_, weight) in block {
                        data.write_f32::<LittleEndian>(*weight)?;
                    }
                }
                WeightQuantization::Float16 => {
                    // Use SIMD-accelerated batch conversion via half::slice
                    use half::slice::HalfFloatSliceExt;
                    let weights: Vec<f32> = block.iter().map(|(_, w)| *w).collect();
                    let mut f16_slice: Vec<half::f16> = vec![half::f16::ZERO; weights.len()];
                    f16_slice.convert_from_f32_slice(&weights);
                    for h in f16_slice {
                        data.write_u16::<LittleEndian>(h.to_bits())?;
                    }
                }
                WeightQuantization::UInt8 => {
                    for (_, weight) in block {
                        let quantized = ((*weight - adjusted_min) / scale).round() as u8;
                        data.write_u8(quantized)?;
                    }
                }
                WeightQuantization::UInt4 => {
                    let mut j = 0;
                    while j < block.len() {
                        let q1 = ((block[j].1 - adjusted_min) / scale).round() as u8 & 0x0F;
                        let q2 = if j + 1 < block.len() {
                            ((block[j + 1].1 - adjusted_min) / scale).round() as u8 & 0x0F
                        } else {
                            0
                        };
                        data.write_u8((q2 << 4) | q1)?;
                        j += 2;
                    }
                }
            }

            i = block_end;
        }

        Ok(Self {
            quantization,
            scale,
            min_val: adjusted_min,
            skip_list,
            data,
            doc_count: postings.len() as u32,
        })
    }

    /// Serialize to bytes
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u8(self.quantization as u8)?;
        writer.write_f32::<LittleEndian>(self.scale)?;
        writer.write_f32::<LittleEndian>(self.min_val)?;
        writer.write_u32::<LittleEndian>(self.doc_count)?;

        // Write skip list with block-max support
        self.skip_list.write(writer)?;

        // Write data
        writer.write_u32::<LittleEndian>(self.data.len() as u32)?;
        writer.write_all(&self.data)?;

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let quant_byte = reader.read_u8()?;
        let quantization = WeightQuantization::from_u8(quant_byte).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Invalid quantization type")
        })?;
        let scale = reader.read_f32::<LittleEndian>()?;
        let min_val = reader.read_f32::<LittleEndian>()?;
        let doc_count = reader.read_u32::<LittleEndian>()?;

        // Read skip list with block-max support
        let skip_list = SparseSkipList::read(reader)?;

        let data_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        Ok(Self {
            quantization,
            scale,
            min_val,
            skip_list,
            data,
            doc_count,
        })
    }

    /// Number of documents
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Number of blocks
    pub fn num_blocks(&self) -> usize {
        self.skip_list.len()
    }

    /// Get quantization format
    pub fn quantization(&self) -> WeightQuantization {
        self.quantization
    }

    /// Global maximum weight across all blocks (for MaxScore pruning)
    pub fn global_max_weight(&self) -> f32 {
        self.skip_list.global_max_weight()
    }

    /// Get block-max weight for a specific block
    pub fn block_max_weight(&self, block_idx: usize) -> Option<f32> {
        self.skip_list.get(block_idx).map(|e| e.max_weight)
    }

    /// Compute maximum possible contribution to dot product with given query weight
    ///
    /// This is used for MaxScore pruning: if `query_weight * global_max_weight < threshold`,
    /// this entire dimension can be skipped.
    #[inline]
    pub fn max_contribution(&self, query_weight: f32) -> f32 {
        query_weight * self.skip_list.global_max_weight()
    }

    /// Create an iterator
    pub fn iterator(&self) -> BlockSparsePostingIterator<'_> {
        BlockSparsePostingIterator::new(self)
    }

    /// Approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        // Header: quantization (1) + scale (4) + min_val (4) + doc_count (4) = 13
        // Skip list: count (4) + global_max (4) + entries * (first_doc + last_doc + offset + max_weight) = 4 + 4 + n * 16
        // Data: data.len()
        13 + 8 + self.skip_list.len() * 16 + self.data.len()
    }

    /// Concatenate multiple posting lists with doc_id remapping
    pub fn concatenate(
        sources: &[(BlockSparsePostingList, u32)],
        target_quantization: WeightQuantization,
    ) -> io::Result<Self> {
        // Decode all postings and merge
        let mut all_postings: Vec<(DocId, f32)> = Vec::new();

        for (source, doc_offset) in sources {
            let decoded = source.decode_all()?;
            for (doc_id, weight) in decoded {
                all_postings.push((doc_id + doc_offset, weight));
            }
        }

        // Re-encode with target quantization
        Self::from_postings(&all_postings, target_quantization)
    }

    /// Decode all postings
    pub fn decode_all(&self) -> io::Result<Vec<(DocId, f32)>> {
        let mut result = Vec::with_capacity(self.doc_count as usize);
        let mut iter = self.iterator();

        while iter.doc() != super::TERMINATED {
            result.push((iter.doc(), iter.weight()));
            iter.advance();
        }

        Ok(result)
    }
}

/// Iterator over block sparse posting list
pub struct BlockSparsePostingIterator<'a> {
    posting_list: &'a BlockSparsePostingList,
    current_block: usize,
    block_postings: Vec<(DocId, f32)>,
    position_in_block: usize,
    exhausted: bool,
}

impl<'a> BlockSparsePostingIterator<'a> {
    fn new(posting_list: &'a BlockSparsePostingList) -> Self {
        let exhausted = posting_list.skip_list.is_empty();
        let mut iter = Self {
            posting_list,
            current_block: 0,
            block_postings: Vec::new(),
            position_in_block: 0,
            exhausted,
        };

        if !iter.exhausted {
            iter.load_block(0);
        }

        iter
    }

    fn load_block(&mut self, block_idx: usize) {
        let entry = match self.posting_list.skip_list.get(block_idx) {
            Some(e) => e,
            None => {
                self.exhausted = true;
                return;
            }
        };

        self.current_block = block_idx;
        self.position_in_block = 0;
        self.block_postings.clear();

        let offset = entry.offset as usize;
        let first_doc_id = entry.first_doc;
        let data = &self.posting_list.data[offset..];

        // Read block header: [count: u16][doc_bit_width: u8]
        if data.len() < 3 {
            self.exhausted = true;
            return;
        }
        let count = u16::from_le_bytes([data[0], data[1]]) as usize;
        let doc_bit_width = RoundedBitWidth::from_u8(data[2]).unwrap_or(RoundedBitWidth::Zero);

        // Unpack doc IDs with SIMD-accelerated delta decoding
        let doc_bytes = doc_bit_width.bytes_per_value() * count.saturating_sub(1);
        let doc_data = &data[3..3 + doc_bytes];
        let mut doc_ids = vec![0u32; count];
        unpack_deltas_fixed(doc_data, doc_bit_width, first_doc_id, count, &mut doc_ids);

        // Weight data starts after doc IDs
        let weight_offset = 3 + doc_bytes;
        let weight_data = &data[weight_offset..];
        let pl = self.posting_list;

        // Decode weights based on quantization (batch SIMD where possible)
        let weights: Vec<f32> = match pl.quantization {
            WeightQuantization::Float32 => {
                let mut weights = Vec::with_capacity(count);
                let mut reader = weight_data;
                for _ in 0..count {
                    if reader.len() >= 4 {
                        weights.push((&mut reader).read_f32::<LittleEndian>().unwrap_or(0.0));
                    } else {
                        weights.push(0.0);
                    }
                }
                weights
            }
            WeightQuantization::Float16 => {
                // Use SIMD-accelerated batch conversion via half::slice
                use half::slice::HalfFloatSliceExt;
                let mut f16_slice: Vec<half::f16> = Vec::with_capacity(count);
                for i in 0..count {
                    let offset = i * 2;
                    if offset + 2 <= weight_data.len() {
                        let bits =
                            u16::from_le_bytes([weight_data[offset], weight_data[offset + 1]]);
                        f16_slice.push(half::f16::from_bits(bits));
                    } else {
                        f16_slice.push(half::f16::ZERO);
                    }
                }
                let mut weights = vec![0.0f32; count];
                f16_slice.convert_to_f32_slice(&mut weights);
                weights
            }
            WeightQuantization::UInt8 => {
                let mut weights = Vec::with_capacity(count);
                for i in 0..count {
                    if i < weight_data.len() {
                        weights.push(weight_data[i] as f32 * pl.scale + pl.min_val);
                    } else {
                        weights.push(0.0);
                    }
                }
                weights
            }
            WeightQuantization::UInt4 => {
                let mut weights = Vec::with_capacity(count);
                for i in 0..count {
                    let byte_idx = i / 2;
                    if byte_idx < weight_data.len() {
                        let byte = weight_data[byte_idx];
                        let quantized = if i % 2 == 0 {
                            byte & 0x0F
                        } else {
                            (byte >> 4) & 0x0F
                        };
                        weights.push(quantized as f32 * pl.scale + pl.min_val);
                    } else {
                        weights.push(0.0);
                    }
                }
                weights
            }
        };

        // Combine doc_ids and weights into block_postings
        for (doc_id, weight) in doc_ids.into_iter().zip(weights.into_iter()) {
            self.block_postings.push((doc_id, weight));
        }
    }

    /// Check if iterator is exhausted
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Current document ID
    pub fn doc(&self) -> DocId {
        if self.exhausted {
            super::TERMINATED
        } else if self.position_in_block < self.block_postings.len() {
            self.block_postings[self.position_in_block].0
        } else {
            super::TERMINATED
        }
    }

    /// Current weight
    pub fn weight(&self) -> f32 {
        if self.exhausted || self.position_in_block >= self.block_postings.len() {
            0.0
        } else {
            self.block_postings[self.position_in_block].1
        }
    }

    /// Get current block's maximum weight (for Block-Max optimization)
    ///
    /// Returns the maximum weight of any posting in the current block.
    /// Used to compute upper bound contribution: `query_weight * current_block_max_weight()`.
    #[inline]
    pub fn current_block_max_weight(&self) -> f32 {
        if self.exhausted {
            0.0
        } else {
            self.posting_list
                .skip_list
                .get(self.current_block)
                .map(|e| e.max_weight)
                .unwrap_or(0.0)
        }
    }

    /// Compute maximum possible contribution from current block
    ///
    /// For Block-Max WAND: if this value < threshold, skip the entire block.
    #[inline]
    pub fn current_block_max_contribution(&self, query_weight: f32) -> f32 {
        query_weight * self.current_block_max_weight()
    }

    /// Advance to next posting
    pub fn advance(&mut self) -> DocId {
        if self.exhausted {
            return super::TERMINATED;
        }

        self.position_in_block += 1;
        if self.position_in_block >= self.block_postings.len() {
            self.load_block(self.current_block + 1);
        }

        self.doc()
    }

    /// Seek to first doc_id >= target
    pub fn seek(&mut self, target: DocId) -> DocId {
        if self.exhausted {
            return super::TERMINATED;
        }

        // Find target block using shared SkipList
        if let Some(block_idx) = self.posting_list.skip_list.find_block(target) {
            if block_idx != self.current_block {
                self.load_block(block_idx);
            }

            // Linear search within block
            while self.position_in_block < self.block_postings.len() {
                if self.block_postings[self.position_in_block].0 >= target {
                    return self.doc();
                }
                self.position_in_block += 1;
            }

            // Try next block
            self.load_block(self.current_block + 1);
            self.seek(target)
        } else {
            self.exhausted = true;
            super::TERMINATED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_dot_product() {
        let v1 = SparseVector::from_entries(&[0, 2, 5], &[1.0, 2.0, 3.0]);
        let v2 = SparseVector::from_entries(&[1, 2, 5], &[1.0, 4.0, 2.0]);

        // dot = 0 + 2*4 + 3*2 = 14
        assert!((v1.dot(&v2) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_posting_list_float32() {
        let postings = vec![(0, 1.5), (5, 2.3), (10, 0.8), (100, 3.15)];
        let pl = SparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        assert_eq!(pl.doc_count(), 4);

        let mut iter = pl.iterator();
        assert_eq!(iter.doc(), 0);
        assert!((iter.weight() - 1.5).abs() < 1e-6);

        iter.advance();
        assert_eq!(iter.doc(), 5);
        assert!((iter.weight() - 2.3).abs() < 1e-6);

        iter.advance();
        assert_eq!(iter.doc(), 10);

        iter.advance();
        assert_eq!(iter.doc(), 100);
        assert!((iter.weight() - 3.15).abs() < 1e-6);

        iter.advance();
        assert_eq!(iter.doc(), super::super::TERMINATED);
    }

    #[test]
    fn test_sparse_posting_list_uint8() {
        let postings = vec![(0, 0.0), (5, 0.5), (10, 1.0)];
        let pl = SparsePostingList::from_postings(&postings, WeightQuantization::UInt8).unwrap();

        let decoded = pl.decode_all().unwrap();
        assert_eq!(decoded.len(), 3);

        // UInt8 quantization should preserve relative ordering
        assert!(decoded[0].1 < decoded[1].1);
        assert!(decoded[1].1 < decoded[2].1);
    }

    #[test]
    fn test_block_sparse_posting_list() {
        // Create enough postings to span multiple blocks
        let postings: Vec<(DocId, f32)> = (0..300).map(|i| (i * 2, (i as f32) * 0.1)).collect();

        let pl =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        assert_eq!(pl.doc_count(), 300);
        assert!(pl.num_blocks() >= 2);

        // Test iteration
        let mut iter = pl.iterator();
        for (expected_doc, expected_weight) in &postings {
            assert_eq!(iter.doc(), *expected_doc);
            assert!((iter.weight() - expected_weight).abs() < 1e-6);
            iter.advance();
        }
        assert_eq!(iter.doc(), super::super::TERMINATED);
    }

    #[test]
    fn test_block_sparse_seek() {
        let postings: Vec<(DocId, f32)> = (0..500).map(|i| (i * 3, i as f32)).collect();

        let pl =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        let mut iter = pl.iterator();

        // Seek to exact match
        assert_eq!(iter.seek(300), 300);

        // Seek to non-exact (should find next)
        assert_eq!(iter.seek(301), 303);

        // Seek beyond end
        assert_eq!(iter.seek(2000), super::super::TERMINATED);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let postings: Vec<(DocId, f32)> = vec![(0, 1.0), (10, 2.0), (100, 3.0)];

        for quant in [
            WeightQuantization::Float32,
            WeightQuantization::Float16,
            WeightQuantization::UInt8,
        ] {
            let pl = BlockSparsePostingList::from_postings(&postings, quant).unwrap();

            let mut buffer = Vec::new();
            pl.serialize(&mut buffer).unwrap();

            let pl2 = BlockSparsePostingList::deserialize(&mut &buffer[..]).unwrap();

            assert_eq!(pl.doc_count(), pl2.doc_count());
            assert_eq!(pl.quantization(), pl2.quantization());

            // Verify iteration produces same results
            let mut iter1 = pl.iterator();
            let mut iter2 = pl2.iterator();

            while iter1.doc() != super::super::TERMINATED {
                assert_eq!(iter1.doc(), iter2.doc());
                // Allow some tolerance for quantization
                assert!((iter1.weight() - iter2.weight()).abs() < 0.1);
                iter1.advance();
                iter2.advance();
            }
        }
    }

    #[test]
    fn test_concatenate() {
        let postings1: Vec<(DocId, f32)> = vec![(0, 1.0), (5, 2.0)];
        let postings2: Vec<(DocId, f32)> = vec![(0, 3.0), (10, 4.0)];

        let pl1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();
        let pl2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge with doc_offset for second list
        let merged = BlockSparsePostingList::concatenate(
            &[(pl1, 0), (pl2, 100)],
            WeightQuantization::Float32,
        )
        .unwrap();

        assert_eq!(merged.doc_count(), 4);

        let decoded = merged.decode_all().unwrap();
        assert_eq!(decoded[0], (0, 1.0));
        assert_eq!(decoded[1], (5, 2.0));
        assert_eq!(decoded[2], (100, 3.0)); // 0 + 100 offset
        assert_eq!(decoded[3], (110, 4.0)); // 10 + 100 offset
    }

    #[test]
    fn test_sparse_vector_config() {
        // Test default config
        let default = SparseVectorConfig::default();
        assert_eq!(default.index_size, IndexSize::U32);
        assert_eq!(default.weight_quantization, WeightQuantization::Float32);
        assert_eq!(default.bytes_per_entry(), 8.0); // 4 + 4

        // Test SPLADE config
        let splade = SparseVectorConfig::splade();
        assert_eq!(splade.index_size, IndexSize::U16);
        assert_eq!(splade.weight_quantization, WeightQuantization::UInt8);
        assert_eq!(splade.bytes_per_entry(), 3.0); // 2 + 1

        // Test compact config
        let compact = SparseVectorConfig::compact();
        assert_eq!(compact.index_size, IndexSize::U16);
        assert_eq!(compact.weight_quantization, WeightQuantization::UInt4);
        assert_eq!(compact.bytes_per_entry(), 2.5); // 2 + 0.5

        // Test serialization roundtrip
        let byte = splade.to_byte();
        let restored = SparseVectorConfig::from_byte(byte).unwrap();
        assert_eq!(restored, splade);
    }

    #[test]
    fn test_index_size() {
        assert_eq!(IndexSize::U16.bytes(), 2);
        assert_eq!(IndexSize::U32.bytes(), 4);
        assert_eq!(IndexSize::U16.max_value(), 65535);
        assert_eq!(IndexSize::U32.max_value(), u32::MAX);
    }

    #[test]
    fn test_block_max_weight() {
        // Create postings with known max weights per block
        // Block 0: docs 0-127, weights 0.0-12.7, max = 12.7
        // Block 1: docs 128-255, weights 12.8-25.5, max = 25.5
        // Block 2: docs 256-299, weights 25.6-29.9, max = 29.9
        let postings: Vec<(DocId, f32)> =
            (0..300).map(|i| (i as DocId, (i as f32) * 0.1)).collect();

        let pl =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        // Verify global max weight
        assert!((pl.global_max_weight() - 29.9).abs() < 0.01);

        // Verify block max weights
        assert!(pl.num_blocks() >= 3);

        // Block 0: max should be around 12.7 (index 127 * 0.1)
        let block0_max = pl.block_max_weight(0).unwrap();
        assert!((block0_max - 12.7).abs() < 0.01);

        // Block 1: max should be around 25.5 (index 255 * 0.1)
        let block1_max = pl.block_max_weight(1).unwrap();
        assert!((block1_max - 25.5).abs() < 0.01);

        // Block 2: max should be around 29.9 (index 299 * 0.1)
        let block2_max = pl.block_max_weight(2).unwrap();
        assert!((block2_max - 29.9).abs() < 0.01);

        // Test max_contribution
        let query_weight = 2.0;
        assert!((pl.max_contribution(query_weight) - 59.8).abs() < 0.1);

        // Test iterator block_max methods
        let mut iter = pl.iterator();
        assert!((iter.current_block_max_weight() - 12.7).abs() < 0.01);
        assert!((iter.current_block_max_contribution(query_weight) - 25.4).abs() < 0.1);

        // Seek to block 1
        iter.seek(128);
        assert!((iter.current_block_max_weight() - 25.5).abs() < 0.01);
    }

    #[test]
    fn test_sparse_skip_list_serialization() {
        let mut skip_list = SparseSkipList::new();
        skip_list.push(0, 127, 0, 12.7);
        skip_list.push(128, 255, 100, 25.5);
        skip_list.push(256, 299, 200, 29.9);

        assert_eq!(skip_list.len(), 3);
        assert!((skip_list.global_max_weight() - 29.9).abs() < 0.01);

        // Serialize
        let mut buffer = Vec::new();
        skip_list.write(&mut buffer).unwrap();

        // Deserialize
        let restored = SparseSkipList::read(&mut buffer.as_slice()).unwrap();

        assert_eq!(restored.len(), 3);
        assert!((restored.global_max_weight() - 29.9).abs() < 0.01);

        // Verify entries
        let e0 = restored.get(0).unwrap();
        assert_eq!(e0.first_doc, 0);
        assert_eq!(e0.last_doc, 127);
        assert!((e0.max_weight - 12.7).abs() < 0.01);

        let e1 = restored.get(1).unwrap();
        assert_eq!(e1.first_doc, 128);
        assert!((e1.max_weight - 25.5).abs() < 0.01);
    }
}
