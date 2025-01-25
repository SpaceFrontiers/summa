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
//!
//! ## Block Format (v2)
//!
//! The block-based format separates data into 3 sub-blocks per 128-entry block:
//! - **Doc IDs**: Delta-encoded, bit-packed (SIMD-friendly)
//! - **Ordinals**: Bit-packed small integers (lazy decode, only for results)
//! - **Weights**: Quantized (f32/f16/u8/u4)

mod block;
mod config;

pub use block::{BlockSparsePostingIterator, BlockSparsePostingList};
pub use config::{
    IndexSize, QueryWeighting, SparseEntry, SparseQueryConfig, SparseVector, SparseVectorConfig,
    WeightQuantization,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use super::posting_common::{read_vint, write_vint};
use crate::DocId;

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
        let postings: Vec<(DocId, u16, f32)> =
            (0..300).map(|i| (i * 2, 0, (i as f32) * 0.1)).collect();

        let pl =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        assert_eq!(pl.doc_count(), 300);
        assert!(pl.num_blocks() >= 2);

        // Test iteration
        let mut iter = pl.iterator();
        for (expected_doc, _, expected_weight) in &postings {
            assert_eq!(iter.doc(), *expected_doc);
            assert!((iter.weight() - expected_weight).abs() < 1e-6);
            iter.advance();
        }
        assert_eq!(iter.doc(), super::super::TERMINATED);
    }

    #[test]
    fn test_block_sparse_seek() {
        let postings: Vec<(DocId, u16, f32)> = (0..500).map(|i| (i * 3, 0, i as f32)).collect();

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
        let postings: Vec<(DocId, u16, f32)> = vec![(0, 0, 1.0), (10, 0, 2.0), (100, 0, 3.0)];

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

            // Verify iteration produces same results
            let mut iter1 = pl.iterator();
            let mut iter2 = pl2.iterator();

            while iter1.doc() != super::super::TERMINATED {
                assert_eq!(iter1.doc(), iter2.doc());
                assert!((iter1.weight() - iter2.weight()).abs() < 0.1);
                iter1.advance();
                iter2.advance();
            }
        }
    }

    #[test]
    fn test_concatenate() {
        let postings1: Vec<(DocId, u16, f32)> = vec![(0, 0, 1.0), (5, 1, 2.0)];
        let postings2: Vec<(DocId, u16, f32)> = vec![(0, 0, 3.0), (10, 1, 4.0)];

        let pl1 =
            BlockSparsePostingList::from_postings(&postings1, WeightQuantization::Float32).unwrap();
        let pl2 =
            BlockSparsePostingList::from_postings(&postings2, WeightQuantization::Float32).unwrap();

        // Merge manually
        let mut all: Vec<(DocId, u16, f32)> = pl1.decode_all();
        for (doc_id, ord, w) in pl2.decode_all() {
            all.push((doc_id + 100, ord, w));
        }
        let merged =
            BlockSparsePostingList::from_postings(&all, WeightQuantization::Float32).unwrap();

        assert_eq!(merged.doc_count(), 4);

        let decoded = merged.decode_all();
        assert_eq!(decoded[0], (0, 0, 1.0));
        assert_eq!(decoded[1], (5, 1, 2.0));
        assert_eq!(decoded[2], (100, 0, 3.0));
        assert_eq!(decoded[3], (110, 1, 4.0));
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
        let postings: Vec<(DocId, u16, f32)> = (0..300)
            .map(|i| (i as DocId, 0, (i as f32) * 0.1))
            .collect();

        let pl =
            BlockSparsePostingList::from_postings(&postings, WeightQuantization::Float32).unwrap();

        assert!((pl.global_max_weight() - 29.9).abs() < 0.01);
        assert!(pl.num_blocks() >= 3);

        let block0_max = pl.block_max_weight(0).unwrap();
        assert!((block0_max - 12.7).abs() < 0.01);

        let block1_max = pl.block_max_weight(1).unwrap();
        assert!((block1_max - 25.5).abs() < 0.01);

        let block2_max = pl.block_max_weight(2).unwrap();
        assert!((block2_max - 29.9).abs() < 0.01);

        // Test iterator block_max methods
        let query_weight = 2.0;
        let mut iter = pl.iterator();
        assert!((iter.current_block_max_weight() - 12.7).abs() < 0.01);
        assert!((iter.current_block_max_contribution(query_weight) - 25.4).abs() < 0.1);

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
