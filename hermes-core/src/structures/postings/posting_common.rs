//! Shared primitives for posting lists
//!
//! This module contains common code used by both text posting lists and sparse vector posting lists:
//! - Variable-length integer encoding (varint)
//! - Skip list structure for block-based access
//! - Block constants

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use crate::DocId;

/// Standard block size for posting lists (SIMD-friendly)
pub const BLOCK_SIZE: usize = 128;

/// Write variable-length integer (1-9 bytes)
///
/// Uses continuation bit encoding: 7 bits of data per byte,
/// high bit indicates more bytes follow.
#[inline]
pub fn write_vint<W: Write>(writer: &mut W, mut value: u64) -> io::Result<()> {
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
#[inline]
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

/// Skip list entry for block-based posting lists
///
/// Enables O(log n) seeking by storing metadata for each block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipEntry {
    /// First doc_id in the block (absolute)
    pub first_doc: DocId,
    /// Last doc_id in the block
    pub last_doc: DocId,
    /// Byte offset to block data
    pub offset: u32,
}

impl SkipEntry {
    pub fn new(first_doc: DocId, last_doc: DocId, offset: u32) -> Self {
        Self {
            first_doc,
            last_doc,
            offset,
        }
    }

    /// Write skip entry to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.first_doc)?;
        writer.write_u32::<LittleEndian>(self.last_doc)?;
        writer.write_u32::<LittleEndian>(self.offset)?;
        Ok(())
    }

    /// Read skip entry from reader
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let first_doc = reader.read_u32::<LittleEndian>()?;
        let last_doc = reader.read_u32::<LittleEndian>()?;
        let offset = reader.read_u32::<LittleEndian>()?;
        Ok(Self {
            first_doc,
            last_doc,
            offset,
        })
    }
}

/// Skip list for block-based posting lists
#[derive(Debug, Clone, Default)]
pub struct SkipList {
    entries: Vec<SkipEntry>,
}

impl SkipList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Add a skip entry
    pub fn push(&mut self, first_doc: DocId, last_doc: DocId, offset: u32) {
        self.entries
            .push(SkipEntry::new(first_doc, last_doc, offset));
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get entry by index
    pub fn get(&self, index: usize) -> Option<&SkipEntry> {
        self.entries.get(index)
    }

    /// Find block index containing doc_id >= target.
    /// Uses binary search on monotonically increasing `last_doc` values.
    ///
    /// Returns None if target is beyond all blocks.
    pub fn find_block(&self, target: DocId) -> Option<usize> {
        let idx = self.entries.partition_point(|e| e.last_doc < target);
        if idx < self.entries.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// Iterate over entries
    pub fn iter(&self) -> impl Iterator<Item = &SkipEntry> {
        self.entries.iter()
    }

    /// Write skip list to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.entries.len() as u32)?;
        for entry in &self.entries {
            entry.write(writer)?;
        }
        Ok(())
    }

    /// Read skip list from reader
    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let count = reader.read_u32::<LittleEndian>()? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(SkipEntry::read(reader)?);
        }
        Ok(Self { entries })
    }

    /// Convert from tuple format (for compatibility)
    pub fn from_tuples(tuples: &[(DocId, DocId, u32)]) -> Self {
        Self {
            entries: tuples
                .iter()
                .map(|(first, last, offset)| SkipEntry::new(*first, *last, *offset))
                .collect(),
        }
    }

    /// Convert to tuple format (for compatibility)
    pub fn to_tuples(&self) -> Vec<(DocId, DocId, u32)> {
        self.entries
            .iter()
            .map(|e| (e.first_doc, e.last_doc, e.offset))
            .collect()
    }
}

/// Write a block of delta-encoded doc_ids
///
/// First doc_id is written as absolute value, rest as deltas.
/// Returns the last doc_id written.
pub fn write_doc_id_block<W: Write>(writer: &mut W, doc_ids: &[DocId]) -> io::Result<DocId> {
    if doc_ids.is_empty() {
        return Ok(0);
    }

    write_vint(writer, doc_ids.len() as u64)?;

    let mut prev = 0u32;
    for (i, &doc_id) in doc_ids.iter().enumerate() {
        if i == 0 {
            // First doc_id: absolute
            write_vint(writer, doc_id as u64)?;
        } else {
            // Rest: delta from previous
            write_vint(writer, (doc_id - prev) as u64)?;
        }
        prev = doc_id;
    }

    Ok(*doc_ids.last().unwrap())
}

/// Read a block of delta-encoded doc_ids
///
/// Returns vector of absolute doc_ids.
pub fn read_doc_id_block<R: Read>(reader: &mut R) -> io::Result<Vec<DocId>> {
    let count = read_vint(reader)? as usize;
    let mut doc_ids = Vec::with_capacity(count);

    let mut prev = 0u32;
    for i in 0..count {
        let value = read_vint(reader)? as u32;
        let doc_id = if i == 0 {
            value // First: absolute
        } else {
            prev + value // Rest: delta
        };
        doc_ids.push(doc_id);
        prev = doc_id;
    }

    Ok(doc_ids)
}

// ============================================================================
// Fixed-width bitpacking for SIMD-friendly delta encoding
// ============================================================================

use crate::structures::simd;

/// Rounded bit width for SIMD-friendly encoding
///
/// Values are rounded up to 0, 8, 16, or 32 bits for efficient SIMD unpacking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RoundedBitWidth {
    /// All values are zero (e.g., consecutive doc IDs)
    Zero = 0,
    /// 8-bit values (0-255)
    Bits8 = 8,
    /// 16-bit values (0-65535)
    Bits16 = 16,
    /// 32-bit values
    Bits32 = 32,
}

impl RoundedBitWidth {
    /// Determine the rounded bit width needed for a maximum value
    pub fn from_max_value(max_val: u32) -> Self {
        if max_val == 0 {
            RoundedBitWidth::Zero
        } else if max_val <= 255 {
            RoundedBitWidth::Bits8
        } else if max_val <= 65535 {
            RoundedBitWidth::Bits16
        } else {
            RoundedBitWidth::Bits32
        }
    }

    /// Bytes per value
    pub fn bytes_per_value(&self) -> usize {
        match self {
            RoundedBitWidth::Zero => 0,
            RoundedBitWidth::Bits8 => 1,
            RoundedBitWidth::Bits16 => 2,
            RoundedBitWidth::Bits32 => 4,
        }
    }

    /// Convert from u8
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(RoundedBitWidth::Zero),
            8 => Some(RoundedBitWidth::Bits8),
            16 => Some(RoundedBitWidth::Bits16),
            32 => Some(RoundedBitWidth::Bits32),
            _ => None,
        }
    }
}

/// Pack delta-encoded doc IDs with fixed-width encoding
///
/// Stores (gap - 1) for each delta to save one bit since gaps are always >= 1.
/// Returns (bit_width, packed_bytes).
pub fn pack_deltas_fixed(doc_ids: &[DocId]) -> (RoundedBitWidth, Vec<u8>) {
    if doc_ids.len() <= 1 {
        return (RoundedBitWidth::Zero, Vec::new());
    }

    // Compute deltas and find max
    let mut max_delta = 0u32;
    let mut deltas = Vec::with_capacity(doc_ids.len() - 1);

    for i in 1..doc_ids.len() {
        let delta = doc_ids[i] - doc_ids[i - 1] - 1; // Store gap-1
        deltas.push(delta);
        max_delta = max_delta.max(delta);
    }

    let bit_width = RoundedBitWidth::from_max_value(max_delta);
    let bytes_per_val = bit_width.bytes_per_value();

    if bytes_per_val == 0 {
        return (bit_width, Vec::new());
    }

    let mut packed = Vec::with_capacity(deltas.len() * bytes_per_val);

    match bit_width {
        RoundedBitWidth::Zero => {}
        RoundedBitWidth::Bits8 => {
            for delta in deltas {
                packed.push(delta as u8);
            }
        }
        RoundedBitWidth::Bits16 => {
            for delta in deltas {
                packed.extend_from_slice(&(delta as u16).to_le_bytes());
            }
        }
        RoundedBitWidth::Bits32 => {
            for delta in deltas {
                packed.extend_from_slice(&delta.to_le_bytes());
            }
        }
    }

    (bit_width, packed)
}

/// Unpack delta-encoded doc IDs with SIMD acceleration
///
/// Uses SIMD for 8/16/32-bit widths, scalar for zero width.
pub fn unpack_deltas_fixed(
    packed: &[u8],
    bit_width: RoundedBitWidth,
    first_doc_id: DocId,
    count: usize,
    output: &mut [DocId],
) {
    if count == 0 {
        return;
    }

    output[0] = first_doc_id;

    if count == 1 {
        return;
    }

    match bit_width {
        RoundedBitWidth::Zero => {
            // All gaps are 1 (consecutive doc IDs)
            for (i, out) in output.iter_mut().enumerate().skip(1).take(count - 1) {
                *out = first_doc_id + i as u32;
            }
        }
        RoundedBitWidth::Bits8 => {
            simd::unpack_8bit_delta_decode(packed, output, first_doc_id, count);
        }
        RoundedBitWidth::Bits16 => {
            simd::unpack_16bit_delta_decode(packed, output, first_doc_id, count);
        }
        RoundedBitWidth::Bits32 => {
            // Unpack and delta decode
            let mut carry = first_doc_id;
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vint_roundtrip() {
        let values = [
            0u64,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
        ];

        for &value in &values {
            let mut buf = Vec::new();
            write_vint(&mut buf, value).unwrap();
            let read_value = read_vint(&mut buf.as_slice()).unwrap();
            assert_eq!(value, read_value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_skip_list_roundtrip() {
        let mut skip_list = SkipList::new();
        skip_list.push(0, 127, 0);
        skip_list.push(128, 255, 100);
        skip_list.push(256, 500, 200);

        let mut buf = Vec::new();
        skip_list.write(&mut buf).unwrap();

        let restored = SkipList::read(&mut buf.as_slice()).unwrap();
        assert_eq!(skip_list.len(), restored.len());

        for (a, b) in skip_list.iter().zip(restored.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_skip_list_find_block() {
        let mut skip_list = SkipList::new();
        skip_list.push(0, 99, 0);
        skip_list.push(100, 199, 100);
        skip_list.push(200, 299, 200);

        assert_eq!(skip_list.find_block(0), Some(0));
        assert_eq!(skip_list.find_block(50), Some(0));
        assert_eq!(skip_list.find_block(99), Some(0));
        assert_eq!(skip_list.find_block(100), Some(1));
        assert_eq!(skip_list.find_block(150), Some(1));
        assert_eq!(skip_list.find_block(250), Some(2));
        assert_eq!(skip_list.find_block(300), None);
    }

    #[test]
    fn test_doc_id_block_roundtrip() {
        let doc_ids: Vec<DocId> = vec![0, 5, 10, 100, 1000, 10000];

        let mut buf = Vec::new();
        let last = write_doc_id_block(&mut buf, &doc_ids).unwrap();
        assert_eq!(last, 10000);

        let restored = read_doc_id_block(&mut buf.as_slice()).unwrap();
        assert_eq!(doc_ids, restored);
    }

    #[test]
    fn test_doc_id_block_single() {
        let doc_ids: Vec<DocId> = vec![42];

        let mut buf = Vec::new();
        write_doc_id_block(&mut buf, &doc_ids).unwrap();

        let restored = read_doc_id_block(&mut buf.as_slice()).unwrap();
        assert_eq!(doc_ids, restored);
    }
}
