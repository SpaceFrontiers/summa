//! Memory-efficient SSTable index structures
//!
//! This module provides two approaches for memory-efficient block indexing:
//!
//! ## Option 1: FST-based Index (native feature)
//! Uses a Finite State Transducer to map keys to block ordinals. The FST can be
//! mmap'd directly without parsing into heap-allocated structures.
//!
//! ## Option 2: Mmap'd Raw Index
//! Keeps the prefix-compressed block index as raw bytes and decodes entries
//! on-demand during binary search. No heap allocation for the index.
//!
//! Both approaches use a compact BlockAddrStore with bitpacked offsets/lengths.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Write};
use std::ops::Range;

use crate::directories::OwnedBytes;

/// Block address - offset and length in the data section
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockAddr {
    pub offset: u64,
    pub length: u32,
}

impl BlockAddr {
    pub fn byte_range(&self) -> Range<u64> {
        self.offset..self.offset + self.length as u64
    }
}

/// Compact storage for block addresses using delta + bitpacking
///
/// Memory layout:
/// - Header: num_blocks (u32) + offset_bits (u8) + length_bits (u8)
/// - Bitpacked data: offsets and lengths interleaved
///
/// Uses delta encoding for offsets (blocks are sequential) and
/// stores lengths directly (typically similar sizes).
#[derive(Debug)]
pub struct BlockAddrStore {
    num_blocks: u32,
    offset_bits: u8,
    length_bits: u8,
    /// Eagerly decoded addresses for O(1) random access
    addrs: Vec<BlockAddr>,
}

impl BlockAddrStore {
    /// Build from a list of block addresses
    pub fn build(addrs: &[BlockAddr]) -> io::Result<Vec<u8>> {
        if addrs.is_empty() {
            let mut buf = Vec::with_capacity(6);
            buf.write_u32::<LittleEndian>(0)?;
            buf.write_u8(0)?;
            buf.write_u8(0)?;
            return Ok(buf);
        }

        // Compute delta offsets and find max values for bit width
        let mut deltas = Vec::with_capacity(addrs.len());
        let mut prev_end: u64 = 0;
        let mut max_delta: u64 = 0;
        let mut max_length: u32 = 0;

        for addr in addrs {
            // Delta from end of previous block (handles gaps)
            let delta = addr.offset.saturating_sub(prev_end);
            deltas.push(delta);
            max_delta = max_delta.max(delta);
            max_length = max_length.max(addr.length);
            prev_end = addr.offset + addr.length as u64;
        }

        // Compute bit widths
        let offset_bits = if max_delta == 0 {
            1
        } else {
            (64 - max_delta.leading_zeros()) as u8
        };
        let length_bits = if max_length == 0 {
            1
        } else {
            (32 - max_length.leading_zeros()) as u8
        };

        // Calculate packed size
        let bits_per_entry = offset_bits as usize + length_bits as usize;
        let total_bits = bits_per_entry * addrs.len();
        let packed_bytes = total_bits.div_ceil(8);

        let mut buf = Vec::with_capacity(6 + packed_bytes);
        buf.write_u32::<LittleEndian>(addrs.len() as u32)?;
        buf.write_u8(offset_bits)?;
        buf.write_u8(length_bits)?;

        // Bitpack the data
        let mut bit_writer = BitWriter::new(&mut buf);
        for (i, addr) in addrs.iter().enumerate() {
            bit_writer.write(deltas[i], offset_bits)?;
            bit_writer.write(addr.length as u64, length_bits)?;
        }
        bit_writer.flush()?;

        Ok(buf)
    }

    /// Load from raw bytes — eagerly decodes all addresses for O(1) access
    pub fn load(data: OwnedBytes) -> io::Result<Self> {
        if data.len() < 6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BlockAddrStore data too short",
            ));
        }

        let mut reader = data.as_slice();
        let num_blocks = reader.read_u32::<LittleEndian>()?;
        let offset_bits = reader.read_u8()?;
        let length_bits = reader.read_u8()?;

        // Eagerly decode all block addresses once at load time
        let packed_data = &data.as_slice()[6..];
        let mut bit_reader = BitReader::new(packed_data);
        let mut addrs = Vec::with_capacity(num_blocks as usize);
        let mut current_offset: u64 = 0;

        for _ in 0..num_blocks {
            if let (Ok(delta), Ok(length)) =
                (bit_reader.read(offset_bits), bit_reader.read(length_bits))
            {
                current_offset += delta;
                addrs.push(BlockAddr {
                    offset: current_offset,
                    length: length as u32,
                });
                current_offset += length;
            }
        }

        Ok(Self {
            num_blocks,
            offset_bits,
            length_bits,
            addrs,
        })
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        self.num_blocks as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_blocks == 0
    }

    /// Get block address by index — O(1) from eagerly decoded array
    #[inline]
    pub fn get(&self, idx: usize) -> Option<BlockAddr> {
        self.addrs.get(idx).copied()
    }

    /// Get all block addresses
    pub fn all(&self) -> Vec<BlockAddr> {
        self.addrs.clone()
    }
}

/// FST-based block index (Option 1)
///
/// Maps keys to block ordinals using an FST. The FST bytes can be mmap'd
/// directly without any parsing or heap allocation.
#[cfg(feature = "native")]
pub struct FstBlockIndex {
    fst: fst::Map<OwnedBytes>,
    block_addrs: BlockAddrStore,
}

#[cfg(feature = "native")]
impl FstBlockIndex {
    /// Build FST index from keys and block addresses
    pub fn build(entries: &[(Vec<u8>, BlockAddr)]) -> io::Result<Vec<u8>> {
        use fst::MapBuilder;

        // Build FST mapping keys to block ordinals
        let mut fst_builder = MapBuilder::memory();
        for (i, (key, _)) in entries.iter().enumerate() {
            fst_builder
                .insert(key, i as u64)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }
        let fst_bytes = fst_builder
            .into_inner()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Build block address store
        let addrs: Vec<BlockAddr> = entries.iter().map(|(_, addr)| *addr).collect();
        let addr_bytes = BlockAddrStore::build(&addrs)?;

        // Combine: fst_len (u32) + fst_bytes + addr_bytes
        let mut result = Vec::with_capacity(4 + fst_bytes.len() + addr_bytes.len());
        result.write_u32::<LittleEndian>(fst_bytes.len() as u32)?;
        result.extend_from_slice(&fst_bytes);
        result.extend_from_slice(&addr_bytes);

        Ok(result)
    }

    /// Load from raw bytes
    pub fn load(data: OwnedBytes) -> io::Result<Self> {
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FstBlockIndex data too short",
            ));
        }

        let fst_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + fst_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "FstBlockIndex FST data truncated",
            ));
        }

        let fst_data = data.slice(4..4 + fst_len);
        let addr_data = data.slice(4 + fst_len..data.len());

        let fst =
            fst::Map::new(fst_data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let block_addrs = BlockAddrStore::load(addr_data)?;

        Ok(Self { fst, block_addrs })
    }

    /// Look up the block index for a key
    /// Returns the block ordinal that could contain this key.
    /// O(key_len) via FST exact lookup + single stream step.
    pub fn locate(&self, key: &[u8]) -> Option<usize> {
        // Fast exact match — O(key_len), no stream allocation
        if let Some(ordinal) = self.fst.get(key) {
            return Some(ordinal as usize);
        }

        // Find the first block whose first_key > target (single stream step)
        use fst::{IntoStreamer, Streamer};
        let mut stream = self.fst.range().gt(key).into_stream();
        match stream.next() {
            Some((_, ordinal)) if ordinal > 0 => Some(ordinal as usize - 1),
            Some(_) => None, // key < first block's first key
            None => {
                // No key > target → target is after all keys; use last block
                let len = self.fst.len();
                if len > 0 { Some(len - 1) } else { None }
            }
        }
    }

    /// Get block address by ordinal
    pub fn get_addr(&self, ordinal: usize) -> Option<BlockAddr> {
        self.block_addrs.get(ordinal)
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        self.block_addrs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.block_addrs.is_empty()
    }

    /// Get all block addresses
    pub fn all_addrs(&self) -> Vec<BlockAddr> {
        self.block_addrs.all()
    }
}

/// Mmap'd raw block index (Option 2)
///
/// Keeps the prefix-compressed block index as raw bytes and decodes
/// entries on-demand. Uses restart points every R entries for O(log N)
/// lookup via binary search instead of O(N) linear scan.
pub struct MmapBlockIndex {
    data: OwnedBytes,
    num_blocks: u32,
    block_addrs: BlockAddrStore,
    /// Offset where the prefix-compressed keys start
    keys_offset: usize,
    /// Offset where the keys section ends (restart array begins)
    keys_end: usize,
    /// Byte offset in data where the restart offsets array starts
    restart_array_offset: usize,
    /// Number of restart points
    restart_count: usize,
    /// Restart interval (R) — a restart point every R entries
    restart_interval: usize,
}

/// Restart interval: store full (uncompressed) key every R entries
const RESTART_INTERVAL: usize = 16;

impl MmapBlockIndex {
    /// Build mmap-friendly index from entries.
    ///
    /// Format: `num_blocks (u32) | BlockAddrStore | prefix-compressed keys
    /// (with restart points) | restart_offsets[..] | restart_count (u32) | restart_interval (u16)`
    pub fn build(entries: &[(Vec<u8>, BlockAddr)]) -> io::Result<Vec<u8>> {
        if entries.is_empty() {
            let mut buf = Vec::with_capacity(16);
            buf.write_u32::<LittleEndian>(0)?; // num_blocks
            buf.extend_from_slice(&BlockAddrStore::build(&[])?);
            // Empty restart array + footer
            buf.write_u32::<LittleEndian>(0)?; // restart_count
            buf.write_u16::<LittleEndian>(RESTART_INTERVAL as u16)?;
            return Ok(buf);
        }

        // Build block address store
        let addrs: Vec<BlockAddr> = entries.iter().map(|(_, addr)| *addr).collect();
        let addr_bytes = BlockAddrStore::build(&addrs)?;

        // Build prefix-compressed keys with restart points
        let mut keys_buf = Vec::new();
        let mut prev_key: Vec<u8> = Vec::new();
        let mut restart_offsets: Vec<u32> = Vec::new();

        for (i, (key, _)) in entries.iter().enumerate() {
            let is_restart = i % RESTART_INTERVAL == 0;

            if is_restart {
                restart_offsets.push(keys_buf.len() as u32);
                // Store full key (no prefix compression)
                write_vint(&mut keys_buf, 0)?;
                write_vint(&mut keys_buf, key.len() as u64)?;
                keys_buf.extend_from_slice(key);
            } else {
                let prefix_len = common_prefix_len(&prev_key, key);
                let suffix = &key[prefix_len..];
                write_vint(&mut keys_buf, prefix_len as u64)?;
                write_vint(&mut keys_buf, suffix.len() as u64)?;
                keys_buf.extend_from_slice(suffix);
            }

            prev_key.clear();
            prev_key.extend_from_slice(key);
        }

        // Combine: num_blocks + addr_bytes + keys + restart_offsets + footer
        let restart_count = restart_offsets.len();
        let mut result =
            Vec::with_capacity(4 + addr_bytes.len() + keys_buf.len() + restart_count * 4 + 6);
        result.write_u32::<LittleEndian>(entries.len() as u32)?;
        result.extend_from_slice(&addr_bytes);
        result.extend_from_slice(&keys_buf);

        // Write restart offsets array
        for &off in &restart_offsets {
            result.write_u32::<LittleEndian>(off)?;
        }

        // Write footer
        result.write_u32::<LittleEndian>(restart_count as u32)?;
        result.write_u16::<LittleEndian>(RESTART_INTERVAL as u16)?;

        Ok(result)
    }

    /// Load from raw bytes
    pub fn load(data: OwnedBytes) -> io::Result<Self> {
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MmapBlockIndex data too short",
            ));
        }

        let num_blocks = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

        // Load block addresses
        let addr_data_start = 4;
        let remaining = data.slice(addr_data_start..data.len());
        let block_addrs = BlockAddrStore::load(remaining.clone())?;

        // Calculate where keys start
        let bits_per_entry = block_addrs.offset_bits as usize + block_addrs.length_bits as usize;
        let total_bits = bits_per_entry * num_blocks as usize;
        let addr_packed_size = total_bits.div_ceil(8);
        let keys_offset = addr_data_start + 6 + addr_packed_size; // 6 = header of BlockAddrStore

        // Read footer (last 6 bytes: restart_count u32 + restart_interval u16)
        if data.len() < keys_offset + 6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "MmapBlockIndex missing restart footer",
            ));
        }
        let footer_start = data.len() - 6;
        let restart_count = u32::from_le_bytes([
            data[footer_start],
            data[footer_start + 1],
            data[footer_start + 2],
            data[footer_start + 3],
        ]) as usize;
        let restart_interval =
            u16::from_le_bytes([data[footer_start + 4], data[footer_start + 5]]) as usize;

        // Restart offsets array: restart_count × 4 bytes, just before footer
        let restart_array_offset = footer_start - restart_count * 4;

        // Keys section spans from keys_offset to restart_array_offset
        let keys_end = restart_array_offset;

        Ok(Self {
            data,
            num_blocks,
            block_addrs,
            keys_offset,
            keys_end,
            restart_array_offset,
            restart_count,
            restart_interval,
        })
    }

    /// Read restart offset at given index directly from mmap'd data
    #[inline]
    fn restart_offset(&self, idx: usize) -> u32 {
        let pos = self.restart_array_offset + idx * 4;
        u32::from_le_bytes([
            self.data[pos],
            self.data[pos + 1],
            self.data[pos + 2],
            self.data[pos + 3],
        ])
    }

    /// Decode the full key at a restart point (prefix_len is always 0)
    fn decode_restart_key<'a>(&self, keys_data: &'a [u8], restart_idx: usize) -> &'a [u8] {
        let offset = self.restart_offset(restart_idx) as usize;
        let mut reader = &keys_data[offset..];

        let prefix_len = read_vint(&mut reader).unwrap_or(0) as usize;
        debug_assert_eq!(prefix_len, 0, "restart point should have prefix_len=0");
        let suffix_len = read_vint(&mut reader).unwrap_or(0) as usize;

        // reader now points to the suffix bytes
        &reader[..suffix_len]
    }

    /// O(log(N/R) + R) lookup using binary search on restart points, then
    /// linear scan with prefix decompression within the interval.
    pub fn locate(&self, target: &[u8]) -> Option<usize> {
        if self.num_blocks == 0 {
            return None;
        }

        let keys_data = &self.data.as_slice()[self.keys_offset..self.keys_end];

        // Binary search on restart points to find the interval
        let mut lo = 0usize;
        let mut hi = self.restart_count;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let key = self.decode_restart_key(keys_data, mid);
            match key.cmp(target) {
                std::cmp::Ordering::Equal => {
                    return Some(mid * self.restart_interval);
                }
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }

        // lo is the first restart point whose key > target (or restart_count)
        // Search in the interval starting at restart (lo - 1), or 0 if lo == 0
        if lo == 0 {
            // target < first restart key — might be before all keys
            // but we still need to scan from the beginning
        }

        let restart_idx = if lo > 0 { lo - 1 } else { 0 };
        let start_ordinal = restart_idx * self.restart_interval;
        let end_ordinal = if restart_idx + 1 < self.restart_count {
            (restart_idx + 1) * self.restart_interval
        } else {
            self.num_blocks as usize
        };

        // Linear scan from restart point through at most R entries
        let scan_offset = self.restart_offset(restart_idx) as usize;
        let mut reader = &keys_data[scan_offset..];
        let mut current_key = Vec::new();
        let mut last_le_block: Option<usize> = None;

        for i in start_ordinal..end_ordinal {
            let prefix_len = match read_vint(&mut reader) {
                Ok(v) => v as usize,
                Err(_) => break,
            };
            let suffix_len = match read_vint(&mut reader) {
                Ok(v) => v as usize,
                Err(_) => break,
            };

            current_key.truncate(prefix_len);
            if suffix_len > reader.len() {
                break;
            }
            current_key.extend_from_slice(&reader[..suffix_len]);
            reader = &reader[suffix_len..];

            match current_key.as_slice().cmp(target) {
                std::cmp::Ordering::Equal => return Some(i),
                std::cmp::Ordering::Less => last_le_block = Some(i),
                std::cmp::Ordering::Greater => return last_le_block,
            }
        }

        last_le_block
    }

    /// Get block address by ordinal
    pub fn get_addr(&self, ordinal: usize) -> Option<BlockAddr> {
        self.block_addrs.get(ordinal)
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        self.num_blocks as usize
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_blocks == 0
    }

    /// Get all block addresses
    pub fn all_addrs(&self) -> Vec<BlockAddr> {
        self.block_addrs.all()
    }

    /// Decode all keys (for debugging/merging)
    pub fn all_keys(&self) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(self.num_blocks as usize);
        let keys_data = &self.data.as_slice()[self.keys_offset..self.keys_end];
        let mut reader = keys_data;
        let mut current_key = Vec::new();

        for _ in 0..self.num_blocks {
            let prefix_len = match read_vint(&mut reader) {
                Ok(v) => v as usize,
                Err(_) => break,
            };
            let suffix_len = match read_vint(&mut reader) {
                Ok(v) => v as usize,
                Err(_) => break,
            };

            current_key.truncate(prefix_len);
            if suffix_len > reader.len() {
                break;
            }
            current_key.extend_from_slice(&reader[..suffix_len]);
            reader = &reader[suffix_len..];

            result.push(current_key.clone());
        }

        result
    }
}

/// Unified block index that can use either FST or mmap'd raw index
pub enum BlockIndex {
    #[cfg(feature = "native")]
    Fst(FstBlockIndex),
    Mmap(MmapBlockIndex),
}

impl BlockIndex {
    /// Locate the block that could contain the key
    pub fn locate(&self, key: &[u8]) -> Option<usize> {
        match self {
            #[cfg(feature = "native")]
            BlockIndex::Fst(idx) => idx.locate(key),
            BlockIndex::Mmap(idx) => idx.locate(key),
        }
    }

    /// Get block address by ordinal
    pub fn get_addr(&self, ordinal: usize) -> Option<BlockAddr> {
        match self {
            #[cfg(feature = "native")]
            BlockIndex::Fst(idx) => idx.get_addr(ordinal),
            BlockIndex::Mmap(idx) => idx.get_addr(ordinal),
        }
    }

    /// Number of blocks
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "native")]
            BlockIndex::Fst(idx) => idx.len(),
            BlockIndex::Mmap(idx) => idx.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all block addresses
    pub fn all_addrs(&self) -> Vec<BlockAddr> {
        match self {
            #[cfg(feature = "native")]
            BlockIndex::Fst(idx) => idx.all_addrs(),
            BlockIndex::Mmap(idx) => idx.all_addrs(),
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn write_vint<W: Write>(writer: &mut W, mut value: u64) -> io::Result<()> {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            writer.write_all(&[byte])?;
            return Ok(());
        } else {
            writer.write_all(&[byte | 0x80])?;
        }
    }
}

fn read_vint(reader: &mut &[u8]) -> io::Result<u64> {
    let mut result = 0u64;
    let mut shift = 0;

    loop {
        if reader.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected end of varint",
            ));
        }
        let byte = reader[0];
        *reader = &reader[1..];
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Varint too long",
            ));
        }
    }
}

/// Simple bit writer for packing
struct BitWriter<'a> {
    output: &'a mut Vec<u8>,
    buffer: u64,
    bits_in_buffer: u8,
}

impl<'a> BitWriter<'a> {
    fn new(output: &'a mut Vec<u8>) -> Self {
        Self {
            output,
            buffer: 0,
            bits_in_buffer: 0,
        }
    }

    fn write(&mut self, value: u64, num_bits: u8) -> io::Result<()> {
        debug_assert!(num_bits <= 64);

        self.buffer |= value << self.bits_in_buffer;
        self.bits_in_buffer += num_bits;

        while self.bits_in_buffer >= 8 {
            self.output.push(self.buffer as u8);
            self.buffer >>= 8;
            self.bits_in_buffer -= 8;
        }

        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.bits_in_buffer > 0 {
            self.output.push(self.buffer as u8);
            self.buffer = 0;
            self.bits_in_buffer = 0;
        }
        Ok(())
    }
}

/// Simple bit reader for unpacking
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read(&mut self, num_bits: u8) -> io::Result<u64> {
        if num_bits == 0 {
            return Ok(0);
        }

        let mut result: u64 = 0;
        let mut bits_read: u8 = 0;

        while bits_read < num_bits {
            if self.byte_pos >= self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Not enough bits",
                ));
            }

            let bits_available = 8 - self.bit_pos;
            let bits_to_read = (num_bits - bits_read).min(bits_available);
            // Handle edge case where bits_to_read == 8 to avoid overflow
            let mask = if bits_to_read >= 8 {
                0xFF
            } else {
                (1u8 << bits_to_read) - 1
            };
            let bits = (self.data[self.byte_pos] >> self.bit_pos) & mask;

            result |= (bits as u64) << bits_read;
            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_addr_store_roundtrip() {
        let addrs = vec![
            BlockAddr {
                offset: 0,
                length: 1000,
            },
            BlockAddr {
                offset: 1000,
                length: 1500,
            },
            BlockAddr {
                offset: 2500,
                length: 800,
            },
            BlockAddr {
                offset: 3300,
                length: 2000,
            },
        ];

        let bytes = BlockAddrStore::build(&addrs).unwrap();
        let store = BlockAddrStore::load(OwnedBytes::new(bytes)).unwrap();

        assert_eq!(store.len(), 4);
        for (i, expected) in addrs.iter().enumerate() {
            let actual = store.get(i).unwrap();
            assert_eq!(actual.offset, expected.offset, "offset mismatch at {}", i);
            assert_eq!(actual.length, expected.length, "length mismatch at {}", i);
        }
    }

    #[test]
    fn test_block_addr_store_empty() {
        let bytes = BlockAddrStore::build(&[]).unwrap();
        let store = BlockAddrStore::load(OwnedBytes::new(bytes)).unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.get(0).is_none());
    }

    #[test]
    fn test_mmap_block_index_roundtrip() {
        let entries = vec![
            (
                b"aaa".to_vec(),
                BlockAddr {
                    offset: 0,
                    length: 100,
                },
            ),
            (
                b"bbb".to_vec(),
                BlockAddr {
                    offset: 100,
                    length: 150,
                },
            ),
            (
                b"ccc".to_vec(),
                BlockAddr {
                    offset: 250,
                    length: 200,
                },
            ),
        ];

        let bytes = MmapBlockIndex::build(&entries).unwrap();
        let index = MmapBlockIndex::load(OwnedBytes::new(bytes)).unwrap();

        assert_eq!(index.len(), 3);

        // Test locate
        assert_eq!(index.locate(b"aaa"), Some(0));
        assert_eq!(index.locate(b"bbb"), Some(1));
        assert_eq!(index.locate(b"ccc"), Some(2));
        assert_eq!(index.locate(b"aab"), Some(0)); // Between aaa and bbb
        assert_eq!(index.locate(b"ddd"), Some(2)); // After all keys
        assert_eq!(index.locate(b"000"), None); // Before all keys
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_fst_block_index_roundtrip() {
        let entries = vec![
            (
                b"aaa".to_vec(),
                BlockAddr {
                    offset: 0,
                    length: 100,
                },
            ),
            (
                b"bbb".to_vec(),
                BlockAddr {
                    offset: 100,
                    length: 150,
                },
            ),
            (
                b"ccc".to_vec(),
                BlockAddr {
                    offset: 250,
                    length: 200,
                },
            ),
        ];

        let bytes = FstBlockIndex::build(&entries).unwrap();
        let index = FstBlockIndex::load(OwnedBytes::new(bytes)).unwrap();

        assert_eq!(index.len(), 3);

        // Test locate
        assert_eq!(index.locate(b"aaa"), Some(0));
        assert_eq!(index.locate(b"bbb"), Some(1));
        assert_eq!(index.locate(b"ccc"), Some(2));
        assert_eq!(index.locate(b"aab"), Some(0)); // Between aaa and bbb
        assert_eq!(index.locate(b"ddd"), Some(2)); // After all keys
    }

    #[test]
    fn test_bit_writer_reader() {
        let mut buf = Vec::new();
        let mut writer = BitWriter::new(&mut buf);

        writer.write(5, 3).unwrap(); // 101
        writer.write(3, 2).unwrap(); // 11
        writer.write(15, 4).unwrap(); // 1111
        writer.flush().unwrap();

        let mut reader = BitReader::new(&buf);
        assert_eq!(reader.read(3).unwrap(), 5);
        assert_eq!(reader.read(2).unwrap(), 3);
        assert_eq!(reader.read(4).unwrap(), 15);
    }
}
