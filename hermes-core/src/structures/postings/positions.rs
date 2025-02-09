//! Position-aware posting list for phrase queries and multi-field element tracking
//!
//! Positions are encoded as: (element_ordinal << 20) | token_position
//! This allows up to 4096 elements per field and ~1M tokens per element.
//!
//! ## Block Format
//!
//! Uses a block-based format with skip list for efficient binary search by doc_id:
//! - Skip list enables O(log n) lookup by doc_id
//! - Blocks of up to 128 documents for cache efficiency
//! - Delta encoding within blocks for compression
//! - Stackable for fast merge (just adjust doc_id_base per block)
//!
//! Format:
//! ```text
//! Header:
//!   - doc_count: u32
//!   - num_blocks: u32
//!   - skip_list: [(base_doc_id, last_doc_id, byte_offset)] per block
//!   - data_len: u32
//! Data:
//!   - blocks: each block contains delta-encoded postings with positions
//! ```

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use crate::DocId;

/// Block size for position posting list (same as regular posting list)
pub const POSITION_BLOCK_SIZE: usize = 128;

/// Maximum token position within an element (20 bits = 1,048,575)
pub const MAX_TOKEN_POSITION: u32 = (1 << 20) - 1;

/// Maximum element ordinal (12 bits = 4095)
pub const MAX_ELEMENT_ORDINAL: u32 = (1 << 12) - 1;

/// Encode element ordinal and token position into a single u32
#[inline]
pub fn encode_position(element_ordinal: u32, token_position: u32) -> u32 {
    debug_assert!(
        element_ordinal <= MAX_ELEMENT_ORDINAL,
        "Element ordinal {} exceeds maximum {}",
        element_ordinal,
        MAX_ELEMENT_ORDINAL
    );
    debug_assert!(
        token_position <= MAX_TOKEN_POSITION,
        "Token position {} exceeds maximum {}",
        token_position,
        MAX_TOKEN_POSITION
    );
    (element_ordinal << 20) | (token_position & MAX_TOKEN_POSITION)
}

/// Decode element ordinal from encoded position
#[inline]
pub fn decode_element_ordinal(position: u32) -> u32 {
    position >> 20
}

/// Decode token position from encoded position
#[inline]
pub fn decode_token_position(position: u32) -> u32 {
    position & MAX_TOKEN_POSITION
}

/// A posting entry with positions (used during building)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostingWithPositions {
    pub doc_id: DocId,
    pub term_freq: u32,
    /// Encoded positions: (element_ordinal << 20) | token_position
    pub positions: Vec<u32>,
}

/// Block-based position posting list with skip list for O(log n) doc_id lookup
///
/// Similar to BlockPostingList but stores positions per document.
/// Uses binary search on skip list to find the right block, then linear scan within block.
#[derive(Debug, Clone)]
pub struct PositionPostingList {
    /// Skip list: (base_doc_id, last_doc_id, byte_offset)
    /// Enables binary search to find the right block
    skip_list: Vec<(DocId, DocId, u32)>,
    /// Compressed block data
    data: Vec<u8>,
    /// Total document count
    doc_count: u32,
}

impl Default for PositionPostingList {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionPostingList {
    pub fn new() -> Self {
        Self {
            skip_list: Vec::new(),
            data: Vec::new(),
            doc_count: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            skip_list: Vec::with_capacity(capacity / POSITION_BLOCK_SIZE + 1),
            data: Vec::with_capacity(capacity * 8), // rough estimate
            doc_count: 0,
        }
    }

    /// Build from a list of postings with positions
    pub fn from_postings(postings: &[PostingWithPositions]) -> io::Result<Self> {
        if postings.is_empty() {
            return Ok(Self::new());
        }

        let mut skip_list = Vec::new();
        let mut data = Vec::new();
        let mut i = 0;

        while i < postings.len() {
            let block_start = data.len() as u32;
            let block_end = (i + POSITION_BLOCK_SIZE).min(postings.len());
            let block = &postings[i..block_end];

            // Record skip entry
            let base_doc_id = block.first().unwrap().doc_id;
            let last_doc_id = block.last().unwrap().doc_id;
            skip_list.push((base_doc_id, last_doc_id, block_start));

            // Write block: fixed u32 count + first_doc (8-byte prefix), then vint deltas
            data.write_u32::<LittleEndian>(block.len() as u32)?;
            data.write_u32::<LittleEndian>(base_doc_id)?;

            let mut prev_doc_id = base_doc_id;
            for (j, posting) in block.iter().enumerate() {
                if j > 0 {
                    let delta = posting.doc_id - prev_doc_id;
                    write_vint(&mut data, delta as u64)?;
                }
                prev_doc_id = posting.doc_id;

                // Write positions count and positions (absolute - delta bad for ordinal<<20)
                write_vint(&mut data, posting.positions.len() as u64)?;
                for &pos in &posting.positions {
                    write_vint(&mut data, pos as u64)?;
                }
            }

            i = block_end;
        }

        Ok(Self {
            skip_list,
            data,
            doc_count: postings.len() as u32,
        })
    }

    /// Add a posting with positions (for building - converts to block format on serialize)
    pub fn push(&mut self, doc_id: DocId, positions: Vec<u32>) {
        // For compatibility: build in-memory, convert to blocks on serialize
        // This is a simplified approach - we rebuild skip_list on serialize
        let posting = PostingWithPositions {
            doc_id,
            term_freq: positions.len() as u32,
            positions,
        };

        // Serialize this posting to data buffer
        let block_start = self.data.len() as u32;

        // If this is first posting or we need a new block
        let need_new_block =
            self.skip_list.is_empty() || self.doc_count.is_multiple_of(POSITION_BLOCK_SIZE as u32);

        if need_new_block {
            // Start new block: fixed u32 count + first_doc (8-byte prefix)
            self.skip_list.push((doc_id, doc_id, block_start));
            self.data.write_u32::<LittleEndian>(1u32).unwrap();
            self.data.write_u32::<LittleEndian>(doc_id).unwrap();
        } else {
            // Add to existing block — update count in-place + add delta
            let last_block = self.skip_list.last_mut().unwrap();
            let prev_doc_id = last_block.1;
            last_block.1 = doc_id;

            // Patch count u32 at block start
            let count_offset = last_block.2 as usize;
            let old_count = u32::from_le_bytes(
                self.data[count_offset..count_offset + 4]
                    .try_into()
                    .unwrap(),
            );
            self.data[count_offset..count_offset + 4]
                .copy_from_slice(&(old_count + 1).to_le_bytes());

            let delta = doc_id - prev_doc_id;
            write_vint(&mut self.data, delta as u64).unwrap();
        }

        // Write positions (absolute - delta encoding bad for ordinal<<20)
        write_vint(&mut self.data, posting.positions.len() as u64).unwrap();
        for &pos in &posting.positions {
            write_vint(&mut self.data, pos as u64).unwrap();
        }

        self.doc_count += 1;
    }

    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    pub fn len(&self) -> usize {
        self.doc_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Get positions for a specific document using binary search on skip list
    pub fn get_positions(&self, target_doc_id: DocId) -> Option<Vec<u32>> {
        if self.skip_list.is_empty() {
            return None;
        }

        // Binary search on skip list to find the right block
        let block_idx = match self.skip_list.binary_search_by(|&(base, last, _)| {
            if target_doc_id < base {
                std::cmp::Ordering::Greater
            } else if target_doc_id > last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(idx) => idx,
            Err(_) => return None, // doc_id not in any block range
        };

        // Decode block and search for doc_id
        let offset = self.skip_list[block_idx].2 as usize;
        let mut reader = &self.data[offset..];

        // Fixed 8-byte prefix: count(u32) + first_doc(u32)
        let count = reader.read_u32::<LittleEndian>().ok()? as usize;
        let first_doc = reader.read_u32::<LittleEndian>().ok()?;
        let mut prev_doc_id = first_doc;

        for i in 0..count {
            let doc_id = if i == 0 {
                first_doc
            } else {
                let delta = read_vint(&mut reader).ok()? as u32;
                prev_doc_id + delta
            };
            prev_doc_id = doc_id;

            let num_positions = read_vint(&mut reader).ok()? as usize;

            if doc_id == target_doc_id {
                // Found it! Read positions (stored absolute)
                let mut positions = Vec::with_capacity(num_positions);
                for _ in 0..num_positions {
                    let pos = read_vint(&mut reader).ok()? as u32;
                    positions.push(pos);
                }
                return Some(positions);
            } else {
                // Skip positions
                for _ in 0..num_positions {
                    let _ = read_vint(&mut reader);
                }
            }
        }

        None
    }

    /// Serialize to bytes with block format
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write doc count
        writer.write_u32::<LittleEndian>(self.doc_count)?;

        // Write skip list
        writer.write_u32::<LittleEndian>(self.skip_list.len() as u32)?;
        for (base_doc_id, last_doc_id, offset) in &self.skip_list {
            writer.write_u32::<LittleEndian>(*base_doc_id)?;
            writer.write_u32::<LittleEndian>(*last_doc_id)?;
            writer.write_u32::<LittleEndian>(*offset)?;
        }

        // Write data
        writer.write_u32::<LittleEndian>(self.data.len() as u32)?;
        writer.write_all(&self.data)?;

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_count = reader.read_u32::<LittleEndian>()?;

        let skip_count = reader.read_u32::<LittleEndian>()? as usize;
        let mut skip_list = Vec::with_capacity(skip_count);
        for _ in 0..skip_count {
            let base_doc_id = reader.read_u32::<LittleEndian>()?;
            let last_doc_id = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u32::<LittleEndian>()?;
            skip_list.push((base_doc_id, last_doc_id, offset));
        }

        let data_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        Ok(Self {
            skip_list,
            data,
            doc_count,
        })
    }

    /// Concatenate blocks from multiple position lists with doc_id remapping (for merge)
    pub fn concatenate_blocks(sources: &[(PositionPostingList, u32)]) -> io::Result<Self> {
        let mut skip_list = Vec::new();
        let mut data = Vec::new();
        let mut total_docs = 0u32;

        for (source, doc_offset) in sources {
            for block_idx in 0..source.skip_list.len() {
                let (base, last, src_offset) = source.skip_list[block_idx];
                let next_offset = if block_idx + 1 < source.skip_list.len() {
                    source.skip_list[block_idx + 1].2 as usize
                } else {
                    source.data.len()
                };

                let new_base = base + doc_offset;
                let new_last = last + doc_offset;
                let new_offset = data.len() as u32;

                // Copy and adjust block data
                let block_bytes = &source.data[src_offset as usize..next_offset];

                // Fixed 8-byte prefix: count(u32) + first_doc(u32)
                let count = u32::from_le_bytes(block_bytes[0..4].try_into().unwrap());
                let first_doc = u32::from_le_bytes(block_bytes[4..8].try_into().unwrap());

                // Write patched prefix + copy rest verbatim
                data.write_u32::<LittleEndian>(count)?;
                data.write_u32::<LittleEndian>(first_doc + doc_offset)?;
                data.extend_from_slice(&block_bytes[8..]);

                skip_list.push((new_base, new_last, new_offset));
                total_docs += count;
            }
        }

        Ok(Self {
            skip_list,
            data,
            doc_count: total_docs,
        })
    }

    /// Concatenate position lists directly from serialized bytes (zero intermediate allocation).
    ///
    /// Same approach as `BlockPostingList::concatenate_from_raw`: parses only
    /// header + skip_list, then writes merged output with patched 8-byte block prefixes.
    /// Returns doc_count.
    pub fn concatenate_from_raw(
        sources: &[(&[u8], u32)], // (serialized_bytes, doc_offset)
        out: &mut Vec<u8>,
    ) -> io::Result<u32> {
        struct RawSource<'a> {
            skip_list: Vec<(u32, u32, u32)>, // (base, last, offset)
            data: &'a [u8],
            doc_count: u32,
            doc_offset: u32,
        }

        let mut parsed: Vec<RawSource<'_>> = Vec::with_capacity(sources.len());
        for (raw, doc_offset) in sources {
            let mut r = &raw[..];
            let doc_count = r.read_u32::<LittleEndian>()?;
            let skip_count = r.read_u32::<LittleEndian>()? as usize;
            let mut skip_list = Vec::with_capacity(skip_count);
            for _ in 0..skip_count {
                let base = r.read_u32::<LittleEndian>()?;
                let last = r.read_u32::<LittleEndian>()?;
                let offset = r.read_u32::<LittleEndian>()?;
                skip_list.push((base, last, offset));
            }
            let data_len = r.read_u32::<LittleEndian>()? as usize;
            let header_size = raw.len() - r.len();
            let data = &raw[header_size..header_size + data_len];
            parsed.push(RawSource {
                skip_list,
                data,
                doc_count,
                doc_offset: *doc_offset,
            });
        }

        let total_docs: u32 = parsed.iter().map(|s| s.doc_count).sum();
        let total_blocks: usize = parsed.iter().map(|s| s.skip_list.len()).sum();
        let data_size: usize = parsed.iter().map(|s| s.data.len()).sum();

        out.reserve(8 + total_blocks * 12 + 4 + data_size);

        // Header: doc_count + num_blocks
        out.write_u32::<LittleEndian>(total_docs)?;
        out.write_u32::<LittleEndian>(total_blocks as u32)?;

        // Skip list with remapped offsets
        let mut data_base_offset = 0u32;
        let skip_list_start = out.len();
        out.resize(skip_list_start + total_blocks * 12, 0);

        let mut skip_idx = 0;
        for src in &parsed {
            let src_data_len = src.data.len() as u32;
            for &(base, last, offset) in &src.skip_list {
                let pos = skip_list_start + skip_idx * 12;
                out[pos..pos + 4].copy_from_slice(&(base + src.doc_offset).to_le_bytes());
                out[pos + 4..pos + 8].copy_from_slice(&(last + src.doc_offset).to_le_bytes());
                out[pos + 8..pos + 12].copy_from_slice(&(data_base_offset + offset).to_le_bytes());
                skip_idx += 1;
            }
            data_base_offset += src_data_len;
        }

        // data_len + block data with patched first_doc
        out.write_u32::<LittleEndian>(data_base_offset)?;

        for src in &parsed {
            for (i, &(_, _, offset)) in src.skip_list.iter().enumerate() {
                let start = offset as usize;
                let end = if i + 1 < src.skip_list.len() {
                    src.skip_list[i + 1].2 as usize
                } else {
                    src.data.len()
                };
                let block = &src.data[start..end];

                out.extend_from_slice(&block[0..4]); // count (unchanged)
                let first_doc = u32::from_le_bytes(block[4..8].try_into().unwrap());
                out.extend_from_slice(&(first_doc + src.doc_offset).to_le_bytes());
                out.extend_from_slice(&block[8..]); // rest verbatim
            }
        }

        Ok(total_docs)
    }

    /// Get iterator over all postings (for phrase queries)
    pub fn iter(&self) -> PositionPostingIterator<'_> {
        PositionPostingIterator::new(self)
    }
}

/// Write variable-length integer (same as posting.rs)
fn write_vint<W: Write>(writer: &mut W, mut value: u64) -> io::Result<()> {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            writer.write_u8(byte)?;
            break;
        } else {
            writer.write_u8(byte | 0x80)?;
        }
    }
    Ok(())
}

/// Read variable-length integer (same as posting.rs)
fn read_vint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        let byte = reader.read_u8()?;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(result)
}

/// Iterator over block-based position posting list
pub struct PositionPostingIterator<'a> {
    list: &'a PositionPostingList,
    current_block: usize,
    position_in_block: usize,
    block_postings: Vec<PostingWithPositions>,
    exhausted: bool,
}

impl<'a> PositionPostingIterator<'a> {
    pub fn new(list: &'a PositionPostingList) -> Self {
        let exhausted = list.skip_list.is_empty();
        let mut iter = Self {
            list,
            current_block: 0,
            position_in_block: 0,
            block_postings: Vec::new(),
            exhausted,
        };
        if !iter.exhausted {
            iter.load_block(0);
        }
        iter
    }

    fn load_block(&mut self, block_idx: usize) {
        if block_idx >= self.list.skip_list.len() {
            self.exhausted = true;
            return;
        }

        self.current_block = block_idx;
        self.position_in_block = 0;

        let offset = self.list.skip_list[block_idx].2 as usize;
        let mut reader = &self.list.data[offset..];

        // Fixed 8-byte prefix: count(u32) + first_doc(u32)
        let count = reader.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        let first_doc = reader.read_u32::<LittleEndian>().unwrap_or(0);
        self.block_postings.clear();
        self.block_postings.reserve(count);

        let mut prev_doc_id = first_doc;

        for i in 0..count {
            let doc_id = if i == 0 {
                first_doc
            } else {
                let delta = read_vint(&mut reader).unwrap_or(0) as u32;
                prev_doc_id + delta
            };
            prev_doc_id = doc_id;

            let num_positions = read_vint(&mut reader).unwrap_or(0) as usize;
            let mut positions = Vec::with_capacity(num_positions);
            for _ in 0..num_positions {
                let pos = read_vint(&mut reader).unwrap_or(0) as u32;
                positions.push(pos);
            }

            self.block_postings.push(PostingWithPositions {
                doc_id,
                term_freq: num_positions as u32,
                positions,
            });
        }
    }

    pub fn doc(&self) -> DocId {
        if self.exhausted || self.position_in_block >= self.block_postings.len() {
            u32::MAX
        } else {
            self.block_postings[self.position_in_block].doc_id
        }
    }

    pub fn term_freq(&self) -> u32 {
        if self.exhausted || self.position_in_block >= self.block_postings.len() {
            0
        } else {
            self.block_postings[self.position_in_block].term_freq
        }
    }

    pub fn positions(&self) -> &[u32] {
        if self.exhausted || self.position_in_block >= self.block_postings.len() {
            &[]
        } else {
            &self.block_postings[self.position_in_block].positions
        }
    }

    pub fn advance(&mut self) {
        if self.exhausted {
            return;
        }

        self.position_in_block += 1;
        if self.position_in_block >= self.block_postings.len() {
            self.load_block(self.current_block + 1);
        }
    }

    pub fn seek(&mut self, target: DocId) {
        if self.exhausted {
            return;
        }

        // Check if target is in current block
        if let Some((_, last, _)) = self.list.skip_list.get(self.current_block)
            && target <= *last
        {
            // Target might be in current block, scan forward
            while self.position_in_block < self.block_postings.len()
                && self.block_postings[self.position_in_block].doc_id < target
            {
                self.position_in_block += 1;
            }
            if self.position_in_block >= self.block_postings.len() {
                self.load_block(self.current_block + 1);
                self.seek(target); // Continue seeking in next block
            }
            return;
        }

        // Binary search on skip list to find the right block
        let block_idx = match self.list.skip_list.binary_search_by(|&(base, last, _)| {
            if target < base {
                std::cmp::Ordering::Greater
            } else if target > last {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(idx) => idx,
            Err(idx) => idx, // Use the next block if not found exactly
        };

        if block_idx >= self.list.skip_list.len() {
            self.exhausted = true;
            return;
        }

        self.load_block(block_idx);

        // Linear scan within block
        while self.position_in_block < self.block_postings.len()
            && self.block_postings[self.position_in_block].doc_id < target
        {
            self.position_in_block += 1;
        }

        if self.position_in_block >= self.block_postings.len() {
            self.load_block(self.current_block + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_encoding() {
        // Element 0, position 5
        let pos = encode_position(0, 5);
        assert_eq!(decode_element_ordinal(pos), 0);
        assert_eq!(decode_token_position(pos), 5);

        // Element 3, position 100
        let pos = encode_position(3, 100);
        assert_eq!(decode_element_ordinal(pos), 3);
        assert_eq!(decode_token_position(pos), 100);

        // Max values
        let pos = encode_position(MAX_ELEMENT_ORDINAL, MAX_TOKEN_POSITION);
        assert_eq!(decode_element_ordinal(pos), MAX_ELEMENT_ORDINAL);
        assert_eq!(decode_token_position(pos), MAX_TOKEN_POSITION);
    }

    #[test]
    fn test_position_posting_list_build() {
        // Build from postings
        let postings = vec![
            PostingWithPositions {
                doc_id: 1,
                term_freq: 2,
                positions: vec![encode_position(0, 0), encode_position(0, 2)],
            },
            PostingWithPositions {
                doc_id: 3,
                term_freq: 1,
                positions: vec![encode_position(1, 0)],
            },
        ];

        let list = PositionPostingList::from_postings(&postings).unwrap();
        assert_eq!(list.doc_count(), 2);

        // Test binary search
        let pos = list.get_positions(1).unwrap();
        assert_eq!(pos.len(), 2);

        let pos = list.get_positions(3).unwrap();
        assert_eq!(pos.len(), 1);

        // Not found
        assert!(list.get_positions(2).is_none());
        assert!(list.get_positions(99).is_none());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let postings = vec![
            PostingWithPositions {
                doc_id: 1,
                term_freq: 2,
                positions: vec![encode_position(0, 0), encode_position(0, 5)],
            },
            PostingWithPositions {
                doc_id: 3,
                term_freq: 1,
                positions: vec![encode_position(1, 0)],
            },
            PostingWithPositions {
                doc_id: 5,
                term_freq: 1,
                positions: vec![encode_position(0, 10)],
            },
        ];

        let list = PositionPostingList::from_postings(&postings).unwrap();

        let mut bytes = Vec::new();
        list.serialize(&mut bytes).unwrap();

        let mut cursor = std::io::Cursor::new(&bytes);
        let deserialized = PositionPostingList::deserialize(&mut cursor).unwrap();

        assert_eq!(list.doc_count(), deserialized.doc_count());

        // Verify binary search works on deserialized
        let pos = deserialized.get_positions(1).unwrap();
        assert_eq!(pos, vec![encode_position(0, 0), encode_position(0, 5)]);

        let pos = deserialized.get_positions(3).unwrap();
        assert_eq!(pos, vec![encode_position(1, 0)]);
    }

    #[test]
    fn test_binary_search_many_blocks() {
        // Create enough postings to span multiple blocks (128 per block)
        let mut postings = Vec::new();
        for i in 0..300 {
            postings.push(PostingWithPositions {
                doc_id: i * 2, // doc_ids: 0, 2, 4, 6, ...
                term_freq: 1,
                positions: vec![encode_position(0, i)],
            });
        }

        let list = PositionPostingList::from_postings(&postings).unwrap();
        assert_eq!(list.doc_count(), 300);

        // Should have 3 blocks (128 + 128 + 44)
        assert_eq!(list.skip_list.len(), 3);

        // Test binary search across blocks
        let pos = list.get_positions(0).unwrap();
        assert_eq!(pos, vec![encode_position(0, 0)]);

        let pos = list.get_positions(256).unwrap(); // doc 128 in block 1
        assert_eq!(pos, vec![encode_position(0, 128)]);

        let pos = list.get_positions(598).unwrap(); // last doc
        assert_eq!(pos, vec![encode_position(0, 299)]);

        // Not found (odd numbers don't exist)
        assert!(list.get_positions(1).is_none());
        assert!(list.get_positions(257).is_none());
    }

    #[test]
    fn test_concatenate_blocks_merge() {
        // Test efficient merge by concatenating blocks
        let postings1 = vec![
            PostingWithPositions {
                doc_id: 0,
                term_freq: 1,
                positions: vec![0],
            },
            PostingWithPositions {
                doc_id: 1,
                term_freq: 1,
                positions: vec![5],
            },
            PostingWithPositions {
                doc_id: 2,
                term_freq: 1,
                positions: vec![10],
            },
        ];
        let list1 = PositionPostingList::from_postings(&postings1).unwrap();

        let postings2 = vec![
            PostingWithPositions {
                doc_id: 0,
                term_freq: 1,
                positions: vec![100],
            },
            PostingWithPositions {
                doc_id: 1,
                term_freq: 1,
                positions: vec![105],
            },
        ];
        let list2 = PositionPostingList::from_postings(&postings2).unwrap();

        // Merge with doc_id offset
        let combined = PositionPostingList::concatenate_blocks(&[
            (list1, 0), // no offset
            (list2, 3), // offset by 3 (list1 has 3 docs)
        ])
        .unwrap();

        assert_eq!(combined.doc_count(), 5);

        // Check doc_ids are correctly remapped
        assert!(combined.get_positions(0).is_some());
        assert!(combined.get_positions(1).is_some());
        assert!(combined.get_positions(2).is_some());
        assert!(combined.get_positions(3).is_some()); // was 0 in list2
        assert!(combined.get_positions(4).is_some()); // was 1 in list2
    }

    #[test]
    fn test_iterator() {
        let postings = vec![
            PostingWithPositions {
                doc_id: 1,
                term_freq: 2,
                positions: vec![0, 5],
            },
            PostingWithPositions {
                doc_id: 3,
                term_freq: 1,
                positions: vec![10],
            },
            PostingWithPositions {
                doc_id: 5,
                term_freq: 1,
                positions: vec![15],
            },
        ];

        let list = PositionPostingList::from_postings(&postings).unwrap();
        let mut iter = list.iter();

        assert_eq!(iter.doc(), 1);
        assert_eq!(iter.positions(), &[0, 5]);

        iter.advance();
        assert_eq!(iter.doc(), 3);

        iter.seek(5);
        assert_eq!(iter.doc(), 5);
        assert_eq!(iter.positions(), &[15]);

        iter.advance();
        assert_eq!(iter.doc(), u32::MAX); // exhausted
    }
}
