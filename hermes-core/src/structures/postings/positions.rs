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
use std::io::{self, Write};

use super::posting_common::{read_vint, write_vint};
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
    skip_list: Vec<(DocId, DocId, u64)>,
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
            let block_start = data.len() as u64;
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
        let block_start = self.data.len() as u64;

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

    /// Size of one serialized skip entry:
    /// first_doc(4) + last_doc(4) + offset(8) + length(4) = 20 bytes
    const SKIP_ENTRY_SIZE: usize = 20;

    /// Serialize to bytes (footer-based: data first).
    ///
    /// Format:
    /// ```text
    /// [block data: data_len bytes]
    /// [skip entries: N × 20 bytes (base_doc, last_doc, offset, length)]
    /// [footer: data_len(8) + skip_count(4) + doc_count(4) = 16 bytes]
    /// ```
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Data first
        writer.write_all(&self.data)?;

        // Skip list — compute length from adjacent entries
        for (i, (base_doc_id, last_doc_id, offset)) in self.skip_list.iter().enumerate() {
            let next_offset = if i + 1 < self.skip_list.len() {
                self.skip_list[i + 1].2
            } else {
                self.data.len() as u64
            };
            let length = (next_offset - offset) as u32;
            writer.write_u32::<LittleEndian>(*base_doc_id)?;
            writer.write_u32::<LittleEndian>(*last_doc_id)?;
            writer.write_u64::<LittleEndian>(*offset)?;
            writer.write_u32::<LittleEndian>(length)?;
        }

        // Footer
        writer.write_u64::<LittleEndian>(self.data.len() as u64)?;
        writer.write_u32::<LittleEndian>(self.skip_list.len() as u32)?;
        writer.write_u32::<LittleEndian>(self.doc_count)?;

        Ok(())
    }

    /// Deserialize from a byte slice (footer-based format).
    pub fn deserialize(raw: &[u8]) -> io::Result<Self> {
        if raw.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "position data too short",
            ));
        }

        // Parse footer (last 16 bytes)
        let f = raw.len() - 16;
        let data_len = u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()) as usize;
        let skip_count = u32::from_le_bytes(raw[f + 8..f + 12].try_into().unwrap()) as usize;
        let doc_count = u32::from_le_bytes(raw[f + 12..f + 16].try_into().unwrap());

        // Parse skip list (20-byte entries; length field not stored in-memory)
        let mut skip_list = Vec::with_capacity(skip_count);
        let mut pos = data_len;
        for _ in 0..skip_count {
            let base = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
            let last = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap());
            let offset = u64::from_le_bytes(raw[pos + 8..pos + 16].try_into().unwrap());
            // pos + 16..pos + 20 = length (not needed in-memory; adjacent entries suffice)
            skip_list.push((base, last, offset));
            pos += Self::SKIP_ENTRY_SIZE;
        }

        let data = raw[..data_len].to_vec();

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
                let new_offset = data.len() as u64;

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

    /// Streaming merge: write blocks directly to output writer (bounded memory).
    ///
    /// **Zero-materializing**: reads skip entries directly from source bytes
    /// without parsing into Vecs. Explicit `length` field in each 20-byte
    /// entry eliminates adjacent-entry lookups.
    ///
    /// Only output skip bytes are buffered (bounded O(total_blocks × 20)).
    /// Block data flows source → output writer without intermediate buffering.
    ///
    /// Returns `(doc_count, bytes_written)`.
    pub fn concatenate_streaming<W: Write>(
        sources: &[(&[u8], u32)],
        writer: &mut W,
    ) -> io::Result<(u32, usize)> {
        // Parse only footers (16 bytes each) — no skip entries materialized
        struct SourceMeta {
            data_len: usize,
            skip_count: usize,
        }

        let mut metas: Vec<SourceMeta> = Vec::with_capacity(sources.len());
        let mut total_docs = 0u32;

        for (raw, _) in sources {
            if raw.len() < 16 {
                continue;
            }
            let f = raw.len() - 16;
            let data_len = u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()) as usize;
            let skip_count = u32::from_le_bytes(raw[f + 8..f + 12].try_into().unwrap()) as usize;
            let doc_count = u32::from_le_bytes(raw[f + 12..f + 16].try_into().unwrap());
            total_docs += doc_count;
            metas.push(SourceMeta {
                data_len,
                skip_count,
            });
        }

        // Phase 1: Stream block data, reading skip entries on-the-fly.
        // Accumulate output skip bytes (bounded: 20 bytes × total_blocks).
        let mut out_skip: Vec<u8> = Vec::new();
        let mut out_skip_count = 0u32;
        let mut data_written = 0u64;
        let mut patch_buf = [0u8; 8];
        let es = Self::SKIP_ENTRY_SIZE;

        for (src_idx, meta) in metas.iter().enumerate() {
            let (raw, doc_offset) = &sources[src_idx];
            let skip_base = meta.data_len;
            let data = &raw[..meta.data_len];

            for i in 0..meta.skip_count {
                // Read source skip entry directly from raw bytes
                let p = skip_base + i * es;
                let base = u32::from_le_bytes(raw[p..p + 4].try_into().unwrap());
                let last = u32::from_le_bytes(raw[p + 4..p + 8].try_into().unwrap());
                let offset = u64::from_le_bytes(raw[p + 8..p + 16].try_into().unwrap());
                let length = u32::from_le_bytes(raw[p + 16..p + 20].try_into().unwrap());

                let block = &data[offset as usize..(offset as usize + length as usize)];

                // Write output skip entry
                out_skip.extend_from_slice(&(base + doc_offset).to_le_bytes());
                out_skip.extend_from_slice(&(last + doc_offset).to_le_bytes());
                out_skip.extend_from_slice(&data_written.to_le_bytes());
                out_skip.extend_from_slice(&length.to_le_bytes());
                out_skip_count += 1;

                // Write patched 8-byte prefix + rest of block verbatim
                patch_buf[0..4].copy_from_slice(&block[0..4]);
                let first_doc = u32::from_le_bytes(block[4..8].try_into().unwrap());
                patch_buf[4..8].copy_from_slice(&(first_doc + doc_offset).to_le_bytes());
                writer.write_all(&patch_buf)?;
                writer.write_all(&block[8..])?;

                data_written += block.len() as u64;
            }
        }

        // Phase 2: Write skip entries + footer
        writer.write_all(&out_skip)?;

        writer.write_u64::<LittleEndian>(data_written)?;
        writer.write_u32::<LittleEndian>(out_skip_count)?;
        writer.write_u32::<LittleEndian>(total_docs)?;

        let total_bytes = data_written as usize + out_skip.len() + 16;
        Ok((total_docs, total_bytes))
    }

    /// Get iterator over all postings (for phrase queries)
    pub fn iter(&self) -> PositionPostingIterator<'_> {
        PositionPostingIterator::new(self)
    }
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

        let deserialized = PositionPostingList::deserialize(&bytes).unwrap();

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

    /// Helper: build a PositionPostingList from (doc_id, positions) pairs
    fn build_ppl(entries: &[(u32, Vec<u32>)]) -> PositionPostingList {
        let postings: Vec<PostingWithPositions> = entries
            .iter()
            .map(|(doc_id, positions)| PostingWithPositions {
                doc_id: *doc_id,
                term_freq: positions.len() as u32,
                positions: positions.clone(),
            })
            .collect();
        PositionPostingList::from_postings(&postings).unwrap()
    }

    /// Helper: serialize a PositionPostingList to bytes
    fn serialize_ppl(ppl: &PositionPostingList) -> Vec<u8> {
        let mut buf = Vec::new();
        ppl.serialize(&mut buf).unwrap();
        buf
    }

    /// Helper: collect all (doc_id, positions) from a PositionPostingIterator
    fn collect_positions(ppl: &PositionPostingList) -> Vec<(u32, Vec<u32>)> {
        let mut result = Vec::new();
        let mut it = ppl.iter();
        while it.doc() != u32::MAX {
            result.push((it.doc(), it.positions().to_vec()));
            it.advance();
        }
        result
    }

    #[test]
    fn test_concatenate_streaming_matches_blocks() {
        // Build 3 segments with positions
        let seg_a: Vec<(u32, Vec<u32>)> = (0..150)
            .map(|i| (i * 2, vec![i * 10, i * 10 + 3]))
            .collect();
        let seg_b: Vec<(u32, Vec<u32>)> = (0..100).map(|i| (i * 5, vec![i * 7])).collect();
        let seg_c: Vec<(u32, Vec<u32>)> = (0..80).map(|i| (i * 3, vec![i, i + 1, i + 2])).collect();

        let ppl_a = build_ppl(&seg_a);
        let ppl_b = build_ppl(&seg_b);
        let ppl_c = build_ppl(&seg_c);

        let offset_b = 500u32;
        let offset_c = 1000u32;

        // Method 1: concatenate_blocks (reference)
        let ref_merged = PositionPostingList::concatenate_blocks(&[
            (ppl_a.clone(), 0),
            (ppl_b.clone(), offset_b),
            (ppl_c.clone(), offset_c),
        ])
        .unwrap();
        let mut ref_buf = Vec::new();
        ref_merged.serialize(&mut ref_buf).unwrap();

        // Method 2: concatenate_streaming
        let bytes_a = serialize_ppl(&ppl_a);
        let bytes_b = serialize_ppl(&ppl_b);
        let bytes_c = serialize_ppl(&ppl_c);

        let sources: Vec<(&[u8], u32)> =
            vec![(&bytes_a, 0), (&bytes_b, offset_b), (&bytes_c, offset_c)];
        let mut stream_buf = Vec::new();
        let (doc_count, bytes_written) =
            PositionPostingList::concatenate_streaming(&sources, &mut stream_buf).unwrap();

        assert_eq!(doc_count, 330);
        assert_eq!(bytes_written, stream_buf.len());

        // Deserialize both and compare all postings + positions
        let ref_posts = collect_positions(&PositionPostingList::deserialize(&ref_buf).unwrap());
        let stream_posts =
            collect_positions(&PositionPostingList::deserialize(&stream_buf).unwrap());

        assert_eq!(ref_posts.len(), stream_posts.len());
        for (i, (r, s)) in ref_posts.iter().zip(stream_posts.iter()).enumerate() {
            assert_eq!(r.0, s.0, "doc_id mismatch at {}", i);
            assert_eq!(r.1, s.1, "positions mismatch at doc {}", r.0);
        }
    }

    #[test]
    fn test_positions_multi_round_merge() {
        // Round 0: 4 segments with distinct positions
        let segments: Vec<Vec<(u32, Vec<u32>)>> = (0..4)
            .map(|seg| {
                (0..200)
                    .map(|i| {
                        let pos_count = (i % 3) + 1;
                        let positions: Vec<u32> = (0..pos_count)
                            .map(|p| (seg * 1000 + i * 10 + p) as u32)
                            .collect();
                        (i as u32 * 3, positions)
                    })
                    .collect()
            })
            .collect();

        let ppls: Vec<PositionPostingList> = segments.iter().map(|s| build_ppl(s)).collect();
        let serialized: Vec<Vec<u8>> = ppls.iter().map(serialize_ppl).collect();

        // Round 1: merge pairs
        let mut merged_01 = Vec::new();
        let sources_01: Vec<(&[u8], u32)> = vec![(&serialized[0], 0), (&serialized[1], 600)];
        let (dc_01, _) =
            PositionPostingList::concatenate_streaming(&sources_01, &mut merged_01).unwrap();
        assert_eq!(dc_01, 400);

        let mut merged_23 = Vec::new();
        let sources_23: Vec<(&[u8], u32)> = vec![(&serialized[2], 0), (&serialized[3], 600)];
        let (dc_23, _) =
            PositionPostingList::concatenate_streaming(&sources_23, &mut merged_23).unwrap();
        assert_eq!(dc_23, 400);

        // Round 2: merge intermediate results
        let mut final_merged = Vec::new();
        let sources_final: Vec<(&[u8], u32)> = vec![(&merged_01, 0), (&merged_23, 1200)];
        let (dc_final, _) =
            PositionPostingList::concatenate_streaming(&sources_final, &mut final_merged).unwrap();
        assert_eq!(dc_final, 800);

        // Verify all positions survived two rounds of merging
        let final_ppl = PositionPostingList::deserialize(&final_merged).unwrap();
        let all = collect_positions(&final_ppl);
        assert_eq!(all.len(), 800);

        // Spot-check: first doc of each original segment
        // Seg0: doc_id=0, positions from seg=0
        assert_eq!(all[0].0, 0);
        assert_eq!(all[0].1, vec![0]); // seg=0, i=0, pos_count=1, pos=0*1000+0*10+0=0

        // Seg1: doc_id=600, positions from seg=1
        assert_eq!(all[200].0, 600);
        assert_eq!(all[200].1, vec![1000]); // seg=1, i=0, pos_count=1, pos=1*1000+0*10+0=1000

        // Seg2: doc_id=1200, positions from seg=2
        assert_eq!(all[400].0, 1200);
        assert_eq!(all[400].1, vec![2000]); // seg=2, i=0, pos_count=1, pos=2*1000+0*10+0=2000

        // Verify seek + positions on the final merged list
        let mut it = final_ppl.iter();
        it.seek(1200);
        assert_eq!(it.doc(), 1200);
        assert_eq!(it.positions(), &[2000]);

        // Verify get_positions (binary search path)
        let pos = final_ppl.get_positions(600).unwrap();
        assert_eq!(pos, vec![1000]);
    }

    #[test]
    fn test_positions_large_scale_merge() {
        // 5 segments × 500 docs, each doc has 1-4 positions
        let num_segments = 5usize;
        let docs_per_segment = 500usize;

        let segments: Vec<Vec<(u32, Vec<u32>)>> = (0..num_segments)
            .map(|seg| {
                (0..docs_per_segment)
                    .map(|i| {
                        let n_pos = (i % 4) + 1;
                        let positions: Vec<u32> =
                            (0..n_pos).map(|p| (p * 5 + seg) as u32).collect();
                        (i as u32 * 2, positions)
                    })
                    .collect()
            })
            .collect();

        let ppls: Vec<PositionPostingList> = segments.iter().map(|s| build_ppl(s)).collect();
        let serialized: Vec<Vec<u8>> = ppls.iter().map(serialize_ppl).collect();

        let max_doc = (docs_per_segment as u32 - 1) * 2;
        let offsets: Vec<u32> = (0..num_segments)
            .map(|i| i as u32 * (max_doc + 1))
            .collect();

        let sources: Vec<(&[u8], u32)> = serialized
            .iter()
            .zip(offsets.iter())
            .map(|(b, o)| (b.as_slice(), *o))
            .collect();

        let mut merged = Vec::new();
        let (doc_count, _) =
            PositionPostingList::concatenate_streaming(&sources, &mut merged).unwrap();
        assert_eq!(doc_count, (num_segments * docs_per_segment) as u32);

        // Deserialize and verify all positions
        let merged_ppl = PositionPostingList::deserialize(&merged).unwrap();
        let all = collect_positions(&merged_ppl);
        assert_eq!(all.len(), num_segments * docs_per_segment);

        // Verify positions for each segment
        for (seg, &offset) in offsets.iter().enumerate() {
            for i in 0..docs_per_segment {
                let idx = seg * docs_per_segment + i;
                let expected_doc = i as u32 * 2 + offset;
                assert_eq!(all[idx].0, expected_doc, "seg={} i={}", seg, i);

                let n_pos = (i % 4) + 1;
                let expected_positions: Vec<u32> =
                    (0..n_pos).map(|p| (p * 5 + seg) as u32).collect();
                assert_eq!(
                    all[idx].1, expected_positions,
                    "positions mismatch seg={} i={}",
                    seg, i
                );
            }
        }
    }

    #[test]
    fn test_positions_streaming_single_source() {
        let entries: Vec<(u32, Vec<u32>)> =
            (0..300).map(|i| (i * 4, vec![i * 2, i * 2 + 1])).collect();
        let ppl = build_ppl(&entries);
        let direct = serialize_ppl(&ppl);

        let sources: Vec<(&[u8], u32)> = vec![(&direct, 0)];
        let mut streamed = Vec::new();
        PositionPostingList::concatenate_streaming(&sources, &mut streamed).unwrap();

        let p1 = collect_positions(&PositionPostingList::deserialize(&direct).unwrap());
        let p2 = collect_positions(&PositionPostingList::deserialize(&streamed).unwrap());
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_positions_edge_single_doc() {
        let ppl_a = build_ppl(&[(0, vec![42, 43, 44])]);
        let ppl_b = build_ppl(&[(0, vec![100])]);

        let merged = PositionPostingList::concatenate_blocks(&[(ppl_a, 0), (ppl_b, 1)]).unwrap();

        assert_eq!(merged.doc_count(), 2);
        assert_eq!(merged.get_positions(0).unwrap(), vec![42, 43, 44]);
        assert_eq!(merged.get_positions(1).unwrap(), vec![100]);
    }
}
