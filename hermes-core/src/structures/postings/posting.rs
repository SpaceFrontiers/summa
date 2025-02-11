//! Posting list implementation with compact representation
//!
//! Uses delta encoding and variable-length integers for compact storage.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use super::posting_common::{read_vint, write_vint};
use crate::DocId;
use crate::directories::OwnedBytes;

/// A posting entry containing doc_id and term frequency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub doc_id: DocId,
    pub term_freq: u32,
}

/// Compact posting list with delta encoding
#[derive(Debug, Clone, Default)]
pub struct PostingList {
    postings: Vec<Posting>,
}

impl PostingList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            postings: Vec::with_capacity(capacity),
        }
    }

    /// Add a posting (must be added in doc_id order)
    pub fn push(&mut self, doc_id: DocId, term_freq: u32) {
        debug_assert!(
            self.postings.is_empty() || self.postings.last().unwrap().doc_id < doc_id,
            "Postings must be added in sorted order"
        );
        self.postings.push(Posting { doc_id, term_freq });
    }

    /// Add a posting, incrementing term_freq if doc already exists
    pub fn add(&mut self, doc_id: DocId, term_freq: u32) {
        if let Some(last) = self.postings.last_mut()
            && last.doc_id == doc_id
        {
            last.term_freq += term_freq;
            return;
        }
        self.postings.push(Posting { doc_id, term_freq });
    }

    /// Get document count
    pub fn doc_count(&self) -> u32 {
        self.postings.len() as u32
    }

    pub fn len(&self) -> usize {
        self.postings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Posting> {
        self.postings.iter()
    }

    /// Serialize to bytes using delta encoding and varint
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write number of postings
        write_vint(writer, self.postings.len() as u64)?;

        let mut prev_doc_id = 0u32;
        for posting in &self.postings {
            // Delta encode doc_id
            let delta = posting.doc_id - prev_doc_id;
            write_vint(writer, delta as u64)?;
            write_vint(writer, posting.term_freq as u64)?;
            prev_doc_id = posting.doc_id;
        }

        Ok(())
    }

    /// Deserialize from bytes
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let count = read_vint(reader)? as usize;
        let mut postings = Vec::with_capacity(count);

        let mut prev_doc_id = 0u32;
        for _ in 0..count {
            let delta = read_vint(reader)? as u32;
            let term_freq = read_vint(reader)? as u32;
            let doc_id = prev_doc_id + delta;
            postings.push(Posting { doc_id, term_freq });
            prev_doc_id = doc_id;
        }

        Ok(Self { postings })
    }
}

/// Iterator over posting list that supports seeking
pub struct PostingListIterator<'a> {
    postings: &'a [Posting],
    position: usize,
}

impl<'a> PostingListIterator<'a> {
    pub fn new(posting_list: &'a PostingList) -> Self {
        Self {
            postings: &posting_list.postings,
            position: 0,
        }
    }

    /// Current document ID, or TERMINATED if exhausted
    pub fn doc(&self) -> DocId {
        if self.position < self.postings.len() {
            self.postings[self.position].doc_id
        } else {
            TERMINATED
        }
    }

    /// Current term frequency
    pub fn term_freq(&self) -> u32 {
        if self.position < self.postings.len() {
            self.postings[self.position].term_freq
        } else {
            0
        }
    }

    /// Advance to next posting, returns new doc_id or TERMINATED
    pub fn advance(&mut self) -> DocId {
        self.position += 1;
        self.doc()
    }

    /// Seek to first doc_id >= target (binary search on remaining postings)
    pub fn seek(&mut self, target: DocId) -> DocId {
        let remaining = &self.postings[self.position..];
        let offset = remaining.partition_point(|p| p.doc_id < target);
        self.position += offset;
        self.doc()
    }

    /// Size hint for remaining elements
    pub fn size_hint(&self) -> usize {
        self.postings.len().saturating_sub(self.position)
    }
}

/// Sentinel value indicating iterator is exhausted
pub const TERMINATED: DocId = DocId::MAX;

/// Block-based posting list for skip-list style access
/// Each block contains up to BLOCK_SIZE postings
pub const BLOCK_SIZE: usize = 128;

#[derive(Debug, Clone)]
pub struct BlockPostingList {
    /// Skip list: (base_doc_id, last_doc_id_in_block, byte_offset, block_max_tf)
    /// base_doc_id is the first doc_id in the block (absolute, not delta)
    /// block_max_tf enables Block-Max WAND optimization
    skip_list: Vec<(DocId, DocId, u64, u32)>,
    /// Compressed posting data (OwnedBytes for zero-copy mmap support)
    data: OwnedBytes,
    /// Total number of postings
    doc_count: u32,
    /// Maximum term frequency across all postings (for WAND upper bound)
    max_tf: u32,
}

impl BlockPostingList {
    /// Build from a posting list
    pub fn from_posting_list(list: &PostingList) -> io::Result<Self> {
        let mut skip_list = Vec::new();
        let mut data = Vec::new();
        let mut max_tf = 0u32;

        let postings = &list.postings;
        let mut i = 0;

        while i < postings.len() {
            let block_start = data.len() as u64;
            let block_end = (i + BLOCK_SIZE).min(postings.len());
            let block = &postings[i..block_end];

            // Compute block's max term frequency for Block-Max WAND
            let block_max_tf = block.iter().map(|p| p.term_freq).max().unwrap_or(0);
            max_tf = max_tf.max(block_max_tf);

            // Record skip entry with base_doc_id (first doc in block)
            let base_doc_id = block.first().unwrap().doc_id;
            let last_doc_id = block.last().unwrap().doc_id;
            skip_list.push((base_doc_id, last_doc_id, block_start, block_max_tf));

            // Write block: fixed u32 count + first_doc (8-byte prefix), then vint deltas
            data.write_u32::<LittleEndian>(block.len() as u32)?;
            data.write_u32::<LittleEndian>(base_doc_id)?;

            let mut prev_doc_id = base_doc_id;
            for (j, posting) in block.iter().enumerate() {
                if j == 0 {
                    // First doc already in fixed prefix, just write tf
                    write_vint(&mut data, posting.term_freq as u64)?;
                } else {
                    let delta = posting.doc_id - prev_doc_id;
                    write_vint(&mut data, delta as u64)?;
                    write_vint(&mut data, posting.term_freq as u64)?;
                }
                prev_doc_id = posting.doc_id;
            }

            i = block_end;
        }

        Ok(Self {
            skip_list,
            data: OwnedBytes::new(data),
            doc_count: postings.len() as u32,
            max_tf,
        })
    }

    /// Serialize the block posting list (footer-based: data first).
    ///
    /// Format:
    /// ```text
    /// [block data: data_len bytes]
    /// [skip entries: N × 20 bytes (base_doc, last_doc, offset_u64, block_max_tf)]
    /// [footer: data_len(8) + skip_count(4) + doc_count(4) + max_tf(4) = 20 bytes]
    /// ```
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Data first (enables streaming writes during merge)
        writer.write_all(&self.data)?;

        // Skip list
        for (base_doc_id, last_doc_id, offset, block_max_tf) in &self.skip_list {
            writer.write_u32::<LittleEndian>(*base_doc_id)?;
            writer.write_u32::<LittleEndian>(*last_doc_id)?;
            writer.write_u64::<LittleEndian>(*offset)?;
            writer.write_u32::<LittleEndian>(*block_max_tf)?;
        }

        // Footer
        writer.write_u64::<LittleEndian>(self.data.len() as u64)?;
        writer.write_u32::<LittleEndian>(self.skip_list.len() as u32)?;
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;

        Ok(())
    }

    /// Deserialize from a byte slice (footer-based format).
    pub fn deserialize(raw: &[u8]) -> io::Result<Self> {
        if raw.len() < 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting data too short",
            ));
        }

        // Parse footer (last 20 bytes)
        let f = raw.len() - 20;
        let data_len = u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()) as usize;
        let skip_count = u32::from_le_bytes(raw[f + 8..f + 12].try_into().unwrap()) as usize;
        let doc_count = u32::from_le_bytes(raw[f + 12..f + 16].try_into().unwrap());
        let max_tf = u32::from_le_bytes(raw[f + 16..f + 20].try_into().unwrap());

        // Parse skip list (between data and footer)
        let mut skip_list = Vec::with_capacity(skip_count);
        let mut pos = data_len;
        for _ in 0..skip_count {
            let base = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
            let last = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap());
            let offset = u64::from_le_bytes(raw[pos + 8..pos + 16].try_into().unwrap());
            let block_max_tf = u32::from_le_bytes(raw[pos + 16..pos + 20].try_into().unwrap());
            skip_list.push((base, last, offset, block_max_tf));
            pos += 20;
        }

        let data = OwnedBytes::new(raw[..data_len].to_vec());

        Ok(Self {
            skip_list,
            data,
            max_tf,
            doc_count,
        })
    }

    /// Zero-copy deserialization from OwnedBytes.
    /// The data section is sliced from the source without copying.
    pub fn deserialize_zero_copy(raw: OwnedBytes) -> io::Result<Self> {
        if raw.len() < 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting data too short",
            ));
        }

        let f = raw.len() - 20;
        let data_len = u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()) as usize;
        let skip_count = u32::from_le_bytes(raw[f + 8..f + 12].try_into().unwrap()) as usize;
        let doc_count = u32::from_le_bytes(raw[f + 12..f + 16].try_into().unwrap());
        let max_tf = u32::from_le_bytes(raw[f + 16..f + 20].try_into().unwrap());

        let mut skip_list = Vec::with_capacity(skip_count);
        let mut pos = data_len;
        for _ in 0..skip_count {
            let base = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
            let last = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap());
            let offset = u64::from_le_bytes(raw[pos + 8..pos + 16].try_into().unwrap());
            let block_max_tf = u32::from_le_bytes(raw[pos + 16..pos + 20].try_into().unwrap());
            skip_list.push((base, last, offset, block_max_tf));
            pos += 20;
        }

        // Zero-copy: slice references the source OwnedBytes (backed by mmap or Arc<Vec>)
        let data = raw.slice(0..data_len);

        Ok(Self {
            skip_list,
            data,
            max_tf,
            doc_count,
        })
    }

    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Get maximum term frequency (for WAND upper bound computation)
    pub fn max_tf(&self) -> u32 {
        self.max_tf
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.skip_list.len()
    }

    /// Get block metadata: (base_doc_id, last_doc_id, data_offset, data_len, block_max_tf)
    pub fn block_info(&self, block_idx: usize) -> Option<(DocId, DocId, usize, usize, u32)> {
        if block_idx >= self.skip_list.len() {
            return None;
        }
        let (base, last, offset, block_max_tf) = self.skip_list[block_idx];
        let next_offset = if block_idx + 1 < self.skip_list.len() {
            self.skip_list[block_idx + 1].2 as usize
        } else {
            self.data.len()
        };
        Some((
            base,
            last,
            offset as usize,
            next_offset - offset as usize,
            block_max_tf,
        ))
    }

    /// Get block's max term frequency for Block-Max WAND
    pub fn block_max_tf(&self, block_idx: usize) -> Option<u32> {
        self.skip_list
            .get(block_idx)
            .map(|(_, _, _, max_tf)| *max_tf)
    }

    /// Get raw block data for direct copying during merge
    pub fn block_data(&self, block_idx: usize) -> Option<&[u8]> {
        let (_, _, offset, len, _) = self.block_info(block_idx)?;
        Some(&self.data[offset..offset + len])
    }

    /// Concatenate blocks from multiple posting lists with doc_id remapping
    /// This is O(num_blocks) instead of O(num_postings)
    pub fn concatenate_blocks(sources: &[(BlockPostingList, u32)]) -> io::Result<Self> {
        let mut skip_list = Vec::new();
        let mut data = Vec::new();
        let mut total_docs = 0u32;
        let mut max_tf = 0u32;

        for (source, doc_offset) in sources {
            max_tf = max_tf.max(source.max_tf);
            for block_idx in 0..source.num_blocks() {
                if let Some((base, last, src_offset, len, block_max_tf)) =
                    source.block_info(block_idx)
                {
                    let new_base = base + doc_offset;
                    let new_last = last + doc_offset;
                    let new_offset = data.len() as u64;

                    // Copy block data, but we need to adjust the first doc_id in the block
                    let block_bytes = &source.data[src_offset..src_offset + len];

                    // Fixed 8-byte prefix: count(u32) + first_doc(u32)
                    let count = u32::from_le_bytes(block_bytes[0..4].try_into().unwrap());
                    let first_doc = u32::from_le_bytes(block_bytes[4..8].try_into().unwrap());

                    // Write patched prefix + copy rest verbatim
                    data.write_u32::<LittleEndian>(count)?;
                    data.write_u32::<LittleEndian>(first_doc + doc_offset)?;
                    data.extend_from_slice(&block_bytes[8..]);

                    skip_list.push((new_base, new_last, new_offset, block_max_tf));
                    total_docs += count;
                }
            }
        }

        Ok(Self {
            skip_list,
            data: OwnedBytes::new(data),
            doc_count: total_docs,
            max_tf,
        })
    }

    /// Streaming merge: write blocks directly to output writer (bounded memory).
    ///
    /// Parses only footer + skip_list from each source (no data copy),
    /// streams block data with patched 8-byte prefixes directly to `writer`,
    /// then appends merged skip_list + footer.
    ///
    /// Memory per term: O(total_blocks × 16) for skip entries only.
    /// Block data flows source mmap → output writer without buffering.
    ///
    /// Returns `(doc_count, bytes_written)`.
    pub fn concatenate_streaming<W: Write>(
        sources: &[(&[u8], u32)], // (serialized_bytes, doc_offset)
        writer: &mut W,
    ) -> io::Result<(u32, usize)> {
        // Parse footer + skip_list from each source (no data copy)
        struct RawSource<'a> {
            skip_list: Vec<(u32, u32, u64, u32)>, // (base, last, offset, block_max_tf)
            data: &'a [u8],                       // slice of block data section
            max_tf: u32,
            doc_count: u32,
            doc_offset: u32,
        }

        let mut parsed: Vec<RawSource<'_>> = Vec::with_capacity(sources.len());
        for (raw, doc_offset) in sources {
            if raw.len() < 20 {
                continue;
            }
            let f = raw.len() - 20;
            let data_len = u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()) as usize;
            let skip_count = u32::from_le_bytes(raw[f + 8..f + 12].try_into().unwrap()) as usize;
            let doc_count = u32::from_le_bytes(raw[f + 12..f + 16].try_into().unwrap());
            let max_tf = u32::from_le_bytes(raw[f + 16..f + 20].try_into().unwrap());

            let mut skip_list = Vec::with_capacity(skip_count);
            let mut pos = data_len;
            for _ in 0..skip_count {
                let base = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
                let last = u32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap());
                let offset = u64::from_le_bytes(raw[pos + 8..pos + 16].try_into().unwrap());
                let block_max_tf = u32::from_le_bytes(raw[pos + 16..pos + 20].try_into().unwrap());
                skip_list.push((base, last, offset, block_max_tf));
                pos += 20;
            }
            parsed.push(RawSource {
                skip_list,
                data: &raw[..data_len],
                max_tf,
                doc_count,
                doc_offset: *doc_offset,
            });
        }

        let total_docs: u32 = parsed.iter().map(|s| s.doc_count).sum();
        let merged_max_tf: u32 = parsed.iter().map(|s| s.max_tf).max().unwrap_or(0);

        // Phase 1: Stream block data with patched first_doc directly to writer.
        // Accumulate merged skip entries (20 bytes each — bounded).
        let mut merged_skip: Vec<(u32, u32, u64, u32)> = Vec::new();
        let mut data_written = 0u64;
        let mut patch_buf = [0u8; 8]; // reusable 8-byte prefix buffer

        for src in &parsed {
            for (i, &(base, last, offset, block_max_tf)) in src.skip_list.iter().enumerate() {
                let start = offset as usize;
                let end = if i + 1 < src.skip_list.len() {
                    src.skip_list[i + 1].2 as usize
                } else {
                    src.data.len()
                };
                let block = &src.data[start..end];

                merged_skip.push((
                    base + src.doc_offset,
                    last + src.doc_offset,
                    data_written,
                    block_max_tf,
                ));

                // Write patched 8-byte prefix + rest of block verbatim
                patch_buf[0..4].copy_from_slice(&block[0..4]); // count unchanged
                let first_doc = u32::from_le_bytes(block[4..8].try_into().unwrap());
                patch_buf[4..8].copy_from_slice(&(first_doc + src.doc_offset).to_le_bytes());
                writer.write_all(&patch_buf)?;
                writer.write_all(&block[8..])?;

                data_written += block.len() as u64;
            }
        }

        // Phase 2: Write skip_list + footer
        for (base, last, offset, block_max_tf) in &merged_skip {
            writer.write_u32::<LittleEndian>(*base)?;
            writer.write_u32::<LittleEndian>(*last)?;
            writer.write_u64::<LittleEndian>(*offset)?;
            writer.write_u32::<LittleEndian>(*block_max_tf)?;
        }

        writer.write_u64::<LittleEndian>(data_written)?;
        writer.write_u32::<LittleEndian>(merged_skip.len() as u32)?;
        writer.write_u32::<LittleEndian>(total_docs)?;
        writer.write_u32::<LittleEndian>(merged_max_tf)?;

        let total_bytes = data_written as usize + merged_skip.len() * 20 + 20;
        Ok((total_docs, total_bytes))
    }

    /// Create an iterator with skip support
    pub fn iterator(&self) -> BlockPostingIterator<'_> {
        BlockPostingIterator::new(self)
    }

    /// Create an owned iterator that doesn't borrow self
    pub fn into_iterator(self) -> BlockPostingIterator<'static> {
        BlockPostingIterator::owned(self)
    }
}

/// Iterator over block posting list with skip support
/// Can be either borrowed or owned via Cow
///
/// Uses struct-of-arrays layout: separate Vec<u32> for doc_ids and term_freqs.
/// This is more cache-friendly for SIMD seek (contiguous doc_ids) and halves
/// memory vs the previous AoS + separate doc_ids approach.
pub struct BlockPostingIterator<'a> {
    block_list: std::borrow::Cow<'a, BlockPostingList>,
    current_block: usize,
    block_doc_ids: Vec<u32>,
    block_tfs: Vec<u32>,
    position_in_block: usize,
    exhausted: bool,
}

/// Type alias for owned iterator
#[allow(dead_code)]
pub type OwnedBlockPostingIterator = BlockPostingIterator<'static>;

impl<'a> BlockPostingIterator<'a> {
    fn new(block_list: &'a BlockPostingList) -> Self {
        let exhausted = block_list.skip_list.is_empty();
        let mut iter = Self {
            block_list: std::borrow::Cow::Borrowed(block_list),
            current_block: 0,
            block_doc_ids: Vec::with_capacity(BLOCK_SIZE),
            block_tfs: Vec::with_capacity(BLOCK_SIZE),
            position_in_block: 0,
            exhausted,
        };
        if !iter.exhausted {
            iter.load_block(0);
        }
        iter
    }

    fn owned(block_list: BlockPostingList) -> BlockPostingIterator<'static> {
        let exhausted = block_list.skip_list.is_empty();
        let mut iter = BlockPostingIterator {
            block_list: std::borrow::Cow::Owned(block_list),
            current_block: 0,
            block_doc_ids: Vec::with_capacity(BLOCK_SIZE),
            block_tfs: Vec::with_capacity(BLOCK_SIZE),
            position_in_block: 0,
            exhausted,
        };
        if !iter.exhausted {
            iter.load_block(0);
        }
        iter
    }

    fn load_block(&mut self, block_idx: usize) {
        if block_idx >= self.block_list.skip_list.len() {
            self.exhausted = true;
            return;
        }

        self.current_block = block_idx;
        self.position_in_block = 0;

        let offset = self.block_list.skip_list[block_idx].2 as usize;
        let mut reader = &self.block_list.data[offset..];

        // Fixed 8-byte prefix: count(u32) + first_doc(u32)
        let count = reader.read_u32::<LittleEndian>().unwrap_or(0) as usize;
        let first_doc = reader.read_u32::<LittleEndian>().unwrap_or(0);
        self.block_doc_ids.clear();
        self.block_doc_ids.reserve(count);
        self.block_tfs.clear();
        self.block_tfs.reserve(count);

        let mut prev_doc_id = first_doc;

        for i in 0..count {
            if i == 0 {
                // First doc from fixed prefix, read only tf
                if let Ok(tf) = read_vint(&mut reader) {
                    self.block_doc_ids.push(first_doc);
                    self.block_tfs.push(tf as u32);
                }
            } else if let (Ok(delta), Ok(tf)) = (read_vint(&mut reader), read_vint(&mut reader)) {
                let doc_id = prev_doc_id + delta as u32;
                self.block_doc_ids.push(doc_id);
                self.block_tfs.push(tf as u32);
                prev_doc_id = doc_id;
            }
        }
    }

    pub fn doc(&self) -> DocId {
        if self.exhausted {
            TERMINATED
        } else if self.position_in_block < self.block_doc_ids.len() {
            self.block_doc_ids[self.position_in_block]
        } else {
            TERMINATED
        }
    }

    pub fn term_freq(&self) -> u32 {
        if self.exhausted || self.position_in_block >= self.block_tfs.len() {
            0
        } else {
            self.block_tfs[self.position_in_block]
        }
    }

    pub fn advance(&mut self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }

        self.position_in_block += 1;
        if self.position_in_block >= self.block_doc_ids.len() {
            self.load_block(self.current_block + 1);
        }
        self.doc()
    }

    pub fn seek(&mut self, target: DocId) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }

        // Binary search on skip_list: find first block whose last_doc >= target
        let block_idx = self
            .block_list
            .skip_list
            .partition_point(|(_, last_doc, _, _)| *last_doc < target);

        if block_idx >= self.block_list.skip_list.len() {
            self.exhausted = true;
            return TERMINATED;
        }

        if block_idx != self.current_block {
            self.load_block(block_idx);
        }

        // SIMD linear scan within block on cached doc_ids
        let remaining = &self.block_doc_ids[self.position_in_block..];
        let pos = crate::structures::simd::find_first_ge_u32(remaining, target);
        self.position_in_block += pos;

        if self.position_in_block >= self.block_doc_ids.len() {
            self.load_block(self.current_block + 1);
        }
        self.doc()
    }

    /// Skip to the next block, returning the first doc_id in the new block
    /// This is used for block-max WAND optimization when the current block's
    /// max score can't beat the threshold.
    pub fn skip_to_next_block(&mut self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }
        self.load_block(self.current_block + 1);
        self.doc()
    }

    /// Get the current block index
    #[inline]
    pub fn current_block_idx(&self) -> usize {
        self.current_block
    }

    /// Get total number of blocks
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.block_list.skip_list.len()
    }

    /// Get the current block's max term frequency for Block-Max WAND
    #[inline]
    pub fn current_block_max_tf(&self) -> u32 {
        if self.exhausted || self.current_block >= self.block_list.skip_list.len() {
            0
        } else {
            self.block_list.skip_list[self.current_block].3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posting_list_basic() {
        let mut list = PostingList::new();
        list.push(1, 2);
        list.push(5, 1);
        list.push(10, 3);

        assert_eq!(list.len(), 3);

        let mut iter = PostingListIterator::new(&list);
        assert_eq!(iter.doc(), 1);
        assert_eq!(iter.term_freq(), 2);

        assert_eq!(iter.advance(), 5);
        assert_eq!(iter.term_freq(), 1);

        assert_eq!(iter.advance(), 10);
        assert_eq!(iter.term_freq(), 3);

        assert_eq!(iter.advance(), TERMINATED);
    }

    #[test]
    fn test_posting_list_serialization() {
        let mut list = PostingList::new();
        for i in 0..100 {
            list.push(i * 3, (i % 5) + 1);
        }

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let deserialized = PostingList::deserialize(&mut &buffer[..]).unwrap();
        assert_eq!(deserialized.len(), list.len());

        for (a, b) in list.iter().zip(deserialized.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_posting_list_seek() {
        let mut list = PostingList::new();
        for i in 0..100 {
            list.push(i * 2, 1);
        }

        let mut iter = PostingListIterator::new(&list);

        assert_eq!(iter.seek(50), 50);
        assert_eq!(iter.seek(51), 52);
        assert_eq!(iter.seek(200), TERMINATED);
    }

    #[test]
    fn test_block_posting_list() {
        let mut list = PostingList::new();
        for i in 0..500 {
            list.push(i * 2, (i % 10) + 1);
        }

        let block_list = BlockPostingList::from_posting_list(&list).unwrap();
        assert_eq!(block_list.doc_count(), 500);

        let mut iter = block_list.iterator();
        assert_eq!(iter.doc(), 0);
        assert_eq!(iter.term_freq(), 1);

        // Test seek across blocks
        assert_eq!(iter.seek(500), 500);
        assert_eq!(iter.seek(998), 998);
        assert_eq!(iter.seek(1000), TERMINATED);
    }

    #[test]
    fn test_block_posting_list_serialization() {
        let mut list = PostingList::new();
        for i in 0..300 {
            list.push(i * 3, i + 1);
        }

        let block_list = BlockPostingList::from_posting_list(&list).unwrap();

        let mut buffer = Vec::new();
        block_list.serialize(&mut buffer).unwrap();

        let deserialized = BlockPostingList::deserialize(&buffer[..]).unwrap();
        assert_eq!(deserialized.doc_count(), block_list.doc_count());

        // Verify iteration produces same results
        let mut iter1 = block_list.iterator();
        let mut iter2 = deserialized.iterator();

        while iter1.doc() != TERMINATED {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
        assert_eq!(iter2.doc(), TERMINATED);
    }

    /// Helper: collect all (doc_id, tf) from a BlockPostingIterator
    fn collect_postings(bpl: &BlockPostingList) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        let mut it = bpl.iterator();
        while it.doc() != TERMINATED {
            result.push((it.doc(), it.term_freq()));
            it.advance();
        }
        result
    }

    /// Helper: build a BlockPostingList from (doc_id, tf) pairs
    fn build_bpl(postings: &[(u32, u32)]) -> BlockPostingList {
        let mut pl = PostingList::new();
        for &(doc_id, tf) in postings {
            pl.push(doc_id, tf);
        }
        BlockPostingList::from_posting_list(&pl).unwrap()
    }

    /// Helper: serialize a BlockPostingList to bytes
    fn serialize_bpl(bpl: &BlockPostingList) -> Vec<u8> {
        let mut buf = Vec::new();
        bpl.serialize(&mut buf).unwrap();
        buf
    }

    #[test]
    fn test_concatenate_blocks_two_segments() {
        // Segment A: docs 0,2,4,...,198 (100 docs, tf=1..100)
        let a: Vec<(u32, u32)> = (0..100).map(|i| (i * 2, i + 1)).collect();
        let bpl_a = build_bpl(&a);

        // Segment B: docs 0,3,6,...,297 (100 docs, tf=2..101)
        let b: Vec<(u32, u32)> = (0..100).map(|i| (i * 3, i + 2)).collect();
        let bpl_b = build_bpl(&b);

        // Merge: segment B starts at doc_offset=200
        let merged =
            BlockPostingList::concatenate_blocks(&[(bpl_a.clone(), 0), (bpl_b.clone(), 200)])
                .unwrap();

        assert_eq!(merged.doc_count(), 200);

        let postings = collect_postings(&merged);
        assert_eq!(postings.len(), 200);

        // First 100 from A (unchanged)
        for (i, p) in postings.iter().enumerate().take(100) {
            assert_eq!(*p, (i as u32 * 2, i as u32 + 1));
        }
        // Next 100 from B (doc_id += 200)
        for i in 0..100 {
            assert_eq!(postings[100 + i], (i as u32 * 3 + 200, i as u32 + 2));
        }
    }

    #[test]
    fn test_concatenate_streaming_matches_blocks() {
        // Build 3 segments with different doc distributions
        let seg_a: Vec<(u32, u32)> = (0..250).map(|i| (i * 2, (i % 7) + 1)).collect();
        let seg_b: Vec<(u32, u32)> = (0..180).map(|i| (i * 5, (i % 3) + 1)).collect();
        let seg_c: Vec<(u32, u32)> = (0..90).map(|i| (i * 10, (i % 11) + 1)).collect();

        let bpl_a = build_bpl(&seg_a);
        let bpl_b = build_bpl(&seg_b);
        let bpl_c = build_bpl(&seg_c);

        let offset_b = 1000u32;
        let offset_c = 2000u32;

        // Method 1: concatenate_blocks (in-memory reference)
        let ref_merged = BlockPostingList::concatenate_blocks(&[
            (bpl_a.clone(), 0),
            (bpl_b.clone(), offset_b),
            (bpl_c.clone(), offset_c),
        ])
        .unwrap();
        let mut ref_buf = Vec::new();
        ref_merged.serialize(&mut ref_buf).unwrap();

        // Method 2: concatenate_streaming (footer-based, writes to output)
        let bytes_a = serialize_bpl(&bpl_a);
        let bytes_b = serialize_bpl(&bpl_b);
        let bytes_c = serialize_bpl(&bpl_c);

        let sources: Vec<(&[u8], u32)> =
            vec![(&bytes_a, 0), (&bytes_b, offset_b), (&bytes_c, offset_c)];
        let mut stream_buf = Vec::new();
        let (doc_count, bytes_written) =
            BlockPostingList::concatenate_streaming(&sources, &mut stream_buf).unwrap();

        assert_eq!(doc_count, 520); // 250 + 180 + 90
        assert_eq!(bytes_written, stream_buf.len());

        // Deserialize both and verify identical postings
        let ref_postings = collect_postings(&BlockPostingList::deserialize(&ref_buf).unwrap());
        let stream_postings =
            collect_postings(&BlockPostingList::deserialize(&stream_buf).unwrap());

        assert_eq!(ref_postings.len(), stream_postings.len());
        for (i, (r, s)) in ref_postings.iter().zip(stream_postings.iter()).enumerate() {
            assert_eq!(r, s, "mismatch at posting {}", i);
        }
    }

    #[test]
    fn test_multi_round_merge() {
        // Simulate 3 rounds of merging (like tiered merge policy)
        //
        // Round 0: 4 small segments built independently
        // Round 1: merge pairs → 2 medium segments
        // Round 2: merge those → 1 large segment

        let segments: Vec<Vec<(u32, u32)>> = (0..4)
            .map(|seg| (0..200).map(|i| (i * 3, (i + seg * 7) % 10 + 1)).collect())
            .collect();

        let bpls: Vec<BlockPostingList> = segments.iter().map(|s| build_bpl(s)).collect();
        let serialized: Vec<Vec<u8>> = bpls.iter().map(serialize_bpl).collect();

        // Round 1: merge seg0+seg1 (offset=0,600), seg2+seg3 (offset=0,600)
        let mut merged_01 = Vec::new();
        let sources_01: Vec<(&[u8], u32)> = vec![(&serialized[0], 0), (&serialized[1], 600)];
        let (dc_01, _) =
            BlockPostingList::concatenate_streaming(&sources_01, &mut merged_01).unwrap();
        assert_eq!(dc_01, 400);

        let mut merged_23 = Vec::new();
        let sources_23: Vec<(&[u8], u32)> = vec![(&serialized[2], 0), (&serialized[3], 600)];
        let (dc_23, _) =
            BlockPostingList::concatenate_streaming(&sources_23, &mut merged_23).unwrap();
        assert_eq!(dc_23, 400);

        // Round 2: merge the two intermediate results (offset=0, 1200)
        let mut final_merged = Vec::new();
        let sources_final: Vec<(&[u8], u32)> = vec![(&merged_01, 0), (&merged_23, 1200)];
        let (dc_final, _) =
            BlockPostingList::concatenate_streaming(&sources_final, &mut final_merged).unwrap();
        assert_eq!(dc_final, 800);

        // Verify final result has all 800 postings with correct doc_ids
        let final_bpl = BlockPostingList::deserialize(&final_merged).unwrap();
        let postings = collect_postings(&final_bpl);
        assert_eq!(postings.len(), 800);

        // Verify doc_id ordering (must be monotonically non-decreasing within segments,
        // and segment boundaries at 0, 600, 1200, 1800)
        // Seg0: 0..597, Seg1: 600..1197, Seg2: 1200..1797, Seg3: 1800..2397
        assert_eq!(postings[0].0, 0); // first doc of seg0
        assert_eq!(postings[199].0, 597); // last doc of seg0 (199*3)
        assert_eq!(postings[200].0, 600); // first doc of seg1 (0+600)
        assert_eq!(postings[399].0, 1197); // last doc of seg1 (597+600)
        assert_eq!(postings[400].0, 1200); // first doc of seg2
        assert_eq!(postings[799].0, 2397); // last doc of seg3

        // Verify TFs preserved through two rounds of merging
        // Creation formula: tf = (i + seg * 7) % 10 + 1
        for seg in 0u32..4 {
            for i in 0u32..200 {
                let idx = (seg * 200 + i) as usize;
                assert_eq!(
                    postings[idx].1,
                    (i + seg * 7) % 10 + 1,
                    "seg{} tf[{}]",
                    seg,
                    i
                );
            }
        }

        // Verify seek works on final merged result
        let mut it = final_bpl.iterator();
        assert_eq!(it.seek(600), 600);
        assert_eq!(it.seek(1200), 1200);
        assert_eq!(it.seek(2397), 2397);
        assert_eq!(it.seek(2398), TERMINATED);
    }

    #[test]
    fn test_large_scale_merge() {
        // 5 segments × 2000 docs each = 10,000 total docs
        // Each segment has 16 blocks (2000/128 = 15.6 → 16 blocks)
        let num_segments = 5;
        let docs_per_segment = 2000;
        let docs_gap = 3; // doc_ids: 0, 3, 6, ...

        let segments: Vec<Vec<(u32, u32)>> = (0..num_segments)
            .map(|seg| {
                (0..docs_per_segment)
                    .map(|i| (i as u32 * docs_gap, (i as u32 + seg as u32) % 20 + 1))
                    .collect()
            })
            .collect();

        let bpls: Vec<BlockPostingList> = segments.iter().map(|s| build_bpl(s)).collect();

        // Verify each segment has multiple blocks
        for bpl in &bpls {
            assert!(
                bpl.num_blocks() >= 15,
                "expected >=15 blocks, got {}",
                bpl.num_blocks()
            );
        }

        let serialized: Vec<Vec<u8>> = bpls.iter().map(serialize_bpl).collect();

        // Compute offsets: each segment occupies max_doc+1 doc_id space
        let max_doc_per_seg = (docs_per_segment as u32 - 1) * docs_gap;
        let offsets: Vec<u32> = (0..num_segments)
            .map(|i| i as u32 * (max_doc_per_seg + 1))
            .collect();

        let sources: Vec<(&[u8], u32)> = serialized
            .iter()
            .zip(offsets.iter())
            .map(|(b, o)| (b.as_slice(), *o))
            .collect();

        let mut merged = Vec::new();
        let (doc_count, _) =
            BlockPostingList::concatenate_streaming(&sources, &mut merged).unwrap();
        assert_eq!(doc_count, (num_segments * docs_per_segment) as u32);

        // Deserialize and verify
        let merged_bpl = BlockPostingList::deserialize(&merged).unwrap();
        let postings = collect_postings(&merged_bpl);
        assert_eq!(postings.len(), num_segments * docs_per_segment);

        // Verify all doc_ids are strictly monotonically increasing across segment boundaries
        for i in 1..postings.len() {
            assert!(
                postings[i].0 > postings[i - 1].0 || (i % docs_per_segment == 0), // new segment can have lower absolute ID
                "doc_id not increasing at {}: {} vs {}",
                i,
                postings[i - 1].0,
                postings[i].0,
            );
        }

        // Verify seek across all block boundaries
        let mut it = merged_bpl.iterator();
        for (seg, &expected_first) in offsets.iter().enumerate() {
            assert_eq!(
                it.seek(expected_first),
                expected_first,
                "seek to segment {} start",
                seg
            );
        }
    }

    #[test]
    fn test_merge_edge_cases() {
        // Single doc per segment
        let bpl_a = build_bpl(&[(0, 5)]);
        let bpl_b = build_bpl(&[(0, 3)]);

        let merged =
            BlockPostingList::concatenate_blocks(&[(bpl_a.clone(), 0), (bpl_b.clone(), 1)])
                .unwrap();
        assert_eq!(merged.doc_count(), 2);
        let p = collect_postings(&merged);
        assert_eq!(p, vec![(0, 5), (1, 3)]);

        // Exactly BLOCK_SIZE docs (single full block)
        let exact_block: Vec<(u32, u32)> = (0..BLOCK_SIZE as u32).map(|i| (i, i % 5 + 1)).collect();
        let bpl_exact = build_bpl(&exact_block);
        assert_eq!(bpl_exact.num_blocks(), 1);

        let bytes = serialize_bpl(&bpl_exact);
        let mut out = Vec::new();
        let sources: Vec<(&[u8], u32)> = vec![(&bytes, 0), (&bytes, BLOCK_SIZE as u32)];
        let (dc, _) = BlockPostingList::concatenate_streaming(&sources, &mut out).unwrap();
        assert_eq!(dc, BLOCK_SIZE as u32 * 2);

        let merged = BlockPostingList::deserialize(&out).unwrap();
        let postings = collect_postings(&merged);
        assert_eq!(postings.len(), BLOCK_SIZE * 2);
        // Second segment's docs offset by BLOCK_SIZE
        assert_eq!(postings[BLOCK_SIZE].0, BLOCK_SIZE as u32);

        // BLOCK_SIZE + 1 docs (two blocks: 128 + 1)
        let over_block: Vec<(u32, u32)> = (0..BLOCK_SIZE as u32 + 1).map(|i| (i * 2, 1)).collect();
        let bpl_over = build_bpl(&over_block);
        assert_eq!(bpl_over.num_blocks(), 2);
    }

    #[test]
    fn test_streaming_roundtrip_single_source() {
        // Streaming merge with a single source should produce equivalent output to serialize
        let docs: Vec<(u32, u32)> = (0..500).map(|i| (i * 7, i % 15 + 1)).collect();
        let bpl = build_bpl(&docs);
        let direct = serialize_bpl(&bpl);

        let sources: Vec<(&[u8], u32)> = vec![(&direct, 0)];
        let mut streamed = Vec::new();
        BlockPostingList::concatenate_streaming(&sources, &mut streamed).unwrap();

        // Both should deserialize to identical postings
        let p1 = collect_postings(&BlockPostingList::deserialize(&direct).unwrap());
        let p2 = collect_postings(&BlockPostingList::deserialize(&streamed).unwrap());
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_max_tf_preserved_through_merge() {
        // Segment A: max_tf = 50
        let mut a = Vec::new();
        for i in 0..200 {
            a.push((i * 2, if i == 100 { 50 } else { 1 }));
        }
        let bpl_a = build_bpl(&a);
        assert_eq!(bpl_a.max_tf(), 50);

        // Segment B: max_tf = 30
        let mut b = Vec::new();
        for i in 0..200 {
            b.push((i * 2, if i == 50 { 30 } else { 2 }));
        }
        let bpl_b = build_bpl(&b);
        assert_eq!(bpl_b.max_tf(), 30);

        // After merge, max_tf should be max(50, 30) = 50
        let bytes_a = serialize_bpl(&bpl_a);
        let bytes_b = serialize_bpl(&bpl_b);
        let sources: Vec<(&[u8], u32)> = vec![(&bytes_a, 0), (&bytes_b, 1000)];
        let mut out = Vec::new();
        BlockPostingList::concatenate_streaming(&sources, &mut out).unwrap();

        let merged = BlockPostingList::deserialize(&out).unwrap();
        assert_eq!(merged.max_tf(), 50);
        assert_eq!(merged.doc_count(), 400);
    }
}
