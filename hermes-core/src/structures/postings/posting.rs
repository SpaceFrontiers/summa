//! Posting list implementation with compact representation
//!
//! Uses delta encoding and variable-length integers for compact storage.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

use crate::DocId;

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

    /// Seek to first doc_id >= target
    pub fn seek(&mut self, target: DocId) -> DocId {
        // Binary search for efficiency
        while self.position < self.postings.len() {
            if self.postings[self.position].doc_id >= target {
                return self.postings[self.position].doc_id;
            }
            self.position += 1;
        }
        TERMINATED
    }

    /// Size hint for remaining elements
    pub fn size_hint(&self) -> usize {
        self.postings.len().saturating_sub(self.position)
    }
}

/// Sentinel value indicating iterator is exhausted
pub const TERMINATED: DocId = DocId::MAX;

/// Write variable-length integer (1-9 bytes)
fn write_vint<W: Write>(writer: &mut W, mut value: u64) -> io::Result<()> {
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
fn read_vint<R: Read>(reader: &mut R) -> io::Result<u64> {
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

/// Compact posting list stored as raw bytes (for memory-mapped access)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CompactPostingList {
    data: Vec<u8>,
    doc_count: u32,
}

#[allow(dead_code)]
impl CompactPostingList {
    /// Create from a posting list
    pub fn from_posting_list(list: &PostingList) -> io::Result<Self> {
        let mut data = Vec::new();
        list.serialize(&mut data)?;
        Ok(Self {
            doc_count: list.len() as u32,
            data,
        })
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Number of documents in the posting list
    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Deserialize back to PostingList
    pub fn to_posting_list(&self) -> io::Result<PostingList> {
        PostingList::deserialize(&mut &self.data[..])
    }
}

/// Block-based posting list for skip-list style access
/// Each block contains up to BLOCK_SIZE postings
pub const BLOCK_SIZE: usize = 128;

#[derive(Debug, Clone)]
pub struct BlockPostingList {
    /// Skip list: (base_doc_id, last_doc_id_in_block, byte_offset, block_max_tf)
    /// base_doc_id is the first doc_id in the block (absolute, not delta)
    /// block_max_tf enables Block-Max WAND optimization
    skip_list: Vec<(DocId, DocId, u32, u32)>,
    /// Compressed posting data
    data: Vec<u8>,
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
            let block_start = data.len() as u32;
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
            data,
            doc_count: postings.len() as u32,
            max_tf,
        })
    }

    /// Serialize the block posting list
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write doc count and max_tf
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;

        // Write skip list (base_doc_id, last_doc_id, offset, block_max_tf)
        writer.write_u32::<LittleEndian>(self.skip_list.len() as u32)?;
        for (base_doc_id, last_doc_id, offset, block_max_tf) in &self.skip_list {
            writer.write_u32::<LittleEndian>(*base_doc_id)?;
            writer.write_u32::<LittleEndian>(*last_doc_id)?;
            writer.write_u32::<LittleEndian>(*offset)?;
            writer.write_u32::<LittleEndian>(*block_max_tf)?;
        }

        // Write data
        writer.write_u32::<LittleEndian>(self.data.len() as u32)?;
        writer.write_all(&self.data)?;

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_count = reader.read_u32::<LittleEndian>()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;

        let skip_count = reader.read_u32::<LittleEndian>()? as usize;
        let mut skip_list = Vec::with_capacity(skip_count);
        for _ in 0..skip_count {
            let base_doc_id = reader.read_u32::<LittleEndian>()?;
            let last_doc_id = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u32::<LittleEndian>()?;
            let block_max_tf = reader.read_u32::<LittleEndian>()?;
            skip_list.push((base_doc_id, last_doc_id, offset, block_max_tf));
        }

        let data_len = reader.read_u32::<LittleEndian>()? as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

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
                    let new_offset = data.len() as u32;

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
            data,
            doc_count: total_docs,
            max_tf,
        })
    }

    /// Concatenate posting lists directly from serialized bytes (zero intermediate allocation).
    ///
    /// Parses only the header + skip_list from each source's raw bytes (no data copy),
    /// then writes the merged posting list with patched 8-byte block prefixes directly
    /// to `out`. Returns `(doc_count, max_tf)`.
    ///
    /// Avoids: deserialize (data Vec alloc per source), concatenate_blocks (second data Vec),
    /// serialize (third copy to buf). Instead: one output write per block.
    pub fn concatenate_from_raw(
        sources: &[(&[u8], u32)], // (serialized_bytes, doc_offset)
        out: &mut Vec<u8>,
    ) -> io::Result<(u32, u32)> {
        // Parse headers + skip_lists from raw bytes (no data section copy)
        struct RawSource<'a> {
            skip_list: Vec<(u32, u32, u32, u32)>, // (base, last, offset, block_max_tf)
            data: &'a [u8],                       // slice into raw bytes (data section only)
            max_tf: u32,
            doc_count: u32,
            doc_offset: u32,
        }

        let mut parsed: Vec<RawSource<'_>> = Vec::with_capacity(sources.len());
        for (raw, doc_offset) in sources {
            let mut r = &raw[..];
            let doc_count = r.read_u32::<LittleEndian>()?;
            let max_tf = r.read_u32::<LittleEndian>()?;
            let skip_count = r.read_u32::<LittleEndian>()? as usize;
            let mut skip_list = Vec::with_capacity(skip_count);
            for _ in 0..skip_count {
                let base = r.read_u32::<LittleEndian>()?;
                let last = r.read_u32::<LittleEndian>()?;
                let offset = r.read_u32::<LittleEndian>()?;
                let block_max_tf = r.read_u32::<LittleEndian>()?;
                skip_list.push((base, last, offset, block_max_tf));
            }
            let data_len = r.read_u32::<LittleEndian>()? as usize;
            // r now points to start of data section
            let header_size = raw.len() - r.len();
            let data = &raw[header_size..header_size + data_len];
            parsed.push(RawSource {
                skip_list,
                data,
                max_tf,
                doc_count,
                doc_offset: *doc_offset,
            });
        }

        // Compute merged header values
        let total_docs: u32 = parsed.iter().map(|s| s.doc_count).sum();
        let merged_max_tf: u32 = parsed.iter().map(|s| s.max_tf).max().unwrap_or(0);
        let total_blocks: usize = parsed.iter().map(|s| s.skip_list.len()).sum();

        // Reserve output: header + skip_list + estimated data
        let data_size: usize = parsed.iter().map(|s| s.data.len()).sum();
        out.reserve(12 + total_blocks * 16 + 4 + data_size);

        // Write header
        out.write_u32::<LittleEndian>(total_docs)?;
        out.write_u32::<LittleEndian>(merged_max_tf)?;
        out.write_u32::<LittleEndian>(total_blocks as u32)?;

        // Build merged skip_list with remapped offsets, write it
        let mut data_base_offset = 0u32;
        let skip_list_start = out.len();
        // Placeholder for skip_list (will be filled in)
        out.resize(skip_list_start + total_blocks * 16, 0);

        let mut skip_idx = 0;
        for src in &parsed {
            let src_data_len = src.data.len() as u32;
            for &(base, last, offset, block_max_tf) in &src.skip_list {
                let pos = skip_list_start + skip_idx * 16;
                out[pos..pos + 4].copy_from_slice(&(base + src.doc_offset).to_le_bytes());
                out[pos + 4..pos + 8].copy_from_slice(&(last + src.doc_offset).to_le_bytes());
                out[pos + 8..pos + 12].copy_from_slice(&(data_base_offset + offset).to_le_bytes());
                out[pos + 12..pos + 16].copy_from_slice(&block_max_tf.to_le_bytes());
                skip_idx += 1;
            }
            data_base_offset += src_data_len;
        }

        // Write data_len
        out.write_u32::<LittleEndian>(data_base_offset)?;

        // Write block data: for each source, copy blocks with patched first_doc
        for src in &parsed {
            for (i, &(_, _, offset, _)) in src.skip_list.iter().enumerate() {
                let start = offset as usize;
                let end = if i + 1 < src.skip_list.len() {
                    src.skip_list[i + 1].2 as usize
                } else {
                    src.data.len()
                };
                let block = &src.data[start..end];

                // Patch 8-byte prefix: count stays, first_doc += offset
                out.extend_from_slice(&block[0..4]); // count (unchanged)
                let first_doc = u32::from_le_bytes(block[4..8].try_into().unwrap());
                out.extend_from_slice(&(first_doc + src.doc_offset).to_le_bytes());
                out.extend_from_slice(&block[8..]); // rest verbatim
            }
        }

        Ok((total_docs, merged_max_tf))
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
pub struct BlockPostingIterator<'a> {
    block_list: std::borrow::Cow<'a, BlockPostingList>,
    current_block: usize,
    block_postings: Vec<Posting>,
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
            block_postings: Vec::new(),
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
        self.block_postings.clear();
        self.block_postings.reserve(count);

        let mut prev_doc_id = first_doc;

        for i in 0..count {
            if i == 0 {
                // First doc from fixed prefix, read only tf
                if let Ok(tf) = read_vint(&mut reader) {
                    self.block_postings.push(Posting {
                        doc_id: first_doc,
                        term_freq: tf as u32,
                    });
                }
            } else if let (Ok(delta), Ok(tf)) = (read_vint(&mut reader), read_vint(&mut reader)) {
                let doc_id = prev_doc_id + delta as u32;
                self.block_postings.push(Posting {
                    doc_id,
                    term_freq: tf as u32,
                });
                prev_doc_id = doc_id;
            }
        }
    }

    pub fn doc(&self) -> DocId {
        if self.exhausted {
            TERMINATED
        } else if self.position_in_block < self.block_postings.len() {
            self.block_postings[self.position_in_block].doc_id
        } else {
            TERMINATED
        }
    }

    pub fn term_freq(&self) -> u32 {
        if self.exhausted || self.position_in_block >= self.block_postings.len() {
            0
        } else {
            self.block_postings[self.position_in_block].term_freq
        }
    }

    pub fn advance(&mut self) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }

        self.position_in_block += 1;
        if self.position_in_block >= self.block_postings.len() {
            self.load_block(self.current_block + 1);
        }
        self.doc()
    }

    pub fn seek(&mut self, target: DocId) -> DocId {
        if self.exhausted {
            return TERMINATED;
        }

        let target_block = self
            .block_list
            .skip_list
            .iter()
            .position(|(_, last_doc, _, _)| *last_doc >= target);

        if let Some(block_idx) = target_block {
            if block_idx != self.current_block {
                self.load_block(block_idx);
            }

            while self.position_in_block < self.block_postings.len() {
                if self.block_postings[self.position_in_block].doc_id >= target {
                    return self.doc();
                }
                self.position_in_block += 1;
            }

            self.load_block(self.current_block + 1);
            self.seek(target)
        } else {
            self.exhausted = true;
            TERMINATED
        }
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

        let deserialized = BlockPostingList::deserialize(&mut &buffer[..]).unwrap();
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
}
