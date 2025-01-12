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
    /// Skip list: (last_doc_id_in_block, byte_offset)
    skip_list: Vec<(DocId, u32)>,
    /// Compressed posting data
    data: Vec<u8>,
    /// Total number of postings
    doc_count: u32,
}

impl BlockPostingList {
    /// Build from a posting list
    pub fn from_posting_list(list: &PostingList) -> io::Result<Self> {
        let mut skip_list = Vec::new();
        let mut data = Vec::new();

        let postings = &list.postings;
        let mut i = 0;

        while i < postings.len() {
            let block_start = data.len() as u32;
            let block_end = (i + BLOCK_SIZE).min(postings.len());
            let block = &postings[i..block_end];

            // Record skip entry
            let last_doc_id = block.last().unwrap().doc_id;
            skip_list.push((last_doc_id, block_start));

            // Write block
            let mut prev_doc_id = if i == 0 { 0 } else { postings[i - 1].doc_id };
            write_vint(&mut data, block.len() as u64)?;

            for posting in block {
                let delta = posting.doc_id - prev_doc_id;
                write_vint(&mut data, delta as u64)?;
                write_vint(&mut data, posting.term_freq as u64)?;
                prev_doc_id = posting.doc_id;
            }

            i = block_end;
        }

        Ok(Self {
            skip_list,
            data,
            doc_count: postings.len() as u32,
        })
    }

    /// Serialize the block posting list
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Write doc count
        writer.write_u32::<LittleEndian>(self.doc_count)?;

        // Write skip list
        writer.write_u32::<LittleEndian>(self.skip_list.len() as u32)?;
        for (doc_id, offset) in &self.skip_list {
            writer.write_u32::<LittleEndian>(*doc_id)?;
            writer.write_u32::<LittleEndian>(*offset)?;
        }

        // Write data
        writer.write_u32::<LittleEndian>(self.data.len() as u32)?;
        writer.write_all(&self.data)?;

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_count = reader.read_u32::<LittleEndian>()?;

        let skip_count = reader.read_u32::<LittleEndian>()? as usize;
        let mut skip_list = Vec::with_capacity(skip_count);
        for _ in 0..skip_count {
            let doc_id = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u32::<LittleEndian>()?;
            skip_list.push((doc_id, offset));
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

    pub fn doc_count(&self) -> u32 {
        self.doc_count
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

        let offset = self.block_list.skip_list[block_idx].1 as usize;
        let mut reader = &self.block_list.data[offset..];

        let count = read_vint(&mut reader).unwrap_or(0) as usize;
        self.block_postings.clear();
        self.block_postings.reserve(count);

        let mut prev_doc_id = if block_idx == 0 {
            0
        } else {
            self.block_list.skip_list[block_idx - 1].0
        };

        for _ in 0..count {
            if let (Ok(delta), Ok(tf)) = (read_vint(&mut reader), read_vint(&mut reader)) {
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
            .position(|(last_doc, _)| *last_doc >= target);

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
