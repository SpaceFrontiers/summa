//! Posting list implementation with compact representation
//!
//! Text blocks use SIMD-friendly packed bit-width encoding:
//! - Doc IDs: delta-encoded, packed at rounded bit width (0/8/16/32)
//! - Term frequencies: packed at rounded bit width
//! - Same SIMD primitives as sparse blocks (`simd::pack_rounded` / `unpack_rounded`)

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{self, Read, Write};

use super::posting_common::{read_vint, write_vint};
use crate::DocId;
use crate::directories::OwnedBytes;
use crate::structures::simd;

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

/// Block-based posting list with 2-level skip index.
///
/// Each block contains up to `BLOCK_SIZE` postings encoded as packed bit-width arrays.
/// Skip entries use a compact 2-level structure for cache-friendly seeking:
/// - **Level-0** (16 bytes/block): `first_doc`, `last_doc`, `offset`, `max_weight`
/// - **Level-1** (4 bytes/group): `last_doc` per `L1_INTERVAL` blocks
///
/// Seek algorithm: binary search L1, then linear scan ≤`L1_INTERVAL` L0 entries.
pub const BLOCK_SIZE: usize = 128;

/// Number of L0 blocks per L1 skip entry.
const L1_INTERVAL: usize = 8;

/// Compact level-0 skip entry — 16 bytes.
/// `length` is omitted: computable from the block's 8-byte header.
const L0_SIZE: usize = 16;

/// Level-1 skip entry — 4 bytes (just `last_doc`).
const L1_SIZE: usize = 4;

/// Legacy footer: stream_len(8) + l0_count(4) + l1_count(4) + doc_count(4) + max_tf(4) = 24 bytes.
const FOOTER_SIZE: usize = 24;

/// Current footer: the legacy footer followed by `total_positions(8) +
/// flags(4) + min_len(4) + magic(4)`. A list ends with the magic iff it has
/// the extended footer; a legacy footer ends with `max_tf`, which the u16
/// term frequency of the builder keeps far below the magic, so both forms
/// remain readable.
const FOOTER_V2_SIZE: usize = FOOTER_SIZE + 20;

/// "BPL2" little-endian.
const FOOTER_MAGIC: u32 = 0x324C_5042;

/// Footer flag: a `u64` position cursor per L0 block follows the L1 entries.
const FLAG_POS_CURSORS: u32 = 1;

/// Footer flag: the fourth L0 word packs `max_tf` (low 16 bits) and the
/// block's minimum scoring-unit length (high 16 bits) instead of an `f32`
/// max tf, so a block bound can use real length normalisation.
const FLAG_LEN_BOUNDS: u32 = 2;

/// Footer flag: a packed `(max_tf, min_len)` word per L1 group follows the
/// L1 `last_doc` entries (superblock bounds: the maximum and minimum over
/// the group's blocks), so an executor can skip eight blocks at once.
const FLAG_L1_BOUNDS: u32 = 4;

/// Superblock bounds derived from packed L0 words: per `L1_INTERVAL` group
/// the maximum `max_tf` and minimum `min_len` of its blocks.
fn group_bounds_from_l0(l0: &[u8], l0_count: usize) -> Vec<u32> {
    let mut groups = Vec::with_capacity(l0_count.div_ceil(L1_INTERVAL));
    let mut idx = 0;
    while idx < l0_count {
        let end = (idx + L1_INTERVAL).min(l0_count);
        let mut max_tf = 0u32;
        let mut min_len = u32::MAX;
        for block in idx..end {
            let (_, _, _, word) = read_l0(l0, block);
            let (tf, len) = unpack_bounds(word, true);
            max_tf = max_tf.max(tf);
            min_len = min_len.min(len.unwrap_or(1));
        }
        groups.push(pack_bounds(max_tf, min_len));
        idx = end;
    }
    groups
}

/// Pack block bounds into the fourth L0 word (both saturate at u16).
#[inline]
fn pack_bounds(max_tf: u32, min_len: u32) -> u32 {
    max_tf.min(u16::MAX as u32) | (min_len.min(u16::MAX as u32) << 16)
}

/// Unpack the fourth L0 word: `(max_tf, min_len)`; `min_len` is `None` for
/// legacy lists whose word is an `f32` max tf.
#[inline]
fn unpack_bounds(word: u32, packed: bool) -> (u32, Option<u32>) {
    if packed {
        (word & 0xFFFF, Some(word >> 16))
    } else {
        (f32::from_bits(word) as u32, None)
    }
}

/// Size of one position cursor (`u64`: values before the block in the
/// term's position stream).
const CURSOR_SIZE: usize = 8;

/// Parsed footer of either format plus the derived section layout.
struct Footer {
    stream_len: usize,
    l0_count: usize,
    l1_count: usize,
    doc_count: u32,
    max_tf: u32,
    total_positions: u64,
    has_cursors: bool,
    len_bounds: bool,
    l1_bounds: bool,
    min_len: u32,
}

impl Footer {
    fn parse(raw: &[u8]) -> io::Result<Self> {
        if raw.len() < FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting data too short",
            ));
        }
        let extended = raw.len() >= FOOTER_V2_SIZE
            && u32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) == FOOTER_MAGIC;
        let f = raw.len()
            - if extended {
                FOOTER_V2_SIZE
            } else {
                FOOTER_SIZE
            };
        let stream_len = u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()) as usize;
        let l0_count = u32::from_le_bytes(raw[f + 8..f + 12].try_into().unwrap()) as usize;
        let l1_count = u32::from_le_bytes(raw[f + 12..f + 16].try_into().unwrap()) as usize;
        let doc_count = u32::from_le_bytes(raw[f + 16..f + 20].try_into().unwrap());
        let max_tf = u32::from_le_bytes(raw[f + 20..f + 24].try_into().unwrap());
        let (total_positions, flags, min_len) = if extended {
            let total = u64::from_le_bytes(raw[f + 24..f + 32].try_into().unwrap());
            let flags = u32::from_le_bytes(raw[f + 32..f + 36].try_into().unwrap());
            let min_len = u32::from_le_bytes(raw[f + 36..f + 40].try_into().unwrap());
            (total, flags, min_len)
        } else {
            (0, 0, 0)
        };
        let footer = Self {
            stream_len,
            l0_count,
            l1_count,
            doc_count,
            max_tf,
            total_positions,
            has_cursors: flags & FLAG_POS_CURSORS != 0,
            len_bounds: flags & FLAG_LEN_BOUNDS != 0,
            l1_bounds: flags & FLAG_L1_BOUNDS != 0,
            min_len,
        };
        if footer.cursors_end() > f {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting list sections exceed the footer offset",
            ));
        }
        Ok(footer)
    }

    fn l0_start(&self) -> usize {
        self.stream_len
    }
    fn l0_end(&self) -> usize {
        self.l0_start() + self.l0_count * L0_SIZE
    }
    fn l1_end(&self) -> usize {
        self.l0_end() + self.l1_count * L1_SIZE
    }
    fn l1_bounds_end(&self) -> usize {
        self.l1_end() + if self.l1_bounds { self.l1_count * 4 } else { 0 }
    }
    fn cursors_end(&self) -> usize {
        self.l1_bounds_end()
            + if self.has_cursors {
                self.l0_count * CURSOR_SIZE
            } else {
                0
            }
    }
}

/// Read a compact L0 entry from raw bytes at the given index: `(first_doc,
/// last_doc, offset, bounds word)`. The bounds word is packed `(max_tf,
/// min_len)` for current lists and an `f32` max tf for legacy ones; see
/// [`unpack_bounds`].
///
/// Uses a single bounds check (`[..L0_SIZE]`) instead of 4× `try_into().unwrap()`.
#[inline]
fn read_l0(bytes: &[u8], idx: usize) -> (u32, u32, u32, u32) {
    let b = &bytes[idx * L0_SIZE..][..L0_SIZE];
    let first_doc = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let last_doc = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let offset = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
    let bounds = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
    (first_doc, last_doc, offset, bounds)
}

/// Write a compact L0 entry.
#[inline]
fn write_l0(buf: &mut Vec<u8>, first_doc: u32, last_doc: u32, offset: u32, bounds: u32) {
    buf.extend_from_slice(&first_doc.to_le_bytes());
    buf.extend_from_slice(&last_doc.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(&bounds.to_le_bytes());
}

/// Compute block data size from the 8-byte header at `stream[pos..]`.
///
/// Header: `[count: u16][first_doc: u32][doc_id_bits: u8][tf_bits: u8]`
/// Data size = 8 + (count-1) × bytes_per_value(doc_id_bits) + count × bytes_per_value(tf_bits)
#[inline]
fn block_data_size(stream: &[u8], pos: usize) -> usize {
    let count = u16::from_le_bytes(stream[pos..pos + 2].try_into().unwrap()) as usize;
    let doc_rounded = simd::RoundedBitWidth::from_u8(stream[pos + 6]);
    let tf_rounded = simd::RoundedBitWidth::from_u8(stream[pos + 7]);
    let delta_bytes = if count > 1 {
        (count - 1) * doc_rounded.bytes_per_value()
    } else {
        0
    };
    8 + delta_bytes + count * tf_rounded.bytes_per_value()
}

#[derive(Debug, Clone)]
pub struct BlockPostingList {
    /// Block data stream (packed blocks laid out sequentially).
    stream: OwnedBytes,
    /// Level-0 skip entries: `(first_doc, last_doc, offset, max_weight)` × `l0_count`.
    /// 16 bytes per entry. Supports O(1) random access by block index.
    l0_bytes: OwnedBytes,
    /// Number of blocks (= number of L0 entries).
    l0_count: usize,
    /// Level-1 skip `last_doc` values — one per `L1_INTERVAL` blocks.
    /// Stored as `Vec<u32>` for direct SIMD-accelerated `find_first_ge_u32`.
    l1_docs: Vec<u32>,
    /// Packed `(max_tf, min_len)` per L1 group (superblock bounds); empty
    /// for legacy lists.
    l1_bounds: Vec<u32>,
    /// Total posting count.
    doc_count: u32,
    /// Max TF across all blocks.
    max_tf: u32,
    /// Per-block position cursors (`u64` × `l0_count`): number of values in
    /// the term's position stream before the block. `None` for terms
    /// without positions and for legacy lists.
    pos_cursors: Option<OwnedBytes>,
    /// Sum of term frequencies (= values in the position stream) when
    /// cursors are present.
    total_positions: u64,
    /// Whether L0 bounds words are packed `(max_tf, min_len)`.
    len_bounds: bool,
    /// Minimum scoring-unit length over the whole list (with `len_bounds`).
    min_len: u32,
}

impl BlockPostingList {
    /// Read L0 entry by block index. Returns `(first_doc, last_doc, offset, bounds word)`.
    #[inline]
    fn read_l0_entry(&self, idx: usize) -> (u32, u32, u32, u32) {
        read_l0(&self.l0_bytes, idx)
    }

    /// Build from a posting list.
    ///
    /// Block format (8-byte header + packed arrays):
    /// ```text
    /// [count: u16][first_doc: u32][doc_id_bits: u8][tf_bits: u8]
    /// [packed doc_id deltas: (count-1) × bytes_per_value(doc_id_bits)]
    /// [packed tfs: count × bytes_per_value(tf_bits)]
    /// ```
    pub fn from_posting_list(list: &PostingList) -> io::Result<Self> {
        Self::build(list, false, None)
    }

    /// Like [`Self::from_posting_list`], for a term whose positions are
    /// stored as a v2 stream: every block records how many positions precede
    /// it (the cumulative term frequency), so a reader can address the
    /// stream from the doc postings alone.
    pub fn from_posting_list_with_positions(list: &PostingList) -> io::Result<Self> {
        Self::build(list, true, None)
    }

    /// Build with position cursors on demand and, when `length_of` is given,
    /// the minimum scoring-unit length per block (and over the list) so
    /// MaxScore bounds use real length normalisation. Without lengths the
    /// minimum is 1, which any real unit satisfies.
    pub fn from_posting_list_with(
        list: &PostingList,
        with_positions: bool,
        length_of: Option<&dyn Fn(DocId) -> u32>,
    ) -> io::Result<Self> {
        Self::build(list, with_positions, length_of)
    }

    fn build(
        list: &PostingList,
        with_positions: bool,
        length_of: Option<&dyn Fn(DocId) -> u32>,
    ) -> io::Result<Self> {
        let mut stream: Vec<u8> = Vec::new();
        let mut l0_buf: Vec<u8> = Vec::new();
        let mut l1_docs: Vec<u32> = Vec::new();
        let mut cursors: Vec<u8> = Vec::new();
        let mut positions_so_far = 0u64;
        let mut l0_count = 0usize;
        let mut max_tf = 0u32;
        let mut list_min_len = u32::MAX;

        let postings = &list.postings;
        let mut i = 0;

        // Temp buffers reused across blocks
        let mut deltas = Vec::with_capacity(BLOCK_SIZE);
        let mut tf_buf = Vec::with_capacity(BLOCK_SIZE);

        while i < postings.len() {
            if stream.len() > u32::MAX as usize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "posting list stream exceeds u32::MAX bytes",
                ));
            }
            let block_start = stream.len() as u32;
            let block_end = (i + BLOCK_SIZE).min(postings.len());
            let block = &postings[i..block_end];
            let count = block.len();

            // Compute block's max term frequency for block-max pruning
            let block_max_tf = block.iter().map(|p| p.term_freq).max().unwrap_or(0);
            max_tf = max_tf.max(block_max_tf);

            let base_doc_id = block.first().unwrap().doc_id;
            let last_doc_id = block.last().unwrap().doc_id;

            // Delta-encode doc IDs (skip first — stored in header)
            deltas.clear();
            let mut prev = base_doc_id;
            for posting in block.iter().skip(1) {
                deltas.push(posting.doc_id - prev);
                prev = posting.doc_id;
            }
            let max_delta = deltas.iter().copied().max().unwrap_or(0);
            let doc_id_bits = simd::round_bit_width(simd::bits_needed(max_delta));

            // Collect TFs
            tf_buf.clear();
            tf_buf.extend(block.iter().map(|p| p.term_freq));
            let tf_bits = simd::round_bit_width(simd::bits_needed(block_max_tf));

            // Write 8-byte header: [count: u16][first_doc: u32][doc_id_bits: u8][tf_bits: u8]
            stream.write_u16::<LittleEndian>(count as u16)?;
            stream.write_u32::<LittleEndian>(base_doc_id)?;
            stream.push(doc_id_bits);
            stream.push(tf_bits);

            // Write packed doc_id deltas ((count-1) values)
            if count > 1 {
                let rounded = simd::RoundedBitWidth::from_u8(doc_id_bits);
                let byte_count = (count - 1) * rounded.bytes_per_value();
                let start = stream.len();
                stream.resize(start + byte_count, 0);
                simd::pack_rounded(&deltas, rounded, &mut stream[start..]);
            }

            // Write packed TFs (count values)
            {
                let rounded = simd::RoundedBitWidth::from_u8(tf_bits);
                let byte_count = count * rounded.bytes_per_value();
                let start = stream.len();
                stream.resize(start + byte_count, 0);
                simd::pack_rounded(&tf_buf, rounded, &mut stream[start..]);
            }

            // L0 skip entry with the block's bounds
            let block_min_len = length_of.map_or(1, |length_of| {
                block
                    .iter()
                    .map(|p| length_of(p.doc_id).max(1))
                    .min()
                    .unwrap_or(1)
            });
            list_min_len = list_min_len.min(block_min_len);
            write_l0(
                &mut l0_buf,
                base_doc_id,
                last_doc_id,
                block_start,
                pack_bounds(block_max_tf, block_min_len),
            );
            l0_count += 1;
            if with_positions {
                cursors.extend_from_slice(&positions_so_far.to_le_bytes());
                positions_so_far += block.iter().map(|p| p.term_freq as u64).sum::<u64>();
            }

            // L1 entry at the end of each L1_INTERVAL group
            if l0_count.is_multiple_of(L1_INTERVAL) {
                l1_docs.push(last_doc_id);
            }

            i = block_end;
        }

        // Final L1 entry for partial group
        if !l0_count.is_multiple_of(L1_INTERVAL) && l0_count > 0 {
            let (_, last_doc, _, _) = read_l0(&l0_buf, l0_count - 1);
            l1_docs.push(last_doc);
        }
        let l1_bounds = group_bounds_from_l0(&l0_buf, l0_count);

        Ok(Self {
            stream: OwnedBytes::new(stream),
            l0_bytes: OwnedBytes::new(l0_buf),
            l0_count,
            l1_docs,
            l1_bounds,
            doc_count: postings.len() as u32,
            max_tf,
            pos_cursors: with_positions.then(|| OwnedBytes::new(cursors)),
            total_positions: positions_so_far,
            len_bounds: true,
            min_len: if list_min_len == u32::MAX {
                1
            } else {
                list_min_len
            },
        })
    }

    /// Serialize the block posting list (footer-based: stream first).
    ///
    /// Format:
    /// ```text
    /// [stream: block data]
    /// [L0 entries: l0_count × 16 bytes (first_doc, last_doc, offset, max_weight)]
    /// [L1 entries: l1_count × 4 bytes (last_doc)]
    /// [L1 bounds: l1_count × 4 bytes (packed max_tf, min_len), FLAG_L1_BOUNDS]
    /// [position cursors: l0_count × 8 bytes, only with positions]
    /// [footer: stream_len(8) + l0_count(4) + l1_count(4) + doc_count(4) + max_tf(4)
    ///          + total_positions(8) + flags(4) + min_len(4) + magic(4) = 44 bytes]
    /// ```
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.stream)?;
        writer.write_all(&self.l0_bytes)?;
        for &doc in &self.l1_docs {
            writer.write_u32::<LittleEndian>(doc)?;
        }
        for &bounds in &self.l1_bounds {
            writer.write_u32::<LittleEndian>(bounds)?;
        }
        if let Some(cursors) = &self.pos_cursors {
            writer.write_all(cursors)?;
        }
        Self::write_footer(
            writer,
            self.stream.len() as u64,
            self.l0_count,
            self.l1_docs.len(),
            self.doc_count,
            self.max_tf,
            self.total_positions,
            self.pos_cursors.is_some(),
            self.len_bounds.then_some(self.min_len),
            !self.l1_bounds.is_empty(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn write_footer<W: Write>(
        writer: &mut W,
        stream_len: u64,
        l0_count: usize,
        l1_count: usize,
        doc_count: u32,
        max_tf: u32,
        total_positions: u64,
        has_cursors: bool,
        min_len: Option<u32>,
        l1_bounds: bool,
    ) -> io::Result<()> {
        writer.write_u64::<LittleEndian>(stream_len)?;
        writer.write_u32::<LittleEndian>(l0_count as u32)?;
        writer.write_u32::<LittleEndian>(l1_count as u32)?;
        writer.write_u32::<LittleEndian>(doc_count)?;
        writer.write_u32::<LittleEndian>(max_tf)?;
        writer.write_u64::<LittleEndian>(total_positions)?;
        let mut flags = 0u32;
        if has_cursors {
            flags |= FLAG_POS_CURSORS;
        }
        if min_len.is_some() {
            flags |= FLAG_LEN_BOUNDS;
        }
        if l1_bounds {
            flags |= FLAG_L1_BOUNDS;
        }
        writer.write_u32::<LittleEndian>(flags)?;
        writer.write_u32::<LittleEndian>(min_len.unwrap_or(0))?;
        writer.write_u32::<LittleEndian>(FOOTER_MAGIC)?;
        Ok(())
    }

    /// Deserialize from a byte slice (either footer format).
    pub fn deserialize(raw: &[u8]) -> io::Result<Self> {
        Self::deserialize_zero_copy(OwnedBytes::new(raw.to_vec()))
    }

    /// Zero-copy deserialization from OwnedBytes.
    /// Stream, L0 and cursors are sliced from the source without copying.
    /// L1 is extracted into a `Vec<u32>` for SIMD-friendly access (tiny: ≤ N/8 entries).
    pub fn deserialize_zero_copy(raw: OwnedBytes) -> io::Result<Self> {
        let footer = Footer::parse(raw.as_slice())?;
        let l1_docs = Self::extract_l1_docs(&raw[footer.l0_end()..], footer.l1_count);
        let l1_bounds = if footer.l1_bounds {
            Self::extract_l1_docs(&raw[footer.l1_end()..], footer.l1_count)
        } else {
            Vec::new()
        };
        let pos_cursors = footer
            .has_cursors
            .then(|| raw.slice(footer.l1_bounds_end()..footer.cursors_end()));

        Ok(Self {
            stream: raw.slice(0..footer.stream_len),
            l0_bytes: raw.slice(footer.l0_start()..footer.l0_end()),
            l0_count: footer.l0_count,
            l1_docs,
            l1_bounds,
            doc_count: footer.doc_count,
            max_tf: footer.max_tf,
            pos_cursors,
            total_positions: footer.total_positions,
            len_bounds: footer.len_bounds,
            min_len: footer.min_len,
        })
    }

    /// Minimum scoring-unit length over the list, when the list stores
    /// length bounds (`None` for legacy lists).
    pub fn min_len(&self) -> Option<u32> {
        self.len_bounds.then_some(self.min_len)
    }

    /// `(max_tf, min_len)` of a block; `min_len` is `None` for legacy lists.
    #[inline]
    pub fn block_bounds(&self, block_idx: usize) -> Option<(u32, Option<u32>)> {
        if block_idx >= self.l0_count {
            return None;
        }
        let (_, _, _, word) = self.read_l0_entry(block_idx);
        Some(unpack_bounds(word, self.len_bounds))
    }

    /// `(max_tf, min_len)` over the L1 group (`L1_INTERVAL` blocks) that
    /// contains `block_idx`; `None` for legacy lists without group bounds.
    #[inline]
    pub fn group_bounds(&self, block_idx: usize) -> Option<(u32, u32)> {
        if block_idx >= self.l0_count {
            return None;
        }
        let word = *self.l1_bounds.get(block_idx / L1_INTERVAL)?;
        let (max_tf, min_len) = unpack_bounds(word, true);
        Some((max_tf, min_len.unwrap_or(1)))
    }

    /// Last doc of the L1 group containing `block_idx`.
    #[inline]
    pub fn group_last_doc(&self, block_idx: usize) -> Option<DocId> {
        self.l1_docs.get(block_idx / L1_INTERVAL).copied()
    }

    /// Whether `block_idx` opens an L1 group.
    #[inline]
    pub fn is_group_start(&self, block_idx: usize) -> bool {
        block_idx.is_multiple_of(L1_INTERVAL)
    }

    /// Index of the first block after the L1 group containing `block_idx`
    /// (clamped to the block count).
    #[inline]
    pub fn next_group_block(&self, block_idx: usize) -> usize {
        ((block_idx / L1_INTERVAL + 1) * L1_INTERVAL).min(self.l0_count)
    }

    /// Whether serialized bytes carry position cursors (cheap footer check).
    pub fn has_cursors_bytes(raw: &[u8]) -> bool {
        Footer::parse(raw).is_ok_and(|footer| footer.has_cursors)
    }

    /// Whether this list carries a position cursor per block.
    pub fn has_position_cursors(&self) -> bool {
        self.pos_cursors.is_some()
    }

    /// Number of values in the term's position stream (0 without cursors).
    pub fn total_positions(&self) -> u64 {
        self.total_positions
    }

    /// Values in the term's position stream before block `block_idx`.
    #[inline]
    pub fn pos_cursor(&self, block_idx: usize) -> Option<u64> {
        let cursors = self.pos_cursors.as_ref()?;
        let p = block_idx * CURSOR_SIZE;
        cursors
            .get(p..p + CURSOR_SIZE)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// Extract L1 last_doc values from raw LE bytes into a Vec<u32>.
    fn extract_l1_docs(bytes: &[u8], count: usize) -> Vec<u32> {
        let mut docs = Vec::with_capacity(count);
        for i in 0..count {
            let p = i * L1_SIZE;
            docs.push(u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()));
        }
        docs
    }

    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    /// Get maximum term frequency (for MaxScore upper bound computation)
    pub fn max_tf(&self) -> u32 {
        self.max_tf
    }

    /// Get number of blocks
    pub fn num_blocks(&self) -> usize {
        self.l0_count
    }

    /// Get block's max term frequency for block-max pruning
    pub fn block_max_tf(&self, block_idx: usize) -> Option<u32> {
        self.block_bounds(block_idx).map(|(max_tf, _)| max_tf)
    }

    /// Concatenate blocks from multiple posting lists with doc_id remapping.
    /// This is O(num_blocks) instead of O(num_postings).
    pub fn concatenate_blocks(sources: &[(BlockPostingList, u32)]) -> io::Result<Self> {
        let mut stream: Vec<u8> = Vec::new();
        let mut l0_buf: Vec<u8> = Vec::new();
        let mut l1_docs: Vec<u32> = Vec::new();
        let mut l0_count = 0usize;
        let mut total_docs = 0u32;
        let mut max_tf = 0u32;
        let all_cursors = sources.iter().all(|(s, _)| s.has_position_cursors());
        if !all_cursors && sources.iter().any(|(s, _)| s.has_position_cursors()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot concatenate posting lists with and without position cursors",
            ));
        }
        let mut cursors: Vec<u8> = Vec::new();
        let mut positions_before = 0u64;
        let mut min_len = u32::MAX;

        for (source, doc_offset) in sources {
            max_tf = max_tf.max(source.max_tf);
            min_len = min_len.min(source.min_len().unwrap_or(1));
            for block_idx in 0..source.num_blocks() {
                if all_cursors {
                    let cursor = source.pos_cursor(block_idx).unwrap_or(0) + positions_before;
                    cursors.extend_from_slice(&cursor.to_le_bytes());
                }
                let (first_doc, last_doc, offset, word) = source.read_l0_entry(block_idx);
                let (block_max_tf, block_min_len) = unpack_bounds(word, source.len_bounds);
                let bounds = pack_bounds(block_max_tf, block_min_len.unwrap_or(1));
                let blk_size = block_data_size(&source.stream, offset as usize);
                let block_bytes = &source.stream[offset as usize..offset as usize + blk_size];

                let count = u16::from_le_bytes(block_bytes[0..2].try_into().unwrap());
                if stream.len() > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "posting list stream exceeds u32::MAX bytes during concatenation",
                    ));
                }
                let new_offset = stream.len() as u32;

                // Write patched header + copy packed arrays verbatim
                stream.write_u16::<LittleEndian>(count)?;
                stream.write_u32::<LittleEndian>(first_doc + doc_offset)?;
                stream.extend_from_slice(&block_bytes[6..]);

                let new_last = last_doc + doc_offset;
                write_l0(
                    &mut l0_buf,
                    first_doc + doc_offset,
                    new_last,
                    new_offset,
                    bounds,
                );
                l0_count += 1;
                total_docs += count as u32;

                if l0_count.is_multiple_of(L1_INTERVAL) {
                    l1_docs.push(new_last);
                }
            }
            positions_before += source.total_positions;
        }

        // Final L1 entry for partial group
        if !l0_count.is_multiple_of(L1_INTERVAL) && l0_count > 0 {
            let (_, last_doc, _, _) = read_l0(&l0_buf, l0_count - 1);
            l1_docs.push(last_doc);
        }
        let l1_bounds = group_bounds_from_l0(&l0_buf, l0_count);

        Ok(Self {
            stream: OwnedBytes::new(stream),
            l0_bytes: OwnedBytes::new(l0_buf),
            l0_count,
            l1_docs,
            l1_bounds,
            doc_count: total_docs,
            max_tf,
            pos_cursors: all_cursors.then(|| OwnedBytes::new(cursors)),
            total_positions: if all_cursors { positions_before } else { 0 },
            len_bounds: true,
            min_len: if min_len == u32::MAX { 1 } else { min_len },
        })
    }

    /// Streaming merge: write blocks directly to output writer (bounded memory).
    ///
    /// **Zero-materializing**: reads L0 entries directly from source bytes
    /// (mmap or &[u8]) without parsing into Vecs. Block sizes computed from
    /// the 8-byte header (deterministic with packed encoding).
    ///
    /// Output L0 + L1 are buffered (bounded O(total_blocks × 16 + total_blocks/8 × 4)).
    /// Block data flows source → output writer without intermediate buffering.
    ///
    /// Returns `(doc_count, bytes_written)`.
    ///
    /// Returns `Error::Corruption` if any source is shorter than its footer:
    /// metas are paired with sources positionally, so a short/corrupt source
    /// must fail loudly instead of misassigning every subsequent source.
    pub fn concatenate_streaming<W: Write>(
        sources: &[(&[u8], u32)], // (serialized_bytes, doc_offset)
        writer: &mut W,
    ) -> crate::Result<(u32, usize)> {
        let mut metas: Vec<Footer> = Vec::with_capacity(sources.len());
        let mut total_docs = 0u32;
        let mut merged_max_tf = 0u32;
        let mut merged_min_len = u32::MAX;

        for (source_index, (raw, _)) in sources.iter().enumerate() {
            let footer = Footer::parse(raw).map_err(|e| {
                crate::Error::Corruption(format!(
                    "posting list source {source_index} has an invalid footer: {e}"
                ))
            })?;
            total_docs += footer.doc_count;
            merged_max_tf = merged_max_tf.max(footer.max_tf);
            merged_min_len = merged_min_len.min(if footer.len_bounds { footer.min_len } else { 1 });
            metas.push(footer);
        }
        let all_cursors = metas.iter().all(|m| m.has_cursors);
        if !all_cursors && metas.iter().any(|m| m.has_cursors) {
            return Err(crate::Error::Corruption(
                "cannot concatenate posting lists with and without position cursors".into(),
            ));
        }

        // Phase 1: Stream block data, reading L0 entries on-the-fly.
        // Accumulate output L0 + L1 + cursors (bounded).
        let mut out_l0: Vec<u8> = Vec::new();
        let mut out_l1_docs: Vec<u32> = Vec::new();
        let mut out_cursors: Vec<u8> = Vec::new();
        let mut positions_before = 0u64;
        let mut out_l0_count = 0usize;
        let mut stream_written = 0u64;
        let mut patch_buf = [0u8; 8];

        for (src_idx, meta) in metas.iter().enumerate() {
            let (raw, doc_offset) = &sources[src_idx];
            let l0_base = meta.l0_start(); // L0 entries start right after stream
            let src_stream = &raw[..meta.stream_len];
            let cursors_base = meta.l1_bounds_end();

            for i in 0..meta.l0_count {
                // Read source L0 entry directly from raw bytes
                let (first_doc, last_doc, offset, word) = read_l0(&raw[l0_base..], i);
                let (block_max_tf, block_min_len) = unpack_bounds(word, meta.len_bounds);
                let bounds = pack_bounds(block_max_tf, block_min_len.unwrap_or(1));
                if all_cursors {
                    let p = cursors_base + i * CURSOR_SIZE;
                    let cursor = u64::from_le_bytes(raw[p..p + CURSOR_SIZE].try_into().unwrap());
                    out_cursors.extend_from_slice(&(cursor + positions_before).to_le_bytes());
                }

                // Compute block size from header
                let blk_size = block_data_size(src_stream, offset as usize);
                let block = &src_stream[offset as usize..offset as usize + blk_size];

                // Write output L0 entry
                let new_last = last_doc + doc_offset;
                if stream_written > u32::MAX as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "posting list stream exceeds u32::MAX bytes during streaming merge",
                    )
                    .into());
                }
                write_l0(
                    &mut out_l0,
                    first_doc + doc_offset,
                    new_last,
                    stream_written as u32,
                    bounds,
                );
                out_l0_count += 1;

                // L1 entry at group boundary
                if out_l0_count.is_multiple_of(L1_INTERVAL) {
                    out_l1_docs.push(new_last);
                }

                // Patch 8-byte header: [count: u16][first_doc: u32][bits: 2 bytes]
                patch_buf.copy_from_slice(&block[0..8]);
                let blk_first = u32::from_le_bytes(patch_buf[2..6].try_into().unwrap());
                patch_buf[2..6].copy_from_slice(&(blk_first + doc_offset).to_le_bytes());
                writer.write_all(&patch_buf)?;
                writer.write_all(&block[8..])?;

                stream_written += blk_size as u64;
            }
            positions_before += meta.total_positions;
        }

        // Final L1 entry for partial group
        if !out_l0_count.is_multiple_of(L1_INTERVAL) && out_l0_count > 0 {
            let (_, last_doc, _, _) = read_l0(&out_l0, out_l0_count - 1);
            out_l1_docs.push(last_doc);
        }

        // Phase 2: Write L0 + L1 + L1 bounds + cursors + footer
        let out_l1_bounds = group_bounds_from_l0(&out_l0, out_l0_count);
        writer.write_all(&out_l0)?;
        for &doc in &out_l1_docs {
            writer.write_u32::<LittleEndian>(doc)?;
        }
        for &bounds in &out_l1_bounds {
            writer.write_u32::<LittleEndian>(bounds)?;
        }
        writer.write_all(&out_cursors)?;
        Self::write_footer(
            writer,
            stream_written,
            out_l0_count,
            out_l1_docs.len(),
            total_docs,
            merged_max_tf,
            if all_cursors { positions_before } else { 0 },
            all_cursors,
            Some(if merged_min_len == u32::MAX {
                1
            } else {
                merged_min_len
            }),
            true,
        )?;

        let l1_bytes_len = out_l1_docs.len() * L1_SIZE + out_l1_bounds.len() * 4;
        let total_bytes = stream_written as usize
            + out_l0.len()
            + l1_bytes_len
            + out_cursors.len()
            + FOOTER_V2_SIZE;
        Ok((total_docs, total_bytes))
    }

    /// Decode a specific block into caller-provided buffers.
    ///
    /// Returns `true` if the block was decoded, `false` if `block_idx` is out of range.
    /// Reuses `doc_ids` and `tfs` buffers (cleared before filling).
    ///
    /// Uses SIMD-accelerated unpack for 8/16/32-bit packed arrays.
    pub fn decode_block_into(
        &self,
        block_idx: usize,
        doc_ids: &mut Vec<u32>,
        tfs: &mut Vec<u32>,
    ) -> bool {
        if let Some((offset, tf_start, count)) = self.decode_block_doc_ids_only(block_idx, doc_ids)
        {
            self.decode_block_tfs_deferred(offset, tf_start, count, tfs);
            true
        } else {
            false
        }
    }

    /// Decode only doc IDs from a block (no TF decoding).
    ///
    /// Returns `(block_data_offset, tf_start_within_block, count)` for deferred TF decode,
    /// or `None` if block_idx is out of range.
    pub fn decode_block_doc_ids_only(
        &self,
        block_idx: usize,
        doc_ids: &mut Vec<u32>,
    ) -> Option<(usize, usize, usize)> {
        if block_idx >= self.l0_count {
            return None;
        }

        let (_, _, offset, _) = self.read_l0_entry(block_idx);
        let pos = offset as usize;
        let blk_size = block_data_size(&self.stream, pos);
        let block_data = &self.stream[pos..pos + blk_size];

        // 8-byte header: [count: u16][first_doc: u32][doc_id_bits: u8][tf_bits: u8]
        let count = u16::from_le_bytes(block_data[0..2].try_into().unwrap()) as usize;
        let first_doc = u32::from_le_bytes(block_data[2..6].try_into().unwrap());
        let doc_id_bits = block_data[6];

        doc_ids.clear();
        doc_ids.resize(count, 0);
        doc_ids[0] = first_doc;

        let doc_rounded = simd::RoundedBitWidth::from_u8(doc_id_bits);
        let deltas_bytes = if count > 1 {
            (count - 1) * doc_rounded.bytes_per_value()
        } else {
            0
        };

        if count > 1 {
            simd::unpack_rounded(
                &block_data[8..8 + deltas_bytes],
                doc_rounded,
                &mut doc_ids[1..],
                count - 1,
            );
            for i in 1..count {
                doc_ids[i] += doc_ids[i - 1];
            }
        }

        let tfs_start = 8 + deltas_bytes;
        Some((pos, tfs_start, count))
    }

    /// Decode TFs from a previously loaded block (deferred decode).
    ///
    /// `block_offset` and `tf_start` are returned by `decode_block_doc_ids_only`.
    pub fn decode_block_tfs_deferred(
        &self,
        block_offset: usize,
        tf_start: usize,
        count: usize,
        tfs: &mut Vec<u32>,
    ) {
        let blk_size = block_data_size(&self.stream, block_offset);
        let block_data = &self.stream[block_offset..block_offset + blk_size];
        let tf_bits = block_data[7];
        let tf_rounded = simd::RoundedBitWidth::from_u8(tf_bits);

        tfs.clear();
        tfs.resize(count, 0);
        simd::unpack_rounded(
            &block_data[tf_start..tf_start + count * tf_rounded.bytes_per_value()],
            tf_rounded,
            tfs,
            count,
        );
    }

    /// First doc_id of a block (from L0 skip entry). Returns `None` if out of range.
    #[inline]
    pub fn block_first_doc(&self, block_idx: usize) -> Option<DocId> {
        if block_idx >= self.l0_count {
            return None;
        }
        let (first_doc, _, _, _) = self.read_l0_entry(block_idx);
        Some(first_doc)
    }

    /// Last doc_id of a block (from L0 skip entry). Returns `None` if out of range.
    #[inline]
    pub fn block_last_doc(&self, block_idx: usize) -> Option<DocId> {
        if block_idx >= self.l0_count {
            return None;
        }
        let (_, last_doc, _, _) = self.read_l0_entry(block_idx);
        Some(last_doc)
    }

    /// Find the first block whose `last_doc >= target`, starting from `from_block`.
    ///
    /// Uses SIMD-accelerated linear scan:
    /// 1. `find_first_ge_u32` on the contiguous L1 `last_doc` array
    /// 2. Extract ≤`L1_INTERVAL` L0 `last_doc` values into a stack buffer → `find_first_ge_u32`
    ///
    /// Returns `None` if no block contains `target`.
    pub fn seek_block(&self, target: DocId, from_block: usize) -> Option<usize> {
        if from_block >= self.l0_count {
            return None;
        }

        let from_l1 = from_block / L1_INTERVAL;

        // SIMD scan L1 to find the group containing target
        let l1_idx = if !self.l1_docs.is_empty() {
            let idx = from_l1 + simd::find_first_ge_u32(&self.l1_docs[from_l1..], target);
            if idx >= self.l1_docs.len() {
                return None;
            }
            idx
        } else {
            return None;
        };

        // Extract L0 last_doc values within the group into a stack buffer for SIMD scan
        let start = (l1_idx * L1_INTERVAL).max(from_block);
        let end = ((l1_idx + 1) * L1_INTERVAL).min(self.l0_count);
        let count = end - start;

        let mut last_docs = [u32::MAX; L1_INTERVAL];
        for (j, idx) in (start..end).enumerate() {
            let (_, ld, _, _) = read_l0(&self.l0_bytes, idx);
            last_docs[j] = ld;
        }
        let within = simd::find_first_ge_u32(&last_docs[..count], target);
        let block_idx = start + within;

        if block_idx < self.l0_count {
            Some(block_idx)
        } else {
            None
        }
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
/// Uses struct-of-arrays layout: separate `Vec<u32>` for doc_ids and term_freqs.
/// This is more cache-friendly for SIMD seek (contiguous doc_ids) and halves
/// memory vs the previous AoS + separate doc_ids approach.
pub struct BlockPostingIterator<'a> {
    block_list: std::borrow::Cow<'a, BlockPostingList>,
    current_block: usize,
    block_doc_ids: Vec<u32>,
    block_tfs: Vec<u32>,
    position_in_block: usize,
    /// Sum of the term frequencies of the postings before
    /// `position_in_block` in the current block (position stream offset
    /// relative to the block's cursor).
    tf_prefix: u64,
    exhausted: bool,
}

impl<'a> BlockPostingIterator<'a> {
    fn new(block_list: &'a BlockPostingList) -> Self {
        let exhausted = block_list.l0_count == 0;
        let mut iter = Self {
            block_list: std::borrow::Cow::Borrowed(block_list),
            current_block: 0,
            block_doc_ids: Vec::with_capacity(BLOCK_SIZE),
            block_tfs: Vec::with_capacity(BLOCK_SIZE),
            position_in_block: 0,
            tf_prefix: 0,
            exhausted,
        };
        if !iter.exhausted {
            iter.load_block(0);
        }
        iter
    }

    fn owned(block_list: BlockPostingList) -> BlockPostingIterator<'static> {
        let exhausted = block_list.l0_count == 0;
        let mut iter = BlockPostingIterator {
            block_list: std::borrow::Cow::Owned(block_list),
            current_block: 0,
            block_doc_ids: Vec::with_capacity(BLOCK_SIZE),
            block_tfs: Vec::with_capacity(BLOCK_SIZE),
            position_in_block: 0,
            tf_prefix: 0,
            exhausted,
        };
        if !iter.exhausted {
            iter.load_block(0);
        }
        iter
    }

    fn load_block(&mut self, block_idx: usize) {
        if block_idx >= self.block_list.l0_count {
            self.exhausted = true;
            return;
        }

        self.current_block = block_idx;
        self.position_in_block = 0;
        self.tf_prefix = 0;

        self.block_list
            .decode_block_into(block_idx, &mut self.block_doc_ids, &mut self.block_tfs);
    }

    /// Offset of the current posting's positions in the term's position
    /// stream (see `structures::postings::positions_v2`): the block's cursor
    /// plus the term frequencies of the postings before it in the block.
    /// Meaningful only for lists built with position cursors.
    #[inline]
    pub fn position_cursor(&self) -> u64 {
        self.block_list.pos_cursor(self.current_block).unwrap_or(0) + self.tf_prefix
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

        if let Some(&tf) = self.block_tfs.get(self.position_in_block) {
            self.tf_prefix += tf as u64;
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

        // SIMD-accelerated 2-level seek (forward from current block)
        let block_idx = match self.block_list.seek_block(target, self.current_block) {
            Some(idx) => idx,
            None => {
                self.exhausted = true;
                return TERMINATED;
            }
        };

        if block_idx != self.current_block {
            self.load_block(block_idx);
        }

        // SIMD linear scan within block on cached doc_ids
        let remaining = &self.block_doc_ids[self.position_in_block..];
        let pos = crate::structures::simd::find_first_ge_u32(remaining, target);
        self.tf_prefix += self.block_tfs[self.position_in_block..self.position_in_block + pos]
            .iter()
            .map(|&tf| tf as u64)
            .sum::<u64>();
        self.position_in_block += pos;

        if self.position_in_block >= self.block_doc_ids.len() {
            self.load_block(self.current_block + 1);
        }
        self.doc()
    }

    /// Skip to the next block, returning the first doc_id in the new block
    /// This is used for block-max pruning when the current block's
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
        self.block_list.l0_count
    }

    /// Get the current block's max term frequency for block-max pruning
    #[inline]
    pub fn current_block_max_tf(&self) -> u32 {
        if self.exhausted || self.current_block >= self.block_list.l0_count {
            0
        } else {
            self.block_list
                .block_max_tf(self.current_block)
                .unwrap_or(0)
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
    fn test_concatenate_streaming_short_source_returns_corruption() {
        // A source shorter than the 24-byte footer (e.g. a corrupt TermInfo
        // (offset, len) pointing at truncated bytes) must fail loudly.
        // Silently skipping it pairs every later source with the wrong
        // metadata (metas[i] vs sources[i]) — panicking or emitting garbage.
        let seg_a: Vec<(u32, u32)> = (0..250).map(|i| (i * 2, (i % 7) + 1)).collect();
        let seg_c: Vec<(u32, u32)> = (0..90).map(|i| (i * 10, (i % 11) + 1)).collect();
        let bytes_a = serialize_bpl(&build_bpl(&seg_a));
        let bytes_c = serialize_bpl(&build_bpl(&seg_c));
        let short = vec![0u8; FOOTER_SIZE - 1]; // corrupt: shorter than footer

        let sources: Vec<(&[u8], u32)> = vec![(&bytes_a, 0), (&short, 1000), (&bytes_c, 2000)];
        let mut out = Vec::new();
        let result = BlockPostingList::concatenate_streaming(&sources, &mut out);
        assert!(
            matches!(result, Err(crate::Error::Corruption(_))),
            "short/corrupt source must be a Corruption error, not silently skipped: {:?}",
            result.map(|r| r.0)
        );
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

    // ── 2-level skip list format tests ──────────────────────────────────

    #[test]
    fn test_l0_l1_counts() {
        // 1 block (< L1_INTERVAL) → 1 L1 entry (partial group)
        let bpl = build_bpl(&(0..50u32).map(|i| (i, 1)).collect::<Vec<_>>());
        assert_eq!(bpl.num_blocks(), 1);
        assert_eq!(bpl.l1_docs.len(), 1);

        // Exactly L1_INTERVAL blocks → 1 L1 entry (full group)
        let n = BLOCK_SIZE * L1_INTERVAL;
        let bpl = build_bpl(&(0..n as u32).map(|i| (i * 2, 1)).collect::<Vec<_>>());
        assert_eq!(bpl.num_blocks(), L1_INTERVAL);
        assert_eq!(bpl.l1_docs.len(), 1);

        // L1_INTERVAL + 1 blocks → 2 L1 entries
        let n = BLOCK_SIZE * L1_INTERVAL + 1;
        let bpl = build_bpl(&(0..n as u32).map(|i| (i * 2, 1)).collect::<Vec<_>>());
        assert_eq!(bpl.num_blocks(), L1_INTERVAL + 1);
        assert_eq!(bpl.l1_docs.len(), 2);

        // 3 × L1_INTERVAL blocks → 3 L1 entries (all full groups)
        let n = BLOCK_SIZE * L1_INTERVAL * 3;
        let bpl = build_bpl(&(0..n as u32).map(|i| (i, 1)).collect::<Vec<_>>());
        assert_eq!(bpl.num_blocks(), L1_INTERVAL * 3);
        assert_eq!(bpl.l1_docs.len(), 3);
    }

    #[test]
    fn test_l1_last_doc_values() {
        // 20 blocks: 2 full L1 groups (8+8) + 1 partial (4) → 3 L1 entries
        let n = BLOCK_SIZE * 20;
        let docs: Vec<(u32, u32)> = (0..n as u32).map(|i| (i * 3, 1)).collect();
        let bpl = build_bpl(&docs);
        assert_eq!(bpl.num_blocks(), 20);
        assert_eq!(bpl.l1_docs.len(), 3); // ceil(20/8) = 3

        // L1[0] = last_doc of block 7 (end of first group)
        let expected_l1_0 = bpl.block_last_doc(7).unwrap();
        assert_eq!(bpl.l1_docs[0], expected_l1_0);

        // L1[1] = last_doc of block 15 (end of second group)
        let expected_l1_1 = bpl.block_last_doc(15).unwrap();
        assert_eq!(bpl.l1_docs[1], expected_l1_1);

        // L1[2] = last_doc of block 19 (end of partial group)
        let expected_l1_2 = bpl.block_last_doc(19).unwrap();
        assert_eq!(bpl.l1_docs[2], expected_l1_2);
    }

    #[test]
    fn test_seek_block_basic() {
        // 20 blocks spanning large doc ID range
        let n = BLOCK_SIZE * 20;
        let docs: Vec<(u32, u32)> = (0..n as u32).map(|i| (i * 10, 1)).collect();
        let bpl = build_bpl(&docs);

        // Seek to doc 0 → block 0
        assert_eq!(bpl.seek_block(0, 0), Some(0));

        // Seek to the first doc of each block
        for blk in 0..20 {
            let first = bpl.block_first_doc(blk).unwrap();
            assert_eq!(
                bpl.seek_block(first, 0),
                Some(blk),
                "seek to block {} first_doc",
                blk
            );
        }

        // Seek to the last doc of each block
        for blk in 0..20 {
            let last = bpl.block_last_doc(blk).unwrap();
            assert_eq!(
                bpl.seek_block(last, 0),
                Some(blk),
                "seek to block {} last_doc",
                blk
            );
        }

        // Seek past all docs
        let max_doc = bpl.block_last_doc(19).unwrap();
        assert_eq!(bpl.seek_block(max_doc + 1, 0), None);

        // Seek with from_block > 0 (skip early blocks)
        let mid_doc = bpl.block_first_doc(10).unwrap();
        assert_eq!(bpl.seek_block(mid_doc, 10), Some(10));
        assert_eq!(
            bpl.seek_block(mid_doc, 11),
            Some(11).or(bpl.seek_block(mid_doc, 11))
        );
    }

    #[test]
    fn test_seek_block_across_l1_boundaries() {
        // 24 blocks = 3 L1 groups of 8
        let n = BLOCK_SIZE * 24;
        let docs: Vec<(u32, u32)> = (0..n as u32).map(|i| (i * 5, 1)).collect();
        let bpl = build_bpl(&docs);
        assert_eq!(bpl.l1_docs.len(), 3);

        // Seek into each L1 group
        for group in 0..3 {
            let blk = group * L1_INTERVAL;
            let target = bpl.block_first_doc(blk).unwrap();
            assert_eq!(
                bpl.seek_block(target, 0),
                Some(blk),
                "seek to group {} block {}",
                group,
                blk
            );
        }

        // Seek to doc in the middle of group 2 (block 20)
        let target = bpl.block_first_doc(20).unwrap() + 1;
        assert_eq!(bpl.seek_block(target, 0), Some(20));
    }

    #[test]
    fn test_block_data_size_helper() {
        // Build a posting list and verify block_data_size matches actual block sizes
        let docs: Vec<(u32, u32)> = (0..500u32).map(|i| (i * 7, (i % 20) + 1)).collect();
        let bpl = build_bpl(&docs);

        for blk in 0..bpl.num_blocks() {
            let (_, _, offset, _) = bpl.read_l0_entry(blk);
            let computed_size = block_data_size(&bpl.stream, offset as usize);

            // Verify: next block's offset - this block's offset should equal computed_size
            // (for all but last block)
            if blk + 1 < bpl.num_blocks() {
                let (_, _, next_offset, _) = bpl.read_l0_entry(blk + 1);
                assert_eq!(
                    computed_size,
                    (next_offset - offset) as usize,
                    "block_data_size mismatch at block {}",
                    blk
                );
            } else {
                // Last block: offset + size should equal stream length
                assert_eq!(
                    offset as usize + computed_size,
                    bpl.stream.len(),
                    "last block size mismatch"
                );
            }
        }
    }

    #[test]
    fn test_l0_entry_roundtrip() {
        // Verify L0 entries survive serialize → deserialize
        let docs: Vec<(u32, u32)> = (0..1000u32).map(|i| (i * 3, (i % 10) + 1)).collect();
        let bpl = build_bpl(&docs);

        let bytes = serialize_bpl(&bpl);
        let bpl2 = BlockPostingList::deserialize(&bytes).unwrap();

        assert_eq!(bpl.num_blocks(), bpl2.num_blocks());
        for blk in 0..bpl.num_blocks() {
            assert_eq!(
                bpl.read_l0_entry(blk),
                bpl2.read_l0_entry(blk),
                "L0 entry mismatch at block {}",
                blk
            );
        }

        // Verify L1 docs match
        assert_eq!(bpl.l1_docs, bpl2.l1_docs);
    }

    #[test]
    fn test_zero_copy_deserialize_matches() {
        let docs: Vec<(u32, u32)> = (0..2000u32).map(|i| (i * 2, (i % 5) + 1)).collect();
        let bpl = build_bpl(&docs);
        let bytes = serialize_bpl(&bpl);

        let copied = BlockPostingList::deserialize(&bytes).unwrap();
        let zero_copy =
            BlockPostingList::deserialize_zero_copy(OwnedBytes::new(bytes.clone())).unwrap();

        // Same structure
        assert_eq!(copied.l0_count, zero_copy.l0_count);
        assert_eq!(copied.l1_docs, zero_copy.l1_docs);
        assert_eq!(copied.doc_count, zero_copy.doc_count);
        assert_eq!(copied.max_tf, zero_copy.max_tf);

        // Same iteration
        let p1 = collect_postings(&copied);
        let p2 = collect_postings(&zero_copy);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_l1_preserved_through_streaming_merge() {
        // Merge 3 segments, verify L1 is correctly rebuilt
        let seg_a = build_bpl(&(0..1000u32).map(|i| (i * 2, 1)).collect::<Vec<_>>());
        let seg_b = build_bpl(&(0..800u32).map(|i| (i * 3, 2)).collect::<Vec<_>>());
        let seg_c = build_bpl(&(0..500u32).map(|i| (i * 5, 3)).collect::<Vec<_>>());

        let bytes_a = serialize_bpl(&seg_a);
        let bytes_b = serialize_bpl(&seg_b);
        let bytes_c = serialize_bpl(&seg_c);

        let sources: Vec<(&[u8], u32)> = vec![(&bytes_a, 0), (&bytes_b, 10000), (&bytes_c, 20000)];
        let mut out = Vec::new();
        BlockPostingList::concatenate_streaming(&sources, &mut out).unwrap();

        let merged = BlockPostingList::deserialize(&out).unwrap();
        let expected_l1_count = merged.num_blocks().div_ceil(L1_INTERVAL);
        assert_eq!(merged.l1_docs.len(), expected_l1_count);

        // Verify L1 values are correct
        for (i, &l1_doc) in merged.l1_docs.iter().enumerate() {
            let last_block_in_group = ((i + 1) * L1_INTERVAL - 1).min(merged.num_blocks() - 1);
            let expected = merged.block_last_doc(last_block_in_group).unwrap();
            assert_eq!(l1_doc, expected, "L1[{}] mismatch", i);
        }

        // Verify seek_block works on merged result
        for blk in 0..merged.num_blocks() {
            let first = merged.block_first_doc(blk).unwrap();
            assert_eq!(merged.seek_block(first, 0), Some(blk));
        }
    }

    #[test]
    fn test_seek_block_single_block() {
        // Edge case: single block (< L1_INTERVAL)
        let bpl = build_bpl(&[(0, 1), (10, 2), (20, 3)]);
        assert_eq!(bpl.num_blocks(), 1);
        assert_eq!(bpl.l1_docs.len(), 1);

        assert_eq!(bpl.seek_block(0, 0), Some(0));
        assert_eq!(bpl.seek_block(10, 0), Some(0));
        assert_eq!(bpl.seek_block(20, 0), Some(0));
        assert_eq!(bpl.seek_block(21, 0), None);
    }

    #[test]
    fn test_footer_size() {
        // Verify serialized size = stream + L0 + L1 + FOOTER_SIZE
        let docs: Vec<(u32, u32)> = (0..500u32).map(|i| (i * 2, 1)).collect();
        let bpl = build_bpl(&docs);
        let bytes = serialize_bpl(&bpl);

        let expected = bpl.stream.len()
            + bpl.l0_count * L0_SIZE
            + bpl.l1_docs.len() * (L1_SIZE + 4)
            + FOOTER_V2_SIZE;
        assert_eq!(bytes.len(), expected);
    }

    fn build_bpl_with_positions(postings: &[(u32, u32)]) -> BlockPostingList {
        let mut list = PostingList::new();
        for &(doc, tf) in postings {
            list.push(doc, tf);
        }
        BlockPostingList::from_posting_list_with_positions(&list).unwrap()
    }

    /// Expected cursor of every posting: the cumulative tf before it.
    fn expected_cursors(postings: &[(u32, u32)]) -> Vec<u64> {
        let mut acc = 0u64;
        postings
            .iter()
            .map(|&(_, tf)| {
                let c = acc;
                acc += tf as u64;
                c
            })
            .collect()
    }

    fn iterator_cursors(bpl: &BlockPostingList) -> Vec<u64> {
        let mut it = bpl.iterator();
        let mut out = Vec::new();
        while it.doc() != TERMINATED {
            out.push(it.position_cursor());
            it.advance();
        }
        out
    }

    #[test]
    fn position_cursors_survive_serialization_and_seeks() {
        let docs: Vec<(u32, u32)> = (0..700u32).map(|i| (i * 3, i % 5 + 1)).collect();
        let bpl = build_bpl_with_positions(&docs);
        assert!(bpl.has_position_cursors());
        assert_eq!(
            bpl.total_positions(),
            docs.iter().map(|&(_, tf)| tf as u64).sum::<u64>()
        );
        assert_eq!(bpl.pos_cursor(0), Some(0));
        assert_eq!(
            bpl.pos_cursor(1),
            Some(docs[..128].iter().map(|&(_, tf)| tf as u64).sum::<u64>())
        );
        assert_eq!(iterator_cursors(&bpl), expected_cursors(&docs));

        let bytes = serialize_bpl(&bpl);
        assert_eq!(
            bytes.len(),
            bpl.stream.len()
                + bpl.l0_count * (L0_SIZE + CURSOR_SIZE)
                + bpl.l1_docs.len() * (L1_SIZE + 4)
                + FOOTER_V2_SIZE
        );
        assert!(BlockPostingList::has_cursors_bytes(&bytes));
        let decoded =
            BlockPostingList::deserialize_zero_copy(OwnedBytes::new(bytes.clone())).unwrap();
        assert_eq!(iterator_cursors(&decoded), expected_cursors(&docs));
        assert_eq!(decoded.total_positions(), bpl.total_positions());

        // Seeking within and across blocks keeps the cursor exact.
        let mut it = decoded.iterator();
        let expected = expected_cursors(&docs);
        for (i, &(doc, _)) in docs.iter().enumerate().step_by(37) {
            assert_eq!(it.seek(doc), doc);
            assert_eq!(it.position_cursor(), expected[i], "cursor at doc {doc}");
        }
        let mut it = decoded.iterator();
        assert_eq!(it.seek(docs[600].0 + 1), docs[601].0);
        assert_eq!(it.position_cursor(), expected[601]);

        // Lists without positions carry no cursors (the iterator's prefix
        // sum is then relative to nothing and never consulted).
        let plain = build_bpl(&docs);
        assert!(!plain.has_position_cursors());
        assert_eq!(plain.pos_cursor(0), None);
        assert_eq!(plain.total_positions(), 0);
    }

    #[test]
    fn length_bounds_are_packed_per_block_and_survive_merges() {
        let docs: Vec<(u32, u32)> = (0..300u32).map(|i| (i, i % 3 + 1)).collect();
        let length_of = |doc: u32| 10 + (doc % 50) * 7;
        let mut list = PostingList::new();
        for &(doc, tf) in &docs {
            list.push(doc, tf);
        }
        let bpl = BlockPostingList::from_posting_list_with(&list, true, Some(&length_of)).unwrap();
        assert_eq!(bpl.min_len(), Some(10));
        assert_eq!(bpl.block_bounds(0), Some((3, Some(10))));
        // Block 2 covers docs 256..300: min length there is doc 256 (256 % 50 = 6 → 52).
        assert_eq!(bpl.block_bounds(2), Some((3, Some(52))));
        assert_eq!(bpl.block_max_tf(2), Some(3));

        let bytes = serialize_bpl(&bpl);
        let decoded = BlockPostingList::deserialize(&bytes).unwrap();
        assert_eq!(decoded.min_len(), Some(10));
        assert_eq!(decoded.block_bounds(2), Some((3, Some(52))));
        // Superblock bounds: one group of three blocks here, max tf 3 and
        // the smallest length of the whole list.
        assert_eq!(decoded.group_bounds(0), Some((3, 10)));
        assert_eq!(decoded.group_bounds(2), Some((3, 10)));
        assert_eq!(decoded.group_bounds(3), None);
        assert_eq!(decoded.group_last_doc(1), Some(299));
        assert_eq!(decoded.next_group_block(1), 3);

        // Without lengths the minimum is 1, which every real unit satisfies.
        let plain = build_bpl(&docs);
        assert_eq!(plain.min_len(), Some(1));
        assert_eq!(plain.block_bounds(0), Some((3, Some(1))));

        // Streaming merge keeps per-block bounds and takes the list minimum.
        let mut out = Vec::new();
        BlockPostingList::concatenate_streaming(&[(&bytes, 0), (&bytes, 1000)], &mut out).unwrap();
        let merged = BlockPostingList::deserialize(&out).unwrap();
        assert_eq!(merged.min_len(), Some(10));
        assert_eq!(merged.block_bounds(2), Some((3, Some(52))));
        assert_eq!(merged.block_bounds(3), Some((3, Some(10))));
        assert_eq!(merged.block_max_tf(5), Some(3));
        // Six blocks: one full group of eight would need more; here both
        // lists' blocks share group 0.
        assert_eq!(merged.group_bounds(5), Some((3, 10)));
        assert_eq!(merged.group_last_doc(5), Some(1299));
        assert_eq!(merged.next_group_block(5), 6);
    }

    #[test]
    fn legacy_footer_without_magic_still_deserializes() {
        let docs: Vec<(u32, u32)> = (0..300u32).map(|i| (i * 2, 1 + i % 3)).collect();
        let bpl = build_bpl(&docs);
        let bytes = serialize_bpl(&bpl);
        // A pre-magic list is the same bytes without the 16-byte extension.
        let legacy = bytes[..bytes.len() - (FOOTER_V2_SIZE - FOOTER_SIZE)].to_vec();
        assert!(!BlockPostingList::has_cursors_bytes(&legacy));
        // Legacy lists carry an f32 max tf per block and no lengths.
        let decoded = BlockPostingList::deserialize(&legacy).unwrap();
        assert_eq!(collect_postings(&decoded), docs);
        assert_eq!(decoded.max_tf(), 3);
        assert!(!decoded.has_position_cursors());
        assert_eq!(decoded.min_len(), None);
        assert_eq!(decoded.group_bounds(0), None);
        // And the legacy bytes concatenate into a current-format list.
        let mut out = Vec::new();
        let (count, written) =
            BlockPostingList::concatenate_streaming(&[(&legacy, 0), (&legacy, 1000)], &mut out)
                .unwrap();
        assert_eq!(count, 600);
        assert_eq!(written, out.len());
        let merged = BlockPostingList::deserialize(&out).unwrap();
        assert_eq!(merged.doc_count(), 600);
        assert!(!merged.has_position_cursors());
    }

    #[test]
    fn streaming_merge_rebases_position_cursors() {
        let a: Vec<(u32, u32)> = (0..200u32).map(|i| (i, i % 4 + 1)).collect();
        let b: Vec<(u32, u32)> = (0..150u32).map(|i| (i * 2, 2)).collect();
        let bytes_a = serialize_bpl(&build_bpl_with_positions(&a));
        let bytes_b = serialize_bpl(&build_bpl_with_positions(&b));
        let mut out = Vec::new();
        let (count, written) =
            BlockPostingList::concatenate_streaming(&[(&bytes_a, 0), (&bytes_b, 1000)], &mut out)
                .unwrap();
        assert_eq!(count, 350);
        assert_eq!(written, out.len());
        let merged = BlockPostingList::deserialize(&out).unwrap();
        assert!(merged.has_position_cursors());
        let all: Vec<(u32, u32)> = a
            .iter()
            .copied()
            .chain(b.iter().map(|&(d, tf)| (d + 1000, tf)))
            .collect();
        assert_eq!(collect_postings(&merged), all);
        assert_eq!(iterator_cursors(&merged), expected_cursors(&all));
        assert_eq!(
            merged.total_positions(),
            all.iter().map(|&(_, tf)| tf as u64).sum::<u64>()
        );
        // The in-memory reference agrees.
        let reference = BlockPostingList::concatenate_blocks(&[
            (build_bpl_with_positions(&a), 0),
            (build_bpl_with_positions(&b), 1000),
        ])
        .unwrap();
        assert_eq!(iterator_cursors(&reference), expected_cursors(&all));
        // Mixing lists with and without cursors is refused.
        let plain = serialize_bpl(&build_bpl(&b));
        assert!(
            BlockPostingList::concatenate_streaming(
                &[(&bytes_a, 0), (&plain, 1000)],
                &mut Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn test_seek_block_from_block_skips_earlier() {
        // 16 blocks: seek with from_block should skip earlier blocks
        let n = BLOCK_SIZE * 16;
        let docs: Vec<(u32, u32)> = (0..n as u32).map(|i| (i * 3, 1)).collect();
        let bpl = build_bpl(&docs);

        // Target is in block 5, but from_block=8 → should find block >= 8
        let target_in_5 = bpl.block_first_doc(5).unwrap() + 1;
        // from_block=8 means we only look at blocks 8+
        // target_in_5 < last_doc of block 8, so seek_block(target, 8) should return 8
        let result = bpl.seek_block(target_in_5, 8);
        assert!(result.is_some());
        assert!(result.unwrap() >= 8);
    }
}
