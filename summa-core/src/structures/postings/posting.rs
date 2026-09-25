//! Posting list implementation with compact representation
//!
//! Text blocks hold 128 postings: delta-coded doc ids followed by term
//! frequencies, each array encoded by one of the [`PostingCodec`]s
//! (`docs/posting-codecs.md`):
//! - `Rounded` (default): widths rounded to 0/8/16/32 bits, SIMD widening
//! - `Packed`: exact bit widths (BP128 style)
//! - `Pfor`: exact width with patched exceptions (OptP4D style)
//! - `Simd4x`: four-lane library packing and integrated strict document deltas
//!
//! The codec is stored per block in the header, so a single list (for example
//! the output of a merge) may mix codecs.

mod groups;
mod impacts;
mod reader;
mod validation;
use groups::GroupWords;
use impacts::{ImpactBuilder, ImpactTable};

pub(crate) use reader::{DeferredPosting, PostingListReader};

#[cfg(feature = "native")]
mod compact;
#[cfg(feature = "native")]
pub(crate) use compact::{PostingBlockSource, PostingStreamWriter};

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{self, Write};

use super::bitpacking4x;
use super::horizontal_bp128::{pack_block_n as pack_bits, unpack_block_n as unpack_bits};
use super::opt_p4d::{find_optimal_bit_width, pack_with_exceptions};
use crate::DocId;
use crate::directories::OwnedBytes;
use crate::structures::simd;

/// Encoding of the packed arrays inside one posting block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PostingCodec {
    /// Widths rounded up to 0/8/16/32 bits; decodes with plain SIMD widening.
    /// This is the low-overhead performance baseline.
    #[default]
    Rounded = 0,
    /// Exact bit widths (BP128 style): ~1.8× smaller than `Rounded` on the
    /// repository benchmark for ~10 % slower decoding.
    Packed = 1,
    /// Exact width with up to 10 % patched exceptions (OptP4D style):
    /// smallest, ~30 % slower decoding than `Rounded`.
    Pfor = 2,
    /// Library SIMD packing of full blocks with exact horizontal tails.
    /// Documents use gap-minus-one values; positions use the same block policy.
    Simd4x = 3,
}

impl PostingCodec {
    /// Codec id stored in the top two bits of the block header's `doc_bits`.
    const HEADER_SHIFT: u32 = 6;
    const WIDTH_MASK: u8 = 0x3F;

    /// The two-bit id field is fully assigned. A fifth codec cannot be
    /// signalled in the block header: it needs a footer flag plus an
    /// `INDEX_META_FORMAT_VERSION` bump (see `docs/posting-codecs.md`).
    fn from_header_byte(doc_bits: u8) -> io::Result<(Self, u8)> {
        let width = doc_bits & Self::WIDTH_MASK;
        let codec = match doc_bits >> Self::HEADER_SHIFT {
            0 => PostingCodec::Rounded,
            1 => PostingCodec::Packed,
            2 => PostingCodec::Pfor,
            _ => PostingCodec::Simd4x,
        };
        if width > 32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("posting block doc-id width {width} exceeds 32 bits"),
            ));
        }
        Ok((codec, width))
    }

    /// The four-lane kernel requires a full block. Existing rounded tails
    /// avoid scalar bit extraction on short runs preserved by normal merge.
    pub(super) fn for_count(self, count: usize) -> Self {
        if self == Self::Simd4x && count < BLOCK_SIZE {
            Self::Rounded
        } else {
            self
        }
    }

    fn header_byte(self, width: u8) -> u8 {
        ((self as u8) << Self::HEADER_SHIFT) | width
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "rounded" | "default" => Some(PostingCodec::Rounded),
            "packed" | "bp128" | "exact" => Some(PostingCodec::Packed),
            "pfor" | "optp4d" | "patched" => Some(PostingCodec::Pfor),
            "simd4x" => Some(PostingCodec::Simd4x),
            _ => None,
        }
    }
}

impl std::fmt::Display for PostingCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PostingCodec::Rounded => "rounded",
            PostingCodec::Packed => "packed",
            PostingCodec::Pfor => "pfor",
            PostingCodec::Simd4x => "simd4x",
        })
    }
}

// ── Exact-width bit packing (Packed codec) ───────────────────────────────

/// Bytes needed for `count` values at `width` bits.
#[inline]
fn packed_bytes(count: usize, width: u8) -> usize {
    (count * width as usize).div_ceil(8)
}

// ── Patched packing (Pfor codec) ─────────────────────────────────────────

/// Payload of one `Pfor` array: `[n_exceptions u8][packed low bits][(pos u8, high u32) × n]`.
fn pack_pfor(values: &[u32], out: &mut Vec<u8>) -> u8 {
    let (width, _, _) = find_optimal_bit_width(values);
    let (packed, exceptions) = pack_with_exceptions(values, width);
    out.push(exceptions.len() as u8);
    out.extend_from_slice(&packed);
    for (pos, high) in exceptions {
        out.push(pos);
        out.extend_from_slice(&high.to_le_bytes());
    }
    width
}

/// Byte length of a `Pfor` array payload for `count` values at `width`.
fn pfor_payload_len(input: &[u8], count: usize, width: u8) -> io::Result<usize> {
    let n_exceptions = *input
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "posting block truncated"))?
        as usize;
    Ok(1 + packed_bytes(count, width) + n_exceptions * 5)
}

/// Decode one `Pfor` array: the packed low bits, then each `(pos, high)`
/// exception patched in place straight from the table. No per-decode scratch.
fn unpack_pfor(input: &[u8], width: u8, out: &mut [u32], count: usize) -> io::Result<()> {
    let n_exceptions = *input
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "posting block truncated"))?
        as usize;
    let table_at = 1 + packed_bytes(count, width);
    let table_end = table_at + n_exceptions * 5;
    if input.len() < table_end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "posting block exception table truncated",
        ));
    }
    let out = &mut out[..count];
    unpack_bits(&input[1..table_at], width, out, count);
    if width < 32 {
        for entry in input[table_at..table_end].chunks_exact(5) {
            let pos = entry[0] as usize;
            let high = u32::from_le_bytes([entry[1], entry[2], entry[3], entry[4]]);
            if pos < count {
                out[pos] |= high << width;
            }
        }
    }
    Ok(())
}

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
        crate::observe::search_work!(posting_seeks += 1);
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

/// Optional downward-rounded length/TF minima, L0 then L1, after cursors.
const FLAG_RATIO_BOUNDS: u32 = 8;

/// Optional complete frequency/length envelopes after ratio metadata.
const FLAG_IMPACT_BOUNDS: u32 = 16;
/// Combined impact directory: L0 records followed by L1 group records.
const FLAG_GROUP_IMPACT_BOUNDS: u32 = 32;
const FLAG_COMPACT_HEADERS: u32 = 64;
const FLAG_SHORT_CURSORS: u32 = 128;

fn append_ratio_groups(ratios: &mut Vec<u8>, blocks: usize) {
    for start in (0..blocks).step_by(L1_INTERVAL) {
        let ratio = (start..(start + L1_INTERVAL).min(blocks))
            .map(|i| read_ratio(ratios, i))
            .fold(f32::INFINITY, f32::min);
        ratios.extend_from_slice(&ratio.to_le_bytes());
    }
}

#[inline]
fn read_ratio(bytes: &[u8], index: usize) -> f32 {
    let at = index * 4;
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn validate_ratios(bytes: &[u8]) -> io::Result<()> {
    if bytes.chunks_exact(4).any(|value| {
        let ratio = f32::from_le_bytes(value.try_into().unwrap());
        !ratio.is_finite() || ratio < 0.0
    }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid posting ratio bound",
        ));
    }
    Ok(())
}

fn lower_length_ratio(length: u32, tf: u32) -> f32 {
    if tf == 0 {
        return 0.0;
    }
    // f64 represents both inputs exactly. One f32 step down covers division
    // and conversion rounding, including an exactly representable quotient.
    ((length as f64 / tf as f64) as f32).next_down().max(0.0)
}

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
#[derive(Debug, Clone, Copy)]
struct Footer {
    compact_headers: bool,
    short_cursors: bool,
    stream_len: usize,
    l0_count: usize,
    l1_count: usize,
    doc_count: u32,
    max_tf: u32,
    total_positions: u64,
    has_cursors: bool,
    len_bounds: bool,
    l1_bounds: bool,
    ratio_bounds: bool,
    impact_bounds: bool,
    group_impact_bounds: bool,
    min_len: u32,
}

impl Footer {
    fn parse(raw: &[u8]) -> io::Result<Self> {
        Self::parse_tail(raw, raw.len())
    }

    fn parse_tail(raw: &[u8], total_len: usize) -> io::Result<Self> {
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
        let stream_len = usize::try_from(u64::from_le_bytes(raw[f..f + 8].try_into().unwrap()))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "posting stream exceeds address space",
                )
            })?;
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
        if flags
            & !(FLAG_POS_CURSORS
                | FLAG_LEN_BOUNDS
                | FLAG_L1_BOUNDS
                | FLAG_RATIO_BOUNDS
                | FLAG_IMPACT_BOUNDS
                | FLAG_GROUP_IMPACT_BOUNDS
                | FLAG_COMPACT_HEADERS
                | FLAG_SHORT_CURSORS)
            != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown posting footer flags",
            ));
        }
        let footer = Self {
            compact_headers: flags & FLAG_COMPACT_HEADERS != 0,
            short_cursors: flags & FLAG_SHORT_CURSORS != 0,
            stream_len,
            l0_count,
            l1_count,
            doc_count,
            max_tf,
            total_positions,
            has_cursors: flags & FLAG_POS_CURSORS != 0,
            len_bounds: flags & FLAG_LEN_BOUNDS != 0,
            l1_bounds: flags & FLAG_L1_BOUNDS != 0,
            ratio_bounds: flags & FLAG_RATIO_BOUNDS != 0,
            impact_bounds: flags & FLAG_IMPACT_BOUNDS != 0,
            group_impact_bounds: flags & FLAG_GROUP_IMPACT_BOUNDS != 0,
            min_len,
        };
        if footer.short_cursors && (!footer.has_cursors || footer.total_positions > u32::MAX as u64)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid short position cursors",
            ));
        }
        if footer.group_impact_bounds && (!footer.impact_bounds || !footer.l1_bounds) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "group impacts require L0 impacts and L1 bounds",
            ));
        }
        let end = l0_count
            .checked_mul(L0_SIZE + if footer.compact_headers { 4 } else { 0 })
            .and_then(|n| {
                l1_count
                    .checked_mul(L1_SIZE + if footer.l1_bounds { 4 } else { 0 })
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| {
                l0_count
                    .checked_mul(if footer.has_cursors {
                        footer.cursor_size()
                    } else {
                        0
                    })
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| {
                if footer.ratio_bounds {
                    l0_count
                        .checked_add(l1_count)
                        .and_then(|m| m.checked_mul(4))
                        .and_then(|m| n.checked_add(m))
                } else {
                    Some(n)
                }
            })
            .and_then(|n| n.checked_add(stream_len));
        let footer_offset = total_len.saturating_sub(raw.len() - f);
        if end.is_none_or(|end| {
            if footer.impact_bounds {
                !footer.len_bounds
                    || !footer.ratio_bounds
                    || l0_count
                        .checked_add(if footer.group_impact_bounds {
                            l1_count
                        } else {
                            0
                        })
                        .and_then(|n| n.checked_add(1))
                        .and_then(|n| n.checked_mul(4))
                        .and_then(|n| n.checked_add(end))
                        .is_none_or(|minimum| minimum > footer_offset)
            } else {
                end != footer_offset
            }
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting list sections do not match the footer offset",
            ));
        }
        Ok(footer)
    }

    fn l0_start(&self) -> usize {
        self.stream_len
    }
    fn impact_record_count(&self) -> usize {
        self.l0_count
            + if self.group_impact_bounds {
                self.l1_count
            } else {
                0
            }
    }
    fn l0_end(&self) -> usize {
        self.l0_start() + self.l0_count * L0_SIZE
    }
    fn l1_start(&self) -> usize {
        self.l0_end()
            + if self.compact_headers {
                self.l0_count * 4
            } else {
                0
            }
    }
    fn cursor_size(&self) -> usize {
        if self.short_cursors { 4 } else { CURSOR_SIZE }
    }
    fn l1_end(&self) -> usize {
        self.l1_start() + self.l1_count * L1_SIZE
    }
    fn l1_bounds_end(&self) -> usize {
        self.l1_end() + if self.l1_bounds { self.l1_count * 4 } else { 0 }
    }
    fn ratios_end(&self) -> usize {
        self.cursors_end()
            + if self.ratio_bounds {
                (self.l0_count + self.l1_count) * 4
            } else {
                0
            }
    }
    fn cursors_end(&self) -> usize {
        self.l1_bounds_end()
            + if self.has_cursors {
                self.l0_count * self.cursor_size()
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

/// Content check that structural admission cannot make without decoding:
/// a block's ids must be strictly increasing and span exactly its L0 range.
/// Branch-free so the 128-value pass vectorises on the decode path.
#[inline]
fn verify_block_docs(
    docs: &[u32],
    first: u32,
    last: u32,
    byte_gaps: Option<&[u8]>,
    strict_gap_width: Option<u8>,
) -> bool {
    let ordered = if strict_gap_width.is_some_and(|width| width <= 25) {
        // Gap-minus-one guarantees positive gaps. At most 127 gaps of at
        // most 2^25 sum to less than 2^32: a wrap would put the endpoint
        // below the start. Ordered matching endpoints prove every prefix.
        // Full blocks' reserved first gap is checked before this call.
        debug_assert!(docs.len() <= BLOCK_SIZE);
        true
    } else if let Some(gaps) = byte_gaps {
        // At most 127 byte-sized gaps: their sum cannot wrap a u32 more than
        // once. Nonzero gaps and matching ordered endpoints therefore prove
        // strict ordering without re-reading the decoded u32 array.
        debug_assert_eq!(gaps.len() + 1, docs.len());
        let mut nonzero = true;
        for &gap in gaps {
            nonzero &= gap != 0;
        }
        nonzero
    } else {
        let mut ordered = true;
        for pair in docs.windows(2) {
            ordered &= pair[0] < pair[1];
        }
        ordered
    };
    ordered
        && first <= last
        && last != TERMINATED
        && docs.first() == Some(&first)
        && docs.last() == Some(&last)
}

/// Write a compact L0 entry.
#[inline]
fn write_l0(buf: &mut Vec<u8>, first_doc: u32, last_doc: u32, offset: u32, bounds: u32) {
    buf.extend_from_slice(&first_doc.to_le_bytes());
    buf.extend_from_slice(&last_doc.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(&bounds.to_le_bytes());
}

/// Byte length of block `idx` from the L0 offsets: the next block's offset
/// (or the stream end) minus this block's offset. Header-independent, so a
/// block payload may carry codec-specific variable-length data.
#[inline]
fn block_len_from_l0(l0_bytes: &[u8], l0_count: usize, stream_len: usize, idx: usize) -> usize {
    let (_, _, offset, _) = read_l0(l0_bytes, idx);
    let end = if idx + 1 < l0_count {
        read_l0(l0_bytes, idx + 1).2 as usize
    } else {
        stream_len
    };
    end.saturating_sub(offset as usize)
}

/// Encoded doc-id delta array and tf array of one block, with the header
/// width bytes to store for them.
struct EncodedBlock {
    doc_bits: u8,
    tf_bits: u8,
}

/// Append the packed arrays of one block to `stream` using `codec`.
fn encode_block_arrays(
    codec: PostingCodec,
    deltas: &[u32],
    tfs: &[u32],
    stream: &mut Vec<u8>,
) -> EncodedBlock {
    let codec = codec.for_count(tfs.len());
    match codec {
        PostingCodec::Simd4x => EncodedBlock {
            doc_bits: codec.header_byte(bitpacking4x::encode_gaps(deltas, stream)),
            tf_bits: bitpacking4x::encode(tfs, stream),
        },
        PostingCodec::Rounded => {
            let max_delta = deltas.iter().copied().max().unwrap_or(0);
            let doc_bits = simd::round_bit_width(simd::bits_needed(max_delta));
            let max_tf = tfs.iter().copied().max().unwrap_or(0);
            let tf_bits = simd::round_bit_width(simd::bits_needed(max_tf));
            if !deltas.is_empty() {
                let rounded = simd::RoundedBitWidth::from_u8(doc_bits);
                let start = stream.len();
                stream.resize(start + deltas.len() * rounded.bytes_per_value(), 0);
                simd::pack_rounded(deltas, rounded, &mut stream[start..]);
            }
            {
                let rounded = simd::RoundedBitWidth::from_u8(tf_bits);
                let start = stream.len();
                stream.resize(start + tfs.len() * rounded.bytes_per_value(), 0);
                simd::pack_rounded(tfs, rounded, &mut stream[start..]);
            }
            EncodedBlock {
                doc_bits: codec.header_byte(doc_bits),
                tf_bits,
            }
        }
        PostingCodec::Packed => {
            let max_delta = deltas.iter().copied().max().unwrap_or(0);
            let doc_bits = simd::bits_needed(max_delta);
            let max_tf = tfs.iter().copied().max().unwrap_or(0);
            let tf_bits = simd::bits_needed(max_tf);
            pack_bits(deltas, doc_bits, stream);
            pack_bits(tfs, tf_bits, stream);
            EncodedBlock {
                doc_bits: codec.header_byte(doc_bits),
                tf_bits,
            }
        }
        PostingCodec::Pfor => {
            let doc_bits = if deltas.is_empty() {
                0
            } else {
                pack_pfor(deltas, stream)
            };
            let tf_bits = pack_pfor(tfs, stream);
            EncodedBlock {
                doc_bits: codec.header_byte(doc_bits),
                tf_bits,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockPostingList {
    compact_headers: bool,
    short_cursors: bool,
    /// Explicit deserialization checks decoded ordering; segment queries trust the writer.
    verify_content: bool,
    /// First decoding failure detected in this immutable segment reader.
    content_error: Option<std::sync::Arc<reader::PostingIntegrity>>,
    /// Block data stream (packed blocks laid out sequentially).
    stream: OwnedBytes,
    /// Level-0 skip entries: `(first_doc, last_doc, offset, max_weight)` × `l0_count`.
    /// 16 bytes per entry, followed by compact descriptors when present.
    /// Supports O(1) random access without another reference-counted slice.
    l0_bytes: OwnedBytes,
    /// Number of blocks (= number of L0 entries).
    l0_count: usize,
    /// Level-1 skip `last_doc` values — one per `L1_INTERVAL` blocks.
    /// Borrowed little-endian words; opening does not copy the group directory.
    l1_docs: GroupWords,
    /// Packed `(max_tf, min_len)` per L1 group (superblock bounds); empty
    /// for legacy lists.
    l1_bounds: GroupWords,
    /// Optional L0 then L1 length/TF ratio minima, borrowed from index bytes.
    ratios: Option<OwnedBytes>,
    /// Validated borrowed offsets and compact integer envelope records.
    impacts: Option<ImpactTable>,
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
        Self::build(list, false, None, PostingCodec::Rounded, false, false)
    }

    /// Build a list using an explicit per-block codec.
    pub fn from_posting_list_with_codec(
        list: &PostingList,
        codec: PostingCodec,
    ) -> io::Result<Self> {
        Self::build(list, false, None, codec, false, false)
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
        Self::build(
            list,
            with_positions,
            length_of,
            PostingCodec::Rounded,
            false,
            false,
        )
    }

    /// Build with the complete physical layout policy used by index writers.
    pub fn from_posting_list_with_options(
        list: &PostingList,
        with_positions: bool,
        length_of: Option<&dyn Fn(DocId) -> u32>,
        codec: PostingCodec,
    ) -> io::Result<Self> {
        Self::build(list, with_positions, length_of, codec, false, false)
    }

    /// Build optional score-independent ratio bounds, retaining the existing codec.
    pub fn from_posting_list_with_ratio_bounds(
        list: &PostingList,
        with_positions: bool,
        length_of: Option<&dyn Fn(DocId) -> u32>,
        codec: PostingCodec,
    ) -> io::Result<Self> {
        Self::build(list, with_positions, length_of, codec, true, false)
    }

    /// Build complete, bounded frequency/length envelopes for multi-block lists.
    /// Implies ratio bounds and requires the effective scoring-length callback.
    pub fn from_posting_list_with_impact_bounds(
        list: &PostingList,
        with_positions: bool,
        length_of: Option<&dyn Fn(DocId) -> u32>,
        codec: PostingCodec,
    ) -> io::Result<Self> {
        if length_of.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "impact bounds require scoring lengths",
            ));
        }
        Self::build(list, with_positions, length_of, codec, true, true)
    }

    fn build(
        list: &PostingList,
        with_positions: bool,
        length_of: Option<&dyn Fn(DocId) -> u32>,
        codec: PostingCodec,
        ratio_bounds: bool,
        impact_bounds: bool,
    ) -> io::Result<Self> {
        // Persisted scoring lengths saturate at `MAX_CHUNK_LENGTH`. A ratio or
        // envelope derived from a longer raw length would be over-tight, so
        // the constructor caps here rather than trusting every caller to.
        let capped = |id: DocId| {
            length_of
                .map_or(0, |length_of| length_of(id))
                .min(crate::segment::chunk_map::MAX_CHUNK_LENGTH)
        };
        let length_of: Option<&dyn Fn(DocId) -> u32> = if ratio_bounds {
            length_of.map(|_| &capped as &dyn Fn(DocId) -> u32)
        } else {
            length_of
        };
        let mut ratios = (ratio_bounds && length_of.is_some()).then(Vec::new);
        let mut impacts = (impact_bounds && length_of.is_some() && list.len() > BLOCK_SIZE)
            .then(|| ImpactBuilder::with_groups(list.len().div_ceil(BLOCK_SIZE)))
            .transpose()?;
        let mut points = [(0u32, 0u32); BLOCK_SIZE];
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

            // Collect TFs
            tf_buf.clear();
            tf_buf.extend(block.iter().map(|p| p.term_freq));

            // Write 8-byte header: [count: u16][first_doc: u32][doc_bits: u8][tf_bits: u8]
            // (`doc_bits` carries the codec id in its top two bits); the
            // packed arrays follow.
            stream.write_u16::<LittleEndian>(count as u16)?;
            stream.write_u32::<LittleEndian>(base_doc_id)?;
            let header_at = stream.len();
            stream.push(0);
            stream.push(0);
            let encoded = encode_block_arrays(codec, &deltas, &tf_buf, &mut stream);
            stream[header_at] = encoded.doc_bits;
            stream[header_at + 1] = encoded.tf_bits;

            // L0 skip entry with the block's bounds
            let block_min_len = length_of.map_or(1, |length_of| {
                block
                    .iter()
                    .map(|p| length_of(p.doc_id).max(1))
                    .min()
                    .unwrap_or(1)
            });
            list_min_len = list_min_len.min(block_min_len);
            if let Some(ratios) = &mut ratios {
                let ratio = block
                    .iter()
                    .map(|p| lower_length_ratio(length_of.unwrap()(p.doc_id).max(1), p.term_freq))
                    .fold(f32::INFINITY, f32::min);
                ratios.extend_from_slice(&ratio.to_le_bytes());
            }
            if let Some(impacts) = &mut impacts {
                for (point, posting) in points.iter_mut().zip(block) {
                    let length = length_of.unwrap()(posting.doc_id);
                    *point = (
                        posting.term_freq,
                        if length == 0 {
                            posting.term_freq
                        } else {
                            length
                        },
                    );
                }
                impacts.append_points(&mut points[..count])?;
            }
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
        if let Some(ratios) = &mut ratios {
            append_ratio_groups(ratios, l0_count);
        }

        Ok(Self {
            compact_headers: false,
            short_cursors: false,
            verify_content: true,
            content_error: None,
            stream: OwnedBytes::new(stream),
            l0_bytes: OwnedBytes::new(l0_buf),
            l0_count,
            l1_docs: l1_docs.into(),
            l1_bounds: l1_bounds.into(),
            ratios: ratios.map(OwnedBytes::new),
            impacts: if let Some(mut impacts) = impacts {
                impacts.append_groups_with(l0_count, |_| Ok(None))?;
                impacts.finish()
            } else {
                None
            },
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
        self.serialize_layout(writer, self.compact_headers)
    }

    /// Representation policy retained by explicit field reordering.
    #[cfg(all(feature = "native", test))]
    pub(crate) fn has_compact_headers(&self) -> bool {
        self.compact_headers
    }

    /// Separate fixed-width block metadata from payload pages. Pfor retains
    /// its existing framing because exception structure lives in the payload.
    pub fn serialize_compact<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let compact =
            (0..self.num_blocks()).all(|i| self.block_codec(i) != Some(PostingCodec::Pfor));
        self.serialize_layout(writer, compact)
    }

    fn serialize_layout<W: Write>(&self, writer: &mut W, compact: bool) -> io::Result<()> {
        let source_header = if self.compact_headers { 0 } else { 8 };
        let output_header = if compact { 0 } else { 8 };
        let output_offset =
            |offset: usize, block: usize| offset - block * source_header + block * output_header;
        let stream_len = output_offset(self.stream.len(), self.num_blocks());
        if compact == self.compact_headers {
            writer.write_all(&self.stream)?;
        } else {
            for i in 0..self.num_blocks() {
                if !compact {
                    writer.write_all(&self.block_header(i))?;
                }
                writer.write_all(self.block_payload(i))?;
            }
        }
        for i in 0..self.num_blocks() {
            let (first, last, offset, bounds) = self.read_l0_entry(i);
            writer.write_u32::<LittleEndian>(first)?;
            writer.write_u32::<LittleEndian>(last)?;
            writer.write_u32::<LittleEndian>(
                u32::try_from(output_offset(offset as usize, i))
                    .map_err(|_| io::Error::other("posting stream offset overflow"))?,
            )?;
            writer.write_u32::<LittleEndian>(bounds)?;
        }
        if compact {
            for i in 0..self.num_blocks() {
                let header = self.block_header(i);
                writer.write_all(&header[..2])?;
                writer.write_all(&header[6..])?;
            }
        }
        writer.write_all(self.l1_docs.bytes())?;
        writer.write_all(self.l1_bounds.bytes())?;
        let short_cursors =
            compact && self.pos_cursors.is_some() && self.total_positions <= u32::MAX as u64;
        if self.pos_cursors.is_some() {
            for i in 0..self.num_blocks() {
                let cursor = self.pos_cursor(i).unwrap();
                if short_cursors {
                    writer.write_u32::<LittleEndian>(cursor as u32)?;
                } else {
                    writer.write_u64::<LittleEndian>(cursor)?;
                }
            }
        }
        if let Some(ratios) = &self.ratios {
            writer.write_all(ratios)?;
        }
        if let Some(impacts) = &self.impacts {
            writer.write_all(impacts.bytes())?;
        }
        Self::write_footer(
            writer,
            stream_len as u64,
            self.l0_count,
            self.l1_docs.len(),
            self.doc_count,
            self.max_tf,
            self.total_positions,
            self.pos_cursors.is_some(),
            self.len_bounds.then_some(self.min_len),
            !self.l1_bounds.is_empty(),
            self.ratios.is_some(),
            self.impacts.is_some(),
            self.has_group_impact_bounds(),
            if compact { FLAG_COMPACT_HEADERS } else { 0 }
                | if short_cursors { FLAG_SHORT_CURSORS } else { 0 },
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
        ratio_bounds: bool,
        impact_bounds: bool,
        group_impact_bounds: bool,
        layout_flags: u32,
    ) -> io::Result<()> {
        writer.write_u64::<LittleEndian>(stream_len)?;
        writer.write_u32::<LittleEndian>(l0_count as u32)?;
        writer.write_u32::<LittleEndian>(l1_count as u32)?;
        writer.write_u32::<LittleEndian>(doc_count)?;
        writer.write_u32::<LittleEndian>(max_tf)?;
        writer.write_u64::<LittleEndian>(total_positions)?;
        let mut flags = layout_flags;
        if has_cursors {
            flags |= FLAG_POS_CURSORS;
        }
        if min_len.is_some() {
            flags |= FLAG_LEN_BOUNDS;
        }
        if l1_bounds {
            flags |= FLAG_L1_BOUNDS;
        }
        if ratio_bounds {
            flags |= FLAG_RATIO_BOUNDS;
        }
        if impact_bounds {
            flags |= FLAG_IMPACT_BOUNDS;
        }
        if group_impact_bounds {
            flags |= FLAG_GROUP_IMPACT_BOUNDS;
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
    /// Stream, L0, L1 and cursors are sliced from the source without copying.
    pub fn deserialize_zero_copy(raw: OwnedBytes) -> io::Result<Self> {
        let footer = Self::validate_bytes(&raw)?;
        Ok(Self::from_layout(raw, footer))
    }

    fn validate_bytes(raw: &[u8]) -> io::Result<Footer> {
        let footer = Footer::parse(raw)?;
        validation::validate_list(raw, &footer)?;
        if footer.ratio_bounds {
            validate_ratios(&raw[footer.cursors_end()..footer.ratios_end()])?;
        }
        if footer.impact_bounds {
            ImpactTable::validate(
                &raw[footer.ratios_end()..raw.len() - FOOTER_V2_SIZE],
                footer.impact_record_count(),
            )?;
        }
        Ok(footer)
    }

    // The footer proves section extents. The owning query reader trusts interior
    // contents; explicit deserialization additionally validates them.
    fn from_layout(raw: OwnedBytes, footer: Footer) -> Self {
        let ratios = footer
            .ratio_bounds
            .then(|| raw.slice(footer.cursors_end()..footer.ratios_end()));
        let l1_docs = GroupWords::borrowed(raw.slice(footer.l1_start()..footer.l1_end()));
        let l1_bounds = GroupWords::borrowed(raw.slice(footer.l1_end()..footer.l1_bounds_end()));
        let pos_cursors = footer
            .has_cursors
            .then(|| raw.slice(footer.l1_bounds_end()..footer.cursors_end()));

        Self {
            compact_headers: footer.compact_headers,
            short_cursors: footer.short_cursors,
            verify_content: true,
            content_error: None,
            stream: raw.slice(0..footer.stream_len),
            l0_bytes: raw.slice(footer.l0_start()..footer.l1_start()),
            l0_count: footer.l0_count,
            l1_docs,
            l1_bounds,
            ratios,
            impacts: footer.impact_bounds.then(|| {
                ImpactTable::from_validated(
                    raw.slice(footer.ratios_end()..raw.len() - FOOTER_V2_SIZE),
                    footer.impact_record_count(),
                )
            }),
            doc_count: footer.doc_count,
            max_tf: footer.max_tf,
            pos_cursors,
            total_positions: footer.total_positions,
            len_bounds: footer.len_bounds,
            min_len: footer.min_len,
        }
    }

    /// Minimum scoring-unit length over the list, when the list stores
    /// length bounds (`None` for legacy lists).
    pub fn min_len(&self) -> Option<u32> {
        self.len_bounds.then_some(self.min_len)
    }

    /// Whether optional ratio bounds are present (zero entries mean unknown).
    pub fn has_ratio_bounds(&self) -> bool {
        self.ratios.is_some()
    }

    /// Whether the list stores an impact directory, including unknown records.
    pub fn has_impact_bounds(&self) -> bool {
        self.impacts.is_some()
    }

    /// Envelope diagnostics: absent directory/out-of-range is `None`; an unknown
    /// record is `Some(0)`; a populated record has 1–8 complete frontier points.
    pub fn block_impact_point_count(&self, block: usize) -> Option<usize> {
        if block >= self.l0_count {
            return None;
        }
        self.impacts
            .as_ref()?
            .record(block)
            .map(|r| r.first().copied().unwrap_or(0) as usize)
    }

    /// Whether the optional table also stores group envelopes.
    pub fn has_group_impact_bounds(&self) -> bool {
        self.impacts
            .as_ref()
            .is_some_and(|t| t.record_count() > self.l0_count)
    }

    /// Point count for the group containing this block; zero means unknown.
    #[cfg(test)]
    pub fn group_impact_point_count(&self, block: usize) -> Option<usize> {
        if block >= self.l0_count {
            return None;
        }
        self.impacts
            .as_ref()?
            .record(self.l0_count + block / L1_INTERVAL)
            .map(|r| r.first().copied().unwrap_or(0) as usize)
    }

    pub(crate) fn group_impact_minimum(
        &self,
        block: usize,
        reciprocal: f64,
        ratio: f64,
    ) -> Option<f64> {
        if block >= self.l0_count {
            return None;
        }
        self.impacts
            .as_ref()?
            .minimum(self.l0_count + block / L1_INTERVAL, reciprocal, ratio)
    }

    pub(crate) fn block_impact_minimum(
        &self,
        block: usize,
        reciprocal: f64,
        ratio: f64,
    ) -> Option<f64> {
        if block >= self.l0_count {
            return None;
        }
        self.impacts.as_ref()?.minimum(block, reciprocal, ratio)
    }

    /// Conservative minimum length/TF for a block, or zero if unavailable.
    pub(crate) fn block_length_ratio(&self, block: usize) -> f32 {
        self.ratios
            .as_ref()
            .map_or(0.0, |ratios| read_ratio(ratios, block))
    }

    pub(crate) fn group_length_ratio(&self, block: usize) -> f32 {
        self.ratios.as_ref().map_or(0.0, |ratios| {
            read_ratio(ratios, self.l0_count + block / L1_INTERVAL)
        })
    }

    /// `(max_tf, min_len)` of a block; `min_len` is `None` for legacy lists.
    #[inline]
    pub fn block_bounds(&self, block_idx: usize) -> Option<(u32, Option<u32>)> {
        if block_idx >= self.l0_count {
            return None;
        }
        let (_, _, _, word) = self.read_l0_entry(block_idx);
        let (max_tf, min_len) = unpack_bounds(word, self.len_bounds);
        // A packed maximum can saturate; the full-width list maximum remains
        // a conservative bound. Never interpret saturation as an actual TF.
        let max_tf = if self.len_bounds && max_tf == u16::MAX as u32 {
            max_tf.max(self.max_tf)
        } else {
            max_tf
        };
        Some((max_tf, min_len))
    }

    /// `(max_tf, min_len)` over the L1 group (`L1_INTERVAL` blocks) that
    /// contains `block_idx`; `None` for legacy lists without group bounds.
    #[inline]
    pub fn group_bounds(&self, block_idx: usize) -> Option<(u32, u32)> {
        if block_idx >= self.l0_count {
            return None;
        }
        let word = self.l1_bounds.get(block_idx / L1_INTERVAL)?;
        let (max_tf, min_len) = unpack_bounds(word, true);
        let max_tf = if max_tf == u16::MAX as u32 {
            max_tf.max(self.max_tf)
        } else {
            max_tf
        };
        Some((max_tf, min_len.unwrap_or(1)))
    }

    /// Last doc of the L1 group containing `block_idx`.
    #[inline]
    pub fn group_last_doc(&self, block_idx: usize) -> Option<DocId> {
        self.l1_docs.get(block_idx / L1_INTERVAL)
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

    #[inline]
    fn block_header(&self, block: usize) -> [u8; 8] {
        let (first, _, offset, _) = self.read_l0_entry(block);
        if self.compact_headers {
            let headers = &self.l0_bytes[self.l0_count * L0_SIZE..];
            let mut header = [0; 8];
            header[..2].copy_from_slice(&headers[block * 4..block * 4 + 2]);
            header[2..6].copy_from_slice(&first.to_le_bytes());
            header[6..].copy_from_slice(&headers[block * 4 + 2..block * 4 + 4]);
            header
        } else {
            self.stream[offset as usize..offset as usize + 8]
                .try_into()
                .unwrap()
        }
    }

    #[inline]
    fn block_payload(&self, block: usize) -> &[u8] {
        let offset = self.read_l0_entry(block).2 as usize;
        let header = if self.compact_headers { 0 } else { 8 };
        &self.stream[offset + header..offset + self.block_len(block)]
    }

    /// Number of values in the term's position stream (0 without cursors).
    pub fn total_positions(&self) -> u64 {
        self.total_positions
    }

    /// Values in the term's position stream before block `block_idx`.
    #[inline]
    pub fn pos_cursor(&self, block_idx: usize) -> Option<u64> {
        let cursors = self.pos_cursors.as_ref()?;
        let size = if self.short_cursors { 4 } else { CURSOR_SIZE };
        let p = block_idx * size;
        cursors.get(p..p + size).map(|b| {
            if self.short_cursors {
                u64::from(u32::from_le_bytes(b.try_into().unwrap()))
            } else {
                u64::from_le_bytes(b.try_into().unwrap())
            }
        })
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
        // Admission precedes all output allocation/copying; typed sources have
        // already validated payloads, but their requested rebasing is new.
        let mut previous_last = None;
        let mut total_docs = 0u32;
        let mut total_positions = 0u64;
        for (source, offset) in sources {
            validation::validate_remap(
                &source.l0_bytes,
                source.l0_count,
                *offset,
                &mut previous_last,
            )?;
            total_docs = total_docs.checked_add(source.doc_count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "merged posting count overflow")
            })?;
            total_positions = total_positions
                .checked_add(source.total_positions)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "merged position cursor overflow",
                    )
                })?;
        }
        let mut stream: Vec<u8> = Vec::new();
        let mut l0_buf: Vec<u8> = Vec::new();
        let mut l1_docs: Vec<u32> = Vec::new();
        let mut l0_count = 0usize;
        let mut ratios = sources
            .iter()
            .any(|(s, _)| s.has_ratio_bounds())
            .then(Vec::new);
        let mut impacts = if sources.iter().any(|(s, _)| s.has_impact_bounds()) {
            let blocks = sources
                .iter()
                .try_fold(0usize, |n, (s, _)| n.checked_add(s.num_blocks()))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "impact block count overflow")
                })?;
            Some(ImpactBuilder::with_groups(blocks)?)
        } else {
            None
        };
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
                if let Some(impacts) = &mut impacts {
                    impacts.append(
                        source
                            .impacts
                            .as_ref()
                            .and_then(|t| t.record(block_idx))
                            .unwrap_or(&[]),
                    )?;
                }
                if let Some(ratios) = &mut ratios {
                    ratios.extend_from_slice(&source.block_length_ratio(block_idx).to_le_bytes());
                }
                if all_cursors {
                    let cursor = source.pos_cursor(block_idx).unwrap_or(0) + positions_before;
                    cursors.extend_from_slice(&cursor.to_le_bytes());
                }
                let (first_doc, last_doc, _, word) = source.read_l0_entry(block_idx);
                let (block_max_tf, block_min_len) = unpack_bounds(word, source.len_bounds);
                let bounds = pack_bounds(block_max_tf, block_min_len.unwrap_or(1));
                let header = source.block_header(block_idx);
                let count = u16::from_le_bytes(header[..2].try_into().unwrap());
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
                stream.extend_from_slice(&header[6..]);
                stream.extend_from_slice(source.block_payload(block_idx));

                let new_last = last_doc + doc_offset;
                write_l0(
                    &mut l0_buf,
                    first_doc + doc_offset,
                    new_last,
                    new_offset,
                    bounds,
                );
                l0_count += 1;

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
        if let Some(ratios) = &mut ratios {
            append_ratio_groups(ratios, l0_count);
        }

        Ok(Self {
            compact_headers: false,
            short_cursors: false,
            verify_content: true,
            content_error: None,
            stream: OwnedBytes::new(stream),
            l0_bytes: OwnedBytes::new(l0_buf),
            l0_count,
            l1_docs: l1_docs.into(),
            l1_bounds: l1_bounds.into(),
            ratios: ratios.map(OwnedBytes::new),
            impacts: if let Some(mut impacts) = impacts {
                let mut source = 0;
                let mut base = 0;
                impacts.append_groups_with(l0_count, |range| {
                    while base + sources[source].0.num_blocks() <= range.start {
                        base += sources[source].0.num_blocks();
                        source += 1;
                    }
                    let list = &sources[source].0;
                    let local = range.start - base;
                    let record = if local.is_multiple_of(L1_INTERVAL)
                        && range.end - base == (local + L1_INTERVAL).min(list.num_blocks())
                    {
                        list.impacts
                            .as_ref()
                            .and_then(|t| t.record(list.l0_count + local / L1_INTERVAL))
                    } else {
                        None
                    };
                    Ok(record)
                })?;
                impacts.finish()
            } else {
                None
            },
            doc_count: total_docs,
            max_tf,
            pos_cursors: all_cursors.then(|| OwnedBytes::new(cursors)),
            total_positions: if all_cursors { total_positions } else { 0 },
            len_bounds: true,
            min_len: if min_len == u32::MAX { 1 } else { min_len },
        })
    }

    /// Streaming merge: write blocks directly to output writer (bounded memory).
    ///
    /// **Zero-materializing**: reads L0 entries directly from source bytes
    /// (mmap or &[u8]) without parsing into Vecs. Block sizes come from the
    /// L0 offsets, so blocks of any codec are copied verbatim.
    ///
    /// Output L0 + L1 are buffered (bounded O(total_blocks × 16 + total_blocks/8 × 4)).
    /// Block data flows source → output writer without intermediate buffering.
    ///
    /// Returns `(doc_count, bytes_written)`.
    ///
    /// Preflights all source headers/directories and remapped document ranges
    /// before writing. Corrupt sources, overlapping ranges, or arithmetic
    /// overflow return `Error::Corruption`, including the single-source copy path.
    pub fn concatenate_streaming<W: Write>(
        sources: &[(&[u8], u32)], // (serialized_bytes, doc_offset)
        writer: &mut W,
    ) -> crate::Result<(u32, usize)> {
        let mut metas: Vec<Footer> = Vec::with_capacity(sources.len());
        let mut total_docs = 0u32;
        let mut merged_max_tf = 0u32;
        let mut merged_min_len = u32::MAX;
        let mut previous_last = None;
        let mut total_positions = 0u64;

        for (source_index, (raw, offset)) in sources.iter().enumerate() {
            let invalid_source = |error: io::Error| {
                crate::Error::Corruption(format!(
                    "posting list source {source_index} is invalid: {error}"
                ))
            };
            // The zero-offset copy shortcut also needs validated payloads.
            // This checks headers/directories; it never decodes postings.
            let footer = Self::validate_bytes(raw).map_err(invalid_source)?;
            validation::validate_remap(
                &raw[footer.l0_start()..footer.l0_end()],
                footer.l0_count,
                *offset,
                &mut previous_last,
            )
            .map_err(invalid_source)?;
            total_docs = total_docs
                .checked_add(footer.doc_count)
                .ok_or_else(|| crate::Error::Corruption("merged posting count overflow".into()))?;
            total_positions = total_positions
                .checked_add(footer.total_positions)
                .ok_or_else(|| {
                    crate::Error::Corruption("merged position cursor overflow".into())
                })?;
            merged_max_tf = merged_max_tf.max(footer.max_tf);
            merged_min_len = merged_min_len.min(if footer.len_bounds { footer.min_len } else { 1 });
            metas.push(footer);
        }

        // The common single-source term in the first segment needs no doc-id
        // rebasing and already has a valid index/footer. Copy it wholesale.
        if sources.len() == 1 && sources[0].1 == 0 {
            writer.write_all(sources[0].0)?;
            return Ok((metas[0].doc_count, sources[0].0.len()));
        }

        let compact_output = !metas.is_empty() && metas.iter().all(|meta| meta.compact_headers);
        let short_cursors = compact_output && total_positions <= u32::MAX as u64;
        let mut out_headers = Vec::new();
        let all_cursors = metas.iter().all(|m| m.has_cursors);
        if !all_cursors && metas.iter().any(|m| m.has_cursors) {
            return Err(crate::Error::Corruption(
                "cannot concatenate posting lists with and without position cursors".into(),
            ));
        }

        // Phase 1: Stream block data, reading L0 entries on-the-fly.
        // Accumulate output L0 + L1 + cursors (bounded).
        let mut out_impacts = if metas.iter().any(|m| m.impact_bounds) {
            let blocks = metas
                .iter()
                .try_fold(0usize, |n, m| n.checked_add(m.l0_count))
                .ok_or_else(|| crate::Error::Corruption("impact block count overflow".into()))?;
            Some(ImpactBuilder::with_groups(blocks)?)
        } else {
            None
        };
        let mut out_ratios = metas.iter().any(|m| m.ratio_bounds).then(Vec::new);
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
                if let Some(impacts) = &mut out_impacts {
                    let record = if meta.impact_bounds {
                        ImpactTable::record_from_validated(
                            &raw[meta.ratios_end()..raw.len() - FOOTER_V2_SIZE],
                            meta.impact_record_count(),
                            i,
                        )
                    } else {
                        &[]
                    };
                    impacts.append(record)?;
                }
                if let Some(ratios) = &mut out_ratios {
                    let ratio = if meta.ratio_bounds {
                        read_ratio(&raw[meta.cursors_end()..], i)
                    } else {
                        0.0
                    };
                    ratios.extend_from_slice(&ratio.to_le_bytes());
                }
                // Read source L0 entry directly from raw bytes
                let (first_doc, last_doc, offset, word) = read_l0(&raw[l0_base..], i);
                let (block_max_tf, block_min_len) = unpack_bounds(word, meta.len_bounds);
                let bounds = pack_bounds(block_max_tf, block_min_len.unwrap_or(1));
                if all_cursors {
                    let size = meta.cursor_size();
                    let p = cursors_base + i * size;
                    let cursor = if meta.short_cursors {
                        u64::from(u32::from_le_bytes(raw[p..p + size].try_into().unwrap()))
                    } else {
                        u64::from_le_bytes(raw[p..p + size].try_into().unwrap())
                    };
                    if short_cursors {
                        out_cursors
                            .extend_from_slice(&((cursor + positions_before) as u32).to_le_bytes());
                    } else {
                        out_cursors.extend_from_slice(&(cursor + positions_before).to_le_bytes());
                    }
                }

                // Block size from the neighbouring L0 offset (codec-independent)
                let blk_size =
                    block_len_from_l0(&raw[l0_base..], meta.l0_count, meta.stream_len, i);
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
                let payload = if meta.compact_headers {
                    let header = &raw[meta.l0_end() + i * 4..meta.l0_end() + i * 4 + 4];
                    patch_buf[..2].copy_from_slice(&header[..2]);
                    patch_buf[6..].copy_from_slice(&header[2..]);
                    block
                } else {
                    patch_buf.copy_from_slice(&block[..8]);
                    &block[8..]
                };
                patch_buf[2..6].copy_from_slice(&(first_doc + doc_offset).to_le_bytes());
                if compact_output {
                    out_headers.extend_from_slice(&patch_buf[..2]);
                    out_headers.extend_from_slice(&patch_buf[6..]);
                } else {
                    writer.write_all(&patch_buf)?;
                }
                writer.write_all(payload)?;
                stream_written += (if compact_output { 0 } else { 8 } + payload.len()) as u64;
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
        writer.write_all(&out_headers)?;
        for &doc in &out_l1_docs {
            writer.write_u32::<LittleEndian>(doc)?;
        }
        for &bounds in &out_l1_bounds {
            writer.write_u32::<LittleEndian>(bounds)?;
        }
        writer.write_all(&out_cursors)?;
        if let Some(ratios) = &mut out_ratios {
            append_ratio_groups(ratios, out_l0_count);
            writer.write_all(ratios)?;
        }
        let out_impacts = if let Some(mut impacts) = out_impacts {
            let mut source = 0;
            let mut base = 0;
            impacts.append_groups_with(out_l0_count, |range| {
                while base + metas[source].l0_count <= range.start {
                    base += metas[source].l0_count;
                    source += 1;
                }
                let meta = &metas[source];
                let local = range.start - base;
                let record = if meta.group_impact_bounds
                    && local.is_multiple_of(L1_INTERVAL)
                    && range.end - base == (local + L1_INTERVAL).min(meta.l0_count)
                {
                    let raw = sources[source].0;
                    Some(ImpactTable::record_from_validated(
                        &raw[meta.ratios_end()..raw.len() - FOOTER_V2_SIZE],
                        meta.impact_record_count(),
                        meta.l0_count + local / L1_INTERVAL,
                    ))
                } else {
                    None
                };
                Ok(record)
            })?;
            impacts.finish()
        } else {
            None
        };
        if let Some(impacts) = &out_impacts {
            writer.write_all(impacts.bytes())?;
        }
        Self::write_footer(
            writer,
            stream_written,
            out_l0_count,
            out_l1_docs.len(),
            total_docs,
            merged_max_tf,
            if all_cursors { total_positions } else { 0 },
            all_cursors,
            Some(if merged_min_len == u32::MAX {
                1
            } else {
                merged_min_len
            }),
            true,
            out_ratios.is_some(),
            out_impacts.is_some(),
            out_impacts.is_some(),
            if compact_output {
                FLAG_COMPACT_HEADERS
            } else {
                0
            } | if short_cursors && all_cursors {
                FLAG_SHORT_CURSORS
            } else {
                0
            },
        )?;

        let l1_bytes_len = out_l1_docs.len() * L1_SIZE + out_l1_bounds.len() * 4;
        let total_bytes = stream_written as usize
            + out_l0.len()
            + out_headers.len()
            + l1_bytes_len
            + out_cursors.len()
            + out_ratios.as_ref().map_or(0, Vec::len)
            + out_impacts.as_ref().map_or(0, |t| t.bytes().len())
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
    /// or `None` if block_idx is out of range. A block whose decoded ids
    /// disagree with the L0 directory (content corruption that structural
    /// admission cannot see) also yields `None`, after an error log naming
    /// the block; callers treat it as the end of the list.
    pub fn decode_block_doc_ids_only(
        &self,
        block_idx: usize,
        doc_ids: &mut Vec<u32>,
    ) -> Option<(usize, usize, usize)> {
        match self.decode_block_doc_ids_checked(block_idx, doc_ids) {
            Ok(state) => state,
            Err(error) => {
                log::error!(
                    "posting block {block_idx} of {} is corrupt; the cursor ends here: {error}",
                    self.l0_count
                );
                None
            }
        }
    }

    /// `Ok(None)` is out of range; `Err` is a block whose payload does not
    /// match its directory entry. On error `doc_ids` is left empty.
    fn decode_block_doc_ids_checked(
        &self,
        block_idx: usize,
        doc_ids: &mut Vec<u32>,
    ) -> io::Result<Option<(usize, usize, usize)>> {
        if block_idx >= self.l0_count {
            return Ok(None);
        }
        let decoded = (|| {
            let (first, last, offset, _) = self.read_l0_entry(block_idx);
            let header = self.block_header(block_idx);
            let payload = self.block_payload(block_idx);
            let state = self.decode_block_doc_ids_unchecked(
                offset as usize,
                block_idx,
                header,
                payload,
                doc_ids,
            )?;
            if self.verify_content {
                // Header 8 denotes Rounded with 8-bit raw gaps (no codec tag).
                let byte_gaps = (header[6] == 8).then(|| &payload[..doc_ids.len() - 1]);
                if header[6] >> PostingCodec::HEADER_SHIFT == PostingCodec::Simd4x as u8
                    && doc_ids.len() == BLOCK_SIZE
                    && !bitpacking4x::first_gap_is_zero(
                        payload,
                        header[6] & PostingCodec::WIDTH_MASK,
                    )
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "SIMD posting first gap must be zero",
                    ));
                }
                let strict_gap_width = (header[6] >> PostingCodec::HEADER_SHIFT
                    == PostingCodec::Simd4x as u8)
                    .then_some(header[6] & PostingCodec::WIDTH_MASK);
                if !verify_block_docs(doc_ids, first, last, byte_gaps, strict_gap_width) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("decoded doc ids leave the directory range {first}..={last}"),
                    ));
                }
            }
            crate::observe::search_work!(
                doc_blocks += 1,
                doc_values += doc_ids.len(),
                doc_payload_bytes += state.1 - if self.compact_headers { 0 } else { 8 }
            );
            Ok(Some(state))
        })();
        if decoded.is_err() {
            doc_ids.clear();
            if let Some(error) = &self.content_error {
                // Write-once: no query may erase another query's failure.
                error.record(block_idx);
            }
        }
        decoded
    }

    fn decode_block_doc_ids_unchecked(
        &self,
        pos: usize,
        block_idx: usize,
        header: [u8; 8],
        payload: &[u8],
        doc_ids: &mut Vec<u32>,
    ) -> io::Result<(usize, usize, usize)> {
        let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
        let count = u16::from_le_bytes(header[..2].try_into().unwrap()) as usize;
        let first_doc = u32::from_le_bytes(header[2..6].try_into().unwrap());
        let (codec, doc_width) = PostingCodec::from_header_byte(header[6])?;
        let header_len = if self.compact_headers { 0 } else { 8 };
        let state = if self.compact_headers { block_idx } else { pos };
        if count == 0 || count > BLOCK_SIZE {
            return Err(invalid("invalid posting block count"));
        }

        // Every decoder overwrites the complete output; retain initialized
        // storage so equal-sized blocks do not need a redundant zero pass.
        doc_ids.resize(count, 0);
        doc_ids[0] = first_doc;

        if codec == PostingCodec::Simd4x {
            let values = if count == BLOCK_SIZE {
                count
            } else {
                count - 1
            };
            let bytes = bitpacking4x::encoded_len(values, doc_width);
            bitpacking4x::decode_docs(&payload[..bytes], doc_width, first_doc, doc_ids);
            return Ok((state, header_len + bytes, count));
        }
        let deltas_bytes = if count > 1 {
            match codec {
                PostingCodec::Rounded => {
                    let rounded = simd::RoundedBitWidth::try_from_u8(doc_width)
                        .ok_or_else(|| invalid("invalid rounded posting width"))?;
                    let bytes = (count - 1) * rounded.bytes_per_value();
                    simd::unpack_rounded_raw_delta_decode(
                        &payload[..bytes],
                        rounded,
                        doc_ids,
                        first_doc,
                        count,
                    );
                    return Ok((state, header_len + bytes, count));
                }
                PostingCodec::Packed => {
                    let bytes = packed_bytes(count - 1, doc_width);
                    unpack_bits(&payload[..bytes], doc_width, &mut doc_ids[1..], count - 1);
                    bytes
                }
                PostingCodec::Pfor => {
                    let bytes = pfor_payload_len(payload, count - 1, doc_width)?;
                    unpack_pfor(&payload[..bytes], doc_width, &mut doc_ids[1..], count - 1)?;
                    bytes
                }
                PostingCodec::Simd4x => unreachable!("SIMD documents were decoded above"),
            }
        } else {
            0
        };
        for i in 1..count {
            doc_ids[i] = doc_ids[i].wrapping_add(doc_ids[i - 1]);
        }

        let tfs_start = header_len + deltas_bytes;
        Ok((state, tfs_start, count))
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
        tfs.resize(count, 0);
        self.decode_block_tfs_slice(block_offset, tf_start, tfs);
    }

    /// Shared frequency decoder for vector and fixed-block consumers.
    fn decode_block_tfs_slice(&self, block_offset: usize, tf_start: usize, tfs: &mut [u32]) {
        let count = tfs.len();
        let (header, block_data) = if self.compact_headers {
            (
                self.block_header(block_offset),
                self.block_payload(block_offset),
            )
        } else {
            (
                self.stream[block_offset..block_offset + 8]
                    .try_into()
                    .unwrap(),
                &self.stream[block_offset..],
            )
        };
        let (codec, _) = PostingCodec::from_header_byte(header[6]).expect("admitted posting codec");
        let tf_bits = header[7];
        let payload = &block_data[tf_start..];
        crate::observe::search_work!(
            tf_blocks += 1,
            tf_values += count,
            tf_payload_bytes += match codec {
                PostingCodec::Pfor =>
                    pfor_payload_len(payload, count, tf_bits).expect("admitted frequency payload"),
                _ => packed_bytes(count, tf_bits),
            }
        );
        match codec {
            PostingCodec::Simd4x => {
                bitpacking4x::decode(&payload[..packed_bytes(count, tf_bits)], tf_bits, tfs);
            }
            PostingCodec::Rounded => {
                let rounded = simd::RoundedBitWidth::try_from_u8(tf_bits)
                    .expect("invalid rounded posting frequency width");
                simd::unpack_rounded(
                    &payload[..count * rounded.bytes_per_value()],
                    rounded,
                    tfs,
                    count,
                );
            }
            PostingCodec::Packed => {
                unpack_bits(
                    &payload[..packed_bytes(count, tf_bits)],
                    tf_bits,
                    tfs,
                    count,
                );
            }
            PostingCodec::Pfor => {
                let len = pfor_payload_len(payload, count, tf_bits)
                    .expect("invalid patched posting frequency payload");
                unpack_pfor(&payload[..len], tf_bits, tfs, count)
                    .expect("invalid patched posting frequency table");
            }
        }
    }

    /// Byte length of block `block_idx` (from the L0 offsets).
    #[inline]
    fn block_len(&self, block_idx: usize) -> usize {
        block_len_from_l0(&self.l0_bytes, self.l0_count, self.stream.len(), block_idx)
    }

    /// Codec of block `block_idx` (diagnostics).
    pub fn block_codec(&self, block_idx: usize) -> Option<PostingCodec> {
        if block_idx >= self.l0_count {
            return None;
        }
        PostingCodec::from_header_byte(self.block_header(block_idx)[6])
            .ok()
            .map(|(codec, _)| codec)
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
    /// Checks the current block first, then gallops over the L1 group ends
    /// before a bounded L0 search. Near seeks are constant time; distant
    /// seeks inspect logarithmically many groups rather than scanning the gap.
    ///
    /// Returns `None` if no block contains `target`.
    pub fn seek_block(&self, target: DocId, from_block: usize) -> Option<usize> {
        if from_block >= self.l0_count {
            return None;
        }

        if self
            .block_last_doc(from_block)
            .is_some_and(|last| last >= target)
        {
            return Some(from_block);
        }
        let from_l1 = from_block / L1_INTERVAL;
        let groups = self.l1_docs.words().get(from_l1..)?;
        let first = u32::from_le_bytes(*groups.first()?);
        let offset = if first >= target {
            0
        } else {
            let mut bound = 1usize;
            while bound < groups.len() && u32::from_le_bytes(groups[bound]) < target {
                bound = bound.saturating_mul(2);
            }
            let lo = bound / 2;
            let hi = bound.saturating_add(1).min(groups.len());
            lo + groups[lo..hi].partition_point(|&last| u32::from_le_bytes(last) < target)
        };
        let l1_idx = from_l1 + offset;
        if l1_idx >= self.l1_docs.len() {
            return None;
        }

        // Search the validated entries in place instead of gathering every
        // strided last ID in the group before finding the lower bound.
        let start = (l1_idx * L1_INTERVAL).max(from_block + 1);
        let end = ((l1_idx + 1) * L1_INTERVAL).min(self.l0_count);
        let entries = self.l0_bytes.as_slice().as_chunks::<L0_SIZE>().0;
        let within = entries[start..end].partition_point(|entry| {
            u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]) < target
        });
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

    /// Point probes start at their first requested block and reuse decode
    /// storage. Ordinary iteration still starts at the first posting.
    pub(crate) fn into_candidate_iterator(
        self,
        first_target: DocId,
        scratch: &mut PostingDecodeScratch,
    ) -> BlockPostingIterator<'static> {
        let first_block = self.seek_block(first_target, 0);
        let PostingDecodeScratch {
            doc_ids,
            term_freqs,
        } = std::mem::take(scratch);
        let mut block_tfs = term_freqs.unwrap_or_default();
        block_tfs.take();
        let mut iterator = BlockPostingIterator {
            block_list: std::borrow::Cow::Owned(self),
            current_block: first_block.unwrap_or(0),
            block_doc_ids: doc_ids,
            block_tfs,
            tf_state: (0, 0, 0),
            position_in_block: 0,
            position_offsets: None,
            position_offsets_ready: false,
            block_position_cursor: 0,
            exhausted: first_block.is_none(),
        };
        if let Some(block) = first_block {
            iterator.load_block(block);
        }
        iterator
    }
}

#[derive(Default)]
pub(crate) struct PostingDecodeScratch {
    doc_ids: Vec<u32>,
    term_freqs: Option<std::sync::OnceLock<[u32; BLOCK_SIZE]>>,
}

/// Iterator over block posting list with skip support
/// Can be either borrowed or owned via Cow
///
/// Document IDs and frequencies have separate bounded decode buffers. Document
/// movement does not initialize frequencies; score and position consumers do.
/// Only the current block is retained, including during scratch recycling.
pub struct BlockPostingIterator<'a> {
    block_list: std::borrow::Cow<'a, BlockPostingList>,
    current_block: usize,
    block_doc_ids: Vec<u32>,
    /// Fixed inline scratch, reused across blocks and initialized only by
    /// frequency/position consumers. OnceLock preserves immutable frequency
    /// access and the iterator's native Send+Sync contract; keeping it inline
    /// avoids a heap allocation per cursor per query.
    block_tfs: std::sync::OnceLock<[u32; BLOCK_SIZE]>,
    tf_state: (usize, usize, usize),
    position_in_block: usize,
    /// Lazily computed absolute position offsets, including the one-past-end
    /// offset. One bounded block (1 KiB) replaces per-document prefix reductions.
    position_offsets: Option<Box<[u64; BLOCK_SIZE + 1]>>,
    position_offsets_ready: bool,
    block_position_cursor: u64,
    exhausted: bool,
}

impl<'a> BlockPostingIterator<'a> {
    pub(crate) fn recycle(self, scratch: &mut PostingDecodeScratch) {
        *scratch = PostingDecodeScratch {
            doc_ids: self.block_doc_ids,
            term_freqs: Some(self.block_tfs),
        };
    }

    fn new(block_list: &'a BlockPostingList) -> Self {
        let exhausted = block_list.l0_count == 0;
        let mut iter = Self {
            block_list: std::borrow::Cow::Borrowed(block_list),
            current_block: 0,
            block_doc_ids: Vec::with_capacity(BLOCK_SIZE),
            block_tfs: std::sync::OnceLock::new(),
            tf_state: (0, 0, 0),
            position_in_block: 0,
            position_offsets: None,
            position_offsets_ready: false,
            block_position_cursor: 0,
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
            block_tfs: std::sync::OnceLock::new(),
            tf_state: (0, 0, 0),
            position_in_block: 0,
            position_offsets: None,
            position_offsets_ready: false,
            block_position_cursor: 0,
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
        self.position_offsets_ready = false;
        self.block_position_cursor = self.block_list.pos_cursor(block_idx).unwrap_or(0);

        self.block_tfs.take();
        match self
            .block_list
            .decode_block_doc_ids_checked(block_idx, &mut self.block_doc_ids)
        {
            Ok(Some(state)) => self.tf_state = state,
            Ok(None) => unreachable!("block index was range-checked above"),
            Err(error) => {
                // The cursor API is infallible: end the list here rather than
                // expose ids outside the directory, and say so.
                log::error!(
                    "posting block {block_idx} of {} is corrupt; the cursor ends here: {error}",
                    self.block_list.l0_count
                );
                self.block_doc_ids.clear();
                self.tf_state = (0, 0, 0);
                self.exhausted = true;
            }
        }
    }

    fn frequencies(&self) -> &[u32] {
        let (offset, start, count) = self.tf_state;
        if count == 0 {
            return &[];
        }
        &self.block_tfs.get_or_init(|| {
            let mut tfs = [0; BLOCK_SIZE];
            self.block_list
                .decode_block_tfs_slice(offset, start, &mut tfs[..count]);
            tfs
        })[..count]
    }

    /// Offset of the current posting's positions in the term's position
    /// stream (see `structures::postings::positions_v2`): the block's cursor
    /// plus the term frequencies of the postings before it in the block.
    /// Meaningful only for lists built with position cursors.
    #[inline]
    pub fn position_cursor(&self) -> u64 {
        if self.position_offsets_ready {
            self.position_offsets.as_ref().unwrap()[self.position_in_block]
        } else {
            self.block_position_cursor
                + self.frequencies()[..self.position_in_block]
                    .iter()
                    .map(|&tf| u64::from(tf))
                    .sum::<u64>()
        }
    }

    pub(crate) fn position_cursor_mut(&mut self) -> u64 {
        if !self.position_offsets_ready {
            self.initialize_position_offsets();
        }
        self.position_offsets.as_ref().unwrap()[self.position_in_block]
    }

    /// Address and length of the current posting's positions. Prefixes are
    /// prepared once per block; document-only iteration does not initialize them.
    #[inline]
    pub(crate) fn position_range(&mut self) -> (u64, u32) {
        let cursor = self.position_cursor_mut();
        (cursor, self.frequencies()[self.position_in_block])
    }

    fn initialize_position_offsets(&mut self) {
        self.frequencies();
        let offsets = self
            .position_offsets
            .get_or_insert_with(|| Box::new([0; BLOCK_SIZE + 1]));
        let mut total = self.block_position_cursor;
        offsets[0] = total;
        if let Some(frequencies) = self.block_tfs.get() {
            for (offset, &tf) in offsets[1..].iter_mut().zip(&frequencies[..self.tf_state.2]) {
                total += u64::from(tf);
                *offset = total;
            }
        }
        self.position_offsets_ready = true;
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
        if self.exhausted || self.position_in_block >= self.block_doc_ids.len() {
            0
        } else {
            self.frequencies()[self.position_in_block]
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

    #[inline]
    pub fn seek(&mut self, target: DocId) -> DocId {
        crate::observe::search_work!(posting_seeks += 1);
        if self.exhausted {
            return TERMINATED;
        }
        let current = self.block_doc_ids[self.position_in_block];
        if target <= current {
            return current;
        }
        if target > *self.block_doc_ids.last().unwrap() {
            return self.seek_later_block(target);
        }
        // Nearby intersection probes often need just one step. Distant probes
        // use logarithmic comparisons instead of scanning the decoded gap.
        let next = self.position_in_block + 1;
        self.position_in_block = if self.block_doc_ids[next] >= target {
            next
        } else {
            let remaining = &self.block_doc_ids[next + 1..];
            let mut bound = 1;
            while bound < remaining.len() && remaining[bound] < target {
                bound *= 2;
            }
            let lo = bound / 2;
            let hi = (bound + 1).min(remaining.len());
            next + 1 + lo + remaining[lo..hi].partition_point(|&doc| doc < target)
        };
        self.block_doc_ids[self.position_in_block]
    }

    /// Select a posting from a bounded decoded suffix, leaving it parked for
    /// ordinary frequency/position reads. None yields after consuming a block.
    pub(crate) fn find_in_block(
        &mut self,
        mut find: impl FnMut(&[u32], &[u32]) -> Option<usize>,
    ) -> Option<DocId> {
        if self.exhausted {
            return Some(TERMINATED);
        }
        let start = self.position_in_block;
        if let Some(offset) = find(&self.block_doc_ids[start..], &self.frequencies()[start..]) {
            self.position_in_block += offset;
            return Some(self.block_doc_ids[self.position_in_block]);
        }
        self.load_block(self.current_block + 1);
        None
    }

    /// Reposition a physical probe without discarding its bounded decode buffers.
    /// Logical document order can move backwards through an RGB permutation.
    pub(crate) fn seek_physical(&mut self, target: DocId) -> DocId {
        if !self.exhausted && target >= self.doc() {
            return self.seek(target);
        }
        let Some(block) = self.block_list.seek_block(target, 0) else {
            self.exhausted = true;
            return TERMINATED;
        };
        self.exhausted = false;
        if block != self.current_block || self.block_doc_ids.is_empty() {
            self.load_block(block);
        }
        if self.exhausted {
            return TERMINATED;
        }
        self.position_in_block = self.block_doc_ids.partition_point(|&doc| doc < target);
        self.doc()
    }

    fn seek_later_block(&mut self, target: DocId) -> DocId {
        let Some(block_idx) = self.block_list.seek_block(target, self.current_block + 1) else {
            self.exhausted = true;
            return TERMINATED;
        };
        self.load_block(block_idx);
        if self.exhausted {
            return TERMINATED;
        }
        // Verified content: the block's last id is at least `target`.
        self.position_in_block = self.block_doc_ids.partition_point(|&doc| doc < target);
        self.block_doc_ids[self.position_in_block]
    }

    /// Copy a bounded prefix without decoding frequencies or positions.
    pub(crate) fn fill_doc_batch(&mut self, docs: &mut [DocId]) -> usize {
        self.fill_batch::<false>(docs, &mut [])
    }

    pub(crate) fn fill_scored_doc_batch(&mut self, docs: &mut [DocId], tfs: &mut [u32]) -> usize {
        assert!(tfs.len() >= docs.len());
        self.fill_batch::<true>(docs, tfs)
    }

    fn fill_batch<const WITH_FREQUENCIES: bool>(
        &mut self,
        docs: &mut [DocId],
        tfs: &mut [u32],
    ) -> usize {
        assert!(docs.len() <= BLOCK_SIZE);
        let mut count = 0;
        while count < docs.len() && !self.exhausted {
            let remaining = &self.block_doc_ids[self.position_in_block..];
            let take = remaining.len().min(docs.len() - count);
            docs[count..count + take].copy_from_slice(&remaining[..take]);
            if WITH_FREQUENCIES {
                tfs[count..count + take].copy_from_slice(
                    &self.frequencies()[self.position_in_block..self.position_in_block + take],
                );
            }
            count += take;
            self.position_in_block += take;
            if self.position_in_block == self.block_doc_ids.len() {
                self.load_block(self.current_block + 1);
            }
        }
        count
    }

    /// Probe sorted IDs within decoded blocks, amortizing directory checks.
    pub(crate) fn retain_doc_batch(&mut self, docs: &mut [DocId]) -> usize {
        self.retain_batch::<false>(docs, &mut [])
    }

    pub(crate) fn retain_scored_doc_batch(&mut self, docs: &mut [DocId], tfs: &mut [u32]) -> usize {
        assert!(tfs.len() >= docs.len());
        self.retain_batch::<true>(docs, tfs)
    }

    fn retain_batch<const WITH_FREQUENCIES: bool>(
        &mut self,
        docs: &mut [DocId],
        tfs: &mut [u32],
    ) -> usize {
        assert!(docs.len() <= BLOCK_SIZE);
        let mut input = 0;
        let mut kept = 0;
        while input < docs.len() {
            if self.seek(docs[input]) == TERMINATED {
                break;
            }
            let last = *self.block_doc_ids.last().unwrap();
            while input < docs.len() && docs[input] <= last {
                let doc = docs[input];
                self.position_in_block = simd::find_first_ge_block_from(
                    &self.block_doc_ids,
                    self.position_in_block,
                    doc,
                );
                if self.block_doc_ids[self.position_in_block] == doc {
                    docs[kept] = doc;
                    if WITH_FREQUENCIES {
                        tfs[kept] = self.frequencies()[self.position_in_block];
                    }
                    kept += 1;
                }
                input += 1;
            }
        }
        kept
    }

    /// Consume a bounded ID range into caller-owned membership words. Retain the
    /// term-frequency prefix so subsequent scoring and position reads stay valid.
    pub(crate) fn fill_doc_window(&mut self, base: DocId, bits: &mut [u64]) {
        let span = u32::try_from(bits.len()).unwrap().checked_mul(64).unwrap();
        let end = base.saturating_add(span);
        bits.fill(0);
        self.seek(base);
        let list = self.block_list.as_ref();
        let dense = list.doc_count() >= 16
            && list
                .block_first_doc(0)
                .zip(list.block_last_doc(list.num_blocks().saturating_sub(1)))
                .is_some_and(|(first, last)| {
                    u64::from(last)
                        .checked_sub(u64::from(first))
                        .is_some_and(|span| span < u64::from(list.doc_count()) * 2)
                });
        if dense {
            self.fill_doc_words::<true>(base, end, bits);
        } else {
            self.fill_doc_words::<false>(base, end, bits);
        }
    }

    fn fill_doc_words<const GROUPED: bool>(&mut self, base: DocId, end: DocId, bits: &mut [u64]) {
        self.visit_until::<false>(end, |docs, _| {
            // Decide once per window, outside the hot decoded-run loop.
            if !GROUPED {
                for &doc in docs {
                    let offset = (doc - base) as usize;
                    bits[offset / 64] |= 1u64 << (offset % 64);
                }
                return true;
            }
            let mut index = 0;
            while index < docs.len() {
                let offset = (docs[index] - base) as usize;
                let word_index = offset / 64;
                let mut mask = 1u64 << (offset % 64);
                index += 1;
                while index < docs.len() {
                    let offset = (docs[index] - base) as usize;
                    if offset / 64 != word_index {
                        break;
                    }
                    mask |= 1u64 << (offset % 64);
                    index += 1;
                }
                bits[word_index] |= mask;
            }
            true
        });
    }

    /// Visit already decoded posting runs before `end` (exclusive). Each run
    /// contains at most one block. Returning false leaves that run unconsumed;
    /// callers can check cancellation without changing storage-layer policy.
    /// Preserve the TF prefix so subsequent position reads remain valid.
    pub(crate) fn visit_postings_until(
        &mut self,
        end: DocId,
        mut visit: impl FnMut(&[u32], &[u32]) -> bool,
    ) {
        self.visit_until::<true>(end, |docs, tfs| visit(docs, tfs.unwrap()));
    }

    /// Consume bounded decoded ID runs without initializing frequencies.
    pub(crate) fn visit_doc_ids_until(
        &mut self,
        end: DocId,
        mut visit: impl FnMut(&[u32]) -> bool,
    ) {
        self.visit_until::<false>(end, |docs, _| visit(docs));
    }

    fn visit_until<const WITH_FREQUENCIES: bool>(
        &mut self,
        end: DocId,
        mut visit: impl FnMut(&[u32], Option<&[u32]>) -> bool,
    ) {
        while self.doc() < end {
            let start = self.position_in_block;
            let count = self.block_doc_ids[start..].partition_point(|&doc| doc < end);
            let tfs = WITH_FREQUENCIES.then(|| &self.frequencies()[start..start + count]);
            if !visit(&self.block_doc_ids[start..start + count], tfs) {
                return;
            }
            self.position_in_block += count;
            if self.position_in_block == self.block_doc_ids.len() {
                self.load_block(self.current_block + 1);
            }
        }
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

    /// Borrow immutable metadata for the currently decoded posting block.
    pub(crate) fn current_block_metadata(&self) -> Option<(&BlockPostingList, usize)> {
        (!self.exhausted).then_some((&self.block_list, self.current_block))
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

/// Bounded intersection scratch for a fixed pair of monotonically advancing
/// posting iterators. Cursor positions stay on a match while its TF and positions
/// are consumed; the SIMD kernel runs once per overlapping block pair.
pub(crate) struct PostingIntersection {
    seek_driven: bool,
    left_is_rare: bool,
    blocks: (usize, usize),
    pairs: [(u8, u8); BLOCK_SIZE],
    next: usize,
    count: usize,
    ends: (usize, usize),
}

impl Default for PostingIntersection {
    fn default() -> Self {
        Self {
            seek_driven: false,
            left_is_rare: true,
            blocks: (usize::MAX, usize::MAX),
            pairs: [(0, 0); BLOCK_SIZE],
            next: 0,
            count: 0,
            ends: (0, 0),
        }
    }
}

impl PostingIntersection {
    pub(crate) fn with_costs(left: u32, right: u32) -> Self {
        Self {
            seek_driven: left.min(right).saturating_mul(4) < left.max(right),
            left_is_rare: left <= right,
            ..Self::default()
        }
    }

    /// Invalidate cached pairs before a physical rewind through a document map.
    pub(crate) fn reset(&mut self) {
        self.blocks = (usize::MAX, usize::MAX);
    }

    /// None yields at a block boundary so the caller can check cancellation.
    pub(crate) fn intersect_block(
        &mut self,
        left: &mut BlockPostingIterator<'_>,
        right: &mut BlockPostingIterator<'_>,
    ) -> Option<DocId> {
        if left.exhausted || right.exhausted {
            return Some(TERMINATED);
        }
        if left.doc() == right.doc() {
            return Some(left.doc());
        }
        if self.seek_driven {
            fn align(
                lead: &mut BlockPostingIterator<'_>,
                other: &mut BlockPostingIterator<'_>,
            ) -> Option<DocId> {
                let candidate = lead.doc();
                let next = other.seek(candidate);
                if next == candidate {
                    return Some(candidate);
                }
                lead.seek(next);
                None
            }
            return if self.left_is_rare {
                align(left, right)
            } else {
                align(right, left)
            };
        }
        if *left.block_doc_ids.last().unwrap() < right.doc() {
            left.seek_later_block(right.doc());
            return None;
        }
        if *right.block_doc_ids.last().unwrap() < left.doc() {
            right.seek_later_block(left.doc());
            return None;
        }
        let blocks = (left.current_block, right.current_block);
        if self.blocks != blocks {
            let mut a = left.position_in_block;
            let mut b = right.position_in_block;
            self.count = simd::intersect_posting_blocks(
                &left.block_doc_ids,
                &mut a,
                &right.block_doc_ids,
                &mut b,
                &mut self.pairs,
            );
            self.blocks = blocks;
            self.ends = (a, b);
            self.next = 0;
        }
        while self.next < self.count {
            let (a, b) = self.pairs[self.next];
            self.next += 1;
            let (a, b) = (usize::from(a), usize::from(b));
            if a >= left.position_in_block && b >= right.position_in_block {
                left.position_in_block = a;
                right.position_in_block = b;
                return Some(left.doc());
            }
        }
        left.position_in_block = left.position_in_block.max(self.ends.0);
        right.position_in_block = right.position_in_block.max(self.ends.1);
        if left.position_in_block == left.block_doc_ids.len() {
            left.load_block(left.current_block + 1);
        }
        if right.position_in_block == right.block_doc_ids.len() {
            right.load_block(right.current_block + 1);
        }
        None
    }
}

#[cfg(test)]
mod compact_layout_tests {
    use super::*;
    #[test]
    fn batched_intersection_preserves_monotone_seeks_frequencies_and_position_offsets() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let make = |divisor| {
                let mut postings = PostingList::new();
                for doc in 0..3001 {
                    if doc % divisor != 1 {
                        postings.push(doc, 1 + doc % 7);
                    }
                }
                BlockPostingList::from_posting_list_with_options(&postings, true, None, codec)
                    .unwrap()
            };
            let a = make(3);
            let b = make(5);
            for (seek_driven, left_is_rare) in [(false, true), (true, true), (true, false)] {
                let mut left = a.iterator();
                let mut right = b.iterator();
                let mut reference_left = a.iterator();
                let mut reference_right = b.iterator();
                let mut intersection = PostingIntersection {
                    seek_driven,
                    left_is_rare,
                    ..Default::default()
                };
                let mut target = 0;
                loop {
                    left.seek(target);
                    right.seek(target);
                    let doc = loop {
                        if let Some(doc) = intersection.intersect_block(&mut left, &mut right) {
                            break doc;
                        }
                    };
                    let expected = (target..3001)
                        .find(|doc| doc % 3 != 1 && doc % 5 != 1)
                        .unwrap_or(TERMINATED);
                    assert_eq!(doc, expected);
                    assert_eq!(
                        intersection.intersect_block(&mut left, &mut right),
                        Some(doc)
                    );
                    if doc == TERMINATED {
                        break;
                    }
                    reference_left.seek(doc);
                    reference_right.seek(doc);
                    assert_eq!(left.term_freq(), reference_left.term_freq());
                    assert_eq!(right.term_freq(), reference_right.term_freq());
                    assert_eq!(left.position_cursor(), reference_left.position_cursor());
                    assert_eq!(right.position_cursor(), reference_right.position_cursor());
                    target = doc + if doc % 7 == 0 { 19 } else { 1 };
                }
                for target in [0, 200, 129, 3, 2048, 3000, 2] {
                    intersection.reset();
                    left.seek_physical(target);
                    right.seek_physical(target);
                    let doc = loop {
                        if let Some(doc) = intersection.intersect_block(&mut left, &mut right) {
                            break doc;
                        }
                    };
                    let expected = (target..3001)
                        .find(|doc| doc % 3 != 1 && doc % 5 != 1)
                        .unwrap_or(TERMINATED);
                    assert_eq!(doc, expected);
                    reference_left.seek_physical(doc);
                    reference_right.seek_physical(doc);
                    assert_eq!(left.position_cursor(), reference_left.position_cursor());
                    assert_eq!(right.position_cursor(), reference_right.position_cursor());
                }
            }
        }
    }

    #[test]
    fn compact_postings_preserve_payload_scores_and_copy_merges() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Simd4x,
            PostingCodec::Pfor,
        ] {
            for count in [1, 2, 127, 128, 129, 1025] {
                let mut postings = PostingList::new();
                for doc in 0..count {
                    postings.push(doc * 3, doc % 13);
                }
                let list = BlockPostingList::from_posting_list_with_ratio_bounds(
                    &postings,
                    true,
                    Some(&|doc| doc % 100 + 1),
                    codec,
                )
                .unwrap();
                let mut old = Vec::new();
                list.serialize(&mut old).unwrap();
                let mut bytes = Vec::new();
                list.serialize_compact(&mut bytes).unwrap();
                let compact = BlockPostingList::deserialize(&bytes).unwrap();
                assert_eq!(compact.compact_headers, codec != PostingCodec::Pfor);
                if codec != PostingCodec::Pfor {
                    assert_eq!(old.len() - bytes.len(), list.num_blocks() * 8);
                }
                let mut a = Vec::new();
                let mut b = Vec::new();
                for i in 0..list.num_blocks() {
                    assert_eq!(list.block_payload(i), compact.block_payload(i));
                    assert_eq!(list.pos_cursor(i), compact.pos_cursor(i));
                    assert!(compact.decode_block_into(i, &mut a, &mut b));
                    assert_eq!(
                        a,
                        postings.postings
                            [i * BLOCK_SIZE..(i * BLOCK_SIZE + BLOCK_SIZE).min(count as usize)]
                            .iter()
                            .map(|p| p.doc_id)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(b, a.iter().map(|doc| (doc / 3) % 13).collect::<Vec<_>>());
                }
                let mut repeated = Vec::new();
                compact.serialize(&mut repeated).unwrap();
                assert_eq!(bytes, repeated);
                for second in [&old, &bytes] {
                    let mut merged = Vec::new();
                    let (docs, len) = BlockPostingList::concatenate_streaming(
                        &[(&bytes, 0), (second, count * 3)],
                        &mut merged,
                    )
                    .unwrap();
                    assert_eq!(len, merged.len());
                    assert_eq!(docs, count * 2);
                    let merged = BlockPostingList::deserialize(&merged).unwrap();
                    for block in 0..merged.num_blocks() {
                        assert_eq!(
                            merged.block_payload(block),
                            list.block_payload(block % list.num_blocks())
                        );
                        assert!(merged.decode_block_into(block, &mut a, &mut b));
                    }
                }
            }
        }
    }

    #[test]
    fn compact_posting_metadata_rejects_bad_descriptors_and_content_checks_remain_observable() {
        let mut postings = PostingList::new();
        for doc in 0..257 {
            postings.push(doc * 2, 1);
        }
        let list = BlockPostingList::from_posting_list_with_options(
            &postings,
            true,
            None,
            PostingCodec::Rounded,
        )
        .unwrap();
        let mut bytes = Vec::new();
        list.serialize_compact(&mut bytes).unwrap();
        let footer = Footer::parse(&bytes).unwrap();
        for (at, value) in [
            (footer.l0_end(), 0),
            (footer.l0_end() + 1, 1),
            (footer.l0_end() + 2, 33),
            (footer.l0_end() + 3, 33),
            (footer.l1_bounds_end(), 1),
        ] {
            let mut corrupt = bytes.clone();
            corrupt[at] = value;
            assert!(
                BlockPostingList::deserialize(&corrupt).is_err(),
                "byte {at}"
            );
        }
        let mut corrupt = bytes.clone();
        corrupt[0] = 0;
        let compact = BlockPostingList::deserialize(&corrupt).unwrap();
        let mut docs = Vec::new();
        assert!(compact.decode_block_doc_ids_checked(0, &mut docs).is_err());
        assert!(docs.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_decoding_overwrites_stale_values_across_lengths_and_codecs() {
        let mut docs = vec![u32::MAX; BLOCK_SIZE * 2];
        let mut tfs = docs.clone();
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            for count in [128, 1, 127, 128, 17, 256, 257] {
                for freq in [0, 1, 255, 65536] {
                    let mut postings = PostingList::new();
                    for i in 0..count {
                        postings.push(i * 3 + 7, freq);
                    }
                    let list = BlockPostingList::from_posting_list_with_options(
                        &postings, true, None, codec,
                    )
                    .unwrap();
                    for compact in [false, true] {
                        let mut bytes = Vec::new();
                        if compact {
                            list.serialize_compact(&mut bytes).unwrap();
                        } else {
                            list.serialize(&mut bytes).unwrap();
                        }
                        let list = BlockPostingList::deserialize(&bytes).unwrap();
                        for block in (0..list.num_blocks()).rev().chain(0..list.num_blocks()) {
                            docs.fill(u32::MAX);
                            tfs.fill(u32::MAX);
                            assert!(list.decode_block_into(block, &mut docs, &mut tfs));
                            let expected: Vec<_> = postings.postings[block * BLOCK_SIZE
                                ..postings.postings.len().min((block + 1) * BLOCK_SIZE)]
                                .iter()
                                .map(|p| p.doc_id)
                                .collect();
                            assert_eq!(docs, expected);
                            assert_eq!(tfs, vec![freq; expected.len()]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compact_membership_batches_preserve_resume_frequencies_and_position_prefixes() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut postings = PostingList::new();
            let mut expected = Vec::new();
            let mut prefix = 0u64;
            for i in 0..701u32 {
                let tf = i % 19 + 1;
                let doc = i * 13 + 5;
                postings.push(doc, tf);
                expected.push((doc, tf, prefix));
                prefix += u64::from(tf);
            }
            let list =
                BlockPostingList::from_posting_list_with_options(&postings, true, None, codec)
                    .unwrap();
            let bytes = serialize_bpl(&list);
            let mut cursor = list.iterator();
            let mut docs = [0; BLOCK_SIZE];
            let mut consumed = 0;
            while cursor.doc() != TERMINATED {
                let count = cursor.fill_doc_batch(&mut docs);
                assert_eq!(
                    &docs[..count],
                    &expected[consumed..consumed + count]
                        .iter()
                        .map(|e| e.0)
                        .collect::<Vec<_>>()
                );
                consumed += count;
                if consumed < expected.len() {
                    assert!(cursor.block_tfs.get().is_none(), "copy initialized TFs");
                    assert_eq!(cursor.doc(), expected[consumed].0);
                    assert_eq!(cursor.term_freq(), expected[consumed].1);
                    assert_eq!(cursor.position_cursor_mut(), expected[consumed].2);
                }
            }
            assert_eq!(consumed, expected.len());
            assert_eq!(cursor.fill_doc_batch(&mut docs), 0);
            let mut cursor = list.iterator();
            for start in (0..10_000u32).step_by(BLOCK_SIZE) {
                let mut candidates: Vec<_> = (start..start + BLOCK_SIZE as u32).collect();
                let wanted: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|doc| expected.iter().any(|e| e.0 == *doc))
                    .collect();
                let kept = cursor.retain_doc_batch(&mut candidates);
                assert_eq!(&candidates[..kept], wanted, "{codec:?} start={start}");
                if cursor.doc() != TERMINATED {
                    let entry = expected.iter().find(|e| e.0 == cursor.doc()).unwrap();
                    assert_eq!(cursor.term_freq(), entry.1);
                    assert_eq!(cursor.position_cursor_mut(), entry.2);
                }
            }
            assert_eq!(cursor.doc(), TERMINATED);
            assert_eq!(serialize_bpl(&list), bytes);
        }
    }

    #[test]
    fn membership_words_preserve_unaligned_dense_sparse_windows_and_resume() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            for stride in [1, 2, 3, 67, 129] {
                let docs: Vec<u32> = (0..10_000).step_by(stride).collect();
                let mut postings = PostingList::new();
                for &doc in &docs {
                    postings.push(doc, 1 + doc % 7);
                }
                let list =
                    BlockPostingList::from_posting_list_with_options(&postings, true, None, codec)
                        .unwrap();
                for start in [0, 1, 17, 63, 64, 127, 511] {
                    let mut cursor = list.iterator();
                    for base in [start, start + 4096, start + 8192] {
                        let mut bits = [u64::MAX; 64];
                        cursor.fill_doc_window(base, &mut bits);
                        let mut expected = [0u64; 64];
                        for &doc in &docs {
                            if (base..base + 4096).contains(&doc) {
                                let offset = (doc - base) as usize;
                                expected[offset / 64] |= 1u64 << (offset % 64);
                            }
                        }
                        assert_eq!(bits, expected, "{codec:?}, stride={stride}, base={base}");
                        let next = docs
                            .iter()
                            .copied()
                            .find(|&doc| doc >= base + 4096)
                            .unwrap_or(TERMINATED);
                        assert_eq!(cursor.doc(), next);
                        if next != TERMINATED {
                            assert_eq!(cursor.term_freq(), 1 + next % 7);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn document_only_navigation_defers_frequencies_and_resumes_exact_position_reads() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut postings = PostingList::new();
            let mut expected = Vec::new();
            let mut prefix = 0u64;
            for i in 0..701u32 {
                let tf = if i % 37 == 0 { u32::MAX } else { i % 23 + 1 };
                let doc = i * 13 + 5;
                postings.push(doc, tf);
                expected.push((doc, tf, prefix));
                prefix += u64::from(tf);
            }
            let list =
                BlockPostingList::from_posting_list_with_options(&postings, true, None, codec)
                    .unwrap();
            let bytes = serialize_bpl(&list);
            for owned in [false, true] {
                let mut cursor = if owned {
                    list.clone().into_iterator()
                } else {
                    list.iterator()
                };
                assert!(
                    cursor.block_tfs.get().is_none(),
                    "{codec:?}: ID-only open decoded frequencies"
                );
                for target in [20, 205, 1800, 3900] {
                    let entry = expected.iter().find(|e| e.0 >= target).unwrap();
                    assert_eq!(cursor.seek(target), entry.0);
                    assert!(
                        cursor.block_tfs.get().is_none(),
                        "ID-only seek decoded frequencies"
                    );
                }
                let entry = expected.iter().find(|e| e.0 == cursor.doc()).unwrap();
                #[cfg(feature = "native")]
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        let shared = &cursor;
                        scope.spawn(move || {
                            assert_eq!(shared.term_freq(), entry.1);
                            assert_eq!(shared.position_cursor(), entry.2);
                        });
                    }
                });
                assert_eq!(cursor.position_cursor(), entry.2);
                assert_eq!(cursor.term_freq(), entry.1);
                assert_eq!(cursor.position_cursor_mut(), entry.2);
                let mut bits = [0u64; 64];
                cursor.fill_doc_window(4096, &mut bits);
                for &(doc, _, _) in &expected {
                    if (4096..8192).contains(&doc) {
                        assert_ne!(bits[(doc as usize - 4096) / 64] & (1 << (doc % 64)), 0);
                    }
                }
                assert_eq!(
                    bits.iter().map(|word| word.count_ones()).sum::<u32>(),
                    expected
                        .iter()
                        .filter(|e| (4096..8192).contains(&e.0))
                        .count() as u32
                );
                assert!(
                    cursor.block_tfs.get().is_none(),
                    "membership windows decoded frequencies"
                );
                let entry = expected.iter().find(|e| e.0 == cursor.doc()).unwrap();
                assert_eq!(cursor.position_cursor_mut(), entry.2);
                assert_eq!(cursor.term_freq(), entry.1);
                let parked = cursor.doc();
                cursor.visit_postings_until(TERMINATED, |docs, tfs| {
                    for (&doc, &tf) in docs.iter().zip(tfs) {
                        assert_eq!(tf, expected.iter().find(|e| e.0 == doc).unwrap().1);
                    }
                    false
                });
                assert_eq!(cursor.doc(), parked);
                assert_eq!(cursor.position_cursor(), entry.2);
                let mut seen = Vec::new();
                cursor.visit_postings_until(TERMINATED, |docs, tfs| {
                    seen.extend(docs.iter().copied().zip(tfs.iter().copied()));
                    true
                });
                assert_eq!(
                    seen,
                    expected
                        .iter()
                        .filter(|e| e.0 >= parked)
                        .map(|e| (e.0, e.1))
                        .collect::<Vec<_>>()
                );
                assert_eq!(cursor.doc(), TERMINATED);
                assert_eq!(cursor.term_freq(), 0);
                assert_eq!(cursor.seek(0), TERMINATED);
            }
            assert_eq!(serialize_bpl(&list), bytes);
        }
    }

    #[test]
    fn document_navigation_defers_position_accounting_without_changing_cursors() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut postings = PostingList::new();
            let mut expected = Vec::new();
            let mut total = 0u64;
            for doc in 0..701u32 {
                expected.push(total);
                let tf = doc % 23 + 1;
                postings.push(doc * 7, tf);
                total += u64::from(tf);
            }
            let list =
                BlockPostingList::from_posting_list_with_options(&postings, true, None, codec)
                    .unwrap();
            let mut cursor = list.iterator();
            for target in [7, 70, 777, 896, 1400, 3500, 4900] {
                assert_eq!(cursor.seek(target), target);
                assert!(
                    cursor.position_offsets.is_none(),
                    "navigation must not prepare unused position offsets"
                );
                assert_eq!(cursor.position_cursor(), expected[target as usize / 7]);
                assert_eq!(cursor.term_freq(), target / 7 % 23 + 1);
                assert_eq!(cursor.seek(target - 1), target);
            }
            assert_eq!(cursor.advance(), TERMINATED);
        }
    }

    #[test]
    fn deferred_position_accounting_resumes_after_reads_windows_and_stopped_runs() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let docs: Vec<_> = (0..600u32).map(|doc| (doc * 11, u32::MAX - doc)).collect();
            let prefixes = expected_cursors(&docs);
            let mut input = PostingList::new();
            for &(doc, tf) in &docs {
                input.push(doc, tf);
            }
            let list = BlockPostingList::build(&input, true, None, codec, false, false).unwrap();
            let before = serialize_bpl(&list);
            let mut cursor = list.iterator();
            for at in [13, 25, 127, 128, 199] {
                cursor.seek(docs[at].0);
                assert_eq!(cursor.position_range(), (prefixes[at], docs[at].1));
                assert_eq!(cursor.position_range(), (prefixes[at], docs[at].1));
                assert_eq!(cursor.position_cursor_mut(), prefixes[at]);
                assert_eq!(cursor.position_cursor_mut(), prefixes[at]);
                assert_eq!(cursor.position_cursor(), prefixes[at]);
                cursor.seek(docs[at].0 - 1);
                assert_eq!(cursor.position_cursor_mut(), prefixes[at]);
            }
            cursor.visit_postings_until(docs[220].0, |_, _| true);
            assert_eq!(cursor.position_cursor_mut(), prefixes[220]);
            cursor.visit_postings_until(docs[250].0, |_, _| false);
            assert_eq!(cursor.position_cursor_mut(), prefixes[220]);
            let mut bits = [0u64; 4];
            cursor.fill_doc_window(docs[220].0, &mut bits);
            let at = docs.partition_point(|&(doc, _)| doc < docs[220].0 + 256);
            assert_eq!(cursor.position_cursor_mut(), prefixes[at]);
            cursor.skip_to_next_block();
            assert_eq!(cursor.position_cursor_mut(), prefixes[256]);
            assert_eq!(serialize_bpl(&list), before);
        }
    }

    #[test]
    fn posting_runs_preserve_frequencies_positions_and_resumption_after_stop() {
        let docs: Vec<_> = (0..1027u32).map(|i| (i * 7, i % 31 + 1)).collect();
        let prefixes = expected_cursors(&docs);
        let mut list = PostingList::new();
        for &(doc, tf) in &docs {
            list.push(doc, tf);
        }
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let postings = BlockPostingList::build(&list, true, None, codec, false, false).unwrap();
            let bytes = serialize_bpl(&postings);
            let mut cursor = postings.iterator();
            cursor.seek(docs[17].0);
            let mut actual = Vec::new();
            let mut calls = 0;
            cursor.visit_postings_until(TERMINATED, |ids, tfs| {
                calls += 1;
                assert_eq!(ids.len(), tfs.len());
                assert!(ids.len() <= BLOCK_SIZE);
                if calls == 3 {
                    return false;
                }
                actual.extend(ids.iter().copied().zip(tfs.iter().copied()));
                true
            });
            assert_eq!(actual, docs[17..256]);
            assert_eq!(cursor.doc(), docs[256].0);
            assert_eq!(cursor.position_cursor(), prefixes[256]);
            cursor.visit_postings_until(docs[331].0, |ids, tfs| {
                actual.extend(ids.iter().copied().zip(tfs.iter().copied()));
                true
            });
            assert_eq!(actual, docs[17..331]);
            assert_eq!(cursor.doc(), docs[331].0);
            assert_eq!(cursor.term_freq(), docs[331].1);
            assert_eq!(cursor.position_cursor(), prefixes[331]);
            cursor.visit_postings_until(TERMINATED, |ids, tfs| {
                actual.extend(ids.iter().copied().zip(tfs.iter().copied()));
                true
            });
            assert_eq!(actual, docs[17..]);
            cursor.visit_postings_until(TERMINATED, |_, _| panic!("already exhausted"));
            assert_eq!(cursor.doc(), TERMINATED);
            assert_eq!(serialize_bpl(&postings), bytes);
        }
    }

    #[test]
    fn document_windows_preserve_posting_frequencies_positions_and_bytes() {
        for near_end in [false, true] {
            let base = if near_end { TERMINATED - 30_000 } else { 0 };
            let docs: Vec<_> = (0..5000u32).map(|i| (base + i * 5, i % 31 + 1)).collect();
            let prefixes = expected_cursors(&docs);
            let mut list = PostingList::new();
            for &(doc, tf) in &docs {
                list.push(doc, tf);
            }
            for codec in [
                PostingCodec::Rounded,
                PostingCodec::Packed,
                PostingCodec::Pfor,
                PostingCodec::Simd4x,
            ] {
                let postings =
                    BlockPostingList::build(&list, true, None, codec, false, false).unwrap();
                let bytes = serialize_bpl(&postings);
                let mut cursor = postings.iterator();
                let mut actual = Vec::new();
                while cursor.doc() != TERMINATED {
                    let base = cursor.doc();
                    let mut bits = [u64::MAX; 64];
                    cursor.fill_doc_window(base, &mut bits);
                    for (index, word) in bits.into_iter().enumerate() {
                        for bit in 0..64 {
                            if word & (1 << bit) != 0 {
                                actual.push(base + index as u32 * 64 + bit);
                            }
                        }
                    }
                    let at = docs.partition_point(|&(doc, _)| doc < base.saturating_add(4096));
                    assert_eq!(cursor.doc(), docs.get(at).map_or(TERMINATED, |p| p.0));
                    if at < docs.len() {
                        assert_eq!(cursor.term_freq(), docs[at].1);
                        assert_eq!(cursor.position_cursor(), prefixes[at]);
                    }
                }
                assert_eq!(actual, docs.iter().map(|p| p.0).collect::<Vec<_>>());
                assert_eq!(serialize_bpl(&postings), bytes);
            }
        }
    }

    #[test]
    fn nearby_and_distant_seeks_preserve_postings_position_cursors_and_bytes() {
        let docs: Vec<_> = (0..32_769u32)
            .map(|i| (i * 5 + i % 3, i % 31 + 1))
            .collect();
        let prefixes = expected_cursors(&docs);
        let mut list = PostingList::new();
        for &(doc, tf) in &docs {
            list.push(doc, tf);
        }
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let postings = BlockPostingList::build(&list, true, None, codec, false, false).unwrap();
            let bytes = serialize_bpl(&postings);
            let lasts: Vec<_> = (0..postings.num_blocks())
                .map(|i| postings.block_last_doc(i).unwrap())
                .collect();
            for from in [0, 1, 7, 8, 64, 200, 256, 257] {
                for target in [0, 1, 100, 639, 640, 641, 700, 160_000, 163_840, TERMINATED] {
                    let expected = (from..lasts.len()).find(|&i| lasts[i] >= target);
                    assert_eq!(
                        postings.seek_block(target, from),
                        expected,
                        "{codec:?} from={from} target={target}"
                    );
                }
            }
            let mut cursor = postings.iterator();
            let mut at = 0;
            for target in [
                0,
                1,
                20,
                19,
                639,
                640,
                641,
                700,
                160_000,
                161_000,
                160_000,
                docs.last().unwrap().0 - 1,
                docs.last().unwrap().0,
                TERMINATED,
                0,
            ] {
                while at < docs.len() && docs[at].0 < target {
                    at += 1;
                }
                let expected = docs.get(at).map_or(TERMINATED, |&(doc, _)| doc);
                assert_eq!(cursor.seek(target), expected, "{codec:?} target={target}");
                if at < docs.len() {
                    assert_eq!(cursor.term_freq(), docs[at].1);
                    assert_eq!(cursor.position_cursor(), prefixes[at]);
                }
            }
            assert_eq!(serialize_bpl(&postings), bytes);
        }
    }

    #[test]
    fn candidate_posting_probes_reuse_buffers_and_preserve_seek_and_position_cursors() {
        let mut list = PostingList::new();
        for i in 0..700 {
            list.push(i * 3, i % 7 + 1);
        }
        let postings = BlockPostingList::from_posting_list(&list).unwrap();
        let mut scratch = PostingDecodeScratch::default();
        let mut doc_buffer = None;
        for first in [0, 385, 900, 1800, 2100, 100, 2097] {
            let mut reference = postings.clone().into_iterator();
            let mut selected = postings
                .clone()
                .into_candidate_iterator(first, &mut scratch);
            assert_eq!(
                selected.current_block_idx(),
                postings.seek_block(first, 0).unwrap_or(0)
            );
            for target in (first..2200).step_by(17) {
                assert_eq!(selected.seek(target), reference.seek(target));
                assert_eq!(selected.term_freq(), reference.term_freq());
                if selected.doc() != TERMINATED {
                    assert_eq!(selected.position_cursor(), reference.position_cursor());
                }
            }
            selected.recycle(&mut scratch);
            assert!(scratch.doc_ids.capacity() >= BLOCK_SIZE);
            // The frequency scratch is inline (no heap allocation to reuse);
            // the document buffer is the one heap allocation and must be.
            assert!(scratch.term_freqs.is_some());
            let address = scratch.doc_ids.as_ptr() as usize;
            if let Some(previous) = doc_buffer {
                assert_eq!(
                    address, previous,
                    "document scratch allocation must be reused"
                );
            }
            doc_buffer = Some(address);
        }
    }

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
    #[test]
    fn deserialization_rejects_unsupported_width_instead_of_empty_results() {
        let list = build_bpl(&[(0, 1), (1, 2)]);
        let mut bytes = Vec::new();
        list.serialize(&mut bytes).unwrap();
        bytes[6] = 0xe1;
        let error = BlockPostingList::deserialize(&bytes).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("exceeds 32 bits"));
    }

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
        assert_eq!(bpl.l1_docs.get(0).unwrap(), expected_l1_0);

        // L1[1] = last_doc of block 15 (end of second group)
        let expected_l1_1 = bpl.block_last_doc(15).unwrap();
        assert_eq!(bpl.l1_docs.get(1).unwrap(), expected_l1_1);

        // L1[2] = last_doc of block 19 (end of partial group)
        let expected_l1_2 = bpl.block_last_doc(19).unwrap();
        assert_eq!(bpl.l1_docs.get(2).unwrap(), expected_l1_2);
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
    fn block_len_matches_l0_offsets() {
        // Block lengths derive from neighbouring L0 offsets and add up to the stream.
        let bpl = build_bpl(&(0..1000).map(|i| (i * 3, 1 + i % 4)).collect::<Vec<_>>());
        let mut total = 0usize;
        for b in 0..bpl.num_blocks() {
            let (_, _, offset, _) = bpl.read_l0_entry(b);
            assert_eq!(offset as usize, total, "block {b} offset");
            total += bpl.block_len(b);
        }
        assert_eq!(total, bpl.stream.len());
    }

    /// Every codec round-trips doc ids and tfs exactly, `seek` agrees with
    /// `Rounded`, and `Rounded` output is byte-identical to the historic
    /// layout (codec id 0, widths 0/8/16/32).
    #[test]
    fn every_codec_round_trips_and_seeks() {
        let mut postings: Vec<(u32, u32)> = Vec::new();
        let mut doc = 0u32;
        for i in 0..5000u32 {
            // Mostly small gaps with rare huge ones (forces exceptions / wide
            // blocks), tfs mostly 1-3 with rare outliers.
            doc += if i % 97 == 0 { 100_000 } else { 1 + i % 7 };
            let tf = if i % 131 == 0 { 5000 } else { 1 + i % 3 };
            postings.push((doc, tf));
        }
        let mut list = PostingList::new();
        for &(d, tf) in &postings {
            list.push(d, tf);
        }
        let rounded = BlockPostingList::from_posting_list(&list).unwrap();
        let mut sizes = Vec::new();
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let bpl = BlockPostingList::from_posting_list_with_codec(&list, codec).unwrap();
            assert_eq!(collect_postings(&bpl), postings, "{codec}");
            for b in 0..bpl.num_blocks() {
                let count = (postings.len() - b * BLOCK_SIZE).min(BLOCK_SIZE);
                let expected_codec = if codec == PostingCodec::Simd4x && count < BLOCK_SIZE {
                    PostingCodec::Rounded
                } else {
                    codec
                };
                assert_eq!(bpl.block_codec(b), Some(expected_codec));
                assert_eq!(bpl.block_max_tf(b), rounded.block_max_tf(b));
            }
            // Serialized round trip (both copying and zero-copy paths).
            let bytes = serialize_bpl(&bpl);
            let back = BlockPostingList::deserialize(&bytes).unwrap();
            assert_eq!(collect_postings(&back), postings, "{codec} deserialize");
            let back =
                BlockPostingList::deserialize_zero_copy(OwnedBytes::new(bytes.clone())).unwrap();
            assert_eq!(collect_postings(&back), postings, "{codec} zero-copy");
            // Seeks land on the same docs as the reference layout.
            let mut a = rounded.iterator();
            let mut b = back.iterator();
            for target in (0..postings.last().unwrap().0 + 10).step_by(2_003) {
                assert_eq!(a.seek(target), b.seek(target), "{codec} seek {target}");
                assert_eq!(a.term_freq(), b.term_freq());
            }
            sizes.push((codec, bytes.len()));
        }
        let rounded_bytes = serialize_bpl(&rounded);
        assert_eq!(
            serialize_bpl(
                &BlockPostingList::from_posting_list_with_codec(&list, PostingCodec::Rounded)
                    .unwrap()
            ),
            rounded_bytes,
            "Rounded must stay byte-identical"
        );
        // Header byte of a Rounded block: codec 0, rounded width.
        assert!(matches!(rounded.stream[6], 0 | 8 | 16 | 32));
        let size = |c: PostingCodec| sizes.iter().find(|(k, _)| *k == c).unwrap().1;
        assert!(size(PostingCodec::Packed) < size(PostingCodec::Rounded));
        assert!(size(PostingCodec::Pfor) < size(PostingCodec::Packed));
    }

    /// Blocks of different codecs merge by verbatim copy and decode correctly.
    #[test]
    fn mixed_codec_sources_concatenate() {
        let a: Vec<(u32, u32)> = (0..300u32).map(|i| (i * 5, 1 + i % 9)).collect();
        let b: Vec<(u32, u32)> = (0..300u32).map(|i| (i * 11 + 3, 2 + i % 5)).collect();
        let list_a = {
            let mut l = PostingList::new();
            a.iter().for_each(|&(d, t)| l.push(d, t));
            BlockPostingList::from_posting_list_with_codec(&l, PostingCodec::Pfor).unwrap()
        };
        let list_b = {
            let mut l = PostingList::new();
            b.iter().for_each(|&(d, t)| l.push(d, t));
            BlockPostingList::from_posting_list_with_codec(&l, PostingCodec::Packed).unwrap()
        };
        let offset_b = a.last().unwrap().0 + 1;
        let expected: Vec<(u32, u32)> = a
            .iter()
            .copied()
            .chain(b.iter().map(|&(d, t)| (d + offset_b, t)))
            .collect();

        let merged = BlockPostingList::concatenate_blocks(&[
            (list_a.clone(), 0),
            (list_b.clone(), offset_b),
        ])
        .unwrap();
        assert_eq!(collect_postings(&merged), expected);

        let bytes_a = serialize_bpl(&list_a);
        let bytes_b = serialize_bpl(&list_b);
        let mut out = Vec::new();
        let (docs, written) = BlockPostingList::concatenate_streaming(
            &[(bytes_a.as_slice(), 0), (bytes_b.as_slice(), offset_b)],
            &mut out,
        )
        .unwrap();
        assert_eq!(docs, 600);
        assert_eq!(written, out.len());
        let streamed = BlockPostingList::deserialize(&out).unwrap();
        assert_eq!(collect_postings(&streamed), expected);
        assert_eq!(streamed.block_codec(0), Some(PostingCodec::Pfor));
        assert_eq!(
            streamed.block_codec(streamed.num_blocks() - 1),
            Some(PostingCodec::Packed)
        );
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
        assert_eq!(bpl.l1_docs.bytes(), bpl2.l1_docs.bytes());
    }

    #[test]
    fn posting_open_borrows_unaligned_group_directories_and_preserves_bytes_and_seeks() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut postings = PostingList::new();
            for i in 0..3457u32 {
                postings.push(i * 7 + 3, i % 13 + 1);
            }
            let built = BlockPostingList::from_posting_list_with_codec(&postings, codec).unwrap();
            let encoded = serialize_bpl(&built);
            for prefix in [1, 3, 7] {
                let mut padded = vec![0; prefix];
                padded.extend_from_slice(&encoded);
                let owner = OwnedBytes::new(padded);
                let raw = owner.slice(prefix..owner.len());
                let footer = Footer::parse(&raw).unwrap();
                let docs_start = raw[footer.l1_start()..].as_ptr();
                let bounds_start = raw[footer.l1_end()..].as_ptr();
                let opened = BlockPostingList::deserialize_zero_copy(raw).unwrap();
                assert_eq!(
                    opened.l1_docs.bytes().as_ptr(),
                    docs_start,
                    "opening must borrow group document metadata"
                );
                assert_eq!(
                    opened.l1_bounds.bytes().as_ptr(),
                    bounds_start,
                    "opening must borrow group bound metadata"
                );
                assert_eq!(serialize_bpl(&opened), encoded);
                for from in 0..=built.num_blocks() {
                    for target in (0..=3457 * 7 + 4).step_by(37) {
                        let expected = (from..built.num_blocks())
                            .find(|&block| built.block_last_doc(block).unwrap() >= target);
                        assert_eq!(opened.seek_block(target, from), expected);
                    }
                }
                assert_eq!(collect_postings(&opened), collect_postings(&built));
            }
        }
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
        assert_eq!(copied.l1_docs.bytes(), zero_copy.l1_docs.bytes());
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
        for (i, word) in merged.l1_docs.words().iter().enumerate() {
            let l1_doc = u32::from_le_bytes(*word);
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
        BlockPostingList::from_posting_list_with(&list, true, None).unwrap()
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
        // A real pre-magic layout has no L1 bounds or footer extension, and
        // stores f32 maxima rather than packed TF/length L0 words.
        let footer = Footer::parse(&bytes).unwrap();
        let mut legacy = bytes[..footer.l1_end()].to_vec();
        for block in 0..bpl.num_blocks() {
            let at = footer.l0_start() + block * L0_SIZE + 12;
            legacy[at..at + 4]
                .copy_from_slice(&(bpl.block_max_tf(block).unwrap() as f32).to_le_bytes());
        }
        legacy.extend_from_slice(
            &bytes[bytes.len() - FOOTER_V2_SIZE..bytes.len() - (FOOTER_V2_SIZE - FOOTER_SIZE)],
        );
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
    #[test]
    fn saturated_tf_bounds_remain_upper_bounds_without_changing_encoded_bytes() {
        let docs: Vec<_> = (0..(BLOCK_SIZE * 18) as u32)
            .map(|doc| {
                (
                    doc,
                    if doc == (BLOCK_SIZE * 17) as u32 {
                        100_000
                    } else {
                        1
                    },
                )
            })
            .collect();
        let list = build_bpl(&docs);
        let bytes = serialize_bpl(&list);
        assert_eq!(list.max_tf(), 100_000);
        assert!(list.block_bounds(17).unwrap().0 >= 100_000);
        assert!(list.group_bounds(17).unwrap().0 >= 100_000);
        assert_eq!(list.block_bounds(0).unwrap().0, 1);
        let decoded = BlockPostingList::deserialize(&bytes).unwrap();
        assert!(decoded.block_bounds(17).unwrap().0 >= 100_000);
        assert!(decoded.group_bounds(17).unwrap().0 >= 100_000);
        assert_eq!(serialize_bpl(&decoded), bytes);
        assert_eq!(collect_postings(&decoded), docs);
    }
    #[test]
    fn ratio_bounds_preserve_payload_bytes_and_copy_merge_with_legacy_blocks() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let mut postings = PostingList::new();
            for i in 0..1100 {
                postings.push(i * 3, 1 + i % 97);
            }
            let length = |id| 10 + id % 997;
            let plain = BlockPostingList::from_posting_list_with_options(
                &postings,
                true,
                Some(&length),
                codec,
            )
            .unwrap();
            let tight = BlockPostingList::from_posting_list_with_ratio_bounds(
                &postings,
                true,
                Some(&length),
                codec,
            )
            .unwrap();
            assert_eq!(plain.stream.as_slice(), tight.stream.as_slice());
            assert_eq!(plain.l0_bytes.as_slice(), tight.l0_bytes.as_slice());
            assert_eq!(
                plain.pos_cursors.as_ref().map(OwnedBytes::as_slice),
                tight.pos_cursors.as_ref().map(OwnedBytes::as_slice)
            );
            let mut raw = Vec::new();
            tight.serialize(&mut raw).unwrap();
            let decoded = BlockPostingList::deserialize(&raw).unwrap();
            assert!(decoded.has_ratio_bounds());
            for block in 0..tight.num_blocks() {
                assert!(decoded.block_length_ratio(block) > 0.0);
                for posting in &postings.postings
                    [block * BLOCK_SIZE..((block + 1) * BLOCK_SIZE).min(postings.len())]
                {
                    assert!(
                        decoded.block_length_ratio(block) as f64
                            <= length(posting.doc_id) as f64 / posting.term_freq as f64
                    );
                    assert!(decoded.group_length_ratio(block) <= decoded.block_length_ratio(block));
                }
            }
            let merged = BlockPostingList::concatenate_blocks(&[
                (tight.clone(), 0),
                (plain.clone(), 10_000),
            ])
            .unwrap();
            let mut plain_raw = Vec::new();
            plain.serialize(&mut plain_raw).unwrap();
            let mut streaming = Vec::new();
            let (_, size) = BlockPostingList::concatenate_streaming(
                &[(&raw, 0), (&plain_raw, 10_000)],
                &mut streaming,
            )
            .unwrap();
            let mut expected = Vec::new();
            merged.serialize(&mut expected).unwrap();
            assert_eq!(streaming, expected);
            assert_eq!(streaming.len(), size);
            for block in 0..tight.num_blocks() {
                assert_eq!(
                    merged.block_length_ratio(block),
                    tight.block_length_ratio(block)
                );
                assert_eq!(merged.block_length_ratio(block + tight.num_blocks()), 0.0);
                let mut before = Vec::new();
                let mut after = Vec::new();
                let mut tfs_before = Vec::new();
                let mut tfs_after = Vec::new();
                tight.decode_block_into(block, &mut before, &mut tfs_before);
                merged.decode_block_into(block, &mut after, &mut tfs_after);
                assert_eq!((before, tfs_before), (after, tfs_after));
            }
            // Model an actual list without the optional extension: remove
            // its bytes as well as its flag. Unaddressed trailers are corrupt.
            let footer = Footer::parse(&raw).unwrap();
            raw.drain(footer.cursors_end()..footer.ratios_end());
            let flags = raw.len() - 12;
            raw[flags] &= !(FLAG_RATIO_BOUNDS as u8);
            let legacy_view = BlockPostingList::deserialize(&raw).unwrap();
            assert!(!legacy_view.has_ratio_bounds());
            assert_eq!(legacy_view.stream.as_slice(), tight.stream.as_slice());
        }
    }

    #[test]
    fn ratio_bounds_reject_truncation_nonfinite_negative_and_unknown_flags() {
        let mut postings = PostingList::new();
        postings.push(1, 3);
        let list = BlockPostingList::from_posting_list_with_ratio_bounds(
            &postings,
            false,
            Some(&|_| 7),
            PostingCodec::Rounded,
        )
        .unwrap();
        let mut raw = Vec::new();
        list.serialize(&mut raw).unwrap();
        let footer = Footer::parse(&raw).unwrap();
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            let mut corrupt = raw.clone();
            corrupt[footer.cursors_end()..footer.cursors_end() + 4]
                .copy_from_slice(&bad.to_le_bytes());
            assert!(BlockPostingList::deserialize(&corrupt).is_err());
            assert!(
                BlockPostingList::concatenate_streaming(&[(&corrupt, 0)], &mut Vec::new()).is_err()
            );
        }
        let mut short = raw.clone();
        short.remove(footer.cursors_end());
        assert!(BlockPostingList::deserialize(&short).is_err());
        let at = raw.len() - 12;
        raw[at] |= 128;
        assert!(BlockPostingList::deserialize(&raw).is_err());
    }

    const ALL_CODECS: [PostingCodec; 4] = [
        PostingCodec::Rounded,
        PostingCodec::Packed,
        PostingCodec::Pfor,
        PostingCodec::Simd4x,
    ];

    /// Byte offset of the first document delta of `block` in the stream.
    fn first_delta_byte(list: &BlockPostingList, block: usize) -> usize {
        let (_, _, offset, _) = list.read_l0_entry(block);
        let header = offset as usize;
        // Pfor arrays start with their exception count.
        header + 8 + usize::from(list.block_codec(block) == Some(PostingCodec::Pfor))
    }

    /// Structurally valid bytes whose block content disagrees with the L0
    /// directory must end the cursor explicitly, never index out of range.
    #[test]
    fn content_corrupt_block_never_panics_or_truncates_silently() {
        for codec in ALL_CODECS {
            // Block 0: docs 0..128 (full); block 1: [200, 300] (a tail, so
            // Simd4x falls back to Rounded there).
            let mut postings = PostingList::new();
            for doc in 0..128 {
                postings.push(doc, doc % 5 + 1);
            }
            postings.push(200, 2);
            postings.push(300, 3);
            let list =
                BlockPostingList::from_posting_list_with_options(&postings, true, None, codec)
                    .unwrap();
            let bytes = serialize_bpl(&list);
            let at = first_delta_byte(&list, 1);
            assert_eq!(
                bytes[at], 100,
                "{codec}: delta 300-200 at the first payload byte"
            );
            let mut corrupt = bytes.clone();
            corrupt[at] = 1; // block 1 now decodes to [200, 201]; L0 still says 200..=300
            let admitted = BlockPostingList::deserialize(&corrupt)
                .unwrap_or_else(|e| panic!("{codec}: structural admission must pass: {e}"));
            // Before the content check this indexed past the decoded block.
            let mut cursor = admitted.iterator();
            assert_eq!(cursor.seek(250), TERMINATED, "{codec}");
            assert_eq!(cursor.term_freq(), 0);
            assert_eq!(cursor.advance(), TERMINATED);
            let mut decoded = Vec::new();
            assert!(
                admitted
                    .decode_block_doc_ids_only(0, &mut decoded)
                    .is_some()
            );
            assert_eq!(decoded.len(), 128);
            assert!(
                admitted
                    .decode_block_doc_ids_only(1, &mut decoded)
                    .is_none(),
                "{codec}: a content-corrupt block must be reported, not decoded"
            );
            assert!(decoded.is_empty(), "no ids escape a corrupt block");
            assert!(!admitted.decode_block_into(1, &mut decoded, &mut Vec::new()));
            // Sequential traversal stops at the corrupt block instead of
            // yielding ids outside the directory range.
            assert_eq!(
                collect_postings(&admitted),
                collect_postings(&list)[..128],
                "{codec}"
            );
            let mut window = admitted.iterator();
            let mut bits = [0u64; 8];
            window.fill_doc_window(0, &mut bits);
            assert_eq!(bits[..2], [u64::MAX, u64::MAX]);
            assert_eq!(window.doc(), TERMINATED);
            // Untouched bytes still decode completely.
            assert_eq!(
                collect_postings(&BlockPostingList::deserialize(&bytes).unwrap()).len(),
                130
            );
        }
        // A full Simd4x block: corrupt one lane word after the reserved zero
        // first gap so the decoded ids drift below the directory's last id.
        let mut postings = PostingList::new();
        for doc in 0..128 {
            postings.push(doc, 1);
        }
        for i in 0..128 {
            postings.push(200 + i * 2, 1);
        }
        let list = BlockPostingList::from_posting_list_with_codec(&postings, PostingCodec::Simd4x)
            .unwrap();
        assert_eq!(list.block_codec(1), Some(PostingCodec::Simd4x));
        let mut bytes = serialize_bpl(&list);
        let at = first_delta_byte(&list, 1) + 1;
        assert_ne!(bytes[at], 0);
        bytes[at] = 0;
        let admitted = BlockPostingList::deserialize(&bytes).unwrap();
        let mut cursor = admitted.iterator();
        assert_eq!(cursor.seek(450), TERMINATED);
        assert!(
            admitted
                .decode_block_doc_ids_only(1, &mut Vec::new())
                .is_none()
        );
        assert_eq!(collect_postings(&admitted).len(), 128);
    }

    /// Single-block list whose tail (count < 128) is encoded with codec id 3,
    /// as the first Simd4x prototype wrote; the builder now emits Rounded
    /// tails but readers keep accepting these bytes.
    fn simd_exact_tail_bytes(docs: &[(u32, u32)]) -> Vec<u8> {
        assert!(!docs.is_empty() && docs.len() < BLOCK_SIZE);
        let count = docs.len();
        let (first, last) = (docs[0].0, docs[count - 1].0);
        let deltas: Vec<u32> = docs.windows(2).map(|w| w[1].0 - w[0].0).collect();
        let tfs: Vec<u32> = docs.iter().map(|d| d.1).collect();
        let max_tf = tfs.iter().copied().max().unwrap();
        let mut stream = Vec::new();
        stream.write_u16::<LittleEndian>(count as u16).unwrap();
        stream.write_u32::<LittleEndian>(first).unwrap();
        let header_at = stream.len();
        stream.extend_from_slice(&[0, 0]);
        let doc_bits = bitpacking4x::encode_gaps(&deltas, &mut stream);
        let tf_bits = bitpacking4x::encode(&tfs, &mut stream);
        stream[header_at] = PostingCodec::Simd4x.header_byte(doc_bits);
        stream[header_at + 1] = tf_bits;
        let mut bytes = stream.clone();
        write_l0(&mut bytes, first, last, 0, pack_bounds(max_tf, 1));
        bytes.extend_from_slice(&last.to_le_bytes());
        bytes.extend_from_slice(&pack_bounds(max_tf, 1).to_le_bytes());
        BlockPostingList::write_footer(
            &mut bytes,
            stream.len() as u64,
            1,
            1,
            count as u32,
            max_tf,
            0,
            false,
            Some(1),
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap();
        bytes
    }

    #[test]
    fn simd4x_exact_tail_blocks_decode_seek_and_copy_through_merge() {
        for count in [1usize, 2, 3, 17, 127] {
            let docs: Vec<(u32, u32)> = (0..count as u32)
                .map(|i| (i * 3 + 7, i % 4 + 1 + (i == 5) as u32 * 900))
                .collect();
            let bytes = simd_exact_tail_bytes(&docs);
            let list = BlockPostingList::deserialize(&bytes).unwrap();
            assert_eq!(
                list.block_codec(0),
                Some(PostingCodec::Simd4x),
                "count={count}"
            );
            assert_eq!(collect_postings(&list), docs, "count={count}");
            let mut cursor = list.iterator();
            for &(doc, tf) in &docs {
                assert_eq!(cursor.seek(doc.saturating_sub(1)), doc);
                assert_eq!(cursor.term_freq(), tf);
            }
            assert_eq!(cursor.seek(docs[count - 1].0 + 1), TERMINATED);
            assert_eq!(serialize_bpl(&list), bytes, "byte-identical round trip");
            // Merges copy the tail verbatim, keeping codec id 3.
            let mut out = Vec::new();
            let (merged_count, written) =
                BlockPostingList::concatenate_streaming(&[(&bytes, 0), (&bytes, 1000)], &mut out)
                    .unwrap();
            assert_eq!((merged_count as usize, written), (2 * count, out.len()));
            let merged = BlockPostingList::deserialize(&out).unwrap();
            assert_eq!(merged.block_codec(1), Some(PostingCodec::Simd4x));
            let expected: Vec<(u32, u32)> = docs
                .iter()
                .copied()
                .chain(docs.iter().map(|&(d, t)| (d + 1000, t)))
                .collect();
            assert_eq!(collect_postings(&merged), expected, "count={count}");
            let typed =
                BlockPostingList::concatenate_blocks(&[(list.clone(), 0), (list.clone(), 1000)])
                    .unwrap();
            assert_eq!(serialize_bpl(&typed), out);
            // Header corruption of the tail is still caught structurally.
            let mut short = bytes.clone();
            short[0] = (count + 1) as u8;
            assert!(BlockPostingList::deserialize(&short).is_err());
        }
    }

    /// A legacy 24-byte footer ending in a `max_tf` equal to the extended
    /// footer magic is ambiguous; it must be rejected, never parsed as the
    /// extended layout.
    #[test]
    fn legacy_footer_whose_max_tf_equals_the_magic_is_rejected_not_misread() {
        for count in [1u32, 2, 130, 700] {
            let mut postings = PostingList::new();
            for i in 0..count {
                postings.push(i * 2, if i == 0 { FOOTER_MAGIC } else { 1 });
            }
            let bpl = BlockPostingList::from_posting_list(&postings).unwrap();
            assert_eq!(bpl.max_tf(), FOOTER_MAGIC);
            let bytes = serialize_bpl(&bpl);
            let footer = Footer::parse(&bytes).unwrap();
            let mut legacy = bytes[..footer.l1_end()].to_vec();
            for block in 0..bpl.num_blocks() {
                let at = footer.l0_start() + block * L0_SIZE + 12;
                legacy[at..at + 4]
                    .copy_from_slice(&(bpl.block_max_tf(block).unwrap() as f32).to_le_bytes());
            }
            legacy.extend_from_slice(
                &bytes[bytes.len() - FOOTER_V2_SIZE..bytes.len() - (FOOTER_V2_SIZE - FOOTER_SIZE)],
            );
            assert_eq!(
                u32::from_le_bytes(legacy[legacy.len() - 4..].try_into().unwrap()),
                FOOTER_MAGIC
            );
            assert!(
                BlockPostingList::deserialize(&legacy).is_err(),
                "count={count}: ambiguous footer must not be read as either layout"
            );
            assert!(!BlockPostingList::has_cursors_bytes(&legacy));
            let mut out = vec![0xab];
            assert!(BlockPostingList::concatenate_streaming(&[(&legacy, 0)], &mut out).is_err());
            assert_eq!(out, [0xab]);
            // The same postings with the extended footer read back exactly.
            assert_eq!(
                BlockPostingList::deserialize(&bytes).unwrap().max_tf(),
                FOOTER_MAGIC
            );
        }
    }

    #[test]
    fn mixed_cursor_and_cursorless_sources_fail_loudly_before_output() {
        let docs: Vec<(u32, u32)> = (0..300u32).map(|i| (i * 2, i % 3 + 1)).collect();
        let with = build_bpl_with_positions(&docs);
        let without = build_bpl(&docs);
        assert!(with.has_position_cursors() && !without.has_position_cursors());
        for sources in [
            [(with.clone(), 0), (without.clone(), 1000)],
            [(without.clone(), 0), (with.clone(), 1000)],
        ] {
            let error = BlockPostingList::concatenate_blocks(&sources).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("with and without position cursors"),
                "{error}"
            );
            let bytes: Vec<Vec<u8>> = sources.iter().map(|(s, _)| serialize_bpl(s)).collect();
            let mut out = vec![0xab];
            let result = BlockPostingList::concatenate_streaming(
                &[(&bytes[0], 0), (&bytes[1], 1000)],
                &mut out,
            );
            match result {
                Err(crate::Error::Corruption(message)) => {
                    assert!(
                        message.contains("with and without position cursors"),
                        "{message}"
                    )
                }
                other => panic!("expected a loud corruption error, got {other:?}"),
            }
            assert_eq!(out, [0xab], "nothing is written for a rejected source set");
        }
    }
}

#[cfg(test)]
#[path = "posting/merge_admission_tests.rs"]
mod merge_admission_tests;

#[cfg(test)]
mod byte_gap_validation_tests {
    use super::*;

    #[test]
    fn strict_gap_validation_agrees_with_full_scan_at_width_and_wrap_boundaries() {
        for count in 1..=BLOCK_SIZE {
            for width in 0..=32 {
                let mask = if width == 0 {
                    0
                } else {
                    u32::MAX >> (32 - width)
                };
                for first in [0u32, 17, u32::MAX / 2, u32::MAX - 1] {
                    for pattern in 0..3 {
                        let mut docs = vec![first];
                        for i in 1..count {
                            let encoded = match pattern {
                                0 => mask,
                                1 => 0,
                                _ => (i as u32).wrapping_mul(0x9e3779b9) & mask,
                            };
                            docs.push(docs.last().unwrap().wrapping_add(encoded).wrapping_add(1));
                        }
                        let end = *docs.last().unwrap();
                        for last in [end, end.wrapping_add(1), end.wrapping_sub(1)] {
                            assert_eq!(
                                verify_block_docs(&docs, first, last, None, Some(width)),
                                verify_block_docs(&docs, first, last, None, None),
                                "count={count}, width={width}, first={first}, pattern={pattern}, last={last}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn byte_gap_validation_agrees_with_decoded_order_for_tails_and_unsigned_wraps() {
        for count in 1..=BLOCK_SIZE {
            for first in [0u32, 17, u32::MAX - 400, u32::MAX - 1] {
                for pattern in 0..5 {
                    let gaps: Vec<u8> = (1..count)
                        .map(|i| match pattern {
                            0 => 1,
                            1 => 255,
                            2 => {
                                if i == count / 2 {
                                    0
                                } else {
                                    1
                                }
                            }
                            3 => (i * 71) as u8,
                            _ => ((i * 71) as u8).max(1),
                        })
                        .collect();
                    let mut docs = vec![first];
                    for &gap in &gaps {
                        docs.push(docs.last().unwrap().wrapping_add(u32::from(gap)));
                    }
                    let end = *docs.last().unwrap();
                    for last in [end, end.wrapping_add(1), end.wrapping_sub(1)] {
                        assert_eq!(
                            verify_block_docs(&docs, first, last, Some(&gaps), None),
                            verify_block_docs(&docs, first, last, None, None),
                            "count={count}, first={first}, pattern={pattern}, last={last}"
                        );
                    }
                }
            }
        }
    }
}
