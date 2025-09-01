//! Position stream v2: positions addressed through the doc postings.
//!
//! One stream per term, referenced by `TermInfo::External { position_offset,
//! position_len }`:
//!
//! ```text
//! [block 0][block 1]...[block n-1]
//! [block offsets: u32 × n]                      byte offset of each block
//! [footer: num_blocks u32, total_positions u64, magic u32 "POS2"]   16 bytes
//! block: [count u16][bits u8][pad u8][packed values: count × bytes_per_value(bits)]
//! ```
//!
//! The values form one flat sequence in posting order: for every document
//! (or chunk) its sorted positions, delta-coded (`p0, p1 - p0, ...`). Blocks
//! hold exactly [`POSITION_STREAM_BLOCK`] values except the last one, so the
//! `i`-th value lives in block `i / 128`. The doc postings record, per doc
//! block, how many values precede the block ([`BlockPostingList::pos_cursor`])
//! and the posting iterator adds the term frequencies of the postings before
//! the current one ([`BlockPostingIterator::position_cursor`]); a reader then
//! decodes only the one or two blocks covering `[cursor, cursor + tf)`.
//!
//! Because the stream is doc-agnostic, a merge re-packs the sources' values
//! into fresh 128-value blocks without decoding deltas, and the cursors of the
//! merged doc postings are the sources' cursors shifted by the number of
//! values that precede each source.
//!
//! [`BlockPostingList::pos_cursor`]: super::BlockPostingList::pos_cursor
//! [`BlockPostingIterator::position_cursor`]: super::BlockPostingIterator::position_cursor

use std::io::{self, Write};

use byteorder::{LittleEndian, WriteBytesExt};

use super::positions::PositionPostingList;
use crate::DocId;
use crate::directories::OwnedBytes;
use crate::structures::simd;

/// Values per position block; the block of value `i` is `i / 128`.
pub const POSITION_STREAM_BLOCK: usize = 128;

const BLOCK_HEADER: usize = 4;
const FOOTER: usize = 16;
/// "POS2" little-endian.
const MAGIC: u32 = 0x3253_4F50;

/// Streaming writer of one term's position stream.
pub struct PositionStreamEncoder<W: Write> {
    writer: W,
    pending: Vec<u32>,
    offsets: Vec<u32>,
    written: u64,
    total: u64,
    scratch: Vec<u8>,
}

impl<W: Write> PositionStreamEncoder<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            pending: Vec::with_capacity(POSITION_STREAM_BLOCK),
            offsets: Vec::new(),
            written: 0,
            total: 0,
            scratch: Vec::with_capacity(BLOCK_HEADER + POSITION_STREAM_BLOCK * 4),
        }
    }

    /// Append one document's positions. They are sorted here and stored as
    /// deltas; the caller must append documents in posting order and keep the
    /// count equal to the term frequency stored in the doc postings.
    pub fn push_doc(&mut self, positions: &mut [u32]) -> io::Result<()> {
        positions.sort_unstable();
        let mut prev = 0u32;
        for &position in positions.iter() {
            self.push_value(position - prev)?;
            prev = position;
        }
        Ok(())
    }

    /// Append already delta-coded values (re-packing another stream).
    pub fn push_values(&mut self, values: &[u32]) -> io::Result<()> {
        for &value in values {
            self.push_value(value)?;
        }
        Ok(())
    }

    #[inline]
    fn push_value(&mut self, value: u32) -> io::Result<()> {
        self.pending.push(value);
        self.total += 1;
        if self.pending.len() == POSITION_STREAM_BLOCK {
            self.flush_block()?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if self.written > u32::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "position stream exceeds u32::MAX bytes",
            ));
        }
        self.offsets.push(self.written as u32);
        let max = self.pending.iter().copied().max().unwrap_or(0);
        let width = simd::RoundedBitWidth::from_exact(simd::bits_needed(max));
        let count = self.pending.len();
        self.scratch.clear();
        self.scratch
            .resize(BLOCK_HEADER + count * width.bytes_per_value(), 0);
        self.scratch[0..2].copy_from_slice(&(count as u16).to_le_bytes());
        self.scratch[2] = width.as_u8();
        self.scratch[3] = 0;
        simd::pack_rounded(&self.pending, width, &mut self.scratch[BLOCK_HEADER..]);
        self.writer.write_all(&self.scratch)?;
        self.written += self.scratch.len() as u64;
        self.pending.clear();
        Ok(())
    }

    /// Flush the tail block and write the offsets and footer. Returns
    /// `(total_positions, bytes_written)`.
    pub fn finish(mut self) -> io::Result<(u64, u64)> {
        self.flush_block()?;
        for &offset in &self.offsets {
            self.writer.write_u32::<LittleEndian>(offset)?;
        }
        self.writer
            .write_u32::<LittleEndian>(self.offsets.len() as u32)?;
        self.writer.write_u64::<LittleEndian>(self.total)?;
        self.writer.write_u32::<LittleEndian>(MAGIC)?;
        let bytes = self.written + (self.offsets.len() * 4) as u64 + FOOTER as u64;
        Ok((self.total, bytes))
    }
}

/// Zero-copy reader of one term's position stream.
#[derive(Debug, Clone)]
pub struct PositionStream {
    bytes: OwnedBytes,
    num_blocks: usize,
    offsets_start: usize,
    total: u64,
}

impl PositionStream {
    /// Whether `raw` ends with a v2 footer (legacy lists end with a doc count).
    pub fn is_stream(raw: &[u8]) -> bool {
        raw.len() >= FOOTER && u32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) == MAGIC
    }

    pub fn open(bytes: OwnedBytes) -> io::Result<Self> {
        let raw = bytes.as_slice();
        if !Self::is_stream(raw) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "position stream footer missing",
            ));
        }
        let f = raw.len() - FOOTER;
        let num_blocks = u32::from_le_bytes(raw[f..f + 4].try_into().unwrap()) as usize;
        let total = u64::from_le_bytes(raw[f + 4..f + 12].try_into().unwrap());
        let offsets_start = f.checked_sub(num_blocks * 4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "position stream offsets table longer than the stream",
            )
        })?;
        Ok(Self {
            bytes,
            num_blocks,
            offsets_start,
            total,
        })
    }

    pub fn total_positions(&self) -> u64 {
        self.total
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn block_range(&self, idx: usize) -> Option<(usize, usize)> {
        if idx >= self.num_blocks {
            return None;
        }
        let raw = self.bytes.as_slice();
        let read_offset = |i: usize| {
            let p = self.offsets_start + i * 4;
            u32::from_le_bytes(raw[p..p + 4].try_into().unwrap()) as usize
        };
        let start = read_offset(idx);
        let end = if idx + 1 < self.num_blocks {
            read_offset(idx + 1)
        } else {
            self.offsets_start
        };
        (start <= end && end <= self.offsets_start).then_some((start, end))
    }

    /// Decode block `idx` (raw delta values) into `out`.
    pub fn decode_block(&self, idx: usize, out: &mut Vec<u32>) -> bool {
        let Some((start, end)) = self.block_range(idx) else {
            return false;
        };
        let raw = &self.bytes.as_slice()[start..end];
        if raw.len() < BLOCK_HEADER {
            return false;
        }
        let count = u16::from_le_bytes(raw[0..2].try_into().unwrap()) as usize;
        let width = simd::RoundedBitWidth::from_u8(raw[2]);
        let needed = BLOCK_HEADER + count * width.bytes_per_value();
        if raw.len() < needed || count > POSITION_STREAM_BLOCK {
            return false;
        }
        out.clear();
        out.resize(count, 0);
        simd::unpack_rounded(&raw[BLOCK_HEADER..needed], width, out, count);
        true
    }

    /// Positions of one document: the `tf` values starting at `cursor`,
    /// delta-decoded into absolute positions. `scratch` holds one decoded
    /// block between calls.
    pub fn read_into(
        &self,
        cursor: u64,
        tf: u32,
        scratch: &mut Vec<u32>,
        out: &mut Vec<u32>,
    ) -> bool {
        out.clear();
        if tf == 0 {
            return true;
        }
        if cursor + tf as u64 > self.total {
            return false;
        }
        let mut idx = (cursor / POSITION_STREAM_BLOCK as u64) as usize;
        let mut in_block = (cursor % POSITION_STREAM_BLOCK as u64) as usize;
        let mut remaining = tf as usize;
        let mut prev = 0u32;
        out.reserve(remaining);
        while remaining > 0 {
            if !self.decode_block(idx, scratch) || scratch.len() <= in_block {
                return false;
            }
            let take = remaining.min(scratch.len() - in_block);
            for &delta in &scratch[in_block..in_block + take] {
                prev = prev.wrapping_add(delta);
                out.push(prev);
            }
            remaining -= take;
            in_block = 0;
            idx += 1;
        }
        true
    }
}

/// Positions of one term in either on-disk format.
#[derive(Debug, Clone)]
pub enum TermPositions {
    /// Pre-v2 list with its own per-block skip entries and absolute positions.
    Legacy(PositionPostingList),
    /// Cursor-addressed stream (see the module docs).
    Stream(PositionStream),
}

impl TermPositions {
    pub fn open(bytes: OwnedBytes) -> io::Result<Self> {
        if PositionStream::is_stream(bytes.as_slice()) {
            Ok(TermPositions::Stream(PositionStream::open(bytes)?))
        } else {
            Ok(TermPositions::Legacy(PositionPostingList::deserialize(
                bytes.as_slice(),
            )?))
        }
    }

    /// Positions of `doc_id`; `cursor` and `tf` come from the doc-posting
    /// iterator positioned on that document.
    pub fn positions_into(
        &self,
        doc_id: DocId,
        cursor: u64,
        tf: u32,
        scratch: &mut Vec<u32>,
        out: &mut Vec<u32>,
    ) -> bool {
        match self {
            TermPositions::Legacy(list) => list.get_positions_into(doc_id, out),
            TermPositions::Stream(stream) => stream.read_into(cursor, tf, scratch, out),
        }
    }

    pub fn positions(&self, doc_id: DocId, cursor: u64, tf: u32) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        let mut scratch = Vec::new();
        self.positions_into(doc_id, cursor, tf, &mut scratch, &mut out)
            .then_some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(docs: &[Vec<u32>]) -> (Vec<u8>, u64) {
        let mut buf = Vec::new();
        let mut encoder = PositionStreamEncoder::new(&mut buf);
        for doc in docs {
            let mut positions = doc.clone();
            encoder.push_doc(&mut positions).unwrap();
        }
        let (total, bytes) = encoder.finish().unwrap();
        assert_eq!(bytes as usize, buf.len());
        (buf, total)
    }

    fn read_all(stream: &PositionStream, docs: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut cursor = 0u64;
        let mut scratch = Vec::new();
        let mut out = Vec::new();
        let mut result = Vec::new();
        for doc in docs {
            assert!(stream.read_into(cursor, doc.len() as u32, &mut scratch, &mut out));
            result.push(out.clone());
            cursor += doc.len() as u64;
        }
        result
    }

    #[test]
    fn stream_round_trips_sorted_positions_across_blocks() {
        let docs: Vec<Vec<u32>> = (0..50)
            .map(|d| (0..(d % 7 + 1) * 13).map(|i| i * 3 + d).collect())
            .chain(std::iter::once((0..300).map(|i| i * 1000).collect()))
            .chain(std::iter::once(vec![70_000, 5, 5, 1 << 21]))
            .collect();
        let (buf, total) = encode(&docs);
        assert_eq!(total, docs.iter().map(|d| d.len() as u64).sum::<u64>());
        assert!(PositionStream::is_stream(&buf));
        let stream = PositionStream::open(OwnedBytes::new(buf)).unwrap();
        assert_eq!(stream.total_positions(), total);
        assert_eq!(stream.num_blocks(), total.div_ceil(128) as usize);
        let expected: Vec<Vec<u32>> = docs
            .iter()
            .map(|d| {
                let mut s = d.clone();
                s.sort_unstable();
                s
            })
            .collect();
        assert_eq!(read_all(&stream, &docs), expected);
        // Out-of-range reads fail instead of aliasing another document.
        let mut scratch = Vec::new();
        let mut out = Vec::new();
        assert!(!stream.read_into(total - 1, 2, &mut scratch, &mut out));
    }

    #[test]
    fn repacking_values_reproduces_the_stream() {
        let a: Vec<Vec<u32>> = (0..40).map(|d| vec![d, d + 2, d + 7]).collect();
        let b: Vec<Vec<u32>> = (0..90).map(|d| (0..d % 5 + 1).collect()).collect();
        let (buf_a, _) = encode(&a);
        let (buf_b, _) = encode(&b);
        let mut merged = Vec::new();
        let mut encoder = PositionStreamEncoder::new(&mut merged);
        let mut values = Vec::new();
        for buf in [buf_a, buf_b] {
            let stream = PositionStream::open(OwnedBytes::new(buf)).unwrap();
            for idx in 0..stream.num_blocks() {
                assert!(stream.decode_block(idx, &mut values));
                encoder.push_values(&values).unwrap();
            }
        }
        encoder.finish().unwrap();
        let all: Vec<Vec<u32>> = a.iter().chain(&b).cloned().collect();
        let (direct, _) = encode(&all);
        assert_eq!(merged, direct);
    }

    /// Size of the two formats on a synthetic "content" term: run with
    /// `cargo test -p hermes-core --lib -- --ignored --nocapture positions_v2`.
    #[test]
    #[ignore]
    fn size_comparison_report() {
        // 20k chunks of ~200 tokens, term frequency 1–4, positions spread
        // over the chunk (the shape of a mid-frequency stem in `content`).
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let docs: Vec<Vec<u32>> = (0..20_000)
            .map(|_| {
                let tf = (rng() % 4 + 1) as usize;
                let mut p: Vec<u32> = (0..tf).map(|_| (rng() % 200) as u32).collect();
                p.sort_unstable();
                p
            })
            .collect();
        let mut legacy = PositionPostingList::new();
        for (i, doc) in docs.iter().enumerate() {
            legacy.push(i as u32 * 3, doc.clone());
        }
        let mut legacy_bytes = Vec::new();
        legacy.serialize(&mut legacy_bytes).unwrap();
        let (v2, total) = encode(&docs);
        let cursors = docs.len().div_ceil(128) * 8;
        eprintln!(
            "positions: legacy {} B, v2 {} B (+{} B cursors) for {} values -> {:.2} vs {:.2} B/pos",
            legacy_bytes.len(),
            v2.len(),
            cursors,
            total,
            legacy_bytes.len() as f64 / total as f64,
            (v2.len() + cursors) as f64 / total as f64
        );
    }

    #[test]
    fn term_positions_bridges_the_legacy_format() {
        let mut legacy = PositionPostingList::new();
        legacy.push(3, vec![1, 4]);
        legacy.push(9, vec![0]);
        let mut bytes = Vec::new();
        legacy.serialize(&mut bytes).unwrap();
        assert!(!PositionStream::is_stream(&bytes));
        let positions = TermPositions::open(OwnedBytes::new(bytes)).unwrap();
        assert!(matches!(positions, TermPositions::Legacy(_)));
        assert_eq!(positions.positions(3, 0, 2), Some(vec![1, 4]));
        assert_eq!(positions.positions(9, 2, 1), Some(vec![0]));
        assert_eq!(positions.positions(4, 0, 1), None);

        let (buf, _) = encode(&[vec![1, 4], vec![0]]);
        let positions = TermPositions::open(OwnedBytes::new(buf)).unwrap();
        assert!(matches!(positions, TermPositions::Stream(_)));
        assert_eq!(positions.positions(3, 0, 2), Some(vec![1, 4]));
        assert_eq!(positions.positions(9, 2, 1), Some(vec![0]));
    }
}
