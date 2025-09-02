//! Position stream v3: positions addressed through the doc postings.
//!
//! One stream per term, referenced by `TermInfo::External { position_offset,
//! position_len }`:
//!
//! ```text
//! [block 0][block 1]...[block n-1]
//! [block index: (byte_offset u32, value_start u64) × n]
//! [footer: num_blocks u32, total_positions u64, magic u32 "POS3"]   16 bytes
//! block: [count u16][bits u8][pad u8][packed values: count × bytes_per_value(bits)]
//! ```
//!
//! The values form one flat sequence in posting order: for every document
//! (or chunk) its sorted positions, delta-coded (`p0, p1 - p0, ...`). Blocks
//! hold at most [`POSITION_STREAM_BLOCK`] values. The block index records both
//! the physical byte offset and logical value start, so interior blocks may be
//! short. The doc postings record, per doc block, how many values precede the
//! block ([`BlockPostingList::pos_cursor`]) and the posting iterator adds the
//! term frequencies of the postings before the current one
//! ([`BlockPostingIterator::position_cursor`]).
//!
//! Because the logical starts do not require interior blocks to be full, merge
//! copies every encoded source block verbatim and rebuilds only the block index
//! and footer. The cursors of merged doc postings are shifted by the number of
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
const INDEX_ENTRY: usize = 12;
const FOOTER: usize = 16;
/// "POS3" little-endian.
const MAGIC: u32 = 0x3353_4F50;

/// Streaming writer of one term's position stream.
pub struct PositionStreamEncoder<W: Write> {
    writer: W,
    pending: Vec<u32>,
    index: Vec<(u32, u64)>,
    written: u64,
    total: u64,
    scratch: Vec<u8>,
}

impl<W: Write> PositionStreamEncoder<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            pending: Vec::with_capacity(POSITION_STREAM_BLOCK),
            index: Vec::new(),
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
        self.index
            .push((self.written as u32, self.total - self.pending.len() as u64));
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
        for &(offset, value_start) in &self.index {
            self.writer.write_u32::<LittleEndian>(offset)?;
            self.writer.write_u64::<LittleEndian>(value_start)?;
        }
        self.writer
            .write_u32::<LittleEndian>(self.index.len() as u32)?;
        self.writer.write_u64::<LittleEndian>(self.total)?;
        self.writer.write_u32::<LittleEndian>(MAGIC)?;
        let bytes = self.written + (self.index.len() * INDEX_ENTRY) as u64 + FOOTER as u64;
        Ok((self.total, bytes))
    }
}

/// Zero-copy reader of one term's position stream.
#[derive(Debug, Clone)]
pub struct PositionStream {
    bytes: OwnedBytes,
    num_blocks: usize,
    index_start: usize,
    total: u64,
    canonical_blocks: bool,
}

impl PositionStream {
    /// Whether `raw` ends with a current position-stream footer.
    pub fn is_stream(raw: &[u8]) -> bool {
        raw.len() >= FOOTER && u32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) == MAGIC
    }

    pub fn open(bytes: OwnedBytes) -> io::Result<Self> {
        let (num_blocks, index_start, total) = Self::parse_layout(bytes.as_slice())?;
        // Freshly encoded streams keep every interior block full, retaining
        // the original O(1) cursor-to-block calculation. Only concatenated
        // streams with partial interior source tails need the index search.
        let canonical_blocks = num_blocks == 0
            || Self::index_entry(bytes.as_slice(), index_start, num_blocks - 1).1
                == (num_blocks as u64 - 1) * POSITION_STREAM_BLOCK as u64;
        Ok(Self {
            bytes,
            num_blocks,
            index_start,
            total,
            canonical_blocks,
        })
    }

    fn parse_layout(raw: &[u8]) -> io::Result<(usize, usize, u64)> {
        if !Self::is_stream(raw) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "current position stream footer missing",
            ));
        }
        let footer_start = raw.len() - FOOTER;
        let num_blocks =
            u32::from_le_bytes(raw[footer_start..footer_start + 4].try_into().unwrap()) as usize;
        let total =
            u64::from_le_bytes(raw[footer_start + 4..footer_start + 12].try_into().unwrap());
        let index_len = num_blocks.checked_mul(INDEX_ENTRY).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "position block index overflows")
        })?;
        let index_start = footer_start.checked_sub(index_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "position block index longer than the stream",
            )
        })?;
        Ok((num_blocks, index_start, total))
    }

    pub fn total_positions(&self) -> u64 {
        self.total
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    #[inline]
    fn index_entry(raw: &[u8], index_start: usize, idx: usize) -> (usize, u64) {
        let p = index_start + idx * INDEX_ENTRY;
        (
            u32::from_le_bytes(raw[p..p + 4].try_into().unwrap()) as usize,
            u64::from_le_bytes(raw[p + 4..p + 12].try_into().unwrap()),
        )
    }

    fn block_range(&self, idx: usize) -> Option<(usize, usize, u64)> {
        if idx >= self.num_blocks {
            return None;
        }
        let raw = self.bytes.as_slice();
        let (start, value_start) = Self::index_entry(raw, self.index_start, idx);
        let end = if idx + 1 < self.num_blocks {
            Self::index_entry(raw, self.index_start, idx + 1).0
        } else {
            self.index_start
        };
        (start <= end && end <= self.index_start).then_some((start, end, value_start))
    }

    fn block_count(raw: &[u8]) -> Option<usize> {
        if raw.len() < BLOCK_HEADER {
            return None;
        }
        let count = u16::from_le_bytes(raw[0..2].try_into().unwrap()) as usize;
        if count == 0 || count > POSITION_STREAM_BLOCK {
            return None;
        }
        let bytes_per_value = match raw[2] {
            0 => 0,
            8 => 1,
            16 => 2,
            32 => 4,
            _ => return None,
        };
        (raw.len() == BLOCK_HEADER + count * bytes_per_value).then_some(count)
    }

    fn locate_value(&self, cursor: u64) -> Option<(usize, usize)> {
        if cursor >= self.total || self.num_blocks == 0 {
            return None;
        }
        if self.canonical_blocks {
            let idx = usize::try_from(cursor / POSITION_STREAM_BLOCK as u64).ok()?;
            return Some((idx, (cursor % POSITION_STREAM_BLOCK as u64) as usize));
        }
        let raw = self.bytes.as_slice();
        let mut low = 0usize;
        let mut high = self.num_blocks;
        while low < high {
            let mid = low + (high - low) / 2;
            let value_start = Self::index_entry(raw, self.index_start, mid).1;
            if value_start <= cursor {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        let idx = low.checked_sub(1)?;
        let (start, end, value_start) = self.block_range(idx)?;
        let count = Self::block_count(&raw[start..end])?;
        let in_block = usize::try_from(cursor.checked_sub(value_start)?).ok()?;
        (in_block < count).then_some((idx, in_block))
    }

    /// Decode block `idx` (raw delta values) into `out`.
    pub fn decode_block(&self, idx: usize, out: &mut Vec<u32>) -> bool {
        let Some((start, end, _)) = self.block_range(idx) else {
            return false;
        };
        let raw = &self.bytes.as_slice()[start..end];
        let Some(count) = Self::block_count(raw) else {
            return false;
        };
        let width = simd::RoundedBitWidth::from_u8(raw[2]);
        let needed = BLOCK_HEADER + count * width.bytes_per_value();
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
        if cursor
            .checked_add(tf as u64)
            .is_none_or(|end| end > self.total)
        {
            return false;
        }
        let Some((mut idx, mut in_block)) = self.locate_value(cursor) else {
            return false;
        };
        let mut remaining = tf as usize;
        let mut next_cursor = cursor;
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
            next_cursor += take as u64;
            in_block = 0;
            idx += 1;
            if remaining > 0
                && self
                    .block_range(idx)
                    .is_none_or(|(_, _, value_start)| value_start != next_cursor)
            {
                return false;
            }
        }
        true
    }

    /// Concatenate current-format streams by copying encoded blocks verbatim.
    /// Only the compact block index and footer are rebuilt.
    pub fn concatenate_streaming<W: Write>(
        sources: &[&[u8]],
        writer: &mut W,
    ) -> crate::Result<(u64, u64)> {
        let layouts: Vec<_> = sources
            .iter()
            .map(|raw| Self::parse_layout(raw))
            .collect::<io::Result<_>>()?;

        // The common single-source/zero-offset merge can preserve the entire
        // stream, including its already valid index and footer.
        if sources.len() == 1 {
            let raw = sources[0];
            let total = layouts[0].2;
            Self::validate_blocks(raw, total)?;
            writer.write_all(raw)?;
            return Ok((total, raw.len() as u64));
        }

        let total_blocks: usize = layouts.iter().map(|(blocks, _, _)| *blocks).sum();
        let total_blocks_u32 = u32::try_from(total_blocks).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "position stream has more than u32::MAX blocks",
            )
        })?;
        let mut out_index = Vec::with_capacity(total_blocks * INDEX_ENTRY);
        let mut data_written = 0u64;
        let mut total_positions = 0u64;

        for (raw, &(num_blocks, index_start, source_total)) in sources.iter().zip(&layouts) {
            let mut expected_start = 0u64;
            for idx in 0..num_blocks {
                let (start, value_start) = Self::index_entry(raw, index_start, idx);
                let end = if idx + 1 < num_blocks {
                    Self::index_entry(raw, index_start, idx + 1).0
                } else {
                    index_start
                };
                if start > end || end > index_start || value_start != expected_start {
                    return Err(crate::Error::Corruption(
                        "invalid position block index during merge".into(),
                    ));
                }
                let block = &raw[start..end];
                let count = Self::block_count(block).ok_or_else(|| {
                    crate::Error::Corruption("invalid position block during merge".into())
                })?;
                if data_written > u32::MAX as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "position stream exceeds u32::MAX bytes during merge",
                    )
                    .into());
                }
                out_index.write_u32::<LittleEndian>(data_written as u32)?;
                out_index.write_u64::<LittleEndian>(
                    total_positions.checked_add(value_start).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "position count overflows u64 during merge",
                        )
                    })?,
                )?;
                writer.write_all(block)?;
                data_written += block.len() as u64;
                expected_start += count as u64;
            }
            if expected_start != source_total {
                return Err(crate::Error::Corruption(
                    "position stream total does not match its blocks".into(),
                ));
            }
            total_positions = total_positions.checked_add(source_total).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "position count overflows u64 during merge",
                )
            })?;
        }

        writer.write_all(&out_index)?;
        writer.write_u32::<LittleEndian>(total_blocks_u32)?;
        writer.write_u64::<LittleEndian>(total_positions)?;
        writer.write_u32::<LittleEndian>(MAGIC)?;
        let bytes_written = data_written + out_index.len() as u64 + FOOTER as u64;
        Ok((total_positions, bytes_written))
    }

    fn validate_blocks(raw: &[u8], expected_total: u64) -> io::Result<()> {
        let (num_blocks, index_start, _) = Self::parse_layout(raw)?;
        let mut total = 0u64;
        for idx in 0..num_blocks {
            let (start, value_start) = Self::index_entry(raw, index_start, idx);
            let end = if idx + 1 < num_blocks {
                Self::index_entry(raw, index_start, idx + 1).0
            } else {
                index_start
            };
            if start > end || end > index_start || value_start != total {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid position block index",
                ));
            }
            total += Self::block_count(&raw[start..end]).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid position block")
            })? as u64;
        }
        if total != expected_total {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "position stream total does not match its blocks",
            ));
        }
        Ok(())
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
        assert!(stream.canonical_blocks);
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

    #[test]
    fn streaming_concatenation_copies_non_aligned_blocks_verbatim() {
        // Both sources end with partial blocks. A merged stream therefore has
        // an interior short block and exercises the v3 logical-start index.
        let a: Vec<Vec<u32>> = (0..43)
            .map(|doc| {
                (0..doc % 5 + 1)
                    .map(|position| doc + position * 7)
                    .collect()
            })
            .collect();
        let b: Vec<Vec<u32>> = (0..51)
            .map(|doc| {
                (0..doc % 4 + 1)
                    .map(|position| doc * 2 + position)
                    .collect()
            })
            .collect();
        let (encoded_a, total_a) = encode(&a);
        let (encoded_b, total_b) = encode(&b);
        assert_ne!(total_a % POSITION_STREAM_BLOCK as u64, 0);
        assert_ne!(total_b % POSITION_STREAM_BLOCK as u64, 0);

        let source_blocks = |raw: &[u8]| {
            let stream = PositionStream::open(OwnedBytes::new(raw.to_vec())).unwrap();
            (0..stream.num_blocks())
                .map(|idx| {
                    let (start, end, _) = stream.block_range(idx).unwrap();
                    raw[start..end].to_vec()
                })
                .collect::<Vec<_>>()
        };
        let expected_blocks: Vec<Vec<u8>> = source_blocks(&encoded_a)
            .into_iter()
            .chain(source_blocks(&encoded_b))
            .collect();

        let mut merged = Vec::new();
        let (total, written) = PositionStream::concatenate_streaming(
            &[encoded_a.as_slice(), encoded_b.as_slice()],
            &mut merged,
        )
        .unwrap();
        assert_eq!(total, total_a + total_b);
        assert_eq!(written as usize, merged.len());

        let stream = PositionStream::open(OwnedBytes::new(merged.clone())).unwrap();
        assert!(!stream.canonical_blocks);
        let actual_blocks: Vec<Vec<u8>> = (0..stream.num_blocks())
            .map(|idx| {
                let (start, end, _) = stream.block_range(idx).unwrap();
                merged[start..end].to_vec()
            })
            .collect();
        assert_eq!(actual_blocks, expected_blocks, "encoded blocks changed");
        assert_eq!(stream.num_blocks(), expected_blocks.len());

        let all: Vec<Vec<u32>> = a.iter().chain(&b).cloned().collect();
        let expected: Vec<Vec<u32>> = all
            .iter()
            .map(|positions| {
                let mut positions = positions.clone();
                positions.sort_unstable();
                positions
            })
            .collect();
        assert_eq!(read_all(&stream, &all), expected);
    }

    #[test]
    fn single_source_streaming_concatenation_is_an_exact_copy() {
        let (encoded, total) = encode(&[(0..137).collect()]);
        let mut copied = Vec::new();
        let result =
            PositionStream::concatenate_streaming(&[encoded.as_slice()], &mut copied).unwrap();
        assert_eq!(result, (total, encoded.len() as u64));
        assert_eq!(copied, encoded);
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
