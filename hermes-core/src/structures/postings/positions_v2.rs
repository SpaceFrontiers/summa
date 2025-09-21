//! Cursor-addressed position stream: positions addressed through the doc
//! postings. Current formats are POS5 (interleaved blocks) and POS6 (compact
//! directory). Earlier formats are rejected and require rebuilding.
//!
//! One stream per term, referenced by `TermInfo::External { position_offset,
//! position_len }`:
//!
//! ```text
//! [block 0][block 1]...[block n-1]
//! [block index: (byte_offset u32, value_start u64) × n]
//! [footer: num_blocks u32, total_positions u64, magic u32 "POS5"]   16 bytes
//! The high bit of num_blocks certifies unique positions within each document.
//! block: [count u16][bits u8][codec u8][packed values]
//! codec 0: rounded widths (0/8/16/32 bits), any count 1..=128.
//! codec 1: BitPacker4x, full 128-value blocks only. The encoder never emits
//!          codec 1 for a short block (short blocks are downgraded to codec 0)
//!          and the reader rejects one as corruption.
//! ```
//!
//! Opt-in POS6 stores payloads without interleaved headers, followed by one
//! `(byte_offset u32, value_start u64)` checkpoint per eight blocks and one
//! two-byte count/width/codec descriptor per block. The footer differs only
//! in its magic. Structural admission reads only this directory.
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

#[cfg(feature = "native")]
mod compact;
mod directory;
#[cfg(feature = "native")]
pub(crate) use compact::PositionRangeSource;

use std::io::{self, Write};

use byteorder::{LittleEndian, WriteBytesExt};

use super::{PostingCodec, bitpacking4x};
use crate::directories::OwnedBytes;
use crate::structures::simd;

/// Values per position block; the block of value `i` is `i / 128`.
pub const POSITION_STREAM_BLOCK: usize = 128;

const BLOCK_HEADER: usize = 4;
const INDEX_ENTRY: usize = 12;
const FOOTER: usize = 16;
/// POS5 uses the interleaved block layout; POS6 uses the compact directory.
const MAGIC: u32 = 0x3553_4F50;
/// High bit of the footer block count certifies unique positions per document.
const UNIQUE_POSITIONS: u32 = 1 << 31;

fn has_unique_positions(raw: &[u8]) -> bool {
    let at = raw.len() - FOOTER;
    u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) & UNIQUE_POSITIONS != 0
}

fn block_count_and_flags(count: usize, unique: bool) -> io::Result<u32> {
    let count = u32::try_from(count)
        .ok()
        .filter(|&count| count < UNIQUE_POSITIONS)
        .ok_or_else(|| io::Error::other("too many position blocks"))?;
    Ok(count | if unique { UNIQUE_POSITIONS } else { 0 })
}

fn footer_magic(compact: bool) -> u32 {
    if compact {
        directory::COMPACT_MAGIC
    } else {
        MAGIC
    }
}

/// Streaming writer of one term's position stream.
pub struct PositionStreamEncoder<W: Write> {
    writer: W,
    pending: Vec<u32>,
    index: Vec<(u32, u64)>,
    index_limit: Option<usize>,
    written: u64,
    total: u64,
    scratch: Vec<u8>,
    codec: PostingCodec,
    compact: bool,
    descriptors: Vec<u16>,
    unique_positions: bool,
}

impl<W: Write> PositionStreamEncoder<W> {
    pub fn new(writer: W) -> Self {
        Self::with_posting_codec(writer, PostingCodec::Rounded)
    }

    /// Use SIMD packing for `Simd4x`; other posting policies retain rounded
    /// positions. Copied encoded blocks preserve their own codec tags.
    pub fn with_posting_codec(writer: W, codec: PostingCodec) -> Self {
        Self {
            writer,
            pending: Vec::with_capacity(POSITION_STREAM_BLOCK),
            index: Vec::new(),
            index_limit: None,
            written: 0,
            total: 0,
            scratch: Vec::with_capacity(BLOCK_HEADER + POSITION_STREAM_BLOCK * 4),
            codec,
            compact: false,
            descriptors: Vec::new(),
            unique_positions: true,
        }
    }

    /// Emit POS6 metadata separately from payloads.
    pub fn with_compact_directory(mut self) -> Self {
        assert!(
            self.index.is_empty() && self.pending.is_empty(),
            "select the position format before appending values"
        );
        self.compact = true;
        self
    }

    /// Append one document's positions. They are sorted here and stored as
    /// deltas; the caller must append documents in posting order and keep the
    /// count equal to the term frequency stored in the doc postings.
    pub fn push_doc(&mut self, positions: &mut [u32]) -> io::Result<()> {
        positions.sort_unstable();
        let mut prev = 0u32;
        for (index, &position) in positions.iter().enumerate() {
            self.unique_positions &= index == 0 || position != prev;
            self.push_value(position - prev)?;
            prev = position;
        }
        Ok(())
    }

    /// Append already delta-coded values (re-packing another stream).
    pub fn push_values(&mut self, values: &[u32]) -> io::Result<()> {
        self.unique_positions = false;
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

    fn reserve_index_entry(&mut self) -> io::Result<()> {
        if let Some(limit) = self.index_limit {
            if self.index.len() >= limit {
                return Err(io::Error::other(
                    "position output directory exceeds compaction scratch budget",
                ));
            }
            if self.index.len() == self.index.capacity() {
                let capacity = self.index.capacity().saturating_mul(2).max(16).min(limit);
                self.index.reserve_exact(capacity - self.index.len());
            }
        }
        if self.compact && self.descriptors.len() == self.descriptors.capacity() {
            let limit = self.index_limit.unwrap_or(usize::MAX);
            let capacity = self
                .descriptors
                .capacity()
                .saturating_mul(2)
                .max(16)
                .min(limit);
            self.descriptors
                .reserve_exact(capacity - self.descriptors.len());
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
        self.reserve_index_entry()?;
        self.index
            .push((self.written as u32, self.total - self.pending.len() as u64));
        let count = self.pending.len();
        self.scratch.clear();
        self.scratch.resize(BLOCK_HEADER, 0);
        self.scratch[0..2].copy_from_slice(&(count as u16).to_le_bytes());
        if self.codec.for_count(count) == PostingCodec::Simd4x {
            let width = bitpacking4x::encode(&self.pending, &mut self.scratch);
            self.scratch[2] = width;
            self.scratch[3] = 1;
        } else {
            let max = self.pending.iter().copied().max().unwrap_or(0);
            let width = simd::RoundedBitWidth::from_exact(simd::bits_needed(max));
            self.scratch
                .resize(BLOCK_HEADER + count * width.bytes_per_value(), 0);
            self.scratch[2] = width.as_u8();
            simd::pack_rounded(&self.pending, width, &mut self.scratch[BLOCK_HEADER..]);
        }
        if self.compact {
            self.descriptors.push(directory::descriptor(&self.scratch)?);
            self.writer.write_all(&self.scratch[BLOCK_HEADER..])?;
            self.written += (self.scratch.len() - BLOCK_HEADER) as u64;
        } else {
            self.writer.write_all(&self.scratch)?;
            self.written += self.scratch.len() as u64;
        }
        self.pending.clear();
        Ok(())
    }

    fn append_encoded_block(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.unique_positions = false;
        let count = PositionStream::block_count(bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid copied position block")
        })?;
        self.flush_block()?;
        self.reserve_index_entry()?;
        let offset = u32::try_from(self.written)
            .map_err(|_| io::Error::other("position output exceeds u32 offsets"))?;
        self.index.push((offset, self.total));
        let payload = if self.compact {
            self.descriptors.push(directory::descriptor(bytes)?);
            &bytes[BLOCK_HEADER..]
        } else {
            bytes
        };
        self.writer.write_all(payload)?;
        self.written += payload.len() as u64;
        self.total += count as u64;
        Ok(())
    }

    /// Flush the tail block and write the offsets and footer. Returns
    /// `(total_positions, bytes_written)`.
    pub fn finish(self) -> io::Result<(u64, u64)> {
        self.finish_checked(|| Ok(()))
    }

    fn finish_checked(
        mut self,
        mut check: impl FnMut() -> io::Result<()>,
    ) -> io::Result<(u64, u64)> {
        check()?;
        self.flush_block()?;
        if self.compact {
            directory::write(&mut self.writer, &self.index, &self.descriptors, &mut check)?;
        } else {
            for (i, &(offset, value_start)) in self.index.iter().enumerate() {
                if i.is_multiple_of(4096) {
                    check()?;
                }
                self.writer.write_u32::<LittleEndian>(offset)?;
                self.writer.write_u64::<LittleEndian>(value_start)?;
            }
        }
        self.writer
            .write_u32::<LittleEndian>(block_count_and_flags(
                self.index.len(),
                self.unique_positions,
            )?)?;
        self.writer.write_u64::<LittleEndian>(self.total)?;
        self.writer
            .write_u32::<LittleEndian>(footer_magic(self.compact))?;
        let index_len = if self.compact {
            directory::directory_len(self.index.len()).unwrap()
        } else {
            self.index.len() * INDEX_ENTRY
        };
        let bytes = self.written + index_len as u64 + FOOTER as u64;
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
    compact: bool,
}

#[derive(Default)]
struct PositionBlockCache {
    index: Option<usize>,
    value_start: u64,
    values: Vec<u32>,
    #[cfg(test)]
    decodes: usize,
    #[cfg(test)]
    lookups: usize,
}

impl PositionBlockCache {
    #[inline]
    fn deltas(&self, cursor: u64, tf: u32) -> Option<&[u32]> {
        self.index?;
        let start = usize::try_from(cursor.checked_sub(self.value_start)?).ok()?;
        self.values.get(start..start.checked_add(tf as usize)?)
    }
}

impl PositionStream {
    /// Whether `raw` ends with a current position-stream footer.
    pub fn is_stream(raw: &[u8]) -> bool {
        directory::is_compact(raw)
            || (raw.len() >= FOOTER
                && u32::from_le_bytes(raw[raw.len() - 4..].try_into().unwrap()) == MAGIC)
    }

    pub fn open(bytes: OwnedBytes) -> io::Result<Self> {
        let (num_blocks, index_start, total) = Self::parse_layout(&bytes)?;
        Self::validate_blocks(&bytes, total)?;
        Ok(Self::from_layout(bytes, num_blocks, index_start, total))
    }

    /// Parse the envelope without auditing writer-produced directory or payload contents.
    pub(super) fn open_for_query(bytes: OwnedBytes) -> io::Result<Self> {
        let (num_blocks, index_start, total) = Self::parse_layout(&bytes)?;
        Ok(Self::from_layout(bytes, num_blocks, index_start, total))
    }

    fn from_layout(bytes: OwnedBytes, num_blocks: usize, index_start: usize, total: u64) -> Self {
        // Freshly encoded streams keep every interior block full, retaining
        // the original O(1) cursor-to-block calculation. Only concatenated
        // streams with partial interior source tails need the index search.
        let canonical_blocks = num_blocks == 0
            || Self::entry_for(&bytes, index_start, num_blocks, num_blocks - 1).1
                == (num_blocks as u64 - 1) * POSITION_STREAM_BLOCK as u64;
        Self {
            compact: directory::is_compact(&bytes),
            bytes,
            num_blocks,
            index_start,
            total,
            canonical_blocks,
        }
    }

    fn parse_layout(raw: &[u8]) -> io::Result<(usize, usize, u64)> {
        Self::parse_layout_tail(raw, raw.len())
    }

    fn parse_layout_tail(raw: &[u8], total_len: usize) -> io::Result<(usize, usize, u64)> {
        if !Self::is_stream(raw) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported position stream format; rebuild the index",
            ));
        }
        let footer_start = raw.len() - FOOTER;
        let num_blocks =
            (u32::from_le_bytes(raw[footer_start..footer_start + 4].try_into().unwrap())
                & !UNIQUE_POSITIONS) as usize;
        let total =
            u64::from_le_bytes(raw[footer_start + 4..footer_start + 12].try_into().unwrap());
        let index_len = (if directory::is_compact(raw) {
            directory::directory_len(num_blocks)
        } else {
            num_blocks.checked_mul(INDEX_ENTRY)
        })
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "position block index overflows")
        })?;
        let index_start = total_len
            .saturating_sub(FOOTER)
            .checked_sub(index_len)
            .ok_or_else(|| {
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

    fn entry_for(raw: &[u8], index_start: usize, blocks: usize, idx: usize) -> (usize, u64) {
        if directory::is_compact(raw) {
            directory::entry(&raw[index_start..raw.len() - FOOTER], blocks, idx)
        } else {
            Self::index_entry(raw, index_start, idx)
        }
    }

    #[inline]
    fn entry(&self, idx: usize) -> (usize, u64) {
        if self.compact {
            directory::entry(
                &self.bytes[self.index_start..self.bytes.len() - FOOTER],
                self.num_blocks,
                idx,
            )
        } else {
            Self::index_entry(self.bytes.as_slice(), self.index_start, idx)
        }
    }

    fn block_range(&self, idx: usize) -> Option<(usize, usize, u64)> {
        if idx >= self.num_blocks {
            return None;
        }
        let (start, value_start) = self.entry(idx);
        let end = if self.compact {
            let tag = directory::tag(&self.bytes[self.index_start..], self.num_blocks, idx);
            start + directory::parts(tag).ok()?.3
        } else if idx + 1 < self.num_blocks {
            self.entry(idx + 1).0
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
        // Mirror of the encoder: codec 1 (BitPacker4x) exists only for full
        // blocks; `flush_block` downgrades any short block to codec 0.
        let payload_len = match (raw[3], raw[2]) {
            (0, 0 | 8 | 16 | 32) => count * (raw[2] as usize / 8),
            (1, 0..=32) if count == POSITION_STREAM_BLOCK => {
                bitpacking4x::encoded_len(count, raw[2])
            }
            _ => return None,
        };
        (raw.len() == BLOCK_HEADER + payload_len).then_some(count)
    }

    fn locate_value(&self, cursor: u64, forward_from: Option<usize>) -> Option<(usize, usize)> {
        if cursor >= self.total || self.num_blocks == 0 {
            return None;
        }
        if self.canonical_blocks {
            let idx = usize::try_from(cursor / POSITION_STREAM_BLOCK as u64).ok()?;
            return Some((idx, (cursor % POSITION_STREAM_BLOCK as u64) as usize));
        }
        if self.compact {
            return directory::locate(
                &self.bytes[self.index_start..],
                self.num_blocks,
                cursor,
                forward_from,
            );
        }
        let mut low = 0usize;
        let mut high = self.num_blocks;
        if let Some(first) = forward_from {
            low = first.min(self.num_blocks);
            high = low.saturating_add(1).min(self.num_blocks);
            let mut step = 1usize;
            while high < self.num_blocks && self.entry(high).1 <= cursor {
                low = high;
                step = step.saturating_mul(2);
                high = high.saturating_add(step).min(self.num_blocks);
            }
        }
        while low < high {
            let mid = low + (high - low) / 2;
            let value_start = self.entry(mid).1;
            if value_start <= cursor {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        let idx = low.checked_sub(1)?;
        // Admission checked every logical span against its block header. The
        // upper-bound search and cursor < total place this cursor in that span.
        let value_start = self.entry(idx).1;
        let in_block = usize::try_from(cursor.checked_sub(value_start)?).ok()?;
        Some((idx, in_block))
    }

    /// Decode block `idx` (raw delta values) into `out`.
    pub fn decode_block(&self, idx: usize, out: &mut Vec<u32>) -> bool {
        let Some((start, end, _)) = self.block_range(idx) else {
            return false;
        };
        self.decode_payload(idx, &self.bytes.as_slice()[start..end], out)
    }

    fn decode_payload(&self, idx: usize, raw: &[u8], out: &mut Vec<u32>) -> bool {
        if !self.compact {
            return Self::decode_block_bytes(raw, out);
        }
        let tag = directory::tag(&self.bytes[self.index_start..], self.num_blocks, idx);
        let Ok((count, width, codec, bytes)) = directory::parts(tag) else {
            return false;
        };
        if bytes != raw.len() {
            return false;
        }
        out.resize(count, 0);
        crate::observe::search_work!(
            position_blocks += 1,
            position_values += count,
            position_payload_bytes += bytes
        );
        if codec == 1 {
            bitpacking4x::decode(raw, width, out);
        } else {
            simd::unpack_rounded(
                raw,
                simd::RoundedBitWidth::try_from_u8(width).unwrap(),
                out,
                count,
            );
        }
        true
    }

    fn decode_block_bytes(raw: &[u8], out: &mut Vec<u32>) -> bool {
        let Some(count) = Self::block_count(raw) else {
            return false;
        };
        out.resize(count, 0);
        crate::observe::search_work!(
            position_blocks += 1,
            position_values += count,
            position_payload_bytes += raw.len() - BLOCK_HEADER
        );
        if raw[3] == 1 {
            bitpacking4x::decode(&raw[BLOCK_HEADER..], raw[2], out);
        } else {
            // `block_count` already admitted only rounded widths; a `None`
            // here is a corrupt block and is reported as a failed decode.
            let Some(width) = simd::RoundedBitWidth::try_from_u8(raw[2]) else {
                out.clear();
                return false;
            };
            simd::unpack_rounded(&raw[BLOCK_HEADER..], width, out, count);
        }
        true
    }

    /// Positions of one document: the `tf` values starting at `cursor`,
    /// delta-decoded into absolute positions. `scratch` reuses allocation.
    ///
    /// Not for the query path: every call builds a fresh block cache, so
    /// consecutive documents in the same block are decoded again each time.
    /// Query iterators bind a `TermPositionCursor` (via
    /// `TermPositions::into_cursor`) which keeps the last decoded block.
    pub fn read_into(
        &self,
        cursor: u64,
        tf: u32,
        scratch: &mut Vec<u32>,
        out: &mut Vec<u32>,
    ) -> bool {
        let mut cache = PositionBlockCache {
            values: std::mem::take(scratch),
            ..Default::default()
        };
        let found = self.read_cached(cursor, tf, &mut cache, out);
        *scratch = cache.values;
        found
    }

    #[inline]
    fn read_cached(
        &self,
        cursor: u64,
        tf: u32,
        cache: &mut PositionBlockCache,
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
        // Most phrase reads stay in the previously decoded block. Its logical
        // range works for both canonical and copied short blocks.
        if let Some(deltas) = cache.deltas(cursor, tf) {
            let mut position = 0u32;
            out.extend(deltas.iter().map(|&delta| {
                position = position.wrapping_add(delta);
                position
            }));
            return true;
        }
        self.read_uncached(cursor, tf, cache, out)
    }

    fn read_uncached(
        &self,
        cursor: u64,
        tf: u32,
        cache: &mut PositionBlockCache,
        out: &mut Vec<u32>,
    ) -> bool {
        let located = if self.canonical_blocks {
            self.locate_value(cursor, None)
        } else {
            let cached = cache.index.and_then(|idx| {
                let offset = cursor.checked_sub(cache.value_start)?;
                (offset < cache.values.len() as u64).then_some((idx, offset as usize))
            });
            if cached.is_some() {
                cached
            } else {
                #[cfg(test)]
                {
                    cache.lookups += 1;
                }
                let forward = cache
                    .index
                    .filter(|_| cursor >= cache.value_start)
                    .map(|idx| idx + 1);
                self.locate_value(cursor, forward)
            }
        };
        let Some((mut idx, mut in_block)) = located else {
            return false;
        };
        let mut remaining = tf as usize;
        let mut next_cursor = cursor;
        let mut prev = 0u32;
        out.reserve(remaining);
        while remaining > 0 {
            if cache.index != Some(idx) {
                cache.index = None;
                let Some((start, end, value_start)) = self.block_range(idx) else {
                    return false;
                };
                // Also validates adjacency when one document spans copied
                // short blocks. A failed replacement cannot retain a stale range.
                if value_start.checked_add(in_block as u64) != Some(next_cursor)
                    || !self.decode_payload(
                        idx,
                        &self.bytes.as_slice()[start..end],
                        &mut cache.values,
                    )
                {
                    return false;
                }
                cache.value_start = value_start;
                cache.index = Some(idx);
                #[cfg(test)]
                {
                    cache.decodes += 1;
                }
            }
            if cache.values.len() <= in_block {
                return false;
            }
            let take = remaining.min(cache.values.len() - in_block);
            for &delta in &cache.values[in_block..in_block + take] {
                prev = prev.wrapping_add(delta);
                out.push(prev);
            }
            remaining -= take;
            next_cursor += take as u64;
            in_block = 0;
            idx += 1;
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

        if sources.iter().any(|raw| directory::is_compact(raw)) {
            let mut encoder = PositionStreamEncoder::new(writer).with_compact_directory();
            let mut block = Vec::with_capacity(BLOCK_HEADER + POSITION_STREAM_BLOCK * 4);
            for (raw, &(blocks, index_start, total)) in sources.iter().zip(&layouts) {
                Self::validate_blocks(raw, total)?;
                for idx in 0..blocks {
                    let (start, _) = Self::entry_for(raw, index_start, blocks, idx);
                    let end = if idx + 1 == blocks {
                        index_start
                    } else {
                        Self::entry_for(raw, index_start, blocks, idx + 1).0
                    };
                    if directory::is_compact(raw) {
                        let tag = directory::tag(&raw[index_start..], blocks, idx);
                        let (count, width, codec, _) = directory::parts(tag)?;
                        block.clear();
                        block.extend_from_slice(&(count as u16).to_le_bytes());
                        block.extend_from_slice(&[width, codec]);
                        block.extend_from_slice(&raw[start..end]);
                        encoder.append_encoded_block(&block)?;
                    } else {
                        encoder.append_encoded_block(&raw[start..end])?;
                    }
                }
            }
            encoder.unique_positions = sources.iter().all(|raw| has_unique_positions(raw));
            return Ok(encoder.finish()?);
        }

        let total_blocks: usize = layouts.iter().map(|(blocks, _, _)| *blocks).sum();
        let count_and_flags = block_count_and_flags(
            total_blocks,
            sources.iter().all(|raw| has_unique_positions(raw)),
        )?;
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
        writer.write_u32::<LittleEndian>(count_and_flags)?;
        writer.write_u64::<LittleEndian>(total_positions)?;
        writer.write_u32::<LittleEndian>(footer_magic(false))?;
        let bytes_written = data_written + out_index.len() as u64 + FOOTER as u64;
        Ok((total_positions, bytes_written))
    }

    fn validate_blocks(raw: &[u8], expected_total: u64) -> io::Result<()> {
        let (num_blocks, index_start, _) = Self::parse_layout(raw)?;
        if directory::is_compact(raw) {
            return directory::validate(
                &raw[index_start..raw.len() - FOOTER],
                num_blocks,
                index_start,
                expected_total,
            );
        }
        if (num_blocks == 0 && index_start != 0)
            || (num_blocks != 0 && Self::index_entry(raw, index_start, 0).0 != 0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid position block index: unaddressed payload prefix",
            ));
        }
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

/// Cursor-addressed positions of one term in the current on-disk format.
#[derive(Debug, Clone)]
pub struct TermPositions(pub(super) PositionStream);

/// Query-local cursor bound to one immutable term stream. At most one decoded
/// block (128 u32 values) is retained; seeking backwards safely replaces it.
pub(crate) struct TermPositionCursor {
    positions: TermPositions,
    cache: PositionBlockCache,
}

impl TermPositionCursor {
    /// A one-occurrence document stores its absolute position as one delta.
    /// Borrow the decoded value directly on a cache hit; block misses use the
    /// shared decoder and the caller's existing scratch.
    #[inline]
    pub(crate) fn read_one(&mut self, cursor: u64, scratch: &mut Vec<u32>) -> Option<u32> {
        if let Some(&position) = self
            .cache
            .deltas(cursor, 1)
            .and_then(|values| values.first())
        {
            crate::observe::search_work!(position_reads += 1, positions_requested += 1);
            return Some(position);
        }
        self.read_into(cursor, 1, scratch).then(|| scratch[0])
    }

    /// Exact membership in one document's sorted positions. Short cached
    /// ranges can stop at the target without materializing absolute positions.
    #[inline]
    pub(crate) fn contains(
        &mut self,
        cursor: u64,
        tf: u32,
        target: u32,
        scratch: &mut Vec<u32>,
    ) -> bool {
        if tf == 1 {
            return self.read_one(cursor, scratch) == Some(target);
        }
        if let Some(deltas) = self.cache.deltas(cursor, tf) {
            crate::observe::search_work!(position_reads += 1, positions_requested += tf);
            let mut position = 0u32;
            for &delta in deltas {
                position = position.wrapping_add(delta);
                if position >= target {
                    return position == target;
                }
            }
            return false;
        }
        self.read_into(cursor, tf, scratch) && scratch.binary_search(&target).is_ok()
    }

    pub(crate) fn read_into(&mut self, cursor: u64, tf: u32, out: &mut Vec<u32>) -> bool {
        crate::observe::search_work!(position_reads += 1, positions_requested += tf);
        self.positions
            .0
            .read_cached(cursor, tf, &mut self.cache, out)
    }
}

impl TermPositions {
    pub(crate) fn has_unique_positions(&self) -> bool {
        has_unique_positions(&self.0.bytes)
    }

    /// Representation policy retained by explicit field reordering.
    #[cfg(all(feature = "native", test))]
    pub(crate) fn has_compact_directory(&self) -> bool {
        self.0.compact
    }

    pub(crate) fn into_cursor(self) -> TermPositionCursor {
        TermPositionCursor {
            positions: self,
            cache: PositionBlockCache::default(),
        }
    }
    pub fn open(bytes: OwnedBytes) -> io::Result<Self> {
        Ok(Self(PositionStream::open(bytes)?))
    }

    /// Positions addressed by `cursor` and `tf` from the doc-posting
    /// iterator positioned on that document.
    pub fn positions_into(
        &self,
        cursor: u64,
        tf: u32,
        scratch: &mut Vec<u32>,
        out: &mut Vec<u32>,
    ) -> bool {
        self.0.read_into(cursor, tf, scratch, out)
    }

    /// Convenience for tests and diagnostics only: allocates a fresh output
    /// and scratch vector per call and uses the uncached
    /// [`PositionStream::read_into`]. Query code must use
    /// `Self::into_cursor` / `TermPositionCursor::read_into` with reused
    /// buffers.
    pub fn positions(&self, cursor: u64, tf: u32) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        let mut scratch = Vec::new();
        self.positions_into(cursor, tf, &mut scratch, &mut out)
            .then_some(out)
    }
}

#[cfg(test)]
mod compact_directory_tests {
    use super::*;

    fn encode(values: &[u32], compact: bool, codec: PostingCodec) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
        if compact {
            encoder = encoder.with_compact_directory();
        }
        encoder.push_values(values).unwrap();
        encoder.finish().unwrap();
        bytes
    }

    #[test]
    fn singleton_reads_preserve_cached_values_across_short_blocks_and_reverse_probes() {
        for codec in [PostingCodec::Rounded, PostingCodec::Simd4x] {
            for compact in [false, true] {
                let values: Vec<_> = (0..267u32).map(|i| i.wrapping_mul(1234567)).collect();
                let first = encode(&values[..13], compact, codec);
                let second = encode(&values[13..], compact, codec);
                let mut bytes = Vec::new();
                PositionStream::concatenate_streaming(&[&first, &second], &mut bytes).unwrap();
                let original = bytes.clone();
                let mut cursor = TermPositions::open(OwnedBytes::new(bytes))
                    .unwrap()
                    .into_cursor();
                let mut scratch = Vec::new();
                for index in (0..values.len()).chain((0..values.len()).rev()) {
                    assert_eq!(
                        cursor.read_one(index as u64, &mut scratch),
                        Some(values[index])
                    );
                    let decodes = cursor.cache.decodes;
                    assert_eq!(
                        cursor.read_one(index as u64, &mut scratch),
                        Some(values[index])
                    );
                    assert_eq!(cursor.cache.decodes, decodes);
                }
                assert_eq!(cursor.read_one(values.len() as u64, &mut scratch), None);
                assert_eq!(cursor.read_one(u64::MAX, &mut scratch), None);
                assert_eq!(cursor.positions.0.bytes.as_slice(), original);
            }
        }
    }

    #[test]
    fn compact_positions_preserve_payloads_and_every_cursor_across_mixed_short_blocks() {
        for codec in [PostingCodec::Rounded, PostingCodec::Simd4x] {
            for count in [0, 1, 2, 127, 128, 129, 1024, 1025, 4097] {
                let values: Vec<u32> = (0..count).map(|i| (i * 31 % 257) as u32).collect();
                let old = encode(&values, false, codec);
                let new = encode(&values, true, codec);
                let interleaved = PositionStream::open(OwnedBytes::new(old.clone())).unwrap();
                let compact = PositionStream::open(OwnedBytes::new(new.clone())).unwrap();
                assert!(compact.compact);
                assert_eq!(
                    new.len(),
                    compact.index_start
                        + directory::directory_len(compact.num_blocks).unwrap()
                        + FOOTER
                );
                for i in 0..compact.num_blocks {
                    let (a, b, _) = interleaved.block_range(i).unwrap();
                    let (c, d, _) = compact.block_range(i).unwrap();
                    assert_eq!(&old[a + BLOCK_HEADER..b], &new[c..d]);
                }
                let tail = encode(&[7, 3, 11], true, codec);
                let mut merged = Vec::new();
                PositionStream::concatenate_streaming(&[&tail, &old, &new, &tail], &mut merged)
                    .unwrap();
                let stream = PositionStream::open(OwnedBytes::new(merged)).unwrap();
                let expected: Vec<_> =
                    [vec![7, 3, 11], values.clone(), values, vec![7, 3, 11]].concat();
                let mut scratch = Vec::new();
                let mut out = Vec::new();
                for (i, &value) in expected.iter().enumerate() {
                    assert!(stream.read_into(i as u64, 1, &mut scratch, &mut out));
                    assert_eq!(out, [value]);
                }
            }
        }
    }

    #[test]
    fn compact_directory_rejects_bad_structure_without_interpreting_payload_values() {
        let bytes = encode(&vec![7; 1153], true, PostingCodec::Rounded);
        let (_, start, _) = PositionStream::parse_layout(&bytes).unwrap();
        // Payload values are arbitrary valid deltas; admission only needs metadata.
        let mut changed = bytes.clone();
        changed[..start].fill(255);
        assert!(PositionStream::open(OwnedBytes::new(changed)).is_ok());
        for at in [
            start,
            start + 4,
            start + 12,
            start + 16,
            bytes.len() - FOOTER - 1,
        ] {
            let mut corrupt = bytes.clone();
            corrupt[at] ^= 128;
            assert!(
                PositionStream::open(OwnedBytes::new(corrupt)).is_err(),
                "byte {at}"
            );
        }
        for n in 0..bytes.len() {
            assert!(PositionStream::open(OwnedBytes::new(bytes[..n].to_vec())).is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_position_open_rejects_overflowing_checkpoint_before_deriving_layout() {
        let mut bytes = Vec::new();
        let mut encoder = PositionStreamEncoder::new(&mut bytes).with_compact_directory();
        encoder.push_values(&[1; 256]).unwrap();
        encoder.finish().unwrap();
        let (_, index_start, _) = PositionStream::parse_layout(&bytes).unwrap();
        bytes[index_start + 4..index_start + 12].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(PositionStream::open(OwnedBytes::new(bytes)).is_err());
    }

    #[test]
    fn position_decoding_overwrites_stale_values_across_lengths_and_codecs() {
        let mut out = vec![u32::MAX; 256];
        for codec in [PostingCodec::Rounded, PostingCodec::Simd4x] {
            for count in [128, 1, 127, 128, 17] {
                for value in [0, 1, 255, 65536] {
                    for compact in [false, true] {
                        let mut bytes = Vec::new();
                        let mut encoder =
                            PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                        if compact {
                            encoder = encoder.with_compact_directory();
                        }
                        encoder.push_values(&vec![value; count]).unwrap();
                        encoder.finish().unwrap();
                        let stream = PositionStream::open(OwnedBytes::new(bytes)).unwrap();
                        out.fill(u32::MAX);
                        assert!(stream.decode_block(0, &mut out));
                        assert_eq!(out, vec![value; count]);
                    }
                }
            }
        }
    }

    #[test]
    fn admitted_mixed_position_directories_locate_every_cursor_across_copied_short_blocks() {
        let counts = [1, 127, 128, 3, 65, 128, 7];
        let mut sources = Vec::new();
        let mut expected = Vec::new();
        for (block, &count) in counts.iter().enumerate() {
            let mut bytes = Vec::new();
            let codec = if block % 2 == 0 {
                PostingCodec::Rounded
            } else {
                PostingCodec::Simd4x
            };
            let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
            encoder.push_values(&vec![1; count]).unwrap();
            encoder.finish().unwrap();
            sources.push(bytes);
            expected.extend((0..count).map(|offset| (block, offset)));
        }
        let mut bytes = Vec::new();
        PositionStream::concatenate_streaming(
            &sources.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut bytes,
        )
        .unwrap();
        let original = bytes.clone();
        let stream = PositionStream::open(OwnedBytes::new(bytes)).unwrap();
        assert!(!stream.canonical_blocks);
        for cursor in (0..expected.len()).rev().chain(0..expected.len()) {
            let expected = expected[cursor];
            assert_eq!(stream.locate_value(cursor as u64, None), Some(expected));
            for first in 0..=expected.0 {
                assert_eq!(
                    stream.locate_value(cursor as u64, Some(first)),
                    Some(expected)
                );
            }
            let mut values = Vec::new();
            assert!(stream.read_into(cursor as u64, 1, &mut Vec::new(), &mut values));
            assert_eq!(values, [1]);
        }
        assert_eq!(stream.locate_value(expected.len() as u64, None), None);
        assert_eq!(stream.locate_value(u64::MAX, Some(usize::MAX)), None);
        assert_eq!(stream.bytes.as_slice(), original);
    }

    #[test]
    fn position_uniqueness_is_certified_per_document_and_conjoined_by_copying_merge() {
        for compact in [false, true] {
            let encode = |docs: &[Vec<u32>]| {
                let mut bytes = Vec::new();
                let mut encoder = PositionStreamEncoder::new(&mut bytes);
                if compact {
                    encoder = encoder.with_compact_directory();
                }
                for doc in docs {
                    encoder.push_doc(&mut doc.clone()).unwrap();
                }
                encoder.finish().unwrap();
                bytes
            };
            let unique = encode(&[vec![7, 0, 1], vec![0, 3, u32::MAX]]);
            let repeated = encode(&[vec![0, 3, 3]]);
            assert!(has_unique_positions(&unique));
            assert!(!has_unique_positions(&repeated));
            assert_eq!(directory::is_compact(&unique), compact);
            let mut uncertified = unique.clone();
            let end = uncertified.len();
            let count = u32::from_le_bytes(
                uncertified[end - FOOTER..end - FOOTER + 4]
                    .try_into()
                    .unwrap(),
            ) & !UNIQUE_POSITIONS;
            uncertified[end - FOOTER..end - FOOTER + 4].copy_from_slice(&count.to_le_bytes());
            assert!(TermPositions::open(OwnedBytes::new(uncertified.clone())).is_ok());
            for second in [&unique, &repeated, &uncertified] {
                let mut merged = Vec::new();
                PositionStream::concatenate_streaming(&[&unique, second], &mut merged).unwrap();
                let result = PositionStream::open(OwnedBytes::new(merged.clone())).unwrap();
                assert_eq!(has_unique_positions(&merged), has_unique_positions(second));
                // Each source's packed blocks remain byte-identical.
                let mut block = 0;
                for source in [&unique, second] {
                    let source = PositionStream::open(OwnedBytes::new(source.clone())).unwrap();
                    for index in 0..source.num_blocks() {
                        let (a, b, _) = source.block_range(index).unwrap();
                        let (c, d, _) = result.block_range(block).unwrap();
                        assert_eq!(&source.bytes[a..b], &result.bytes[c..d]);
                        block += 1;
                    }
                }
            }
            let mut raw = Vec::new();
            let mut encoder = PositionStreamEncoder::new(&mut raw);
            encoder.push_values(&[0, 1, 2]).unwrap();
            encoder.finish().unwrap();
            assert!(
                !has_unique_positions(&raw),
                "raw values cannot prove document boundaries"
            );
        }
    }

    #[test]
    fn sequential_merged_position_reads_reuse_logical_block_addresses() {
        let docs = vec![vec![1, 5]; 13];
        let (source, _) = encode(&docs);
        let mut bytes = Vec::new();
        PositionStream::concatenate_streaming(&vec![source.as_slice(); 30], &mut bytes).unwrap();
        let original = bytes.clone();
        let mut cursor = TermPositions::open(OwnedBytes::new(bytes))
            .unwrap()
            .into_cursor();
        let mut out = Vec::new();
        for doc in 0..390u32 {
            assert!(cursor.read_into(u64::from(doc) * 2, 2, &mut out));
            assert_eq!(out, [1, 5]);
        }
        assert_eq!(cursor.cache.decodes, 30);
        assert_eq!(
            cursor.cache.lookups, 30,
            "cached block addresses must serve all covered documents"
        );
        assert!(cursor.read_into(0, 2, &mut out));
        assert_eq!(out, [1, 5]);
        assert!(cursor.read_into(778, 2, &mut out));
        assert_eq!(out, [1, 5]);
        let stream = cursor.positions.0;
        assert_eq!(stream.bytes.as_slice(), original);
    }

    #[test]
    fn merged_position_cursor_handles_forward_gaps_backward_reads_and_spanning_documents() {
        let mut docs = Vec::new();
        let mut sources = Vec::new();
        for source in 0..40 {
            let part: Vec<Vec<u32>> = (0..17)
                .map(|doc| {
                    (0..1 + (source * 17 + doc) % 301)
                        .map(|p| p * 3 + doc)
                        .collect()
                })
                .collect();
            sources.push(encode(&part).0);
            docs.extend(part);
        }
        let mut bytes = Vec::new();
        PositionStream::concatenate_streaming(
            &sources.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut bytes,
        )
        .unwrap();
        let before = bytes.clone();
        let mut cursor = TermPositions::open(OwnedBytes::new(bytes))
            .unwrap()
            .into_cursor();
        let mut starts = Vec::new();
        let mut next = 0u64;
        for doc in &docs {
            starts.push(next);
            next += doc.len() as u64;
        }
        let mut out = Vec::new();
        for at in (0..docs.len())
            .step_by(7)
            .chain((0..docs.len()).rev())
            .chain(0..docs.len())
        {
            assert!(cursor.read_into(starts[at], docs[at].len() as u32, &mut out));
            assert_eq!(out, docs[at], "document {at}");
        }
        let stream = cursor.positions.0;
        assert_eq!(stream.bytes.as_slice(), before);
    }

    #[test]
    fn failed_cross_block_position_read_does_not_publish_a_stale_cached_range() {
        let (a, _) = encode(&[vec![1, 5, 9]]);
        let (b, _) = encode(&[(0..300).collect()]);
        let mut bytes = Vec::new();
        PositionStream::concatenate_streaming(&[&a, &b], &mut bytes).unwrap();
        let valid = PositionStream::open(OwnedBytes::new(bytes.clone())).unwrap();
        let (start, _, _) = valid.block_range(2).unwrap();
        bytes[start + 2] = 7; // Unsupported width in the middle of a spanning document.
        assert!(TermPositions::open(OwnedBytes::new(bytes.clone())).is_err());
        // Bypass public admission only inside this test to exercise defensive
        // cache replacement after a failed decode. Public open rejects it.
        let mut malformed = valid;
        malformed.bytes = OwnedBytes::new(bytes);
        let mut cursor = TermPositions(malformed).into_cursor();
        let mut out = Vec::new();
        assert!(cursor.read_into(0, 3, &mut out));
        assert!(!cursor.read_into(3, 300, &mut out));
        assert_eq!(cursor.cache.index, None);
        assert!(cursor.read_into(0, 3, &mut out));
        assert_eq!(out, [1, 5, 9]);
    }

    #[test]
    fn position_open_rejects_invalid_headers_directories_and_unaddressed_bytes() {
        let (bytes, _) = encode(&[(0..300).collect()]);
        let (_, index_start, _) = PositionStream::parse_layout(&bytes).unwrap();
        for case in 0..7 {
            let mut bad = bytes.clone();
            match case {
                0 => bad[2] = 7,
                1 => bad[3] = 2,
                2 => bad[index_start..index_start + 4].copy_from_slice(&1u32.to_le_bytes()),
                3 => bad[index_start + 4..index_start + 12].copy_from_slice(&1u64.to_le_bytes()),
                4 => bad[index_start + INDEX_ENTRY..index_start + INDEX_ENTRY + 4]
                    .copy_from_slice(&u32::MAX.to_le_bytes()),
                5 => bad[index_start + INDEX_ENTRY + 4..index_start + INDEX_ENTRY + 12]
                    .copy_from_slice(&0u64.to_le_bytes()),
                6 => {
                    let total_at = bad.len() - FOOTER + 4;
                    bad[total_at..total_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
                }
                _ => unreachable!(),
            }
            assert!(
                PositionStream::open(OwnedBytes::new(bad)).is_err(),
                "case {case}"
            );
        }
        let (mut empty, _) = encode(&[]);
        assert!(PositionStream::open(OwnedBytes::new(empty.clone())).is_ok());
        empty.insert(0, 0);
        assert!(PositionStream::open(OwnedBytes::new(empty)).is_err());
        let stream = PositionStream::open(OwnedBytes::new(bytes.clone())).unwrap();
        assert_eq!(stream.bytes.as_slice(), bytes);
    }

    #[test]
    fn position_codecs_round_trip_every_width_and_short_tail() {
        for codec in [PostingCodec::Rounded, PostingCodec::Simd4x] {
            for count in [1, 3, 31, 127, 128, 129, 257] {
                for width in 0..=32 {
                    let max = if width == 0 {
                        0
                    } else {
                        u32::MAX >> (32 - width)
                    };
                    let values: Vec<_> = (0..count)
                        .map(|i| if i % 3 == 0 { max } else { 0 })
                        .collect();
                    let mut bytes = Vec::new();
                    let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
                    encoder.push_values(&values).unwrap();
                    encoder.finish().unwrap();
                    let stream = PositionStream::open(OwnedBytes::new(bytes.clone())).unwrap();
                    let mut actual = Vec::new();
                    let mut block = Vec::new();
                    for idx in 0..stream.num_blocks() {
                        assert!(stream.decode_block(idx, &mut block));
                        actual.extend_from_slice(&block);
                        let (start, end, _) = stream.block_range(idx).unwrap();
                        assert_eq!(
                            bytes[start + 3],
                            u8::from(
                                codec == PostingCodec::Simd4x
                                    && block.len() == POSITION_STREAM_BLOCK
                            )
                        );
                        if bytes[start + 3] == 0 {
                            let rounded = simd::RoundedBitWidth::from_exact(simd::bits_needed(
                                block.iter().copied().max().unwrap(),
                            ));
                            let mut expected = Vec::new();
                            expected.extend_from_slice(&(block.len() as u16).to_le_bytes());
                            expected.extend_from_slice(&[rounded.as_u8(), 0]);
                            for value in &block {
                                expected.extend_from_slice(
                                    &value.to_le_bytes()[..rounded.bytes_per_value()],
                                );
                            }
                            assert_eq!(&bytes[start..end], expected);
                        }
                    }
                    assert_eq!(actual, values);
                    for (at, replacement) in [(2, 33), (3, 2)] {
                        let mut corrupt = bytes.clone();
                        corrupt[at] = replacement;
                        assert!(PositionStream::open(OwnedBytes::new(corrupt)).is_err());
                    }
                    let mut copied = Vec::new();
                    PositionStream::concatenate_streaming(&[&bytes], &mut copied).unwrap();
                    assert_eq!(copied, bytes);
                }
            }
        }
    }

    #[test]
    fn mixed_position_codecs_copy_short_interior_blocks_and_preserve_cursors() {
        let mut encoded = Vec::new();
        let mut docs = Vec::new();
        for codec in [
            PostingCodec::Simd4x,
            PostingCodec::Rounded,
            PostingCodec::Simd4x,
        ] {
            let mut bytes = Vec::new();
            let mut encoder = PositionStreamEncoder::with_posting_codec(&mut bytes, codec);
            for count in [1, 130, 7, 127] {
                let mut positions: Vec<_> = (0..count).map(|i| i * 3 + 4).collect();
                encoder.push_doc(&mut positions).unwrap();
                docs.push(positions);
            }
            encoder.finish().unwrap();
            encoded.push(bytes);
        }
        let mut output = Vec::new();
        PositionStream::concatenate_streaming(
            &encoded.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut output,
        )
        .unwrap();
        let stream = PositionStream::open(OwnedBytes::new(output.clone())).unwrap();
        assert!(!stream.canonical_blocks);
        let mut next = 0;
        for source in encoded {
            let source = PositionStream::open(OwnedBytes::new(source)).unwrap();
            for idx in 0..source.num_blocks() {
                let (start, end, _) = source.block_range(idx).unwrap();
                let (out_start, out_end, _) = stream.block_range(next).unwrap();
                assert_eq!(
                    &output[out_start..out_end],
                    &source.bytes.as_slice()[start..end]
                );
                next += 1;
            }
        }
        let mut starts = vec![0u64];
        for doc in &docs {
            starts.push(starts.last().unwrap() + doc.len() as u64);
        }
        let mut cursor = TermPositions(stream).into_cursor();
        let mut actual = Vec::new();
        for i in (0..docs.len()).chain((0..docs.len()).rev()) {
            assert!(cursor.read_into(starts[i], docs[i].len() as u32, &mut actual));
            assert_eq!(actual, docs[i]);
        }
    }

    #[test]
    fn term_cursor_reuses_blocks_and_isolates_terms_and_backward_seeks() {
        let docs: Vec<Vec<u32>> = (0..200).map(|i| vec![i, i + 3]).collect();
        let (bytes, _) = encode(&docs);
        let mut cursor = TermPositions::open(OwnedBytes::new(bytes.clone()))
            .unwrap()
            .into_cursor();
        let mut out = Vec::new();
        for (id, expected) in docs.iter().enumerate() {
            assert!(cursor.read_into(id as u64 * 2, 2, &mut out));
            assert_eq!(&out, expected);
        }
        assert_eq!(
            cursor.cache.decodes,
            400usize.div_ceil(POSITION_STREAM_BLOCK)
        );
        assert!(cursor.read_into(0, 2, &mut out));
        assert_eq!(out, docs[0]);
        let decodes = cursor.cache.decodes;
        assert!(cursor.read_into(2, 2, &mut out));
        assert_eq!(cursor.cache.decodes, decodes);
        let (other_bytes, _) = encode(&[vec![99, 100]]);
        let mut other = TermPositions::open(OwnedBytes::new(other_bytes))
            .unwrap()
            .into_cursor();
        assert!(other.read_into(0, 2, &mut out));
        assert_eq!(out, vec![99, 100]);
        assert!(!cursor.read_into(u64::MAX, 2, &mut out));
        assert!(cursor.read_into(0, 0, &mut out));
        assert!(out.is_empty());
        let stream = cursor.positions.0;
        assert_eq!(
            stream.bytes.as_slice(),
            bytes,
            "reading must not modify encoded blocks"
        );
    }

    #[test]
    fn term_cursor_handles_copied_short_blocks_and_cross_block_documents() {
        let first = vec![vec![1, 4, 9]; 13];
        let second = vec![vec![10, 20, 30]; 80];
        let (a, _) = encode(&first);
        let (b, _) = encode(&second);
        let mut merged = Vec::new();
        PositionStream::concatenate_streaming(&[&a, &b], &mut merged).unwrap();
        let mut cursor = TermPositions::open(OwnedBytes::new(merged))
            .unwrap()
            .into_cursor();
        let mut out = Vec::new();
        for (doc, expected) in first.iter().chain(&second).enumerate() {
            assert!(cursor.read_into(doc as u64 * 3, 3, &mut out));
            assert_eq!(&out, expected);
        }
        assert_eq!(cursor.cache.decodes, 3);
        assert!(cursor.cache.values.capacity() <= POSITION_STREAM_BLOCK);
    }

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
    fn raw_repacking_preserves_payload_bytes_without_certifying_document_boundaries() {
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
        let (mut direct, _) = encode(&all);
        assert!(has_unique_positions(&direct));
        assert!(!has_unique_positions(&merged));
        let at = direct.len() - FOOTER;
        let count = u32::from_le_bytes(direct[at..at + 4].try_into().unwrap()) & !UNIQUE_POSITIONS;
        direct[at..at + 4].copy_from_slice(&count.to_le_bytes());
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

    /// The encoder only emits BitPacker4x (codec 1) for full 128-value
    /// blocks; a short codec-1 block is an encoding no writer produces and
    /// must be rejected at open instead of being decoded by the tail path.
    #[test]
    fn short_bitpacker_block_is_rejected_as_corruption() {
        fn assemble(count: usize, width: u8, codec: u8, payload_len: usize) -> Vec<u8> {
            let mut raw = Vec::new();
            raw.extend_from_slice(&(count as u16).to_le_bytes());
            raw.extend_from_slice(&[width, codec]);
            raw.extend(std::iter::repeat_n(0u8, payload_len));
            let block_len = raw.len();
            raw.write_u32::<LittleEndian>(0).unwrap();
            raw.write_u64::<LittleEndian>(0).unwrap();
            raw.write_u32::<LittleEndian>(1).unwrap();
            raw.write_u64::<LittleEndian>(count as u64).unwrap();
            raw.write_u32::<LittleEndian>(MAGIC).unwrap();
            assert_eq!(raw.len(), block_len + INDEX_ENTRY + FOOTER);
            raw
        }
        for count in [1usize, 5, 64, 127] {
            for width in [0u8, 3, 8, 32] {
                let short = assemble(count, width, 1, bitpacking4x::encoded_len(count, width));
                let block = &short[..short.len() - INDEX_ENTRY - FOOTER];
                assert_eq!(
                    PositionStream::block_count(block),
                    None,
                    "count={count} width={width}"
                );
                assert!(
                    PositionStream::open(OwnedBytes::new(short)).is_err(),
                    "count={count} width={width}"
                );
            }
        }
        // The same shapes with codec 0 (what the encoder actually emits for a
        // short block) and a full codec-1 block remain admitted.
        for count in [1usize, 5, 64, 127] {
            let rounded = assemble(count, 8, 0, count);
            assert!(PositionStream::open(OwnedBytes::new(rounded)).is_ok());
        }
        let full = assemble(
            POSITION_STREAM_BLOCK,
            3,
            1,
            bitpacking4x::encoded_len(POSITION_STREAM_BLOCK, 3),
        );
        let stream = PositionStream::open(OwnedBytes::new(full)).unwrap();
        let mut block = Vec::new();
        assert!(stream.decode_block(0, &mut block));
        assert_eq!(block, vec![0; POSITION_STREAM_BLOCK]);
        // The encoder agrees: a Simd4x stream with a short tail tags it 0.
        let mut bytes = Vec::new();
        let mut encoder =
            PositionStreamEncoder::with_posting_codec(&mut bytes, PostingCodec::Simd4x);
        encoder.push_values(&[1; 130]).unwrap();
        encoder.finish().unwrap();
        let stream = PositionStream::open(OwnedBytes::new(bytes.clone())).unwrap();
        let (start, _, _) = stream.block_range(0).unwrap();
        assert_eq!(bytes[start + 3], 1);
        let (start, _, _) = stream.block_range(1).unwrap();
        assert_eq!(bytes[start + 3], 0);
    }

    #[test]
    fn term_positions_reject_old_stream_revisions() {
        let (buf, _) = encode(&[vec![1, 4], vec![0]]);
        let positions = TermPositions::open(OwnedBytes::new(buf.clone())).unwrap();
        assert_eq!(positions.positions(0, 2), Some(vec![1, 4]));
        for magic in [b"POS3", b"POS4", b"POS7"] {
            let mut bytes = buf.clone();
            let at = bytes.len() - 4;
            bytes[at..].copy_from_slice(magic);
            assert!(TermPositions::open(OwnedBytes::new(bytes)).is_err());
        }
    }
}
