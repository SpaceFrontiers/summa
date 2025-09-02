//! Virtual-id maps of chunked text fields (`seg_<id>.chunks`).
//!
//! A text field declared `chunked` indexes every value as its own scoring
//! unit: term postings and positions are keyed by a dense, segment-local
//! **virtual id** instead of the document id. This file maps each virtual id
//! back to `(doc_id, ordinal)` and records the chunk's token count for BM25
//! length normalisation. See `docs/chunked-text-fields.md`.
//!
//! ```text
//! [magic "CHNK"][version u32 = 2][num_sections u32]
//! TOC × num_sections: [field_id u32][kind u32][count u32][total_tokens u64][data_offset u64]
//! kind 0 (chunk map):   doc_ids u32 × n | ordinals u16 × n | lengths u16 × n
//! kind 1 (doc lengths): lengths u16 × num_docs        (norms of a plain text field)
//! ```
//!
//! Version 1 files have 24-byte entries without `kind` and hold chunk maps
//! only; they are still read.
//!
//! Virtual ids are assigned in indexing order, and documents are indexed in
//! doc-id order, so `doc_ids` starts out non-decreasing. A reorder pass on a
//! field with the `reorder` attribute permutes the virtual ids (BP over the
//! field's postings, `segment/text_reorder.rs`); no query path depends on the
//! order. Merges concatenate sections and add the document offset to
//! `doc_ids`; ordinals and lengths are copied verbatim.
//!
//! A doc-length section stores the token count of the field in every
//! document of the segment (0 when the document has no value), so BM25 can
//! normalise plain fields by their real length instead of `tf`.

use std::io::{self, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rustc_hash::FxHashMap;

use crate::DocId;
use crate::directories::OwnedBytes;

const MAGIC: u32 = 0x4B4E_4843; // "CHNK"
const VERSION: u32 = 2;
const HEADER_SIZE: usize = 12;
const TOC_ENTRY_SIZE_V1: usize = 24;
const TOC_ENTRY_SIZE: usize = 28;
const KIND_CHUNK_MAP: u32 = 0;
const KIND_DOC_LENGTHS: u32 = 1;

/// Token count stored per chunk; longer chunks saturate.
pub const MAX_CHUNK_LENGTH: u32 = u16::MAX as u32;

/// In-memory map of one chunked field while a segment is being built.
#[derive(Debug, Default, Clone)]
pub struct ChunkMapBuilder {
    doc_ids: Vec<DocId>,
    ordinals: Vec<u16>,
    lengths: Vec<u16>,
    total_tokens: u64,
}

impl ChunkMapBuilder {
    /// Number of chunks so far (the next virtual id).
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Register the next chunk. Returns its virtual id.
    pub fn push(&mut self, doc_id: DocId, ordinal: u16, token_count: u32) -> io::Result<u32> {
        let vid = u32::try_from(self.doc_ids.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunked text field exceeds u32::MAX chunks in one segment",
            )
        })?;
        self.doc_ids.push(doc_id);
        self.ordinals.push(ordinal);
        self.lengths.push(token_count.min(MAX_CHUNK_LENGTH) as u16);
        self.total_tokens += u64::from(token_count);
        Ok(vid)
    }

    /// Heap bytes held by this builder (memory-budget accounting).
    pub fn estimated_bytes(&self) -> usize {
        self.doc_ids.capacity() * 4 + self.ordinals.capacity() * 2 + self.lengths.capacity() * 2
    }

    fn section_bytes(&self) -> u64 {
        self.doc_ids.len() as u64 * 8
    }

    /// Token count of virtual id `vid` (saturated at `MAX_CHUNK_LENGTH`).
    pub fn length(&self, vid: u32) -> u32 {
        self.lengths
            .get(vid as usize)
            .map_or(0, |len| u32::from(*len))
    }
}

/// Per-document token counts of one plain text field, ready to be written.
pub struct DocLengthsColumn<'a> {
    pub field_id: u32,
    /// One entry per document of the segment (0 = no value).
    pub lengths: &'a [u16],
    /// Sum of the unsaturated token counts.
    pub total_tokens: u64,
}

/// Write every chunked field's map and every plain field's length column as
/// one `.chunks` file.
///
/// `fields` must be sorted by field id and contain only non-empty builders;
/// `norms` likewise sorted, one column per field.
pub fn write_chunk_maps<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, &ChunkMapBuilder)],
    norms: &[DocLengthsColumn<'_>],
) -> io::Result<u64> {
    let sections = fields.len() + norms.len();
    let mut offset = (HEADER_SIZE + TOC_ENTRY_SIZE * sections) as u64;
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(VERSION)?;
    writer.write_u32::<LittleEndian>(sections as u32)?;
    for (field_id, map) in fields {
        writer.write_u32::<LittleEndian>(*field_id)?;
        writer.write_u32::<LittleEndian>(KIND_CHUNK_MAP)?;
        writer.write_u32::<LittleEndian>(map.len() as u32)?;
        writer.write_u64::<LittleEndian>(map.total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += map.section_bytes();
    }
    for column in norms {
        writer.write_u32::<LittleEndian>(column.field_id)?;
        writer.write_u32::<LittleEndian>(KIND_DOC_LENGTHS)?;
        writer.write_u32::<LittleEndian>(column.lengths.len() as u32)?;
        writer.write_u64::<LittleEndian>(column.total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += column.lengths.len() as u64 * 2;
    }
    for (_, map) in fields {
        for doc_id in &map.doc_ids {
            writer.write_u32::<LittleEndian>(*doc_id)?;
        }
        for ordinal in &map.ordinals {
            writer.write_u16::<LittleEndian>(*ordinal)?;
        }
        for length in &map.lengths {
            writer.write_u16::<LittleEndian>(*length)?;
        }
    }
    for column in norms {
        for length in column.lengths {
            writer.write_u16::<LittleEndian>(*length)?;
        }
    }
    Ok(offset)
}

/// Read-only per-document lengths of one plain text field, backed by the
/// mapped `.chunks` file.
#[derive(Debug, Clone)]
pub struct DocLengths {
    lengths: OwnedBytes,
    num_docs: u32,
    total_tokens: u64,
}

impl DocLengths {
    /// In-memory lengths column (tests).
    #[cfg(test)]
    pub(crate) fn from_lengths(lengths: &[u16]) -> Self {
        let mut bytes = Vec::with_capacity(lengths.len() * 2);
        for len in lengths {
            bytes.extend_from_slice(&len.to_le_bytes());
        }
        Self {
            lengths: OwnedBytes::new(bytes),
            num_docs: lengths.len() as u32,
            total_tokens: lengths.iter().map(|&l| u64::from(l)).sum(),
        }
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Average length over documents that have the field (1.0 when none).
    pub fn avg_len(&self) -> f32 {
        let with_value = self
            .lengths
            .as_slice()
            .chunks_exact(2)
            .filter(|b| b[0] != 0 || b[1] != 0)
            .count();
        if with_value == 0 {
            1.0
        } else {
            (self.total_tokens as f64 / with_value as f64) as f32
        }
    }

    /// Token count of the field in `doc_id` (0 when absent or out of range,
    /// saturated at `MAX_CHUNK_LENGTH`).
    #[inline]
    pub fn length(&self, doc_id: DocId) -> u32 {
        let at = doc_id as usize * 2;
        self.lengths
            .as_slice()
            .get(at..at + 2)
            .map_or(0, |b| u32::from(u16::from_le_bytes([b[0], b[1]])))
    }

    pub(crate) fn length_bytes(&self) -> &[u8] {
        self.lengths.as_slice()
    }
}

/// Everything a `.chunks` file holds.
#[derive(Debug, Default)]
pub struct ChunkMapFile {
    pub chunk_maps: FxHashMap<u32, ChunkMap>,
    pub doc_lengths: FxHashMap<u32, DocLengths>,
}

/// Read-only chunk map of one field, backed by the mapped `.chunks` file.
#[derive(Debug, Clone)]
pub struct ChunkMap {
    doc_ids: OwnedBytes,
    ordinals: OwnedBytes,
    lengths: OwnedBytes,
    num_chunks: u32,
    total_tokens: u64,
    /// Nominal chunk length: the 90th-percentile chunk length of the field in
    /// this segment. BM25 floors every chunk length at this value so a short
    /// tail chunk is not rewarded for being short (`docs/chunked-bm25.md`).
    length_floor: u32,
}

/// 90th-percentile of a little-endian `u16` length column (0 when empty).
fn nominal_chunk_length(lengths: &[u8]) -> u32 {
    let n = lengths.len() / 2;
    if n == 0 {
        return 0;
    }
    let mut histogram = vec![0u32; u16::MAX as usize + 1];
    for pair in lengths.chunks_exact(2) {
        histogram[u16::from_le_bytes([pair[0], pair[1]]) as usize] += 1;
    }
    // Smallest length such that at least 90 % of the chunks are ≤ it.
    let target = (n as u64 * 9).div_ceil(10);
    let mut seen = 0u64;
    for (len, &count) in histogram.iter().enumerate() {
        seen += u64::from(count);
        if seen >= target {
            return len as u32;
        }
    }
    u16::MAX as u32
}

impl ChunkMap {
    /// Number of chunks (virtual ids) in this segment.
    #[inline]
    pub fn num_chunks(&self) -> u32 {
        self.num_chunks
    }

    /// Sum of all chunk token counts.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Average chunk length in tokens (1.0 when empty).
    pub fn avg_len(&self) -> f32 {
        if self.num_chunks == 0 {
            1.0
        } else {
            (self.total_tokens as f64 / f64::from(self.num_chunks)) as f32
        }
    }

    /// Nominal chunk length (90th percentile), the BM25 length floor.
    #[inline]
    pub fn length_floor(&self) -> u32 {
        self.length_floor
    }

    /// BM25 length of virtual id `vid`: its token count floored at the
    /// nominal chunk length.
    #[inline]
    pub fn bm25_length(&self, vid: u32) -> u32 {
        self.length(vid).max(self.length_floor)
    }

    /// Document owning virtual id `vid`.
    #[inline]
    pub fn doc_id(&self, vid: u32) -> DocId {
        let at = vid as usize * 4;
        let b = &self.doc_ids.as_slice()[at..at + 4];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Ordinal (value index within the document) of virtual id `vid`.
    #[inline]
    pub fn ordinal(&self, vid: u32) -> u16 {
        let at = vid as usize * 2;
        let b = &self.ordinals.as_slice()[at..at + 2];
        u16::from_le_bytes([b[0], b[1]])
    }

    /// Token count of virtual id `vid` (saturated at `MAX_CHUNK_LENGTH`).
    #[inline]
    pub fn length(&self, vid: u32) -> u32 {
        let at = vid as usize * 2;
        let b = &self.lengths.as_slice()[at..at + 2];
        u32::from(u16::from_le_bytes([b[0], b[1]]))
    }

    /// `(doc_id, ordinal)` of virtual id `vid`.
    #[inline]
    pub fn resolve(&self, vid: u32) -> (DocId, u16) {
        (self.doc_id(vid), self.ordinal(vid))
    }

    /// Raw little-endian document-id column (merge copy).
    pub(crate) fn doc_id_bytes(&self) -> &[u8] {
        self.doc_ids.as_slice()
    }

    /// Raw little-endian ordinal column (merge copy).
    pub(crate) fn ordinal_bytes(&self) -> &[u8] {
        self.ordinals.as_slice()
    }

    /// Raw little-endian length column (merge copy).
    pub(crate) fn length_bytes(&self) -> &[u8] {
        self.lengths.as_slice()
    }
}

/// Parse a `.chunks` file into per-field chunk maps and length columns.
pub fn read_chunk_maps(bytes: OwnedBytes) -> io::Result<ChunkMapFile> {
    let data = bytes.as_slice();
    if data.len() < HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunk map file shorter than its header",
        ));
    }
    let mut cursor = io::Cursor::new(data);
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("chunk map magic mismatch: {magic:#x}"),
        ));
    }
    let version = cursor.read_u32::<LittleEndian>()?;
    let entry_size = match version {
        1 => TOC_ENTRY_SIZE_V1,
        VERSION => TOC_ENTRY_SIZE,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported chunk map version {other} (expected {VERSION})"),
            ));
        }
    };
    let num_sections = cursor.read_u32::<LittleEndian>()? as usize;
    if data.len() < HEADER_SIZE + entry_size * num_sections {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunk map table of contents truncated",
        ));
    }
    let overflow = || io::Error::new(io::ErrorKind::InvalidData, "chunk map size overflow");
    let mut file = ChunkMapFile::default();
    for _ in 0..num_sections {
        let field_id = cursor.read_u32::<LittleEndian>()?;
        let kind = if version == 1 {
            KIND_CHUNK_MAP
        } else {
            cursor.read_u32::<LittleEndian>()?
        };
        let count = cursor.read_u32::<LittleEndian>()?;
        let total_tokens = cursor.read_u64::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()? as usize;
        let n = count as usize;
        let bytes_per_entry = match kind {
            KIND_CHUNK_MAP => 8,
            KIND_DOC_LENGTHS => 2,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown chunk map section kind {other} for field {field_id}"),
                ));
            }
        };
        let end = offset
            .checked_add(n.checked_mul(bytes_per_entry).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        if end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("chunk map section of field {field_id} exceeds file length"),
            ));
        }
        match kind {
            KIND_CHUNK_MAP => {
                let doc_ids = bytes.slice(offset..offset + n * 4);
                let ordinals = bytes.slice(offset + n * 4..offset + n * 6);
                let lengths = bytes.slice(offset + n * 6..end);
                let length_floor = nominal_chunk_length(lengths.as_slice());
                file.chunk_maps.insert(
                    field_id,
                    ChunkMap {
                        doc_ids,
                        ordinals,
                        lengths,
                        num_chunks: count,
                        total_tokens,
                        length_floor,
                    },
                );
            }
            _ => {
                file.doc_lengths.insert(
                    field_id,
                    DocLengths {
                        lengths: bytes.slice(offset..end),
                        num_docs: count,
                        total_tokens,
                    },
                );
            }
        }
    }
    Ok(file)
}

/// One source section of a merged chunk map.
pub struct ChunkMapSource<'a> {
    pub map: &'a ChunkMap,
    /// Added to every document id of the source.
    pub doc_offset: u32,
}

/// One source of a merged length column: the source segment's column when it
/// has one, and its document count (zeros are written for a missing column).
pub struct DocLengthsSource<'a> {
    pub lengths: Option<&'a DocLengths>,
    pub num_docs: u32,
}

/// Write the merged `.chunks` file: per field, the sources' sections are
/// concatenated in order (virtual ids of a later source are offset by the
/// chunk counts of the earlier ones, matching the posting merge; length
/// columns follow the document order of the merge).
///
/// `fields` and `norms` must be sorted by field id; a field with zero total
/// chunks is skipped.
pub fn write_merged_chunk_maps<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, Vec<ChunkMapSource<'_>>)],
    norms: &[(u32, Vec<DocLengthsSource<'_>>)],
) -> io::Result<u64> {
    let live: Vec<&(u32, Vec<ChunkMapSource<'_>>)> = fields
        .iter()
        .filter(|(_, sources)| sources.iter().any(|s| s.map.num_chunks() > 0))
        .collect();
    let sections = live.len() + norms.len();
    let mut offset = (HEADER_SIZE + TOC_ENTRY_SIZE * sections) as u64;
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(VERSION)?;
    writer.write_u32::<LittleEndian>(sections as u32)?;
    for (field_id, sources) in &live {
        let mut num_chunks = 0u64;
        let mut total_tokens = 0u64;
        for source in sources {
            num_chunks += u64::from(source.map.num_chunks());
            total_tokens += source.map.total_tokens();
        }
        let num_chunks = u32::try_from(num_chunks).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("chunked field {field_id} exceeds u32::MAX chunks after merge"),
            )
        })?;
        writer.write_u32::<LittleEndian>(*field_id)?;
        writer.write_u32::<LittleEndian>(KIND_CHUNK_MAP)?;
        writer.write_u32::<LittleEndian>(num_chunks)?;
        writer.write_u64::<LittleEndian>(total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += u64::from(num_chunks) * 8;
    }
    for (field_id, sources) in norms {
        let num_docs: u64 = sources.iter().map(|s| u64::from(s.num_docs)).sum();
        let num_docs = u32::try_from(num_docs).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("field {field_id} exceeds u32::MAX documents after merge"),
            )
        })?;
        let total_tokens: u64 = sources
            .iter()
            .filter_map(|s| s.lengths.map(DocLengths::total_tokens))
            .sum();
        writer.write_u32::<LittleEndian>(*field_id)?;
        writer.write_u32::<LittleEndian>(KIND_DOC_LENGTHS)?;
        writer.write_u32::<LittleEndian>(num_docs)?;
        writer.write_u64::<LittleEndian>(total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += u64::from(num_docs) * 2;
    }
    let mut patched: Vec<u8> = Vec::new();
    for (_, sources) in &live {
        for source in sources {
            if source.doc_offset == 0 {
                writer.write_all(source.map.doc_id_bytes())?;
                continue;
            }
            patched.clear();
            patched.reserve(source.map.doc_id_bytes().len());
            for chunk in source.map.doc_id_bytes().chunks_exact(4) {
                let doc = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let remapped = doc.checked_add(source.doc_offset).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "document id overflow while merging chunk maps",
                    )
                })?;
                patched.extend_from_slice(&remapped.to_le_bytes());
            }
            writer.write_all(&patched)?;
        }
        for source in sources {
            writer.write_all(source.map.ordinal_bytes())?;
        }
        for source in sources {
            writer.write_all(source.map.length_bytes())?;
        }
    }
    let zeros = [0u8; 2 * 1024];
    for (_, sources) in norms {
        for source in sources {
            match source.lengths {
                Some(lengths) if lengths.num_docs() == source.num_docs => {
                    writer.write_all(lengths.length_bytes())?;
                }
                Some(lengths) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "length column covers {} documents, segment has {}",
                            lengths.num_docs(),
                            source.num_docs
                        ),
                    ));
                }
                None => {
                    let mut remaining = source.num_docs as usize * 2;
                    while remaining > 0 {
                        let take = remaining.min(zeros.len());
                        writer.write_all(&zeros[..take])?;
                        remaining -= take;
                    }
                }
            }
        }
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(entries: &[(u32, u16, u32)]) -> ChunkMapBuilder {
        let mut builder = ChunkMapBuilder::default();
        for &(doc, ord, len) in entries {
            builder.push(doc, ord, len).unwrap();
        }
        builder
    }

    #[test]
    fn round_trips_two_fields() {
        let a = build(&[(0, 0, 10), (0, 1, 20), (3, 0, 70_000)]);
        let b = build(&[(1, 0, 5)]);
        let mut out = Vec::new();
        write_chunk_maps(&mut out, &[(2, &a), (7, &b)], &[]).unwrap();
        let maps = read_chunk_maps(OwnedBytes::new(out)).unwrap().chunk_maps;
        let a = &maps[&2];
        assert_eq!(a.num_chunks(), 3);
        assert_eq!(a.resolve(0), (0, 0));
        assert_eq!(a.resolve(1), (0, 1));
        assert_eq!(a.resolve(2), (3, 0));
        assert_eq!(a.length(1), 20);
        assert_eq!(a.length(2), MAX_CHUNK_LENGTH, "lengths saturate at u16");
        assert_eq!(a.total_tokens(), 70_030);
        assert_eq!(maps[&7].resolve(0), (1, 0));
        assert_eq!(maps[&7].avg_len(), 5.0);
    }

    #[test]
    fn merged_maps_offset_doc_ids_and_keep_ordinals() {
        let first = build(&[(0, 0, 10), (1, 0, 11), (1, 1, 12)]);
        let second = build(&[(0, 0, 20), (0, 1, 21)]);
        let mut raw_first = Vec::new();
        write_chunk_maps(&mut raw_first, &[(4, &first)], &[]).unwrap();
        let mut raw_second = Vec::new();
        write_chunk_maps(&mut raw_second, &[(4, &second)], &[]).unwrap();
        let first = read_chunk_maps(OwnedBytes::new(raw_first))
            .unwrap()
            .chunk_maps;
        let second = read_chunk_maps(OwnedBytes::new(raw_second))
            .unwrap()
            .chunk_maps;

        let mut merged = Vec::new();
        write_merged_chunk_maps(
            &mut merged,
            &[(
                4,
                vec![
                    ChunkMapSource {
                        map: &first[&4],
                        doc_offset: 0,
                    },
                    ChunkMapSource {
                        map: &second[&4],
                        doc_offset: 2,
                    },
                ],
            )],
            &[],
        )
        .unwrap();
        let merged = read_chunk_maps(OwnedBytes::new(merged)).unwrap().chunk_maps;
        let map = &merged[&4];
        assert_eq!(map.num_chunks(), 5);
        assert_eq!(map.total_tokens(), 74);
        assert_eq!(
            (0..5).map(|v| map.resolve(v)).collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
        );
        assert_eq!(
            (0..5).map(|v| map.length(v)).collect::<Vec<_>>(),
            vec![10, 11, 12, 20, 21]
        );
    }

    #[test]
    fn rejects_foreign_or_truncated_files() {
        assert!(read_chunk_maps(OwnedBytes::new(vec![0u8; 4])).is_err());
        let mut bad_magic = Vec::new();
        bad_magic.write_u32::<LittleEndian>(0xDEAD_BEEF).unwrap();
        bad_magic.write_u32::<LittleEndian>(VERSION).unwrap();
        bad_magic.write_u32::<LittleEndian>(0).unwrap();
        assert!(read_chunk_maps(OwnedBytes::new(bad_magic)).is_err());

        let a = build(&[(0, 0, 10)]);
        let mut out = Vec::new();
        write_chunk_maps(&mut out, &[(1, &a)], &[]).unwrap();
        out.truncate(out.len() - 1);
        assert!(read_chunk_maps(OwnedBytes::new(out)).is_err());
    }

    #[test]
    fn doc_length_columns_round_trip_and_merge_with_zero_fill() {
        let a = build(&[(0, 0, 10)]);
        let column = [7u16, 0, 300];
        let mut out = Vec::new();
        write_chunk_maps(
            &mut out,
            &[(1, &a)],
            &[DocLengthsColumn {
                field_id: 5,
                lengths: &column,
                total_tokens: 307,
            }],
        )
        .unwrap();
        let file = read_chunk_maps(OwnedBytes::new(out)).unwrap();
        assert_eq!(file.chunk_maps[&1].num_chunks(), 1);
        let norms = &file.doc_lengths[&5];
        assert_eq!(norms.num_docs(), 3);
        assert_eq!(
            (0..4).map(|d| norms.length(d)).collect::<Vec<_>>(),
            vec![7, 0, 300, 0]
        );
        assert_eq!(norms.total_tokens(), 307);
        assert!(
            (norms.avg_len() - 153.5).abs() < 1e-3,
            "{}",
            norms.avg_len()
        );

        // Merge: a source without the column contributes zeros for its docs.
        let mut merged = Vec::new();
        write_merged_chunk_maps(
            &mut merged,
            &[],
            &[(
                5,
                vec![
                    DocLengthsSource {
                        lengths: None,
                        num_docs: 2,
                    },
                    DocLengthsSource {
                        lengths: Some(norms),
                        num_docs: 3,
                    },
                ],
            )],
        )
        .unwrap();
        let merged = read_chunk_maps(OwnedBytes::new(merged)).unwrap();
        assert!(merged.chunk_maps.is_empty());
        let norms = &merged.doc_lengths[&5];
        assert_eq!(norms.num_docs(), 5);
        assert_eq!(
            (0..5).map(|d| norms.length(d)).collect::<Vec<_>>(),
            vec![0, 0, 7, 0, 300]
        );
        assert_eq!(norms.total_tokens(), 307);
    }

    #[test]
    fn version_one_files_still_read() {
        let a = build(&[(0, 0, 10), (2, 0, 4)]);
        let mut out = Vec::new();
        out.write_u32::<LittleEndian>(MAGIC).unwrap();
        out.write_u32::<LittleEndian>(1).unwrap();
        out.write_u32::<LittleEndian>(1).unwrap();
        out.write_u32::<LittleEndian>(9).unwrap();
        out.write_u32::<LittleEndian>(2).unwrap();
        out.write_u64::<LittleEndian>(14).unwrap();
        out.write_u64::<LittleEndian>((HEADER_SIZE + TOC_ENTRY_SIZE_V1) as u64)
            .unwrap();
        for doc in &a.doc_ids {
            out.write_u32::<LittleEndian>(*doc).unwrap();
        }
        for ord in &a.ordinals {
            out.write_u16::<LittleEndian>(*ord).unwrap();
        }
        for len in &a.lengths {
            out.write_u16::<LittleEndian>(*len).unwrap();
        }
        let file = read_chunk_maps(OwnedBytes::new(out)).unwrap();
        assert!(file.doc_lengths.is_empty());
        let map = &file.chunk_maps[&9];
        assert_eq!(map.resolve(1), (2, 0));
        assert_eq!(map.length(1), 4);
    }
}
