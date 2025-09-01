//! Virtual-id maps of chunked text fields (`seg_<id>.chunks`).
//!
//! A text field declared `chunked` indexes every value as its own scoring
//! unit: term postings and positions are keyed by a dense, segment-local
//! **virtual id** instead of the document id. This file maps each virtual id
//! back to `(doc_id, ordinal)` and records the chunk's token count for BM25
//! length normalisation. See `docs/chunked-text-fields.md`.
//!
//! ```text
//! [magic "CHNK"][version u32 = 1][num_fields u32]
//! TOC × num_fields: [field_id u32][num_chunks u32][total_tokens u64][data_offset u64]
//! per field:        doc_ids u32 × n | ordinals u16 × n | lengths u16 × n
//! ```
//!
//! Virtual ids are assigned in indexing order, and documents are indexed in
//! doc-id order, so `doc_ids` is non-decreasing and `(doc_id, ordinal)` is
//! strictly increasing. Merges concatenate sections and add the document
//! offset to `doc_ids`; ordinals and lengths are copied verbatim.

use std::io::{self, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rustc_hash::FxHashMap;

use crate::DocId;
use crate::directories::OwnedBytes;

const MAGIC: u32 = 0x4B4E_4843; // "CHNK"
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 12;
const TOC_ENTRY_SIZE: usize = 24;

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
}

/// Write every chunked field's map as one `.chunks` file.
///
/// `fields` must be sorted by field id and contain only non-empty builders.
pub fn write_chunk_maps<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, &ChunkMapBuilder)],
) -> io::Result<u64> {
    let mut offset = (HEADER_SIZE + TOC_ENTRY_SIZE * fields.len()) as u64;
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(VERSION)?;
    writer.write_u32::<LittleEndian>(fields.len() as u32)?;
    for (field_id, map) in fields {
        writer.write_u32::<LittleEndian>(*field_id)?;
        writer.write_u32::<LittleEndian>(map.len() as u32)?;
        writer.write_u64::<LittleEndian>(map.total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += map.section_bytes();
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
    Ok(offset)
}

/// Read-only chunk map of one field, backed by the mapped `.chunks` file.
#[derive(Debug, Clone)]
pub struct ChunkMap {
    doc_ids: OwnedBytes,
    ordinals: OwnedBytes,
    lengths: OwnedBytes,
    num_chunks: u32,
    total_tokens: u64,
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

/// Parse a `.chunks` file into per-field maps.
pub fn read_chunk_maps(bytes: OwnedBytes) -> io::Result<FxHashMap<u32, ChunkMap>> {
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
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported chunk map version {version} (expected {VERSION})"),
        ));
    }
    let num_fields = cursor.read_u32::<LittleEndian>()? as usize;
    if data.len() < HEADER_SIZE + TOC_ENTRY_SIZE * num_fields {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunk map table of contents truncated",
        ));
    }
    let mut maps = FxHashMap::default();
    for _ in 0..num_fields {
        let field_id = cursor.read_u32::<LittleEndian>()?;
        let num_chunks = cursor.read_u32::<LittleEndian>()?;
        let total_tokens = cursor.read_u64::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()? as usize;
        let n = num_chunks as usize;
        let end = offset
            .checked_add(n.checked_mul(8).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "chunk map size overflow")
            })?)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "chunk map size overflow"))?;
        if end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("chunk map section of field {field_id} exceeds file length"),
            ));
        }
        let doc_ids = bytes.slice(offset..offset + n * 4);
        let ordinals = bytes.slice(offset + n * 4..offset + n * 6);
        let lengths = bytes.slice(offset + n * 6..end);
        maps.insert(
            field_id,
            ChunkMap {
                doc_ids,
                ordinals,
                lengths,
                num_chunks,
                total_tokens,
            },
        );
    }
    Ok(maps)
}

/// One source section of a merged chunk map.
pub struct ChunkMapSource<'a> {
    pub map: &'a ChunkMap,
    /// Added to every document id of the source.
    pub doc_offset: u32,
}

/// Write the merged `.chunks` file: per field, the sources' sections are
/// concatenated in order (virtual ids of a later source are offset by the
/// chunk counts of the earlier ones, matching the posting merge).
///
/// `fields` must be sorted by field id; a field with zero total chunks is
/// skipped.
pub fn write_merged_chunk_maps<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, Vec<ChunkMapSource<'_>>)],
) -> io::Result<u64> {
    let live: Vec<&(u32, Vec<ChunkMapSource<'_>>)> = fields
        .iter()
        .filter(|(_, sources)| sources.iter().any(|s| s.map.num_chunks() > 0))
        .collect();
    let mut offset = (HEADER_SIZE + TOC_ENTRY_SIZE * live.len()) as u64;
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(VERSION)?;
    writer.write_u32::<LittleEndian>(live.len() as u32)?;
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
        writer.write_u32::<LittleEndian>(num_chunks)?;
        writer.write_u64::<LittleEndian>(total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += u64::from(num_chunks) * 8;
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
        write_chunk_maps(&mut out, &[(2, &a), (7, &b)]).unwrap();
        let maps = read_chunk_maps(OwnedBytes::new(out)).unwrap();
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
        write_chunk_maps(&mut raw_first, &[(4, &first)]).unwrap();
        let mut raw_second = Vec::new();
        write_chunk_maps(&mut raw_second, &[(4, &second)]).unwrap();
        let first = read_chunk_maps(OwnedBytes::new(raw_first)).unwrap();
        let second = read_chunk_maps(OwnedBytes::new(raw_second)).unwrap();

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
        )
        .unwrap();
        let merged = read_chunk_maps(OwnedBytes::new(merged)).unwrap();
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
        write_chunk_maps(&mut out, &[(1, &a)]).unwrap();
        out.truncate(out.len() - 1);
        assert!(read_chunk_maps(OwnedBytes::new(out)).is_err());
    }
}
