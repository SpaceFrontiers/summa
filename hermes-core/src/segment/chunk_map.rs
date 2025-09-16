//! Virtual-id maps of chunked text fields (`seg_<id>.chunks`).
//!
//! A text field declared `chunked` indexes every value as its own scoring
//! unit: term postings and positions are keyed by a dense, segment-local
//! **virtual id** instead of the document id. This file maps each virtual id
//! back to `(doc_id, ordinal)` and records the chunk's token count for BM25
//! length normalisation. See `docs/chunked-text-fields.md`.
//!
//! ```text
//! [magic "CHNK"][version u32, 1..=5][num_sections u32]
//! TOC × num_sections: [field_id u32][kind u32][count u32][total_tokens u64][data_offset u64]
//! kind 0 (chunk map):   doc_ids u32 × n | ordinals u16 × n | lengths u16 × n
//! kind 2 (addressed map): kind 0 columns | logical-order physical slots u32 × n
//! kind 1 (doc lengths): lengths u16 × num_docs        (norms of a plain text field)
//! kind 3 (document map, V4): addressed map with document scoring semantics
//! kind 4 (byte norms, V5): byte4 norm codes × num_docs, exact total_tokens in TOC
//! ```
//!
//! Version 1 files have 24-byte entries without `kind` and hold chunk maps
//! only; they are still read.
//!
//! Virtual ids are assigned in indexing order, and documents are indexed in
//! doc-id order, so `doc_ids` starts out non-decreasing. A reorder pass on a
//! field with the `reorder` attribute permutes the virtual ids (BP over the
//! field's postings, `segment/text_reorder.rs`). Readers verify doc-id order
//! at open before enabling ordered query paths. Merges concatenate sections and add the document offset to
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
const VERSION: u32 = 5;
const DOCUMENT_VERSION: u32 = 4;
const ADDRESSED_VERSION: u32 = 3;
const HEADER_SIZE: usize = 12;
const TOC_ENTRY_SIZE_V1: usize = 24;
const TOC_ENTRY_SIZE: usize = 28;
const KIND_CHUNK_MAP: u32 = 0;
const KIND_DOC_LENGTHS: u32 = 1;
const KIND_ADDRESSED_CHUNK_MAP: u32 = 2;
const KIND_DOCUMENT_MAP: u32 = 3;
const KIND_BYTE_NORMS: u32 = 4;

/// Token count stored per chunk; longer chunks saturate.
pub const MAX_CHUNK_LENGTH: u32 = u16::MAX as u32;

/// In-memory map of one chunked field while a segment is being built.
#[derive(Debug, Default, Clone)]
pub struct ChunkMapBuilder {
    doc_ids: Vec<DocId>,
    ordinals: Vec<u16>,
    lengths: Vec<u16>,
    total_tokens: u64,
    document_units: bool,
}

impl ChunkMapBuilder {
    #[cfg(feature = "native")]
    pub(crate) fn with_capacity(count: usize) -> Self {
        Self {
            doc_ids: Vec::with_capacity(count),
            ordinals: Vec::with_capacity(count),
            lengths: Vec::with_capacity(count),
            total_tokens: 0,
            document_units: false,
        }
    }

    /// Preserve the scoring-unit policy when rebuilding mapped text.
    #[cfg(any(feature = "native", feature = "wasm", test))]
    pub(crate) fn set_document_units(&mut self, document_units: bool) {
        self.document_units = document_units;
    }

    fn section_kind(&self) -> u32 {
        if self.document_units {
            KIND_DOCUMENT_MAP
        } else if self.logically_ordered() {
            KIND_CHUNK_MAP
        } else {
            KIND_ADDRESSED_CHUNK_MAP
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn set_total_tokens(&mut self, total: u64) {
        self.total_tokens = total;
    }

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

    fn logically_ordered(&self) -> bool {
        self.doc_ids.iter().zip(&self.ordinals).is_sorted()
            && self
                .doc_ids
                .iter()
                .zip(&self.ordinals)
                .zip(self.doc_ids.iter().zip(&self.ordinals).skip(1))
                .all(|(a, b)| a != b)
    }

    fn section_bytes(&self) -> u64 {
        self.doc_ids.len() as u64
            * if self.section_kind() == KIND_CHUNK_MAP {
                8
            } else {
                12
            }
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
    write_chunk_maps_with_norms(writer, fields, norms, false)
}

/// Norm inputs share one CHNK section writer. Existing columns keep their
/// encoding and payload bytes; only newly built columns choose quantization.
enum NormColumn<'a> {
    New(&'a DocLengthsColumn<'a>, bool),
    #[cfg(feature = "native")]
    Encoded(u32, &'a DocLengths),
}

impl NormColumn<'_> {
    fn field_id(&self) -> u32 {
        match self {
            Self::New(column, _) => column.field_id,
            #[cfg(feature = "native")]
            Self::Encoded(field, _) => *field,
        }
    }
    fn quantized(&self) -> bool {
        match self {
            Self::New(_, quantized) => *quantized,
            #[cfg(feature = "native")]
            Self::Encoded(_, lengths) => lengths.is_quantized(),
        }
    }
    fn count(&self) -> usize {
        match self {
            Self::New(column, _) => column.lengths.len(),
            #[cfg(feature = "native")]
            Self::Encoded(_, lengths) => lengths.num_docs() as usize,
        }
    }
    fn total_tokens(&self) -> u64 {
        match self {
            Self::New(column, _) => column.total_tokens,
            #[cfg(feature = "native")]
            Self::Encoded(_, lengths) => lengths.total_tokens(),
        }
    }
    fn write(&self, writer: &mut (impl Write + ?Sized)) -> io::Result<()> {
        match self {
            #[cfg(feature = "native")]
            Self::Encoded(_, lengths) => writer.write_all(lengths.length_bytes()),
            Self::New(column, quantized) => {
                for &length in column.lengths {
                    if *quantized {
                        writer.write_all(&[super::norms::encode(u32::from(length))])?;
                    } else {
                        writer.write_u16::<LittleEndian>(length)?;
                    }
                }
                Ok(())
            }
        }
    }
}

#[cfg(feature = "native")]
pub(crate) fn write_chunk_maps_with_copied_norms<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, &ChunkMapBuilder)],
    norms: &[(u32, &DocLengths)],
) -> io::Result<u64> {
    let columns: Vec<_> = norms
        .iter()
        .map(|&(field, lengths)| NormColumn::Encoded(field, lengths))
        .collect();
    write_chunk_map_columns(
        writer,
        fields,
        &columns,
        norms.iter().any(|(_, lengths)| lengths.is_quantized()),
    )
}

pub(crate) fn write_chunk_maps_with_norms<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, &ChunkMapBuilder)],
    norms: &[DocLengthsColumn<'_>],
    quantized: bool,
) -> io::Result<u64> {
    let columns: Vec<_> = norms
        .iter()
        .map(|column| NormColumn::New(column, quantized))
        .collect();
    write_chunk_map_columns(writer, fields, &columns, quantized)
}

/// Preserve per-field normalization policy during row compaction of mixed
/// generations. A representative length is never requantized into another value.
#[cfg(feature = "native")]
pub(crate) fn write_chunk_maps_with_norm_policy<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, &ChunkMapBuilder)],
    norms: &[DocLengthsColumn<'_>],
    quantized: impl Fn(u32) -> bool,
) -> io::Result<u64> {
    let columns: Vec<_> = norms
        .iter()
        .map(|column| NormColumn::New(column, quantized(column.field_id)))
        .collect();
    write_chunk_map_columns(
        writer,
        fields,
        &columns,
        columns.iter().any(NormColumn::quantized),
    )
}

fn write_chunk_map_columns<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, &ChunkMapBuilder)],
    norms: &[NormColumn<'_>],
    quantized_version: bool,
) -> io::Result<u64> {
    let sections = fields.len() + norms.len();
    let mut offset = (HEADER_SIZE + TOC_ENTRY_SIZE * sections) as u64;
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(if quantized_version {
        VERSION
    } else if fields.iter().any(|(_, m)| m.document_units) {
        DOCUMENT_VERSION
    } else {
        ADDRESSED_VERSION
    })?;
    writer.write_u32::<LittleEndian>(sections as u32)?;
    for (field_id, map) in fields {
        writer.write_u32::<LittleEndian>(*field_id)?;
        writer.write_u32::<LittleEndian>(map.section_kind())?;
        writer.write_u32::<LittleEndian>(map.len() as u32)?;
        writer.write_u64::<LittleEndian>(map.total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += map.section_bytes();
    }
    for column in norms {
        writer.write_u32::<LittleEndian>(column.field_id())?;
        writer.write_u32::<LittleEndian>(if column.quantized() {
            KIND_BYTE_NORMS
        } else {
            KIND_DOC_LENGTHS
        })?;
        writer.write_u32::<LittleEndian>(column.count() as u32)?;
        writer.write_u64::<LittleEndian>(column.total_tokens())?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += column.count() as u64 * if column.quantized() { 1 } else { 2 };
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
        if map.section_kind() != KIND_CHUNK_MAP {
            let mut slots: Vec<u32> = (0..map.len() as u32).collect();
            slots.sort_unstable_by_key(|&slot| {
                (map.doc_ids[slot as usize], map.ordinals[slot as usize])
            });
            let mut previous = None;
            for slot in slots {
                let key = (map.doc_ids[slot as usize], map.ordinals[slot as usize]);
                if previous == Some(key) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate logical text chunk",
                    ));
                }
                previous = Some(key);
                writer.write_u32::<LittleEndian>(slot)?;
            }
        }
    }
    for column in norms {
        column.write(writer)?;
    }
    Ok(offset)
}

/// Set once a posting referenced a virtual chunk id past its field's chunk
/// map, so the corruption is logged a single time per process instead of
/// once per scored posting.
static INVALID_CHUNK_ID_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Scoring length substituted for a virtual id the chunk map does not hold.
/// One token is the loosest valid normalisation: search keeps running and the
/// (logged) corruption is visible instead of a bounds-check panic.
const INVALID_CHUNK_LENGTH: u32 = 1;

#[cold]
#[inline(never)]
fn invalid_chunk_id(vid: u32, num_chunks: usize) -> u32 {
    if !INVALID_CHUNK_ID_REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::error!(
            "chunk map lookup out of range: virtual chunk id {vid} >= {num_chunks} chunks; \
             the postings and .chunks file disagree (corrupt segment). Scoring that chunk \
             with length {INVALID_CHUNK_LENGTH}; further occurrences are not logged"
        );
    }
    INVALID_CHUNK_LENGTH
}

/// Whether an out-of-range virtual chunk id has been reported (tests).
#[cfg(test)]
mod quantized_norm_tests {
    use super::*;
    #[test]
    fn byte_norm_columns_and_mixed_merges_preserve_their_scoring_lengths() {
        let values = [0, 1, 41, 999, u16::MAX];
        let make = |quantized| {
            let mut bytes = Vec::new();
            write_chunk_maps_with_norms(
                &mut bytes,
                &[],
                &[DocLengthsColumn {
                    field_id: 0,
                    lengths: &values,
                    total_tokens: 90000,
                }],
                quantized,
            )
            .unwrap();
            read_chunk_maps(OwnedBytes::new(bytes))
                .unwrap()
                .doc_lengths
                .remove(&0)
                .unwrap()
        };
        let old = make(false);
        let quantized = make(true);
        assert_eq!(quantized.length_bytes().len(), values.len());
        assert_eq!(old.length_bytes().len(), values.len() * 2);
        assert_eq!(quantized.total_tokens(), 90000);
        assert_eq!(old.avg_len(), quantized.avg_len());
        for (doc, &value) in values.iter().enumerate() {
            assert_eq!(
                quantized.length(doc as u32),
                crate::segment::norms::quantize(u32::from(value))
            );
        }
        for second in [&old, &quantized] {
            let mut bytes = Vec::new();
            write_merged_chunk_maps(
                &mut bytes,
                &[],
                &[(
                    0,
                    vec![
                        DocLengthsSource {
                            lengths: Some(&quantized),
                            num_docs: 5,
                        },
                        DocLengthsSource {
                            lengths: None,
                            num_docs: 2,
                        },
                        DocLengthsSource {
                            lengths: Some(second),
                            num_docs: 5,
                        },
                    ],
                )],
            )
            .unwrap();
            let merged = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
            let merged = &merged.doc_lengths[&0];
            assert_eq!(merged.is_quantized(), second.is_quantized());
            for i in 0..5 {
                assert_eq!(merged.length(i), quantized.length(i));
                assert_eq!(merged.length(i + 7), second.length(i));
            }
            assert_eq!(merged.length(5), 0);
            assert_eq!(merged.length(6), 0);
            assert_eq!(merged.total_tokens(), 180000);
        }
    }
}

#[cfg(test)]
fn invalid_chunk_id_reported() -> bool {
    INVALID_CHUNK_ID_REPORTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Read a bounded batch from the existing little-endian length column.
/// Resolve the backing byte view once; two-byte arrays need no alignment.
/// The bounds check is the only work on the common path; a missing id takes
/// the cold reporting path (`STRICT`) or reads as absent (0).
fn gather_lengths<const STRICT: bool>(bytes: &[u8], ids: &[u32], out: &mut [u32], floor: u32) {
    let pairs = bytes.as_chunks::<2>().0;
    for (&id, slot) in ids.iter().zip(&mut out[..ids.len()]) {
        let value = match pairs.get(id as usize) {
            Some(pair) => u32::from(u16::from_le_bytes(*pair)),
            None if STRICT => invalid_chunk_id(id, pairs.len()),
            None => 0,
        };
        *slot = value.max(floor);
    }
}

/// Read-only per-document lengths of one plain text field, backed by the
/// mapped `.chunks` file.
#[derive(Debug, Clone)]
pub struct DocLengths {
    quantized: bool,
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
            quantized: false,
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
            .chunks_exact(if self.quantized { 1 } else { 2 })
            .filter(|b| b.iter().any(|&value| value != 0))
            .count();
        if with_value == 0 {
            1.0
        } else {
            (self.total_tokens as f64 / with_value as f64) as f32
        }
    }

    /// Scoring length of the field in `doc_id` (0 when absent or out of range).
    /// U16 columns are saturated; byte columns return their rounded-down
    /// representative. Token totals retain the original unsaturated counts.
    #[inline]
    pub fn length(&self, doc_id: DocId) -> u32 {
        if self.quantized {
            return super::norms::decode(self.norm_code(doc_id));
        }
        self.lengths
            .as_slice()
            .as_chunks::<2>()
            .0
            .get(doc_id as usize)
            .copied()
            .map_or(0, |b| u32::from(u16::from_le_bytes(b)))
    }

    pub(crate) fn gather_lengths(&self, ids: &[u32], out: &mut [u32]) {
        if self.quantized {
            for (&id, value) in ids.iter().zip(out) {
                *value = self.length(id);
            }
        } else {
            gather_lengths::<false>(self.length_bytes(), ids, out, 0);
        }
    }

    pub(crate) fn is_quantized(&self) -> bool {
        self.quantized
    }

    #[inline]
    pub(crate) fn norm_code(&self, doc: DocId) -> u8 {
        self.lengths
            .as_slice()
            .get(doc as usize)
            .copied()
            .unwrap_or(0)
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
    logically_ordered: bool,
    logical_slots: Option<OwnedBytes>,
    /// Derived once at open; never persisted or assumed from schema flags.
    doc_ids_monotonic: bool,
    document_units: bool,
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
    pub(crate) fn is_document_map(&self) -> bool {
        self.document_units
    }

    /// Represent an older unpermuted plain field without rebuilding postings.
    /// The caller admits six bytes per document before creating these columns;
    /// existing length bytes remain borrowed from the immutable segment.
    #[cfg(feature = "native")]
    pub(crate) fn identity_documents(
        num_docs: u32,
        lengths: Option<&DocLengths>,
    ) -> io::Result<Self> {
        if lengths.is_some_and(|lengths| lengths.num_docs() != num_docs) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "document length count mismatch",
            ));
        }
        let mut ids = Vec::with_capacity(num_docs as usize * 4);
        for doc in 0..num_docs {
            ids.extend_from_slice(&doc.to_le_bytes());
        }
        Ok(Self {
            doc_ids: OwnedBytes::new(ids),
            ordinals: OwnedBytes::new(vec![0; num_docs as usize * 2]),
            lengths: lengths.map_or_else(
                || OwnedBytes::new(vec![0; num_docs as usize * 2]),
                |lengths| {
                    if !lengths.quantized {
                        return lengths.lengths.clone();
                    }
                    let mut bytes = Vec::with_capacity(num_docs as usize * 2);
                    for doc in 0..num_docs {
                        bytes.extend_from_slice(&(lengths.length(doc) as u16).to_le_bytes());
                    }
                    OwnedBytes::new(bytes)
                },
            ),
            num_chunks: num_docs,
            total_tokens: lengths.map_or(0, DocLengths::total_tokens),
            length_floor: 0,
            logically_ordered: true,
            logical_slots: None,
            doc_ids_monotonic: true,
            document_units: true,
        })
    }

    pub(crate) fn has_logical_addressing(&self) -> bool {
        self.logically_ordered || self.logical_slots.is_some()
    }

    fn logical_slot(&self, index: u32) -> u32 {
        self.logical_slots.as_ref().map_or(index, |slots| {
            let offset = index as usize * 4;
            u32::from_le_bytes(slots.as_slice()[offset..offset + 4].try_into().unwrap())
        })
    }

    /// Document maps are validated dense permutations: logical index equals
    /// document ID, so their existing slot column is already the inverse map.
    pub(crate) fn document_slot(&self, doc: DocId) -> Option<u32> {
        (self.document_units && doc < self.num_chunks).then(|| self.logical_slot(doc))
    }

    pub(crate) fn slots_for_document(&self, doc: DocId) -> impl Iterator<Item = (u16, u32)> + '_ {
        super::logical_address::ordered_document_slots(self.num_chunks, doc, |index| {
            let (doc, ordinal) = self.resolve(self.logical_slot(index));
            Some(super::logical_address::LogicalUnit { doc, ordinal })
        })
        .map(|(ordinal, index)| (ordinal, self.logical_slot(index)))
    }

    pub(crate) fn slot_for_unit(&self, unit: super::logical_address::LogicalUnit) -> Option<u32> {
        super::logical_address::ordered_slot_for_unit(self.num_chunks, unit, |index| {
            let (doc, ordinal) = self.resolve(self.logical_slot(index));
            Some(super::logical_address::LogicalUnit { doc, ordinal })
        })
        .map(|index| self.logical_slot(index))
    }

    pub(crate) fn logically_ordered(&self) -> bool {
        self.logically_ordered
    }

    pub(crate) fn is_doc_ordered(&self) -> bool {
        self.doc_ids_monotonic
    }

    /// First virtual id owned by a document at or after `target`.
    /// Only valid for a verified doc-ordered map; num_chunks means exhausted.
    pub(crate) fn lower_bound_doc(&self, target: DocId) -> u32 {
        debug_assert!(self.is_doc_ordered());
        let (mut lo, mut hi) = (0, self.num_chunks);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.doc_id(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

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
    /// An id past the map is corrupt: it is logged once and scored as one
    /// token instead of panicking.
    #[inline]
    pub fn length(&self, vid: u32) -> u32 {
        let pairs = self.lengths.as_slice().as_chunks::<2>().0;
        match pairs.get(vid as usize) {
            Some(pair) => u32::from(u16::from_le_bytes(*pair)),
            None => invalid_chunk_id(vid, pairs.len()),
        }
    }

    pub(crate) fn gather_bm25_lengths(&self, ids: &[u32], out: &mut [u32]) {
        gather_lengths::<true>(self.length_bytes(), ids, out, self.length_floor);
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
        2 | ADDRESSED_VERSION | DOCUMENT_VERSION | VERSION => TOC_ENTRY_SIZE,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported chunk map version {other} (expected {VERSION})"),
            ));
        }
    };
    let num_sections = cursor.read_u32::<LittleEndian>()? as usize;
    let overflow = || io::Error::new(io::ErrorKind::InvalidData, "chunk map size overflow");
    let toc_end = num_sections
        .checked_mul(entry_size)
        .and_then(|n| HEADER_SIZE.checked_add(n))
        .ok_or_else(overflow)?;
    if data.len() < toc_end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunk map table of contents truncated",
        ));
    }
    let mut expected_offset = toc_end;
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
        let offset = usize::try_from(cursor.read_u64::<LittleEndian>()?).map_err(|_| overflow())?;
        if offset != expected_offset
            || file.chunk_maps.contains_key(&field_id)
            || file.doc_lengths.contains_key(&field_id)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk map has overlapping sections, gaps or duplicate fields",
            ));
        }
        let n = count as usize;
        let bytes_per_entry = match kind {
            KIND_CHUNK_MAP => 8,
            KIND_ADDRESSED_CHUNK_MAP if version >= 3 => 12,
            KIND_DOCUMENT_MAP if version >= 4 => 12,
            KIND_DOC_LENGTHS => 2,
            KIND_BYTE_NORMS if version >= 5 => 1,
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
        expected_offset = end;
        match kind {
            KIND_CHUNK_MAP | KIND_ADDRESSED_CHUNK_MAP | KIND_DOCUMENT_MAP => {
                let doc_ids = bytes.slice(offset..offset + n * 4);
                let ordinals = bytes.slice(offset + n * 4..offset + n * 6);
                let lengths = bytes.slice(offset + n * 6..offset + n * 8);
                let logical_slots =
                    (kind != KIND_CHUNK_MAP).then(|| bytes.slice(offset + n * 8..end));
                let document_units = kind == KIND_DOCUMENT_MAP;
                let length_floor = if document_units {
                    0
                } else {
                    nominal_chunk_length(lengths.as_slice())
                };
                let logically_ordered = super::logical_address::logically_ordered(
                    doc_ids
                        .as_slice()
                        .chunks_exact(4)
                        .zip(ordinals.as_slice().chunks_exact(2))
                        .map(|(doc, ordinal)| {
                            Some(super::logical_address::LogicalUnit {
                                doc: u32::from_le_bytes(doc.try_into().unwrap()),
                                ordinal: u16::from_le_bytes(ordinal.try_into().unwrap()),
                            })
                        }),
                );
                let doc_ids_monotonic = doc_ids
                    .as_slice()
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .is_sorted();
                let map = ChunkMap {
                    doc_ids,
                    ordinals,
                    lengths,
                    num_chunks: count,
                    total_tokens,
                    length_floor,
                    logically_ordered,
                    logical_slots,
                    doc_ids_monotonic,
                    document_units,
                };
                if version >= 3 && kind == KIND_CHUNK_MAP && !map.logically_ordered {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "V3 chunk map requires logical ordering or an addressed section",
                    ));
                }
                if let Some(slots) = &map.logical_slots {
                    let mut previous = None;
                    for raw in slots.as_slice().chunks_exact(4) {
                        let slot = u32::from_le_bytes(raw.try_into().unwrap());
                        if slot >= count {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "text logical slot out of range",
                            ));
                        }
                        let key = map.resolve(slot);
                        if document_units && (key.0 >= count || key.1 != 0) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "document map must cover each document once with ordinal zero",
                            ));
                        }
                        if previous.is_some_and(|p| p >= key) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "text logical slots are not a sorted unique permutation",
                            ));
                        }
                        previous = Some(key);
                    }
                }
                file.chunk_maps.insert(field_id, map);
            }
            _ => {
                if kind == KIND_BYTE_NORMS
                    && bytes[offset..end]
                        .iter()
                        .any(|&code| super::norms::decode(code) > MAX_CHUNK_LENGTH)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "byte norm exceeds saturated document length",
                    ));
                }
                file.doc_lengths.insert(
                    field_id,
                    DocLengths {
                        quantized: kind == KIND_BYTE_NORMS,
                        lengths: bytes.slice(offset..end),
                        num_docs: count,
                        total_tokens,
                    },
                );
            }
        }
    }
    if expected_offset != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing chunk map data",
        ));
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

fn all_quantized(sources: &[DocLengthsSource<'_>]) -> bool {
    sources
        .iter()
        .filter_map(|source| source.lengths)
        .all(DocLengths::is_quantized)
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
    write_merged_chunk_maps_ordered(writer, fields, norms, &FxHashMap::default(), || Ok(()))
}

/// A validated permutation of concatenated source physical units.
pub(crate) struct ChunkMapOrder<'a> {
    pub order: &'a [u32],
    pub inverse: &'a [u32],
}

pub(crate) fn write_merged_chunk_maps_ordered<W: Write + ?Sized>(
    writer: &mut W,
    fields: &[(u32, Vec<ChunkMapSource<'_>>)],
    norms: &[(u32, Vec<DocLengthsSource<'_>>)],
    orders: &FxHashMap<u32, ChunkMapOrder<'_>>,
    mut check_cancelled: impl FnMut() -> io::Result<()>,
) -> io::Result<u64> {
    let live: Vec<&(u32, Vec<ChunkMapSource<'_>>)> = fields
        .iter()
        .filter(|(_, sources)| sources.iter().any(|s| s.map.num_chunks() > 0))
        .collect();
    // Never silently discard addressing on a prepared source. Pure legacy
    // maps can still copy-merge in their original version until explicit reorder.
    let mut legacy = false;
    let mut addressed = false;
    let mut document_units = false;
    for (field, sources) in &live {
        check_cancelled()?;
        if let Some(plan) = orders.get(field) {
            let count: u64 = sources.iter().map(|s| u64::from(s.map.num_chunks())).sum();
            if count != plan.order.len() as u64 || plan.order.len() != plan.inverse.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "text map permutation has the wrong extent",
                ));
            }
            for (new, &old) in plan.order.iter().enumerate() {
                if new.is_multiple_of(4096) {
                    check_cancelled()?;
                }
                if plan.inverse.get(old as usize).copied() != Some(new as u32) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "text map order and inverse disagree",
                    ));
                }
            }
        }
        let documents = sources.iter().any(|s| s.map.document_units);
        if documents && sources.iter().any(|s| !s.map.document_units) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot merge document and chunk scoring units for one text field",
            ));
        }
        document_units |= documents;
        let needs_migration =
            !orders.contains_key(field) && sources.iter().any(|s| !s.map.has_logical_addressing());
        if needs_migration
            && sources
                .iter()
                .any(|s| s.map.num_chunks() > 0 && s.map.has_logical_addressing())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot merge prepared and unprepared text chunk maps; explicitly reorder legacy segments first",
            ));
        }
        legacy |= needs_migration;
        addressed |= documents
            || orders.contains_key(field)
            || (!needs_migration && sources.iter().any(|s| !s.map.logically_ordered()));
    }
    if legacy && addressed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy text chunks require explicit reorder before merging with V3 addressed fields",
        ));
    }
    let sections = live.len() + norms.len();
    let mut offset = (HEADER_SIZE + TOC_ENTRY_SIZE * sections) as u64;
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(
        if !legacy && norms.iter().any(|(_, sources)| all_quantized(sources)) {
            VERSION
        } else if legacy {
            2
        } else if document_units {
            DOCUMENT_VERSION
        } else {
            ADDRESSED_VERSION
        },
    )?;
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
        writer.write_u32::<LittleEndian>(if sources.iter().any(|s| s.map.document_units) {
            KIND_DOCUMENT_MAP
        } else if orders.contains_key(field_id)
            || (sources.iter().all(|s| s.map.has_logical_addressing())
                && sources.iter().any(|s| !s.map.logically_ordered()))
        {
            KIND_ADDRESSED_CHUNK_MAP
        } else {
            KIND_CHUNK_MAP
        })?;
        writer.write_u32::<LittleEndian>(num_chunks)?;
        writer.write_u64::<LittleEndian>(total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += u64::from(num_chunks)
            * if orders.contains_key(field_id)
                || sources.iter().any(|s| s.map.document_units)
                || (sources.iter().all(|s| s.map.has_logical_addressing())
                    && sources.iter().any(|s| !s.map.logically_ordered()))
            {
                12
            } else {
                8
            };
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
        writer.write_u32::<LittleEndian>(if !legacy && all_quantized(sources) {
            KIND_BYTE_NORMS
        } else {
            KIND_DOC_LENGTHS
        })?;
        writer.write_u32::<LittleEndian>(num_docs)?;
        writer.write_u64::<LittleEndian>(total_tokens)?;
        writer.write_u64::<LittleEndian>(offset)?;
        offset += u64::from(num_docs)
            * if !legacy && all_quantized(sources) {
                1
            } else {
                2
            };
    }
    let mut patched = Vec::with_capacity(64 * 1024);
    for (field, sources) in &live {
        check_cancelled()?;
        if let Some(plan) = orders.get(field) {
            let mut bases = Vec::with_capacity(sources.len());
            let mut base = 0u32;
            for source in sources {
                bases.push(base);
                base += source.map.num_chunks();
            }
            for column in 0..3 {
                for batch in plan.order.chunks(4096) {
                    check_cancelled()?;
                    patched.clear();
                    for &old in batch {
                        let index = bases.partition_point(|&base| base <= old) - 1;
                        let source = &sources[index];
                        let slot = old - bases[index];
                        match column {
                            0 => {
                                let doc = source
                                    .map
                                    .doc_id(slot)
                                    .checked_add(source.doc_offset)
                                    .ok_or_else(|| {
                                        io::Error::new(
                                            io::ErrorKind::InvalidData,
                                            "reordered document ID overflow",
                                        )
                                    })?;
                                patched.extend_from_slice(&doc.to_le_bytes());
                            }
                            1 => patched.extend_from_slice(&source.map.ordinal(slot).to_le_bytes()),
                            _ => patched
                                .extend_from_slice(&(source.map.length(slot) as u16).to_le_bytes()),
                        }
                    }
                    writer.write_all(&patched)?;
                }
            }
            for (source, &base) in sources.iter().zip(&bases) {
                for first in (0..source.map.num_chunks()).step_by(4096) {
                    check_cancelled()?;
                    patched.clear();
                    for logical in first..source.map.num_chunks().min(first.saturating_add(4096)) {
                        let old = source.map.logical_slot(logical) + base;
                        patched.extend_from_slice(&plan.inverse[old as usize].to_le_bytes());
                    }
                    writer.write_all(&patched)?;
                }
            }
            continue;
        }
        for source in sources {
            if source.doc_offset == 0 {
                writer.write_all(source.map.doc_id_bytes())?;
                continue;
            }
            for batch in source.map.doc_id_bytes().chunks(64 * 1024) {
                patched.clear();
                for chunk in batch.chunks_exact(4) {
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
        }
        for source in sources {
            writer.write_all(source.map.ordinal_bytes())?;
        }
        for source in sources {
            writer.write_all(source.map.length_bytes())?;
        }
        let addressed = sources.iter().any(|s| s.map.document_units)
            || (sources.iter().all(|s| s.map.has_logical_addressing())
                && sources.iter().any(|s| !s.map.logically_ordered()));
        if addressed {
            let mut base = 0u32;
            for source in sources {
                for i in 0..source.map.num_chunks() {
                    writer.write_u32::<LittleEndian>(source.map.logical_slot(i) + base)?;
                }
                base += source.map.num_chunks();
            }
        }
    }
    let zeros = [0u8; 2 * 1024];
    for (_, sources) in norms {
        let quantized = !legacy && all_quantized(sources);
        for source in sources {
            match source.lengths {
                Some(lengths) if lengths.num_docs() == source.num_docs => {
                    if lengths.quantized && !quantized {
                        for batch in lengths.length_bytes().chunks(32 * 1024) {
                            patched.clear();
                            for &code in batch {
                                patched.extend_from_slice(
                                    &(super::norms::decode(code) as u16).to_le_bytes(),
                                );
                            }
                            writer.write_all(&patched)?;
                        }
                    } else {
                        writer.write_all(lengths.length_bytes())?;
                    }
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
                    let mut remaining = source.num_docs as usize * if quantized { 1 } else { 2 };
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

    #[cfg(feature = "native")]
    #[test]
    fn ordered_merge_map_validates_permutations_and_cancels_during_column_output() {
        let map = ChunkMap::identity_documents(8192, None).unwrap();
        let order: Vec<u32> = (0..8192).rev().collect();
        let fields = [(
            1,
            vec![ChunkMapSource {
                map: &map,
                doc_offset: 0,
            }],
        )];
        let plans = FxHashMap::from_iter([(
            1,
            ChunkMapOrder {
                order: &order,
                inverse: &order,
            },
        )]);
        let mut bytes = Vec::new();
        let mut checks = 0;
        let error = write_merged_chunk_maps_ordered(&mut bytes, &fields, &[], &plans, || {
            checks += 1;
            if checks == 6 {
                Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert!(
            bytes.len() > HEADER_SIZE + TOC_ENTRY_SIZE,
            "cancellation must interrupt an in-progress column"
        );
        let invalid = vec![0; order.len()];
        let plans = FxHashMap::from_iter([(
            1,
            ChunkMapOrder {
                order: &order,
                inverse: &invalid,
            },
        )]);
        bytes.clear();
        assert!(
            write_merged_chunk_maps_ordered(&mut bytes, &fields, &[], &plans, || Ok(())).is_err()
        );
        assert!(
            bytes.is_empty(),
            "reject inconsistent permutations before writing a header"
        );
    }

    #[test]
    fn document_maps_preserve_plain_lengths_and_require_the_new_version() {
        let mut builder = ChunkMapBuilder::default();
        for (doc, length) in [(2, 10), (0, 20), (1, 100)] {
            builder.push(doc, 0, length).unwrap();
        }
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
        // V4 document maps reuse addressed columns but do not apply the
        // chunked scoring policy. Older readers must reject this section.
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 3);
        builder.set_document_units(true);
        bytes.clear();
        write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4);
        let file = read_chunk_maps(OwnedBytes::new(bytes.clone())).unwrap();
        let map = &file.chunk_maps[&0];
        assert!(map.is_document_map());
        assert_eq!(map.resolve(0), (2, 0));
        for doc in 0..3 {
            let slot = map.document_slot(doc).unwrap();
            assert_eq!(map.resolve(slot), (doc, 0));
        }
        assert_eq!(map.document_slot(3), None);
        assert_eq!(map.document_slot(u32::MAX), None);
        assert_eq!(map.bm25_length(0), 10);
        assert_eq!(map.bm25_length(1), 20);
        assert_eq!(map.slots_for_document(0).collect::<Vec<_>>(), vec![(0, 1)]);
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert!(read_chunk_maps(OwnedBytes::new(bytes)).is_err());
    }

    fn build(entries: &[(u32, u16, u32)]) -> ChunkMapBuilder {
        let mut builder = ChunkMapBuilder::default();
        for &(doc, ord, len) in entries {
            builder.push(doc, ord, len).unwrap();
        }
        builder
    }

    #[test]
    fn batched_length_reads_preserve_missing_values_chunk_floors_and_column_bytes() {
        let docs = DocLengths::from_lengths(&[0, 1, u16::MAX, 120]);
        let original = docs.length_bytes().to_vec();
        let ids = [u32::MAX, 1, 2, 3, 0, 2];
        let mut out = [999; 8];
        docs.gather_lengths(&ids, &mut out);
        assert_eq!(
            out,
            [
                0,
                1,
                u32::from(u16::MAX),
                120,
                0,
                u32::from(u16::MAX),
                999,
                999
            ]
        );
        for (&id, &value) in ids.iter().zip(&out) {
            assert_eq!(value, docs.length(id));
        }
        docs.gather_lengths(&[], &mut []);
        assert_eq!(docs.length_bytes(), original);

        let mut builder = ChunkMapBuilder::default();
        for id in 0..100 {
            builder
                .push(
                    id,
                    0,
                    if id == 99 {
                        300
                    } else if id == 0 {
                        0
                    } else {
                        10
                    },
                )
                .unwrap();
        }
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(0, &builder)], &[]).unwrap();
        let file = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
        let map = &file.chunk_maps[&0];
        let original = map.length_bytes().to_vec();
        let ids = [0, 99, 50, 1];
        let mut out = [999; 6];
        map.gather_bm25_lengths(&ids, &mut out);
        assert_eq!(out, [10, 300, 10, 10, 999, 999]);
        for (&id, &value) in ids.iter().zip(&out) {
            assert_eq!(value, map.bm25_length(id));
        }
        assert_eq!(map.length_bytes(), original);
    }

    #[test]
    fn invalid_virtual_ids_score_as_one_token_and_are_reported_once() {
        let mut out = [0; 2];
        gather_lengths::<true>(&[7, 0], &[u32::MAX, 0], &mut out, 0);
        assert_eq!(out, [INVALID_CHUNK_LENGTH, 7]);
        assert!(invalid_chunk_id_reported());
        let mut floored = [0];
        gather_lengths::<true>(&[0, 0], &[u32::MAX], &mut floored, 10);
        assert_eq!(floored, [10], "the substitute still honours the BM25 floor");

        let map = build(&[(0, 0, 10)]);
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(0, &map)], &[]).unwrap();
        let map = &read_chunk_maps(OwnedBytes::new(bytes)).unwrap().chunk_maps[&0];
        assert_eq!(map.length(0), 10);
        assert_eq!(map.length(1), INVALID_CHUNK_LENGTH);
        assert_eq!(
            map.bm25_length(1),
            INVALID_CHUNK_LENGTH.max(map.length_floor())
        );
        // Non-strict document lengths keep reporting absence as 0.
        let mut plain = [9];
        gather_lengths::<false>(&[0, 0], &[5], &mut plain, 0);
        assert_eq!(plain, [0]);
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

    #[test]
    fn addressed_chunk_maps_copy_and_remap_slots_without_losing_missing_ordinals() {
        let source = build(&[(2, 7, 11), (0, 3, 12), (2, 1, 13)]);
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(1, &source)], &[]).unwrap();
        let file = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
        let map = &file.chunk_maps[&1];
        assert_eq!(
            map.slot_for_unit(super::super::logical_address::LogicalUnit { doc: 2, ordinal: 7 }),
            Some(0)
        );
        assert_eq!(
            map.slot_for_unit(super::super::logical_address::LogicalUnit { doc: 2, ordinal: 0 }),
            None
        );
        let mut merged = Vec::new();
        write_merged_chunk_maps(
            &mut merged,
            &[(
                1,
                vec![
                    ChunkMapSource { map, doc_offset: 0 },
                    ChunkMapSource { map, doc_offset: 4 },
                ],
            )],
            &[],
        )
        .unwrap();
        let file = read_chunk_maps(OwnedBytes::new(merged)).unwrap();
        let result = &file.chunk_maps[&1];
        assert_eq!(
            result.ordinal_bytes(),
            [map.ordinal_bytes(), map.ordinal_bytes()].concat()
        );
        assert_eq!(
            result.length_bytes(),
            [map.length_bytes(), map.length_bytes()].concat()
        );
        assert_eq!(
            result.slot_for_unit(super::super::logical_address::LogicalUnit { doc: 6, ordinal: 7 }),
            Some(3)
        );
        assert_eq!(
            result.slot_for_unit(super::super::logical_address::LogicalUnit { doc: 4, ordinal: 3 }),
            Some(4)
        );
        assert_eq!(
            result.slot_for_unit(super::super::logical_address::LogicalUnit { doc: 6, ordinal: 1 }),
            Some(5)
        );
        assert_eq!(
            result.slot_for_unit(super::super::logical_address::LogicalUnit { doc: 5, ordinal: 0 }),
            None
        );
    }

    #[test]
    fn chunk_map_versions_preserve_legacy_capability_and_reject_corrupt_addressing() {
        let source = build(&[(2, 0, 11), (0, 0, 12)]);
        let mut bytes = Vec::new();
        write_chunk_maps(&mut bytes, &[(1, &source)], &[]).unwrap();
        let mut invalid = bytes.clone();
        let end = invalid.len();
        invalid[end - 4..].copy_from_slice(&1u32.to_le_bytes());
        assert!(
            read_chunk_maps(OwnedBytes::new(invalid)).is_err(),
            "duplicate slots"
        );
        let mut legacy = bytes.clone();
        legacy.truncate(legacy.len() - 8);
        legacy[16..20].copy_from_slice(&KIND_CHUNK_MAP.to_le_bytes());
        assert!(
            read_chunk_maps(OwnedBytes::new(legacy.clone())).is_err(),
            "V3 ordered kind must be ordered"
        );
        legacy[4..8].copy_from_slice(&2u32.to_le_bytes());
        let old = read_chunk_maps(OwnedBytes::new(legacy)).unwrap();
        let old_map = &old.chunk_maps[&1];
        assert!(!old_map.has_logical_addressing());
        let mut merged = Vec::new();
        write_merged_chunk_maps(
            &mut merged,
            &[(
                1,
                vec![ChunkMapSource {
                    map: old_map,
                    doc_offset: 0,
                }],
            )],
            &[],
        )
        .unwrap();
        assert!(
            !read_chunk_maps(OwnedBytes::new(merged)).unwrap().chunk_maps[&1]
                .has_logical_addressing()
        );
        let current = read_chunk_maps(OwnedBytes::new(bytes)).unwrap();
        let mut output = Vec::new();
        assert!(
            write_merged_chunk_maps(
                &mut output,
                &[(
                    1,
                    vec![
                        ChunkMapSource {
                            map: old_map,
                            doc_offset: 0
                        },
                        ChunkMapSource {
                            map: &current.chunk_maps[&1],
                            doc_offset: 3
                        },
                    ]
                )],
                &[]
            )
            .is_err()
        );
        assert!(
            output.is_empty(),
            "reject incompatible sources before writing"
        );
    }
}
