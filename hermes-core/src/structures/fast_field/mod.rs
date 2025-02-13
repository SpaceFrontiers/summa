//! Fast field columnar storage for efficient filtering and sorting.
//!
//! Stores one column per fast-field, indexed by doc_id for O(1) access.
//! Supports u64, i64, f64, and text (dictionary-encoded ordinal) columns.
//! Both single-valued and multi-valued columns are supported.
//!
//! ## File format (`.fast` — version FST2)
//!
//! ```text
//! [column 0 blocked data] [column 1 blocked data] ... [column N blocked data]
//! [TOC: FastFieldTocEntry × num_columns]
//! [footer: toc_offset(8) + num_columns(4) + magic(4)]  = 16 bytes
//! ```
//!
//! ## Blocked column format
//!
//! Each column's data region is a sequence of independently-decodable blocks:
//!
//! ```text
//! [num_blocks: u32]
//! [block_index: BlockIndexEntry × num_blocks]   (16 bytes each)
//! [block_0 data] [block_0 dict?] [block_1 data] [block_1 dict?] ...
//! ```
//!
//! `BlockIndexEntry`: num_docs(4) + data_len(4) + dict_count(4) + dict_len(4)
//!
//! Fresh segments produce a single block. Merges stack blocks from source
//! segments via raw byte copy (memcpy) — no per-value decode/re-encode.
//!
//! ## Codecs (auto-selected per block at build time)
//!
//! | ID | Codec           | Description                               |
//! |----|-----------------|-------------------------------------------|
//! |  0 | Constant        | All values identical — 0 data bytes       |
//! |  1 | Bitpacked       | min-subtract + global bitpack             |
//! |  2 | Linear          | Regression line + bitpacked residuals     |
//! |  3 | BlockwiseLinear | Per-512-block linear + residuals          |

pub mod codec;

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// ── Constants ─────────────────────────────────────────────────────────────

/// Magic number for `.fast` file footer — FST2 (auto-codec + multi-value)
pub const FAST_FIELD_MAGIC: u32 = 0x32545346;

/// Footer size: toc_offset(8) + num_columns(4) + magic(4) = 16
pub const FAST_FIELD_FOOTER_SIZE: u64 = 16;

/// Sentinel for missing / absent values in any fast-field column type.
///
/// - **Text**: document has no value → ordinal stored as `u64::MAX`
/// - **Numeric (u64/i64/f64)**: document has no value → raw stored as `u64::MAX`
///
/// Callers should check `raw != FAST_FIELD_MISSING` before interpreting
/// the value as a real number or ordinal.
pub const FAST_FIELD_MISSING: u64 = u64::MAX;

// ── Column type ───────────────────────────────────────────────────────────

/// Type of a fast-field column (stored in TOC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FastFieldColumnType {
    U64 = 0,
    I64 = 1,
    F64 = 2,
    TextOrdinal = 3,
}

impl FastFieldColumnType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::U64),
            1 => Some(Self::I64),
            2 => Some(Self::F64),
            3 => Some(Self::TextOrdinal),
            _ => None,
        }
    }
}

// ── Encoding helpers ──────────────────────────────────────────────────────

/// Zigzag-encode an i64 to u64 (small absolute values → small u64).
#[inline]
pub fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Zigzag-decode a u64 back to i64.
#[inline]
pub fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Encode f64 to u64 preserving total order.
/// Positive floats: flip sign bit (so they sort above negatives).
/// Negative floats: flip all bits (so they sort in reverse magnitude).
#[inline]
pub fn f64_to_sortable_u64(f: f64) -> u64 {
    let bits = f.to_bits();
    if (bits >> 63) == 0 {
        bits ^ (1u64 << 63) // positive: flip sign bit
    } else {
        !bits // negative: flip all bits
    }
}

/// Decode sortable u64 back to f64.
#[inline]
pub fn sortable_u64_to_f64(v: u64) -> f64 {
    let bits = if (v >> 63) != 0 {
        v ^ (1u64 << 63) // was positive: unflip sign bit
    } else {
        !v // was negative: unflip all bits
    };
    f64::from_bits(bits)
}

/// Minimum number of bits needed to represent `val`.
#[inline]
pub fn bits_needed_u64(val: u64) -> u8 {
    if val == 0 {
        0
    } else {
        64 - val.leading_zeros() as u8
    }
}

// ── Bit-packing ───────────────────────────────────────────────────────────

/// Pack `values` at `bits_per_value` bits each into `out`.
/// `out` must be large enough: `ceil(values.len() * bits_per_value / 8)` bytes.
pub fn bitpack_write(values: &[u64], bits_per_value: u8, out: &mut Vec<u8>) {
    if bits_per_value == 0 {
        return; // all values are the same (constant column)
    }
    let bpv = bits_per_value as usize;
    let total_bits = values.len() * bpv;
    let total_bytes = total_bits.div_ceil(8);
    out.reserve(total_bytes);

    let start = out.len();
    out.resize(start + total_bytes, 0);
    let buf = &mut out[start..];

    for (i, &val) in values.iter().enumerate() {
        let bit_offset = i * bpv;
        let byte_offset = bit_offset / 8;
        let bit_shift = bit_offset % 8;

        // Write across byte boundaries (up to 9 bytes for 64-bit values)
        let mut remaining_bits = bpv;
        let mut v = val;
        let mut bo = byte_offset;
        let mut bs = bit_shift;

        while remaining_bits > 0 {
            let can_write = (8 - bs).min(remaining_bits);
            let mask = (1u64 << can_write) - 1;
            buf[bo] |= ((v & mask) << bs) as u8;
            v >>= can_write;
            remaining_bits -= can_write;
            bo += 1;
            bs = 0;
        }
    }
}

/// Read value at `index` from bit-packed data.
///
/// Fast path: reads a single unaligned u64 (LE) covering the target bits,
/// shifts and masks. This compiles to ~4 instructions on x86/ARM and avoids
/// the per-byte loop entirely for bpv ≤ 56.
#[inline]
pub fn bitpack_read(data: &[u8], bits_per_value: u8, index: usize) -> u64 {
    if bits_per_value == 0 {
        return 0;
    }
    let bpv = bits_per_value as usize;
    let bit_offset = index * bpv;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;

    // Fast path: single unaligned LE u64 load, shift, and mask.
    // Valid when all needed bits fit within 8 bytes: bit_shift + bpv ≤ 64.
    if bit_shift + bpv <= 64 && byte_offset + 8 <= data.len() {
        let raw = u64::from_le_bytes(data[byte_offset..byte_offset + 8].try_into().unwrap());
        let mask = if bpv >= 64 {
            u64::MAX
        } else {
            (1u64 << bpv) - 1
        };
        return (raw >> bit_shift) & mask;
    }

    // Slow path for the last few values near the end of the buffer
    let mut result: u64 = 0;
    let mut remaining_bits = bpv;
    let mut bo = byte_offset;
    let mut bs = bit_shift;
    let mut out_shift = 0;

    while remaining_bits > 0 {
        let can_read = (8 - bs).min(remaining_bits);
        let mask = ((1u64 << can_read) - 1) as u8;
        let byte_val = if bo < data.len() { data[bo] } else { 0 };
        result |= (((byte_val >> bs) & mask) as u64) << out_shift;
        remaining_bits -= can_read;
        out_shift += can_read;
        bo += 1;
        bs = 0;
    }

    result
}

// ── TOC entry ─────────────────────────────────────────────────────────────

/// On-disk TOC entry for a fast-field column (FST2 format).
///
/// Wire: field_id(4) + column_type(1) + flags(1) + data_offset(8) + data_len(8) +
///       num_docs(4) + dict_offset(8) + dict_count(4) = 38 bytes
///
/// The `flags` byte encodes:
///   bit 0: multi-valued column (offset+value sub-columns)
///
/// For multi-valued columns, the data region contains:
///   [offset column (auto-codec)] [value column (auto-codec)]
///   with a 4-byte length prefix for the offset column so the reader knows where
///   the value column starts.
#[derive(Debug, Clone)]
pub struct FastFieldTocEntry {
    pub field_id: u32,
    pub column_type: FastFieldColumnType,
    pub multi: bool,
    pub data_offset: u64,
    pub data_len: u64,
    pub num_docs: u32,
    /// Byte offset of the text dictionary section (0 for numeric columns).
    pub dict_offset: u64,
    /// Number of entries in the text dictionary (0 for numeric columns).
    pub dict_count: u32,
}

/// FST2 TOC entry size: field_id(4)+column_type(1)+flags(1)+data_offset(8)+data_len(8)+num_docs(4)+dict_offset(8)+dict_count(4) = 38
pub const FAST_FIELD_TOC_ENTRY_SIZE: usize = 4 + 1 + 1 + 8 + 8 + 4 + 8 + 4; // 38

// ── Block index entry ─────────────────────────────────────────────────────

/// On-disk index entry for one block within a blocked column.
///
/// Wire: num_docs(4) + data_len(4) + dict_count(4) + dict_len(4) = 16 bytes
#[derive(Debug, Clone)]
pub struct BlockIndexEntry {
    pub num_docs: u32,
    pub data_len: u32,
    pub dict_count: u32,
    pub dict_len: u32,
}

pub const BLOCK_INDEX_ENTRY_SIZE: usize = 16;

impl BlockIndexEntry {
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.num_docs)?;
        w.write_u32::<LittleEndian>(self.data_len)?;
        w.write_u32::<LittleEndian>(self.dict_count)?;
        w.write_u32::<LittleEndian>(self.dict_len)?;
        Ok(())
    }

    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        let num_docs = r.read_u32::<LittleEndian>()?;
        let data_len = r.read_u32::<LittleEndian>()?;
        let dict_count = r.read_u32::<LittleEndian>()?;
        let dict_len = r.read_u32::<LittleEndian>()?;
        Ok(Self {
            num_docs,
            data_len,
            dict_count,
            dict_len,
        })
    }
}

impl FastFieldTocEntry {
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.field_id)?;
        w.write_u8(self.column_type as u8)?;
        let flags: u8 = if self.multi { 1 } else { 0 };
        w.write_u8(flags)?;
        w.write_u64::<LittleEndian>(self.data_offset)?;
        w.write_u64::<LittleEndian>(self.data_len)?;
        w.write_u32::<LittleEndian>(self.num_docs)?;
        w.write_u64::<LittleEndian>(self.dict_offset)?;
        w.write_u32::<LittleEndian>(self.dict_count)?;
        Ok(())
    }

    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        let field_id = r.read_u32::<LittleEndian>()?;
        let ct = r.read_u8()?;
        let column_type = FastFieldColumnType::from_u8(ct)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad column type"))?;
        let flags = r.read_u8()?;
        let multi = (flags & 1) != 0;
        let data_offset = r.read_u64::<LittleEndian>()?;
        let data_len = r.read_u64::<LittleEndian>()?;
        let num_docs = r.read_u32::<LittleEndian>()?;
        let dict_offset = r.read_u64::<LittleEndian>()?;
        let dict_count = r.read_u32::<LittleEndian>()?;
        Ok(Self {
            field_id,
            column_type,
            multi,
            data_offset,
            data_len,
            num_docs,
            dict_offset,
            dict_count,
        })
    }
}

// ── Writer ────────────────────────────────────────────────────────────────

/// Collects values during indexing and serializes a single fast-field column.
///
/// Supports both single-valued and multi-valued columns.
/// For multi-valued columns, values are stored in a flat array with an
/// offset column that maps doc_id → value range.
pub struct FastFieldWriter {
    pub column_type: FastFieldColumnType,
    /// Whether this is a multi-valued column.
    pub multi: bool,

    // ── Single-valued state ──
    /// Raw u64 values indexed by local doc_id (single-value mode).
    values: Vec<u64>,

    // ── Multi-valued state ──
    /// Flat list of all values (multi-value mode).
    multi_values: Vec<u64>,
    /// Per-doc cumulative offset into `multi_values`. Length = num_docs + 1.
    /// offsets[doc_id]..offsets[doc_id+1] is the value range for doc_id.
    multi_offsets: Vec<u32>,
    /// Current doc_id being filled (for multi-value sequential writes).
    multi_current_doc: u32,

    // ── Text state (shared) ──
    /// For TextOrdinal: maps original string → insertion order.
    text_values: Option<BTreeMap<String, u32>>,
    /// For TextOrdinal single-value: per-doc string values (parallel to `values`).
    text_per_doc: Option<Vec<Option<String>>>,
    /// For TextOrdinal multi-value: per-value strings (parallel to `multi_values`).
    text_multi_values: Option<Vec<String>>,
}

impl FastFieldWriter {
    /// Create a writer for a single-valued numeric column (u64/i64/f64).
    pub fn new_numeric(column_type: FastFieldColumnType) -> Self {
        debug_assert!(matches!(
            column_type,
            FastFieldColumnType::U64 | FastFieldColumnType::I64 | FastFieldColumnType::F64
        ));
        Self {
            column_type,
            multi: false,
            values: Vec::new(),
            multi_values: Vec::new(),
            multi_offsets: vec![0],
            multi_current_doc: 0,
            text_values: None,
            text_per_doc: None,
            text_multi_values: None,
        }
    }

    /// Create a writer for a multi-valued numeric column.
    pub fn new_numeric_multi(column_type: FastFieldColumnType) -> Self {
        debug_assert!(matches!(
            column_type,
            FastFieldColumnType::U64 | FastFieldColumnType::I64 | FastFieldColumnType::F64
        ));
        Self {
            column_type,
            multi: true,
            values: Vec::new(),
            multi_values: Vec::new(),
            multi_offsets: vec![0],
            multi_current_doc: 0,
            text_values: None,
            text_per_doc: None,
            text_multi_values: None,
        }
    }

    /// Create a writer for a single-valued text ordinal column.
    pub fn new_text() -> Self {
        Self {
            column_type: FastFieldColumnType::TextOrdinal,
            multi: false,
            values: Vec::new(),
            multi_values: Vec::new(),
            multi_offsets: vec![0],
            multi_current_doc: 0,
            text_values: Some(BTreeMap::new()),
            text_per_doc: Some(Vec::new()),
            text_multi_values: None,
        }
    }

    /// Create a writer for a multi-valued text ordinal column.
    pub fn new_text_multi() -> Self {
        Self {
            column_type: FastFieldColumnType::TextOrdinal,
            multi: true,
            values: Vec::new(),
            multi_values: Vec::new(),
            multi_offsets: vec![0],
            multi_current_doc: 0,
            text_values: Some(BTreeMap::new()),
            text_per_doc: None,
            text_multi_values: Some(Vec::new()),
        }
    }

    /// Record a numeric value for `doc_id`. Fills gaps with 0.
    /// For single-value mode only.
    pub fn add_u64(&mut self, doc_id: u32, value: u64) {
        if self.multi {
            self.add_multi_u64(doc_id, value);
            return;
        }
        let idx = doc_id as usize;
        if idx >= self.values.len() {
            self.values.resize(idx + 1, FAST_FIELD_MISSING);
            if let Some(ref mut tpd) = self.text_per_doc {
                tpd.resize(idx + 1, None);
            }
        }
        self.values[idx] = value;
    }

    /// Record a value in multi-value mode.
    fn add_multi_u64(&mut self, doc_id: u32, value: u64) {
        // Pad offsets for any skipped doc_ids
        while self.multi_current_doc < doc_id {
            self.multi_current_doc += 1;
            self.multi_offsets.push(self.multi_values.len() as u32);
        }
        // Ensure offset exists for current doc
        if self.multi_current_doc == doc_id && self.multi_offsets.len() == doc_id as usize + 1 {
            // offset for doc_id already exists as the last entry
        }
        self.multi_values.push(value);
    }

    /// Record an i64 value (zigzag-encoded).
    pub fn add_i64(&mut self, doc_id: u32, value: i64) {
        self.add_u64(doc_id, zigzag_encode(value));
    }

    /// Record an f64 value (sortable-encoded).
    pub fn add_f64(&mut self, doc_id: u32, value: f64) {
        self.add_u64(doc_id, f64_to_sortable_u64(value));
    }

    /// Record a text value (dictionary-encoded at build time).
    pub fn add_text(&mut self, doc_id: u32, value: &str) {
        if let Some(ref mut dict) = self.text_values {
            let next_id = dict.len() as u32;
            dict.entry(value.to_string()).or_insert(next_id);
        }

        if self.multi {
            if let Some(ref mut tmv) = self.text_multi_values {
                // Pad offsets for skipped docs
                while self.multi_current_doc < doc_id {
                    self.multi_current_doc += 1;
                    self.multi_offsets.push(self.multi_values.len() as u32);
                }
                if self.multi_current_doc == doc_id
                    && self.multi_offsets.len() == doc_id as usize + 1
                {
                    // offset already exists
                }
                self.multi_values.push(0); // placeholder, resolved later
                tmv.push(value.to_string());
            }
        } else {
            let idx = doc_id as usize;
            if idx >= self.values.len() {
                self.values.resize(idx + 1, FAST_FIELD_MISSING);
            }
            if let Some(ref mut tpd) = self.text_per_doc {
                if idx >= tpd.len() {
                    tpd.resize(idx + 1, None);
                }
                tpd[idx] = Some(value.to_string());
            }
        }
    }

    /// Ensure the column covers `num_docs` entries.
    ///
    /// Absent entries are filled with [`FAST_FIELD_MISSING`] for single-value
    /// columns, or with empty offset ranges for multi-value columns.
    pub fn pad_to(&mut self, num_docs: u32) {
        let n = num_docs as usize;
        if self.multi {
            while (self.multi_offsets.len() as u32) <= num_docs {
                self.multi_offsets.push(self.multi_values.len() as u32);
            }
            self.multi_current_doc = num_docs;
        } else {
            if self.values.len() < n {
                self.values.resize(n, FAST_FIELD_MISSING);
                if let Some(ref mut tpd) = self.text_per_doc {
                    tpd.resize(n, None);
                }
            }
        }
    }

    /// Number of documents in this column.
    pub fn num_docs(&self) -> u32 {
        if self.multi {
            // offsets has num_docs+1 entries
            (self.multi_offsets.len() as u32).saturating_sub(1)
        } else {
            self.values.len() as u32
        }
    }

    /// Serialize column data using blocked format with auto-selecting codec.
    ///
    /// Writes a single block: [num_blocks(4)] [BlockIndexEntry] [block_data] [block_dict?]
    /// Returns `(toc_entry, total_bytes_written)`.
    pub fn serialize(
        &mut self,
        writer: &mut dyn Write,
        data_offset: u64,
    ) -> io::Result<(FastFieldTocEntry, u64)> {
        // For text ordinal: resolve strings to sorted ordinals
        if self.column_type == FastFieldColumnType::TextOrdinal {
            self.resolve_text_ordinals();
        }

        let num_docs = self.num_docs();

        // Serialize block data into a temp buffer to measure lengths
        let mut block_data = Vec::new();
        if self.multi {
            // Multi-value: write [offset_col_len(4)] [offset_col] [value_col]
            let offsets_u64: Vec<u64> = self.multi_offsets.iter().map(|&v| v as u64).collect();
            let mut offset_buf = Vec::new();
            codec::serialize_auto(&offsets_u64, &mut offset_buf)?;

            block_data.write_u32::<LittleEndian>(offset_buf.len() as u32)?;
            block_data.write_all(&offset_buf)?;

            codec::serialize_auto(&self.multi_values, &mut block_data)?;
        } else {
            codec::serialize_auto(&self.values, &mut block_data)?;
        }

        // Serialize text dictionary into temp buffer
        let mut dict_buf = Vec::new();
        let dict_count = if self.column_type == FastFieldColumnType::TextOrdinal {
            let (count, _) = self.write_text_dictionary(&mut dict_buf)?;
            count
        } else {
            0u32
        };

        // Build block index entry
        let block_entry = BlockIndexEntry {
            num_docs,
            data_len: block_data.len() as u32,
            dict_count,
            dict_len: dict_buf.len() as u32,
        };

        // Write: num_blocks + block_index + block_data + block_dict
        let mut total_bytes = 0u64;

        writer.write_u32::<LittleEndian>(1u32)?; // num_blocks
        total_bytes += 4;

        block_entry.write_to(writer)?;
        total_bytes += BLOCK_INDEX_ENTRY_SIZE as u64;

        writer.write_all(&block_data)?;
        total_bytes += block_data.len() as u64;

        writer.write_all(&dict_buf)?;
        total_bytes += dict_buf.len() as u64;

        let toc = FastFieldTocEntry {
            field_id: 0, // set by caller
            column_type: self.column_type,
            multi: self.multi,
            data_offset,
            data_len: total_bytes,
            num_docs,
            dict_offset: 0, // no longer used at TOC level (per-block dicts)
            dict_count: 0,
        };

        Ok((toc, total_bytes))
    }

    /// Resolve text per-doc values to sorted ordinals.
    fn resolve_text_ordinals(&mut self) {
        let dict = self.text_values.as_ref().expect("text_values required");

        // Build sorted ordinal map: BTreeMap iterates in sorted order
        let sorted_ordinals: BTreeMap<&str, u64> = dict
            .keys()
            .enumerate()
            .map(|(ord, key)| (key.as_str(), ord as u64))
            .collect();

        if self.multi {
            // Multi-value: resolve multi_values via text_multi_values
            if let Some(ref tmv) = self.text_multi_values {
                for (i, text) in tmv.iter().enumerate() {
                    self.multi_values[i] = sorted_ordinals[text.as_str()];
                }
            }
        } else {
            // Single-value: resolve values via text_per_doc
            let tpd = self.text_per_doc.as_ref().expect("text_per_doc required");
            for (i, doc_text) in tpd.iter().enumerate() {
                match doc_text {
                    Some(text) => {
                        self.values[i] = sorted_ordinals[text.as_str()];
                    }
                    None => {
                        self.values[i] = FAST_FIELD_MISSING;
                    }
                }
            }
        }
    }

    /// Write len-prefixed sorted strings. Returns (dict_count, bytes_written).
    fn write_text_dictionary(&self, writer: &mut dyn Write) -> io::Result<(u32, u64)> {
        let dict = self.text_values.as_ref().expect("text_values required");
        let mut bytes_written = 0u64;

        // BTreeMap keys are already sorted
        let count = dict.len() as u32;
        for key in dict.keys() {
            let key_bytes = key.as_bytes();
            writer.write_u32::<LittleEndian>(key_bytes.len() as u32)?;
            writer.write_all(key_bytes)?;
            bytes_written += 4 + key_bytes.len() as u64;
        }

        Ok((count, bytes_written))
    }
}

// ── Reader ────────────────────────────────────────────────────────────────

use crate::directories::OwnedBytes;

/// One independently-decodable block within a blocked column.
///
/// All byte slices are zero-copy borrows from the mmap'd `.fast` file.
pub struct ColumnBlock {
    /// Number of docs before this block (for doc_id → block lookup).
    pub cumulative_docs: u32,
    /// Number of docs in this block.
    pub num_docs: u32,
    /// Auto-codec encoded data for this block (single-value or raw multi-value region).
    pub data: OwnedBytes,
    /// For multi-value blocks: offset sub-column.
    pub offset_data: OwnedBytes,
    /// For multi-value blocks: value sub-column.
    pub value_data: OwnedBytes,
    /// Per-block text dictionary (text columns only).
    pub dict: Option<TextDictReader>,
    /// Block-local ordinal → global ordinal mapping (text columns with >1 block).
    pub ordinal_map: Vec<u32>,
    /// Raw dictionary bytes for this block (for merge: memcpy).
    pub raw_dict: OwnedBytes,
}

/// Reads a single fast-field column from mmap/buffer with O(1) doc_id access.
///
/// A column is a sequence of independently-decodable blocks. Fresh segments
/// have one block; merged segments may have multiple (one per source segment).
///
/// **Zero-copy**: all data is borrowed from the underlying mmap / `OwnedBytes`.
pub struct FastFieldReader {
    pub column_type: FastFieldColumnType,
    pub num_docs: u32,
    pub multi: bool,

    /// Blocks in doc_id order.
    blocks: Vec<ColumnBlock>,

    /// Global merged text dictionary (for text_ordinal lookups and filters).
    /// Built at open time by merging per-block dictionaries.
    global_text_dict: Option<TextDictReader>,
}

impl FastFieldReader {
    /// Open a blocked column from an `OwnedBytes` file buffer using a TOC entry.
    pub fn open(file_data: &OwnedBytes, toc: &FastFieldTocEntry) -> io::Result<Self> {
        let region_start = toc.data_offset as usize;
        let region_end = region_start + toc.data_len as usize;

        if region_end > file_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "fast field data out of bounds",
            ));
        }

        let raw = file_data.as_slice();

        // Read num_blocks
        let mut pos = region_start;
        if pos + 4 > region_end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "fast field: missing num_blocks",
            ));
        }
        let num_blocks = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
        pos += 4;

        // Read block index
        let idx_size = num_blocks as usize * BLOCK_INDEX_ENTRY_SIZE;
        if pos + idx_size > region_end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "fast field: block index truncated",
            ));
        }
        let mut block_entries = Vec::with_capacity(num_blocks as usize);
        {
            let mut cursor = std::io::Cursor::new(&raw[pos..pos + idx_size]);
            for _ in 0..num_blocks {
                block_entries.push(BlockIndexEntry::read_from(&mut cursor)?);
            }
        }
        pos += idx_size;

        let empty = OwnedBytes::new(Vec::new());

        // Parse each block's data + dict slices
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        let mut cumulative = 0u32;

        for entry in &block_entries {
            let data_start = pos;
            let data_end = data_start + entry.data_len as usize;
            let dict_start = data_end;
            let dict_end = dict_start + entry.dict_len as usize;

            if dict_end > file_data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "fast field: block data/dict truncated",
                ));
            }

            // Parse multi-value sub-columns from block data
            let (block_data, offset_data, value_data) = if toc.multi {
                let block_raw = &raw[data_start..data_end];
                if block_raw.len() < 4 {
                    (empty.clone(), empty.clone(), empty.clone())
                } else {
                    let offset_col_len =
                        u32::from_le_bytes(block_raw[0..4].try_into().unwrap()) as usize;
                    let o_start = data_start + 4;
                    let o_end = o_start + offset_col_len;
                    let v_start = o_end;
                    let v_end = data_end;
                    (
                        file_data.slice(data_start..data_end),
                        file_data.slice(o_start..o_end),
                        file_data.slice(v_start..v_end),
                    )
                }
            } else {
                (
                    file_data.slice(data_start..data_end),
                    empty.clone(),
                    empty.clone(),
                )
            };

            // Parse block dict
            let dict = if entry.dict_count > 0 {
                Some(TextDictReader::open(
                    file_data,
                    dict_start,
                    entry.dict_count,
                )?)
            } else {
                None
            };

            let raw_dict = if entry.dict_len > 0 {
                file_data.slice(dict_start..dict_end)
            } else {
                empty.clone()
            };

            blocks.push(ColumnBlock {
                cumulative_docs: cumulative,
                num_docs: entry.num_docs,
                data: block_data,
                offset_data,
                value_data,
                dict,
                ordinal_map: Vec::new(),
                raw_dict,
            });

            cumulative += entry.num_docs;
            pos = dict_end;
        }

        // Build global text dictionary for multi-block text columns
        let global_text_dict = if toc.column_type == FastFieldColumnType::TextOrdinal {
            Some(Self::build_global_text_dict(&mut blocks)?)
        } else {
            None
        };

        Ok(Self {
            column_type: toc.column_type,
            num_docs: toc.num_docs,
            multi: toc.multi,
            blocks,
            global_text_dict,
        })
    }

    /// Build a global merged dictionary from all per-block dictionaries.
    /// Also populates `ordinal_map` on each block (block-local → global ordinal).
    fn build_global_text_dict(blocks: &mut [ColumnBlock]) -> io::Result<TextDictReader> {
        use std::collections::BTreeSet;

        // Collect all unique strings from all blocks
        let mut all_strings = BTreeSet::new();
        for block in blocks.iter() {
            if let Some(ref dict) = block.dict {
                for s in dict.iter() {
                    all_strings.insert(s.to_string());
                }
            }
        }

        // Build global sorted list
        let global_sorted: Vec<String> = all_strings.into_iter().collect();

        // Build per-block ordinal maps: block-local ordinal → global ordinal
        for block in blocks.iter_mut() {
            if let Some(ref dict) = block.dict {
                let mut map = Vec::with_capacity(dict.len() as usize);
                for local_ord in 0..dict.len() {
                    let text = dict.get(local_ord).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "block dict ordinal {} out of range (dict len {})",
                                local_ord,
                                dict.len()
                            ),
                        )
                    })?;
                    let global_ord = global_sorted
                        .binary_search_by(|s| s.as_str().cmp(text))
                        .map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "block dict entry {:?} not found in merged global dict",
                                    text
                                ),
                            )
                        })? as u32;
                    map.push(global_ord);
                }
                block.ordinal_map = map;
            }
        }

        // Serialize global dict into a buffer and create a TextDictReader
        let mut dict_buf = Vec::new();
        for s in &global_sorted {
            let bytes = s.as_bytes();
            dict_buf.write_u32::<LittleEndian>(bytes.len() as u32)?;
            dict_buf.extend_from_slice(bytes);
        }
        let dict_data = OwnedBytes::new(dict_buf);
        TextDictReader::open(&dict_data, 0, global_sorted.len() as u32)
    }

    /// Remap a block-local raw ordinal to a global ordinal using the block's ordinal_map.
    /// Returns raw unchanged for non-text columns, missing ordinals, or if no map exists.
    #[inline]
    fn remap_ordinal(&self, block: &ColumnBlock, raw: u64) -> u64 {
        if self.column_type == FastFieldColumnType::TextOrdinal
            && raw != FAST_FIELD_MISSING
            && !block.ordinal_map.is_empty()
        {
            let idx = raw as usize;
            if idx < block.ordinal_map.len() {
                block.ordinal_map[idx] as u64
            } else {
                // Corrupt ordinal — treat as missing
                FAST_FIELD_MISSING
            }
        } else {
            raw
        }
    }

    /// Find the block containing `doc_id`. Returns (block_index, local_doc_id).
    #[inline]
    fn find_block(&self, doc_id: u32) -> (usize, u32) {
        debug_assert!(!self.blocks.is_empty());
        // Single block fast path (common: fresh segments)
        if self.blocks.len() == 1 {
            return (0, doc_id);
        }
        // Binary search: find the last block whose cumulative_docs <= doc_id
        let bi = self
            .blocks
            .partition_point(|b| b.cumulative_docs <= doc_id)
            .saturating_sub(1);
        (bi, doc_id - self.blocks[bi].cumulative_docs)
    }

    /// Get raw u64 value for a doc_id.
    ///
    /// Returns [`FAST_FIELD_MISSING`] for out-of-range doc_ids **and** for docs
    /// that were never assigned a value (absent docs).
    ///
    /// For text columns, returns the global ordinal (remapped from block-local).
    /// For multi-valued columns, returns the first value (or `FAST_FIELD_MISSING` if empty).
    #[inline]
    pub fn get_u64(&self, doc_id: u32) -> u64 {
        if doc_id >= self.num_docs {
            return FAST_FIELD_MISSING;
        }
        let (bi, local) = self.find_block(doc_id);
        let block = &self.blocks[bi];

        if self.multi {
            let start = codec::auto_read(block.offset_data.as_slice(), local as usize) as u32;
            let end = codec::auto_read(block.offset_data.as_slice(), local as usize + 1) as u32;
            if start >= end {
                return FAST_FIELD_MISSING;
            }
            let raw = codec::auto_read(block.value_data.as_slice(), start as usize);
            return self.remap_ordinal(block, raw);
        }

        let raw = codec::auto_read(block.data.as_slice(), local as usize);
        self.remap_ordinal(block, raw)
    }

    /// Get the value range for a multi-valued column within its block.
    /// Returns (block_index, start_index, end_index) into the block's flat value array.
    #[inline]
    fn block_value_range(&self, doc_id: u32) -> (usize, u32, u32) {
        if !self.multi || doc_id >= self.num_docs {
            return (0, 0, 0);
        }
        let (bi, local) = self.find_block(doc_id);
        let block = &self.blocks[bi];
        let start = codec::auto_read(block.offset_data.as_slice(), local as usize) as u32;
        let end = codec::auto_read(block.offset_data.as_slice(), local as usize + 1) as u32;
        (bi, start, end)
    }

    /// Get the value range for a multi-valued column.
    /// Returns (start_index, end_index) — for single-block columns these are
    /// direct indices; for multi-block, use `get_multi_values` instead.
    #[inline]
    pub fn value_range(&self, doc_id: u32) -> (u32, u32) {
        let (_, start, end) = self.block_value_range(doc_id);
        (start, end)
    }

    /// Get a specific value from the flat value array (multi-value mode).
    /// For single-block columns only. For multi-block, use `get_multi_values`.
    #[inline]
    pub fn get_value_at(&self, index: u32) -> u64 {
        // For single-block (common case), delegate directly
        if self.blocks.len() == 1 {
            let raw = codec::auto_read(self.blocks[0].value_data.as_slice(), index as usize);
            return self.remap_ordinal(&self.blocks[0], raw);
        }
        // Multi-block fallback — index is block-local, caller should use get_multi_values
        0
    }

    /// Get all values for a multi-valued doc_id. Handles multi-block correctly.
    pub fn get_multi_values(&self, doc_id: u32) -> Vec<u64> {
        let (bi, start, end) = self.block_value_range(doc_id);
        if start >= end {
            return Vec::new();
        }
        let block = &self.blocks[bi];
        (start..end)
            .map(|idx| {
                let raw = codec::auto_read(block.value_data.as_slice(), idx as usize);
                self.remap_ordinal(block, raw)
            })
            .collect()
    }

    /// Iterate multi-values for a doc, calling `f` for each. Returns true if `f` ever returns true (short-circuit).
    /// Handles multi-block columns correctly by finding the right block.
    #[inline]
    pub fn for_each_multi_value(&self, doc_id: u32, mut f: impl FnMut(u64) -> bool) -> bool {
        let (bi, start, end) = self.block_value_range(doc_id);
        if start >= end {
            return false;
        }
        let block = &self.blocks[bi];
        for idx in start..end {
            let raw = codec::auto_read(block.value_data.as_slice(), idx as usize);
            if f(self.remap_ordinal(block, raw)) {
                return true;
            }
        }
        false
    }

    /// Batch-scan all values in a single-value column, calling `f(doc_id, raw_value)` for each.
    ///
    /// Uses `auto_read_batch` internally (one codec dispatch per block, not per value),
    /// enabling compiler auto-vectorization for byte-aligned bitpacked columns.
    /// For text columns, returned values are global ordinals (remapped).
    /// For multi-value columns, use `for_each_multi_value` instead.
    pub fn scan_single_values(&self, mut f: impl FnMut(u32, u64)) {
        if self.multi {
            return;
        }
        const BATCH: usize = 256;
        let mut buf = [0u64; BATCH];
        let needs_remap = self.column_type == FastFieldColumnType::TextOrdinal;

        for block in &self.blocks {
            let n = block.num_docs as usize;
            let mut pos = 0;
            while pos < n {
                let chunk = (n - pos).min(BATCH);
                codec::auto_read_batch(block.data.as_slice(), pos, &mut buf[..chunk]);

                if needs_remap && !block.ordinal_map.is_empty() {
                    for (i, &raw) in buf[..chunk].iter().enumerate() {
                        let val = if raw != FAST_FIELD_MISSING {
                            let idx = raw as usize;
                            if idx < block.ordinal_map.len() {
                                block.ordinal_map[idx] as u64
                            } else {
                                FAST_FIELD_MISSING
                            }
                        } else {
                            raw
                        };
                        f(block.cumulative_docs + pos as u32 + i as u32, val);
                    }
                } else {
                    for (i, &val) in buf[..chunk].iter().enumerate() {
                        f(block.cumulative_docs + pos as u32 + i as u32, val);
                    }
                }
                pos += chunk;
            }
        }
    }

    /// Check if this doc has a value (not [`FAST_FIELD_MISSING`]).
    ///
    /// For single-value columns, checks the raw sentinel.
    /// For multi-value columns, checks if the offset range is non-empty.
    #[inline]
    pub fn has_value(&self, doc_id: u32) -> bool {
        if !self.multi {
            return doc_id < self.num_docs && self.get_u64(doc_id) != FAST_FIELD_MISSING;
        }
        let (_, start, end) = self.block_value_range(doc_id);
        start < end
    }

    /// Get decoded i64 value (zigzag-decoded).
    ///
    /// Returns `i64::MIN` for absent docs (zigzag_decode of `FAST_FIELD_MISSING`).
    /// Use [`has_value`](Self::has_value) to distinguish absent from real values.
    #[inline]
    pub fn get_i64(&self, doc_id: u32) -> i64 {
        zigzag_decode(self.get_u64(doc_id))
    }

    /// Get decoded f64 value (sortable-decoded).
    ///
    /// Returns `NaN` for absent docs (`sortable_u64_to_f64(FAST_FIELD_MISSING)`).
    /// Use [`has_value`](Self::has_value) to distinguish absent from real values.
    #[inline]
    pub fn get_f64(&self, doc_id: u32) -> f64 {
        sortable_u64_to_f64(self.get_u64(doc_id))
    }

    /// Get the text ordinal for a doc_id. Returns FAST_FIELD_MISSING if missing.
    #[inline]
    pub fn get_ordinal(&self, doc_id: u32) -> u64 {
        self.get_u64(doc_id)
    }

    /// Get the text string for a doc_id (looks up ordinal in block-local dictionary).
    /// Returns None if the doc has no value or ordinal is missing.
    pub fn get_text(&self, doc_id: u32) -> Option<&str> {
        if doc_id >= self.num_docs {
            return None;
        }
        let (bi, local) = self.find_block(doc_id);
        let block = &self.blocks[bi];
        let raw_ordinal = if self.multi {
            let start = codec::auto_read(block.offset_data.as_slice(), local as usize) as u32;
            let end = codec::auto_read(block.offset_data.as_slice(), local as usize + 1) as u32;
            if start >= end {
                return None;
            }
            codec::auto_read(block.value_data.as_slice(), start as usize)
        } else {
            codec::auto_read(block.data.as_slice(), local as usize)
        };
        if raw_ordinal == FAST_FIELD_MISSING {
            return None;
        }
        block.dict.as_ref().and_then(|d| d.get(raw_ordinal as u32))
    }

    /// Look up text string → global ordinal. Returns None if not found.
    pub fn text_ordinal(&self, text: &str) -> Option<u64> {
        self.global_text_dict.as_ref().and_then(|d| d.ordinal(text))
    }

    /// Access the global text dictionary reader (if this is a text column).
    pub fn text_dict(&self) -> Option<&TextDictReader> {
        self.global_text_dict.as_ref()
    }

    /// Number of blocks in this column.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Access blocks for raw stacking during merge.
    pub fn blocks(&self) -> &[ColumnBlock] {
        &self.blocks
    }
}

// ── Text dictionary ───────────────────────────────────────────────────────

/// Sorted dictionary for text ordinal columns.
///
/// **Zero-copy**: the dictionary data is a shared slice of the `.fast` file.
/// An offset table (one u32 per entry) is built at open time so that
/// individual strings can be accessed without scanning.
pub struct TextDictReader {
    /// The raw dictionary bytes from the `.fast` file (zero-copy).
    data: OwnedBytes,
    /// Per-entry (offset, len) pairs into `data` — built at open time.
    offsets: Vec<(u32, u32)>,
}

impl TextDictReader {
    /// Open a zero-copy text dictionary from `file_data` starting at `dict_start`.
    pub fn open(file_data: &OwnedBytes, dict_start: usize, count: u32) -> io::Result<Self> {
        // First pass: scan len-prefixed entries to build offset table
        let dict_slice = file_data.as_slice();
        let mut pos = dict_start;
        let mut offsets = Vec::with_capacity(count as usize);

        for _ in 0..count {
            if pos + 4 > dict_slice.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "text dict truncated",
                ));
            }
            let len = u32::from_le_bytes([
                dict_slice[pos],
                dict_slice[pos + 1],
                dict_slice[pos + 2],
                dict_slice[pos + 3],
            ]) as usize;
            pos += 4;
            if pos + len > dict_slice.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "text dict entry truncated",
                ));
            }
            // Validate UTF-8 eagerly
            std::str::from_utf8(&dict_slice[pos..pos + len]).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("invalid utf8: {}", e))
            })?;
            offsets.push((pos as u32, len as u32));
            pos += len;
        }

        // Slice the full dict region (dict_start..pos) zero-copy
        let data = file_data.slice(dict_start..pos);

        // Adjust offsets to be relative to `data` (subtract dict_start)
        for entry in offsets.iter_mut() {
            entry.0 -= dict_start as u32;
        }

        Ok(Self { data, offsets })
    }

    /// Get string by ordinal — zero-copy borrow from the underlying file data.
    pub fn get(&self, ordinal: u32) -> Option<&str> {
        let &(off, len) = self.offsets.get(ordinal as usize)?;
        let slice = &self.data.as_slice()[off as usize..off as usize + len as usize];
        // Safety: we validated UTF-8 in open()
        Some(unsafe { std::str::from_utf8_unchecked(slice) })
    }

    /// Binary search for a string → ordinal.
    pub fn ordinal(&self, text: &str) -> Option<u64> {
        self.offsets
            .binary_search_by(|&(off, len)| {
                let slice = &self.data.as_slice()[off as usize..off as usize + len as usize];
                // Safety: validated UTF-8 in open()
                let entry = unsafe { std::str::from_utf8_unchecked(slice) };
                entry.cmp(text)
            })
            .ok()
            .map(|i| i as u64)
    }

    /// Number of entries in the dictionary.
    pub fn len(&self) -> u32 {
        self.offsets.len() as u32
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Iterate all entries.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.offsets.iter().map(|&(off, len)| {
            let slice = &self.data.as_slice()[off as usize..off as usize + len as usize];
            unsafe { std::str::from_utf8_unchecked(slice) }
        })
    }
}

// ── File-level write/read ─────────────────────────────────────────────────

/// Write fast-field TOC + footer.
pub fn write_fast_field_toc_and_footer(
    writer: &mut dyn Write,
    toc_offset: u64,
    entries: &[FastFieldTocEntry],
) -> io::Result<()> {
    for e in entries {
        e.write_to(writer)?;
    }
    writer.write_u64::<LittleEndian>(toc_offset)?;
    writer.write_u32::<LittleEndian>(entries.len() as u32)?;
    writer.write_u32::<LittleEndian>(FAST_FIELD_MAGIC)?;
    Ok(())
}

/// Read fast-field footer from the last 16 bytes.
/// Returns (toc_offset, num_columns).
pub fn read_fast_field_footer(file_data: &[u8]) -> io::Result<(u64, u32)> {
    let len = file_data.len();
    if len < FAST_FIELD_FOOTER_SIZE as usize {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "fast field file too small for footer",
        ));
    }
    let footer = &file_data[len - FAST_FIELD_FOOTER_SIZE as usize..];
    let mut cursor = std::io::Cursor::new(footer);
    let toc_offset = cursor.read_u64::<LittleEndian>()?;
    let num_columns = cursor.read_u32::<LittleEndian>()?;
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != FAST_FIELD_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad fast field magic: 0x{:08x}", magic),
        ));
    }
    Ok((toc_offset, num_columns))
}

/// Read all TOC entries from file data (FST2 format).
pub fn read_fast_field_toc(
    file_data: &[u8],
    toc_offset: u64,
    num_columns: u32,
) -> io::Result<Vec<FastFieldTocEntry>> {
    let start = toc_offset as usize;
    let expected = num_columns as usize * FAST_FIELD_TOC_ENTRY_SIZE;
    if start + expected > file_data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "fast field TOC out of bounds",
        ));
    }
    let mut cursor = std::io::Cursor::new(&file_data[start..start + expected]);
    let mut entries = Vec::with_capacity(num_columns as usize);
    for _ in 0..num_columns {
        entries.push(FastFieldTocEntry::read_from(&mut cursor)?);
    }
    Ok(entries)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        for v in [0i64, 1, -1, 42, -42, i64::MAX, i64::MIN] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }
    }

    #[test]
    fn test_f64_sortable_roundtrip() {
        for v in [0.0f64, 1.0, -1.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE] {
            assert_eq!(sortable_u64_to_f64(f64_to_sortable_u64(v)), v);
        }
    }

    #[test]
    fn test_f64_sortable_order() {
        let values = [-100.0f64, -1.0, -0.0, 0.0, 0.5, 1.0, 100.0];
        let encoded: Vec<u64> = values.iter().map(|&v| f64_to_sortable_u64(v)).collect();
        for i in 1..encoded.len() {
            assert!(
                encoded[i] >= encoded[i - 1],
                "{} >= {} failed for {} vs {}",
                encoded[i],
                encoded[i - 1],
                values[i],
                values[i - 1]
            );
        }
    }

    #[test]
    fn test_bitpack_roundtrip() {
        let values: Vec<u64> = vec![0, 3, 7, 15, 0, 1, 6, 12];
        let bpv = 4u8;
        let mut packed = Vec::new();
        bitpack_write(&values, bpv, &mut packed);

        for (i, &expected) in values.iter().enumerate() {
            let got = bitpack_read(&packed, bpv, i);
            assert_eq!(got, expected, "index {}", i);
        }
    }

    #[test]
    fn test_bitpack_high_bpv_regression() {
        // Regression: bpv > 56 with non-zero bit_shift used to read wrong bits
        // because the old 8-byte fast path didn't check bit_shift + bpv <= 64.
        for bpv in [57u8, 58, 59, 60, 63, 64] {
            let max_val = if bpv == 64 {
                u64::MAX
            } else {
                (1u64 << bpv) - 1
            };
            let values: Vec<u64> = (0..32)
                .map(|i: u64| {
                    if max_val == u64::MAX {
                        i * 7
                    } else {
                        (i * 7) % (max_val + 1)
                    }
                })
                .collect();
            let mut packed = Vec::new();
            bitpack_write(&values, bpv, &mut packed);
            for (i, &expected) in values.iter().enumerate() {
                let got = bitpack_read(&packed, bpv, i);
                assert_eq!(got, expected, "high bpv={} index={}", bpv, i);
            }
        }
    }

    #[test]
    fn test_bitpack_various_widths() {
        for bpv in [1u8, 2, 3, 5, 7, 8, 13, 16, 32, 64] {
            let max_val = if bpv == 64 {
                u64::MAX
            } else {
                (1u64 << bpv) - 1
            };
            let values: Vec<u64> = (0..100)
                .map(|i: u64| {
                    if max_val == u64::MAX {
                        i
                    } else {
                        i % (max_val + 1)
                    }
                })
                .collect();
            let mut packed = Vec::new();
            bitpack_write(&values, bpv, &mut packed);

            for (i, &expected) in values.iter().enumerate() {
                let got = bitpack_read(&packed, bpv, i);
                assert_eq!(got, expected, "bpv={} index={}", bpv, i);
            }
        }
    }

    /// Helper: wrap a Vec<u8> in OwnedBytes for tests.
    fn owned(buf: Vec<u8>) -> OwnedBytes {
        OwnedBytes::new(buf)
    }

    #[test]
    fn test_writer_reader_u64_roundtrip() {
        let mut writer = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        writer.add_u64(0, 100);
        writer.add_u64(1, 200);
        writer.add_u64(2, 150);
        writer.add_u64(4, 300); // gap at doc_id=3
        writer.pad_to(5);

        let mut buf = Vec::new();
        let (mut toc, _bytes) = writer.serialize(&mut buf, 0).unwrap();
        toc.field_id = 42;

        // Write TOC + footer
        let toc_offset = buf.len() as u64;
        write_fast_field_toc_and_footer(&mut buf, toc_offset, &[toc]).unwrap();

        // Read back
        let ob = owned(buf);
        let (toc_off, num_cols) = read_fast_field_footer(&ob).unwrap();
        assert_eq!(num_cols, 1);
        let tocs = read_fast_field_toc(&ob, toc_off, num_cols).unwrap();
        assert_eq!(tocs.len(), 1);
        assert_eq!(tocs[0].field_id, 42);

        let reader = FastFieldReader::open(&ob, &tocs[0]).unwrap();
        assert_eq!(reader.get_u64(0), 100);
        assert_eq!(reader.get_u64(1), 200);
        assert_eq!(reader.get_u64(2), 150);
        assert_eq!(reader.get_u64(3), FAST_FIELD_MISSING); // gap → absent sentinel
        assert_eq!(reader.get_u64(4), 300);
    }

    #[test]
    fn test_writer_reader_i64_roundtrip() {
        let mut writer = FastFieldWriter::new_numeric(FastFieldColumnType::I64);
        writer.add_i64(0, -100);
        writer.add_i64(1, 50);
        writer.add_i64(2, 0);
        writer.pad_to(3);

        let mut buf = Vec::new();
        let (toc, _) = writer.serialize(&mut buf, 0).unwrap();
        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();
        assert_eq!(reader.get_i64(0), -100);
        assert_eq!(reader.get_i64(1), 50);
        assert_eq!(reader.get_i64(2), 0);
    }

    #[test]
    fn test_writer_reader_f64_roundtrip() {
        let mut writer = FastFieldWriter::new_numeric(FastFieldColumnType::F64);
        writer.add_f64(0, -1.5);
        writer.add_f64(1, 3.15);
        writer.add_f64(2, 0.0);
        writer.pad_to(3);

        let mut buf = Vec::new();
        let (toc, _) = writer.serialize(&mut buf, 0).unwrap();
        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();
        assert_eq!(reader.get_f64(0), -1.5);
        assert_eq!(reader.get_f64(1), 3.15);
        assert_eq!(reader.get_f64(2), 0.0);
    }

    #[test]
    fn test_writer_reader_text_roundtrip() {
        let mut writer = FastFieldWriter::new_text();
        writer.add_text(0, "banana");
        writer.add_text(1, "apple");
        writer.add_text(2, "cherry");
        writer.add_text(3, "apple"); // duplicate
        // doc_id=4 has no value
        writer.pad_to(5);

        let mut buf = Vec::new();
        let (toc, _) = writer.serialize(&mut buf, 0).unwrap();
        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();

        // Dictionary is sorted: apple=0, banana=1, cherry=2
        assert_eq!(reader.get_text(0), Some("banana"));
        assert_eq!(reader.get_text(1), Some("apple"));
        assert_eq!(reader.get_text(2), Some("cherry"));
        assert_eq!(reader.get_text(3), Some("apple"));
        assert_eq!(reader.get_text(4), None); // missing

        // Ordinal lookups
        assert_eq!(reader.text_ordinal("apple"), Some(0));
        assert_eq!(reader.text_ordinal("banana"), Some(1));
        assert_eq!(reader.text_ordinal("cherry"), Some(2));
        assert_eq!(reader.text_ordinal("durian"), None);
    }

    #[test]
    fn test_constant_column() {
        let mut writer = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        for i in 0..100 {
            writer.add_u64(i, 42);
        }

        let mut buf = Vec::new();
        let (toc, _) = writer.serialize(&mut buf, 0).unwrap();

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();
        for i in 0..100 {
            assert_eq!(reader.get_u64(i), 42);
        }
    }

    // ── Multi-value tests ──

    #[test]
    fn test_multi_value_u64_roundtrip() {
        let mut writer = FastFieldWriter::new_numeric_multi(FastFieldColumnType::U64);
        // doc 0: [10, 20, 30]
        writer.add_u64(0, 10);
        writer.add_u64(0, 20);
        writer.add_u64(0, 30);
        // doc 1: [] (empty)
        // doc 2: [100]
        writer.add_u64(2, 100);
        // doc 3: [5, 15]
        writer.add_u64(3, 5);
        writer.add_u64(3, 15);
        writer.pad_to(4);

        let mut buf = Vec::new();
        let (toc, _) = writer.serialize(&mut buf, 0).unwrap();
        assert!(toc.multi);
        assert_eq!(toc.num_docs, 4);

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();
        assert!(reader.multi);

        // doc 0: first value
        assert_eq!(reader.get_u64(0), 10);
        let (s, e) = reader.value_range(0);
        assert_eq!(e - s, 3);
        assert_eq!(reader.get_value_at(s), 10);
        assert_eq!(reader.get_value_at(s + 1), 20);
        assert_eq!(reader.get_value_at(s + 2), 30);

        // doc 1: empty → sentinel
        assert_eq!(reader.get_u64(1), FAST_FIELD_MISSING);
        let (s, e) = reader.value_range(1);
        assert_eq!(s, e);
        assert!(!reader.has_value(1));

        // doc 2: [100]
        assert_eq!(reader.get_u64(2), 100);
        assert!(reader.has_value(2));

        // doc 3: [5, 15]
        assert_eq!(reader.get_u64(3), 5);
        let (s, e) = reader.value_range(3);
        assert_eq!(e - s, 2);
        assert_eq!(reader.get_value_at(s), 5);
        assert_eq!(reader.get_value_at(s + 1), 15);
    }

    #[test]
    fn test_multi_value_text_roundtrip() {
        let mut writer = FastFieldWriter::new_text_multi();
        // doc 0: ["banana", "apple"]
        writer.add_text(0, "banana");
        writer.add_text(0, "apple");
        // doc 1: ["cherry"]
        writer.add_text(1, "cherry");
        // doc 2: [] empty
        writer.pad_to(3);

        let mut buf = Vec::new();
        let (toc, _) = writer.serialize(&mut buf, 0).unwrap();
        assert!(toc.multi);

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();

        // doc 0: first value ordinal → banana is ordinal 1 (apple=0, banana=1, cherry=2)
        let (s, e) = reader.value_range(0);
        assert_eq!(e - s, 2);
        let ord0 = reader.get_value_at(s);
        let ord1 = reader.get_value_at(s + 1);
        assert_eq!(reader.text_dict().unwrap().get(ord0 as u32), Some("banana"));
        assert_eq!(reader.text_dict().unwrap().get(ord1 as u32), Some("apple"));

        // doc 1: cherry
        let (s, e) = reader.value_range(1);
        assert_eq!(e - s, 1);
        let ord = reader.get_value_at(s);
        assert_eq!(reader.text_dict().unwrap().get(ord as u32), Some("cherry"));

        // doc 2: empty
        assert!(!reader.has_value(2));
    }

    #[test]
    fn test_multi_value_full_toc_roundtrip() {
        let mut writer = FastFieldWriter::new_numeric_multi(FastFieldColumnType::U64);
        writer.add_u64(0, 1);
        writer.add_u64(0, 2);
        writer.add_u64(1, 3);
        writer.pad_to(2);

        let mut buf = Vec::new();
        let (mut toc, _) = writer.serialize(&mut buf, 0).unwrap();
        toc.field_id = 7;

        let toc_offset = buf.len() as u64;
        write_fast_field_toc_and_footer(&mut buf, toc_offset, &[toc]).unwrap();

        let ob = owned(buf);
        let (toc_off, num_cols) = read_fast_field_footer(&ob).unwrap();
        let tocs = read_fast_field_toc(&ob, toc_off, num_cols).unwrap();
        assert_eq!(tocs[0].field_id, 7);
        assert!(tocs[0].multi);

        let reader = FastFieldReader::open(&ob, &tocs[0]).unwrap();
        assert_eq!(reader.get_u64(0), 1);
        assert_eq!(reader.get_u64(1), 3);
    }

    /// Helper: serialize a writer into a blocked column, return (block_data, block_dict, block_index_entry)
    /// by stripping the blocked header.
    fn serialize_single_block(writer: &mut FastFieldWriter) -> (Vec<u8>, Vec<u8>, BlockIndexEntry) {
        let mut buf = Vec::new();
        let (_toc, _) = writer.serialize(&mut buf, 0).unwrap();
        // Strip: [num_blocks(4)] [BlockIndexEntry(16)] [data...] [dict...]
        let mut cursor = std::io::Cursor::new(&buf[4..4 + BLOCK_INDEX_ENTRY_SIZE]);
        let entry = BlockIndexEntry::read_from(&mut cursor).unwrap();
        let data_start = 4 + BLOCK_INDEX_ENTRY_SIZE;
        let data_end = data_start + entry.data_len as usize;
        let dict_end = data_end + entry.dict_len as usize;
        let data = buf[data_start..data_end].to_vec();
        let dict = if dict_end > data_end {
            buf[data_end..dict_end].to_vec()
        } else {
            Vec::new()
        };
        (data, dict, entry)
    }

    /// Manually assemble a multi-block column from individual block payloads.
    fn assemble_blocked_column(
        field_id: u32,
        column_type: FastFieldColumnType,
        multi: bool,
        blocks: &[(u32, &[u8], u32, &[u8])], // (num_docs, data, dict_count, dict)
    ) -> (Vec<u8>, FastFieldTocEntry) {
        use byteorder::{LittleEndian, WriteBytesExt};

        let mut buf = Vec::new();
        let num_blocks = blocks.len() as u32;

        // num_blocks
        buf.write_u32::<LittleEndian>(num_blocks).unwrap();

        // block index
        for &(num_docs, data, dict_count, dict) in blocks {
            let entry = BlockIndexEntry {
                num_docs,
                data_len: data.len() as u32,
                dict_count,
                dict_len: dict.len() as u32,
            };
            entry.write_to(&mut buf).unwrap();
        }

        // block data + dicts
        let mut total_docs = 0u32;
        for &(num_docs, data, _, dict) in blocks {
            buf.extend_from_slice(data);
            buf.extend_from_slice(dict);
            total_docs += num_docs;
        }

        let data_len = buf.len() as u64;

        // Write TOC + footer
        let toc = FastFieldTocEntry {
            field_id,
            column_type,
            multi,
            data_offset: 0,
            data_len,
            num_docs: total_docs,
            dict_offset: 0,
            dict_count: 0,
        };

        let toc_offset = buf.len() as u64;
        write_fast_field_toc_and_footer(&mut buf, toc_offset, std::slice::from_ref(&toc)).unwrap();

        (buf, toc)
    }

    #[test]
    fn test_multi_block_numeric_roundtrip() {
        // Block A: 3 docs [10, 20, 30]
        let mut wa = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        wa.add_u64(0, 10);
        wa.add_u64(1, 20);
        wa.add_u64(2, 30);
        let (data_a, dict_a, entry_a) = serialize_single_block(&mut wa);

        // Block B: 2 docs [40, 50]
        let mut wb = FastFieldWriter::new_numeric(FastFieldColumnType::U64);
        wb.add_u64(0, 40);
        wb.add_u64(1, 50);
        let (data_b, dict_b, entry_b) = serialize_single_block(&mut wb);

        let (buf, toc) = assemble_blocked_column(
            1,
            FastFieldColumnType::U64,
            false,
            &[
                (entry_a.num_docs, &data_a, entry_a.dict_count, &dict_a),
                (entry_b.num_docs, &data_b, entry_b.dict_count, &dict_b),
            ],
        );

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();

        assert_eq!(reader.num_docs, 5);
        assert_eq!(reader.num_blocks(), 2);
        assert_eq!(reader.get_u64(0), 10);
        assert_eq!(reader.get_u64(1), 20);
        assert_eq!(reader.get_u64(2), 30);
        assert_eq!(reader.get_u64(3), 40);
        assert_eq!(reader.get_u64(4), 50);
    }

    #[test]
    fn test_multi_block_text_roundtrip() {
        // Block A: 2 docs ["alpha", "beta"]
        let mut wa = FastFieldWriter::new_text();
        wa.add_text(0, "alpha");
        wa.add_text(1, "beta");
        let (data_a, dict_a, entry_a) = serialize_single_block(&mut wa);

        // Block B: 2 docs ["gamma", "alpha"]  (alpha shared with block A)
        let mut wb = FastFieldWriter::new_text();
        wb.add_text(0, "gamma");
        wb.add_text(1, "alpha");
        let (data_b, dict_b, entry_b) = serialize_single_block(&mut wb);

        let (buf, toc) = assemble_blocked_column(
            2,
            FastFieldColumnType::TextOrdinal,
            false,
            &[
                (entry_a.num_docs, &data_a, entry_a.dict_count, &dict_a),
                (entry_b.num_docs, &data_b, entry_b.dict_count, &dict_b),
            ],
        );

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();

        assert_eq!(reader.num_docs, 4);
        assert_eq!(reader.num_blocks(), 2);

        // Global dict should be: alpha(0), beta(1), gamma(2)
        assert_eq!(reader.text_dict().unwrap().len(), 3);

        // Block A: alpha=local0→global0, beta=local1→global1
        assert_eq!(reader.get_text(0), Some("alpha"));
        assert_eq!(reader.get_text(1), Some("beta"));

        // Block B: gamma=local1→global2, alpha=local0→global0
        assert_eq!(reader.get_text(2), Some("gamma"));
        assert_eq!(reader.get_text(3), Some("alpha"));

        // Global ordinal lookups
        assert_eq!(reader.text_ordinal("alpha"), Some(0));
        assert_eq!(reader.text_ordinal("beta"), Some(1));
        assert_eq!(reader.text_ordinal("gamma"), Some(2));

        // get_u64 returns global ordinals
        assert_eq!(reader.get_u64(0), 0); // alpha
        assert_eq!(reader.get_u64(1), 1); // beta
        assert_eq!(reader.get_u64(2), 2); // gamma
        assert_eq!(reader.get_u64(3), 0); // alpha
    }

    #[test]
    fn test_multi_block_multi_value_numeric() {
        // Block A: doc0=[1,2], doc1=[3]
        let mut wa = FastFieldWriter::new_numeric_multi(FastFieldColumnType::U64);
        wa.add_u64(0, 1);
        wa.add_u64(0, 2);
        wa.add_u64(1, 3);
        wa.pad_to(2);
        let (data_a, dict_a, entry_a) = serialize_single_block(&mut wa);

        // Block B: doc0=[4,5,6], doc1=[]
        let mut wb = FastFieldWriter::new_numeric_multi(FastFieldColumnType::U64);
        wb.add_u64(0, 4);
        wb.add_u64(0, 5);
        wb.add_u64(0, 6);
        wb.pad_to(2);
        let (data_b, dict_b, entry_b) = serialize_single_block(&mut wb);

        let (buf, toc) = assemble_blocked_column(
            3,
            FastFieldColumnType::U64,
            true,
            &[
                (entry_a.num_docs, &data_a, entry_a.dict_count, &dict_a),
                (entry_b.num_docs, &data_b, entry_b.dict_count, &dict_b),
            ],
        );

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();

        assert_eq!(reader.num_docs, 4);
        assert_eq!(reader.num_blocks(), 2);

        // doc0 (block A): [1, 2]
        assert_eq!(reader.get_multi_values(0), vec![1, 2]);
        // doc1 (block A): [3]
        assert_eq!(reader.get_multi_values(1), vec![3]);
        // doc2 (block B, local 0): [4, 5, 6]
        assert_eq!(reader.get_multi_values(2), vec![4, 5, 6]);
        // doc3 (block B, local 1): []
        assert_eq!(reader.get_multi_values(3), Vec::<u64>::new());
    }
}
