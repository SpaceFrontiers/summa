//! Fast field columnar storage for efficient filtering and sorting.
//!
//! Stores one column per fast-field, indexed by doc_id for O(1) access.
//! Supports u64, i64, f64, and text (dictionary-encoded ordinal) columns.
//!
//! ## File format (`.fast`)
//!
//! ```text
//! [column 0 bit-packed data] [column 1 data] ... [column N data]
//! [dict 0: len-prefixed sorted strings] [dict 1] ...  (text columns only)
//! [TOC: FastFieldTocEntry × num_columns]
//! [footer: toc_offset(8) + num_columns(4) + magic(4)]  = 16 bytes
//! ```

use std::collections::BTreeMap;
use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// ── Constants ─────────────────────────────────────────────────────────────

/// Magic number for `.fast` file footer ("FST1" in LE)
pub const FAST_FIELD_MAGIC: u32 = 0x31545346;

/// Footer size: toc_offset(8) + num_columns(4) + magic(4) = 16
pub const FAST_FIELD_FOOTER_SIZE: u64 = 16;

/// Sentinel ordinal for missing text values
pub const TEXT_MISSING_ORDINAL: u64 = u64::MAX;

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

    // Fast path: if we can read 8 bytes starting at byte_offset, do a single
    // unaligned LE u64 load, shift, and mask. Covers bpv ≤ 56 (56+7=63 < 64).
    if byte_offset + 8 <= data.len() {
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

/// On-disk TOC entry for a fast-field column.
///
/// Wire: field_id(4) + column_type(1) + data_offset(8) + num_docs(4) +
///       min_value(8) + bits_per_value(1) + dict_offset(8) + dict_count(4) = 38 bytes
#[derive(Debug, Clone)]
pub struct FastFieldTocEntry {
    pub field_id: u32,
    pub column_type: FastFieldColumnType,
    pub data_offset: u64,
    pub num_docs: u32,
    pub min_value: u64,
    pub bits_per_value: u8,
    /// Byte offset of the text dictionary section (0 for numeric columns).
    pub dict_offset: u64,
    /// Number of entries in the text dictionary (0 for numeric columns).
    pub dict_count: u32,
}

pub const FAST_FIELD_TOC_ENTRY_SIZE: usize = 4 + 1 + 8 + 4 + 8 + 1 + 8 + 4; // 38

impl FastFieldTocEntry {
    pub fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_u32::<LittleEndian>(self.field_id)?;
        w.write_u8(self.column_type as u8)?;
        w.write_u64::<LittleEndian>(self.data_offset)?;
        w.write_u32::<LittleEndian>(self.num_docs)?;
        w.write_u64::<LittleEndian>(self.min_value)?;
        w.write_u8(self.bits_per_value)?;
        w.write_u64::<LittleEndian>(self.dict_offset)?;
        w.write_u32::<LittleEndian>(self.dict_count)?;
        Ok(())
    }

    pub fn read_from(r: &mut dyn Read) -> io::Result<Self> {
        let field_id = r.read_u32::<LittleEndian>()?;
        let ct = r.read_u8()?;
        let column_type = FastFieldColumnType::from_u8(ct)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad column type"))?;
        let data_offset = r.read_u64::<LittleEndian>()?;
        let num_docs = r.read_u32::<LittleEndian>()?;
        let min_value = r.read_u64::<LittleEndian>()?;
        let bits_per_value = r.read_u8()?;
        let dict_offset = r.read_u64::<LittleEndian>()?;
        let dict_count = r.read_u32::<LittleEndian>()?;
        Ok(Self {
            field_id,
            column_type,
            data_offset,
            num_docs,
            min_value,
            bits_per_value,
            dict_offset,
            dict_count,
        })
    }
}

// ── Writer ────────────────────────────────────────────────────────────────

/// Collects values during indexing and serializes a single fast-field column.
pub struct FastFieldWriter {
    pub column_type: FastFieldColumnType,
    /// Raw u64 values indexed by local doc_id.
    /// For text: stores temporary placeholder 0, replaced by ordinals at build time.
    values: Vec<u64>,
    /// For TextOrdinal: maps original string → insertion order.
    /// Sorted dictionary is built at serialization time.
    text_values: Option<BTreeMap<String, u32>>,
    /// For TextOrdinal: per-doc string values (parallel to `values`).
    text_per_doc: Option<Vec<Option<String>>>,
}

impl FastFieldWriter {
    /// Create a writer for a numeric column (u64/i64/f64).
    pub fn new_numeric(column_type: FastFieldColumnType) -> Self {
        debug_assert!(matches!(
            column_type,
            FastFieldColumnType::U64 | FastFieldColumnType::I64 | FastFieldColumnType::F64
        ));
        Self {
            column_type,
            values: Vec::new(),
            text_values: None,
            text_per_doc: None,
        }
    }

    /// Create a writer for a text ordinal column.
    pub fn new_text() -> Self {
        Self {
            column_type: FastFieldColumnType::TextOrdinal,
            values: Vec::new(),
            text_values: Some(BTreeMap::new()),
            text_per_doc: Some(Vec::new()),
        }
    }

    /// Record a numeric value for `doc_id`. Fills gaps with 0.
    pub fn add_u64(&mut self, doc_id: u32, value: u64) {
        let idx = doc_id as usize;
        if idx >= self.values.len() {
            self.values.resize(idx + 1, 0);
            if let Some(ref mut tpd) = self.text_per_doc {
                tpd.resize(idx + 1, None);
            }
        }
        self.values[idx] = value;
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
        let idx = doc_id as usize;
        if idx >= self.values.len() {
            self.values.resize(idx + 1, 0);
        }
        if let Some(ref mut tpd) = self.text_per_doc {
            if idx >= tpd.len() {
                tpd.resize(idx + 1, None);
            }
            tpd[idx] = Some(value.to_string());
        }
        if let Some(ref mut dict) = self.text_values {
            let next_id = dict.len() as u32;
            dict.entry(value.to_string()).or_insert(next_id);
        }
    }

    /// Ensure the values array covers `num_docs` entries.
    pub fn pad_to(&mut self, num_docs: u32) {
        let n = num_docs as usize;
        if self.values.len() < n {
            self.values.resize(n, 0);
            if let Some(ref mut tpd) = self.text_per_doc {
                tpd.resize(n, None);
            }
        }
    }

    /// Number of documents in this column.
    pub fn num_docs(&self) -> u32 {
        self.values.len() as u32
    }

    /// Serialize column data to writer. Returns (data_offset_start, data_size).
    ///
    /// For text columns, also writes dictionary after the column data.
    /// Returns `(toc_entry_without_offsets, dict_bytes_written)`.
    pub fn serialize(
        &mut self,
        writer: &mut dyn Write,
        data_offset: u64,
    ) -> io::Result<(FastFieldTocEntry, u64)> {
        let num_docs = self.values.len() as u32;

        // For text ordinal: resolve per-doc strings to sorted ordinals
        if self.column_type == FastFieldColumnType::TextOrdinal {
            self.resolve_text_ordinals();
        }

        // Compute min/max for bit-packing
        let (min_value, bits_per_value) = if self.values.is_empty() {
            (0u64, 0u8)
        } else {
            let min_val = *self.values.iter().min().unwrap();
            let max_val = *self.values.iter().max().unwrap();
            let range = max_val - min_val;
            (min_val, bits_needed_u64(range))
        };

        // Bit-pack values (min-subtracted)
        let mut packed = Vec::new();
        if bits_per_value > 0 {
            let shifted: Vec<u64> = self.values.iter().map(|&v| v - min_value).collect();
            bitpack_write(&shifted, bits_per_value, &mut packed);
        }

        // Write column data
        writer.write_all(&packed)?;
        let data_end = data_offset + packed.len() as u64;

        // Write text dictionary (if applicable)
        let (dict_offset, dict_count, dict_bytes) =
            if self.column_type == FastFieldColumnType::TextOrdinal {
                let dict_off = data_end;
                let (count, bytes) = self.write_text_dictionary(writer)?;
                (dict_off, count, bytes)
            } else {
                (0u64, 0u32, 0u64)
            };

        let toc = FastFieldTocEntry {
            field_id: 0, // set by caller
            column_type: self.column_type,
            data_offset,
            num_docs,
            min_value,
            bits_per_value,
            dict_offset,
            dict_count,
        };

        Ok((toc, packed.len() as u64 + dict_bytes))
    }

    /// Resolve text per-doc values to sorted ordinals.
    fn resolve_text_ordinals(&mut self) {
        let dict = self.text_values.as_ref().expect("text_values required");
        let tpd = self.text_per_doc.as_ref().expect("text_per_doc required");

        // Build sorted ordinal map: BTreeMap iterates in sorted order
        let sorted_ordinals: BTreeMap<&str, u64> = dict
            .keys()
            .enumerate()
            .map(|(ord, key)| (key.as_str(), ord as u64))
            .collect();

        // Assign ordinals to values
        for (i, doc_text) in tpd.iter().enumerate() {
            match doc_text {
                Some(text) => {
                    self.values[i] = sorted_ordinals[text.as_str()];
                }
                None => {
                    self.values[i] = TEXT_MISSING_ORDINAL;
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

/// Reads a single fast-field column from mmap/buffer with O(1) doc_id access.
///
/// **Zero-copy**: column data and text dictionary are borrowed slices of the
/// underlying mmap / `OwnedBytes`; no heap allocation per column.
pub struct FastFieldReader {
    pub column_type: FastFieldColumnType,
    pub num_docs: u32,
    min_value: u64,
    bits_per_value: u8,
    /// Bit-packed column data — zero-copy slice of the `.fast` file.
    data: OwnedBytes,
    /// Text dictionary: sorted strings (only for TextOrdinal).
    text_dict: Option<TextDictReader>,
}

impl FastFieldReader {
    /// Open a column from an `OwnedBytes` file buffer using a TOC entry.
    /// Zero-copy: the returned reader borrows slices of `file_data`.
    pub fn open(file_data: &OwnedBytes, toc: &FastFieldTocEntry) -> io::Result<Self> {
        let data_start = toc.data_offset as usize;
        let bpv = toc.bits_per_value as usize;
        let data_len = if bpv == 0 {
            0
        } else {
            (toc.num_docs as usize * bpv).div_ceil(8)
        };
        let data_end = data_start + data_len;

        if data_end > file_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "fast field data out of bounds",
            ));
        }

        // Zero-copy slice — shares the underlying mmap/Arc<Vec>
        let data = file_data.slice(data_start..data_end);

        let text_dict = if toc.column_type == FastFieldColumnType::TextOrdinal && toc.dict_count > 0
        {
            let dict_start = toc.dict_offset as usize;
            Some(TextDictReader::open(file_data, dict_start, toc.dict_count)?)
        } else {
            None
        };

        Ok(Self {
            column_type: toc.column_type,
            num_docs: toc.num_docs,
            min_value: toc.min_value,
            bits_per_value: toc.bits_per_value,
            data,
            text_dict,
        })
    }

    /// Get raw u64 value for a doc_id. Returns 0/sentinel for out-of-range.
    #[inline]
    pub fn get_u64(&self, doc_id: u32) -> u64 {
        if doc_id >= self.num_docs {
            return 0;
        }
        if self.bits_per_value == 0 {
            return self.min_value;
        }
        let raw = bitpack_read(self.data.as_slice(), self.bits_per_value, doc_id as usize);
        raw + self.min_value
    }

    /// Get decoded i64 value (zigzag-decoded).
    #[inline]
    pub fn get_i64(&self, doc_id: u32) -> i64 {
        zigzag_decode(self.get_u64(doc_id))
    }

    /// Get decoded f64 value (sortable-decoded).
    #[inline]
    pub fn get_f64(&self, doc_id: u32) -> f64 {
        sortable_u64_to_f64(self.get_u64(doc_id))
    }

    /// Get the text ordinal for a doc_id. Returns TEXT_MISSING_ORDINAL if missing.
    #[inline]
    pub fn get_ordinal(&self, doc_id: u32) -> u64 {
        self.get_u64(doc_id)
    }

    /// Get the text string for a doc_id (looks up ordinal in dictionary).
    /// Returns None if the doc has no value or ordinal is missing.
    pub fn get_text(&self, doc_id: u32) -> Option<&str> {
        let ordinal = self.get_ordinal(doc_id);
        if ordinal == TEXT_MISSING_ORDINAL {
            return None;
        }
        self.text_dict.as_ref().and_then(|d| d.get(ordinal as u32))
    }

    /// Look up text string → ordinal. Returns None if not found.
    pub fn text_ordinal(&self, text: &str) -> Option<u64> {
        self.text_dict.as_ref().and_then(|d| d.ordinal(text))
    }

    /// Access the text dictionary reader (if this is a text column).
    pub fn text_dict(&self) -> Option<&TextDictReader> {
        self.text_dict.as_ref()
    }

    /// Min value stored in this column (before encoding).
    pub fn min_value(&self) -> u64 {
        self.min_value
    }

    /// Bits per value in this column.
    pub fn bits_per_value(&self) -> u8 {
        self.bits_per_value
    }

    /// Raw packed column bytes — for stackable merge (zero-copy).
    pub fn raw_data(&self) -> &OwnedBytes {
        &self.data
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

/// Read all TOC entries from file data.
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
        assert_eq!(reader.get_u64(3), 0); // gap filled with 0
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
        assert_eq!(toc.bits_per_value, 0); // constant — no data bytes
        assert_eq!(toc.min_value, 42);

        let ob = owned(buf);
        let reader = FastFieldReader::open(&ob, &toc).unwrap();
        for i in 0..100 {
            assert_eq!(reader.get_u64(i), 42);
        }
    }
}
