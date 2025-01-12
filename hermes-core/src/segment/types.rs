//! Segment types and metadata

use std::io::{self, Cursor};
use std::path::PathBuf;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rustc_hash::FxHashMap;

use crate::dsl::Field;

/// Unique segment identifier (UUID7-like: 48-bit timestamp + 80-bit random)
///
/// Stored as u128 internally for full 128-bit support.
/// Format: [48-bit timestamp ms][80-bit random]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentId(pub u128);

impl SegmentId {
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        // UUID7-like: 48 bits timestamp (ms) + 80 bits random
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let random_bits: u128 =
            ((rand::random::<u64>() as u128) << 16) | (rand::random::<u16>() as u128);

        // Combine: timestamp in upper 48 bits, random in lower 80 bits
        Self((timestamp_ms << 80) | random_bits)
    }

    pub fn from_u128(id: u128) -> Self {
        Self(id)
    }

    /// For backwards compatibility with u64-based IDs
    pub fn from_u64(id: u64) -> Self {
        Self(id as u128)
    }

    /// Create from hex string (32 chars)
    pub fn from_hex(s: &str) -> Option<Self> {
        u128::from_str_radix(s, 16).ok().map(Self)
    }

    /// Convert to hex string (32 chars, zero-padded)
    pub fn to_hex(&self) -> String {
        format!("{:032x}", self.0)
    }
}

impl Default for SegmentId {
    fn default() -> Self {
        Self::new()
    }
}

/// Field statistics for BM25F scoring
#[derive(Debug, Clone, Default)]
pub struct FieldStats {
    /// Total number of tokens across all documents for this field
    pub total_tokens: u64,
    /// Number of documents that have this field
    pub doc_count: u32,
}

impl FieldStats {
    /// Average field length (tokens per document)
    pub fn avg_field_len(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.doc_count as f32
        }
    }
}

/// Segment metadata
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    pub id: u128,
    pub num_docs: u32,
    /// Per-field statistics for BM25F scoring (field_id -> stats)
    pub field_stats: FxHashMap<u32, FieldStats>,
}

impl SegmentMeta {
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.write_u128::<LittleEndian>(self.id)?;
        buf.write_u32::<LittleEndian>(self.num_docs)?;

        // Write field stats
        buf.write_u32::<LittleEndian>(self.field_stats.len() as u32)?;
        for (&field_id, stats) in &self.field_stats {
            buf.write_u32::<LittleEndian>(field_id)?;
            buf.write_u64::<LittleEndian>(stats.total_tokens)?;
            buf.write_u32::<LittleEndian>(stats.doc_count)?;
        }

        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut reader = Cursor::new(data);
        let id = reader.read_u128::<LittleEndian>()?;
        let num_docs = reader.read_u32::<LittleEndian>()?;

        // Read field stats (handle legacy format without field stats)
        let mut field_stats = FxHashMap::default();
        if reader.position() < data.len() as u64 {
            let num_fields = reader.read_u32::<LittleEndian>()?;
            for _ in 0..num_fields {
                let field_id = reader.read_u32::<LittleEndian>()?;
                let total_tokens = reader.read_u64::<LittleEndian>()?;
                let doc_count = reader.read_u32::<LittleEndian>()?;
                field_stats.insert(
                    field_id,
                    FieldStats {
                        total_tokens,
                        doc_count,
                    },
                );
            }
        }

        Ok(Self {
            id,
            num_docs,
            field_stats,
        })
    }

    /// Get average field length for a field
    pub fn avg_field_len(&self, field: Field) -> f32 {
        self.field_stats
            .get(&field.0)
            .map(|s| s.avg_field_len())
            .unwrap_or(0.0)
    }
}

/// Paths for segment files
pub struct SegmentFiles {
    pub term_dict: PathBuf,
    pub postings: PathBuf,
    pub store: PathBuf,
    pub meta: PathBuf,
}

impl SegmentFiles {
    pub fn new(segment_id: u128) -> Self {
        let prefix = format!("seg_{:032x}", segment_id);
        Self {
            term_dict: PathBuf::from(format!("{}.terms", prefix)),
            postings: PathBuf::from(format!("{}.post", prefix)),
            store: PathBuf::from(format!("{}.store", prefix)),
            meta: PathBuf::from(format!("{}.meta", prefix)),
        }
    }
}
