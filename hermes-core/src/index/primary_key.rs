//! Primary key deduplication index.
//!
//! Uses a bloom filter + `FxHashSet` for uncommitted keys to reject duplicates
//! at `add_document()` time. Committed keys are checked via fast-field
//! `TextDictReader::ordinal()` (binary search, O(log n)).
//!
//! The bloom filter is persisted to `pk_bloom.bin` so that restarts don't need
//! to re-iterate every committed key. On load, only keys from segments that
//! appeared since the last persist are iterated.

use std::collections::HashSet;

use byteorder::{LittleEndian, WriteBytesExt};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::dsl::Field;
use crate::error::{Error, Result};
use crate::segment::SegmentSnapshot;
use crate::structures::BloomFilter;

/// Bloom filter sizing: 10 bits/key ≈ 1% false positive rate.
const BLOOM_BITS_PER_KEY: usize = 10;

/// Extra capacity added to bloom filter beyond known keys.
const BLOOM_HEADROOM: usize = 100_000;

/// File name for the persisted primary-key bloom filter.
pub const PK_BLOOM_FILE: &str = "pk_bloom.bin";

/// Magic bytes for the persisted bloom file.
const PK_BLOOM_MAGIC: u32 = 0x504B424C; // "PKBL"

/// Lightweight per-segment data for primary key lookups.
///
/// Only holds fast-field readers (text dictionaries), not full `SegmentReader`s.
/// This avoids loading DimensionTables, SSTable FSTs, bloom filters, etc.
pub struct PkSegmentData {
    pub segment_id: String,
    pub fast_fields: FxHashMap<u32, crate::structures::fast_field::FastFieldReader>,
}

/// Thread-safe primary key deduplication index.
///
/// Sync dedup in the hot path: `BloomFilter::may_contain()`,
/// `FxHashSet::contains()`, and `TextDictReader::ordinal()` are all sync.
///
/// Interior mutability for the mutable state (bloom + uncommitted set) is
/// behind `parking_lot::Mutex`. The committed data is only mutated via
/// `&mut self` methods (commit/abort path), so no lock is needed for it.
pub struct PrimaryKeyIndex {
    field: Field,
    state: parking_lot::Mutex<PrimaryKeyState>,
    /// Lightweight per-segment fast-field data for checking committed keys.
    /// Only mutated by `&mut self` methods (refresh/clear) — no lock needed.
    committed_data: Vec<PkSegmentData>,
    /// Holds ref counts so segments aren't deleted while we hold readers.
    _snapshot: Option<SegmentSnapshot>,
}

struct PrimaryKeyState {
    bloom: BloomFilter,
    uncommitted: FxHashSet<Vec<u8>>,
}

impl PrimaryKeyIndex {
    /// Create a new PrimaryKeyIndex by scanning committed segments.
    ///
    /// Iterates each segment's fast-field text dictionary to populate the bloom
    /// filter with all existing primary key values. The snapshot keeps ref counts
    /// alive so segments aren't deleted while we hold data.
    ///
    /// **CPU-intensive** — call from `spawn_blocking`, not the async runtime.
    pub fn new(field: Field, pk_data: Vec<PkSegmentData>, snapshot: SegmentSnapshot) -> Self {
        // Count total unique keys across all segments for bloom sizing.
        let mut total_keys: usize = 0;
        for data in &pk_data {
            if let Some(ff) = data.fast_fields.get(&field.0)
                && let Some(dict) = ff.text_dict()
            {
                total_keys += dict.len() as usize;
            }
        }

        let mut bloom = BloomFilter::new(total_keys + BLOOM_HEADROOM, BLOOM_BITS_PER_KEY);

        // Insert all committed keys into the bloom filter.
        for data in &pk_data {
            if let Some(ff) = data.fast_fields.get(&field.0)
                && let Some(dict) = ff.text_dict()
            {
                for key in dict.iter() {
                    bloom.insert(key.as_bytes());
                }
            }
        }

        let bloom_bytes = bloom.size_bytes();
        log::info!(
            "[primary_key] bloom filter: {} keys, {:.2} MB",
            total_keys,
            bloom_bytes as f64 / (1024.0 * 1024.0),
        );

        Self {
            field,
            state: parking_lot::Mutex::new(PrimaryKeyState {
                bloom,
                uncommitted: FxHashSet::default(),
            }),
            committed_data: pk_data,
            _snapshot: Some(snapshot),
        }
    }

    /// Create from a pre-loaded bloom filter (loaded from `pk_bloom.bin`).
    ///
    /// Skips dictionary iteration entirely when the persisted bloom covers
    /// all current segments. `pk_data` contains data for ALL current segments.
    /// If `new_data` is non-empty, their keys are inserted into the bloom
    /// before returning (incremental update). `new_data` is a borrowed slice
    /// pointing to the subset of segments not covered by the persisted bloom.
    pub fn from_persisted(
        field: Field,
        mut bloom: BloomFilter,
        pk_data: Vec<PkSegmentData>,
        new_data: &[PkSegmentData],
        snapshot: SegmentSnapshot,
    ) -> Self {
        let mut added = 0usize;
        for data in new_data {
            if let Some(ff) = data.fast_fields.get(&field.0)
                && let Some(dict) = ff.text_dict()
            {
                for key in dict.iter() {
                    bloom.insert(key.as_bytes());
                    added += 1;
                }
            }
        }

        log::info!(
            "[primary_key] bloom filter loaded from cache: {:.2} MB{}",
            bloom.size_bytes() as f64 / (1024.0 * 1024.0),
            if added > 0 {
                format!(
                    ", added {} keys from {} new segment(s)",
                    added,
                    new_data.len()
                )
            } else {
                String::new()
            },
        );

        Self {
            field,
            state: parking_lot::Mutex::new(PrimaryKeyState {
                bloom,
                uncommitted: FxHashSet::default(),
            }),
            committed_data: pk_data,
            _snapshot: Some(snapshot),
        }
    }

    /// Serialize the bloom filter for persistence to `pk_bloom.bin`.
    pub fn bloom_to_bytes(&self) -> Vec<u8> {
        self.state.lock().bloom.to_bytes()
    }

    /// Memory used by the bloom filter and uncommitted set.
    pub fn memory_bytes(&self) -> usize {
        let state = self.state.lock();
        state.bloom.size_bytes() + state.uncommitted.len() * 32 // estimate 32 bytes per key
    }

    /// Check whether a document's primary key is unique, and if so, register it.
    ///
    /// Returns `Ok(())` if the key is new (inserted into bloom + uncommitted set).
    /// Returns `Err(DuplicatePrimaryKey)` if the key already exists.
    /// Returns `Err(Document)` if the primary key field is missing or empty.
    pub fn check_and_insert(&self, doc: &crate::dsl::Document) -> Result<()> {
        let value = doc
            .get_first(self.field)
            .ok_or_else(|| Error::Document("Missing primary key field".into()))?;
        let key = value
            .as_text()
            .ok_or_else(|| Error::Document("Primary key must be text".into()))?;
        if key.is_empty() {
            return Err(Error::Document("Primary key must not be empty".into()));
        }

        let key_bytes = key.as_bytes();

        {
            let mut state = self.state.lock();

            // Fast path: bloom says definitely not present → new key.
            if !state.bloom.may_contain(key_bytes) {
                state.bloom.insert(key_bytes);
                state.uncommitted.insert(key_bytes.to_vec());
                return Ok(());
            }

            // Bloom positive → check uncommitted set first (fast, in-memory).
            if state.uncommitted.contains(key_bytes) {
                return Err(Error::DuplicatePrimaryKey(key.to_string()));
            }
        }
        // Lock released — check committed segments without holding mutex.
        // committed_data is immutable (only changed via &mut self methods).
        for data in &self.committed_data {
            if let Some(ff) = data.fast_fields.get(&self.field.0)
                && let Some(dict) = ff.text_dict()
                && dict.ordinal(key).is_some()
            {
                return Err(Error::DuplicatePrimaryKey(key.to_string()));
            }
        }

        // Re-acquire lock to insert. Re-check uncommitted in case another
        // thread inserted the same key while we were scanning committed segments.
        let mut state = self.state.lock();
        if state.uncommitted.contains(key_bytes) {
            return Err(Error::DuplicatePrimaryKey(key.to_string()));
        }

        // Bloom false positive — key is genuinely new.
        state.bloom.insert(key_bytes);
        state.uncommitted.insert(key_bytes.to_vec());
        Ok(())
    }

    /// Refresh after commit: merge new segment data, prune removed segments,
    /// insert new keys into bloom, and clear uncommitted set.
    ///
    /// Only `new_data` (segments not already held) need to be loaded by the
    /// caller. Existing data for segments still in `snapshot` is retained.
    /// The snapshot keeps ref counts alive so segments aren't deleted.
    pub fn refresh_incremental(&mut self, new_data: Vec<PkSegmentData>, snapshot: SegmentSnapshot) {
        let new_seg_ids: HashSet<&str> =
            snapshot.segment_ids().iter().map(|s| s.as_str()).collect();

        // Insert new segments' keys into bloom (these were uncommitted before).
        // get_mut() bypasses the mutex — safe because we have &mut self.
        let state = self.state.get_mut();
        for data in &new_data {
            if let Some(ff) = data.fast_fields.get(&self.field.0)
                && let Some(dict) = ff.text_dict()
            {
                for key in dict.iter() {
                    state.bloom.insert(key.as_bytes());
                }
            }
        }
        state.uncommitted.clear();

        // Keep existing data for segments still in the snapshot
        let mut kept: Vec<PkSegmentData> = self
            .committed_data
            .drain(..)
            .filter(|d| new_seg_ids.contains(d.segment_id.as_str()))
            .collect();
        kept.extend(new_data);
        self.committed_data = kept;
        self._snapshot = Some(snapshot);
    }

    /// Iterator over segment IDs already held in this PK index.
    pub fn committed_segment_ids(&self) -> impl Iterator<Item = &str> {
        self.committed_data.iter().map(|d| d.segment_id.as_str())
    }

    /// Roll back an uncommitted key registration (e.g. when channel send fails
    /// after check_and_insert succeeded). Bloom may retain the key but that only
    /// causes harmless false positives, never missed duplicates.
    pub fn rollback_uncommitted_key(&self, doc: &crate::dsl::Document) {
        if let Some(value) = doc.get_first(self.field)
            && let Some(key) = value.as_text()
        {
            self.state.lock().uncommitted.remove(key.as_bytes());
        }
    }

    /// Clear uncommitted keys (e.g. on abort). Bloom may retain stale entries
    /// but that only causes harmless false positives (extra committed-segment
    /// lookups), never missed duplicates.
    pub fn clear_uncommitted(&mut self) {
        self.state.get_mut().uncommitted.clear();
    }
}

/// Serialize a bloom filter with the segment IDs it covers into `pk_bloom.bin` format.
///
/// Layout: `[magic:u32][num_segs:u32][seg_id_hex × 32 bytes each...][bloom_bytes...]`
pub fn serialize_pk_bloom(segment_ids: &[String], bloom_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::with_capacity(8 + segment_ids.len() * 32 + bloom_bytes.len());
    data.write_u32::<LittleEndian>(PK_BLOOM_MAGIC).unwrap();
    data.write_u32::<LittleEndian>(segment_ids.len() as u32)
        .unwrap();
    for seg_id in segment_ids {
        let bytes = seg_id.as_bytes();
        data.extend_from_slice(bytes);
        // Pad to 32 bytes (segment IDs are 32-char hex strings)
        data.extend(std::iter::repeat_n(0u8, 32 - bytes.len()));
    }
    data.extend_from_slice(bloom_bytes);
    data
}

/// Deserialize `pk_bloom.bin`. Returns the set of covered segment IDs and the bloom filter,
/// or `None` if the data is corrupt / wrong magic.
pub fn deserialize_pk_bloom(data: &[u8]) -> Option<(HashSet<String>, BloomFilter)> {
    if data.len() < 8 {
        return None;
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != PK_BLOOM_MAGIC {
        return None;
    }
    let num_segments = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let header_end = 8 + num_segments * 32;
    if data.len() < header_end + 12 {
        return None;
    }
    let mut segment_ids = HashSet::with_capacity(num_segments);
    for i in 0..num_segments {
        let start = 8 + i * 32;
        let raw = &data[start..start + 32];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(32);
        let hex = std::str::from_utf8(&raw[..end]).ok()?;
        segment_ids.insert(hex.to_string());
    }
    let bloom = BloomFilter::from_bytes_mutable(&data[header_end..]).ok()?;
    Some((segment_ids, bloom))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::dsl::{Document, Field};
    use crate::segment::SegmentTracker;

    fn make_doc(field: Field, key: &str) -> Document {
        let mut doc = Document::new();
        doc.add_text(field, key);
        doc
    }

    fn empty_snapshot() -> SegmentSnapshot {
        SegmentSnapshot::new(Arc::new(SegmentTracker::new()), vec![])
    }

    #[test]
    fn test_new_empty_readers() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());
        // Should construct without panicking
        let doc = make_doc(field, "key1");
        assert!(pk.check_and_insert(&doc).is_ok());
    }

    #[test]
    fn test_unique_keys_accepted() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        assert!(pk.check_and_insert(&make_doc(field, "a")).is_ok());
        assert!(pk.check_and_insert(&make_doc(field, "b")).is_ok());
        assert!(pk.check_and_insert(&make_doc(field, "c")).is_ok());
    }

    #[test]
    fn test_duplicate_uncommitted_rejected() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
        let result = pk.check_and_insert(&make_doc(field, "key1"));
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::DuplicatePrimaryKey(k) => assert_eq!(k, "key1"),
            other => panic!("Expected DuplicatePrimaryKey, got {:?}", other),
        }
    }

    #[test]
    fn test_missing_field_rejected() {
        let field = Field(0);
        let other_field = Field(1);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        // Document has a different field, not the primary key field
        let doc = make_doc(other_field, "value");
        let result = pk.check_and_insert(&doc);
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Document(msg) => assert!(msg.contains("Missing"), "{}", msg),
            other => panic!("Expected Document error, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_key_rejected() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        let result = pk.check_and_insert(&make_doc(field, ""));
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Document(msg) => assert!(msg.contains("empty"), "{}", msg),
            other => panic!("Expected Document error, got {:?}", other),
        }
    }

    #[test]
    fn test_clear_uncommitted() {
        let field = Field(0);
        let mut pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        // Insert key1
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
        // Duplicate should fail
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_err());

        // Clear uncommitted
        pk.clear_uncommitted();

        // After clear, bloom still has key1 but uncommitted doesn't.
        // With no committed readers, the key should be allowed again
        // (bloom positive → check uncommitted (not found) → check committed (empty) → accept)
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
    }

    #[test]
    fn test_many_unique_keys() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        for i in 0..1000 {
            let key = format!("key_{}", i);
            assert!(pk.check_and_insert(&make_doc(field, &key)).is_ok());
        }

        // All should be duplicates now
        for i in 0..1000 {
            let key = format!("key_{}", i);
            assert!(pk.check_and_insert(&make_doc(field, &key)).is_err());
        }
    }

    #[test]
    fn test_refresh_clears_uncommitted() {
        let field = Field(0);
        let mut pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());

        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_err());

        // Refresh with empty data (simulates commit where segments
        // don't have fast fields — edge case)
        pk.refresh_incremental(vec![], empty_snapshot());

        // After refresh, uncommitted is cleared and no committed data has
        // the key, so it should be accepted again
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
    }

    #[test]
    fn test_pk_bloom_serialize_roundtrip() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![], empty_snapshot());
        for i in 0..100 {
            pk.check_and_insert(&make_doc(field, &format!("key_{}", i)))
                .unwrap();
        }

        let seg_ids = vec![
            "00000000000000000000000000000001".to_string(),
            "00000000000000000000000000000002".to_string(),
        ];
        let bloom_bytes = pk.bloom_to_bytes();
        let data = serialize_pk_bloom(&seg_ids, &bloom_bytes);
        let (got_ids, got_bloom) = deserialize_pk_bloom(&data).expect("deserialize failed");

        assert_eq!(got_ids.len(), 2);
        assert!(got_ids.contains(&seg_ids[0]));
        assert!(got_ids.contains(&seg_ids[1]));

        // Verify the loaded bloom recognizes previously inserted keys.
        for i in 0..100 {
            let key = format!("key_{}", i);
            assert!(
                got_bloom.may_contain(key.as_bytes()),
                "bloom miss for {}",
                key
            );
        }
    }

    #[test]
    fn test_pk_bloom_deserialize_bad_data() {
        assert!(deserialize_pk_bloom(&[]).is_none());
        assert!(deserialize_pk_bloom(&[0; 7]).is_none());
        assert!(deserialize_pk_bloom(&[0; 8]).is_none()); // wrong magic
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let field = Field(0);
        let pk = Arc::new(PrimaryKeyIndex::new(field, vec![], empty_snapshot()));

        // Spawn multiple threads trying to insert the same key
        let mut handles = vec![];
        for _ in 0..10 {
            let pk = Arc::clone(&pk);
            handles.push(std::thread::spawn(move || {
                pk.check_and_insert(&make_doc(field, "contested_key"))
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let successes = results.iter().filter(|r| r.is_ok()).count();
        let failures = results.iter().filter(|r| r.is_err()).count();

        // Exactly one thread should succeed, rest should get DuplicatePrimaryKey
        assert_eq!(successes, 1, "Exactly one insert should succeed");
        assert_eq!(failures, 9, "Rest should fail with duplicate");
    }
}
