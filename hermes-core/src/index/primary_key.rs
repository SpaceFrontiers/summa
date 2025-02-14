//! Primary key deduplication index.
//!
//! Uses a bloom filter + `FxHashSet` for uncommitted keys to reject duplicates
//! at `add_document()` time. Committed keys are checked via fast-field
//! `TextDictReader::ordinal()` (binary search, O(log n)).

use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::dsl::Field;
use crate::error::{Error, Result};
use crate::segment::SegmentReader;
use crate::structures::BloomFilter;

/// Bloom filter sizing: 10 bits/key ≈ 1% false positive rate.
const BLOOM_BITS_PER_KEY: usize = 10;

/// Extra capacity added to bloom filter beyond known keys.
const BLOOM_HEADROOM: usize = 100_000;

/// Thread-safe primary key deduplication index.
///
/// Sync dedup in the hot path: `BloomFilter::may_contain()`,
/// `FxHashSet::contains()`, and `TextDictReader::ordinal()` are all sync.
///
/// Interior mutability for the mutable state (bloom + uncommitted set) is
/// behind `parking_lot::Mutex`. The committed readers are only mutated via
/// `&mut self` methods (commit/abort path), so no lock is needed for them.
pub struct PrimaryKeyIndex {
    field: Field,
    state: parking_lot::Mutex<PrimaryKeyState>,
    /// Segment readers for checking committed keys.
    /// Only mutated by `&mut self` methods (refresh/clear) — no lock needed.
    committed_readers: Vec<Arc<SegmentReader>>,
}

struct PrimaryKeyState {
    bloom: BloomFilter,
    uncommitted: FxHashSet<Vec<u8>>,
}

impl PrimaryKeyIndex {
    /// Create a new PrimaryKeyIndex by scanning committed segments.
    ///
    /// Iterates each reader's fast-field text dictionary to populate the bloom
    /// filter with all existing primary key values.
    pub fn new(field: Field, readers: Vec<Arc<SegmentReader>>) -> Self {
        // Count total unique keys across all segments for bloom sizing.
        let mut total_keys: usize = 0;
        for reader in &readers {
            if let Some(ff) = reader.fast_field(field.0)
                && let Some(dict) = ff.text_dict()
            {
                total_keys += dict.len() as usize;
            }
        }

        let mut bloom = BloomFilter::new(total_keys + BLOOM_HEADROOM, BLOOM_BITS_PER_KEY);

        // Insert all committed keys into the bloom filter.
        for reader in &readers {
            if let Some(ff) = reader.fast_field(field.0)
                && let Some(dict) = ff.text_dict()
            {
                for key in dict.iter() {
                    bloom.insert(key.as_bytes());
                }
            }
        }

        Self {
            field,
            state: parking_lot::Mutex::new(PrimaryKeyState {
                bloom,
                uncommitted: FxHashSet::default(),
            }),
            committed_readers: readers,
        }
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

        // Check committed segments via fast-field text dictionary.
        for reader in &self.committed_readers {
            if let Some(ff) = reader.fast_field(self.field.0)
                && let Some(dict) = ff.text_dict()
                && dict.ordinal(key).is_some()
            {
                return Err(Error::DuplicatePrimaryKey(key.to_string()));
            }
        }

        // Bloom false positive — key is genuinely new.
        state.bloom.insert(key_bytes);
        state.uncommitted.insert(key_bytes.to_vec());
        Ok(())
    }

    /// Refresh after commit: replace committed readers and clear uncommitted set.
    ///
    /// The new readers include the just-committed segments, so their text
    /// dictionaries already contain the previously-uncommitted keys.
    pub fn refresh(&mut self, new_readers: Vec<Arc<SegmentReader>>) {
        self.committed_readers = new_readers;
        // get_mut() bypasses the mutex — safe because we have &mut self.
        let state = self.state.get_mut();
        state.uncommitted.clear();
    }

    /// Clear uncommitted keys (e.g. on abort). Bloom may retain stale entries
    /// but that only causes harmless false positives (extra committed-segment
    /// lookups), never missed duplicates.
    pub fn clear_uncommitted(&mut self) {
        self.state.get_mut().uncommitted.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::{Document, Field};

    fn make_doc(field: Field, key: &str) -> Document {
        let mut doc = Document::new();
        doc.add_text(field, key);
        doc
    }

    #[test]
    fn test_new_empty_readers() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![]);
        // Should construct without panicking
        let doc = make_doc(field, "key1");
        assert!(pk.check_and_insert(&doc).is_ok());
    }

    #[test]
    fn test_unique_keys_accepted() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![]);

        assert!(pk.check_and_insert(&make_doc(field, "a")).is_ok());
        assert!(pk.check_and_insert(&make_doc(field, "b")).is_ok());
        assert!(pk.check_and_insert(&make_doc(field, "c")).is_ok());
    }

    #[test]
    fn test_duplicate_uncommitted_rejected() {
        let field = Field(0);
        let pk = PrimaryKeyIndex::new(field, vec![]);

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
        let pk = PrimaryKeyIndex::new(field, vec![]);

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
        let pk = PrimaryKeyIndex::new(field, vec![]);

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
        let mut pk = PrimaryKeyIndex::new(field, vec![]);

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
        let pk = PrimaryKeyIndex::new(field, vec![]);

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
        let mut pk = PrimaryKeyIndex::new(field, vec![]);

        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_err());

        // Refresh with empty readers (simulates commit where segment readers
        // don't have fast fields — edge case)
        pk.refresh(vec![]);

        // After refresh, uncommitted is cleared and no committed readers have
        // the key, so it should be accepted again
        assert!(pk.check_and_insert(&make_doc(field, "key1")).is_ok());
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let field = Field(0);
        let pk = Arc::new(PrimaryKeyIndex::new(field, vec![]));

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
