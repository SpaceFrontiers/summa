//! Zstd compression backend with dictionary support
//!
//! For static indexes, we use:
//! - Maximum compression level (22) for best compression ratio
//! - Trained dictionaries for even better compression of similar documents
//! - Larger block sizes to improve compression efficiency

use std::io;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DICTIONARY_ID: AtomicU64 = AtomicU64::new(1);

/// Compression level (1-22 for zstd)
#[derive(Debug, Clone, Copy)]
pub struct CompressionLevel(pub i32);

impl CompressionLevel {
    /// Fast compression (level 1)
    pub const FAST: Self = Self(1);
    /// Default compression (level 3)
    pub const DEFAULT: Self = Self(3);
    /// Better compression (level 9)
    pub const BETTER: Self = Self(9);
    /// Best compression (level 19)
    pub const BEST: Self = Self(19);
    /// Maximum compression (level 22) - slowest but smallest
    pub const MAX: Self = Self(22);
}

impl Default for CompressionLevel {
    fn default() -> Self {
        Self::FAST // Level 3: good balance of speed and compression
    }
}

/// Trained Zstd dictionary for improved compression
#[derive(Clone)]
pub struct CompressionDict {
    raw_dict: crate::directories::OwnedBytes,
    /// Stable across clones and never derived from an allocator address. The
    /// thread-local codec caches can outlive a dictionary, so raw pointers are
    /// vulnerable to allocator ABA reuse.
    cache_id: u64,
}

impl CompressionDict {
    /// Train a dictionary from sample data
    ///
    /// For best results, provide many small samples (e.g., serialized documents)
    /// The dictionary size should typically be 16KB-112KB
    pub fn train(samples: &[&[u8]], dict_size: usize) -> io::Result<Self> {
        let raw_dict = zstd::dict::from_samples(samples, dict_size).map_err(io::Error::other)?;
        Ok(Self {
            raw_dict: crate::directories::OwnedBytes::new(raw_dict),
            cache_id: NEXT_DICTIONARY_ID.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Create dictionary from raw bytes (for loading saved dictionaries)
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            raw_dict: crate::directories::OwnedBytes::new(bytes),
            cache_id: NEXT_DICTIONARY_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Create dictionary from OwnedBytes (zero-copy for mmap)
    pub fn from_owned_bytes(bytes: crate::directories::OwnedBytes) -> Self {
        Self {
            raw_dict: bytes,
            cache_id: NEXT_DICTIONARY_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Get raw dictionary bytes (for saving)
    pub fn as_bytes(&self) -> &[u8] {
        self.raw_dict.as_slice()
    }

    /// Dictionary size in bytes
    pub fn len(&self) -> usize {
        self.raw_dict.len()
    }

    /// Check if dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.raw_dict.is_empty()
    }

    #[inline]
    fn cache_id(&self) -> u64 {
        self.cache_id
    }
}

/// Compress data using Zstd
///
/// Uses a thread-local bulk compressor to avoid per-call encoder allocation.
/// Only rebuilds when the compression level changes.
pub fn compress(data: &[u8], level: CompressionLevel) -> io::Result<Vec<u8>> {
    thread_local! {
        static COMPRESSOR: std::cell::RefCell<Option<(i32, zstd::bulk::Compressor<'static>)>> =
            const { std::cell::RefCell::new(None) };
    }
    COMPRESSOR.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().is_none_or(|(l, _)| *l != level.0) {
            let cmp = zstd::bulk::Compressor::new(level.0).map_err(io::Error::other)?;
            *slot = Some((level.0, cmp));
        }
        slot.as_mut()
            .unwrap()
            .1
            .compress(data)
            .map_err(io::Error::other)
    })
}

/// Compress data using Zstd with a trained dictionary
///
/// Caches the dictionary compressor in a thread-local, keyed by dictionary
/// pointer + compression level. Only rebuilt when dict or level changes.
pub fn compress_with_dict(
    data: &[u8],
    level: CompressionLevel,
    dict: &CompressionDict,
) -> io::Result<Vec<u8>> {
    thread_local! {
        static DICT_CMP: std::cell::RefCell<Option<(u64, i32, zstd::bulk::Compressor<'static>)>> =
            const { std::cell::RefCell::new(None) };
    }
    let dict_key = dict.cache_id();

    DICT_CMP.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|(k, l, _)| *k != dict_key || *l != level.0)
        {
            let cmp = zstd::bulk::Compressor::with_dictionary(level.0, dict.as_bytes())
                .map_err(io::Error::other)?;
            *slot = Some((dict_key, level.0, cmp));
        }
        slot.as_mut()
            .unwrap()
            .2
            .compress(data)
            .map_err(io::Error::other)
    })
}

/// Capacity hint for bulk decompressor (covers typical 256KB store blocks).
/// Blocks that decompress larger than this fall back to streaming decode.
const DECOMPRESS_CAPACITY: usize = 512 * 1024;

/// Decompress data using Zstd
///
/// Fast path: reuses a thread-local bulk `Decompressor` with a 512KB
/// capacity hint. Falls back to streaming decode for oversized blocks.
pub fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    thread_local! {
        static DECOMPRESSOR: std::cell::RefCell<zstd::bulk::Decompressor<'static>> =
            std::cell::RefCell::new(zstd::bulk::Decompressor::new().unwrap());
    }
    DECOMPRESSOR.with(|dc| {
        dc.borrow_mut()
            .decompress(data, DECOMPRESS_CAPACITY)
            .or_else(|_| zstd::decode_all(data))
    })
}

/// Decompress while rejecting output larger than `max_output` bytes.
///
/// Index files are trusted only after validation. Using an explicit bound at
/// compressed block boundaries prevents a tiny corrupt frame from expanding
/// until the process runs out of memory.
pub fn decompress_limited(data: &[u8], max_output: usize) -> io::Result<Vec<u8>> {
    thread_local! {
        static DECOMPRESSOR: std::cell::RefCell<zstd::bulk::Decompressor<'static>> =
            std::cell::RefCell::new(zstd::bulk::Decompressor::new().unwrap());
    }
    DECOMPRESSOR.with(|dc| {
        dc.borrow_mut().decompress(data, max_output).or_else(|_| {
            let decoder = zstd::Decoder::new(data)?;
            read_limited(decoder, max_output)
        })
    })
}

/// Decompress data using Zstd with a trained dictionary
///
/// Caches the dictionary decompressor in a thread-local, keyed by the
/// dictionary's data pointer. Since a given `AsyncStoreReader` always holds
/// the same `CompressionDict` (behind `Arc<OwnedBytes>`), the pointer is
/// stable for the reader's lifetime. The decompressor is only rebuilt when
/// a different dictionary is encountered (e.g., switching between segments).
pub fn decompress_with_dict(data: &[u8], dict: &CompressionDict) -> io::Result<Vec<u8>> {
    thread_local! {
        static DICT_DC: std::cell::RefCell<Option<(u64, zstd::bulk::Decompressor<'static>)>> =
            const { std::cell::RefCell::new(None) };
    }
    // Use the raw dict slice pointer as a stable identity key.
    let dict_key = dict.cache_id();

    DICT_DC.with(|cell| {
        let mut slot = cell.borrow_mut();
        // Rebuild decompressor only if dict changed
        if slot.as_ref().is_none_or(|(k, _)| *k != dict_key) {
            let dc = zstd::bulk::Decompressor::with_dictionary(dict.as_bytes())
                .map_err(io::Error::other)?;
            *slot = Some((dict_key, dc));
        }
        slot.as_mut()
            .unwrap()
            .1
            .decompress(data, DECOMPRESS_CAPACITY)
            .or_else(|_| {
                let mut decoder = zstd::Decoder::with_dictionary(data, dict.as_bytes())?;
                let mut output = Vec::new();
                io::Read::read_to_end(&mut decoder, &mut output)?;
                Ok(output)
            })
    })
}

/// Dictionary variant of [`decompress_limited`].
pub fn decompress_with_dict_limited(
    data: &[u8],
    dict: &CompressionDict,
    max_output: usize,
) -> io::Result<Vec<u8>> {
    thread_local! {
        static DICT_DC: std::cell::RefCell<Option<(u64, zstd::bulk::Decompressor<'static>)>> =
            const { std::cell::RefCell::new(None) };
    }
    let dict_key = dict.cache_id();

    DICT_DC.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().is_none_or(|(key, _)| *key != dict_key) {
            let dc = zstd::bulk::Decompressor::with_dictionary(dict.as_bytes())
                .map_err(io::Error::other)?;
            *slot = Some((dict_key, dc));
        }
        slot.as_mut()
            .unwrap()
            .1
            .decompress(data, max_output)
            .or_else(|_| {
                let decoder = zstd::Decoder::with_dictionary(data, dict.as_bytes())?;
                read_limited(decoder, max_output)
            })
    })
}

fn read_limited(mut reader: impl Read, max_output: usize) -> io::Result<Vec<u8>> {
    let read_limit = u64::try_from(max_output)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let initial_capacity = max_output.min(DECOMPRESS_CAPACITY);
    let mut output = Vec::with_capacity(initial_capacity);
    reader.by_ref().take(read_limit).read_to_end(&mut output)?;
    if output.len() > max_output {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "decompressed data exceeds configured limit",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let data = b"Hello, World! This is a test of compression.".repeat(100);
        let compressed = compress(&data, CompressionLevel::default()).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_empty_data() {
        let data: &[u8] = &[];
        let compressed = compress(data, CompressionLevel::default()).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compression_levels() {
        let data = b"Test data for compression levels".repeat(100);
        for level in [1, 3, 9, 19] {
            let compressed = compress(&data, CompressionLevel(level)).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(data.as_slice(), decompressed.as_slice());
        }
    }

    #[test]
    fn test_limited_decompression_rejects_oversized_output() {
        let data = vec![7u8; 4096];
        let compressed = compress(&data, CompressionLevel::default()).unwrap();
        assert!(decompress_limited(&compressed, 1024).is_err());
        assert_eq!(decompress_limited(&compressed, data.len()).unwrap(), data);
    }

    #[test]
    fn test_dictionary_cache_identity_is_stable_and_unique() {
        let first = CompressionDict::from_bytes(b"first dictionary material".repeat(16));
        let first_clone = first.clone();
        let second = CompressionDict::from_bytes(b"second dictionary material".repeat(16));
        assert_eq!(first.cache_id(), first_clone.cache_id());
        assert_ne!(first.cache_id(), second.cache_id());

        let payload = b"dictionary cache switches must rebuild their codec state".repeat(64);
        for dict in [&first, &second, &first_clone] {
            let compressed =
                compress_with_dict(&payload, CompressionLevel::default(), dict).unwrap();
            assert_eq!(
                decompress_with_dict_limited(&compressed, dict, payload.len()).unwrap(),
                payload
            );
        }
    }
}
