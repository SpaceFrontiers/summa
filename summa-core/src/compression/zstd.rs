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

/// The stable first-frame size is a capacity hint, not validation. A stream
/// can contain more frames, so a short bulk attempt still needs the bounded
/// streaming fallback. Unknown sizes never reserve the entire safety limit.
fn limited_bulk_capacity(data: &[u8], max_output: usize) -> io::Result<usize> {
    match zstd::zstd_safe::get_frame_content_size(data) {
        Ok(Some(size)) => {
            let size = usize::try_from(size).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "decompressed size exceeds address space",
                )
            })?;
            if size > max_output {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "decompressed data exceeds configured limit",
                ));
            }
            Ok(size)
        }
        // Let the actual decoder report malformed/truncated input. Empty and
        // size-less streams retain their existing library-defined semantics.
        Ok(None) | Err(_) => Ok(max_output.min(DECOMPRESS_CAPACITY)),
    }
}

/// Decompress while rejecting output larger than `max_output` bytes.
///
/// Index files are trusted only after validation. Using an explicit bound at
/// compressed block boundaries prevents a tiny corrupt frame from expanding
/// until the process runs out of memory.
pub fn decompress_limited(data: &[u8], max_output: usize) -> io::Result<Vec<u8>> {
    let capacity = limited_bulk_capacity(data, max_output)?;
    thread_local! {
        static DECOMPRESSOR: std::cell::RefCell<zstd::bulk::Decompressor<'static>> =
            std::cell::RefCell::new(zstd::bulk::Decompressor::new().unwrap());
    }
    DECOMPRESSOR.with(|dc| {
        dc.borrow_mut().decompress(data, capacity).or_else(|_| {
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
    let capacity = limited_bulk_capacity(data, max_output)?;
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
            .decompress(data, capacity)
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
    fn bounded_small_block_decoding_does_not_reserve_the_safety_limit() {
        let payload = vec![7u8; 16 * 1024];
        let limit = 256 * 1024 * 1024;
        let compressed = compress(&payload, CompressionLevel::default()).unwrap();
        let output = decompress_limited(&compressed, limit).unwrap();
        assert_eq!(output, payload);
        assert!(
            output.capacity() <= payload.len(),
            "small block reserved {} bytes",
            output.capacity()
        );
        let dict = CompressionDict::from_bytes(b"dictionary material".repeat(64));
        let compressed = compress_with_dict(&payload, CompressionLevel::default(), &dict).unwrap();
        let output = decompress_with_dict_limited(&compressed, &dict, limit).unwrap();
        assert_eq!(output, payload);
        assert!(output.capacity() <= payload.len());
    }

    #[test]
    fn bounded_unknown_size_frames_preserve_large_output_and_reject_overflow() {
        let payload = vec![17u8; DECOMPRESS_CAPACITY * 2];
        let compressed = zstd::stream::encode_all(payload.as_slice(), 3).unwrap();
        assert_eq!(
            zstd::zstd_safe::get_frame_content_size(&compressed).unwrap(),
            None
        );
        assert_eq!(
            decompress_limited(&compressed, payload.len()).unwrap(),
            payload
        );
        assert!(decompress_limited(&compressed, payload.len() - 1).is_err());
        let dict = CompressionDict::from_bytes(b"dictionary material".repeat(64));
        let mut encoder =
            zstd::stream::Encoder::with_dictionary(Vec::new(), 3, dict.as_bytes()).unwrap();
        std::io::Write::write_all(&mut encoder, &payload).unwrap();
        let compressed = encoder.finish().unwrap();
        assert_eq!(
            decompress_with_dict_limited(&compressed, &dict, payload.len()).unwrap(),
            payload
        );
        assert!(decompress_with_dict_limited(&compressed, &dict, payload.len() - 1).is_err());
    }

    #[test]
    fn test_limited_dictionary_decompression_rejects_oversized_output() {
        let dict = CompressionDict::from_bytes(b"seven seven seven ".repeat(64));
        let data = b"seven ".repeat(1024);
        let compressed = compress_with_dict(&data, CompressionLevel::default(), &dict).unwrap();
        // Known content size: rejected before any decompression buffer grows.
        assert!(decompress_with_dict_limited(&compressed, &dict, 1024).is_err());
        assert!(decompress_with_dict_limited(&compressed, &dict, data.len() - 1).is_err());
        assert_eq!(
            decompress_with_dict_limited(&compressed, &dict, data.len()).unwrap(),
            data
        );
        // Unknown content size: the streaming fallback enforces the same cap.
        let mut encoder = zstd::Encoder::with_dictionary(Vec::new(), 3, dict.as_bytes()).unwrap();
        encoder.include_contentsize(false).unwrap();
        std::io::Write::write_all(&mut encoder, &data).unwrap();
        let unknown = encoder.finish().unwrap();
        assert_eq!(
            zstd::zstd_safe::get_frame_content_size(&unknown).unwrap(),
            None
        );
        assert!(decompress_with_dict_limited(&unknown, &dict, 1024).is_err());
        assert!(decompress_with_dict_limited(&unknown, &dict, data.len() - 1).is_err());
        assert_eq!(
            decompress_with_dict_limited(&unknown, &dict, data.len()).unwrap(),
            data
        );
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

#[cfg(test)]
mod bounded_capacity_tests {
    use super::*;

    #[test]
    fn small_bounded_zstd_frames_do_not_reserve_the_entire_safety_limit() {
        let payload = b"small independently compressed dictionary block".repeat(20);
        let encoded = compress(&payload, CompressionLevel::BETTER).unwrap();
        let decoded = decompress_limited(&encoded, 64 * 1024 * 1024).unwrap();
        assert_eq!(decoded, payload);
        assert!(
            decoded.capacity() <= 4096,
            "capacity={}",
            decoded.capacity()
        );
    }

    #[test]
    fn small_dictionary_frames_do_not_reserve_the_entire_safety_limit() {
        let dict = CompressionDict::from_bytes(b"dictionary material for small blocks".repeat(20));
        let payload = b"dictionary material for small blocks and metadata".repeat(20);
        let encoded = compress_with_dict(&payload, CompressionLevel::BETTER, &dict).unwrap();
        let decoded = decompress_with_dict_limited(&encoded, &dict, 64 * 1024 * 1024).unwrap();
        assert_eq!(decoded, payload);
        assert!(
            decoded.capacity() <= 4096,
            "capacity={}",
            decoded.capacity()
        );
    }
}

#[cfg(test)]
mod bounded_frame_tests {
    use super::*;
    use std::io::Write;

    fn assert_plain_and_dictionary(encoded: &[u8], expected: &[u8], dict: &CompressionDict) {
        assert_eq!(
            decompress_limited(encoded, expected.len()).unwrap(),
            expected
        );
        assert_eq!(
            decompress_with_dict_limited(encoded, dict, expected.len()).unwrap(),
            expected
        );
        if !expected.is_empty() {
            assert!(decompress_limited(encoded, expected.len() - 1).is_err());
            assert!(decompress_with_dict_limited(encoded, dict, expected.len() - 1).is_err());
        }
    }

    #[test]
    fn bounded_zstd_decoding_preserves_empty_unknown_size_and_concatenated_frames() {
        let dict = CompressionDict::from_bytes(Vec::new());
        let empty = compress(&[], CompressionLevel::FAST).unwrap();
        assert_plain_and_dictionary(&empty, &[], &dict);
        let payload = vec![77u8; DECOMPRESS_CAPACITY * 2 + 17];
        let mut encoder = zstd::Encoder::new(Vec::new(), 1).unwrap();
        encoder.include_contentsize(false).unwrap();
        encoder.write_all(&payload).unwrap();
        let unknown = encoder.finish().unwrap();
        assert_eq!(
            zstd::zstd_safe::get_frame_content_size(&unknown).unwrap(),
            None
        );
        assert_eq!(
            limited_bulk_capacity(&unknown, 64 * 1024 * 1024).unwrap(),
            DECOMPRESS_CAPACITY
        );
        assert_plain_and_dictionary(&unknown, &payload, &dict);
        let first = compress(b"first", CompressionLevel::FAST).unwrap();
        let second = compress(b"second", CompressionLevel::FAST).unwrap();
        let joined = [first.as_slice(), second.as_slice()].concat();
        assert_eq!(zstd::decode_all(joined.as_slice()).unwrap(), b"firstsecond");
        assert_plain_and_dictionary(&joined, b"firstsecond", &dict);
        // Skippable frame is a standard Zstd frame with an eight-byte header.
        let skipped = [
            0x184D2A50u32.to_le_bytes().as_slice(),
            3u32.to_le_bytes().as_slice(),
            b"tag",
            second.as_slice(),
        ]
        .concat();
        assert_eq!(zstd::decode_all(skipped.as_slice()).unwrap(), b"second");
        assert_plain_and_dictionary(&skipped, b"second", &dict);
    }

    #[test]
    fn bounded_zstd_decoding_does_not_hide_truncation_checksum_or_trailing_corruption() {
        let payload = b"known size does not prove frame integrity".repeat(100);
        let mut encoder = zstd::Encoder::new(Vec::new(), 1).unwrap();
        encoder.include_checksum(true).unwrap();
        encoder.write_all(&payload).unwrap();
        let good = encoder.finish().unwrap();
        let mut bad_checksum = good.clone();
        *bad_checksum.last_mut().unwrap() ^= 1;
        let mut trailing = good.clone();
        trailing.extend_from_slice(b"bad trailing frame");
        for broken in [
            &good[..good.len() - 1],
            bad_checksum.as_slice(),
            trailing.as_slice(),
        ] {
            assert!(zstd::decode_all(broken).is_err());
            assert!(decompress_limited(broken, payload.len()).is_err());
            let dict = CompressionDict::from_bytes(Vec::new());
            assert!(decompress_with_dict_limited(broken, &dict, payload.len()).is_err());
            assert_eq!(decompress_limited(&good, payload.len()).unwrap(), payload);
            assert_eq!(
                decompress_with_dict_limited(&good, &dict, payload.len()).unwrap(),
                payload
            );
        }
    }

    #[test]
    fn bounded_dictionary_decoding_preserves_concatenation_and_dictionary_switches() {
        let first = CompressionDict::from_bytes(b"first raw dictionary contents".repeat(20));
        let second = CompressionDict::from_bytes(b"second raw dictionary contents".repeat(20));
        for dict in [&first, &second, &first] {
            let a = compress_with_dict(
                b"first raw dictionary contents",
                CompressionLevel::FAST,
                dict,
            )
            .unwrap();
            let b = compress_with_dict(
                b"second raw dictionary contents",
                CompressionLevel::FAST,
                dict,
            )
            .unwrap();
            let joined = [a, b].concat();
            let expected = b"first raw dictionary contentssecond raw dictionary contents";
            assert_eq!(
                decompress_with_dict_limited(&joined, dict, expected.len()).unwrap(),
                expected
            );
            assert!(decompress_with_dict_limited(&joined, dict, expected.len() - 1).is_err());
        }
    }
}
