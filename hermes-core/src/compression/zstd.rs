//! Zstd compression backend with dictionary support
//!
//! For static indexes, we use:
//! - Maximum compression level (22) for best compression ratio
//! - Trained dictionaries for even better compression of similar documents
//! - Larger block sizes to improve compression efficiency

use std::io::{self, Write};

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
        Self::MAX // Use maximum compression for static indexes
    }
}

/// Trained Zstd dictionary for improved compression
#[derive(Clone)]
pub struct CompressionDict {
    raw_dict: crate::directories::OwnedBytes,
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
        })
    }

    /// Create dictionary from raw bytes (for loading saved dictionaries)
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            raw_dict: crate::directories::OwnedBytes::new(bytes),
        }
    }

    /// Create dictionary from OwnedBytes (zero-copy for mmap)
    pub fn from_owned_bytes(bytes: crate::directories::OwnedBytes) -> Self {
        Self { raw_dict: bytes }
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
}

/// Compress data using Zstd
pub fn compress(data: &[u8], level: CompressionLevel) -> io::Result<Vec<u8>> {
    zstd::encode_all(data, level.0).map_err(io::Error::other)
}

/// Compress data using Zstd with a trained dictionary
pub fn compress_with_dict(
    data: &[u8],
    level: CompressionLevel,
    dict: &CompressionDict,
) -> io::Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::with_dictionary(Vec::new(), level.0, dict.raw_dict.as_slice())
        .map_err(io::Error::other)?;
    encoder.write_all(data)?;
    encoder.finish().map_err(io::Error::other)
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

/// Decompress data using Zstd with a trained dictionary
///
/// Caches the dictionary decompressor in a thread-local, keyed by the
/// dictionary's data pointer. Since a given `AsyncStoreReader` always holds
/// the same `CompressionDict` (behind `Arc<OwnedBytes>`), the pointer is
/// stable for the reader's lifetime. The decompressor is only rebuilt when
/// a different dictionary is encountered (e.g., switching between segments).
pub fn decompress_with_dict(data: &[u8], dict: &CompressionDict) -> io::Result<Vec<u8>> {
    thread_local! {
        static DICT_DC: std::cell::RefCell<Option<(usize, zstd::bulk::Decompressor<'static>)>> =
            const { std::cell::RefCell::new(None) };
    }
    // Use the raw dict slice pointer as a stable identity key.
    let dict_key = dict.as_bytes().as_ptr() as usize;

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
}
