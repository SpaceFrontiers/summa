//! Zstd compression backend with dictionary support
//!
////! For static indexes, we use:
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
    raw_dict: Vec<u8>,
}

impl CompressionDict {
    /// Train a dictionary from sample data
    ///
    /// For best results, provide many small samples (e.g., serialized documents)
    /// The dictionary size should typically be 16KB-112KB
    pub fn train(samples: &[&[u8]], dict_size: usize) -> io::Result<Self> {
        let raw_dict = zstd::dict::from_samples(samples, dict_size).map_err(io::Error::other)?;
        Ok(Self { raw_dict })
    }

    /// Create dictionary from raw bytes (for loading saved dictionaries)
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { raw_dict: bytes }
    }

    /// Get raw dictionary bytes (for saving)
    pub fn as_bytes(&self) -> &[u8] {
        &self.raw_dict
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
    let mut encoder = zstd::Encoder::with_dictionary(Vec::new(), level.0, &dict.raw_dict)
        .map_err(io::Error::other)?;
    encoder.write_all(data)?;
    encoder.finish().map_err(io::Error::other)
}

/// Upper bound for decompressed output (512KB covers 256KB store blocks).
const DECOMPRESS_CAPACITY: usize = 512 * 1024;

/// Decompress data using Zstd
///
/// Reuses a thread-local `Decompressor` to avoid re-initializing the
/// zstd context on every call. The bulk API reads the content-size
/// field from the frame header and allocates the exact output buffer.
pub fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    thread_local! {
        static DECOMPRESSOR: std::cell::RefCell<zstd::bulk::Decompressor<'static>> =
            std::cell::RefCell::new(zstd::bulk::Decompressor::new().unwrap());
    }
    DECOMPRESSOR.with(|dc| {
        dc.borrow_mut()
            .decompress(data, DECOMPRESS_CAPACITY)
            .map_err(io::Error::other)
    })
}

/// Decompress data using Zstd with a trained dictionary
///
/// Note: dictionary decompressors are NOT reused via thread-local because
/// each store/sstable may use a different dictionary. The caller (block
/// cache) ensures this is called only on cache misses.
pub fn decompress_with_dict(data: &[u8], dict: &CompressionDict) -> io::Result<Vec<u8>> {
    let mut decompressor =
        zstd::bulk::Decompressor::with_dictionary(&dict.raw_dict).map_err(io::Error::other)?;
    decompressor
        .decompress(data, DECOMPRESS_CAPACITY)
        .map_err(io::Error::other)
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
