//! Compression using Zstd with dictionary support
//!
//! For static indexes, we use maximum compression (level 22) and trained
//! dictionaries for optimal compression ratios.
//!
//! # Usage
//!
//! ```rust
//! use hermes_core::compression::{compress, decompress, CompressionLevel};
//!
//! let data = b"Hello, World!";
//! let compressed = compress(data, CompressionLevel::MAX).unwrap();
//! let decompressed = decompress(&compressed).unwrap();
//! ```

mod zstd;

pub use self::zstd::{
    CompressionDict, CompressionLevel, compress, compress_with_dict, decompress,
    decompress_with_dict,
};
