//! Storage backends and the async I/O contracts used by the engine.
//!
//! [`Directory`] is the read-only boundary used by search, while
//! [`DirectoryWriter`] adds mutation and streaming output for indexing.
//! [`RamDirectory`] is portable; filesystem and mmap implementations require
//! `native`, and HTTP access requires the `http` feature.

#[cfg(feature = "native")]
mod cold_io;
mod directory;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "native")]
mod mmap;
mod slice_cache;

#[cfg(feature = "native")]
pub(crate) use cold_io::ColdStreamingWriter;
#[cfg(feature = "native")]
pub(crate) use directory::FileStreamingWriter;
#[cfg(feature = "native")]
pub use directory::FsDirectory;
pub use directory::{
    CachingDirectory, Directory, DirectoryWriter, FileHandle, IndexLabel, OwnedBytes, RamDirectory,
    RangeReadFn, StreamingWriter,
};
#[cfg(feature = "http")]
pub use http::*;
#[cfg(feature = "native")]
pub use mmap::MmapDirectory;
pub use slice_cache::*;
