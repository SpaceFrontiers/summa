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
