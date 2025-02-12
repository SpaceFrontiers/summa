mod directory;
#[cfg(feature = "http")]
mod http;
#[cfg(feature = "native")]
mod mmap;
mod slice_cache;

#[cfg(feature = "native")]
pub(crate) use directory::FileStreamingWriter;
#[cfg(feature = "native")]
pub use directory::FsDirectory;
pub use directory::{
    CachingDirectory, Directory, DirectoryWriter, FileHandle, OwnedBytes, RamDirectory,
    RangeReadFn, StreamingWriter,
};
#[cfg(feature = "http")]
pub use http::*;
#[cfg(feature = "native")]
pub use mmap::MmapDirectory;
pub use slice_cache::*;
