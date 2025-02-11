//! Document store streaming build.
//!
//! Reads pre-serialized document bytes from temp file and passes them
//! directly to the store writer, avoiding deserialize→Document→reserialize.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use crate::Result;
use crate::compression::CompressionLevel;

/// Stream compressed document store directly to disk.
///
/// Reads pre-serialized document bytes from temp file and passes them
/// directly to the store writer via `store_raw`, avoiding the
/// deserialize→Document→reserialize roundtrip entirely.
pub(super) fn build_store_streaming(
    store_path: &PathBuf,
    num_compression_threads: usize,
    compression_level: CompressionLevel,
    writer: &mut dyn Write,
) -> Result<()> {
    use crate::segment::store::EagerParallelStoreWriter;

    let file = File::open(store_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let mut store_writer = EagerParallelStoreWriter::with_compression_level(
        writer,
        num_compression_threads,
        compression_level,
    );

    // Stream pre-serialized doc bytes directly — no deserialization needed.
    // Temp file format: [doc_len: u32 LE][doc_bytes: doc_len bytes] repeated.
    let mut offset = 0usize;
    while offset + 4 <= mmap.len() {
        let doc_len = u32::from_le_bytes([
            mmap[offset],
            mmap[offset + 1],
            mmap[offset + 2],
            mmap[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + doc_len > mmap.len() {
            break;
        }

        let doc_bytes = &mmap[offset..offset + doc_len];
        store_writer.store_raw(doc_bytes)?;
        offset += doc_len;
    }

    store_writer.finish()?;
    Ok(())
}
