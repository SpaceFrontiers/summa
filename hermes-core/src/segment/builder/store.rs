//! Document store streaming build.
//!
//! Reads pre-serialized document bytes from temp file and passes them
//! directly to the store writer, avoiding deserialize→Document→reserialize.

#[cfg(feature = "native")]
use std::fs::File;
use std::io::Write;
#[cfg(feature = "native")]
use std::path::PathBuf;

use crate::Result;
use crate::compression::CompressionLevel;

/// Stream compressed document store directly to disk.
///
/// Reads pre-serialized document bytes from temp file and passes them
/// directly to the store writer via `store_raw`, avoiding the
/// deserialize→Document→reserialize roundtrip entirely.
///
/// `expected_docs` is the number of documents the builder added. The store
/// writer's final doc count is verified against this — a mismatch means the
/// temp file was truncated or a compression block was lost, which would
/// silently corrupt the segment (postings/store doc_id desync).
#[cfg(feature = "native")]
pub(super) fn build_store_streaming(
    store_path: &PathBuf,
    num_compression_threads: usize,
    compression_level: CompressionLevel,
    writer: &mut dyn Write,
    expected_docs: u32,
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

    let store_docs = store_writer.finish()?;
    if store_docs != expected_docs {
        return Err(crate::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Store doc count mismatch: store wrote {} docs but builder expected {}",
                store_docs, expected_docs
            ),
        )));
    }
    Ok(())
}

/// Stream compressed document store from in-memory buffer (WASM path).
///
/// Uses single-threaded inline compression instead of EagerParallelStoreWriter
/// which requires std::thread::spawn (unavailable on WASM).
#[cfg(not(feature = "native"))]
pub(super) fn build_store_streaming_from_buffer(
    store_data: &[u8],
    compression_level: CompressionLevel,
    writer: &mut dyn Write,
    expected_docs: u32,
) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};

    let block_size = crate::segment::store::STORE_BLOCK_SIZE;

    struct BlockIndex {
        first_doc_id: u32,
        offset: u64,
        length: u32,
        num_docs: u32,
    }

    let mut index: Vec<BlockIndex> = Vec::new();
    let mut current_offset = 0u64;
    let mut next_doc_id: u32 = 0;
    let mut block_first_doc: u32 = 0;
    let mut block_buffer = Vec::with_capacity(block_size);

    let mut offset = 0usize;
    while offset + 4 <= store_data.len() {
        let doc_len = u32::from_le_bytes([
            store_data[offset],
            store_data[offset + 1],
            store_data[offset + 2],
            store_data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + doc_len > store_data.len() {
            break;
        }

        // Write doc length prefix + doc bytes to block buffer
        block_buffer.write_u32::<LittleEndian>(doc_len as u32)?;
        block_buffer.extend_from_slice(&store_data[offset..offset + doc_len]);
        next_doc_id += 1;
        offset += doc_len;

        // Flush block when it exceeds block_size
        if block_buffer.len() >= block_size {
            let compressed = crate::compression::compress(&block_buffer, compression_level)?;
            writer.write_all(&compressed)?;
            index.push(BlockIndex {
                first_doc_id: block_first_doc,
                offset: current_offset,
                length: compressed.len() as u32,
                num_docs: next_doc_id - block_first_doc,
            });
            current_offset += compressed.len() as u64;
            block_first_doc = next_doc_id;
            block_buffer.clear();
        }
    }

    // Flush remaining data
    if !block_buffer.is_empty() {
        let compressed = crate::compression::compress(&block_buffer, compression_level)?;
        writer.write_all(&compressed)?;
        index.push(BlockIndex {
            first_doc_id: block_first_doc,
            offset: current_offset,
            length: compressed.len() as u32,
            num_docs: next_doc_id - block_first_doc,
        });
        current_offset += compressed.len() as u64;
    }

    // Write store index + footer (same format as EagerParallelStoreWriter)
    let data_end_offset = current_offset;
    writer.write_u32::<LittleEndian>(index.len() as u32)?;
    for entry in &index {
        writer.write_u32::<LittleEndian>(entry.first_doc_id)?;
        writer.write_u64::<LittleEndian>(entry.offset)?;
        writer.write_u32::<LittleEndian>(entry.length)?;
        writer.write_u32::<LittleEndian>(entry.num_docs)?;
    }
    writer.write_u64::<LittleEndian>(data_end_offset)?;
    writer.write_u64::<LittleEndian>(0)?; // dict_offset
    writer.write_u32::<LittleEndian>(next_doc_id)?;
    writer.write_u32::<LittleEndian>(0)?; // has_dict = false
    writer.write_u32::<LittleEndian>(2)?; // STORE_VERSION
    writer.write_u32::<LittleEndian>(0x53544F52)?; // STORE_MAGIC

    if next_doc_id != expected_docs {
        return Err(crate::Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Store doc count mismatch: store wrote {} docs but builder expected {}",
                next_doc_id, expected_docs
            ),
        )));
    }
    Ok(())
}
