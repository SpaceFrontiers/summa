//! Document store with Zstd compression and lazy loading
//!
//! Optimized for static indexes:
//! - Maximum compression level (22) for best compression ratio
//! - Larger block sizes (64KB) for better compression efficiency
//! - Optional trained dictionary support for even better compression
//! - Parallel compression support for faster indexing
//!
//! Writer stores documents in compressed blocks.
//! Reader only loads index into memory, blocks are loaded on-demand.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::io::{self, Write};
use std::sync::Arc;

use crate::DocId;
use crate::compression::CompressionDict;
#[cfg(feature = "native")]
use crate::compression::CompressionLevel;
use crate::directories::{AsyncFileRead, LazyFileHandle, LazyFileSlice};
use crate::dsl::{Document, Schema};

const STORE_MAGIC: u32 = 0x53544F52; // "STOR"
const STORE_VERSION: u32 = 2; // Version 2 supports dictionaries

/// Write block index + footer to a store file.
///
/// Shared by `StoreWriter::finish`, `StoreWriter::finish` (empty), and `StoreMerger::finish`.
fn write_store_index_and_footer(
    writer: &mut (impl Write + ?Sized),
    index: &[StoreBlockIndex],
    data_end_offset: u64,
    dict_offset: u64,
    num_docs: u32,
    has_dict: bool,
) -> io::Result<()> {
    writer.write_u32::<LittleEndian>(index.len() as u32)?;
    for entry in index {
        writer.write_u32::<LittleEndian>(entry.first_doc_id)?;
        writer.write_u64::<LittleEndian>(entry.offset)?;
        writer.write_u32::<LittleEndian>(entry.length)?;
        writer.write_u32::<LittleEndian>(entry.num_docs)?;
    }
    writer.write_u64::<LittleEndian>(data_end_offset)?;
    writer.write_u64::<LittleEndian>(dict_offset)?;
    writer.write_u32::<LittleEndian>(num_docs)?;
    writer.write_u32::<LittleEndian>(if has_dict { 1 } else { 0 })?;
    writer.write_u32::<LittleEndian>(STORE_VERSION)?;
    writer.write_u32::<LittleEndian>(STORE_MAGIC)?;
    Ok(())
}

/// Block size for document store (256KB for better compression)
/// Larger blocks = better compression ratio but more memory per block load
pub const STORE_BLOCK_SIZE: usize = 256 * 1024;

/// Default dictionary size (64KB is a good balance)
pub const DEFAULT_DICT_SIZE: usize = 4 * 1024;

/// Default compression level for document store
#[cfg(feature = "native")]
const DEFAULT_COMPRESSION_LEVEL: CompressionLevel = CompressionLevel(7);

/// Binary document format:
///   num_fields: u16
///   per field: field_id: u16, type_tag: u8, value data
///     0=Text:         len:u32 + utf8
///     1=U64:          u64 LE
///     2=I64:          i64 LE
///     3=F64:          f64 LE
///     4=Bytes:        len:u32 + raw
///     5=SparseVector: count:u32 + count*(u32+f32)
///     6=DenseVector:  count:u32 + count*f32
///     7=Json:         len:u32 + json utf8
pub fn serialize_document(doc: &Document, schema: &Schema) -> io::Result<Vec<u8>> {
    use crate::dsl::FieldValue;

    let stored: Vec<_> = doc
        .field_values()
        .iter()
        .filter(|(field, value)| {
            // Dense vectors live in .vectors (LazyFlatVectorData), not in .store
            if matches!(value, crate::dsl::FieldValue::DenseVector(_)) {
                return false;
            }
            schema.get_field_entry(*field).is_some_and(|e| e.stored)
        })
        .collect();

    let mut buf = Vec::with_capacity(256);
    buf.write_u16::<LittleEndian>(stored.len() as u16)?;

    for (field, value) in &stored {
        buf.write_u16::<LittleEndian>(field.0 as u16)?;
        match value {
            FieldValue::Text(s) => {
                buf.push(0);
                let bytes = s.as_bytes();
                buf.write_u32::<LittleEndian>(bytes.len() as u32)?;
                buf.extend_from_slice(bytes);
            }
            FieldValue::U64(v) => {
                buf.push(1);
                buf.write_u64::<LittleEndian>(*v)?;
            }
            FieldValue::I64(v) => {
                buf.push(2);
                buf.write_i64::<LittleEndian>(*v)?;
            }
            FieldValue::F64(v) => {
                buf.push(3);
                buf.write_f64::<LittleEndian>(*v)?;
            }
            FieldValue::Bytes(b) => {
                buf.push(4);
                buf.write_u32::<LittleEndian>(b.len() as u32)?;
                buf.extend_from_slice(b);
            }
            FieldValue::SparseVector(entries) => {
                buf.push(5);
                buf.write_u32::<LittleEndian>(entries.len() as u32)?;
                for (idx, val) in entries {
                    buf.write_u32::<LittleEndian>(*idx)?;
                    buf.write_f32::<LittleEndian>(*val)?;
                }
            }
            FieldValue::DenseVector(values) => {
                buf.push(6);
                buf.write_u32::<LittleEndian>(values.len() as u32)?;
                // Write raw f32 bytes directly
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
                };
                buf.extend_from_slice(byte_slice);
            }
            FieldValue::Json(v) => {
                buf.push(7);
                let json_bytes = serde_json::to_vec(v)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                buf.write_u32::<LittleEndian>(json_bytes.len() as u32)?;
                buf.extend_from_slice(&json_bytes);
            }
        }
    }

    Ok(buf)
}

/// Compressed block result
#[cfg(feature = "native")]
struct CompressedBlock {
    seq: usize,
    first_doc_id: DocId,
    num_docs: u32,
    compressed: Vec<u8>,
}

/// Parallel document store writer - compresses blocks immediately when queued
///
/// Spawns compression tasks as soon as blocks are ready, overlapping document
/// ingestion with compression to reduce total indexing time.
///
/// Uses background threads to compress blocks while the main thread continues
/// accepting documents.
#[cfg(feature = "native")]
pub struct EagerParallelStoreWriter<'a> {
    writer: &'a mut dyn Write,
    block_buffer: Vec<u8>,
    /// Compressed blocks ready to be written (may arrive out of order)
    compressed_blocks: Vec<CompressedBlock>,
    /// Handles for in-flight compression tasks
    pending_handles: Vec<std::thread::JoinHandle<CompressedBlock>>,
    next_seq: usize,
    next_doc_id: DocId,
    block_first_doc: DocId,
    dict: Option<Arc<CompressionDict>>,
    compression_level: CompressionLevel,
}

#[cfg(feature = "native")]
impl<'a> EagerParallelStoreWriter<'a> {
    /// Create a new eager parallel store writer
    pub fn new(writer: &'a mut dyn Write, _num_threads: usize) -> Self {
        Self::with_compression_level(writer, _num_threads, DEFAULT_COMPRESSION_LEVEL)
    }

    /// Create with specific compression level
    pub fn with_compression_level(
        writer: &'a mut dyn Write,
        _num_threads: usize,
        compression_level: CompressionLevel,
    ) -> Self {
        Self {
            writer,
            block_buffer: Vec::with_capacity(STORE_BLOCK_SIZE),
            compressed_blocks: Vec::new(),
            pending_handles: Vec::new(),
            next_seq: 0,
            next_doc_id: 0,
            block_first_doc: 0,
            dict: None,
            compression_level,
        }
    }

    /// Create with dictionary
    pub fn with_dict(
        writer: &'a mut dyn Write,
        dict: CompressionDict,
        _num_threads: usize,
    ) -> Self {
        Self::with_dict_and_level(writer, dict, _num_threads, DEFAULT_COMPRESSION_LEVEL)
    }

    /// Create with dictionary and specific compression level
    pub fn with_dict_and_level(
        writer: &'a mut dyn Write,
        dict: CompressionDict,
        _num_threads: usize,
        compression_level: CompressionLevel,
    ) -> Self {
        Self {
            writer,
            block_buffer: Vec::with_capacity(STORE_BLOCK_SIZE),
            compressed_blocks: Vec::new(),
            pending_handles: Vec::new(),
            next_seq: 0,
            next_doc_id: 0,
            block_first_doc: 0,
            dict: Some(Arc::new(dict)),
            compression_level,
        }
    }

    pub fn store(&mut self, doc: &Document, schema: &Schema) -> io::Result<DocId> {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        let doc_bytes = serialize_document(doc, schema)?;

        self.block_buffer
            .write_u32::<LittleEndian>(doc_bytes.len() as u32)?;
        self.block_buffer.extend_from_slice(&doc_bytes);

        if self.block_buffer.len() >= STORE_BLOCK_SIZE {
            self.spawn_compression();
        }

        Ok(doc_id)
    }

    /// Spawn compression for the current block immediately
    fn spawn_compression(&mut self) {
        if self.block_buffer.is_empty() {
            return;
        }

        let num_docs = self.next_doc_id - self.block_first_doc;
        let data = std::mem::replace(&mut self.block_buffer, Vec::with_capacity(STORE_BLOCK_SIZE));
        let seq = self.next_seq;
        let first_doc_id = self.block_first_doc;
        let dict = self.dict.clone();

        self.next_seq += 1;
        self.block_first_doc = self.next_doc_id;

        // Spawn compression task using thread
        let level = self.compression_level;
        let handle = std::thread::spawn(move || {
            let compressed = if let Some(ref d) = dict {
                crate::compression::compress_with_dict(&data, level, d).expect("compression failed")
            } else {
                crate::compression::compress(&data, level).expect("compression failed")
            };

            CompressedBlock {
                seq,
                first_doc_id,
                num_docs,
                compressed,
            }
        });

        self.pending_handles.push(handle);
    }

    /// Collect any completed compression tasks
    fn collect_completed(&mut self) {
        let mut remaining = Vec::new();
        for handle in self.pending_handles.drain(..) {
            if handle.is_finished() {
                if let Ok(block) = handle.join() {
                    self.compressed_blocks.push(block);
                }
            } else {
                remaining.push(handle);
            }
        }
        self.pending_handles = remaining;
    }

    pub fn finish(mut self) -> io::Result<u32> {
        // Spawn compression for any remaining data
        self.spawn_compression();

        // Collect any already-completed tasks
        self.collect_completed();

        // Wait for all remaining compression tasks
        for handle in self.pending_handles.drain(..) {
            if let Ok(block) = handle.join() {
                self.compressed_blocks.push(block);
            }
        }

        if self.compressed_blocks.is_empty() {
            write_store_index_and_footer(&mut self.writer, &[], 0, 0, 0, false)?;
            return Ok(0);
        }

        // Sort by sequence to maintain order
        self.compressed_blocks.sort_by_key(|b| b.seq);

        // Write blocks in order and build index
        let mut index = Vec::with_capacity(self.compressed_blocks.len());
        let mut current_offset = 0u64;

        for block in &self.compressed_blocks {
            index.push(StoreBlockIndex {
                first_doc_id: block.first_doc_id,
                offset: current_offset,
                length: block.compressed.len() as u32,
                num_docs: block.num_docs,
            });

            self.writer.write_all(&block.compressed)?;
            current_offset += block.compressed.len() as u64;
        }

        let data_end_offset = current_offset;

        // Write dictionary if present
        let dict_offset = if let Some(ref dict) = self.dict {
            let offset = current_offset;
            let dict_bytes = dict.as_bytes();
            self.writer
                .write_u32::<LittleEndian>(dict_bytes.len() as u32)?;
            self.writer.write_all(dict_bytes)?;
            Some(offset)
        } else {
            None
        };

        // Write index + footer
        write_store_index_and_footer(
            &mut self.writer,
            &index,
            data_end_offset,
            dict_offset.unwrap_or(0),
            self.next_doc_id,
            self.dict.is_some(),
        )?;

        Ok(self.next_doc_id)
    }
}

/// Block index entry for document store
#[derive(Debug, Clone)]
pub(crate) struct StoreBlockIndex {
    pub(crate) first_doc_id: DocId,
    pub(crate) offset: u64,
    pub(crate) length: u32,
    pub(crate) num_docs: u32,
}

/// Async document store reader - loads blocks on demand
pub struct AsyncStoreReader {
    /// LazyFileSlice for the data portion - fetches ranges on demand
    data_slice: LazyFileSlice,
    /// Block index
    index: Vec<StoreBlockIndex>,
    num_docs: u32,
    /// Optional compression dictionary
    dict: Option<CompressionDict>,
    /// Block cache
    cache: RwLock<StoreBlockCache>,
}

/// FIFO block cache — O(1) lookup, insert, and eviction.
///
/// Uses VecDeque for eviction order (pop_front = O(1)) instead of
/// Vec::remove(0) which is O(n). get() is &self for read-lock compatibility.
struct StoreBlockCache {
    blocks: FxHashMap<DocId, Arc<Vec<u8>>>,
    insert_order: std::collections::VecDeque<DocId>,
    max_blocks: usize,
}

impl StoreBlockCache {
    fn new(max_blocks: usize) -> Self {
        Self {
            blocks: FxHashMap::default(),
            insert_order: std::collections::VecDeque::with_capacity(max_blocks),
            max_blocks,
        }
    }

    fn get(&self, first_doc_id: DocId) -> Option<Arc<Vec<u8>>> {
        self.blocks.get(&first_doc_id).map(Arc::clone)
    }

    fn insert(&mut self, first_doc_id: DocId, block: Arc<Vec<u8>>) {
        if self.blocks.contains_key(&first_doc_id) {
            return; // already cached
        }
        while self.blocks.len() >= self.max_blocks {
            if let Some(evict) = self.insert_order.pop_front() {
                self.blocks.remove(&evict);
            } else {
                break;
            }
        }
        self.blocks.insert(first_doc_id, block);
        self.insert_order.push_back(first_doc_id);
    }
}

impl AsyncStoreReader {
    /// Open a document store from LazyFileHandle
    /// Only loads footer and index into memory, data blocks are fetched on-demand
    pub async fn open(file_handle: LazyFileHandle, cache_blocks: usize) -> io::Result<Self> {
        let file_len = file_handle.len();
        // Footer: data_end(8) + dict_offset(8) + num_docs(4) + has_dict(4) + version(4) + magic(4) = 32 bytes
        if file_len < 32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Store too small",
            ));
        }

        // Read footer (32 bytes)
        let footer = file_handle
            .read_bytes_range(file_len - 32..file_len)
            .await?;
        let mut reader = footer.as_slice();
        let data_end_offset = reader.read_u64::<LittleEndian>()?;
        let dict_offset = reader.read_u64::<LittleEndian>()?;
        let num_docs = reader.read_u32::<LittleEndian>()?;
        let has_dict = reader.read_u32::<LittleEndian>()? != 0;
        let version = reader.read_u32::<LittleEndian>()?;
        let magic = reader.read_u32::<LittleEndian>()?;

        if magic != STORE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid store magic",
            ));
        }
        if version != STORE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported store version: {}", version),
            ));
        }

        // Load dictionary if present
        let dict = if has_dict && dict_offset > 0 {
            let dict_start = dict_offset;
            let dict_len_bytes = file_handle
                .read_bytes_range(dict_start..dict_start + 4)
                .await?;
            let dict_len = (&dict_len_bytes[..]).read_u32::<LittleEndian>()? as u64;
            let dict_bytes = file_handle
                .read_bytes_range(dict_start + 4..dict_start + 4 + dict_len)
                .await?;
            Some(CompressionDict::from_bytes(dict_bytes.to_vec()))
        } else {
            None
        };

        // Calculate index location
        let index_start = if has_dict && dict_offset > 0 {
            let dict_start = dict_offset;
            let dict_len_bytes = file_handle
                .read_bytes_range(dict_start..dict_start + 4)
                .await?;
            let dict_len = (&dict_len_bytes[..]).read_u32::<LittleEndian>()? as u64;
            dict_start + 4 + dict_len
        } else {
            data_end_offset
        };
        let index_end = file_len - 32;

        let index_bytes = file_handle.read_bytes_range(index_start..index_end).await?;
        let mut reader = index_bytes.as_slice();

        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;
        let mut index = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            let first_doc_id = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u64::<LittleEndian>()?;
            let length = reader.read_u32::<LittleEndian>()?;
            let num_docs_in_block = reader.read_u32::<LittleEndian>()?;

            index.push(StoreBlockIndex {
                first_doc_id,
                offset,
                length,
                num_docs: num_docs_in_block,
            });
        }

        // Create lazy slice for data portion only
        let data_slice = file_handle.slice(0..data_end_offset);

        Ok(Self {
            data_slice,
            index,
            num_docs,
            dict,
            cache: RwLock::new(StoreBlockCache::new(cache_blocks)),
        })
    }

    /// Number of documents
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Number of blocks currently in the cache
    pub fn cached_blocks(&self) -> usize {
        self.cache.read().blocks.len()
    }

    /// Get a document by doc_id (async - may load block)
    pub async fn get(&self, doc_id: DocId, schema: &Schema) -> io::Result<Option<Document>> {
        if doc_id >= self.num_docs {
            return Ok(None);
        }

        // Find block containing this doc_id
        let block_idx = self
            .index
            .binary_search_by(|entry| {
                if doc_id < entry.first_doc_id {
                    std::cmp::Ordering::Greater
                } else if doc_id >= entry.first_doc_id + entry.num_docs {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Doc not found in index"))?;

        let entry = &self.index[block_idx];
        let block_data = self.load_block(entry).await?;

        // Find document within block
        let doc_offset_in_block = doc_id - entry.first_doc_id;
        let mut reader = &block_data[..];

        for _ in 0..doc_offset_in_block {
            let doc_len = reader.read_u32::<LittleEndian>()? as usize;
            if doc_len > reader.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid doc length",
                ));
            }
            reader = &reader[doc_len..];
        }

        let doc_len = reader.read_u32::<LittleEndian>()? as usize;
        let doc_bytes = &reader[..doc_len];

        deserialize_document(doc_bytes, schema).map(Some)
    }

    async fn load_block(&self, entry: &StoreBlockIndex) -> io::Result<Arc<Vec<u8>>> {
        // Check cache first (read lock — concurrent cache hits don't serialize)
        {
            if let Some(block) = self.cache.read().get(entry.first_doc_id) {
                return Ok(block);
            }
        }

        // Load from FileSlice
        let start = entry.offset;
        let end = start + entry.length as u64;
        let compressed = self.data_slice.read_bytes_range(start..end).await?;

        // Use dictionary decompression if available
        let decompressed = if let Some(ref dict) = self.dict {
            crate::compression::decompress_with_dict(compressed.as_slice(), dict)?
        } else {
            crate::compression::decompress(compressed.as_slice())?
        };

        let block = Arc::new(decompressed);

        // Insert into cache
        {
            let mut cache = self.cache.write();
            cache.insert(entry.first_doc_id, Arc::clone(&block));
        }

        Ok(block)
    }
}

pub fn deserialize_document(data: &[u8], _schema: &Schema) -> io::Result<Document> {
    use crate::dsl::Field;

    let mut reader = data;
    let num_fields = reader.read_u16::<LittleEndian>()? as usize;
    let mut doc = Document::new();

    for _ in 0..num_fields {
        let field_id = reader.read_u16::<LittleEndian>()?;
        let field = Field(field_id as u32);
        let type_tag = reader.read_u8()?;

        match type_tag {
            0 => {
                // Text
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let s = std::str::from_utf8(&reader[..len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                doc.add_text(field, s);
                reader = &reader[len..];
            }
            1 => {
                // U64
                doc.add_u64(field, reader.read_u64::<LittleEndian>()?);
            }
            2 => {
                // I64
                doc.add_i64(field, reader.read_i64::<LittleEndian>()?);
            }
            3 => {
                // F64
                doc.add_f64(field, reader.read_f64::<LittleEndian>()?);
            }
            4 => {
                // Bytes
                let len = reader.read_u32::<LittleEndian>()? as usize;
                doc.add_bytes(field, reader[..len].to_vec());
                reader = &reader[len..];
            }
            5 => {
                // SparseVector
                let count = reader.read_u32::<LittleEndian>()? as usize;
                let mut entries = Vec::with_capacity(count);
                for _ in 0..count {
                    let idx = reader.read_u32::<LittleEndian>()?;
                    let val = reader.read_f32::<LittleEndian>()?;
                    entries.push((idx, val));
                }
                doc.add_sparse_vector(field, entries);
            }
            6 => {
                // DenseVector
                let count = reader.read_u32::<LittleEndian>()? as usize;
                let byte_len = count * 4;
                if reader.len() < byte_len {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "dense vector field {}: need {} bytes but only {} remain",
                            field.0,
                            byte_len,
                            reader.len()
                        ),
                    ));
                }
                let mut values = vec![0.0f32; count];
                // Read raw f32 bytes directly
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        reader.as_ptr(),
                        values.as_mut_ptr() as *mut u8,
                        byte_len,
                    );
                }
                reader = &reader[byte_len..];
                doc.add_dense_vector(field, values);
            }
            7 => {
                // Json
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let v: serde_json::Value = serde_json::from_slice(&reader[..len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                doc.add_json(field, v);
                reader = &reader[len..];
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown field type tag: {}", type_tag),
                ));
            }
        }
    }

    Ok(doc)
}

/// Raw block info for store merging (without decompression)
#[derive(Debug, Clone)]
pub struct RawStoreBlock {
    pub first_doc_id: DocId,
    pub num_docs: u32,
    pub offset: u64,
    pub length: u32,
}

/// Store merger - concatenates compressed blocks from multiple stores without recompression
///
/// This is much faster than rebuilding stores since it avoids:
/// - Decompressing blocks from source stores
/// - Re-serializing documents
/// - Re-compressing blocks at level 22
///
/// Limitations:
/// - All source stores must NOT use dictionaries (or use the same dictionary)
/// - Doc IDs are remapped sequentially
pub struct StoreMerger<'a, W: Write> {
    writer: &'a mut W,
    index: Vec<StoreBlockIndex>,
    current_offset: u64,
    next_doc_id: DocId,
}

impl<'a, W: Write> StoreMerger<'a, W> {
    pub fn new(writer: &'a mut W) -> Self {
        Self {
            writer,
            index: Vec::new(),
            current_offset: 0,
            next_doc_id: 0,
        }
    }

    /// Append raw compressed blocks from a store file
    ///
    /// `data_slice` should be the data portion of the store (before index/footer)
    /// `blocks` contains the block metadata from the source store
    pub async fn append_store<F: AsyncFileRead>(
        &mut self,
        data_slice: &F,
        blocks: &[RawStoreBlock],
    ) -> io::Result<()> {
        for block in blocks {
            // Read raw compressed block data
            let start = block.offset;
            let end = start + block.length as u64;
            let compressed_data = data_slice.read_bytes_range(start..end).await?;

            // Write to output
            self.writer.write_all(compressed_data.as_slice())?;

            // Add to index with remapped doc IDs
            self.index.push(StoreBlockIndex {
                first_doc_id: self.next_doc_id,
                offset: self.current_offset,
                length: block.length,
                num_docs: block.num_docs,
            });

            self.current_offset += block.length as u64;
            self.next_doc_id += block.num_docs;
        }

        Ok(())
    }

    /// Append blocks from a dict-compressed store by decompressing and recompressing.
    ///
    /// For stores that use dictionary compression, raw blocks can't be stacked
    /// directly because the decompressor needs the original dictionary.
    /// This method decompresses each block with the source dict, then
    /// recompresses without a dictionary so the merged output is self-contained.
    pub async fn append_store_recompressing(&mut self, store: &AsyncStoreReader) -> io::Result<()> {
        let dict = store.dict();
        let data_slice = store.data_slice();
        let blocks = store.block_index();

        for block in blocks {
            let start = block.offset;
            let end = start + block.length as u64;
            let compressed = data_slice.read_bytes_range(start..end).await?;

            // Decompress with source dict (or without if no dict)
            let decompressed = if let Some(d) = dict {
                crate::compression::decompress_with_dict(compressed.as_slice(), d)?
            } else {
                crate::compression::decompress(compressed.as_slice())?
            };

            // Recompress without dictionary
            let recompressed = crate::compression::compress(
                &decompressed,
                crate::compression::CompressionLevel::default(),
            )?;

            self.writer.write_all(&recompressed)?;

            self.index.push(StoreBlockIndex {
                first_doc_id: self.next_doc_id,
                offset: self.current_offset,
                length: recompressed.len() as u32,
                num_docs: block.num_docs,
            });

            self.current_offset += recompressed.len() as u64;
            self.next_doc_id += block.num_docs;
        }

        Ok(())
    }

    /// Finish writing the merged store
    pub fn finish(self) -> io::Result<u32> {
        let data_end_offset = self.current_offset;

        // No dictionary support for merged stores (would need same dict across all sources)
        let dict_offset = 0u64;

        // Write index + footer
        write_store_index_and_footer(
            self.writer,
            &self.index,
            data_end_offset,
            dict_offset,
            self.next_doc_id,
            false,
        )?;

        Ok(self.next_doc_id)
    }
}

impl AsyncStoreReader {
    /// Get raw block metadata for merging (without loading block data)
    pub fn raw_blocks(&self) -> Vec<RawStoreBlock> {
        self.index
            .iter()
            .map(|entry| RawStoreBlock {
                first_doc_id: entry.first_doc_id,
                num_docs: entry.num_docs,
                offset: entry.offset,
                length: entry.length,
            })
            .collect()
    }

    /// Get the data slice for raw block access
    pub fn data_slice(&self) -> &LazyFileSlice {
        &self.data_slice
    }

    /// Check if this store uses a dictionary (incompatible with raw merging)
    pub fn has_dict(&self) -> bool {
        self.dict.is_some()
    }

    /// Get the decompression dictionary (if any)
    pub fn dict(&self) -> Option<&CompressionDict> {
        self.dict.as_ref()
    }

    /// Get block index for iteration
    pub(crate) fn block_index(&self) -> &[StoreBlockIndex] {
        &self.index
    }
}
