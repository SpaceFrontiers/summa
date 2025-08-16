//! Document store with Zstd compression and lazy loading
//!
//! Optimized for static indexes:
//! - Maximum compression level (22) for best compression ratio
//! - Larger block sizes (256KB) for better compression efficiency
//! - Optional trained dictionary support for even better compression
//! - Parallel compression support for faster indexing
//!
//! Writer stores documents in compressed blocks.
//! Reader only loads index into memory, blocks are loaded on-demand.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
#[cfg(feature = "native")]
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};
use std::sync::Arc;
#[cfg(feature = "native")]
use std::sync::OnceLock;
#[cfg(feature = "native")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "native")]
use std::sync::mpsc::{Receiver, Sender, SyncSender};
#[cfg(feature = "native")]
use std::thread::JoinHandle;

use crate::DocId;
use crate::compression::CompressionDict;
#[cfg(feature = "native")]
use crate::compression::CompressionLevel;
use crate::directories::FileHandle;
use crate::dsl::{Document, Schema};

const STORE_MAGIC: u32 = 0x53544F52; // "STOR"
/// Version 3 widens each document's stored field-value count from u16 to u32.
const STORE_VERSION: u32 = 3;

/// Block size for document store (16KB).
/// Smaller blocks reduce read amplification for single-doc fetches at the
/// cost of slightly worse compression ratio. Zstd dictionary training
/// recovers most of the compression loss.
pub const STORE_BLOCK_SIZE: usize = 16 * 1024;

/// Default dictionary size (4KB is a good balance)
pub const DEFAULT_DICT_SIZE: usize = 4 * 1024;

/// Hard safety bounds for individual on-disk store objects. Writers normally
/// emit ~16 KiB blocks and 4 KiB dictionaries; these generous limits preserve
/// unusually large stored documents while bounding corrupt compressed frames.
const MAX_STORE_BLOCK_BYTES: usize = 256 * 1024 * 1024;
const MAX_STORE_DICTIONARY_BYTES: u64 = 16 * 1024 * 1024;

/// Default compression level for document store
#[cfg(feature = "native")]
const DEFAULT_COMPRESSION_LEVEL: CompressionLevel = CompressionLevel(3);

/// Write block index + footer to a store file.
///
/// Shared by `EagerParallelStoreWriter::finish` and `StoreMerger::finish`.
fn write_store_index_and_footer(
    writer: &mut (impl Write + ?Sized),
    index: &[StoreBlockIndex],
    data_end_offset: u64,
    dict_offset: u64,
    num_docs: u32,
    has_dict: bool,
) -> io::Result<()> {
    writer.write_u32::<LittleEndian>(u32::try_from(index.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "too many document store blocks",
        )
    })?)?;
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

/// Binary document format:
///   num_fields: u32
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
    let mut buf = Vec::with_capacity(256);
    serialize_document_into(doc, schema, &mut buf)?;
    Ok(buf)
}

/// Serialize a document into a reusable buffer (clears it first).
/// Avoids per-document allocation when called in a loop.
pub fn serialize_document_into(
    doc: &Document,
    schema: &Schema,
    buf: &mut Vec<u8>,
) -> io::Result<()> {
    use crate::dsl::FieldValue;

    buf.clear();

    // Two-pass approach avoids allocating a Vec just to count + iterate stored fields.
    let is_stored = |field: &crate::dsl::Field, value: &FieldValue| -> bool {
        // Dense/binary vectors live in .vectors (LazyFlatVectorData), not in .store
        if matches!(
            value,
            FieldValue::DenseVector(_) | FieldValue::BinaryDenseVector(_)
        ) {
            return false;
        }
        schema.get_field_entry(*field).is_some_and(|e| e.stored)
    };

    let stored_count = doc
        .field_values()
        .iter()
        .filter(|(field, value)| is_stored(field, value))
        .count();

    buf.write_u32::<LittleEndian>(u32::try_from(stored_count).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "too many stored field-values")
    })?)?;

    for (field, value) in doc.field_values().iter().filter(|(f, v)| is_stored(f, v)) {
        buf.write_u16::<LittleEndian>(u16::try_from(field.0).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "stored field id exceeds u16")
        })?)?;
        match value {
            FieldValue::Text(s) => {
                buf.push(0);
                let bytes = s.as_bytes();
                buf.write_u32::<LittleEndian>(u32::try_from(bytes.len()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "stored text is too large")
                })?)?;
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
                buf.write_u32::<LittleEndian>(u32::try_from(b.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "stored byte field is too large",
                    )
                })?)?;
                buf.extend_from_slice(b);
            }
            FieldValue::SparseVector(entries) => {
                buf.push(5);
                buf.write_u32::<LittleEndian>(u32::try_from(entries.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "stored sparse vector is too large",
                    )
                })?)?;
                for (idx, val) in entries {
                    buf.write_u32::<LittleEndian>(*idx)?;
                    buf.write_f32::<LittleEndian>(*val)?;
                }
            }
            FieldValue::DenseVector(values) => {
                buf.push(6);
                buf.write_u32::<LittleEndian>(u32::try_from(values.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "stored dense vector is too large",
                    )
                })?)?;
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
                buf.write_u32::<LittleEndian>(u32::try_from(json_bytes.len()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "stored JSON is too large")
                })?)?;
                buf.extend_from_slice(&json_bytes);
            }
            FieldValue::BinaryDenseVector(b) => {
                buf.push(8);
                buf.write_u32::<LittleEndian>(u32::try_from(b.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "stored binary dense vector is too large",
                    )
                })?)?;
                buf.extend_from_slice(b);
            }
        }
    }

    Ok(())
}

/// Compressed block result
#[cfg(feature = "native")]
struct CompressedBlock {
    seq: usize,
    first_doc_id: DocId,
    num_docs: u32,
    compressed: Vec<u8>,
}

#[cfg(feature = "native")]
struct CompressionJob {
    seq: usize,
    first_doc_id: DocId,
    num_docs: u32,
    data: Vec<u8>,
}

#[cfg(feature = "native")]
struct CompressionRequest {
    job: CompressionJob,
    dict: Option<Arc<CompressionDict>>,
    compression_level: CompressionLevel,
    results: Sender<io::Result<CompressedBlock>>,
    cancelled: Arc<AtomicBool>,
}

/// Process-wide fixed-width compression executor.
///
/// Segment builders with the same configured width share one executor. This
/// keeps simultaneous flushes from multiplying `num_threads` by the number of
/// indexing workers while retaining parallel compression across all stores.
#[cfg(feature = "native")]
struct StoreCompressionExecutor {
    jobs: Option<SyncSender<CompressionRequest>>,
    workers: Vec<JoinHandle<()>>,
    num_threads: usize,
}

#[cfg(feature = "native")]
impl StoreCompressionExecutor {
    fn new(num_threads: usize) -> Self {
        let (job_sender, job_receiver) =
            std::sync::mpsc::sync_channel::<CompressionRequest>(num_threads);
        let job_receiver = Arc::new(std::sync::Mutex::new(job_receiver));
        let mut workers = Vec::with_capacity(num_threads);

        for worker_id in 0..num_threads {
            let jobs = Arc::clone(&job_receiver);
            let worker = std::thread::Builder::new()
                .name(format!("hermes-store-compress-{num_threads}-{worker_id}"))
                .spawn(move || {
                    loop {
                        // `std::sync::mpsc::Receiver` is single-consumer. Keep
                        // only the receive operation under the mutex; workers
                        // compress independently after taking a request.
                        let request = {
                            let receiver = jobs
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            receiver.recv()
                        };
                        let Ok(request) = request else {
                            break;
                        };
                        let CompressionRequest {
                            job,
                            dict,
                            compression_level,
                            results,
                            cancelled,
                        } = request;
                        if cancelled.load(Ordering::Acquire) {
                            continue;
                        }

                        // Always produce exactly one result for every accepted
                        // job. Without catch_unwind, one unexpected codec panic
                        // would leave the ordered producer waiting forever for
                        // the missing sequence number.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let compressed = if let Some(ref dict) = dict {
                                crate::compression::compress_with_dict(
                                    &job.data,
                                    compression_level,
                                    dict,
                                )
                            } else {
                                crate::compression::compress(&job.data, compression_level)
                            }?;
                            Ok(CompressedBlock {
                                seq: job.seq,
                                first_doc_id: job.first_doc_id,
                                num_docs: job.num_docs,
                                compressed,
                            })
                        }))
                        .unwrap_or_else(|_| {
                            Err(io::Error::other(
                                "document-store compression worker panicked",
                            ))
                        });

                        // A writer can disappear after an earlier I/O error.
                        // Its remaining bounded jobs should not stop the shared
                        // executor from serving other segment builders.
                        if !cancelled.load(Ordering::Acquire) {
                            let _ = results.send(result);
                        }
                    }
                })
                .expect("failed to spawn document-store compression worker");
            workers.push(worker);
        }

        Self {
            jobs: Some(job_sender),
            workers,
            num_threads,
        }
    }
}

#[cfg(feature = "native")]
impl Drop for StoreCompressionExecutor {
    fn drop(&mut self) {
        self.jobs.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(feature = "native")]
static STORE_COMPRESSION_EXECUTORS: OnceLock<
    parking_lot::Mutex<HashMap<usize, Arc<StoreCompressionExecutor>>>,
> = OnceLock::new();

#[cfg(feature = "native")]
fn shared_store_compression_executor(num_threads: usize) -> Arc<StoreCompressionExecutor> {
    let num_threads = num_threads.max(1);
    let mut executors = STORE_COMPRESSION_EXECUTORS
        .get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
        .lock();
    if let Some(executor) = executors.get(&num_threads) {
        return Arc::clone(executor);
    }

    let executor = Arc::new(StoreCompressionExecutor::new(num_threads));
    // Keep one process-lifetime owner. A weak-only registry could create a new
    // width-N generation while the previous generation was still joining its
    // N workers after an error, transiently multiplying the promised bound.
    executors.insert(num_threads, Arc::clone(&executor));
    log::info!(
        "[store] process-wide compression pool: {} thread(s)",
        num_threads
    );
    executor
}

/// Per-writer handle into the shared fixed-width compression executor.
///
/// The producer separately limits the number of submitted-but-not-written
/// blocks. The bounded job channel provides backpressure as a second line of
/// defence and prevents a fast store-file scan from creating an unbounded task
/// queue.
#[cfg(feature = "native")]
struct StoreCompressionPool {
    executor: Arc<StoreCompressionExecutor>,
    results: Receiver<io::Result<CompressedBlock>>,
    result_sender: Sender<io::Result<CompressedBlock>>,
    dict: Option<Arc<CompressionDict>>,
    compression_level: CompressionLevel,
    cancelled: Arc<AtomicBool>,
}

#[cfg(feature = "native")]
impl StoreCompressionPool {
    fn new(
        num_threads: usize,
        dict: Option<Arc<CompressionDict>>,
        compression_level: CompressionLevel,
    ) -> Self {
        // A zero-width pool can never make progress. Treat zero as the
        // conservative single-worker setting; the public configuration did
        // not historically reject zero.
        let num_threads = num_threads.max(1);
        let (result_sender, result_receiver) = std::sync::mpsc::channel();

        Self {
            executor: shared_store_compression_executor(num_threads),
            results: result_receiver,
            result_sender,
            dict,
            compression_level,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    #[inline]
    fn num_threads(&self) -> usize {
        self.executor.num_threads
    }

    fn submit(&self, job: CompressionJob) -> io::Result<()> {
        self.executor
            .jobs
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "compression pool is closed"))?
            .send(CompressionRequest {
                job,
                dict: self.dict.clone(),
                compression_level: self.compression_level,
                results: self.result_sender.clone(),
                cancelled: Arc::clone(&self.cancelled),
            })
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "document-store compression workers stopped",
                )
            })
    }

    fn receive(&self) -> io::Result<CompressedBlock> {
        self.results.recv().map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "document-store compression result channel closed",
            )
        })?
    }

    fn shutdown(&mut self) -> io::Result<()> {
        // All jobs for this writer have yielded results before the normal
        // finish path reaches here. The process-wide executor stays available
        // to other concurrent segment builders.
        self.cancelled.store(true, Ordering::Release);
        Ok(())
    }
}

#[cfg(feature = "native")]
impl Drop for StoreCompressionPool {
    fn drop(&mut self) {
        // Requests not yet started are skipped by shared workers. A block
        // already inside Zstd completes, but it no longer publishes a result
        // into the abandoned writer's channel.
        self.cancelled.store(true, Ordering::Release);
    }
}

/// Parallel document store writer - compresses blocks immediately when queued
///
/// A fixed number of workers compress blocks while the caller continues
/// accepting documents. Submitted blocks and out-of-order results are bounded,
/// and completed blocks are written continuously in sequence order.
#[cfg(feature = "native")]
pub struct EagerParallelStoreWriter<'a> {
    writer: &'a mut dyn Write,
    block_buffer: Vec<u8>,
    /// Reusable buffer for document serialization (avoids per-doc allocation)
    serialize_buf: Vec<u8>,
    workers: StoreCompressionPool,
    /// Completed blocks waiting for an earlier sequence number.
    ready_blocks: BTreeMap<usize, CompressedBlock>,
    /// Number of submitted jobs whose result has not yet been received.
    pending_results: usize,
    /// Maximum submitted-but-not-written blocks.
    max_in_flight: usize,
    next_seq: usize,
    next_write_seq: usize,
    next_doc_id: DocId,
    block_first_doc: DocId,
    index: Vec<StoreBlockIndex>,
    current_offset: u64,
    dict: Option<Arc<CompressionDict>>,
}

#[cfg(feature = "native")]
impl<'a> EagerParallelStoreWriter<'a> {
    /// Create a new eager parallel store writer
    pub fn new(writer: &'a mut dyn Write, num_threads: usize) -> Self {
        Self::with_compression_level(writer, num_threads, DEFAULT_COMPRESSION_LEVEL)
    }

    /// Create with specific compression level
    pub fn with_compression_level(
        writer: &'a mut dyn Write,
        num_threads: usize,
        compression_level: CompressionLevel,
    ) -> Self {
        Self::with_optional_dict(writer, None, num_threads, compression_level)
    }

    /// Create with dictionary
    pub fn with_dict(writer: &'a mut dyn Write, dict: CompressionDict, num_threads: usize) -> Self {
        Self::with_dict_and_level(writer, dict, num_threads, DEFAULT_COMPRESSION_LEVEL)
    }

    /// Create with dictionary and specific compression level
    pub fn with_dict_and_level(
        writer: &'a mut dyn Write,
        dict: CompressionDict,
        num_threads: usize,
        compression_level: CompressionLevel,
    ) -> Self {
        Self::with_optional_dict(writer, Some(Arc::new(dict)), num_threads, compression_level)
    }

    fn with_optional_dict(
        writer: &'a mut dyn Write,
        dict: Option<Arc<CompressionDict>>,
        num_threads: usize,
        compression_level: CompressionLevel,
    ) -> Self {
        let workers = StoreCompressionPool::new(num_threads, dict.clone(), compression_level);
        // One active block plus one queued block per worker keeps the pool
        // saturated without retaining the complete compressed store.
        let max_in_flight = workers.num_threads().saturating_mul(2).max(1);
        Self {
            writer,
            block_buffer: Vec::with_capacity(STORE_BLOCK_SIZE),
            serialize_buf: Vec::with_capacity(512),
            workers,
            ready_blocks: BTreeMap::new(),
            pending_results: 0,
            max_in_flight,
            next_seq: 0,
            next_write_seq: 0,
            next_doc_id: 0,
            block_first_doc: 0,
            index: Vec::new(),
            current_offset: 0,
            dict,
        }
    }

    pub fn store(&mut self, doc: &Document, schema: &Schema) -> io::Result<DocId> {
        serialize_document_into(doc, schema, &mut self.serialize_buf)?;
        if self.serialize_buf.len() > MAX_STORE_BLOCK_BYTES.saturating_sub(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "serialized document exceeds store block limit",
            ));
        }
        let doc_id = self.next_doc_id;
        self.next_doc_id = self
            .next_doc_id
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "document id overflow"))?;
        self.block_buffer
            .write_u32::<LittleEndian>(self.serialize_buf.len() as u32)?;
        self.block_buffer.extend_from_slice(&self.serialize_buf);
        if self.block_buffer.len() >= STORE_BLOCK_SIZE {
            self.queue_compression()?;
        }
        Ok(doc_id)
    }

    /// Store pre-serialized document bytes directly (avoids deserialize+reserialize roundtrip).
    pub fn store_raw(&mut self, doc_bytes: &[u8]) -> io::Result<DocId> {
        if doc_bytes.len() > MAX_STORE_BLOCK_BYTES.saturating_sub(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "serialized document exceeds store block limit",
            ));
        }
        let doc_id = self.next_doc_id;
        self.next_doc_id = self
            .next_doc_id
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "document id overflow"))?;

        self.block_buffer
            .write_u32::<LittleEndian>(doc_bytes.len() as u32)?;
        self.block_buffer.extend_from_slice(doc_bytes);

        if self.block_buffer.len() >= STORE_BLOCK_SIZE {
            self.queue_compression()?;
        }

        Ok(doc_id)
    }

    /// Queue the current block and apply ordered backpressure when the bounded
    /// in-flight window is full.
    fn queue_compression(&mut self) -> io::Result<()> {
        if self.block_buffer.is_empty() {
            return Ok(());
        }

        let num_docs = self.next_doc_id - self.block_first_doc;
        let data = std::mem::replace(&mut self.block_buffer, Vec::with_capacity(STORE_BLOCK_SIZE));
        let seq = self.next_seq;
        let first_doc_id = self.block_first_doc;

        self.workers.submit(CompressionJob {
            seq,
            first_doc_id,
            num_docs,
            data,
        })?;
        self.next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "store block overflow"))?;
        self.pending_results += 1;
        self.block_first_doc = self.next_doc_id;

        while self.outstanding_blocks() >= self.max_in_flight {
            self.receive_and_write_ready()?;
        }

        Ok(())
    }

    #[inline]
    fn outstanding_blocks(&self) -> usize {
        self.next_seq - self.next_write_seq
    }

    fn receive_and_write_ready(&mut self) -> io::Result<()> {
        if self.pending_results == 0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "missing document-store compression result",
            ));
        }
        let block = self.workers.receive()?;
        self.pending_results -= 1;
        if block.seq < self.next_write_seq || self.ready_blocks.insert(block.seq, block).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "duplicate document-store compression result",
            ));
        }
        self.write_ready_blocks()
    }

    fn write_ready_blocks(&mut self) -> io::Result<()> {
        while let Some(block) = self.ready_blocks.remove(&self.next_write_seq) {
            let length = u32::try_from(block.compressed.len()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "compressed store block too large",
                )
            })?;
            self.writer.write_all(&block.compressed)?;
            self.index.push(StoreBlockIndex {
                first_doc_id: block.first_doc_id,
                offset: self.current_offset,
                length,
                num_docs: block.num_docs,
            });
            self.current_offset = self
                .current_offset
                .checked_add(u64::from(length))
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "store size overflow"))?;
            self.next_write_seq += 1;
        }
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<u32> {
        // Queue any remaining data, then receive and stream every bounded
        // out-of-order result before writing the footer.
        self.queue_compression()?;
        while self.next_write_seq < self.next_seq {
            self.receive_and_write_ready()?;
        }
        debug_assert_eq!(self.pending_results, 0);
        debug_assert!(self.ready_blocks.is_empty());
        self.workers.shutdown()?;

        if self.index.is_empty() {
            write_store_index_and_footer(&mut self.writer, &[], 0, 0, 0, false)?;
            return Ok(0);
        }

        // Write dictionary if present
        let dict_offset = if let Some(ref dict) = self.dict {
            let offset = self.current_offset;
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
            &self.index,
            self.current_offset,
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
    /// FileHandle for the data portion - fetches ranges on demand
    data_slice: FileHandle,
    /// Block index
    index: Vec<StoreBlockIndex>,
    num_docs: u32,
    /// Optional compression dictionary
    dict: Option<CompressionDict>,
    /// Process-wide byte-bounded block cache.
    cache: Arc<SharedStoreCache>,
    /// Stable directory + segment namespace for shared-cache keys.
    cache_namespace: StoreCacheNamespace,
}

/// Decompressed block with pre-built doc offset table.
///
/// The offset table is built once on decompression: `offsets[i]` is the byte
/// position in `data` where doc `i`'s length prefix starts. This turns the
/// O(n) linear scan per `get()` into O(1) direct indexing.
struct CachedBlock {
    data: Vec<u8>,
    /// Byte offset of each doc's length prefix within `data`.
    /// `offsets.len()` == number of docs in the block.
    offsets: Vec<u32>,
}

impl CachedBlock {
    fn build(data: Vec<u8>, num_docs: u32) -> io::Result<Self> {
        if num_docs as usize > data.len() / 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store block document count exceeds block length",
            ));
        }
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(num_docs as usize).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "store block has too many documents",
            )
        })?;
        let mut pos = 0usize;
        for _ in 0..num_docs {
            let length_end = pos.checked_add(4).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "store block offset overflow")
            })?;
            if length_end > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated block while building offset table",
                ));
            }
            offsets.push(u32::try_from(pos).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "store block offset exceeds u32")
            })?);
            let doc_len =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos = length_end.checked_add(doc_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "store document length overflow")
            })?;
            if pos > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "store document is truncated",
                ));
            }
        }
        if pos != data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store block contains trailing data",
            ));
        }
        Ok(Self { data, offsets })
    }

    /// Get doc bytes by index within the block (O(1))
    fn doc_bytes(&self, doc_offset_in_block: u32) -> io::Result<&[u8]> {
        let idx = doc_offset_in_block as usize;
        if idx >= self.offsets.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "doc offset out of range",
            ));
        }
        let start = self.offsets[idx] as usize;
        let data_start = start.checked_add(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "store document offset overflow")
        })?;
        if data_start > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated doc length",
            ));
        }
        let doc_len = u32::from_le_bytes([
            self.data[start],
            self.data[start + 1],
            self.data[start + 2],
            self.data[start + 3],
        ]) as usize;
        let data_end = data_start.checked_add(doc_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "store document length overflow")
        })?;
        if data_end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "doc data overflow",
            ));
        }
        Ok(&self.data[data_start..data_end])
    }

    #[inline]
    fn retained_bytes(&self) -> usize {
        self.data.capacity() + self.offsets.capacity() * std::mem::size_of::<u32>()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StoreCacheNamespace {
    directory: usize,
    segment: u128,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StoreCacheKey {
    namespace: StoreCacheNamespace,
    first_doc_id: DocId,
}

struct SharedStoreCacheState {
    blocks: LruCache<StoreCacheKey, Arc<CachedBlock>>,
    retained_bytes: usize,
    namespace_bytes: FxHashMap<StoreCacheNamespace, usize>,
    namespace_readers: FxHashMap<StoreCacheNamespace, usize>,
}

/// Process-wide byte-bounded cache for decompressed document-store blocks.
///
/// The old cache bounded each segment by an entry count. One large stored
/// body can make a block close to `MAX_STORE_BLOCK_BYTES`, so 32 entries per
/// segment retained up to 2 GiB and multiplied that by segment fan-out. This
/// cache has one hard byte ceiling across all indexes using the same policy.
/// Hits take a shared read lock; eviction is insertion-ordered rather than
/// serializing every hit solely for exact LRU promotion.
pub(crate) struct SharedStoreCache {
    state: RwLock<SharedStoreCacheState>,
    max_bytes: usize,
    /// Very large one-document blocks have almost no spatial reuse and can
    /// evict thousands of ordinary result blocks. The OS compressed-file page
    /// cache remains available when decompressed admission is bypassed.
    max_entry_bytes: usize,
}

impl std::fmt::Debug for SharedStoreCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedStoreCache")
            .field("max_bytes", &self.max_bytes)
            .field("max_entry_bytes", &self.max_entry_bytes)
            .field("retained_bytes", &self.total_bytes())
            .finish()
    }
}

impl SharedStoreCache {
    const MAX_ADMITTED_ENTRY_BYTES: usize = 8 * 1024 * 1024;

    pub(crate) fn new(max_bytes: usize) -> Self {
        Self::with_limits(max_bytes, max_bytes.min(Self::MAX_ADMITTED_ENTRY_BYTES))
    }

    fn with_limits(max_bytes: usize, max_entry_bytes: usize) -> Self {
        Self {
            state: RwLock::new(SharedStoreCacheState {
                blocks: LruCache::unbounded(),
                retained_bytes: 0,
                namespace_bytes: FxHashMap::default(),
                namespace_readers: FxHashMap::default(),
            }),
            max_bytes,
            max_entry_bytes: max_entry_bytes.min(max_bytes),
        }
    }

    fn register(&self, namespace: StoreCacheNamespace) {
        if self.max_bytes == 0 {
            return;
        }
        let mut state = self.state.write();
        *state.namespace_readers.entry(namespace).or_default() += 1;
    }

    fn unregister(&self, namespace: StoreCacheNamespace) {
        if self.max_bytes == 0 {
            return;
        }
        let mut state = self.state.write();
        let Some(readers) = state.namespace_readers.get_mut(&namespace) else {
            return;
        };
        *readers -= 1;
        if *readers > 0 {
            return;
        }
        state.namespace_readers.remove(&namespace);

        // A merged-away segment will never hit these entries again. Remove
        // them immediately instead of waiting for unrelated searches to
        // create enough pressure for ordinary LRU eviction.
        let keys: Vec<_> = state
            .blocks
            .iter()
            .filter_map(|(key, _)| (key.namespace == namespace).then_some(*key))
            .collect();
        for key in keys {
            if let Some(block) = state.blocks.pop(&key) {
                state.retained_bytes = state.retained_bytes.saturating_sub(block.retained_bytes());
            }
        }
        state.namespace_bytes.remove(&namespace);
    }

    fn get(&self, key: StoreCacheKey) -> Option<Arc<CachedBlock>> {
        // Do not serialize all process-wide hits merely to update exact LRU
        // order. Concurrent readers use a shared lock; insertion and
        // decompression-race resolution still promote entries.
        self.state.read().blocks.peek(&key).map(Arc::clone)
    }

    /// Admit one block and return the canonical cached allocation if another
    /// request won the decompression race.
    fn insert(&self, key: StoreCacheKey, block: Arc<CachedBlock>) -> Arc<CachedBlock> {
        let bytes = block.retained_bytes();
        if self.max_bytes == 0 || bytes == 0 || bytes > self.max_entry_bytes {
            return block;
        }

        let mut state = self.state.write();
        if let Some(existing) = state.blocks.get(&key) {
            return Arc::clone(existing);
        }

        state.retained_bytes = state.retained_bytes.saturating_add(bytes);
        *state.namespace_bytes.entry(key.namespace).or_default() = state
            .namespace_bytes
            .get(&key.namespace)
            .copied()
            .unwrap_or(0)
            .saturating_add(bytes);
        state.blocks.put(key, Arc::clone(&block));

        while state.retained_bytes > self.max_bytes {
            let Some((evicted_key, evicted)) = state.blocks.pop_lru() else {
                state.retained_bytes = 0;
                state.namespace_bytes.clear();
                break;
            };
            let evicted_bytes = evicted.retained_bytes();
            state.retained_bytes = state.retained_bytes.saturating_sub(evicted_bytes);
            if let Some(namespace_bytes) = state.namespace_bytes.get_mut(&evicted_key.namespace) {
                *namespace_bytes = namespace_bytes.saturating_sub(evicted_bytes);
                if *namespace_bytes == 0 {
                    state.namespace_bytes.remove(&evicted_key.namespace);
                }
            }
        }
        block
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.state.read().retained_bytes
    }

    pub(crate) fn total_blocks(&self) -> usize {
        self.state.read().blocks.len()
    }

    fn namespace_bytes(&self, namespace: StoreCacheNamespace) -> usize {
        self.state
            .read()
            .namespace_bytes
            .get(&namespace)
            .copied()
            .unwrap_or(0)
    }

    fn namespace_blocks(&self, namespace: StoreCacheNamespace) -> usize {
        self.state
            .read()
            .blocks
            .iter()
            .filter(|(key, _)| key.namespace == namespace)
            .count()
    }
}

impl Drop for AsyncStoreReader {
    fn drop(&mut self) {
        self.cache.unregister(self.cache_namespace);
    }
}

impl AsyncStoreReader {
    /// Open a document store from FileHandle
    /// Only loads footer and index into memory, data blocks are fetched on-demand
    pub(crate) async fn open(
        file_handle: FileHandle,
        directory_namespace: usize,
        segment_namespace: u128,
        cache: Arc<SharedStoreCache>,
    ) -> io::Result<Self> {
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

        let index_end = file_len - 32;
        if data_end_offset > index_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store data section extends past its footer",
            ));
        }

        // Load dictionary if present, and compute index_start in one pass
        let (dict, index_start) = if has_dict {
            if dict_offset < data_end_offset || dict_offset >= index_end {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "store dictionary offset is out of bounds",
                ));
            }
            let dict_start = dict_offset;
            let dict_header_end = dict_start.checked_add(4).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "store dictionary range overflow",
                )
            })?;
            if dict_header_end > index_end {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "store dictionary length is truncated",
                ));
            }
            let dict_len_bytes = file_handle
                .read_bytes_range(dict_start..dict_header_end)
                .await?;
            let dict_len = (&dict_len_bytes[..]).read_u32::<LittleEndian>()? as u64;
            if dict_len > MAX_STORE_DICTIONARY_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "store dictionary exceeds safety limit",
                ));
            }
            let dict_end = dict_header_end.checked_add(dict_len).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "store dictionary range overflow",
                )
            })?;
            if dict_end > index_end {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "store dictionary is truncated",
                ));
            }
            let dict_bytes = file_handle
                .read_bytes_range(dict_header_end..dict_end)
                .await?;
            (
                Some(CompressionDict::from_owned_bytes(dict_bytes)),
                dict_end,
            )
        } else {
            if dict_offset != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "store without a dictionary has a dictionary offset",
                ));
            }
            (None, data_end_offset)
        };

        if index_start > index_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store index offset is out of bounds",
            ));
        }

        let index_bytes = file_handle.read_bytes_range(index_start..index_end).await?;
        let mut reader = index_bytes.as_slice();

        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;
        let required_index_bytes = num_blocks.checked_mul(20).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "store index size overflow")
        })?;
        if reader.len() != required_index_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store index length is inconsistent",
            ));
        }
        let mut index = Vec::new();
        index
            .try_reserve_exact(num_blocks)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many store blocks"))?;

        let mut expected_doc = 0u32;
        let mut expected_offset = 0u64;

        for _ in 0..num_blocks {
            let first_doc_id = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u64::<LittleEndian>()?;
            let length = reader.read_u32::<LittleEndian>()?;
            let num_docs_in_block = reader.read_u32::<LittleEndian>()?;

            let end = offset.checked_add(length as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "store block range overflow")
            })?;
            if first_doc_id != expected_doc
                || num_docs_in_block == 0
                || offset != expected_offset
                || end > data_end_offset
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "store block index is inconsistent",
                ));
            }
            expected_doc = expected_doc.checked_add(num_docs_in_block).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "store document count overflow")
            })?;
            expected_offset = end;

            index.push(StoreBlockIndex {
                first_doc_id,
                offset,
                length,
                num_docs: num_docs_in_block,
            });
        }

        if expected_doc != num_docs || expected_offset != data_end_offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "store footer totals do not match its block index",
            ));
        }

        // Create lazy slice for data portion only
        let data_slice = file_handle.slice(0..data_end_offset);

        let cache_namespace = StoreCacheNamespace {
            directory: directory_namespace,
            segment: segment_namespace,
        };
        cache.register(cache_namespace);
        Ok(Self {
            data_slice,
            index,
            num_docs,
            dict,
            cache,
            cache_namespace,
        })
    }

    /// Number of documents
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Number of blocks currently in the cache
    pub fn cached_blocks(&self) -> usize {
        self.cache.namespace_blocks(self.cache_namespace)
    }

    /// Heap bytes retained by decompressed blocks and their document offsets.
    pub fn cached_bytes(&self) -> usize {
        self.cache.namespace_bytes(self.cache_namespace)
    }

    /// Get a document by doc_id (async - may load block)
    pub async fn get(&self, doc_id: DocId, schema: &Schema) -> io::Result<Option<Document>> {
        if doc_id >= self.num_docs {
            return Ok(None);
        }

        let t = crate::observe::Timer::start();
        let (entry, block) = self.find_and_load_block(doc_id).await?;
        let doc_bytes = block.doc_bytes(doc_id - entry.first_doc_id)?;
        let result = deserialize_document(doc_bytes, schema).map(Some);
        crate::observe::store_get(schema.index_label(), t.secs());
        result
    }

    /// Get specific fields of a document by doc_id (async - may load block)
    ///
    /// Only deserializes the requested fields, skipping over unwanted data.
    /// Much faster than `get()` when documents have large fields (text bodies,
    /// vectors) that aren't needed for the response.
    pub async fn get_fields(
        &self,
        doc_id: DocId,
        schema: &Schema,
        field_ids: &[u32],
    ) -> io::Result<Option<Document>> {
        if doc_id >= self.num_docs {
            return Ok(None);
        }

        let t = crate::observe::Timer::start();
        let (entry, block) = self.find_and_load_block(doc_id).await?;
        let doc_bytes = block.doc_bytes(doc_id - entry.first_doc_id)?;
        let result = deserialize_document_fields(doc_bytes, schema, field_ids).map(Some);
        crate::observe::store_get(schema.index_label(), t.secs());
        result
    }

    /// Find the block index entry and load/cache the block for a given doc_id
    async fn find_and_load_block(
        &self,
        doc_id: DocId,
    ) -> io::Result<(&StoreBlockIndex, Arc<CachedBlock>)> {
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
        let block = self.load_block(entry).await?;
        Ok((entry, block))
    }

    async fn load_block(&self, entry: &StoreBlockIndex) -> io::Result<Arc<CachedBlock>> {
        let key = StoreCacheKey {
            namespace: self.cache_namespace,
            first_doc_id: entry.first_doc_id,
        };
        if let Some(block) = self.cache.get(key) {
            return Ok(block);
        }

        // Load from FileSlice
        let start = entry.offset;
        let end = start.checked_add(entry.length as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "store block range overflow")
        })?;
        let compressed = self.data_slice.read_bytes_range(start..end).await?;

        // Use dictionary decompression if available
        let decompressed = if let Some(ref dict) = self.dict {
            crate::compression::decompress_with_dict_limited(
                compressed.as_slice(),
                dict,
                MAX_STORE_BLOCK_BYTES,
            )?
        } else {
            crate::compression::decompress_limited(compressed.as_slice(), MAX_STORE_BLOCK_BYTES)?
        };

        // Build offset table for O(1) doc lookup within the block
        let cached = CachedBlock::build(decompressed, entry.num_docs)?;
        Ok(self.cache.insert(key, Arc::new(cached)))
    }
}

/// Deserialize only specific fields from document bytes.
///
/// Skips over unwanted fields without allocating their values — just advances
/// the reader past their length-prefixed data. For large documents with many
/// fields (e.g., full text body), this avoids allocating/copying data that
/// the caller doesn't need.
pub fn deserialize_document_fields(
    data: &[u8],
    schema: &Schema,
    field_ids: &[u32],
) -> io::Result<Document> {
    deserialize_document_inner(data, schema, Some(field_ids))
}

/// Deserialize all fields from document bytes.
///
/// Delegates to the shared field-parsing core with no field filter.
pub fn deserialize_document(data: &[u8], schema: &Schema) -> io::Result<Document> {
    deserialize_document_inner(data, schema, None)
}

/// Shared deserialization core. `field_filter = None` means all fields wanted.
fn deserialize_document_inner(
    data: &[u8],
    _schema: &Schema,
    field_filter: Option<&[u32]>,
) -> io::Result<Document> {
    use crate::dsl::Field;

    let mut reader = data;
    let num_fields = reader.read_u32::<LittleEndian>()? as usize;
    let mut doc = Document::new();

    for _ in 0..num_fields {
        let field_id = reader.read_u16::<LittleEndian>()?;
        let type_tag = reader.read_u8()?;

        let wanted = field_filter.is_none_or(|ids| ids.contains(&(field_id as u32)));

        match type_tag {
            0 => {
                // Text
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let bytes = take_document_bytes(&mut reader, len, "text field")?;
                if wanted {
                    let s = std::str::from_utf8(bytes)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    doc.add_text(Field(field_id as u32), s);
                }
            }
            1 => {
                // U64
                let v = reader.read_u64::<LittleEndian>()?;
                if wanted {
                    doc.add_u64(Field(field_id as u32), v);
                }
            }
            2 => {
                // I64
                let v = reader.read_i64::<LittleEndian>()?;
                if wanted {
                    doc.add_i64(Field(field_id as u32), v);
                }
            }
            3 => {
                // F64
                let v = reader.read_f64::<LittleEndian>()?;
                if wanted {
                    doc.add_f64(Field(field_id as u32), v);
                }
            }
            4 => {
                // Bytes
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let bytes = take_document_bytes(&mut reader, len, "byte field")?;
                if wanted {
                    doc.add_bytes(Field(field_id as u32), bytes.to_vec());
                }
            }
            5 => {
                // SparseVector
                let count = reader.read_u32::<LittleEndian>()? as usize;
                let byte_len = count.checked_mul(8).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "sparse vector size overflow")
                })?;
                let bytes = take_document_bytes(&mut reader, byte_len, "sparse vector")?;
                if wanted {
                    let mut entries = Vec::new();
                    entries.try_reserve_exact(count).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "sparse vector is too large")
                    })?;
                    let mut vector_reader = bytes;
                    for _ in 0..count {
                        let idx = vector_reader.read_u32::<LittleEndian>()?;
                        let val = vector_reader.read_f32::<LittleEndian>()?;
                        entries.push((idx, val));
                    }
                    doc.add_sparse_vector(Field(field_id as u32), entries);
                }
            }
            6 => {
                // DenseVector
                let count = reader.read_u32::<LittleEndian>()? as usize;
                let byte_len = count.checked_mul(4).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "dense vector size overflow")
                })?;
                let bytes = take_document_bytes(&mut reader, byte_len, "dense vector")?;
                if wanted {
                    let mut values = vec![0.0f32; count];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            values.as_mut_ptr() as *mut u8,
                            byte_len,
                        );
                    }
                    doc.add_dense_vector(Field(field_id as u32), values);
                }
            }
            7 => {
                // Json
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let bytes = take_document_bytes(&mut reader, len, "JSON field")?;
                if wanted {
                    let v: serde_json::Value = serde_json::from_slice(bytes)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    doc.add_json(Field(field_id as u32), v);
                }
            }
            8 => {
                // BinaryDenseVector
                let len = reader.read_u32::<LittleEndian>()? as usize;
                let bytes = take_document_bytes(&mut reader, len, "binary dense vector")?;
                if wanted {
                    doc.add_binary_dense_vector(Field(field_id as u32), bytes.to_vec());
                }
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

fn take_document_bytes<'a>(reader: &mut &'a [u8], len: usize, field: &str) -> io::Result<&'a [u8]> {
    if len > reader.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{field} is truncated"),
        ));
    }
    let (value, remaining) = reader.split_at(len);
    *reader = remaining;
    Ok(value)
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
    pub async fn append_store(
        &mut self,
        data_slice: &FileHandle,
        blocks: &[RawStoreBlock],
        cancellation: Option<&std::sync::atomic::AtomicBool>,
    ) -> io::Result<()> {
        for block in blocks {
            if cancellation
                .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::Relaxed))
            {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "store merge cancelled",
                ));
            }
            // Read raw compressed block data
            let start = block.offset;
            let end = start.checked_add(block.length as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "store block range overflow")
            })?;
            if end > data_slice.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "store block range is out of bounds",
                ));
            }
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

            self.current_offset = self
                .current_offset
                .checked_add(block.length as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "store size overflow"))?;
            self.next_doc_id = self
                .next_doc_id
                .checked_add(block.num_docs)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "store document count overflow")
                })?;
        }

        Ok(())
    }

    /// Append blocks from a dict-compressed store by decompressing and recompressing.
    ///
    /// For stores that use dictionary compression, raw blocks can't be stacked
    /// directly because the decompressor needs the original dictionary.
    /// This method decompresses each block with the source dict, then
    /// recompresses without a dictionary so the merged output is self-contained.
    pub async fn append_store_recompressing(
        &mut self,
        store: &AsyncStoreReader,
        cancellation: Option<&std::sync::atomic::AtomicBool>,
    ) -> io::Result<()> {
        let dict = store.dict();
        let data_slice = store.data_slice();
        let blocks = store.block_index();

        for block in blocks {
            if cancellation
                .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::Relaxed))
            {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "store merge cancelled",
                ));
            }
            let start = block.offset;
            let end = start.checked_add(block.length as u64).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "store block range overflow")
            })?;
            if end > data_slice.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "store block range is out of bounds",
                ));
            }
            let compressed = data_slice.read_bytes_range(start..end).await?;

            // Decompress with source dict (or without if no dict)
            let decompressed = if let Some(d) = dict {
                crate::compression::decompress_with_dict_limited(
                    compressed.as_slice(),
                    d,
                    MAX_STORE_BLOCK_BYTES,
                )?
            } else {
                crate::compression::decompress_limited(
                    compressed.as_slice(),
                    MAX_STORE_BLOCK_BYTES,
                )?
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
                length: u32::try_from(recompressed.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "compressed store block too large",
                    )
                })?,
                num_docs: block.num_docs,
            });

            self.current_offset = self
                .current_offset
                .checked_add(recompressed.len() as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "store size overflow"))?;
            self.next_doc_id = self
                .next_doc_id
                .checked_add(block.num_docs)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "store document count overflow")
                })?;
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
    pub fn data_slice(&self) -> &FileHandle {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    fn raw_store_bytes(
        num_threads: usize,
        compression_level: CompressionLevel,
        documents: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut output = Vec::new();
        let mut store = EagerParallelStoreWriter::with_compression_level(
            &mut output,
            num_threads,
            compression_level,
        );
        for document in documents {
            store.store_raw(document).unwrap();
        }
        assert_eq!(store.finish().unwrap() as usize, documents.len());
        output
    }

    fn cached_test_block(byte: u8) -> Arc<CachedBlock> {
        Arc::new(CachedBlock::build(vec![4, 0, 0, 0, byte, byte, byte, byte], 1).unwrap())
    }

    #[cfg(feature = "native")]
    #[test]
    fn parallel_store_is_byte_identical_across_worker_counts() {
        // Variable-sized, non-uniform blocks make completion order differ
        // readily without changing block boundaries or encoded bytes.
        let documents: Vec<Vec<u8>> = (0..64usize)
            .map(|doc| {
                let len = 1_000 + (doc * 7_919 % 40_000);
                (0..len)
                    .map(|offset| ((doc * 17 + offset * 31 + offset / 7) % 251) as u8)
                    .collect()
            })
            .collect();

        let single_worker = raw_store_bytes(1, CompressionLevel::FAST, &documents);
        let four_workers = raw_store_bytes(4, CompressionLevel::FAST, &documents);
        assert_eq!(four_workers, single_worker);
    }

    #[cfg(feature = "native")]
    #[test]
    fn parallel_stores_share_same_width_compression_executor() {
        let first_documents: Vec<Vec<u8>> = (0..12)
            .map(|doc| vec![(doc * 17 + 3) as u8; STORE_BLOCK_SIZE + doc * 97])
            .collect();
        let second_documents: Vec<Vec<u8>> = (0..12)
            .map(|doc| vec![(doc * 29 + 7) as u8; STORE_BLOCK_SIZE + doc * 131])
            .collect();
        let expected_first = raw_store_bytes(1, CompressionLevel::FAST, &first_documents);
        let expected_second = raw_store_bytes(1, CompressionLevel::BETTER, &second_documents);

        let mut first_output = Vec::new();
        let mut second_output = Vec::new();
        let mut first = EagerParallelStoreWriter::with_compression_level(
            &mut first_output,
            3,
            CompressionLevel::FAST,
        );
        let mut second = EagerParallelStoreWriter::with_compression_level(
            &mut second_output,
            3,
            CompressionLevel::BETTER,
        );

        assert!(Arc::ptr_eq(
            &first.workers.executor,
            &second.workers.executor
        ));
        assert_eq!(first.workers.num_threads(), 3);
        assert_eq!(second.workers.num_threads(), 3);

        for (first_document, second_document) in first_documents.iter().zip(&second_documents) {
            first.store_raw(first_document).unwrap();
            second.store_raw(second_document).unwrap();
        }
        assert_eq!(first.finish().unwrap(), first_documents.len() as u32);
        assert_eq!(second.finish().unwrap(), second_documents.len() as u32);
        assert_eq!(first_output, expected_first);
        assert_eq!(second_output, expected_second);
    }

    #[cfg(feature = "native")]
    #[test]
    fn parallel_store_bounds_in_flight_blocks_and_streams_before_finish() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingWriter(Arc<AtomicUsize>);
        impl Write for CountingWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.0.fetch_add(bytes.len(), Ordering::Relaxed);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let bytes_written = Arc::new(AtomicUsize::new(0));
        let mut output = CountingWriter(Arc::clone(&bytes_written));
        let mut store = EagerParallelStoreWriter::new(&mut output, 3);
        assert_eq!(store.workers.num_threads(), 3);
        assert_eq!(store.max_in_flight, 6);

        let document = vec![7u8; STORE_BLOCK_SIZE];
        for _ in 0..24 {
            store.store_raw(&document).unwrap();
            assert!(store.outstanding_blocks() < store.max_in_flight);
            assert_eq!(
                store.outstanding_blocks(),
                store.pending_results + store.ready_blocks.len()
            );
        }

        // Hitting the in-flight bound drains sequence-ready results directly
        // to the destination instead of retaining the complete store.
        assert!(bytes_written.load(Ordering::Relaxed) > 0);
        assert_eq!(store.finish().unwrap(), 24);
    }

    #[cfg(feature = "native")]
    #[test]
    fn parallel_store_zero_threads_falls_back_to_one_worker() {
        let mut output = Vec::new();
        let store = EagerParallelStoreWriter::new(&mut output, 0);
        assert_eq!(store.workers.num_threads(), 1);
        assert_eq!(store.max_in_flight, 2);
        assert_eq!(store.finish().unwrap(), 0);
        // Empty block index (u32 count) plus the 32-byte footer.
        assert_eq!(output.len(), 36);
    }

    #[test]
    fn cached_block_rejects_truncated_and_trailing_documents() {
        assert!(CachedBlock::build(vec![8, 0, 0, 0, 1], 1).is_err());
        assert!(CachedBlock::build(vec![0, 0, 0, 0, 1], 1).is_err());
    }

    #[test]
    fn document_store_block_limit_is_256_mib() {
        assert_eq!(MAX_STORE_BLOCK_BYTES, 256 * 1024 * 1024);
    }

    #[test]
    fn document_deserializer_rejects_length_prefixed_slice_overrun() {
        let schema = Schema::builder().build();
        let truncated_text = [1, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, b'x'];
        assert!(deserialize_document(&truncated_text, &schema).is_err());

        let truncated_sparse = [1, 0, 0, 0, 0, 0, 5, 2, 0, 0, 0, 1, 0, 0, 0];
        assert!(deserialize_document(&truncated_sparse, &schema).is_err());
    }

    #[test]
    fn shared_store_cache_is_byte_bounded_and_read_concurrent() {
        let block_bytes = cached_test_block(1).retained_bytes();
        let cache = SharedStoreCache::with_limits(block_bytes * 2, block_bytes);
        let key = |first_doc_id| StoreCacheKey {
            namespace: StoreCacheNamespace {
                directory: 1,
                segment: 7,
            },
            first_doc_id,
        };

        cache.insert(key(1), cached_test_block(1));
        cache.insert(key(2), cached_test_block(2));
        assert!(cache.get(key(1)).is_some());
        cache.insert(key(3), cached_test_block(3));

        assert!(cache.get(key(1)).is_none());
        assert!(cache.get(key(2)).is_some());
        assert!(cache.get(key(3)).is_some());
        assert!(cache.total_bytes() <= block_bytes * 2);
    }

    #[test]
    fn shared_store_cache_bypasses_oversized_entries() {
        let block = cached_test_block(1);
        let cache = SharedStoreCache::with_limits(1024, block.retained_bytes() - 1);
        let key = StoreCacheKey {
            namespace: StoreCacheNamespace {
                directory: 1,
                segment: 9,
            },
            first_doc_id: 0,
        };
        cache.insert(key, block);
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.get(key).is_none());
    }

    #[test]
    fn shared_store_cache_purges_closed_segment_namespace() {
        let block = cached_test_block(1);
        let cache = SharedStoreCache::with_limits(1024, 1024);
        let key = StoreCacheKey {
            namespace: StoreCacheNamespace {
                directory: 1,
                segment: 11,
            },
            first_doc_id: 0,
        };
        cache.register(key.namespace);
        cache.insert(key, block);
        assert!(cache.total_bytes() > 0);
        cache.unregister(key.namespace);
        assert_eq!(cache.total_bytes(), 0);
        assert!(cache.get(key).is_none());
    }

    #[test]
    fn shared_store_cache_isolates_equal_segment_ids_across_directories() {
        let cache = SharedStoreCache::with_limits(1024, 1024);
        let key = |directory| StoreCacheKey {
            namespace: StoreCacheNamespace {
                directory,
                segment: 42,
            },
            first_doc_id: 0,
        };
        let left = cache.insert(key(1), cached_test_block(1));
        let right = cache.insert(key(2), cached_test_block(2));

        assert!(!Arc::ptr_eq(&left, &right));
        assert_eq!(cache.total_blocks(), 2);
    }
}
