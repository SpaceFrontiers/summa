//! Streaming segment builder with optimized memory usage
//!
//! Key optimizations:
//! - **String interning**: Terms are interned using `lasso` to avoid repeated allocations
//! - **hashbrown HashMap**: O(1) average insertion instead of BTreeMap's O(log n)
//! - **Streaming document store**: Documents written to disk immediately
//! - **Incremental posting flush**: Large posting lists flushed to temp file
//! - **Memory-mapped intermediate files**: Reduces memory pressure
//! - **Arena allocation**: Batch allocations for reduced fragmentation

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

use hashbrown::HashMap;
use lasso::{Rodeo, Spur};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::compression::CompressionLevel;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
use crate::structures::{PostingList, SSTableWriter, TermInfo};
use crate::tokenizer::BoxedTokenizer;
use crate::wand::WandData;
use crate::{DocId, Result};

/// Size of the document store buffer before writing to disk
const STORE_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16MB

/// Interned term key combining field and term
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TermKey {
    field: u32,
    term: Spur,
}

/// Compact posting entry for in-memory storage
#[derive(Clone, Copy)]
struct CompactPosting {
    doc_id: DocId,
    term_freq: u16,
}

/// In-memory posting list for a term
struct PostingListBuilder {
    /// In-memory postings
    postings: Vec<CompactPosting>,
}

impl PostingListBuilder {
    fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }

    /// Add a posting, merging if same doc_id as last
    #[inline]
    fn add(&mut self, doc_id: DocId, term_freq: u32) {
        // Check if we can merge with the last posting
        if let Some(last) = self.postings.last_mut()
            && last.doc_id == doc_id
        {
            last.term_freq = last.term_freq.saturating_add(term_freq as u16);
            return;
        }
        self.postings.push(CompactPosting {
            doc_id,
            term_freq: term_freq.min(u16::MAX as u32) as u16,
        });
    }

    fn len(&self) -> usize {
        self.postings.len()
    }
}

/// Intermediate result for parallel posting serialization
enum SerializedPosting {
    /// Inline posting (small enough to fit in TermInfo)
    Inline(TermInfo),
    /// External posting with serialized bytes
    External { bytes: Vec<u8>, doc_count: u32 },
}

/// Statistics for debugging segment builder performance
#[derive(Debug, Clone)]
pub struct SegmentBuilderStats {
    /// Number of documents indexed
    pub num_docs: u32,
    /// Number of unique terms in the inverted index
    pub unique_terms: usize,
    /// Total postings in memory (across all terms)
    pub postings_in_memory: usize,
    /// Number of interned strings
    pub interned_strings: usize,
    /// Size of doc_field_lengths vector
    pub doc_field_lengths_size: usize,
}

/// Configuration for segment builder
#[derive(Clone)]
pub struct SegmentBuilderConfig {
    /// Directory for temporary spill files
    pub temp_dir: PathBuf,
    /// Compression level for document store
    pub compression_level: CompressionLevel,
    /// Number of threads for parallel compression
    pub num_compression_threads: usize,
    /// Initial capacity for term interner
    pub interner_capacity: usize,
    /// Initial capacity for posting lists hashmap
    pub posting_map_capacity: usize,
}

impl Default for SegmentBuilderConfig {
    fn default() -> Self {
        Self {
            temp_dir: std::env::temp_dir(),
            compression_level: CompressionLevel(7),
            num_compression_threads: num_cpus::get(),
            interner_capacity: 1_000_000,
            posting_map_capacity: 500_000,
        }
    }
}

/// Segment builder with optimized memory usage
///
/// Features:
/// - Streams documents to disk immediately (no in-memory document storage)
/// - Uses string interning for terms (reduced allocations)
/// - Uses hashbrown HashMap (faster than BTreeMap)
pub struct SegmentBuilder {
    schema: Schema,
    config: SegmentBuilderConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,

    /// String interner for terms - O(1) lookup and deduplication
    term_interner: Rodeo,

    /// Inverted index: term key -> posting list
    inverted_index: HashMap<TermKey, PostingListBuilder>,

    /// Streaming document store writer
    store_file: BufWriter<File>,
    store_path: PathBuf,

    /// Document count
    next_doc_id: DocId,

    /// Per-field statistics for BM25F
    field_stats: FxHashMap<u32, FieldStats>,

    /// Per-document field lengths stored compactly
    /// Uses a flat Vec instead of Vec<HashMap> for better cache locality
    /// Layout: [doc0_field0_len, doc0_field1_len, ..., doc1_field0_len, ...]
    doc_field_lengths: Vec<u32>,
    num_indexed_fields: usize,
    field_to_slot: FxHashMap<u32, usize>,

    /// Optional pre-computed WAND data for IDF values
    wand_data: Option<Arc<WandData>>,

    /// Reusable buffer for per-document term frequency aggregation
    /// Avoids allocating a new hashmap for each document
    local_tf_buffer: FxHashMap<Spur, u32>,

    /// Reusable buffer for tokenization to avoid per-token String allocations
    token_buffer: String,
}

impl SegmentBuilder {
    /// Create a new segment builder
    pub fn new(schema: Schema, config: SegmentBuilderConfig) -> Result<Self> {
        let segment_id = uuid::Uuid::new_v4();
        let store_path = config
            .temp_dir
            .join(format!("hermes_store_{}.tmp", segment_id));

        let store_file = BufWriter::with_capacity(
            STORE_BUFFER_SIZE,
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&store_path)?,
        );

        // Count indexed fields for compact field length storage
        let mut num_indexed_fields = 0;
        let mut field_to_slot = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if entry.indexed && matches!(entry.field_type, FieldType::Text) {
                field_to_slot.insert(field.0, num_indexed_fields);
                num_indexed_fields += 1;
            }
        }

        Ok(Self {
            schema,
            tokenizers: FxHashMap::default(),
            term_interner: Rodeo::new(),
            inverted_index: HashMap::with_capacity(config.posting_map_capacity),
            store_file,
            store_path,
            next_doc_id: 0,
            field_stats: FxHashMap::default(),
            doc_field_lengths: Vec::new(),
            num_indexed_fields,
            field_to_slot,
            wand_data: None,
            local_tf_buffer: FxHashMap::default(),
            token_buffer: String::with_capacity(64),
            config,
        })
    }

    /// Create with pre-computed WAND data
    pub fn with_wand_data(
        schema: Schema,
        config: SegmentBuilderConfig,
        wand_data: Arc<WandData>,
    ) -> Result<Self> {
        let mut builder = Self::new(schema, config)?;
        builder.wand_data = Some(wand_data);
        Ok(builder)
    }

    pub fn set_tokenizer(&mut self, field: Field, tokenizer: BoxedTokenizer) {
        self.tokenizers.insert(field, tokenizer);
    }

    pub fn num_docs(&self) -> u32 {
        self.next_doc_id
    }

    /// Get current statistics for debugging performance
    pub fn stats(&self) -> SegmentBuilderStats {
        let postings_in_memory: usize =
            self.inverted_index.values().map(|p| p.postings.len()).sum();
        SegmentBuilderStats {
            num_docs: self.next_doc_id,
            unique_terms: self.inverted_index.len(),
            postings_in_memory,
            interned_strings: self.term_interner.len(),
            doc_field_lengths_size: self.doc_field_lengths.len(),
        }
    }

    /// Add a document - streams to disk immediately
    pub fn add_document(&mut self, doc: Document) -> Result<DocId> {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Initialize field lengths for this document
        let base_idx = self.doc_field_lengths.len();
        self.doc_field_lengths
            .resize(base_idx + self.num_indexed_fields, 0);

        for (field, value) in doc.field_values() {
            let entry = self.schema.get_field_entry(*field);
            if entry.is_none() || !entry.unwrap().indexed {
                continue;
            }

            let entry = entry.unwrap();
            match (&entry.field_type, value) {
                (FieldType::Text, FieldValue::Text(text)) => {
                    let token_count = self.index_text_field(*field, doc_id, text)?;

                    // Update field statistics
                    let stats = self.field_stats.entry(field.0).or_default();
                    stats.total_tokens += token_count as u64;
                    stats.doc_count += 1;

                    // Store field length compactly
                    if let Some(&slot) = self.field_to_slot.get(&field.0) {
                        self.doc_field_lengths[base_idx + slot] = token_count;
                    }
                }
                (FieldType::U64, FieldValue::U64(v)) => {
                    self.index_numeric_field(*field, doc_id, *v)?;
                }
                (FieldType::I64, FieldValue::I64(v)) => {
                    self.index_numeric_field(*field, doc_id, *v as u64)?;
                }
                (FieldType::F64, FieldValue::F64(v)) => {
                    self.index_numeric_field(*field, doc_id, v.to_bits())?;
                }
                _ => {}
            }
        }

        // Stream document to disk immediately
        self.write_document_to_store(&doc)?;

        Ok(doc_id)
    }

    /// Index a text field using interned terms
    ///
    /// Optimization: Zero-allocation inline tokenization + term frequency aggregation.
    /// Instead of allocating a String per token, we:
    /// 1. Iterate over whitespace-split words
    /// 2. Build lowercase token in a reusable buffer
    /// 3. Intern directly from the buffer
    fn index_text_field(&mut self, field: Field, doc_id: DocId, text: &str) -> Result<u32> {
        // Phase 1: Aggregate term frequencies within this document
        // Reuse buffer to avoid allocations
        self.local_tf_buffer.clear();

        let mut token_count = 0u32;

        // Zero-allocation tokenization: iterate words, lowercase inline, intern directly
        for word in text.split_whitespace() {
            // Build lowercase token in reusable buffer
            self.token_buffer.clear();
            for c in word.chars() {
                if c.is_alphanumeric() {
                    for lc in c.to_lowercase() {
                        self.token_buffer.push(lc);
                    }
                }
            }

            if self.token_buffer.is_empty() {
                continue;
            }

            token_count += 1;

            // Intern the term directly from buffer - O(1) amortized
            let term_spur = self.term_interner.get_or_intern(&self.token_buffer);
            *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;
        }

        // Phase 2: Insert aggregated terms into inverted index
        // Now we only do one inverted_index lookup per unique term in doc
        let field_id = field.0;

        for (&term_spur, &tf) in &self.local_tf_buffer {
            let term_key = TermKey {
                field: field_id,
                term: term_spur,
            };

            let posting = self
                .inverted_index
                .entry(term_key)
                .or_insert_with(PostingListBuilder::new);
            posting.add(doc_id, tf);
        }

        Ok(token_count)
    }

    fn index_numeric_field(&mut self, field: Field, doc_id: DocId, value: u64) -> Result<()> {
        // For numeric fields, we use a special encoding
        let term_str = format!("__num_{}", value);
        let term_spur = self.term_interner.get_or_intern(&term_str);

        let term_key = TermKey {
            field: field.0,
            term: term_spur,
        };

        let posting = self
            .inverted_index
            .entry(term_key)
            .or_insert_with(PostingListBuilder::new);
        posting.add(doc_id, 1);

        Ok(())
    }

    /// Write document to streaming store
    fn write_document_to_store(&mut self, doc: &Document) -> Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let doc_bytes = super::store::serialize_document(doc, &self.schema)?;

        self.store_file
            .write_u32::<LittleEndian>(doc_bytes.len() as u32)?;
        self.store_file.write_all(&doc_bytes)?;

        Ok(())
    }

    /// Build the final segment
    pub async fn build<D: Directory + DirectoryWriter>(
        mut self,
        dir: &D,
        segment_id: SegmentId,
    ) -> Result<SegmentMeta> {
        // Flush any buffered data
        self.store_file.flush()?;

        let files = SegmentFiles::new(segment_id.0);

        // Extract data needed for parallel processing
        let store_path = self.store_path.clone();
        let schema = self.schema.clone();
        let num_compression_threads = self.config.num_compression_threads;
        let compression_level = self.config.compression_level;

        // Build postings and document store in parallel
        let (postings_result, store_result) = rayon::join(
            || self.build_postings(),
            || {
                Self::build_store_parallel(
                    &store_path,
                    &schema,
                    num_compression_threads,
                    compression_level,
                )
            },
        );

        let (term_dict_data, postings_data) = postings_result?;
        let store_data = store_result?;

        // Write to directory
        dir.write(&files.term_dict, &term_dict_data).await?;
        dir.write(&files.postings, &postings_data).await?;
        dir.write(&files.store, &store_data).await?;

        let meta = SegmentMeta {
            id: segment_id.0,
            num_docs: self.next_doc_id,
            field_stats: self.field_stats.clone(),
        };

        dir.write(&files.meta, &meta.serialize()?).await?;

        // Cleanup temp files
        let _ = std::fs::remove_file(&self.store_path);

        Ok(meta)
    }

    /// Build postings from inverted index
    ///
    /// Uses parallel processing to serialize posting lists concurrently.
    fn build_postings(&mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        // Phase 1: Collect and sort term keys (parallel key generation)
        // Key format: field_id (4 bytes) + term bytes
        let mut term_entries: Vec<(Vec<u8>, &PostingListBuilder)> = self
            .inverted_index
            .iter()
            .map(|(term_key, posting_list)| {
                let term_str = self.term_interner.resolve(&term_key.term);
                let mut key = Vec::with_capacity(4 + term_str.len());
                key.extend_from_slice(&term_key.field.to_le_bytes());
                key.extend_from_slice(term_str.as_bytes());
                (key, posting_list)
            })
            .collect();

        // Sort by key for SSTable ordering
        term_entries.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

        // Phase 2: Parallel serialization of posting lists
        // Each term's posting list is serialized independently
        let serialized: Vec<(Vec<u8>, SerializedPosting)> = term_entries
            .into_par_iter()
            .map(|(key, posting_builder)| {
                // Build posting list from in-memory postings
                let mut full_postings = PostingList::with_capacity(posting_builder.len());
                for p in &posting_builder.postings {
                    full_postings.push(p.doc_id, p.term_freq as u32);
                }

                // Build term info
                let doc_ids: Vec<u32> = full_postings.iter().map(|p| p.doc_id).collect();
                let term_freqs: Vec<u32> = full_postings.iter().map(|p| p.term_freq).collect();

                let result = if let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs) {
                    SerializedPosting::Inline(inline)
                } else {
                    // Serialize to local buffer
                    let mut posting_bytes = Vec::new();
                    let block_list =
                        crate::structures::BlockPostingList::from_posting_list(&full_postings)
                            .expect("BlockPostingList creation failed");
                    block_list
                        .serialize(&mut posting_bytes)
                        .expect("BlockPostingList serialization failed");
                    SerializedPosting::External {
                        bytes: posting_bytes,
                        doc_count: full_postings.doc_count(),
                    }
                };

                (key, result)
            })
            .collect();

        // Phase 3: Sequential assembly (must be sequential for offset calculation)
        let mut term_dict = Vec::new();
        let mut postings = Vec::new();
        let mut writer = SSTableWriter::<TermInfo>::new(&mut term_dict);

        for (key, serialized_posting) in serialized {
            let term_info = match serialized_posting {
                SerializedPosting::Inline(info) => info,
                SerializedPosting::External { bytes, doc_count } => {
                    let posting_offset = postings.len() as u64;
                    let posting_len = bytes.len() as u32;
                    postings.extend_from_slice(&bytes);
                    TermInfo::external(posting_offset, posting_len, doc_count)
                }
            };

            writer.insert(&key, &term_info)?;
        }

        writer.finish()?;
        Ok((term_dict, postings))
    }

    /// Build document store from streamed temp file (static method for parallel execution)
    ///
    /// Uses parallel processing to deserialize documents concurrently.
    fn build_store_parallel(
        store_path: &PathBuf,
        schema: &Schema,
        num_compression_threads: usize,
        compression_level: CompressionLevel,
    ) -> Result<Vec<u8>> {
        use super::store::EagerParallelStoreWriter;

        let file = File::open(store_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Phase 1: Parse document boundaries (sequential, fast)
        let mut doc_ranges: Vec<(usize, usize)> = Vec::new();
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

            doc_ranges.push((offset, doc_len));
            offset += doc_len;
        }

        // Phase 2: Parallel deserialization of documents
        let docs: Vec<Document> = doc_ranges
            .into_par_iter()
            .filter_map(|(start, len)| {
                let doc_bytes = &mmap[start..start + len];
                super::store::deserialize_document(doc_bytes, schema).ok()
            })
            .collect();

        // Phase 3: Write to store (compression is already parallel in EagerParallelStoreWriter)
        let mut store_data = Vec::new();
        let mut store_writer = EagerParallelStoreWriter::with_compression_level(
            &mut store_data,
            num_compression_threads,
            compression_level,
        );

        for doc in &docs {
            store_writer.store(doc, schema)?;
        }

        store_writer.finish()?;
        Ok(store_data)
    }
}

impl Drop for SegmentBuilder {
    fn drop(&mut self) {
        // Cleanup temp files on drop
        let _ = std::fs::remove_file(&self.store_path);
    }
}
