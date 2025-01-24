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
use crate::{DocId, Result};

// Re-export from vector_data for backwards compatibility
pub use super::vector_data::{FlatVectorData, IVFRaBitQIndexData, ScaNNIndexData};

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

/// In-memory position posting list for a term (for fields with record_positions=true)
struct PositionPostingListBuilder {
    /// Doc ID -> list of positions (encoded as element_ordinal << 20 | token_position)
    postings: Vec<(DocId, Vec<u32>)>,
}

impl PositionPostingListBuilder {
    fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }

    /// Add a position for a document
    #[inline]
    fn add_position(&mut self, doc_id: DocId, position: u32) {
        if let Some((last_doc, positions)) = self.postings.last_mut()
            && *last_doc == doc_id
        {
            positions.push(position);
            return;
        }
        self.postings.push((doc_id, vec![position]));
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
    /// Estimated total memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Memory breakdown by component
    pub memory_breakdown: MemoryBreakdown,
}

/// Detailed memory breakdown by component
#[derive(Debug, Clone, Default)]
pub struct MemoryBreakdown {
    /// Postings memory (CompactPosting structs)
    pub postings_bytes: usize,
    /// Inverted index HashMap overhead
    pub index_overhead_bytes: usize,
    /// Term interner memory
    pub interner_bytes: usize,
    /// Document field lengths
    pub field_lengths_bytes: usize,
    /// Dense vector storage
    pub dense_vectors_bytes: usize,
    /// Number of dense vectors
    pub dense_vector_count: usize,
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

    /// Reusable buffer for per-document term frequency aggregation
    /// Avoids allocating a new hashmap for each document
    local_tf_buffer: FxHashMap<Spur, u32>,

    /// Reusable buffer for tokenization to avoid per-token String allocations
    token_buffer: String,

    /// Dense vector storage per field: field -> (doc_ids, vectors)
    /// Vectors are stored as flat f32 arrays for efficient RaBitQ indexing
    dense_vectors: FxHashMap<u32, DenseVectorBuilder>,

    /// Sparse vector storage per field: field -> SparseVectorBuilder
    /// Uses proper BlockSparsePostingList with configurable quantization
    sparse_vectors: FxHashMap<u32, SparseVectorBuilder>,

    /// Position index for fields with positions enabled
    /// term key -> position posting list
    position_index: HashMap<TermKey, PositionPostingListBuilder>,

    /// Fields that have position tracking enabled, with their mode
    position_enabled_fields: FxHashMap<u32, Option<crate::dsl::PositionMode>>,

    /// Current element ordinal for multi-valued fields (reset per document)
    current_element_ordinal: FxHashMap<u32, u32>,
}

/// Builder for dense vector index using RaBitQ
struct DenseVectorBuilder {
    /// Dimension of vectors
    dim: usize,
    /// Document IDs with vectors
    doc_ids: Vec<DocId>,
    /// Flat vector storage (doc_ids.len() * dim floats)
    vectors: Vec<f32>,
}

impl DenseVectorBuilder {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            doc_ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    fn add(&mut self, doc_id: DocId, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");
        self.doc_ids.push(doc_id);
        self.vectors.extend_from_slice(vector);
    }

    fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Get all vectors as Vec<Vec<f32>> for RaBitQ indexing
    fn get_vectors(&self) -> Vec<Vec<f32>> {
        self.doc_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = i * self.dim;
                self.vectors[start..start + self.dim].to_vec()
            })
            .collect()
    }

    /// Get vectors trimmed to specified dimension for matryoshka/MRL indexing
    fn get_vectors_trimmed(&self, trim_dim: usize) -> Vec<Vec<f32>> {
        debug_assert!(trim_dim <= self.dim, "trim_dim must be <= dim");
        self.doc_ids
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = i * self.dim;
                self.vectors[start..start + trim_dim].to_vec()
            })
            .collect()
    }
}

/// Builder for sparse vector index using BlockSparsePostingList
///
/// Collects (doc_id, weight) postings per dimension, then builds
/// BlockSparsePostingList with proper quantization during commit.
struct SparseVectorBuilder {
    /// Postings per dimension: dim_id -> Vec<(doc_id, weight)>
    postings: FxHashMap<u32, Vec<(DocId, f32)>>,
}

impl SparseVectorBuilder {
    fn new() -> Self {
        Self {
            postings: FxHashMap::default(),
        }
    }

    /// Add a sparse vector entry
    #[inline]
    fn add(&mut self, dim_id: u32, doc_id: DocId, weight: f32) {
        self.postings
            .entry(dim_id)
            .or_default()
            .push((doc_id, weight));
    }

    fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }
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
        // Also track which fields have position recording enabled
        let mut num_indexed_fields = 0;
        let mut field_to_slot = FxHashMap::default();
        let mut position_enabled_fields = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if entry.indexed && matches!(entry.field_type, FieldType::Text) {
                field_to_slot.insert(field.0, num_indexed_fields);
                num_indexed_fields += 1;
                if entry.positions.is_some() {
                    position_enabled_fields.insert(field.0, entry.positions);
                }
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
            local_tf_buffer: FxHashMap::default(),
            token_buffer: String::with_capacity(64),
            config,
            dense_vectors: FxHashMap::default(),
            sparse_vectors: FxHashMap::default(),
            position_index: HashMap::new(),
            position_enabled_fields,
            current_element_ordinal: FxHashMap::default(),
        })
    }

    pub fn set_tokenizer(&mut self, field: Field, tokenizer: BoxedTokenizer) {
        self.tokenizers.insert(field, tokenizer);
    }

    pub fn num_docs(&self) -> u32 {
        self.next_doc_id
    }

    /// Get current statistics for debugging performance
    pub fn stats(&self) -> SegmentBuilderStats {
        use std::mem::size_of;

        let postings_in_memory: usize =
            self.inverted_index.values().map(|p| p.postings.len()).sum();

        // Precise memory calculation using actual struct sizes
        // CompactPosting: doc_id (u32) + term_freq (u16) = 6 bytes, but may have padding
        let compact_posting_size = size_of::<CompactPosting>();

        // Postings: actual Vec capacity * element size + Vec overhead (24 bytes on 64-bit)
        let postings_bytes: usize = self
            .inverted_index
            .values()
            .map(|p| {
                p.postings.capacity() * compact_posting_size + size_of::<Vec<CompactPosting>>()
            })
            .sum();

        // Inverted index: HashMap overhead per entry
        // Each entry: TermKey (field u32 + Spur 4 bytes = 8 bytes) + PostingListBuilder + HashMap bucket overhead
        // HashMap typically uses ~1.5x capacity, each bucket ~16-24 bytes
        let term_key_size = size_of::<TermKey>();
        let posting_builder_size = size_of::<PostingListBuilder>();
        let hashmap_entry_overhead = 24; // bucket pointer + metadata
        let index_overhead_bytes = self.inverted_index.len()
            * (term_key_size + posting_builder_size + hashmap_entry_overhead);

        // Term interner: Rodeo stores strings + metadata
        // Each interned string: actual string bytes + Spur (4 bytes) + internal overhead (~16 bytes)
        // We can't get exact string lengths, so estimate average term length of 8 bytes
        let avg_term_len = 8;
        let interner_overhead_per_string = size_of::<lasso::Spur>() + 16;
        let interner_bytes =
            self.term_interner.len() * (avg_term_len + interner_overhead_per_string);

        // Doc field lengths: Vec<u32> with capacity
        let field_lengths_bytes =
            self.doc_field_lengths.capacity() * size_of::<u32>() + size_of::<Vec<u32>>();

        // Dense vectors: actual capacity used
        let mut dense_vectors_bytes: usize = 0;
        let mut dense_vector_count: usize = 0;
        for b in self.dense_vectors.values() {
            // vectors: Vec<f32> capacity + doc_ids: Vec<DocId> capacity
            dense_vectors_bytes += b.vectors.capacity() * size_of::<f32>()
                + b.doc_ids.capacity() * size_of::<DocId>()
                + size_of::<Vec<f32>>()
                + size_of::<Vec<DocId>>();
            dense_vector_count += b.doc_ids.len();
        }

        // Local buffers
        let local_tf_buffer_bytes =
            self.local_tf_buffer.capacity() * (size_of::<lasso::Spur>() + size_of::<u32>() + 16);

        let estimated_memory_bytes = postings_bytes
            + index_overhead_bytes
            + interner_bytes
            + field_lengths_bytes
            + dense_vectors_bytes
            + local_tf_buffer_bytes;

        let memory_breakdown = MemoryBreakdown {
            postings_bytes,
            index_overhead_bytes,
            interner_bytes,
            field_lengths_bytes,
            dense_vectors_bytes,
            dense_vector_count,
        };

        SegmentBuilderStats {
            num_docs: self.next_doc_id,
            unique_terms: self.inverted_index.len(),
            postings_in_memory,
            interned_strings: self.term_interner.len(),
            doc_field_lengths_size: self.doc_field_lengths.len(),
            estimated_memory_bytes,
            memory_breakdown,
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

        // Reset element ordinals for this document (for multi-valued fields)
        self.current_element_ordinal.clear();

        for (field, value) in doc.field_values() {
            let entry = self.schema.get_field_entry(*field);
            if entry.is_none() || !entry.unwrap().indexed {
                continue;
            }

            let entry = entry.unwrap();
            match (&entry.field_type, value) {
                (FieldType::Text, FieldValue::Text(text)) => {
                    // Get current element ordinal for multi-valued fields
                    let element_ordinal = *self.current_element_ordinal.get(&field.0).unwrap_or(&0);
                    let token_count =
                        self.index_text_field(*field, doc_id, text, element_ordinal)?;
                    // Increment element ordinal for next value of this field
                    *self.current_element_ordinal.entry(field.0).or_insert(0) += 1;

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
                (FieldType::DenseVector, FieldValue::DenseVector(vec)) => {
                    self.index_dense_vector_field(*field, doc_id, vec)?;
                }
                (FieldType::SparseVector, FieldValue::SparseVector(entries)) => {
                    self.index_sparse_vector_field(*field, doc_id, entries)?;
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
    ///
    /// If position recording is enabled for this field, also records token positions
    /// encoded as (element_ordinal << 20) | token_position.
    fn index_text_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        text: &str,
        element_ordinal: u32,
    ) -> Result<u32> {
        use crate::dsl::PositionMode;

        let field_id = field.0;
        let position_mode = self
            .position_enabled_fields
            .get(&field_id)
            .copied()
            .flatten();

        // Phase 1: Aggregate term frequencies within this document
        // Also collect positions if enabled
        // Reuse buffer to avoid allocations
        self.local_tf_buffer.clear();

        // For position tracking: term -> list of positions in this text
        let mut local_positions: FxHashMap<Spur, Vec<u32>> = FxHashMap::default();

        let mut token_position = 0u32;

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

            // Intern the term directly from buffer - O(1) amortized
            let term_spur = self.term_interner.get_or_intern(&self.token_buffer);
            *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;

            // Record position based on mode
            if let Some(mode) = position_mode {
                let encoded_pos = match mode {
                    // Ordinal only: just store element ordinal (token position = 0)
                    PositionMode::Ordinal => element_ordinal << 20,
                    // Token position only: just store token position (ordinal = 0)
                    PositionMode::TokenPosition => token_position,
                    // Full: encode both
                    PositionMode::Full => (element_ordinal << 20) | token_position,
                };
                local_positions
                    .entry(term_spur)
                    .or_default()
                    .push(encoded_pos);
            }

            token_position += 1;
        }

        // Phase 2: Insert aggregated terms into inverted index
        // Now we only do one inverted_index lookup per unique term in doc
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

            // Add positions if enabled
            if position_mode.is_some()
                && let Some(positions) = local_positions.get(&term_spur)
            {
                let pos_posting = self
                    .position_index
                    .entry(term_key)
                    .or_insert_with(PositionPostingListBuilder::new);
                for &pos in positions {
                    pos_posting.add_position(doc_id, pos);
                }
            }
        }

        Ok(token_position)
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

    /// Index a dense vector field
    fn index_dense_vector_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        vector: &[f32],
    ) -> Result<()> {
        let dim = vector.len();

        let builder = self
            .dense_vectors
            .entry(field.0)
            .or_insert_with(|| DenseVectorBuilder::new(dim));

        // Verify dimension consistency
        if builder.dim != dim && builder.len() > 0 {
            return Err(crate::Error::Schema(format!(
                "Dense vector dimension mismatch: expected {}, got {}",
                builder.dim, dim
            )));
        }

        builder.add(doc_id, vector);
        Ok(())
    }

    /// Index a sparse vector field using dedicated sparse posting lists
    ///
    /// Collects (doc_id, weight) postings per dimension. During commit, these are
    /// converted to BlockSparsePostingList with proper quantization from SparseVectorConfig.
    ///
    /// Weights below the configured `weight_threshold` are not indexed.
    fn index_sparse_vector_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        entries: &[(u32, f32)],
    ) -> Result<()> {
        // Get weight threshold from field config (default 0.0 = no filtering)
        let weight_threshold = self
            .schema
            .get_field_entry(field)
            .and_then(|entry| entry.sparse_vector_config.as_ref())
            .map(|config| config.weight_threshold)
            .unwrap_or(0.0);

        let builder = self
            .sparse_vectors
            .entry(field.0)
            .or_insert_with(SparseVectorBuilder::new);

        for &(dim_id, weight) in entries {
            // Skip weights below threshold
            if weight.abs() < weight_threshold {
                continue;
            }

            builder.add(dim_id, doc_id, weight);
        }

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

        // Build positions FIRST to get offsets for TermInfo
        let (positions_data, position_offsets) = self.build_positions_file()?;

        // Extract data needed for parallel processing
        let store_path = self.store_path.clone();
        let schema = self.schema.clone();
        let num_compression_threads = self.config.num_compression_threads;
        let compression_level = self.config.compression_level;

        // Build postings and document store in parallel
        let (postings_result, store_result) = rayon::join(
            || self.build_postings(&position_offsets),
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

        // Write positions file (data only, offsets are in TermInfo)
        if !positions_data.is_empty() {
            dir.write(&files.positions, &positions_data).await?;
        }

        // Build and write dense vector indexes (RaBitQ) - all in one file
        if !self.dense_vectors.is_empty() {
            let vectors_data = self.build_vectors_file()?;
            if !vectors_data.is_empty() {
                dir.write(&files.vectors, &vectors_data).await?;
            }
        }

        // Build and write sparse vector posting lists
        if !self.sparse_vectors.is_empty() {
            let sparse_data = self.build_sparse_file()?;
            if !sparse_data.is_empty() {
                dir.write(&files.sparse, &sparse_data).await?;
            }
        }

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

    /// Build unified vectors file containing all dense vector indexes
    ///
    /// File format:
    /// - Header: num_fields (u32)
    /// - For each field: field_id (u32), index_type (u8), offset (u64), length (u64)
    /// - Data: concatenated serialized indexes (RaBitQ, IVF-RaBitQ, or ScaNN)
    fn build_vectors_file(&self) -> Result<Vec<u8>> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // Build all indexes first: (field_id, index_type, data)
        let mut field_indexes: Vec<(u32, u8, Vec<u8>)> = Vec::new();

        for (&field_id, builder) in &self.dense_vectors {
            if builder.len() == 0 {
                continue;
            }

            let field = crate::dsl::Field(field_id);

            // Get dense vector config
            let dense_config = self
                .schema
                .get_field_entry(field)
                .and_then(|e| e.dense_vector_config.as_ref());

            // Get vectors, potentially trimmed for matryoshka/MRL indexing
            let index_dim = dense_config.map(|c| c.index_dim()).unwrap_or(builder.dim);
            let vectors = if index_dim < builder.dim {
                // Trim vectors to mrl_dim for indexing
                builder.get_vectors_trimmed(index_dim)
            } else {
                builder.get_vectors()
            };

            // During normal indexing, segments always store Flat (raw vectors).
            // ANN indexes are built at index-level via build_vector_index() which
            // trains centroids/codebooks once from all vectors and triggers rebuild.
            let flat_data = FlatVectorData {
                dim: index_dim,
                vectors: vectors.clone(),
                doc_ids: builder.doc_ids.clone(),
            };
            let index_bytes = serde_json::to_vec(&flat_data)
                .map_err(|e| crate::Error::Serialization(e.to_string()))?;
            let index_type = 3u8; // 3 = Flat

            field_indexes.push((field_id, index_type, index_bytes));
        }

        if field_indexes.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by field_id for consistent ordering
        field_indexes.sort_by_key(|(id, _, _)| *id);

        // Calculate header size: num_fields + (field_id, index_type, offset, len) per field
        let header_size = 4 + field_indexes.len() * (4 + 1 + 8 + 8);

        // Build output
        let mut output = Vec::new();

        // Write number of fields
        output.write_u32::<LittleEndian>(field_indexes.len() as u32)?;

        // Calculate offsets and write header entries
        let mut current_offset = header_size as u64;
        for (field_id, index_type, data) in &field_indexes {
            output.write_u32::<LittleEndian>(*field_id)?;
            output.write_u8(*index_type)?;
            output.write_u64::<LittleEndian>(current_offset)?;
            output.write_u64::<LittleEndian>(data.len() as u64)?;
            current_offset += data.len() as u64;
        }

        // Write data
        for (_, _, data) in field_indexes {
            output.extend_from_slice(&data);
        }

        Ok(output)
    }

    /// Build sparse vectors file containing BlockSparsePostingList per field/dimension
    ///
    /// File format (direct-indexed table for O(1) dimension lookup):
    /// - Header: num_fields (u32)
    /// - For each field:
    ///   - field_id (u32)
    ///   - quantization (u8)
    ///   - max_dim_id (u32)          ← highest dimension ID + 1 (table size)
    ///   - table: [(offset: u64, length: u32)] × max_dim_id  ← direct indexed by dim_id
    ///     (offset=0, length=0 means dimension not present)
    /// - Data: concatenated serialized BlockSparsePostingList
    fn build_sparse_file(&self) -> Result<Vec<u8>> {
        use crate::structures::{BlockSparsePostingList, WeightQuantization};
        use byteorder::{LittleEndian, WriteBytesExt};

        if self.sparse_vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Collect field data: (field_id, quantization, max_dim_id, dim_id -> serialized_bytes)
        type SparseFieldData = (u32, WeightQuantization, u32, FxHashMap<u32, Vec<u8>>);
        let mut field_data: Vec<SparseFieldData> = Vec::new();

        for (&field_id, builder) in &self.sparse_vectors {
            if builder.is_empty() {
                continue;
            }

            let field = crate::dsl::Field(field_id);

            // Get quantization from field config
            let quantization = self
                .schema
                .get_field_entry(field)
                .and_then(|e| e.sparse_vector_config.as_ref())
                .map(|c| c.weight_quantization)
                .unwrap_or(WeightQuantization::Float32);

            // Find max dimension ID
            let max_dim_id = builder.postings.keys().max().copied().unwrap_or(0);

            // Build BlockSparsePostingList for each dimension
            let mut dim_bytes: FxHashMap<u32, Vec<u8>> = FxHashMap::default();

            for (&dim_id, postings) in &builder.postings {
                // Sort postings by doc_id
                let mut sorted_postings = postings.clone();
                sorted_postings.sort_by_key(|(doc_id, _)| *doc_id);

                // Build BlockSparsePostingList
                let block_list =
                    BlockSparsePostingList::from_postings(&sorted_postings, quantization)
                        .map_err(crate::Error::Io)?;

                // Serialize
                let mut bytes = Vec::new();
                block_list.serialize(&mut bytes).map_err(crate::Error::Io)?;

                dim_bytes.insert(dim_id, bytes);
            }

            field_data.push((field_id, quantization, max_dim_id + 1, dim_bytes));
        }

        if field_data.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by field_id
        field_data.sort_by_key(|(id, _, _, _)| *id);

        // Calculate header size
        // Header: num_fields (4)
        // Per field: field_id (4) + quant (1) + max_dim_id (4) + table (12 * max_dim_id)
        let mut header_size = 4u64;
        for (_, _, max_dim_id, _) in &field_data {
            header_size += 4 + 1 + 4; // field_id + quant + max_dim_id
            header_size += (*max_dim_id as u64) * 12; // table entries: (offset: u64, length: u32)
        }

        // Build output
        let mut output = Vec::new();

        // Write num_fields
        output.write_u32::<LittleEndian>(field_data.len() as u32)?;

        // Track current data offset (after all headers)
        let mut current_offset = header_size;

        // First, collect all data bytes in order and build offset tables
        let mut all_data: Vec<u8> = Vec::new();
        let mut field_tables: Vec<Vec<(u64, u32)>> = Vec::new();

        for (_, _, max_dim_id, dim_bytes) in &field_data {
            let mut table: Vec<(u64, u32)> = vec![(0, 0); *max_dim_id as usize];

            // Process dimensions in order
            for dim_id in 0..*max_dim_id {
                if let Some(bytes) = dim_bytes.get(&dim_id) {
                    table[dim_id as usize] = (current_offset, bytes.len() as u32);
                    current_offset += bytes.len() as u64;
                    all_data.extend_from_slice(bytes);
                }
                // else: table entry stays (0, 0) meaning dimension not present
            }

            field_tables.push(table);
        }

        // Write field headers and tables
        for (i, (field_id, quantization, max_dim_id, _)) in field_data.iter().enumerate() {
            output.write_u32::<LittleEndian>(*field_id)?;
            output.write_u8(*quantization as u8)?;
            output.write_u32::<LittleEndian>(*max_dim_id)?;

            // Write table (direct indexed by dim_id)
            for &(offset, length) in &field_tables[i] {
                output.write_u64::<LittleEndian>(offset)?;
                output.write_u32::<LittleEndian>(length)?;
            }
        }

        // Write data
        output.extend_from_slice(&all_data);

        Ok(output)
    }

    /// Build positions file for phrase queries
    ///
    /// File format:
    /// - Data only: concatenated serialized PositionPostingList
    /// - Position offsets are stored in TermInfo (no separate header needed)
    ///
    /// Returns: (positions_data, term_key -> (offset, len) mapping)
    #[allow(clippy::type_complexity)]
    fn build_positions_file(&self) -> Result<(Vec<u8>, FxHashMap<Vec<u8>, (u64, u32)>)> {
        use crate::structures::PositionPostingList;

        let mut position_offsets: FxHashMap<Vec<u8>, (u64, u32)> = FxHashMap::default();

        if self.position_index.is_empty() {
            return Ok((Vec::new(), position_offsets));
        }

        // Collect and sort entries by key
        let mut entries: Vec<(Vec<u8>, &PositionPostingListBuilder)> = self
            .position_index
            .iter()
            .map(|(term_key, pos_list)| {
                let term_str = self.term_interner.resolve(&term_key.term);
                let mut key = Vec::with_capacity(4 + term_str.len());
                key.extend_from_slice(&term_key.field.to_le_bytes());
                key.extend_from_slice(term_str.as_bytes());
                (key, pos_list)
            })
            .collect();

        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Serialize all position lists and track offsets
        let mut output = Vec::new();

        for (key, pos_builder) in entries {
            // Convert builder to PositionPostingList
            let mut pos_list = PositionPostingList::with_capacity(pos_builder.postings.len());
            for (doc_id, positions) in &pos_builder.postings {
                pos_list.push(*doc_id, positions.clone());
            }

            // Serialize and track offset
            let offset = output.len() as u64;
            pos_list.serialize(&mut output).map_err(crate::Error::Io)?;
            let len = (output.len() as u64 - offset) as u32;

            position_offsets.insert(key, (offset, len));
        }

        Ok((output, position_offsets))
    }

    /// Build postings from inverted index
    ///
    /// Uses parallel processing to serialize posting lists concurrently.
    /// Position offsets are looked up and embedded in TermInfo.
    fn build_postings(
        &mut self,
        position_offsets: &FxHashMap<Vec<u8>, (u64, u32)>,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
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

                // Don't inline if term has positions (inline format doesn't support position offsets)
                let has_positions = position_offsets.contains_key(&key);
                let result = if !has_positions
                    && let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs)
                {
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

                    // Look up position offset for this term
                    if let Some(&(pos_offset, pos_len)) = position_offsets.get(&key) {
                        TermInfo::external_with_positions(
                            posting_offset,
                            posting_len,
                            doc_count,
                            pos_offset,
                            pos_len,
                        )
                    } else {
                        TermInfo::external(posting_offset, posting_len, doc_count)
                    }
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
