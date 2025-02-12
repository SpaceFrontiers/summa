//! Streaming segment builder with optimized memory usage
//!
//! Key optimizations:
//! - **String interning**: Terms are interned using `lasso` to avoid repeated allocations
//! - **hashbrown HashMap**: O(1) average insertion instead of BTreeMap's O(log n)
//! - **Streaming document store**: Documents written to disk immediately
//! - **Zero-copy store build**: Pre-serialized doc bytes passed directly to compressor
//! - **Parallel posting serialization**: Rayon parallel sort + serialize
//! - **Inline posting fast path**: Small terms skip PostingList/BlockPostingList entirely

mod config;
mod dense;
#[cfg(feature = "diagnostics")]
mod diagnostics;
mod postings;
mod sparse;
mod store;

pub use config::{MemoryBreakdown, SegmentBuilderConfig, SegmentBuilderStats};

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::PathBuf;

use hashbrown::HashMap;
use lasso::{Rodeo, Spur};
use rustc_hash::FxHashMap;

use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use std::sync::Arc;

use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
use crate::tokenizer::BoxedTokenizer;
use crate::{DocId, Result};

use dense::DenseVectorBuilder;
use postings::{CompactPosting, PositionPostingListBuilder, PostingListBuilder, TermKey};
use sparse::SparseVectorBuilder;

/// Size of the document store buffer before writing to disk
const STORE_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16MB

/// Memory overhead per new term in the inverted index:
/// HashMap entry control byte + padding + TermKey + PostingListBuilder + Vec header
const NEW_TERM_OVERHEAD: usize = size_of::<TermKey>() + size_of::<PostingListBuilder>() + 24;

/// Memory overhead per newly interned string: Spur + arena pointers (2 × usize)
const INTERN_OVERHEAD: usize = size_of::<Spur>() + 2 * size_of::<usize>();

/// Memory overhead per new term in the position index
const NEW_POS_TERM_OVERHEAD: usize =
    size_of::<TermKey>() + size_of::<PositionPostingListBuilder>() + 24;

/// Segment builder with optimized memory usage
///
/// Features:
/// - Streams documents to disk immediately (no in-memory document storage)
/// - Uses string interning for terms (reduced allocations)
/// - Uses hashbrown HashMap (faster than BTreeMap)
pub struct SegmentBuilder {
    schema: Arc<Schema>,
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

    /// Reusable buffer for per-document position tracking (when positions enabled)
    /// Avoids allocating a new hashmap for each text field per document
    local_positions: FxHashMap<Spur, Vec<u32>>,

    /// Reusable buffer for tokenization to avoid per-token String allocations
    token_buffer: String,

    /// Reusable buffer for numeric field term encoding (avoids format!() alloc per call)
    numeric_buffer: String,

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

    /// Incrementally tracked memory estimate (avoids expensive stats() calls)
    estimated_memory: usize,

    /// Reusable buffer for document serialization (avoids per-document allocation)
    doc_serialize_buffer: Vec<u8>,

    /// Fast-field columnar writers per field_id (only for fields with fast=true)
    fast_fields: FxHashMap<u32, crate::structures::fast_field::FastFieldWriter>,
}

impl SegmentBuilder {
    /// Create a new segment builder
    pub fn new(schema: Arc<Schema>, config: SegmentBuilderConfig) -> Result<Self> {
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

        // Initialize fast-field writers for fields with fast=true
        use crate::structures::fast_field::{FastFieldColumnType, FastFieldWriter};
        let mut fast_fields = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if entry.fast {
                let writer = match entry.field_type {
                    FieldType::U64 => FastFieldWriter::new_numeric(FastFieldColumnType::U64),
                    FieldType::I64 => FastFieldWriter::new_numeric(FastFieldColumnType::I64),
                    FieldType::F64 => FastFieldWriter::new_numeric(FastFieldColumnType::F64),
                    FieldType::Text => FastFieldWriter::new_text(),
                    _ => continue, // fast not supported for other types
                };
                fast_fields.insert(field.0, writer);
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
            local_positions: FxHashMap::default(),
            token_buffer: String::with_capacity(64),
            numeric_buffer: String::with_capacity(32),
            config,
            dense_vectors: FxHashMap::default(),
            sparse_vectors: FxHashMap::default(),
            position_index: HashMap::new(),
            position_enabled_fields,
            current_element_ordinal: FxHashMap::default(),
            estimated_memory: 0,
            doc_serialize_buffer: Vec::with_capacity(256),
            fast_fields,
        })
    }

    pub fn set_tokenizer(&mut self, field: Field, tokenizer: BoxedTokenizer) {
        self.tokenizers.insert(field, tokenizer);
    }

    /// Get the current element ordinal for a field and increment it.
    /// Used for multi-valued fields (text, dense_vector, sparse_vector).
    fn next_element_ordinal(&mut self, field_id: u32) -> u32 {
        let ordinal = *self.current_element_ordinal.get(&field_id).unwrap_or(&0);
        *self.current_element_ordinal.entry(field_id).or_insert(0) += 1;
        ordinal
    }

    pub fn num_docs(&self) -> u32 {
        self.next_doc_id
    }

    /// Fast O(1) memory estimate - updated incrementally during indexing
    #[inline]
    pub fn estimated_memory_bytes(&self) -> usize {
        self.estimated_memory
    }

    /// Count total unique sparse dimensions across all fields
    pub fn sparse_dim_count(&self) -> usize {
        self.sparse_vectors.values().map(|b| b.postings.len()).sum()
    }

    /// Get current statistics for debugging performance (expensive - iterates all data)
    pub fn stats(&self) -> SegmentBuilderStats {
        use std::mem::size_of;

        let postings_in_memory: usize =
            self.inverted_index.values().map(|p| p.postings.len()).sum();

        // Size constants computed from actual types
        let compact_posting_size = size_of::<CompactPosting>();
        let vec_overhead = size_of::<Vec<u8>>(); // Vec header: ptr + len + cap = 24 bytes on 64-bit
        let term_key_size = size_of::<TermKey>();
        let posting_builder_size = size_of::<PostingListBuilder>();
        let spur_size = size_of::<lasso::Spur>();
        let sparse_entry_size = size_of::<(DocId, u16, f32)>();

        // hashbrown HashMap entry overhead: key + value + 1 byte control + padding
        // Measured: ~(key_size + value_size + 8) per entry on average
        let hashmap_entry_base_overhead = 8usize;

        // FxHashMap uses same layout as hashbrown
        let fxhashmap_entry_overhead = hashmap_entry_base_overhead;

        // Postings memory
        let postings_bytes: usize = self
            .inverted_index
            .values()
            .map(|p| p.postings.capacity() * compact_posting_size + vec_overhead)
            .sum();

        // Inverted index overhead
        let index_overhead_bytes = self.inverted_index.len()
            * (term_key_size + posting_builder_size + hashmap_entry_base_overhead);

        // Term interner: Rodeo stores strings + metadata
        // Rodeo internal: string bytes + Spur + arena overhead (~2 pointers per string)
        let interner_arena_overhead = 2 * size_of::<usize>();
        let avg_term_len = 8; // Estimated average term length
        let interner_bytes =
            self.term_interner.len() * (avg_term_len + spur_size + interner_arena_overhead);

        // Doc field lengths
        let field_lengths_bytes =
            self.doc_field_lengths.capacity() * size_of::<u32>() + vec_overhead;

        // Dense vectors
        let mut dense_vectors_bytes: usize = 0;
        let mut dense_vector_count: usize = 0;
        let doc_id_ordinal_size = size_of::<(DocId, u16)>();
        for b in self.dense_vectors.values() {
            dense_vectors_bytes += b.vectors.capacity() * size_of::<f32>()
                + b.doc_ids.capacity() * doc_id_ordinal_size
                + 2 * vec_overhead; // Two Vecs
            dense_vector_count += b.doc_ids.len();
        }

        // Local buffers
        let local_tf_entry_size = spur_size + size_of::<u32>() + fxhashmap_entry_overhead;
        let local_tf_buffer_bytes = self.local_tf_buffer.capacity() * local_tf_entry_size;

        // Sparse vectors
        let mut sparse_vectors_bytes: usize = 0;
        for builder in self.sparse_vectors.values() {
            for postings in builder.postings.values() {
                sparse_vectors_bytes += postings.capacity() * sparse_entry_size + vec_overhead;
            }
            // Inner FxHashMap overhead: u32 key + Vec value ptr + overhead
            let inner_entry_size = size_of::<u32>() + vec_overhead + fxhashmap_entry_overhead;
            sparse_vectors_bytes += builder.postings.len() * inner_entry_size;
        }
        // Outer FxHashMap overhead
        let outer_sparse_entry_size =
            size_of::<u32>() + size_of::<SparseVectorBuilder>() + fxhashmap_entry_overhead;
        sparse_vectors_bytes += self.sparse_vectors.len() * outer_sparse_entry_size;

        // Position index
        let mut position_index_bytes: usize = 0;
        for pos_builder in self.position_index.values() {
            for (_, positions) in &pos_builder.postings {
                position_index_bytes += positions.capacity() * size_of::<u32>() + vec_overhead;
            }
            // Vec<(DocId, Vec<u32>)> entry size
            let pos_entry_size = size_of::<DocId>() + vec_overhead;
            position_index_bytes += pos_builder.postings.capacity() * pos_entry_size;
        }
        // HashMap overhead for position_index
        let pos_index_entry_size =
            term_key_size + size_of::<PositionPostingListBuilder>() + hashmap_entry_base_overhead;
        position_index_bytes += self.position_index.len() * pos_index_entry_size;

        let estimated_memory_bytes = postings_bytes
            + index_overhead_bytes
            + interner_bytes
            + field_lengths_bytes
            + dense_vectors_bytes
            + local_tf_buffer_bytes
            + sparse_vectors_bytes
            + position_index_bytes;

        let memory_breakdown = MemoryBreakdown {
            postings_bytes,
            index_overhead_bytes,
            interner_bytes,
            field_lengths_bytes,
            dense_vectors_bytes,
            dense_vector_count,
            sparse_vectors_bytes,
            position_index_bytes,
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
        self.estimated_memory += self.num_indexed_fields * std::mem::size_of::<u32>();

        // Reset element ordinals for this document (for multi-valued fields)
        self.current_element_ordinal.clear();

        for (field, value) in doc.field_values() {
            let Some(entry) = self.schema.get_field_entry(*field) else {
                continue;
            };

            // Dense vectors are written to .vectors when indexed || stored
            // Other field types require indexed
            if !matches!(&entry.field_type, FieldType::DenseVector) && !entry.indexed {
                continue;
            }

            match (&entry.field_type, value) {
                (FieldType::Text, FieldValue::Text(text)) => {
                    let element_ordinal = self.next_element_ordinal(field.0);
                    let token_count =
                        self.index_text_field(*field, doc_id, text, element_ordinal)?;

                    let stats = self.field_stats.entry(field.0).or_default();
                    stats.total_tokens += token_count as u64;
                    if element_ordinal == 0 {
                        stats.doc_count += 1;
                    }

                    if let Some(&slot) = self.field_to_slot.get(&field.0) {
                        self.doc_field_lengths[base_idx + slot] = token_count;
                    }

                    // Fast-field: store raw text for text ordinal column
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_text(doc_id, text);
                    }
                }
                (FieldType::U64, FieldValue::U64(v)) => {
                    self.index_numeric_field(*field, doc_id, *v)?;
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_u64(doc_id, *v);
                    }
                }
                (FieldType::I64, FieldValue::I64(v)) => {
                    self.index_numeric_field(*field, doc_id, *v as u64)?;
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_i64(doc_id, *v);
                    }
                }
                (FieldType::F64, FieldValue::F64(v)) => {
                    self.index_numeric_field(*field, doc_id, v.to_bits())?;
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_f64(doc_id, *v);
                    }
                }
                (FieldType::DenseVector, FieldValue::DenseVector(vec))
                    if entry.indexed || entry.stored =>
                {
                    let ordinal = self.next_element_ordinal(field.0);
                    self.index_dense_vector_field(*field, doc_id, ordinal as u16, vec)?;
                }
                (FieldType::SparseVector, FieldValue::SparseVector(entries)) => {
                    let ordinal = self.next_element_ordinal(field.0);
                    self.index_sparse_vector_field(*field, doc_id, ordinal as u16, entries)?;
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
    /// Uses a custom tokenizer when set for the field (via `set_tokenizer`),
    /// otherwise falls back to an inline zero-allocation path (split_whitespace
    /// + lowercase + strip non-alphanumeric).
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
        // Reuse buffers to avoid allocations
        self.local_tf_buffer.clear();
        // Clear position Vecs in-place (keeps allocated capacity for reuse)
        for v in self.local_positions.values_mut() {
            v.clear();
        }

        let mut token_position = 0u32;

        // Tokenize: use custom tokenizer if set, else inline zero-alloc path.
        // The owned Vec<Token> is computed first so the immutable borrow of
        // self.tokenizers ends before we mutate other fields.
        let custom_tokens = self.tokenizers.get(&field).map(|t| t.tokenize(text));

        if let Some(tokens) = custom_tokens {
            // Custom tokenizer path
            for token in &tokens {
                let is_new_string = !self.term_interner.contains(&token.text);
                let term_spur = self.term_interner.get_or_intern(&token.text);
                if is_new_string {
                    self.estimated_memory += token.text.len() + INTERN_OVERHEAD;
                }
                *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;

                if let Some(mode) = position_mode {
                    let encoded_pos = match mode {
                        PositionMode::Ordinal => element_ordinal << 20,
                        PositionMode::TokenPosition => token.position,
                        PositionMode::Full => (element_ordinal << 20) | token.position,
                    };
                    self.local_positions
                        .entry(term_spur)
                        .or_default()
                        .push(encoded_pos);
                }
            }
            token_position = tokens.len() as u32;
        } else {
            // Inline zero-allocation path: split_whitespace + lowercase + strip non-alphanumeric
            for word in text.split_whitespace() {
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

                let is_new_string = !self.term_interner.contains(&self.token_buffer);
                let term_spur = self.term_interner.get_or_intern(&self.token_buffer);
                if is_new_string {
                    self.estimated_memory += self.token_buffer.len() + INTERN_OVERHEAD;
                }
                *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;

                if let Some(mode) = position_mode {
                    let encoded_pos = match mode {
                        PositionMode::Ordinal => element_ordinal << 20,
                        PositionMode::TokenPosition => token_position,
                        PositionMode::Full => (element_ordinal << 20) | token_position,
                    };
                    self.local_positions
                        .entry(term_spur)
                        .or_default()
                        .push(encoded_pos);
                }

                token_position += 1;
            }
        }

        // Phase 2: Insert aggregated terms into inverted index
        // Now we only do one inverted_index lookup per unique term in doc
        for (&term_spur, &tf) in &self.local_tf_buffer {
            let term_key = TermKey {
                field: field_id,
                term: term_spur,
            };

            let is_new_term = !self.inverted_index.contains_key(&term_key);
            let posting = self
                .inverted_index
                .entry(term_key)
                .or_insert_with(PostingListBuilder::new);
            posting.add(doc_id, tf);

            self.estimated_memory += size_of::<CompactPosting>();
            if is_new_term {
                self.estimated_memory += NEW_TERM_OVERHEAD;
            }

            if position_mode.is_some()
                && let Some(positions) = self.local_positions.get(&term_spur)
            {
                let is_new_pos_term = !self.position_index.contains_key(&term_key);
                let pos_posting = self
                    .position_index
                    .entry(term_key)
                    .or_insert_with(PositionPostingListBuilder::new);
                for &pos in positions {
                    pos_posting.add_position(doc_id, pos);
                }
                self.estimated_memory += positions.len() * size_of::<u32>();
                if is_new_pos_term {
                    self.estimated_memory += NEW_POS_TERM_OVERHEAD;
                }
            }
        }

        Ok(token_position)
    }

    fn index_numeric_field(&mut self, field: Field, doc_id: DocId, value: u64) -> Result<()> {
        use std::fmt::Write;

        self.numeric_buffer.clear();
        write!(self.numeric_buffer, "__num_{}", value).unwrap();
        let is_new_string = !self.term_interner.contains(&self.numeric_buffer);
        let term_spur = self.term_interner.get_or_intern(&self.numeric_buffer);

        let term_key = TermKey {
            field: field.0,
            term: term_spur,
        };

        let is_new_term = !self.inverted_index.contains_key(&term_key);
        let posting = self
            .inverted_index
            .entry(term_key)
            .or_insert_with(PostingListBuilder::new);
        posting.add(doc_id, 1);

        self.estimated_memory += size_of::<CompactPosting>();
        if is_new_term {
            self.estimated_memory += NEW_TERM_OVERHEAD;
        }
        if is_new_string {
            self.estimated_memory += self.numeric_buffer.len() + INTERN_OVERHEAD;
        }

        Ok(())
    }

    /// Index a dense vector field with ordinal tracking
    fn index_dense_vector_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        ordinal: u16,
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

        builder.add(doc_id, ordinal, vector);

        self.estimated_memory += std::mem::size_of_val(vector) + size_of::<(DocId, u16)>();

        Ok(())
    }

    /// Index a sparse vector field using dedicated sparse posting lists
    ///
    /// Collects (doc_id, ordinal, weight) postings per dimension. During commit, these are
    /// converted to BlockSparsePostingList with proper quantization from SparseVectorConfig.
    ///
    /// Weights below the configured `weight_threshold` are not indexed.
    fn index_sparse_vector_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        ordinal: u16,
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

        builder.inc_vector_count();

        for &(dim_id, weight) in entries {
            // Skip weights below threshold
            if weight.abs() < weight_threshold {
                continue;
            }

            let is_new_dim = !builder.postings.contains_key(&dim_id);
            builder.add(dim_id, doc_id, ordinal, weight);
            self.estimated_memory += size_of::<(DocId, u16, f32)>();
            if is_new_dim {
                // HashMap entry overhead + Vec header
                self.estimated_memory += size_of::<u32>() + size_of::<Vec<(DocId, u16, f32)>>() + 8; // 8 = hashmap control byte + padding
            }
        }

        Ok(())
    }

    /// Write document to streaming store (reuses internal buffer to avoid per-doc allocation)
    fn write_document_to_store(&mut self, doc: &Document) -> Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        super::store::serialize_document_into(doc, &self.schema, &mut self.doc_serialize_buffer)?;

        self.store_file
            .write_u32::<LittleEndian>(self.doc_serialize_buffer.len() as u32)?;
        self.store_file.write_all(&self.doc_serialize_buffer)?;

        Ok(())
    }

    /// Build the final segment
    ///
    /// Streams all data directly to disk via StreamingWriter to avoid buffering
    /// entire serialized outputs in memory. Each phase consumes and drops its
    /// source data before the next phase begins.
    pub async fn build<D: Directory + DirectoryWriter>(
        mut self,
        dir: &D,
        segment_id: SegmentId,
        trained: Option<&super::TrainedVectorStructures>,
    ) -> Result<SegmentMeta> {
        // Flush any buffered data
        self.store_file.flush()?;

        let files = SegmentFiles::new(segment_id.0);

        // Phase 1: Stream positions directly to disk (consumes position_index)
        let position_index = std::mem::take(&mut self.position_index);
        let position_offsets = if !position_index.is_empty() {
            let mut pos_writer = dir.streaming_writer(&files.positions).await?;
            let offsets = postings::build_positions_streaming(
                position_index,
                &self.term_interner,
                &mut *pos_writer,
            )?;
            pos_writer.finish()?;
            offsets
        } else {
            FxHashMap::default()
        };

        // Phase 2: 4-way parallel build — postings, store, dense vectors, sparse vectors
        // These are fully independent: different source data, different output files.
        let inverted_index = std::mem::take(&mut self.inverted_index);
        let term_interner = std::mem::replace(&mut self.term_interner, Rodeo::new());
        let store_path = self.store_path.clone();
        let num_compression_threads = self.config.num_compression_threads;
        let compression_level = self.config.compression_level;
        let dense_vectors = std::mem::take(&mut self.dense_vectors);
        let mut sparse_vectors = std::mem::take(&mut self.sparse_vectors);
        let schema = &self.schema;

        // Pre-create all streaming writers (async) before entering sync rayon scope
        let mut term_dict_writer = dir.streaming_writer(&files.term_dict).await?;
        let mut postings_writer = dir.streaming_writer(&files.postings).await?;
        let mut store_writer = dir.streaming_writer(&files.store).await?;
        let mut vectors_writer = if !dense_vectors.is_empty() {
            Some(dir.streaming_writer(&files.vectors).await?)
        } else {
            None
        };
        let mut sparse_writer = if !sparse_vectors.is_empty() {
            Some(dir.streaming_writer(&files.sparse).await?)
        } else {
            None
        };
        let mut fast_fields = std::mem::take(&mut self.fast_fields);
        let num_docs = self.next_doc_id;
        let mut fast_writer = if !fast_fields.is_empty() {
            Some(dir.streaming_writer(&files.fast).await?)
        } else {
            None
        };

        let ((postings_result, store_result), ((vectors_result, sparse_result), fast_result)) =
            rayon::join(
                || {
                    rayon::join(
                        || {
                            postings::build_postings_streaming(
                                inverted_index,
                                term_interner,
                                &position_offsets,
                                &mut *term_dict_writer,
                                &mut *postings_writer,
                            )
                        },
                        || {
                            store::build_store_streaming(
                                &store_path,
                                num_compression_threads,
                                compression_level,
                                &mut *store_writer,
                                num_docs,
                            )
                        },
                    )
                },
                || {
                    rayon::join(
                        || {
                            rayon::join(
                                || -> Result<()> {
                                    if let Some(ref mut w) = vectors_writer {
                                        dense::build_vectors_streaming(
                                            dense_vectors,
                                            schema,
                                            trained,
                                            &mut **w,
                                        )?;
                                    }
                                    Ok(())
                                },
                                || -> Result<()> {
                                    if let Some(ref mut w) = sparse_writer {
                                        sparse::build_sparse_streaming(
                                            &mut sparse_vectors,
                                            schema,
                                            &mut **w,
                                        )?;
                                    }
                                    Ok(())
                                },
                            )
                        },
                        || -> Result<()> {
                            if let Some(ref mut w) = fast_writer {
                                build_fast_fields_streaming(&mut fast_fields, num_docs, &mut **w)?;
                            }
                            Ok(())
                        },
                    )
                },
            );
        postings_result?;
        store_result?;
        vectors_result?;
        sparse_result?;
        fast_result?;
        term_dict_writer.finish()?;
        postings_writer.finish()?;
        store_writer.finish()?;
        if let Some(w) = vectors_writer {
            w.finish()?;
        }
        if let Some(w) = sparse_writer {
            w.finish()?;
        }
        if let Some(w) = fast_writer {
            w.finish()?;
        }
        drop(position_offsets);
        drop(sparse_vectors);

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
}

/// Serialize all fast-field columns to a `.fast` file.
fn build_fast_fields_streaming(
    fast_fields: &mut FxHashMap<u32, crate::structures::fast_field::FastFieldWriter>,
    num_docs: u32,
    writer: &mut dyn Write,
) -> Result<()> {
    use crate::structures::fast_field::{FastFieldTocEntry, write_fast_field_toc_and_footer};

    if fast_fields.is_empty() {
        return Ok(());
    }

    // Sort fields by id for deterministic output
    let mut field_ids: Vec<u32> = fast_fields.keys().copied().collect();
    field_ids.sort_unstable();

    let mut toc_entries: Vec<FastFieldTocEntry> = Vec::with_capacity(field_ids.len());
    let mut current_offset = 0u64;

    for &field_id in &field_ids {
        let ff = fast_fields.get_mut(&field_id).unwrap();
        ff.pad_to(num_docs);

        let (mut toc, bytes_written) = ff.serialize(writer, current_offset)?;
        toc.field_id = field_id;
        current_offset += bytes_written;
        toc_entries.push(toc);
    }

    // Write TOC + footer
    let toc_offset = current_offset;
    write_fast_field_toc_and_footer(writer, toc_offset, &toc_entries)?;

    log::debug!(
        "[fast-fields] wrote {} columns, {} docs, toc at offset {}",
        toc_entries.len(),
        num_docs,
        toc_offset
    );

    Ok(())
}

impl Drop for SegmentBuilder {
    fn drop(&mut self) {
        // Cleanup temp files on drop
        let _ = std::fs::remove_file(&self.store_path);
    }
}
