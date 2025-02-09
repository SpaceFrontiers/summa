//! Streaming segment builder with optimized memory usage
//!
//! Key optimizations:
//! - **String interning**: Terms are interned using `lasso` to avoid repeated allocations
//! - **hashbrown HashMap**: O(1) average insertion instead of BTreeMap's O(log n)
//! - **Streaming document store**: Documents written to disk immediately
//! - **Incremental posting flush**: Large posting lists flushed to temp file
//! - **Memory-mapped intermediate files**: Reduces memory pressure
//! - **Arena allocation**: Batch allocations for reduced fragmentation

mod config;
mod posting;
mod vectors;

pub use config::{MemoryBreakdown, SegmentBuilderConfig, SegmentBuilderStats};

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::PathBuf;

use hashbrown::HashMap;
use lasso::{Rodeo, Spur};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::compression::CompressionLevel;

use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
use crate::structures::{PostingList, SSTableWriter, TermInfo};
use crate::tokenizer::BoxedTokenizer;
use crate::{DocId, Result};

use posting::{
    CompactPosting, PositionPostingListBuilder, PostingListBuilder, SerializedPosting, TermKey,
};
use vectors::{DenseVectorBuilder, SparseVectorBuilder};

use super::vector_data::FlatVectorData;

/// Size of the document store buffer before writing to disk
const STORE_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16MB

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

    /// Incrementally tracked memory estimate (avoids expensive stats() calls)
    estimated_memory: usize,
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
            estimated_memory: 0,
        })
    }

    pub fn set_tokenizer(&mut self, field: Field, tokenizer: BoxedTokenizer) {
        self.tokenizers.insert(field, tokenizer);
    }

    pub fn num_docs(&self) -> u32 {
        self.next_doc_id
    }

    /// Fast O(1) memory estimate - updated incrementally during indexing
    #[inline]
    pub fn estimated_memory_bytes(&self) -> usize {
        self.estimated_memory
    }

    /// Recalibrate incremental memory estimate using capacity-based calculation.
    /// More expensive than estimated_memory_bytes() — O(terms + dims) vs O(1) —
    /// but accounts for Vec capacity growth (doubling) and HashMap table overhead.
    /// Call periodically (e.g. every 1000 docs) to prevent drift.
    pub fn recalibrate_memory(&mut self) {
        self.estimated_memory = self.stats().estimated_memory_bytes;
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
            let entry = self.schema.get_field_entry(*field);
            if entry.is_none() {
                continue;
            }

            let entry = entry.unwrap();
            // Dense vectors are written to .vectors when indexed || stored
            // Other field types require indexed
            let dominated_by_index = matches!(&entry.field_type, FieldType::DenseVector);
            if !dominated_by_index && !entry.indexed {
                continue;
            }

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
                    // Only count each document once, even for multi-value fields
                    if element_ordinal == 0 {
                        stats.doc_count += 1;
                    }

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
                (FieldType::DenseVector, FieldValue::DenseVector(vec))
                    if entry.indexed || entry.stored =>
                {
                    // Dense vectors written to .vectors (not .store) when indexed || stored
                    let element_ordinal = *self.current_element_ordinal.get(&field.0).unwrap_or(&0);
                    self.index_dense_vector_field(*field, doc_id, element_ordinal as u16, vec)?;
                    // Increment element ordinal for next value of this field
                    *self.current_element_ordinal.entry(field.0).or_insert(0) += 1;
                }
                (FieldType::SparseVector, FieldValue::SparseVector(entries)) => {
                    // Get current element ordinal for multi-valued fields
                    let element_ordinal = *self.current_element_ordinal.get(&field.0).unwrap_or(&0);
                    self.index_sparse_vector_field(
                        *field,
                        doc_id,
                        element_ordinal as u16,
                        entries,
                    )?;
                    // Increment element ordinal for next value of this field
                    *self.current_element_ordinal.entry(field.0).or_insert(0) += 1;
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
            let is_new_string = !self.term_interner.contains(&self.token_buffer);
            let term_spur = self.term_interner.get_or_intern(&self.token_buffer);
            if is_new_string {
                use std::mem::size_of;
                // string bytes + Spur + arena overhead (2 pointers)
                self.estimated_memory +=
                    self.token_buffer.len() + size_of::<Spur>() + 2 * size_of::<usize>();
            }
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

            let is_new_term = !self.inverted_index.contains_key(&term_key);
            let posting = self
                .inverted_index
                .entry(term_key)
                .or_insert_with(PostingListBuilder::new);
            posting.add(doc_id, tf);

            // Incremental memory tracking
            use std::mem::size_of;
            self.estimated_memory += size_of::<CompactPosting>();
            if is_new_term {
                // HashMap entry overhead + PostingListBuilder + Vec header
                self.estimated_memory +=
                    size_of::<TermKey>() + size_of::<PostingListBuilder>() + 24;
            }

            // Add positions if enabled
            if position_mode.is_some()
                && let Some(positions) = local_positions.get(&term_spur)
            {
                let is_new_pos_term = !self.position_index.contains_key(&term_key);
                let pos_posting = self
                    .position_index
                    .entry(term_key)
                    .or_insert_with(PositionPostingListBuilder::new);
                for &pos in positions {
                    pos_posting.add_position(doc_id, pos);
                }
                // Incremental memory tracking for position index
                self.estimated_memory += positions.len() * size_of::<u32>();
                if is_new_pos_term {
                    self.estimated_memory +=
                        size_of::<TermKey>() + size_of::<PositionPostingListBuilder>() + 24;
                }
            }
        }

        Ok(token_position)
    }

    fn index_numeric_field(&mut self, field: Field, doc_id: DocId, value: u64) -> Result<()> {
        use std::mem::size_of;

        // For numeric fields, we use a special encoding
        let term_str = format!("__num_{}", value);
        let is_new_string = !self.term_interner.contains(&term_str);
        let term_spur = self.term_interner.get_or_intern(&term_str);

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

        // Incremental memory tracking
        self.estimated_memory += size_of::<CompactPosting>();
        if is_new_term {
            self.estimated_memory += size_of::<TermKey>() + size_of::<PostingListBuilder>() + 24;
        }
        if is_new_string {
            self.estimated_memory += term_str.len() + size_of::<Spur>() + 2 * size_of::<usize>();
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

        // Incremental memory tracking
        use std::mem::{size_of, size_of_val};
        self.estimated_memory += size_of_val(vector) + size_of::<(DocId, u16)>();

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

        for &(dim_id, weight) in entries {
            // Skip weights below threshold
            if weight.abs() < weight_threshold {
                continue;
            }

            // Incremental memory tracking
            use std::mem::size_of;
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
            let offsets = Self::build_positions_streaming(
                position_index,
                &self.term_interner,
                &mut *pos_writer,
            )?;
            pos_writer.finish()?;
            offsets
        } else {
            FxHashMap::default()
        };

        // Phase 2: Build postings + store — stream directly to disk
        let inverted_index = std::mem::take(&mut self.inverted_index);
        let term_interner = std::mem::replace(&mut self.term_interner, Rodeo::new());
        let store_path = self.store_path.clone();
        let schema_clone = self.schema.clone();
        let num_compression_threads = self.config.num_compression_threads;
        let compression_level = self.config.compression_level;

        // Create streaming writers (async) before entering sync build phases
        let mut term_dict_writer = dir.streaming_writer(&files.term_dict).await?;
        let mut postings_writer = dir.streaming_writer(&files.postings).await?;
        let mut store_writer = dir.streaming_writer(&files.store).await?;

        let (postings_result, store_result) = rayon::join(
            || {
                Self::build_postings_streaming(
                    inverted_index,
                    term_interner,
                    &position_offsets,
                    &mut *term_dict_writer,
                    &mut *postings_writer,
                )
            },
            || {
                Self::build_store_streaming(
                    &store_path,
                    &schema_clone,
                    num_compression_threads,
                    compression_level,
                    &mut *store_writer,
                )
            },
        );
        postings_result?;
        store_result?;
        term_dict_writer.finish()?;
        postings_writer.finish()?;
        store_writer.finish()?;
        drop(position_offsets);

        // Phase 3: Dense vectors — stream directly to disk
        let dense_vectors = std::mem::take(&mut self.dense_vectors);
        if !dense_vectors.is_empty() {
            let mut writer = dir.streaming_writer(&files.vectors).await?;
            Self::build_vectors_streaming(dense_vectors, &self.schema, trained, &mut *writer)?;
            writer.finish()?;
        }

        // Phase 4: Sparse vectors — stream directly to disk
        let mut sparse_vectors = std::mem::take(&mut self.sparse_vectors);
        if !sparse_vectors.is_empty() {
            let mut writer = dir.streaming_writer(&files.sparse).await?;
            Self::build_sparse_streaming(&mut sparse_vectors, &self.schema, &mut *writer)?;
            drop(sparse_vectors);
            writer.finish()?;
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

    /// Stream dense vectors directly to disk (zero-buffer for vector data).
    ///
    /// Computes sizes deterministically (no trial serialization needed), writes
    /// a small header, then streams each field's raw f32 data directly to the writer.
    fn build_vectors_streaming(
        dense_vectors: FxHashMap<u32, DenseVectorBuilder>,
        schema: &Schema,
        trained: Option<&super::TrainedVectorStructures>,
        writer: &mut dyn Write,
    ) -> Result<()> {
        use crate::dsl::{DenseVectorQuantization, VectorIndexType};
        use byteorder::{LittleEndian, WriteBytesExt};

        let mut fields: Vec<(u32, DenseVectorBuilder)> = dense_vectors
            .into_iter()
            .filter(|(_, b)| b.len() > 0)
            .collect();
        fields.sort_by_key(|(id, _)| *id);

        if fields.is_empty() {
            return Ok(());
        }

        // Resolve quantization config per field from schema
        let quants: Vec<DenseVectorQuantization> = fields
            .iter()
            .map(|(field_id, _)| {
                schema
                    .get_field_entry(Field(*field_id))
                    .and_then(|e| e.dense_vector_config.as_ref())
                    .map(|c| c.quantization)
                    .unwrap_or(DenseVectorQuantization::F32)
            })
            .collect();

        // Compute sizes using deterministic formula (no serialization needed)
        let mut field_sizes: Vec<usize> = Vec::with_capacity(fields.len());
        for (i, (_field_id, builder)) in fields.iter().enumerate() {
            field_sizes.push(FlatVectorData::serialized_binary_size(
                builder.dim,
                builder.len(),
                quants[i],
            ));
        }

        /// Magic number for vectors file footer ("VEC2" in LE)
        const VECTORS_FOOTER_MAGIC: u32 = 0x32434556;

        // Data-first format: stream field data, then write TOC + footer at end.
        // Data starts at file offset 0 → mmap page-aligned, no alignment copies.
        struct TocEntry {
            field_id: u32,
            index_type: u8,
            offset: u64,
            size: u64,
        }
        let mut toc: Vec<TocEntry> = Vec::with_capacity(fields.len() * 2);
        let mut current_offset = 0u64;

        // Pre-build ANN indexes while we still have access to the raw vectors.
        // Each ANN blob is (field_id, index_type, serialized_bytes).
        let mut ann_blobs: Vec<(u32, u8, Vec<u8>)> = Vec::new();
        if let Some(trained) = trained {
            for (field_id, builder) in &fields {
                let config = schema
                    .get_field_entry(Field(*field_id))
                    .and_then(|e| e.dense_vector_config.as_ref());

                if let Some(config) = config {
                    let dim = builder.dim;
                    let blob = match config.index_type {
                        VectorIndexType::IvfRaBitQ if trained.centroids.contains_key(field_id) => {
                            let centroids = &trained.centroids[field_id];
                            let (mut index, codebook) =
                                super::ann_build::new_ivf_rabitq(dim, centroids);
                            for (i, (doc_id, ordinal)) in builder.doc_ids.iter().enumerate() {
                                let v = &builder.vectors[i * dim..(i + 1) * dim];
                                index.add_vector(centroids, &codebook, *doc_id, *ordinal, v);
                            }
                            super::ann_build::serialize_ivf_rabitq(index, codebook)
                                .map(|b| (super::ann_build::IVF_RABITQ_TYPE, b))
                        }
                        VectorIndexType::ScaNN
                            if trained.centroids.contains_key(field_id)
                                && trained.codebooks.contains_key(field_id) =>
                        {
                            let centroids = &trained.centroids[field_id];
                            let codebook = &trained.codebooks[field_id];
                            let mut index = super::ann_build::new_scann(dim, centroids, codebook);
                            for (i, (doc_id, ordinal)) in builder.doc_ids.iter().enumerate() {
                                let v = &builder.vectors[i * dim..(i + 1) * dim];
                                index.add_vector(centroids, codebook, *doc_id, *ordinal, v);
                            }
                            super::ann_build::serialize_scann(index, codebook)
                                .map(|b| (super::ann_build::SCANN_TYPE, b))
                        }
                        _ => continue,
                    };
                    match blob {
                        Ok((index_type, bytes)) => {
                            log::info!(
                                "[segment_build] built ANN(type={}) for field {} ({} vectors, {} bytes)",
                                index_type,
                                field_id,
                                builder.doc_ids.len(),
                                bytes.len()
                            );
                            ann_blobs.push((*field_id, index_type, bytes));
                        }
                        Err(e) => {
                            log::warn!(
                                "[segment_build] ANN serialize failed for field {}: {}",
                                field_id,
                                e
                            );
                        }
                    }
                }
            }
        }

        // Stream each field's flat data directly (builder → disk, no intermediate buffer)
        for (i, (_field_id, builder)) in fields.into_iter().enumerate() {
            let data_offset = current_offset;
            FlatVectorData::serialize_binary_from_flat_streaming(
                builder.dim,
                &builder.vectors,
                &builder.doc_ids,
                quants[i],
                writer,
            )
            .map_err(crate::Error::Io)?;
            current_offset += field_sizes[i] as u64;
            toc.push(TocEntry {
                field_id: _field_id,
                index_type: super::ann_build::FLAT_TYPE,
                offset: data_offset,
                size: field_sizes[i] as u64,
            });
            // Pad to 8-byte boundary so next field's mmap slice is aligned
            let pad = (8 - (current_offset % 8)) % 8;
            if pad > 0 {
                writer.write_all(&[0u8; 8][..pad as usize])?;
                current_offset += pad;
            }
            // builder dropped here, freeing vector memory before next field
        }

        // Write ANN blob entries after flat entries
        for (field_id, index_type, blob) in ann_blobs {
            let data_offset = current_offset;
            let blob_len = blob.len() as u64;
            writer.write_all(&blob)?;
            current_offset += blob_len;
            toc.push(TocEntry {
                field_id,
                index_type,
                offset: data_offset,
                size: blob_len,
            });
            let pad = (8 - (current_offset % 8)) % 8;
            if pad > 0 {
                writer.write_all(&[0u8; 8][..pad as usize])?;
                current_offset += pad;
            }
        }

        // Write TOC entries
        let toc_offset = current_offset;
        for entry in &toc {
            writer.write_u32::<LittleEndian>(entry.field_id)?;
            writer.write_u8(entry.index_type)?;
            writer.write_u64::<LittleEndian>(entry.offset)?;
            writer.write_u64::<LittleEndian>(entry.size)?;
        }

        // Write footer: toc_offset(8) + num_fields(4) + magic(4)
        writer.write_u64::<LittleEndian>(toc_offset)?;
        writer.write_u32::<LittleEndian>(toc.len() as u32)?;
        writer.write_u32::<LittleEndian>(VECTORS_FOOTER_MAGIC)?;

        Ok(())
    }

    /// Stream sparse vectors directly to disk (footer-based format).
    ///
    /// Data is written first (one dim at a time), then the TOC and footer
    /// are appended. This matches the dense vectors format pattern.
    fn build_sparse_streaming(
        sparse_vectors: &mut FxHashMap<u32, SparseVectorBuilder>,
        schema: &Schema,
        writer: &mut dyn Write,
    ) -> Result<()> {
        use crate::segment::sparse_format::SPARSE_FOOTER_MAGIC;
        use crate::structures::{BlockSparsePostingList, WeightQuantization};
        use byteorder::{LittleEndian, WriteBytesExt};

        if sparse_vectors.is_empty() {
            return Ok(());
        }

        // Collect and sort fields
        let mut field_ids: Vec<u32> = sparse_vectors.keys().copied().collect();
        field_ids.sort_unstable();

        // Per-field TOC accumulated in memory (~16 bytes per dim)
        struct FieldToc {
            field_id: u32,
            quantization: u8,
            dims: Vec<(u32, u64, u32)>, // (dim_id, data_offset, data_length)
        }
        let mut field_tocs: Vec<FieldToc> = Vec::new();
        let mut current_offset = 0u64;

        for &field_id in &field_ids {
            let builder = sparse_vectors.get_mut(&field_id).unwrap();
            if builder.is_empty() {
                continue;
            }

            let field = crate::dsl::Field(field_id);
            let sparse_config = schema
                .get_field_entry(field)
                .and_then(|e| e.sparse_vector_config.as_ref());

            let quantization = sparse_config
                .map(|c| c.weight_quantization)
                .unwrap_or(WeightQuantization::Float32);

            let block_size = sparse_config.map(|c| c.block_size).unwrap_or(128);
            let pruning_fraction = sparse_config.and_then(|c| c.posting_list_pruning);

            // Sort dim IDs for deterministic output
            let mut dim_ids: Vec<u32> = builder.postings.keys().copied().collect();
            dim_ids.sort_unstable();

            let mut dim_entries: Vec<(u32, u64, u32)> = Vec::with_capacity(dim_ids.len());
            let mut serialize_buf = Vec::new();

            for dim_id in dim_ids {
                let postings = builder.postings.get_mut(&dim_id).unwrap();
                postings.sort_unstable_by_key(|(doc_id, ordinal, _)| (*doc_id, *ordinal));

                if let Some(fraction) = pruning_fraction
                    && postings.len() > 1
                    && fraction < 1.0
                {
                    let original_len = postings.len();
                    postings.sort_by(|a, b| {
                        b.2.abs()
                            .partial_cmp(&a.2.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let keep = ((original_len as f64 * fraction as f64).ceil() as usize).max(1);
                    postings.truncate(keep);
                    postings.sort_unstable_by_key(|(d, o, _)| (*d, *o));
                }

                let block_list = BlockSparsePostingList::from_postings_with_block_size(
                    postings,
                    quantization,
                    block_size,
                )
                .map_err(crate::Error::Io)?;

                serialize_buf.clear();
                block_list
                    .serialize(&mut serialize_buf)
                    .map_err(crate::Error::Io)?;
                writer.write_all(&serialize_buf)?;

                dim_entries.push((dim_id, current_offset, serialize_buf.len() as u32));
                current_offset += serialize_buf.len() as u64;
            }

            if !dim_entries.is_empty() {
                field_tocs.push(FieldToc {
                    field_id,
                    quantization: quantization as u8,
                    dims: dim_entries,
                });
            }
        }

        if field_tocs.is_empty() {
            return Ok(());
        }

        // Write TOC at end
        let toc_offset = current_offset;
        for ftoc in &field_tocs {
            writer.write_u32::<LittleEndian>(ftoc.field_id)?;
            writer.write_u8(ftoc.quantization)?;
            writer.write_u32::<LittleEndian>(ftoc.dims.len() as u32)?;
            for &(dim_id, offset, length) in &ftoc.dims {
                writer.write_u32::<LittleEndian>(dim_id)?;
                writer.write_u64::<LittleEndian>(offset)?;
                writer.write_u32::<LittleEndian>(length)?;
            }
        }

        // Write footer: toc_offset(8) + num_fields(4) + magic(4) = 16 bytes
        writer.write_u64::<LittleEndian>(toc_offset)?;
        writer.write_u32::<LittleEndian>(field_tocs.len() as u32)?;
        writer.write_u32::<LittleEndian>(SPARSE_FOOTER_MAGIC)?;

        Ok(())
    }

    /// Stream positions directly to disk, returning only the offset map.
    ///
    /// Consumes the position_index and writes each position posting list
    /// directly to the writer, tracking offsets for the postings phase.
    fn build_positions_streaming(
        position_index: HashMap<TermKey, PositionPostingListBuilder>,
        term_interner: &Rodeo,
        writer: &mut dyn Write,
    ) -> Result<FxHashMap<Vec<u8>, (u64, u32)>> {
        use crate::structures::PositionPostingList;

        let mut position_offsets: FxHashMap<Vec<u8>, (u64, u32)> = FxHashMap::default();

        // Consume HashMap into Vec for sorting (owned, no borrowing)
        let mut entries: Vec<(Vec<u8>, PositionPostingListBuilder)> = position_index
            .into_iter()
            .map(|(term_key, pos_builder)| {
                let term_str = term_interner.resolve(&term_key.term);
                let mut key = Vec::with_capacity(size_of::<u32>() + term_str.len());
                key.extend_from_slice(&term_key.field.to_le_bytes());
                key.extend_from_slice(term_str.as_bytes());
                (key, pos_builder)
            })
            .collect();

        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut current_offset = 0u64;

        for (key, pos_builder) in entries {
            let mut pos_list = PositionPostingList::with_capacity(pos_builder.postings.len());
            for (doc_id, positions) in pos_builder.postings {
                pos_list.push(doc_id, positions);
            }

            // Serialize to a temp buffer to get exact length, then write
            let mut buf = Vec::new();
            pos_list.serialize(&mut buf).map_err(crate::Error::Io)?;
            writer.write_all(&buf)?;

            position_offsets.insert(key, (current_offset, buf.len() as u32));
            current_offset += buf.len() as u64;
        }

        Ok(position_offsets)
    }

    /// Stream postings directly to disk.
    ///
    /// Parallel serialization of posting lists, then sequential streaming of
    /// term dict and postings data directly to writers (no Vec<u8> accumulation).
    fn build_postings_streaming(
        inverted_index: HashMap<TermKey, PostingListBuilder>,
        term_interner: Rodeo,
        position_offsets: &FxHashMap<Vec<u8>, (u64, u32)>,
        term_dict_writer: &mut dyn Write,
        postings_writer: &mut dyn Write,
    ) -> Result<()> {
        // Phase 1: Consume HashMap into sorted Vec (frees HashMap overhead)
        let mut term_entries: Vec<(Vec<u8>, PostingListBuilder)> = inverted_index
            .into_iter()
            .map(|(term_key, posting_list)| {
                let term_str = term_interner.resolve(&term_key.term);
                let mut key = Vec::with_capacity(4 + term_str.len());
                key.extend_from_slice(&term_key.field.to_le_bytes());
                key.extend_from_slice(term_str.as_bytes());
                (key, posting_list)
            })
            .collect();

        drop(term_interner);

        term_entries.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

        // Phase 2: Parallel serialization
        let serialized: Vec<(Vec<u8>, SerializedPosting)> = term_entries
            .into_par_iter()
            .map(|(key, posting_builder)| {
                let mut full_postings = PostingList::with_capacity(posting_builder.len());
                for p in &posting_builder.postings {
                    full_postings.push(p.doc_id, p.term_freq as u32);
                }

                let doc_ids: Vec<u32> = full_postings.iter().map(|p| p.doc_id).collect();
                let term_freqs: Vec<u32> = full_postings.iter().map(|p| p.term_freq).collect();

                let has_positions = position_offsets.contains_key(&key);
                let result = if !has_positions
                    && let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs)
                {
                    SerializedPosting::Inline(inline)
                } else {
                    let mut posting_bytes = Vec::new();
                    let block_list =
                        crate::structures::BlockPostingList::from_posting_list(&full_postings)?;
                    block_list.serialize(&mut posting_bytes)?;
                    SerializedPosting::External {
                        bytes: posting_bytes,
                        doc_count: full_postings.doc_count(),
                    }
                };

                Ok((key, result))
            })
            .collect::<Result<Vec<_>>>()?;

        // Phase 3: Stream directly to writers (no intermediate Vec<u8> accumulation)
        let mut postings_offset = 0u64;
        let mut writer = SSTableWriter::<_, TermInfo>::new(term_dict_writer);

        for (key, serialized_posting) in serialized {
            let term_info = match serialized_posting {
                SerializedPosting::Inline(info) => info,
                SerializedPosting::External { bytes, doc_count } => {
                    let posting_len = bytes.len() as u32;
                    postings_writer.write_all(&bytes)?;

                    let info = if let Some(&(pos_offset, pos_len)) = position_offsets.get(&key) {
                        TermInfo::external_with_positions(
                            postings_offset,
                            posting_len,
                            doc_count,
                            pos_offset,
                            pos_len,
                        )
                    } else {
                        TermInfo::external(postings_offset, posting_len, doc_count)
                    };
                    postings_offset += posting_len as u64;
                    info
                }
            };

            writer.insert(&key, &term_info)?;
        }

        let _ = writer.finish()?;
        Ok(())
    }

    /// Stream compressed document store directly to disk.
    ///
    /// Reads documents from temp file, compresses in parallel batches,
    /// and writes directly to the streaming writer (no Vec<u8> accumulation).
    fn build_store_streaming(
        store_path: &PathBuf,
        schema: &Schema,
        num_compression_threads: usize,
        compression_level: CompressionLevel,
        writer: &mut dyn Write,
    ) -> Result<()> {
        use super::store::EagerParallelStoreWriter;

        let file = File::open(store_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

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

        const BATCH_SIZE: usize = 10_000;
        let mut store_writer = EagerParallelStoreWriter::with_compression_level(
            writer,
            num_compression_threads,
            compression_level,
        );

        for batch in doc_ranges.chunks(BATCH_SIZE) {
            let batch_docs: Vec<Document> = batch
                .par_iter()
                .filter_map(|&(start, len)| {
                    let doc_bytes = &mmap[start..start + len];
                    super::store::deserialize_document(doc_bytes, schema).ok()
                })
                .collect();

            for doc in &batch_docs {
                store_writer.store(doc, schema)?;
            }
        }

        store_writer.finish()?;
        Ok(())
    }
}

impl Drop for SegmentBuilder {
    fn drop(&mut self) {
        // Cleanup temp files on drop
        let _ = std::fs::remove_file(&self.store_path);
    }
}
