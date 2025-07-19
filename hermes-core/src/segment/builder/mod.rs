//! Streaming segment builder with optimized memory usage
//!
//! Key optimizations:
//! - **String interning**: Terms are interned using `lasso` to avoid repeated allocations
//! - **hashbrown HashMap**: O(1) average insertion instead of BTreeMap's O(log n)
//! - **Streaming document store**: Documents written to disk immediately
//! - **Zero-copy store build**: Pre-serialized doc bytes passed directly to compressor
//! - **Parallel posting serialization**: Rayon parallel sort + serialize
//! - **Inline posting fast path**: Small terms skip PostingList/BlockPostingList entirely

#[cfg_attr(not(feature = "native"), allow(dead_code))]
pub(crate) mod bmp;
mod config;
mod dense;
#[cfg(feature = "diagnostics")]
mod diagnostics;
#[cfg_attr(not(feature = "native"), allow(dead_code))]
pub(crate) mod graph_bisection;
pub use graph_bisection::BpBudget;
mod postings;
mod sparse;
mod store;

pub use config::{MemoryBreakdown, SegmentBuilderConfig, SegmentBuilderStats};

#[cfg(feature = "native")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "native")]
use std::io::BufWriter;
use std::io::Write;
use std::mem::size_of;
#[cfg(feature = "native")]
use std::path::PathBuf;

use hashbrown::HashMap;
use rustc_hash::FxHashMap;

// String interning: lasso on native (fast arena), HashMap on WASM (no C deps)
#[cfg(feature = "native")]
use lasso::{Rodeo, Spur};

#[cfg(not(feature = "native"))]
pub(crate) mod simple_interner {
    use hashbrown::HashMap;

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Spur(u32);

    /// Simple string interner for WASM (replaces lasso::Rodeo).
    /// Stores each string once in a Vec; HashMap maps &str → index.
    pub struct Rodeo {
        /// Canonical storage — each string lives here exactly once.
        strings: Vec<Box<str>>,
        /// Maps borrowed string slices (pointing into `strings`) to their index.
        /// Safety: entries are never removed and Box<str> has a stable address.
        map: HashMap<&'static str, u32>,
    }

    impl Rodeo {
        pub fn new() -> Self {
            Self {
                strings: Vec::new(),
                map: HashMap::new(),
            }
        }

        pub fn get(&self, key: &str) -> Option<Spur> {
            self.map.get(key).map(|&id| Spur(id))
        }

        pub fn get_or_intern(&mut self, key: &str) -> Spur {
            if let Some(&id) = self.map.get(key) {
                return Spur(id);
            }
            let id = self.strings.len() as u32;
            let boxed: Box<str> = key.into();
            // Safety: the Box<str> is stored in self.strings (append-only Vec)
            // and never moved or freed while the Rodeo is alive.
            let static_ref: &'static str = unsafe { &*(boxed.as_ref() as *const str) };
            self.strings.push(boxed);
            self.map.insert(static_ref, id);
            Spur(id)
        }

        pub fn resolve(&self, spur: &Spur) -> &str {
            &self.strings[spur.0 as usize]
        }

        pub fn len(&self) -> usize {
            self.strings.len()
        }
    }
}

#[cfg(not(feature = "native"))]
use simple_interner::{Rodeo, Spur};

use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use std::sync::Arc;

use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
use crate::tokenizer::BoxedTokenizer;
use crate::{DocId, Result};

use dense::{BinaryDenseVectorBuilder, DenseVectorBuilder};
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

/// Packed position encoding is `(element_ordinal << 20) | token_position`:
/// 12 bits of element ordinal, 20 bits of token position. Values beyond these
/// maxima must saturate — a plain shift/or silently corrupts the neighboring
/// bit field (ordinal 4096 wraps to 0 and aliases element 0; token positions
/// >= 2^20 bleed into the ordinal bits).
const MAX_POSITION_ELEMENT_ORDINAL: u32 = (1 << 12) - 1;
const MAX_TOKEN_POSITION: u32 = (1 << 20) - 1;

/// Default BMP vocabulary size when `dims` is unset in the sparse vector
/// config (SPLADE unigram vocabulary). Must match the build-time defaults in
/// `builder/sparse.rs` and `merger/sparse.rs`.
const DEFAULT_BMP_SPARSE_DIMS: u32 = 105879;

/// Human-readable name of a schema field type (matches the SDL/serde names).
fn field_type_name(field_type: &FieldType) -> &'static str {
    match field_type {
        FieldType::Text => "text",
        FieldType::U64 => "u64",
        FieldType::I64 => "i64",
        FieldType::F64 => "f64",
        FieldType::Bytes => "bytes",
        FieldType::SparseVector => "sparse_vector",
        FieldType::DenseVector => "dense_vector",
        FieldType::Json => "json",
        FieldType::BinaryDenseVector => "binary_dense_vector",
    }
}

/// Human-readable name of a document field value's type (matches SDL names).
fn field_value_type_name(value: &FieldValue) -> &'static str {
    match value {
        FieldValue::Text(_) => "text",
        FieldValue::U64(_) => "u64",
        FieldValue::I64(_) => "i64",
        FieldValue::F64(_) => "f64",
        FieldValue::Bytes(_) => "bytes",
        FieldValue::SparseVector(_) => "sparse_vector",
        FieldValue::DenseVector(_) => "dense_vector",
        FieldValue::Json(_) => "json",
        FieldValue::BinaryDenseVector(_) => "binary_dense_vector",
    }
}

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

    /// Spill file for high-frequency posting lists (lazily created on first spill).
    #[cfg(feature = "native")]
    posting_spill_file: Option<BufWriter<File>>,
    #[cfg(feature = "native")]
    posting_spill_path: PathBuf,
    /// Tracks spilled ranges per term key: (file_offset, posting_count).
    #[cfg(feature = "native")]
    posting_spill_index: HashMap<TermKey, Vec<(u64, u32)>>,
    #[cfg(feature = "native")]
    posting_spill_offset: u64,

    /// Streaming document store writer (native: temp file on disk, WASM: in-memory buffer)
    #[cfg(feature = "native")]
    store_file: BufWriter<File>,
    #[cfg(feature = "native")]
    store_path: PathBuf,
    #[cfg(not(feature = "native"))]
    store_buffer: Vec<u8>,

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
    /// Vectors are stored as flat f32 arrays for global IVF-PQ indexing.
    dense_vectors: FxHashMap<u32, DenseVectorBuilder>,

    /// Binary dense vector storage per field: field -> packed-bit vectors
    binary_dense_vectors: FxHashMap<u32, BinaryDenseVectorBuilder>,

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

    /// Whether the once-per-segment position-encoding saturation warning
    /// has already been emitted (see MAX_POSITION_ELEMENT_ORDINAL).
    position_saturation_warned: bool,

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
        #[cfg(feature = "native")]
        let (store_file, store_path, spill_path) = {
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
            let spill_path = config
                .temp_dir
                .join(format!("hermes_spill_{}.tmp", segment_id));
            (store_file, store_path, spill_path)
        };

        // Count indexed fields, track positions, and auto-configure tokenizers
        let registry = crate::tokenizer::TokenizerRegistry::new();
        let mut num_indexed_fields = 0;
        let mut field_to_slot = FxHashMap::default();
        let mut position_enabled_fields = FxHashMap::default();
        let mut tokenizers = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if entry.indexed && matches!(entry.field_type, FieldType::Text) {
                field_to_slot.insert(field.0, num_indexed_fields);
                num_indexed_fields += 1;
                if entry.positions.is_some() {
                    position_enabled_fields.insert(field.0, entry.positions);
                }
                if let Some(ref tok_name) = entry.tokenizer
                    && let Some(tokenizer) = registry.get(tok_name)
                {
                    tokenizers.insert(field, tokenizer);
                }
            }
        }

        // Initialize fast-field writers for fields with fast=true
        use crate::structures::fast_field::{FastFieldColumnType, FastFieldWriter};
        let mut fast_fields = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if entry.fast {
                let writer = if entry.multi {
                    match entry.field_type {
                        FieldType::U64 => {
                            FastFieldWriter::new_numeric_multi(FastFieldColumnType::U64)
                        }
                        FieldType::I64 => {
                            FastFieldWriter::new_numeric_multi(FastFieldColumnType::I64)
                        }
                        FieldType::F64 => {
                            FastFieldWriter::new_numeric_multi(FastFieldColumnType::F64)
                        }
                        FieldType::Text => FastFieldWriter::new_text_multi(),
                        _ => continue,
                    }
                } else {
                    match entry.field_type {
                        FieldType::U64 => FastFieldWriter::new_numeric(FastFieldColumnType::U64),
                        FieldType::I64 => FastFieldWriter::new_numeric(FastFieldColumnType::I64),
                        FieldType::F64 => FastFieldWriter::new_numeric(FastFieldColumnType::F64),
                        FieldType::Text => FastFieldWriter::new_text(),
                        _ => continue,
                    }
                };
                fast_fields.insert(field.0, writer);
            }
        }

        Ok(Self {
            schema,
            tokenizers,
            term_interner: Rodeo::new(),
            inverted_index: HashMap::with_capacity(config.posting_map_capacity),
            #[cfg(feature = "native")]
            posting_spill_file: None,
            #[cfg(feature = "native")]
            posting_spill_path: spill_path,
            #[cfg(feature = "native")]
            posting_spill_index: HashMap::new(),
            #[cfg(feature = "native")]
            posting_spill_offset: 0,
            #[cfg(feature = "native")]
            store_file,
            #[cfg(feature = "native")]
            store_path,
            #[cfg(not(feature = "native"))]
            store_buffer: Vec::with_capacity(STORE_BUFFER_SIZE),
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
            binary_dense_vectors: FxHashMap::default(),
            sparse_vectors: FxHashMap::default(),
            position_index: HashMap::new(),
            position_enabled_fields,
            current_element_ordinal: FxHashMap::default(),
            position_saturation_warned: false,
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

    fn next_vector_ordinal(&mut self, field_id: u32) -> Result<u16> {
        let ordinal = self.next_element_ordinal(field_id);
        u16::try_from(ordinal).map_err(|_| {
            crate::Error::Document(format!(
                "field {field_id} has more than {} vector values in one document",
                u16::MAX as usize + 1
            ))
        })
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
        let spur_size = size_of::<Spur>();
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
        // Binary dense vectors
        for b in self.binary_dense_vectors.values() {
            dense_vectors_bytes += b.vectors.capacity()
                + b.doc_ids.capacity() * doc_id_ordinal_size
                + 2 * vec_overhead;
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

    /// Fail-loud pre-validation of a document's field values against the
    /// schema. Runs BEFORE any builder state is mutated, so a rejected
    /// document never poisons the builder (doc id advanced, postings written,
    /// store write skipped).
    ///
    /// - A value whose runtime type does not match the schema field type
    ///   would previously fall through `add_document`'s match silently: the
    ///   value was stored but never indexed, so queries on the field could
    ///   never match the document. Reject it loudly instead.
    /// - Sparse entries destined for a BMP-format field must fit the
    ///   configured `dims`: the block-max grid only has rows for
    ///   `dim_id < dims`, so out-of-range entries would be silently dropped
    ///   from the grid and silently filtered from queries — permanently
    ///   unsearchable.
    fn validate_document_against_schema(&self, doc: &Document) -> Result<()> {
        for (field, value) in doc.field_values() {
            let Some(entry) = self.schema.get_field_entry(*field) else {
                continue;
            };

            // Mirror the indexing skip below: values that are neither indexed
            // nor fast (and are not vector types) are only stored verbatim.
            if !matches!(
                &entry.field_type,
                FieldType::DenseVector | FieldType::BinaryDenseVector
            ) && !entry.indexed
                && !entry.fast
            {
                continue;
            }

            match (&entry.field_type, value) {
                (FieldType::SparseVector, FieldValue::SparseVector(entries)) => {
                    if let Some(config) = entry.sparse_vector_config.as_ref()
                        && config.format == crate::structures::SparseFormat::Bmp
                    {
                        let dims = config.dims.unwrap_or(DEFAULT_BMP_SPARSE_DIMS);
                        if let Some(&(dim_id, _)) =
                            entries.iter().find(|&&(dim_id, _)| dim_id >= dims)
                        {
                            return Err(crate::Error::Schema(format!(
                                "sparse vector for field '{}' contains dim_id {} out of \
                                 range for the configured BMP dims={}: dimensions >= dims \
                                 are never written to the block-max grid and can never \
                                 match a query; raise `dims` in the field's sparse_vector \
                                 config or fix the embedding model",
                                entry.name, dim_id, dims
                            )));
                        }
                    }
                }
                // Matching (type, value) pairs — indexed by `add_document`.
                (FieldType::Text, FieldValue::Text(_))
                | (FieldType::U64, FieldValue::U64(_))
                | (FieldType::I64, FieldValue::I64(_))
                | (FieldType::F64, FieldValue::F64(_))
                | (FieldType::DenseVector, FieldValue::DenseVector(_))
                | (FieldType::BinaryDenseVector, FieldValue::BinaryDenseVector(_))
                // Stored-only types: no indexing support, value stored verbatim.
                | (FieldType::Bytes, FieldValue::Bytes(_))
                | (FieldType::Json, FieldValue::Json(_)) => {}
                (expected, got) => {
                    return Err(crate::Error::Schema(format!(
                        "type mismatch for field '{}': schema expects a {} value, got {}; \
                         the value would be stored but never indexed, so queries on this \
                         field could never match the document — fix the document or the \
                         schema",
                        entry.name,
                        field_type_name(expected),
                        field_value_type_name(got),
                    )));
                }
            }
        }
        Ok(())
    }

    /// Add a document - streams to disk immediately
    pub fn add_document(&mut self, doc: Document) -> Result<DocId> {
        // Reject schema-mismatched values before mutating any builder state.
        self.validate_document_against_schema(&doc)?;

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

            // Dense/binary vectors are written to .vectors when indexed || stored
            // Other field types require indexed or fast
            if !matches!(
                &entry.field_type,
                FieldType::DenseVector | FieldType::BinaryDenseVector
            ) && !entry.indexed
                && !entry.fast
            {
                continue;
            }

            match (&entry.field_type, value) {
                (FieldType::Text, FieldValue::Text(text)) => {
                    if entry.indexed {
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
                    }

                    // Fast-field: store raw text for text ordinal column
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_text(doc_id, text);
                    }
                }
                (FieldType::U64, FieldValue::U64(v)) => {
                    if entry.indexed {
                        self.index_numeric_field(*field, doc_id, *v)?;
                    }
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_u64(doc_id, *v);
                    }
                }
                (FieldType::I64, FieldValue::I64(v)) => {
                    if entry.indexed {
                        self.index_numeric_field(*field, doc_id, *v as u64)?;
                    }
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_i64(doc_id, *v);
                    }
                }
                (FieldType::F64, FieldValue::F64(v)) => {
                    if entry.indexed {
                        self.index_numeric_field(*field, doc_id, v.to_bits())?;
                    }
                    if let Some(ff) = self.fast_fields.get_mut(&field.0) {
                        ff.add_f64(doc_id, *v);
                    }
                }
                (FieldType::DenseVector, FieldValue::DenseVector(vec))
                    if entry.indexed || entry.stored =>
                {
                    let ordinal = self.next_vector_ordinal(field.0)?;
                    self.index_dense_vector_field(*field, doc_id, ordinal, vec)?;
                }
                (FieldType::BinaryDenseVector, FieldValue::BinaryDenseVector(bytes))
                    if entry.indexed || entry.stored =>
                {
                    let ordinal = self.next_vector_ordinal(field.0)?;
                    self.index_binary_dense_vector_field(*field, doc_id, ordinal, bytes)?;
                }
                (FieldType::SparseVector, FieldValue::SparseVector(entries)) => {
                    let ordinal = self.next_vector_ordinal(field.0)?;
                    self.index_sparse_vector_field(*field, doc_id, ordinal, entries)?;
                }
                // Only reachable for stored-only types (bytes/json) and for
                // vector values on fields that are neither indexed nor
                // stored: type-mismatched values are rejected loudly by
                // `validate_document_against_schema` before this loop.
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

        // Saturate the packed 12-bit ordinal field instead of letting the
        // shift silently wrap (ordinal 4096 << 20 == 0, aliasing element 0).
        let encoded_ordinal = if position_mode.is_some_and(|m| m.tracks_ordinal())
            && element_ordinal > MAX_POSITION_ELEMENT_ORDINAL
        {
            self.warn_position_saturation(
                "element ordinal",
                element_ordinal,
                MAX_POSITION_ELEMENT_ORDINAL,
            );
            MAX_POSITION_ELEMENT_ORDINAL
        } else {
            element_ordinal
        };

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
                let term_spur = if let Some(spur) = self.term_interner.get(&token.text) {
                    spur
                } else {
                    let spur = self.term_interner.get_or_intern(&token.text);
                    self.estimated_memory += token.text.len() + INTERN_OVERHEAD;
                    spur
                };
                *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;

                if let Some(mode) = position_mode {
                    let encoded_pos = match mode {
                        PositionMode::Ordinal => encoded_ordinal << 20,
                        PositionMode::TokenPosition => token.position,
                        PositionMode::Full => {
                            (encoded_ordinal << 20) | self.saturate_token_position(token.position)
                        }
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

                let term_spur = if let Some(spur) = self.term_interner.get(&self.token_buffer) {
                    spur
                } else {
                    let spur = self.term_interner.get_or_intern(&self.token_buffer);
                    self.estimated_memory += self.token_buffer.len() + INTERN_OVERHEAD;
                    spur
                };
                *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;

                if let Some(mode) = position_mode {
                    let encoded_pos = match mode {
                        PositionMode::Ordinal => encoded_ordinal << 20,
                        PositionMode::TokenPosition => token_position,
                        PositionMode::Full => {
                            (encoded_ordinal << 20) | self.saturate_token_position(token_position)
                        }
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

            match self.inverted_index.entry(term_key) {
                hashbrown::hash_map::Entry::Occupied(mut o) => {
                    o.get_mut().add(doc_id, tf);
                    self.estimated_memory += size_of::<CompactPosting>();
                    // Spill large posting lists to disk to reduce peak memory
                    #[cfg(feature = "native")]
                    if o.get().should_spill() {
                        use byteorder::{LittleEndian, WriteBytesExt};

                        let builder = o.get_mut();
                        let count = builder.postings.len() as u32;
                        let offset = self.posting_spill_offset;

                        // Lazily create the spill file on first spill
                        let spill_file = if let Some(ref mut f) = self.posting_spill_file {
                            f
                        } else {
                            self.posting_spill_file = Some(BufWriter::with_capacity(
                                256 * 1024,
                                OpenOptions::new()
                                    .create(true)
                                    .write(true)
                                    .truncate(true)
                                    .open(&self.posting_spill_path)?,
                            ));
                            self.posting_spill_file.as_mut().unwrap()
                        };
                        for p in &builder.postings {
                            spill_file.write_u32::<LittleEndian>(p.doc_id)?;
                            spill_file.write_u16::<LittleEndian>(p.term_freq)?;
                        }
                        self.posting_spill_offset += count as u64 * 6;
                        self.posting_spill_index
                            .entry(term_key)
                            .or_default()
                            .push((offset, count));

                        let freed = builder.postings.len() * size_of::<CompactPosting>();
                        builder.spilled_count += count;
                        builder.postings.clear();
                        self.estimated_memory -= freed;
                    }
                }
                hashbrown::hash_map::Entry::Vacant(v) => {
                    let mut posting = PostingListBuilder::new();
                    posting.add(doc_id, tf);
                    v.insert(posting);
                    self.estimated_memory += size_of::<CompactPosting>() + NEW_TERM_OVERHEAD;
                }
            }

            if position_mode.is_some()
                && let Some(positions) = self.local_positions.get(&term_spur)
            {
                match self.position_index.entry(term_key) {
                    hashbrown::hash_map::Entry::Occupied(mut o) => {
                        for &pos in positions {
                            o.get_mut().add_position(doc_id, pos);
                        }
                        self.estimated_memory += positions.len() * size_of::<u32>();
                    }
                    hashbrown::hash_map::Entry::Vacant(v) => {
                        let mut pos_posting = PositionPostingListBuilder::new();
                        for &pos in positions {
                            pos_posting.add_position(doc_id, pos);
                        }
                        self.estimated_memory +=
                            positions.len() * size_of::<u32>() + NEW_POS_TERM_OVERHEAD;
                        v.insert(pos_posting);
                    }
                }
            }
        }

        Ok(token_position)
    }

    /// Saturate a token position at the 20-bit packed-encoding maximum so it
    /// cannot bleed into the element-ordinal bits.
    #[inline]
    fn saturate_token_position(&mut self, token_position: u32) -> u32 {
        if token_position > MAX_TOKEN_POSITION {
            self.warn_position_saturation("token position", token_position, MAX_TOKEN_POSITION);
            MAX_TOKEN_POSITION
        } else {
            token_position
        }
    }

    /// Warn once per segment when the packed position encoding saturates.
    #[cold]
    fn warn_position_saturation(&mut self, what: &str, value: u32, max: u32) {
        if !self.position_saturation_warned {
            self.position_saturation_warned = true;
            log::warn!(
                "[segment_builder] {what} {value} exceeds the position-encoding limit {max}; \
                 saturating — phrase/ordinal matching degrades for the overflowing \
                 elements/tokens instead of corrupting other documents' matches \
                 (further occurrences in this segment are not logged)"
            );
        }
    }

    fn index_numeric_field(&mut self, field: Field, doc_id: DocId, value: u64) -> Result<()> {
        use std::fmt::Write;

        self.numeric_buffer.clear();
        write!(self.numeric_buffer, "__num_{}", value).unwrap();
        let term_spur = if let Some(spur) = self.term_interner.get(&self.numeric_buffer) {
            spur
        } else {
            let spur = self.term_interner.get_or_intern(&self.numeric_buffer);
            self.estimated_memory += self.numeric_buffer.len() + INTERN_OVERHEAD;
            spur
        };

        let term_key = TermKey {
            field: field.0,
            term: term_spur,
        };

        match self.inverted_index.entry(term_key) {
            hashbrown::hash_map::Entry::Occupied(mut o) => {
                o.get_mut().add(doc_id, 1);
                self.estimated_memory += size_of::<CompactPosting>();
            }
            hashbrown::hash_map::Entry::Vacant(v) => {
                let mut posting = PostingListBuilder::new();
                posting.add(doc_id, 1);
                v.insert(posting);
                self.estimated_memory += size_of::<CompactPosting>() + NEW_TERM_OVERHEAD;
            }
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
        let expected_dim = self
            .schema
            .get_field_entry(field)
            .and_then(|entry| entry.dense_vector_config.as_ref())
            .map(|config| config.dim)
            .ok_or_else(|| crate::Error::Schema("DenseVector field missing config".to_string()))?;
        if dim != expected_dim {
            return Err(crate::Error::Schema(format!(
                "Dense vector dimension mismatch: schema expects {}, got {}",
                expected_dim, dim
            )));
        }
        if let Some((index, value)) = vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(crate::Error::Document(format!(
                "dense vector contains non-finite value {value} at index {index}"
            )));
        }

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

    /// Index a binary dense vector field with ordinal tracking
    fn index_binary_dense_vector_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        ordinal: u16,
        bytes: &[u8],
    ) -> Result<()> {
        let dim_bits = self
            .schema
            .get_field_entry(field)
            .and_then(|e| e.binary_dense_vector_config.as_ref())
            .map(|c| c.dim)
            .ok_or_else(|| {
                crate::Error::Schema("BinaryDenseVector field missing config".to_string())
            })?;

        let expected_byte_len = dim_bits.div_ceil(8);
        if dim_bits == 0 || !dim_bits.is_multiple_of(8) {
            return Err(crate::Error::Schema(format!(
                "Binary vector dimension must be a positive multiple of 8, got {dim_bits}"
            )));
        }
        if bytes.len() != expected_byte_len {
            return Err(crate::Error::Schema(format!(
                "Binary vector byte length mismatch: expected {} (dim={}), got {}",
                expected_byte_len,
                dim_bits,
                bytes.len()
            )));
        }

        let builder = self
            .binary_dense_vectors
            .entry(field.0)
            .or_insert_with(|| BinaryDenseVectorBuilder::new(dim_bits));

        builder.add(doc_id, ordinal, bytes);
        self.estimated_memory += bytes.len() + size_of::<(DocId, u16)>();

        Ok(())
    }

    /// Index a sparse vector field using dedicated sparse posting lists
    ///
    /// Collects (doc_id, ordinal, weight) postings per dimension. During commit, these are
    /// converted to BlockSparsePostingList with proper quantization from SparseVectorConfig.
    ///
    /// Weights below the configured `weight_threshold` are not indexed. When
    /// `doc_mass` is configured, only the top-|weight| entries covering that
    /// fraction of the vector's total |weight| mass are kept (the excessive
    /// tail of SPLADE-style vectors is cropped).
    fn index_sparse_vector_field(
        &mut self,
        field: Field,
        doc_id: DocId,
        ordinal: u16,
        entries: &[(u32, f32)],
    ) -> Result<()> {
        if let Some((index, (_, weight))) = entries
            .iter()
            .enumerate()
            .find(|(_, (_, weight))| !weight.is_finite())
        {
            return Err(crate::Error::Document(format!(
                "sparse vector contains non-finite weight {weight} at index {index}"
            )));
        }
        let (weight_threshold, doc_mass, min_terms) = self
            .schema
            .get_field_entry(field)
            .and_then(|entry| entry.sparse_vector_config.as_ref())
            .map(|config| (config.weight_threshold, config.doc_mass, config.min_terms))
            .unwrap_or((0.0, None, 0));

        let builder = self
            .sparse_vectors
            .entry(field.0)
            .or_insert_with(SparseVectorBuilder::new);

        builder.inc_vector_count();

        // Document-side mass cropping: determine the per-vector weight cutoff
        // below which entries fall outside the doc_mass fraction of total mass.
        // Short vectors (<= min_terms entries) are never cropped.
        let mass_cutoff = match doc_mass {
            Some(mass) if mass < 1.0 && entries.len() > min_terms => {
                let mut weights: Vec<f32> = entries
                    .iter()
                    .map(|&(_, w)| w.abs())
                    .filter(|w| *w >= weight_threshold)
                    .collect();
                weights.sort_unstable_by(|a, b| b.total_cmp(a));
                let total: f64 = weights.iter().map(|&w| w as f64).sum();
                let target = total * mass as f64;
                let mut cumulative = 0.0f64;
                let mut cutoff = 0.0f32;
                for &w in &weights {
                    if cumulative >= target {
                        break;
                    }
                    cumulative += w as f64;
                    cutoff = w;
                }
                cutoff
            }
            _ => 0.0,
        };

        for &(dim_id, weight) in entries {
            // Skip weights below threshold or outside the doc_mass prefix
            if weight.abs() < weight_threshold || weight.abs() < mass_cutoff {
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

        #[cfg(feature = "native")]
        {
            self.store_file
                .write_u32::<LittleEndian>(self.doc_serialize_buffer.len() as u32)?;
            self.store_file.write_all(&self.doc_serialize_buffer)?;
        }
        #[cfg(not(feature = "native"))]
        {
            self.store_buffer
                .write_u32::<LittleEndian>(self.doc_serialize_buffer.len() as u32)?;
            self.store_buffer.write_all(&self.doc_serialize_buffer)?;
            // The in-memory store buffer is often the largest allocation on
            // the wasm branch (native streams docs to a temp file instead).
            // Count it so the memory-budget flush check can see it.
            self.estimated_memory += size_of::<u32>() + self.doc_serialize_buffer.len();
        }

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
        #[cfg(feature = "native")]
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
        #[cfg(feature = "native")]
        let store_path = self.store_path.clone();
        #[cfg(feature = "native")]
        let num_compression_threads = self.config.num_compression_threads;
        let compression_level = self.config.compression_level;
        let dense_vectors = std::mem::take(&mut self.dense_vectors);
        let binary_dense_vectors = std::mem::take(&mut self.binary_dense_vectors);
        let mut sparse_vectors = std::mem::take(&mut self.sparse_vectors);
        let schema = &self.schema;

        // Pre-create all streaming writers (async) before entering sync rayon scope
        // Wrapped in OffsetWriter to track bytes written per phase.
        let mut term_dict_writer =
            super::OffsetWriter::new(dir.streaming_writer(&files.term_dict).await?);
        let mut postings_writer =
            super::OffsetWriter::new(dir.streaming_writer(&files.postings).await?);
        let mut store_writer = super::OffsetWriter::new(dir.streaming_writer(&files.store).await?);
        let mut vectors_writer = if !dense_vectors.is_empty() || !binary_dense_vectors.is_empty() {
            Some(super::OffsetWriter::new(
                dir.streaming_writer(&files.vectors).await?,
            ))
        } else {
            None
        };
        let mut sparse_writer = if !sparse_vectors.is_empty() {
            Some(super::OffsetWriter::new(
                dir.streaming_writer(&files.sparse).await?,
            ))
        } else {
            None
        };
        let mut fast_fields = std::mem::take(&mut self.fast_fields);
        let num_docs = self.next_doc_id;
        let mut fast_writer = if !fast_fields.is_empty() {
            Some(super::OffsetWriter::new(
                dir.streaming_writer(&files.fast).await?,
            ))
        } else {
            None
        };

        #[cfg(feature = "native")]
        {
            if let Some(ref mut f) = self.posting_spill_file {
                f.flush()?;
            }
            let posting_spill_index = std::mem::take(&mut self.posting_spill_index);
            let mut spill_reader_opt = if !posting_spill_index.is_empty() {
                let spill_file = std::fs::File::open(&self.posting_spill_path)?;
                Some((std::io::BufReader::new(spill_file), posting_spill_index))
            } else {
                None
            };

            let ((postings_result, store_result), ((vectors_result, sparse_result), fast_result)) =
                rayon::join(
                    || {
                        rayon::join(
                            || {
                                let spill_arg = spill_reader_opt.as_mut().map(|(r, idx)| {
                                    (
                                        r as &mut std::io::BufReader<std::fs::File>,
                                        idx as &postings::SpillIndex,
                                    )
                                });
                                postings::build_postings_streaming(
                                    inverted_index,
                                    term_interner,
                                    &position_offsets,
                                    &mut term_dict_writer,
                                    &mut postings_writer,
                                    spill_arg,
                                )
                            },
                            || {
                                store::build_store_streaming(
                                    &store_path,
                                    num_compression_threads,
                                    compression_level,
                                    &mut store_writer,
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
                                                binary_dense_vectors,
                                                schema,
                                                trained,
                                                w,
                                            )?;
                                        }
                                        Ok(())
                                    },
                                    || -> Result<()> {
                                        if let Some(ref mut w) = sparse_writer {
                                            sparse::build_sparse_streaming(
                                                &mut sparse_vectors,
                                                schema,
                                                w,
                                            )?;
                                        }
                                        Ok(())
                                    },
                                )
                            },
                            || -> Result<()> {
                                if let Some(ref mut w) = fast_writer {
                                    build_fast_fields_streaming(&mut fast_fields, num_docs, w)?;
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
        }

        #[cfg(not(feature = "native"))]
        {
            postings::build_postings_streaming(
                inverted_index,
                term_interner,
                &position_offsets,
                &mut term_dict_writer,
                &mut postings_writer,
            )?;
            store::build_store_streaming_from_buffer(
                &self.store_buffer,
                compression_level,
                &mut store_writer,
                num_docs,
            )?;
            if let Some(ref mut w) = vectors_writer {
                dense::build_vectors_streaming(
                    dense_vectors,
                    binary_dense_vectors,
                    schema,
                    trained,
                    w,
                )?;
            }
            if let Some(ref mut w) = sparse_writer {
                sparse::build_sparse_streaming(&mut sparse_vectors, schema, w)?;
            }
            if let Some(ref mut w) = fast_writer {
                build_fast_fields_streaming(&mut fast_fields, num_docs, w)?;
            }
        }

        let term_dict_bytes = term_dict_writer.offset() as usize;
        let postings_bytes = postings_writer.offset() as usize;
        let store_bytes = store_writer.offset() as usize;
        let vectors_bytes = vectors_writer.as_ref().map_or(0, |w| w.offset() as usize);
        let sparse_bytes = sparse_writer.as_ref().map_or(0, |w| w.offset() as usize);
        let fast_bytes = fast_writer.as_ref().map_or(0, |w| w.offset() as usize);

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

        log::info!(
            "[segment_build] {} docs: term_dict={}, postings={}, store={}, vectors={}, sparse={}, fast={}",
            num_docs,
            super::format_bytes(term_dict_bytes),
            super::format_bytes(postings_bytes),
            super::format_bytes(store_bytes),
            super::format_bytes(vectors_bytes),
            super::format_bytes(sparse_bytes),
            super::format_bytes(fast_bytes),
        );

        let meta = SegmentMeta {
            id: segment_id.0,
            num_docs: self.next_doc_id,
            field_stats: self.field_stats.clone(),
        };

        // Durable: committed metadata.json will reference this segment, so a
        // torn/unsynced .meta after power loss would make the commit
        // unreadable (every other segment file is fsynced by its streaming
        // writer's finish()).
        dir.write_durable(&files.meta, &meta.serialize()?).await?;

        // Cleanup temp files
        #[cfg(feature = "native")]
        {
            let _ = std::fs::remove_file(&self.store_path);
        }

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

    Ok(())
}

#[cfg(feature = "native")]
impl Drop for SegmentBuilder {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.store_path);
        if self.posting_spill_file.is_some() {
            let _ = std::fs::remove_file(&self.posting_spill_path);
        }
    }
}

#[cfg(test)]
impl SegmentBuilder {
    /// Test helper: all encoded positions recorded for `(field, term)`.
    fn positions_for_term(&self, field: Field, term: &str) -> Vec<u32> {
        let Some(spur) = self.term_interner.get(term) else {
            return Vec::new();
        };
        let key = TermKey {
            field: field.0,
            term: spur,
        };
        self.position_index
            .get(&key)
            .map(|b| {
                b.postings
                    .iter()
                    .flat_map(|(_, ps)| ps.iter().copied())
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::SchemaBuilder;

    fn builder_for(schema: Schema) -> SegmentBuilder {
        SegmentBuilder::new(Arc::new(schema), SegmentBuilderConfig::default()).unwrap()
    }

    // ------------------------------------------------------------------
    // Finding: field values whose runtime type does not match the schema
    // field type fell into `_ => {}` and were silently not indexed while
    // still being stored — queries could never match the document.
    // ------------------------------------------------------------------
    #[test]
    fn test_add_document_rejects_type_mismatched_field_value() {
        let mut sb = SchemaBuilder::default();
        let views = sb.add_u64_field("views", true, true);
        let mut builder = builder_for(sb.build());

        let mut doc = Document::new();
        doc.add_text(views, "123");
        let err = builder
            .add_document(doc)
            .expect_err("schema-mismatched value must be rejected loudly, not silently unindexed");
        let msg = err.to_string();
        assert!(msg.contains("views"), "error must name the field: {msg}");
        assert!(
            msg.contains("u64"),
            "error must name the expected type: {msg}"
        );
        assert!(msg.contains("text"), "error must name the got type: {msg}");

        // The rejected document must not have consumed a doc id (no poisoning).
        assert_eq!(builder.num_docs(), 0);

        // A well-typed document still indexes fine afterwards.
        let mut doc = Document::new();
        doc.add_u64(views, 123);
        builder.add_document(doc).unwrap();
        assert_eq!(builder.num_docs(), 1);
    }

    // ------------------------------------------------------------------
    // Finding: sparse entries with dim_id >= the configured BMP `dims`
    // were accepted at index time but silently dropped from the BMP grid
    // and silently filtered from queries — permanently unsearchable.
    // ------------------------------------------------------------------
    #[test]
    fn test_add_document_rejects_bmp_sparse_dim_out_of_range() {
        use crate::structures::{SparseFormat, SparseVectorConfig};

        let mut sb = SchemaBuilder::default();
        let config = SparseVectorConfig {
            format: SparseFormat::Bmp,
            dims: Some(100),
            ..Default::default()
        };
        let spv = sb.add_sparse_vector_field_with_config("spv", true, false, config);
        let mut builder = builder_for(sb.build());

        // In-range dims are accepted.
        let mut doc = Document::new();
        doc.add_sparse_vector(spv, vec![(50, 1.0)]);
        builder.add_document(doc).unwrap();

        // dim_id >= dims must be rejected with an actionable error.
        let mut doc = Document::new();
        doc.add_sparse_vector(spv, vec![(50, 1.0), (150, 2.0)]);
        let err = builder
            .add_document(doc)
            .expect_err("out-of-range BMP dim must be rejected, not silently unsearchable");
        let msg = err.to_string();
        assert!(msg.contains("spv"), "error must name the field: {msg}");
        assert!(msg.contains("150"), "error must name the dim_id: {msg}");
        assert!(
            msg.contains("100"),
            "error must name the configured dims: {msg}"
        );
        assert_eq!(
            builder.num_docs(),
            1,
            "rejected doc must not consume a doc id"
        );
    }

    #[test]
    fn test_add_document_maxscore_sparse_dims_unbounded() {
        // MaxScore-format sparse fields have no dims bound — large dim ids
        // stay legal (the per-dim TOC addresses any u32 dimension).
        let mut sb = SchemaBuilder::default();
        let spv = sb.add_sparse_vector_field("spv", true, false);
        let mut builder = builder_for(sb.build());

        let mut doc = Document::new();
        doc.add_sparse_vector(spv, vec![(3_000_000, 1.0)]);
        builder.add_document(doc).unwrap();
    }

    // ------------------------------------------------------------------
    // Finding: `(element_ordinal << 20) | token_position` silently
    // corrupted when element_ordinal >= 4096 (shifted out of the u32,
    // aliasing element 0) or token_position >= 2^20 (bleeding into the
    // ordinal bits). Both must saturate at their field maxima.
    // ------------------------------------------------------------------
    #[test]
    fn test_position_element_ordinal_overflow_saturates_instead_of_wrapping() {
        use crate::dsl::PositionMode;

        let mut sb = SchemaBuilder::default();
        let body = sb.add_text_field("body", true, false);
        sb.set_positions(body, PositionMode::Full);
        let mut builder = builder_for(sb.build());

        // 4097 values: element ordinal 4096 does not fit the 12-bit ordinal
        // field ((4096u32 << 20) wraps to 0, colliding with element 0).
        let mut doc = Document::new();
        doc.add_text(body, "anchor");
        for _ in 0..4095 {
            doc.add_text(body, "filler");
        }
        doc.add_text(body, "needle");
        builder.add_document(doc).unwrap();

        let positions = builder.positions_for_term(body, "needle");
        assert_eq!(positions.len(), 1);
        let encoded = positions[0];
        assert_ne!(
            encoded >> 20,
            0,
            "element ordinal 4096 must not alias element 0"
        );
        assert_eq!(
            encoded >> 20,
            4095,
            "overflowing element ordinal must saturate at 4095"
        );
    }

    #[test]
    fn test_position_token_position_overflow_saturates_instead_of_bleeding() {
        use crate::dsl::PositionMode;

        let mut sb = SchemaBuilder::default();
        let body = sb.add_text_field("body", true, false);
        sb.set_positions(body, PositionMode::Full);
        let mut builder = builder_for(sb.build());

        // One value with 2^20 + 1 tokens: the last token's position does not
        // fit the 20-bit position field and would bleed into ordinal bit 0.
        let mut text = "w ".repeat(1 << 20);
        text.push_str("needle");
        let mut doc = Document::new();
        doc.add_text(body, text);
        builder.add_document(doc).unwrap();

        let positions = builder.positions_for_term(body, "needle");
        assert_eq!(positions.len(), 1);
        let encoded = positions[0];
        assert_eq!(
            encoded >> 20,
            0,
            "token position overflow must not decode as a different element ordinal"
        );
        assert_eq!(
            encoded & 0xFFFFF,
            0xFFFFF,
            "overflowing token position must saturate at 2^20 - 1"
        );
    }

    // ------------------------------------------------------------------
    // Finding: a posting-list spill firing between two values of the same
    // document split that document's postings across the spilled range and
    // the in-memory tail; the build-time merge concatenated them without
    // deduplication (inflated doc_freq, doc visited twice, split tf).
    // ------------------------------------------------------------------
    #[cfg(feature = "native")]
    #[tokio::test]
    async fn test_spill_mid_document_does_not_duplicate_postings() {
        use crate::directories::RamDirectory;
        use crate::structures::TERMINATED;

        let mut sb = SchemaBuilder::default();
        let body = sb.add_text_field("body", true, false);
        let schema = Arc::new(sb.build());
        let mut builder =
            SegmentBuilder::new(Arc::clone(&schema), SegmentBuilderConfig::default()).unwrap();

        // Docs 0..16382 each contribute one posting for "hot", leaving the
        // in-memory posting list one entry short of SPILL_THRESHOLD (16384).
        for _ in 0..16383 {
            let mut doc = Document::new();
            doc.add_text(body, "hot");
            builder.add_document(doc).unwrap();
        }

        // Doc 16383 has TWO values containing "hot": indexing the first value
        // reaches the spill threshold and spills the list INCLUDING this doc's
        // entry; the second value then re-adds the same doc to the now-empty
        // in-memory tail.
        let mut doc = Document::new();
        doc.add_text(body, "hot");
        doc.add_text(body, "hot");
        let boundary_doc = builder.add_document(doc).unwrap();
        assert_eq!(boundary_doc, 16383);

        let dir = RamDirectory::new();
        let segment_id = crate::segment::SegmentId::new();
        builder.build(&dir, segment_id, None).await.unwrap();

        let reader = crate::segment::SegmentReader::open(&dir, segment_id, schema, 16)
            .await
            .unwrap();
        let postings = reader
            .get_postings(body, b"hot")
            .await
            .unwrap()
            .expect("postings for 'hot'");
        assert_eq!(
            postings.doc_count(),
            16384,
            "each document must appear exactly once per term (spill-boundary duplicate)"
        );

        // Doc ids must be strictly increasing and the boundary document's
        // split term frequency must be merged into a single posting.
        let mut it = postings.iterator();
        let mut prev: Option<DocId> = None;
        let mut boundary_tf = 0u32;
        let mut d = it.doc();
        while d != TERMINATED {
            if let Some(p) = prev {
                assert!(p < d, "duplicate/unordered doc id {d} after {p}");
            }
            if d == boundary_doc {
                boundary_tf = it.term_freq();
            }
            prev = Some(d);
            d = it.advance();
        }
        assert_eq!(prev, Some(boundary_doc));
        assert_eq!(
            boundary_tf, 2,
            "boundary doc's term frequency must combine both values"
        );
    }
}
