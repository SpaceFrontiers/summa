//! Single-threaded IndexWriter for WASM.
//!
//! Provides the same logical API as the native `IndexWriter` (create, open,
//! add_document, commit) but runs everything inline — no OS threads, no
//! channels, no condvars.
//!
//! # Architecture
//!
//! ```text
//! add_document() ──► SegmentBuilder (in-memory)
//!                         │
//!                         ▼  (memory budget exceeded or commit())
//!                    build segment ──► DirectoryWriter
//! ```

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::directories::DirectoryWriter;
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
use crate::error::Result;
use crate::index::IndexMetadata;
use crate::segment::{
    SegmentBuilder, SegmentBuilderConfig, SegmentId, validate_vector_value_counts,
};
use crate::tokenizer::BoxedTokenizer;

use super::IndexConfig;

mod mutations;

/// Default memory budget for WASM (32 MB — conservative for browser).
const DEFAULT_WASM_MEMORY_BUDGET: usize = 32 * 1024 * 1024;

/// Minimum docs before auto-flush (avoids tiny segments).
const MIN_DOCS_BEFORE_FLUSH: u32 = 100;

fn default_builder_config(index_config: &IndexConfig) -> SegmentBuilderConfig {
    SegmentBuilderConfig {
        optimization: index_config.optimization,
        posting_codec: index_config.effective_posting_codec(),
        ..SegmentBuilderConfig::default()
    }
}

/// Single-threaded IndexWriter for WASM targets.
///
/// Documents are buffered in a `SegmentBuilder` and flushed to segments
/// when the memory budget is exceeded or `commit()` is called.
pub struct IndexWriter<D: DirectoryWriter + 'static> {
    directory: Arc<D>,
    schema: Arc<Schema>,
    builder_config: SegmentBuilderConfig,
    builder: Option<SegmentBuilder>,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    /// In-memory metadata — avoids reload from directory on commit.
    metadata: IndexMetadata,
    /// Segments built but not yet committed to metadata
    pending_segments: Vec<(String, u32)>,
    /// Memory budget per builder (bytes)
    memory_budget: usize,
    primary_key: Option<super::primary_key::PrimaryKeyIndex>,
    /// A consumed/poisoned builder cannot be retried without losing replacements.
    poisoned: bool,
    publication_pending: bool,
    /// Claims survive failed/cancelled output writes; cleanup runs on retry/abort.
    owned_outputs: std::collections::HashSet<String>,
}

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    /// Create a new index in the directory.
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        let builder_config = default_builder_config(&config);
        Self::create_with_config(directory, schema, config, builder_config).await
    }

    /// Create a new index with custom builder config.
    pub async fn create_with_config(
        directory: D,
        schema: Schema,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
    ) -> Result<Self> {
        crate::dsl::reject_removed_vector_index_types(&schema)
            .map_err(crate::error::Error::Schema)?;
        let directory = Arc::new(directory);
        let schema = Arc::new(schema);
        let metadata = IndexMetadata::new((*schema).clone());
        metadata.save(directory.as_ref()).await?;

        Self::new_with_parts(directory, schema, config, builder_config, metadata).await
    }

    /// Open an existing index for writing.
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        let builder_config = default_builder_config(&config);
        Self::open_with_config(directory, config, builder_config).await
    }

    /// Open an existing index with custom builder config.
    pub async fn open_with_config(
        directory: D,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
    ) -> Result<Self> {
        let directory = Arc::new(directory);
        let metadata = IndexMetadata::load_persisting_migration(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());

        let mut writer =
            Self::new_with_parts(directory, schema, config, builder_config, metadata).await?;
        writer.reclaim_orphans().await?;
        Ok(writer)
    }

    async fn new_with_parts(
        directory: Arc<D>,
        schema: Arc<Schema>,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
        metadata: IndexMetadata,
    ) -> Result<Self> {
        let registry = crate::tokenizer::TokenizerRegistry::new();
        let mut tokenizers = FxHashMap::default();
        for (field, entry) in schema.fields() {
            if matches!(entry.field_type, FieldType::Text)
                && let Some(ref tok_name) = entry.tokenizer
                && let Some(tok) = registry.get(tok_name)
            {
                tokenizers.insert(field, tok);
            }
        }

        let memory_budget = config
            .max_indexing_memory_bytes
            .min(DEFAULT_WASM_MEMORY_BUDGET);

        let mut writer = Self {
            directory,
            schema,
            builder_config,
            builder: None,
            tokenizers,
            metadata,
            pending_segments: Vec::new(),
            memory_budget,
            primary_key: None,
            poisoned: false,
            publication_pending: false,
            owned_outputs: Default::default(),
        };
        writer.initialize_primary_key().await?;
        Ok(writer)
    }

    /// Get the schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the in-memory metadata (reflects latest commit).
    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    /// Get the directory.
    pub fn directory(&self) -> &Arc<D> {
        &self.directory
    }

    /// Set tokenizer for a field.
    pub fn set_tokenizer<T: crate::tokenizer::Tokenizer>(&mut self, field: Field, tokenizer: T) {
        self.tokenizers.insert(field, Box::new(tokenizer));
    }

    fn ensure_builder(&mut self) -> Result<&mut SegmentBuilder> {
        if self.builder.is_none() {
            let mut b = SegmentBuilder::new(Arc::clone(&self.schema), self.builder_config.clone())?;
            for (field, tokenizer) in &self.tokenizers {
                b.set_tokenizer(*field, tokenizer.clone_box());
            }
            self.builder = Some(b);
        }
        Ok(self.builder.as_mut().unwrap())
    }

    /// Pre-validate a document against the schema before it reaches the
    /// `SegmentBuilder`.
    ///
    /// `SegmentBuilder::add_document` is not transactional: it advances its
    /// internal doc id and writes postings before the fallible per-field work
    /// and only writes the document store afterwards, so an error part-way
    /// through leaves the store one document short of the doc count. Every
    /// later `build()` of that builder then fails with a "Store doc count
    /// mismatch", losing the whole buffered batch. The builder cannot be
    /// rolled back from here, so invalid documents are rejected BEFORE any
    /// builder state is mutated. These checks mirror the fallible validations
    /// reachable in `SegmentBuilder::add_document` on the wasm build (the
    /// native-only spill paths do not exist here); all of them are pure
    /// functions of (document, schema).
    fn validate_document(&self, doc: &Document) -> Result<()> {
        validate_vector_value_counts(doc, &self.schema)?;
        let mut stored_count = 0usize;

        for (field, value) in doc.field_values() {
            let Some(entry) = self.schema.get_field_entry(*field) else {
                continue;
            };

            // Stored field IDs remain u16 on disk; only the number of stored
            // field-values was widened to u32 in store v3.
            let stored_in_store = entry.stored
                && !matches!(
                    value,
                    FieldValue::DenseVector(_) | FieldValue::BinaryDenseVector(_)
                );
            if stored_in_store {
                stored_count += 1;
                if field.0 > u16::MAX as u32 {
                    return Err(crate::Error::Document(format!(
                        "stored field id {} exceeds u16",
                        field.0
                    )));
                }
            }

            match (&entry.field_type, value) {
                (FieldType::DenseVector, FieldValue::DenseVector(vec))
                    if entry.indexed || entry.stored =>
                {
                    let expected_dim = entry
                        .dense_vector_config
                        .as_ref()
                        .map(|config| config.dim)
                        .ok_or_else(|| {
                            crate::Error::Schema("DenseVector field missing config".to_string())
                        })?;
                    if vec.len() != expected_dim {
                        return Err(crate::Error::Schema(format!(
                            "Dense vector dimension mismatch: schema expects {}, got {}",
                            expected_dim,
                            vec.len()
                        )));
                    }
                    if let Some((index, v)) = vec.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                        return Err(crate::Error::Document(format!(
                            "dense vector contains non-finite value {v} at index {index}"
                        )));
                    }
                }
                (FieldType::BinaryDenseVector, FieldValue::BinaryDenseVector(bytes))
                    if entry.indexed || entry.stored =>
                {
                    let dim_bits = entry
                        .binary_dense_vector_config
                        .as_ref()
                        .map(|c| c.dim)
                        .ok_or_else(|| {
                            crate::Error::Schema(
                                "BinaryDenseVector field missing config".to_string(),
                            )
                        })?;
                    if dim_bits == 0 || !dim_bits.is_multiple_of(8) {
                        return Err(crate::Error::Schema(format!(
                            "Binary vector dimension must be a positive multiple of 8, got {dim_bits}"
                        )));
                    }
                    let expected_byte_len = dim_bits.div_ceil(8);
                    if bytes.len() != expected_byte_len {
                        return Err(crate::Error::Schema(format!(
                            "Binary vector byte length mismatch: expected {} (dim={}), got {}",
                            expected_byte_len,
                            dim_bits,
                            bytes.len()
                        )));
                    }
                }
                (FieldType::SparseVector, FieldValue::SparseVector(entries))
                    if entry.indexed || entry.fast =>
                {
                    if let Some((index, (_, weight))) = entries
                        .iter()
                        .enumerate()
                        .find(|(_, (_, weight))| !weight.is_finite())
                    {
                        return Err(crate::Error::Document(format!(
                            "sparse vector contains non-finite weight {weight} at index {index}"
                        )));
                    }
                }
                _ => {}
            }
        }

        if u32::try_from(stored_count).is_err() {
            return Err(crate::Error::Document(format!(
                "too many stored field-values in one document (max {})",
                u32::MAX
            )));
        }

        Ok(())
    }

    /// Add a document. Automatically builds a segment when memory budget is exceeded.
    ///
    /// All-or-nothing: an invalid document is rejected without mutating any
    /// writer/builder state, so buffered documents stay committable.
    pub async fn add_document(&mut self, doc: Document) -> Result<()> {
        self.ensure_healthy()?;
        self.validate_document(&doc)?;
        self.ensure_builder()?;
        if let Some(pk) = &self.primary_key {
            pk.check_and_insert(&doc)?;
        }
        self.poisoned = true;
        let b = self.builder.as_mut().unwrap();
        if let Err(e) = b.add_document(doc) {
            // Defensive: `validate_document` mirrors every fallible path in
            // `SegmentBuilder::add_document`, so this should be unreachable.
            // If a new fallible path slips through, the builder is poisoned
            // (doc id advanced without a store write) and committing it would
            // fail with a doc-count mismatch, silently losing every buffered
            // document. Drop the poisoned builder loudly. The transaction must be aborted
            // before this writer can accept or publish further mutations.
            let buffered = self.builder.take().map(|b| b.num_docs()).unwrap_or(0);
            let lost = buffered.saturating_sub(1);
            log::warn!(
                "[wasm_writer] index={} segment builder poisoned by failed add_document ({e}); \
                 discarding {lost} buffered document(s)",
                self.schema.index_label()
            );
            return Err(crate::Error::Internal(format!(
                "document failed mid-indexing and poisoned the segment builder: {e}; \
                 {lost} buffered document(s) were discarded; abort the pending transaction before continuing"
            )));
        }

        self.poisoned = false;

        // Check memory budget (with 20% headroom for build overhead)
        let effective_budget = self.memory_budget * 4 / 5;
        if b.estimated_memory_bytes() >= effective_budget && b.num_docs() >= MIN_DOCS_BEFORE_FLUSH {
            self.flush_builder().await?;
        }

        Ok(())
    }

    /// Add multiple documents.
    pub async fn add_documents(&mut self, documents: Vec<Document>) -> Result<usize> {
        let total = documents.len();
        for doc in documents {
            self.add_document(doc).await?;
        }
        Ok(total)
    }

    /// Flush the current builder to a segment on disk.
    async fn flush_builder(&mut self) -> Result<()> {
        if let Some(builder) = self.builder.take()
            && builder.num_docs() > 0
        {
            let segment_id = SegmentId::new();
            let segment_hex = segment_id.to_hex();
            let doc_count = builder.num_docs();

            log::info!(
                "[wasm_writer] index={} building segment: id={} docs={}",
                self.schema.index_label(),
                segment_hex,
                doc_count
            );

            self.owned_outputs.insert(segment_hex.clone());
            self.poisoned = true;
            builder
                .build(self.directory.as_ref(), segment_id, None)
                .await?;

            self.pending_segments.push((segment_hex, doc_count));
            self.poisoned = false;
        }
        Ok(())
    }

    /// Number of documents in the current (uncommitted) builder.
    pub fn pending_docs(&self) -> u32 {
        self.builder.as_ref().map(|b| b.num_docs()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::default_builder_config;

    #[test]
    fn standard_builder_config_honors_index_posting_policy() {
        let config = crate::index::IndexConfig {
            optimization: crate::structures::IndexOptimization::SizeOptimized,
            ..Default::default()
        };
        let builder = default_builder_config(&config);
        assert_eq!(builder.optimization, config.optimization);
        assert_eq!(builder.posting_codec, crate::structures::PostingCodec::Pfor);
    }
}
