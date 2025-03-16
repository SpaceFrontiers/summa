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
use crate::dsl::{Document, Field, FieldType, Schema};
use crate::error::Result;
use crate::index::IndexMetadata;
use crate::segment::{SegmentBuilder, SegmentBuilderConfig, SegmentId};
use crate::tokenizer::BoxedTokenizer;

use super::IndexConfig;

/// Default memory budget for WASM (32 MB — conservative for browser).
const DEFAULT_WASM_MEMORY_BUDGET: usize = 32 * 1024 * 1024;

/// Minimum docs before auto-flush (avoids tiny segments).
const MIN_DOCS_BEFORE_FLUSH: u32 = 100;

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
}

impl<D: DirectoryWriter + 'static> IndexWriter<D> {
    /// Create a new index in the directory.
    pub async fn create(directory: D, schema: Schema, config: IndexConfig) -> Result<Self> {
        Self::create_with_config(directory, schema, config, SegmentBuilderConfig::default()).await
    }

    /// Create a new index with custom builder config.
    pub async fn create_with_config(
        directory: D,
        schema: Schema,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
    ) -> Result<Self> {
        let directory = Arc::new(directory);
        let schema = Arc::new(schema);
        let metadata = IndexMetadata::new((*schema).clone());
        metadata.save(directory.as_ref()).await?;

        Ok(Self::new_with_parts(
            directory,
            schema,
            config,
            builder_config,
            metadata,
        ))
    }

    /// Open an existing index for writing.
    pub async fn open(directory: D, config: IndexConfig) -> Result<Self> {
        Self::open_with_config(directory, config, SegmentBuilderConfig::default()).await
    }

    /// Open an existing index with custom builder config.
    pub async fn open_with_config(
        directory: D,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
    ) -> Result<Self> {
        let directory = Arc::new(directory);
        let metadata = IndexMetadata::load(directory.as_ref()).await?;
        let schema = Arc::new(metadata.schema.clone());

        Ok(Self::new_with_parts(
            directory,
            schema,
            config,
            builder_config,
            metadata,
        ))
    }

    fn new_with_parts(
        directory: Arc<D>,
        schema: Arc<Schema>,
        config: IndexConfig,
        builder_config: SegmentBuilderConfig,
        metadata: IndexMetadata,
    ) -> Self {
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

        Self {
            directory,
            schema,
            builder_config,
            builder: None,
            tokenizers,
            metadata,
            pending_segments: Vec::new(),
            memory_budget,
        }
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

    /// Add a document. Automatically builds a segment when memory budget is exceeded.
    pub async fn add_document(&mut self, doc: Document) -> Result<()> {
        self.ensure_builder()?;
        let b = self.builder.as_mut().unwrap();
        b.add_document(doc)?;

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
        if let Some(builder) = self.builder.take() {
            if builder.num_docs() > 0 {
                let segment_id = SegmentId::new();
                let segment_hex = segment_id.to_hex();
                let doc_count = builder.num_docs();

                log::info!(
                    "[wasm_writer] building segment: id={} docs={}",
                    segment_hex,
                    doc_count
                );

                builder
                    .build(self.directory.as_ref(), segment_id, None)
                    .await?;

                self.pending_segments.push((segment_hex, doc_count));
            }
        }
        Ok(())
    }

    /// Commit all pending documents and register segments in metadata.
    ///
    /// Returns `true` if new segments were committed.
    pub async fn commit(&mut self) -> Result<bool> {
        self.flush_builder().await?;

        if self.pending_segments.is_empty() {
            return Ok(false);
        }

        // Update in-memory metadata and save to directory
        for (seg_hex, num_docs) in self.pending_segments.drain(..) {
            self.metadata.add_segment(seg_hex, num_docs);
        }
        self.metadata.save(self.directory.as_ref()).await?;

        Ok(true)
    }

    /// Number of documents in the current (uncommitted) builder.
    pub fn pending_docs(&self) -> u32 {
        self.builder.as_ref().map(|b| b.num_docs()).unwrap_or(0)
    }
}
