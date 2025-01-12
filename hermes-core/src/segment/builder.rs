//! Segment builder for creating new segments

use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::store::StoreWriter;
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
use crate::structures::{PostingList, SSTableWriter, TermInfo};
use crate::tokenizer::{BoxedTokenizer, LowercaseTokenizer};
use crate::wand::WandData;
use crate::{DocId, Result};

/// Builder for creating a single segment
pub struct SegmentBuilder {
    schema: Schema,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,
    inverted_index: FxHashMap<Field, BTreeMap<Vec<u8>, PostingList>>,
    documents: Vec<Document>,
    next_doc_id: DocId,
    /// Per-field statistics for BM25F
    field_stats: FxHashMap<u32, FieldStats>,
    /// Per-document field lengths (doc_id -> field_id -> token_count)
    doc_field_lengths: Vec<FxHashMap<u32, u32>>,
    /// Optional pre-computed WAND data for IDF values
    wand_data: Option<Arc<WandData>>,
}

impl SegmentBuilder {
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            tokenizers: FxHashMap::default(),
            inverted_index: FxHashMap::default(),
            documents: Vec::new(),
            next_doc_id: 0,
            field_stats: FxHashMap::default(),
            doc_field_lengths: Vec::new(),
            wand_data: None,
        }
    }

    /// Create a new segment builder with pre-computed WAND data
    ///
    /// The WAND data provides IDF values for terms, enabling more accurate
    /// block-max scores during indexing. This is useful when you have
    /// pre-computed term statistics from `hermes-tool term-stats`.
    pub fn with_wand_data(schema: Schema, wand_data: Arc<WandData>) -> Self {
        Self {
            schema,
            tokenizers: FxHashMap::default(),
            inverted_index: FxHashMap::default(),
            documents: Vec::new(),
            next_doc_id: 0,
            field_stats: FxHashMap::default(),
            doc_field_lengths: Vec::new(),
            wand_data: Some(wand_data),
        }
    }

    /// Set WAND data for IDF computation
    pub fn set_wand_data(&mut self, wand_data: Arc<WandData>) {
        self.wand_data = Some(wand_data);
    }

    pub fn set_tokenizer(&mut self, field: Field, tokenizer: BoxedTokenizer) {
        self.tokenizers.insert(field, tokenizer);
    }

    pub fn num_docs(&self) -> u32 {
        self.next_doc_id
    }

    pub fn add_document(&mut self, doc: Document) -> Result<DocId> {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // Track field lengths for this document
        let mut doc_lengths: FxHashMap<u32, u32> = FxHashMap::default();

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

                    // Track per-document field length
                    doc_lengths.insert(field.0, token_count);
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

        self.doc_field_lengths.push(doc_lengths);
        self.documents.push(doc);
        Ok(doc_id)
    }

    /// Index a text field and return the number of tokens
    fn index_text_field(&mut self, field: Field, doc_id: DocId, text: &str) -> Result<u32> {
        let default_tokenizer = LowercaseTokenizer;
        let tokenizer: &dyn crate::tokenizer::TokenizerClone = self
            .tokenizers
            .get(&field)
            .map(|t| t.as_ref())
            .unwrap_or(&default_tokenizer);

        let tokens = tokenizer.tokenize(text);
        let token_count = tokens.len() as u32;

        let field_index = self.inverted_index.entry(field).or_default();

        for token in tokens {
            let term = token.text.as_bytes().to_vec();
            let posting = field_index.entry(term).or_default();
            posting.add(doc_id, 1);
        }

        Ok(token_count)
    }

    fn index_numeric_field(&mut self, field: Field, doc_id: DocId, value: u64) -> Result<()> {
        let term = value.to_le_bytes().to_vec();
        let field_index = self.inverted_index.entry(field).or_default();
        let posting = field_index.entry(term).or_default();
        posting.add(doc_id, 1);
        Ok(())
    }

    pub async fn build<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segment_id: SegmentId,
    ) -> Result<SegmentMeta> {
        self.build_with_threads(dir, segment_id, 1).await
    }

    /// Build segment with parallel compression
    ///
    /// Uses `num_threads` for parallel block compression in the document store.
    #[cfg(feature = "native")]
    pub async fn build_with_threads<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segment_id: SegmentId,
        num_threads: usize,
    ) -> Result<SegmentMeta> {
        let files = SegmentFiles::new(segment_id.0);

        let mut term_dict_data = Vec::new();
        let mut postings_data = Vec::new();
        let mut store_data = Vec::new();

        self.build_postings(&mut term_dict_data, &mut postings_data)?;

        // Use parallel compression if num_threads > 1
        if num_threads > 1 {
            self.build_store_parallel(&mut store_data, num_threads)?;
        } else {
            self.build_store(&mut store_data)?;
        }

        dir.write(&files.term_dict, &term_dict_data).await?;
        dir.write(&files.postings, &postings_data).await?;
        dir.write(&files.store, &store_data).await?;

        let meta = SegmentMeta {
            id: segment_id.0,
            num_docs: self.next_doc_id,
            field_stats: self.field_stats.clone(),
        };

        dir.write(&files.meta, &meta.serialize()?).await?;

        Ok(meta)
    }

    /// Build segment without parallel compression (non-native fallback)
    #[cfg(not(feature = "native"))]
    pub async fn build_with_threads<D: Directory + DirectoryWriter>(
        &self,
        dir: &D,
        segment_id: SegmentId,
        _num_threads: usize,
    ) -> Result<SegmentMeta> {
        let files = SegmentFiles::new(segment_id.0);

        let mut term_dict_data = Vec::new();
        let mut postings_data = Vec::new();
        let mut store_data = Vec::new();

        self.build_postings(&mut term_dict_data, &mut postings_data)?;
        self.build_store(&mut store_data)?;

        dir.write(&files.term_dict, &term_dict_data).await?;
        dir.write(&files.postings, &postings_data).await?;
        dir.write(&files.store, &store_data).await?;

        let meta = SegmentMeta {
            id: segment_id.0,
            num_docs: self.next_doc_id,
            field_stats: self.field_stats.clone(),
        };

        dir.write(&files.meta, &meta.serialize()?).await?;

        Ok(meta)
    }

    fn build_postings(&self, term_dict: &mut Vec<u8>, postings: &mut Vec<u8>) -> Result<()> {
        let mut all_terms: BTreeMap<Vec<u8>, (Field, &PostingList)> = BTreeMap::new();

        for (field, terms) in &self.inverted_index {
            for (term, posting_list) in terms {
                let mut key = Vec::with_capacity(4 + term.len());
                key.extend_from_slice(&field.0.to_le_bytes());
                key.extend_from_slice(term);
                all_terms.insert(key, (*field, posting_list));
            }
        }

        let mut writer = SSTableWriter::<TermInfo>::new(term_dict);

        for (key, (_field, posting_list)) in &all_terms {
            // Try to inline small posting lists
            let doc_ids: Vec<u32> = posting_list.iter().map(|p| p.doc_id).collect();
            let term_freqs: Vec<u32> = posting_list.iter().map(|p| p.term_freq).collect();

            let term_info = if let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs) {
                // Small posting list - inline it directly
                inline
            } else {
                // Large posting list - write to external file
                let posting_offset = postings.len() as u64;
                let block_list =
                    crate::structures::BlockPostingList::from_posting_list(posting_list)?;
                block_list.serialize(postings)?;
                TermInfo::external(
                    posting_offset,
                    (postings.len() as u64 - posting_offset) as u32,
                    posting_list.doc_count(),
                )
            };

            writer.insert(key, &term_info)?;
        }

        writer.finish()?;
        Ok(())
    }

    fn build_store(&self, store_data: &mut Vec<u8>) -> Result<()> {
        let mut writer = StoreWriter::new(store_data);

        for doc in &self.documents {
            writer.store(doc, &self.schema)?;
        }

        writer.finish()?;
        Ok(())
    }

    /// Build store with parallel compression
    #[cfg(feature = "native")]
    fn build_store_parallel(&self, store_data: &mut Vec<u8>, num_threads: usize) -> Result<()> {
        use super::store::ParallelStoreWriter;

        let mut writer = ParallelStoreWriter::new(store_data, num_threads);

        for doc in &self.documents {
            writer.store(doc, &self.schema)?;
        }

        writer.finish()?;
        Ok(())
    }
}
