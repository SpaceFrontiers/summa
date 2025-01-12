//! Streaming segment builder with optimized memory usage
//!
//! Key optimizations:
//! - **String interning**: Terms are interned using `lasso` to avoid repeated allocations
//! - **hashbrown HashMap**: O(1) average insertion instead of BTreeMap's O(log n)
//! - **Streaming document store**: Documents written to disk immediately
//! - **Incremental posting flush**: Large posting lists flushed to temp file
//! - **Memory-mapped intermediate files**: Reduces memory pressure
//! - **Arena allocation**: Batch allocations for reduced fragmentation

#[cfg(feature = "native")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "native")]
use std::io::{BufWriter, Write};
#[cfg(feature = "native")]
use std::path::PathBuf;
#[cfg(feature = "native")]
use std::sync::Arc;

#[cfg(feature = "native")]
use hashbrown::HashMap;
#[cfg(feature = "native")]
use lasso::{Rodeo, Spur};
#[cfg(feature = "native")]
use rustc_hash::FxHashMap;

#[cfg(feature = "native")]
use super::store::StoreWriter;
#[cfg(feature = "native")]
use super::types::{FieldStats, SegmentFiles, SegmentId, SegmentMeta};
#[cfg(feature = "native")]
use crate::compression::CompressionLevel;
#[cfg(feature = "native")]
use crate::directories::{Directory, DirectoryWriter};
#[cfg(feature = "native")]
use crate::dsl::{Document, Field, FieldType, FieldValue, Schema};
#[cfg(feature = "native")]
use crate::structures::{PostingList, SSTableWriter, TermInfo};
#[cfg(feature = "native")]
use crate::tokenizer::{BoxedTokenizer, LowercaseTokenizer};
#[cfg(feature = "native")]
use crate::wand::WandData;
#[cfg(feature = "native")]
use crate::{DocId, Result};

/// Threshold for flushing a posting list to disk (number of postings)
#[cfg(feature = "native")]
const POSTING_FLUSH_THRESHOLD: usize = 100_000;

/// Size of the posting spill buffer before writing to disk
#[cfg(feature = "native")]
const SPILL_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16MB

/// Number of shards for the inverted index
/// Power of 2 for fast modulo via bitwise AND
#[cfg(feature = "native")]
const NUM_INDEX_SHARDS: usize = 64;

/// Interned term key combining field and term
#[cfg(feature = "native")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TermKey {
    field: u32,
    term: Spur,
}

#[cfg(feature = "native")]
impl TermKey {
    /// Get shard index for this term key (uses term's internal id for distribution)
    #[inline]
    fn shard(&self) -> usize {
        // Spur wraps NonZero<u32>, get the raw value for sharding
        // Using bitwise AND since NUM_INDEX_SHARDS is power of 2
        (self.term.into_inner().get() as usize) & (NUM_INDEX_SHARDS - 1)
    }
}

/// Sharded inverted index for better cache locality
/// Each shard is a smaller hashmap that fits better in CPU cache
#[cfg(feature = "native")]
struct ShardedInvertedIndex {
    shards: Vec<HashMap<TermKey, SpillablePostingList>>,
}

#[cfg(feature = "native")]
impl ShardedInvertedIndex {
    fn new(capacity_per_shard: usize) -> Self {
        let mut shards = Vec::with_capacity(NUM_INDEX_SHARDS);
        for _ in 0..NUM_INDEX_SHARDS {
            shards.push(HashMap::with_capacity(capacity_per_shard));
        }
        Self { shards }
    }

    #[inline]
    fn get_mut(&mut self, key: &TermKey) -> Option<&mut SpillablePostingList> {
        self.shards[key.shard()].get_mut(key)
    }

    /// Get or insert a posting list for the given term key
    #[inline]
    fn get_or_insert(&mut self, key: TermKey) -> &mut SpillablePostingList {
        self.shards[key.shard()]
            .entry(key)
            .or_insert_with(SpillablePostingList::new)
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
}

/// Iterator over all entries in the sharded index
#[cfg(feature = "native")]
struct ShardedIndexIter<'a> {
    shards: std::slice::Iter<'a, HashMap<TermKey, SpillablePostingList>>,
    current: Option<hashbrown::hash_map::Iter<'a, TermKey, SpillablePostingList>>,
}

#[cfg(feature = "native")]
impl<'a> Iterator for ShardedIndexIter<'a> {
    type Item = (&'a TermKey, &'a SpillablePostingList);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut current) = self.current
                && let Some(item) = current.next()
            {
                return Some(item);
            }
            // Move to next shard
            match self.shards.next() {
                Some(shard) => self.current = Some(shard.iter()),
                None => return None,
            }
        }
    }
}

#[cfg(feature = "native")]
impl<'a> IntoIterator for &'a ShardedInvertedIndex {
    type Item = (&'a TermKey, &'a SpillablePostingList);
    type IntoIter = ShardedIndexIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ShardedIndexIter {
            shards: self.shards.iter(),
            current: None,
        }
    }
}

/// Compact posting entry for in-memory storage
#[cfg(feature = "native")]
#[derive(Clone, Copy)]
struct CompactPosting {
    doc_id: DocId,
    term_freq: u16, // Most term frequencies fit in u16
}

/// Posting list that can spill to disk when too large
#[cfg(feature = "native")]
struct SpillablePostingList {
    /// In-memory postings (hot data)
    memory: Vec<CompactPosting>,
    /// Offset in spill file where flushed postings start (-1 if none)
    spill_offset: i64,
    /// Number of postings in spill file
    spill_count: u32,
}

#[cfg(feature = "native")]
impl SpillablePostingList {
    fn new() -> Self {
        Self {
            memory: Vec::new(),
            spill_offset: -1,
            spill_count: 0,
        }
    }

    #[allow(dead_code)]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            memory: Vec::with_capacity(capacity),
            spill_offset: -1,
            spill_count: 0,
        }
    }

    #[inline]
    fn add(&mut self, doc_id: DocId, term_freq: u32) {
        // Merge with last posting if same doc_id
        if let Some(last) = self.memory.last_mut()
            && last.doc_id == doc_id
        {
            last.term_freq = last.term_freq.saturating_add(term_freq as u16);
            return;
        }
        self.memory.push(CompactPosting {
            doc_id,
            term_freq: term_freq.min(u16::MAX as u32) as u16,
        });
    }

    fn total_count(&self) -> usize {
        self.memory.len() + self.spill_count as usize
    }

    fn needs_spill(&self) -> bool {
        self.memory.len() >= POSTING_FLUSH_THRESHOLD
    }
}

/// Configuration for segment builder
#[cfg(feature = "native")]
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

#[cfg(feature = "native")]
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
/// - Spills large posting lists to disk (bounded memory)
#[cfg(feature = "native")]
pub struct SegmentBuilder {
    schema: Schema,
    config: SegmentBuilderConfig,
    tokenizers: FxHashMap<Field, BoxedTokenizer>,

    /// String interner for terms - O(1) lookup and deduplication
    term_interner: Rodeo,

    /// Sharded inverted index for better cache locality
    /// Each shard is a smaller hashmap that fits better in CPU cache
    inverted_index: ShardedInvertedIndex,

    /// Streaming document store writer
    store_file: BufWriter<File>,
    store_path: PathBuf,

    /// Spill file for large posting lists
    spill_file: Option<BufWriter<File>>,
    spill_path: PathBuf,
    spill_offset: u64,

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

    /// Reusable buffer for terms that need spilling
    terms_to_spill_buffer: Vec<TermKey>,
}

#[cfg(feature = "native")]
impl SegmentBuilder {
    /// Create a new segment builder
    pub fn new(schema: Schema, config: SegmentBuilderConfig) -> Result<Self> {
        let segment_id = uuid::Uuid::new_v4();
        let store_path = config
            .temp_dir
            .join(format!("hermes_store_{}.tmp", segment_id));
        let spill_path = config
            .temp_dir
            .join(format!("hermes_spill_{}.tmp", segment_id));

        let store_file = BufWriter::with_capacity(
            SPILL_BUFFER_SIZE,
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
            inverted_index: ShardedInvertedIndex::new(
                config.posting_map_capacity / NUM_INDEX_SHARDS,
            ),
            store_file,
            store_path,
            spill_file: None,
            spill_path,
            spill_offset: 0,
            next_doc_id: 0,
            field_stats: FxHashMap::default(),
            doc_field_lengths: Vec::new(),
            num_indexed_fields,
            field_to_slot,
            wand_data: None,
            local_tf_buffer: FxHashMap::default(),
            terms_to_spill_buffer: Vec::new(),
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
    /// Optimization: aggregate term frequencies within the document first,
    /// then do a single insert per unique term. This reduces hash lookups
    /// from O(total_tokens) to O(unique_terms_in_doc).
    fn index_text_field(&mut self, field: Field, doc_id: DocId, text: &str) -> Result<u32> {
        let default_tokenizer = LowercaseTokenizer;
        let tokenizer: &dyn crate::tokenizer::TokenizerClone = self
            .tokenizers
            .get(&field)
            .map(|t| t.as_ref())
            .unwrap_or(&default_tokenizer);

        let tokens = tokenizer.tokenize(text);
        let token_count = tokens.len() as u32;

        // Phase 1: Aggregate term frequencies within this document
        // Reuse buffer to avoid allocations
        self.local_tf_buffer.clear();

        for token in tokens {
            // Intern the term - O(1) amortized
            let term_spur = self.term_interner.get_or_intern(&token.text);
            *self.local_tf_buffer.entry(term_spur).or_insert(0) += 1;
        }

        // Phase 2: Insert aggregated terms into inverted index
        // Now we only do one inverted_index lookup per unique term in doc
        // Reuse buffer for terms to spill
        let field_id = field.0;
        self.terms_to_spill_buffer.clear();

        for (&term_spur, &tf) in &self.local_tf_buffer {
            let term_key = TermKey {
                field: field_id,
                term: term_spur,
            };

            let posting = self.inverted_index.get_or_insert(term_key);
            posting.add(doc_id, tf);

            // Mark for spilling if needed
            if posting.needs_spill() {
                self.terms_to_spill_buffer.push(term_key);
            }
        }

        // Phase 3: Spill large posting lists
        for i in 0..self.terms_to_spill_buffer.len() {
            let term_key = self.terms_to_spill_buffer[i];
            self.spill_posting_list(term_key)?;
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

        let posting = self.inverted_index.get_or_insert(term_key);
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

    /// Spill a large posting list to disk
    fn spill_posting_list(&mut self, term_key: TermKey) -> Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let posting = self.inverted_index.get_mut(&term_key).unwrap();

        // Initialize spill file if needed
        if self.spill_file.is_none() {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(true)
                .open(&self.spill_path)?;
            self.spill_file = Some(BufWriter::with_capacity(SPILL_BUFFER_SIZE, file));
        }

        let spill_file = self.spill_file.as_mut().unwrap();

        // Record spill offset if this is first spill for this term
        if posting.spill_offset < 0 {
            posting.spill_offset = self.spill_offset as i64;
        }

        // Write postings to spill file
        for p in &posting.memory {
            spill_file.write_u32::<LittleEndian>(p.doc_id)?;
            spill_file.write_u16::<LittleEndian>(p.term_freq)?;
            self.spill_offset += 6; // 4 bytes doc_id + 2 bytes term_freq
        }

        posting.spill_count += posting.memory.len() as u32;
        posting.memory.clear();
        posting.memory.shrink_to(POSTING_FLUSH_THRESHOLD / 4); // Keep some capacity

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
        if let Some(ref mut spill) = self.spill_file {
            spill.flush()?;
        }

        let files = SegmentFiles::new(segment_id.0);

        // Build term dictionary and postings
        let (term_dict_data, postings_data) = self.build_postings()?;

        // Build document store from streamed data
        let store_data = self.build_store_from_stream()?;

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
        let _ = std::fs::remove_file(&self.spill_path);

        Ok(meta)
    }

    /// Build postings from inverted index
    fn build_postings(&mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        use std::collections::BTreeMap;

        // We need to sort terms for SSTable, so collect into BTreeMap
        // Key format: field_id (4 bytes) + term bytes
        let mut sorted_terms: BTreeMap<Vec<u8>, &SpillablePostingList> = BTreeMap::new();

        for (term_key, posting_list) in &self.inverted_index {
            let term_str = self.term_interner.resolve(&term_key.term);
            let mut key = Vec::with_capacity(4 + term_str.len());
            key.extend_from_slice(&term_key.field.to_le_bytes());
            key.extend_from_slice(term_str.as_bytes());
            sorted_terms.insert(key, posting_list);
        }

        let mut term_dict = Vec::new();
        let mut postings = Vec::new();
        let mut writer = SSTableWriter::<TermInfo>::new(&mut term_dict);

        // Memory-map spill file if it exists
        let spill_mmap = if self.spill_file.is_some() {
            drop(self.spill_file.take()); // Close writer
            let file = File::open(&self.spill_path)?;
            Some(unsafe { memmap2::Mmap::map(&file)? })
        } else {
            None
        };

        for (key, spill_posting) in sorted_terms {
            // Reconstruct full posting list
            let mut full_postings = PostingList::with_capacity(spill_posting.total_count());

            // Read spilled postings first (they come before in-memory ones)
            if spill_posting.spill_offset >= 0
                && let Some(ref mmap) = spill_mmap
            {
                let mut offset = spill_posting.spill_offset as usize;
                for _ in 0..spill_posting.spill_count {
                    let doc_id = u32::from_le_bytes([
                        mmap[offset],
                        mmap[offset + 1],
                        mmap[offset + 2],
                        mmap[offset + 3],
                    ]);
                    let term_freq = u16::from_le_bytes([mmap[offset + 4], mmap[offset + 5]]);
                    full_postings.push(doc_id, term_freq as u32);
                    offset += 6;
                }
            }

            // Add in-memory postings
            for p in &spill_posting.memory {
                full_postings.push(p.doc_id, p.term_freq as u32);
            }

            // Build term info
            let doc_ids: Vec<u32> = full_postings.iter().map(|p| p.doc_id).collect();
            let term_freqs: Vec<u32> = full_postings.iter().map(|p| p.term_freq).collect();

            let term_info = if let Some(inline) = TermInfo::try_inline(&doc_ids, &term_freqs) {
                inline
            } else {
                let posting_offset = postings.len() as u64;
                let block_list =
                    crate::structures::BlockPostingList::from_posting_list(&full_postings)?;
                block_list.serialize(&mut postings)?;
                TermInfo::external(
                    posting_offset,
                    (postings.len() as u64 - posting_offset) as u32,
                    full_postings.doc_count(),
                )
            };

            writer.insert(&key, &term_info)?;
        }

        writer.finish()?;
        Ok((term_dict, postings))
    }

    /// Build document store from streamed temp file
    fn build_store_from_stream(&mut self) -> Result<Vec<u8>> {
        // Memory-map the temp store file
        drop(std::mem::replace(
            &mut self.store_file,
            BufWriter::new(File::create("/dev/null")?),
        ));

        let file = File::open(&self.store_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Re-compress with proper block structure
        let mut store_data = Vec::new();
        let mut store_writer =
            StoreWriter::with_compression_level(&mut store_data, self.config.compression_level);

        let mut offset = 0usize;
        while offset < mmap.len() {
            if offset + 4 > mmap.len() {
                break;
            }

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

            let doc_bytes = &mmap[offset..offset + doc_len];
            offset += doc_len;

            // Deserialize and re-store with proper compression
            if let Ok(doc) = super::store::deserialize_document(doc_bytes, &self.schema) {
                store_writer.store(&doc, &self.schema)?;
            }
        }

        store_writer.finish()?;
        Ok(store_data)
    }
}

#[cfg(feature = "native")]
impl Drop for SegmentBuilder {
    fn drop(&mut self) {
        // Cleanup temp files on drop
        let _ = std::fs::remove_file(&self.store_path);
        let _ = std::fs::remove_file(&self.spill_path);
    }
}
