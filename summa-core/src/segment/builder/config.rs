//! Configuration and statistics types for segment builder

use std::path::PathBuf;

use crate::compression::CompressionLevel;

/// Statistics about segment builder state
#[derive(Debug, Clone, Default)]
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
    /// Documents indexed into a dynamically tokenized field
    /// (`text<lex(by: ...)>`) without any value in the hint field, so the
    /// tokenizer's default applied.
    pub unhinted_dynamic_docs: u64,
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
    /// Sparse vector storage
    pub sparse_vectors_bytes: usize,
    /// Position index storage
    pub position_index_bytes: usize,
}

/// Configuration for segment builder
#[derive(Clone)]
pub struct SegmentBuilderConfig {
    /// Directory for temporary spill files
    pub temp_dir: PathBuf,
    /// Compression level for document store
    pub compression_level: CompressionLevel,
    /// Width of the document-store compression pool. Concurrent builders with
    /// the same width share one process-wide executor.
    pub num_compression_threads: usize,
    /// Initial capacity for term interner
    pub interner_capacity: usize,
    /// Initial capacity for posting lists hashmap
    pub posting_map_capacity: usize,
    /// Term-dictionary compression / bloom configuration (from the index
    /// optimization mode).
    pub optimization: crate::structures::IndexOptimization,
    /// Posting block codec (`docs/posting-codecs.md`).
    pub posting_codec: crate::structures::PostingCodec,
    /// New plain-text columns use versioned byte4 norms. Existing segments retain their scores.
    pub quantized_norms: bool,
    /// New position streams use a compact directory separate from payload pages.
    pub compact_text: bool,
    /// Opt in to compact, score-independent length/TF block bounds.
    pub posting_ratio_bounds: bool,
    /// Opt in to bounded competitive frequency/length envelopes. Implies ratio bounds.
    pub posting_impact_bounds: bool,
    /// Validated flush target for this segment's term dictionary.
    pub term_dict_block_size: crate::structures::SSTableBlockSize,
}

impl SegmentBuilderConfig {
    /// Block-bound metadata this builder writes; impact bounds imply ratio bounds.
    pub fn effective_posting_bounds(&self) -> crate::index::PostingBounds {
        crate::index::PostingBounds::new(self.posting_ratio_bounds, self.posting_impact_bounds)
    }
}

impl Default for SegmentBuilderConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "native")]
            temp_dir: std::env::temp_dir(),
            #[cfg(not(feature = "native"))]
            temp_dir: PathBuf::from("/tmp"),
            compression_level: CompressionLevel(3),
            #[cfg(feature = "native")]
            num_compression_threads: crate::default_compression_threads(),
            #[cfg(not(feature = "native"))]
            num_compression_threads: 1,
            interner_capacity: 1_000_000,
            posting_map_capacity: 500_000,
            optimization: crate::structures::IndexOptimization::default(),
            posting_codec: crate::structures::PostingCodec::default(),
            quantized_norms: false,
            compact_text: false,
            posting_ratio_bounds: false,
            posting_impact_bounds: false,
            term_dict_block_size: crate::structures::SSTableBlockSize::default(),
        }
    }
}
