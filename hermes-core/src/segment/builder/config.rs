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
