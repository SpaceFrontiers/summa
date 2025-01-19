//! Core data structures for indexing
//!
//! Organized into submodules:
//! - `postings` - Posting list compression formats
//! - `vector` - Vector indexing (RaBitQ, IVF-RaBitQ, ScaNN)
//! - `simd` - SIMD utilities
//! - `sstable` - SSTable for term dictionary

pub mod postings;
pub mod simd;
mod sstable;
pub mod vector;

// Re-export postings
pub use postings::{
    BlockPostingIterator,
    BlockPostingList,
    // Sparse vector
    BlockSparsePostingIterator,
    BlockSparsePostingList,
    // Posting common
    COMMON_BLOCK_SIZE,
    // Posting format
    CompressedPostingIterator,
    CompressedPostingList,
    CompressionStats,
    // Elias-Fano
    EliasFano,
    EliasFanoIterator,
    EliasFanoPostingIterator,
    EliasFanoPostingList,
    // Horizontal BP128
    HORIZONTAL_BP128_BLOCK_SIZE,
    HorizontalBP128Block,
    HorizontalBP128Iterator,
    HorizontalBP128PostingList,
    INLINE_THRESHOLD,
    IndexOptimization,
    IndexSize,
    // OptP4D
    OPT_P4D_BLOCK_SIZE,
    OptP4DBlock,
    OptP4DIterator,
    OptP4DPostingList,
    PARTITIONED_EF_THRESHOLD,
    // Partitioned EF
    PEF_BLOCK_SIZE,
    PEFBlockInfo,
    // Basic posting
    POSTING_BLOCK_SIZE,
    PartitionedEFPostingIterator,
    PartitionedEFPostingList,
    PartitionedEliasFano,
    Posting,
    PostingFormat,
    PostingList,
    PostingListIterator,
    // Roaring
    ROARING_BLOCK_SIZE,
    ROARING_THRESHOLD_RATIO,
    // Rounded BP128
    ROUNDED_BP128_BLOCK_SIZE,
    RoaringBitmap,
    RoaringBlockInfo,
    RoaringIterator,
    RoaringPostingIterator,
    RoaringPostingList,
    RoundedBP128Block,
    RoundedBP128Iterator,
    RoundedBP128PostingList,
    RoundedBitWidth,
    SMALL_BLOCK_SIZE,
    SMALL_BLOCK_THRESHOLD,
    SPARSE_BLOCK_SIZE,
    SkipEntry,
    SkipList,
    SparseEntry,
    SparsePosting,
    SparsePostingIterator,
    SparsePostingList,
    SparseSkipEntry,
    SparseSkipList,
    SparseVector,
    SparseVectorConfig,
    TERMINATED,
    // Vertical BP128
    VERTICAL_BP128_BLOCK_SIZE,
    VerticalBP128Block,
    VerticalBP128Iterator,
    VerticalBP128PostingList,
    WeightQuantization,
    binary_search_block,
    pack_block,
    pack_deltas_fixed,
    pack_vertical,
    read_doc_id_block,
    read_vint,
    unpack_block,
    unpack_block_n,
    unpack_deltas_fixed,
    unpack_vertical,
    write_doc_id_block,
    write_vint,
};

// Re-export vector
pub use vector::{
    // IVF core
    ClusterData,
    ClusterStorage,
    CoarseCentroids,
    CoarseConfig,
    // Quantization
    DistanceTable,
    // Indexes
    IVFPQConfig,
    IVFPQIndex,
    IVFRaBitQConfig,
    IVFRaBitQIndex,
    MultiAssignment,
    PQCodebook,
    PQConfig,
    PQVector,
    QuantizedCode,
    QuantizedQuery,
    QuantizedVector,
    Quantizer,
    RaBitQCodebook,
    RaBitQConfig,
    RaBitQIndex,
    SoarConfig,
};

// Re-export simd
pub use simd::bits_needed;

// Re-export sstable
pub use sstable::{
    AsyncSSTableIterator, AsyncSSTableReader, BLOCK_SIZE as SSTABLE_BLOCK_SIZE, BloomFilter,
    SSTABLE_MAGIC, SSTableStats, SSTableValue, SSTableWriter, SSTableWriterConfig, TermInfo,
};
