//! Posting list data structures and compression formats
//!
//! This module contains various posting list implementations optimized for
//! different use cases:
//! - `posting` - Basic block-based posting lists
//! - `posting_common` - Shared utilities for posting list encoding
//! - `posting_format` - Adaptive format selection based on list characteristics
//! - `elias_fano` - Elias-Fano encoding for sparse lists
//! - `partitioned_ef` - Partitioned Elias-Fano for better cache locality
//! - `roaring` - Roaring bitmap for dense lists
//! - `horizontal_bp128` - Horizontal bit-packing (SIMD-friendly)
//! - `vertical_bp128` - Vertical bit-packing
//! - `rounded_bp128` - Rounded bit-packing
//! - `opt_p4d` - Optimized Patched Frame-of-Reference Delta
//! - `sparse_vector` - Sparse vector posting lists

mod elias_fano;
mod horizontal_bp128;
mod opt_p4d;
mod partitioned_ef;
mod positions;
mod posting;
mod posting_common;
mod posting_format;
mod roaring;
mod rounded_bp128;
mod sparse;
#[allow(dead_code)]
mod vertical_bp128;

pub use elias_fano::{
    EliasFano, EliasFanoIterator, EliasFanoPostingIterator, EliasFanoPostingList,
};
pub use horizontal_bp128::{
    HORIZONTAL_BP128_BLOCK_SIZE, HorizontalBP128Block, HorizontalBP128Iterator,
    HorizontalBP128PostingList, SMALL_BLOCK_SIZE, SMALL_BLOCK_THRESHOLD, binary_search_block,
    pack_block, unpack_block, unpack_block_n,
};
pub use opt_p4d::{OPT_P4D_BLOCK_SIZE, OptP4DBlock, OptP4DIterator, OptP4DPostingList};
pub use partitioned_ef::{
    PEF_BLOCK_SIZE, PEFBlockInfo, PartitionedEFPostingIterator, PartitionedEFPostingList,
    PartitionedEliasFano,
};
pub use positions::{
    MAX_ELEMENT_ORDINAL, MAX_TOKEN_POSITION, PositionPostingIterator, PositionPostingList,
    PostingWithPositions, decode_element_ordinal, decode_token_position, encode_position,
};
pub use posting::{
    BLOCK_SIZE as POSTING_BLOCK_SIZE, BlockPostingIterator, BlockPostingList, Posting, PostingList,
    PostingListIterator, TERMINATED,
};
pub use posting_common::{
    BLOCK_SIZE as COMMON_BLOCK_SIZE, RoundedBitWidth, SkipEntry, SkipList, pack_deltas_fixed,
    read_doc_id_block, read_vint, unpack_deltas_fixed, write_doc_id_block, write_vint,
};
pub use posting_format::{
    CompressedPostingIterator, CompressedPostingList, CompressionStats, INLINE_THRESHOLD,
    IndexOptimization, PARTITIONED_EF_THRESHOLD, PostingFormat, ROARING_THRESHOLD_RATIO,
};
pub use roaring::{
    ROARING_BLOCK_SIZE, RoaringBitmap, RoaringBlockInfo, RoaringIterator, RoaringPostingIterator,
    RoaringPostingList,
};
pub use rounded_bp128::{
    ROUNDED_BP128_BLOCK_SIZE, RoundedBP128Block, RoundedBP128Iterator, RoundedBP128PostingList,
};
pub use sparse::{
    BlockSparsePostingIterator, BlockSparsePostingList, IndexSize, QueryWeighting,
    SPARSE_BLOCK_SIZE, SparseEntry, SparsePosting, SparsePostingIterator, SparsePostingList,
    SparseQueryConfig, SparseSkipEntry, SparseSkipList, SparseVector, SparseVectorConfig,
    WeightQuantization,
};
pub use vertical_bp128::{
    VERTICAL_BP128_BLOCK_SIZE, VerticalBP128Block, VerticalBP128Iterator, VerticalBP128PostingList,
    pack_vertical, unpack_vertical,
};
