mod elias_fano;
mod horizontal_bp128;
mod opt_p4d;
mod partitioned_ef;
mod posting;
mod posting_format;
mod roaring;
pub mod simd;
mod sstable;
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
pub use posting::{
    BLOCK_SIZE as POSTING_BLOCK_SIZE, BlockPostingIterator, BlockPostingList, Posting, PostingList,
    PostingListIterator, TERMINATED,
};
pub use posting_format::{
    CompressedPostingIterator, CompressedPostingList, CompressionStats, INLINE_THRESHOLD,
    IndexOptimization, PARTITIONED_EF_THRESHOLD, PostingFormat, ROARING_THRESHOLD_RATIO,
};
pub use roaring::{
    ROARING_BLOCK_SIZE, RoaringBitmap, RoaringBlockInfo, RoaringIterator, RoaringPostingIterator,
    RoaringPostingList,
};
pub use simd::bits_needed;
pub use sstable::{
    AsyncSSTableReader, BLOCK_SIZE as SSTABLE_BLOCK_SIZE, BloomFilter, SSTABLE_MAGIC, SSTableStats,
    SSTableValue, SSTableWriter, SSTableWriterConfig, TermInfo,
};
pub use vertical_bp128::{
    VERTICAL_BP128_BLOCK_SIZE, VerticalBP128Block, VerticalBP128Iterator, VerticalBP128PostingList,
    pack_vertical, unpack_vertical,
};
