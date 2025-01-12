//! Hermes - A minimal async search engine library
//!
//! Inspired by tantivy/summavy, this library provides:
//! - Fully async IO with Directory abstraction for network/local/memory storage
//! - SSTable-based term dictionary with hot cache and lazy loading
//! - Bitpacked posting lists with block-level skip info
//! - Document store with Zstd compression
//! - Multiple segments with merge support
//! - Text and numeric field support
//! - Term, boolean, and boost queries
//! - BlockWAND / MaxScore query optimizations

pub mod compression;
pub mod directories;
pub mod dsl;
pub mod error;
pub mod index;
pub mod query;
pub mod segment;
pub mod structures;
pub mod tokenizer;
pub mod wand;

// Re-exports from dsl
pub use dsl::{
    Document, Field, FieldDef, FieldEntry, FieldType, FieldValue, IndexDef, QueryLanguageParser,
    Schema, SchemaBuilder, SdlParser, parse_sdl, parse_single_index,
};

// Backwards compatibility alias
pub mod schema {
    pub use crate::dsl::{
        Document, Field, FieldEntry, FieldType, FieldValue, Schema, SchemaBuilder,
    };
}

// Re-exports from structures
pub use structures::{
    AsyncSSTableReader, BlockPostingList, HorizontalBP128Iterator, HorizontalBP128PostingList,
    PostingList, PostingListIterator, SSTableValue, TERMINATED, TermInfo,
};

// Re-exports from directories
#[cfg(feature = "native")]
pub use directories::FsDirectory;
#[cfg(feature = "http")]
pub use directories::HttpDirectory;
pub use directories::{
    AsyncFileRead, CachingDirectory, Directory, DirectoryWriter, FileSlice, LazyFileHandle,
    LazyFileSlice, OwnedBytes, RamDirectory, SliceCacheStats, SliceCachingDirectory,
};

// Re-exports from segment
pub use segment::{
    AsyncSegmentReader, AsyncStoreReader, FieldStats, SegmentId, SegmentMeta, SegmentReader,
};
#[cfg(feature = "native")]
pub use segment::{SegmentBuilder, SegmentBuilderConfig, SegmentBuilderStats};

// Re-exports from query
pub use query::{
    BlockWand, Bm25Params, BooleanQuery, BoostQuery, MaxScoreWand, Query, Scorer, SearchHit,
    SearchResponse, SearchResult, TermQuery, TopKCollector, WandResult, search_segment,
};

// Re-exports from tokenizer
pub use tokenizer::{
    BoxedTokenizer, Language, LanguageAwareTokenizer, LowercaseTokenizer, MultiLanguageStemmer,
    SimpleTokenizer, StemmerTokenizer, Token, Tokenizer, TokenizerRegistry, parse_language,
};

// Re-exports from other modules
pub use directories::SLICE_CACHE_EXTENSION;
pub use error::{Error, Result};
pub use index::{Index, IndexConfig, SLICE_CACHE_FILENAME};
#[cfg(feature = "native")]
pub use index::{IndexWriter, warmup_and_save_slice_cache};

// Re-exports from wand
pub use wand::{TermWandInfo, WandData};

pub type DocId = u32;
pub type TermFreq = u32;
pub type Score = f32;
