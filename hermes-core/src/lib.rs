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
pub mod merge;
pub mod query;
pub mod segment;
pub mod structures;
pub mod tokenizer;

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
#[cfg(feature = "native")]
pub use directories::MmapDirectory;
pub use directories::{
    AsyncFileRead, CachingDirectory, Directory, DirectoryWriter, FileSlice, LazyFileHandle,
    LazyFileSlice, OwnedBytes, RamDirectory, SliceCacheStats, SliceCachingDirectory,
};

/// Default directory type for native builds - uses memory-mapped files for efficient access
#[cfg(feature = "native")]
pub type DefaultDirectory = MmapDirectory;

// Re-exports from segment
pub use segment::{
    AsyncSegmentReader, AsyncStoreReader, FieldStats, SegmentId, SegmentMeta, SegmentReader,
};
#[cfg(feature = "native")]
pub use segment::{SegmentBuilder, SegmentBuilderConfig, SegmentBuilderStats};

// Re-exports from query
pub use query::{
    BlockMaxScoreExecutor, Bm25Params, BooleanQuery, BoostQuery, Query, ScoredDoc, Scorer,
    SearchHit, SearchResponse, SearchResult, TermQuery, TopKCollector, WandExecutor, WandOrQuery,
    search_segment,
};

// Re-exports from tokenizer
pub use tokenizer::{
    BoxedTokenizer, Language, LanguageAwareTokenizer, LowercaseTokenizer, MultiLanguageStemmer,
    SimpleTokenizer, StemmerTokenizer, Token, Tokenizer, TokenizerRegistry, parse_language,
};

// Re-exports from other modules
pub use directories::SLICE_CACHE_EXTENSION;
pub use error::{Error, Result};
pub use index::Searcher;
#[cfg(feature = "native")]
pub use index::{Index, IndexReader, IndexWriter};
pub use index::{IndexConfig, IndexMetadata, SLICE_CACHE_FILENAME};
#[cfg(feature = "native")]
pub use index::{
    IndexingStats, SchemaConfig, SchemaFieldConfig, create_index_at_path, create_index_from_sdl,
    index_documents_from_reader, index_json_document, parse_schema,
};

// Re-exports from merge
#[cfg(feature = "native")]
pub use merge::SegmentManager;
pub use merge::{MergeCandidate, MergePolicy, NoMergePolicy, SegmentInfo, TieredMergePolicy};

pub type DocId = u32;
pub type TermFreq = u32;
pub type Score = f32;
