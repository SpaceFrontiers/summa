mod custom_serializer;
mod default_tokenizers;
mod frozen_log_merge_policy;
mod fruit_extractors;
mod index_holder;
mod index_registry;
mod index_writer_holder;
mod query_parser;
mod segment_attributes;
mod summa_document;
mod summa_tokenizer;

pub use default_tokenizers::default_tokenizers;
pub use fruit_extractors::{build_fruit_extractor, FruitExtractor};
pub use index_holder::IndexHolder;
pub use index_registry::{IndexQuery, IndexRegistry};
pub use index_writer_holder::{ComponentFile, IndexWriterHolder, SegmentComponent};
use once_cell::sync::Lazy;
pub use query_parser::QueryParser;
pub use segment_attributes::SummaSegmentAttributes;
pub use summa_document::{DocumentParsingError, SummaDocument};
pub use summa_tokenizer::SummaTokenizer;

use crate::metrics::CacheMetrics;

pub static CACHE_METRICS: Lazy<CacheMetrics> = Lazy::new(CacheMetrics::default);
