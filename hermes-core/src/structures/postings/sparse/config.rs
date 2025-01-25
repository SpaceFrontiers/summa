//! Configuration types for sparse vector posting lists

use serde::{Deserialize, Serialize};

/// Size of the index (term/dimension ID) in sparse vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum IndexSize {
    /// 16-bit index (0-65535), ideal for SPLADE vocabularies
    U16 = 0,
    /// 32-bit index (0-4B), for large vocabularies
    #[default]
    U32 = 1,
}

impl IndexSize {
    /// Bytes per index
    pub fn bytes(&self) -> usize {
        match self {
            IndexSize::U16 => 2,
            IndexSize::U32 => 4,
        }
    }

    /// Maximum value representable
    pub fn max_value(&self) -> u32 {
        match self {
            IndexSize::U16 => u16::MAX as u32,
            IndexSize::U32 => u32::MAX,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(IndexSize::U16),
            1 => Some(IndexSize::U32),
            _ => None,
        }
    }
}

/// Quantization format for sparse vector weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum WeightQuantization {
    /// Full 32-bit float precision
    #[default]
    Float32 = 0,
    /// 16-bit float (half precision)
    Float16 = 1,
    /// 8-bit unsigned integer with scale factor
    UInt8 = 2,
    /// 4-bit unsigned integer with scale factor (packed, 2 per byte)
    UInt4 = 3,
}

impl WeightQuantization {
    /// Bytes per weight (approximate for UInt4)
    pub fn bytes_per_weight(&self) -> f32 {
        match self {
            WeightQuantization::Float32 => 4.0,
            WeightQuantization::Float16 => 2.0,
            WeightQuantization::UInt8 => 1.0,
            WeightQuantization::UInt4 => 0.5,
        }
    }

    pub(crate) fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(WeightQuantization::Float32),
            1 => Some(WeightQuantization::Float16),
            2 => Some(WeightQuantization::UInt8),
            3 => Some(WeightQuantization::UInt4),
            _ => None,
        }
    }
}

/// Query-time weighting strategy for sparse vector queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QueryWeighting {
    /// All terms get weight 1.0
    #[default]
    One,
    /// Terms weighted by IDF (inverse document frequency) from the index
    Idf,
}

/// Query-time configuration for sparse vectors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseQueryConfig {
    /// HuggingFace tokenizer path/name for query-time tokenization
    /// Example: "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    /// Weighting strategy for tokenized query terms
    #[serde(default)]
    pub weighting: QueryWeighting,
    /// Heap factor for approximate search (SEISMIC-style optimization)
    /// A block is skipped if its max possible score < heap_factor * threshold
    /// - 1.0 = exact search (default)
    /// - 0.8 = approximate, ~20% faster with minor recall loss
    /// - 0.5 = very approximate, much faster
    #[serde(default = "default_heap_factor")]
    pub heap_factor: f32,
    /// Maximum number of query dimensions to process (query pruning)
    /// Processes only the top-k dimensions by weight
    /// - None = process all dimensions (default)
    /// - Some(10) = process top 10 dimensions only
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_query_dims: Option<usize>,
}

fn default_heap_factor() -> f32 {
    1.0
}

impl Default for SparseQueryConfig {
    fn default() -> Self {
        Self {
            tokenizer: None,
            weighting: QueryWeighting::One,
            heap_factor: 1.0,
            max_query_dims: None,
        }
    }
}

/// Configuration for sparse vector storage
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVectorConfig {
    /// Size of dimension/term indices
    pub index_size: IndexSize,
    /// Quantization for weights
    pub weight_quantization: WeightQuantization,
    /// Minimum weight threshold - weights below this value are not indexed
    /// This reduces index size and can improve query speed at the cost of recall
    #[serde(default)]
    pub weight_threshold: f32,
    /// Block size for posting lists (must be power of 2, default 128 for SIMD)
    /// Larger blocks = better compression, smaller blocks = faster seeks
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Static pruning: fraction of postings to keep per inverted list (SEISMIC-style)
    /// Lists are sorted by weight descending and truncated to top fraction.
    /// - None = keep all postings (default, exact)
    /// - Some(0.1) = keep top 10% of postings per dimension
    ///
    /// Applied only during initial segment build, not during merge.
    /// This exploits "concentration of importance" - top entries preserve most of inner product.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posting_list_pruning: Option<f32>,
    /// Query-time configuration (tokenizer, weighting)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_config: Option<SparseQueryConfig>,
}

fn default_block_size() -> usize {
    128
}

impl Default for SparseVectorConfig {
    fn default() -> Self {
        Self {
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float32,
            weight_threshold: 0.0,
            block_size: 128,
            posting_list_pruning: None,
            query_config: None,
        }
    }
}

impl SparseVectorConfig {
    /// SPLADE-optimized config: u16 indices, int8 weights
    pub fn splade() -> Self {
        Self {
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt8,
            weight_threshold: 0.0,
            block_size: 128,
            posting_list_pruning: None,
            query_config: None,
        }
    }

    /// Compact config: u16 indices, 4-bit weights
    pub fn compact() -> Self {
        Self {
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt4,
            weight_threshold: 0.0,
            block_size: 128,
            posting_list_pruning: None,
            query_config: None,
        }
    }

    /// Full precision config
    pub fn full_precision() -> Self {
        Self {
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float32,
            weight_threshold: 0.0,
            block_size: 128,
            posting_list_pruning: None,
            query_config: None,
        }
    }

    /// Set weight threshold (builder pattern)
    pub fn with_weight_threshold(mut self, threshold: f32) -> Self {
        self.weight_threshold = threshold;
        self
    }

    /// Set posting list pruning fraction (builder pattern)
    /// e.g., 0.1 = keep top 10% of postings per dimension
    pub fn with_pruning(mut self, fraction: f32) -> Self {
        self.posting_list_pruning = Some(fraction.clamp(0.0, 1.0));
        self
    }

    /// Bytes per entry (index + weight)
    pub fn bytes_per_entry(&self) -> f32 {
        self.index_size.bytes() as f32 + self.weight_quantization.bytes_per_weight()
    }

    /// Serialize config to a single byte
    pub fn to_byte(&self) -> u8 {
        ((self.index_size as u8) << 4) | (self.weight_quantization as u8)
    }

    /// Deserialize config from a single byte
    /// Note: weight_threshold, block_size and query_config are not serialized in the byte
    pub fn from_byte(b: u8) -> Option<Self> {
        let index_size = IndexSize::from_u8(b >> 4)?;
        let weight_quantization = WeightQuantization::from_u8(b & 0x0F)?;
        Some(Self {
            index_size,
            weight_quantization,
            weight_threshold: 0.0,
            block_size: 128,
            posting_list_pruning: None,
            query_config: None,
        })
    }

    /// Set block size (builder pattern)
    /// Must be power of 2, recommended: 64, 128, 256
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size.next_power_of_two();
        self
    }

    /// Set query configuration (builder pattern)
    pub fn with_query_config(mut self, config: SparseQueryConfig) -> Self {
        self.query_config = Some(config);
        self
    }
}

/// A sparse vector entry: (dimension_id, weight)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseEntry {
    pub dim_id: u32,
    pub weight: f32,
}

/// Sparse vector representation
#[derive(Debug, Clone, Default)]
pub struct SparseVector {
    pub(super) entries: Vec<SparseEntry>,
}

impl SparseVector {
    /// Create a new sparse vector
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Create from dimension IDs and weights
    pub fn from_entries(dim_ids: &[u32], weights: &[f32]) -> Self {
        assert_eq!(dim_ids.len(), weights.len());
        let mut entries: Vec<SparseEntry> = dim_ids
            .iter()
            .zip(weights.iter())
            .map(|(&dim_id, &weight)| SparseEntry { dim_id, weight })
            .collect();
        // Sort by dimension ID for efficient intersection
        entries.sort_by_key(|e| e.dim_id);
        Self { entries }
    }

    /// Add an entry (must maintain sorted order by dim_id)
    pub fn push(&mut self, dim_id: u32, weight: f32) {
        debug_assert!(
            self.entries.is_empty() || self.entries.last().unwrap().dim_id < dim_id,
            "Entries must be added in sorted order by dim_id"
        );
        self.entries.push(SparseEntry { dim_id, weight });
    }

    /// Number of non-zero entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over entries
    pub fn iter(&self) -> impl Iterator<Item = &SparseEntry> {
        self.entries.iter()
    }

    /// Sort by dimension ID (required for posting list encoding)
    pub fn sort_by_dim(&mut self) {
        self.entries.sort_by_key(|e| e.dim_id);
    }

    /// Sort by weight descending
    pub fn sort_by_weight_desc(&mut self) {
        self.entries.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get top-k entries by weight
    pub fn top_k(&self, k: usize) -> Vec<SparseEntry> {
        let mut sorted = self.entries.clone();
        sorted.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        sorted
    }

    /// Compute dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.entries.len() && j < other.entries.len() {
            let a = &self.entries[i];
            let b = &other.entries[j];

            match a.dim_id.cmp(&b.dim_id) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += a.weight * b.weight;
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    /// L2 norm squared
    pub fn norm_squared(&self) -> f32 {
        self.entries.iter().map(|e| e.weight * e.weight).sum()
    }

    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Prune dimensions below a weight threshold
    pub fn filter_by_weight(&self, min_weight: f32) -> Self {
        let entries: Vec<SparseEntry> = self
            .entries
            .iter()
            .filter(|e| e.weight.abs() >= min_weight)
            .cloned()
            .collect();
        Self { entries }
    }
}

impl From<Vec<(u32, f32)>> for SparseVector {
    fn from(pairs: Vec<(u32, f32)>) -> Self {
        Self {
            entries: pairs
                .into_iter()
                .map(|(dim_id, weight)| SparseEntry { dim_id, weight })
                .collect(),
        }
    }
}

impl From<SparseVector> for Vec<(u32, f32)> {
    fn from(vec: SparseVector) -> Self {
        vec.entries
            .into_iter()
            .map(|e| (e.dim_id, e.weight))
            .collect()
    }
}
