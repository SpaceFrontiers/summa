//! Configuration types for sparse vector posting lists

use serde::{Deserialize, Serialize};

/// Sparse vector index format
///
/// Determines the on-disk layout and query execution strategy:
/// - **MaxScore**: Per-dimension variable-size blocks (DAAT — document-at-a-time).
///   Default, optimal for general sparse retrieval with block-max pruning.
/// - **Bmp**: Fixed doc_id range blocks (BAAT — block-at-a-time).
///   Based on Mallia, Suel & Tonellotto (SIGIR 2024). Divides the document
///   space into fixed-size blocks and processes them in decreasing upper-bound
///   order, enabling aggressive early termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SparseFormat {
    /// Per-dimension variable-size blocks (existing format, DAAT MaxScore)
    #[default]
    MaxScore,
    /// Fixed doc_id range blocks (BMP, BAAT block-at-a-time)
    Bmp,
}

impl SparseFormat {
    fn is_default(&self) -> bool {
        *self == Self::MaxScore
    }
}

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
///
/// Research-validated compression/effectiveness trade-offs (Pati, 2025):
/// - **UInt8**: 4x compression, ~1-2% nDCG@10 loss (RECOMMENDED for production)
/// - **Float16**: 2x compression, <1% nDCG@10 loss
/// - **Float32**: No compression, baseline effectiveness
/// - **UInt4**: 8x compression, ~3-5% nDCG@10 loss (experimental)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum WeightQuantization {
    /// Full 32-bit float precision
    #[default]
    Float32 = 0,
    /// 16-bit float (half precision) - 2x compression, <1% effectiveness loss
    Float16 = 1,
    /// 8-bit unsigned integer with scale factor - 4x compression, ~1-2% effectiveness loss (RECOMMENDED)
    UInt8 = 2,
    /// 4-bit unsigned integer with scale factor (packed, 2 per byte) - 8x compression, ~3-5% effectiveness loss
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
#[serde(rename_all = "snake_case")]
pub enum QueryWeighting {
    /// All terms get weight 1.0
    #[default]
    One,
    /// Terms weighted by IDF (inverse document frequency) from global index statistics
    /// Uses ln(N/df) where N = total docs, df = docs containing dimension
    Idf,
    /// Terms weighted by pre-computed IDF from model's idf.json file
    /// Loaded from HuggingFace model repo. No fallback to global stats.
    IdfFile,
}

/// Query-time configuration for sparse vectors
///
/// Research-validated query optimization strategies:
/// - **weight_threshold (0.01-0.05)**: Drop query dimensions with weight below threshold
///   - Filters low-IDF tokens that add latency without improving relevance
/// - **max_query_dims (10-20)**: Process only top-k dimensions by weight
///   - 30-50% latency reduction with <2% nDCG loss (Qiao et al., 2023)
/// - **heap_factor (0.8)**: Skip blocks with low max score contribution
///   - ~20% speedup with minor recall loss (SEISMIC-style)
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
    ///
    /// Research recommendation:
    /// - 1.0 = exact search (default)
    /// - 0.8 = approximate, ~20% faster with minor recall loss (RECOMMENDED for production)
    /// - 0.5 = very approximate, much faster but higher recall loss
    #[serde(default = "default_heap_factor")]
    pub heap_factor: f32,
    /// Minimum weight for query dimensions (query-time pruning)
    /// Dimensions with abs(weight) below this threshold are dropped before search.
    /// Useful for filtering low-IDF tokens that add latency without improving relevance.
    ///
    /// - 0.0 = no filtering (default)
    /// - 0.01-0.05 = recommended for SPLADE/learned sparse models
    #[serde(default)]
    pub weight_threshold: f32,
    /// Maximum number of query dimensions to process (query pruning)
    /// Processes only the top-k dimensions by weight
    ///
    /// Research recommendation (Multiple papers 2022-2024):
    /// - None = process all dimensions (default, exact)
    /// - Some(10-20) = process top 10-20 dimensions only (RECOMMENDED for SPLADE)
    ///   - 30-50% latency reduction
    ///   - <2% nDCG@10 loss
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_query_dims: Option<usize>,
    /// Fraction of query dimensions to keep (0.0-1.0), same semantics as
    /// indexing-time `pruning`: sort by abs(weight) descending,
    /// keep top fraction. None or 1.0 = no pruning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruning: Option<f32>,
    /// Minimum number of query dimensions before pruning and weight_threshold
    /// filtering are applied. Protects short queries from losing most signal.
    ///
    /// Default: 4. Set to 0 to always apply pruning/filtering.
    #[serde(default = "default_min_terms")]
    pub min_query_dims: usize,
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
            weight_threshold: 0.0,
            max_query_dims: None,
            pruning: None,
            min_query_dims: 4,
        }
    }
}

/// Configuration for sparse vector storage
///
/// Research-validated optimizations for learned sparse retrieval (SPLADE, uniCOIL, etc.):
/// - **Weight threshold (0.01-0.05)**: Removes ~30-50% of postings with minimal nDCG impact
/// - **Posting list pruning (0.1)**: Keeps top 10% per dimension, 50-70% index reduction, <1% nDCG loss
/// - **Query pruning (top 10-20 dims)**: 30-50% latency reduction, <2% nDCG loss
/// - **UInt8 quantization**: 4x compression, 1-2% nDCG loss (optimal trade-off)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVectorConfig {
    /// Index format: MaxScore (DAAT) or BMP (BAAT)
    #[serde(default, skip_serializing_if = "SparseFormat::is_default")]
    pub format: SparseFormat,
    /// Size of dimension/term indices
    pub index_size: IndexSize,
    /// Quantization for weights (see WeightQuantization docs for trade-offs)
    pub weight_quantization: WeightQuantization,
    /// Minimum weight threshold - weights below this value are not indexed
    ///
    /// Research recommendation (Guo et al., 2022; SPLADE v2):
    /// - 0.01-0.05 for SPLADE models removes ~30-50% of postings
    /// - Minimal impact on nDCG@10 (<1% loss)
    /// - Major reduction in index size and query latency
    #[serde(default)]
    pub weight_threshold: f32,
    /// Block size for posting lists (must be power of 2, default 128 for SIMD)
    /// Larger blocks = better compression, smaller blocks = faster seeks.
    /// Used by MaxScore format only.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// BMP block size: number of consecutive doc_ids per block (must be power of 2).
    /// Default 64. Only used when format = Bmp.
    /// Smaller = better pruning granularity, larger = less overhead.
    #[serde(default = "default_bmp_block_size")]
    pub bmp_block_size: u32,
    /// Maximum BMP grid memory in bytes. If the grid (num_dims × num_blocks)
    /// would exceed this, bmp_block_size is automatically increased to cap memory.
    /// Default: 256MB. Set to 0 to disable the cap.
    #[serde(default = "default_max_bmp_grid_bytes")]
    pub max_bmp_grid_bytes: u64,
    /// BMP superblock size: number of consecutive blocks grouped for hierarchical
    /// pruning (Carlson et al., SIGIR 2025). Must be power of 2.
    /// Default 64. Set to 0 to disable superblock pruning (flat BMP scoring).
    /// Only used when format = Bmp.
    #[serde(default = "default_bmp_superblock_size")]
    pub bmp_superblock_size: u32,
    /// Static pruning: fraction of postings to keep per inverted list (SEISMIC-style)
    /// Lists are sorted by weight descending and truncated to top fraction.
    ///
    /// Research recommendation (SPLADE v2, Formal et al., 2021):
    /// - None = keep all postings (default, exact)
    /// - Some(0.1) = keep top 10% of postings per dimension
    ///   - 50-70% index size reduction
    ///   - <1% nDCG@10 loss
    ///   - Exploits "concentration of importance" in learned representations
    ///
    /// Applied only during initial segment build, not during merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruning: Option<f32>,
    /// Query-time configuration (tokenizer, weighting)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_config: Option<SparseQueryConfig>,
    /// Fixed vocabulary size (number of dimensions) for BMP format.
    ///
    /// When set, all BMP segments use the same grid dimensions (rows = dims),
    /// enabling zero-copy block-copy merge. The grid is indexed by dim_id directly
    /// (no dim_ids Section C needed).
    ///
    /// Required for BMP V11 format. Typical values:
    /// - SPLADE/BERT: 30522 or 105879 (WordPiece / Unigram vocabulary)
    /// - uniCOIL: 30522
    /// - Custom models: set to vocabulary size
    ///
    /// If None, BMP builder derives dims from observed data (V10 behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<u32>,
    /// Fixed max weight scale for BMP format.
    ///
    /// When set, all BMP segments use the same quantization scale
    /// (`max_weight_scale = max_weight`), eliminating rescaling during merge.
    ///
    /// For SPLADE models: 5.0 (covers typical weight range 0-5).
    /// If None, BMP builder derives scale from data (V10 behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_weight: Option<f32>,
    /// Minimum number of postings in a dimension before pruning and
    /// weight_threshold filtering are applied. Protects dimensions with
    /// very few postings from losing most of their signal.
    ///
    /// Default: 4. Set to 0 to always apply pruning/filtering.
    #[serde(default = "default_min_terms")]
    pub min_terms: usize,
}

fn default_block_size() -> usize {
    128
}

fn default_bmp_block_size() -> u32 {
    64
}

fn default_max_bmp_grid_bytes() -> u64 {
    0 // disabled by default — masks eliminate DRAM stalls during scoring
}

fn default_bmp_superblock_size() -> u32 {
    64
}

fn default_min_terms() -> usize {
    4
}

impl Default for SparseVectorConfig {
    fn default() -> Self {
        Self {
            format: SparseFormat::MaxScore,
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float32,
            weight_threshold: 0.0,
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: None,
            query_config: None,

            dims: None,
            max_weight: None,
            min_terms: 4,
        }
    }
}

impl SparseVectorConfig {
    /// SPLADE-optimized config with research-validated defaults
    ///
    /// Optimized for SPLADE, uniCOIL, and similar learned sparse retrieval models.
    /// Based on research findings from:
    /// - Pati (2025): UInt8 quantization = 4x compression, 1-2% nDCG loss
    /// - Formal et al. (2021): SPLADE v2 posting list pruning
    /// - Qiao et al. (2023): Query dimension pruning and approximate search
    /// - Guo et al. (2022): Weight thresholding for efficiency
    ///
    /// Expected performance vs. full precision baseline:
    /// - Index size: ~15-25% of original (combined effect of all optimizations)
    /// - Query latency: 40-60% faster
    /// - Effectiveness: 2-4% nDCG@10 loss (typically acceptable for production)
    ///
    /// Vocabulary: ~30K dimensions (fits in u16)
    pub fn splade() -> Self {
        Self {
            format: SparseFormat::MaxScore,
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt8,
            weight_threshold: 0.01, // Remove ~30-50% of low-weight postings
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: Some(0.1), // Keep top 10% per dimension
            query_config: Some(SparseQueryConfig {
                tokenizer: None,
                weighting: QueryWeighting::One,
                heap_factor: 0.8,         // 20% faster approximate search
                weight_threshold: 0.01,   // Drop low-IDF query tokens
                max_query_dims: Some(20), // Process top 20 query dimensions
                pruning: Some(0.1),       // Keep top 10% of query dims
                min_query_dims: 4,
            }),

            dims: None,
            max_weight: None,
            min_terms: 4,
        }
    }

    /// SPLADE-optimized config with BMP (Block-Max Pruning) format
    ///
    /// Same optimization settings as `splade()` but uses the BMP block-at-a-time
    /// format (Mallia, Suel & Tonellotto, SIGIR 2024) instead of MaxScore.
    /// BMP divides the document space into fixed-size blocks and processes them
    /// in decreasing upper-bound order, enabling aggressive early termination.
    pub fn splade_bmp() -> Self {
        Self {
            format: SparseFormat::Bmp,
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt8,
            weight_threshold: 0.01,
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: Some(0.1),
            query_config: Some(SparseQueryConfig {
                tokenizer: None,
                weighting: QueryWeighting::One,
                heap_factor: 0.8,
                weight_threshold: 0.01,
                max_query_dims: Some(20),
                pruning: Some(0.1),
                min_query_dims: 4,
            }),

            dims: Some(105879),
            max_weight: Some(5.0),
            min_terms: 4,
        }
    }

    /// Compact config: Maximum compression (experimental)
    ///
    /// Uses aggressive UInt4 quantization for smallest possible index size.
    /// Expected trade-offs:
    /// - Index size: ~10-15% of Float32 baseline
    /// - Effectiveness: ~3-5% nDCG@10 loss
    ///
    /// Recommended for: Memory-constrained environments, cache-heavy workloads
    pub fn compact() -> Self {
        Self {
            format: SparseFormat::MaxScore,
            index_size: IndexSize::U16,
            weight_quantization: WeightQuantization::UInt4,
            weight_threshold: 0.02, // Slightly higher threshold for UInt4
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: Some(0.15), // Keep top 15% per dimension
            query_config: Some(SparseQueryConfig {
                tokenizer: None,
                weighting: QueryWeighting::One,
                heap_factor: 0.7,         // More aggressive approximate search
                weight_threshold: 0.02,   // Drop low-IDF query tokens
                max_query_dims: Some(15), // Fewer query dimensions
                pruning: Some(0.15),      // Keep top 15% of query dims
                min_query_dims: 4,
            }),

            dims: None,
            max_weight: None,
            min_terms: 4,
        }
    }

    /// Full precision config: No compression, baseline effectiveness
    ///
    /// Use for: Research baselines, when effectiveness is critical
    pub fn full_precision() -> Self {
        Self {
            format: SparseFormat::MaxScore,
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float32,
            weight_threshold: 0.0,
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: None,
            query_config: None,

            dims: None,
            max_weight: None,
            min_terms: 4,
        }
    }

    /// Conservative config: Mild optimizations, minimal effectiveness loss
    ///
    /// Balances compression and effectiveness with conservative defaults.
    /// Expected trade-offs:
    /// - Index size: ~40-50% of Float32 baseline
    /// - Query latency: ~20-30% faster
    /// - Effectiveness: <1% nDCG@10 loss
    ///
    /// Recommended for: Production deployments prioritizing effectiveness
    pub fn conservative() -> Self {
        Self {
            format: SparseFormat::MaxScore,
            index_size: IndexSize::U32,
            weight_quantization: WeightQuantization::Float16,
            weight_threshold: 0.005, // Minimal pruning
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: None, // No posting list pruning
            query_config: Some(SparseQueryConfig {
                tokenizer: None,
                weighting: QueryWeighting::One,
                heap_factor: 0.9,         // Nearly exact search
                weight_threshold: 0.005,  // Minimal query pruning
                max_query_dims: Some(50), // Process more dimensions
                pruning: None,            // No fraction-based pruning
                min_query_dims: 4,
            }),

            dims: None,
            max_weight: None,
            min_terms: 4,
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
        self.pruning = Some(fraction.clamp(0.0, 1.0));
        self
    }

    /// Bytes per entry (index + weight)
    pub fn bytes_per_entry(&self) -> f32 {
        self.index_size.bytes() as f32 + self.weight_quantization.bytes_per_weight()
    }

    /// Serialize config to a single byte.
    ///
    /// Layout: bits 7-4 = IndexSize, bit 3 = format (0=MaxScore, 1=BMP), bits 2-0 = WeightQuantization
    pub fn to_byte(&self) -> u8 {
        let format_bit = if self.format == SparseFormat::Bmp {
            0x08
        } else {
            0
        };
        ((self.index_size as u8) << 4) | format_bit | (self.weight_quantization as u8)
    }

    /// Deserialize config from a single byte.
    ///
    /// Note: weight_threshold, block_size, bmp_block_size, and query_config are not
    /// serialized in the byte — they come from the schema.
    pub fn from_byte(b: u8) -> Option<Self> {
        let index_size = IndexSize::from_u8((b >> 4) & 0x03)?;
        let format = if b & 0x08 != 0 {
            SparseFormat::Bmp
        } else {
            SparseFormat::MaxScore
        };
        let weight_quantization = WeightQuantization::from_u8(b & 0x07)?;
        Some(Self {
            format,
            index_size,
            weight_quantization,
            weight_threshold: 0.0,
            block_size: 128,
            bmp_block_size: 64,
            max_bmp_grid_bytes: 0,
            bmp_superblock_size: 64,
            pruning: None,
            query_config: None,

            dims: None,
            max_weight: None,
            min_terms: 4,
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
