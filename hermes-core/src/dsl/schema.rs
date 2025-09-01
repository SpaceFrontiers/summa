//! Schema definitions for documents and fields

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Field identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field(pub u32);

/// Types of fields supported
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Text field - tokenized and indexed
    #[serde(rename = "text")]
    Text,
    /// Unsigned 64-bit integer
    #[serde(rename = "u64")]
    U64,
    /// Signed 64-bit integer
    #[serde(rename = "i64")]
    I64,
    /// 64-bit floating point
    #[serde(rename = "f64")]
    F64,
    /// Raw bytes (not tokenized)
    #[serde(rename = "bytes")]
    Bytes,
    /// Sparse vector field - indexed as inverted posting lists with quantized weights
    #[serde(rename = "sparse_vector")]
    SparseVector,
    /// Dense vector field indexed with the global IVF-PQ ANN implementation.
    #[serde(rename = "dense_vector")]
    DenseVector,
    /// JSON field - arbitrary JSON data, stored but not indexed
    #[serde(rename = "json")]
    Json,
    /// Binary dense vector field - packed-bit storage with Hamming distance scoring
    #[serde(rename = "binary_dense_vector")]
    BinaryDenseVector,
}

/// Field options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEntry {
    pub name: String,
    pub field_type: FieldType,
    pub indexed: bool,
    pub stored: bool,
    /// Name of the tokenizer to use for this field (for text fields)
    pub tokenizer: Option<String>,
    /// Whether this field can have multiple values (serialized as array in JSON)
    #[serde(default)]
    pub multi: bool,
    /// Position tracking mode for phrase queries and multi-field element tracking
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<PositionMode>,
    /// Configuration for sparse vector fields (index size, weight quantization)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_vector_config: Option<crate::structures::SparseVectorConfig>,
    /// Configuration for dense vector fields (dimension, quantization)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_vector_config: Option<DenseVectorConfig>,
    /// Configuration for binary dense vector fields (dimension in bits)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_dense_vector_config: Option<BinaryDenseVectorConfig>,
    /// Whether this field has columnar fast-field storage for O(1) doc→value access.
    /// Valid for u64, i64, f64, and text fields.
    #[serde(default)]
    pub fast: bool,
    /// Whether this field is a primary key (unique constraint, at most one per schema)
    #[serde(default)]
    pub primary_key: bool,
    /// Whether build-time document reordering (Recursive Graph Bisection) is enabled.
    /// Valid for sparse_vector fields with BMP format. Clusters similar documents
    /// into the same blocks for better pruning effectiveness.
    #[serde(default)]
    pub reorder: bool,
    /// Chunked text field: every value is its own BM25 scoring unit with a
    /// per-chunk ordinal in results (`docs/chunked-text-fields.md`). Text only.
    #[serde(default)]
    pub chunked: bool,
    /// BM25 k1 of a text field; `None` = `BM25_K1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_k1: Option<f32>,
    /// BM25 b of a text field; `None` = `BM25_B`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bm25_b: Option<f32>,
}

impl FieldEntry {
    /// Parsed tokenizer spec of a text field (`None` for non-text fields,
    /// fields without a tokenizer, or unparsable names).
    pub fn tokenizer_spec(&self) -> Option<crate::tokenizer::TokenizerSpec> {
        if self.field_type != FieldType::Text {
            return None;
        }
        crate::tokenizer::TokenizerSpec::parse(self.tokenizer.as_deref()?).ok()
    }
}

/// Position tracking mode for text fields
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionMode {
    /// Track only element ordinal for multi-valued fields (which array element)
    /// Useful for returning which element matched without full phrase query support
    Ordinal,
    /// Track only token position within text (for phrase queries)
    /// Does not track element ordinal - all positions are relative to concatenated text
    TokenPosition,
    /// Track both element ordinal and token position (full support)
    /// Position format: (element_ordinal << 20) | token_position
    Full,
}

impl PositionMode {
    /// Whether this mode tracks element ordinals
    pub fn tracks_ordinal(&self) -> bool {
        matches!(self, PositionMode::Ordinal | PositionMode::Full)
    }

    /// Whether this mode tracks token positions
    pub fn tracks_token_position(&self) -> bool {
        matches!(self, PositionMode::TokenPosition | PositionMode::Full)
    }
}

/// Vector index algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexType {
    /// Flat - brute-force search over raw vectors (accumulating state)
    Flat,
    /// Removed: global IVF with residual product quantization. The variant is
    /// kept only so schemas from older indexes deserialize into an actionable
    /// error instead of an unknown-variant failure. See
    /// `docs/turboquant-quantization.md` for the IVF-TQ replacement.
    IvfPq,
    /// TurboQuant: training-free per-segment compressed flat scan
    /// (`docs/turboquant-quantization.md`). Available from the first segment
    /// build with no global artifacts.
    Tq,
    /// Trained global IVF router with TurboQuant-coded centroid residuals:
    /// sub-linear probing with a derived (never trained) leaf codec. The
    /// default trained float ANN format.
    #[default]
    IvfTq,
    /// ScaNN: a shared, index-wide hierarchical partitioner with
    /// asymmetric-hashing leaf codes. Immutable segments reference one
    /// trained generation and can therefore be merged without retraining.
    Scann,
}

/// Reject schemas that reference removed index types. Called on every schema
/// entry point (index create, metadata load), so SDL/JSON/programmatic
/// construction all fail loudly with the same actionable message.
pub(crate) fn reject_removed_vector_index_types(schema: &Schema) -> Result<(), String> {
    for (_, entry) in schema.fields() {
        if let Some(config) = entry.dense_vector_config.as_ref() {
            validate_target_vectors(
                &entry.name,
                config.target_vectors,
                !matches!(
                    config.index_type,
                    VectorIndexType::Flat | VectorIndexType::Tq
                ),
            )?;
            if config.index_type == VectorIndexType::IvfPq {
                return Err(format!(
                    "dense field '{}' uses index_type `ivf_pq`, which was removed; \
                     recreate the index with `ivf_tq` (trained router, training-free \
                     TurboQuant leaves) and reindex — see docs/turboquant-quantization.md",
                    entry.name,
                ));
            }
            if config.index_type == VectorIndexType::Scann && config.soar.is_some() {
                return Err(format!(
                    "dense field '{}' enables SOAR for ScaNN, but ScaNN SOAR secondary assignments are not implemented; set soar to null/off",
                    entry.name,
                ));
            }
            validate_persisted_scann_options(
                &entry.name,
                config.index_type == VectorIndexType::Scann,
                config.num_clusters,
                config.tree_levels,
                config.nprobe,
                config.ivf_routing,
            )?;
        }
        if let Some(config) = entry.binary_dense_vector_config.as_ref() {
            validate_target_vectors(
                &entry.name,
                config.target_vectors,
                config.index_type != BinaryIndexType::Flat,
            )?;
            if config.soar.is_some() && config.index_type != BinaryIndexType::Scann {
                return Err(format!(
                    "binary dense field '{}' enables binary SOAR spilling, but it requires the ScaNN index",
                    entry.name,
                ));
            }
            if config.index_type == BinaryIndexType::Scann && !config.dim.is_multiple_of(8) {
                return Err(format!(
                    "binary dense field '{}' uses ScaNN with dimension {}; binary ScaNN dimensions must be a multiple of 8 bits",
                    entry.name, config.dim,
                ));
            }
            validate_persisted_scann_options(
                &entry.name,
                config.index_type == BinaryIndexType::Scann,
                config.num_clusters,
                config.tree_levels,
                config.nprobe,
                config.ivf_routing,
            )?;
        }
    }
    Ok(())
}

fn validate_target_vectors(
    field_name: &str,
    target_vectors: Option<u64>,
    topology_is_automatic: bool,
) -> Result<(), String> {
    if target_vectors == Some(0) {
        return Err(format!(
            "field '{field_name}' has target_vectors 0; expected a positive steady-state vector count"
        ));
    }
    if target_vectors.is_some() && !topology_is_automatic {
        return Err(format!(
            "field '{field_name}' sets target_vectors for a flat/training-free index; the hint is only valid for IVF or ScaNN automatic topology"
        ));
    }
    Ok(())
}

fn validate_persisted_scann_options(
    field_name: &str,
    is_scann: bool,
    num_clusters: Option<usize>,
    tree_levels: Option<u8>,
    nprobe: usize,
    routing: IvfRoutingMode,
) -> Result<(), String> {
    if !is_scann {
        if tree_levels.is_some() {
            return Err(format!(
                "field '{field_name}' sets tree_levels but does not use the ScaNN index"
            ));
        }
        return Ok(());
    }
    if routing != IvfRoutingMode::Auto {
        return Err(format!(
            "field '{field_name}' sets routing {routing:?} for ScaNN, but ScaNN owns its hierarchical routing; remove the routing option"
        ));
    }

    if let Some(levels) = tree_levels
        && !(1..=3).contains(&levels)
    {
        return Err(format!(
            "field '{field_name}' has ScaNN tree_levels {levels}; expected 1..=3"
        ));
    }
    if let Some(leaves) = num_clusters {
        if !(2..=30_000_000).contains(&leaves) {
            return Err(format!(
                "field '{field_name}' has ScaNN num_clusters {leaves}; expected 2..=30000000"
            ));
        }
        if nprobe > leaves {
            return Err(format!(
                "field '{field_name}' has ScaNN nprobe {nprobe} greater than num_clusters {leaves}"
            ));
        }
    }
    if nprobe == 0 {
        return Err(format!(
            "field '{field_name}' has ScaNN nprobe 0; expected a positive probe count"
        ));
    }
    Ok(())
}

/// How an IVF coarse codebook is searched.
///
/// This is shared by floating-point and packed-binary dense fields. It only
/// controls centroid routing; vector encoding and the distance metric remain
/// properties of the concrete dense index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IvfRoutingMode {
    /// Select flat routing for small codebooks and HNSW routing for large
    /// codebooks where scanning every centroid would dominate query latency.
    #[default]
    Auto,
    /// Score every leaf centroid exactly.
    Flat,
    /// Use a two-level, beam-routed hierarchy over the leaf centroids.
    TwoLevel,
    /// Use an HNSW graph over the global leaf centroids.
    Hnsw,
}

/// Storage quantization for dense vector elements
///
/// Controls the precision of each vector coordinate in `.vectors` files.
/// Lower precision reduces storage and memory bandwidth; scoring uses
/// native-precision SIMD (no dequantization on the hot path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenseVectorQuantization {
    /// 32-bit IEEE 754 float (4 bytes/dim) — full precision, baseline
    #[default]
    F32,
    /// 16-bit IEEE 754 half-float (2 bytes/dim) — <0.1% recall loss for normalized embeddings
    F16,
    /// 8-bit unsigned scalar quantization (1 byte/dim) — maps `[-1, 1]` to `[0, 255]`
    UInt8,
    /// Binary packed-bit storage (1 bit per dimension, ceil(dim/8) bytes per vector).
    /// Used internally by BinaryDenseVector fields. Not selectable for DenseVector fields.
    Binary,
}

impl DenseVectorQuantization {
    /// Bytes per element for non-binary quantization types.
    /// Panics for Binary — use `dim.div_ceil(8)` for binary vector byte size.
    pub fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::UInt8 => 1,
            Self::Binary => panic!("element_size() not valid for Binary; use dim.div_ceil(8)"),
        }
    }

    /// Wire format tag (stored in .vectors header)
    pub fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::UInt8 => 2,
            Self::Binary => 3,
        }
    }

    /// Decode wire format tag
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::UInt8),
            3 => Some(Self::Binary),
            _ => None,
        }
    }
}

/// Configuration for dense vector fields using exact Flat accumulation or the
/// single production IVF-PQ ANN format.
///
/// Indexes operate in two states:
/// - **Flat (accumulating)**: Brute-force search over raw vectors before
///   `build_vector_index` is called.
/// - **Built (ANN)**: Fast approximate nearest neighbor search using trained structures.
///   Centroids and codebooks are trained from index-wide data and shared by
///   every segment; segment payloads contain only assignments and PQ codes.
#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseVectorConfig {
    /// Dimensionality of vectors
    pub dim: usize,
    /// Target vector index algorithm (Flat or IVF-PQ).
    /// When in accumulating state, search uses brute-force regardless of this setting.
    #[serde(default)]
    pub index_type: VectorIndexType,
    /// Storage quantization for vector elements (f32, f16, uint8)
    #[serde(default)]
    pub quantization: DenseVectorQuantization,
    /// Number of IVF leaf clusters. If omitted, the selected index algorithm's
    /// corpus-size cost model determines the value.
    /// If None, automatically determined based on dataset size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_clusters: Option<usize>,
    /// Expected steady-state vector count used only for automatic topology
    /// sizing. Training readiness still depends on the observed live corpus.
    /// Explicit `num_clusters` takes precedence over this hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_vectors: Option<u64>,
    /// Number of levels in the ScaNN routing tree. When omitted, training
    /// derives the depth from corpus size. Only meaningful for ScaNN.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_levels: Option<u8>,
    /// Coarse-codebook routing strategy. This setting is metric agnostic and
    /// is applied to every IVF-backed dense index.
    #[serde(default)]
    pub ivf_routing: IvfRoutingMode,
    /// Number of leaf clusters to probe during search (default: 64)
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    /// Whether stored vectors are pre-normalized to unit L2 norm.
    /// When true, scoring skips per-vector norm computation (cosine = dot / ||q||),
    /// reducing compute by ~40%. Common for embedding models (e.g. OpenAI, Cohere).
    /// New IVF-TQ generations index a normalized ANN-only copy while retaining
    /// the original values for exact reranking. Legacy unnormalized IVF-TQ
    /// generations must be rebuilt before they can be searched.
    /// Default: true (most embedding models produce L2-normalized vectors).
    #[serde(default = "default_unit_norm")]
    pub unit_norm: bool,
    /// SOAR spilled cluster assignments for IVF-TQ.
    /// Assigns vectors to a secondary cluster with an orthogonality-amplified
    /// residual, improving recall at the same nprobe for ~1.2-2x assignment storage.
    /// Default: selective spilling calibrated to at most 30% of vectors for
    /// IVF-TQ. Set this to `None` to disable SOAR. Ignored by non-IVF formats.
    ///
    /// Unlike optional fields whose `None` value is omitted on serialization,
    /// this field serializes `None` as `null`: omission means "use the new
    /// selective default", while an explicit `null` must continue to mean off
    /// across a schema round trip.
    #[serde(default = "default_soar")]
    pub soar: Option<crate::structures::SoarConfig>,
}

#[derive(Default)]
enum PersistedSoar {
    #[default]
    Unspecified,
    Specified(Option<crate::structures::SoarConfig>),
}

impl<'de> Deserialize<'de> for PersistedSoar {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Option::<crate::structures::SoarConfig>::deserialize(deserializer).map(Self::Specified)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DenseVectorConfigWire {
    dim: usize,
    #[serde(default)]
    index_type: VectorIndexType,
    #[serde(default)]
    quantization: DenseVectorQuantization,
    #[serde(default)]
    num_clusters: Option<usize>,
    #[serde(default)]
    target_vectors: Option<u64>,
    #[serde(default)]
    tree_levels: Option<u8>,
    #[serde(default)]
    ivf_routing: IvfRoutingMode,
    #[serde(default = "default_nprobe")]
    nprobe: usize,
    #[serde(default = "default_unit_norm")]
    unit_norm: bool,
    #[serde(default)]
    soar: PersistedSoar,
}

impl<'de> Deserialize<'de> for DenseVectorConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = DenseVectorConfigWire::deserialize(deserializer)?;
        let soar = match wire.soar {
            PersistedSoar::Specified(soar) => soar,
            PersistedSoar::Unspecified if wire.index_type == VectorIndexType::IvfTq => {
                default_soar()
            }
            PersistedSoar::Unspecified => None,
        };
        Ok(Self {
            dim: wire.dim,
            index_type: wire.index_type,
            quantization: wire.quantization,
            num_clusters: wire.num_clusters,
            target_vectors: wire.target_vectors,
            tree_levels: wire.tree_levels,
            ivf_routing: wire.ivf_routing,
            nprobe: wire.nprobe,
            unit_norm: wire.unit_norm,
            soar,
        })
    }
}

fn default_nprobe() -> usize {
    64
}

fn default_unit_norm() -> bool {
    true
}

fn default_soar() -> Option<crate::structures::SoarConfig> {
    Some(crate::structures::SoarConfig::default())
}

impl DenseVectorConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            index_type: VectorIndexType::IvfTq,
            quantization: DenseVectorQuantization::F32,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: IvfRoutingMode::Auto,
            nprobe: 64,
            unit_norm: true,
            soar: Some(crate::structures::SoarConfig::default()),
        }
    }

    /// Create Flat (brute-force) configuration - no ANN index
    pub fn flat(dim: usize) -> Self {
        Self {
            dim,
            index_type: VectorIndexType::Flat,
            quantization: DenseVectorQuantization::F32,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: IvfRoutingMode::Auto,
            nprobe: 0,
            unit_norm: true,
            soar: None,
        }
    }

    /// Create TurboQuant configuration: training-free compressed flat scan.
    pub fn tq(dim: usize) -> Self {
        Self {
            dim,
            index_type: VectorIndexType::Tq,
            quantization: DenseVectorQuantization::F32,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: IvfRoutingMode::Flat,
            nprobe: 0,
            unit_norm: true,
            soar: None,
        }
    }

    /// Create IVF-TQ configuration: trained coarse router, TurboQuant leaves.
    pub fn ivf_tq(dim: usize, num_clusters: Option<usize>, nprobe: usize) -> Self {
        Self {
            dim,
            index_type: VectorIndexType::IvfTq,
            quantization: DenseVectorQuantization::F32,
            num_clusters,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: IvfRoutingMode::Auto,
            nprobe,
            unit_norm: true,
            soar: Some(crate::structures::SoarConfig::default()),
        }
    }

    /// Set storage quantization
    pub fn with_quantization(mut self, quantization: DenseVectorQuantization) -> Self {
        self.quantization = quantization;
        self
    }

    /// Mark vectors as pre-normalized to unit L2 norm
    pub fn with_unit_norm(mut self) -> Self {
        self.unit_norm = true;
        self
    }

    /// Set number of IVF clusters
    pub fn with_num_clusters(mut self, num_clusters: usize) -> Self {
        self.num_clusters = Some(num_clusters);
        self
    }

    /// Hint the expected steady-state corpus size for automatic topology.
    pub fn with_target_vectors(mut self, target_vectors: u64) -> Self {
        self.target_vectors = Some(target_vectors);
        self
    }

    /// Set flat, two-level, or HNSW IVF centroid routing explicitly.
    pub fn with_ivf_routing(mut self, routing: IvfRoutingMode) -> Self {
        self.ivf_routing = routing;
        self
    }
    /// Enable SOAR spilled secondary cluster assignments (IVF-based indexes only)
    pub fn with_soar(mut self, soar: crate::structures::SoarConfig) -> Self {
        self.soar = Some(soar);
        self
    }

    /// Explicitly disable SOAR secondary assignments.
    pub fn without_soar(mut self) -> Self {
        self.soar = None;
        self
    }

    /// Check if this config uses IVF
    pub fn uses_ivf(&self) -> bool {
        self.index_type == VectorIndexType::IvfTq
    }

    /// Whether the partitioner supports SOAR secondary assignments.
    pub fn supports_soar(&self) -> bool {
        self.index_type == VectorIndexType::IvfTq
    }

    /// Check if this config is flat (brute-force)
    pub fn is_flat(&self) -> bool {
        self.index_type == VectorIndexType::Flat
    }

    /// Calculate optimal number of clusters for given vector count
    pub fn optimal_num_clusters(&self, num_vectors: usize) -> usize {
        self.num_clusters.unwrap_or_else(|| {
            let num_vectors = self.target_vectors.map_or(num_vectors, |target| {
                usize::try_from(target)
                    .unwrap_or(usize::MAX)
                    .max(num_vectors)
            });
            // Balanced IVF cost model: practical values are commonly in the
            // 4-16×sqrt(N) range. Eight is a conservative midpoint; training
            // quality and artifact memory impose the final bounds.
            let optimal = 8.0 * (num_vectors as f64).sqrt();
            (optimal as usize).clamp(16, 1_048_576)
        })
    }
}

/// Configuration for binary dense vector fields
///
/// Binary dense vectors store packed bits (1 bit per dimension) and use
/// Hamming distance for scoring. Segments accumulate exact packed codes and
/// use the same global IVF router after `build_vector_index`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinaryDenseVectorConfig {
    /// Number of bits (dimensions). Storage is ceil(dim/8) bytes per vector.
    pub dim: usize,
    /// ANN index type: Flat (brute-force SIMD Hamming) or Ivf (default)
    /// (k-majority Hamming clusters — probe `nprobe` clusters at query time).
    /// IVF pays off for segments past a few million vectors.
    #[serde(default)]
    pub index_type: BinaryIndexType,
    /// Number of IVF leaf clusters, selected from corpus and sample size by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_clusters: Option<usize>,
    /// Expected steady-state vector count used only for automatic topology
    /// sizing. Training readiness still depends on the observed live corpus.
    /// Explicit `num_clusters` takes precedence over this hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_vectors: Option<u64>,
    /// Number of levels in the ScaNN Hamming routing tree. When omitted,
    /// training derives the depth from corpus size. Only meaningful for ScaNN.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_levels: Option<u8>,
    /// Coarse-codebook routing strategy. Uses the same routing planner as
    /// floating-point IVF indexes.
    #[serde(default)]
    pub ivf_routing: IvfRoutingMode,
    /// Clusters to probe during search (default: 64)
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    /// Optional one-secondary selective spilling for binary ScaNN. The
    /// alternate leaf is chosen by exact centroid Hamming distance. Unlike
    /// float SOAR, packed bits have no continuous residual geometry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soar: Option<crate::structures::SoarConfig>,
}

/// ANN index type for binary dense vector fields
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryIndexType {
    /// Brute-force SIMD Hamming scan
    Flat,
    /// IVF with a global k-majority Hamming quantizer
    #[default]
    Ivf,
    /// Hierarchical Hamming partitioning with exact packed-code leaf scoring.
    Scann,
}

/// Complete target ANN configuration for an atomic vector-index ALTER.
/// Storage shape is deliberately included for validation but cannot change:
/// ALTER rewrites ANN payloads from retained flat vectors, not stored vectors.
#[derive(Debug, Clone)]
pub enum VectorIndexAlter {
    Dense(DenseVectorConfig),
    Binary(BinaryDenseVectorConfig),
}

impl BinaryDenseVectorConfig {
    pub fn new(dim: usize) -> Self {
        assert!(
            dim.is_multiple_of(8),
            "BinaryDenseVector dimension must be a multiple of 8, got {dim}"
        );
        Self {
            dim,
            index_type: BinaryIndexType::Ivf,
            num_clusters: None,
            target_vectors: None,
            tree_levels: None,
            ivf_routing: IvfRoutingMode::Auto,
            nprobe: 64,
            soar: None,
        }
    }

    /// Enable the IVF index (builder pattern)
    pub fn with_ivf(mut self, num_clusters: Option<usize>, nprobe: usize) -> Self {
        self.index_type = BinaryIndexType::Ivf;
        self.num_clusters = num_clusters;
        self.nprobe = nprobe;
        self
    }

    /// Hint the expected steady-state corpus size for automatic topology.
    pub fn with_target_vectors(mut self, target_vectors: u64) -> Self {
        self.target_vectors = Some(target_vectors);
        self
    }

    /// Set flat, two-level, or HNSW IVF centroid routing explicitly.
    pub fn with_ivf_routing(mut self, routing: IvfRoutingMode) -> Self {
        self.ivf_routing = routing;
        self
    }

    /// Enable selective secondary-leaf spilling for binary ScaNN.
    pub fn with_soar(mut self, soar: crate::structures::SoarConfig) -> Self {
        self.soar = Some(soar);
        self
    }

    /// Disable binary ScaNN secondary-leaf spilling.
    pub fn without_soar(mut self) -> Self {
        self.soar = None;
        self
    }

    /// Balanced binary IVF cluster count for a given vector count.
    pub fn optimal_num_clusters(&self, num_vectors: usize) -> usize {
        self.num_clusters.unwrap_or_else(|| {
            let num_vectors = self.target_vectors.map_or(num_vectors, |target| {
                usize::try_from(target)
                    .unwrap_or(usize::MAX)
                    .max(num_vectors)
            });
            // The 15M-row packed-Hamming sweep found the balanced sqrt(N)
            // geometry Pareto-optimal for practical recall/latency targets.
            // Larger, search-quality geometries remain available explicitly.
            let balanced = (num_vectors as f64).sqrt().ceil() as usize;
            balanced.clamp(16, 1_048_576)
        })
    }

    /// Number of bytes needed to store one vector
    pub fn byte_len(&self) -> usize {
        self.dim.div_ceil(8)
    }
}

use super::query_field_router::QueryRouterRule;

/// Schema defining document structure
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schema {
    fields: Vec<FieldEntry>,
    name_to_field: HashMap<String, Field>,
    /// Default fields for query parsing (when no field is specified)
    #[serde(default)]
    default_fields: Vec<Field>,
    /// Query router rules for routing queries to specific fields based on regex patterns
    #[serde(default)]
    query_routers: Vec<QueryRouterRule>,
    /// Run BP (graph bisection) reordering of `reorder`-attributed BMP fields
    /// inside segment merges. SDL: `reorder_on_merge: true` at index level.
    /// Absent = disabled (merges block-copy; the standalone reorder pass
    /// handles ordering).
    #[serde(default)]
    reorder_on_merge: bool,
    /// Index name used as the `index` label on metrics. Set from the SDL
    /// index name at parse time and overridden with the registry name at
    /// server-side index creation. Empty on old metadata → "unknown".
    #[serde(default)]
    index_name: String,
}

impl Schema {
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    pub fn get_field(&self, name: &str) -> Option<Field> {
        self.name_to_field.get(name).copied()
    }

    pub fn get_field_entry(&self, field: Field) -> Option<&FieldEntry> {
        self.fields.get(field.0 as usize)
    }

    /// Field whose values hint the dynamic tokenizer of `field`
    /// (`text<lex(by: <hint field>, ...)>`), if any.
    pub fn tokenizer_hint_field(&self, field: Field) -> Option<Field> {
        let spec = self.get_field_entry(field)?.tokenizer_spec()?;
        self.get_field(spec.hint_field()?)
    }

    /// Clone this schema with one vector field's ANN parameters replaced.
    /// Field type, dimension, and storage quantization are immutable.
    pub fn with_vector_index_alter(
        &self,
        field: Field,
        alter: VectorIndexAlter,
    ) -> Result<Self, String> {
        let mut next = self.clone();
        let entry = next
            .fields
            .get_mut(field.0 as usize)
            .ok_or_else(|| format!("vector ALTER references unknown field {}", field.0))?;
        match alter {
            VectorIndexAlter::Dense(config) => {
                let current = entry
                    .dense_vector_config
                    .as_ref()
                    .ok_or_else(|| format!("field '{}' is not a dense vector field", entry.name))?;
                if config.dim != current.dim || config.quantization != current.quantization {
                    return Err(format!(
                        "field '{}' ALTER cannot change dimension or storage quantization",
                        entry.name
                    ));
                }
                if matches!(
                    config.index_type,
                    VectorIndexType::Flat | VectorIndexType::Tq
                ) {
                    return Err(format!(
                        "field '{}' ALTER target must be `ivf_tq` or `scann`",
                        entry.name
                    ));
                }
                entry.dense_vector_config = Some(config);
            }
            VectorIndexAlter::Binary(config) => {
                let current = entry.binary_dense_vector_config.as_ref().ok_or_else(|| {
                    format!("field '{}' is not a binary dense vector field", entry.name)
                })?;
                if config.dim != current.dim {
                    return Err(format!(
                        "field '{}' ALTER cannot change binary dimension",
                        entry.name
                    ));
                }
                if config.index_type == BinaryIndexType::Flat {
                    return Err(format!(
                        "field '{}' ALTER target must be `ivf` or `scann`",
                        entry.name
                    ));
                }
                entry.binary_dense_vector_config = Some(config);
            }
        }
        reject_removed_vector_index_types(&next)?;
        Ok(next)
    }

    pub fn get_field_name(&self, field: Field) -> Option<&str> {
        self.fields.get(field.0 as usize).map(|e| e.name.as_str())
    }

    pub fn fields(&self) -> impl Iterator<Item = (Field, &FieldEntry)> {
        self.fields
            .iter()
            .enumerate()
            .map(|(i, e)| (Field(i as u32), e))
    }

    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// Whether any field has the `reorder` attribute set.
    /// Used by the background optimizer to determine which indexes need BP reordering.
    pub fn has_reorder_fields(&self) -> bool {
        self.fields.iter().any(|e| e.reorder)
    }

    /// Whether merges BP-reorder `reorder`-attributed BMP fields while writing
    /// the merged segment (index-level SDL option `reorder_on_merge: true`).
    pub fn reorder_on_merge(&self) -> bool {
        self.reorder_on_merge
    }

    /// Index name for metric labels ("unknown" when not set — pre-existing
    /// metadata or programmatic schemas without a name).
    pub fn index_label(&self) -> &str {
        if self.index_name.is_empty() {
            "unknown"
        } else {
            &self.index_name
        }
    }

    /// Set the index name used as the metrics `index` label.
    pub fn set_index_name(&mut self, name: impl Into<String>) {
        self.index_name = name.into();
    }

    /// Get the default fields for query parsing
    pub fn default_fields(&self) -> &[Field] {
        &self.default_fields
    }

    /// Set default fields (used by builder)
    pub fn set_default_fields(&mut self, fields: Vec<Field>) {
        self.default_fields = fields;
    }

    /// Get the query router rules
    pub fn query_routers(&self) -> &[QueryRouterRule] {
        &self.query_routers
    }

    /// Set query router rules
    pub fn set_query_routers(&mut self, rules: Vec<QueryRouterRule>) {
        self.query_routers = rules;
    }

    /// Get the primary key field, if one is defined
    pub fn primary_field(&self) -> Option<Field> {
        self.fields
            .iter()
            .enumerate()
            .find(|(_, e)| e.primary_key)
            .map(|(i, _)| Field(i as u32))
    }
}

/// Builder for Schema
#[derive(Debug, Default)]
pub struct SchemaBuilder {
    fields: Vec<FieldEntry>,
    default_fields: Vec<String>,
    query_routers: Vec<QueryRouterRule>,
    reorder_on_merge: bool,
    index_name: String,
}

impl SchemaBuilder {
    pub fn add_text_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field_with_tokenizer(
            name,
            FieldType::Text,
            indexed,
            stored,
            Some("simple".to_string()),
        )
    }

    pub fn add_text_field_with_tokenizer(
        &mut self,
        name: &str,
        indexed: bool,
        stored: bool,
        tokenizer: &str,
    ) -> Field {
        self.add_field_with_tokenizer(
            name,
            FieldType::Text,
            indexed,
            stored,
            Some(tokenizer.to_string()),
        )
    }

    pub fn add_u64_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field(name, FieldType::U64, indexed, stored)
    }

    pub fn add_i64_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field(name, FieldType::I64, indexed, stored)
    }

    pub fn add_f64_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field(name, FieldType::F64, indexed, stored)
    }

    pub fn add_bytes_field(&mut self, name: &str, stored: bool) -> Field {
        self.add_field(name, FieldType::Bytes, false, stored)
    }

    /// Add a JSON field for storing arbitrary JSON data
    ///
    /// JSON fields are never indexed, only stored. They can hold any valid JSON value
    /// (objects, arrays, strings, numbers, booleans, null).
    pub fn add_json_field(&mut self, name: &str, stored: bool) -> Field {
        self.add_field(name, FieldType::Json, false, stored)
    }

    /// Add a sparse vector field with default configuration
    ///
    /// Sparse vectors are indexed as inverted posting lists where each dimension
    /// becomes a "term" and documents have quantized weights for each dimension.
    pub fn add_sparse_vector_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_sparse_vector_field_with_config(
            name,
            indexed,
            stored,
            crate::structures::SparseVectorConfig::default(),
        )
    }

    /// Add a sparse vector field with custom configuration
    ///
    /// Use `SparseVectorConfig::splade()` for SPLADE models (u16 indices, uint8 weights).
    /// Use `SparseVectorConfig::compact()` for maximum compression (u16 indices, uint4 weights).
    pub fn add_sparse_vector_field_with_config(
        &mut self,
        name: &str,
        indexed: bool,
        stored: bool,
        config: crate::structures::SparseVectorConfig,
    ) -> Field {
        let field = Field(self.fields.len() as u32);
        self.fields.push(FieldEntry {
            name: name.to_string(),
            field_type: FieldType::SparseVector,
            indexed,
            stored,
            tokenizer: None,
            multi: false,
            positions: None,
            sparse_vector_config: Some(config),
            dense_vector_config: None,
            binary_dense_vector_config: None,
            fast: false,
            primary_key: false,
            reorder: false,
            chunked: false,
            bm25_k1: None,
            bm25_b: None,
        });
        field
    }

    /// Set sparse vector configuration for an existing field
    pub fn set_sparse_vector_config(
        &mut self,
        field: Field,
        config: crate::structures::SparseVectorConfig,
    ) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.sparse_vector_config = Some(config);
        }
    }

    /// Add a dense vector field with default configuration
    ///
    /// Dense vectors use the global IVF-PQ ANN implementation. The dimension
    /// determines both the stored vector shape and PQ structure.
    pub fn add_dense_vector_field(
        &mut self,
        name: &str,
        dim: usize,
        indexed: bool,
        stored: bool,
    ) -> Field {
        self.add_dense_vector_field_with_config(name, indexed, stored, DenseVectorConfig::new(dim))
    }

    /// Add a dense vector field with custom configuration
    pub fn add_dense_vector_field_with_config(
        &mut self,
        name: &str,
        indexed: bool,
        stored: bool,
        config: DenseVectorConfig,
    ) -> Field {
        let field = Field(self.fields.len() as u32);
        self.fields.push(FieldEntry {
            name: name.to_string(),
            field_type: FieldType::DenseVector,
            indexed,
            stored,
            tokenizer: None,
            multi: false,
            positions: None,
            sparse_vector_config: None,
            dense_vector_config: Some(config),
            binary_dense_vector_config: None,
            fast: false,
            primary_key: false,
            reorder: false,
            chunked: false,
            bm25_k1: None,
            bm25_b: None,
        });
        field
    }

    /// Add a binary dense vector field
    ///
    /// Binary dense vectors use packed-bit storage (1 bit per dimension),
    /// exact Hamming scoring inside globally routed IVF leaves, and a flat
    /// SIMD fallback while the index is accumulating.
    pub fn add_binary_dense_vector_field(
        &mut self,
        name: &str,
        dim: usize,
        indexed: bool,
        stored: bool,
    ) -> Field {
        self.add_binary_dense_vector_field_with_config(
            name,
            indexed,
            stored,
            BinaryDenseVectorConfig::new(dim),
        )
    }

    /// Add a binary dense vector field with custom configuration
    pub fn add_binary_dense_vector_field_with_config(
        &mut self,
        name: &str,
        indexed: bool,
        stored: bool,
        config: BinaryDenseVectorConfig,
    ) -> Field {
        let field = Field(self.fields.len() as u32);
        self.fields.push(FieldEntry {
            name: name.to_string(),
            field_type: FieldType::BinaryDenseVector,
            indexed,
            stored,
            tokenizer: None,
            multi: false,
            positions: None,
            sparse_vector_config: None,
            dense_vector_config: None,
            binary_dense_vector_config: Some(config),
            fast: false,
            primary_key: false,
            reorder: false,
            chunked: false,
            bm25_k1: None,
            bm25_b: None,
        });
        field
    }

    fn add_field(
        &mut self,
        name: &str,
        field_type: FieldType,
        indexed: bool,
        stored: bool,
    ) -> Field {
        self.add_field_with_tokenizer(name, field_type, indexed, stored, None)
    }

    fn add_field_with_tokenizer(
        &mut self,
        name: &str,
        field_type: FieldType,
        indexed: bool,
        stored: bool,
        tokenizer: Option<String>,
    ) -> Field {
        self.add_field_full(name, field_type, indexed, stored, tokenizer, false)
    }

    fn add_field_full(
        &mut self,
        name: &str,
        field_type: FieldType,
        indexed: bool,
        stored: bool,
        tokenizer: Option<String>,
        multi: bool,
    ) -> Field {
        let field = Field(self.fields.len() as u32);
        self.fields.push(FieldEntry {
            name: name.to_string(),
            field_type,
            indexed,
            stored,
            tokenizer,
            multi,
            positions: None,
            sparse_vector_config: None,
            dense_vector_config: None,
            binary_dense_vector_config: None,
            fast: false,
            primary_key: false,
            reorder: false,
            chunked: false,
            bm25_k1: None,
            bm25_b: None,
        });
        field
    }

    /// Set the multi attribute on the last added field
    pub fn set_multi(&mut self, field: Field, multi: bool) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.multi = multi;
        }
    }

    /// Set fast-field columnar storage for O(1) doc→value access.
    /// Valid for u64, i64, f64, and text fields.
    pub fn set_fast(&mut self, field: Field, fast: bool) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.fast = fast;
        }
    }

    /// Mark a field as the primary key (unique constraint).
    ///
    /// Primary key implies fast + indexed (dedup looks committed keys up in
    /// the fast-field text dictionary) — kept in sync with the SDL path,
    /// which forces the same attributes.
    pub fn set_primary_key(&mut self, field: Field) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.primary_key = true;
            entry.fast = true;
            entry.indexed = true;
        }
    }

    /// Enable build-time document reordering (Recursive Graph Bisection) for BMP fields
    pub fn set_reorder(&mut self, field: Field, reorder: bool) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.reorder = reorder;
        }
    }

    /// Set the BM25 parameters of a text field (`None` keeps the default).
    pub fn set_bm25_params(&mut self, field: Field, k1: Option<f32>, b: Option<f32>) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.bm25_k1 = k1;
            entry.bm25_b = b;
        }
    }

    /// Mark a text field as chunked: each value is indexed as its own BM25
    /// unit and results carry per-chunk ordinals. Implies `multi`.
    pub fn set_chunked(&mut self, field: Field, chunked: bool) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.chunked = chunked;
            if chunked {
                entry.multi = true;
            }
        }
    }

    /// Enable BP reordering of `reorder`-attributed BMP fields inside merges
    /// (index-level; SDL `reorder_on_merge: true`). Default: disabled.
    pub fn set_reorder_on_merge(&mut self, on: bool) {
        self.reorder_on_merge = on;
    }

    /// Set the index name used as the metrics `index` label.
    pub fn set_index_name(&mut self, name: impl Into<String>) {
        self.index_name = name.into();
    }

    /// Set position tracking mode for phrase queries and multi-field element tracking
    pub fn set_positions(&mut self, field: Field, mode: PositionMode) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.positions = Some(mode);
        }
    }

    /// Set default fields by name
    pub fn set_default_fields(&mut self, field_names: Vec<String>) {
        self.default_fields = field_names;
    }

    /// Set query router rules
    pub fn set_query_routers(&mut self, rules: Vec<QueryRouterRule>) {
        self.query_routers = rules;
    }

    pub fn build(self) -> Schema {
        let mut name_to_field = HashMap::new();
        for (i, entry) in self.fields.iter().enumerate() {
            name_to_field.insert(entry.name.clone(), Field(i as u32));
        }

        // Resolve default field names to Field IDs
        let default_fields: Vec<Field> = self
            .default_fields
            .iter()
            .filter_map(|name| name_to_field.get(name).copied())
            .collect();

        Schema {
            fields: self.fields,
            name_to_field,
            default_fields,
            query_routers: self.query_routers,
            reorder_on_merge: self.reorder_on_merge,
            index_name: self.index_name,
        }
    }
}

/// Value that can be stored in a field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "u64")]
    U64(u64),
    #[serde(rename = "i64")]
    I64(i64),
    #[serde(rename = "f64")]
    F64(f64),
    #[serde(rename = "bytes")]
    Bytes(Vec<u8>),
    /// Sparse vector: list of (dimension_id, weight) pairs
    #[serde(rename = "sparse_vector")]
    SparseVector(Vec<(u32, f32)>),
    /// Dense vector: float32 values
    #[serde(rename = "dense_vector")]
    DenseVector(Vec<f32>),
    /// Arbitrary JSON value
    #[serde(rename = "json")]
    Json(serde_json::Value),
    /// Binary dense vector: packed bits (ceil(dim/8) bytes)
    #[serde(rename = "binary_dense_vector")]
    BinaryDenseVector(Vec<u8>),
}

impl FieldValue {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            FieldValue::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            FieldValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            FieldValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            FieldValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_sparse_vector(&self) -> Option<&[(u32, f32)]> {
        match self {
            FieldValue::SparseVector(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn as_dense_vector(&self) -> Option<&[f32]> {
        match self {
            FieldValue::DenseVector(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            FieldValue::Json(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_binary_dense_vector(&self) -> Option<&[u8]> {
        match self {
            FieldValue::BinaryDenseVector(v) => Some(v),
            _ => None,
        }
    }
}

/// A document to be indexed
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Document {
    field_values: Vec<(Field, FieldValue)>,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_text(&mut self, field: Field, value: impl Into<String>) {
        self.field_values
            .push((field, FieldValue::Text(value.into())));
    }

    pub fn add_u64(&mut self, field: Field, value: u64) {
        self.field_values.push((field, FieldValue::U64(value)));
    }

    pub fn add_i64(&mut self, field: Field, value: i64) {
        self.field_values.push((field, FieldValue::I64(value)));
    }

    pub fn add_f64(&mut self, field: Field, value: f64) {
        self.field_values.push((field, FieldValue::F64(value)));
    }

    pub fn add_bytes(&mut self, field: Field, value: Vec<u8>) {
        self.field_values.push((field, FieldValue::Bytes(value)));
    }

    pub fn add_sparse_vector(&mut self, field: Field, entries: Vec<(u32, f32)>) {
        self.field_values
            .push((field, FieldValue::SparseVector(entries)));
    }

    pub fn add_dense_vector(&mut self, field: Field, values: Vec<f32>) {
        self.field_values
            .push((field, FieldValue::DenseVector(values)));
    }

    pub fn add_json(&mut self, field: Field, value: serde_json::Value) {
        self.field_values.push((field, FieldValue::Json(value)));
    }

    pub fn add_binary_dense_vector(&mut self, field: Field, values: Vec<u8>) {
        self.field_values
            .push((field, FieldValue::BinaryDenseVector(values)));
    }

    pub fn get_first(&self, field: Field) -> Option<&FieldValue> {
        self.field_values
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, v)| v)
    }

    pub fn get_all(&self, field: Field) -> impl Iterator<Item = &FieldValue> {
        self.field_values
            .iter()
            .filter(move |(f, _)| *f == field)
            .map(|(_, v)| v)
    }

    pub fn field_values(&self) -> &[(Field, FieldValue)] {
        &self.field_values
    }

    /// Return a new Document containing only fields marked as `stored` in the schema
    pub fn filter_stored(&self, schema: &Schema) -> Document {
        Document {
            field_values: self
                .field_values
                .iter()
                .filter(|(field, _)| {
                    schema
                        .get_field_entry(*field)
                        .is_some_and(|entry| entry.stored)
                })
                .cloned()
                .collect(),
        }
    }

    /// Convert document to a JSON object using field names from schema
    ///
    /// Fields marked as `multi` in the schema are always returned as JSON arrays.
    /// Other fields with multiple values are also returned as arrays.
    /// Fields with a single value (and not marked multi) are returned as scalar values.
    pub fn to_json(&self, schema: &Schema) -> serde_json::Value {
        use std::collections::HashMap;

        // Group values by field, keeping track of field entry for multi check
        let mut field_values_map: HashMap<Field, (String, bool, Vec<serde_json::Value>)> =
            HashMap::new();

        for (field, value) in &self.field_values {
            if let Some(entry) = schema.get_field_entry(*field) {
                let json_value = match value {
                    FieldValue::Text(s) => serde_json::Value::String(s.clone()),
                    FieldValue::U64(n) => serde_json::Value::Number((*n).into()),
                    FieldValue::I64(n) => serde_json::Value::Number((*n).into()),
                    FieldValue::F64(n) => serde_json::json!(n),
                    FieldValue::Bytes(b) => {
                        use base64::Engine;
                        serde_json::Value::String(
                            base64::engine::general_purpose::STANDARD.encode(b),
                        )
                    }
                    FieldValue::SparseVector(entries) => {
                        let indices: Vec<u32> = entries.iter().map(|(i, _)| *i).collect();
                        let values: Vec<f32> = entries.iter().map(|(_, v)| *v).collect();
                        serde_json::json!({
                            "indices": indices,
                            "values": values
                        })
                    }
                    FieldValue::DenseVector(values) => {
                        serde_json::json!(values)
                    }
                    FieldValue::Json(v) => v.clone(),
                    FieldValue::BinaryDenseVector(b) => {
                        use base64::Engine;
                        serde_json::Value::String(
                            base64::engine::general_purpose::STANDARD.encode(b),
                        )
                    }
                };
                field_values_map
                    .entry(*field)
                    .or_insert_with(|| (entry.name.clone(), entry.multi, Vec::new()))
                    .2
                    .push(json_value);
            }
        }

        // Convert to JSON object, using arrays for multi fields or when multiple values exist
        let mut map = serde_json::Map::new();
        for (_field, (name, is_multi, values)) in field_values_map {
            let json_value = if is_multi || values.len() > 1 {
                serde_json::Value::Array(values)
            } else {
                values.into_iter().next().unwrap()
            };
            map.insert(name, json_value);
        }

        serde_json::Value::Object(map)
    }

    /// Create a Document from a JSON object using field names from schema
    ///
    /// Supports:
    /// - String values -> Text fields
    /// - Number values -> U64/I64/F64 fields (based on schema type)
    /// - Array values -> Multiple values for the same field (multifields)
    ///
    /// Unknown fields (not in schema) are silently ignored.
    pub fn from_json(json: &serde_json::Value, schema: &Schema) -> Option<Self> {
        let obj = json.as_object()?;
        let mut doc = Document::new();

        for (key, value) in obj {
            if let Some(field) = schema.get_field(key) {
                let field_entry = schema.get_field_entry(field)?;
                Self::add_json_value(&mut doc, field, &field_entry.field_type, value);
            }
        }

        Some(doc)
    }

    /// Helper to add a JSON value to a document, handling type conversion
    fn add_json_value(
        doc: &mut Document,
        field: Field,
        field_type: &FieldType,
        value: &serde_json::Value,
    ) {
        match value {
            serde_json::Value::String(s) => {
                if matches!(field_type, FieldType::Text) {
                    doc.add_text(field, s.clone());
                }
            }
            serde_json::Value::Number(n) => {
                match field_type {
                    FieldType::I64 => {
                        if let Some(i) = n.as_i64() {
                            doc.add_i64(field, i);
                        }
                    }
                    FieldType::U64 => {
                        if let Some(u) = n.as_u64() {
                            doc.add_u64(field, u);
                        } else if let Some(i) = n.as_i64() {
                            // Allow positive i64 as u64
                            if i >= 0 {
                                doc.add_u64(field, i as u64);
                            }
                        }
                    }
                    FieldType::F64 => {
                        if let Some(f) = n.as_f64() {
                            doc.add_f64(field, f);
                        }
                    }
                    _ => {}
                }
            }
            // Handle arrays (multifields) - add each element separately
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::add_json_value(doc, field, field_type, item);
                }
            }
            // Handle sparse vector objects
            serde_json::Value::Object(obj) if matches!(field_type, FieldType::SparseVector) => {
                if let (Some(indices_val), Some(values_val)) =
                    (obj.get("indices"), obj.get("values"))
                {
                    let indices: Vec<u32> = indices_val
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as u32))
                                .collect()
                        })
                        .unwrap_or_default();
                    let values: Vec<f32> = values_val
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_f64().map(|n| n as f32))
                                .collect()
                        })
                        .unwrap_or_default();
                    if indices.len() == values.len() {
                        let entries: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
                        doc.add_sparse_vector(field, entries);
                    }
                }
            }
            // Handle JSON fields - accept any value directly
            _ if matches!(field_type, FieldType::Json) => {
                doc.add_json(field, value.clone());
            }
            serde_json::Value::Object(_) => {}
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder() {
        let mut builder = Schema::builder();
        let title = builder.add_text_field("title", true, true);
        let body = builder.add_text_field("body", true, false);
        let count = builder.add_u64_field("count", true, true);
        let schema = builder.build();

        assert_eq!(schema.get_field("title"), Some(title));
        assert_eq!(schema.get_field("body"), Some(body));
        assert_eq!(schema.get_field("count"), Some(count));
        assert_eq!(schema.get_field("nonexistent"), None);
    }

    #[test]
    fn ivf_tq_defaults_to_selective_soar() {
        for config in [
            DenseVectorConfig::new(8),
            DenseVectorConfig::ivf_tq(8, Some(4), 2),
        ] {
            let soar = config.soar.expect("IVF-TQ should enable SOAR by default");
            assert_eq!(soar.num_secondary, 1);
            assert!(soar.selective);
            assert_eq!(soar.calibration_target(), Some(0.30));
        }

        assert!(DenseVectorConfig::flat(8).soar.is_none());
        assert!(DenseVectorConfig::tq(8).soar.is_none());
    }

    #[test]
    fn binary_ivf_uses_measured_balanced_fifteen_million_geometry() {
        let config = BinaryDenseVectorConfig::new(2_560);
        assert_eq!(config.optimal_num_clusters(15_000_000), 3_873);

        let explicit = config.with_ivf(Some(8_192), 128);
        assert_eq!(explicit.optimal_num_clusters(15_000_000), 8_192);
    }

    #[test]
    fn target_vectors_sizes_automatic_topology_but_explicit_clusters_win() {
        let hinted = BinaryDenseVectorConfig::new(2_560).with_target_vectors(1_000_000_000);
        assert_eq!(hinted.optimal_num_clusters(1_000_000), 31_623);

        let lower_hint = BinaryDenseVectorConfig::new(2_560).with_target_vectors(1_000_000);
        assert_eq!(
            lower_hint.optimal_num_clusters(15_000_000),
            BinaryDenseVectorConfig::new(2_560).optimal_num_clusters(15_000_000),
            "a steady-state hint is a lower bound and must not shrink live-corpus geometry"
        );

        let explicit = hinted.with_ivf(Some(8_192), 128);
        assert_eq!(explicit.optimal_num_clusters(1_000_000), 8_192);

        let float = DenseVectorConfig::ivf_tq(1_024, None, 64).with_target_vectors(1_000_000_000);
        assert_eq!(float.optimal_num_clusters(1_000_000), 252_982);
    }

    #[test]
    fn persisted_target_vectors_must_be_positive_and_topology_bearing() {
        let mut zero = BinaryDenseVectorConfig::new(256);
        zero.target_vectors = Some(0);
        let mut builder = Schema::builder();
        builder.add_binary_dense_vector_field_with_config("hash", true, false, zero);
        let error = reject_removed_vector_index_types(&builder.build()).unwrap_err();
        assert!(error.contains("positive steady-state"), "{error}");

        let hinted = DenseVectorConfig::ivf_tq(128, None, 64).with_target_vectors(1_000_000_000);
        let encoded = serde_json::to_value(&hinted).unwrap();
        let decoded: DenseVectorConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.target_vectors, Some(1_000_000_000));

        let binary = BinaryDenseVectorConfig::new(2_560).with_target_vectors(1_000_000_000);
        let encoded = serde_json::to_value(&binary).unwrap();
        let decoded: BinaryDenseVectorConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.target_vectors, Some(1_000_000_000));

        let old_dense: DenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 128,
            "index_type": "ivf_tq"
        }))
        .unwrap();
        assert_eq!(old_dense.target_vectors, None);
        let old_binary: BinaryDenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 256,
            "index_type": "ivf"
        }))
        .unwrap();
        assert_eq!(old_binary.target_vectors, None);

        let flat = DenseVectorConfig::flat(128).with_target_vectors(1_000_000);
        let mut builder = Schema::builder();
        builder.add_dense_vector_field_with_config("embedding", true, false, flat);
        let error = reject_removed_vector_index_types(&builder.build()).unwrap_err();
        assert!(error.contains("flat/training-free"), "{error}");

        let mut binary_flat = BinaryDenseVectorConfig::new(256);
        binary_flat.index_type = BinaryIndexType::Flat;
        binary_flat.target_vectors = Some(1_000_000);
        let mut builder = Schema::builder();
        builder.add_binary_dense_vector_field_with_config("hash", true, false, binary_flat);
        let error = reject_removed_vector_index_types(&builder.build()).unwrap_err();
        assert!(error.contains("flat/training-free"), "{error}");
    }

    #[test]
    fn omitted_and_explicitly_disabled_soar_are_distinct_in_serde() {
        let omitted: DenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 8,
            "index_type": "ivf_tq"
        }))
        .unwrap();
        let default_soar = omitted
            .soar
            .as_ref()
            .expect("an omitted SOAR setting should enable the selective default");
        assert_eq!(default_soar.num_secondary, 1);
        assert!(default_soar.selective);
        assert_eq!(default_soar.calibration_target(), Some(0.30));

        let disabled: DenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 8,
            "index_type": "ivf_tq",
            "soar": null
        }))
        .unwrap();
        assert!(disabled.soar.is_none());

        let encoded = serde_json::to_value(&disabled).unwrap();
        assert_eq!(encoded.get("soar"), Some(&serde_json::Value::Null));
        let round_trip: DenseVectorConfig = serde_json::from_value(encoded).unwrap();
        assert!(
            round_trip.soar.is_none(),
            "explicit off must survive a schema round trip"
        );
    }

    #[test]
    fn scann_config_serde_preserves_old_json_defaults_and_new_parameters() {
        let old_dense: DenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 768,
            "index_type": "ivf_tq"
        }))
        .unwrap();
        assert_eq!(old_dense.tree_levels, None);
        let old_json = serde_json::to_value(&old_dense).unwrap();
        assert!(old_json.get("tree_levels").is_none());

        let scann: DenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 1024,
            "index_type": "scann",
            "num_clusters": 10_000_000,
            "tree_levels": 2,
            "nprobe": 1024
        }))
        .unwrap();
        assert_eq!(scann.index_type, VectorIndexType::Scann);
        assert_eq!(scann.tree_levels, Some(2));
        assert!(scann.soar.is_none());

        let binary: BinaryDenseVectorConfig = serde_json::from_value(serde_json::json!({
            "dim": 1024,
            "index_type": "scann",
            "tree_levels": 3
        }))
        .unwrap();
        assert_eq!(binary.index_type, BinaryIndexType::Scann);
        assert_eq!(binary.tree_levels, Some(3));
    }

    #[test]
    fn persisted_scann_geometry_is_validated_on_schema_load() {
        let mut invalid_levels = DenseVectorConfig::new(128);
        invalid_levels.index_type = VectorIndexType::Scann;
        invalid_levels.tree_levels = Some(4);
        invalid_levels.soar = None;
        let mut builder = Schema::builder();
        builder.add_dense_vector_field_with_config("embedding", true, false, invalid_levels);
        let error = reject_removed_vector_index_types(&builder.build())
            .expect_err("invalid persisted ScaNN levels must fail at the schema gate");
        assert!(error.contains("1..=3"), "{error}");

        let mut wrong_algorithm = BinaryDenseVectorConfig::new(256);
        wrong_algorithm.tree_levels = Some(2);
        let mut builder = Schema::builder();
        builder.add_binary_dense_vector_field_with_config("hash", true, false, wrong_algorithm);
        let error = reject_removed_vector_index_types(&builder.build())
            .expect_err("ScaNN-only persisted options must fail on IVF");
        assert!(error.contains("does not use the ScaNN index"), "{error}");

        let mut invalid_soar = DenseVectorConfig::flat(128);
        invalid_soar.index_type = VectorIndexType::Scann;
        invalid_soar.nprobe = 1;
        invalid_soar.soar = Some(crate::structures::SoarConfig::default());
        let mut builder = Schema::builder();
        builder.add_dense_vector_field_with_config("embedding", true, false, invalid_soar);
        let error = reject_removed_vector_index_types(&builder.build())
            .expect_err("persisted ScaNN SOAR must fail until assignments exist");
        assert!(error.contains("not implemented"), "{error}");

        let mut one_leaf = DenseVectorConfig::flat(128);
        one_leaf.index_type = VectorIndexType::Scann;
        one_leaf.num_clusters = Some(1);
        one_leaf.nprobe = 1;
        let mut builder = Schema::builder();
        builder.add_dense_vector_field_with_config("embedding", true, false, one_leaf);
        let error = reject_removed_vector_index_types(&builder.build())
            .expect_err("one-leaf ScaNN geometry must fail at schema load");
        assert!(error.contains("2..=30000000"), "{error}");

        let binary = BinaryDenseVectorConfig {
            dim: 255,
            index_type: BinaryIndexType::Scann,
            num_clusters: Some(2),
            target_vectors: None,
            tree_levels: Some(1),
            ivf_routing: IvfRoutingMode::Auto,
            nprobe: 1,
            soar: None,
        };
        let mut builder = Schema::builder();
        builder.add_binary_dense_vector_field_with_config("hash", true, false, binary);
        let error = reject_removed_vector_index_types(&builder.build())
            .expect_err("binary ScaNN dimensions must be byte-aligned");
        assert!(error.contains("multiple of 8"), "{error}");
    }

    #[test]
    fn test_set_primary_key_forces_fast_and_indexed() {
        // Regression: the SDL path forces fast + indexed on primary-key fields
        // (needed for dedup lookups against the fast-field text dict). The
        // programmatic builder must do the same, otherwise committed-key dedup
        // is silently inert after every commit.
        let mut builder = Schema::builder();
        let id = builder.add_text_field("id", false, true);
        builder.set_primary_key(id);
        let schema = builder.build();

        let entry = schema.get_field_entry(id).unwrap();
        assert!(entry.primary_key);
        assert!(
            entry.fast,
            "primary key must imply fast (dedup reads the fast-field text dict)"
        );
        assert!(entry.indexed, "primary key must imply indexed");
    }

    #[test]
    fn test_document() {
        let mut builder = Schema::builder();
        let title = builder.add_text_field("title", true, true);
        let count = builder.add_u64_field("count", true, true);
        let _schema = builder.build();

        let mut doc = Document::new();
        doc.add_text(title, "Hello World");
        doc.add_u64(count, 42);

        assert_eq!(doc.get_first(title).unwrap().as_text(), Some("Hello World"));
        assert_eq!(doc.get_first(count).unwrap().as_u64(), Some(42));
    }

    #[test]
    fn test_document_serialization() {
        let mut builder = Schema::builder();
        let title = builder.add_text_field("title", true, true);
        let count = builder.add_u64_field("count", true, true);
        let _schema = builder.build();

        let mut doc = Document::new();
        doc.add_text(title, "Hello World");
        doc.add_u64(count, 42);

        // Serialize
        let json = serde_json::to_string(&doc).unwrap();
        println!("Serialized doc: {}", json);

        // Deserialize
        let doc2: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(
            doc2.field_values().len(),
            2,
            "Should have 2 field values after deserialization"
        );
        assert_eq!(
            doc2.get_first(title).unwrap().as_text(),
            Some("Hello World")
        );
        assert_eq!(doc2.get_first(count).unwrap().as_u64(), Some(42));
    }

    #[test]
    fn test_multivalue_field() {
        let mut builder = Schema::builder();
        let uris = builder.add_text_field("uris", true, true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        // Create document with multiple values for the same field
        let mut doc = Document::new();
        doc.add_text(uris, "one");
        doc.add_text(uris, "two");
        doc.add_text(title, "Test Document");

        // Verify get_first returns the first value
        assert_eq!(doc.get_first(uris).unwrap().as_text(), Some("one"));

        // Verify get_all returns all values
        let all_uris: Vec<_> = doc.get_all(uris).collect();
        assert_eq!(all_uris.len(), 2);
        assert_eq!(all_uris[0].as_text(), Some("one"));
        assert_eq!(all_uris[1].as_text(), Some("two"));

        // Verify to_json returns array for multi-value field
        let json = doc.to_json(&schema);
        let uris_json = json.get("uris").unwrap();
        assert!(uris_json.is_array(), "Multi-value field should be an array");
        let uris_arr = uris_json.as_array().unwrap();
        assert_eq!(uris_arr.len(), 2);
        assert_eq!(uris_arr[0].as_str(), Some("one"));
        assert_eq!(uris_arr[1].as_str(), Some("two"));

        // Verify single-value field is NOT an array
        let title_json = json.get("title").unwrap();
        assert!(
            title_json.is_string(),
            "Single-value field should be a string"
        );
        assert_eq!(title_json.as_str(), Some("Test Document"));
    }

    #[test]
    fn test_multivalue_from_json() {
        let mut builder = Schema::builder();
        let uris = builder.add_text_field("uris", true, true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        // Create JSON with array value
        let json = serde_json::json!({
            "uris": ["one", "two"],
            "title": "Test Document"
        });

        // Parse from JSON
        let doc = Document::from_json(&json, &schema).unwrap();

        // Verify all values are present
        let all_uris: Vec<_> = doc.get_all(uris).collect();
        assert_eq!(all_uris.len(), 2);
        assert_eq!(all_uris[0].as_text(), Some("one"));
        assert_eq!(all_uris[1].as_text(), Some("two"));

        // Verify single value
        assert_eq!(
            doc.get_first(title).unwrap().as_text(),
            Some("Test Document")
        );

        // Verify roundtrip: to_json should produce equivalent JSON
        let json_out = doc.to_json(&schema);
        let uris_out = json_out.get("uris").unwrap().as_array().unwrap();
        assert_eq!(uris_out.len(), 2);
        assert_eq!(uris_out[0].as_str(), Some("one"));
        assert_eq!(uris_out[1].as_str(), Some("two"));
    }

    #[test]
    fn test_multi_attribute_forces_array() {
        // Test that fields marked as 'multi' are always serialized as arrays,
        // even when they have only one value
        let mut builder = Schema::builder();
        let uris = builder.add_text_field("uris", true, true);
        builder.set_multi(uris, true); // Mark as multi
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        // Verify the multi attribute is set
        assert!(schema.get_field_entry(uris).unwrap().multi);
        assert!(!schema.get_field_entry(title).unwrap().multi);

        // Create document with single value for multi field
        let mut doc = Document::new();
        doc.add_text(uris, "only_one");
        doc.add_text(title, "Test Document");

        // Verify to_json returns array for multi field even with single value
        let json = doc.to_json(&schema);

        let uris_json = json.get("uris").unwrap();
        assert!(
            uris_json.is_array(),
            "Multi field should be array even with single value"
        );
        let uris_arr = uris_json.as_array().unwrap();
        assert_eq!(uris_arr.len(), 1);
        assert_eq!(uris_arr[0].as_str(), Some("only_one"));

        // Verify non-multi field with single value is NOT an array
        let title_json = json.get("title").unwrap();
        assert!(
            title_json.is_string(),
            "Non-multi single-value field should be a string"
        );
        assert_eq!(title_json.as_str(), Some("Test Document"));
    }

    #[test]
    fn test_sparse_vector_field() {
        let mut builder = Schema::builder();
        let embedding = builder.add_sparse_vector_field("embedding", true, true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        assert_eq!(schema.get_field("embedding"), Some(embedding));
        assert_eq!(
            schema.get_field_entry(embedding).unwrap().field_type,
            FieldType::SparseVector
        );

        // Create document with sparse vector
        let mut doc = Document::new();
        doc.add_sparse_vector(embedding, vec![(0, 1.0), (5, 2.5), (10, 0.5)]);
        doc.add_text(title, "Test Document");

        // Verify accessor
        let entries = doc
            .get_first(embedding)
            .unwrap()
            .as_sparse_vector()
            .unwrap();
        assert_eq!(entries, &[(0, 1.0), (5, 2.5), (10, 0.5)]);

        // Verify JSON roundtrip
        let json = doc.to_json(&schema);
        let embedding_json = json.get("embedding").unwrap();
        assert!(embedding_json.is_object());
        assert_eq!(
            embedding_json
                .get("indices")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            3
        );

        // Parse back from JSON
        let doc2 = Document::from_json(&json, &schema).unwrap();
        let entries2 = doc2
            .get_first(embedding)
            .unwrap()
            .as_sparse_vector()
            .unwrap();
        assert_eq!(entries2[0].0, 0);
        assert!((entries2[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(entries2[1].0, 5);
        assert!((entries2[1].1 - 2.5).abs() < 1e-6);
        assert_eq!(entries2[2].0, 10);
        assert!((entries2[2].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_json_field() {
        let mut builder = Schema::builder();
        let metadata = builder.add_json_field("metadata", true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        assert_eq!(schema.get_field("metadata"), Some(metadata));
        assert_eq!(
            schema.get_field_entry(metadata).unwrap().field_type,
            FieldType::Json
        );
        // JSON fields are never indexed
        assert!(!schema.get_field_entry(metadata).unwrap().indexed);
        assert!(schema.get_field_entry(metadata).unwrap().stored);

        // Create document with JSON value (object)
        let json_value = serde_json::json!({
            "author": "John Doe",
            "tags": ["rust", "search"],
            "nested": {"key": "value"}
        });
        let mut doc = Document::new();
        doc.add_json(metadata, json_value.clone());
        doc.add_text(title, "Test Document");

        // Verify accessor
        let stored_json = doc.get_first(metadata).unwrap().as_json().unwrap();
        assert_eq!(stored_json, &json_value);
        assert_eq!(
            stored_json.get("author").unwrap().as_str(),
            Some("John Doe")
        );

        // Verify JSON roundtrip via to_json/from_json
        let doc_json = doc.to_json(&schema);
        let metadata_out = doc_json.get("metadata").unwrap();
        assert_eq!(metadata_out, &json_value);

        // Parse back from JSON
        let doc2 = Document::from_json(&doc_json, &schema).unwrap();
        let stored_json2 = doc2.get_first(metadata).unwrap().as_json().unwrap();
        assert_eq!(stored_json2, &json_value);
    }

    #[test]
    fn test_json_field_various_types() {
        let mut builder = Schema::builder();
        let data = builder.add_json_field("data", true);
        let _schema = builder.build();

        // Test with array
        let arr_value = serde_json::json!([1, 2, 3, "four", null]);
        let mut doc = Document::new();
        doc.add_json(data, arr_value.clone());
        assert_eq!(doc.get_first(data).unwrap().as_json().unwrap(), &arr_value);

        // Test with string
        let str_value = serde_json::json!("just a string");
        let mut doc2 = Document::new();
        doc2.add_json(data, str_value.clone());
        assert_eq!(doc2.get_first(data).unwrap().as_json().unwrap(), &str_value);

        // Test with number
        let num_value = serde_json::json!(42.5);
        let mut doc3 = Document::new();
        doc3.add_json(data, num_value.clone());
        assert_eq!(doc3.get_first(data).unwrap().as_json().unwrap(), &num_value);

        // Test with null
        let null_value = serde_json::Value::Null;
        let mut doc4 = Document::new();
        doc4.add_json(data, null_value.clone());
        assert_eq!(
            doc4.get_first(data).unwrap().as_json().unwrap(),
            &null_value
        );

        // Test with boolean
        let bool_value = serde_json::json!(true);
        let mut doc5 = Document::new();
        doc5.add_json(data, bool_value.clone());
        assert_eq!(
            doc5.get_first(data).unwrap().as_json().unwrap(),
            &bool_value
        );
    }
}
