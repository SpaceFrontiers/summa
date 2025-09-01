//! Schema Definition Language (SDL) for Hermes
//!
//! A simple, readable format for defining index schemas using pest parser.
//!
//! # Example SDL
//!
//! ```text
//! # Article index schema
//! index articles {
//!     # Primary text field for full-text search
//!     field title: text [indexed, stored]
//!
//!     # Body content - indexed but not stored (save space)
//!     field body: text [indexed]
//!
//!     # Author name
//!     field author: text [indexed, stored]
//!
//!     # Publication timestamp
//!     field published_at: i64 [indexed, stored]
//!
//!     # View count
//!     field views: u64 [indexed, stored]
//!
//!     # Rating score
//!     field rating: f64 [indexed, stored]
//!
//!     # Raw content hash (not indexed, just stored)
//!     field content_hash: bytes [stored]
//!
//!     # Dense vector with the production IVF-PQ index
//!     field embedding: dense_vector<768> [indexed<ivf_tq, routing: hnsw, nprobe: 64>]
//!
//! }
//! ```
//!
//! # Dense Vector Index Configuration
//!
//! Index-related parameters for dense vectors are specified in `indexed<...>`:
//! - `ivf_tq` - index type
//! - `centroids: "path"` - path to pre-trained centroids file
//! - `nprobe: N` - number of clusters to probe (default: 64)

use pest::Parser;
use pest_derive::Parser;

use super::query_field_router::{QueryRouterRule, RoutingMode};
use super::schema::{DenseVectorQuantization, FieldType, Schema, SchemaBuilder};
use crate::Result;
use crate::error::Error;

#[derive(Parser)]
#[grammar = "dsl/sdl/sdl.pest"]
pub struct SdlParser;

use super::schema::{BinaryDenseVectorConfig, DenseVectorConfig};
use crate::structures::{
    IndexSize, QueryWeighting, SparseFormat, SparseQueryConfig, SparseVectorConfig,
    WeightQuantization,
};

/// Parsed field definition
#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,
    pub indexed: bool,
    pub stored: bool,
    /// Tokenizer name for text fields (e.g., "simple", "en_stem", "german")
    pub tokenizer: Option<String>,
    /// Whether this field can have multiple values (serialized as array in JSON)
    pub multi: bool,
    /// Position tracking mode for phrase queries and multi-field element tracking
    pub positions: Option<super::schema::PositionMode>,
    /// Configuration for sparse vector fields
    pub sparse_vector_config: Option<SparseVectorConfig>,
    /// Configuration for dense vector fields
    pub dense_vector_config: Option<DenseVectorConfig>,
    /// Configuration for binary dense vector fields
    pub binary_dense_vector_config: Option<BinaryDenseVectorConfig>,
    /// Whether this field has columnar fast-field storage
    pub fast: bool,
    /// Whether this field is a primary key (unique constraint)
    pub primary: bool,
    /// Whether build-time document reordering (BP) is enabled for BMP fields
    pub reorder: bool,
    /// BM25 k1 of a text field (`indexed<k1: ...>`), `None` = default
    pub bm25_k1: Option<f32>,
    /// BM25 b of a text field (`indexed<b: ...>`), `None` = default
    pub bm25_b: Option<f32>,
    /// Chunked text field (`indexed<chunked>`): each value is its own BM25 unit
    pub chunked: bool,
}

/// Parsed index definition
#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub default_fields: Vec<String>,
    /// Query router rules for routing queries to specific fields
    pub query_routers: Vec<QueryRouterRule>,
    /// BP-reorder `reorder`-attributed BMP fields inside merges
    /// (index-level `reorder_on_merge: true`). Absent = disabled.
    pub reorder_on_merge: bool,
}

impl IndexDef {
    /// Convert to a Schema
    pub fn to_schema(&self) -> Schema {
        let mut builder = SchemaBuilder::default();

        for field in &self.fields {
            let f = match field.field_type {
                FieldType::Text => {
                    let tokenizer = field.tokenizer.as_deref().unwrap_or("simple");
                    builder.add_text_field_with_tokenizer(
                        &field.name,
                        field.indexed,
                        field.stored,
                        tokenizer,
                    )
                }
                FieldType::U64 => builder.add_u64_field(&field.name, field.indexed, field.stored),
                FieldType::I64 => builder.add_i64_field(&field.name, field.indexed, field.stored),
                FieldType::F64 => builder.add_f64_field(&field.name, field.indexed, field.stored),
                FieldType::Bytes => builder.add_bytes_field(&field.name, field.stored),
                FieldType::Json => builder.add_json_field(&field.name, field.stored),
                FieldType::SparseVector => {
                    if let Some(config) = &field.sparse_vector_config {
                        builder.add_sparse_vector_field_with_config(
                            &field.name,
                            field.indexed,
                            field.stored,
                            config.clone(),
                        )
                    } else {
                        builder.add_sparse_vector_field(&field.name, field.indexed, field.stored)
                    }
                }
                FieldType::DenseVector => {
                    // Dense vector dimension must be specified via config
                    let config = field
                        .dense_vector_config
                        .as_ref()
                        .expect("DenseVector field requires dimension to be specified");
                    builder.add_dense_vector_field_with_config(
                        &field.name,
                        field.indexed,
                        field.stored,
                        config.clone(),
                    )
                }
                FieldType::BinaryDenseVector => {
                    let config = field
                        .binary_dense_vector_config
                        .as_ref()
                        .expect("BinaryDenseVector field requires dimension to be specified");
                    builder.add_binary_dense_vector_field_with_config(
                        &field.name,
                        field.indexed,
                        field.stored,
                        config.clone(),
                    )
                }
            };
            if field.multi {
                builder.set_multi(f, true);
            }
            if field.fast {
                builder.set_fast(f, true);
            }
            if field.primary {
                builder.set_primary_key(f);
            }
            if field.reorder {
                builder.set_reorder(f, true);
            }
            if field.chunked {
                builder.set_chunked(f, true);
            }
            if field.bm25_k1.is_some() || field.bm25_b.is_some() {
                builder.set_bm25_params(f, field.bm25_k1, field.bm25_b);
            }
            // Set positions: explicit > auto (ordinal for multi vectors)
            let positions = field.positions.or({
                // Auto-set ordinal positions for multi-valued vector fields
                if field.multi
                    && matches!(
                        field.field_type,
                        FieldType::SparseVector
                            | FieldType::DenseVector
                            | FieldType::BinaryDenseVector
                    )
                {
                    Some(super::schema::PositionMode::Ordinal)
                } else {
                    None
                }
            });
            if let Some(mode) = positions {
                builder.set_positions(f, mode);
            }
        }

        // Set default fields if specified
        if !self.default_fields.is_empty() {
            builder.set_default_fields(self.default_fields.clone());
        }

        // Set query routers if specified
        if !self.query_routers.is_empty() {
            builder.set_query_routers(self.query_routers.clone());
        }

        builder.set_index_name(self.name.clone());

        if self.reorder_on_merge {
            if self.fields.iter().any(|f| f.reorder) {
                builder.set_reorder_on_merge(true);
            } else {
                // Fail loud: the option would silently do nothing without at
                // least one `reorder`-attributed field.
                log::warn!(
                    "index '{}': reorder_on_merge is set but no field has the `reorder` attribute — merges will not reorder anything",
                    self.name,
                );
                builder.set_reorder_on_merge(true);
            }
        }

        builder.build()
    }

    /// Create a QueryFieldRouter from the query router rules
    ///
    /// Returns None if there are no query router rules defined.
    /// Returns Err if any regex pattern is invalid.
    pub fn to_query_router(&self) -> Result<Option<super::query_field_router::QueryFieldRouter>> {
        if self.query_routers.is_empty() {
            return Ok(None);
        }

        super::query_field_router::QueryFieldRouter::from_rules(&self.query_routers)
            .map(Some)
            .map_err(Error::Schema)
    }
}

/// Parse field type from string
fn parse_field_type(type_str: &str) -> Result<FieldType> {
    match type_str {
        "text" | "string" | "str" => Ok(FieldType::Text),
        "u64" | "uint" | "unsigned" => Ok(FieldType::U64),
        "i64" | "int" | "integer" => Ok(FieldType::I64),
        "f64" | "float" | "double" => Ok(FieldType::F64),
        "bytes" | "binary" | "blob" => Ok(FieldType::Bytes),
        "json" => Ok(FieldType::Json),
        "sparse_vector" => Ok(FieldType::SparseVector),
        "dense_vector" | "vector" => Ok(FieldType::DenseVector),
        "binary_dense_vector" | "binary_vector" => Ok(FieldType::BinaryDenseVector),
        _ => Err(Error::Schema(format!("Unknown field type: {}", type_str))),
    }
}

/// Index configuration parsed from indexed<...> attribute
#[derive(Debug, Clone, Default)]
enum SoarDirective {
    /// No `soar:` keyword was present. IVF-TQ resolves this to its selective
    /// default after the final index type is known.
    #[default]
    Unspecified,
    /// `soar: off` was explicitly requested.
    Disabled,
    /// An explicit selective/full/aggressive preset.
    Enabled(crate::structures::SoarConfig),
}

#[derive(Debug, Clone, Default)]
struct IndexConfig {
    index_type: Option<super::schema::VectorIndexType>,
    num_clusters: Option<usize>,
    target_vectors: Option<u64>,
    tree_levels: Option<u8>,
    nprobe: Option<usize>,
    ivf_routing: Option<super::schema::IvfRoutingMode>,
    soar: SoarDirective,
    binary_index_type: Option<super::schema::BinaryIndexType>,
    // Sparse vector index params
    sparse_format: Option<SparseFormat>,
    quantization: Option<WeightQuantization>,
    weight_threshold: Option<f32>,
    block_size: Option<usize>,
    bmp_block_size: Option<u32>,
    bmp_grid_bits: Option<u8>,
    pruning: Option<f32>,
    min_terms: Option<usize>,
    doc_mass: Option<f32>,
    // Sparse vector query-time config
    query_tokenizer: Option<String>,
    query_weighting: Option<QueryWeighting>,
    query_weight_threshold: Option<f32>,
    query_max_dims: Option<usize>,
    query_pruning: Option<f32>,
    query_min_query_dims: Option<usize>,
    query_lsp_gamma: Option<usize>,
    // BMP fixed dims (vocabulary size) and max weight scale
    dims: Option<u32>,
    max_weight: Option<f32>,
    // Position tracking mode for phrase queries
    positions: Option<super::schema::PositionMode>,
    // Chunked text field: every value is its own BM25 unit
    chunked: bool,
    // BM25 parameters of a text field
    bm25_k1: Option<f32>,
    bm25_b: Option<f32>,
}

/// Parsed attributes from SDL field definition
struct ParsedAttributes {
    indexed: bool,
    stored: bool,
    multi: bool,
    fast: bool,
    primary: bool,
    reorder: bool,
    index_config: Option<IndexConfig>,
}

/// Parse attributes from pest pair
fn parse_attributes(pair: pest::iterators::Pair<Rule>) -> Result<ParsedAttributes> {
    let mut attrs = ParsedAttributes {
        indexed: false,
        stored: false,
        multi: false,
        fast: false,
        primary: false,
        reorder: false,
        index_config: None,
    };

    for attr in pair.into_inner() {
        if attr.as_rule() == Rule::attribute {
            let mut found_config = false;
            for inner in attr.clone().into_inner() {
                match inner.as_rule() {
                    Rule::indexed_with_config => {
                        attrs.indexed = true;
                        attrs.index_config = Some(parse_index_config(inner)?);
                        found_config = true;
                        break;
                    }
                    Rule::stored_with_config => {
                        attrs.stored = true;
                        attrs.multi = true; // stored<multi>
                        found_config = true;
                        break;
                    }
                    _ => {}
                }
            }
            if !found_config {
                match attr.as_str() {
                    "indexed" => attrs.indexed = true,
                    "stored" => attrs.stored = true,
                    "fast" => attrs.fast = true,
                    "primary" => attrs.primary = true,
                    "reorder" => attrs.reorder = true,
                    _ => {}
                }
            }
        }
    }

    Ok(attrs)
}

/// Parse index configuration from indexed<...> attribute
fn parse_index_config(pair: pest::iterators::Pair<Rule>) -> Result<IndexConfig> {
    let mut config = IndexConfig::default();

    // indexed_with_config = { "indexed" ~ "<" ~ index_config_params ~ ">" }
    // index_config_params = { index_config_param ~ ("," ~ index_config_param)* }
    // index_config_param = { index_type_kwarg | centroids_kwarg | codebook_kwarg | nprobe_kwarg | index_type_spec }

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::index_config_params {
            for param in inner.into_inner() {
                if param.as_rule() == Rule::index_config_param {
                    for p in param.into_inner() {
                        parse_single_index_config_param(&mut config, p)?;
                    }
                }
            }
        }
    }

    Ok(config)
}

/// Parse a single index config parameter
fn parse_single_index_config_param(
    config: &mut IndexConfig,
    p: pest::iterators::Pair<Rule>,
) -> Result<()> {
    use super::schema::VectorIndexType;

    match p.as_rule() {
        Rule::index_type_spec => match p.as_str() {
            "flat" => {
                config.index_type = Some(VectorIndexType::Flat);
                config.binary_index_type = Some(super::schema::BinaryIndexType::Flat);
            }
            "ivf" => config.binary_index_type = Some(super::schema::BinaryIndexType::Ivf),
            "ivf_pq" => config.index_type = Some(VectorIndexType::IvfPq),
            "ivf_tq" => config.index_type = Some(VectorIndexType::IvfTq),
            "scann" => {
                config.index_type = Some(VectorIndexType::Scann);
                config.binary_index_type = Some(super::schema::BinaryIndexType::Scann);
            }
            "tq" => config.index_type = Some(VectorIndexType::Tq),
            _ => {}
        },
        Rule::index_type_kwarg => {
            // index_type_kwarg = { "index" ~ ":" ~ index_type_spec }
            if let Some(t) = p.into_inner().next() {
                match t.as_str() {
                    "flat" => {
                        config.index_type = Some(VectorIndexType::Flat);
                        config.binary_index_type = Some(super::schema::BinaryIndexType::Flat);
                    }
                    "ivf" => config.binary_index_type = Some(super::schema::BinaryIndexType::Ivf),
                    "ivf_pq" => config.index_type = Some(VectorIndexType::IvfPq),
                    "ivf_tq" => config.index_type = Some(VectorIndexType::IvfTq),
                    "scann" => {
                        config.index_type = Some(VectorIndexType::Scann);
                        config.binary_index_type = Some(super::schema::BinaryIndexType::Scann);
                    }
                    "tq" => config.index_type = Some(VectorIndexType::Tq),
                    _ => {}
                }
            }
        }
        Rule::num_clusters_kwarg => {
            // num_clusters_kwarg = { "num_clusters" ~ ":" ~ num_clusters_spec }
            if let Some(n) = p.into_inner().next() {
                config.num_clusters = Some(n.as_str().parse().map_err(|_| {
                    Error::Schema(format!(
                        "num_clusters '{}' does not fit on this platform",
                        n.as_str()
                    ))
                })?);
            }
        }
        Rule::target_vectors_kwarg => {
            if let Some(value) = p.into_inner().next() {
                config.target_vectors = Some(value.as_str().parse().map_err(|_| {
                    Error::Schema(format!(
                        "target_vectors '{}' does not fit in an unsigned 64-bit integer",
                        value.as_str()
                    ))
                })?);
            }
        }
        Rule::nprobe_kwarg => {
            // nprobe_kwarg = { "nprobe" ~ ":" ~ nprobe_spec }
            if let Some(n) = p.into_inner().next() {
                config.nprobe = Some(n.as_str().parse().map_err(|_| {
                    Error::Schema(format!(
                        "nprobe '{}' does not fit on this platform",
                        n.as_str()
                    ))
                })?);
            }
        }
        Rule::tree_levels_kwarg => {
            if let Some(value) = p.into_inner().next() {
                config.tree_levels = Some(value.as_str().parse().map_err(|_| {
                    Error::Schema(format!(
                        "tree_levels '{}' does not fit in an unsigned 8-bit integer",
                        value.as_str()
                    ))
                })?);
            }
        }
        Rule::routing_kwarg => {
            if let Some(value) = p.into_inner().next() {
                config.ivf_routing = Some(match value.as_str() {
                    "flat" => super::schema::IvfRoutingMode::Flat,
                    "two_level" => super::schema::IvfRoutingMode::TwoLevel,
                    "hnsw" => super::schema::IvfRoutingMode::Hnsw,
                    _ => super::schema::IvfRoutingMode::Auto,
                });
            }
        }
        Rule::soar_kwarg => {
            // soar_kwarg = { "soar" ~ ":" ~ soar_spec }
            if let Some(s) = p.into_inner().next() {
                use crate::structures::SoarConfig;
                config.soar = match s.as_str() {
                    "selective" => SoarDirective::Enabled(SoarConfig::new()),
                    "full" => SoarDirective::Enabled(SoarConfig::full()),
                    "aggressive" => SoarDirective::Enabled(SoarConfig::aggressive()),
                    _ => SoarDirective::Disabled, // "off"
                };
            }
        }
        Rule::quantization_kwarg => {
            // quantization_kwarg = { "quantization" ~ ":" ~ quantization_spec }
            if let Some(q) = p.into_inner().next() {
                config.quantization = Some(match q.as_str() {
                    "float32" | "f32" => WeightQuantization::Float32,
                    "float16" | "f16" => WeightQuantization::Float16,
                    "uint8" | "u8" => WeightQuantization::UInt8,
                    "uint4" | "u4" => WeightQuantization::UInt4,
                    _ => WeightQuantization::default(),
                });
            }
        }
        Rule::weight_threshold_kwarg => {
            // weight_threshold_kwarg = { "weight_threshold" ~ ":" ~ weight_threshold_spec }
            if let Some(t) = p.into_inner().next() {
                config.weight_threshold = Some(t.as_str().parse().unwrap_or_else(|_| {
                    log::warn!(
                        "Invalid weight_threshold value '{}', using default 0.0",
                        t.as_str()
                    );
                    0.0
                }));
            }
        }
        Rule::block_size_kwarg => {
            // block_size_kwarg = { "block_size" ~ ":" ~ block_size_spec }
            if let Some(n) = p.into_inner().next() {
                config.block_size = Some(n.as_str().parse().unwrap_or_else(|_| {
                    log::warn!(
                        "Invalid block_size value '{}', using default 128",
                        n.as_str()
                    );
                    128
                }));
            }
        }
        Rule::bmp_grid_bits_kwarg => {
            // bmp_grid_bits_kwarg = { "bmp_grid_bits" ~ ":" ~ bits_spec }
            if let Some(n) = p.into_inner().next() {
                config.bmp_grid_bits = Some(n.as_str().parse().unwrap_or_else(|_| {
                    log::warn!(
                        "Invalid bmp_grid_bits value '{}', using default {}",
                        n.as_str(),
                        SparseVectorConfig::DEFAULT_BMP_GRID_BITS,
                    );
                    SparseVectorConfig::DEFAULT_BMP_GRID_BITS
                }));
            }
        }
        Rule::bmp_block_size_kwarg => {
            // bmp_block_size_kwarg = { "bmp_block_size" ~ ":" ~ block_size_spec }
            if let Some(n) = p.into_inner().next() {
                config.bmp_block_size = Some(n.as_str().parse().unwrap_or_else(|_| {
                    log::warn!(
                        "Invalid bmp_block_size value '{}', using default {}",
                        n.as_str(),
                        SparseVectorConfig::DEFAULT_BMP_BLOCK_SIZE,
                    );
                    SparseVectorConfig::DEFAULT_BMP_BLOCK_SIZE
                }));
            }
        }
        Rule::pruning_kwarg => {
            // pruning_kwarg = { "pruning" ~ ":" ~ pruning_spec }
            if let Some(f) = p.into_inner().next() {
                config.pruning = Some(f.as_str().parse().unwrap_or_else(|_| {
                    log::warn!("Invalid pruning value '{}', using default 1.0", f.as_str());
                    1.0
                }));
            }
        }
        Rule::doc_mass_kwarg => {
            // doc_mass_kwarg = { "doc_mass" ~ ":" ~ pruning_spec }
            if let Some(f) = p.into_inner().next() {
                config.doc_mass = Some(f.as_str().parse().unwrap_or_else(|_| {
                    log::warn!("Invalid doc_mass value '{}', using 1.0 (off)", f.as_str());
                    1.0
                }));
            }
        }
        Rule::min_terms_kwarg => {
            if let Some(n) = p.into_inner().next() {
                config.min_terms = Some(n.as_str().parse().unwrap_or_else(|_| {
                    log::warn!("Invalid min_terms value '{}', using default 4", n.as_str());
                    4
                }));
            }
        }
        Rule::sparse_format_kwarg => {
            // sparse_format_kwarg = { "format" ~ ":" ~ sparse_format_spec }
            if let Some(f) = p.into_inner().next() {
                config.sparse_format = Some(match f.as_str() {
                    "bmp" => SparseFormat::Bmp,
                    "maxscore" => SparseFormat::MaxScore,
                    _ => SparseFormat::default(),
                });
            }
        }
        Rule::sparse_dims_kwarg => {
            if let Some(n) = p.into_inner().next() {
                config.dims = Some(n.as_str().parse().unwrap_or_else(|_| {
                    log::warn!("Invalid dims value '{}', using default 105879", n.as_str());
                    105879
                }));
            }
        }
        Rule::sparse_max_weight_kwarg => {
            if let Some(f) = p.into_inner().next() {
                config.max_weight = Some(f.as_str().parse().unwrap_or_else(|_| {
                    log::warn!(
                        "Invalid max_weight value '{}', using default 5.0",
                        f.as_str()
                    );
                    5.0
                }));
            }
        }
        Rule::query_config_block => {
            // query_config_block = { "query" ~ "<" ~ query_config_params ~ ">" }
            parse_query_config_block(config, p);
        }
        Rule::positions_kwarg => {
            // positions_kwarg = { "positions" | "ordinal" | "token_position" }
            use super::schema::PositionMode;
            config.positions = Some(match p.as_str() {
                "ordinal" => PositionMode::Ordinal,
                "token_position" => PositionMode::TokenPosition,
                _ => PositionMode::Full, // "positions" or any other value defaults to Full
            });
        }
        Rule::chunked_kwarg => {
            config.chunked = true;
        }
        Rule::bm25_k1_kwarg => {
            if let Some(v) = p.into_inner().next() {
                config.bm25_k1 = Some(v.as_str().parse().map_err(|_| {
                    Error::Schema(format!("invalid BM25 k1 value '{}'", v.as_str()))
                })?);
            }
        }
        Rule::bm25_b_kwarg => {
            if let Some(v) = p.into_inner().next() {
                let b: f32 = v
                    .as_str()
                    .parse()
                    .map_err(|_| Error::Schema(format!("invalid BM25 b value '{}'", v.as_str())))?;
                if !(0.0..=1.0).contains(&b) {
                    return Err(Error::Schema(format!(
                        "BM25 b must be between 0 and 1, got {b}"
                    )));
                }
                config.bm25_b = Some(b);
            }
        }
        _ => {}
    }

    Ok(())
}

/// Parse query configuration block: query<tokenizer: "...", weighting: idf>
fn parse_query_config_block(config: &mut IndexConfig, pair: pest::iterators::Pair<Rule>) {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::query_config_params {
            for param in inner.into_inner() {
                if param.as_rule() == Rule::query_config_param {
                    for p in param.into_inner() {
                        match p.as_rule() {
                            Rule::query_tokenizer_kwarg => {
                                // query_tokenizer_kwarg = { "tokenizer" ~ ":" ~ tokenizer_path }
                                if let Some(path) = p.into_inner().next()
                                    && let Some(inner_path) = path.into_inner().next()
                                {
                                    config.query_tokenizer = Some(inner_path.as_str().to_string());
                                }
                            }
                            Rule::query_weighting_kwarg => {
                                // query_weighting_kwarg = { "weighting" ~ ":" ~ weighting_spec }
                                if let Some(w) = p.into_inner().next() {
                                    config.query_weighting = Some(match w.as_str() {
                                        "one" => QueryWeighting::One,
                                        "idf" => QueryWeighting::Idf,
                                        "idf_file" => QueryWeighting::IdfFile,
                                        _ => QueryWeighting::One,
                                    });
                                }
                            }
                            Rule::query_weight_threshold_kwarg => {
                                if let Some(t) = p.into_inner().next() {
                                    config.query_weight_threshold =
                                        Some(t.as_str().parse().unwrap_or_else(|_| {
                                            log::warn!(
                                                "Invalid query weight_threshold '{}', using 0.0",
                                                t.as_str()
                                            );
                                            0.0
                                        }));
                                }
                            }
                            Rule::query_max_dims_kwarg => {
                                if let Some(t) = p.into_inner().next() {
                                    config.query_max_dims =
                                        Some(t.as_str().parse().unwrap_or_else(|_| {
                                            log::warn!(
                                                "Invalid query max_dims '{}', using 0",
                                                t.as_str()
                                            );
                                            0
                                        }));
                                }
                            }
                            Rule::query_pruning_kwarg => {
                                if let Some(t) = p.into_inner().next() {
                                    config.query_pruning =
                                        Some(t.as_str().parse().unwrap_or_else(|_| {
                                            log::warn!(
                                                "Invalid query pruning '{}', using 1.0",
                                                t.as_str()
                                            );
                                            1.0
                                        }));
                                }
                            }
                            Rule::query_min_query_dims_kwarg => {
                                if let Some(t) = p.into_inner().next() {
                                    config.query_min_query_dims =
                                        Some(t.as_str().parse().unwrap_or_else(|_| {
                                            log::warn!(
                                                "Invalid query min_query_dims '{}', using 4",
                                                t.as_str()
                                            );
                                            4
                                        }));
                                }
                            }
                            Rule::query_lsp_gamma_kwarg => {
                                if let Some(value) = p.into_inner().next() {
                                    config.query_lsp_gamma =
                                        Some(value.as_str().parse().unwrap_or_else(|_| {
                                            log::warn!(
                                                "Invalid query lsp_gamma '{}', using 0",
                                                value.as_str()
                                            );
                                            0
                                        }));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

/// Parse a field definition from pest pair
fn parse_field_def(pair: pest::iterators::Pair<Rule>) -> Result<FieldDef> {
    let mut inner = pair.into_inner();

    let name = inner
        .next()
        .ok_or_else(|| Error::Schema("Missing field name".to_string()))?
        .as_str()
        .to_string();

    let field_type_str = inner
        .next()
        .ok_or_else(|| Error::Schema("Missing field type".to_string()))?
        .as_str();

    let field_type = parse_field_type(field_type_str)?;

    // Parse optional tokenizer spec, sparse_vector_config, dense_vector_config, and attributes
    let mut tokenizer = None;
    let mut sparse_vector_config = None;
    let mut dense_vector_config = None;
    let mut binary_dense_vector_config = None;
    let mut indexed = true;
    let mut stored = true;
    let mut multi = false;
    let mut fast = false;
    let mut primary = false;
    let mut reorder = false;
    let mut index_config: Option<IndexConfig> = None;

    for item in inner {
        match item.as_rule() {
            Rule::tokenizer_spec => {
                // `<name>` or `<stem(by: field, default: simple)>`: store the
                // canonical spec string (validated against the index in
                // `parse_index_def`).
                let raw = item.as_str().trim();
                let raw = raw
                    .strip_prefix('<')
                    .and_then(|s| s.strip_suffix('>'))
                    .unwrap_or(raw);
                let spec = crate::tokenizer::TokenizerSpec::parse(raw)
                    .map_err(|e| Error::Schema(format!("Field '{name}': {e}")))?;
                tokenizer = Some(spec.to_string());
            }
            Rule::sparse_vector_config => {
                // Parse named parameters: <index_size: u16, quantization: uint8, weight_threshold: 0.1>
                sparse_vector_config = Some(parse_sparse_vector_config(item));
            }
            Rule::dense_vector_config => {
                // Parse dense_vector_params (keyword or positional) - only dims
                dense_vector_config = Some(parse_dense_vector_config(item));
            }
            Rule::binary_dense_vector_config => {
                // Parse binary dense vector config - just dimension (number of bits)
                let dim: usize = item
                    .into_inner()
                    .next()
                    .map(|d| d.as_str().parse().unwrap_or(0))
                    .unwrap_or(0);
                if dim == 0 || !dim.is_multiple_of(8) {
                    return Err(Error::Schema(format!(
                        "BinaryDenseVector dimension must be a positive multiple of 8, got {dim}"
                    )));
                }
                binary_dense_vector_config = Some(BinaryDenseVectorConfig::new(dim));
            }
            Rule::attributes => {
                let attrs = parse_attributes(item)?;
                indexed = attrs.indexed;
                stored = attrs.stored;
                multi = attrs.multi;
                fast = attrs.fast;
                primary = attrs.primary;
                reorder = attrs.reorder;
                index_config = attrs.index_config;
            }
            _ => {}
        }
    }

    // PEG grammar ambiguity: both dense_vector_config and binary_dense_vector_config
    // match `<N>`, and dense_vector_config comes first in the ordered choice. When the
    // field_type is BinaryDenseVector, remap the matched dense_vector_config.
    if field_type == FieldType::BinaryDenseVector
        && binary_dense_vector_config.is_none()
        && let Some(ref dv_config) = dense_vector_config
    {
        let dim = dv_config.dim;
        if dim == 0 || !dim.is_multiple_of(8) {
            return Err(Error::Schema(format!(
                "BinaryDenseVector dimension must be a positive multiple of 8, got {dim}"
            )));
        }
        binary_dense_vector_config = Some(BinaryDenseVectorConfig::new(dim));
        dense_vector_config = None;
    }

    // Primary key implies fast + indexed (needed for dedup lookups)
    if primary {
        fast = true;
        indexed = true;
    }

    // Merge index config into vector configs if both exist
    let mut positions = None;
    let mut chunked = false;
    let mut bm25_k1 = None;
    let mut bm25_b = None;
    if let Some(idx_cfg) = index_config {
        positions = idx_cfg.positions;
        chunked = idx_cfg.chunked;
        bm25_k1 = idx_cfg.bm25_k1;
        bm25_b = idx_cfg.bm25_b;
        if (bm25_k1.is_some() || bm25_b.is_some()) && field_type != FieldType::Text {
            return Err(Error::Schema(format!(
                "field '{name}': BM25 `k1`/`b` require a text field, got {field_type:?}"
            )));
        }
        if chunked {
            if field_type != FieldType::Text {
                return Err(Error::Schema(format!(
                    "field '{name}': `chunked` requires a text field, got {field_type:?}"
                )));
            }
            if let Some(mode) = positions
                && mode != super::schema::PositionMode::TokenPosition
            {
                return Err(Error::Schema(format!(
                    "field '{name}': a chunked text field may only declare `token_position` \
                     (positions restart in every chunk and the chunk is the ordinal); \
                     `{}` is not allowed",
                    match mode {
                        super::schema::PositionMode::Ordinal => "ordinal",
                        super::schema::PositionMode::Full => "positions",
                        super::schema::PositionMode::TokenPosition => unreachable!(),
                    }
                )));
            }
            // Stored values of a chunked field round-trip as an array.
            multi = true;
        }
        if let Some(ref mut bv_config) = binary_dense_vector_config {
            apply_index_config_to_binary_dense_vector(bv_config, idx_cfg)?;
        } else if let Some(ref mut dv_config) = dense_vector_config {
            apply_index_config_to_dense_vector(dv_config, idx_cfg)?;
        } else if field_type == FieldType::SparseVector {
            reject_scann_options_for_non_dense_vector(&idx_cfg, "sparse vector")?;
            // For sparse vectors, create default config if not present and apply index params
            let sv_config = sparse_vector_config.get_or_insert(SparseVectorConfig::default());
            apply_index_config_to_sparse_vector(sv_config, idx_cfg);
        } else {
            reject_scann_options_for_non_dense_vector(&idx_cfg, "non-vector field")?;
        }
    }

    Ok(FieldDef {
        name,
        field_type,
        indexed,
        stored,
        tokenizer,
        multi,
        positions,
        sparse_vector_config,
        dense_vector_config,
        binary_dense_vector_config,
        fast,
        primary,
        reorder,
        chunked,
        bm25_k1,
        bm25_b,
    })
}

fn reject_scann_options_for_non_dense_vector(config: &IndexConfig, field_kind: &str) -> Result<()> {
    if config.index_type == Some(super::schema::VectorIndexType::Scann)
        || config.binary_index_type == Some(super::schema::BinaryIndexType::Scann)
        || config.tree_levels.is_some()
        || config.target_vectors.is_some()
    {
        return Err(Error::Schema(format!(
            "vector index options require a dense or binary dense vector field, not a {field_kind}"
        )));
    }
    Ok(())
}

/// Apply index configuration from indexed<...> to BinaryDenseVectorConfig
fn apply_index_config_to_binary_dense_vector(
    config: &mut BinaryDenseVectorConfig,
    idx_cfg: IndexConfig,
) -> Result<()> {
    if idx_cfg.target_vectors == Some(0) {
        return Err(Error::Schema(
            "target_vectors must be greater than zero".to_string(),
        ));
    }
    if idx_cfg.index_type.is_some() && idx_cfg.binary_index_type.is_none() {
        return Err(Error::Schema(
            "binary dense vectors support only 'flat', 'ivf', or 'scann' index types".to_string(),
        ));
    }
    if let Some(index_type) = idx_cfg.binary_index_type {
        config.index_type = index_type;
    }
    if idx_cfg.target_vectors.is_some() && config.index_type == super::schema::BinaryIndexType::Flat
    {
        return Err(Error::Schema(
            "'target_vectors' is only valid for binary IVF or ScaNN automatic topology".to_string(),
        ));
    }
    validate_scann_index_options(
        "binary dense vector",
        config.index_type == super::schema::BinaryIndexType::Scann,
        &idx_cfg,
    )?;
    match &idx_cfg.soar {
        SoarDirective::Unspecified | SoarDirective::Disabled => config.soar = None,
        SoarDirective::Enabled(soar)
            if config.index_type == super::schema::BinaryIndexType::Scann =>
        {
            config.soar = Some(soar.clone());
        }
        SoarDirective::Enabled(_) => {
            return Err(Error::Schema(
                "'soar' on a binary dense vector requires the ScaNN index".to_string(),
            ));
        }
    }
    if idx_cfg.num_clusters.is_some() {
        config.num_clusters = idx_cfg.num_clusters;
    }
    if idx_cfg.target_vectors.is_some() {
        config.target_vectors = idx_cfg.target_vectors;
    }
    if idx_cfg.tree_levels.is_some() {
        config.tree_levels = idx_cfg.tree_levels;
    }
    if let Some(nprobe) = idx_cfg.nprobe {
        config.nprobe = nprobe;
    }
    if let Some(routing) = idx_cfg.ivf_routing {
        config.ivf_routing = routing;
    }
    Ok(())
}

/// Apply index configuration from indexed<...> to DenseVectorConfig
fn apply_index_config_to_dense_vector(
    config: &mut DenseVectorConfig,
    idx_cfg: IndexConfig,
) -> Result<()> {
    if idx_cfg.target_vectors == Some(0) {
        return Err(Error::Schema(
            "target_vectors must be greater than zero".to_string(),
        ));
    }
    if idx_cfg.binary_index_type.is_some() && idx_cfg.index_type.is_none() {
        return Err(Error::Schema(
            "float dense vectors do not support the binary-only 'ivf' index type; use 'ivf_tq' or 'scann'"
                .to_string(),
        ));
    }
    // Apply index type if specified
    if let Some(index_type) = idx_cfg.index_type {
        config.index_type = index_type;
    }
    if idx_cfg.target_vectors.is_some()
        && matches!(
            config.index_type,
            super::schema::VectorIndexType::Flat | super::schema::VectorIndexType::Tq
        )
    {
        return Err(Error::Schema(
            "'target_vectors' is only valid for IVF-TQ or ScaNN automatic topology".to_string(),
        ));
    }

    validate_scann_index_options(
        "dense vector",
        config.index_type == super::schema::VectorIndexType::Scann,
        &idx_cfg,
    )?;
    if idx_cfg.target_vectors.is_some() {
        config.target_vectors = idx_cfg.target_vectors;
    }

    // TQ scans every code (no probing, no clusters, no routing); accepting
    // these knobs silently would misrepresent how the field is searched.
    if config.index_type == super::schema::VectorIndexType::Tq {
        for (option, present) in [
            ("num_clusters", idx_cfg.num_clusters.is_some()),
            ("nprobe", idx_cfg.nprobe.is_some()),
            ("routing", idx_cfg.ivf_routing.is_some()),
        ] {
            if present {
                log::warn!(
                    "'{option}' has no effect on the 'tq' index (training-free full \
                     scan); ignoring"
                );
            }
        }
        // Canonicalize to the same shape as DenseVectorConfig::tq() so every
        // construction path yields an identical config for a `tq` field.
        config.num_clusters = None;
        config.nprobe = 0;
        config.ivf_routing = super::schema::IvfRoutingMode::Flat;
        apply_soar_to_dense_vector(config, idx_cfg)?;
        return Ok(());
    }

    // Apply num_clusters for IVF-based indexes
    if idx_cfg.num_clusters.is_some() {
        config.num_clusters = idx_cfg.num_clusters;
    }
    if idx_cfg.tree_levels.is_some() {
        config.tree_levels = idx_cfg.tree_levels;
    }

    // Apply nprobe if specified
    if let Some(nprobe) = idx_cfg.nprobe {
        config.nprobe = nprobe;
    }
    if let Some(routing) = idx_cfg.ivf_routing {
        config.ivf_routing = routing;
    }

    apply_soar_to_dense_vector(config, idx_cfg)?;
    Ok(())
}

const MAX_SCANN_TREE_LEVELS: u8 = 3;
const MAX_SCANN_LEAVES: usize = 30_000_000;

fn validate_scann_index_options(
    field_kind: &str,
    is_scann: bool,
    config: &IndexConfig,
) -> Result<()> {
    if !is_scann {
        if config.tree_levels.is_some() {
            return Err(Error::Schema(format!(
                "'tree_levels' is only valid for a ScaNN {field_kind} index"
            )));
        }
        return Ok(());
    }

    if config.ivf_routing.is_some() {
        return Err(Error::Schema(format!(
            "'routing' is not configurable for ScaNN {field_kind} indexes; ScaNN owns its hierarchical routing"
        )));
    }

    if let Some(tree_levels) = config.tree_levels
        && !(1..=MAX_SCANN_TREE_LEVELS).contains(&tree_levels)
    {
        return Err(Error::Schema(format!(
            "ScaNN tree_levels must be in 1..={MAX_SCANN_TREE_LEVELS}, got {tree_levels}"
        )));
    }
    if let Some(num_clusters) = config.num_clusters {
        if num_clusters < 2 {
            return Err(Error::Schema(
                "ScaNN num_clusters (terminal leaf count) must be at least 2".to_string(),
            ));
        }
        if num_clusters > MAX_SCANN_LEAVES {
            return Err(Error::Schema(format!(
                "ScaNN num_clusters cannot exceed {MAX_SCANN_LEAVES}, got {num_clusters}"
            )));
        }
        let nprobe = config.nprobe.unwrap_or(64);
        if nprobe > num_clusters {
            return Err(Error::Schema(format!(
                "ScaNN nprobe ({nprobe}) cannot exceed explicit num_clusters ({num_clusters})"
            )));
        }
    }
    if config.nprobe == Some(0) {
        return Err(Error::Schema("ScaNN nprobe must be positive".to_string()));
    }
    Ok(())
}

/// Apply SOAR spilling if specified (IVF-based indexes only)
fn apply_soar_to_dense_vector(config: &mut DenseVectorConfig, idx_cfg: IndexConfig) -> Result<()> {
    match idx_cfg.soar {
        SoarDirective::Unspecified => {
            config.soar = config
                .supports_soar()
                .then(crate::structures::SoarConfig::default);
        }
        SoarDirective::Disabled => {
            config.soar = None;
        }
        SoarDirective::Enabled(soar) => {
            if config.supports_soar() {
                config.soar = Some(soar);
            } else {
                config.soar = None;
                return Err(Error::Schema(format!(
                    "'soar' requires the IVF-TQ index and is not implemented for {:?}",
                    config.index_type
                )));
            }
        }
    }
    Ok(())
}

/// Parse sparse_vector_config - only index_size (positional)
/// Example: <u16> or <u32>
fn parse_sparse_vector_config(pair: pest::iterators::Pair<Rule>) -> SparseVectorConfig {
    let mut index_size = IndexSize::default();

    // Parse positional index_size_spec
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::index_size_spec {
            index_size = match inner.as_str() {
                "u16" => IndexSize::U16,
                "u32" => IndexSize::U32,
                _ => IndexSize::default(),
            };
        }
    }

    SparseVectorConfig {
        index_size,
        ..SparseVectorConfig::default()
    }
}

/// Apply index configuration from indexed<...> to SparseVectorConfig
fn apply_index_config_to_sparse_vector(config: &mut SparseVectorConfig, idx_cfg: IndexConfig) {
    if let Some(f) = idx_cfg.sparse_format {
        config.format = f;
    }
    if let Some(q) = idx_cfg.quantization {
        config.weight_quantization = q;
    }
    if let Some(t) = idx_cfg.weight_threshold {
        config.weight_threshold = t;
    }
    if let Some(bs) = idx_cfg.block_size {
        let adjusted = bs.next_power_of_two();
        if adjusted != bs {
            log::warn!(
                "block_size {} adjusted to next power of two: {}",
                bs,
                adjusted
            );
        }
        config.block_size = adjusted;
    }
    if let Some(bs) = idx_cfg.bmp_block_size {
        let adjusted = bs.next_power_of_two().clamp(1, 256);
        if adjusted != bs {
            log::warn!(
                "bmp_block_size {} adjusted to power of two in 1..=256: {}",
                bs,
                adjusted
            );
        }
        config.bmp_block_size = adjusted;
    }
    if let Some(bits) = idx_cfg.bmp_grid_bits {
        if bits == 2 || bits == 4 {
            config.bmp_grid_bits = bits;
        } else {
            log::warn!(
                "bmp_grid_bits {} unsupported (must be 2 or 4), using {}",
                bits,
                SparseVectorConfig::DEFAULT_BMP_GRID_BITS,
            );
            config.bmp_grid_bits = SparseVectorConfig::DEFAULT_BMP_GRID_BITS;
        }
    }
    if let Some(p) = idx_cfg.pruning {
        let clamped = p.clamp(0.0, 1.0);
        if (clamped - p).abs() > f32::EPSILON {
            log::warn!(
                "pruning {} clamped to valid range [0.0, 1.0]: {}",
                p,
                clamped
            );
        }
        config.pruning = Some(clamped);
    }
    if let Some(mt) = idx_cfg.min_terms {
        config.min_terms = mt;
    }
    if let Some(dm) = idx_cfg.doc_mass {
        let clamped = dm.clamp(0.0, 1.0);
        if (clamped - dm).abs() > f32::EPSILON {
            log::warn!(
                "doc_mass {} clamped to valid range [0.0, 1.0]: {}",
                dm,
                clamped
            );
        }
        config.doc_mass = Some(clamped);
    }
    if let Some(d) = idx_cfg.dims {
        config.dims = Some(d);
    }
    if let Some(mw) = idx_cfg.max_weight {
        config.max_weight = Some(mw);
    }
    // Apply query-time configuration if present
    if idx_cfg.query_tokenizer.is_some()
        || idx_cfg.query_weighting.is_some()
        || idx_cfg.query_weight_threshold.is_some()
        || idx_cfg.query_max_dims.is_some()
        || idx_cfg.query_pruning.is_some()
        || idx_cfg.query_min_query_dims.is_some()
        || idx_cfg.query_lsp_gamma.is_some()
    {
        let query_config = config
            .query_config
            .get_or_insert(SparseQueryConfig::default());
        if let Some(tokenizer) = idx_cfg.query_tokenizer {
            query_config.tokenizer = Some(tokenizer);
        }
        if let Some(weighting) = idx_cfg.query_weighting {
            query_config.weighting = weighting;
        }
        if let Some(t) = idx_cfg.query_weight_threshold {
            query_config.weight_threshold = t;
        }
        if let Some(d) = idx_cfg.query_max_dims {
            query_config.max_query_dims = Some(d);
        }
        if let Some(p) = idx_cfg.query_pruning {
            query_config.pruning = Some(p);
        }
        if let Some(m) = idx_cfg.query_min_query_dims {
            query_config.min_query_dims = m;
        }
        if let Some(gamma) = idx_cfg.query_lsp_gamma {
            query_config.lsp_gamma = Some(gamma);
        }
    }
}

/// Parse dense_vector_config - dims and optional quantization type
/// All index-related params are in indexed<...> attribute
fn parse_dense_vector_config(pair: pest::iterators::Pair<Rule>) -> DenseVectorConfig {
    let mut dim: usize = 0;
    let mut quantization = DenseVectorQuantization::F32;

    // Navigate to dense_vector_params
    for params in pair.into_inner() {
        if params.as_rule() == Rule::dense_vector_params {
            for inner in params.into_inner() {
                match inner.as_rule() {
                    Rule::dense_vector_keyword_params => {
                        for kwarg in inner.into_inner() {
                            match kwarg.as_rule() {
                                Rule::dims_kwarg => {
                                    if let Some(d) = kwarg.into_inner().next() {
                                        dim = d.as_str().parse().unwrap_or(0);
                                    }
                                }
                                Rule::quant_type_spec => {
                                    quantization = parse_quant_type(kwarg.as_str());
                                }
                                _ => {}
                            }
                        }
                    }
                    Rule::dense_vector_positional_params => {
                        for item in inner.into_inner() {
                            match item.as_rule() {
                                Rule::dimension_spec => {
                                    dim = item.as_str().parse().unwrap_or(0);
                                }
                                Rule::quant_type_spec => {
                                    quantization = parse_quant_type(item.as_str());
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    DenseVectorConfig::new(dim).with_quantization(quantization)
}

fn parse_quant_type(s: &str) -> DenseVectorQuantization {
    match s.trim() {
        "f16" => DenseVectorQuantization::F16,
        "uint8" | "u8" => DenseVectorQuantization::UInt8,
        _ => DenseVectorQuantization::F32,
    }
}

/// Parse default_fields definition
fn parse_default_fields_def(pair: pest::iterators::Pair<Rule>) -> Vec<String> {
    pair.into_inner().map(|p| p.as_str().to_string()).collect()
}

/// Parse a query router definition
fn parse_query_router_def(pair: pest::iterators::Pair<Rule>) -> Result<QueryRouterRule> {
    let mut pattern = String::new();
    let mut substitution = String::new();
    let mut target_field = String::new();
    let mut mode = RoutingMode::Additional;

    for prop in pair.into_inner() {
        if prop.as_rule() != Rule::query_router_prop {
            continue;
        }

        for inner in prop.into_inner() {
            match inner.as_rule() {
                Rule::query_router_pattern => {
                    if let Some(regex_str) = inner.into_inner().next() {
                        pattern = parse_string_value(regex_str);
                    }
                }
                Rule::query_router_substitution => {
                    if let Some(quoted) = inner.into_inner().next() {
                        substitution = parse_string_value(quoted);
                    }
                }
                Rule::query_router_target => {
                    if let Some(ident) = inner.into_inner().next() {
                        target_field = ident.as_str().to_string();
                    }
                }
                Rule::query_router_mode => {
                    if let Some(mode_val) = inner.into_inner().next() {
                        mode = match mode_val.as_str() {
                            "exclusive" => RoutingMode::Exclusive,
                            "additional" => RoutingMode::Additional,
                            _ => RoutingMode::Additional,
                        };
                    }
                }
                _ => {}
            }
        }
    }

    if pattern.is_empty() {
        return Err(Error::Schema("query_router missing 'pattern'".to_string()));
    }
    if substitution.is_empty() {
        return Err(Error::Schema(
            "query_router missing 'substitution'".to_string(),
        ));
    }
    if target_field.is_empty() {
        return Err(Error::Schema(
            "query_router missing 'target_field'".to_string(),
        ));
    }

    Ok(QueryRouterRule {
        pattern,
        substitution,
        target_field,
        mode,
    })
}

/// Parse a string value from quoted_string, raw_string, or regex_string
fn parse_string_value(pair: pest::iterators::Pair<Rule>) -> String {
    let s = pair.as_str();
    match pair.as_rule() {
        Rule::regex_string => {
            // regex_string contains either raw_string or quoted_string
            if let Some(inner) = pair.into_inner().next() {
                parse_string_value(inner)
            } else {
                s.to_string()
            }
        }
        Rule::raw_string => {
            // r"..." - strip r" prefix and " suffix
            s[2..s.len() - 1].to_string()
        }
        Rule::quoted_string => {
            // "..." - strip quotes and handle escapes
            let inner = &s[1..s.len() - 1];
            // Simple escape handling
            inner
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
        }
        _ => s.to_string(),
    }
}

/// Parse an index definition from pest pair
fn parse_index_def(pair: pest::iterators::Pair<Rule>) -> Result<IndexDef> {
    let mut inner = pair.into_inner();

    let name = inner
        .next()
        .ok_or_else(|| Error::Schema("Missing index name".to_string()))?
        .as_str()
        .to_string();

    let mut fields = Vec::new();
    let mut default_fields = Vec::new();
    let mut query_routers = Vec::new();
    let mut reorder_on_merge = false;

    for item in inner {
        match item.as_rule() {
            Rule::field_def => {
                fields.push(parse_field_def(item)?);
            }
            Rule::default_fields_def => {
                default_fields = parse_default_fields_def(item);
            }
            Rule::query_router_def => {
                query_routers.push(parse_query_router_def(item)?);
            }
            Rule::reorder_on_merge_def => {
                let value = item
                    .into_inner()
                    .next()
                    .map(|b| b.as_str() == "true")
                    .unwrap_or(false);
                reorder_on_merge = value;
            }
            _ => {}
        }
    }

    validate_tokenizer_specs(&name, &fields)?;

    // Validate primary key constraints
    let primary_fields: Vec<&FieldDef> = fields.iter().filter(|f| f.primary).collect();
    if primary_fields.len() > 1 {
        return Err(Error::Schema(format!(
            "Index '{}' has {} primary key fields, but at most one is allowed",
            name,
            primary_fields.len()
        )));
    }
    if let Some(pk) = primary_fields.first() {
        if pk.field_type != FieldType::Text {
            return Err(Error::Schema(format!(
                "Primary key field '{}' must be of type text, got {:?}",
                pk.name, pk.field_type
            )));
        }
        if pk.multi {
            return Err(Error::Schema(format!(
                "Primary key field '{}' cannot be multi-valued",
                pk.name
            )));
        }
    }

    Ok(IndexDef {
        name,
        fields,
        default_fields,
        query_routers,
        reorder_on_merge,
    })
}

/// Fail loudly on tokenizer specs that would otherwise degrade silently:
/// unknown tokenizer names (the builder would fall back to plain lowercasing)
/// and dynamic stemmers whose hint field does not exist or is not text.
fn validate_tokenizer_specs(index_name: &str, fields: &[FieldDef]) -> Result<()> {
    use crate::tokenizer::{TokenizerRegistry, TokenizerSpec};
    let mut registry: Option<TokenizerRegistry> = None;
    for field in fields {
        let Some(raw) = field.tokenizer.as_deref() else {
            continue;
        };
        let spec = TokenizerSpec::parse(raw).map_err(|e| {
            Error::Schema(format!("Index '{index_name}', field '{}': {e}", field.name))
        })?;
        match spec {
            TokenizerSpec::Named(tokenizer) => {
                let registry = registry.get_or_insert_with(TokenizerRegistry::new);
                if !registry.contains(&tokenizer) {
                    return Err(Error::Schema(format!(
                        "Index '{index_name}', field '{}': unknown tokenizer '{tokenizer}'",
                        field.name
                    )));
                }
            }
            TokenizerSpec::DynamicStem { by: None, .. } => {}
            TokenizerSpec::DynamicStem { by: Some(by), .. } => {
                match fields.iter().find(|f| f.name == by) {
                    None => {
                        return Err(Error::Schema(format!(
                            "Index '{index_name}', field '{}': tokenizer hint field '{by}' does not exist",
                            field.name
                        )));
                    }
                    Some(hint) if hint.field_type != FieldType::Text => {
                        return Err(Error::Schema(format!(
                            "Index '{index_name}', field '{}': tokenizer hint field '{by}' must be a text field, got {:?}",
                            field.name, hint.field_type
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    Ok(())
}

/// Parse SDL from a string
pub fn parse_sdl(input: &str) -> Result<Vec<IndexDef>> {
    let pairs = SdlParser::parse(Rule::file, input)
        .map_err(|e| Error::Schema(format!("Parse error: {}", e)))?;

    let mut indexes = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::index_def {
                    indexes.push(parse_index_def(inner)?);
                }
            }
        }
    }

    Ok(indexes)
}

/// Parse SDL and return a single index definition
pub fn parse_single_index(input: &str) -> Result<IndexDef> {
    let indexes = parse_sdl(input)?;

    if indexes.is_empty() {
        return Err(Error::Schema("No index definition found".to_string()));
    }

    if indexes.len() > 1 {
        return Err(Error::Schema(
            "Multiple index definitions found, expected one".to_string(),
        ));
    }

    Ok(indexes.into_iter().next().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_schema() {
        let sdl = r#"
            index articles {
                field title: text [indexed, stored]
                field body: text [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);

        let index = &indexes[0];
        assert_eq!(index.name, "articles");
        assert_eq!(index.fields.len(), 2);

        assert_eq!(index.fields[0].name, "title");
        assert!(matches!(index.fields[0].field_type, FieldType::Text));
        assert!(index.fields[0].indexed);
        assert!(index.fields[0].stored);

        assert_eq!(index.fields[1].name, "body");
        assert!(matches!(index.fields[1].field_type, FieldType::Text));
        assert!(index.fields[1].indexed);
        assert!(!index.fields[1].stored);
    }

    #[test]
    fn test_parse_all_field_types() {
        let sdl = r#"
            index test {
                field text_field: text [indexed, stored]
                field u64_field: u64 [indexed, stored]
                field i64_field: i64 [indexed, stored]
                field f64_field: f64 [indexed, stored]
                field bytes_field: bytes [stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert!(matches!(index.fields[0].field_type, FieldType::Text));
        assert!(matches!(index.fields[1].field_type, FieldType::U64));
        assert!(matches!(index.fields[2].field_type, FieldType::I64));
        assert!(matches!(index.fields[3].field_type, FieldType::F64));
        assert!(matches!(index.fields[4].field_type, FieldType::Bytes));
    }

    #[test]
    fn test_parse_with_comments() {
        let sdl = r#"
            # This is a comment
            index articles {
                # Title field
                field title: text [indexed, stored]
                field body: text [indexed] # inline comment not supported yet
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes[0].fields.len(), 2);
    }

    #[test]
    fn test_parse_type_aliases() {
        let sdl = r#"
            index test {
                field a: string [indexed]
                field b: int [indexed]
                field c: uint [indexed]
                field d: float [indexed]
                field e: binary [stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert!(matches!(index.fields[0].field_type, FieldType::Text));
        assert!(matches!(index.fields[1].field_type, FieldType::I64));
        assert!(matches!(index.fields[2].field_type, FieldType::U64));
        assert!(matches!(index.fields[3].field_type, FieldType::F64));
        assert!(matches!(index.fields[4].field_type, FieldType::Bytes));
    }

    #[test]
    fn test_to_schema() {
        let sdl = r#"
            index articles {
                field title: text [indexed, stored]
                field views: u64 [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let schema = indexes[0].to_schema();

        assert!(schema.get_field("title").is_some());
        assert!(schema.get_field("views").is_some());
        assert!(schema.get_field("nonexistent").is_none());
    }

    #[test]
    fn test_default_attributes() {
        let sdl = r#"
            index test {
                field title: text
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let field = &indexes[0].fields[0];

        // Default should be indexed and stored
        assert!(field.indexed);
        assert!(field.stored);
    }

    #[test]
    fn chunked_text_field_parses_and_implies_multi() {
        let sdl = r#"
            index documents {
                field languages: text<raw_ci> [fast]
                field content: text<stem(by: languages, default: simple)> [indexed<chunked, token_position>]
                field notes: text<simple> [indexed<chunked>, stored]
            }
        "#;
        let index = parse_single_index(sdl).unwrap();
        let content = &index.fields[1];
        assert!(content.chunked);
        assert!(content.multi, "chunked implies multi-valued storage");
        assert_eq!(
            content.positions,
            Some(crate::dsl::PositionMode::TokenPosition)
        );
        let notes = &index.fields[2];
        assert!(notes.chunked && notes.stored && notes.positions.is_none());

        let schema = index.to_schema();
        let entry = schema
            .get_field_entry(schema.get_field("content").unwrap())
            .unwrap();
        assert!(entry.chunked && entry.multi);
        assert!(
            !schema
                .get_field_entry(schema.get_field("languages").unwrap())
                .unwrap()
                .chunked
        );
    }

    #[test]
    fn chunked_rejects_non_text_and_ordinal_position_modes() {
        let non_text = parse_sdl("index i { field n: u64 [indexed<chunked>] }").unwrap_err();
        assert!(
            non_text.to_string().contains("requires a text field"),
            "{non_text}"
        );

        for mode in ["positions", "ordinal"] {
            let sdl = format!("index i {{ field c: text<simple> [indexed<chunked, {mode}>] }}");
            let error = parse_sdl(&sdl).unwrap_err();
            assert!(
                error.to_string().contains("token_position"),
                "{mode}: {error}"
            );
        }
    }

    #[test]
    fn test_multiple_indexes() {
        let sdl = r#"
            index articles {
                field title: text [indexed, stored]
            }

            index users {
                field name: text [indexed, stored]
                field email: text [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 2);
        assert_eq!(indexes[0].name, "articles");
        assert_eq!(indexes[1].name, "users");
    }

    #[test]
    fn test_tokenizer_spec() {
        let sdl = r#"
            index articles {
                field title: text<en_stem> [indexed, stored]
                field body: text<simple> [indexed]
                field author: text [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert_eq!(index.fields[0].name, "title");
        assert_eq!(index.fields[0].tokenizer, Some("en_stem".to_string()));

        assert_eq!(index.fields[1].name, "body");
        assert_eq!(index.fields[1].tokenizer, Some("simple".to_string()));

        assert_eq!(index.fields[2].name, "author");
        assert_eq!(index.fields[2].tokenizer, None); // No tokenizer specified
    }

    #[test]
    fn test_dynamic_tokenizer_spec() {
        let sdl = r#"
            index documents {
                field languages: text<raw_ci> [fast]
                field content: text<stem(by: languages, default: simple)> [indexed<token_position>]
                field title: text<stem(by:languages,default:english)> [indexed]
                field embedding: dense_vector<768> [indexed]
                field hash: binary_dense_vector<64> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];
        assert_eq!(
            index.fields[1].tokenizer,
            Some("stem(by: languages, default: simple)".to_string())
        );
        assert_eq!(
            index.fields[1].positions,
            Some(super::super::schema::PositionMode::TokenPosition)
        );
        // Canonical rendering normalises spacing and language names.
        assert_eq!(
            index.fields[2].tokenizer,
            Some("stem(by: languages, default: en)".to_string())
        );
        // Vector `<N>` configs are unaffected by the extended tokenizer grammar.
        assert_eq!(
            index.fields[3].dense_vector_config.as_ref().unwrap().dim,
            768
        );
        assert_eq!(
            index.fields[4]
                .binary_dense_vector_config
                .as_ref()
                .unwrap()
                .dim,
            64
        );

        let schema = index.to_schema();
        let content = schema.get_field("content").unwrap();
        let languages = schema.get_field("languages").unwrap();
        assert_eq!(schema.tokenizer_hint_field(content), Some(languages));
        assert_eq!(schema.tokenizer_hint_field(languages), None);
        let entry = schema.get_field_entry(content).unwrap();
        assert_eq!(
            entry.tokenizer_spec().unwrap().hint_field(),
            Some("languages")
        );
    }

    #[test]
    fn test_tokenizer_specs_fail_loud() {
        let missing_hint_field = r#"
            index documents {
                field content: text<stem(by: languages, default: simple)> [indexed]
            }
        "#;
        let err = parse_sdl(missing_hint_field).unwrap_err().to_string();
        assert!(
            err.contains("hint field 'languages' does not exist"),
            "{err}"
        );

        let numeric_hint_field = r#"
            index documents {
                field languages: u64 [fast]
                field content: text<stem(by: languages)> [indexed]
            }
        "#;
        let err = parse_sdl(numeric_hint_field).unwrap_err().to_string();
        assert!(err.contains("must be a text field"), "{err}");

        let unknown_default = r#"
            index documents {
                field languages: text [fast]
                field content: text<stem(by: languages, default: klingon)> [indexed]
            }
        "#;
        let err = parse_sdl(unknown_default).unwrap_err().to_string();
        assert!(err.contains("unknown default language 'klingon'"), "{err}");

        let unknown_tokenizer = r#"
            index documents {
                field content: text<klingon_stem> [indexed]
            }
        "#;
        let err = parse_sdl(unknown_tokenizer).unwrap_err().to_string();
        assert!(err.contains("unknown tokenizer 'klingon_stem'"), "{err}");
    }

    #[test]
    fn test_tokenizer_in_schema() {
        let sdl = r#"
            index articles {
                field title: text<german> [indexed, stored]
                field body: text<en_stem> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let schema = indexes[0].to_schema();

        let title_field = schema.get_field("title").unwrap();
        let title_entry = schema.get_field_entry(title_field).unwrap();
        assert_eq!(title_entry.tokenizer, Some("german".to_string()));

        let body_field = schema.get_field("body").unwrap();
        let body_entry = schema.get_field_entry(body_field).unwrap();
        assert_eq!(body_entry.tokenizer, Some("en_stem".to_string()));
    }

    #[test]
    fn test_query_router_basic() {
        let sdl = r#"
            index documents {
                field title: text [indexed, stored]
                field uri: text [indexed, stored]

                query_router {
                    pattern: "10\\.\\d{4,}/[^\\s]+"
                    substitution: "doi://{0}"
                    target_field: uris
                    mode: exclusive
                }
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert_eq!(index.query_routers.len(), 1);
        let router = &index.query_routers[0];
        assert_eq!(router.pattern, r"10\.\d{4,}/[^\s]+");
        assert_eq!(router.substitution, "doi://{0}");
        assert_eq!(router.target_field, "uris");
        assert_eq!(router.mode, RoutingMode::Exclusive);
    }

    #[test]
    fn test_query_router_raw_string() {
        let sdl = r#"
            index documents {
                field uris: text [indexed, stored]

                query_router {
                    pattern: r"^pmid:(\d+)$"
                    substitution: "pubmed://{1}"
                    target_field: uris
                    mode: additional
                }
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let router = &indexes[0].query_routers[0];

        assert_eq!(router.pattern, r"^pmid:(\d+)$");
        assert_eq!(router.substitution, "pubmed://{1}");
        assert_eq!(router.mode, RoutingMode::Additional);
    }

    #[test]
    fn test_multiple_query_routers() {
        let sdl = r#"
            index documents {
                field uris: text [indexed, stored]

                query_router {
                    pattern: r"^doi:(10\.\d{4,}/[^\s]+)$"
                    substitution: "doi://{1}"
                    target_field: uris
                    mode: exclusive
                }

                query_router {
                    pattern: r"^pmid:(\d+)$"
                    substitution: "pubmed://{1}"
                    target_field: uris
                    mode: exclusive
                }

                query_router {
                    pattern: r"^arxiv:(\d+\.\d+)$"
                    substitution: "arxiv://{1}"
                    target_field: uris
                    mode: additional
                }
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes[0].query_routers.len(), 3);
    }

    #[test]
    fn test_query_router_default_mode() {
        let sdl = r#"
            index documents {
                field uris: text [indexed, stored]

                query_router {
                    pattern: r"test"
                    substitution: "{0}"
                    target_field: uris
                }
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        // Default mode should be Additional
        assert_eq!(indexes[0].query_routers[0].mode, RoutingMode::Additional);
    }

    #[test]
    fn test_multi_attribute() {
        let sdl = r#"
            index documents {
                field uris: text [indexed, stored<multi>]
                field title: text [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);

        let fields = &indexes[0].fields;
        assert_eq!(fields.len(), 2);

        // uris should have multi=true
        assert_eq!(fields[0].name, "uris");
        assert!(fields[0].multi, "uris field should have multi=true");

        // title should have multi=false
        assert_eq!(fields[1].name, "title");
        assert!(!fields[1].multi, "title field should have multi=false");

        // Verify schema conversion preserves multi attribute
        let schema = indexes[0].to_schema();
        let uris_field = schema.get_field("uris").unwrap();
        let title_field = schema.get_field("title").unwrap();

        assert!(schema.get_field_entry(uris_field).unwrap().multi);
        assert!(!schema.get_field_entry(title_field).unwrap().multi);
    }

    #[test]
    fn test_sparse_vector_field() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].fields.len(), 1);
        assert_eq!(indexes[0].fields[0].name, "embedding");
        assert_eq!(indexes[0].fields[0].field_type, FieldType::SparseVector);
        assert!(indexes[0].fields[0].sparse_vector_config.is_none());
    }

    #[test]
    fn test_sparse_vector_with_config() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<quantization: uint8>, stored]
                field dense: sparse_vector<u32> [indexed<quantization: float32>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes[0].fields.len(), 2);

        // First field: u16 indices, uint8 quantization
        let f1 = &indexes[0].fields[0];
        assert_eq!(f1.name, "embedding");
        let config1 = f1.sparse_vector_config.as_ref().unwrap();
        assert_eq!(config1.index_size, IndexSize::U16);
        assert_eq!(config1.weight_quantization, WeightQuantization::UInt8);

        // Second field: u32 indices, float32 quantization
        let f2 = &indexes[0].fields[1];
        assert_eq!(f2.name, "dense");
        let config2 = f2.sparse_vector_config.as_ref().unwrap();
        assert_eq!(config2.index_size, IndexSize::U32);
        assert_eq!(config2.weight_quantization, WeightQuantization::Float32);
    }

    #[test]
    fn test_sparse_vector_bmp_block_size() {
        let sdl = r#"
            index documents {
                field emb: sparse_vector<u32> [indexed<format: bmp, dims: 105879, bmp_block_size: 256>]
                field emb2: sparse_vector<u32> [indexed<format: bmp, dims: 30522>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config1 = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config1.format, SparseFormat::Bmp);
        assert_eq!(config1.bmp_block_size, 256);

        // Default block size stays 32.
        let config2 = indexes[0].fields[1].sparse_vector_config.as_ref().unwrap();
        assert_eq!(
            config2.bmp_block_size,
            SparseVectorConfig::DEFAULT_BMP_BLOCK_SIZE
        );
    }

    /// Regression: `bmp_grid_bits` parsed but was never applied to the field
    /// config — SDL said 2, segments silently built 4-bit grids.
    #[test]
    fn test_sparse_vector_bmp_grid_bits() {
        let sdl = r#"
            index documents {
                field emb: sparse_vector<u32> [indexed<format: bmp, dims: 105879, bmp_block_size: 256, bmp_grid_bits: 2>]
                field emb2: sparse_vector<u32> [indexed<format: bmp, dims: 30522>]
                field emb3: sparse_vector<u32> [indexed<format: bmp, dims: 30522, bmp_grid_bits: 3>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config1 = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config1.bmp_grid_bits, 2);
        // Default stays 4
        let config2 = indexes[0].fields[1].sparse_vector_config.as_ref().unwrap();
        assert_eq!(
            config2.bmp_grid_bits,
            SparseVectorConfig::DEFAULT_BMP_GRID_BITS
        );
        // Unsupported width falls back to 4 with a warning
        let config3 = indexes[0].fields[2].sparse_vector_config.as_ref().unwrap();
        assert_eq!(
            config3.bmp_grid_bits,
            SparseVectorConfig::DEFAULT_BMP_GRID_BITS
        );
    }

    #[test]
    fn test_sparse_vector_with_weight_threshold() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<quantization: uint8, weight_threshold: 0.1>, stored]
                field embedding2: sparse_vector<u32> [indexed<quantization: float16, weight_threshold: 0.05>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes[0].fields.len(), 2);

        // First field: u16 indices, uint8 quantization, threshold 0.1
        let f1 = &indexes[0].fields[0];
        assert_eq!(f1.name, "embedding");
        let config1 = f1.sparse_vector_config.as_ref().unwrap();
        assert_eq!(config1.index_size, IndexSize::U16);
        assert_eq!(config1.weight_quantization, WeightQuantization::UInt8);
        assert!((config1.weight_threshold - 0.1).abs() < 0.001);

        // Second field: u32 indices, float16 quantization, threshold 0.05
        let f2 = &indexes[0].fields[1];
        assert_eq!(f2.name, "embedding2");
        let config2 = f2.sparse_vector_config.as_ref().unwrap();
        assert_eq!(config2.index_size, IndexSize::U32);
        assert_eq!(config2.weight_quantization, WeightQuantization::Float16);
        assert!((config2.weight_threshold - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_sparse_vector_with_pruning() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector [indexed<quantization: uint8, pruning: 0.1>, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let f = &indexes[0].fields[0];
        assert_eq!(f.name, "embedding");
        let config = f.sparse_vector_config.as_ref().unwrap();
        assert_eq!(config.weight_quantization, WeightQuantization::UInt8);
        assert_eq!(config.pruning, Some(0.1));
    }

    #[test]
    fn test_sparse_vector_with_doc_mass() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector [indexed<quantization: uint8, doc_mass: 0.9>, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config.doc_mass, Some(0.9));

        // Not specified → off
        let sdl = r#"
            index documents {
                field embedding: sparse_vector [indexed<quantization: uint8>]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config.doc_mass, None);
    }

    #[test]
    fn test_dense_vector_field() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].fields.len(), 1);

        let f = &indexes[0].fields[0];
        assert_eq!(f.name, "embedding");
        assert_eq!(f.field_type, FieldType::DenseVector);

        let config = f.dense_vector_config.as_ref().unwrap();
        assert_eq!(config.dim, 768);
    }

    #[test]
    fn test_dense_vector_alias() {
        let sdl = r#"
            index documents {
                field embedding: vector<1536> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes[0].fields[0].field_type, FieldType::DenseVector);
        assert_eq!(
            indexes[0].fields[0]
                .dense_vector_config
                .as_ref()
                .unwrap()
                .dim,
            1536
        );
    }

    #[test]
    fn test_dense_vector_with_num_clusters() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<ivf_tq, num_clusters: 256>, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);

        let f = &indexes[0].fields[0];
        assert_eq!(f.name, "embedding");
        assert_eq!(f.field_type, FieldType::DenseVector);

        let config = f.dense_vector_config.as_ref().unwrap();
        assert_eq!(config.dim, 768);
        assert_eq!(config.num_clusters, Some(256));
        assert_eq!(config.nprobe, 64); // billion-scale default
    }

    #[test]
    fn scann_float_and_binary_parse_billion_scale_settings() {
        let indexes = parse_sdl(
            r#"
            index billion_vectors {
                field embedding: dense_vector<1024, f16> [indexed<scann, num_clusters: 10000000, tree_levels: 2, nprobe: 1024>]
                field hash: binary_dense_vector<1024> [indexed<scann, num_clusters: 10000000, tree_levels: 3, nprobe: 2048>]
            }
            "#,
        )
        .unwrap();

        let dense = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();
        assert_eq!(
            dense.index_type,
            super::super::schema::VectorIndexType::Scann
        );
        assert_eq!(dense.num_clusters, Some(10_000_000));
        assert_eq!(dense.tree_levels, Some(2));
        assert_eq!(dense.nprobe, 1024);

        let binary = indexes[0].fields[1]
            .binary_dense_vector_config
            .as_ref()
            .unwrap();
        assert_eq!(
            binary.index_type,
            super::super::schema::BinaryIndexType::Scann
        );
        assert_eq!(binary.num_clusters, Some(10_000_000));
        assert_eq!(binary.tree_levels, Some(3));
        assert_eq!(binary.nprobe, 2048);
    }

    #[test]
    fn target_vectors_parses_for_float_and_binary_indexes() {
        let indexes = parse_sdl(
            r#"
            index streaming_vectors {
                field embedding: dense_vector<1024, f16> [indexed<scann, num_clusters: 1000000, target_vectors: 1000000000>]
                field hash: binary_dense_vector<2560> [indexed<ivf, target_vectors: 1000000000>]
            }
            "#,
        )
        .unwrap();

        assert_eq!(
            indexes[0].fields[0]
                .dense_vector_config
                .as_ref()
                .unwrap()
                .target_vectors,
            Some(1_000_000_000)
        );
        assert_eq!(
            indexes[0].fields[0]
                .dense_vector_config
                .as_ref()
                .unwrap()
                .num_clusters,
            Some(1_000_000),
            "target_vectors may be persisted alongside an explicit, overriding topology"
        );
        assert_eq!(
            indexes[0].fields[1]
                .binary_dense_vector_config
                .as_ref()
                .unwrap()
                .target_vectors,
            Some(1_000_000_000)
        );

        let error = parse_sdl(
            "index invalid { field hash: binary_dense_vector<256> [indexed<ivf, target_vectors: 0>] }",
        )
        .expect_err("zero target must fail");
        assert!(error.to_string().contains("greater than zero"), "{error}");

        let error = parse_sdl(
            "index invalid { field embedding: dense_vector<256> [indexed<tq, target_vectors: 1000000>] }",
        )
        .expect_err("training-free topology hint must fail");
        assert!(error.to_string().contains("automatic topology"), "{error}");

        for sdl in [
            "index invalid { field embedding: dense_vector<256> [indexed<flat, target_vectors: 1000000>] }",
            "index invalid { field hash: binary_dense_vector<256> [indexed<flat, target_vectors: 1000000>] }",
        ] {
            let error = parse_sdl(sdl).expect_err("flat topology hint must fail");
            assert!(error.to_string().contains("automatic topology"), "{error}");
        }

        let error = parse_sdl(
            "index invalid { field hash: binary_dense_vector<256> [indexed<ivf, target_vectors: 18446744073709551616>] }",
        )
        .expect_err("u64 overflow must fail");
        assert!(error.to_string().contains("unsigned 64-bit"), "{error}");
    }

    #[test]
    fn scann_rejects_invalid_geometry_and_algorithm_specific_options() {
        for (fragment, expected) in [
            ("scann, tree_levels: 0", "tree_levels"),
            ("scann, tree_levels: 4", "tree_levels"),
            ("scann, num_clusters: 30000001", "num_clusters"),
            ("scann, num_clusters: 1", "at least 2"),
            ("scann, routing: flat", "not configurable for ScaNN"),
            (
                "scann, num_clusters: 32, nprobe: 33",
                "cannot exceed explicit num_clusters",
            ),
            ("ivf_tq, tree_levels: 2", "only valid for a ScaNN"),
        ] {
            let sdl = format!(
                "index invalid {{ field embedding: dense_vector<128> [indexed<{fragment}>] }}"
            );
            let error = parse_sdl(&sdl).expect_err(fragment);
            assert!(error.to_string().contains(expected), "{fragment}: {error}");
        }
    }

    #[test]
    fn binary_scann_accepts_selective_spilling_but_binary_ivf_rejects_it() {
        let indexes = parse_sdl(
            "index valid { field hash: binary_dense_vector<256> [indexed<scann, soar: selective>] }",
        )
        .unwrap();
        let soar = indexes[0].fields[0]
            .binary_dense_vector_config
            .as_ref()
            .unwrap()
            .soar
            .as_ref()
            .expect("binary ScaNN should retain explicit spilling");
        assert_eq!(soar.calibration_target(), Some(0.30));

        let error = parse_sdl(
            "index invalid { field hash: binary_dense_vector<256> [indexed<ivf, soar: selective>] }",
        )
        .expect_err("binary IVF spilling must fail loudly");
        assert!(error.to_string().contains("requires the ScaNN"), "{error}");
    }

    #[test]
    fn test_dense_vector_with_soar() {
        // Omission resolves to the selective one-secondary default.
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<ivf_tq>]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();
        let soar = config
            .soar
            .as_ref()
            .expect("omitted SOAR should enable selective spilling");
        assert_eq!(soar.num_secondary, 1);
        assert!(soar.selective);
        assert_eq!(soar.calibration_target(), Some(0.30));

        // The explicit selective preset resolves to the same policy.
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<ivf_tq, num_clusters: 256, soar: selective>, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        let soar = config.soar.as_ref().expect("soar should be enabled");
        assert_eq!(soar.num_secondary, 1);
        assert!(soar.selective);

        // aggressive is a compatibility alias for full one-secondary spilling
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<ivf_tq, soar: aggressive>]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();
        let soar = config.soar.as_ref().expect("soar should be enabled");
        assert_eq!(soar.num_secondary, 1);
        assert!(!soar.selective);

        // off keeps soar disabled
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<ivf_tq, soar: off>]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();
        assert!(config.soar.is_none());
    }

    #[test]
    fn omitted_soar_is_canonicalized_off_for_non_ivf_formats() {
        let sdl = r#"
            index documents {
                field tq: dense_vector<768> [indexed<tq>]
                field flat: dense_vector<768> [indexed<flat>]
                field scann: dense_vector<768> [indexed<scann>]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        for field in &indexes[0].fields {
            assert!(
                field
                    .dense_vector_config
                    .as_ref()
                    .expect("dense config")
                    .soar
                    .is_none(),
                "{} should not retain an ignored SOAR default",
                field.name,
            );
        }
    }

    #[test]
    fn float_scann_rejects_explicit_soar_until_secondary_assignments_exist() {
        let error = parse_sdl(
            "index invalid { field embedding: dense_vector<256> [indexed<scann, soar: selective>] }",
        )
        .unwrap_err();
        assert!(error.to_string().contains("not implemented"));
        assert!(error.to_string().contains("Scann") || error.to_string().contains("ScaNN"));
    }

    #[test]
    fn test_ivf_routing_modes_apply_to_float_and_binary_fields() {
        let indexes = parse_sdl(
            r#"
            index vectors {
                field embedding: dense_vector<768> [indexed<ivf_tq, routing: hnsw>]
                field hash: binary_dense_vector<512> [indexed<ivf, routing: two_level>]
            }
            "#,
        )
        .unwrap();
        let schema = indexes[0].to_schema();
        let embedding = schema.get_field("embedding").unwrap();
        let hash = schema.get_field("hash").unwrap();
        assert_eq!(
            schema
                .get_field_entry(embedding)
                .unwrap()
                .dense_vector_config
                .as_ref()
                .unwrap()
                .ivf_routing,
            super::super::schema::IvfRoutingMode::Hnsw
        );
        assert_eq!(
            schema
                .get_field_entry(hash)
                .unwrap()
                .binary_dense_vector_config
                .as_ref()
                .unwrap()
                .ivf_routing,
            super::super::schema::IvfRoutingMode::TwoLevel
        );
    }

    #[test]
    fn test_binary_dense_vector_with_ivf() {
        let sdl = r#"
            index documents {
                field hash: binary_dense_vector<512> [indexed<ivf, num_clusters: 128, nprobe: 16>, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0]
            .binary_dense_vector_config
            .as_ref()
            .unwrap();
        assert_eq!(config.dim, 512);
        assert_eq!(
            config.index_type,
            super::super::schema::BinaryIndexType::Ivf
        );
        assert_eq!(config.num_clusters, Some(128));
        assert_eq!(config.nprobe, 16);

        // Default targets the global IVF index; segments remain flat until
        // build_vector_index is requested.
        let sdl = r#"
            index documents {
                field hash: binary_dense_vector<512> [indexed]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0]
            .binary_dense_vector_config
            .as_ref()
            .unwrap();
        assert_eq!(
            config.index_type,
            super::super::schema::BinaryIndexType::Ivf
        );
    }

    #[test]
    fn test_dense_vector_with_num_clusters_and_nprobe() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<1536> [indexed<ivf_tq, num_clusters: 512, nprobe: 64>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(config.num_clusters, Some(512));
        assert_eq!(config.nprobe, 64);
    }

    #[test]
    fn test_dense_vector_keyword_syntax() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 1536> [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert!(config.num_clusters.is_none());
    }

    #[test]
    fn test_dense_vector_keyword_syntax_full() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 1536> [indexed<ivf_tq, num_clusters: 256, nprobe: 64>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(config.num_clusters, Some(256));
        assert_eq!(config.nprobe, 64);
    }

    #[test]
    fn test_dense_vector_keyword_syntax_partial() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<ivf_tq, num_clusters: 128>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.num_clusters, Some(128));
        assert_eq!(config.nprobe, 64); // billion-scale default
    }

    #[test]
    fn test_dense_vector_ivf_tq_index_with_probe() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<ivf_tq, num_clusters: 256, nprobe: 64>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::IvfTq);
        assert_eq!(config.num_clusters, Some(256));
        assert_eq!(config.nprobe, 64);
    }

    #[test]
    fn test_dense_vector_ivf_tq_index_without_explicit_probe() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 1536> [indexed<ivf_tq, num_clusters: 512>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(config.index_type, VectorIndexType::IvfTq);
        assert_eq!(config.num_clusters, Some(512));
    }

    #[test]
    fn test_dense_vector_ivf_tq_no_clusters() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<ivf_tq>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::IvfTq);
        assert!(config.num_clusters.is_none());
    }

    #[test]
    fn removed_ivf_pq_still_parses_to_the_reserved_variant() {
        use crate::dsl::schema::VectorIndexType;

        // The SDL keeps accepting `ivf_pq` purely so index create/open can
        // reject it with an actionable message instead of a grammar error.
        let sdl = r#"
            index test {
                field embedding: dense_vector<8> [indexed<ivf_pq>]
            }
        "#;
        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();
        assert_eq!(config.index_type, VectorIndexType::IvfPq);

        let mut builder = crate::dsl::SchemaBuilder::default();
        builder.add_dense_vector_field_with_config("embedding", true, true, config.clone());
        let schema = builder.build();
        let error = crate::dsl::schema::reject_removed_vector_index_types(&schema)
            .expect_err("removed index types must be rejected at the index gate");
        assert!(error.contains("ivf_tq"), "{error}");
        assert!(error.contains("removed"), "{error}");
    }

    #[test]
    fn test_dense_vector_flat_index() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<flat>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::Flat);
    }

    #[test]
    fn test_dense_vector_default_index_type() {
        use crate::dsl::schema::VectorIndexType;

        // Omitting an index type selects the production IVF-PQ path.
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::IvfTq);
    }

    #[test]
    fn test_dense_vector_f16_quantization() {
        use crate::dsl::schema::{DenseVectorQuantization, VectorIndexType};

        let sdl = r#"
            index documents {
                field embedding: dense_vector<768, f16> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.quantization, DenseVectorQuantization::F16);
        assert_eq!(config.index_type, VectorIndexType::IvfTq);
    }

    #[test]
    fn test_dense_vector_uint8_quantization() {
        use crate::dsl::schema::DenseVectorQuantization;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<1024, uint8> [indexed<ivf_tq>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1024);
        assert_eq!(config.quantization, DenseVectorQuantization::UInt8);
    }

    #[test]
    fn test_dense_vector_u8_alias() {
        use crate::dsl::schema::DenseVectorQuantization;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<512, u8> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 512);
        assert_eq!(config.quantization, DenseVectorQuantization::UInt8);
    }

    #[test]
    fn test_dense_vector_default_f32_quantization() {
        use crate::dsl::schema::DenseVectorQuantization;

        // No quantization type → default f32
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.quantization, DenseVectorQuantization::F32);
    }

    #[test]
    fn test_dense_vector_keyword_with_quantization() {
        use crate::dsl::schema::DenseVectorQuantization;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768, f16> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.quantization, DenseVectorQuantization::F16);
    }

    #[test]
    fn test_json_field_type() {
        let sdl = r#"
            index documents {
                field title: text [indexed, stored]
                field metadata: json [stored]
                field extra: json
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert_eq!(index.fields.len(), 3);

        // Check JSON field
        assert_eq!(index.fields[1].name, "metadata");
        assert!(matches!(index.fields[1].field_type, FieldType::Json));
        assert!(index.fields[1].stored);
        // JSON fields should not be indexed (enforced by add_json_field)

        // Check default attributes for JSON field
        assert_eq!(index.fields[2].name, "extra");
        assert!(matches!(index.fields[2].field_type, FieldType::Json));

        // Verify schema conversion
        let schema = index.to_schema();
        let metadata_field = schema.get_field("metadata").unwrap();
        let entry = schema.get_field_entry(metadata_field).unwrap();
        assert_eq!(entry.field_type, FieldType::Json);
        assert!(!entry.indexed); // JSON fields are never indexed
        assert!(entry.stored);
    }

    #[test]
    fn test_sparse_vector_query_config() {
        use crate::structures::QueryWeighting;

        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<quantization: uint8, query<tokenizer: "Alibaba-NLP/gte-Qwen2-1.5B-instruct", weighting: idf>>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert_eq!(index.fields.len(), 1);
        assert_eq!(index.fields[0].name, "embedding");
        assert!(matches!(
            index.fields[0].field_type,
            FieldType::SparseVector
        ));

        let config = index.fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config.index_size, IndexSize::U16);
        assert_eq!(config.weight_quantization, WeightQuantization::UInt8);

        // Check query config
        let query_config = config.query_config.as_ref().unwrap();
        assert_eq!(
            query_config.tokenizer.as_deref(),
            Some("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        );
        assert_eq!(query_config.weighting, QueryWeighting::Idf);

        // Verify schema conversion preserves query config
        let schema = index.to_schema();
        let embedding_field = schema.get_field("embedding").unwrap();
        let entry = schema.get_field_entry(embedding_field).unwrap();
        let sv_config = entry.sparse_vector_config.as_ref().unwrap();
        let qc = sv_config.query_config.as_ref().unwrap();
        assert_eq!(
            qc.tokenizer.as_deref(),
            Some("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
        );
        assert_eq!(qc.weighting, QueryWeighting::Idf);
    }

    #[test]
    fn test_sparse_vector_query_config_weighting_one() {
        use crate::structures::QueryWeighting;

        let sdl = r#"
            index documents {
                field embedding: sparse_vector [indexed<query<weighting: one>>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();

        let query_config = config.query_config.as_ref().unwrap();
        assert!(query_config.tokenizer.is_none());
        assert_eq!(query_config.weighting, QueryWeighting::One);
    }

    #[test]
    fn test_sparse_vector_query_config_weighting_idf_file() {
        use crate::structures::QueryWeighting;

        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<quantization: uint8, query<tokenizer: "opensearch-neural-sparse-encoding-v1", weighting: idf_file>>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();

        let query_config = config.query_config.as_ref().unwrap();
        assert_eq!(
            query_config.tokenizer.as_deref(),
            Some("opensearch-neural-sparse-encoding-v1")
        );
        assert_eq!(query_config.weighting, QueryWeighting::IdfFile);

        // Verify schema conversion preserves idf_file
        let schema = indexes[0].to_schema();
        let field = schema.get_field("embedding").unwrap();
        let entry = schema.get_field_entry(field).unwrap();
        let sc = entry.sparse_vector_config.as_ref().unwrap();
        let qc = sc.query_config.as_ref().unwrap();
        assert_eq!(qc.weighting, QueryWeighting::IdfFile);
    }

    #[test]
    fn test_sparse_vector_query_config_pruning_params() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<quantization: uint8, query<weighting: idf, weight_threshold: 0.03, max_dims: 25, pruning: 0.2, lsp_gamma: 500>>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();

        let qc = config.query_config.as_ref().unwrap();
        assert_eq!(qc.weighting, QueryWeighting::Idf);
        assert!((qc.weight_threshold - 0.03).abs() < 0.001);
        assert_eq!(qc.max_query_dims, Some(25));
        assert!((qc.pruning.unwrap() - 0.2).abs() < 0.001);
        assert_eq!(qc.lsp_gamma, Some(500));

        // Verify schema roundtrip
        let schema = indexes[0].to_schema();
        let field = schema.get_field("embedding").unwrap();
        let entry = schema.get_field_entry(field).unwrap();
        let sc = entry.sparse_vector_config.as_ref().unwrap();
        let rqc = sc.query_config.as_ref().unwrap();
        assert!((rqc.weight_threshold - 0.03).abs() < 0.001);
        assert_eq!(rqc.max_query_dims, Some(25));
        assert!((rqc.pruning.unwrap() - 0.2).abs() < 0.001);
        assert_eq!(rqc.lsp_gamma, Some(500));
    }

    #[test]
    fn test_sparse_vector_format_maxscore() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<format: maxscore, quantization: uint8>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config.format, SparseFormat::MaxScore);
        assert_eq!(config.weight_quantization, WeightQuantization::UInt8);

        // Verify schema roundtrip
        let schema = indexes[0].to_schema();
        let field = schema.get_field("embedding").unwrap();
        let entry = schema.get_field_entry(field).unwrap();
        let sc = entry.sparse_vector_config.as_ref().unwrap();
        assert_eq!(sc.format, SparseFormat::MaxScore);
    }

    #[test]
    fn test_sparse_vector_format_bmp() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<format: bmp, quantization: uint8>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].sparse_vector_config.as_ref().unwrap();
        assert_eq!(config.format, SparseFormat::Bmp);
    }

    #[test]
    fn test_fast_attribute() {
        let sdl = r#"
            index products {
                field name: text [indexed, stored]
                field price: f64 [indexed, fast]
                field category: text [indexed, stored, fast]
                field count: u64 [fast]
                field score: i64 [indexed, stored, fast]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);
        let index = &indexes[0];
        assert_eq!(index.fields.len(), 5);

        // name: no fast
        assert!(!index.fields[0].fast);
        // price: fast
        assert!(index.fields[1].fast);
        assert!(matches!(index.fields[1].field_type, FieldType::F64));
        // category: fast text
        assert!(index.fields[2].fast);
        assert!(matches!(index.fields[2].field_type, FieldType::Text));
        // count: fast only
        assert!(index.fields[3].fast);
        assert!(matches!(index.fields[3].field_type, FieldType::U64));
        // score: fast i64
        assert!(index.fields[4].fast);
        assert!(matches!(index.fields[4].field_type, FieldType::I64));

        // Verify schema roundtrip preserves fast flag
        let schema = index.to_schema();
        let price_field = schema.get_field("price").unwrap();
        assert!(schema.get_field_entry(price_field).unwrap().fast);

        let category_field = schema.get_field("category").unwrap();
        assert!(schema.get_field_entry(category_field).unwrap().fast);

        let name_field = schema.get_field("name").unwrap();
        assert!(!schema.get_field_entry(name_field).unwrap().fast);
    }

    #[test]
    fn test_primary_attribute() {
        let sdl = r#"
            index documents {
                field id: text [primary, stored]
                field title: text [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);
        let index = &indexes[0];
        assert_eq!(index.fields.len(), 2);

        // id should be primary, and auto-set fast + indexed
        let id_field = &index.fields[0];
        assert!(id_field.primary, "id should be primary");
        assert!(id_field.fast, "primary implies fast");
        assert!(id_field.indexed, "primary implies indexed");

        // title should NOT be primary
        assert!(!index.fields[1].primary);

        // Verify schema conversion preserves primary_key
        let schema = index.to_schema();
        let id = schema.get_field("id").unwrap();
        let id_entry = schema.get_field_entry(id).unwrap();
        assert!(id_entry.primary_key);
        assert!(id_entry.fast);
        assert!(id_entry.indexed);

        let title = schema.get_field("title").unwrap();
        assert!(!schema.get_field_entry(title).unwrap().primary_key);

        // primary_field() should return the primary field
        assert_eq!(schema.primary_field(), Some(id));
    }

    #[test]
    fn test_primary_with_other_attributes() {
        let sdl = r#"
            index documents {
                field id: text<simple> [primary, indexed, stored]
                field body: text [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let id_field = &indexes[0].fields[0];
        assert!(id_field.primary);
        assert!(id_field.indexed);
        assert!(id_field.stored);
        assert!(id_field.fast);
        assert_eq!(id_field.tokenizer, Some("simple".to_string()));
    }

    #[test]
    fn test_primary_only_one_allowed() {
        let sdl = r#"
            index documents {
                field id: text [primary]
                field alt_id: text [primary]
            }
        "#;

        let result = parse_sdl(sdl);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("primary key"),
            "Error should mention primary key: {}",
            err
        );
    }

    #[test]
    fn test_primary_must_be_text() {
        let sdl = r#"
            index documents {
                field id: u64 [primary]
            }
        "#;

        let result = parse_sdl(sdl);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("text"),
            "Error should mention text type: {}",
            err
        );
    }

    #[test]
    fn test_primary_cannot_be_multi() {
        let sdl = r#"
            index documents {
                field id: text [primary, stored<multi>]
            }
        "#;

        let result = parse_sdl(sdl);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("multi"), "Error should mention multi: {}", err);
    }

    #[test]
    fn test_no_primary_field() {
        // Schema without primary field should work fine
        let sdl = r#"
            index documents {
                field title: text [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let schema = indexes[0].to_schema();
        assert!(schema.primary_field().is_none());
    }

    #[test]
    fn bm25_parameters_parse_per_text_field() {
        let sdl = r#"
            index documents {
                field title: text<en_stem> [indexed<token_position, k1: 0.9, b: 0.4>]
                field body: text<en_stem> [indexed<chunked, token_position, b: 0.3>]
                field plain: text<en_stem> [indexed]
            }
        "#;
        let schema = parse_sdl(sdl).unwrap()[0].to_schema();
        let entry = |name: &str| {
            schema
                .get_field_entry(schema.get_field(name).unwrap())
                .unwrap()
                .clone()
        };
        assert_eq!(entry("title").bm25_k1, Some(0.9));
        assert_eq!(entry("title").bm25_b, Some(0.4));
        assert_eq!(entry("body").bm25_k1, None);
        assert_eq!(entry("body").bm25_b, Some(0.3));
        assert!(entry("body").chunked);
        assert_eq!(entry("plain").bm25_k1, None);
        assert_eq!(entry("plain").bm25_b, None);
        let params =
            crate::query::Bm25Params::for_field(&schema, schema.get_field("title").unwrap());
        assert_eq!((params.k1, params.b), (0.9, 0.4));
        let params =
            crate::query::Bm25Params::for_field(&schema, schema.get_field("plain").unwrap());
        assert_eq!((params.k1, params.b), (1.2, 0.75));

        // Validation: b outside 0..=1, and parameters on a non-text field.
        assert!(parse_sdl("index i { field t: text [indexed<b: 1.5>] }").is_err());
        assert!(parse_sdl("index i { field n: u64 [indexed<k1: 0.9>] }").is_err());
    }

    #[test]
    fn test_reorder_attribute() {
        let sdl = r#"
            index documents {
                field embedding: sparse_vector<u16> [indexed<format: bmp, quantization: uint8>, reorder]
                field embedding2: sparse_vector [indexed<format: bmp>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes[0].fields.len(), 2);

        // First field should have reorder=true
        assert!(indexes[0].fields[0].reorder);
        // Second field should have reorder=false
        assert!(!indexes[0].fields[1].reorder);

        // Verify schema roundtrip
        let schema = indexes[0].to_schema();
        let f1 = schema.get_field("embedding").unwrap();
        assert!(schema.get_field_entry(f1).unwrap().reorder);

        let f2 = schema.get_field("embedding2").unwrap();
        assert!(!schema.get_field_entry(f2).unwrap().reorder);

        // Index-level reorder_on_merge absent → disabled (current behaviour)
        assert!(!schema.reorder_on_merge());
    }

    #[test]
    fn test_reorder_on_merge_index_option() {
        let sdl = r#"
            index documents {
                reorder_on_merge: true
                field embedding: sparse_vector<u16> [indexed<format: bmp>, reorder]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert!(indexes[0].reorder_on_merge);
        let schema = indexes[0].to_schema();
        assert!(schema.reorder_on_merge());

        // Explicit false parses and stays disabled
        let sdl_off = r#"
            index documents {
                reorder_on_merge: false
                field embedding: sparse_vector<u16> [indexed<format: bmp>, reorder]
            }
        "#;
        let indexes = parse_sdl(sdl_off).unwrap();
        assert!(!indexes[0].reorder_on_merge);
        assert!(!indexes[0].to_schema().reorder_on_merge());

        // Schema serde roundtrip preserves the flag (persisted in metadata)
        let schema_on = parse_sdl(sdl).unwrap()[0].to_schema();
        let json = serde_json::to_string(&schema_on).unwrap();
        let back: crate::dsl::Schema = serde_json::from_str(&json).unwrap();
        assert!(back.reorder_on_merge());
    }
}
