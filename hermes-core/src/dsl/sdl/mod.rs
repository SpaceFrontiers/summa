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
//!     # Dense vector with IVF-RaBitQ index
//!     field embedding: dense_vector<768> [indexed<rabitq, centroids: "centroids.bin", nprobe: 32>]
//!
//!     # Dense vector with ScaNN index and MRL dimension
//!     field embedding2: dense_vector<1536> [indexed<scann, centroids: "c.bin", codebook: "pq.bin", mrl_dim: 256>]
//! }
//! ```
//!
//! # Dense Vector Index Configuration
//!
//! Index-related parameters for dense vectors are specified in `indexed<...>`:
//! - `rabitq` or `scann` - index type
//! - `centroids: "path"` - path to pre-trained centroids file
//! - `codebook: "path"` - path to PQ codebook (ScaNN only)
//! - `nprobe: N` - number of clusters to probe (default: 32)
//! - `mrl_dim: N` - Matryoshka dimension for index (uses truncated vectors)

use pest::Parser;
use pest_derive::Parser;

use super::query_field_router::{QueryRouterRule, RoutingMode};
use super::schema::{FieldType, Schema, SchemaBuilder};
use crate::Result;
use crate::error::Error;

#[derive(Parser)]
#[grammar = "dsl/sdl/sdl.pest"]
pub struct SdlParser;

use super::schema::DenseVectorConfig;
use crate::structures::{
    IndexSize, QueryWeighting, SparseQueryConfig, SparseVectorConfig, WeightQuantization,
};

/// Parsed field definition
#[derive(Debug, Clone)]
pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,
    pub indexed: bool,
    pub stored: bool,
    /// Tokenizer name for text fields (e.g., "default", "en_stem", "german")
    pub tokenizer: Option<String>,
    /// Whether this field can have multiple values (serialized as array in JSON)
    pub multi: bool,
    /// Configuration for sparse vector fields
    pub sparse_vector_config: Option<SparseVectorConfig>,
    /// Configuration for dense vector fields
    pub dense_vector_config: Option<DenseVectorConfig>,
}

/// Parsed index definition
#[derive(Debug, Clone)]
pub struct IndexDef {
    pub name: String,
    pub fields: Vec<FieldDef>,
    pub default_fields: Vec<String>,
    /// Query router rules for routing queries to specific fields
    pub query_routers: Vec<QueryRouterRule>,
}

impl IndexDef {
    /// Convert to a Schema
    pub fn to_schema(&self) -> Schema {
        let mut builder = SchemaBuilder::default();

        for field in &self.fields {
            let f = match field.field_type {
                FieldType::Text => {
                    let tokenizer = field.tokenizer.as_deref().unwrap_or("default");
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
            };
            if field.multi {
                builder.set_multi(f, true);
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
        _ => Err(Error::Schema(format!("Unknown field type: {}", type_str))),
    }
}

/// Index configuration parsed from indexed<...> attribute
#[derive(Debug, Clone, Default)]
struct IndexConfig {
    index_type: Option<super::schema::VectorIndexType>,
    centroids_path: Option<String>,
    codebook_path: Option<String>,
    nprobe: Option<usize>,
    mrl_dim: Option<usize>,
    // Sparse vector index params
    quantization: Option<WeightQuantization>,
    weight_threshold: Option<f32>,
    // Sparse vector query-time config
    query_tokenizer: Option<String>,
    query_weighting: Option<QueryWeighting>,
}

/// Parse attributes from pest pair
/// Returns (indexed, stored, multi, index_config)
fn parse_attributes(pair: pest::iterators::Pair<Rule>) -> (bool, bool, bool, Option<IndexConfig>) {
    let mut indexed = false;
    let mut stored = false;
    let mut multi = false;
    let mut index_config = None;

    for attr in pair.into_inner() {
        if attr.as_rule() == Rule::attribute {
            // attribute = { indexed_with_config | "indexed" | "stored" | "multi" }
            // Check if it contains indexed_with_config
            let mut found_indexed_with_config = false;
            for inner in attr.clone().into_inner() {
                if inner.as_rule() == Rule::indexed_with_config {
                    indexed = true;
                    index_config = Some(parse_index_config(inner));
                    found_indexed_with_config = true;
                    break;
                }
            }
            if !found_indexed_with_config {
                // Simple attribute
                match attr.as_str() {
                    "indexed" => indexed = true,
                    "stored" => stored = true,
                    "multi" => multi = true,
                    _ => {}
                }
            }
        }
    }

    (indexed, stored, multi, index_config)
}

/// Parse index configuration from indexed<...> attribute
fn parse_index_config(pair: pest::iterators::Pair<Rule>) -> IndexConfig {
    let mut config = IndexConfig::default();

    // indexed_with_config = { "indexed" ~ "<" ~ index_config_params ~ ">" }
    // index_config_params = { index_config_param ~ ("," ~ index_config_param)* }
    // index_config_param = { index_type_kwarg | centroids_kwarg | codebook_kwarg | nprobe_kwarg | index_type_spec }

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::index_config_params {
            for param in inner.into_inner() {
                if param.as_rule() == Rule::index_config_param {
                    for p in param.into_inner() {
                        parse_single_index_config_param(&mut config, p);
                    }
                }
            }
        }
    }

    config
}

/// Parse a single index config parameter
fn parse_single_index_config_param(config: &mut IndexConfig, p: pest::iterators::Pair<Rule>) {
    use super::schema::VectorIndexType;

    match p.as_rule() {
        Rule::index_type_spec => {
            config.index_type = Some(match p.as_str() {
                "scann" => VectorIndexType::ScaNN,
                "rabitq" => VectorIndexType::IvfRaBitQ,
                _ => VectorIndexType::IvfRaBitQ,
            });
        }
        Rule::index_type_kwarg => {
            // index_type_kwarg = { "index" ~ ":" ~ index_type_spec }
            if let Some(t) = p.into_inner().next() {
                config.index_type = Some(match t.as_str() {
                    "scann" => VectorIndexType::ScaNN,
                    "rabitq" => VectorIndexType::IvfRaBitQ,
                    _ => VectorIndexType::IvfRaBitQ,
                });
            }
        }
        Rule::centroids_kwarg => {
            // centroids_kwarg = { "centroids" ~ ":" ~ centroids_path }
            // centroids_path = { "\"" ~ path_chars ~ "\"" }
            if let Some(path) = p.into_inner().next()
                && let Some(inner_path) = path.into_inner().next()
            {
                config.centroids_path = Some(inner_path.as_str().to_string());
            }
        }
        Rule::codebook_kwarg => {
            // codebook_kwarg = { "codebook" ~ ":" ~ codebook_path }
            if let Some(path) = p.into_inner().next()
                && let Some(inner_path) = path.into_inner().next()
            {
                config.codebook_path = Some(inner_path.as_str().to_string());
            }
        }
        Rule::nprobe_kwarg => {
            // nprobe_kwarg = { "nprobe" ~ ":" ~ nprobe_spec }
            if let Some(n) = p.into_inner().next() {
                config.nprobe = Some(n.as_str().parse().unwrap_or(32));
            }
        }
        Rule::mrl_dim_kwarg => {
            // mrl_dim_kwarg = { "mrl_dim" ~ ":" ~ mrl_dim_spec }
            if let Some(n) = p.into_inner().next() {
                config.mrl_dim = Some(n.as_str().parse().unwrap_or(0));
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
                config.weight_threshold = Some(t.as_str().parse().unwrap_or(0.0));
            }
        }
        Rule::query_config_block => {
            // query_config_block = { "query" ~ "<" ~ query_config_params ~ ">" }
            parse_query_config_block(config, p);
        }
        _ => {}
    }
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
                                        _ => QueryWeighting::One,
                                    });
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
    let mut indexed = true;
    let mut stored = true;
    let mut multi = false;
    let mut index_config: Option<IndexConfig> = None;

    for item in inner {
        match item.as_rule() {
            Rule::tokenizer_spec => {
                // Extract tokenizer name from <name>
                if let Some(tok_name) = item.into_inner().next() {
                    tokenizer = Some(tok_name.as_str().to_string());
                }
            }
            Rule::sparse_vector_config => {
                // Parse named parameters: <index_size: u16, quantization: uint8, weight_threshold: 0.1>
                sparse_vector_config = Some(parse_sparse_vector_config(item));
            }
            Rule::dense_vector_config => {
                // Parse dense_vector_params (keyword or positional) - only dims and mrl_dim
                dense_vector_config = Some(parse_dense_vector_config(item));
            }
            Rule::attributes => {
                let (idx, sto, mul, idx_cfg) = parse_attributes(item);
                indexed = idx;
                stored = sto;
                multi = mul;
                index_config = idx_cfg;
            }
            _ => {}
        }
    }

    // Merge index config into vector configs if both exist
    if let Some(idx_cfg) = index_config {
        if let Some(ref mut dv_config) = dense_vector_config {
            apply_index_config_to_dense_vector(dv_config, idx_cfg);
        } else if field_type == FieldType::SparseVector {
            // For sparse vectors, create default config if not present and apply index params
            let sv_config = sparse_vector_config.get_or_insert(SparseVectorConfig::default());
            apply_index_config_to_sparse_vector(sv_config, idx_cfg);
        }
    }

    Ok(FieldDef {
        name,
        field_type,
        indexed,
        stored,
        tokenizer,
        multi,
        sparse_vector_config,
        dense_vector_config,
    })
}

/// Apply index configuration from indexed<...> to DenseVectorConfig
fn apply_index_config_to_dense_vector(config: &mut DenseVectorConfig, idx_cfg: IndexConfig) {
    use super::schema::VectorIndexType;

    let nprobe = idx_cfg.nprobe.unwrap_or(32);

    match idx_cfg.index_type {
        Some(VectorIndexType::ScaNN) => {
            config.index_type = VectorIndexType::ScaNN;
            config.coarse_centroids_path = idx_cfg.centroids_path;
            config.pq_codebook_path = idx_cfg.codebook_path;
            config.nprobe = nprobe;
        }
        Some(VectorIndexType::IvfRaBitQ) => {
            config.index_type = VectorIndexType::IvfRaBitQ;
            config.coarse_centroids_path = idx_cfg.centroids_path;
            config.nprobe = nprobe;
        }
        Some(VectorIndexType::RaBitQ) | None => {
            // If centroids provided, use IVF-RaBitQ, otherwise plain RaBitQ
            if idx_cfg.centroids_path.is_some() {
                config.index_type = VectorIndexType::IvfRaBitQ;
                config.coarse_centroids_path = idx_cfg.centroids_path;
                config.nprobe = nprobe;
            }
            // else keep default RaBitQ
        }
    }

    // Apply mrl_dim if specified
    if idx_cfg.mrl_dim.is_some() {
        config.mrl_dim = idx_cfg.mrl_dim;
    }
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
        weight_quantization: WeightQuantization::default(),
        weight_threshold: 0.0,
        posting_list_pruning: None,
        query_config: None,
    }
}

/// Apply index configuration from indexed<...> to SparseVectorConfig
fn apply_index_config_to_sparse_vector(config: &mut SparseVectorConfig, idx_cfg: IndexConfig) {
    if let Some(q) = idx_cfg.quantization {
        config.weight_quantization = q;
    }
    if let Some(t) = idx_cfg.weight_threshold {
        config.weight_threshold = t;
    }
    // Apply query-time configuration if present
    if idx_cfg.query_tokenizer.is_some() || idx_cfg.query_weighting.is_some() {
        let query_config = config
            .query_config
            .get_or_insert(SparseQueryConfig::default());
        if let Some(tokenizer) = idx_cfg.query_tokenizer {
            query_config.tokenizer = Some(tokenizer);
        }
        if let Some(weighting) = idx_cfg.query_weighting {
            query_config.weighting = weighting;
        }
    }
}

/// Parse dense_vector_config - only dims
/// All index-related params (including mrl_dim) are now in indexed<...> attribute
fn parse_dense_vector_config(pair: pest::iterators::Pair<Rule>) -> DenseVectorConfig {
    let mut dim: usize = 0;

    // Navigate to dense_vector_params
    for params in pair.into_inner() {
        if params.as_rule() == Rule::dense_vector_params {
            for inner in params.into_inner() {
                match inner.as_rule() {
                    Rule::dense_vector_keyword_params => {
                        // Parse keyword args: dims: N
                        for kwarg in inner.into_inner() {
                            if kwarg.as_rule() == Rule::dims_kwarg
                                && let Some(d) = kwarg.into_inner().next()
                            {
                                dim = d.as_str().parse().unwrap_or(0);
                            }
                        }
                    }
                    Rule::dense_vector_positional_params => {
                        // Parse positional: just dimension
                        if let Some(dim_pair) = inner.into_inner().next() {
                            dim = dim_pair.as_str().parse().unwrap_or(0);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    DenseVectorConfig::new(dim)
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
            _ => {}
        }
    }

    Ok(IndexDef {
        name,
        fields,
        default_fields,
        query_routers,
    })
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
                field body: text<default> [indexed]
                field author: text [indexed, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let index = &indexes[0];

        assert_eq!(index.fields[0].name, "title");
        assert_eq!(index.fields[0].tokenizer, Some("en_stem".to_string()));

        assert_eq!(index.fields[1].name, "body");
        assert_eq!(index.fields[1].tokenizer, Some("default".to_string()));

        assert_eq!(index.fields[2].name, "author");
        assert_eq!(index.fields[2].tokenizer, None); // No tokenizer specified
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
                field uris: text [indexed, stored, multi]
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
    fn test_dense_vector_with_centroids() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<centroids: "centroids.bin">, stored]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        assert_eq!(indexes.len(), 1);

        let f = &indexes[0].fields[0];
        assert_eq!(f.name, "embedding");
        assert_eq!(f.field_type, FieldType::DenseVector);

        let config = f.dense_vector_config.as_ref().unwrap();
        assert_eq!(config.dim, 768);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("centroids.bin")
        );
        assert_eq!(config.nprobe, 32); // default
    }

    #[test]
    fn test_dense_vector_with_centroids_and_nprobe() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<1536> [indexed<centroids: "/path/to/centroids.bin", nprobe: 64>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("/path/to/centroids.bin")
        );
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
        assert!(config.coarse_centroids_path.is_none());
    }

    #[test]
    fn test_dense_vector_keyword_syntax_full() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 1536> [indexed<centroids: "/path/to/centroids.bin", nprobe: 64>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("/path/to/centroids.bin")
        );
        assert_eq!(config.nprobe, 64);
    }

    #[test]
    fn test_dense_vector_keyword_syntax_partial() {
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<centroids: "centroids.bin">]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("centroids.bin")
        );
        assert_eq!(config.nprobe, 32); // default
    }

    #[test]
    fn test_dense_vector_scann_index() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<scann, centroids: "centroids.bin", codebook: "pq_codebook.bin", nprobe: 64>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::ScaNN);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("centroids.bin")
        );
        assert_eq!(config.pq_codebook_path.as_deref(), Some("pq_codebook.bin"));
        assert_eq!(config.nprobe, 64);
    }

    #[test]
    fn test_dense_vector_rabitq_index() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 1536> [indexed<rabitq, centroids: "centroids.bin">]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(config.index_type, VectorIndexType::IvfRaBitQ);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("centroids.bin")
        );
        assert!(config.pq_codebook_path.is_none());
    }

    #[test]
    fn test_dense_vector_rabitq_no_centroids() {
        use crate::dsl::schema::VectorIndexType;

        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed<rabitq>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::IvfRaBitQ);
        assert!(config.coarse_centroids_path.is_none());
    }

    #[test]
    fn test_dense_vector_default_index_type() {
        use crate::dsl::schema::VectorIndexType;

        // When no index type specified, should default to RaBitQ (basic)
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.index_type, VectorIndexType::RaBitQ);
    }

    #[test]
    fn test_dense_vector_mrl_dim() {
        // Test matryoshka/MRL dimension trimming (new syntax: mrl_dim in indexed<...>)
        let sdl = r#"
            index documents {
                field embedding: dense_vector<1536> [indexed<mrl_dim: 256>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 1536);
        assert_eq!(config.mrl_dim, Some(256));
        assert_eq!(config.index_dim(), 256);
    }

    #[test]
    fn test_dense_vector_mrl_dim_with_centroids() {
        // Test mrl_dim combined with other index options
        let sdl = r#"
            index documents {
                field embedding: dense_vector<768> [indexed<centroids: "centroids.bin", nprobe: 64, mrl_dim: 128>]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.mrl_dim, Some(128));
        assert_eq!(config.index_dim(), 128);
        assert_eq!(
            config.coarse_centroids_path.as_deref(),
            Some("centroids.bin")
        );
        assert_eq!(config.nprobe, 64);
    }

    #[test]
    fn test_dense_vector_no_mrl_dim() {
        // Test that index_dim() returns full dim when mrl_dim is not set
        let sdl = r#"
            index documents {
                field embedding: dense_vector<dims: 768> [indexed]
            }
        "#;

        let indexes = parse_sdl(sdl).unwrap();
        let config = indexes[0].fields[0].dense_vector_config.as_ref().unwrap();

        assert_eq!(config.dim, 768);
        assert_eq!(config.mrl_dim, None);
        assert_eq!(config.index_dim(), 768);
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
}
