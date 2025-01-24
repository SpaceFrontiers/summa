//! Model Architecture Language (MAL) for Hermes LLM
//!
//! A composable DSL for defining LLM model architectures using pest parser.
//!
//! # Example MAL - Simple (flat) style
//!
//! ```text
//! model tiny {
//!     vocab_size: 32000
//!     hidden_size: 128
//!     num_layers: 4
//!     num_heads: 4
//!     intermediate_size: 512
//! }
//! ```
//!
//! # Example MAL - Composable style
//!
//! ```text
//! # Define attention mechanism
//! attention gqa {
//!     num_heads: 32
//!     num_kv_heads: 8
//!     head_dim: 128
//!     position_encoding: rope { theta: 10000.0 }
//! }
//!
//! # Define FFN
//! ffn swiglu_mlp {
//!     hidden_dim: 14336
//!     activation: swiglu
//!     bias: false
//! }
//!
//! # Define transformer block
//! block llama_block {
//!     attention: gqa
//!     ffn: swiglu_mlp
//!     norm: rmsnorm { eps: 1e-5 }
//!     norm_position: pre
//! }
//!
//! # Define model using the block
//! model llama_7b {
//!     vocab_size: 32000
//!     max_seq_len: 4096
//!     hidden_size: 4096
//!     block: llama_block
//!     num_layers: 32
//! }
//! ```

use anyhow::{Result, anyhow};
use pest::Parser;
use pest_derive::Parser;
use rust_embed::Embed;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Embedded well-known model definitions
#[derive(Embed)]
#[folder = "well-known/"]
#[include = "*.mal"]
struct WellKnown;

#[derive(Parser)]
#[grammar = "mal/mal.pest"]
pub struct MalParser;

// ============================================================================
// AST Types
// ============================================================================

/// Position encoding type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionEncoding {
    Rope { theta: f64, scaling: Option<f64> },
    Alibi { learned_slopes: bool },
    Learned { max_positions: usize },
    None,
}

impl Default for PositionEncoding {
    fn default() -> Self {
        Self::Rope {
            theta: 10000.0,
            scaling: None,
        }
    }
}

/// Attention mechanism definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDef {
    pub name: String,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub dropout: f64,
    pub bias: bool,
    pub position_encoding: PositionEncoding,
    pub window_size: Option<usize>,
    pub causal: bool,
}

impl Default for AttentionDef {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            num_heads: None,
            num_kv_heads: None,
            head_dim: None,
            dropout: 0.0,
            bias: false,
            position_encoding: PositionEncoding::default(),
            window_size: None,
            causal: true,
        }
    }
}

/// Normalization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormType {
    #[default]
    RmsNorm,
    LayerNorm,
    None,
}

/// Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NormConfig {
    pub norm_type: NormType,
    pub eps: f64,
}

/// Activation function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Activation {
    #[default]
    SwiGLU,
    GELU,
    SiLU,
    ReLU,
    GELUNew,
    GELUTanh,
}

/// Feed-forward network definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnDef {
    pub name: String,
    pub hidden_dim: Option<usize>,
    pub activation: Activation,
    pub bias: bool,
    pub dropout: f64,
    pub gate: bool,
}

impl Default for FfnDef {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            hidden_dim: None,
            activation: Activation::default(),
            bias: false,
            dropout: 0.0,
            gate: true,
        }
    }
}

/// Transformer block definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDef {
    pub name: String,
    pub attention: AttentionDef,
    pub ffn: FfnDef,
    pub norm: NormConfig,
    pub norm_position: NormPosition,
    pub residual: bool,
    pub dropout: f64,
}

/// Normalization position in block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormPosition {
    #[default]
    Pre,
    Post,
}

impl Default for BlockDef {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            attention: AttentionDef::default(),
            ffn: FfnDef::default(),
            norm: NormConfig {
                norm_type: NormType::RmsNorm,
                eps: 1e-5,
            },
            norm_position: NormPosition::Pre,
            residual: true,
            dropout: 0.0,
        }
    }
}

/// Embeddings configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingsConfig {
    pub tie_weights: bool,
    pub dropout: f64,
    pub scale: Option<f64>,
}

/// Output head configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    pub bias: bool,
    pub norm: Option<NormConfig>,
}

/// Parsed model definition from MAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDef {
    pub name: String,
    pub description: Option<String>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub block: BlockDef,
    pub embeddings: EmbeddingsConfig,
    pub output: OutputConfig,
}

impl Default for ModelDef {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: None,
            vocab_size: 32000,
            max_seq_len: 2048,
            hidden_size: 768,
            num_layers: 12,
            block: BlockDef::default(),
            embeddings: EmbeddingsConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl std::fmt::Display for ModelDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "model {} {{", self.name)?;
        if let Some(desc) = &self.description {
            writeln!(f, "    description: \"{}\"", desc)?;
        }
        writeln!(f, "    vocab_size: {}", self.vocab_size)?;
        writeln!(f, "    max_seq_len: {}", self.max_seq_len)?;
        writeln!(f, "    hidden_size: {}", self.hidden_size)?;
        writeln!(f, "    num_layers: {}", self.num_layers)?;
        writeln!(f, "}}")?;
        writeln!(f)?;

        // Attention
        writeln!(f, "attention {{")?;
        if let Some(h) = self.block.attention.num_heads {
            writeln!(f, "    num_heads: {}", h)?;
        }
        if let Some(kv) = self.block.attention.num_kv_heads {
            writeln!(f, "    num_kv_heads: {}", kv)?;
        }
        if let Some(hd) = self.block.attention.head_dim {
            writeln!(f, "    head_dim: {}", hd)?;
        }
        writeln!(f, "    bias: {}", self.block.attention.bias)?;
        writeln!(f, "}}")?;
        writeln!(f)?;

        // FFN
        writeln!(f, "ffn {{")?;
        if let Some(dim) = self.block.ffn.hidden_dim {
            writeln!(f, "    hidden_dim: {}", dim)?;
        }
        writeln!(f, "    activation: {:?}", self.block.ffn.activation)?;
        writeln!(f, "    bias: {}", self.block.ffn.bias)?;
        writeln!(f, "}}")?;
        writeln!(f)?;

        // Block
        writeln!(f, "block {{")?;
        writeln!(f, "    norm: {:?}", self.block.norm.norm_type)?;
        writeln!(f, "    norm_position: {:?}", self.block.norm_position)?;
        writeln!(f, "    residual: {}", self.block.residual)?;
        writeln!(f, "}}")?;
        writeln!(f)?;

        // Estimated parameters
        let params = self.estimated_params();
        writeln!(
            f,
            "Estimated parameters: {:.2}B",
            params as f64 / 1_000_000_000.0
        )
    }
}

impl ModelDef {
    // ========================================================================
    // Computed properties for model construction
    // ========================================================================

    pub fn num_heads(&self) -> usize {
        self.block.attention.num_heads.unwrap_or(12)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.block
            .attention
            .num_kv_heads
            .unwrap_or(self.num_heads())
    }

    pub fn head_dim(&self) -> usize {
        self.block
            .attention
            .head_dim
            .unwrap_or(self.hidden_size / self.num_heads())
    }

    pub fn intermediate_size(&self) -> usize {
        self.block.ffn.hidden_dim.unwrap_or(self.hidden_size * 4)
    }

    pub fn dropout(&self) -> f64 {
        self.block.dropout
    }

    pub fn use_bias(&self) -> bool {
        self.block.ffn.bias || self.block.attention.bias
    }

    pub fn norm_eps(&self) -> f64 {
        if self.block.norm.eps > 0.0 {
            self.block.norm.eps
        } else {
            1e-5
        }
    }

    pub fn rope_theta(&self) -> f64 {
        match &self.block.attention.position_encoding {
            PositionEncoding::Rope { theta, .. } => *theta,
            _ => 10000.0,
        }
    }

    pub fn use_swiglu(&self) -> bool {
        matches!(self.block.ffn.activation, Activation::SwiGLU)
    }

    pub fn use_rmsnorm(&self) -> bool {
        matches!(self.block.norm.norm_type, NormType::RmsNorm)
    }

    /// Estimate total parameters
    pub fn estimated_params(&self) -> usize {
        let embed_params = self.vocab_size * self.hidden_size;
        let attn_params = 4 * self.hidden_size * self.hidden_size;
        let ff_params = 3 * self.hidden_size * self.intermediate_size();
        let layer_params = attn_params + ff_params + 2 * self.hidden_size;
        let head_params = self.hidden_size * self.vocab_size;
        embed_params + self.num_layers * layer_params + head_params
    }

    /// Load from JSON file
    pub fn from_json(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Save to JSON file
    pub fn save_json(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Complete parsed MAL file with all definitions
#[derive(Debug, Clone, Default)]
pub struct MalFile {
    pub attentions: HashMap<String, AttentionDef>,
    pub ffns: HashMap<String, FfnDef>,
    pub blocks: HashMap<String, BlockDef>,
    pub models: HashMap<String, ModelDef>,
}

// ============================================================================
// Parsing Functions
// ============================================================================

/// Parse activation type from string
fn parse_activation(s: &str) -> Activation {
    match s {
        "swiglu" => Activation::SwiGLU,
        "gelu" => Activation::GELU,
        "silu" => Activation::SiLU,
        "relu" => Activation::ReLU,
        "gelu_new" => Activation::GELUNew,
        "gelu_tanh" => Activation::GELUTanh,
        _ => Activation::SwiGLU,
    }
}

/// Parse a model property (block-based only)
fn parse_model_prop(
    pair: pest::iterators::Pair<Rule>,
    def: &mut ModelDef,
    file: &MalFile,
) -> Result<()> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::vocab_size_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.vocab_size = val.as_str().parse()?;
                }
            }
            Rule::max_seq_len_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.max_seq_len = val.as_str().parse()?;
                }
            }
            Rule::hidden_size_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.hidden_size = val.as_str().parse()?;
                }
            }
            Rule::num_layers_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.num_layers = val.as_str().parse()?;
                }
            }
            Rule::block_ref_prop => {
                for child in inner.into_inner() {
                    match child.as_rule() {
                        Rule::identifier => {
                            let name = child.as_str();
                            if let Some(block) = file.blocks.get(name) {
                                def.block = block.clone();
                            }
                        }
                        Rule::inline_block => {
                            let mut block = BlockDef::default();
                            for prop in child.into_inner() {
                                if prop.as_rule() == Rule::block_prop {
                                    parse_block_prop(prop, &mut block, file)?;
                                }
                            }
                            def.block = block;
                        }
                        _ => {}
                    }
                }
            }
            Rule::description_prop => {
                if let Some(val) = inner.into_inner().next() {
                    let s = val.as_str();
                    def.description = Some(s[1..s.len() - 1].to_string());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parse a model definition from pest pair
fn parse_model_def(pair: pest::iterators::Pair<Rule>, file: &MalFile) -> Result<ModelDef> {
    let mut def = ModelDef::default();
    let mut inner = pair.into_inner();

    // Get model name
    if let Some(name) = inner.next() {
        def.name = name.as_str().to_string();
    }

    // Parse properties
    for prop in inner {
        if prop.as_rule() == Rule::model_prop {
            parse_model_prop(prop, &mut def, file)?;
        }
    }

    Ok(def)
}

/// Parse MAL from a string (returns first model found)
pub fn parse_mal(input: &str) -> Result<ModelDef> {
    let file = parse_mal_full(input)?;
    file.models
        .into_values()
        .next()
        .ok_or_else(|| anyhow!("No model definition found"))
}

/// Parse complete MAL file with all definitions
pub fn parse_mal_full(input: &str) -> Result<MalFile> {
    let pairs = MalParser::parse(Rule::file, input).map_err(|e| anyhow!("Parse error: {}", e))?;

    let mut file = MalFile::default();

    for pair in pairs {
        if pair.as_rule() == Rule::file {
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::definition {
                    for def in inner.into_inner() {
                        match def.as_rule() {
                            Rule::model_def => {
                                let model = parse_model_def(def, &file)?;
                                file.models.insert(model.name.clone(), model);
                            }
                            Rule::attention_def => {
                                let attn = parse_attention_def(def)?;
                                file.attentions.insert(attn.name.clone(), attn);
                            }
                            Rule::ffn_def => {
                                let ffn = parse_ffn_def(def)?;
                                file.ffns.insert(ffn.name.clone(), ffn);
                            }
                            Rule::block_def => {
                                let block = parse_block_def(def, &file)?;
                                file.blocks.insert(block.name.clone(), block);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    Ok(file)
}

/// Parse an attention definition
fn parse_attention_def(pair: pest::iterators::Pair<Rule>) -> Result<AttentionDef> {
    let mut def = AttentionDef::default();
    let mut inner = pair.into_inner();

    if let Some(name) = inner.next() {
        def.name = name.as_str().to_string();
    }

    for prop in inner {
        if prop.as_rule() == Rule::attention_prop {
            parse_attention_prop(prop, &mut def)?;
        }
    }

    Ok(def)
}

/// Parse attention properties
fn parse_attention_prop(pair: pest::iterators::Pair<Rule>, def: &mut AttentionDef) -> Result<()> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::num_heads_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.num_heads = Some(val.as_str().parse()?);
                }
            }
            Rule::num_kv_heads_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.num_kv_heads = Some(val.as_str().parse()?);
                }
            }
            Rule::head_dim_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.head_dim = Some(val.as_str().parse()?);
                }
            }
            Rule::dropout_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.dropout = val.as_str().parse()?;
                }
            }
            Rule::bias_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.bias = val.as_str() == "true";
                }
            }
            Rule::causal_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.causal = val.as_str() == "true";
                }
            }
            Rule::window_size_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.window_size = Some(val.as_str().parse()?);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parse an FFN definition
fn parse_ffn_def(pair: pest::iterators::Pair<Rule>) -> Result<FfnDef> {
    let mut def = FfnDef::default();
    let mut inner = pair.into_inner();

    if let Some(name) = inner.next() {
        def.name = name.as_str().to_string();
    }

    for prop in inner {
        if prop.as_rule() == Rule::ffn_prop {
            parse_ffn_prop(prop, &mut def)?;
        }
    }

    Ok(def)
}

/// Parse FFN properties
fn parse_ffn_prop(pair: pest::iterators::Pair<Rule>, def: &mut FfnDef) -> Result<()> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::hidden_dim_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.hidden_dim = Some(val.as_str().parse()?);
                }
            }
            Rule::activation_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.activation = parse_activation(val.as_str());
                }
            }
            Rule::bias_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.bias = val.as_str() == "true";
                }
            }
            Rule::dropout_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.dropout = val.as_str().parse()?;
                }
            }
            Rule::gate_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.gate = val.as_str() == "true";
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parse a block definition
fn parse_block_def(pair: pest::iterators::Pair<Rule>, file: &MalFile) -> Result<BlockDef> {
    let mut def = BlockDef::default();
    let mut inner = pair.into_inner();

    if let Some(name) = inner.next() {
        def.name = name.as_str().to_string();
    }

    for prop in inner {
        if prop.as_rule() == Rule::block_prop {
            parse_block_prop(prop, &mut def, file)?;
        }
    }

    Ok(def)
}

/// Parse block properties
fn parse_block_prop(
    pair: pest::iterators::Pair<Rule>,
    def: &mut BlockDef,
    file: &MalFile,
) -> Result<()> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::attention_ref_prop => {
                // Can be identifier or inline definition
                for child in inner.into_inner() {
                    match child.as_rule() {
                        Rule::identifier => {
                            let name = child.as_str();
                            if let Some(attn) = file.attentions.get(name) {
                                def.attention = attn.clone();
                            }
                        }
                        Rule::inline_attention => {
                            let mut attn = AttentionDef::default();
                            for prop in child.into_inner() {
                                if prop.as_rule() == Rule::attention_prop {
                                    parse_attention_prop(prop, &mut attn)?;
                                }
                            }
                            def.attention = attn;
                        }
                        _ => {}
                    }
                }
            }
            Rule::ffn_ref_prop => {
                for child in inner.into_inner() {
                    match child.as_rule() {
                        Rule::identifier => {
                            let name = child.as_str();
                            if let Some(ffn) = file.ffns.get(name) {
                                def.ffn = ffn.clone();
                            }
                        }
                        Rule::inline_ffn => {
                            let mut ffn = FfnDef::default();
                            for prop in child.into_inner() {
                                if prop.as_rule() == Rule::ffn_prop {
                                    parse_ffn_prop(prop, &mut ffn)?;
                                }
                            }
                            def.ffn = ffn;
                        }
                        _ => {}
                    }
                }
            }
            Rule::norm_position_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.norm_position = match val.as_str() {
                        "pre" => NormPosition::Pre,
                        "post" => NormPosition::Post,
                        _ => NormPosition::Pre,
                    };
                }
            }
            Rule::residual_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.residual = val.as_str() == "true";
                }
            }
            Rule::dropout_prop => {
                if let Some(val) = inner.into_inner().next() {
                    def.dropout = val.as_str().parse()?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Parse MAL from a file
pub fn parse_mal_file<P: AsRef<std::path::Path>>(path: P) -> Result<ModelDef> {
    let content = std::fs::read_to_string(path)?;
    parse_mal(&content)
}

// ============================================================================
// Built-in model definitions
// ============================================================================

/// Get a well-known model definition by name
///
/// Accepts:
/// - Short names: "nano", "tiny", "gpt2-small", etc.
/// - Well-known paths: "well-known/nano.mal", "well-known/gpt2_small.mal"
/// - Filenames: "nano.mal", "gpt2_small.mal"
pub fn get_builtin_model(name: &str) -> Option<ModelDef> {
    let mal = get_wellknown_mal(name)?;
    parse_mal(&mal).ok()
}

/// Get the raw MAL content for a well-known model
///
/// Dynamically loads from embedded well-known/ directory.
pub fn get_wellknown_mal(name: &str) -> Option<String> {
    // Normalize: strip well-known/ prefix, ensure .mal suffix
    let name = name.strip_prefix("well-known/").unwrap_or(name);
    let filename = if name.ends_with(".mal") {
        name.to_string()
    } else {
        // Convert kebab-case to snake_case for filename
        format!("{}.mal", name.replace('-', "_"))
    };

    WellKnown::get(&filename).map(|f| String::from_utf8_lossy(&f.data).into_owned())
}

/// List all well-known model names (auto-discovered from embedded files)
pub fn list_wellknown_models() -> Vec<String> {
    WellKnown::iter()
        .filter_map(|path| {
            let path: &str = path.as_ref();
            if path.ends_with(".mal") {
                Some(path.strip_suffix(".mal").unwrap().replace('_', "-"))
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_model() {
        let mal = r#"
            attention test_attn {
                num_heads: 8
                bias: false
            }

            ffn test_ffn {
                hidden_dim: 2048
                activation: gelu
            }

            block test_block {
                attention: test_attn
                ffn: test_ffn
                norm_position: pre
            }

            model test {
                vocab_size: 32000
                hidden_size: 512
                num_layers: 8
                block: test_block
            }
        "#;

        let def = parse_mal(mal).unwrap();
        assert_eq!(def.name, "test");
        assert_eq!(def.vocab_size, 32000);
        assert_eq!(def.hidden_size, 512);
        assert_eq!(def.num_layers, 8);
    }

    #[test]
    fn test_parse_with_block_props() {
        let mal = r#"
            attention full_attn {
                num_heads: 16
                num_kv_heads: 4
                bias: true
                dropout: 0.1
            }

            ffn full_ffn {
                hidden_dim: 4096
                activation: gelu
                bias: true
                dropout: 0.1
            }

            block full_block {
                attention: full_attn
                ffn: full_ffn
                norm: layernorm { eps: 1e-6 }
                norm_position: pre
                residual: true
            }

            model full_test {
                description: "A test model"
                vocab_size: 50000
                max_seq_len: 4096
                hidden_size: 1024
                num_layers: 12
                block: full_block
            }
        "#;

        let def = parse_mal(mal).unwrap();
        assert_eq!(def.description, Some("A test model".to_string()));
        assert_eq!(def.vocab_size, 50000);
        assert_eq!(def.max_seq_len, 4096);
        assert_eq!(def.block.attention.num_heads, Some(16));
        assert_eq!(def.block.attention.num_kv_heads, Some(4));
        assert_eq!(def.block.ffn.hidden_dim, Some(4096));
        assert!(matches!(def.block.ffn.activation, Activation::GELU));
    }

    #[test]
    fn test_wellknown_models() {
        for name in list_wellknown_models() {
            let def = get_builtin_model(&name).expect(&format!("Failed to get {}", name));
            // Verify computed properties work
            assert!(def.num_heads() > 0);
            assert!(def.intermediate_size() > 0);
        }
    }

    #[test]
    fn test_model_properties() {
        let def = get_builtin_model("tiny").unwrap();

        assert_eq!(def.vocab_size, 32000);
        assert_eq!(def.hidden_size, 128);
        assert_eq!(def.num_layers, 4);
        assert_eq!(def.num_heads(), 4);
    }

    #[test]
    fn test_comments() {
        let mal = r#"
            # This is a comment
            attention test_attn {
                # Comment in attention
                num_heads: 2
            }

            ffn test_ffn {
                hidden_dim: 256
            }

            block test_block {
                attention: test_attn
                ffn: test_ffn
            }

            # Comment before model
            model test {
                vocab_size: 1000
                hidden_size: 64
                num_layers: 2
                block: test_block
            }
        "#;

        let def = parse_mal(mal).unwrap();
        assert_eq!(def.vocab_size, 1000);
    }

    #[test]
    fn test_composable_architecture() {
        let mal = r#"
            attention my_attn {
                num_heads: 16
                num_kv_heads: 4
                head_dim: 128
                bias: false
            }

            ffn my_ffn {
                hidden_dim: 11008
                activation: swiglu
                bias: false
            }

            block my_block {
                attention: my_attn
                ffn: my_ffn
                norm: rmsnorm { eps: 1e-5 }
                norm_position: pre
                residual: true
            }

            model my_model {
                description: "LLaMA 7B architecture"
                vocab_size: 32000
                max_seq_len: 4096
                hidden_size: 4096
                num_layers: 32
                block: my_block
            }
        "#;

        let file = parse_mal_full(mal).unwrap();

        assert!(file.attentions.contains_key("my_attn"));
        assert!(file.ffns.contains_key("my_ffn"));
        assert!(file.blocks.contains_key("my_block"));
        assert!(file.models.contains_key("my_model"));

        let attn = file.attentions.get("my_attn").unwrap();
        assert_eq!(attn.num_heads, Some(16));
        assert_eq!(attn.num_kv_heads, Some(4));

        let ffn = file.ffns.get("my_ffn").unwrap();
        assert_eq!(ffn.hidden_dim, Some(11008));
        assert!(matches!(ffn.activation, Activation::SwiGLU));

        let block = file.blocks.get("my_block").unwrap();
        assert!(matches!(block.norm_position, NormPosition::Pre));
        assert!(block.residual);
    }
}
