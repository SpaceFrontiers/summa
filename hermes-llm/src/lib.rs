//! # Hermes LLM
//!
//! Inference and model-definition library for Hermes LLMs.
//!
//! Training lives in the `hermes-train` Python package (PyTorch); this crate
//! covers everything needed to define and serve models:
//!
//! - **Model Architecture Language (MAL)**: Define any transformer architecture using a composable DSL
//! - **Generation**: Text generation with temperature, top-k sampling
//! - **Tokenization**: HuggingFace tokenizer loading (local file or HF hub)
//! - **Export**: MAL → JSON model config consumed by `hermes-train`
//!
//! Checkpoints are plain safetensors with a stable tensor-naming contract
//! (`embedding.*`, `layers.{i}.attention.*`, `layers.{i}.feed_forward.*`,
//! `final_norm.*`, `lm_head.*`) shared with `hermes-train`.
//!
//! ## Quick Start
//!
//! ```ignore
//! use hermes_llm::{Transformer, get_builtin_model};
//!
//! // Load a predefined model architecture
//! let model_def = get_builtin_model("tiny").unwrap();
//!
//! // Or parse from MAL file
//! let model_def = hermes_llm::parse_mal_file("model.mal").unwrap();
//! ```
//!
//! ## Model Architecture Language (MAL)
//!
//! MAL allows defining transformer architectures in a readable, composable format:
//!
//! ```text
//! attention my_attn {
//!     num_heads: 32
//!     num_kv_heads: 8
//! }
//!
//! ffn my_ffn {
//!     hidden_dim: 4096
//!     activation: swiglu
//! }
//!
//! block my_block {
//!     attention: my_attn
//!     ffn: my_ffn
//!     norm: rmsnorm { eps: 1e-5 }
//!     norm_position: pre
//! }
//!
//! model my_model {
//!     vocab_size: 32000
//!     hidden_size: 1024
//!     num_layers: 32
//!     block: my_block
//! }
//! ```

pub mod generate;
pub mod mal;
pub mod model;
pub mod tokenizer;

// Core types
pub use model::Transformer;

// Generation
pub use generate::TextGenerator;

// Model Architecture Language (MAL)
pub use mal::{
    Activation, AttentionDef, BlockDef, FfnDef, MalFile, ModelDef, NormPosition, NormType,
    PositionEncoding, get_builtin_model, get_wellknown_mal, list_wellknown_models, parse_mal,
    parse_mal_file, parse_mal_full,
};

// Tokenization
pub use tokenizer::Tokenizer;
