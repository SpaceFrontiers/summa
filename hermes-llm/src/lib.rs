//! # Hermes LLM
//!
//! Inference and model-definition library for Hermes LLMs.
//!
//! Burn Autodiff training lives in the `hermes-train` crate; this crate owns the
//! shared model and everything needed for inference:
//!
//! - **Model Architecture Language (MAL)**: Define any transformer architecture using a composable DSL
//! - **Generation**: Text generation with temperature, top-k sampling
//! - **Tokenization**: HuggingFace `tokenizer.json` loading
//! - **Export**: MAL → JSON model config
//!
//! Checkpoints are Burn-native safetensors written directly from the same
//! [`Transformer`] module tree used by `hermes-train`, with no conversion layer.
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

pub mod burn_model;
pub mod generate;
/// Model Architecture Language (MAL) — re-exported from the standalone
/// `hermes-mal` crate, which is the single source of truth.
pub use hermes_mal as mal;
pub mod remote;
pub mod tokenizer;

// Core types
pub use burn_model::{
    Backend, Device, InferenceState, MambaBackend, Transformer, default_device, load_safetensors,
    save_safetensors,
};

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
