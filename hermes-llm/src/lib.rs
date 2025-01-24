//! # Hermes LLM
//!
//! A Rust library for training and running Large Language Models from scratch.
//!
//! ## Features
//!
//! - **Model Architecture Language (MAL)**: Define any transformer architecture using a composable DSL
//! - **Training**: Distributed training with NCCL, gradient accumulation, checkpointing
//! - **Generation**: Text generation with temperature, top-k sampling
//! - **Tokenization**: BPE tokenizer training and inference
//! - **DPO**: Direct Preference Optimization for RLHF
//!
//! ## Quick Start
//!
//! ```ignore
//! use hermes_llm::{Transformer, Trainer, get_builtin_model};
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

pub mod config;
pub mod data;
pub mod distributed;
pub mod dpo;
pub mod generate;
pub mod io;
pub mod mal;
pub mod model;
pub mod tokenizer;
pub mod training;

// Core types
pub use config::TrainingConfig;
pub use model::Transformer;

// Training
pub use training::{Trainer, TrainingState, create_progress_bar};

// Generation
pub use generate::{TextGenerator, get_lr_with_warmup};

// Distributed
pub use distributed::{DistributedConfig, NcclCommunicator};

// Model Architecture Language (MAL)
pub use mal::{
    Activation, AttentionDef, BlockDef, FfnDef, MalFile, ModelDef, NormPosition, NormType,
    PositionEncoding, get_builtin_model, get_wellknown_mal, list_wellknown_models, parse_mal,
    parse_mal_file, parse_mal_full,
};

// Data loading
pub use data::{DataLoader, Dataset};

// Tokenization
pub use tokenizer::{BPETrainer, Tokenizer};
