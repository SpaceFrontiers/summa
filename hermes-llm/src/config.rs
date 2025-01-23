use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length (context window)
    pub max_seq_len: usize,
    /// Embedding dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Intermediate size in FFN (typically 4x hidden_size)
    pub intermediate_size: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Whether to use bias in linear layers
    pub use_bias: bool,
    /// RoPE base frequency
    pub rope_theta: f64,
}

impl Config {
    /// GPT-2 Small configuration (124M parameters)
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 1024,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            rope_theta: 10000.0,
        }
    }

    /// GPT-2 Medium configuration (355M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 1024,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            rope_theta: 10000.0,
        }
    }

    /// GPT-2 Large configuration (774M parameters)
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            max_seq_len: 1024,
            hidden_size: 1280,
            num_layers: 36,
            num_heads: 20,
            intermediate_size: 5120,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            rope_theta: 10000.0,
        }
    }

    /// Nano configuration (~500K params) - fastest for testing
    pub fn nano() -> Self {
        Self {
            vocab_size: 1000,
            max_seq_len: 128,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 256,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            rope_theta: 10000.0,
        }
    }

    /// Tiny configuration for testing/debugging (~9M params)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            max_seq_len: 256,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 4,
            intermediate_size: 512,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: true,
            rope_theta: 10000.0,
        }
    }

    /// LLaMA-style configuration (no bias, RMSNorm, SwiGLU)
    pub fn llama_small() -> Self {
        Self {
            vocab_size: 32000,
            max_seq_len: 2048,
            hidden_size: 1024,
            num_layers: 16,
            num_heads: 16,
            intermediate_size: 2752, // 8/3 * hidden_size for SwiGLU
            dropout: 0.0,
            layer_norm_eps: 1e-6,
            use_bias: false,
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    pub fn save_json(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay for AdamW
    pub weight_decay: f64,
    /// Adam beta1
    pub beta1: f64,
    /// Adam beta2
    pub beta2: f64,
    /// Gradient clipping max norm
    pub grad_clip: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Warmup steps for learning rate scheduler
    pub warmup_steps: usize,
    /// Save checkpoint every N steps
    pub save_every: usize,
    /// Evaluate every N steps
    pub eval_every: usize,
    /// Log every N steps
    pub log_every: usize,
    /// Sequence length for training
    pub seq_len: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            grad_clip: 1.0,
            batch_size: 32,
            epochs: 1,
            warmup_steps: 1000,
            save_every: 1000,
            eval_every: 500,
            log_every: 10,
            seq_len: 512,
            gradient_accumulation_steps: 1,
        }
    }
}
