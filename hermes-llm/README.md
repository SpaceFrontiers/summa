# hermes-llm

Train Large Language Models from scratch in Rust using [Candle](https://github.com/huggingface/candle).

## Features

- **Model Architecture Language (MAL)**: Define any transformer architecture using a composable DSL
- **Transformer Architecture** with configurable attention (GQA, sliding window), normalization, and FFN
- **Well-Known Models**: Bundled architectures (nano, tiny, GPT-2, LLaMA, Mistral)
- **BPE Tokenizer Training** using HuggingFace tokenizers
- **Training Infrastructure**: AdamW optimizer, gradient clipping, checkpointing, interruptible training
- **Text Generation**: Temperature sampling, top-k sampling
- **Distributed Training**: Multi-GPU support with NCCL
- **Backend Support**: CPU, CUDA, Metal (Apple Silicon), Accelerate

## Installation

```bash
# CPU only (default)
cargo build --release -p hermes-llm

# With CUDA support
cargo build --release -p hermes-llm --features cuda

# With Metal support (macOS)
cargo build --release -p hermes-llm --features metal

# With Accelerate (macOS)
cargo build --release -p hermes-llm --features accelerate
```

## Usage

### Train a tokenizer

```bash
hermes-llm train-tokenizer \
  --input data/corpus.txt \
  --output tokenizer.json \
  --vocab-size 32000
```

### Train a model

```bash
# Using a well-known model
hermes-llm train \
  --data data/corpus.txt \
  --tokenizer tokenizer.json \
  --model tiny \
  --output checkpoints

# Or use full well-known path
hermes-llm train \
  --model well-known/mistral-7b.mal \
  ...

# Or use a custom .mal file
hermes-llm train \
  --model my_custom_model.mal \
  ...
```

**Well-known models:** `nano`, `tiny`, `gpt2-small`, `gpt2-medium`, `gpt2-large`, `llama-small`, `llama-7b`, `mistral-7b`

### Generate text

```bash
hermes-llm generate \
  --checkpoint checkpoints/checkpoint_epoch_10.safetensors \
  --config checkpoints/config.json \
  --tokenizer tokenizer.json \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.8
```

### Show model info

```bash
hermes-llm info --model gpt2-small
```

## Multi-GPU Training (NCCL)

For distributed training, just add `--num-gpus`:

```bash
# Build with NCCL support
cargo build --release -p hermes-llm --features cuda --features nccl

# Single GPU
hermes-llm train --data corpus.jsonl --tokenizer tok.json --model gpt2-small

# 4 GPUs (automatically uses NCCL)
hermes-llm train --data corpus.jsonl --tokenizer tok.json --model gpt2-small --num-gpus 4
```

### Consumer GPUs (RTX 3090, 4090, etc.)

Consumer GPUs without NVLink don't support GPU peer-to-peer access. If you see errors like `peer access is not supported between these two devices`, disable P2P and SHM:

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 hermes-llm train \
  --data corpus.jsonl --tokenizer tok.json --model gpt2-medium --num-gpus 3
```

| Variable             | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| `NCCL_P2P_DISABLE=1` | Disable direct GPU-to-GPU communication                       |
| `NCCL_SHM_DISABLE=1` | Disable shared memory (uses CUDA IPC which needs peer access) |
| `NCCL_DEBUG=INFO`    | Enable debug logging (troubleshooting)                        |

**Note:** With both disabled, NCCL uses socket-based communication which is slower but works on any multi-GPU setup.

### Training Options

| Option         | Default     | Description                      |
| -------------- | ----------- | -------------------------------- |
| `--data`       | (stdin)     | Training data file               |
| `--tokenizer`  | required    | Tokenizer file path              |
| `--model`      | tiny        | Model preset                     |
| `--num-gpus`   | 1           | Number of GPUs (>1 enables NCCL) |
| `--batch-size` | 32          | Batch size per GPU               |
| `--grad-accum` | 1           | Gradient accumulation steps      |
| `--epochs`     | 1           | Training epochs                  |
| `--lr`         | 3e-4        | Learning rate                    |
| `--output`     | checkpoints | Output directory                 |

### Effective Batch Size

```
effective_batch = batch_size × grad_accum × num_gpus
```

Example: `--batch-size 32 --grad-accum 4 --num-gpus 4` = 512 effective batch

## Fine-tuning

Continue training from a pre-trained checkpoint:

```bash
hermes-llm train \
  --checkpoint pretrained.safetensors \
  --data finetune-data.jsonl \
  --tokenizer tok.json \
  --model gpt2-small \
  --lr 1e-5 \
  --epochs 3
```

### Fine-tuning Options

| Option            | Description                                         |
| ----------------- | --------------------------------------------------- |
| `--checkpoint`    | Path to pre-trained weights (.safetensors)          |
| `--freeze-layers` | Number of layers to freeze from bottom (default: 0) |
| `--lr`            | Use lower LR for fine-tuning (e.g., 1e-5)           |

### Freezing Layers

Freeze early layers to preserve general knowledge while adapting top layers:

```bash
hermes-llm train \
  --checkpoint pretrained.safetensors \
  --data domain-data.jsonl \
  --tokenizer tok.json \
  --freeze-layers 8 \
  --lr 5e-5
```

## Direct Preference Optimization (DPO)

Align your model to human preferences without a separate reward model:

```bash
hermes-llm dpo \
  --checkpoint sft-model.safetensors \
  --config checkpoints/config.json \
  --data preferences.jsonl \
  --tokenizer tok.json \
  --beta 0.1 \
  --lr 5e-7 \
  --epochs 1
```

### Preference Data Format

JSONL file with `prompt`, `chosen`, and `rejected` fields:

```json
{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
{"prompt": "Explain gravity:", "chosen": "Gravity is...", "rejected": "Idk lol"}
```

### DPO Options

| Option         | Default         | Description                      |
| -------------- | --------------- | -------------------------------- |
| `--checkpoint` | required        | SFT model to start from          |
| `--config`     | required        | Model config JSON                |
| `--data`       | required        | Preference pairs (JSONL)         |
| `--beta`       | 0.1             | KL divergence penalty            |
| `--lr`         | 5e-7            | Learning rate (very low for DPO) |
| `--max-len`    | 512             | Max sequence length              |
| `--output`     | checkpoints-dpo | Output directory                 |

## Model Configurations

| Config      | Layers | Hidden | Heads | Params (32K vocab) |
| ----------- | ------ | ------ | ----- | ------------------ |
| nano        | 2      | 64     | 2     | ~4M                |
| tiny        | 4      | 128    | 4     | ~9M                |
| gpt2-small  | 12     | 768    | 12    | ~124M              |
| gpt2-medium | 24     | 1024   | 16    | ~355M              |
| gpt2-large  | 36     | 1280   | 20    | ~774M              |
| llama-small | 16     | 1024   | 16    | ~268M              |
| llama-7b    | 32     | 4096   | 32    | ~7B                |

_Note: Parameter count depends heavily on vocab size. Run `hermes-llm info --model <name>` for exact counts._

## Model Architecture Language (MAL)

MAL is a composable DSL for defining LLM architectures. Models are built from reusable components: **attention**, **ffn**, and **block**.

### Example

```mal
# my_model.mal

# Define attention mechanism
attention my_attn {
    num_heads: 16
    num_kv_heads: 4      # Grouped Query Attention
    bias: false
}

# Define FFN
ffn my_ffn {
    hidden_dim: 4096
    activation: swiglu
    bias: false
}

# Define transformer block
block my_block {
    attention: my_attn
    ffn: my_ffn
    norm: rmsnorm { eps: 1e-5 }
    norm_position: pre
    residual: true
}

# Define complete model
model my_model {
    description: "Custom model"
    vocab_size: 32000
    max_seq_len: 4096
    hidden_size: 1024
    num_layers: 16
    block: my_block
}
```

Use it with:

```bash
hermes-llm train --model my_model.mal --data corpus.jsonl --tokenizer tok.json
```

### MAL Components

| Component     | Properties                                                                             |
| ------------- | -------------------------------------------------------------------------------------- |
| **attention** | `num_heads`, `num_kv_heads`, `head_dim`, `bias`, `dropout`, `causal`, `window_size`    |
| **ffn**       | `hidden_dim`, `activation` (swiglu/gelu/silu/relu), `bias`, `dropout`, `gate`          |
| **block**     | `attention`, `ffn`, `norm` (rmsnorm/layernorm), `norm_position` (pre/post), `residual` |
| **model**     | `vocab_size`, `hidden_size`, `max_seq_len`, `num_layers`, `block`, `description`       |

## Architecture

The model implements a modern transformer architecture:

- **Embeddings**: Token embeddings (no position embeddings - uses RoPE)
- **Attention**: Multi-head self-attention with RoPE (Rotary Position Embedding)
- **Normalization**: RMSNorm (pre-normalization)
- **FFN**: SwiGLU activation for LLaMA-style, GELU for GPT-style
- **Output**: Tied embeddings with language modeling head

## Library Usage

```rust
use hermes_llm::{Config, GPT, Trainer};
use hermes_llm::config::TrainingConfig;
use hermes_llm::data::{Dataset, DataLoader};
use hermes_llm::tokenizer::Tokenizer;
use candle_core::Device;

// Load or train tokenizer
let tokenizer = Tokenizer::from_file("tokenizer.json")?;

// Create model config
let mut config = Config::tiny();
config.vocab_size = tokenizer.vocab_size();

// Load dataset
let dataset = Dataset::from_file("data.txt", &tokenizer, 256)?;
let mut loader = DataLoader::new(dataset, 32, true);

// Create trainer
let device = Device::Cpu;
let training_config = TrainingConfig::default();
let mut trainer = Trainer::new(config, training_config, device)?;

// Train
trainer.train(&mut loader, None, Some("checkpoints"))?;
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Candle ML Framework](https://github.com/huggingface/candle)

## License

MIT
