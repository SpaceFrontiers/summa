# hermes-llm

Train Large Language Models from scratch in Rust using [Candle](https://github.com/huggingface/candle).

## Features

- **GPT-style Transformer Architecture** with RoPE positional embeddings
- **BPE Tokenizer Training** using HuggingFace tokenizers
- **Multiple Model Configurations**: tiny, GPT-2 small/medium/large, LLaMA-style
- **Training Infrastructure**: AdamW optimizer, gradient clipping, checkpointing
- **Text Generation**: Temperature sampling, top-k sampling
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
hermes-llm train \
  --data data/corpus.txt \
  --tokenizer tokenizer.json \
  --model tiny \
  --output checkpoints \
  --lr 3e-4 \
  --batch-size 32 \
  --epochs 10 \
  --seq-len 256
```

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

For distributed training across multiple NVIDIA GPUs with gradient synchronization:

### Quick Start

```bash
# Build with NCCL support
cargo build --release -p hermes-llm --features nccl

# Launch on 4 GPUs
./scripts/train_nccl.sh 4 corpus.jsonl tokenizer.json gpt2-small
```

### Manual Launch

```bash
# Terminal 1 (GPU 0, rank 0 - creates NCCL ID)
CUDA_VISIBLE_DEVICES=0 cargo run --release -p hermes-llm --features nccl -- train \
  --data corpus.jsonl --tokenizer tokenizer.json --model gpt2-small \
  --world-size 4 --rank 0 --batch-size 32 --grad-accum 4

# Terminal 2 (GPU 1, rank 1)
CUDA_VISIBLE_DEVICES=1 cargo run --release -p hermes-llm --features nccl -- train \
  --data corpus.jsonl --tokenizer tokenizer.json --model gpt2-small \
  --world-size 4 --rank 1 --batch-size 32 --grad-accum 4

# ... repeat for ranks 2, 3
```

### CLI Options for Distributed Training

| Option             | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `--world-size N`   | Total number of GPUs/processes                            |
| `--rank N`         | This process's rank (0 to N-1)                            |
| `--gpu-id N`       | GPU device index (auto-set from rank in distributed mode) |
| `--grad-accum N`   | Gradient accumulation steps                               |
| `--comm-file PATH` | File for NCCL ID exchange (default: nccl_id.txt)          |

### Effective Batch Size

```
effective_batch = batch_size × grad_accum × world_size
```

Example: `--batch-size 32 --grad-accum 4 --world-size 4` = 512 effective batch

## Model Configurations

| Config      | Layers | Hidden | Heads | Params |
| ----------- | ------ | ------ | ----- | ------ |
| tiny        | 4      | 128    | 4     | ~1M    |
| gpt2-small  | 12     | 768    | 12    | ~124M  |
| gpt2-medium | 24     | 1024   | 16    | ~355M  |
| gpt2-large  | 36     | 1280   | 20    | ~774M  |
| llama-small | 16     | 1024   | 16    | ~150M  |

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
