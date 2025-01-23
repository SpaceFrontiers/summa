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

_Note: Parameter count depends heavily on vocab size. Run `hermes-llm info --model <name>` for exact counts._

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
