# hermes-train

Burn-native training for the same MAL-driven `Transformer` used by
`hermes-llm` inference. There is no Python model mirror or checkpoint adapter.

## Build

```bash
# CPU
cargo build --release -p hermes-train

# Apple Metal
cargo build --release -p hermes-train --features metal

# NVIDIA CUDA
cargo build --release -p hermes-train --features cuda
```

## Train

```bash
hermes-train train \
  --config models/hybrid-tiny.mal \
  --tokenizer tokenizer.json \
  --data corpus.jsonl \
  --output checkpoint \
  --batch-size 8 \
  --grad-accum 4 \
  --shuffle-buffer 8192 \
  --checkpoint-every 100 \
  --seq-len 256 \
  --epochs 1
```

Training data is either a text file or JSONL with a string `text` field; both
formats may be Zstandard-compressed (`.zst`). The reader streams documents and
fixed-length samples through a deterministic bounded shuffle buffer instead of
retaining the corpus in memory. Repeat `--data` to combine files. Samples never
cross document boundaries. Set `--shuffle-buffer 0` only for ordered diagnostic
runs.

The trainer uses Burn Autodiff with batched Muon updates for hidden 2D matrices
and AdamW for embeddings, output weights, norms, biases, and convolution
kernels. Muon uses a 20x learning rate; AdamW uses beta2 0.95; global gradient
norm clipping covers both parameter sets. It supports cosine or
warmup-stable-decay scheduling and fine-tuning from Burn-native safetensors.
On CUDA, Muon's Newton-Schulz iterations use BF16 while model parameters and
optimizer state remain FP32.
It atomically replaces the latest native checkpoint every 100 optimizer steps
by default; pass `--checkpoint-every 0` to save only at completion.
On Mamba models, training and inference use fused CubeCL selective-scan kernels
on Metal and CUDA; CPU uses the tensor-operation reference implementation.

Outputs are deliberately minimal:

- `config.json`, with tokenizer vocabulary size applied
- `metrics.jsonl`, flushed after every optimizer step for live reporters
- `weights.safetensors`, using Burn-native parameter names

The checkpoint loads directly in `hermes-llm` with strict tensor and shape
validation. Experiment services such as W&B can tail `metrics.jsonl` without
being linked into the training process.
