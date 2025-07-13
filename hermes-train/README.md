# hermes-train

Training for the same MAL-driven `Transformer` used by `hermes-llm` inference.
There is no Python model mirror or checkpoint adapter.

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
formats may be Zstandard-compressed (`.zst`). The reader EOS-joins documents,
packs every complete fixed-length sample, and streams samples through a
deterministic bounded shuffle buffer instead of retaining the corpus in memory.
Repeat `--data` for curriculum stages; each file is trained completely before
the next. Set `--shuffle-buffer 0` only for ordered diagnostic runs.

The trainer uses batched Muon updates for hidden 2D matrices and AdamW for
embeddings, output weights, norms, biases, and convolution kernels. Muon uses a
20x learning rate; AdamW uses beta2 0.95; global gradient norm clipping covers
both parameter sets. It supports cosine or warmup-stable-decay scheduling and
fine-tuning from safetensors. On CUDA, linear algebra uses relaxed FP32 storage
with FP16 Tensor Core staging and FP32 accumulation. Model parameters and
optimizer state remain FP32; Muon's Newton-Schulz iterations use BF16.
It writes the latest checkpoint every 100 optimizer steps by default; pass
`--checkpoint-every 0` to save only at completion. Files are staged behind an
in-progress marker and the training-state file is published last, so resume and
remote sync never consume a partially replaced checkpoint.
Each training checkpoint includes weights, AdamW and Muon state, and the exact
curriculum position. Relaunch the same command with `--resume` to replay the
deterministic bounded shuffle up to that position and continue the schedule.
On Mamba models, training and inference use fused CubeCL selective-scan kernels
on Metal and CUDA; CPU uses the tensor-operation reference implementation.

Outputs are deliberately minimal:

- `config.json`, with tokenizer vocabulary size applied
- `metrics.jsonl`, flushed after every optimizer step for live reporters
- `weights.safetensors`, using the shared model's parameter names
- `adamw-state.bpk`, `muon-state.bpk`, and `training-state.json` for resume

The checkpoint loads directly in `hermes-llm` with strict tensor and shape
validation. Experiment services such as W&B can tail `metrics.jsonl` without
being linked into the training process.
