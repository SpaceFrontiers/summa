# hermes-train

PyTorch training for Hermes LLMs. Architectures are defined in MAL (the Rust-side
DSL in `hermes-llm`); this package trains them and emits safetensors checkpoints
that `hermes-llm` serves directly — the tensor naming contract is shared
(`embedding.*`, `layers.{i}.attention.*`, `layers.{i}.feed_forward.*`,
`final_norm.*`, `lm_head.*`).

## Pipeline

```bash
# 1. Export the architecture from MAL (Rust side owns the parser)
hermes-llm export --model tiny --output config.json          # preset
hermes-llm export --model my_model.mal --output config.json  # or MAL file

# 2. Train a BPE tokenizer (optional — any HF tokenizer.json works)
hermes-train train-tokenizer --input data.jsonl --output tokenizer.json --vocab-size 32000

# 3. Train (bf16, Muon+AdamW, grad accumulation, resumable)
hermes-train train --config config.json --tokenizer tokenizer.json \
    --data data.jsonl.zst --output checkpoints --batch-size 32 --epochs 1

# multi-GPU: same command under torchrun
torchrun --nproc-per-node 8 -m hermes_train.cli train --config config.json ...

# 4. DPO fine-tuning
hermes-train dpo --config config.json --tokenizer tokenizer.json \
    --data prefs.jsonl --checkpoint checkpoints/weights.safetensors --output checkpoints-dpo

# 5. Serve with the Rust side
hermes-llm generate --checkpoint checkpoints/weights.safetensors \
    --config config.json --tokenizer tokenizer.json --prompt "hello"
```

## Training stack

- **Precision**: bf16 autocast with fp32 master weights (fp32 fallback on CPU/MPS)
- **Optimizer**: Muon (Newton–Schulz orthogonalized momentum) for 2D weight
  matrices, AdamW for embeddings/norms/head — the hybrid scheme used by
  Kimi K2 / GLM-scale runs
- **Schedule**: WSD (warmup–stable–decay, the 2026 default) or cosine via
  `--schedule`; the flat plateau makes run extension and curricula sane
- **Data**: JSONL with a `text` field; `.gz` / `.zst` supported; documents are
  tokenized, EOS-joined and packed into fixed windows
- **Document masking** (default on): attention is block-diagonal across
  packed-document boundaries, SSM recurrent state resets at doc starts, and
  the reference path uses a segment-aware causal conv — doc 2 is provably
  independent of doc 1 (tested). The fused CUDA kernel (Mamba-1) has no
  `seq_idx`, so under the fused path SSM state crosses boundaries
  (warned once); attention layers still mask correctly. `--no-doc-masking`
  disables.
- **Curriculum stages**: repeat `-d` — each file trains sequentially under
  one LR schedule, so with WSD the decay lands on your final
  (highest-quality / task-specific) stage
- **QK-Norm**: `qk_norm: true` in a MAL attention def adds per-head RMSNorm
  on Q/K before RoPE (OLMo2/Gemma-style stabilizer); tensors
  `layers.{i}.attention.{q,k}_norm.weight` in both engines
- **Checkpoints**: `weights.safetensors` (Candle-compatible names) +
  `training_state.json`; Ctrl+C saves and exits, `--resume` continues
- **Distributed**: DDP via `torchrun` (gradient sync); FSDP is a planned upgrade

## Hybrid Transformer + Mamba

MAL expresses heterogeneous layer stacks: blocks whose mixer is a Mamba-1
selective SSM instead of attention, cycled via `pattern:`:

```mal
ssm my_ssm { state_dim: 16, conv_kernel: 4, expand: 2 }
block mamba_block { ssm: my_ssm, ffn: my_ffn, norm: rmsnorm { eps: 1e-5 } }
model hybrid { ..., pattern: [mamba_block, mamba_block, attn_block] }
```

The Mamba state is a compact persistent memory (fixed size, O(1)/token);
the attention layers provide exact retrieval. Training uses a reference fp32
scan by default; install the `mamba` extra (`uv sync --extra mamba`, CUDA
only) and the fused `selective_scan_fn` kernel is used automatically on CUDA.
SSM tensors live under `layers.{i}.ssm.*` (mamba-ssm naming) and load
directly into `hermes-llm generate`, which serves incrementally: prompt
prefill once, then one step per token (Mamba recurrent state + attention KV
cache — no full recompute).
