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
  --curriculum curriculum.json \
  --output checkpoint \
  --checkpoint-every 100
```

### Build the education curriculum

[`education-curriculum.example.json`](education-curriculum.example.json)
defines four ordered tiers: language foundations, school fundamentals,
university material, and advanced scholarship. Each tier emits a causal-LM
stage followed by a contrastive-retrieval stage, with deterministic replay of
earlier tiers.

The builder has a strict data boundary: Search API performs embedding-backed
hybrid discovery and returns IDs plus lightweight metadata;
`public.documents_assembled` in `alloydb-documents` supplies the canonical full
copy for only those IDs. Search-result text can never enter the output. URI
scheme and prefix do not affect selection; URIs are metadata only. The builder
requires Search API's `hybrid` mode, which fuses sparse and dense retrieval.

Install the live-only dependencies, set the standard libpq environment
variables (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and `PGPASSWORD`), and
run:

```bash
python -m pip install asyncpg zstandard
python hermes-train/scripts/build_education_curriculum.py \
  --config hermes-train/education-curriculum.example.json \
  --output /data/education-curriculum
```

Use `--search-limit 10` for a live smoke build. The output includes the staged
JSONL files, validation splits, a directly consumable `curriculum.json`, and a
checksummed `manifest.json` with Search API query and rejection counts. Selection
and replay are covered without live services by
`python3 hermes-train/scripts/test_education_curriculum.py`.

Training is defined by a versioned JSON curriculum; stage geometry is not split
between CLI flags and a data manifest. Start from
[`curriculum.example.json`](curriculum.example.json), then set its data paths,
step budgets, and measured batch sizes. Relative paths resolve against the
curriculum file. Set a stage's `shuffle_buffer` to zero only for ordered
diagnostic runs.

The four objectives are causal LM (`text`), target-only summarization
(`document`, `summary`), target-only retrieval planning (`request`, `plan`, and
optional `context`), and normalized in-batch contrastive retrieval (`query`,
`positive`, and optional `negatives`). Structured objectives require JSONL;
causal LM also accepts plain text. All formats may be Zstandard-compressed.
The complete schema, loss masks, truncation behavior, and resume contract are in
[`docs/training-objectives-and-curricula.md`](../docs/training-objectives-and-curricula.md).

The trainer uses batched Muon updates for hidden 2D matrices and AdamW for
embeddings, output weights, norms, biases, and convolution kernels. Muon uses a
20x learning rate; AdamW uses beta2 0.95; global gradient norm clipping covers
both parameter sets. It supports cosine or warmup-stable-decay scheduling and
fine-tuning from safetensors. CUDA training uses BF16 Tensor Core operands while
model parameters and optimizer state remain FP32; Muon's Newton-Schulz
iterations also use BF16.
It writes the latest checkpoint every 100 optimizer steps by default; pass
`--checkpoint-every 0` to save only at completion. Files are staged behind an
in-progress marker and the training-state file is published last, so resume and
remote sync never consume a partially replaced checkpoint.
Each training checkpoint includes weights, AdamW and Muon state, and the exact
curriculum position. Relaunch the same command with `--resume` to replay the
deterministic bounded shuffle up to that position and continue the schedule.
For causal stages, `.token-cache/<run-signature>/` stores an append-only local
cache of tokenized documents. Resume reconstructs the exact packer/shuffle
state from this stream without re-running the tokenizer; an interrupted tail
record is discarded and rebuilt from the corpus. The cache is derived, may be
deleted safely between runs, and is intentionally not uploaded as part of the
authoritative model/optimizer checkpoint.
Resume verifies the entire curriculum and optimization signature. Use
`--checkpoint` instead when warm-starting a new curriculum from existing
safetensors.
On Mamba models, training and inference use fused CubeCL selective-scan kernels
on Metal and CUDA; CPU uses the tensor-operation reference implementation.

Outputs are deliberately minimal:

- `config.json`, with the logical tokenizer vocabulary size applied; embedding
  and output tensors use a derived 64-row storage alignment
- `resolved-curriculum.json`, with defaults applied and relative paths resolved
- `metrics.jsonl`, flushed after every optimizer step for live reporters
- `.token-cache/`, a repairable local causal-token cache for fast resume
- `weights.safetensors`, using the shared model's parameter names
- `adamw-state.bpk`, `muon-state.bpk`, and `training-state.json` for resume

The checkpoint loads directly in `hermes-llm` with strict tensor and shape
validation. Experiment services such as W&B can tail `metrics.jsonl` without
being linked into the training process.

MoE models add `router_aux_loss` and `optimized_loss` to each metrics row. The
task-specific `loss` remains directly comparable with dense runs. See
[`docs/moe-design.md`](../docs/moe-design.md) for the MAL schema, routing
semantics, research basis, and current grouped-kernel limitation.

Pass `--layer-metrics-every N` to add pre-clipping `layer_grad_norms` every N
optimizer steps for the model visualization lab. This diagnostic is disabled by
default because it walks every layer and copies the resulting norms to the CPU.
The W&B sidecar expands the array into `layer_grad_norm/layer_N` scalar series,
while the local JSONL retains the dense row used by the lab heatmap.

## Reliable relaunch and W&B

[`scripts/relaunch.sh`](scripts/relaunch.sh) is the boot-safe supervisor for
long-running or spot-instance jobs. It owns `--output` and automatically adds
`--resume` only when all model, AdamW, Muon, and training-state files form a
complete checkpoint. A lock makes repeated boot hooks idempotent, failed
trainer processes are relaunched after a configurable delay, and termination
attempts one final remote sync.

Remote backups use either `gs://` (through `gcloud storage`) or `file://`.
Checkpoints are uploaded to an immutable `checkpoints/<step>/` directory and
`latest.json` is published last. On boot, a newer complete remote checkpoint
is restored, while a newer persistent-disk checkpoint is never overwritten by
an older backup. An interrupted local checkpoint is not resumed unless a
complete remote copy can replace it.

Copy and edit the example configuration, then run the supervisor as the same
user that owns the training files:

```bash
cp hermes-train/curriculum.example.json /opt/hermes-run/curriculum.json
cp hermes-train/scripts/relaunch.conf.example /opt/hermes-run/relaunch.conf
hermes-train/scripts/relaunch.sh /opt/hermes-run/relaunch.conf
```

For boot and process supervision, use that command as `ExecStart` in a systemd
service with `Restart=on-failure`, or from an `@reboot` cron entry. The script
itself keeps the trainer alive after ordinary process failures, so systemd is
mainly protection for the supervisor and machine lifecycle.

Set `HERMES_TRAIN_WANDB_ENV` and `HERMES_TRAIN_WANDB_PYTHON` in the run
configuration to supervise [`scripts/wandb_tail.py`](scripts/wandb_tail.py)
with the trainer. The environment file should be mode 600 and contain the API
key plus a stable `WANDB_RUN_ID`; the reporter then backfills `metrics.jsonl`,
survives file replacement during restore, and reconnects to the same run after
every restart. W&B configuration is validated before training starts so a
requested reporter cannot silently disappear.
