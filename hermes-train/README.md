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
On Mamba models, training and inference use fused CubeCL selective-scan kernels
on Metal and CUDA; CPU uses the tensor-operation reference implementation.

Outputs are deliberately minimal:

- `config.json`, with the logical tokenizer vocabulary size applied; embedding
  and output tensors use a derived 64-row storage alignment
- `metrics.jsonl`, flushed after every optimizer step for live reporters
- `weights.safetensors`, using the shared model's parameter names
- `adamw-state.bpk`, `muon-state.bpk`, and `training-state.json` for resume

The checkpoint loads directly in `hermes-llm` with strict tensor and shape
validation. Experiment services such as W&B can tail `metrics.jsonl` without
being linked into the training process.

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
an older backup. The first sync migrates the earlier flat `gcloud rsync`
layout automatically. An interrupted local checkpoint is not resumed unless a
complete remote copy can replace it.

Copy and edit the example configuration, then run the supervisor as the same
user that owns the training files:

```bash
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
