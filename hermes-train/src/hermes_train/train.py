"""Training loop — successor of hermes-llm's old training.rs.

Upgrades over the Rust trainer:
- bf16 autocast with fp32 master weights (CUDA; fp32 on CPU/MPS)
- Muon for hidden 2D matrices + AdamW for the rest
- correct gradient accumulation (every micro-batch contributes)
- cosine LR schedule with linear warmup
- DDP via torchrun (RANK/WORLD_SIZE env)

Kept from the Rust trainer:
- Ctrl+C saves an interrupt checkpoint; --resume continues from it
- training_state.json schema (epoch, batch_position, global_step, shuffle_seed)
- weights.safetensors with Candle-compatible tensor names
- layer freezing for fine-tuning
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import signal
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from hermes_train.config import ModelDef
from hermes_train.data import DataLoader
from hermes_train.model import Transformer
from hermes_train.muon import build_optimizers


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)  # noqa: NPY002 — global legacy-RNG seed is intentional
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr_with_warmup(
    step: int, warmup_steps: int, max_lr: float, min_lr: float, total_steps: int
) -> float:
    """Cosine schedule with linear warmup (port of the old generate.rs helper)."""
    if step < warmup_steps:
        return max_lr * (step / max(warmup_steps, 1))
    decay_ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(decay_ratio, 1.0)))
    return min_lr + coeff * (max_lr - min_lr)


def get_lr_wsd(
    step: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
    total_steps: int,
    decay_frac: float = 0.1,
) -> float:
    """Warmup-Stable-Decay: flat peak LR for the bulk of training, cosine
    decay only over the final `decay_frac` of steps. Makes mid-run extension
    and multi-stage curricula sane (the 2026 default over cosine)."""
    if step < warmup_steps:
        return max_lr * (step / max(warmup_steps, 1))
    decay_start = int(total_steps * (1.0 - decay_frac))
    if step < decay_start:
        return max_lr
    decay_ratio = (step - decay_start) / max(total_steps - decay_start, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(decay_ratio, 1.0)))
    return min_lr + coeff * (max_lr - min_lr)


class Trainer:
    def __init__(
        self,
        config: ModelDef,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        warmup_steps: int = 1000,
        schedule: str = "wsd",  # wsd | cosine
        doc_masking: bool = True,
        grad_checkpoint: bool = False,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or pick_device()
        self.lr = lr
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.doc_masking = doc_masking
        self.grad_checkpoint = grad_checkpoint
        self.seed = seed
        self.global_step = 0
        self.interrupted = False

        # Distributed setup (torchrun sets RANK/WORLD_SIZE)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Seed every RNG (offset by rank so DDP replicas draw distinct dropout
        # masks) — model init still matches because DDP broadcasts rank-0 params.
        _seed_everything(seed + self.rank)
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo"
            )

        self.model = Transformer(config).to(self.device)
        self.model.gradient_checkpointing = grad_checkpoint
        self.ddp_model: torch.nn.Module = self.model
        if self.world_size > 1:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )

        self.optimizers = build_optimizers(self.model, lr=lr, weight_decay=weight_decay)
        self.autocast_dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else None
        )

        if self.is_main:
            print(
                f"Model: {config.name}, {self.model.num_parameters():,} params, "
                f"device={self.device}, "
                f"precision={'bf16' if self.autocast_dtype else 'fp32'}, "
                f"world_size={self.world_size}"
            )

        # Optional Weights & Biases live curves — enabled only when
        # WANDB_API_KEY is set (main rank). No key → silently off.
        self.wandb = None
        if self.is_main and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb  # noqa: PLC0415 — optional dep, imported only when WANDB_API_KEY set

                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "hermes-retriever"),
                    name=os.environ.get("WANDB_NAME", config.name),
                    config={
                        "params": self.model.num_parameters(),
                        "hidden_size": config.hidden_size,
                        "num_layers": config.num_layers,
                        "vocab_size": config.vocab_size,
                        "lr": lr,
                        "schedule": schedule,
                        "grad_accum_steps": grad_accum_steps,
                        "world_size": self.world_size,
                    },
                )
                self.wandb = wandb
                print(f"wandb: logging to {wandb.run.url}")
            except Exception as e:  # noqa: BLE001 — never let logging break training
                print(f"wandb disabled ({e})")

        signal.signal(signal.SIGINT, self._on_interrupt)

    def _on_interrupt(self, signum, frame) -> None:
        print("\nInterrupt received, saving checkpoint...")
        self.interrupted = True

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def freeze_layers(self, num_layers: int) -> None:
        """Freeze the first N layers plus embeddings (fine-tuning)."""
        if num_layers <= 0:
            return
        frozen = 0
        prefixes = (*(f"layers.{i}." for i in range(num_layers)), "embedding.")
        for name, p in self.model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad = False
                frozen += 1
        # Rebuild optimizers over the remaining trainable params
        self.optimizers = build_optimizers(self.model, lr=self.lr)
        if self.is_main:
            print(f"Frozen {frozen} parameter tensors")

    def _wandb_log(self, data: dict) -> None:
        """Log to wandb, tolerating transient failures — a network blip must
        never crash a multi-hour run. Disable after repeated failures."""
        if not self.wandb:
            return
        try:
            self.wandb.log(data, step=self.global_step)
        except Exception as e:  # noqa: BLE001
            self._wandb_fail = getattr(self, "_wandb_fail", 0) + 1
            if self._wandb_fail == 1:
                print(f"wandb.log failed ({e}); continuing")
            if self._wandb_fail >= 10:
                print("wandb.log failed 10x — disabling wandb for this run")
                self.wandb = None

    def _wandb_finish(self) -> None:
        if self.wandb:
            with contextlib.suppress(Exception):
                self.wandb.finish()

    def _set_lr(self, total_steps: int) -> float:
        schedule_fn = get_lr_wsd if self.schedule == "wsd" else get_lr_with_warmup
        lr = schedule_fn(
            self.global_step, self.warmup_steps, self.lr, self.lr * 0.1, total_steps
        )
        scale = lr / self.lr
        for opt in self.optimizers:
            for group in opt.param_groups:
                group.setdefault("base_lr", group["lr"])
                group["lr"] = group["base_lr"] * scale
        return lr

    def _forward_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pass targets so the model computes chunked cross-entropy internally
        # (never materializing the full [B, S, V] logits) inside the DDP forward.
        if self.autocast_dtype is not None:
            with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                return self.ddp_model(input_ids, doc_ids=doc_ids, targets=targets)
        return self.ddp_model(input_ids, doc_ids=doc_ids, targets=targets)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        checkpoint_dir: str | Path | None = None,
        resume_state: dict | None = None,
        total_steps: int | None = None,
        max_steps: int | None = None,
        save_every: int | None = None,
    ) -> bool:
        """Run training. Returns True if completed, False if interrupted.

        save_every>0 writes ``weights.safetensors`` every N optimizer steps
        (atomic temp+rename, so a concurrent reader — e.g. an eval process
        loading the snapshot — never sees a half-written file). This is what
        makes mid-run evaluation possible without stopping training.
        """
        start_epoch, start_position = 0, 0
        if resume_state:
            self.global_step = resume_state["global_step"]
            start_epoch = resume_state["epoch"]
            start_position = resume_state["batch_position"]
            if self.is_main:
                print(
                    f"Resuming from epoch {start_epoch + 1}, "
                    f"batch {start_position}, step {self.global_step}"
                )

        steps_per_epoch = max(train_loader.num_batches() // self.grad_accum_steps, 1)
        total_steps = total_steps or min(
            steps_per_epoch * epochs, max_steps or (1 << 62)
        )
        if self.warmup_steps >= total_steps and self.is_main:
            print(
                f"WARNING: warmup_steps={self.warmup_steps} >= total_steps="
                f"{total_steps}; LR never reaches peak — the model will "
                "under-train. Lower --warmup-steps."
            )

        for epoch in range(start_epoch, epochs):
            train_loader.reset(seed=epoch)
            if epoch == start_epoch and start_position > 0:
                train_loader.set_position(start_position)

            self.ddp_model.train()
            pbar = tqdm(
                total=train_loader.num_batches(),
                desc=f"epoch {epoch + 1}/{epochs}",
                disable=not self.is_main,
            )

            # Accumulate the loss as a device tensor and read it once per
            # optimizer step, so we don't force a host sync every micro-batch.
            accumulated_loss = torch.zeros((), device=self.device)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            for micro_step, batch_indices in enumerate(train_loader, start=1):
                if self.interrupted:
                    if self.is_main and checkpoint_dir:
                        self.save_training_state(
                            checkpoint_dir,
                            epoch=epoch,
                            batch_position=train_loader.position(),
                        )
                    pbar.close()
                    return False

                input_ids, targets, doc_ids = train_loader.dataset.get_batch(
                    batch_indices, self.device, with_doc_ids=self.doc_masking
                )
                is_boundary = micro_step % self.grad_accum_steps == 0
                # Under DDP, only all-reduce grads on the accumulation boundary;
                # no_sync() suppresses the reduce on the intermediate micro-steps
                # (else we pay grad_accum_stepsx the communication).
                if self.world_size > 1 and not is_boundary:
                    sync_ctx = self.ddp_model.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()
                with sync_ctx:
                    loss = self._forward_loss(input_ids, targets, doc_ids)
                    (loss / self.grad_accum_steps).backward()
                accumulated_loss += loss.detach()

                if is_boundary:
                    lr = self._set_lr(total_steps)
                    grad_norm = 0.0
                    if self.grad_clip > 0:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        )
                    for opt in self.optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    self.global_step += 1

                    avg_loss = accumulated_loss.item() / self.grad_accum_steps
                    accumulated_loss = torch.zeros((), device=self.device)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                    if self.wandb:
                        tokens = (
                            input_ids.shape[0]
                            * input_ids.shape[1]
                            * self.grad_accum_steps
                            * self.world_size
                        )
                        self._wandb_log(
                            {
                                "loss": avg_loss,
                                "lr": lr,
                                "grad_norm": grad_norm,
                                "tokens": self.global_step * tokens,
                                "epoch": epoch + micro_step / max(len(train_loader), 1),
                            }
                        )

                    if (
                        save_every
                        and self.global_step % save_every == 0
                        and self.is_main
                        and checkpoint_dir
                    ):
                        snapshot = Path(checkpoint_dir) / "weights.safetensors"
                        self.save_checkpoint_atomic(snapshot)
                        print(f"[step {self.global_step}] snapshot → {snapshot}")

                    if max_steps is not None and self.global_step >= max_steps:
                        pbar.close()
                        if self.is_main and checkpoint_dir:
                            self.save_training_state(
                                checkpoint_dir,
                                epoch=epoch,
                                batch_position=train_loader.position(),
                            )
                            print(f"Reached max_steps={max_steps}")
                        self._wandb_finish()
                        return True

                pbar.update(1)

            pbar.close()

            if self.is_main and checkpoint_dir:
                # Named epoch weights (for keeping history) + a resumable state.
                path = (
                    Path(checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.safetensors"
                )
                self.save_checkpoint_atomic(path)
                self.save_training_state(
                    checkpoint_dir, epoch=epoch + 1, batch_position=0
                )
                print(f"Saved checkpoint to {path}")
            if self.world_size > 1:
                dist.barrier()

        self._wandb_finish()
        return True

    # ------------------------------------------------------------------
    # Checkpointing (Candle-compatible weights + resume sidecar)
    # ------------------------------------------------------------------

    def state_dict_for_export(self) -> dict[str, torch.Tensor]:
        """Contiguous fp32 CPU tensors, RoPE buffers excluded (non-persistent)."""
        return {
            k: v.detach().float().contiguous().cpu()
            for k, v in self.model.state_dict().items()
        }

    def save_checkpoint(self, path: str | Path) -> None:
        save_file(self.state_dict_for_export(), str(path))

    def save_checkpoint_atomic(self, path: str | Path) -> None:
        """Weights-only atomic snapshot for concurrent mid-run eval readers."""
        self._atomic_write(
            Path(path), lambda p: save_file(self.state_dict_for_export(), str(p))
        )

    def load_checkpoint(self, path: str | Path) -> None:
        state = load_file(str(path))
        self.model.load_state_dict(state, strict=True)

    def _atomic_write(self, path: Path, write: Callable[[Path], None]) -> None:
        """Write via a temp sibling + rename so readers/resumes never observe a
        partial file (safetensors, optimizer state, or json)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        write(tmp)
        tmp.replace(path)

    def save_training_state(
        self, checkpoint_dir: str | Path, epoch: int, batch_position: int
    ) -> None:
        """Save a fully resumable checkpoint: weights + optimizer moments +
        metadata, each written atomically. Without optimizer state, every
        resume restarts AdamW/Muon cold (LR spike + instability)."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(
            checkpoint_dir / "weights.safetensors",
            lambda p: save_file(self.state_dict_for_export(), str(p)),
        )
        self._atomic_write(
            checkpoint_dir / "optim_state.pt",
            lambda p: torch.save([o.state_dict() for o in self.optimizers], p),
        )
        state = {
            "epoch": epoch,
            "batch_position": batch_position,
            "global_step": self.global_step,
            "shuffle_seed": epoch,
            "seed": self.seed,
            "world_size": self.world_size,
            "lr": self.lr,
            "schedule": self.schedule,
            "grad_accum_steps": self.grad_accum_steps,
            "torch_version": torch.__version__,
            "wandb_run": getattr(self.wandb, "run", None) and self.wandb.run.id,
        }
        self._atomic_write(
            checkpoint_dir / "training_state.json",
            lambda p: p.write_text(json.dumps(state, indent=2)),
        )
        print(f"Saved resumable checkpoint to {checkpoint_dir}")

    def load_training_state(self, checkpoint_dir: str | Path) -> dict | None:
        checkpoint_dir = Path(checkpoint_dir)
        weights = checkpoint_dir / "weights.safetensors"
        state_path = checkpoint_dir / "training_state.json"
        optim_path = checkpoint_dir / "optim_state.pt"
        if not weights.exists():
            return None
        self.load_checkpoint(weights)
        if optim_path.exists():
            opt_states = torch.load(
                optim_path, map_location=self.device, weights_only=False
            )
            if len(opt_states) != len(self.optimizers):
                raise ValueError(
                    f"optim_state has {len(opt_states)} optimizers, "
                    f"trainer has {len(self.optimizers)} — cannot resume"
                )
            for opt, st in zip(self.optimizers, opt_states, strict=True):
                opt.load_state_dict(st)
        elif self.is_main:
            print(
                "WARNING: resuming without optim_state.pt — optimizer moments "
                "reset (older checkpoint?); expect a transient loss bump"
            )
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.global_step = state["global_step"]
            return state
        return None
