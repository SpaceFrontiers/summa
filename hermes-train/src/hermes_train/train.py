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

import json
import math
import os
import signal
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from hermes_train.config import ModelDef
from hermes_train.data import DataLoader
from hermes_train.model import Transformer, cross_entropy_loss
from hermes_train.muon import build_optimizers


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
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


class Trainer:
    def __init__(
        self,
        config: ModelDef,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        warmup_steps: int = 1000,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or pick_device()
        self.lr = lr
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.interrupted = False

        # Distributed setup (torchrun sets RANK/WORLD_SIZE)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo"
            )

        self.model = Transformer(config).to(self.device)
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
        prefixes = tuple(f"layers.{i}." for i in range(num_layers)) + ("embedding.",)
        for name, p in self.model.named_parameters():
            if name.startswith(prefixes):
                p.requires_grad = False
                frozen += 1
        # Rebuild optimizers over the remaining trainable params
        self.optimizers = build_optimizers(self.model, lr=self.lr)
        if self.is_main:
            print(f"Frozen {frozen} parameter tensors")

    def _set_lr(self, total_steps: int) -> float:
        lr = get_lr_with_warmup(
            self.global_step, self.warmup_steps, self.lr, self.lr * 0.1, total_steps
        )
        scale = lr / self.lr
        for opt in self.optimizers:
            for group in opt.param_groups:
                group.setdefault("base_lr", group["lr"])
                group["lr"] = group["base_lr"] * scale
        return lr

    def _forward_loss(
        self, input_ids: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.autocast_dtype is not None:
            with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                logits = self.ddp_model(input_ids)
                return cross_entropy_loss(logits, targets)
        logits = self.ddp_model(input_ids)
        return cross_entropy_loss(logits, targets)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        checkpoint_dir: str | Path | None = None,
        resume_state: dict | None = None,
        total_steps: int | None = None,
        max_steps: int | None = None,
    ) -> bool:
        """Run training. Returns True if completed, False if interrupted."""
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

            accumulated_loss = 0.0
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

                input_ids, targets = train_loader.dataset.get_batch(
                    batch_indices, self.device
                )
                loss = self._forward_loss(input_ids, targets)
                (loss / self.grad_accum_steps).backward()
                accumulated_loss += loss.item()

                if micro_step % self.grad_accum_steps == 0:
                    lr = self._set_lr(total_steps)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    for opt in self.optimizers:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    self.global_step += 1

                    avg_loss = accumulated_loss / self.grad_accum_steps
                    accumulated_loss = 0.0
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    if max_steps is not None and self.global_step >= max_steps:
                        pbar.close()
                        if self.is_main and checkpoint_dir:
                            path = Path(checkpoint_dir) / "checkpoint_final.safetensors"
                            self.save_checkpoint(path)
                            print(f"Reached max_steps={max_steps}, saved {path}")
                        return True

                pbar.update(1)

            pbar.close()

            if self.is_main and checkpoint_dir:
                path = (
                    Path(checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.safetensors"
                )
                self.save_checkpoint(path)
                print(f"Saved checkpoint to {path}")
            if self.world_size > 1:
                dist.barrier()

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

    def load_checkpoint(self, path: str | Path) -> None:
        state = load_file(str(path))
        self.model.load_state_dict(state, strict=True)

    def save_training_state(
        self, checkpoint_dir: str | Path, epoch: int, batch_position: int
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint(checkpoint_dir / "weights.safetensors")
        state = {
            "epoch": epoch,
            "batch_position": batch_position,
            "global_step": self.global_step,
            "shuffle_seed": epoch,
        }
        (checkpoint_dir / "training_state.json").write_text(json.dumps(state, indent=2))
        print(f"Saved interrupt checkpoint to {checkpoint_dir}")

    def load_training_state(self, checkpoint_dir: str | Path) -> dict | None:
        checkpoint_dir = Path(checkpoint_dir)
        weights = checkpoint_dir / "weights.safetensors"
        state_path = checkpoint_dir / "training_state.json"
        if not weights.exists():
            return None
        self.load_checkpoint(weights)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.global_step = state["global_step"]
            return state
        return None
