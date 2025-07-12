"""Direct Preference Optimization — port of hermes-llm's old dpo.rs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from hermes_train.config import ModelDef
from hermes_train.model import Transformer
from hermes_train.tokenizer import Tokenizer


class PreferenceDataset:
    """JSONL of {prompt, chosen, rejected}."""

    def __init__(self, pairs: list[dict]) -> None:
        self.pairs = pairs

    @classmethod
    def from_file(cls, path: str | Path) -> PreferenceDataset:
        from hermes_train.data import open_text

        with open_text(path) as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(pairs)} preference pairs")
        return cls(pairs)

    def __len__(self) -> int:
        return len(self.pairs)


class DpoTrainer:
    def __init__(
        self,
        config: ModelDef,
        checkpoint_path: str,
        device: torch.device,
        lr: float = 5e-7,
        beta: float = 0.1,
        max_len: int = 512,
    ) -> None:
        self.device = device
        self.beta = beta
        self.max_len = max_len

        state = load_file(checkpoint_path)
        self.policy = Transformer(config).to(device)
        self.policy.load_state_dict(state, strict=True)
        self.reference = Transformer(config).to(device)
        self.reference.load_state_dict(state, strict=True)
        self.reference.requires_grad_(False)
        self.reference.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0
        )
        print(f"DPO Trainer initialized: beta={beta}, max_len={max_len}, lr={lr}")

    def _sequence_log_probs(
        self, model: Transformer, ids: torch.Tensor
    ) -> torch.Tensor:
        """Sum of per-token log-probs of ids[1:] under the model."""
        logits = model(ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        target = ids[:, 1:].unsqueeze(-1)
        return log_probs.gather(-1, target).squeeze(-1).sum(dim=1)

    def _tokenize_batch(self, texts: list[str], tokenizer: Tokenizer) -> torch.Tensor:
        rows = []
        for text in texts:
            ids = tokenizer.encode(text)[: self.max_len]
            ids += [tokenizer.pad_token_id] * (self.max_len - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def dpo_loss(
        self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor
    ) -> torch.Tensor:
        policy_chosen = self._sequence_log_probs(self.policy, chosen_ids)
        policy_rejected = self._sequence_log_probs(self.policy, rejected_ids)
        with torch.no_grad():
            ref_chosen = self._sequence_log_probs(self.reference, chosen_ids)
            ref_rejected = self._sequence_log_probs(self.reference, rejected_ids)

        chosen_rewards = policy_chosen - ref_chosen
        rejected_rewards = policy_rejected - ref_rejected
        return -F.logsigmoid(self.beta * (chosen_rewards - rejected_rewards)).mean()

    def train(
        self,
        dataset: PreferenceDataset,
        tokenizer: Tokenizer,
        epochs: int,
        batch_size: int,
        output_dir: str | None = None,
    ) -> None:
        for epoch in range(epochs):
            self.policy.train()
            total_loss, steps = 0.0, 0
            pbar = tqdm(
                range(0, len(dataset), batch_size),
                desc=f"dpo epoch {epoch + 1}/{epochs}",
            )
            for start in pbar:
                batch = dataset.pairs[start : start + batch_size]
                chosen = self._tokenize_batch(
                    [p["prompt"] + p["chosen"] for p in batch], tokenizer
                )
                rejected = self._tokenize_batch(
                    [p["prompt"] + p["rejected"] for p in batch], tokenizer
                )

                loss = self.dpo_loss(chosen, rejected)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            print(
                f"Epoch {epoch + 1} complete, avg loss: {total_loss / max(steps, 1):.4f}"
            )
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                path = Path(output_dir) / f"dpo_epoch_{epoch + 1}.safetensors"
                save_file(
                    {
                        k: v.detach().float().contiguous().cpu()
                        for k, v in self.policy.state_dict().items()
                    },
                    str(path),
                )
                print(f"Saved checkpoint to {path}")
