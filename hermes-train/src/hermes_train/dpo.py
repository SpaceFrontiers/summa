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
from hermes_train.textio import open_text
from hermes_train.tokenizer import Tokenizer


class PreferenceDataset:
    """JSONL of {prompt, chosen, rejected}."""

    def __init__(self, pairs: list[dict]) -> None:
        self.pairs = pairs

    @classmethod
    def from_file(cls, path: str | Path) -> PreferenceDataset:
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
        self, model: Transformer, ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Sum of per-token log-probs of ids[1:] under the model, restricted to
        the completion (non-prompt, non-pad) positions via `mask` [B, L-1].
        Summing over prompt/pad tokens would dilute and bias the DPO signal."""
        logits = model(ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        target = ids[:, 1:].unsqueeze(-1)
        token_lp = log_probs.gather(-1, target).squeeze(-1)
        return (token_lp * mask).sum(dim=1)

    def _tokenize_pairs(
        self, prompts: list[str], completions: list[str], tokenizer: Tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt+completion, returning padded ids [B, L] and a
        completion-target mask [B, L-1] (1 where the target ids[:,1:] is a
        completion token and not padding)."""
        pad = tokenizer.pad_token_id
        rows, masks = [], []
        for prompt, completion in zip(prompts, completions, strict=True):
            p_ids = tokenizer.encode(prompt)
            c_ids = tokenizer.encode(completion)
            ids = (p_ids + c_ids)[: self.max_len]
            plen = min(len(p_ids), len(ids))
            pad_n = self.max_len - len(ids)
            # target position i predicts ids[i+1]; keep it iff that target is a
            # completion token (index >= plen) and not padding.
            mask = [1.0 if (i + 1) >= plen else 0.0 for i in range(len(ids) - 1)]
            ids += [pad] * pad_n
            mask += [0.0] * pad_n  # pad targets contribute nothing
            rows.append(ids)
            masks.append(mask)
        ids_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        mask_t = torch.tensor(masks, dtype=torch.float32, device=self.device)
        return ids_t, mask_t

    def dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> torch.Tensor:
        policy_chosen = self._sequence_log_probs(self.policy, chosen_ids, chosen_mask)
        policy_rejected = self._sequence_log_probs(
            self.policy, rejected_ids, rejected_mask
        )
        with torch.no_grad():
            ref_chosen = self._sequence_log_probs(
                self.reference, chosen_ids, chosen_mask
            )
            ref_rejected = self._sequence_log_probs(
                self.reference, rejected_ids, rejected_mask
            )

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
                prompts = [p["prompt"] for p in batch]
                chosen_ids, chosen_mask = self._tokenize_pairs(
                    prompts, [p["chosen"] for p in batch], tokenizer
                )
                rejected_ids, rejected_mask = self._tokenize_pairs(
                    prompts, [p["rejected"] for p in batch], tokenizer
                )

                loss = self.dpo_loss(
                    chosen_ids, chosen_mask, rejected_ids, rejected_mask
                )
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
