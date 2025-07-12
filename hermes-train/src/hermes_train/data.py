"""Data loading — port of hermes-llm's old data.rs.

JSONL with a ``text`` field (``.gz``/``.zst``/``.zstd`` supported); documents are
tokenized, EOS-joined into one flat token stream, and sampled as random fixed
windows. The loader keeps the Rust semantics: seeded shuffle per epoch, batches
strided across ranks, and a checkpointable position for resume.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np
import torch
import torch.distributed as dist

from hermes_train.textio import open_text
from hermes_train.tokenizer import Tokenizer

__all__ = ["DataLoader", "Dataset", "open_text"]


def _iter_texts(lines: Iterator[str]) -> Iterator[str]:
    """Yield the ``text`` field of each JSONL line. Malformed lines are skipped
    and counted (logged at the end) rather than aborting a corpus-scale run."""
    bad = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "")
        except (json.JSONDecodeError, AttributeError):
            bad += 1
            continue
        if text:
            yield text
    if bad:
        print(f"WARNING: skipped {bad} malformed JSONL line(s) during tokenization")


TOKENIZE_BATCH = 20_000


class Dataset:
    """Flat EOS-joined token stream sampled as strided (non-overlapping)
    windows — one epoch is one pass over the tokens.

    Tokenization is parallel (HF ``encode_batch``) and cached next to the
    source file as raw uint32 (``<file>.tokens.bin``), memory-mapped on
    reload — corpus-scale runs restart in seconds instead of hours.
    """

    def __init__(
        self, tokens: np.ndarray, seq_len: int, eos_token_id: int | None = None
    ) -> None:
        self.tokens = tokens if tokens.dtype == np.uint32 else tokens.astype(np.uint32)
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id

    @classmethod
    def _tokenize_to(
        cls, lines: Iterator[str], tokenizer: Tokenizer, out: IO[bytes]
    ) -> int:
        total = 0
        batch: list[str] = []

        def flush() -> int:
            if not batch:
                return 0
            encodings = tokenizer.inner.encode_batch(batch)
            n = 0
            for enc in encodings:
                arr = np.array([*enc.ids, tokenizer.eos_token_id], dtype=np.uint32)
                out.write(arr.tobytes())
                n += len(arr)
            batch.clear()
            return n

        for text in _iter_texts(lines):
            batch.append(text)
            if len(batch) >= TOKENIZE_BATCH:
                total += flush()
        total += flush()
        return total

    @classmethod
    def from_files(
        cls, paths: list[str | Path], tokenizer: Tokenizer, seq_len: int
    ) -> Dataset:
        def lines() -> Iterator[str]:
            for path in paths:
                with open_text(path) as f:
                    yield from f

        buf = io.BytesIO()
        cls._tokenize_to(lines(), tokenizer, buf)
        tokens = np.frombuffer(buf.getvalue(), dtype=np.uint32)
        return cls(tokens, seq_len, eos_token_id=tokenizer.eos_token_id)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> Dataset:
        """Load with a sidecar token cache: tokenize once, memory-map on later
        runs. The cache key includes the tokenizer fingerprint, so changing the
        tokenizer invalidates it instead of silently reusing stale tokens. Under
        DDP only rank 0 tokenizes; other ranks wait on a barrier, avoiding the
        concurrent-write race on a cold cache."""
        path = Path(path)
        # Fingerprinted, rank-independent cache name.
        cache = path.with_name(f"{path.name}.{tokenizer.fingerprint()}.tokens.bin")
        fresh = cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime
        if not fresh:
            if rank == 0:
                tmp = cache.with_suffix(f".{rank}.tmp")
                with tmp.open("wb") as out, open_text(path) as f:
                    n = cls._tokenize_to(iter(f), tokenizer, out)
                tmp.replace(cache)  # atomic publish
                print(f"tokenized {path.name}: {n} tokens → {cache.name}")
            if world_size > 1 and dist.is_initialized():
                dist.barrier()  # non-rank-0 ranks wait for the cache to exist
        tokens = np.memmap(cache, dtype=np.uint32, mode="r")
        return cls(tokens, seq_len, eos_token_id=tokenizer.eos_token_id)

    @classmethod
    def from_stdin(cls, tokenizer: Tokenizer, seq_len: int) -> Dataset:
        buf = io.BytesIO()
        cls._tokenize_to(iter(sys.stdin), tokenizer, buf)
        tokens = np.frombuffer(buf.getvalue(), dtype=np.uint32)
        return cls(tokens, seq_len, eos_token_id=tokenizer.eos_token_id)

    def __len__(self) -> int:
        """Number of non-overlapping (seq_len+1)-token windows."""
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def get_batch(
        self, indices: np.ndarray, device: torch.device, with_doc_ids: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        starts = indices * self.seq_len
        windows = np.stack(
            [np.asarray(self.tokens[i : i + self.seq_len + 1]) for i in starts]
        )
        batch = torch.from_numpy(windows.astype(np.int64)).to(device)
        inputs = batch[:, :-1].contiguous()

        doc_ids = None
        if with_doc_ids and self.eos_token_id is not None:
            # Token t belongs to document number = EOS count before position t
            # within the window (EOS closes its document).
            is_eos = (inputs == self.eos_token_id).long()
            doc_ids = torch.zeros_like(inputs)
            doc_ids[:, 1:] = is_eos[:, :-1].cumsum(dim=1)

        return inputs, batch[:, 1:].contiguous(), doc_ids


class DataLoader:
    """Seeded, rank-strided batch loader with checkpointable position."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.indices = np.arange(len(dataset))
        self.current_pos = 0
        self.batches_yielded = 0
        self.max_batches = (len(dataset) // batch_size) // world_size
        self.shuffle_seed = 42
        # Fail loud rather than "complete" instantly with zero steps and write
        # an untrained checkpoint.
        if self.max_batches == 0:
            raise ValueError(
                f"dataset yields no batches: {len(dataset)} windows, "
                f"batch_size={batch_size}, world_size={world_size}. "
                "Provide more data or lower --batch-size/--seq-len."
            )

    def num_batches(self) -> int:
        return self.max_batches

    def __len__(self) -> int:
        # Load-bearing: the wandb logging path computes fractional epoch as
        # micro_step / len(train_loader). Without __len__ that path raises
        # TypeError only once wandb is enabled (dead code otherwise), so the
        # crash hides until a live run. Keep len() == the yielded batch count.
        return self.max_batches

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.shuffle_seed = seed
        self.current_pos = 0
        self.batches_yielded = 0
        if self.shuffle:
            # Shuffle a fresh arange so the permutation depends only on the
            # seed, not on prior resets — resume is deterministic either way.
            self.indices = np.arange(len(self.dataset))
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(self.indices)

    def position(self) -> int:
        return self.current_pos

    def set_position(self, pos: int) -> None:
        self.current_pos = pos
        self.batches_yielded = pos // self.batch_size // self.world_size

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield batch index arrays assigned to this rank."""
        while self.batches_yielded < self.max_batches:
            if self.current_pos + self.batch_size > len(self.indices):
                return
            batch_num = self.current_pos // self.batch_size
            batch = self.indices[self.current_pos : self.current_pos + self.batch_size]
            self.current_pos += self.batch_size
            if batch_num % self.world_size == self.rank:
                self.batches_yielded += 1
                yield batch
