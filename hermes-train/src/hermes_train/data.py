"""Data loading — port of hermes-llm's old data.rs.

JSONL with a ``text`` field (``.gz``/``.zst``/``.zstd`` supported); documents are
tokenized, EOS-joined into one flat token stream, and sampled as random fixed
windows. The loader keeps the Rust semantics: seeded shuffle per epoch, batches
strided across ranks, and a checkpointable position for resume.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import numpy as np
import torch
import zstandard

from hermes_train.tokenizer import Tokenizer


def open_text(path: str | Path) -> IO[str]:
    """Open a possibly-compressed text file for line reading."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if suffix in (".zst", ".zstd"):
        raw = open(path, "rb")  # noqa: SIM115 — ownership passes to the returned wrapper
        return io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8"
        )
    return open(path, encoding="utf-8")


def _iter_texts(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        text = json.loads(line).get("text", "")
        if text:
            yield text


class Dataset:
    """Flat EOS-joined token stream with fixed-window sampling."""

    def __init__(
        self, tokens: np.ndarray, seq_len: int, eos_token_id: int | None = None
    ) -> None:
        self.tokens = tokens.astype(np.uint32, copy=False)
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id

    @classmethod
    def _from_lines(
        cls, lines: Iterator[str], tokenizer: Tokenizer, seq_len: int
    ) -> Dataset:
        all_tokens: list[int] = []
        for text in _iter_texts(lines):
            all_tokens.extend(tokenizer.encode(text))
            all_tokens.append(tokenizer.eos_token_id)
        return cls(
            np.array(all_tokens, dtype=np.uint32),
            seq_len,
            eos_token_id=tokenizer.eos_token_id,
        )

    @classmethod
    def from_files(
        cls, paths: list[str | Path], tokenizer: Tokenizer, seq_len: int
    ) -> Dataset:
        def lines() -> Iterator[str]:
            for path in paths:
                with open_text(path) as f:
                    yield from f

        return cls._from_lines(lines(), tokenizer, seq_len)

    @classmethod
    def from_file(cls, path: str | Path, tokenizer: Tokenizer, seq_len: int) -> Dataset:
        return cls.from_files([path], tokenizer, seq_len)

    @classmethod
    def from_stdin(cls, tokenizer: Tokenizer, seq_len: int) -> Dataset:
        return cls._from_lines(iter(sys.stdin), tokenizer, seq_len)

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def get_batch(
        self, indices: np.ndarray, device: torch.device, with_doc_ids: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        windows = np.stack([self.tokens[i : i + self.seq_len + 1] for i in indices])
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

    def num_batches(self) -> int:
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
