"""Tokenizer wrapper + BPE training (port of hermes-llm's old tokenizer.rs)."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer as HfTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Tokenizer:
    def __init__(self, inner: HfTokenizer) -> None:
        self.inner = inner
        vocab_size = inner.get_vocab_size(True)
        self.pad_token_id = self._special_id("<pad>", 0)
        self.bos_token_id = self._special_id("<bos>", 1)
        self.eos_token_id = self._special_id("<eos>", min(2, vocab_size - 1))

    def _special_id(self, token: str, default: int) -> int:
        token_id = self.inner.token_to_id(token)
        return token_id if token_id is not None else default

    @classmethod
    def from_file(cls, path: str | Path) -> Tokenizer:
        return cls(HfTokenizer.from_file(str(path)))

    @classmethod
    def from_pretrained(cls, identifier: str) -> Tokenizer:
        return cls(HfTokenizer.from_pretrained(identifier))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.inner.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.inner.get_vocab_size(True)


def train_bpe(
    inputs: list[str],
    output_path: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer from text/JSONL files and save it."""
    from hermes_train.data import open_text

    tokenizer = HfTokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or SPECIAL_TOKENS,
    )

    def line_iterator():
        for path in inputs:
            with open_text(path) as f:
                yield from (line for line in f if line.strip())

    tokenizer.train_from_iterator(line_iterator(), trainer)
    tokenizer.save(output_path, pretty=True)
    return Tokenizer.from_file(output_path)
