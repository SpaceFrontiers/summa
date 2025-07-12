"""Tokenizer wrapper + BPE training (port of hermes-llm's old tokenizer.rs)."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

from tokenizers import Tokenizer as HfTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from hermes_train.textio import open_text

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Tokenizer:
    # Common special-token spellings across tokenizer families
    _EOS_NAMES = ("<eos>", "<|endoftext|>", "</s>", "<|end_of_text|>", "<|eot_id|>")
    _PAD_NAMES = ("<pad>", "<|padding|>", "<|pad|>")
    _BOS_NAMES = ("<bos>", "<|begin_of_text|>", "<s>", "<|startoftext|>")

    def __init__(self, inner: HfTokenizer) -> None:
        self.inner = inner
        vocab_size = inner.get_vocab_size(True)
        if vocab_size == 0:
            raise ValueError("tokenizer has an empty vocabulary")
        # EOS is load-bearing (document joins, doc-masking boundaries, stop):
        # resolve it from known names before falling back to a raw id, and warn
        # loudly if the fallback is used (it drives doc boundaries).
        eos = self._first_id(self._EOS_NAMES, None)
        if eos is None:
            eos = min(2, vocab_size - 1)
            warnings.warn(
                f"no known EOS token in tokenizer; falling back to id {eos}. "
                "Document joins/masking may be wrong — add an EOS special token.",
                stacklevel=2,
            )
        self.eos_token_id = eos
        self.pad_token_id = self._first_id(self._PAD_NAMES, self.eos_token_id)
        self.bos_token_id = self._first_id(self._BOS_NAMES, self.eos_token_id)

    def _first_id(self, names: tuple[str, ...], default: int | None) -> int | None:
        for name in names:
            token_id = self.inner.token_to_id(name)
            if token_id is not None:
                return token_id
        return default

    def fingerprint(self) -> str:
        """Short stable hash of the tokenizer definition, for cache keys — so a
        changed tokenizer never silently reuses tokens from the old one."""
        return hashlib.sha256(self.inner.to_str().encode()).hexdigest()[:12]

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
    tokenizer = HfTokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or SPECIAL_TOKENS,
    )

    def text_iterator():
        # Accept JSONL ({"text": ...}) or plain text, streamed line by line so
        # a multi-GB corpus never loads into memory at once.
        for path in inputs:
            with open_text(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line[0] == "{":
                        try:
                            yield json.loads(line).get("text", "")
                            continue
                        except json.JSONDecodeError:
                            pass
                    yield line

    tokenizer.train_from_iterator(text_iterator(), trainer)
    tokenizer.save(output_path, pretty=True)
    return Tokenizer.from_file(output_path)
