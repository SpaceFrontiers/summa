"""Compressed-text reading. A leaf module (no intra-package imports) so both
``data`` and ``tokenizer`` can import it at top level without a cycle."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import IO

import zstandard


def open_text(path: str | Path) -> IO[str]:
    """Open a possibly-compressed (``.gz``/``.zst``/``.zstd``) text file for
    line reading, UTF-8."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if suffix in (".zst", ".zstd"):
        raw = path.open("rb")  # ownership passes to the returned wrapper
        return io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8"
        )
    return path.open(encoding="utf-8")
