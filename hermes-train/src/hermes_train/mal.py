"""Thin wrapper over the Rust MAL parser (single source of truth).

The Model Architecture Language is parsed by the Rust ``hermes-mal`` crate,
exposed to Python through the ``hermes_mal`` PyO3 extension module (built from
``hermes-mal-py``). This module is a thin shim so callers keep importing
``hermes_train.mal`` unchanged.

``hermes_mal.parse_mal`` returns a JSON string that is byte-for-byte what
``hermes-llm export`` emits (serde JSON of ``mal::ModelDef``), so
``ModelDef.from_dict`` consumes the parsed dict identically. Having exactly one
parser (Rust) removes the drift risk of the former hand-written Python parser.

Fail-loud: any syntax error, unknown property key, or undefined
block/attention/ffn/ssm reference is raised by the Rust parser as a
``ValueError`` (aliased here as ``MalError`` for backward compatibility).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_mal import parse_mal as _rust_parse_mal

__all__ = ["MalError", "parse_mal", "parse_mal_file"]

# The Rust binding raises ValueError on any parse/validation failure. Keep the
# historical name as an alias so existing `except MalError` / `pytest.raises`
# call sites (and any downstream users) keep working.
MalError = ValueError


def parse_mal(source: str) -> dict[str, Any]:
    """Parse MAL source and return the model dict (serde-compatible).

    Delegates to the Rust ``hermes_mal`` parser. Raises ``ValueError`` (aka
    :data:`MalError`) if no model is defined or on any syntax/reference error.
    """
    return json.loads(_rust_parse_mal(source))


def parse_mal_file(path: str | Path) -> dict[str, Any]:
    """Parse a ``.mal`` file and return the model dict (serde-compatible)."""
    return parse_mal(Path(path).read_text())
