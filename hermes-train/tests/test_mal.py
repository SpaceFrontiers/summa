"""Drift guard for the in-process MAL parser.

For every well-known preset, the pure-Python parser in `hermes_train.mal` must
produce a dict byte-for-byte identical to the canonical JSON emitted by
`hermes-llm export`. The fixtures under tests/fixtures/mal/ pair each `.mal`
source with its expected `.json` (regenerate with `hermes-llm export` when the
Rust schema changes). Plus negative tests for fail-loud behavior.
"""

import json
from pathlib import Path

import pytest

from hermes_train.config import ModelDef
from hermes_train.mal import MalError, parse_mal

FIXTURES = Path(__file__).parent / "fixtures" / "mal"
PRESETS = sorted(p.stem for p in FIXTURES.glob("*.mal"))


@pytest.mark.parametrize("preset", PRESETS)
def test_mal_matches_export_json(preset: str) -> None:
    """Python parser output equals `hermes-llm export` JSON for each preset."""
    mal_src = (FIXTURES / f"{preset}.mal").read_text()
    expected = json.loads((FIXTURES / f"{preset}.json").read_text())
    assert parse_mal(mal_src) == expected


@pytest.mark.parametrize("preset", PRESETS)
def test_mal_builds_model_def(preset: str) -> None:
    """Parsed dict is accepted by ModelDef.from_dict (same as the JSON path)."""
    mal_src = (FIXTURES / f"{preset}.mal").read_text()
    from_mal = ModelDef.from_dict(parse_mal(mal_src))
    from_json = ModelDef.from_json(FIXTURES / f"{preset}.json")
    assert from_mal == from_json


def test_undefined_block_ref_raises() -> None:
    with pytest.raises(MalError, match="undefined block"):
        parse_mal(
            "model m { vocab_size: 100 max_seq_len: 64 hidden_size: 32 "
            "num_layers: 2 block: nope }"
        )


def test_undefined_attention_ref_raises() -> None:
    with pytest.raises(MalError, match="undefined attention"):
        parse_mal(
            "block b { attention: nope ffn: { hidden_dim: 64 } }\n"
            "model m { vocab_size: 100 max_seq_len: 64 hidden_size: 32 "
            "num_layers: 2 block: b }"
        )


def test_undefined_ffn_ref_raises() -> None:
    with pytest.raises(MalError, match="undefined ffn"):
        parse_mal(
            "block b { attention: { num_heads: 4 } ffn: nope }\n"
            "model m { vocab_size: 100 max_seq_len: 64 hidden_size: 32 "
            "num_layers: 2 block: b }"
        )


def test_undefined_ssm_ref_raises() -> None:
    with pytest.raises(MalError, match="undefined ssm"):
        parse_mal(
            "block b { ssm: nope ffn: { hidden_dim: 64 } }\n"
            "model m { vocab_size: 100 max_seq_len: 64 hidden_size: 32 "
            "num_layers: 2 block: b }"
        )


def test_pattern_undefined_block_raises() -> None:
    with pytest.raises(MalError, match="undefined block"):
        parse_mal(
            "block b { attention: { num_heads: 4 } ffn: { hidden_dim: 64 } }\n"
            "model m { vocab_size: 100 max_seq_len: 64 hidden_size: 32 "
            "num_layers: 2 block: b pattern: [b, nope] }"
        )


def test_unknown_property_key_raises() -> None:
    with pytest.raises(MalError, match="unknown attention property"):
        parse_mal(
            "attention a { num_heads: 4 bogus: 5 }\n"
            "ffn f { hidden_dim: 64 }\n"
            "block b { attention: a ffn: f }\n"
            "model m { vocab_size: 100 max_seq_len: 64 hidden_size: 32 "
            "num_layers: 2 block: b }"
        )


def test_unknown_model_property_raises() -> None:
    with pytest.raises(MalError, match="unknown model property"):
        parse_mal(
            "block b { attention: { num_heads: 4 } ffn: { hidden_dim: 64 } }\n"
            "model m { vocab_size: 100 hidden_size: 32 num_layers: 2 "
            "block: b nonsense: 7 }"
        )


def test_no_model_raises() -> None:
    with pytest.raises(MalError, match="no model definition"):
        parse_mal("ffn f { hidden_dim: 64 }")
