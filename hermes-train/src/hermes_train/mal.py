"""Self-contained MAL (Model Architecture Language) parser.

Parses a ``.mal`` file into the exact same dict structure that
``hermes-llm export`` emits (serde JSON of ``mal::ModelDef``), so that
``ModelDef.from_dict`` consumes it identically. This lets a training box turn a
``.mal`` source into a model config with no Rust binary present.

The grammar mirrors ``hermes-llm/src/mal/mal.pest`` and the AST/serde structs in
``hermes-llm/src/mal/mod.rs``. Field names, defaults, enum tagging and property
aliases are kept in lockstep; ``tests/test_mal.py`` pins the output byte-for-byte
against the canonical ``hermes-llm export`` JSON for every well-known preset.

Fail-loud: undefined block/attention/ffn/ssm references and unknown property
keys raise ``MalError`` (matching the Rust parser, which rejects both).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["MalError", "parse_mal", "parse_mal_file"]


class MalError(ValueError):
    """Raised on any MAL syntax error, unknown key, or undefined reference."""


# ---------------------------------------------------------------------------
# Property-name aliases (mirror mal.pest shared-property rules)
# ---------------------------------------------------------------------------

_NUM_HEADS = {"num_heads", "n_heads", "attention_heads"}
_NUM_KV_HEADS = {"num_kv_heads", "kv_heads"}
_MAX_SEQ_LEN = {"max_seq_len", "context_length", "seq_len"}
_HIDDEN_SIZE = {"hidden_size", "d_model", "embed_dim"}
_NUM_LAYERS = {"num_layers", "n_layers", "depth"}
_HIDDEN_DIM = {"hidden_dim", "intermediate_size"}
_STATE_DIM = {"state_dim", "d_state"}
_CONV_KERNEL = {"conv_kernel", "d_conv"}
_ROPE_THETA = {"rope_theta", "theta"}
_NORM_EPS = {"norm_eps", "eps"}

_ACTIVATIONS = {
    "swiglu": "SwiGLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "relu": "ReLU",
    "gelu_new": "GELUNew",
    "gelu_tanh": "GELUTanh",
}


# ---------------------------------------------------------------------------
# Default constructors (mirror the Default impls in mal/mod.rs)
# ---------------------------------------------------------------------------


def _default_attention() -> dict[str, Any]:
    return {
        "name": "default",
        "num_heads": None,
        "num_kv_heads": None,
        "head_dim": None,
        "dropout": 0.0,
        "bias": False,
        "position_encoding": {"Rope": {"theta": 10000.0, "scaling": None}},
        "window_size": None,
        "causal": True,
        "qk_norm": False,
    }


def _default_ssm() -> dict[str, Any]:
    return {
        "name": "default",
        "state_dim": 16,
        "conv_kernel": 4,
        "expand": 2,
        "dt_rank": None,
    }


def _default_ffn() -> dict[str, Any]:
    return {
        "name": "default",
        "hidden_dim": None,
        "activation": "SwiGLU",
        "bias": False,
        "dropout": 0.0,
        "gate": True,
    }


def _default_block() -> dict[str, Any]:
    # NB: BlockDef::default() sets norm eps 1e-5, unlike NormConfig::default()
    # (eps 0.0). This 1e-5 is only used when a block omits `norm:` entirely.
    return {
        "name": "default",
        "attention": _default_attention(),
        "ssm": None,
        "ffn": _default_ffn(),
        "norm": {"norm_type": "RmsNorm", "eps": 1e-5},
        "norm_position": "Pre",
        "residual": True,
        "dropout": 0.0,
    }


def _default_model() -> dict[str, Any]:
    return {
        "name": "default",
        "description": None,
        "vocab_size": 32000,
        "max_seq_len": 2048,
        "hidden_size": 768,
        "num_layers": 12,
        "block": _default_block(),
        "pattern": None,
        "embeddings": {"tie_weights": False, "dropout": 0.0, "scale": None},
        "output": {"bias": False, "norm": None},
    }


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# number: optional leading '-', digits, optional fraction, optional exponent.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# identifier: (alpha|_) then (alnum|_|-)*
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass
class _Token:
    kind: str  # 'ident' | 'number' | 'string' | 'punct' | 'eof'
    value: str
    pos: int


def _tokenize(src: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "#":  # comment to end of line
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if c in "{}[]:,":
            tokens.append(_Token("punct", c, i))
            i += 1
            continue
        if c == '"':
            m = _STRING_RE.match(src, i)
            if not m:
                raise MalError(f"unterminated string at position {i}")
            tokens.append(_Token("string", m.group(), i))
            i = m.end()
            continue
        # number: '-' followed by a digit, or a leading digit
        if c.isdigit() or (c == "-" and i + 1 < n and src[i + 1].isdigit()):
            m = _NUMBER_RE.match(src, i)
            assert m is not None
            tokens.append(_Token("number", m.group(), i))
            i = m.end()
            continue
        m = _IDENT_RE.match(src, i)
        if m:
            tokens.append(_Token("ident", m.group(), i))
            i = m.end()
            continue
        raise MalError(f"unexpected character {c!r} at position {i}")
    tokens.append(_Token("eof", "", n))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, src: str) -> None:
        self._toks = _tokenize(src)
        self._i = 0
        self.attentions: dict[str, dict[str, Any]] = {}
        self.ssms: dict[str, dict[str, Any]] = {}
        self.ffns: dict[str, dict[str, Any]] = {}
        self.blocks: dict[str, dict[str, Any]] = {}
        self.models: list[dict[str, Any]] = []

    # -- token cursor helpers --------------------------------------------

    def _peek(self) -> _Token:
        return self._toks[self._i]

    def _next(self) -> _Token:
        tok = self._toks[self._i]
        self._i += 1
        return tok

    def _at_punct(self, value: str) -> bool:
        tok = self._peek()
        return tok.kind == "punct" and tok.value == value

    def _expect_punct(self, value: str) -> None:
        tok = self._next()
        if tok.kind != "punct" or tok.value != value:
            raise MalError(f"expected {value!r}, found {tok.value!r} at {tok.pos}")

    def _expect_ident(self) -> str:
        tok = self._next()
        if tok.kind != "ident":
            raise MalError(f"expected identifier, found {tok.value!r} at {tok.pos}")
        return tok.value

    def _expect_colon(self) -> None:
        self._expect_punct(":")

    # -- scalar readers (after a ':') ------------------------------------

    def _read_int(self) -> int:
        tok = self._next()
        if tok.kind != "number" or "." in tok.value or "e" in tok.value.lower():
            raise MalError(f"expected integer, found {tok.value!r} at {tok.pos}")
        return int(tok.value)

    def _read_number(self) -> float:
        tok = self._next()
        if tok.kind != "number":
            raise MalError(f"expected number, found {tok.value!r} at {tok.pos}")
        return float(tok.value)

    def _read_bool(self) -> bool:
        tok = self._next()
        if tok.kind != "ident" or tok.value not in ("true", "false"):
            raise MalError(f"expected boolean, found {tok.value!r} at {tok.pos}")
        return tok.value == "true"

    def _read_string(self) -> str:
        tok = self._next()
        if tok.kind != "string":
            raise MalError(f"expected string, found {tok.value!r} at {tok.pos}")
        # strip quotes; the Rust side keeps the raw inner bytes verbatim.
        return tok.value[1:-1]

    # -- top level -------------------------------------------------------

    def parse(self) -> None:
        while self._peek().kind != "eof":
            kind = self._expect_ident()
            if kind == "attention":
                attn = self._parse_attention_def()
                self.attentions[attn["name"]] = attn
            elif kind == "ssm":
                ssm = self._parse_ssm_def()
                self.ssms[ssm["name"]] = ssm
            elif kind == "ffn":
                ffn = self._parse_ffn_def()
                self.ffns[ffn["name"]] = ffn
            elif kind == "block":
                block = self._parse_block_def()
                self.blocks[block["name"]] = block
            elif kind == "model":
                self.models.append(self._parse_model_def())
            else:
                raise MalError(f"unknown top-level definition {kind!r}")

    # -- attention -------------------------------------------------------

    def _parse_attention_def(self) -> dict[str, Any]:
        name = self._expect_ident()
        attn = _default_attention()
        attn["name"] = name
        self._parse_attention_body(attn)
        return attn

    def _parse_attention_body(self, attn: dict[str, Any]) -> None:
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            if key in _NUM_HEADS:
                self._expect_colon()
                attn["num_heads"] = self._read_int()
            elif key in _NUM_KV_HEADS:
                self._expect_colon()
                attn["num_kv_heads"] = self._read_int()
            elif key == "head_dim":
                self._expect_colon()
                attn["head_dim"] = self._read_int()
            elif key == "dropout":
                self._expect_colon()
                attn["dropout"] = self._read_number()
            elif key == "bias":
                self._expect_colon()
                attn["bias"] = self._read_bool()
            elif key == "window_size":
                self._expect_colon()
                attn["window_size"] = self._read_int()
            elif key == "causal":
                self._expect_colon()
                attn["causal"] = self._read_bool()
            elif key == "qk_norm":
                self._expect_colon()
                attn["qk_norm"] = self._read_bool()
            elif key == "position_encoding":
                self._expect_colon()
                attn["position_encoding"] = self._parse_position_encoding()
            else:
                raise MalError(f"unknown attention property {key!r}")
        self._expect_punct("}")

    def _parse_position_encoding(self) -> dict[str, Any] | str:
        kind = self._expect_ident()
        if kind == "rope":
            theta = 10000.0
            scaling: float | None = None
            self._expect_punct("{")
            while not self._at_punct("}"):
                key = self._expect_ident()
                self._expect_colon()
                if key in _ROPE_THETA or key == "base":
                    theta = self._read_number()
                elif key == "rope_scaling":
                    scaling = self._read_number()
                else:
                    raise MalError(f"unknown rope property {key!r}")
            self._expect_punct("}")
            return {"Rope": {"theta": theta, "scaling": scaling}}
        if kind == "alibi":
            learned_slopes = False
            self._expect_punct("{")
            while not self._at_punct("}"):
                key = self._expect_ident()
                self._expect_colon()
                if key == "slopes":
                    slopes = self._expect_ident()
                    if slopes not in ("learned", "fixed"):
                        raise MalError(f"unknown alibi slopes {slopes!r}")
                    learned_slopes = slopes == "learned"
                else:
                    raise MalError(f"unknown alibi property {key!r}")
            self._expect_punct("}")
            return {"Alibi": {"learned_slopes": learned_slopes}}
        if kind == "learned":
            max_positions = 0
            self._expect_punct("{")
            while not self._at_punct("}"):
                key = self._expect_ident()
                self._expect_colon()
                if key == "max_positions":
                    max_positions = self._read_int()
                else:
                    raise MalError(f"unknown learned property {key!r}")
            self._expect_punct("}")
            return {"Learned": {"max_positions": max_positions}}
        if kind == "none":
            return "None"
        raise MalError(f"unknown position_encoding {kind!r}")

    # -- ssm -------------------------------------------------------------

    def _parse_ssm_def(self) -> dict[str, Any]:
        name = self._expect_ident()
        ssm = _default_ssm()
        ssm["name"] = name
        self._parse_ssm_body(ssm)
        return ssm

    def _parse_ssm_body(self, ssm: dict[str, Any]) -> None:
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            self._expect_colon()
            if key in _STATE_DIM:
                ssm["state_dim"] = self._read_int()
            elif key in _CONV_KERNEL:
                ssm["conv_kernel"] = self._read_int()
            elif key == "expand":
                ssm["expand"] = self._read_int()
            elif key == "dt_rank":
                ssm["dt_rank"] = self._read_int()
            else:
                raise MalError(f"unknown ssm property {key!r}")
        self._expect_punct("}")

    # -- ffn -------------------------------------------------------------

    def _parse_ffn_def(self) -> dict[str, Any]:
        name = self._expect_ident()
        ffn = _default_ffn()
        ffn["name"] = name
        self._parse_ffn_body(ffn)
        return ffn

    def _parse_ffn_body(self, ffn: dict[str, Any]) -> None:
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            self._expect_colon()
            if key in _HIDDEN_DIM:
                ffn["hidden_dim"] = self._read_int()
            elif key == "activation":
                act = self._expect_ident()
                if act not in _ACTIVATIONS:
                    raise MalError(f"unknown activation {act!r}")
                ffn["activation"] = _ACTIVATIONS[act]
            elif key == "bias":
                ffn["bias"] = self._read_bool()
            elif key == "dropout":
                ffn["dropout"] = self._read_number()
            elif key == "gate":
                ffn["gate"] = self._read_bool()
            else:
                raise MalError(f"unknown ffn property {key!r}")
        self._expect_punct("}")

    # -- norm ------------------------------------------------------------

    def _parse_norm_config(self) -> dict[str, Any]:
        # Mirrors parse_norm_config: starts from NormConfig::default() (eps 0.0),
        # sets the type, and only overrides eps if given.
        kind = self._expect_ident()
        if kind == "none":
            return {"norm_type": "None", "eps": 0.0}
        if kind == "rmsnorm":
            norm_type = "RmsNorm"
        elif kind == "layernorm":
            norm_type = "LayerNorm"
        else:
            raise MalError(f"unknown norm config {kind!r}")
        eps = 0.0
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            self._expect_colon()
            if key in _NORM_EPS:
                eps = self._read_number()
            else:
                raise MalError(f"unknown norm property {key!r}")
        self._expect_punct("}")
        return {"norm_type": norm_type, "eps": eps}

    # -- block -----------------------------------------------------------

    def _parse_block_def(self) -> dict[str, Any]:
        name = self._expect_ident()
        block = _default_block()
        block["name"] = name
        self._parse_block_body(block)
        return block

    def _parse_block_body(self, block: dict[str, Any]) -> None:
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            if key == "attention":
                self._expect_colon()
                block["attention"] = self._parse_mixer_ref(
                    self.attentions, _default_attention, self._parse_attention_body
                )
            elif key == "ssm":
                self._expect_colon()
                block["ssm"] = self._parse_mixer_ref(
                    self.ssms, _default_ssm, self._parse_ssm_body
                )
            elif key == "ffn":
                self._expect_colon()
                block["ffn"] = self._parse_mixer_ref(
                    self.ffns, _default_ffn, self._parse_ffn_body
                )
            elif key == "norm":
                self._expect_colon()
                block["norm"] = self._parse_norm_config()
            elif key == "norm_position":
                self._expect_colon()
                pos = self._expect_ident()
                if pos not in ("pre", "post"):
                    raise MalError(f"unknown norm_position {pos!r}")
                block["norm_position"] = "Pre" if pos == "pre" else "Post"
            elif key == "residual":
                self._expect_colon()
                block["residual"] = self._read_bool()
            elif key == "dropout":
                self._expect_colon()
                block["dropout"] = self._read_number()
            else:
                raise MalError(f"unknown block property {key!r}")
        self._expect_punct("}")

    def _parse_mixer_ref(
        self,
        registry: dict[str, dict[str, Any]],
        default_factory: Any,
        body_parser: Any,
    ) -> dict[str, Any]:
        """A mixer/ffn slot: either an inline ``{ ... }`` def or a name ref."""
        if self._at_punct("{"):
            item = default_factory()
            body_parser(item)
            return item
        tok = self._next()
        if tok.kind != "ident":
            raise MalError(f"expected reference or inline def at {tok.pos}")
        if tok.value not in registry:
            # Fail loud, like the (fixed) Rust parser.
            what = {id(self.attentions): "attention", id(self.ssms): "ssm"}.get(
                id(registry), "ffn"
            )
            raise MalError(f"undefined {what} {tok.value!r}")
        return deepcopy(registry[tok.value])

    # -- model -----------------------------------------------------------

    def _parse_model_def(self) -> dict[str, Any]:
        name = self._expect_ident()
        model = _default_model()
        model["name"] = name
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            if key == "vocab_size":
                self._expect_colon()
                model["vocab_size"] = self._read_int()
            elif key in _MAX_SEQ_LEN:
                self._expect_colon()
                model["max_seq_len"] = self._read_int()
            elif key in _HIDDEN_SIZE:
                self._expect_colon()
                model["hidden_size"] = self._read_int()
            elif key in _NUM_LAYERS:
                self._expect_colon()
                model["num_layers"] = self._read_int()
            elif key == "description":
                self._expect_colon()
                model["description"] = self._read_string()
            elif key == "block":
                self._expect_colon()
                if self._at_punct("{"):
                    block = _default_block()
                    self._parse_block_body(block)
                    model["block"] = block
                else:
                    ref = self._expect_ident()
                    if ref not in self.blocks:
                        raise MalError(f"undefined block {ref!r}")
                    model["block"] = deepcopy(self.blocks[ref])
            elif key == "pattern":
                self._expect_colon()
                model["pattern"] = self._parse_pattern()
            elif key == "embeddings":
                model["embeddings"] = self._parse_embeddings()
            elif key == "output":
                model["output"] = self._parse_output()
            else:
                raise MalError(f"unknown model property {key!r}")
        self._expect_punct("}")
        return model

    def _parse_pattern(self) -> list[dict[str, Any]] | None:
        self._expect_punct("[")
        blocks: list[dict[str, Any]] = []
        while not self._at_punct("]"):
            ref = self._expect_ident()
            if ref not in self.blocks:
                raise MalError(f"pattern references undefined block {ref!r}")
            blocks.append(deepcopy(self.blocks[ref]))
            if self._at_punct(","):
                self._next()
            else:
                break
        self._expect_punct("]")
        # Rust only sets pattern when non-empty.
        return blocks or None

    def _parse_embeddings(self) -> dict[str, Any]:
        emb = {"tie_weights": False, "dropout": 0.0, "scale": None}
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            self._expect_colon()
            if key == "tie_weights":
                emb["tie_weights"] = self._read_bool()
            elif key == "dropout":
                emb["dropout"] = self._read_number()
            elif key == "scale":
                emb["scale"] = self._read_number()
            else:
                raise MalError(f"unknown embeddings property {key!r}")
        self._expect_punct("}")
        return emb

    def _parse_output(self) -> dict[str, Any]:
        out: dict[str, Any] = {"bias": False, "norm": None}
        self._expect_punct("{")
        while not self._at_punct("}"):
            key = self._expect_ident()
            if key == "bias":
                self._expect_colon()
                out["bias"] = self._read_bool()
            elif key == "norm":
                self._expect_colon()
                out["norm"] = self._parse_norm_config()
            else:
                raise MalError(f"unknown output property {key!r}")
        self._expect_punct("}")
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_mal(src: str) -> dict[str, Any]:
    """Parse MAL source and return the model dict (serde-compatible).

    Mirrors ``mal::parse_mal``: returns the (single) model definition. Raises
    :class:`MalError` if no model is defined or on any syntax/reference error.
    """
    parser = _Parser(src)
    parser.parse()
    if not parser.models:
        raise MalError("no model definition found")
    return parser.models[0]


def parse_mal_file(path: str | Path) -> dict[str, Any]:
    """Parse a ``.mal`` file and return the model dict (serde-compatible)."""
    return parse_mal(Path(path).read_text())
