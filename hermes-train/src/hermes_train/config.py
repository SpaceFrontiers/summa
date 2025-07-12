"""Model configuration mirroring the MAL `ModelDef` (hermes-llm/src/mal/mod.rs).

Configs are produced by `hermes-llm export --model <preset|file.mal>` and are
plain serde JSON. Field names, defaults and computed properties must stay in
lockstep with the Rust side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AttentionDef:
    name: str = "default"
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    dropout: float = 0.0
    bias: bool = False
    # serde-externally-tagged: {"Rope": {"theta": ..., "scaling": ...}} | "None" | ...
    position_encoding: dict[str, Any] | str = field(
        default_factory=lambda: {"Rope": {"theta": 10000.0, "scaling": None}}
    )
    window_size: int | None = None
    causal: bool = True
    # Per-head RMSNorm on Q/K before RoPE (OLMo2/Gemma-style stabilizer)
    qk_norm: bool = False


@dataclass
class NormConfig:
    norm_type: str = "RmsNorm"  # RmsNorm | LayerNorm | None
    eps: float = 1e-5


@dataclass
class SsmDef:
    """Selective state-space (Mamba) mixer definition."""

    name: str = "default"
    state_dim: int = 16
    conv_kernel: int = 4
    expand: int = 2
    dt_rank: int | None = None


@dataclass
class FfnDef:
    name: str = "default"
    hidden_dim: int | None = None
    activation: str = "SwiGLU"  # SwiGLU | GELU | SiLU | ReLU | GELUNew | GELUTanh
    bias: bool = False
    dropout: float = 0.0
    gate: bool = True


@dataclass
class BlockDef:
    name: str = "default"
    attention: AttentionDef = field(default_factory=AttentionDef)
    ssm: SsmDef | None = None  # when set, the block is a Mamba block
    ffn: FfnDef = field(default_factory=FfnDef)
    norm: NormConfig = field(default_factory=NormConfig)
    norm_position: str = "Pre"  # Pre | Post
    residual: bool = True
    dropout: float = 0.0

    @property
    def is_ssm(self) -> bool:
        return self.ssm is not None

    # Per-block computed properties — mirror BlockDef impl in mal/mod.rs

    @property
    def n_heads(self) -> int:
        return self.attention.num_heads or 12

    @property
    def n_kv_heads(self) -> int:
        return self.attention.num_kv_heads or self.n_heads

    def head_dim(self, hidden_size: int) -> int:
        return self.attention.head_dim or hidden_size // self.n_heads

    def intermediate_size(self, hidden_size: int) -> int:
        return self.ffn.hidden_dim or hidden_size * 4

    @property
    def use_bias(self) -> bool:
        return self.ffn.bias or self.attention.bias

    @property
    def norm_eps(self) -> float:
        return self.norm.eps if self.norm.eps > 0.0 else 1e-5

    @property
    def rope_theta(self) -> float:
        pe = self.attention.position_encoding
        if isinstance(pe, dict) and "Rope" in pe:
            return float(pe["Rope"]["theta"])
        return 10000.0

    @property
    def use_swiglu(self) -> bool:
        return self.ffn.activation == "SwiGLU"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlockDef:
        ssm = d.get("ssm")
        return cls(
            name=d.get("name", "default"),
            attention=AttentionDef(**d.get("attention", {})),
            ssm=SsmDef(**ssm) if ssm else None,
            ffn=FfnDef(**d.get("ffn", {})),
            norm=NormConfig(**d.get("norm", {})),
            norm_position=d.get("norm_position", "Pre"),
            residual=d.get("residual", True),
            dropout=d.get("dropout", 0.0),
        )


@dataclass
class EmbeddingsConfig:
    tie_weights: bool = False
    dropout: float = 0.0
    scale: float | None = None


@dataclass
class OutputConfig:
    bias: bool = False
    norm: NormConfig | None = None


@dataclass
class ModelDef:
    name: str = "default"
    description: str | None = None
    vocab_size: int = 32000
    max_seq_len: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    block: BlockDef = field(default_factory=BlockDef)
    # Heterogeneous layer pattern, repeated cyclically (overrides `block`)
    pattern: list[BlockDef] | None = None
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def block_for_layer(self, i: int) -> BlockDef:
        """Block definition for layer i — mirrors ModelDef::block_for_layer."""
        if self.pattern:
            return self.pattern[i % len(self.pattern)]
        return self.block

    def dt_rank(self, ssm: SsmDef) -> int:
        """Effective Δ rank (paper default ceil(hidden/16))."""
        return ssm.dt_rank or -(-self.hidden_size // 16)

    # ------------------------------------------------------------------
    # Computed properties — mirror ModelDef impl in mal/mod.rs exactly
    # ------------------------------------------------------------------

    @property
    def n_heads(self) -> int:
        return self.block.attention.num_heads or 12

    @property
    def n_kv_heads(self) -> int:
        return self.block.attention.num_kv_heads or self.n_heads

    @property
    def head_dim(self) -> int:
        return self.block.attention.head_dim or self.hidden_size // self.n_heads

    @property
    def intermediate_size(self) -> int:
        return self.block.ffn.hidden_dim or self.hidden_size * 4

    @property
    def dropout(self) -> float:
        return self.block.dropout

    @property
    def use_bias(self) -> bool:
        return self.block.ffn.bias or self.block.attention.bias

    @property
    def norm_eps(self) -> float:
        return self.block.norm.eps if self.block.norm.eps > 0.0 else 1e-5

    @property
    def rope_theta(self) -> float:
        pe = self.block.attention.position_encoding
        if isinstance(pe, dict) and "Rope" in pe:
            return float(pe["Rope"]["theta"])
        return 10000.0

    @property
    def use_swiglu(self) -> bool:
        return self.block.ffn.activation == "SwiGLU"

    # ------------------------------------------------------------------
    # JSON I/O (serde-compatible)
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelDef:
        pattern = d.get("pattern")
        return cls(
            name=d.get("name", "default"),
            description=d.get("description"),
            vocab_size=d["vocab_size"],
            max_seq_len=d["max_seq_len"],
            hidden_size=d["hidden_size"],
            num_layers=d["num_layers"],
            block=BlockDef.from_dict(d.get("block", {})),
            pattern=[BlockDef.from_dict(b) for b in pattern] if pattern else None,
            embeddings=EmbeddingsConfig(**d.get("embeddings", {})),
            output=OutputConfig(
                bias=d.get("output", {}).get("bias", False),
                norm=(
                    NormConfig(**d["output"]["norm"])
                    if d.get("output", {}).get("norm")
                    else None
                ),
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelDef:
        with open(path) as f:
            return cls.from_dict(json.load(f))
