"""Decoder-only Transformer matching hermes-llm/src/model.rs.

The module tree is named so that ``state_dict()`` keys are byte-identical to
the Candle ``VarMap`` names — checkpoints flow both ways with zero conversion:

    embedding.weight
    layers.{i}.attention.{q,k,v,o}_proj.weight[/bias]
    layers.{i}.feed_forward.{gate,up,down}_proj.weight[/bias]
    layers.{i}.attn_norm.weight[/bias]
    layers.{i}.ffn_norm.weight[/bias]
    final_norm.weight[/bias]
    lm_head.weight

Math parity notes (vs model.rs):
- RoPE is NeoX-style rotate-half with cos/sin duplicated across the head dim.
- Norms compute in fp32 and cast back, like the Candle impls.
- FFN activation mirrors model.rs: SiLU when SwiGLU, exact (erf) GELU otherwise.
- lm_head is never weight-tied (Candle side keeps it separate).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hermes_train.config import BlockDef, ModelDef, SsmDef

# Fused selective-scan kernels (CUDA only; `uv sync --extra mamba`)
try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _selective_scan_fn,
    )
except ImportError:
    _selective_scan_fn = None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, size: int, eps: float, use_bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if use_bias else None
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
        x = x.to(dtype) * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def make_norm(config: ModelDef, block: BlockDef) -> nn.Module:
    # NormType::None falls back to RmsNorm, matching Norm::new in model.rs
    if block.norm.norm_type == "LayerNorm":
        return LayerNorm(config.hidden_size, block.norm_eps, block.use_bias)
    return RMSNorm(config.hidden_size, block.norm_eps)


class RotaryEmbedding(nn.Module):
    """NeoX-style RoPE, matching RotaryEmbedding in model.rs."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
        # Duplicate across the head dim so cos/sin cover the full rotation
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: Tensor, k: Tensor, start_pos: int = 0
    ) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        cos = self.cos[start_pos : start_pos + seq_len].to(q.dtype)
        sin = self.sin[start_pos : start_pos + seq_len].to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelDef, block: BlockDef) -> None:
        super().__init__()
        self.num_heads = block.n_heads
        self.num_kv_heads = block.n_kv_heads
        self.head_dim = block.head_dim(config.hidden_size)
        self.dropout_p = block.dropout
        self.window_size = block.attention.window_size
        self.causal = block.attention.causal
        hidden = config.hidden_size
        kv_dim = self.num_kv_heads * self.head_dim
        bias = block.use_bias

        self.q_proj = nn.Linear(hidden, hidden, bias=bias)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=bias)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x: Tensor, rope: RotaryEmbedding, start_pos: int = 0) -> Tensor:
        bsz, seq_len, _ = x.shape

        q = (
            self.q_proj(x)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        q, k = rope(q, k, start_pos)

        # GQA: repeat KV heads
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        attn_mask = None
        is_causal = self.causal and seq_len > 1
        if self.window_size is not None and seq_len > 1:
            # Combined causal + sliding-window additive mask
            i = torch.arange(seq_len, device=x.device).unsqueeze(1)
            j = torch.arange(seq_len, device=x.device).unsqueeze(0)
            allowed = (i - j).abs() <= self.window_size
            if self.causal:
                allowed &= j <= i
            attn_mask = torch.where(allowed, 0.0, float("-inf")).to(q.dtype)
            is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: ModelDef, block: BlockDef) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = block.intermediate_size(config.hidden_size)
        bias = block.use_bias

        self.gate_proj = nn.Linear(hidden, inter, bias=bias) if block.ffn.gate else None
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)
        self.dropout = nn.Dropout(block.dropout)
        self.use_swiglu = block.use_swiglu

    def _act(self, x: Tensor) -> Tensor:
        # model.rs: silu for SwiGLU, exact (erf) GELU for everything else
        return F.silu(x) if self.use_swiglu else F.gelu(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.gate_proj is not None:
            h = self._act(self.gate_proj(x)) * self.up_proj(x)
        else:
            h = self._act(self.up_proj(x))
        return self.down_proj(self.dropout(h))


class MambaMixer(nn.Module):
    """Selective state-space (Mamba-1) mixer, reference implementation.

    Tensor names follow the mamba-ssm convention (in_proj, conv1d, x_proj,
    dt_proj, A_log, D, out_proj) under ``layers.{i}.ssm.*`` — shared with the
    Candle MambaMixer in model.rs. The scan runs in fp32 for stability.
    On CUDA with the ``mamba`` extra installed, swap in fused kernels.
    """

    def __init__(self, config: ModelDef, ssm: SsmDef) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.d_inner = ssm.expand * hidden
        self.state_dim = ssm.state_dim
        self.dt_rank = config.dt_rank(ssm)
        self.conv_kernel = ssm.conv_kernel

        self.in_proj = nn.Linear(hidden, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.conv_kernel,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.state_dim, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, hidden, bias=False)

        # S4D-real init: A = -[1..N] per row, stored as log
        a = torch.arange(1, self.state_dim + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt bias init so softplus(bias) lands in [1e-3, 1e-1] (mamba-ssm default)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        )
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

    def forward(self, x: Tensor) -> Tensor:
        if _selective_scan_fn is not None and x.is_cuda:
            return self._forward_fused(x)
        return self._forward_reference(x)

    def _forward_reference(self, x: Tensor) -> Tensor:
        _, seq_len, _ = x.shape
        in_dtype = x.dtype

        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)

        # Depthwise causal conv over time
        xs = xs.transpose(1, 2)  # [B, di, L]
        xs = self.conv1d(F.pad(xs, (self.conv_kernel - 1, 0)))[..., :seq_len]
        xs = F.silu(xs.transpose(1, 2))  # [B, L, di]

        # Input-dependent Δ, B, C — scan in fp32
        x_dbl = self.x_proj(xs).float()
        delta, b_mat, c_mat = torch.split(
            x_dbl, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(
            F.linear(delta, self.dt_proj.weight.float(), self.dt_proj.bias.float())
        )  # [B, L, di]
        a = -torch.exp(self.A_log.float())  # [di, N]
        xs_f = xs.float()

        delta_a = torch.exp(delta.unsqueeze(-1) * a)  # [B, L, di, N]
        delta_bx = delta.unsqueeze(-1) * b_mat.unsqueeze(2) * xs_f.unsqueeze(-1)

        h = x.new_zeros(x.shape[0], self.d_inner, self.state_dim, dtype=torch.float32)
        ys = []
        for t in range(seq_len):
            h = delta_a[:, t] * h + delta_bx[:, t]
            ys.append((h * c_mat[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)  # [B, L, di]

        y = y + xs_f * self.D.float()
        y = (y.to(in_dtype)) * F.silu(z)
        return self.out_proj(y)

    def _forward_fused(self, x: Tensor) -> Tensor:
        """Tri Dao's fused selective-scan kernel (CUDA, `mamba` extra).

        Mirrors mamba_simple.py: Δ softplus, D skip and the silu(z) gate all
        happen inside the kernel. Math-equivalent to the reference path.
        """
        _, seq_len, _ = x.shape

        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)

        xs = xs.transpose(1, 2)  # [B, di, L]
        xs = self.conv1d(F.pad(xs, (self.conv_kernel - 1, 0)))[..., :seq_len]
        xs = F.silu(xs)  # stay [B, di, L] for the kernel

        x_dbl = self.x_proj(xs.transpose(1, 2))  # [B, L, R+2N]
        delta, b_mat, c_mat = torch.split(
            x_dbl, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        # Δ projection without bias — the kernel adds delta_bias + softplus
        delta = (delta @ self.dt_proj.weight.t()).transpose(1, 2).contiguous()

        y = _selective_scan_fn(
            xs,  # u [B, di, L]
            delta,  # [B, di, L]
            -torch.exp(self.A_log.float()),  # A [di, N]
            b_mat.transpose(1, 2).contiguous(),  # B [B, N, L]
            c_mat.transpose(1, 2).contiguous(),  # C [B, N, L]
            self.D.float(),
            z=z.transpose(1, 2).contiguous(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )  # [B, di, L]
        return self.out_proj(y.transpose(1, 2))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelDef, block: BlockDef) -> None:
        super().__init__()
        # Attribute name defines the state_dict prefix: attention.* or ssm.*
        if block.ssm is not None:
            self.ssm: MambaMixer | None = MambaMixer(config, block.ssm)
            self.attention = None
        else:
            self.ssm = None
            self.attention = MultiHeadAttention(config, block)
        self.feed_forward = FeedForward(config, block)
        self.attn_norm = make_norm(config, block)
        self.ffn_norm = make_norm(config, block)
        self.pre_norm = block.norm_position == "Pre"
        self.use_residual = block.residual

    def _mix(self, x: Tensor, rope: RotaryEmbedding, start_pos: int) -> Tensor:
        if self.ssm is not None:
            return self.ssm(x)
        return self.attention(x, rope, start_pos)

    def forward(self, x: Tensor, rope: RotaryEmbedding, start_pos: int = 0) -> Tensor:
        if self.pre_norm:
            h = self._mix(self.attn_norm(x), rope, start_pos)
            x = x + h if self.use_residual else h
            h = self.feed_forward(self.ffn_norm(x))
            return x + h if self.use_residual else h
        else:
            h = self._mix(x, rope, start_pos)
            x = self.attn_norm(x + h if self.use_residual else h)
            h = self.feed_forward(x)
            return self.ffn_norm(x + h if self.use_residual else h)


class Transformer(nn.Module):
    def __init__(self, config: ModelDef) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared RoPE table: all attention blocks must agree (mirrors model.rs)
        attn_blocks = [
            config.block_for_layer(i)
            for i in range(config.num_layers)
            if not config.block_for_layer(i).is_ssm
        ]
        if attn_blocks:
            head_dims = {b.head_dim(config.hidden_size) for b in attn_blocks}
            thetas = {b.rope_theta for b in attn_blocks}
            if len(head_dims) > 1 or len(thetas) > 1:
                raise ValueError(
                    "all attention blocks must share head_dim and rope theta"
                )
            rope_head_dim, rope_theta = head_dims.pop(), thetas.pop()
        else:
            rope_head_dim, rope_theta = 2, 10000.0  # pure-SSM: unused

        self.layers = nn.ModuleList(
            TransformerBlock(config, config.block_for_layer(i))
            for i in range(config.num_layers)
        )
        self.final_norm = make_norm(config, config.block_for_layer(0))
        # Tied embeddings: no lm_head parameter at all (matches the Candle
        # side — tied checkpoints contain no lm_head.weight tensor)
        self.lm_head = (
            None
            if config.embeddings.tie_weights
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        self.rope = RotaryEmbedding(rope_head_dim, config.max_seq_len, rope_theta)

    def forward(self, input_ids: Tensor, start_pos: int = 0) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope, start_pos)
        x = self.final_norm(x)
        if self.lm_head is None:
            return F.linear(x, self.embedding.weight)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten().long())
