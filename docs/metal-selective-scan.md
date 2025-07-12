# Fused Metal selective-scan kernel for Mamba inference

## Problem

`MambaMixer::forward_with_state` (hermes-llm/src/model.rs) runs the selective
scan as a per-timestep Rust loop of ~8 small Candle ops (`narrow`, `broadcast_mul`,
`exp`, `mul`, `sum`, …). On CUDA/CPU this is merely suboptimal; on Metal it is
pathological:

- Every op is a separate command-buffer dispatch (~µs each) for ~ns of math.
- Ops without native Metal kernels (or with strided-view restrictions — see the
  `.contiguous()` workarounds already in `forward_with_state`) silently fall
  back to CPU, forcing a synchronous GPU→CPU→GPU round-trip per op.

Measured on retriever-100m (115M hybrid, 16 Mamba layers, M-series Mac,
2026-07-14): Metal decode 7+ min / 40 tokens with 122% CPU (the fallback tell);
pure CPU decode ≈ 0.5 s / 10 tokens. ~100× inversion.

## Approach

One custom Metal compute kernel executes the **entire scan** (all timesteps,
all channels, state resident on-device) in a **single dispatch** per layer
call. This is the same parallelization as the CUDA `selective_scan` kernel
that makes training fast:

- Thread grid: one thread per `(batch, channel)` pair — `B × d_inner` threads.
- Each thread keeps its `h[N]` state (N = state_dim, 16 for retriever-100m) in
  registers, iterates `t = 0..L` sequentially (the scan's inherent data
  dependence), and writes `y[b, t, c]` per step plus the final `h` at the end.
- Per step, per thread: `dA = exp(Δ·A[c])`, `h = h·dA + Δ·B[t]·x`,
  `y = Σ h·C[t] + D[c]·x` — all in fp32 registers.

Decode (`L = 1`) collapses from ~10 dispatches + potential CPU round-trips per
layer to exactly one dispatch; prefill fuses `L × ~10` dispatches into one.

## Candle integration

Candle 0.11's custom-op surface is `CustomOp1/2/3` — at most **3 tensor
inputs** and **1 output**. The scan needs 7 inputs and 2 outputs, so we pack:

| slot | contents                           | shape                  |
| ---- | ---------------------------------- | ---------------------- |
| in 1 | `concat(xs, Δ)` last-dim           | `[B, L, 2·di]`         |
| in 2 | `concat(Bmat, Cmat)` last-dim      | `[B, L, 2·N]`          |
| in 3 | `concat(A.flat, D, h0.flat)` 1-D   | `[di·N + di + B·di·N]` |
| out  | `concat(y.flat, h_final.flat)` 1-D | `[B·L·di + B·di·N]`    |

The wrapper (`MetalSelectiveScan` implementing `CustomOp3`) carries the scalar
dims (`b, l, di, n`) and splits/reshapes the output. Packing cost is three
`Tensor::cat` calls — native Metal, negligible next to what it replaces.

`cpu_fwd` is implemented as a straight port of the reference loop, so the op
is correct on any backend (repo rule: SIMD/GPU kernels always ship with a
scalar fallback). `metal_fwd` encodes the fused kernel.

## Dispatch policy (fail loud)

`MambaMixer::forward_with_state` uses the fused op when the tensor lives on a
Metal device (compiled only under `feature = "metal"`); otherwise the existing
reference loop. If the Metal pipeline fails to build at first use, we
`log::warn!` with the compile error and fall back to the reference loop —
degraded, but stated loudly, and generation still works.

The kernel source (MSL) is embedded via `include_str!` and compiled once per
process into a `MTLComputePipelineState` cached in a `OnceLock`.

## Numerics

fp32 throughout (serve weights are fp32). `exp` uses Metal's fast-math
`exp()`; parity tolerance vs the reference scan is `max_abs_diff / max_abs`
< 1e-4 (the CUDA fused kernel showed 4.7e-3 relative in bf16 training; fp32
inference is far tighter).

## Testing

- **Parity (CPU)**: `cpu_fwd` vs the existing reference loop on random
  tensors — shapes covering `L = 1` (decode), `L > 1` (prefill), `B > 1`,
  and non-zero `h0` (resumed state).
- **Parity (Metal)**: same comparison with the tensors on a Metal device;
  `#[cfg(feature = "metal")]`, runs on Apple-silicon dev machines and is a
  no-op elsewhere.
- **End-to-end**: `hermes-llm generate` on retriever-100m weights on Metal
  must produce identical tokens to CPU at temperature 0 and drop from
  ~10 s/token to interactive speed. Perf numbers go in the PR description
  (repo rule: perf claims are measured).

## Follow-up: depthwise conv1d (measured, then fused)

Profiling after the scan fusion (macOS `sample`, retriever-100m decode) showed
98% of CPU time in `conv1d_single_group`: candle lowers grouped conv1d to one
conv **per group**, and the Mamba depthwise conv has `groups == d_inner ==
1024` → ~16k dispatches per decoded token across 16 layers. A second kernel
(`depthwise_conv1d_f32`, bias folded in, same thread layout) replaces it with
one dispatch per layer call.

## Non-goals

- Batched-decode serving and CUDA inference fusion (tracked separately; the
  CUDA path can reuse the same packing wrapper with a CUDA kernel later).
- Training on Metal. Kernel is forward-only; training stays on CUDA/PyTorch.