# Fused attention training

Hermes uses one attention abstraction for inference and training. CPU and Metal
retain a tensor-op reference. CUDA uses CubeCL only; there is no native cuDNN,
C++, or Python dependency.

CubeCL 0.10 provides the accelerated Flash Attention forward used here, but not
a fused training backward. A materialized autodiff graph retains multiple
`[batch, heads, sequence, sequence]` tensors per layer at a 4096-token context.
Hermes therefore supplies a bounded-memory backward behind the same
`AttentionBackend` trait.

The CUDA path:

- makes BHSD inputs contiguous once and stages Q/K/V as FP16;
- uses CubeCL's accelerated causal Flash Attention with FP32 accumulation;
- saves low-precision Q/K/V/output, but no attention matrix;
- recomputes probabilities in fixed query-row chunks using CubeCL's accelerated
  matmuls, so the A100 uses Tensor Cores in backward as well as forward; and
- returns gradients in the backend's default relaxed-FP32 dtype.

The backward is exact but is not presented as FlashAttention-2 backward: it
trades additional probability recomputation and launches for a small,
sequence-bounded live score buffer. This is substantially more efficient than
the former scalar per-row kernel while remaining portable Rust/CubeCL code.

Grouped-query K/V heads are expanded immediately before the custom operation;
autodiff reduces their gradients back to the compact head count. This is a
small linear copy for the current 2:1 head ratio, not a quadratic allocation.

FlashAttention-2 is the relevant algorithm family for the SM80 A100.
FlashAttention-3 targets Hopper, while FlashAttention-4 targets Hopper and
Blackwell; their hardware-specific mechanisms are not emulated on Ampere.

Correctness tests compare the custom forward and Q/K/V gradients with the CPU
reference for causal grouped-query attention. The CUDA smoke gate runs the same
parity check on the A100 before a corpus launch.
