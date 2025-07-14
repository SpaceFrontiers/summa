# Attention kernels

Hermes uses Burn tensor and module operations by default. CUDA attention calls
Burn's CubeCL integration, which launches CubeK Flash Attention with FP16 inputs
and FP32 accumulation. CPU and Metal use Burn's portable attention operations.
There are no cuDNN, C++, or Python runtime dependencies.

The only custom attention component is the CUDA training autodiff boundary.
Burn's current attention API returns the output but not the per-row log-sum-exp
needed by CubeK's experimental fused backward. Materializing the full
`[batch, heads, sequence, sequence]` autodiff graph is not viable at a
4096-token context, so Hermes saves Q/K/V and the output, then recomputes exact
probabilities in fixed query-row chunks during backward. The chunks use Burn's
CubeCL matmuls and reductions; no scalar attention kernel is maintained here.

Grouped-query K/V heads are expanded immediately before the attention operation.
Autodiff reduces their gradients back to the compact head count. This is linear
in sequence length for the current 2:1 head ratio.

The kernel policy is deliberately narrow:

- use Burn/CubeK implementations whenever they support the required operation;
- keep a local kernel boundary only for a measured throughput or peak-memory win;
- compare forward output and Q/K/V gradients with the Burn reference; and
- gate changes with steady-state A100 tokens/s, utilization, power, and peak
  memory at the production shape (batch 2, sequence 4096).

FlashAttention-2 is the relevant algorithm family for the SM80 A100.
FlashAttention-3 targets Hopper; FlashAttention-4 targets Hopper and Blackwell.
Their hardware-specific mechanisms are not emulated on Ampere.

CUDA training enables Burn's lazy fusion for the ordinary elementwise and
reduction graph. Attention and selective scan are explicit custom-operation
boundaries, so their kernels are scheduled on the same stream without exposing
their internals to the fusion planner.

## Fused backward probabilities

The chunked attention backward previously materialized each score chunk and
ran mask, softmax, dtype casts, and the score-gradient elementwise as ~8
separate full `[batch, heads, rows, seq]` passes. Two kernels replace them
(`attention_softmax_stats` + `attention_backward_probabilities_kernel` in
`cube_attention.rs`): a per-row online-softmax statistics pass that iterates
only the causally-visible prefix (the bound replaces any mask tensor, the
softmax scale is folded in), and a single pass emitting both the
probabilities and `P ⊙ (dP − correction) · scale` directly in BF16 for the
following tensor-core matmuls. Positions past the causal bound are exact
zeros. The op crosses the lazy-fusion boundary through the same CustomOpIr
bridge as the scan and cross-entropy ops.

Measured (A100, retriever-100m, T1024/ga8, 2026-07-17): 37,029 → 39,253
tok/s at batch 16 (+6.0%) and 38,164 → 40,607 at batch 20 (+6.4%), with the
`cuda_attention_backward_matches_cpu_reference_for_causal_gqa` autodiff
parity test green and identical loss trajectories. The standalone causal-mask
kernel and the 1.9 ms score casts no longer appear in the profile.
