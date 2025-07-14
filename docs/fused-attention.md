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

## Shape guard on the fused probabilities op

The fused probabilities custom op only runs when the chunk row count and the
sequence length are both powers of two — the class every production shape
belongs to (T = 1024, backward chunk = 2048). Other shapes take a tensor-op
branch inside `chunked_attention_backward` (masked softmax + the same
correction algebra in FP32, quantized once by the matmul input casts).

Why: the BF16-residual-stream end-to-end gate (`hybrid_tiny`, seq 48) caught
NaN gradients whose source took a long bisect — loss finite, all parameter
gradients NaN, nondeterministic onset. The trail, for the record:

- CPU recompute from the very buffers the kernel reads (scores, stats) is
  finite and correct, yet the op's output rows land displaced (`row r`'s
  values written at `32·r + c + 16` for a 48-column chunk) with whole
  regions unwritten — recycled pool garbage then reads as NaN once the pool
  fills with poisoned gradients.
- `compute-sanitizer --tool memcheck`: zero invalid accesses.
- The compiled CUDA source of the kernel (via `CUBECL_DEBUG_LOG`) is
  provably correct — dense scalar indexing, runtime scalars, clamped writes.
- Failures reproduce only at non-power-of-two sequence lengths (40, 48, 96);
  32, 64, and every production shape pass a strict CPU-reference parity
  sweep repeatedly.

A correct kernel whose writes land displaced points at the fork's
multi-stream fusion runtime (the run that first tripped also crashed with
burn-fusion's "Ordering is bigger than operations"), not at the kernel or the
bridge — unresolved upstream, tracked for the cubecl fork. Until then the
guard keeps the fast path exactly where it is proven, and
`cuda_attention_backward_matches_cpu_reference_for_small_shapes` pins the
fallback branch (gradient tolerance 0.08 there: BF16 tensor-core gradient
GEMMs at small odd-K shapes reach 0.0403 deterministically; the canonical
64/64 test stays at 0.02). Sequence length 96 is excluded from that sweep:
the runtime defect produces run-to-run varying gradients there on both
branches (query 0.024/0.051, value 0.131 across processes), so the shape
gates nothing but the runtime lottery.

Two hardening changes shipped with the guard: the backward kernels assert
their FP32 input dtypes at the launch boundary (a mismatched buffer was
previously reinterpreted bit-for-bit — the BF16 stream's first failure mode),
and the elementwise kernel takes an explicit element count instead of
trusting `Tensor::len()`.
