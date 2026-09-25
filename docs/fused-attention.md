# Attention kernels

Summa uses Burn tensor and module operations by default. CUDA attention calls
Burn's CubeCL integration, which launches CubeK Flash Attention with FP16 inputs
and FP32 accumulation. CPU and Metal use Burn's portable attention operations.
There are no cuDNN, C++, or Python runtime dependencies.

The only custom attention component is the CUDA training autodiff boundary.
Burn's current attention API returns the output but not the per-row log-sum-exp
needed by CubeK's experimental fused backward. Materializing the full
`[batch, heads, sequence, sequence]` autodiff graph is not viable at a
4096-token context, so Summa saves Q/K/V and the output, then recomputes exact
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

CubeK's fused backward can replace the chunked backward once its forward API
exposes compatible log-sum-exp state and the end-to-end A100 benchmark wins.
Until then, using it would require an extra score/LSE pass and is not assumed to
be faster merely because it is fused.

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

## Forward LSE emission (landed)

The flash forward now runs through cubek's `launch_ref_with_lse` directly
(summa depends on cubek; burn's module op has no LSE surface), emitting the
per-row softmax log-sum-exp as a sixth saved tensor (`[batch * heads,
seq_q]`, FP32, scaled-score units, natural log, exactly `-inf` on
fully-masked rows). The cubek side (fork branch `fwd-lse`, rev b4fe978)
threads an additive `execute_with_lse` path through the batch → global →
stage layers; `BounceTile::store_row_lse` maps unit-local rows to absolute
rows through the whitebox fragment layout, and the unit owning a row's
first column writes after the cross-plane reductions synchronize the state.

The chunked backward consumes the LSE instead of recomputing softmax
statistics: `attention_softmax_stats` (a block-per-row reduction over every
score chunk plus a sync boundary) is deleted, and the probabilities kernel
computes `P = exp(score·scale − lse)` in one expression. Shapes the flash
kernel rejects fall back to burn's tensor-op attention with the LSE
recomputed from a materialized score matrix (test-scale shapes only).

Measured (A100 40GB, retriever-100m, T1024/ga8, 2026-07-17): 43,217 →
**44,044 tok/s** @B20 (+1.9%) and 44,201 → **45,077** @B26 (+2.0%), all
parity gates green — the canonical 64/64 CPU-reference test now validates
cubek's emitted LSE against CPU autodiff directly. This was steps 1–2 of
the flash-backward arc; step 3 (below) was measured and rejected.

## Tensor-core flash backward (measured, rejected)

Step 3 replaced the chunked backward's five GEMMs + materialized
scores/P/dS with true FlashAttention-style backward kernels: a `D =
rowsum(dO ⊙ O)` prepass, a query-outer dQ kernel, and a key-outer dK/dV
kernel, all bf16 cmma (m16n16k16) against fp32 accumulators, staging
operands through shared memory, recomputing `S`/`P` per tile from the
forward LSE, never materializing a `[seq, seq]` tensor. Specialized to
`head_dim == 64`, `seq % 64 == 0`, square self-attention (all production
shapes); the fusion-layer arbitration fell back to the chunked path
elsewhere. Kernels live in the cubek fork, branch `fwd-lse`, rev ee57892
(`crates/cubek-attention/src/backward/launch/tiled.rs`) with CPU-reference
tests (4/4 green on A100, n64/n128/n256, causal and dense).

Numerics were fully validated: the canonical 64/64 CPU-autodiff parity
test, the whole CUDA suite, and the e2e bf16-stream gradient tests all
passed; bench grad norms matched the chunked path to three decimals.

Measured (same box/config, 2026-07-17): 44,044 → **43,442** @B20 (−1.4%),
45,077 → **44,248** @B26 (−1.8%). Rejected. Two compounding reasons:

- The hand-rolled kernels run 4 planes (128 threads, ~25% occupancy by
  shared memory) with scalar staging loads, no cp.async double-buffering,
  and a per-tile fp32 shared-memory bounce with two plane-syncs between
  cmma stages — well below the ~80%-of-peak cuBLAS GEMMs they replace,
  which recompute nothing.
- The traffic saving the flash structure buys is small here: the chunked
  path already bounds materialization to pow2 sequence chunks, so at
  seq 1024 the recoverable bandwidth is ~1–2% of step time, not enough to
  amortize a 2–3× less efficient tensor-core inner loop.

Closing the kernel-efficiency gap needs cutlass-grade work (8-plane
128-row blocks, cp.async pipelines, vectorized staging, smem swizzling)
for a ~1–2% end-to-end ceiling — poor EV against the remaining arcs. The
summa-side dispatch was reverted; the chunked backward (with LSE reuse)
remains the production path. The kernels stay parked in the cubek fork
should long-sequence configs (seq ≥ 4k, where materialization traffic
dominates) ever matter.

## Runtime defect, second signature (state-tensor stride displacement)

The scan-kernel geometry sweep produced a cleaner signature of the same
fork-runtime defect this document tracks for attention shapes: with a
`[batch, seq, state]` atomic gradient tensor of minor width 5, every value
the kernel emits is correct but lands as if the tensor had minor stride 8
(`gpu[5t + n] == truth[8t + n]`, deterministic, identical across launch
geometries; delta/x gradients through the same kernel are clean). Widths
4/8/16/32 are clean, 2/5/6/12 corrupt, and Metal is correct at every
width, so this is CUDA-runtime tensor binding, not kernel logic or shuffle
semantics. The scan dispatch now refuses non-power-of-two state widths
loudly; if the fork runtime's binding layer is ever fixed, that assert and
this note are the two things to relax.
