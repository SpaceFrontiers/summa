# Fused chunked cross-entropy (GPU)

The chunked output-projection loss keeps memory bounded by processing the
`[tokens, padded_vocab]` logits in chunks, but the reference tensor-op
implementation costs ~18 full-logit passes per chunk on GPU (bias add, padded
mask `slice_assign`, softmax's max/sub/exp/sum/div chain, scatter, scale —
several of them unvectorized because the odd 50,277-column mask offset breaks
contiguity analysis). At `[5120, 50304]` f32 chunks that was ~400 ms per
optimizer step on an A100 — the single largest elementwise cost in training.

`CubeBackend` now implements `LinearCrossEntropyBackend` with two row-wise
kernels instead (`hermes-llm/src/model/linear_cross_entropy.rs`, `mod gpu`):

- `ce_row_statistics` — one block per row computes online log-sum-exp
  `(max, sum)` and captures the target logit in a single pass over the
  _logical_ vocabulary. Padded columns are never read, so they carry exactly
  zero probability with no mask tensor at all; the bias add is folded into
  the same pass.
- `ce_row_gradient` — one flat pass emitting
  `(softmax(x) − onehot(target)) · scale`, with padded columns written as
  exact zeros and the loss scale folded in.

Forward loss per chunk = one producing matmul + one statistics pass + a
`[rows]`-sized epilogue; backward = matmul + statistics + gradient pass +
the two gradient matmuls. The CPU/NdArray reference path is unchanged and
remains the parity oracle: `gpu_fused_loss_and_gradients_match_cpu_reference`
checks loss and all three gradients on the GPU backend against it (both bias
modes, padding narrower than the 256-thread reduction so empty lanes are
exercised, chunked offsets), and asserts padded gradients are exactly zero.

Numerics: statistics accumulate in f32 with the standard online-softmax
update; empty reduction lanes carry `(-inf, 0)` and combine through a guarded
merge so they contribute exact zeros rather than NaN. The chunk matmuls run
on BF16 tensor cores on CUDA (mirroring `matmul_2`), so the CUDA parity test
bounds against the f32 CPU reference at 1e-2 while Metal (f32 matmuls) lands
at ~1e-7.

Measured on the A100 stack (retriever-100m, T1024/ga8, steps 9-10,
2026-07-16): 34,139 → **37,029 tok/s** at batch 16 (+8.5%) and 34,984 →
**38,164 tok/s** at batch 20 (+9.1%), with identical loss and gradient-norm
trajectories; the former ~400 ms/step of width-1 f32 logit passes no longer
appears among the top profile kernels.
