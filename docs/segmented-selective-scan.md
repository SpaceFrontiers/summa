# Segment-parallel selective scan

The Mamba selective scan is a linear recurrence `h_t = α_t·h_{t-1} + β_t` with
a diagonal state transition, so transitions compose associatively: a run of
timesteps collapses to one `(decay, partial)` pair. The CUDA/Metal training
kernels exploit this by cutting each sequence at the existing checkpoint
boundaries (`CHECKPOINTED_SCAN_INTERVAL = 32`) and running every segment
concurrently instead of walking the whole sequence serially per scan.

## Forward (training, `save_states` path)

1. `selective_scan_forward_segment_partials` — one thread per
   `(batch, segment, channel)`: scans its segment from a zero state, emitting
   the segment's `partial` state and `decay = exp(A·Σdt)` (the exact product
   of per-step decays for a diagonal `A`).
2. `selective_scan_forward_segment_carry` — one thread per
   `(batch, channel, n)`: serially folds the per-segment transitions into the
   state _entering_ each segment. Touches `segments × state_dim` values per
   scan — microseconds.
3. `selective_scan_forward_segment_apply` — same grid as (1): re-runs each
   segment from its stitched entering state, writing outputs, per-segment
   checkpoints (the same tensor backward always consumed), and the final
   state.

## Backward

The adjoint recurrence `adj_{t-1} = (adj_t + dy_t·C_t)·α_t` is linear in
`adj`, so the same trick applies right-to-left:
`selective_scan_backward_segment_partials` + `_carry` stitch the adjoint
across segments, then `selective_scan_backward_segmented` launches one block
per `(batch, channel-tile, segment)` — the fused gradient kernel with the
sequence walk replaced by a single segment and the incoming adjoint read from
the carry. Gradients for `A`/`D` accumulate through the pre-existing atomics.

Inference decode, prefill without saved states, and the small-batch
full-state path (`checkpoint_interval == 1`) keep the serial/parallel/step
kernels.

The segment kernels run with CubeCL fast-math (`ReducedPrecision | NotNaN |
NotInf`), which lowers `exp` to the hardware `__expf`; state loops are
unrolled so the 16-wide recurrent state stays in registers.

## Measured (A100 40GB, retriever-100m, B16/T1024/ga8, steps 5–8, 2026-07-16)

- Serial baseline: forward 2.13 ms, fused backward 3.99 ms per layer call;
  26,362 tok/s end-to-end (HEAD), 28,786 on the padded-vocab tree.
- Naive segment kernels (accurate exp, runtime-indexed state): forward
  2.38 ms, backward 4.78 ms — _slower_; the two-pass structure doubles the
  per-element exp and local-memory cost, which dominates.
- With fast-math exp + register-resident state: **29,530 tok/s @B16 /
  30,487 @B20** on the same tree (CubeK matmuls), and **34,139 / 34,984**
  combined with the autotuned cuBLAS GEMM dispatch
  (see `cublas-gemm-dispatch.md`).
- GPU parity suites pass on CUDA and Metal, including the ragged tail-segment
  case (seq 37, interval 32).

Known limits: loop-unrolling the _serial_ kernels does nothing (they are
occupancy-bound, not local-memory-bound), and a 32-wide unroll of the fused
backward regresses 2.8× — both measured and rejected. Shrinking the segment
length is also measured and rejected (interval 16 → −2.3% end-to-end,
8 → −8.3%): block-level parallelism is already sufficient, so further gains
must come from removing the per-step barrier + serial walk inside the block.

## Measured and rejected: warp-cooperative backward

A lanes-over-time rewrite (one warp per `(batch, channel, segment)`,
Kogge–Stone shuffle scans for the state rebuild and the adjoint, channel-major
shared staging for coalescing, block-level `grad_B`/`grad_C` reduction before
one atomic per cell) passed all CUDA/Metal parity suites but ran **1.85×
slower** than the segmented kernel (7.51 ms vs 4.06 ms per call; end-to-end
40,607 → 36,592 tok/s @B20). Halving the channel tile to shrink the ~41 KB
shared footprint (occupancy hypothesis) changed nothing (36,453). The
per-state-lane shuffle chains and staging round-trips cost more than the
segmented kernel's one barrier per step at this problem size
(`d_inner 1024 × N 16 × segment 32`). Together with the interval sweep and
the unroll experiment, this pins the segmented backward as the local optimum
for f32 traffic; the follow-up that did land is BF16 kernel I/O below.

A fourth rewrite was also measured and rejected (2026-07-17): moving the
per-thread rebuilt-state array (`segment_len` floats, runtime-indexed local
memory) into shared memory. Parity held, but end-to-end throughput dropped
2.0% (44,193 → 43,321 tok/s @B26): the 36 KB shared footprint halves
resident blocks per SM (8 → 4), and the kernel is latency-bound — occupancy
buys more than the spill traffic costs. The local array stays.

Fifth and final variant (2026-07-17): halving the per-step barriers (one
`sync_cube` per pair of reverse steps, quad-buffered term slabs,
occupancy-neutral) measured flat to slightly negative (44,193 → 44,073
tok/s @B26). With unrolling, warp-cooperative lanes, shorter segments,
shared-state storage, and barrier batching all measured and rejected, the
segmented backward is at its local optimum for this work decomposition;
further gains require a ground-up redesign of the parallelization (e.g. a
tri-dao-style chunked backward), not tuning.

## BF16 kernel I/O (landed)

The segment kernels (and the depthwise conv) are generic over the sequence
dtype: `x`, `B`, `C`, `delta_raw`, `grad_y` are read in BF16 and `y`,
`grad_delta`, `grad_x` are emitted in BF16, while the recurrent state,
`A`/`D`, checkpoints, carries, and parameter gradients stay f32 (compute is
f32 in registers either way). `grad_B`/`grad_C` accumulate through f32
atomics and are cast to the sequence dtype on the way out so autodiff
composes without dtype mismatches — the fusion IR builder rejects a f32
gradient flowing into a BF16 slice backward at runtime, which the end-to-end
gate caught (the plain-CUDA test suite does not exercise lazy fusion).
The Mamba projections feed the chain via `linear_low_precision`, so the
promote-to-f32 / recast passes around every SSM layer disappear. Non-training
paths (decode, prefill-without-states, the small-batch full-state path) are
f32-only and guarded with loud asserts.

Measured (same A100 setup, 2026-07-17): 39,253 → **41,237 tok/s** @B16 and
40,607 → **42,664** @B20 (+5.1%), peak memory **−7.3 GB** (29.6 GB @B20),
loss and gradient-norm trajectories on the established curve. Parity:
`test_cubecl_selective_scan_bf16_io_matches_f32_reference` compares the BF16
path against the f32 CPU reference over bf16-pre-quantized inputs with
bounded dynamics (strictly negative A), so the bound measures kernel
arithmetic rather than the quantization of a growing state.
