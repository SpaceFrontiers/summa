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
the unroll experiment, this pins the segmented backward as the local optimum;
further scan gains would need a different decomposition (e.g. BF16 kernel
I/O to halve traffic) rather than more parallelism.
