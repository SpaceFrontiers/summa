# BF16 residual stream (CUDA training)

Under `training-fusion`, the whole activation stream between blocks runs in
BF16 — the same layout PyTorch autocast produces. `stream_cast` (in
`model/matmul.rs`) casts the embedding output once per forward; from there
residual adds, dropout, the FFN activation chain, and every projection
input/output stay in BF16, so the elementwise passes move half the bytes and
the promote-to-FP32 / recast kernels that used to sit at every projection
boundary disappear.

What deliberately stays FP32:

- **Norm statistics** (`Norm::forward`): a BF16 input is cast up on entry and
  back on exit — PyTorch autocast's layer-norm policy. Both casts fuse into
  the surrounding elementwise chains; a BF16 mean-of-squares over the hidden
  dimension would lose ~2 mantissa bits to sequential accumulation.
- **The attention Q/K/V path**: `qkv_proj` keeps the FP32 promotion, so RoPE
  rotates in FP32 and the fused attention kernel still sees the F16 saved
  tensors whose BF16 variant measurably inflated gradient norms (~2.5×,
  bisected earlier). Only the output projection joins the BF16 stream.
- **Parameters, parameter gradients, optimizer state** — unchanged; matmul
  weight casts already happened per call before this change.
- **Incremental decode** (`forward_hidden_with_state`): the FP32 stream is
  kept because the scan/conv step kernels are FP32-only. The residual add
  aligns a BF16 branch to the FP32 stream when the two meet
  (`TransformerBlock::residual`).

Boundary contracts this forced:

- `linear_cross_entropy` accepts a BF16 `hidden` (the chunk matmuls wanted
  BF16 operands anyway) and hands `grad_hidden` back in the activation dtype;
  the loss and the weight/bias gradients stay FP32. The fusion-IR loss output
  is pinned to FP32 — under lazy fusion a wrong gradient or output dtype only
  fails at runtime, which the plain-CUDA suite never exercises. The
  end-to-end gate for that class is
  `training_fusion_hybrid_bf16_stream_loss_and_gradients_are_finite`.
- The selective scan's inference paths (decode and prefill without saved
  states) have no BF16 kernels: the dispatcher normalizes those to FP32 at the
  boundary and returns BF16 to the caller. The single checkpointed training
  path is BF16-native. The e2e gate caught this boundary on a small hybrid
- The same gate then surfaced two attention-backward issues at small-model
  shapes: the fused probabilities kernel read a BF16 `correction` tensor as
  raw FP32 (fixed by pinning the correction product to FP32 + loud dtype
  asserts at the launch boundary), and a fork-runtime defect that displaces
  the op's writes at non-power-of-two sequence lengths (shape-guarded to the
  proven power-of-two class — see `fused-attention.md` for the full trail).

Measured (A100 40GB, retriever-100m, T1024, grad-accum 8, steps 5–10,
2026-07-17): 42,664 → **42,793 tok/s** @B20 and 43,422 → **43,787** @B26
(+0.8% end-to-end; the mamba two-thirds of the elementwise pool was already
converted by the BF16 kernel-I/O round, so this round captured the
attention/FFN third). Loss and gradient-norm trajectories match the previous
round to five decimals at B26 (loss 10.925361 vs .925367, grad_norm 1.801
identical). Batch 28 now allocates but aborts fail-loud with a non-finite
gradient norm at the 40 GB edge — the usable ceiling stays B26. The fresh
profile keeps `elemwise_fuse` at 13.9% (norm statistics stay FP32 by design)
and the bf16→f32 cast pool at 2.5% (the qkv promote and chunked
cross-entropy logit promotes remain deliberately FP32).
