# Native cuBLAS GEMM dispatch: proof results and verdict

Investigation of routing Burn/CubeCL BF16 matmuls to native cuBLAS on CUDA,
aimed at closing the training-throughput gap against PyTorch. All numbers
measured 2026-07-16 on a GCP `a2-highgpu-1g` (A100-SXM4 40GB, driver 580,
CUDA 12.9), `retriever-100m` (115.8M params), batch 16 × seq 1024 ×
grad-accum 8, steady-state steps 5–8.

## Prototype

- CubeCL fork (`ppodolsky/cubecl` @ `463c2952`, fast-forward of the production
  `73bd0ea3` pin): asynchronous cuBLAS BF16 GEMM server dispatch
  (`GemmDescriptor`/`GemmMatrix` server API), hardened contracts (rejects
  foreign-stream outputs, buffer overlap, non-empty zero-K problems).
- Burn fork (production pin `bd6e8fa2f` + BF16 dispatch in
  `burn-cubecl::kernel::matmul` + zero-contract fallback + backport of
  upstream `f31e7513a` "drop from foreign stream drains home stream").

## Proof order results (all passed on A100)

1. BF16 numerical matrix — padding, binding offsets, all transpose pairs,
   regular/zero-stride batches, async error propagation: **pass**
   (`cublas_gemm_check`).
2. Layout correctness matrix (transpose × batch × broadcast): **pass**.
3. Three-stream producer → GEMM → consumer ordering and Burn foreign-thread
   drop ordering: **pass**.
4. Burn BF16 linear forward and `dx`/`dW`/`db` under CUDA + fusion + autotune
   - autodiff: **pass** (non-uniform upstream gradient).
5. Async dispatch: 7.5 µs/call CPU-side enqueue over 10,000 calls with a
   single final drain — no per-GEMM host synchronization.

## End-to-end verdicts

**Unconditional dispatch is a net regression under fusion.**

| Configuration                            | tok/s  | Δ vs baseline |
| ---------------------------------------- | ------ | ------------- |
| baseline (CubeK fused matmuls)           | 26,362 | —             |
| unconditional cuBLAS dispatch, same tree | 21,837 | **−17%**      |

Loss/grad parity with baseline is exact per step, so the dispatch is
_correct_ — but slower in context. The nsys profiles explain it:

- Baseline spends 34.5% of GPU time in `matmul_entry_*` — CubeK matmuls with
  **fused elementwise prologues/epilogues** (casts, binops) — plus 16.5% in
  `elemwise_fuse`.
- With the dispatch, the raw GEMMs collapse to ~7% (cutlass/ampere tensor-core
  kernels — kernel selection confirmed), but the formerly fused elementwise
  work materializes as **~33% of GPU time in unvectorized f32 kernels**
  (`kernel_binop_c_f32_n_1` avg 875 µs, `unary_float_f_f32_n_1`,
  `kernel_scalar_binop_c_f32_n_1`) plus ~5% extra bf16↔f32 casts.

A global "always dispatch BF16 matmuls to cuBLAS" policy trades a fused 34.5%
for 7% + 38% unfused — a net loss. The per-shape wins are real but smaller
than the fusion they forfeit.

**Autotune-arbitrated dispatch is a large win.** The early fuser close was
the mistake, not the GEMM. With matmuls fusing normally and the plain
backend matmul (which now dispatches to cuBLAS) serving as the existing
`fused_matmul_fallback` autotune candidate, the tuner measures
"CubeK fused" against "cuBLAS + separately-fused epilogue" per key and keeps
whichever wins on that shape:

| Configuration (identical tree per row pair) | tok/s                                 |
| ------------------------------------------- | ------------------------------------- |
| CubeK only, B16 / B20                       | 29,530 / 30,487                       |
| autotuned cuBLAS, B16 / B20                 | **34,139 / 34,984 (+15.6% / +14.8%)** |

Loss and gradient norms are identical to five decimals between the paired
builds. The Burn-side change is two commits on the pinned fork: the
`kernel/matmul` server-GEMM dispatch, and the `f31e7513a` foreign-drop
backport (`.context/burn-cublas`, tip `e2fe66511`); the CubeCL side is the
async cuBLAS server GEMM (`463c2952`, fast-forward of the production pin).

F16 prepared-weight decode dispatch (M=1, N≈50304) remains unevaluated — it
targets inference decode, not training, and needs the F16 dtype mapping first.
