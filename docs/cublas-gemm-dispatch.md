# Native cuBLAS GEMM dispatch: proof results and verdict

Investigation of routing Burn/CubeCL BF16 matmuls to native cuBLAS on CUDA,
aimed at closing the training-throughput gap against PyTorch. All numbers
measured 2026-07-16 on a GCP `a2-highgpu-1g` (A100-SXM4 40GB, driver 580,
CUDA 12.9), `retriever-100m` (115.8M params), batch 16 × seq 1024 ×
grad-accum 8, steady-state steps 5–8.

## Prototype

- CubeCL [upstream PR #1440](https://github.com/tracel-ai/cubecl/pull/1440)
  (the prototype was measured at `463c2952`; Summa now pins the PR head
  `c0efe74d` from the official repository): asynchronous cuBLAS BF16 GEMM server dispatch
  (`GemmDescriptor`/`GemmMatrix` server API), hardened contracts (rejects
  foreign-stream outputs, buffer overlap, non-empty zero-K problems).
- Burn [upstream PR #5190](https://github.com/tracel-ai/burn/pull/5190)
  (Summa pins `973605c4` from the official repository) adds BF16 dispatch in
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

## Isolated GEMM benchmarks: cuBLAS dispatch vs PyTorch 2.9.1

BF16, CUDA events, 10–100 reps per shape. The dispatch **beats or matches
`torch.matmul` on 14 of 17 training shapes** (typically +10–25%, e.g.
16384×512×1536: 197 vs 156 TFLOPS; 16384×2048×512: 212 vs 173), losing only
on two large-K vocab shapes (4096×512×50304ᵀ: 217 vs 243; 512×50304×4096ᵀ:
199 vs 240) and two mid shapes (−5/−6%). CubeK's own BF16 matmul kernels trail
both on these shapes (autotune logs, `.context/matmul-autotune-*.json.log`).

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
builds. The Burn-side change is the `kernel/matmul` server-GEMM dispatch plus
the `f31e7513a` foreign-drop prerequisite, now carried by upstream PR #5190;
the CubeCL side is the async cuBLAS server GEMM in upstream PR #1440.

F16 prepared-weight decode dispatch (M=1, N≈50304) remains unevaluated — it
targets inference decode, not training, and needs the F16 dtype mapping first.

## cublasLt (measured neutral, kept)

The dispatch now goes through `cublasLtMatmul` with a per-shape cached
execution plan: descriptor + layouts + the algorithm
`cublasLtMatmulAlgoGetHeuristic` selects with a 32 MiB per-stream workspace
(introduced on the upstream PR branch and now included in its pinned head).
This is the same machinery PyTorch's
linear uses, replacing `cublasGemmEx` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

Measured (A100, retriever-100m, T1024/ga8, 2026-07-17): 43,214 → 43,217
tok/s @B20 and 44,193 → 44,201 @B26 — flat within noise, gates green,
gradient norms identical. At these shapes the legacy heuristic was already
picking equivalent kernels, and the autotune arbitration against
CubeK-fused had already captured the remaining arbitrage. Kept anyway: no
regression, and the heuristic + workspace machinery is what future shapes
(and an eventual FLOPS-aware preference pass) build on. Together with the
five measured scan-backward rejections this pins BOTH the GEMM pool
(~23.6%) and the scan pool (15.1%) at their practical floors for the
current kernel architectures.
