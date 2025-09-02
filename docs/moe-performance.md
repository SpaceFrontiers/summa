# MoE A100 performance

These are historical measurements from 2026-07-21, not a fresh run of the
current target selection and reporting requirements.

The measured geometry is 8,192 tokens, hidden size 512, expert width 768,
eight experts, and top-2 routing. Each result below is the median of 100 timed
iterations after 20 warmups; lower latency is better.

| Implementation            |      Forward | Core forward + backward | With router losses |
| ------------------------- | -----------: | ----------------------: | -----------------: |
| Hermes grouped + fused    | **1.503 ms** |            **3.363 ms** |           4.237 ms |
| PyTorch 2.12 `grouped_mm` |     1.625 ms |                3.610 ms |       **4.235 ms** |

Hermes is 7.5% faster in forward and 6.8% faster in the apples-to-apples core
training step. With the model's load-balancing coefficient `0.01` and router
z-loss coefficient `0.001` enabled in both implementations, the results differ
by 0.04%, which is measurement noise. A repeated process-level run preserved
the forward and core-training wins; the full-loss result stayed within 1.3%.

The optimized Hermes path consists of:

- deterministic GPU-resident route counting and stable packing;
- direct route gather and weighted route combine kernels, with specialized
  permutation-aware backward kernels;
- one backend operation for the expert MLP, so the eight host-visible route
  counts are read once and both grouped GEMMs are launched without a framework
  round trip;
- native CUDA BF16 grouped GEMM dispatch on CUDA 12.5+ and SM80+;
- fused BF16 SwiGLU forward and backward kernels with FP32 intermediate math;
- a tensor-operation fallback when native grouped GEMM is unavailable.

The core training measurement minimizes the squared FP32 output and computes
input, router, and expert-parameter gradients. The router-loss measurement adds
the same Switch-style load-balancing term and `logsumexp(logits)^2` z-loss to
both frameworks.

Reproduce the measurements on Linux CUDA with: