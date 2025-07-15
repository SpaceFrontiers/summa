# Kernel size-generality and tuning surface

What happens to the training/inference optimizations when the model shape
changes (hidden size, layers, heads, state dim, sequence length, batch),
and which constants encode measured tuning rather than structure. Rule of
thumb: **everything is correct at any shape** (comptime-generic kernels,
runtime guards, loud asserts, tensor-op fallbacks — pinned by parity tests
at deliberately awkward geometries), and **most things stay fast
automatically**; the short list of measured constants below is the part
worth re-checking for a very different regime.

## Automatically shape-adaptive (nothing to do)

- **Matmul dispatch**: the CubeK fused matmul vs. cuBLAS/cublasLt choice is
  autotune-arbitrated _per shape_ at runtime; cublasLt additionally caches
  a per-shape heuristic plan. New layer sizes re-tune themselves on first
  run (autotune cache).
- **Chunked cross-entropy**: vocabulary padding is derived
  (`ModelDef::padded_vocab_size()`, 64-row alignment); the row kernels are
  generic over vocab and hidden width.
- **Depthwise conv**: comptime-generic over `conv_kernel`.
- **Flash attention forward (+LSE)**: any shape cubek rejects falls back to
  burn's tensor-op attention with a recomputed LSE — head_dim is capped at
  128 with a loud assert. The chunked attention backward derives its
  power-of-two chunking from the sequence and falls back to tensor ops for
  shapes the fused probabilities op cannot take (a fork-runtime defect at
  non-power-of-two chunk widths, documented in `docs/fused-attention.md`).
- **Selective scan**: runtime-generic over batch/sequence/channels
  (channel tiles guard partial tiles, segments guard partial tails; the
  channel-overhang and tail geometries are pinned against CPU autodiff),
  comptime-generic over `state_dim` with a **measured support contract of
  power-of-two widths in 4..=16 on GPU** (any width on the CPU reference).
  The kernels themselves are shape-generic — Metal passes 2/5/6/12 — but
  the CUDA fork runtime displaces writes into state tensors whose minor
  stride is not a power of two (state 5 lands at stride 8; 4/8/16/32
  measured clean, 2/5/6/12 corrupt), and 32 exceeds Metal's threadgroup
  budget, so the dispatch refuses everything outside {4, 8, 16} with an
  actionable panic (`test_gpu_selective_scan_rejects_unsupported_state_dim`
  pins the refusal; widths 4, 8, 16 are each pinned green). Training
  always takes the swept checkpointed kernels regardless of batch or
  sequence — there is exactly one training scan implementation to
  maintain.
- **Hybrid layer patterns**: MAL-driven (`pattern:` cycling); nothing in
  the kernels assumes a layer count or ordering.

## Structural constants (change the kernel if you change them)

These encode the sweep kernels' warp geometry, not tuning. Compile-time
asserts in `scan.rs` pin the invariants:

- `BACKWARD_CHANNELS = 16` — one plane spans two state rows of a 16-wide
  channel tile; the partner-row shuffle offset and `half_plane_sum`'s
  reduction ladder both derive from it, and `BACKWARD_CHANNELS * 2 ==
PLANE_WIDTH` is statically asserted (both CUDA warps and Metal SIMD
  groups are 32-wide).
- `PLANE_WIDTH = 32` — warp/simdgroup width on every supported target.

## Measured constants (re-measure for a very different regime)

Tuned on A100 40GB at retriever-100m scale (d_inner 1536, state 16,
T1024). They are _correct_ everywhere; their optimality was measured here:

- `CHECKPOINTED_SCAN_INTERVAL = 32` — the recompute/parallelism balance;
  16 measured −2.3%, 8 −8.3% at this scale. Statically asserted to tile by
  `BACKWARD_FLUSH = 8` (which itself is sized so the flush-window slots
  fit Metal's 32KB threadgroup budget at state_dim 16 — at the state_dim
  32 cap the segmented backward exceeds Metal's budget and fails loudly at
  launch; CUDA is fine).
- `SERIAL_SCAN_MIN_BLOCKS = 128` — when the non-segmented forward prefers
  serial recurrence over parallel state lanes.
- Launcher widths (`THREADS_PER_CUBE`, `CE_THREADS`, elementwise thread
  counts) — occupancy defaults, shape-independent.
- **Batch size** — the throughput-optimal batch is a property of the
  memory budget, not the code (B27 at retriever-100m/A100-40GB, B28
  OOMs). Sweep it per config.
- The BF16-stream layout (norms FP32-internal, F16 saved attention
  tensors, per-tensor cast points) mirrors PyTorch autocast and is
  shape-independent, but its _wins_ were measured at this scale.

## Language surface

There is no first-party C, C++, CUDA-C, or Metal source in the build. All
kernels are Rust (CubeCL `#[cube]` functions) compiled to the target at
runtime; CUDA-C is _generated_ by CubeCL and fed to NVRTC, never written
or maintained by hand. `.context/` holds gitignored upstream reference
material (mamba-ssm CUDA sources) used for study only. Remaining non-Rust
in the dependency tree is vendored C from crates.io (`zstd-sys` and
similar `-sys` crates) plus NVIDIA's binary cuBLAS/cublasLt — none of it
ours to maintain. The pest grammars, protobuf codegen, and tokenizers are
pure Rust.
