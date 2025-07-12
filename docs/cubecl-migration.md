# CubeCL migration — portable Mamba inference kernels

Status: **experimental, opt-in.** Landed behind the OFF-by-default `cubecl`
cargo feature in `hermes-llm`. The default build and all existing code paths are
untouched.

## What CubeCL is

[CubeCL](https://github.com/tracel-ai/cubecl) (tracel-ai, the Burn authors) is a
Rust language extension + JIT compiler + set of runtimes for GPU compute. You
write **one** `#[cube]` Rust function and it compiles on demand to CUDA, HIP,
Metal/SPIR-V/WGSL (via `wgpu`), or CPU — while still emitting good
platform-specific instructions. It replaces the "write the kernel N times, once
per backend" problem.

## Why

Mamba (selective state-space) inference has two compute-heavy steps — a
depthwise causal conv1d and the selective scan — that a hand-written backend
would need re-implemented per target (a Metal `.metal` file, a CUDA `.cu` file,
a scalar CPU fallback, each parity-tested separately). CubeCL lets us keep **one
Rust source** for all three. This is the migration target: `#[cube]` kernels
that serve on CUDA, Metal (through wgpu), and eventually CPU.

## Scope note (important, read this)

The original task brief referenced files that **do not exist on this branch**:
`hermes-llm/src/metal_scan.rs`, `hermes-llm/src/model/mamba.rs`,
`selective_scan.metal`, and `docs/metal-selective-scan.md`. There are **no
hand-written Metal MSL kernels** and **no Candle `CustomOp3` wiring** in this
repo. What actually exists is a Candle-tensor Mamba mixer,
`hermes-llm/src/model.rs :: MambaMixer::forward_with_state`, which does the conv
via `Tensor::conv1d` and the scan as a sequential Rust loop over Candle ops.

So there was no MSL kernel to "port"; instead we implemented the CubeCL kernels
from the **Candle reference math** and pinned them to that reference with parity
tests. The `metal,remote` test command in the brief also references a `remote`
feature that `hermes-llm` does not have (features are `cpu`/`cuda`/`metal`/
`accelerate`/`flash-attn`/`cubecl`/`cubecl-cuda`); `--features metal` is the
working equivalent and stays green.

## Versions pinned

`hermes-llm/Cargo.toml`, optional, gated by the `cubecl` feature:

```toml
cubecl = { version = "=0.10.0", default-features = false,
           features = ["wgpu", "std"], optional = true }
```

- `cubecl 0.10.0` — current crates.io release as of 2026-07 (the pre-0.10 line,
  e.g. 0.4.x, has a materially different API). Pinned with `=` because CubeCL is
  pre-1.0 and churns launch/scalar/index signatures between minors.
- Runtime: **wgpu** (`cubecl-wgpu 0.10.0`), which targets Metal on macOS,
  Vulkan/DX elsewhere — pure Rust, no system toolchain needed.
- CUDA runtime is available behind the nested `cubecl-cuda` feature
  (`cubecl = ["dep:cubecl"]`, `cubecl-cuda = ["cubecl", "cubecl/cuda"]`).

### Why not the CPU runtime (yet)

`cubecl-cpu 0.10.0` depends on `tracel-llvm` / `tracel-llvm-bundler`
(bundled LLVM/MLIR 20.1) — a heavy system build we deliberately do **not** pull
in. On macOS the wgpu runtime already gives us a Metal path, and it doubles as
the test backend here. Adding `cubecl-cpu` for a true CPU JIT is the follow-up
below.

## Feature flag

```
cubecl        # kernels + wgpu (Metal/Vulkan/DX) runtime
cubecl-cuda   # additionally enable the CUDA runtime
```

Everything CubeCL lives under `#[cfg(feature = "cubecl")]`
(`hermes-llm/src/cubecl_kernels.rs`, module declared in `lib.rs`). With the
feature off, the crate compiles byte-identically to before — CubeCL is not in
the dependency graph.

## What's ported

`hermes-llm/src/cubecl_kernels.rs`:

| kernel             | maps to (Candle `MambaMixer`)                       | threading                                                                                 |
| ------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `conv1d_depthwise` | `padded.conv1d(w, 0, 1, 1, d_inner)` + bias         | one thread per `(batch, channel)`, loop over time                                         |
| `selective_scan`   | the sequential `for t` scan loop (stateless, `h=0`) | one thread per `(batch, channel)`, `h[N]` in a register array (`state_dim` is `comptime`) |

Recurrence in `selective_scan`, matching `model.rs` exactly:

```
da = exp(dt * A)          # A = -exp(A_log), precomputed host-side
h  = h * da + dt * B * x
y  = sum_n(h * C) + x * D
```

Both kernels are `f32`-only (the inference dtype). Public host wrappers
`run_conv1d::<R>` / `run_selective_scan::<R>` take plain slices and return
`Vec<f32>`; scalar CPU references `reference_conv1d` / `reference_selective_scan`
encode the identical math.

### Integration status

The kernels are wired as a **library capability**, not yet swapped into
`MambaMixer::forward`. Swapping in requires Candle-tensor ⇄ device-buffer
plumbing (and, for decode, threading the persistent conv tail + `h` state
through the kernel). The default forward path is intentionally left untouched so
the migration is risk-free to land. Turning it on inside `forward_with_state`
under `#[cfg(feature = "cubecl")]` is a mechanical follow-up.

## Parity results

`cargo test -p hermes-llm --features cubecl` (wgpu → Metal on this macOS host):

- `conv1d_matches_reference` — kernel vs scalar reference — **pass**
- `selective_scan_matches_reference` — kernel vs scalar reference — **pass**
- `conv1d_matches_candle` — kernel vs the actual `Tensor::conv1d` + bias — **pass**
- `selective_scan_matches_candle` — kernel vs the actual Candle sequential scan — **pass**
- `wgpu_smoke` — runtime init + dispatch — **pass**

Tolerance: `max_abs_diff / max_abs < 1e-4` (all comfortably under). The
Candle-parity tests reproduce `model.rs`'s ops op-for-op on CPU tensors and
compare against the kernel output, so the kernels are pinned to the real
inference reference, not just a private re-derivation.

## CubeCL 0.10 API notes / limitations hit

Captured so the next person doesn't re-discover them:

- **Scalar kernel args** are passed to `kernel::launch::<R>(...)` as **raw
  values** (`channels as u32`), _not_ wrapped in `ScalarArg` (that older idiom
  is gone from 0.10's generated launch signature).
- **`ABSOLUTE_POS` and `Array::len()` are `usize`** inside a `#[cube]` body.
  Mixing them with `u32` scalar params is a type error — cast params to `usize`
  at the top of the kernel (`let channels = channels as usize;`).
- **Register/local arrays** (`Array::<f32>::new(len)`) require a **`comptime`**
  length. Hence `state_dim` is a `#[comptime] usize` param; the kernel is
  JIT-specialised per `state_dim`. Fine — it's a small fixed model constant.
- **`CubeDim::new`** in 0.10 is `new(client, working_units)`; for explicit dims
  use `CubeDim::new_1d/2d/3d`. `CubeCount::Static(x,y,z)` still exists.
- `client.create_from_slice(f32::as_bytes(&v))` to upload, `client.empty(bytes)`
  to allocate output, `client.read_one(handle).unwrap()` (returns
  `Result<Bytes>`) to download; `Handle` is `Clone`.
- **wgpu/Metal init works in this sandbox** — the smoke + parity tests actually
  dispatch to the GPU. No headless/adapter issues encountered on this host.
- CubeCL 0.10 **builds on the repo's nightly** (`nightly-2026-07-07`,
  rustc 1.99) without patching.

## Follow-ups

1. **Chunked / associative scan.** The current `selective_scan` is a _sequential_
   per-`(b,c)` scan — correct but it underutilizes the GPU (one thread walks the
   whole time axis; no parallelism across `t`). The real perf win is a
   chunked/parallel-prefix (associative) scan à la the Mamba CUDA kernel. This
   port deliberately does correctness-first; the sequential kernel is the
   baseline to beat.
2. **Stateful decode.** Emit and resume the persistent conv tail and `h` state so
   the kernels can back `forward_with_state`'s O(1)-per-token decode, not just
   the stateless prefill.
3. **Swap into `MambaMixer::forward`** under `#[cfg(feature = "cubecl")]` with
   Candle ⇄ CubeCL buffer bridging (zero-copy where the device matches).
4. **Add `cubecl-cpu`** for a genuine CPU JIT fallback (accepting the bundled
   LLVM/MLIR build cost), so the "one kernel → CUDA + Metal + CPU" story is
   complete without relying on wgpu's CPU backend.
5. **bf16/f16** kernels for parity with the training dtype path.

```

```
