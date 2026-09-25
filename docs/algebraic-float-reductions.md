# Algebraic float reductions

Rust 1.98 stabilised `f32`/`f64` **algebraic** arithmetic —
`algebraic_add`, `algebraic_sub`, `algebraic_mul`, `algebraic_div`,
`algebraic_rem`. Summa pins `1.98.1` in `rust-toolchain.toml` and uses these
operations in the dense-vector reduction kernels.

## Why they matter here

A dot product written with strict IEEE `+` has a loop-carried dependency on the
accumulator. LLVM is not permitted to break it, so the loop runs at one scalar
add per iteration no matter how wide the machine is — which is exactly why
`structures/simd.rs` carries hand-written NEON, SSE, AVX2 and AVX-512 kernels
with four explicit accumulator chains each.

Algebraic operations grant the reassociation permission that was missing. The
same plain `fold` then vectorises on its own:

```rust
a.iter()
    .zip(b)
    .fold(0.0f32, |acc, (&x, &y)| acc.algebraic_add(x.algebraic_mul(y)))
```

They are **not** `-ffast-math`. Algebraic operations never assume finite
inputs: NaN and infinity still propagate, so a degenerate stored vector cannot
silently score as a finite number. They also never cause undefined behaviour.
`structures::simd::algebraic_reduction_tests` pins both properties.

## Where they are used

| Site                                                 | Path          | Role                                                          |
| ---------------------------------------------------- | ------------- | ------------------------------------------------------------- |
| `simd::dot_product_f32` scalar fallback              | query         | used on targets with no SIMD kernel, including WASM           |
| `simd::fused_dot_norm` scalar fallback               | query         | dot + self-norm in one pass                                   |
| `simd::{dot_product,fused_dot_norm}_{f16,u8}_scalar` | query         | quantized vector fallbacks                                    |
| `simd::squared_l2_f32`                               | query + build | shared by ScaNN AH, ScaNN engine, k-means, IVF coarse routing |
| `simd::norm_squared_f32` / `norm_f32`                | query + build | squared / L2 norm                                             |
| `scann::ah::AhQuery::score_{unpacked,packed}`        | query         | asymmetric-hashing LUT accumulation                           |
| `ivf::coarse::soar_secondary_loss`                   | build         | SOAR secondary-assignment loss                                |
| `index::vector_builder` routing distortion           | build         | exact vs routed cluster distance                              |

The four previously duplicated `squared_l2` helpers (`scann/ah.rs`,
`scann/engine.rs`, `kmeans.rs`, `ivf/coarse.rs`) now delegate to
`simd::squared_l2_f32`.

The hand-written SIMD kernels are untouched. Their scalar remainder tails are
shorter than one vector lane group, so algebraic ops buy nothing there.

## Measured

aarch64 (Apple Silicon), rustc 1.98.0, `opt-level = 3`, baseline `target-cpu`,
before → after ns/op:

| dim  | squared_l2             | scalar dot         | fused dot+norm     | SOAR loss          |
| ---- | ---------------------- | ------------------ | ------------------ | ------------------ |
| 128  | 55.5 → 17.0 (3.3x)     | 108 → 16 (6.7x)    | 145 → 27 (5.3x)    | 76.6 → 18.8 (4.1x) |
| 384  | 220.8 → 26.1 (8.4x)    | 360 → 32 (11.1x)   | 365 → 30 (12.3x)   | 290 → 80.0 (3.6x)  |
| 768  | 592.8 → 63.2 (9.4x)    | 956 → 55 (17.3x)   | 752 → 52 (14.5x)   | 538 → 67.7 (7.9x)  |
| 1536 | 1827.8 → 123.6 (14.8x) | 2032 → 131 (15.5x) | 1404 → 123 (11.4x) | 1464 → 287 (5.1x)  |

The ScaNN AH LUT accumulation is gather-bound rather than arithmetic-bound, so
it gains less — 199 → 117 ns/op at 96 blocks (1.7x) and 570 → 437 ns/op at 384
blocks (1.3x) — but the reassociated accumulator chain still pays for itself.

Relative error against a strict `f64` reference stays under `1e-6` at every
dimension (e.g. dim 768 squared L2: `478.5987549` → `478.5982971`).

### End to end

`cargo bench -p summa-core --bench vector_indexing -- ivf_` on the same
machine, same toolchain, source-only diff (criterion, 10 samples, 3s
measurement, all `p < 0.05`):

| benchmark                          | before   | after    | change |
| ---------------------------------- | -------- | -------- | ------ |
| `ivf_tq_plan/16`                   | 134.0 µs | 123.9 µs | −9.4%  |
| `ivf_tq_plan/64`                   | 142.5 µs | 113.5 µs | −21.9% |
| `ivf_coarse_training/clusters/64`  | 4.93 ms  | 4.32 ms  | −16.4% |
| `ivf_coarse_training/clusters/257` | 32.7 ms  | 22.1 ms  | −29.1% |

Coarse training is the k-means path; the plan benchmarks are query-side IVF
routing.

WASM was not measured. `summa-wasm` builds without `simd128`, so LLVM cannot
vectorise there; the scalar fallbacks still gain from the broken accumulator
dependency chain, but do not assume the native multiples carry over.

x86 AVX2/AVX-512 machines keep dispatching to their hand-written kernels for
`dot_product_f32` and `fused_dot_norm`, so the win there is confined to
`squared_l2_f32`, `norm_squared_f32` and the SOAR/vector-builder loops, which
had no SIMD path at all.

## Reproducibility contract

Algebraic reductions are **not bit-reproducible across builds**: a different
rustc version, target CPU, or inlining decision may pick a different reduction
order and move the last few ULPs.

That variance is not new. `dot_product_f32` already rounds differently on NEON,
AVX2, AVX-512 and the scalar path because each uses a different number of
accumulators, so a query scored on an Apple Silicon replica already does not
bit-match the same query on an AVX-512 replica. Using algebraic ops adds no new
_class_ of variance.

What this does change:

- **k-means and IVF/ScaNN centroid training** now produce build-dependent
  centroids at the ULP level. Codebooks are serialised into the index and
  carry a version, so this affects only what a _fresh_ build produces, never
  the readability of an existing segment. Cluster assignment can flip for a
  point that sits on a decision boundary; recall is unaffected in aggregate.
- **Scores** shift by <1e-6 relative, well below any ranking tie that BM25 or
  cosine cutoffs resolve.

Do not use algebraic operations where a float is compared for bit-exact
equality, hashed, or written into a content-addressed artifact. Specifically
they are **not** used in `summa-train`: checkpoint resume is content-addressed
(hence `serde_json`'s `float_roundtrip`) and training must stay bit-reproducible
across restarts.

## Deferred: `chunks_exact` → `as_chunks`

Clippy 1.98 also added `chunks_exact_to_as_chunks`, which fires on ~40 call
sites across `summa-core` and `summa-train`. The suggestion is worth taking —
a const-generic chunk width lets LLVM drop the per-chunk length check and
replaces `bytes.try_into().unwrap()` with a plain array deref — but most of the
sites sit inside BMP, ScaNN and fast-field wire-format parsers, so the rewrite
belongs in its own reviewed change rather than riding along with a toolchain
bump.

The lint is currently allowed at the crate roots of `summa-core`,
`summa-train`, the `summa-train` binary, and `summa-core`'s
`vector_indexing` bench. Removing those four `#![allow(...)]` lines is the
entry point for the migration.
