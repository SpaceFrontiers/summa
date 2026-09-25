# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Shared engineering rules and the focused core/server harness live in
[`AGENTS.md`](AGENTS.md) and [the search system contract](docs/search-system-contract.md).
Read them before changing the search stack; keep system rules there to avoid drift.

## Workflow Rules

- When asked to "commit", "commit and push", or "push" — do it immediately. Do NOT continue making additional changes, running tests, or doing further work unless explicitly asked.
- When asked to "publish" or "trigger publish", run `gh workflow run publish.yml` and stop.
- Do not refactor, rename, or "improve" code beyond what was explicitly requested.
- When fixing a bug, write a failing test first if one doesn't exist, then fix.

## Development & Testing Rules

- **Fail loud, never silent.** Config/schema misuse must error or `log::warn!` at
  load time with an actionable message — never silently ignore an option, match
  zero items, or return empty results due to capability mismatch. A feature that
  ends up disabled at runtime must say so loudly.
- **Regression tests are named for the behavior they pin** (e.g.
  `test_binary_dense_search_multi_thread_runtime`), reproduce the failure first,
  and encode the scenario end-to-end — not just the unit.
- **Format/wire compatibility is designed, not accidental.** New serialized
  fields use `#[serde(default)]` (+ `skip_serializing_if` for emitters); formats
  carry version gates that loudly refuse data too new to read; incompatible
  builds must fail merge via global quantizer/codebook version checks.
  When a change claims "no behavioral change," verify byte-identical output.
- **Hot-path allocation hygiene.** Per-query buffers come from reusable /
  thread-local scratch where appropriate, `Vec::with_capacity` over realloc
  chains, sort+dedup in place over hash maps where possible. New per-query
  allocations on the search path need justification.
- **Every fallback or drop is observable.** Silent truncation, silently skipped
  candidates, or degraded modes need a counter, log line, or stats field.
- **Perf claims are measured.** Quote numbers (before/after, dataset, arch) in
  the PR/commit; validate cross-arch (x86 AVX2 + aarch64 NEON) before changing
  defaults. SIMD kernels always ship with a scalar fallback and dispatch safely.
- **Substantial features get a short design doc** under `docs/` before
  implementation (format changes, new index structures, protocol additions).
- **Float reductions on the vector path use algebraic ops** (Rust 1.98
  `f32::algebraic_add`/`algebraic_mul`) so LLVM can reassociate and vectorize
  them — see `docs/algebraic-float-reductions.md` for the measured numbers and
  the reproducibility contract. Never use them where a float is compared
  bit-exactly, hashed, or written into a content-addressed artifact; in
  particular `summa-train` stays strict IEEE.

## Project Overview

Summa is a high-performance, embeddable full-text search engine written in Rust. It's a monorepo containing:

- **summa-core**: Core search engine library (async, BM25 ranking, WAND optimization)
- **summa-tool**: CLI for index management and data processing pipelines
- **summa-server**: gRPC server for remote search operations
- **summa-broker**: gRPC routing, partitioned indexes, and cross-shard BM25 statistics
- **summa-wasm**: WebAssembly bindings for browsers (search + indexing)
- **summa-web**: Vue/WASM search UI
- **summa-model-lab**: Standalone local LLM trace and observability UI
- **summa-client-python**: Async Python gRPC client
- **summa-client-typescript**: TypeScript gRPC client
- **summa-proto**: Shared gRPC protocol definition
- **summa-mal**: Shared Model Architecture Language parser
- **summa-mal-python**: Thin PyO3 binding around `summa-mal`
- **summa-tokenizer**: Stable-Rust byte-level BPE tokenizer
- **summa-llm**: Shared Transformer/Mamba model, inference, generation, and MAL integration
- **summa-train**: Autodiff training for the shared summa-llm model

LLM pipeline: `summa-train train --config <.mal|.json>` trains the same `summa_llm::Transformer` used by inference, saves a safetensors checkpoint, and `summa-llm generate` loads it strictly. Do not add a second model implementation or checkpoint adapter.

**MAL parser is a single source of truth:** the pest grammar + AST live in the standalone `summa-mal` crate (`summa-mal/src/{lib.rs,mal.pest}`, embeds `summa-mal/well-known/*.mal`). `summa-llm` re-exports it as `crate::mal`; `summa-train` consumes that re-export. Change the grammar/AST in one place.

MAL supports hybrid Transformer+Mamba models: an `ssm { state_dim, conv_kernel, expand, dt_rank }` def makes a block a Mamba (selective state-space) block, and `pattern: [mamba_block, mamba_block, attn_block]` in a model cycles block types across num_layers (see `summa-mal/well-known/hybrid_tiny.mal`). GPU training and inference use the custom CubeCL selective-scan kernel.

## Build Commands

```bash
# Build all Rust packages
cargo build --release

# Run all tests
cargo test --workspace

# Lint and format
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Build WASM (requires Homebrew LLVM for zstd cross-compilation)
(cd summa-wasm && bash build.sh)

# Build Python packages
(cd summa-client-python && uv build)
(cd summa-mal-python && maturin build --release)

# summa-train (LLM training) tests
cargo test -p summa-train

# Run pre-commit hooks (rustfmt, clippy, ruff, prettier)
pre-commit run --all-files
```

Documentation navigation and benchmark commands live in [docs/README.md](docs/README.md)
and [docs/benchmarks.md](docs/benchmarks.md). Run `uv run scripts/check_docs.py`
when changing documentation links or adding benchmark targets.

## Architecture

### Core Library (summa-core/src/)

The index is the central abstraction. Documents are stored in **segments** (write-once chunks that get merged over time).

Key modules:

- `index/` - Main `Index` struct with async operations (search, write, merge)
- `segment/` - Segment building, reading, merging; document storage; vector indexes (flat, TurboQuant tq/ivf_tq and binary IVF)
- `query/` - Query execution with BM25 ranking, WAND/MaxScore optimizations
- `directories/` - Storage abstraction layer (filesystem, HTTP, RAM, memory-mapped, caching)
- `dsl/` - Schema Definition Language parser (pest-based)
- `structures/` - Low-level data structures (SSTables, bitpacked posting lists, skip lists)
- `tokenizer/` - Language-aware tokenization, dynamic stemming, and optional CJK morphology
- `compression/` - Zstd compression with configurable levels
- `merge/` - Segment merge strategies (tiered, no-merge)

### CLI Tool (summa-tool)

`main.rs` owns clap dispatch; `index_ops.rs`, `data_processing.rs`, and
`vector_ops.rs` own the corresponding command implementations:

- `create` - Create index from SDL schema
- `index` - Index documents from JSONL/stdin
- `merge` - Merge segments
- `commit` - Commit pending changes
- `info` - Show index statistics
- `diagnose` - Index health report (ANN leaf skew/fragmentation, per-field disk usage; `--sample`/`--probe-cost`/`--residency`/`--terms`/`--sparse-stats` opt-ins) — see docs/diagnostics.md
- `simhash` - Calculate SimHash for near-duplicate detection
- `sort` - Sort documents by field

Pipeline example: `zstdcat dump.zst | summa-tool simhash -f title -o hash | summa-tool sort -f hash -N | summa-tool index -i ./my_index --stdin`

### gRPC Server (summa-server)

- Proto definitions in `summa-proto/summa.proto`
- Two services: `SearchService` (search, get document, get info) and `IndexService` (create, index, commit, merge, delete)
- `IndexRegistry` for multi-index management
- Default port: 50051

### WASM (summa-wasm)

Browser-compatible search engine with both remote search and local indexing:

- **RemoteIndex** — loads pre-built indexes over HTTP with slice caching + IndexedDB persistence
- **IpfsIndex** — same as RemoteIndex but with JS fetch callbacks for IPFS
- **LocalIndex** — full in-browser indexing: create from SDL, add documents, commit, search. Pluggable `IFilesStorage` for persistence (IDB, encrypted, OPFS)
- **IndexRegistry** — manages multiple named indexes

The WASM build uses `summa-core` with features `["wasm", "http"]`. The `wasm` feature enables:

- `fst-index` — FST block index for reading native-built indexes
- `tokenizers` — HuggingFace tokenizers (pure Rust via `fancy-regex`)
- Sequential fallbacks for all `rayon` parallel operations
- In-memory `Vec<u8>` buffer instead of temp files for document store
- `simple_interner` HashMap-based string interner instead of `lasso`

Key constraint: WASM has no threads, no filesystem, no `SystemTime`. All native-only code is behind `#[cfg(feature = "native")]`.

### Web interfaces

- `summa-web` is the Vue/WASM search application. Its source may depend on
  `summa-wasm` but must not contain LLM trace or Model Lab code.
- `summa-model-lab` is a standalone, dependency-light LLM trace UI served by
  `summa-llm lab`. It must not depend on Vue, `summa-web`, or `summa-wasm`.
- The historical `pnpm lab:*` commands in `summa-web` are forwarding aliases
  only. `/model-lab.html` remains the stable Lab entry route.

### summa-core Feature Flags

- **`native`** (default via `sync`): Full native build — tokio, rayon, threads, mmap, lasso, uuid, etc.
- **`fst-index`**: FST block index support (included in both `native` and `wasm`)
- **`wasm`**: WASM-compatible build — sequential builders, in-memory store, simple interner
- **`http`**: HTTP directory with reqwest (works on both native and WASM)
- **`sync`**: Enables synchronous scoring and depends on `native`. Native
  without sync remains a supported async execution boundary.

### Schema Definition Language (SDL)

```sdl
index articles {
    field title: text<en_stem> [indexed, stored]
    field body: text [indexed]
    field views: u64 [indexed, stored]
}
```

Field types: `text`, `u64`, `i64`, `f64`, `bytes`, `json`, `dense_vector<dim>`, `sparse_vector`
Attributes: `indexed`, `stored` (default: both), `primary`, `fast`, `reorder`
Tokenizers: `default`, `simple`, `en_stem`, `de_stem`, `fr_stem`, `es_stem`, `it_stem`, `pt_stem`, `ru_stem`, `ar_stem`, and more

Full SDL reference: `docs/schema.md`

## Development Requirements

- Rust 1.98.1+ (see `rust-toolchain.toml`)
- Python 3.12+ (for Python bindings)
- Node.js 22.12+ (for WASM and web)
- pnpm 10+ (for TypeScript and web packages)
- uv (for the Python gRPC client)
- maturin (for the MAL Python wheel)
- wasm-pack (for WASM builds)
- protoc (for gRPC)

## CI/CD

Uses GitHub Actions. Trigger workflows with the `gh` CLI:

```bash
# Publish (bumps version, publishes to crates.io/npm/pypi/docker)
gh workflow run publish.yml

# Check CI status
gh run list --workflow=ci.yml --limit=5
```

**Workflows:**

- **ci.yml**: Runs on push/PR to main. Rust format/clippy/feature checks,
  workspace tests/build/docs, shell training harnesses, WASM and search UI,
  Model Lab, Python/TypeScript clients with generated-binding freshness, MAL
  wheel smoke tests, cargo audit, and unused-dependency checks.
- **publish.yml**: Manual trigger (`workflow_dispatch`). Bumps version, publishes to crates.io, NPM (WASM + TS client), PyPI, and GHCR Docker.

## Key Dependencies

- **tokio**: Async runtime
- **zstd**: Compression
- **pest**: SDL parsing
- **tonic/prost**: gRPC
- **Burn + CubeCL**: ML runtime and GPU kernels (`summa-llm` inference)
