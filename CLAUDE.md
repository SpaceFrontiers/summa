# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
  builds must fail merge via version checks (see RaBitQ codebook `version`).
  When a change claims "no behavioral change," verify byte-identical output.
- **Hot-path allocation hygiene.** Per-query buffers come from reusable /
  thread-local scratch (see `BmpScratch`), `Vec::with_capacity` over realloc
  chains, sort+dedup in place over hash maps where possible. New per-query
  allocations on the search path need justification.
- **Every fallback or drop is observable.** Silent truncation, silently skipped
  candidates, or degraded modes need a counter, log line, or stats field.
- **Perf claims are measured.** Quote numbers (before/after, dataset, arch) in
  the PR/commit; validate cross-arch (x86 AVX2 + aarch64 NEON) before changing
  defaults. SIMD kernels always ship with a scalar fallback and dispatch safely.
- **Substantial features get a short design doc** under `docs/` before
  implementation (format changes, new index structures, protocol additions).

## Project Overview

Hermes is a high-performance, embeddable full-text search engine written in Rust. It's a monorepo containing:

- **hermes-core**: Core search engine library (async, BM25 ranking, WAND optimization)
- **hermes-tool**: CLI for index management and data processing pipelines
- **hermes-server**: gRPC server for remote search operations
- **hermes-wasm**: WebAssembly bindings for browsers (search + indexing)
- **hermes-llm**: Shared Transformer/Mamba model, inference, generation, and MAL integration
- **hermes-train**: Autodiff training for the shared hermes-llm model
- **hermes-client-python**: Python gRPC client
- **hermes-web**: Vue.js web UI

LLM pipeline: `hermes-train train --config <.mal|.json>` trains the same `hermes_llm::Transformer` used by inference, saves a safetensors checkpoint, and `hermes-llm generate` loads it strictly. Do not add a second model implementation or checkpoint adapter.

**MAL parser is a single source of truth:** the pest grammar + AST live in the standalone `hermes-mal` crate (`hermes-mal/src/{lib.rs,mal.pest}`, embeds `hermes-mal/well-known/*.mal`). `hermes-llm` re-exports it as `crate::mal`; `hermes-train` consumes that re-export. Change the grammar/AST in one place.

MAL supports hybrid Transformer+Mamba models: an `ssm { state_dim, conv_kernel, expand, dt_rank }` def makes a block a Mamba (selective state-space) block, and `pattern: [mamba_block, mamba_block, attn_block]` in a model cycles block types across num_layers (see `hermes-mal/well-known/hybrid_tiny.mal`). GPU training and inference use the custom CubeCL selective-scan kernel.

## Build Commands

```bash
# Build all Rust packages
cargo build --release

# Run all tests
cargo test --all-features

# Lint and format
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Build WASM (requires Homebrew LLVM for zstd cross-compilation)
cd hermes-wasm && bash build.sh

# Build Python wheel
cd hermes-core-python && maturin build --release

# hermes-train (LLM training) tests
cargo test -p hermes-train

# Run pre-commit hooks (rustfmt, clippy, ruff, prettier)
pre-commit run --all-files
```

## Architecture

### Core Library (hermes-core/src/)

The index is the central abstraction. Documents are stored in **segments** (write-once chunks that get merged over time).

Key modules:

- `index/` - Main `Index` struct with async operations (search, write, merge)
- `segment/` - Segment building, reading, merging; document storage; vector indexes (flat, IVF-RaBitQ, ScaNN)
- `query/` - Query execution with BM25 ranking, WAND/MaxScore optimizations
- `directories/` - Storage abstraction layer (filesystem, HTTP, RAM, memory-mapped, caching)
- `dsl/` - Schema Definition Language parser (pest-based)
- `structures/` - Low-level data structures (SSTables, bitpacked posting lists, skip lists)
- `tokenizer/` - Language-aware tokenization with 15+ Snowball stemmers
- `compression/` - Zstd compression with configurable levels
- `merge/` - Segment merge strategies (tiered, no-merge)

### CLI Tool (hermes-tool)

Single `main.rs` with clap-based subcommands:

- `create` - Create index from SDL schema
- `index` - Index documents from JSONL/stdin
- `merge` - Merge segments
- `commit` - Commit pending changes
- `info` - Show index statistics
- `simhash` - Calculate SimHash for near-duplicate detection
- `sort` - Sort documents by field

Pipeline example: `zstdcat dump.zst | hermes-tool simhash -f title -o hash | hermes-tool sort -f hash -N | hermes-tool index -i ./my_index --stdin`

### gRPC Server (hermes-server)

- Proto definitions in `hermes-proto/hermes.proto`
- Two services: `SearchService` (search, get document, get info) and `IndexService` (create, index, commit, merge, delete)
- `IndexRegistry` for multi-index management
- Default port: 50051

### WASM (hermes-wasm)

Browser-compatible search engine with both remote search and local indexing:

- **RemoteIndex** — loads pre-built indexes over HTTP with slice caching + IndexedDB persistence
- **IpfsIndex** — same as RemoteIndex but with JS fetch callbacks for IPFS
- **LocalIndex** — full in-browser indexing: create from SDL, add documents, commit, search. Pluggable `IFilesStorage` for persistence (IDB, encrypted, OPFS)
- **IndexRegistry** — manages multiple named indexes

The WASM build uses `hermes-core` with features `["wasm", "http"]`. The `wasm` feature enables:

- `fst-index` — FST block index for reading native-built indexes
- `tokenizers` — HuggingFace tokenizers (pure Rust via `fancy-regex`)
- Sequential fallbacks for all `rayon` parallel operations
- In-memory `Vec<u8>` buffer instead of temp files for document store
- `simple_interner` HashMap-based string interner instead of `lasso`

Key constraint: WASM has no threads, no filesystem, no `SystemTime`. All native-only code is behind `#[cfg(feature = "native")]`.

### hermes-core Feature Flags

- **`native`** (default via `sync`): Full native build — tokio, rayon, threads, mmap, lasso, uuid, etc.
- **`fst-index`**: FST block index support (included in both `native` and `wasm`)
- **`wasm`**: WASM-compatible build — sequential builders, in-memory store, simple interner
- **`http`**: HTTP directory with reqwest (works on both native and WASM)
- **`sync`**: Alias for `native`

### Schema Definition Language (SDL)

```sdl
index articles {
    field title: text<en_stem> [indexed, stored]
    field body: text [indexed]
    field views: u64 [indexed, stored]
}
```

Field types: `text`, `u64`, `i64`, `f64`, `bytes`
Attributes: `indexed`, `stored` (default: both)
Tokenizers: `default`, `simple`, `en_stem`, `de_stem`, `fr_stem`, `es_stem`, `it_stem`, `pt_stem`, `ru_stem`, `ar_stem`, and more

Full SDL reference: `docs/schema.md`

## Development Requirements

- Rust 1.97+ (see `rust-toolchain.toml`)
- Python 3.12+ (for Python bindings)
- Node.js 20+ (for WASM and web)
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

- **ci.yml**: Runs on push/PR to main. Rust (fmt, clippy, test, build), WASM build, Python client lint, TypeScript build, cargo audit.
- **publish.yml**: Manual trigger (`workflow_dispatch`). Bumps version, publishes to crates.io, NPM (WASM + TS client), PyPI, and GHCR Docker.

## Key Dependencies

- **tokio**: Async runtime
- **zstd**: Compression
- **pest**: SDL parsing
- **tonic/prost**: gRPC
- **Burn + CubeCL**: ML runtime and GPU kernels (`hermes-llm` inference)