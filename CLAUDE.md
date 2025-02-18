# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hermes is a high-performance, embeddable full-text search engine written in Rust. It's a monorepo containing:

- **hermes-core**: Core search engine library (async, BM25 ranking, WAND optimization)
- **hermes-tool**: CLI for index management and data processing pipelines
- **hermes-server**: gRPC server for remote search operations
- **hermes-wasm**: WebAssembly bindings for browsers
- **hermes-llm**: LLM training framework (Candle ML, separate module)
- **hermes-client-python**: Python gRPC client
- **hermes-web**: Vue.js web UI

## Build Commands

```bash
# Build all Rust packages
cargo build --release

# Run all tests
cargo test --all-features

# Lint and format
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# Build WASM
cd hermes-wasm && wasm-pack build --release --target web

# Build Python wheel
cd hermes-core-python && maturin build --release

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

- Rust 1.92+ (nightly, see `rust-toolchain.toml`)
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

- **ci.yml**: Runs on push/PR to main. Rust (fmt, clippy, test, build), WASM build, Python lint, TypeScript build, cargo audit.
- **publish.yml**: Manual trigger (`workflow_dispatch`). Bumps version, publishes to crates.io, NPM (WASM + TS client), PyPI, and GHCR Docker.

When the user says "publish" or "trigger publish", run `gh workflow run publish.yml`.

## Key Dependencies

- **tokio**: Async runtime
- **zstd**: Compression
- **pest**: SDL parsing
- **tonic/prost**: gRPC
- **candle**: ML framework (hermes-llm)