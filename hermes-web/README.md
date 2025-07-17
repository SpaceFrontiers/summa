# Hermes Web

A Vue 3 + Tailwind CSS web application for searching Hermes indexes via WASM.

## Model Lab

The standalone Model Lab visualizes a versioned trace produced by the shared
Hermes LLM model. Start the local live server to keep one checkpoint resident,
submit queries in the page, and play the resulting residual stream from the
embedding through every layer:

```bash
cargo run --release -p hermes-llm --features metal -- lab \
  --checkpoint checkpoint/weights.safetensors \
  --config checkpoint/config.json \
  --tokenizer tokenizer.json \
  --metrics checkpoint/metrics.jsonl
```

Open http://127.0.0.1:4173/model-lab.html. The live server binds to loopback by
default, accepts one bounded inference job at a time, and serves the page and
API from the same origin. Use `--max-new-tokens`, `--trace-tokens`,
`--channel-bins`, and `--attention-heads` to change its explicit limits.

For bundle-only inspection without loading a checkpoint:

```bash
pnpm lab:dev
pnpm lab:test
pnpm lab:build
```

`lab:dev` needs no JavaScript install or WASM build. It starts with a clearly
marked synthetic trace and can open real JSON bundles entirely in the browser;
selected files never leave the machine. Live query controls remain disabled in
this static mode. The production build uses the repository's pinned Vite
dependency.

Create a bundle from a checkpoint, optionally attaching trainer metrics:

```bash
cargo run -p hermes-llm -- trace \
  --checkpoint checkpoint/weights.safetensors \
  --config checkpoint/config.json \
  --tokenizer tokenizer.json \
  --prompt "Plan what evidence to retrieve" \
  --max-tokens 32 \
  --metrics checkpoint/metrics.jsonl \
  --output checkpoint/model-trace.json
```

The trace command prints every capture reduction. Use `--trace-tokens`,
`--channel-bins`, `--attention-heads`, and `--metrics-points` to change the
bounded defaults. Training can add the optional layer-gradient heatmap with
`hermes-train train --layer-metrics-every N`.

## Quick Start

### 1. Start the index server (from project root)

```bash
# Build test index and serve it
./scripts/serve-index.sh

# Or just serve an existing index
./scripts/serve-index.sh --serve-only
```

### 2. Start the web app

```bash
# Build WASM, install deps, and launch dev server
./scripts/dev.sh

# Or skip WASM build if already built
./scripts/dev.sh --skip-wasm
```

Open http://localhost:5173 in your browser.

## Manual Setup

### Build WASM

```bash
cd ../hermes-wasm
wasm-pack build --target web --release
```

### Install and run

```bash
pnpm install
pnpm dev
```

## Production Build

```bash
pnpm build
```

The static files will be in the `dist/` directory. Serve them with any static file server.

## Usage

1. Enter the server URL (e.g., `http://localhost:8765`)
2. Click **Connect** to load the index
3. Enter a search query and click **Search**
4. Click on results to expand and load document details

## Features

- **Serverless**: Compiles to static files, no backend required
- **WASM-powered**: Uses hermes-wasm for client-side search
- **Network stats**: View HTTP requests and cache statistics
- **Lazy loading**: Documents are fetched on demand when expanded