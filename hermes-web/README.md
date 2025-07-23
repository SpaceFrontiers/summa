# Hermes Web

A Vue 3 + Tailwind CSS application for searching Hermes indexes via WASM.
This project is search-only; the LLM observability UI lives in
[`../hermes-model-lab`](../hermes-model-lab/README.md).

The historical `pnpm lab:dev`, `lab:test`, `lab:lint`, `lab:build`, and
`lab:preview` commands remain as forwarding aliases during the project split.

## Quick Start

### 1. Start the index server (from the repository root)

```bash
cargo run -p hermes-server --bin serve-index -- /path/to/index 8765
```

### 2. Start the web app

```bash
# Build WASM, install deps, and launch dev server
./hermes-web/scripts/dev.sh

# Or skip WASM build if already built
./hermes-web/scripts/dev.sh --skip-wasm
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

## Quality checks

```bash
pnpm test
pnpm lint
pnpm build
```

The unit suite covers framework-independent URL/configuration helpers. Keep
protocol parsing in `src/lib` so it can be shared and tested without booting
Vue or WASM.

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