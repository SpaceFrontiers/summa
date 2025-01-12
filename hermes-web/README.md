# Hermes Web

A Vue 3 + Tailwind CSS web application for searching Hermes indexes via WASM.

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