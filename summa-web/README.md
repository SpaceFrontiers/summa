# Summa Web

Vue/WASM search UI for static indexes over HTTP or IPFS, with lazy document
loading and cache/network diagnostics. LLM traces live in the separate
[Model Lab](../summa-model-lab/README.md).

## Quick start

From the repository root, serve an existing index:

```bash
cargo run -p summa-server --bin serve-index -- /path/to/index 8765
```

In another terminal, build WASM, install dependencies, and start the UI:

```bash
./summa-web/scripts/dev.sh
# Use --skip-wasm when pkg/ is already built.
```

Open <http://localhost:5173>, connect to `http://localhost:8765`, and search.
See [UI configuration](../docs/ux-config.md) and [query syntax](../docs/query-language.md).

## Build and checks

Requires Node.js 22.12+, pnpm 10+, and the [WASM build tools](../summa-wasm/README.md#building).
From the repository root:

```bash
(cd summa-wasm && bash build.sh)
pnpm --dir summa-web install --frozen-lockfile
pnpm --dir summa-web test
pnpm --dir summa-web lint
pnpm --dir summa-web build
```

Serve `summa-web/dist/` as static files. Keep protocol/configuration helpers in
`src/lib` so tests need neither Vue nor WASM. Historical `pnpm lab:*` scripts
forward to Model Lab.
