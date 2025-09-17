# Hermes Web

Vue/WASM search UI for static indexes over HTTP or IPFS, with lazy document
loading and cache/network diagnostics. LLM traces live in the separate
[Model Lab](../hermes-model-lab/README.md).

## Quick start

From the repository root, serve an existing index:

```bash
cargo run -p hermes-server --bin serve-index -- /path/to/index 8765
```

In another terminal, build WASM, install dependencies, and start the UI:

```bash
./hermes-web/scripts/dev.sh
# Use --skip-wasm when pkg/ is already built.
```

Open <http://localhost:5173>, connect to `http://localhost:8765`, and search.
See [UI configuration](../docs/ux-config.md) and [query syntax](../docs/query-language.md).

## Build and checks

Requires Node.js 22.12+, pnpm 10+, and the [WASM build tools](../hermes-wasm/README.md#building).
From the repository root:

```bash
(cd hermes-wasm && bash build.sh)
pnpm --dir hermes-web install --frozen-lockfile
pnpm --dir hermes-web test
pnpm --dir hermes-web lint
pnpm --dir hermes-web build
```

Serve `hermes-web/dist/` as static files. Keep protocol/configuration helpers in
`src/lib` so tests need neither Vue nor WASM. Historical `pnpm lab:*` scripts
forward to Model Lab.
