# Summa Model Lab

Standalone, local-first observability UI for versioned traces produced by the
shared Summa LLM model. It has no Vue, Tailwind, or Summa WASM dependency.

## Live model

From the repository root, serve a checkpoint with its training config and
tokenizer. Replace `sha256-HASH` with the generation in `checkpoint/current.json`:

```bash
cargo run --release -p summa-llm --features metal -- lab \
  --checkpoint checkpoint/generations/sha256-HASH/weights.safetensors \
  --config model.mal \
  --tokenizer tokenizer.json \
  --metrics checkpoint/metrics.jsonl
```

Open <http://127.0.0.1:4173/model-lab.html>. The server binds to loopback by
default, accepts one bounded inference job at a time, and serves this directory
and the trace API from the same origin. Use `--max-new-tokens`,
`--trace-tokens`, `--channel-bins`, and `--attention-heads` to adjust explicit
limits.

`summa-llm lab --web-root` defaults to this project. A custom web root must
contain `model-lab.html` and its referenced `src/` assets.

## Static trace inspection

The development server needs no JavaScript install or WASM build:

```bash
cd summa-model-lab
pnpm dev
```

It starts with a clearly marked synthetic trace. Real JSON bundles are opened
entirely in the browser; selected files never leave the machine. Live query
controls remain disabled when no `/api/status` endpoint is present.

From the repository root, create a bundle from a checkpoint, optionally including trainer metrics:

```bash
cargo run -p summa-llm -- trace \
  --checkpoint checkpoint/generations/sha256-HASH/weights.safetensors \
  --config model.mal \
  --tokenizer tokenizer.json \
  --prompt "Plan what evidence to retrieve" \
  --max-tokens 32 \
  --metrics checkpoint/metrics.jsonl \
  --output checkpoint/model-trace.json
```

The trace command reports every capture reduction. `--trace-tokens`,
`--channel-bins`, `--attention-heads`, and `--metrics-points` adjust the
bounded defaults. Training can add the optional layer-gradient heatmap with
`summa-train train --layer-metrics-every N`.

## Quality checks and production bundle

From `summa-model-lab`, using Node.js 22.12+ and pnpm 10+:

```bash
pnpm install --frozen-lockfile
pnpm check
```

`pnpm check` runs strict ESLint checks, framework-independent Node tests, and a
Vite production build. The build retains `model-lab.html` and writes to
`summa-model-lab/dist/`.

For command compatibility, the historical `pnpm lab:*` scripts in
`summa-web` forward to this project.

## Project boundary

```text
summa-model-lab/
├── model-lab.html       stable /model-lab.html entry
├── src/                 trace UI, styles, fixtures, and unit tests
├── eslint.config.js     Model Lab-only static checks
└── vite.config.js       standalone production bundle
```

Search UI code belongs in `summa-web`; Model Lab must not import from it.

## License

MIT
