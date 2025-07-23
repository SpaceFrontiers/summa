# Hermes Model Lab

Standalone, local-first observability UI for versioned traces produced by the
shared Hermes LLM model. It has no Vue, Tailwind, or Hermes WASM dependency.

## Live model

From the repository root, keep one checkpoint resident and serve Model Lab:

```bash
cargo run --release -p hermes-llm --features metal -- lab \
  --checkpoint checkpoint/weights.safetensors \
  --config checkpoint/config.json \
  --tokenizer tokenizer.json \
  --metrics checkpoint/metrics.jsonl
```

Open <http://127.0.0.1:4173/model-lab.html>. The server binds to loopback by
default, accepts one bounded inference job at a time, and serves this directory
and the trace API from the same origin. Use `--max-new-tokens`,
`--trace-tokens`, `--channel-bins`, and `--attention-heads` to adjust explicit
limits.

`hermes-llm lab --web-root` defaults to this project. A custom web root must
contain `model-lab.html` and its referenced `src/` assets.

## Static trace inspection

The development server needs no JavaScript install or WASM build:

```bash
cd hermes-model-lab
pnpm dev
```

It starts with a clearly marked synthetic trace. Real JSON bundles are opened
entirely in the browser; selected files never leave the machine. Live query
controls remain disabled when no `/api/status` endpoint is present.

Create a bundle from a checkpoint, optionally including trainer metrics:

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

The trace command reports every capture reduction. `--trace-tokens`,
`--channel-bins`, `--attention-heads`, and `--metrics-points` adjust the
bounded defaults. Training can add the optional layer-gradient heatmap with
`hermes-train train --layer-metrics-every N`.

## Quality checks and production bundle

```bash
pnpm install --frozen-lockfile
pnpm check
```

`pnpm check` runs strict ESLint checks, framework-independent Node tests, and a
Vite production build. The build retains `model-lab.html` and writes to
`hermes-model-lab/dist/`.

For command compatibility, the historical `pnpm lab:*` scripts in
`hermes-web` forward to this project.

## Project boundary

```text
hermes-model-lab/
├── model-lab.html       stable /model-lab.html entry
├── src/                 trace UI, styles, fixtures, and unit tests
├── eslint.config.js     Model Lab-only static checks
└── vite.config.js       standalone production bundle
```

Search UI code belongs in `hermes-web`; Model Lab must not import from it.

## License

MIT
