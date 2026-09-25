# Contributing

Read [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md), and the
[search system contract](docs/search-system-contract.md) before changing search
code. Package guides are listed in the [README](README.md#packages).

## Setup

- Rust pinned by [rust-toolchain.toml](rust-toolchain.toml); `protoc` for gRPC.
- Python 3.12+ and `uv` for development; `maturin` for MAL bindings.
- Node.js 22.12+ and pnpm 10+ for web/TypeScript; `wasm-pack` and LLVM for WASM.

```bash
cargo build --release
pre-commit install
```

## Checks

Run from the repository root:

| Change                                          | Check                                                                                                      |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Search stack                                    | `python3 scripts/check_search.py check`                                                                    |
| Lifecycle or RPC                                | `python3 scripts/check_search.py full`                                                                     |
| Portable Rust workspace                         | `cargo test --workspace`                                                                                   |
| Rust formatting                                 | `cargo fmt --all -- --check`                                                                               |
| Rust lints                                      | `cargo clippy --workspace --all-targets -- -D warnings`                                                    |
| Rust API docs                                   | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`                                               |
| Markdown links, navigation, benchmark inventory | `uv run scripts/check_docs.py`                                                                             |
| WASM                                            | `(cd summa-wasm && bash build.sh && npm ci && npm test -- --run)`                                          |
| Python client                                   | `(cd summa-client-python && uv sync --group dev --group test && uv run pytest tests/test_client_unit.py)`  |
| TypeScript client                               | `pnpm --dir summa-client-typescript install --frozen-lockfile && pnpm --dir summa-client-typescript check` |
| MAL Python wheel                                | `(cd summa-mal-python && maturin build --release)`                                                         |
| Search UI                                       | `pnpm --dir summa-web test && pnpm --dir summa-web lint && pnpm --dir summa-web build`                     |
| Model Lab                                       | `pnpm --dir summa-model-lab install --frozen-lockfile && pnpm --dir summa-model-lab check`                 |

Install web dependencies and build WASM before checking the search UI. Python
integration tests need `target/debug/summa-server`. Protocol changes require
[regenerating both clients](summa-proto/README.md#regeneration-and-validation).

The search harness checks native-without-sync and standalone broker builds.
GPU backends require separate Metal/CUDA hosts and checks; `--all-features`
is not a portable test profile. See the [LLM code map](docs/llm-code-map.md)
and [dependency register](docs/upstream-dependencies.md).

Run all hooks with `pre-commit run --all-files` and
`pre-commit run --all-files --hook-stage pre-push`.

## Changes and evidence

- Reproduce bugs with behavior-named regression tests.
- Update the owning design document before substantial changes; preserve format,
  lifecycle, and native/WASM contracts.
- Keep documentation concise and source links relative. Label proposals and
  historical results; preserve their dates, workloads, and raw evidence.
- Follow the [benchmark protocol](docs/benchmarks.md#recorded-results-and-reporting)
  for performance claims.
- Open PRs against `main`; describe the behavior, validation, and unrun checks.

Use the [issue templates](.github/ISSUE_TEMPLATE/) for bugs and feature requests.
Contributions use the repository's MIT license.
