# Contributing to Hermes

Thank you for your interest in contributing to Hermes. This guide will help you get started.

## Development Setup

### Prerequisites

- **Rust 1.98+** (see `rust-toolchain.toml` for the exact version)
- **protoc** (Protocol Buffers compiler, for gRPC builds)
- **Node.js 20+** (for WASM and web UI development)
- **Python 3.12+** (for Python bindings)
- **wasm-pack** (for WASM builds)
- **pre-commit** (for automated code quality checks)
- **uv**, **pnpm**, and **maturin** (for Python, web, and binding projects)

### Getting Started

```bash
git clone https://github.com/<your-fork>/hermes.git
cd hermes
```

Install pre-commit hooks:

```bash
pre-commit install
```

Verify your setup by running the full build and test suite:

```bash
cargo build --release
cargo test --workspace
```

## Build Commands

| Command                                                                            | Description                       |
| ---------------------------------------------------------------------------------- | --------------------------------- |
| `cargo build --release`                                                            | Build all Rust packages           |
| `cargo test --workspace`                                                           | Run the portable Rust test suite  |
| `cargo fmt --all -- --check`                                                       | Check Rust formatting             |
| `cargo clippy --workspace --all-targets -- -D warnings`                            | Run portable Rust lints           |
| `cargo check -p hermes-core --no-default-features --features native --all-targets` | Check Core's native-only boundary |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`                       | Check all Rust API documentation  |
| `cd hermes-wasm && bash build.sh && npm ci && npm test`                            | Build and test the WASM package   |
| `cd hermes-client-python && uv build`                                              | Build the Python gRPC client      |
| `cd hermes-client-typescript && pnpm install && pnpm run build`                    | Build the TypeScript gRPC client  |
| `cd hermes-mal-python && maturin build --release`                                  | Build the MAL Python binding      |
| `cd hermes-web && pnpm test`                                                       | Run the search web unit tests     |
| `cd hermes-model-lab && pnpm check`                                                | Check and build the LLM Model Lab |
| `pre-commit run --all-files`                                                       | Run commit-stage hooks            |
| `pre-commit run --all-files --hook-stage pre-push`                                 | Run push-stage hooks              |

## Project Structure

| Project                      | Description                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| **hermes-core**              | Core search engine library (async, BM25 ranking, WAND optimization, segment-based storage) |
| **hermes-server**            | gRPC server for remote search and index operations                                         |
| **hermes-broker**            | gRPC broker routing the same protocol across many hermes-server instances                  |
| **hermes-tool**              | CLI for index management and data processing pipelines                                     |
| **hermes-wasm**              | WebAssembly bindings for browser-based search and indexing                                 |
| **hermes-web**               | Vue/WASM search UI                                                                         |
| **hermes-model-lab**         | Standalone local LLM trace and observability UI                                            |
| **hermes-client-python**     | Async Python gRPC client                                                                   |
| **hermes-client-typescript** | TypeScript gRPC client                                                                     |
| **hermes-proto**             | Protocol Buffer definitions shared by the server and clients                               |
| **hermes-mal**               | Model Architecture Language parser and well-known definitions                              |
| **hermes-mal-python**        | Thin PyO3 binding around the shared `hermes-mal` parser                                    |
| **hermes-tokenizer**         | Stable-Rust byte-level BPE tokenizer used by training and inference                        |
| **hermes-llm**               | Burn-based shared model, inference, generation, and accelerator kernels                    |
| **hermes-train**             | Autodiff training for the same `hermes-llm` model and safetensors checkpoints              |

For a deeper look at the core architecture, see `CLAUDE.md`. The shared LLM
stack is mapped in `docs/llm-code-map.md`; temporary official GPU revisions and
their release exit criteria live in `docs/upstream-dependencies.md`.

## Submitting Pull Requests

1. **Fork** the repository and create a new branch from `main`:

   ```bash
   git checkout -b my-feature main
   ```

2. **Make your changes.** Write clear, focused commits. Include tests for new functionality.

3. **Run checks locally** before pushing:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
   cargo test --workspace
   ```

4. **Push** your branch and open a pull request against `main`.

5. In the PR description, explain what the change does and why. Reference any related issues.

6. Address review feedback. Once approved, a maintainer will merge your PR.

## Code Style

- **Formatting**: All Rust code must pass `cargo fmt --all -- --check`. Repository-wide settings live in `rustfmt.toml`.
- **Linting**: All code must pass `cargo clippy` with warnings treated as errors.
- **Pre-commit hooks**: The project uses pre-commit hooks that run rustfmt, clippy, Ruff, and Prettier for repository documents/configuration. Install them with `pre-commit install` so checks run automatically before each commit.
- **Tests**: Refactors must keep focused regression tests around the extracted behavior. Run `cargo test --workspace` for the portable suite, then run the affected Python, TypeScript, WASM, or web harness listed above.
- **GPU backends**: Metal and CUDA are validated separately; do not use `--all-features` as a portable substitute for backend-specific checks. Run `pre-commit run clippy-metal --all-files --hook-stage manual` or the corresponding `clippy-cuda` hook on a compatible host.

## Good First Issues

If you are looking for a place to start, these areas are well-suited for first-time contributors:

- **Add a new stemmer language**: The tokenizer module (`hermes-core/src/tokenizer/`) supports 15+ Snowball stemmers. Adding a new language involves registering the stemmer and adding tests.
- **CLI improvements**: The CLI (`hermes-tool/src/main.rs`) uses clap for argument parsing. Improvements to help text, new utility subcommands, or better error messages are welcome.
- **Client library examples**: Add usage examples or improve documentation for `hermes-client-python` or other client libraries.
- **Documentation**: Improve inline docs, add examples to public APIs, or expand the schema reference in `docs/schema.md`.

Look for issues labeled `good first issue` in the issue tracker to find specific tasks.

## Reporting Bugs and Requesting Features

Please use the GitHub issue templates when filing bug reports or feature requests. See `.github/ISSUE_TEMPLATE/` for the available templates.

## License

By contributing to Hermes, you agree that your contributions will be licensed under the same license as the project.
