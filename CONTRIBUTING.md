# Contributing to Hermes

Thank you for your interest in contributing to Hermes. This guide will help you get started.

## Development Setup

### Prerequisites

- **Rust nightly** (see `rust-toolchain.toml` for the exact version; 1.92+ required)
- **protoc** (Protocol Buffers compiler, for gRPC builds)
- **Node.js 20+** (for WASM and web UI development)
- **Python 3.12+** (for Python bindings)
- **wasm-pack** (for WASM builds)
- **pre-commit** (for automated code quality checks)

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
cargo test --all-features
```

## Build Commands

| Command                                                    | Description              |
| ---------------------------------------------------------- | ------------------------ |
| `cargo build --release`                                    | Build all Rust packages  |
| `cargo test --all-features`                                | Run the full test suite  |
| `cargo fmt --all`                                          | Format all Rust code     |
| `cargo clippy --all-targets --all-features -- -D warnings` | Run lints                |
| `cd hermes-wasm && wasm-pack build --release --target web` | Build WASM package       |
| `cd hermes-core-python && maturin build --release`         | Build Python wheel       |
| `pre-commit run --all-files`                               | Run all pre-commit hooks |

## Project Structure

| Crate                    | Description                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| **hermes-core**          | Core search engine library (async, BM25 ranking, WAND optimization, segment-based storage) |
| **hermes-server**        | gRPC server for remote search and index operations                                         |
| **hermes-tool**          | CLI for index management and data processing pipelines                                     |
| **hermes-wasm**          | WebAssembly bindings for browser-based search                                              |
| **hermes-web**           | Vue.js web UI                                                                              |
| **hermes-llm**           | LLM training framework built on Candle ML                                                  |
| **hermes-proto**         | Protocol Buffer definitions for gRPC services                                              |
| **hermes-client-python** | Python gRPC client library                                                                 |

For a deeper look at the core architecture, see `CLAUDE.md`.

## Submitting Pull Requests

1. **Fork** the repository and create a new branch from `main`:

   ```bash
   git checkout -b my-feature main
   ```

2. **Make your changes.** Write clear, focused commits. Include tests for new functionality.

3. **Run checks locally** before pushing:

   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features
   ```

4. **Push** your branch and open a pull request against `main`.

5. In the PR description, explain what the change does and why. Reference any related issues.

6. Address review feedback. Once approved, a maintainer will merge your PR.

## Code Style

- **Formatting**: All Rust code must pass `cargo fmt`. The repository uses the default rustfmt configuration.
- **Linting**: All code must pass `cargo clippy` with warnings treated as errors.
- **Pre-commit hooks**: The project uses pre-commit hooks that run rustfmt, clippy, ruff (Python), and prettier (JS/TS). Install them with `pre-commit install` so checks run automatically before each commit.
- **Tests**: New features and bug fixes should include tests. Run `cargo test --all-features` to verify.

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
