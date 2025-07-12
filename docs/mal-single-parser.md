# MAL: one parser, two languages

The Model Architecture Language (MAL) is now parsed by **exactly one**
implementation — the Rust `hermes-mal` crate. The former hand-written Python
parser (`hermes-train/src/hermes_train/mal.py`) has been replaced by a thin
shim over the Rust parser, exposed to Python via PyO3/maturin. This removes the
drift risk of keeping two parsers (a pest grammar and a recursive-descent Python
port) byte-for-byte in sync.

## Crates

| Crate           | Path             | Role                                                                                                                                                                                               |
| --------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hermes-mal`    | `hermes-mal/`    | The parser: pest grammar + AST/serde structs + embedded `well-known/*.mal`. Light deps only (pest, serde, serde_json, anyhow, rust-embed) — no candle/tokenizers, so the Python wheel stays small. |
| `hermes-mal-py` | `hermes-mal-py/` | PyO3 bindings. `cdylib` named `hermes_mal`, exposes `parse_mal(source: str) -> str` (JSON), errors mapped to `ValueError`.                                                                         |
| `hermes-llm`    | `hermes-llm/`    | Re-exports the parser: `pub use hermes_mal as mal;` — every existing `crate::mal::…` / `hermes_llm::mal::…` path resolves unchanged.                                                               |

`hermes-mal` is where `hermes-llm/src/mal/` (`mod.rs` → `src/lib.rs`, `mal.pest`)
and `hermes-llm/well-known/` used to live.

## How hermes-train imports it

`hermes-train/pyproject.toml` declares a dependency `hermes-mal` with a uv
source pointing at the local `hermes-mal-py` crate:

```toml
[tool.uv.sources]
hermes-mal = { path = "../hermes-mal-py" }
```

`uv sync` builds the wheel (maturin backend) automatically. `hermes_train.mal`
is then a thin wrapper:

```python
from hermes_mal import parse_mal as _rust_parse_mal

def parse_mal(source: str) -> dict:
    return json.loads(_rust_parse_mal(source))
```

`MalError` is kept as an alias of `ValueError` (what the Rust binding raises) for
backward compatibility. The `hermes-train train --config x.mal` path is
unchanged — `cli.load_model_dict` still calls `parse_mal`.

## Building / installing the wheel

- **Dev (editable, into the hermes-train venv):**
  ```bash
  cd hermes-mal-py && maturin develop
  ```
  or just `cd hermes-train && uv sync` (uv builds it from the path source).
- **Release wheel:**
  ```bash
  cd hermes-mal-py && maturin build --release
  # -> target/wheels/hermes_mal-<ver>-cp3xx-...whl
  ```

### macOS note

A PyO3 `extension-module` `cdylib` references Python symbols resolved at load
time. maturin injects `-undefined dynamic_lookup`, but a bare workspace
`cargo build` does not — so `hermes-mal-py/build.rs` emits that linker flag on
macOS/iOS. This keeps `cargo build` (whole workspace) green without maturin.

## CI

The `hermes-train` job now installs a Rust toolchain (`dtolnay/rust-toolchain`)
because `uv sync` compiles `hermes-mal` via maturin.

## Verified

- `cargo build` (workspace), `cargo test -p hermes-mal`, `cargo test -p hermes-llm --features metal,remote`
- `cargo clippy -p hermes-mal -p hermes-mal-py -- -D warnings`
- `cd hermes-mal-py && maturin build --release`
- `cd hermes-train && uv run pytest` (55 passed) and `uv run ruff check src/ tests/`
- `hermes-train train --config <preset>.mal` model-config load, end-to-end.

## Regenerating fixtures

`hermes-train/tests/fixtures/mal/*.json` are the canonical `hermes-llm export`
outputs. When the Rust schema changes, regenerate them with
`hermes-llm export --model <name|file.mal>`. `tests/test_mal.py` pins the
PyO3 parser output byte-for-byte against them (and thus also smoke-tests the
wheel + wiring).
