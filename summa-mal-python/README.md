# summa-mal Python bindings

PyO3 bindings for the Summa Model Architecture Language (MAL) parser.

The repository directory and Rust extension crate are named
`summa-mal-python`; the published Python distribution remains `summa-mal`
and its import module remains `summa_mal`.

This wheel is a thin wrapper around the Rust `summa-mal` crate — the single
source of truth for parsing `.mal` model definitions. It exposes one function:

```python
from summa_mal import parse_mal

json_str = parse_mal(source)  # -> str (serde JSON of ModelDef)
```

The returned compact JSON represents the same `ModelDef` as `summa-llm
export` (which writes pretty-printed JSON) and can be consumed by any
serde-compatible tool.
Syntax errors, unknown keys, and undefined references raise `ValueError`.

## Development

From this directory, build and install the extension in the active virtual
environment, then smoke-test the import:

```bash
maturin develop
python -c 'from pathlib import Path; from summa_mal import parse_mal; print(parse_mal(Path("../summa-mal/well-known/tiny.mal").read_text()))'
```

The Rust seam has a fast workspace test:

```bash
cargo test -p summa-mal-python
```
