# hermes-mal (Python)

PyO3 bindings for the Hermes Model Architecture Language (MAL) parser.

This wheel is a thin wrapper around the Rust `hermes-mal` crate — the single
source of truth for parsing `.mal` model definitions. It exposes one function:

```python
from hermes_mal import parse_mal

json_str = parse_mal(source)  # -> str (serde JSON of ModelDef)
```

The returned JSON is byte-for-byte identical to what `hermes-llm export`
emits, i.e. the dict consumed by `hermes_train.config.ModelDef.from_dict`.
Syntax errors, unknown keys, and undefined references raise `ValueError`.
