# hermes-mal

`hermes-mal` is the single parser and data model for Hermes' Model
Architecture Language (MAL). Both `hermes-llm` and `hermes-train` consume the
same `ModelDef`, and `hermes-mal-python` is a thin binding over this crate.

MAL supports reusable attention, state-space, FFN, and block definitions as
well as inline definitions:

```text
attention local_gqa {
    num_heads: 16
    num_kv_heads: 4
    window_size: 2048
    position_encoding: rope { theta: 10000 }
}

ffn gated {
    hidden_dim: 4096
    activation: swiglu
}

block transformer {
    attention: local_gqa
    ffn: gated
    norm: rmsnorm { eps: 1e-5 }
}

model example {
    vocab_size: 32000
    max_seq_len: 4096
    hidden_size: 1024
    num_layers: 24
    block: transformer
}
```

Parse a source containing one model with `parse_mal`, retain every named
definition with `parse_mal_full`, or use `parse_mal_file` for a local file:

```rust
let model = hermes_mal::parse_mal(source)?;
let layer = model.block_for_layer(0);
println!("heads: {}", layer.num_heads());
# Ok::<(), anyhow::Error>(())
```

Named references are resolved in source order. Duplicate names, undefined
references, unknown properties, unsupported syntax, and a source passed to
`parse_mal` with zero or multiple models are reported as errors.

The `well-known/` directory is embedded into the crate. Use
`list_wellknown_models`, `get_wellknown_mal`, and `get_builtin_model` to
discover or load those definitions without filesystem access.

For heterogeneous models, `pattern` is repeated cyclically across
`num_layers`. Use `ModelDef::block_for_layer` and the computed methods on
`BlockDef` when inspecting such a model; the computed methods directly on
`ModelDef` describe its homogeneous/default `block`.

Run the parser, built-in-model, parameter-estimation, and serialization
regression suite with:

```bash
cargo test -p hermes-mal
```
