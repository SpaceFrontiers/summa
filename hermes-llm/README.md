# hermes-llm

Inference for Hermes Transformer, Mamba, and hybrid models. Model architectures
are defined with the shared MAL parser; training lives in `hermes-train`.

## Backends

```bash
# CPU (default when no GPU feature is selected)
cargo build --release -p hermes-llm

# Apple Metal
cargo build --release -p hermes-llm --features metal

# NVIDIA CUDA
cargo build --release -p hermes-llm --features cuda
```

The runtime supplies embeddings, linear layers, normalization, RoPE, and
optimized CubeCL kernels. Hermes adds CubeCL training kernels for Mamba's
selective scan, depthwise convolution, and attention backward. GPU tensors stay
resident throughout inference and training.

## Generate text

```bash
hermes-llm generate \
  --checkpoint checkpoint/weights.safetensors \
  --config checkpoint/config.json \
  --tokenizer tokenizer.json \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.9 \
  --top-k 40 \
  --repetition-penalty 1.1
```

`--repetition-penalty` uses the standard sign-aware logit adjustment for
tokens already present in the context. `1.0` disables it; values around
`1.05`–`1.2` are useful starting points for repetitive checkpoints.

The checkpoint, config, and tokenizer arguments accept local paths. With the
default `remote` feature they also accept `s3://`, `gs://`, and HTTP(S) URIs;
downloads are cached under `~/.hermes-cache` or `$HERMES_CACHE`.

Backend choice is a build decision. A build without `metal` or `cuda` uses CPU.

## Inspect or export MAL models

```bash
hermes-llm info --model gpt2-small
hermes-llm export --model models/custom.mal --output config.json
```

Well-known MAL definitions include `nano`, `tiny`, GPT-2, LLaMA, Mistral, and
hybrid Transformer/Mamba presets.

## Library usage

```rust,no_run
use hermes_llm::{TextGenerator, Transformer, default_device, load_safetensors};

# fn main() -> anyhow::Result<()> {
let config = hermes_llm::ModelDef::from_json("config.json")?;
let device = default_device();
let mut model = Transformer::new(&config, &device)?;
load_safetensors(&mut model, "weights.safetensors")?;
let generator = TextGenerator::new(&model, &device);
# Ok(())
# }
```

The safetensors loader is strict and consumes the checkpoint written by
`hermes-train` without conversion. Checkpoint configs retain the tokenizer's
logical vocabulary size while embedding and output tensors use a derived
64-row storage alignment.

## Architecture support

- Multi-head and grouped-query causal attention
- RoPE and sliding-window attention
- RMSNorm and LayerNorm
- gated and non-gated FFNs
- Mamba selective state-space blocks
- mixed Transformer/Mamba layer patterns
- tied or untied output embeddings
- incremental KV-cache and recurrent-state decoding

## License

MIT
