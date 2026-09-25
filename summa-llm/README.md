# summa-llm

Inference for Summa Transformer, Mamba, and hybrid models. Model architectures
are defined with the shared MAL parser; training lives in `summa-train`.

## Backends

```bash
# CPU (default when no GPU feature is selected)
cargo build --release -p summa-llm

# Apple Metal
cargo build --release -p summa-llm --features metal

# NVIDIA CUDA
cargo build --release -p summa-llm --features cuda
```

The runtime supplies embeddings, linear layers, normalization, RoPE, and
optimized CubeCL kernels. Summa adds CubeCL training kernels for Mamba's
selective scan, depthwise convolution, and attention backward. GPU tensors stay
resident throughout inference and training.

The commands below assume `target/release` is on `PATH` (for example,
`export PATH="$PWD/target/release:$PATH"` from the repository root).

## Generate text

Replace `sha256-HASH` with the generation named by `checkpoint/current.json`;
use the same MAL/JSON config and tokenizer as training. The trainer does not
write weights or config at the checkpoint root.

```bash
summa-llm generate \
  --checkpoint checkpoint/generations/sha256-HASH/weights.safetensors \
  --config model.mal \
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

Config accepts MAL or JSON. With the default `remote` feature, all three
artifact arguments also accept `s3://`, `gs://`, and HTTP(S) URIs;
downloads are cached under `~/.summa-cache` or `$SUMMA_CACHE`.

Backend choice is a build decision. A build without `metal` or `cuda` uses CPU.
Generation preallocates only the KV positions that the prompt and requested
continuation can consume, instead of every model position. It skips the unused
final decode step and rebuilds a bounded half-window only when the configured
context limit is reached. `Transformer::make_state_with_capacity` exposes the
same right-sized behavior to library callers; `make_state` retains the
max-context convenience behavior.

## Inspect or export MAL models

```bash
summa-llm info --model gpt2-small
summa-llm export --model models/custom.mal --output config.json
```

Well-known MAL definitions include `nano`, `tiny`, GPT-2, LLaMA, Mistral, and
hybrid Transformer/Mamba presets.

## Library usage

```rust,no_run
use summa_llm::{TextGenerator, Transformer, default_device, load_safetensors};

# fn main() -> anyhow::Result<()> {
let config = summa_llm::ModelDef::from_json("config.json")?;
let device = default_device();
let mut model = Transformer::new(&config, &device)?;
load_safetensors(&mut model, "weights.safetensors")?;
let generator = TextGenerator::new(&model, &device);
# Ok(())
# }
```

The safetensors loader is strict and consumes the checkpoint written by
`summa-train` without conversion. Checkpoint configs retain the tokenizer's
logical vocabulary size while embedding and output tensors use a derived
64-row storage alignment.

## Architecture support

- Multi-head and grouped-query causal attention
- RoPE and sliding-window attention
- RMSNorm and LayerNorm
- gated and non-gated FFNs
- optional configurable top-k MoE FFNs (see [MoE design](../docs/moe-design.md))
- Mamba selective state-space blocks
- mixed Transformer/Mamba layer patterns
- tied or untied output embeddings
- right-sized incremental KV-cache and recurrent-state decoding

## Benchmarks and diagnostics

See the [benchmark guide](../docs/benchmarks.md) for CPU smoke runs and CUDA
MoE/memory acceptance runs, and [Model Lab](../summa-model-lab/README.md)
for live inference and trace inspection.

## License

MIT
