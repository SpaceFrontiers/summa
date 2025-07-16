# LLM inference and training code map

Hermes uses one MAL model definition and one Burn `Transformer` implementation
for training, generation, and retrieval. There is no alternate PyTorch or
Candle model stack.

| Area                     | Entry points                                                                   | Responsibility                                                                                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture             | `hermes-mal/src/lib.rs`, `hermes-mal/src/mal.pest`, `hermes-mal/well-known/`   | Parse composable MAL definitions, resolve references, and expose the serializable `ModelDef`. `parse_mal` requires exactly one model; tools that select among several use `parse_mal_full`. |
| Model assembly           | `hermes-llm/src/model/transformer.rs`, `block.rs`                              | Validate dimensions and numeric settings, construct homogeneous or patterned attention/Mamba layers, and own the shared forward/loss/stateful paths.                                        |
| Attention                | `hermes-llm/src/model/attention.rs`, `fused_attention.rs`, `cube_attention.rs` | Grouped-query projection, RoPE, causal/window masks, KV caching, backend selection, and the CUDA training backward.                                                                         |
| Mamba                    | `hermes-llm/src/model/mamba.rs`, `model/scan/`, `model/conv.rs`                | Stateful selective-SSM mixing, CPU correctness references, autodiff nodes, and checkpointed CUDA/Metal kernels.                                                                             |
| Loss and numerics        | `hermes-llm/src/model/linear_cross_entropy.rs`, `norm.rs`, `matmul.rs`         | Chunked vocabulary loss, normalization, precision policy, and native/fused matmul entry points.                                                                                             |
| Generation and artifacts | `hermes-llm/src/generate.rs`, `tokenizer.rs`, `remote.rs`, `model/weights.rs`  | Tokenization/sampling, local or cached remote artifact resolution, and safetensors loading/saving.                                                                                          |
| Corpus pipeline          | `hermes-train/src/data.rs`                                                     | Stream text/JSONL/Zstandard data, batch-tokenize, EOS-pack without padding, and bounded deterministic shuffle.                                                                              |
| Checkpoints              | `hermes-train/src/checkpoint.rs`                                               | Atomically publish model/optimizer/training state and restore Burn parameter IDs for exact resume.                                                                                          |
| Optimization loop        | `hermes-train/src/main.rs`, `muon.rs`                                          | CLI validation, curriculum/epoch position, gradient accumulation/clipping, schedule, Muon + AdamW steps, and metrics.                                                                       |

## Validation layers

- `cargo test -p hermes-mal -p hermes-llm -p hermes-train` covers the parser,
  CPU model paths, streaming corpus logic, optimizer behavior, and checkpoint
  resume.
- `cargo clippy -p hermes-mal -p hermes-llm -p hermes-train --all-targets -- -D warnings`
  is the required host lint gate.
- CUDA and Metal kernel parity tests compare accelerator results with the
  tensor-operation references. Performance changes additionally require the
  end-to-end loss/gradient and steady-state throughput gates documented in the
  relevant file under `docs/`.

Temporary dependency forks and their upstream exit criteria are tracked in
`docs/forked-dependencies.md`.
