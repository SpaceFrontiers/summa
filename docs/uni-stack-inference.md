# LLM compute architecture: Burn + CubeCL

Hermes has one MAL-driven `Transformer` implementation in `hermes-llm`.
`hermes-train` instantiates it with Burn Autodiff; generation instantiates it
with the plain backend.

## Backends

| Cargo features | Backend                    |
| -------------- | -------------------------- |
| none           | Burn ndarray (CPU)         |
| `metal`        | Burn WGPU (Metal on macOS) |
| `cuda`         | Burn CUDA                  |

Training wraps the selected backend in `burn_autodiff::Autodiff`. Inference
does not build an autodiff graph.

## Reused operations

Hermes uses Burn's ready implementations for:

- linear layers, embeddings, RMSNorm, LayerNorm, and dropout
- grouped `Conv1d` for Mamba's causal depthwise convolution
- `RotaryEncoding`; training follows its adjacent-pair convention
- scaled-dot-product attention, including CubeCL flash attention on eligible
  GPU shapes
- tensor matmul, activation, gather, reshape, and elementwise operations
- AdamW and norm gradient clipping

Mamba selective scan is the only custom compute operation. Burn/CubeCL does not
provide a selective state-space scan, and general prefix sum/product cannot
express its recurrence efficiently. GPU inference therefore uses a small
stateful `#[cube]` kernel on Burn's resident `CubeTensor` handles. CPU inference
and Autodiff training use the same readable tensor-operation recurrence; this
keeps backward correct without a second model implementation.

## Checkpoints

Training and inference use the same Burn-native module paths and safetensors
store. There is no PyTorch/Candle adapter, alternate schema, or permissive load.
Missing, unexpected, and shape-mismatched tensors fail loading.

The checkpoint contains only model parameters. RoPE tables and inference state
are derived/runtime data and are not serialized.
