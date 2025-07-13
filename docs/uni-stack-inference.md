# LLM compute architecture

Hermes has one MAL-driven `Transformer` implementation in `hermes-llm`.
`hermes-train` places it on an autodiff device; generation uses the same module
on the plain device.

## Backends

| Cargo features | Backend               |
| -------------- | --------------------- |
| none           | ndarray (CPU)         |
| `metal`        | WGPU (Metal on macOS) |
| `cuda`         | CubeCL CUDA           |

Training wraps the selected backend in `burn_autodiff::Autodiff`. Inference
does not build an autodiff graph.

## Reused operations

Hermes uses Burn's ready implementations for:

- linear layers, embeddings, RMSNorm, LayerNorm, and dropout
- grouped `Conv1d` for the CPU Mamba depthwise-convolution reference
- `RotaryEncoding`; training follows its adjacent-pair convention
- scaled-dot-product attention, including CubeCL flash attention on eligible
  GPU shapes
- tensor matmul, activation, gather, reshape, and elementwise operations
- AdamW and the tensor reductions used by global gradient clipping

Hermes adds custom CubeCL forward and backward operations where the runtime has
no efficient training primitive: Mamba selective scan, GPU depthwise
convolution, and CUDA attention backward. The memory-bounded output loss and
attention backward are composed from Burn tensor operations and CubeCL
matmuls. CUDA attention forward reuses CubeK Flash Attention through Burn;
RoPE reuses `RotaryEncoding`. CPU keeps readable Burn tensor-op references for
parity tests. There is still only one model implementation.

## Checkpoints

Training and inference use the same module paths and safetensors store. There is
no PyTorch/Candle adapter, alternate schema, or permissive load.
Missing, unexpected, and shape-mismatched tensors fail loading.

The checkpoint contains only model parameters. RoPE tables and inference state
are derived/runtime data and are not serialized.
