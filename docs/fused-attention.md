# Attention kernels

Hermes uses Burn tensor and module operations by default. CUDA attention calls
Burn's CubeCL integration, which launches CubeK Flash Attention with FP16 inputs
and FP32 accumulation. CPU and Metal use Burn's portable attention operations.
There are no cuDNN, C++, or Python runtime dependencies.

The only custom attention component is the CUDA training autodiff boundary.
Burn's current attention API returns the output but not the per-row log-sum-exp
needed by CubeK's experimental fused backward. Materializing the full
`[batch, heads, sequence, sequence]` autodiff graph is not viable at a
4096-token context, so Hermes saves Q/K/V and the output, then recomputes exact
probabilities in fixed query-row chunks during backward. The chunks use Burn's
CubeCL matmuls and reductions; no scalar attention kernel is maintained here.

Grouped-query K/V heads are expanded immediately before the attention operation.
Autodiff reduces their gradients back to the compact head count. This is linear
in sequence length for the current 2:1 head ratio.

The kernel policy is deliberately narrow:

- use Burn/CubeK implementations whenever they support the required operation;
- keep a local kernel boundary only for a measured throughput or peak-memory win;
- compare forward output and Q/K/V gradients with the Burn reference; and
- gate changes with steady-state A100 tokens/s, utilization, power, and peak
  memory at the production shape (batch 2, sequence 4096).

FlashAttention-2 is the relevant algorithm family for the SM80 A100.
FlashAttention-3 targets Hopper; FlashAttention-4 targets Hopper and Blackwell.
Their hardware-specific mechanisms are not emulated on Ampere.

Burn's optional whole-graph fusion is separate from these explicit kernels. It
is disabled because the tested global fusion bridge produced invalid CubeCL MSL
for this model. Explicit Burn/CubeK kernels remain enabled and are the optimized
path.
