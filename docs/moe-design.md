# Configurable MoE design

Hermes supports mixture-of-experts as an opt-in FFN property. An FFN without a
`moe` block is the existing dense implementation with the same parameter names,
count, execution path, and checkpoint layout. In particular,
`retriever_100m.mal` remains dense; enabling MoE is a new-model or explicit
upcycling decision, never an implicit migration.

```mal
ffn retrieval_moe {
    hidden_dim: 768
    activation: swiglu
    bias: false
    moe {
        experts: 8
        top_k: 2
        shared_experts: 0
        load_balance_loss_weight: 0.01
        router_z_loss_weight: 0.001
    }
}
```

The router is an unbiased FP32 linear projection. It selects the top-k routed
experts per token, renormalizes their probabilities, and never drops tokens.
Optional shared experts are always active. Routed and shared experts retain the
configured dense FFN activation, width, gate, bias, and dropout. Router weights
use AdamW; expert matrices retain Muon. The trainer adds and separately reports
`router_aux_loss`, while `loss` remains the task loss and `optimized_loss`
includes both.

The portable Burn dispatch currently evaluates static expert shapes and masks
inactive routes. This gives deterministic semantics, autodiff, checkpointing,
and CPU/Metal/CUDA portability, but does not yet reduce expert FLOPs. Production
MoE throughput requires replacing that internal dispatch with a grouped
block-sparse CubeCL kernel; the MAL and checkpoint schema do not need to change.

## Research conclusions

- Start with dropless token-choice routing. OLMoE found it better than
  expert-choice routing in its controlled study, and MegaBlocks shows why a
  dropless block-sparse implementation avoids the quality/capacity trade-off of
  padding or token dropping.
- Fine-grained experts are particularly attractive for a small model. OLMoE and
  the fine-grained scaling-law study find gains from more, smaller experts,
  with diminishing returns. Mixtral's top-2 routing is a well-tested baseline.
- Keep classical load balancing (`0.01`) and router z-loss (`0.001`) as the
  stable first experiment. OLMoE reports better stability with both. An
  auxiliary-loss-free bias controller, as used by DeepSeek-V3, is a worthwhile
  later A/B, not a silent default change.
- Shared experts are configurable but default off. DeepSeekMoE motivates shared
  expert isolation, while OLMoE's matched-compute experiment did not improve
  with a shared expert. It should therefore be measured on Hermes data.
- Soft MoE is not the first choice for autoregressive Hermes serving: its soft
  token-slot mixing is less natural for causal one-token decoding than token
  choice and complicates an efficient KV-aware decode path.
- Sparse upcycling can initialize experts from a trained dense FFN and is the
  safest way to reuse the current weights. It still creates a different model
  signature and must write a new checkpoint directory.

Primary sources:

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/html/2409.02060v2)
- [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70.pdf)
- [MegaBlocks](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)
- [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871)
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [From Sparse to Soft Mixtures of Experts](https://proceedings.iclr.cc/paper_files/paper/2024/file/79fea214543ba263952ac3f4e5452b14-Paper-Conference.pdf)

## First Hermes experiment (not applied to the current model)

For a controlled small-model A/B, retain the current 24-layer mixer order and
replace only four FFNs (layers 6, 12, 18, and 24) with 8 routed experts,
top-2, width 768, and no shared expert. Two active width-768 experts match the
active FFN matmul size of one width-1536 dense FFN. The resulting model is
approximately 144.14M total parameters versus 115.81M today, while active FFN
compute in converted layers is approximately unchanged. Compare task loss per
token, wall-clock convergence, router entropy/load skew, retrieval quality,
and decode latency before expanding MoE to more layers.
