# Configurable MoE design

The model stack supports mixture-of-experts as an opt-in FFN property. An FFN
without a `moe` block is the existing dense implementation with the same
parameter names, count, execution path, and checkpoint layout. In particular,
`retriever_100m.mal` remains dense; enabling MoE is a new-model or explicit
upcycling decision, never an implicit migration.

`well-known/retriever_200m_moe.mal` is the first sized MoE retriever. It keeps
the 24x512 hybrid backbone and changes exactly 12 of 24 FFNs to 8-expert,
top-2 MoE layers. Each expert has width 768, so the two routed experts have the
same aggregate width as a dense width-1536 FFN. MAL estimates 200,795,648
stored parameters and approximately 115,860,992 parameters touched per token
(including routers). MoE placement is balanced across Mamba and attention
blocks, and layer 24 remains a global-attention retrieval layer.

The model deliberately has no shared expert. OLMoE's controlled comparison
found a small regression from forcing one always-active expert at fixed total
and active budgets. Its load-balancing and router z-loss weights follow the
same study's stable dropless token-choice recipe.

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
and the rank-3 routed-expert banks use AdamW; ordinary 2-D hidden matrices retain
Muon. The trainer adds and separately reports `router_aux_loss`, while `loss`
remains the task loss and `optimized_loss` includes both.

The Burn dispatch groups routes by expert and restores token order with an
inverse permutation. Inference evaluates compact expert batches without
padding. Training keeps expert weights in their batched execution layout and,
when the largest expert batch keeps total padded work at or below 1.5x useful
routes, executes two batched GEMMs; more skewed routing falls back to compact
expert batches. The permutation has a matching inverse-permutation backward,
so it avoids the generic scatter-add path. A small CubeCL kernel handles top-2
selection on GPU; other top-k values retain the portable implementation. The
measured A100 comparison is in [MoE performance](moe-performance.md).

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
  with a shared expert. It should therefore be measured on the target corpus.
- Soft MoE is not the first choice for autoregressive serving: its soft
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