# SOTA LLM design (2024–2026) + shared retrieval embeddings — research notes

Compiled 2026-07-15 for the Hermes retriever-100m direction (hybrid
Transformer+Mamba, agentic retrieval-oriented LLM). Status tags: **[Proven]**
in ≥1 frontier/production model; **[Emerging]** strong results, spreading;
**[Frontier]** promising research, not yet validated at scale.

## Part A — LLM architecture & training

### Attention

- **GQA** [Proven, universal default] — the baseline KV reducer (Llama 3, Qwen3,
  Mistral, Gemma). Hermes already does GQA.
- **MLA (Multi-head Latent Attention)** [Proven at scale] — DeepSeek-V2/V3 cache a
  low-rank latent per token (~an order of magnitude smaller KV than MHA) with a
  decoupled RoPE dim; matches/beats MHA quality but mostly pays off >~100B.
  https://arxiv.org/abs/2405.04434
- **Local/global sliding-window interleave** [Proven] — Gemma 2 ~1:1 local:global,
  Gemma 3 pushes 5:1 (local window 1024) to cut long-context KV memory to <15% at
  32K with minimal ppl impact. Trend 2024→26 is _more_ local per global.
  https://arxiv.org/html/2503.19786v1
- **Attention sinks** [Proven] — keep ~4 initial "sink" tokens (StreamingLLM) or a
  per-head learnable sink logit (GPT-OSS) so heads can "attend to nothing"; needed
  for windowed/streaming inference + bf16 stability. https://arxiv.org/abs/2309.17453
- **RoPE + θ-scaling** [Proven] default; **NoPE** interleaved in a subset of layers
  (Llama 4 iRoPE 3:1, Cohere Command A) helps length-generalization [Emerging];
  pure NoPE at scale [Frontier/failed].
- **Long-context extension**: **YaRN** (NTK-by-parts + attention-temperature) is the
  production standard, reaches 128K in a few hundred finetune steps; PI/NTK simpler;
  DCA for ≥256K–1M. Applied in a dedicated late phase, not from scratch.
  https://arxiv.org/abs/2309.00071

### Normalization & stability

- **Pre-norm + RMSNorm** [Proven, universal]. **Peri-LN / sandwich norm** (norm
  before AND after each sublayer; Gemma 2, OLMo 2) reduces variance blowup + loss
  spikes [Emerging]. https://arxiv.org/pdf/2502.02732
- **QK-norm** [Proven, spreading fast] — RMSNorm on Q,K pre-RoPE bounds attention
  logits, near-"free" stability at high LR/bf16. Gemma 3 _replaced_ logit
  soft-capping with QK-norm. **Hermes already has QK-norm.**
- **z-loss** [Proven] — λ·(logZ)² keeps softmax logits from drifting (PaLM, OLMo 2).
  Cheap; worth adding for long bf16 runs.
- **MuonClip / QK-clip** [Emerging, frontier] — rescale Q/K weights when max logit
  exceeds τ; let Muon train Kimi K2 (1T) with zero loss spikes.

### FFN / MoE

- **SwiGLU** [Proven, universal]. **Hermes already uses SwiGLU.**
- **Fine-grained + shared-expert MoE** [Proven, frontier default] — DeepSeekMoE:
  many small experts + 1 always-on shared expert; **aux-loss-free bias balancing**
  (per-expert routing bias nudged online, no gradient interference) is the 2024→25
  shift, widely copied. https://arxiv.org/abs/2412.19437 · https://arxiv.org/pdf/2408.15664
  Ultra-sparse is the frontier (Qwen3-Next 80B total / 3B active).
  _Not relevant at 100M dense, but the recipe to adopt if Hermes ever scales._

### Hybrid attention + SSM/linear-attention — **directly validates Hermes**

- Core thesis, repeatedly confirmed: **a few full-attention layers give precise
  recall; the rest can be cheap linear/SSM** — the hybrid beats both pure stacks at
  lower cost. Pure Mamba underperforms on **recall/copy/in-context retrieval**
  (fixed-size state loses exact tokens); attention is the backstop.
  https://arxiv.org/pdf/2410.03810
- Ratios in the wild cluster at **1 attention : 3–7 linear/SSM**: Jamba 1:7,
  MiniMax 7:1, Hunyuan ~8:1, Granite 4 9:1, Nemotron-H ~8% attention, Zamba 6:1;
  the newer gated-delta models (Qwen3-Next, Kimi Linear) run a heavier **3:1**.
- **Hermes retriever-100m is 2:1 mamba:attn (pattern [mamba, mamba, attn]).** That's
  slightly attention-heavy vs the ~1:3–1:7 band — reasonable for a _retrieval_
  model (recall-sensitive), and cheap to revisit. Mamba-2 / gated-DeltaNet / KDA are
  the modern linear cores if we upgrade from Mamba-1.
  Kimi Linear https://arxiv.org/abs/2510.26692 · Jamba https://arxiv.org/pdf/2403.19887

### Training

- **Optimizers**: AdamW [Proven baseline]; **Muon** [Emerging→frontier] ~2× compute
  efficiency on 2D matrices, needs weight decay + update-RMS matching; **MuonClip**
  scaled it to 1T (Kimi K2). **Hermes already uses Muon+AdamW.** SOAP/Shampoo
  [Emerging] heavier; Adam-mini [Emerging] cuts optimizer memory ~50%.
- **Schedule**: **WSD** [Proven] (warmup → flat peak → ~10% decay) beats/matches
  cosine and doesn't fix the token budget upfront (enables continued training +
  mixing high-quality data in the decay). **Hermes already uses WSD.**
- **Precision**: bf16 [Proven universal]; **FP8** [Emerging→frontier] (DeepSeek-V3,
  Kimi K2) ~2× throughput with <0.25% loss error via fine-grained per-tile scaling.
  Caveat: bf16 can break RoPE at long context (https://arxiv.org/pdf/2411.13476).
- **μP / Tensor Programs V** [Emerging] — tune LR on a small proxy, transfer across
  width. Weight-decay/depth don't transfer. Useful if we sweep HPs before scaling.
- **Data**: **intra-document masking + best-fit packing** improve ICL/knowledge and
  become crucial 4k→64k (https://arxiv.org/html/2402.13991). **Hermes already does
  doc-masking.** **Mid-training / quality annealing** — switch to a curated
  high-quality mix during the LR decay phase (OLMo 2, Phi-4). Long-context is a
  separate late phase (Llama 3.1: θ=500k, 8K→128K over 6 stages, ~800B tokens).

**Net for Hermes:** the architecture already lands on the proven defaults
(GQA, SwiGLU, QK-norm, RMSNorm, Muon+AdamW, WSD, doc-masking, hybrid attn+Mamba).
The highest-value _additions_ if we push quality: **z-loss** (cheap stability),
**mid-training quality-anneal** in the WSD decay, and — if scaling up — Mamba-2 or
gated-DeltaNet cores and a proper long-context phase with YaRN.

## Part B — Sharing retrieval embeddings between the retriever and the LLM

**The question:** can the embedding model and the generator LLM share
representations to cut compute or improve quality? **Short answer: share the
_backbone_, specialize the _head_ — don't naively reuse raw states.**

### What production does

Production RAG uses a **separate, dedicated embedding model** (E5/BGE/NV-Embed/
Voyage/OpenAI). Reasons: documents are embedded **once, offline** for the ANN index
(cheap bi-encoder economics); retrievers need **query/passage asymmetry + hard-
negative contrastive** geometry a generator LM lacks; and decoupling lets you re-
tune the retriever without touching the (often frozen) generator. Note the modern
twist: the _best embedders are themselves decoder LLMs converted to embedders_
(E5-Mistral, NV-Embed, GritLM, LLM2Vec) — so the field already reuses LLM
_backbones_, just as separately-tuned instances, not by reading raw hidden states.

### The three specific ideas, assessed

1. **Reuse the input embedding table / raw hidden states as the retrieval vector.**
   Embedding table: essentially doesn't work (per-token lookup, no sequence
   semantics). Raw causal hidden states: work **only with adaptation** — a
   projection head + contrastive/distillation (and ideally a bidirectional mode,
   LLM2Vec-style). _"One Model Is Enough" (2026)_ adds a ~25M projection head to a
   generating LLM and keeps **~97% of a dedicated retriever's quality at ~21.8×
   lower latency** — but single-dataset/[Frontier]. Not "for free."
   https://arxiv.org/abs/2404.05961 · https://arxiv.org/abs/2603.08429
2. **One model does retrieval + generation (GritLM/GRIT).** [Proven] — bidirectional
   attention + mean-pool for embedding, causal for generation, switched by
   instruction; jointly trained "at no performance loss." SOTA on MTEB _and_ strong
   generative; and because one model does both, cached doc/query representations
   give **>60% RAG speedup on long docs**. Cost: joint training, and doc-KV caching
   balloons storage (~30 TB vs 43 GB for a vector index). This is the strongest,
   most defensible form of sharing. https://arxiv.org/abs/2402.09906
3. **Feed doc embeddings into the LLM latent space (xRAG).** [Emerging] — a small
   trained MLP bridge injects a frozen retriever's doc embedding as **one token**;
   ~175× token reduction, ~3.5× FLOPs, near-lossless on gist QA (TriviaQA) but
   **−8–12% on multi-hop/NQ** where relational detail matters. Good for
   latency/context-budget-bound serving; validate recall-heavy tasks.
   https://arxiv.org/abs/2405.13792 (see also PISCO/COCOM at 5–16× soft compression,
   CLaRa's shared retriever-generator latent space — https://arxiv.org/abs/2501.16075)

### Verdict & recommendation for Hermes (100M hybrid Mamba retriever)

**Yes to sharing the backbone — as GRIT-style "share backbone, specialize head",
not raw-state reuse.** Concretely:

- Train Hermes with a **dual objective**: causal LM loss (agentic generation) + a
  **contrastive retrieval loss on a pooled projection head**, with query/passage
  instruction prefixes and hard negatives. One small model, one KV cache, no second
  embedder to serve — a big win at 100M where you're compute/memory bound.
- **Mamba caveat (load-bearing):** retrieval needs precise associative recall — the
  known weakness of SSM layers, and the exact reason every hybrid keeps full-
  attention layers. **Read the pooled retrieval embedding from (or after) a
  full-attention layer, not a pure-Mamba layer**, and keep enough global-attention
  layers (the field's 1:3–1:7 band) that the representation supports fine-grained
  matching. Our 2:1 mamba:attn is fine for this; don't go more Mamba-heavy without
  checking recall.
- Keep a **separately-servable index path that reuses the same weights** to embed
  the corpus offline (one model, but still a precomputable ANN index — avoids
  GRIT's KV-storage blowup).
- **Good when:** agentic/conversational retrieval (model already holds context),
  tight compute/memory, you can afford a contrastive stage + hard negatives.
  **Bad / keep a dedicated embedder when:** you need a static precomputable index
  over millions of docs coupled to a frequently-retrained generator (operationally
  painful — REPLUG keeps them separate and just tunes the retriever), or you need
  last-mile retrieval SOTA (a specialist + reranker still wins).
- Treat **raw-hidden-state reuse** and **xRAG latent injection** as phase-2
  experiments — real and promising [Frontier], but validate on recall-critical
  workloads before betting the architecture on them.

Key sources: GRIT https://arxiv.org/abs/2402.09906 · LLM2Vec https://arxiv.org/abs/2404.05961
· E5-Mistral https://arxiv.org/abs/2401.00368 · native hidden-states
https://arxiv.org/abs/2603.08429 · xRAG https://arxiv.org/abs/2405.13792 ·
RETRO https://arxiv.org/abs/2112.04426 · REPLUG https://arxiv.org/abs/2301.12652
