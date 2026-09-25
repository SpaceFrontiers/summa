# RL Training Pipeline for Agentic Search over Summa

> Design doc. Goal: train an LLM, with reinforcement learning, to drive Summa as an
> agentic retriever — searching, reading, navigating references, and refining queries
> until it can return a ranked list of the documents that answer a question. The
> approach is modeled on **SID-1** (SID AI, _"SID-1 Technical Report: Test-Time
> Compute for Retrieval"_, Dec 2025) and adapted to Summa' concrete API surface.

The current executable workflow and supported training task contracts are
documented in [training workflows](training-objectives-and-curricula.md) and
the [trainer guide](../summa-train/README.md). This document retains the
agentic-search research proposal and does not imply that every proposed tool
or rollout service is implemented.

---

## 1. What SID-1 actually does (the parts worth copying)

Distilled from the SID-1 technical report and the turbopuffer infrastructure writeup.

| Aspect                 | SID-1 choice                                                                                                                                                                  | Why it matters for us                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Base model**         | Qwen3-14B, RL from base **without SFT**                                                                                                                                       | A capable open instruct model + GRPO is enough; no expensive SFT bootstrap.                                                                                           |
| **Task framing**       | Model reports **documents ranked by relevance**, _not_ a generated answer                                                                                                     | Separates search from synthesis; gives dense partial-credit reward; prevents the model from "answering from memory" instead of retrieving. Composable as a sub-agent. |
| **RL algorithm**       | Modified **GRPO** (Magistral-style), 16 rollouts/question, 256 questions/step → 4096 trajectories/step                                                                        | Group-relative advantage needs no value network; partial-credit reward (NDCG) makes the 16-way comparison informative.                                                |
| **Reward**             | **NDCG** primary, plus recall, plus _timeliness/speed_, plus a format reward added later                                                                                      | Found-the-docs + ranked-them-right + did-it-fast. Speed reward is what makes parallel tool calls emerge.                                                              |
| **Agentic loop**       | Multi-turn: search → read excerpts → optionally `read` full doc → refine query → repeat → submit ranked list. As many steps as needed.                                        | Hierarchical retrieval (excerpts first, full doc on demand) controls context length.                                                                                  |
| **Emergent behaviors** | Prefers ANN over BM25 over time; learns **HyDE** late in training; issues **4–8 parallel searches/turn** (up to ~20 tool calls total)                                         | We don't hand-design these — the reward shapes them. But the tool interface must _allow_ them.                                                                        |
| **Synthetic data**     | Multi-hop questions built from **document-to-document similarity** (no hyperlinks needed); a seed doc must be in the targets; LLM-judge verification; explicit error taxonomy | Summa already computes doc-doc similarity (ANN over its own dense vectors) — we can generate multi-hop data on _any_ corpus.                                          |
| **Stability traps**    | (a) Tokens-In/Tokens-Out retokenization → collapse; (b) length-normalization debiasing → OOV-token blowup                                                                     | These are the two things that silently kill agentic-RL runs. Documented fixes below.                                                                                  |
| **Eval**               | 191 questions across general / finance / science / legal / email; report recall + NDCG + latency + cost; fuse k rollouts with **RRF**                                         | Public benchmarks (HotpotQA, SciFact) saturate — build a custom multi-hop eval.                                                                                       |

Key reported numbers (context, not targets): SID-1 4× = **0.84 recall / 0.73 NDCG / ~6 s / $0.0006 per question**; GPT-5.1 high = 0.78 recall / 144 s; embedding+rerank baseline = 0.45 recall.

Reference RL-search literature that informs the same choices: **Search-R1** (retrieved-token
masking + outcome reward, multi-turn), **R1-Searcher** (two-stage retrieve-reward then
answer-reward), **InfoFlow** (reward-density shaping for sparse-reward search), and the
_Survey of LLM-based Deep Search Agents_ (reward taxonomy: format, correctness, efficiency,
diversity, evidence-quality, retrieval-gain; rule-based vs ORM vs PRM).

---

## 2. Mapping SID-1 onto Summa

Summa exposes the retrieval primitives SID-1 assumes. The grounding (verified against the
codebase) and the gaps:

| SID-1 capability                 | Summa equivalent                                                                                                                                        | Source / note                                                                                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ANN / dense search               | `DenseVectorQuery` (`nprobe`, `rerank_factor`)                                                                                                          | `summa-core/src/query/vector/dense.rs`. **Client must pass the f32 vector** — server does _not_ embed text → we need an embedding service in the loop. |
| BM25 / lexical                   | `TermQuery`, `MatchQuery`, `BooleanQuery` (MUST/SHOULD/MUST_NOT, MaxScore/WAND)                                                                         | `query/term.rs`, `query/boolean.rs`. `MatchQuery` tokenizes server-side.                                                                               |
| Sparse / SPLADE                  | `SparseVectorQuery` — **server tokenizes `text`** and IDF-weights, or accepts precomputed `(indices, values)`; BMP + MaxScore pruning                   | `query/vector/sparse.rs`, `summa-server/src/converters.rs:156`. This one _can_ take raw text.                                                          |
| Metadata / numeric filter        | `RangeQuery` (u64/i64/f64) composed via `BooleanQuery` MUST                                                                                             | `query/range.rs`. Filter-style (score 1.0).                                                                                                            |
| Reranking (L2)                   | `Reranker` field on `SearchRequest` (dense or binary), RRF (`rrf_k`), Matryoshka prefilter                                                              | `summa.proto:157`.                                                                                                                                     |
| Fetch full document              | `GetDocument(DocAddress{segment_id, doc_id})`                                                                                                           | `SearchService.GetDocument`. This is our **`fetch` tool**.                                                                                             |
| Excerpts vs full text            | **Gap**: no snippet/highlight. Stored fields come back whole.                                                                                           | We truncate stored text at the _environment_ layer to make excerpts; `fetch` returns full.                                                             |
| Document references / navigation | **No native graph.** Convention: store reference IDs/URIs in a (multi-valued) stored field; resolve via `GetDocument` / a term lookup on an `id` field. | `FieldValueList` supports multi-value. We build a **`navigate` tool** on top.                                                                          |
| regex search                     | **Not supported.**                                                                                                                                      | Drop it from the toolset (SID-1 had it; non-essential).                                                                                                |
| Hybrid / parallel tools          | Multiple queries per turn = multiple gRPC `Search` calls; fuse with RRF                                                                                 | Parallelism is a serving concern (§6), not a query-type concern.                                                                                       |

**Two things Summa forces that SID-1's stack hid:**

1. **Dense queries need vectors, not text.** The policy model emits _text_; the environment
   must embed it (query text, and HyDE pseudo-docs) with the _same_ embedding model used to
   build the index. So the environment owns an embedding service. Sparse search can take raw
   text directly (server tokenizes), which is convenient for early training.
2. **No snippets.** The environment defines the excerpt policy (e.g. first N stored chars +
   the matched field), keeping `fetch` as the only way to see full text. This _is_ the
   hierarchical-retrieval mechanism that controls context growth.

---

## 3. Environment design

A Gym-style environment wrapping a Summa deployment + an embedding service. One **episode**
= one question against one corpus.

### 3.1 Episode lifecycle

```
reset(question, corpus_id) → observation_0 (system prompt + question + tool schema)
loop:
    action = policy(observation_t)          # model emits tool call(s) or <submit>
    obs_{t+1}, done = env.step(action)       # execute against Summa, append results
until done (submit OR step/token budget hit)
reward = score(submitted_ranking, gold_targets, trajectory_stats)
```

### 3.2 Action space (tools exposed to the model)

Tools are JSON function calls. Multiple calls per turn are allowed and encouraged (the
speed reward makes parallelism pay off — as in SID-1).

```jsonc
// 1. Lexical / structured search
search_bm25(query: string, fields?: [string], filters?: Filter[], k?: int=10)
// 2. Sparse search — Summa tokenizes & IDF-weights server-side
search_sparse(query: string, field: string, filters?: Filter[], k?: int=10)
// 3. Dense ANN — environment embeds `query` (and `hyde_doc` if given) with the index's model
search_dense(query: string, field: string, hyde_doc?: string,
             nprobe?: int, filters?: Filter[], k?: int=10)
// 4. Hybrid — run several of the above and RRF-fuse (server reranker or env-side RRF)
search_hybrid(query: string, modes: ["bm25"|"sparse"|"dense"]+, k?: int=10)
// 5. Read a full document (the SID-1 "read" tool)
fetch(doc_id: string)                       // → full stored fields
// 6. Follow references for navigation
navigate(doc_id: string, ref_field?: string) // → fetch docs referenced by doc_id
// 7. Terminal action
submit(doc_ids: [string])                    // ranked, best-first
```

`Filter` compiles to a `RangeQuery`/`TermQuery` ANDed into a `BooleanQuery` MUST clause
(e.g. `date >= ...`, `source == "email"`). Each `search_*` is one `SearchService.Search`
RPC; `search_hybrid`/`navigate` may fan out to several.

**Doc IDs:** Summa' `DocAddress{segment_id, doc_id}` is unstable across merges. The
environment maps each result to a **stable external id** (a `stored` `id`/`uri` field in the
schema) and translates back. Gold targets are expressed in those stable ids.

### 3.3 Observation space

Search results are rendered as **excerpts** (truncated stored text + score + stable id),
not full docs. `fetch`/`navigate` return full fields. The environment maintains a running
transcript; context-length pressure is real and intended — the policy must learn to
`fetch` selectively. Token budget per episode is a curriculum parameter (§5, length
scheduling).

### 3.4 Why document-centric (not answer-generation)

We copy SID-1's framing: the model's terminal output is a **ranked list of stable doc ids**,
scored against gold target docs. This gives:

- a smooth, dense reward (NDCG/recall over the ranking) instead of a 0/1 answer match;
- no reward hacking via parametric memory (you can't "know" a corpus-specific doc id);
- a drop-in sub-agent: the ranked docs feed any downstream reader/synthesis model.

---

## 4. Reward design

Per trajectory `i` in a GRPO group of 16, against gold target set `T` (with optional graded
relevance), produced ranking `R_i`:

```
r_i = w_ndcg · NDCG@K(R_i, T)
    + w_rec  · Recall@K(R_i, T)
    + w_fmt  · format_ok_i                  # valid tool calls, valid submit, schema-clean
    - w_time · cost_i                        # normalized latency/step/token cost
    - w_red  · redundancy_i                  # repeated identical queries / re-fetches
```

- **NDCG@K** is primary (SID-1's main signal): `DCG@K / IDCG@K`, gain `2^rel-1` over log
  position discount. With binary targets this still rewards _ordering_ the right docs first.
- **Recall@K** stabilizes early training when rankings are mostly wrong (broad credit for
  finding any target). Anneal `w_rec` down as NDCG takes over.
- **format_ok** turned out to need its own term in SID-1 (format regressed late). Keep it
  small but nonzero throughout; covers: parseable tool calls, exactly one `submit`, ids that
  exist, no duplicate ids.
- **cost / time**: normalized so that the _median_ trajectory gets ~0; faster-than-median is
  positive. This is the term that makes parallel tool use and early stopping emerge. Measure
  cost as wall-clock from the Summa `SearchTimings` plus a per-turn and per-token penalty —
  not just step count — so the model is paid for _issuing searches in parallel_ rather than
  serially.
- **redundancy** penalty discourages spamming the same query (a known GRPO failure mode).

Start `w = {ndcg: 1.0, rec: 0.5, fmt: 0.1, time: 0.1, red: 0.05}`; the time weight is the
main knob for the latency/recall trade-off (SID-1 ships multiple compute settings by
varying it).

**Reward source:** purely _rule-based / verifiable_ (NDCG over gold ids) — no neural reward
model in the loop, which is what keeps it cheap and un-hackable (Search-R1's lesson). The
only LLM-judge usage is **offline**, during data generation (§5).

---

## 5. Synthetic data pipeline (Summa-native)

SID-1's headline data trick maps directly onto Summa: build multi-hop questions from
**document-to-document similarity**, which Summa already gives us via ANN over its own dense
vectors. No hyperlinks/Wikipedia structure required → works on _their_ corpus.

### 5.1 Generation

```
1. Seed:    sample a seed doc d0 from the corpus.
2. Chain:   for each hop, search_dense(embed(d0)) over the dense field to get
            top-N similar docs; pick d1 (semantically linked but distinct).
            Repeat to build a chain d0 → d1 → … → dH (H = 1..3).
            (Summa ANN *is* the dynamic link graph.)
3. Question: prompt a strong LLM to write a question whose answer requires *all*
            docs in the chain, with d0 (the seed) guaranteed to be a target —
            SID-1 found a forced seed is required for question diversity.
4. Targets: T = {the chain docs} (+ any near-duplicates flagged below).
```

### 5.2 Verification & noise control (SID-1 error taxonomy)

Run an LLM judge over `(question, T)` to filter:

- **Type 1** — targets contain unnecessary docs (hurts precision → over-reporting). Most
  common; trim.
- **Type 2** — a relevant doc is missing from `T` (adds label noise → model penalized for
  good retrieval). Use Summa search to find likely-missing relevant docs and add them, or
  drop the question.
- **Type 3** — unanswerable despite non-empty `T`. Drop.

Public datasets (HotpotQA etc.) are noisy in exactly these ways and caused SID-1 models to
"over-report documents in hope of catching spurious targets" — so weight synthetic data
heavily and treat public QA as small, audited add-ins.

### 5.3 Difficulty curriculum

- **Single-hop**: trivial (model saturates) — use only for warmup/format learning.
- **Multi-hop (H=2,3)**: the main signal.
- Mix domains to match eval (general / finance / science / legal / email). For "email"-style
  filtering, generate questions that _require_ a `RangeQuery` date filter or a metadata
  `TermQuery` — this teaches the `filters` argument.

### 5.4 Multi-epoch

SID-1 trained 100 epochs on 100-question subsets (with obfuscated doc ids) with minimal
degradation — so a modest, _high-quality_ set reused many times beats a large noisy one.
Obfuscate stable ids per-epoch so the model can't memorize id→target.

---

## 6. Serving & infrastructure for rollouts

This is where SID-1/turbopuffer spent real effort, and the part most likely to bottleneck.

- **QPS bursts**: 256 questions × 16 rollouts × ~20 tool calls ≈ **80k searches/step**, and
  all groups fire their _first_ search in a ~10 s window → **1k+ QPS spikes**. Plan for the
  burst, not the average.
- **Summa serving**: run a **read-only replica pool** over a shared, immutable index
  snapshot (Summa' segment files are write-once; mmap + the caching directory layer make
  replicas cheap). Pin the corpus snapshot for the whole RL run so doc ids/gold stay valid.
  Scale replicas horizontally; the gRPC `SearchService` is stateless per request.
- **Embedding service**: a batched GPU embedder (query text + HyDE docs) co-located with the
  rollout workers. Cache embeddings of repeated query strings within a step.
- **Async rollouts**: decouple generation from search. Each rollout worker issues `search_*`
  RPCs asynchronously so the 4–8 parallel calls/turn actually hit Summa concurrently (this
  is what the _time_ reward is supposed to reward — don't serialize them in the harness).
- **Determinism for reward**: fix `nprobe`, segment layout, and snapshot so NDCG is
  reproducible across the 16 group members (variance should come from the _policy_, not from
  ANN nondeterminism).
- **Throughput sanity**: cache `GetDocument` results per episode; dedupe identical in-flight
  searches across rollouts of the same question.

---

## 7. RL algorithm & stability

**Algorithm:** GRPO, group size G=16, ~256 questions/step (tune to hardware). Group-relative
advantage `A_i = r_i − mean(r)`; optionally `/ std(r)`. No critic, no SFT (SID-1 went
straight from base).

**The two traps SID-1 documents — both have caused silent collapse in agentic RL:**

1. **Tokens-In / Tokens-Out (TI/TO) retokenization.** Converting messages↔tokens across
   turns is _lossy_: re-tokenizing the assembled transcript produces token sequences the
   model would deem "extremely unlikely," and training on them drives instability →
   reward rises then **catastrophically collapses** (tool-calling accuracy craters).
   **Fix:** keep token ids exactly as generated through the whole multi-turn rollout; never
   re-tokenize a reconstructed message list. Append tool-result tokens to the _same_ id
   stream. This alone removed the need for importance-sampling corrections in their setup.

2. **Length-normalization debiasing.** Following Dr.GRPO's per-sequence length debiasing
   makes the model emit out-of-vocab tokens: with a negative correlation between rollout
   length and advantage, length-debiased GRPO yields negative per-token advantages that
   suppress logits globally. **Fix:** keep length-biased normalization — per-token advantage
   `A_i / (L_i · G)` — and instead control length with:
   - **length scheduling**: start with a short max-rollout budget, grow it over training;
   - a **soft length penalty** folded into reward (the `cost_i` term) rather than via the
     normalizer.

**Masking:** mask tool-result / retrieved tokens out of the policy loss (Search-R1's
retrieved-token masking) — the model is graded on _its_ tokens (queries, reasoning, submit),
not on text Summa handed back.

**Warm-up:** first run with sparse search only (raw text → server tokenizes, no embedder
needed) and short budgets to get format + basic search working; then enable dense/HyDE and
grow the length budget.

---

## 8. Evaluation

- **Held-out benchmark**: build ~150–200 multi-hop questions across the 5 domains via §5,
  _audited by hand_, never seen in training. Report **Recall@K, NDCG@K, latency (p50/p95),
  cost/question** — the SID-1 table format.
- **Test-time compute**: evaluate `1×` (single rollout) and `k×` (k rollouts **RRF-fused**,
  `rrf_k≈60`). The k× curve is the "test-time compute for retrieval" story.
- **Baselines** to beat, all on the _same Summa index_: (a) dense-only top-K; (b)
  dense+rerank; (c) BM25; (d) a frontier LLM (GPT-5.x / Sonnet / Gemini) given the same tool
  schema via the API. SID-1's bar: nearly 2× the embedding+rerank recall.
- **Retire saturated tasks**: if HotpotQA/SciFact-style sets hit ~perfect NDCG, drop them
  from the headline (as SID-1 recommends) and lean on the custom multi-hop set.
- **Ablations**: reward terms (drop time / drop recall), tool subsets (no dense, no fetch,
  no navigate), data (synthetic-only vs +public), and the two stability fixes (TI/TO,
  length norm) — these double as regression tests.

---

## 9. Phased implementation plan

| Phase                      | Deliverable                                                                                                                                           | Depends on |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **0. Snapshot & replicas** | Pin a corpus snapshot; stand up a read-only Summa replica pool + batched embedding service; load test to 1k QPS.                                      | §6         |
| **1. Environment**         | `SummaSearchEnv` (reset/step), the 7 tools → Summa RPCs, stable-id mapping, excerpt truncation, transcript builder. Unit-tested against a tiny index. | §3         |
| **2. Reward**              | NDCG/recall/format/time/redundancy scorer over stable ids; deterministic, no neural RM. Golden tests.                                                 | §4         |
| **3. Data**                | Synthetic multi-hop generator (Summa-ANN doc-doc chains) + LLM-judge verifier + curriculum buckets; obfuscated ids. Produce v0 train + held-out eval. | §5         |
| **4. RL loop**             | GRPO trainer with exact TI/TO token handling, length-biased advantage, retrieved-token masking, length scheduling. Warm-up = sparse-only.             | §7         |
| **5. Scale & eval**        | Full multi-domain training; eval harness (1×/k×/RRF) + baselines table; ablations.                                                                    | 1–4        |
| **6. Serve**               | Export policy; inference path = same env tools against production Summa; expose as a retrieval sub-agent.                                             | 5          |

### Suggested repo layout

```
summa-rl/                      # new sibling crate/package
  env/         summa_search_env.py   # tools → summa-client-python RPCs
  reward/      ndcg.py, scorer.py
  data/        gen_multihop.py, verify_judge.py, curriculum.py
  train/       grpo.py (TI/TO-safe), length_schedule.py, masking.py
  eval/        benchmark.py, baselines.py, rrf.py
  serve/       replica_pool/, embedder/
```

Reuses `summa-client-python` for all retrieval; no changes required to the Summa search
algorithm itself (SID-1 explicitly works with existing search tools — and so does this).

---

## 10. Open parameters (decide before Phase 4)

- **Base model**: Qwen3-14B (SID-1's choice) vs smaller (3B/7B) for cheaper iteration.
- **Embedding model**: must match whatever built the dense field in your Summa index — the
  environment embeds query text with _that_ model, or HyDE/ANN silently degrades.
- **Group size / batch**: 16 / 256 is SID-1's; scale to GPU budget (group size ≥8 keeps the
  group-relative signal meaningful).
- **K in NDCG@K / Recall@K**: set to the realistic downstream consumption (e.g. 10–20).
- **Time-reward weight**: the single knob that trades recall for latency; ship a small grid
  as "compute settings."

---

## 11. Literature-grounded refinements

A broader sweep of the agentic-RL / RL-search literature (full map in §12) surfaced several
findings that sharpen the design above. The ones that change a decision:

### 11.1 External validation of the core thesis

- **RL for Long-Horizon Multi-Turn Search Agents** (Kalyan & Andrews, 2510.24126) is
  effectively an independent reproduction of SID-1's claim: a **14B RL-trained model beats
  frontier API models on legal document search (85% vs 78%)**, and **longer multi-turn
  horizons help** — both at train and test time. This is the strongest outside evidence that
  the Qwen3-14B + multi-turn + verifiable-reward recipe is sound, and it directly motivates
  our **turn-budget curriculum** (train short, lengthen later; allow long horizons at eval).

### 11.2 Reward: gate, don't just sum (the multi-objective collapse trap)

Our §4 reward linearly sums NDCG + recall + format − time − redundancy. The multi-objective
RL literature warns this is fragile:

- Naive **sum-then-normalize** loses signal resolution — each criterion's contribution gets
  washed out by group statistics, and the model **reward-hacks the easy term** (e.g. emits an
  empty/short `submit` to bank the speed/format bonus). The **GDPO** line and Magistral-style
  recipes fix this two ways we should adopt:
  1. **Conditioned rewards**: award secondary terms _only if the primary is satisfied_ — e.g.
     grant the speed bonus and format bonus **only when recall > 0** (the model found at least
     one target). This kills the "fast empty answer" exploit at the root.
  2. **Batch-normalize the summed advantage** across the whole step, not just within the
     16-group, to keep variance bounded as we add reward terms.
- **PURE / min-form credit assignment** (2504.15275): summation-form credit lets a model
  "compensate" a bad step with a good one; a **min-form** (value = weakest step) suppresses
  this. Worth considering if we ever add per-turn rewards.

### 11.3 A denser turn-level signal: information gain

Outcome-only rewards cause **advantage collapse** in long rollouts (all 16 group members get
the same score → zero gradient). Two cheap, intrinsic fixes:

- **IGPO** (Information Gain-based Policy Optimization, 2510.14967): define a per-turn reward
  as the **marginal increase in the policy's probability of the correct answer** after that
  turn — dense, model-intrinsic, no external RM. Our document-centric analog: **marginal
  NDCG/recall gain of the best achievable ranking after each turn's new evidence**. This gives
  per-turn credit without a process reward model and is a strong upgrade to §4 if outcome-only
  training stalls.
- **Tree-GRPO** (2509.21240): sample rollouts as a **tree with shared prefixes** → more
  rollouts per fixed token/tool-call budget, and free step-wise process supervision from the
  outcome reward alone. Given our ~80k-searches/step cost (§6), prefix-sharing is a direct
  throughput win.

### 11.4 The "do-nothing" local optimum (query-rewriting trap)

**SAGE** (2506.19783) reports that with a strong retriever, the agent's best safe move is to
**not reformulate** — a deceptive high-reward local optimum that stalls exploration. Summa'
dense retriever is strong, so expect this. Mitigations we should bake in: an explicit
exploration incentive early, the **identical-query penalty** already in our `redundancy` term,
and reward shaping that pays for _information gain_ (11.3) rather than mere query emission.

### 11.5 Extra stability levers for long multi-turn RL

Beyond SID-1's TI/TO + length-norm fixes (§7), the GRPO-mechanics literature adds three that
are cheap insurance for long, multi-turn rollouts:

- **FP32 logits on the LM head**: generator vs trainer kernels differ numerically; in
  importance-sampling regimes this destabilizes — FP32 on the final layer nearly removes it.
- **Sequence-level importance sampling (GSPO) / CISPO-style clipping**: more stable than
  token-level GRPO clipping on long responses; CISPO preserves gradient on rare-but-important
  "fork" tokens that GRPO clipping suppresses.
- **Low-probability-token domination** (TR-GRPO, 2511.00066): low-prob tokens carry outsized
  gradients and destabilize; down-weighting them is the same failure family as SID-1's
  OOV-token blowup — another reason to keep the length-biased normalizer and watch token entropy.

### 11.6 Harden the environment against reward hacking

The **Reward Hacking Benchmark** (2605.02964) finds RL post-training _raises_ exploit rates
(0.6%→13.9% in a controlled pair) and that **simple environment hardening cuts exploits ~88%**
without hurting task success. Concretely for us: never leak gold doc-ids into excerpt metadata
or filters; keep the per-epoch **id obfuscation** SID-1 uses; verify `submit` ids exist in the
_corpus_, not a cached candidate list; and log per-term reward to catch a term being gamed
(RewardScope-style monitoring). Note the open question on **SFT contamination** (Countdown-Code,
2603.07084): SFT can inject hacking priors that resurface under RL — a point in favor of SID-1's
**no-SFT** choice, though warm-up behavior cloning (WebAgent-R1) trades this off against faster
format acquisition.

### 11.7 Infra blueprint

**AgentRL** (2510.04206) is the closest published infra match to what §6 describes:
fully-**asynchronous generation↔training**, a **unified function-call API** across
environments, **cross-policy sampling** for exploration, and **task-advantage normalization**
for multi-task stability — all worth borrowing if we later train across multiple corpora/domains
at once. **WebAgent-R1** and **UserRL** corroborate the async-rollout + warm-up pattern (UserRL
finds an SFT cold-start helps multi-turn; SID-1 deliberately skips it — resolve empirically).

---

## 12. Reading map (RL & agentic flows)

Grouped by relevance to this pipeline. All arXiv unless noted.

**Closest analogs — RL search agents**

- RL for Long-Horizon Multi-Turn Search Agents — 2510.24126 (14B > frontier on legal search)
- Search-R1 — 2503.09516 (retrieved-token masking, outcome reward, multi-turn)
- R1-Searcher — 2503.05592 (two-stage retrieve-then-answer reward)
- IGPO: Information Gain-based Policy Optimization — 2510.14967 (dense turn-level reward)
- InfoFlow — 2510.26575 (reward-density shaping for sparse search reward)
- Agentic Conversational Search w/ RL — 2601.13115 ; HARIS — 2506.07528 (multi-hop verification)
- SAGE — 2506.19783 (query-rewriting RL; do-nothing trap; NDCG@10 reward; identical-query penalty)
- Beyond Outcome Reward: decoupling search & answering — 2510.04695

**Surveys (orientation)**

- Survey of LLM-based Deep Search Agents — 2508.05668 (reward taxonomy)
- Comprehensive Survey on RL-based Agentic Search — 2510.16724
- The Landscape of Agentic RL for LLMs — 2509.02547
- Agentic Tool Use in LLMs — 2604.00835 ; A Brief Overview: Agentic RL in LLMs — 2604.27859

**GRPO mechanics & stability**

- Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning — 2508.08221
- λ-GRPO — 2510.06870 ; TR-GRPO — 2511.00066 (length bias / low-prob-token domination)
- LSPO — 2510.01459 ; Temporal Scheduling for RLVR — 2605.25381 (length/credit scheduling)
- Demystifying Long CoT — 2502.03373 (length reward hacking, repetition penalty)
- DAPO (2503.14476), CISPO (2506.13585), GSPO — clipping/IS variants; FP32-logits recipe (Magistral/Minimax)

**Credit assignment & process rewards**

- From Reasoning to Agentic: Credit Assignment in RL for LLMs — 2604.09459 (taxonomy)
- PURE: min-form credit assignment — 2504.15275
- BEACON: milestone-guided long-horizon — 2605.06078 ; Tree-GRPO — 2509.21240
- AgentPRM — 2502.10325 ; RLTR (tool-use reward) — 2508.19598 ; CAPO — 2508.02298

**Reward hacking (failure modes to guard)**

- Reward Hacking Benchmark (tool-use agents) — 2605.02964
- Reward Misspecification / phase transitions — 2201.03544
- Countdown-Code (SFT contamination → hacking) — 2603.07084
- Reward Modeling for RL-based LLM Reasoning (survey) — 2602.09305

**Infra & multi-task agentic RL**

- AgentRL — 2510.04206 (async gen↔train, function-call API, cross-policy sampling)
- WebAgent-R1 — 2505.16421 ; UserRL — 2509.19736 ; ARTIST — 2505.01441

**Synthetic environments/tasks**

- AgentGen — 2408.00764 (env + task generation, bidirectional difficulty)
- Synthetic Data RL — 2505.17063

---

### Sources

- SID-1 Technical Report — https://www.sid.ai/research/sid-1-technical-report ; intro: https://www.sid.ai/research/sid-1
- turbopuffer, _Training SID-1 to beat GPT-5 at search with 1k+ QPS RL_ — https://turbopuffer.com/blog/reinforcement-learning-sid-ai
- Search-R1 (Jin et al., 2025) — https://arxiv.org/abs/2503.09516
- R1-Searcher — https://arxiv.org/pdf/2503.05592
- InfoFlow (reward-density optimization) — https://arxiv.org/html/2510.26575
- _A Survey of LLM-based Deep Search Agents_ — https://doi.org/10.48550/arxiv.2508.05668
- _Comprehensive Survey on RL-based Agentic Search_ — https://arxiv.org/pdf/2510.16724
