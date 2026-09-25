# Large-candidate-pool retrieval evaluation

Implemented by [`summa-train/src/retrieval_pool.rs`](../summa-train/src/retrieval_pool.rs).
The run-specific examples below describe the motivating experiment.

## Why

`summa-train eval --objective contrastive_retrieval` scores retrieval **in
batch**: the candidate set for a query is whatever the other rows of the same
batch contributed. At the geometry the 300M MoE run used — `--sequence-length
1024 --batch-size 4`, one positive and two mined negatives per record — that is
**12 candidates per query**.

The metric saturated. Every checkpoint from step 457.5k onward reports
`top1_accuracy 1.0, mrr 1.0, recall_at_k 1.0` on all three curriculum stages,
before and after two rounds of SFT. A metric pinned at its ceiling cannot
detect a regression, and retrieval is the capability this model exists to
provide. Ranking one document above eleven in-batch peers says nothing about
ranking it above thousands of plausible alternatives, which is the only
question a search engine asks.

This document specifies a second, harder measurement. It does not replace the
in-batch number — that one stays comparable with training-time metrics, which
is its purpose — it answers a different question.

## What is measured

Given retrieval shards, build one **global candidate pool** from every unique
document they mention, keyed by `document_id` (positives and mined negatives
alike; negatives in the current eval shards are other queries' positives, so
each shard is already a closed ranking set). Embed the whole pool once, embed
every query, and rank each query's own positive against the **entire pool**.

Historical pool sizes from the run-local `.context/heldout-eval/` artifacts
(not checked in; supply your own held-out shards to reproduce an evaluation):

| shard                                    | queries | unique documents |
| ---------------------------------------- | ------- | ---------------- |
| `02-school-fundamentals-eval-retrieval`  | 156     | 156              |
| `03-university-core-eval-retrieval`      | 641     | 641              |
| `04-advanced-scholarship-eval-retrieval` | 2652    | 2652             |

2652 candidates is 221× the current advanced-stage candidate set. Additional
shards may be passed as pure distractors to grow the pool further without
adding queries.

Reported per run:

- `top1_accuracy`, `mrr`, `recall@k` — same definitions as the in-batch eval,
  so the two are directly comparable at differing pool sizes.
- `ndcg_at_10` — with exactly one relevant document per query this is
  `1/log2(rank+1)` when `rank <= 10` and `0` otherwise.
- `ranks.mean`, `median`, `p90`, `p99`, `worst`, and `beyond_100` — the tail
  shape that aggregate ratios and a maximum alone can hide. `beyond_100`
  counts positives outside a plausible first-stage re-ranking window.
- `pool.documents`, `pool.queries` — always emitted. If the pool is ever
  subsampled the report must say by how much; a silently truncated pool would
  read as a harder benchmark than it was.

## Correctness constraints

**Prefixes and truncation come from the task adapter, never from this
command.** Retrieval text is framed by `TaskConfig::RetrievalRepresentation`'s
`query_prefix` / `document_prefix` (defaults `"Represent this query for
retrieval:\n"` and `"Represent this document for retrieval:\n"`) and encoded by
`data::structured::encode_retrieval_text`, which owns EOS placement,
`end_position`, and truncation of the document body. A prompt assembled by hand
here would be out of distribution and would report a fake failure — this has
already happened once on the `qa_reasoning` path.

**A document id names one model-visible text.** Repeated `document_id`s collapse
to one pool slot only when their adapter-framed, truncated tokens are equal. A
collision carrying different visible text fails the run; keeping the first
silently could score a later query against the wrong positive.

**The read-out layer must match the training phase.** `--retrieval-layer` is
validated against the model by `validate_retrieval_layer_for_model`, exactly as
in `eval`. The 300M run trained retrieval at layer 24 with temperature 0.05;
reading any other layer measures an untrained projection.

**Ranking must not be optimistic on ties.** Scores use the same deterministic
tie-break as first-index argmax: a candidate tied with the positive ranks ahead
of it when its pool index is lower. Embeddings are L2-normalized by
`forward_embeddings`, so the dot product is cosine similarity and temperature
does not affect ordering.

**Padding is asserted, not assumed, to be inert.** `encode_retrieval_text` pads
to `--sequence-length` with EOS and records the true `end_position`. Reading the
embedding at `end_position` under a causal model (both the attention and SSM
blocks are causal) cannot depend on later positions, so shorter padding would
be mathematically identical. v1 pads everything to `--sequence-length` anyway,
matching training geometry exactly; a length-bucketing optimization may follow
only with a test pinning bitwise-equal embeddings.

## Shape

New subcommand rather than a flag on `eval`, because the flow is two-pass
(embed pool, then rank queries) where `eval` is single-pass streaming, and
because it emits a different report schema:

```
summa-train retrieval-pool-eval \
  --config <config.json> --tokenizer <tokenizer.json> --checkpoint <weights.safetensors> \
  --data <shard.jsonl.zst>... [--distractors <shard.jsonl.zst>...] \
  --sequence-length 1024 --batch-size 16 \
  --retrieval-layer 24 --recall-k 10 \
  -o report.json
```

It inherits `eval`'s safety properties: forward-only, never `.autodiff()`, no
token cache writes, and `RESERVED_OUTPUT_NAMES` refused for `--output` so a
mistyped path cannot overwrite a live run's state.

## Tests

- Pool larger than one batch ranks correctly across the batch boundary — the
  regression that a per-batch implementation would pass and this must not.
- Metrics are finite, deterministic across two runs, and ordered
  `top1 <= mrr <= 1` and `top1 <= recall@k <= 1`.
- Rank-distribution percentiles distinguish one catastrophic outlier from a
  systematically heavy tail even when both inputs have the same worst rank.
- Prefixes in the emitted `task` block equal the adapter defaults, pinning that
  they were not reconstructed locally.
- A `--retrieval-layer` outside the model is rejected.
- Duplicate `document_id`s across shards collapse to one pool entry, and a
  query whose positive appears twice still resolves to its own document;
  duplicate ids with different model-visible text are rejected.
- Reported `pool.documents` equals the distinct ids actually embedded.
