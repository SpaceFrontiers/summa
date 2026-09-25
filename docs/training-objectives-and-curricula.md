# Training workflows and task contracts

Summa training consumes a strict `WorkflowV2` document. A workflow is an
ordered sequence of named phases; every task-bearing phase selects one task
adapter and declares its data and execution geometry. Relative paths resolve
against the workflow file.

Only version 2 workflows are accepted; phase intent is always explicit and
unknown versions fail before execution.

The checked-in [`workflow.example.json`](../summa-train/workflow.example.json)
is directly executable by the built-in trainer. The
[`workflow.education.example.json`](../summa-train/workflow.education.example.json)
and [`workflow.sleep.example.json`](../summa-train/workflow.sleep.example.json)
examples combine phase kinds that require `NativeWorkflowHost`; neither whole
file is accepted by the wake-only `train` command. To run their wake work with
the stock trainer, extract a separate wake-only WorkflowV2 file and put the
same `periodic_sleep` object on every phase trained with a memory-enabled MAL.
Bind that projection to a content-pinned periodic runtime. The final promotion
phase remains trainer-owned and is never sent to an external worker.

## Phases

| Phase                | Purpose                                    | Required task signal  |
| -------------------- | ------------------------------------------ | --------------------- |
| `pretrain`           | Initial model optimization                 | Any registered task   |
| `continued_pretrain` | Domain/capability continuation             | Any registered task   |
| `sft`                | Supervised fine-tuning                     | Supervised generation |
| `preference`         | Pairwise preference optimization           | Pairwise preference   |
| `rl`                 | Reward-driven post-training                | Verifiable reward     |
| `distillation`       | Teacher/student optimization               | Any registered task   |
| `sleep`              | In-model memory consolidation and dreaming | No external task data |
| `quantization`       | QAT or quantization distillation           | Any registered task   |
| `evaluation`         | Read-only acceptance measurement           | Any registered task   |
| `promotion`          | Candidate acceptance and publication       | No task data          |

Optimization phases require `task`, `data`, `sequence_length`, `batch_size`,
and `gradient_accumulation`. `epochs` defaults to one, `shuffle_buffer` to
8192, and the loss and learning-rate weights to one. `steps`, when present,
caps the phase. Evaluation requires task/data plus sequence and batch size but
rejects optimizer-only fields. Sleep and promotion reject task data and
optimizer geometry. Executor-specific settings are contained in `parameters`
instead of being interpreted by task adapters; promotion instead requires its
strict typed `promotion` evidence contract.

```json
{
  "version": 2,
  "name": "retrieval-model",
  "phases": [
    {
      "name": "foundation",
      "type": "pretrain",
      "task": { "type": "causal_lm" },
      "data": "corpus/foundation.jsonl.zst",
      "sequence_length": 2048,
      "batch_size": 16,
      "gradient_accumulation": 8,
      "steps": 10000
    },
    {
      "name": "representations",
      "type": "continued_pretrain",
      "task": {
        "type": "retrieval_representation",
        "temperature": 0.05,
        "layer": 24
      },
      "data": "corpus/retrieval.jsonl.zst",
      "sequence_length": 512,
      "batch_size": 16,
      "gradient_accumulation": 8,
      "steps": 2000
    }
  ]
}
```

## Execution surfaces

`summa-train train` owns the in-process streaming implementation for causal
LM, summarization, retrieval representation/planning, instruction tuning,
QA/reasoning, and quantization phases over those objectives. It also executes
periodic sleep for a memory-enabled MAL when every projected phase carries one
identical sleep configuration and the operator supplies `--sleep-runtime` plus
its exact `sha256:` identity. `--print-run-signature` computes the periodic
runtime's `workflow_signature` from the model, tokenizer, data identities,
workflow, seed, and optimizer settings and exits before creating any state. It
rejects unsupported task/phase combinations.

The strict periodic runtime shape is shown in
[`sleep-runtime.periodic.example.json`](../summa-train/sleep-runtime.periodic.example.json).
It intentionally has no wake journal or initial tier/parameter-ID state: the
integrated trainer seals those artifacts at each boundary.

`summa-train run-workflow` is the algorithm-neutral lifecycle runner. It
launches the configured worker executable and sends one protocol-v2 JSON
request for each non-sleep, non-promotion phase. Promotion
uses the built-in acceptance executor; an external worker cannot authorize
release. With `--sleep-runtime PATH` and
`--sleep-runtime-sha256 sha256:...`, the stock CLI registers the first-party
factory for standalone `sleep`. It rejects every `periodic_sleep` setting
because this lifecycle surface does not own the worker's optimizer boundary.
`validate-workflow --signature-only` prints the resolved signature required by
the standalone runtime without creating lifecycle state.

The standalone shape is shown in
[`sleep-runtime.standalone.example.json`](../summa-train/sleep-runtime.standalone.example.json).
Its outer and Dreaming wake-journal pins must be identical, and its initial
tier-optimizer and parameter-ID artifacts must belong to the phase input
checkpoint. All-zero hashes in both examples are placeholders and must be
replaced before the finalized runtime JSON itself is hashed.

Progress cursors, typed metric events, yield, and completion are the only
accepted worker responses. New and resumed runs use:

```bash
summa-train run-workflow \
  --workflow summa-train/workflow.example.json \
  --executor /opt/summa/bin/phase-worker \
  --state /data/run/workflow-runtime.json \
  --metrics /data/run/metrics.jsonl \
  --run-id workflow-seed-1 \
  --initial-checkpoint-uri checkpoint://initial/generation-manifest.json \
  --initial-checkpoint-sha256 "sha256:$INITIAL_CHECKPOINT_SHA256"

summa-train run-workflow \
  --workflow summa-train/workflow.example.json \
  --executor /opt/summa/bin/phase-worker \
  --state /data/run/workflow-runtime.json \
  --metrics /data/run/metrics.jsonl \
  --run-id workflow-seed-1 \
  --resume
```

The initial checkpoint options are a pair and are forbidden on resume. Metrics
and run id are also a pair. A yielded worker commits its cursor and exits
successfully; its service supervisor can invoke the resume form. The runtime
will not silently change worker identity, metric history, or workflow
configuration.

`NativeWorkflowHost` is the embedded execution surface for the complete
education workflow and other sleep-enabled recipes. It combines the same
atomic runtime checkpoint and metric journal with explicit registered adapter
identities. Standalone sleep is routed only to a
`NativeSleepPhaseContextFactory`; DPO/forward-KL/GRPO use the native resumable
post-training executor when a `NativePostTrainingContextFactory` lends its
model/provider/publisher adapters and, for periodic sleep, its idempotent
`PostTrainingBoundaryHook`; other periodic optimizer phases are routed only to
a registered in-process `NativePeriodicWakeExecutor`, which owns the complete
phase.
The supplied context also authenticates cumulative optimizer-step and exact
trainable-policy model-token clocks against the phase input. The first-party
`NativePostTrainingBoundaryController` wraps every inner native-sleep cursor in
the post-training resume envelope and drains all crossed tier boundaries in
clock order before the next update may commit. It persists the optimizer receipt
before sleep begins. The registered controller must match the lent controller
and clock authority; each output checkpoint must authenticate both cumulative
clocks for the next phase factory.
Promotion remains built in, and only ordinary non-periodic phases may use the
external worker. The host validates the complete dispatch plan before
it creates or loads state and binds workflow, worker, and factory identities
into the resume digest. It deliberately supplies no site-specific loader,
judge, evaluator, or storage implementation.

The typed promotion object contains the run against the strongest fixed
ablation, exactly ten other runs completing the eleven-ablation catalog,
resource evidence, a required acceptance policy, and a decision directory.
Every input is given as a `{"path": "..."}` reference that resolves relative to
the workflow file. The gate resolves the model transport referenced by every
target, binds the accepted report to the exact current checkpoint, and publishes
a deterministic receipt. A retry can reuse only the same bytes. Rejected
decisions remain auditable but cannot advance the release phase.

The acceptance suite, benchmark suite manifest, acceptance policy, and resource
comparison use strict v2 schemas. Resource comparison v2 is produced by the
first-party resource host: a timeout-bounded worker returns raw observations and
relative artifact references, while the host derives the strongest matched
baseline and verifies the exact-resume artifacts. The policy owns the
stable-anchor catalog set, resource-evaluator identity, sample minima, and all
parity/performance limits. The resource comparison contains raw paired wake
trials, contiguous per-cycle capacity observations, and raw kernel
reference/candidate values. Exact-resume paths stay relative in the artifact and
are resolved only on a clone for I/O verification. Promotion recomputes
throughput, p95 latency, capacity extrema, and parity errors; evidence cannot
submit aggregates or choose its own tolerances. Stored and routed-active
capacity must remain identical to cycle zero across every observed sleep cycle.

## Held-out evaluation

`summa-train eval` scores an existing checkpoint on held-out shards with a
forward-only pass. It reuses the trainer's objective forward pass, batching, and
JSONL/`.jsonl.zst` streaming, so its numbers are directly comparable with
training-time metrics. It constructs no optimizer, records no autodiff tape,
writes no token cache, and touches no `TrainingState` or metric journal, so a
live run stays exactly resumable while it is evaluated. The evaluation device is
never `.autodiff()` — dropout is therefore the identity — and the model is never
`prepare_inference()`d, because mixed-precision decode weights would change the
reported loss.

```bash
summa-train eval \
  --config model.mal --tokenizer tokenizer.json \
  --checkpoint out/generations/sha256-.../weights.safetensors \
  --data holdout-a.jsonl.zst --data holdout-b.jsonl.zst \
  --objective causal_lm --sequence-length 1024 --batch-size 8 \
  --max-batches 200 --output eval-causal.json
```

`--objective causal_lm` reports the token-weighted mean cross-entropy over every
supervised token plus its perplexity; per-batch means are never averaged
unweighted, which would bias the result toward short batches.
`--objective contrastive_retrieval` reports in-batch top-1 accuracy from the same
similarity matrix the training loss uses, plus MRR and recall@`--recall-k`, with
per-batch candidate counts. Use `--retrieval-layer` and `--temperature` to match
the retrieval training phase; query and document prefixes come from the built-in
task defaults. In-batch negatives make retrieval accuracy depend on the
candidate-pool size, so a trailing batch that cannot be filled is dropped and
counted in `dropped_samples`; token-weighted language losses score it exactly.

For measurements that the streaming evaluator cannot make, use
[`retrieval-pool-eval`](retrieval-pool-eval.md) to rank against one global
candidate pool and [`generate-eval`](generation-eval.md) to score free-running
decodes. The former exposes retrieval regressions after the in-batch metric
saturates; the latter catches generation failures that teacher-forced loss can
hide.

### Supervised-generation objectives

`--objective summarization`, `instruction_tuning`, `qa_reasoning`, and
`retrieval_planning` score the four supervised-generation tasks on their own
JSONL contracts (`document`/`summary`, `instruction`/`response` with optional
`system` and `input`, `question`/`answer` with optional `reasoning`,
`request`/`plan` with optional `context`). Each reports
`supervised_generation.loss` and `.perplexity`: the token-weighted mean
cross-entropy over **target positions only**. Prompt and EOS-padding positions
never contribute, because the command reuses the trainer's masked-loss forward
pass and the same task adapter framing, so the reported number is the training
loss for that phase minus gradients — comparable straight across a train/eval
boundary. `supervised_tokens` is therefore much smaller than `compute_tokens`,
and `truncated_tokens` counts source text dropped to reserve room for the target.

Prompt framing must match the training phase exactly, so the report echoes the
complete task in its `task` block. `--instruction` overrides the built-in
instruction for phases that set their own, and `--require-reasoning` mirrors
`qa_reasoning`'s `require_reasoning`, which demands a `reasoning` field and
supervises the `Reasoning:`/`Answer:` target framing. Retrieval-only flags are
rejected on these objectives and vice versa. Because targets are never
truncated and padding cannot reach earlier positions through causal attention,
the reported loss is unchanged by `--batch-size` and by any
`--sequence-length` long enough to hold the record; use the training phase's
sequence length so truncation behaviour matches too.

`--sequence-length` is where evaluation deliberately differs from training. A
supervised record whose prompt framing, complete target, and EOS do not fit
makes `train` abort — a run must not optimize a silently reduced dataset — but
one over-long held-out record must not void a whole evaluation. `eval` skips it,
counts it in `oversized_records` (per shard and in total), and warns on stderr
and in `warnings`; it is excluded from every reported number.

`--shuffle-buffer` (default `0`, source order) is the only setting `--seed`
affects, and the JSON report deliberately excludes timing, so identical inputs
produce a byte-identical report. Config, tokenizer, checkpoint, and every shard
are recorded with their digests: the shard identity is the same authenticated
value the trainer's run signature consumes. A checkpoint/model mismatch, an
out-of-range retrieval layer, a sequence length past `max_seq_len`, and data too
small for one batch all fail loudly. Adjusted conditions — a config vocabulary
overridden by the tokenizer, a `--seed` that cannot matter, dropped trailing
samples — are written to stderr and to the report's `warnings` array, because
the CLI's default log filter would otherwise hide them. `--output` refuses
trainer artifact names such as `current.json` or `weights.safetensors`, so a
mistyped path cannot overwrite a live run.

## Configured corpus discovery

The corpus pipeline accepts replaceable search, record-materialization,
tokenization, and deduplication adapters. The production recipe uses Search API
for discovery and `PostgresRecordMaterializer` for canonical bodies. It is
configured in
[`corpus.production.example.json`](../summa-train/corpus.production.example.json),
not in task or trainer code.

- `request_template` contains the complete provider request shape.
- `request_mapping` maps query, offset, limit, arbitrary query parameters, and
  switches that must be false. Both reranker switches in the example are
  mandatory false values.
- `fusion.sparse` and `fusion.dense` each configure a clause pointer, a vector
  field pointer, and its required field name. Both clauses must survive request
  construction; Search API performs query embedding.
- `response_mapping` maps hits, keys, scores, URI arrays, optional inline text,
  and arbitrary metadata without compiled field names.
- `postgres.statement` and `postgres.columns` define full-record lookup. URI
  strings are opaque metadata and no prefix affects inclusion.
- `postgres.transport_security` is mandatory. `verified_tls` forces encrypted
  transport and exact hostname/certificate validation; the production example
  pins the PEM trust bundle by SHA-256 without serializing its contents. The
  only plaintext mode requires an explicit acknowledgement and is restricted
  to numeric loopback addresses or Unix sockets for a trusted local proxy or
  tests. A DSN cannot downgrade either policy.

Discovery rejects overfull pages and inconsistent exact totals before any hit
is staged. Once a backend declares `total_hits`, the strict v2 progress journal
pins it across subsequent pages and resume; totals may not shrink, grow,
disappear, promise records after a short page, or fall below the returned
offset.

The index revision, tokenizer digest, source manifest, and every output shard
are content-addressed. Replace all checked-in placeholder values before a live
build. The configured post-dedup bound is 10–20B unique tokens.

## Reproducible corpus composition

Corpus discovery remains separate from curriculum construction. A version-2
composition file pins one immutable source manifest and declares any number of
stages. Each stage has a token target, shard size, eligibility predicate, and
one or more positively weighted strata. Predicates operate on the generic
tokenized-row schema (`topic`, `difficulty`, `view`, and exact metadata); search
request fields never enter trainer logic.

`summa-train compose-curriculum` streams the verified source into bounded
stratum spools, then uses a deterministic weighted scheduler to emit each
stage. It validates the configured versus actual token fractions, rejects
ambiguous multi-stratum matches, checks every output shard, and atomically
publishes the complete multi-stage build. The source manifest hash and all
mixture settings are included in the composition identity, while filesystem
paths are excluded so the same inputs reproduce the same manifests on another
machine.

The production education example emits fixed `foundation`, `school`,
`university`, and `scholarly` manifests. Those labels are recipe values, not
special cases in the composer; another workflow may use entirely different
stages or metadata keys.

Those four causal manifests are the only education-workflow inputs emitted by
`compose-curriculum`. Retrieval, summarization, reasoning, planning,
preference, RL, and evaluation data paths are independent immutable datasets
with their own task schemas, manifests, and content identities. The composer
does not derive them from causal rows. Fixed stratified mixtures are the
baseline; generated QA and adaptive sampling or transformations are separate,
ablation-gated preparation jobs whose outputs must be pinned explicitly.

Run discovery/materialization first, copy its printed `manifest` digest into
`source_manifest_sha256`, and then compose the stages:

```bash
summa-train prepare-corpus \
  --recipe summa-train/corpus.production.example.json \
  --tokenizer tokenizer.json \
  --output /data/corpora \
  --work-directory /data/corpus-work

summa-train compose-curriculum \
  --config summa-train/curriculum-composition.example.json \
  --output summa-train/corpus \
  --work-directory /data/curriculum-work
```

## Built-in task packages

`TaskAdapter` exposes a stable name, data framing, input/target fields, loss-mask
policy, execution signal, requested metrics, configuration validation, and
record validation. It contains no optimizer, model, search-engine, or
RL-algorithm choices.

| Task                       | JSONL fields                                              | Signal                            |
| -------------------------- | --------------------------------------------------------- | --------------------------------- |
| `causal_lm`                | `text` (plain text is also accepted by the data reader)   | Next-token prediction             |
| `summarization`            | `document`, `summary`                                     | Target-only supervised generation |
| `retrieval_representation` | `query`, `positive`, optional `negatives[]`               | Contrastive representation        |
| `retrieval_ranking`        | `query`, `documents[]` with `document` and `relevance`    | Listwise ranking                  |
| `retrieval_planning`       | `request`, `plan`, optional `context`                     | Target-only supervised generation |
| `instruction_tuning`       | `instruction`, `response`, optional `system` and `input`  | Target-only supervised generation |
| `qa_reasoning`             | `question`, `answer`, optional `reasoning`                | Target-only supervised generation |
| `pairwise_preference`      | `prompt`, `chosen`, `rejected`                            | Pairwise preference               |
| `verifiable_rl`            | `prompt`, `verifier_payload`, optional `reference_answer` | Named verifier reward             |

Retrieval ranking requires at least one positive and one non-positive
candidate. Pairwise responses must differ. A `qa_reasoning` task may require
the reasoning field. Verifiable RL configuration names a verifier adapter and
passes opaque parameters to that adapter; the task schema does not execute
commands or embed a particular RL algorithm.

Existing causal, summarization, retrieval-representation, and
retrieval-planning batches retain their fixed-shape execution. Padding and
prompts do not contribute to supervised target loss, complete targets are
reserved before truncating source text, and retrieval embeddings use the last
meaningful token at the configured one-based layer.

## Post-training algorithms

Preference, general distillation, and RL require a typed `post_training`
object. The library implements the objective math and a deterministic native
phase executor. It binds its resume cursor to the workflow, phase, immutable
input checkpoint, complete data bytes, model/provider/tokenizer identities,
publisher implementation, record/epoch/update counters, and rollout RNG range.
Each update first persists a prepared transaction and then an immutable
model/optimizer receipt; an injected idempotent publisher makes retries after
either boundary safe. Site-specific model loading remains outside trainer core.

- DPO consumes `pairwise_preference`, an immutable reference policy, `beta`,
  label smoothing, and explicit sum/mean sequence reduction.
- Forward KL consumes a frozen teacher distribution aligned to the student's
  selected target-token rows. Its optional temperature-squared scaling is
  explicit.
- GRPO consumes `verifiable_rl`, grouped sampled continuations, clipped policy
  ratios, group-normalized advantages, and an optional immutable reference.
  Positive KL weight requires that reference. `exact_answer` is the only
  built-in verifier; other judges are registered executor adapters.

A local frozen model declares `adapter`, `artifact`, and exact `sha256`;
non-local models declare an immutable `revision`. Artifact paths resolve against
the workflow, symlinks are rejected, and bytes are verified before loading.
Periodic in-model sleep is rejected unless the execution surface owns an
explicit, content-pinned, transaction-idempotent optimizer-boundary hook. The
stock `train` command supplies that hook for its own optimization phases.
The education workflow contains complete DPO, forward-KL, and GRPO phase
examples.

## Benchmark privacy boundary

Public and sealed suites use the same evaluator contract. Their manifests and
examples stay in separately materialized local artifacts and are not copied
into result files. The benchmark-run evidence does retain per-case ids and
scores for sealed suites so an auditor can reproduce the gate. The promotion report is the narrower release
artifact: it exposes public per-case results but only a sealed case count and
aggregate pass/fail result.

## In-model sleep

A standalone `sleep` phase carries its controls under `sleep`; an optimization
phase may install the same object under `periodic_sleep`. The ordered tier
periods must be positive, strictly increasing, and divisible. Coincident
boundaries execute fastest-to-slowest. MAL owns the fast/medium/slow memory
topology and fixed reserve geometry; workflow tier IDs and capacities must
match it.

The stock wake loop accepts `optimizer_steps` schedules. It rejects a
`model_tokens` schedule because one optimizer update cannot be split safely
across multiple crossed token boundaries; lower-level hosts may use that clock
only when they partition work exactly. Wake gradients are partitioned into
ordinary and tier-local scopes, globally norm-clipped, and committed to each
tier's independent AdamW accumulator only after a complete optimizer window.

The tensor backend stages the prospective sender update without mutating the
teacher, runs chunked forward-KL Knowledge Seeding, GRPO-style semantic/edit
imitation, retention gates, and atomic publication. Dream generation reads
model-owned wake contexts rather than corpus/search data; its random extra
expert is enabled only during candidate generation. Gradient-alignment
selection, isolated frozen-model LM-head LoRA trials, and ReSTEM policy updates
cannot mutate the shared candidate on rejection. Each adapter receipt binds the
immutable base checkpoint and parameter digest, logical output-projection
target, exact `[rank, hidden]`/`[rank, vocabulary]` shapes, candidate, and
evaluator.

The first-party runtime implements rollout generation, the frozen
token-semantic judge, likelihood retention evaluation, prospective tier
updates, atomic publication, isolated LM-head LoRA trials, and ReSTEM policy
updates.
Embedding applications can replace the same typed components; all identities
remain content-pinned.

The checked-in sleep workflow uses placeholder evaluator/reference hashes that
must be replaced before execution. The implementation is paper-inspired and
experimental: unlike the paper's growing expert set, Summa preallocates a
bounded reserve and performs a versioned final-tier distillation into the base.
An untrainable rank-matched zero route occupies the reserve lane until a real
slot activates, so the first activation replaces rather than adds top-k expert
compute. Acceptance compares the initial wake count with the complete
min/max envelope over every sleep cycle; it has no post-activation baseline
exception. Router-pool overhead is bounded and separately guarded by the wake
throughput and latency gates.

## Quantization phase

The quantization phase accepts `binary_g128`, `ternary_g128`, and
`ternary_entropy_g128`. `group_size` defaults to and must remain 128.
`embeddings` and `lm_head` default to true. `start_step`, optional `end_step`,
and optional `warmup_format` control progressive quantization.

The nested `training` object is either:

- `{"type":"qat","warmup_steps":0,"straight_through":true}`; or
- `{"type":"distillation","teacher_checkpoint":"...","teacher_sha256":"sha256:<64 lowercase hex>","temperature":1.0,"loss_weight":1.0}`.

Distillation requires both the teacher checkpoint and its exact digest; the
relative path resolves against the workflow. The built-in trainer rejects
symlinks, verifies the artifact, disables teacher dropout, and records the
teacher identity in checkpoint state. Temperature must be positive and
distillation loss weight must be non-negative.

`start_step` and `end_step` are absolute optimizer-step boundaries. QAT may set
a different `warmup_format` plus a positive `warmup_steps`; distillation cannot
use a format warm-up. Matrix fake quantization runs on the active device, uses
the archive codec's deterministic group-128 code and scale estimator, and
preserves parameter IDs so straight-through gradients update the authoritative
full-precision masters. QAT retains scales in the training dtype; HQUANT rounds
them to the declared FP16 storage representation during export. Binary, dense
ternary, and compact ternary archives use
1.125, 2.125, and 1.75 encoded bits per matrix weight respectively for full
128-value groups, including FP16 group scales. Archive-wide bits/weight uses
actual file sizes and also counts every floating tensor.

At the end of a successful built-in quantization phase, the trainer atomically
publishes canonical safetensors, the complete HQUANT archive, and
`candidate.json` under a stable candidate key. It reopens every member before
checkpointing the receipt. Resume repeats that validation and accepts only the
same content-addressed candidate.

## Determinism and metrics

Resolved workflow configuration is part of the run signature. A periodic
sleep runtime is additionally recorded as exactly one canonical-path and
digest artifact in checkpoint v2 and is verified before restoring its tier
state. Resume must preserve phase position, phase-local counters,
optimizer/RNG state, task configuration, and every pinned runtime identity;
reordered or changed inputs fail rather than replaying data under a different
objective.

Metrics identify the phase, phase kind, task, named loss, raw and weighted
loss, optimizer geometry, token/example/truncation counts, throughput, and
task-specific measurements. The JSONL stream remains the source for the W&B
sidecar; lifecycle executors add their own namespaced metrics without changing
task contracts. The closed event catalog also includes phase timing, memory-tier
updates, active capacity, distillation divergence, imitation reward, dream
selection/trials, retention deltas, quantization state, device utilization, and
input/transfer/GPU timing.

The built-in trainer creates `metrics.jsonl` under its checkpoint output and
derives a stable run id from the run signature. The lifecycle runner requires
`--metrics` and `--run-id` together. At each runtime boundary the host syncs the
journal before checkpointing the committed record count. Resume validates the
exact prefix and truncates only an uncommitted/torn tail. The W&B sidecar uses
metric sequence as its W&B step, so a stable `WANDB_RUN_ID` backfills and joins
preemption segments without affecting training.
