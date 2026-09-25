# summa-train

`summa-train` trains and evaluates the same MAL-defined `Transformer` used by
`summa-llm`. Model weights remain safetensors; there is no Python model mirror.

## Build

```bash
cargo build --release -p summa-train                 # CPU
cargo build --release -p summa-train --features metal
cargo build --release -p summa-train --features cuda
```

Build outputs live in `target/release`. To use the commands below from the
repository root, run `export PATH="$PWD/target/release:$PATH"`.
See the [benchmark guide](../docs/benchmarks.md) for wake-tier profiling and
[training contracts](../docs/training-objectives-and-curricula.md) for task semantics.

## Commands

| Command                                 | Purpose                                                   |
| --------------------------------------- | --------------------------------------------------------- |
| `train`                                 | Built-in streaming objectives and optional periodic sleep |
| `validate-workflow`, `run-workflow`     | Validate and orchestrate phase workflows                  |
| `eval`                                  | Held-out loss and task metrics                            |
| `generate-eval`                         | Free-running generation quality                           |
| `retrieval-pool-eval`                   | Retrieval against a shared candidate pool                 |
| `prepare-corpus`, `compose-curriculum`  | Immutable corpus and curriculum builds                    |
| `quantize`, `upgrade-memory-checkpoint` | Explicit weight transformations                           |
| `verify-checkpoint`                     | Validate a sealed generation and metric prefix            |

Run `summa-train <command> --help` for flags. See
[generation evaluation](../docs/generation-eval.md) and
[retrieval evaluation](../docs/retrieval-pool-eval.md).

## WorkflowV2

Use strict [WorkflowV2](../docs/training-objectives-and-curricula.md) configuration.
Unknown fields and unsupported versions fail before execution.

Validate and inspect resolved paths before a run:

```bash
summa-train validate-workflow --workflow summa-train/workflow.example.json
```

Optimization phases may use `steps` for an exact optimizer-step request, or
omit it to consume their configured epochs. Use `max_steps` with the latter to
cap a natural-epoch plan: the resolved length is
`min(epoch_steps, max_steps)`. `steps` and `max_steps` are mutually exclusive.
This distinction matters for replay shards: an exact request can overrun a
small shard and fail, while an uncapped large shard can silently dominate the
workflow. The SFT example uses `max_steps` for retrieval replay for that reason.

For the complete education recipe, also bind validation to the exact
sleep-capable MAL before creating any run state. This rejects missing sleep
hooks, mixed memory/ordinary layers, and tier-name or reserve-capacity drift:

```bash
summa-train validate-workflow \
  --workflow summa-train/workflow.education.example.json \
  --config summa-mal/well-known/retriever_300m_moe_sleep.mal
```

The built-in streaming trainer runs causal LM, summarization, retrieval
planning/representation, instruction-tuning, and QA/reasoning phases, including
QAT or quantization distillation over those objectives. For a MAL model with an
explicit memory hierarchy it also runs periodic in-model sleep at optimizer
boundaries through the content-pinned first-party runtime. It rejects retrieval
ranking, preference, general distillation, RL, standalone sleep, evaluation,
and promotion; dispatch those phases through `NativeWorkflowHost`.

```bash
summa-train train \
  --config summa-mal/well-known/retriever_300m_moe.mal \
  --tokenizer tokenizer.json \
  --workflow summa-train/workflow.example.json \
  --output checkpoint \
  --checkpoint-every 500
```

For the sleep-capable model, use a wake-only WorkflowV2 projection whose every
phase carries the same `periodic_sleep` configuration, and bind the runtime
configuration itself by digest:

```bash
summa-train train \
  --config summa-mal/well-known/retriever_300m_moe_sleep.mal \
  --tokenizer tokenizer.json \
  --workflow wake-with-periodic-sleep.json \
  --sleep-runtime summa-train/sleep-runtime.periodic.example.json \
  --sleep-runtime-sha256 "sha256:$SLEEP_RUNTIME_SHA256" \
  --output checkpoint
```

For the schedule- and capacity-matched no-sleep ablation in the stock wake
trainer, replace `periodic_sleep` on every wake phase with the same typed
`memory_update_mode` object and omit the sleep-runtime flags:

```json
{
  "memory_update_mode": {
    "type": "wake_only",
    "schedule": {
      "clock": "optimizer_steps",
      "terminal_consolidation": "distill_into_base_v1",
      "tiers": [
        { "id": "fast", "update_period": 100, "reserve_slots": 2 },
        { "id": "medium", "update_period": 400, "reserve_slots": 4 },
        { "id": "slow", "update_period": 3200, "reserve_slots": 8 }
      ]
    },
    "tier_optimizer": {
      "learning_rate": 0.0003,
      "beta_1": 0.9,
      "beta_2": 0.95,
      "epsilon": 1e-8,
      "weight_decay": 0.1
    }
  }
}
```

`wake_only` retains each tier's independent clock and pending gradients and
applies due base updates fastest-to-slowest. It performs no consolidation,
transfer, reserve activation/reset, or Dreaming. Checkpoints bind the exact
mode, schedule, optimizer settings, clocks, moments, and pending accumulators.
The native DPO, forward-KL, and GRPO executor rejects `memory_update_mode`
instead of silently applying its ordinary optimizer; for memory-model tier
scheduling, those phases currently support `periodic_sleep` only.
The stock trainer accepts either `[fast, slow]` or `[fast, medium, slow]` so its
typed tier metrics remain unambiguous. Memory training currently requires the
memory hierarchy in every model layer; mixed memory/ordinary-FFN layer
topologies are rejected explicitly.

Profile the complete tier wake step—not only model forward/backward—with the
harness-less `wake_tier_step` benchmark. It emits interleaved raw non-boundary
and boundary timings for at least three paired seeds, including gradient
partitioning, tier accumulation/commit, and the wake-only update path:

```bash
cargo bench -p summa-train --bench wake_tier_step --features cuda -- \
  --model "$PWD/summa-mal/well-known/retriever_300m_moe_sleep.mal" \
  --batch-size 4 --sequence-length 1024 --periods 100,400,3200 \
  --non-due-clock 99 --due-clock 100 --require-cuda \
  --output "$PWD/wake-tier-step-cuda.json"
```

The report records that each paired case starts from a fresh tier bank. This
keeps the compared model, gradients, and optimizer state byte-identical and
profiles the cold first-boundary path; use the separate `memory_reserve`
benchmark's matched static-model gate for the steady-state 5% wake-overhead
acceptance decision.

Runs without the CUDA plus `training-fusion` stack are labelled smoke-only in
the JSON and are not CUDA acceptance evidence. The embedded compact model is
intended only for quick local smoke runs.

Create the runtime's `workflow_signature` without touching checkpoint or
runtime state. Use every training-semantic option from the eventual invocation;
the output path itself is not part of the signature:

```bash
summa-train train \
  --config summa-mal/well-known/retriever_300m_moe_sleep.mal \
  --tokenizer tokenizer.json \
  --workflow wake-with-periodic-sleep.json \
  --print-run-signature
```

For standalone `sleep` under `run-workflow`, derive that field from the
resolved lifecycle workflow instead:

```bash
summa-train validate-workflow \
  --workflow summa-train/workflow.sleep.example.json \
  --signature-only
```

Runtime schema version 1 resolves every relative path against the runtime JSON.
Start from
[`sleep-runtime.periodic.example.json`](sleep-runtime.periodic.example.json) or
[`sleep-runtime.standalone.example.json`](sleep-runtime.standalone.example.json).
Every all-zero digest in those files is an intentionally invalid operational
placeholder: replace it with the exact artifact or execution signature, then
hash the finalized runtime JSON for `--sleep-runtime-sha256`. Periodic mode must
omit `wake_context_journal`,
`initial_tier_optimizer_state`, `initial_model_parameter_ids`, and the nested
Dreaming journal because the trainer seals fresh boundary artifacts. Standalone
mode requires those three top-level artifacts and, when Dreaming is enabled,
the same wake journal in both sections. The five transaction, prospective,
tier-optimizer, candidate, and rejection output directories must be distinct;
the loader creates missing directories and rejects symlinks or non-directories.

The trainer seals a fresh, bounded model-owned wake-context journal at each due
boundary. It checkpoints the exact model, tier optimizers, active masks, RNG
reservations, transaction cursor, and generated-artifact identities after each
sleep subphase. Resume reopens every pinned input and refuses a different
runtime identity. The stock loop schedules sleep on completed
`optimizer_steps`; it rejects a `model_tokens` clock because it cannot split an
indivisible optimizer update across token boundaries. The lower-level sleep
controller retains the typed clock for hosts that can partition work exactly.

Relative data paths resolve against the workflow file. Causal documents are
EOS-joined and token-packed. Corpus decoding, tokenization, bounded shuffling,
and prefetch run on a reader thread. Structured generation losses cover target
tokens only; retrieval uses configured representation layers and explicit
positive/negative grouping.

[`workflow.education.example.json`](workflow.education.example.json) is the
full 300M production recipe: roughly 13.8B scheduled causal tokens progress through
language foundations, school knowledge, university material, and scholarly
material before retrieval, summarization, reasoning, planning, distillation,
DPO, GRPO, binary QAT, public/sealed evaluation, and promotion. Only the four
causal tiers are outputs of `compose-curriculum`. Every later task data path is
a separately prepared immutable input with its own task schema, manifest, and
content identity; the causal composer does not synthesize retrieval, QA,
preference, RL, or evaluation records. Fixed stratified mixtures are the
baseline. Generated QA and adaptive sampling or transformations must be built
as separately identified, ablation-gated artifacts.
Every optimizer-bearing phase in this sleep-model recipe—including forward-KL,
DPO, GRPO, and binary QAT—carries the same explicit `periodic_sleep` object and
uses the phase-neutral `candidates/education-sleep` store. Evaluation and
promotion do not install wake hooks.

For a backend that implements multiple phase algorithms, the stock lifecycle
CLI uses one JSONL executable for ordinary phases. Promotion is always
executed by the trainer's built-in verified gate and is never delegated to that
executable:

```bash
summa-train run-workflow \
  --workflow summa-train/workflow.example.json \
  --executor /opt/summa/bin/phase-worker \
  --state /data/run/workflow-runtime.json \
  --metrics /data/run/metrics.jsonl \
  --run-id workflow-seed-1 \
  --initial-checkpoint-uri checkpoint://initial/generation-manifest.json \
  --initial-checkpoint-sha256 "sha256:$INITIAL_CHECKPOINT_SHA256"
```

The stock command sends each non-sleep, non-promotion phase—including typed
DPO, forward-KL, and GRPO phases—to that worker. Add `--sleep-runtime PATH` and
`--sleep-runtime-sha256 sha256:...` to execute standalone `sleep` phases with
the first-party model loader, deterministic rollouts, frozen token-semantic
judge, likelihood retention evaluator, tier optimizer, Dreaming backend, and
durable stores; use the checked-in standalone runtime example above. Its input
checkpoint URI must name a local regular safetensors file because this
first-party runtime is deliberately local-artifact-only. `run-workflow` still
rejects `periodic_sleep`, because only the integrated `train` loop owns real
optimizer boundaries. The worker receives one
strict protocol-v2 JSON request on stdin and emits
typed `metric` messages (event plus global step and optional checkpoint hash),
newline-delimited progress cursors, and exactly one yielded or complete
message. The host derives phase identity instead of trusting worker-supplied
labels. At every runtime boundary it syncs the metric journal before atomically
recording the committed metric count; resume validates that prefix and removes
any uncommitted or torn tail. Resume also requires the same resolved workflow,
worker dispatch identity, and metric run id. Optimization/mutation results must be new
immutable checkpoint URIs and assessment cannot change weights. The native
promotion gate releases only the exact accepted current candidate.

All phase, benchmark, and resource workers run with an empty inherited
environment and `/` as their working directory. A script worker must therefore
name its interpreter directly with an absolute shebang such as `#!/bin/sh` or
`#!/opt/summa/venv/bin/python`; `#!/usr/bin/env ...` cannot be relied on
without an ambient `PATH`. Worker subprocess paths must likewise be absolute or
be resolved internally by a self-contained binary—the host never injects an
ambient `PATH`.

If a worker yields, invoke the same command with `--resume` and without either
initial-checkpoint option. The runtime verifies the workflow, worker dispatch
identity, metric run id, and committed metric prefix before continuing. The concrete
post-training library implements DPO, forward `KL(teacher || student)`, clipped
GRPO with a non-negative KL estimator, JSONL/zstd streaming, and an exact-answer
verifier. Its native `PhaseExecutor` checkpoints a content-bound cursor before
each optimizer publication and after its immutable model/optimizer receipt, so
either interruption window resumes without applying an update twice. The
trainer injects task-aligned model, frozen-reference, teacher-distribution,
rollout, idempotent publisher, and any non-built-in verifier adapters; there is
no repository-specific model loader. Adapter, input, workflow, provider, RNG,
and publisher identities are verified on resume. A phase with `periodic_sleep`
is accepted only when the trainer supplies an authenticated
`PostTrainingClockReceipt` for the exact input checkpoint and the explicit
idempotent boundary hook. Trainable policy adapters report a monotonic exact
model-token counter; the executor samples it around each deterministic update
instead of estimating token usage from padded sequence geometry.

Forward-KL accepts causal-LM and supervised-generation task examples, including
instruction tuning, and applies the task's explicit prompt/target structure.
Every local frozen teacher or reference uses the same canonical
`sha256:<64 lowercase hex>` identity as runtime checkpoints; alternate digest
spellings are rejected rather than normalized. Revision-pinned remote models
derive that canonical runtime identity from the adapter, immutable revision,
and adapter parameters, so equal revision labels from different providers or
model repositories cannot alias one another. Context factories should use
`FrozenModelSpec::immutable_identity()` as their exact expected value.

The complete education recipe needs the embedded `NativeWorkflowHost`, with
registered model, storage, judge, and evaluator adapters. Missing routes fail
before state is created. See [execution surfaces](../docs/training-objectives-and-curricula.md#execution-surfaces)
for native DPO/KL/GRPO, periodic wake, standalone sleep, and promotion routing.

## Corpus preparation

The generic pipeline is discovery → materialization → normalization → exact
deduplication → topic/difficulty classification → controlled transformations
and repetition → tokenization/sharding → immutable manifest. `SearchBackend`,
`RecordMaterializer`, `CorpusTokenizer`, and `Deduplicator` are replaceable.

The production example uses Search API hybrid sparse+dense fusion with both
rerankers forced off, followed by `PostgresRecordMaterializer` for canonical
document bodies. Query clauses, request fields, result mappings, SQL fields,
pagination, and URI values are configuration—not trainer constants. Secrets are
read only through configured environment-variable names. Its post-dedup target
is 10–20B unique tokens.

PostgreSQL transport policy is mandatory and cannot be weakened by the DSN.
The production example uses `verified_tls`, forces TLS 1.2 or newer, verifies
the exact configured server name, disables platform roots, and reads a PEM CA
bundle from `CORPUS_POSTGRES_ROOT_CERT_PEM` only after its configured SHA-256
matches. Set the DSN's `host` to the same DNS name; an optional `hostaddr` may
select its network address. Use `trust.source: system` only when platform trust
is an intentional deployment input. There are no invalid-certificate or
invalid-hostname switches.

`plaintext_local_proxy` exists for a trusted local AlloyDB Auth Proxy or local
tests. It requires `acknowledge_plaintext: true` and accepts only numeric
loopback IPs or Unix sockets, while forcing TLS off for that one local hop; the
proxy remains responsible for its authenticated encrypted upstream. Direct
PostgreSQL endpoints should use `verified_tls`. If an AlloyDB direct endpoint
cannot present a verifiable certificate/name, use the Auth Proxy rather than
weakening verification.

`request_mapping.query_pointer`, pagination pointers, parameter mappings, and
response mappings define the provider wire format. `fusion.sparse` and
`fusion.dense` independently declare their clause and vector-field pointers;
the pipeline verifies both clauses on every request. Every pointer in
`disabled_reranker_pointers` must resolve to `false`. Search API owns query
embedding, while the corpus process sends text and materializes full records
from PostgreSQL. `snapshot.request_revision_pointer` pins the configured index
generation on every request; the configured response provider/revision pointers
must echo that exact identity on every page. A missing or drifted proof aborts
before page contents enter the discovery catalog. Discovery also rejects a page
larger than its requested limit, a total below the returned offset, a short page
that contradicts a declared total, or a total that changes or disappears on a
later page. The strict v2 progress journal retains the declared total across
resume. The checked-in endpoint,
snapshot revision, digest values, and credentials are placeholders and must be
replaced with pinned production values.

```bash
summa-train prepare-corpus \
  --recipe summa-train/corpus.production.example.json \
  --tokenizer tokenizer.json \
  --output /data/corpora \
  --work-directory /data/corpus-work
```

Completed build directories are immutable and contain checksummed token shards
plus `manifest.json`. Discovery pages are first merged by record key and then
materialized in canonical key order, so overlapping query order cannot select a
record's curriculum tier. Topic and difficulty rules inspect canonical text and
configured canonical metadata; explicit priorities and specificity resolve
overlap independently of rule declaration order.

`prepare-corpus` is crash-resumable. Each discovery page or materialization
batch commits its SQLite discovery/dedup mutations together with a
content-hashed cursor, after syncing any completed output shards. Re-running the
same command resumes `.building`, repairs only a journal mirror that is one
or more committed transactions behind, removes uncommitted regular shard tails, and
fails closed on identity, journal, committed-file, or symlink drift. The search
index proof, PostgreSQL `snapshot_statement`, tokenizer revision, recipe, and
dedup policy must remain identical across resume. Completed publication is
idempotent. A build aborts if a source snapshot changes or the final unique-token
count falls outside the configured bounds.

The PostgreSQL `snapshot_statement` must return a stable, remotely maintained
dataset generation that changes whenever materialized source rows change. An
MVCC value such as `txid_current_snapshot()` is intentionally unsuitable for
cross-process resume: a restarted build will fail closed because that value is
different. Keep rows for a published generation immutable, or include the
generation in the materialization query.

Turn one classified build into fixed topic/capability mixtures with the generic
composition pass. First replace `source_manifest_sha256` with the canonical
`manifest_sha256` printed by `prepare-corpus`:

```bash
summa-train compose-curriculum \
  --config summa-train/curriculum-composition.example.json \
  --output summa-train/corpus \
  --work-directory /data/curriculum-work
```

The example atomically publishes
`corpus/education-curriculum-2026-08/{foundation,school,university,scholarly}`,
matching the paths in the education workflow. Stage and stratum predicates may
use classified `topic`, `difficulty`, view names, or exact arbitrary metadata;
no search provider or field name is compiled into composition. Positive integer
weights define the fixed token mixture, and `max_fraction_deviation` is checked
before publication. A row matching two strata in one stage is rejected rather
than silently depending on configuration order; eligible unmatched rows must
be explicitly set to `exclude` or they fail the build. Source rows are copied with
their URI strings untouched, and bounded transient spools avoid loading the
10–20B-token corpus into memory. Composition syncs a content-pinned progress
journal after every source and output shard. Re-running the same command
resumes an interrupted `.building` generation, verifies every committed
spool/shard and identity, and replays only the uncommitted shard; configuration,
source, journal, file, or symlink drift fails closed.

## In-model sleep memory

MAL can opt a block into an ordered fast-to-slow `memory` chain. Fixed low-rank
reserve experts, router rows, activation masks, and generation counters are
preallocated. Dormant learned slot parameters have no forward, backward,
auxiliary-loss, or optimizer path. The router projection nevertheless has its
full configured capacity width from the first checkpoint: parameter-free zero
columns occupy dormant lanes, and each active learned row replaces its lane.
A deterministic untrainable low-rank fallback executes the exact zero-output
top-1 route until an active slot replaces it, so both router width and wake
reserve top-k stay constant through every sleep cycle. The additional random
expert is available only through dream-generation forwards.

`retriever_300m_moe_sleep.mal` is experimental and leaves the ordinary 300M
preset unchanged. Upgrade an existing checkpoint explicitly:

```bash
summa-train upgrade-memory-checkpoint \
  --source-config summa-mal/well-known/retriever_300m_moe.mal \
  --target-config summa-mal/well-known/retriever_300m_moe_sleep.mal \
  --checkpoint weights.safetensors \
  --output sleep-weights.safetensors
```

The upgrader strictly maps each source FFN into the fast tier, initializes
slower tiers as residual no-ops, leaves reserves dormant, and verifies topology
and logit parity in tests. KV caches and Mamba recurrent state remain ordinary
inference state and are not long-term memory.

Sleep schedules, Knowledge Seeding, semantic/edit imitation with GRPO,
retention transactions, dream selection by model-loss gradient alignment,
isolated frozen-model LM-head LoRA trials (paper geometry rank 64 / alpha 128),
and ReSTEM updates have an in-process tensor implementation. Each built-in
trial supervises only the generated continuation, derives its final-normalized
hidden features once, caches the immutable evaluation features across trials,
and trains only `[rank, hidden]` and `[rank, vocabulary]` adapter tensors with
batched device GEMMs. It adds their delta to the frozen base projection. Its
content-addressed receipt binds the adapter to the
immutable base checkpoint and exact model-parameter digest, logical
output-projection target, exact shapes, candidate, and evaluator. The stock
runtime provides model-owned rollout generation, a frozen
content-pinned token n-gram overlap judge, a pinned likelihood retention
evaluator, prospective tier updates, and atomic publication. The n-gram judge
is deterministic but is not a neural semantic judge; deployments can provide a
frozen semantic model through the same typed interface. A ReSTEM cycle with no
positive isolated trials publishes an idempotent no-op policy node, and each
later periodic cycle authenticates and uses the newest committed node as its
parent, including after exact resume.
Transaction IDs, teacher/student hashes, subphases, masks, reserve generations,
evaluator hashes, artifacts, optimizer namespaces, and RNG streams are
checkpoint-v2 state. Failed consolidation restores the immutable teacher; a
sender slot is reset only after accepted transfer.

Wake training partitions gradients into ordinary and per-memory-tier scopes.
Each tier has its own AdamW state, accumulator, and update clock; a partial
gradient-accumulation window is discarded for every scope. One global norm and
clip covers wake and tier gradients before a completed optimizer window is
committed.

Dream generation prefills each wake prefix once and then uses the ordinary KV
or recurrent inference state for linear-time single-token decoding. It adds
exactly one deterministic-random expert per token from outside each persistent
FFN MoE's ordinary top-k. Reserve-memory routers remain
ordinary top-1 from the initial zero fallback onward. Their capacity-width
projection is fixed; dormant learned rows stay off graph behind parameter-free
zero columns, and activation replaces columns without growing router work.
Routed-active acceptance counts that fixed low-rank lane before activation and
every completed cycle; there is no first-activation baseline exception. The
independent 5% wake throughput and latency gates cover this fixed-width router
and fixed expert-matmul path.

This is a versioned, paper-inspired implementation of [_Language Models Need
Sleep_](https://arxiv.org/abs/2606.03979). The paper grows low-rank experts;
Summa instead preallocates bounded reserves and uses the versioned
`distill_into_base_v1` terminal transaction so stored and routed-active capacity
remain bounded. That adaptation, plus unspecified paper hyperparameters, means
the implementation is experimental until paired acceptance runs reproduce the
reported trends.

## Ultra-low-bit quantization

In the command below, replace `sha256-HASH` with the generation named by
`checkpoint/current.json`.

Summa supports group-128 binary (`1.125` encoded bits/weight for full groups),
dense two-bit ternary (`2.125`), and compact base-3 ternary (`1.75`) checkpoint
codecs, including one FP16 scale per group. The training path performs deterministic
fake quantization on the active device and uses a straight-through estimator
while full-precision master weights and optimizer state remain authoritative.
Embeddings and LM heads are included by default; scalar norms and other
non-matrix tensors stay in their original dtype.

```bash
summa-train quantize \
  --checkpoint checkpoint/generations/sha256-HASH/weights.safetensors \
  --format binary-g128 \
  --output checkpoint/weights.binary-g128
```

The output is a complete, immutable weight archive: HQUANT matrix files, byte-exact
floating tensors, per-tensor error measurements, source SHA-256, and a manifest
whose average bits/weight includes every stored tensor. A direct packed
matrix-vector implementation is a correctness oracle, not a claim that a fused
CUDA/Metal production kernel or runnable quantized Transformer backend already
exists. Model topology and tokenizer assets remain separate. The binary weight alphabet,
group size, and FP16 scaling follow PrismML's published
[Bonsai representation](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf); HQUANT is
a native Summa archive, not a GGUF file. Summa does not claim to reproduce
PrismML's undisclosed conversion or training algorithm. The binary scale
estimator is the deterministic least-squares optimum for Summa's fixed sign
codes; both that conversion and Summa's ternary L2/base-3 paths are explicitly
Summa-native and are not attributed to PrismML.

Quantization is also a task-data WorkflowV2 training phase. The education
workflow demonstrates ternary warm-up followed by binary QAT and treats QAT as
an optimizer-bearing periodic-sleep phase. If its final optimizer step is a
sleep boundary, the trainer commits and checkpoints that boundary first, binds
the resulting model digest, and only then atomically publishes and reopens
`quantized-candidates/<stable-key>/{weights.safetensors,hquant/,candidate.json}`,
records the immutable candidate receipt in checkpoint v2, and emits exact
archive/error metrics. Resume can finish an interrupted post-sleep publication,
while a candidate recorded before an outstanding boundary is rejected. A retry
accepts only the same validated bytes. To use a frozen teacher instead,
configure the training object with an exact, lowercase, prefixed digest:

```json
{
  "name": "binary-distillation",
  "type": "quantization",
  "task": { "type": "causal_lm" },
  "data": "corpus/quantization-calibration/manifest.json",
  "sequence_length": 2048,
  "batch_size": 12,
  "gradient_accumulation": 8,
  "steps": 1000,
  "quantization": {
    "format": "binary_g128",
    "group_size": 128,
    "start_step": 0,
    "embeddings": true,
    "lm_head": true,
    "training": {
      "type": "distillation",
      "teacher_checkpoint": "teachers/full-precision.safetensors",
      "teacher_sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      "temperature": 1.0,
      "loss_weight": 1.0
    }
  }
}
```

Relative teacher paths resolve against the workflow. The built-in trainer
rejects symlinks, verifies the teacher bytes before loading, disables teacher
dropout, and persists the teacher identity in quantization checkpoint state.
`start_step` and `end_step` are absolute optimizer-step boundaries.
Replace the all-zero example digest with the teacher file's real SHA-256 before
running it.

## Acceptance and promotion

Public and sealed suites use paired seeds and a capacity-/compute-matched
baseline. Promotion requires at least three paired seeds, a positive paired
confidence bound for improvement cases, stable-anchor regression no greater
than one absolute point or measured baseline variation, exact resume, constant
routed-active parameters, at least 95% wake throughput, and at most 5% wake
latency regression.

Acceptance suites and benchmark suite manifests are strict schema v2. Each case
has an explicit catalog identity and `stable_anchor` value. The acceptance-policy
v2 artifact prescribes the complete anchor catalog set; promotion rejects any
suite flag that differs, so a suite author cannot exempt an improvement case.
The policy also names the resource evaluator, and pins minimum trial/sample
counts, parity error ceilings, and wake performance ratios. Resource evidence
contains no tolerances.

An untrainable rank-matched zero route occupies each memory tier's reserve route lane, so
the first receiver activation replaces equal active compute instead of adding
it. Resource-comparison v2 capacity evidence contains cycle zero followed by
every completed sleep cycle without gaps. Promotion recomputes each envelope
and requires routed-active parameters, stored parameters, and stored bytes to
remain exactly equal to cycle zero and the candidate target; that target may
not exceed the matched baseline. There is no activation or topology resize
exception.

Both public and sealed manifests must cover causal pretraining, summarization,
retrieval representation/ranking, retrieval planning, reasoning/QA, preference,
verifiable RL, synthetic retention, CLINC, Banking77, MK-NIAH, QASPER,
no-context SQuAD incorporation, and ARC Dreaming. Later sweeps also require
Manchu, Kalamang, and BABILong. Catalog IDs and benchmark families are fixed by
the runner rather than inferred from filenames.

Benchmark runs are produced by `BenchmarkRunner::run` with a configured
evaluator (`summa_train::benchmark_worker::ExternalBenchmarkEvaluator` speaks
the JSONL worker protocol). `evaluator_arguments` in the run config is the exact
UTF-8 argument vector passed to that worker. The host clears the inherited
environment and runs the evaluator from `/`, so the evaluator must be
self-contained and may not depend on ambient shell, working-directory, or secret
state.

Each target JSON points to its checkpoint manifest and must declare the bytes
evaluated with a required `representation`. Use `{"type":"full_precision"}` for
the sealed generation weights, or
`{"type":"hquant","candidate_manifest":".../candidate.json","candidate_manifest_sha256":"sha256:<64 lowercase hex>"}`
for a published QAT candidate. There is no implicit full-precision target.
Resource accounting is sealed inside the checkpoint generation as
`training-accounting.json`, where the generation manifest authenticates it and
the actual `weights.safetensors` bytes.

For this single-accelerator trainer, `training_gpu_hours` is the measured
committed training time: the integer-nanosecond sum of optimizer-step
`Throughput.elapsed_seconds` windows plus successful, non-overlapping sleep and
quantization-export phase windows, divided by 3600. Wake windows include
input/host stalls; failed attempts and ordinary checkpoint serialization are
not represented by this checkpoint-bound metric. External benchmark jobs also
enforce their measured per-evaluation GPU-hour budget. `gpu_busy_seconds`
remains a utilization diagnostic and is not substituted for compute budget.
For a full-precision target, `stored_bytes` is exactly the sealed
`weights.safetensors` length. For an HQUANT target it is the sum of every
validated packed and retained-floating weight member in the archive; metadata,
the candidate's canonical FP master, candidate-container/manifest overhead,
optimizer, and trainer state are excluded. `parameters` is checked against the
HQUANT archive inventory.
`routed_active_parameters` includes all
non-expert parameters, complete routers and shared experts, and ordinary top-k
routed experts. Dense, MoE, and sleep-memory models are measured from the live
module; memory accounting follows the synchronized active-slot masks while its
fixed fallback/active reserve route lane remains constant.

The persistent benchmark evaluator receives local artifact paths, target
checkpoint manifests, the concrete FP weights or HQUANT archive transport,
fixed model/example-order seeds, target role, pair ordinal, and the hard
per-evaluation GPU-hour budget. An HQUANT evaluator must supply its own real
backend; the trainer does not dequantize to FP as a benchmark substitute. It
returns only a finite score, measured GPU hours, example count, and optional
finite metrics. The runner itself verifies the complete public/sealed catalog,
equal example counts, target manifests, ordering, budgets, and at least three
strictly ordered paired seeds.

`summa_train::resource_worker::run_resource_benchmark` executes the resource
host after all eleven benchmark runs exist. It sends one bounded JSON line and
accepts exactly one bounded JSON line. The response contains raw wake trials,
capacity observations, grouped-mm and PyTorch samples, and relative exact-resume
artifact references. The host derives the strongest matched baseline, verifies
the exact-resume artifacts, and writes `resource-comparison.json` into the
output directory. A timeout, extra output, or a symlinked evaluator binary fails
closed and the child process group is reaped.

`summa_train::benchmark::evaluate_verified_promotion` takes the selected run,
the complete eleven-ablation comparison matrix, the resource comparison, and the
acceptance policy. The selected run must appear exactly once in the matrix.
Every run must use the same evaluator, suites, candidate, ordering, paired
seeds, and measured training-compute allowance. Resource-comparison v2 stores
raw paired token/elapsed/latency observations, raw per-cycle capacity
observations, and raw reference/candidate kernel values. Promotion recomputes
aggregate throughput, nearest-rank p95 latency, capacity minima/maxima, and
maximum absolute/relative parity errors — evidence never submits an aggregate.

Exact-resume evidence names the interrupted generation, both distinct final
generations, and both metric journals. Promotion reads all five, requires the
two final generation manifests to be byte-identical, checks the recorded
interruption/final steps, and recomputes the semantic metric digest of both
journals. The semantic digest omits run IDs, record sequences, timestamps,
device samples, and elapsed/rate fields while retaining steps, phases, token
counts, objectives, losses, rewards, and checkpoint identities. Raw logs from
interrupted and uninterrupted runs are therefore allowed to differ in honest
wall-clock observations; final state and semantic progress must still match
exactly.

Benchmark artifacts retain sealed case IDs and scores for audit, never private
examples. Promotion reports expose public results and an aggregate sealed gate.

For WorkflowV2 release, configure the typed `promotion` object shown in
[`workflow.sleep.example.json`](workflow.sleep.example.json): one selected run,
exactly ten distinct comparison runs, resource evidence, and an acceptance
policy, each given as `{"path": "..."}`. Paths resolve against the workflow. The
built-in gate requires the selected candidate manifest hash to equal the
runtime's current checkpoint hash. It writes a deterministic decision under
`artifact_directory`. Retries accept only byte-identical existing decisions;
rejected decisions are retained for audit but never create a release receipt.
Accepted receipts remain explicit inputs to serving promotion; training never
mutates serving weights.

## Checkpoints, metrics, and relaunch

Checkpoint v2 contains model and optimizer state plus workflow phase counters,
token counts, metric journal position, parameter IDs, data/artifact references,
independent optimizer namespaces, sleep state, quantization state, evaluator
hashes, and RNG counters. Each save seals an immutable
`generations/sha256-<manifest-hash>/` directory,
including `generation-manifest.json`, then atomically replaces `current.json`.
Every manifest file is size- and SHA-256-verified before resume. Unsupported
checkpoint schema versions fail explicitly.

`summa-train verify-checkpoint --root CHECKPOINT --metrics
CHECKPOINT/metrics.jsonl` exposes the same strict generation, training-state,
accounting, optimizer-reference, and committed-metric-prefix checks for
automation. The relaunch supervisor invokes this command for every local,
downloaded, and pre-publication generation; its Python helper is limited to
transport envelopes and immutable artifact copying. The built-in trainer also
holds `.trainer.lock` for its full process lifetime, so a second direct trainer
cannot mutate the same output root behind the supervisor.

`metrics.jsonl` is a strict append-only schema-v2 stream. Typed events include
optimization losses, throughput, phase timing, tier updates, active capacity,
post-training update receipts, distillation divergence, imitation rewards,
dream selection/trials, retention, quantization, and device utilization. Resume
validates the committed prefix and removes only metric records newer than the model checkpoint. Already committed
optimizer-window durations remain byte-for-byte unchanged, so later checkpoint
evidence accumulates prior measured time plus newly observed windows; it does
not claim wall-clock timing is reproducible across executions. The built-in
trainer creates this journal in `--output`; `run-workflow` requires `--metrics`
and `--run-id` together. `--layer-metrics-every N` opts the built-in trainer into
the more expensive per-layer gradient-norm series used by the visualization
lab.

CUDA builds start one persistent, asynchronous `nvidia-smi` sampler at a
one-second interval. Select the physical GPU by index, UUID, or PCI bus ID with
`--gpu-physical-device`; set `--gpu-metrics-interval-ms 0` to disable it or a
value of at least 100 to change the cadence. Non-CUDA builds default to zero.
Both values are part of the run signature, so a resume cannot silently change
the monitored device or cadence. Samples contain real GPU utilization,
used/total memory, power, and temperature and retain their exact collection
timestamp; the trainer drains a bounded channel into `metrics.jsonl` at safe
boundaries so the metric writer remains single-owner. Missing `nvidia-smi`, an
unsupported counter, malformed output, process exit, or channel pressure emits
a warning and never stops or synchronously polls the accelerator. The W&B
sidecar forwards these `device_utilization` events unchanged.

[`scripts/relaunch.sh`](scripts/relaunch.sh) supervises preemptible jobs, resumes
only complete checkpoints, and derives an exact generated-artifact closure from
the sealed `training-state.json`. Sleep transactions, tier optimizer snapshots,
Dreaming trials, wake journals, QAT candidates/archives, and checkpoint-bound
training evidence are stored once in a global SHA-256 object store; each
generation receives a small immutable closure manifest. The supervisor uploads
and re-verifies the checkpoint generation, closure manifest, and every referenced
object before publishing `current.json` last. The remote pointer binds the
generation, closure digest, and an exact committed metrics prefix stored under
`checkpoint-metrics/<generation>/metrics.jsonl`. Pointer publication is
monotonic and compare-and-swap protected; an equal-step generation fork fails
closed. Restore verifies those bindings before atomically installing immutable
artifacts and refuses to overwrite a different local file. Mutable pointers,
staging files, and artifacts from a later generation are never captured. An
external Dreaming `initial_policy` is a deployment-owned input: its path and
digest are pinned by the sleep runtime and must be provisioned unchanged on a
replacement host; generated descendants and adapters remain in the artifact
closure. The transport supports `gs://` and `file://`. The supervisor wraps the
built-in `train` command; a service
supervising `run-workflow` must preserve its runtime state/metric files and add
`--resume` after a yielded or interrupted run. Copy
[`scripts/relaunch.conf.example`](scripts/relaunch.conf.example), keep W&B
credentials in its referenced mode-600 environment file, and run:

```bash
summa-train/scripts/relaunch.sh /opt/summa-run/relaunch.conf
```

When configured, [`scripts/wandb_tail.py`](scripts/wandb_tail.py) validates and
flattens typed JSONL events into a stable W&B run. Install `wandb` in the Python
environment named by `SUMMA_TRAIN_WANDB_PYTHON`, set a stable `WANDB_RUN_ID`,
and keep `SUMMA_TRAIN_WANDB_ENV` outside the repository. W&B remains a sidecar:
a reporting or network failure cannot stop training.
