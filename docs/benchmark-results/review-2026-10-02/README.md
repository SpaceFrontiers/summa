# Search and I/O review — October 2

This reviews the accumulated search and payload I/O changes and repeats the full
Searchbench comparison on the retained 10M-document indexes. Measurements use
new release binaries from the reviewed source. The companion fixed-arrival I/O
run compares mmap, positional workers and io_uring under a 256-MiB memory cap.

## Full matrix results

All **121,717,668 requests completed without errors**. The 677 admitted expressions preserve
each Summa index’s retained counts, plans, top-100 IDs and raw score bits; all three
index inventories match before/after and the retained September 30 inventories.
The [file inventory](index-inventory.json) retains each size and SHA-256.
The [total table](table.md) contains every one of the 57 family/operation cells.
The [228-row CSV](total.csv) adds CPU/request, peak RSS, anonymous RSS and observed
p99 ranges for all four variants; [validated JSON](campaign-results.json) retains
every phase and repetition. Different workloads are not summed into a misleading total QPS.

| Variant          | Cells ≥ Luxir | Cells 0.95–<1× Luxir | Cells <0.95× Luxir | Index GiB | Peak RSS MiB | Peak anonymous RSS MiB | End-of-phase index mapping RSS MiB |
| ---------------- | ------------: | -------------------: | -----------------: | --------: | -----------: | ---------------------: | ---------------------------------: |
| Default          |            28 |                    6 |                 23 |     17.42 |      3,907.6 |                  390.9 |                            3,502.9 |
| Word pairs       |            31 |                    4 |                 22 |     17.93 |      3,975.1 |                  391.5 |                            3,570.1 |
| RGB + word pairs |            47 |                    0 |                 10 |     13.35 |      3,692.1 |                  387.8 |                            3,290.9 |
| Luxir 0.1.0      |             — |                    — |                  — |      8.93 |      1,212.2 |                   79.7 |                    1,107.5–1,109.4 |

Across the two complete passes, the largest phase-median QPS difference is
4.72% for default, 5.81% for pairs, 7.79% for RGB + pairs and 9.03% for Luxir
(measured as max/min − 1 within one cell). Repetition and phase values remain
visible in the JSON; small ratios around 1× should not be read as proven wins.

The 0.95 boundary is a descriptive reporting bucket, not a statistical test.
Peak RSS is the largest sampled process value across cells; anonymous RSS is a
subset, not a separate amount to add. Mapping residency is sampled after each
phase and includes evictable pages. Index sizes are Summa’s hashed file totals
and Luxir’s reported searchable segment size. These engines have different
schemas/layouts, and 37 shared-input expressions have different retained counts.

RGB + pairs leads in 47/57 cells, rather than the historical 50/57 near-or-ahead
summary from the earlier campaign. The ten slower cells are explicit below;
none is hidden by averaging with phrase or broad-pattern gains.

| Remaining RGB + pairs gap | Operation | × Luxir |
| ------------------------- | --------- | ------: |
| and_high_med              | TOP_10    |  0.929× |
| and_high_med              | TOP_100   |  0.925× |
| high_term                 | COUNT     |  0.877× |
| low_term                  | TOP_10    |  0.892× |
| low_term                  | COUNT     |  0.869× |
| med_term                  | COUNT     |  0.881× |
| prefix3                   | TOP_10    |  0.791× |
| prefix3                   | TOP_100   |  0.783× |
| regex                     | TOP_10    |  0.930× |
| regex                     | TOP_100   |  0.940× |

High-frequency phrase TOP_10/TOP_100 gains are 72.40×/82.98× with RGB + pairs;
COUNT is 1,570.54× on this particular materialized-pair workload. These are
workload-specific results, not general engine speedups. Broad wildcard remains
4.36×/4.37× Luxir for ranked retrieval and 4.20× for COUNT. Prefix ranked retrieval
is only 0.79×/0.78× Luxir. Very cheap high-term COUNT illustrates a different
tradeoff: Summa is 0.877× Luxir in throughput but uses 217.6 versus 57.2 server
CPU microseconds per request. HTTP admission/scheduling costs matter at this scale.

The CSV includes CPU and memory evidence for these tradeoffs. No index,
backend, cache or concurrency default is changed from this one machine.

## Matched four-binary control

All **49,686,066 requests complete without errors** across eight phases and
154 admitted expressions; 102 budget exclusions remain explicit. Every arm
preserves HTTP counts and top-10/100 IDs against the same RGB + pairs audit.
The fixture hashes also match after the control, and the isolated-network
wrapper restores the host namespace restriction.

The [18-cell control table](control-table.md) shows all four absolute QPS values,
review/baseline throughput and CPU ratios. [Control JSON](control-results.json)
retains phase/repetition ranges and memory. The reviewed build measures
**0.974–1.031×** the normal baseline. Its weakest mean is wildcard TOP_100
(0.974×, CPU/request 1.004×); that cell is slightly faster in the first reviewed
pass and slower in the second. This is a measured small difference, not grounds
for claiming either a proven regression or universal performance neutrality.
Prefix TOP_100/COUNT and regex COUNT measure 0.983×/0.984×/0.986× baseline.

The older diagnostic count binary reproduces the broad-wildcard ranked
regression: **0.875×** its old baseline for both limits, with CPU/request
**14.7%/14.2% higher**. However, the normal review baseline—whose production Rust
sources match that diagnostic count binary—already avoids this slowdown.
The reviewed build is 0.994×/1.002× that normal baseline for broad-wildcard
TOP_10/TOP_100, and 1.003×/1.013× the old pre-count baseline. Thus this review's
helper cleanup must not be credited with fixing the old regression. The
retained diagnostic binary remains slower; compiler/layout causation is
unproven, and no speculative hot-path rewrite was introduced to chase it.

Against the old pre-count baseline, the reviewed build preserves the count
optimization: wildcard **2.131×**, prefix **1.241×**, regex **1.433×**, and broad
wildcard **1.148×**. The old diagnostic count binary shows only 0.992× for broad
COUNT on this boot, another reason to keep all four arms visible. High-term
COUNT spans 96,641–99,601 QPS across these arms; reviewed/baseline is 1.031×.
The lower cheap-query throughput relative to older campaigns therefore cannot
be attributed to this review's source changes. Focused-control absolute QPS
should not be combined with full-matrix Luxir rows as if they were paired.

Peak sampled RSS is 2,970.0 MiB for the normal baseline and 2,969.0 MiB reviewed;
peak anonymous RSS is 385.8 versus 385.0 MiB. The shared helper cleanup adds no
cache or resident index structure. Remaining engine gaps are those shown in
the full matrix, not erased by this control.

## Fixed-arrival I/O results

All **54 load cells and three smoke cells pass**. The load cells contain
**259,200 offers, 228,080 accepted and verified queries, 31,120 rejected offers,
and 7,298,560 decoded documents**. Smoke cells verify another 192 queries and
6,144 documents. Fixture bytes are unchanged. I/O, worker, notification,
quarantine and OOM failures are zero. Rejection is an observed admission outcome,
not a successful execution of the rejected query.

The [complete I/O table](io-table.md) includes every workload/rate/backend group;
[validated I/O JSON](io-results.json) retains all 54 cells, per-query acceptance,
resource counters and raw-observation hashes. Table entries are means of three
observations. Completed/s and CPU/query include the eight-second arrival window
and final drain. First-second and later rejection use intended arrival times;
“later” means seconds 1–8, not proven long-term stationarity. The displayed p95/p99
are means of per-cell quantiles over accepted queries, not pooled percentiles.
CPU includes oracle verification. Cgroup memory includes the engine, page cache,
bounded samples and observation export.

At MaxScore 400/800 offered queries/s, io_uring uses **5.94%/3.49% less CPU per
completion than pool**. At mixed 200/400/600 it saves **5.39%/5.23%/4.13%**.
Every paired repetition shows a CPU saving at these five settings. Later
rejection is zero for both explicit-read backends, and ring's mean later p95/p99
is slightly lower. This is evidence for an explicit option on this workload,
not a universal backend ranking.

At MaxScore 1,200/s, pool completes **1,169.5/s** versus ring's **1,151.1/s**;
ring rejects **3.15%** of later arrivals versus **1.65%** for pool, and its later
p99 is **8.37 versus 7.92 ms**. CPU/completion is effectively tied (4.469 versus
4.475 ms), with mixed directions across repetitions. The overload result does
not support a ring efficiency claim.

Mmap uses only **2.666 ms CPU/completion** at MaxScore 1,200/s, with zero later
rejection and a 4.36-ms later p99. Its cold first-second rejection is **22.25%**,
versus pool's 8.36% and ring's 10.14%. Its mean peak cgroup memory is 217.8 MiB,
versus 111.8/109.8 MiB for pool/ring. Warm-only throughput would hide this
cold-start and memory tradeoff.

For mixed queries, mmap reaches the 256-MiB cap and rejects **82.67–94.40%** of
later arrivals. It reads 2.0–2.5 GiB per cell and records 461,433–596,915 file
refaults. Pool/ring read about 140 MiB, record zero file refaults, and keep every
cell below 198 MiB. This supports memory-pressure thrashing as the mapped
control's failure mode. It does not establish a general 15× engine or ring
speedup: rejected queries alter the surviving workload, and BMP scoring remains
mapped for all backends. Pool/ring also reject **17–27%** of first-second mixed
offers, which the table retains. Larger cold working sets, merge interference,
long-duration saturation and ARM ring throughput remain unmeasured.

## Review changes

Two issues were fixed. `PrefixQuery` previously turned unknown, unindexed and
non-text fields into empty results. Its scorer, estimate, exact count and optional
planner bitset now share field admission with wildcard and regex queries. The
regression first failed in direct count collection; extending it to the optional
bitset reproduced a second route through which the planner could hide the error.
Invalid optional predicates now decline the shortcut so ordinary execution can
report the error. Native sync and async paths are covered.

The two new AVX2 posting kernels declared both AVX2 and POPCNT requirements, but
runtime dispatch checked only AVX2. Dispatch now checks both before entering
either unsafe kernel; other hosts retain the existing portable implementation.
This fixes the feature precondition even on CPUs or virtual machines that expose
an unusual feature combination. We did not emulate an AVX2-without-POPCNT CPU.

Pattern queries now have one async expansion helper used by ranked execution,
estimates and exact counts, plus a sync helper over the same validation and
dictionary plan. This removes repeated matcher instantiations and keeps admission
at one owner. No writer, cache, decoder, scorer, storage format or configuration
default was added.

| Reviewed boundary                              | Ownership and invariant checked                                                                                                                                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pattern query → segment expansion → term union | Query proves dictionary ranges; reader admits bounded expansions; union owns deduplication and physical/logical ID handling. Count shortcuts cannot bypass deletions, collector capabilities or expansion errors.                                                 |
| Word-pair writer → phrase execution            | Pair encoding stays in the shared writer; exact phrase execution reuses term scoring with phrase statistics and frequency semantics. Existing tests compare pair/position counts and raw scores, including merge, deletion and RGB cases.                         |
| Ranked conjunction/disjunction execution       | Block probes and window bounds stay in the shared executor/cursors. Pruning uses the shared relative floating-point margin and preserves equal-score candidates; the full corpus audit compares optimized results with exhaustive counts and retained raw scores. |
| Posting codec → copying merge                  | Canonical readers own bitmap and compact blocks. Compatible payload copying and remapping remain separate from explicit integrity admission. Normal search does not gain an extra payload scan.                                                                   |
| Dictionary decode → process cache              | One bounded cache owns decoded blocks and suffix filters, including replacement charges and namespace cleanup. Corpus payloads stay evictable.                                                                                                                    |
| Payload service → pool/ring → returned views   | Shared admission bounds jobs and bytes; backends retain submitted I/O ownership through cancellation. Idle buffers and returned views have separate lifetimes. Deletion drains accepted leases.                                                                   |
| RPC admission → search pool → hydration        | Search work keeps its admission permit after caller cancellation; hydration delegates to the canonical bounded batch reader. Frontends translate requests without a second search implementation.                                                                 |

This was a targeted review of these changed boundaries, supported by the full
harness and corpus audits. It is not a proof that every repository subsystem is
free of defects.

## Measurement protocol

The full matrix has 19 families × TOP_10/TOP_100/COUNT, with default, word-pairs,
and RGB + word-pairs indexes plus Luxir 0.1.0. All three Summa fixtures contain the same 10M
source documents as the earlier campaign; their file hashes are checked before
and after. All three fresh audits must exactly match their own retained counts,
plans, ranked IDs and raw score bits. Every phase also validates HTTP counts and
top-10/100 IDs against that audit.

GCP Cascade Lake, 32 vCPUs: 30 server CPUs, two driver CPUs, 32 clients,
query cache off, isolated loopback. Rust 1.98.1, release, `-C target-cpu=native`.
Phases D1 P1 G1 L1 L2 G2 P2 D2; fresh server per phase; 40-second session warmup;
three 3-second repetitions with one second of per-cell warmup. QPS is the mean
of the two phase medians. Raw repetition ranges, CPU, anonymous RSS, mapped
residency, and individual observed latency quantiles are retained. CPU/request
is total measured server CPU divided by completed requests across all six repetitions. A range of
six observed p99 values is not a pooled campaign p99 or a confidence interval.

This is the existing **shared-input** comparison: 677 expressions execute on
both engines and 149 retain explicit Summa budget exclusions. Of the admitted
expressions, 640 have equal retained counts across engines and 37 differ. It does not assert
identical cross-engine counts, ranking, analyzers or index layouts. The HTTP
frontend is a benchmark adapter over core search, not the production gRPC server.
The paired pattern control separately measures the review's source change and
the earlier broad-wildcard regression on the same boot and RGB + pairs fixture.

The matched control uses six families: prefix, regex, wildcard, broad wildcard,
high-frequency terms and high/medium conjunctions (154 admitted expressions).
Four arms run in forward and reverse order: the earlier pre-count binary,
the retained count binary, this review's baseline with the normal diagnostic
command, and the reviewed binary. The two middle arms have identical production
Rust sources; their diagnostic command, test source and build path differ.
All arms use the same RGB + pairs fixture, CPU allocation, replay settings and
HTTP correctness checks. Binary hashes and source differences are retained in
[source provenance](source.json). The control does not by itself identify an
instruction-cache or compiler cause for any binary-to-binary difference.

The I/O workload retains the 262,144-document, two-segment MaxScore/BMP fixture
and exact ID/score/document oracle. It offers fixed arrivals for eight seconds,
limits admission to eight tasks, uses four runtime workers on CPUs 0–7, and
hydrates 32 documents per accepted query. Three orders per backend/rate produce
54 load cells plus three smoke cells. Each cell starts in a fresh 256-MiB cgroup
with swap and metadata pinning disabled and verifies zero sparse/store residency.
CPU includes oracle checking; latency excludes rejected offers, which are
reported separately. BMP scoring remains mapped. These short in-process runs do
not establish general production saturation capacity or justify default changes.

## Validation

Native `full` passes 2,165 tests, including five real-server RPC tests, with 25
ignored tests. The final `check` after tightening SIMD dispatch passes 2,160 tests
with the same 25 ignored tests. Both include strict Clippy, formatting and the
required feature checks. The WASM release build and all 41 tests pass.

The first Linux harness launch could not capture provenance because the source
snapshot had no Git repository. The snapshot was initialized locally on the
validation VM and the harness rerun; no workspace history was changed. This was
an environment/setup failure before tests ran, retained in the evidence.

Linux `io-uring` validation passes all 73 tests with no ignored tests; the
standalone arrival scheduler passes four tests and strict Clippy. A separate
x86 run passes all 75 SIMD tests and the prefix-field regression. All 800
source files in the build snapshot match the reviewed workspace manifest.
[Validation evidence](validation.json) records both successful runs and the
initial setup failure.

The ARM diagnostic controls preserve all 59 counts and top-100 ID/score
sequences on ordinary and RGB 1M-document indexes, in both attempts. Timing is
**inconclusive**: the first attempt's ordinary-index baseline pass drifts sharply,
and the second attempt's RGB candidate pass drifts sharply; other passes on the
same binaries do not reproduce those changes. The shared desktop also had
substantial concurrent application load. We retain the [initial summary](arm-initial-summary.json)
and [repeated measurements](arm-results.json), but make no ARM speedup or
regression claim. An isolated ARM throughput campaign remains unrun.

## Coverage and limits

“Complete” here means the retained 57-cell Searchbench query matrix, the matched
18-cell binary control, and every cell of the retained fixed-arrival payload
comparison. This run does not rebuild the 10M-document indexes or remeasure
indexing/merge throughput, ANN recall/training, distributed broker throughput,
or production gRPC saturation. Their applicable correctness/lifecycle tests run
in the harness, but that is not performance evidence. The ARM diagnostic replay
ran twice and remains inconclusive; isolated ARM throughput is unrun.

The correctness oracle is exhaustive count plus retained same-index ranked IDs
and raw score bits, with HTTP count/ID validation before timed phases. It does
not establish cross-engine ranking equivalence. Budget exclusions remain
observable and are not counted as successful empty queries. Raw evidence is
kept under `.context/review-all-20261002/`; archive and binary hashes are in
[source provenance](source.json). Exporters beside this report validate the
artifact bundles before producing tables.

## Re-exporting the evidence

The exporters reuse the repository's existing metric/percentile owners and
reject incomplete runs, unexpected binaries, changed fixtures, admission or
replay mismatches, errors and invalid timing/resource accounting. With the
retained bundles extracted under the paths used here:

```sh
python3 docs/benchmark-results/review-2026-10-02/summarize.py \
  .context/review-all-20261002/evidence \
  .context/union-count/campaign7-artifacts
python3 docs/benchmark-results/review-2026-10-02/summarize_control.py \
  .context/review-all-20261002/evidence
python3 docs/benchmark-results/review-2026-10-02/summarize_io.py \
  .context/review-all-20261002/evidence/io
```

Both cloud VMs were explicitly stopped and independently confirmed terminated;
the scratch filesystem was unmounted and the temporary host namespace setting
was restored. The stop client reported a connection reset, but follow-up API
reads confirmed shutdown. See the [final cloud state](cloud-final.json).
