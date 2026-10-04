# Posting intersection follow-up — October 4, 2026

This continues the [October 3 opportunities review](../posting-opportunities-2026-10-03/README.md).
It reduces the rejected BSR corpus failure, repeats the OR/phrase regression
screen, tests a pruned-specialization control, and measures later-clause batching
on a dedicated ARM host. The final x86 campaign compares current `main`, the
AVX2 dispatcher, corrected batching, and corrected batching with transient BSR
in the same run. [The complete table](table.md) includes every family, paired
rounds and process resources. [Machine-readable results](results.json) retain
summary rows; `individual.jsonl.gz` contains every per-query comparison.

## Selected result and BSR decision

Retain the AVX2 balanced/galloping dispatcher, corrected density-gated later-clause
batching on x86-64/aarch64, and the resume-before-compaction invariant. Reject the
corrected BSR candidate on integrated performance grounds: it passes every audit
but lowers targeted multi-term latency only 0.9–2.1%, leaves ranked-suite latency
between 0.04% better and 0.38% worse, and regresses the four boundary controls by
8.1–9.3%. Its additional conversion and dispatch code is not retained. Restoring
the unreachable pruned body is also rejected.

The final x86 campaign uses Cascade Lake, generic Rust 1.98.1 release binaries,
CPU 2, both 10M-document layouts, and `head/D/G/BSR/BSR/G/D/head` phase order.
Each query has five warmups and seven recorded samples per round at top-10,
top-100 and count-only. `head` is main commit
`8f9c3f667928216a52037d16cfa906c593ef0c7a`; D is the dispatcher, G adds corrected
batching. This directly measures the combined change against main.

| x86 layout | Limit | Main suite total ms | Selected total ms | Suite latency reduction | 26 targeted queries reduction |
| ---------- | ----: | ------------------: | ----------------: | ----------------------: | ----------------------------: |
| Default    |    10 |            3962.575 |          3445.757 |                   13.0% |                         15.1% |
| Default    |   100 |            5205.168 |          4680.698 |                   10.1% |                         15.2% |
| RGB        |    10 |            3453.091 |          2822.990 |                   18.2% |                         18.8% |
| RGB        |   100 |            3829.435 |          3171.356 |                   17.2% |                         19.5% |

Ranked process CPU falls 9.8–17.7%; mean peak RSS differs by less than 0.3 MiB.
Count-only totals move 1.4–2.9% faster, but the count algorithm is unchanged and
this is not credited as an optimization. Both index inventories match before
and after, and the ARM fixture inventories match these x86 fixtures byte for byte.

The selected path still has tradeoffs. Against main, default medium-frequency OR
top-100 is 4.7% slower (paired ratios 0.974/0.935). Default high-frequency phrase
top-10 is 6.3% slower, with widely separated paired ratios 1.046/0.856. Default
high/low-frequency OR top-100 is 11.3% slower, with paired ratios 0.815/1.005.
Those unstable pairs prevent assigning all pooled movement to the code, but do
not justify discarding the measured regressions. The full table includes them;
this change is not a universal latency improvement or a production traffic estimate.

## Correctness finding

The BSR audit failure was a caller contract violation in the unmerged batching
prototype; baseline `main` uses scalar later-clause filtering. Document 4,502,871 really
matches `+the +of +in +and`, but its first two frequencies became 13 and 8 instead
of 3 and 4. Its inflated score displaced another top-100 document.

The balanced four-lane kernel can return a conservative candidate resume position
before its last emitted pair. In-place candidate compaction then overwrote that
retained prefix with duplicate/decreasing IDs. A later call no longer received a
strictly increasing suffix. BSR recovered ordinals by counting state-mask bits,
which made that invariant violation visible as incorrect frequency attribution.
This is separate from the previously fixed BSR terminal-group tail loss.

The owning conjunction caller now consumes candidates through the last emitted
pair before compacting. Cost: one index maximum per nonempty intersection, no
allocation or extra decode. The primitive explicitly documents conservative
resume positions; a debug assertion guards the caller's remaining suffix.

The regression reduces to 259 documents and three terms, two with identical rare
postings. Seeking the common term leaves 28 postings in its first block; a
four-lane match can emit through candidate slot 26 while retaining slot 24.
The test checks all 127 hits, exact score bits and counts in sync, async and counted
execution, with rounded and bitmap codecs. On actual x86 the old caller fails
the strict-suffix assertion and the fixed caller passes. A portable model of the
x86 balanced kernel plus BSR also reproduces wrong IDs/frequencies before the fix;
afterward it passes the reduced case and 160 random/clustered corpora. Model
results are supporting evidence, not a replacement for native corpus audits.

## OR and phrase investigation

The original 3–6% family regressions did not reproduce consistently in a focused
42-query ABBA repeat (21 recorded samples and five warmups per query). Default
OR top-100 remains approximately 1% slower; the paired rounds straddle parity.
Restoring the old, unreachable multi-term body in the two-term pruned
specialization changes most totals by less than 1% and does not consistently
recover OR performance. That control is rejected; no dead body is retained.

Normalized instruction sequences match between the original dispatcher and
batching binaries for three phrase candidate-finding functions, posting
intersection, and both window executors. This comparison removes addresses and
RIP displacements; it does not prove identical relocated data or indirect callees.
CPU-clock profiles place OR time in window scoring/length gathering/seek/decode
and phrase time in frequency-bound candidate scans and positional scoring.
The pruned conjunction was not a sampled hot path. Hardware performance counters
were unavailable, so no cache or branch-miss explanation is claimed. Profiles
include startup; worker-filtered reports are retained separately. Profiled timings
are excluded from latency tables.

## Dedicated ARM result

An Ampere Altra (Neoverse N1) `t2a-standard-8` VM runs both immutable 10M-document
indexes from a read-only disk clone. Rust 1.98.1, generic release flags, one worker
pinned to CPU 2, ABBA order, seven recorded samples plus five warmups, top-10,
top-100 and count-only. This removes the shared-Mac contention that made the
October 3 ARM confirmation inconclusive.

Later-clause batching lowers latency for the 26 targeted multi-term queries by
18.2–21.2%, and the 199-query ranked suite by 12.2–19.7%. Process CPU improves
12.0–19.1%; mean peak RSS differs by less than 0.1 MiB for ranked runs. Count-only
moves −0.3% to −0.6% in the ratio and is not an algorithmic gain. Some OR/phrase
families regress by up to 3.6%; both rounds and individual cases are preserved.
The final architecture gate includes x86-64 and aarch64; WASM retains cursor seeks.
The ARM candidate uses the same effective code with a compile-time-always-enabled
experimental gate; archived source files distinguish it from the final gate.

## Validation

The four completed follow-up campaigns validate **2,598 exhaustive audits** and
**121,464 recorded query executions**, excluding warmups. All recorded query
IDs, score bits, counts and plans match; all index inventories remain unchanged.
The complete audit outputs also match exactly across x86 and ARM.

The selected batching source passes `python3 scripts/check_search.py check`:
2,170 tests (25 ignored), strict focused Clippy, native without sync, and the
standalone broker build. Workspace-wide all-targets Clippy also passes. The WASM release build and all 41 JavaScript tests pass.
Linux independently passes 1,913 core library tests (15 ignored) and strict
release Clippy for both the corrected batching and corrected BSR variants.
Extended lifecycle/RPC checks were not rerun because those boundaries are unchanged.

A final-source x86 rebuild after extending the architecture gate to aarch64 has
an identical `.text` section to the measured corrected batching executable
(SHA-256 `d0ac04e32ed2e5958dc40db0ed8929e2d0fb40d4b9b04178c12b9a1be6ae816d`).
The complete binaries have different hashes; this assertion concerns machine
instructions, not debug metadata or all ELF sections. Both hashes and source
snapshots are archived.

## Execution notes

The first corrected-binary extraction exhausted the benchmark VM's boot disk.
No partially extracted executable was timed. Only this experiment's output
folder moved to the existing scratch disk; all three binary hashes were verified
before measurements, and index inventories remained unchanged. GCloud reported
connection resets after some resource operations; actual resource state was
checked before retrying. The temporary ARM VM, cloned disk and snapshot were
removed after verified collection. Both original x86 VMs were verified stopped;
their disks were preserved. Hardware PMU limitations are described above.

## Reproduction and limits

Run `python3 docs/benchmark-results/posting-followup-2026-10-04/summarize.py`.
It checks the split evidence archive and each member hash, exhaustive audits,
recorded IDs/score bits/counts/plans, sample counts, and unchanged index inventories
before regenerating tables. Totals pool both rounds' samples per query, take each
query's median, then sum. Earlier working notes averaged per-round totals instead;
the published table consistently uses pooled medians. Ratios are before/after.
Do not multiply improvements from separate campaigns.

The archive preserves executed runners, fixture queries, source variants,
compiler/host metadata, binary hashes, raw timings, resource reports, failure
reproducers and validation logs. Binaries, index payloads and raw perf recordings
are not committed. Process CPU includes startup/warmups; raw calling-thread CPU
samples do not include worker CPU. RSS includes mapped index pages. Added scratch
is one fixed 256-byte pair buffer and there is no new query allocation, persisted
format, public API, scorer, or writer.

These are warm, single-worker search measurements on two layouts of one corpus.
Cold storage, concurrent ingestion/merge, HTTP throughput, production latency
percentiles, other CPU families, full QFilter/persistent BSR, fused decoding,
run/complement/trie formats and variable impact partitions remain unmeasured.
Union and count-only algorithms are unchanged; incidental count timing changes
must not be presented as optimizations.
