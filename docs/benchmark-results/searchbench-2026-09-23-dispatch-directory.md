# Searchbench throughput: dispatch, bounded ID lookup, and owned envelopes

Coverage remains **15/826 queries**: seven conjunctions, seven low-frequency
phrases, and one medium-frequency phrase (`"references reflist"`). This is a
restricted compatibility comparison, not the complete published benchmark.

All variants use the same ordinary, non-RGB, 10M-document index, one segment,
compiler (Rust 1.98.1), `-C target-cpu=native`, and 32-vCPU Cascade Lake benchmark
host. Thirty server hardware threads cover 15 physical cores; the driver uses
the remaining physical core (CPUs 15,31). Query caches are off. Summa uses the
benchmark HTTP frontend over core, not the production gRPC server. Luxir is the
same official 0.1.0 x86-64-v4 executable used in the preceding comparison.

The final runs retain 30 search/blocking workers, four HTTP workers, and
64-request admission. Each cell has three ten-second repetitions after a
30-second session warmup and untimed response validation. A separate short
screen uses three five-second repetitions, ten-second session warmup, and both
one and 32 clients. Connection warmup and native Searchbench driver behavior
are unchanged. No indexing, compilation or binary transfers overlap timing.
The final comparison measures both dispatch choices across all 15 queries.

The leading tables below use the **corrected final executable**, which moves
query/class strings with borrowed `get_mut` lookups. An allocation-counting
regression test proves that valid envelope conversion performs zero allocations.
The initial prototype moved those buffers with mutable JSON indexing, which
silently allocated two temporary key strings. The test first reproduced those
two allocations, then passed after the correction. Earlier prototype timings are
retained separately; they are not evidence of removing two allocations.

The final variants are:

- **Before:** the preceding borrowed-ID frontend, rebuilt with identical SHA-256
  to its earlier frozen executable.
- **Directory / blocking:** allocation-free envelope conversion plus at most 256
  sparse codec checkpoints per fast-field reader. Existing numeric/text accessors
  use the shared decoder; no benchmark-only public core API, new persisted format
  or growing per-query cache is introduced.
- **In-place:** the identical corrected executable with the explicit
  `serve INDEX PORT 30 4 in-place` override. The default remains `blocking`.
  Tokio can still transfer its runtime core through its blocking pool.

The initial screen separates the ownership prototype from the directory, then
compares both dispatch modes. Its strings variant also includes dispatch-option
plumbing, so it does not isolate allocation effects from that plumbing. The
reader's CPU cost is additionally checked in stage probes and a separate ARM
kernel benchmark. These establish local costs, not standalone HTTP gains for
each change.

A four-instance prototype ABBA repeat and the corrected all-15 campaign run
after a machine restart. Both use the same CPU model, executable hashes and index;
only paired comparisons within each boot are interpreted. The first boot's
absolute QPS is not used as the corrected campaign's baseline.

Equal counts do not establish general analyzer or relevance equivalence across
engines. Within Summa, all admitted queries preserve exact ranked IDs, score
bits, counts, and all 45 HTTP response bodies. Repetition ranges, CPU cost and
memory are retained in the accompanying CSVs. These runs do not establish p99
latency or cold-cache behavior.

**Throughput, queries per second; higher is better.** Each entry is the median
of three repetitions at 32 clients.

| Family       | Operation | Summa before (QPS) | Summa directory, blocking (QPS) | Summa directory, in-place (QPS) | Luxir (QPS) |
| ------------ | --------- | ------------------ | ------------------------------- | ------------------------------- | ----------- |
| and_high_low | TOP_10    | 43,106.1           | 43,681.9                        | 43,024.0                        | 52,523.8    |
| and_high_low | TOP_100   | 37,927.2           | 39,493.5                        | 38,488.2                        | 38,393.7    |
| and_high_low | COUNT     | 67,728.5           | 67,761.5                        | 69,054.1                        | 64,576.9    |
| low_phrase   | TOP_10    | 15,708.3           | 15,655.4                        | 16,201.5                        | 7,448.5     |
| low_phrase   | TOP_100   | 4,956.4            | 4,925.2                         | 5,033.1                         | 3,779.1     |
| low_phrase   | COUNT     | 1,990.4            | 1,952.0                         | 1,970.9                         | 2,050.9     |
| med_phrase   | TOP_10    | 15,335.9           | 15,284.6                        | 16,019.1                        | 17,552.3    |
| med_phrase   | TOP_100   | 10,205.8           | 10,319.5                        | 10,769.1                        | 5,982.8     |
| med_phrase   | COUNT     | 716.4              | 713.5                           | 717.8                           | 605.5       |

## Result and decision

The corrected blocking configuration changes conjunction top-100 throughput by
**+4.1%** versus its same-session baseline
(37,927 → 39,493 QPS). Its CPU cost changes
-3.8% (658.8 →
633.7 CPU µs/request). Conjunction top-10 changes
+1.3%,
and exact count changes +0.0%.
Count does not project IDs, so its movement cannot be attributed to faster ID
projection.

In-place dispatch changes conjunction top-100 throughput by
-2.5% versus updated blocking execution,
while CPU cost changes +7.8%.
It is an explicit experiment, not a new default. Medium-phrase top-10 changes
+4.8%
and top-100 changes
+4.4%
relative to updated blocking execution, but this family contains only one query.
The tradeoff does not establish a generally better scheduling policy.

Updated Summa is **16.8% behind fresh Luxir on conjunction top-10** and
**2.9% ahead on top-100** in this run. This does not establish a general lead:
Luxir top-100 itself varies from 41,581 QPS in the first prototype campaign to
38,394 in the corrected campaign. The bounded directory is retained for its demonstrated
lookup/CPU efficiency, and envelope conversion now avoids two allocations;
neither result implies a universal QPS improvement. Blocking dispatch and the
CPU-derived HTTP-worker default remain unchanged.

For conjunction top-100, updated Summa consumes 25.01 busy
CPU equivalents, compared with 24.99 before and
28.64 for Luxir. A CPU equivalent is process CPU seconds divided
by elapsed seconds, not a physical-core count. Lower work per request does not
by itself fill the remaining execution capacity. The outer in-place experiment
still incurs runtime-core transfers and the existing Searcher handoff; it does
not demonstrate that all dispatch costs have been removed.

## CPU and memory

CPU cost is the median of per-repetition process CPU seconds divided by requests.
Process RSS includes mapped pages; anonymous RSS is not a complete allocator
measurement. The checkpoint payload itself is bounded independently of the
corpus size.

| Family       | Operation | Before CPU µs/request | Directory CPU µs/request | In-place CPU µs/request | Luxir CPU µs/request |
| ------------ | --------- | --------------------- | ------------------------ | ----------------------- | -------------------- |
| and_high_low | TOP_10    | 578.8                 | 571.7                    | 612.4                   | 553.0                |
| and_high_low | TOP_100   | 658.8                 | 633.7                    | 683.3                   | 745.9                |
| and_high_low | COUNT     | 339.3                 | 339.3                    | 361.2                   | 442.7                |
| low_phrase   | TOP_10    | 1,676.7               | 1,680.9                  | 1,669.8                 | 3,997.1              |
| low_phrase   | TOP_100   | 5,477.0               | 5,532.1                  | 5,455.3                 | 7,911.7              |
| low_phrase   | COUNT     | 14,685.1              | 14,953.4                 | 14,855.2                | 14,581.8             |
| med_phrase   | TOP_10    | 1,671.3               | 1,678.2                  | 1,668.4                 | 1,691.4              |
| med_phrase   | TOP_100   | 2,526.7               | 2,507.0                  | 2,494.7                 | 4,975.1              |
| med_phrase   | COUNT     | 41,462.7              | 41,630.8                 | 40,970.6                | 49,317.6             |

| Variant   | Peak process RSS (MiB) | Peak anonymous RSS (MiB) |
| --------- | ---------------------- | ------------------------ |
| baseline  | 1,150.9                | 109.2                    |
| directory | 1,150.6                | 109.1                    |
| inplace   | 1,151.1                | 109.7                    |
| luxir     | 159.0                  | 12.4                     |

## Prototype screen and repeatability

The following earlier screen uses the ownership prototype that still allocated
temporary JSON keys. It separates the codec-directory effect from that prototype
and compares dispatch modes, but does not prove a two-allocation reduction.
Single-client results are unstable: baseline top-100 moves from 1,504 to 1,902 QPS
between the two controls. Do not infer a precise single-client latency gain.

| Screen variant  | Clients | Conjunction TOP_10 (QPS) | Conjunction TOP_100 (QPS) | Conjunction COUNT (QPS) |
| --------------- | ------- | ------------------------ | ------------------------- | ----------------------- |
| baseline        | 1       | 2,132.0                  | 1,504.1                   | 2,280.5                 |
| baseline        | 32      | 44,105.4                 | 38,817.2                  | 66,537.6                |
| strings         | 1       | 1,911.9                  | 1,666.5                   | 1,968.8                 |
| strings         | 32      | 43,551.8                 | 37,830.2                  | 67,376.7                |
| directory       | 1       | 2,134.6                  | 1,649.6                   | 2,654.8                 |
| directory       | 32      | 44,472.7                 | 40,446.7                  | 67,417.6                |
| inplace         | 1       | 2,358.3                  | 2,115.6                   | 2,692.9                 |
| inplace         | 32      | 43,048.2                 | 38,534.3                  | 68,745.9                |
| baseline-repeat | 1       | 2,081.2                  | 1,901.7                   | 2,296.1                 |
| baseline-repeat | 32      | 44,421.1                 | 38,415.8                  | 64,486.6                |

The first all-15 prototype run recorded conjunction top-10 2.8% below baseline
and essentially unchanged top-100 throughput. Because its short screen did not
reproduce that top-10 loss, a separate-boot ABBA repeat uses four fresh servers
with the unchanged prototype binaries, 32 clients, 15-second warmup and three
ten-second repetitions per operation. Absolute QPS is not compared across boots.

| Prototype ABBA instance | TOP_10 (QPS) | TOP_100 (QPS) | COUNT (QPS) |
| ----------------------- | -----------: | ------------: | ----------: |
| before-1                |     44,020.4 |      38,321.0 |    68,215.4 |
| after-1                 |     44,587.1 |      40,022.4 |    65,000.7 |
| after-2                 |     44,066.9 |      39,850.1 |    65,538.7 |
| before-2                |     44,258.9 |      38,707.3 |    64,762.8 |

Averages of the two session medians change top-10 by
+0.4% and top-100 by
+3.7%.
The earlier top-10 loss is not reproduced. Top-100 CPU cost changes
-3.3%.
Count controls drift substantially, so the first run's count gain is not a
repeatable attribution. All prototype measurements remain in the supplemental
CSVs; the leading table uses the subsequently corrected executable.

## ID projection probes

Four sequential prototype stage probes run in baseline/directory/directory/
baseline order, outside HTTP timing. Each query/limit has 20 warmups and 100
samples. For conjunction top-100, the median across the seven per-query medians
is 33.401 and 32.347 µs before, versus 18.773 and 18.720 µs with the directory:
about **43% less projection time**. The lookup implementation is unchanged in
the corrected executable. These are isolated stage measurements, not 32-client
response latency or a promise of a matching throughput increase. Other phrase
stage probes vary more and all raw observations are retained.

## Correctness, memory bounds, and validation

The directory owns at most 256 three-u32 checkpoints: **3 KiB of heap payload per
reader**, plus its small inline descriptor. This is a bound across all merged
source blocks in that reader, not 3 KiB per block or an entry per document. It
requires a second header walk at open after the existing validation. Encoded
values and dictionaries retain their original byte owner. Multivalue decoding,
missing values, local/global ordinal semantics and persisted bytes are unchanged.

Fast-field block metadata and checkpoints now contribute to segment estimated
heap totals. Existing lazy text-dictionary tables and ordinal maps remain outside
that estimate; process anonymous RSS is reported separately and is not identical
to allocated heap. No pinning setting or payload residency policy changes.

New tests cover merged numeric blocks, sparse/tail boundaries, bounded metadata,
missing values, local text dictionaries, duplicate/rank-ordered access, and
unchanged encoded bytes. Envelope tests preserve validation order and limits and
prove that both string buffers move without copying and that successful envelope
conversion performs zero allocations. The latter test caught the prototype’s
two temporary JSON-key allocations before the final build. Dispatch tests exercise
64 submitted requests, panic cleanup, cancellation retaining reader/permit
ownership, and runtime shutdown waiting for started work. Real HTTP smoke tests
pass for both modes, ordinary/RGB indexes, one HTTP worker, successful/error
responses, concurrent searches, and shutdown.

Validation:

- `python3 scripts/check_search.py full`: all nine stages passed, including
  2,031 native tests (25 normally ignored), native without sync, portable core, API docs and real-server
  broker end-to-end tests (`20260923T170615.743556Z-full`).
- Final `python3 scripts/check_search.py check`: all five stages passed
  (`20260923T180425.771100Z-check`). One preceding run hit a ten-second mock-broker
  discovery timeout in `ambiguous_index_reads_are_deterministic_and_writes_refused`.
  The focused retry and full check retry passed without code changes; the failed
  log is retained (`20260923T171723.416019Z-check`).
- Final WASM release build and all 39 JavaScript tests passed after the metadata
  accounting change; the 39 tests also passed after a fresh `npm ci`.
  Diagnostic-feature Clippy and all seven example tests passed;
  four real HTTP smoke configurations passed.
- Documentation/contracts and diff checks passed. No default thread-count,
  search-pool ownership, persisted format or production RPC schema change is introduced.

## Separate ARM decoder experiment

Two local Apple M4 Criterion passes compare the existing scalar codec header walk
with the public reader's checkpoint lookup on identical encoded values. Each
sample performs 100 deterministic sparse probes; fixture construction is outside
timing. Rust 1.98.1, normal local release flags, one-second warmup, thirty samples
and three-second measurement windows. The raw scalar reference excludes reader
accessor overhead, so these are kernel comparisons, not end-to-end before/after
measurements or HTTP speedups.

| Values in column                  | Scalar reference, pass 1 / 2 (µs per 100 probes) | Reader, pass 1 / 2 (µs per 100 probes) |
| --------------------------------- | -----------------------------------------------: | -------------------------------------: |
| 512 (small non-blockwise control) |                                    0.205 / 0.205 |                          0.282 / 0.282 |
| 65,536                            |                                    9.742 / 9.676 |                          1.876 / 1.837 |
| 1,048,576                         |                                  136.25 / 108.15 |                          3.349 / 3.317 |

The large scalar case is noisy (first-pass estimate interval 114–161 µs), but
both passes establish the expected reduction in repeated header work. The small
control includes the reader accessor overhead and does not isolate a regression
from the change. This synthetic fixture does not justify changing scheduling
defaults. Reproduce with:

```sh
cargo bench --locked -p summa-core --bench core_structures -- \
  fast_field_sparse_lookup --warm-up-time 1 --measurement-time 3 --sample-size 30
```

Raw Criterion estimates, samples, logs and environment metadata are retained with
the campaign evidence.

## Artifacts and lifecycle

All **114 cells / 342 repetitions** pass their error and memory-sampling checks.
Fifteen Summa server instances preserve all 45 response bodies, and all saved
exhaustive IDs/score-bits/count audits agree. Coverage remains 15/826 queries.
The separate Luxir validation/control probe is excluded from throughput tables;
its four-driver-thread CPU assignment can overlap server CPUs, unlike the timed
runs' reserved driver core.

The first campaign archive was downloaded and verified before its automatic
shutdown. The extra repeat was initially attempted while that shutdown was
already in flight; both machines were independently confirmed stopped, then only the
benchmark machine restarted. The corrected adapter required a short build-machine restart.
A corrected-run output-path collision failed before any timing began; the output
directory was renamed and the clean run completed. Those logs remain in the
archive. No timed sample from an aborted launch is included.

Final combined evidence: SHA-256
`6d9dffed84a78f3f3ad999dd930e70234729e699866bb5022e23371bbea35818` (1,828,573 bytes).
The separate source/build artifacts and the first campaign archive are also
retained under `.context/yonik-benchmark/dispatch-directory/`, together with raw
Criterion data, allocation-test failure/success, validation logs and cloud-state
receipts. Host restrictions are restored. **Both machines are confirmed
`TERMINATED` after verified collection.** Nothing is committed or pushed.

- [Corrected comparison CSV](searchbench-2026-09-23-dispatch-directory.csv)
- [Prototype screen CSV](searchbench-2026-09-23-dispatch-directory-screen.csv)
- [Initial prototype comparison CSV](searchbench-2026-09-23-dispatch-directory-prototype.csv)
- [Prototype ABBA repeat CSV](searchbench-2026-09-23-dispatch-directory-confirm.csv)

| Executable     | SHA-256                                                            |
| -------------- | ------------------------------------------------------------------ |
| next-baseline  | `61f485a0ac1964389f4ec01b87b855fc8155911972f0c9f1b339d402b241c778` |
| next-strings   | `d9c17341d9ea62040442b89635892e2aa316ba9c3598c67d2ccdd2cde4bb6480` |
| next-directory | `5bd4306038cceebb58b76e0b38cef911603d3a7ed2d006cb5321bdb8b7ca3769` |
| next-corrected | `edaeb00b413eef381b972ac017372402d40619e5ddedf9cda9af4fa783de2f09` |
