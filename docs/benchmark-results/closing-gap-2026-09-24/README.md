# Closing the Searchbench gap: measured checkpoints

**September 25 decision:** the reverse-FST investigation has been discontinued
at the user's direction. FST results below are retained as historical evidence;
they are not a production recommendation or an active implementation plan.

September 24–26, 2026. This is an ongoing investigation, not a claim of parity across all query families. The September 25 full baseline and September 26 targeted follow-ups supersede the older mixed-checkpoint comparison for the current source. Earlier sections retain their original measurement scope; historical gains must not be multiplied into the fresh results. The gap is not yet closed.

## September 26: finite regex alternatives and posting investigation

Bounded finite regex languages now use grouped exact SSTable lookups. The query
retains full overlapping literals such as `http` and `https` before prefix
subsumption; infinite/truncated extraction keeps the existing bounded scan.
The segment expansion owner, block decoder and constant-score union remain
shared. No writer, index format, FST, cache, worker or backend default changes.

The same immutable 10M-document index, Rust 1.98.1 release compiler and
`-C target-cpu=native` flags were used. Build-host CPU model/flags match the
baseline. Old/new Summa follow **A–B–B–A**, with Luxir controls interleaved as
A1–B1–L1–B2–A2–L2. Server/driver affinity, worker counts, admission, result-cache
policy and the pinned replay remain as documented below. Each cell retains three
three-second repetitions after session warmup. No builds, profiles or bulk
artifact transfers overlap timing; two small future-I/O runner updates occurred
during the HTTP campaign.

The 90 cells cover five selected families, not a fresh 19-family campaign:
four admitted regex expressions, 42 broad wildcards, 47 low terms, 50 high/medium
conjunctions and 46 high/low disjunctions. They contain **24,100,592 requests with
zero errors**. All **677 exhaustive count/top-100 ID/score-bit audits** pass for
both finite-regex and posting candidates. Four Summa phases additionally pass
8,124 HTTP checks; two Luxir phases pass 4,062 own-count/result-structure checks.
Both engines receive the same untimed 677-query × three-operation coverage.
The fresh 826-expression probe still has 677 successful counts and 149 visible
budget errors. Equal ranking between engines is not asserted.

Values below are means of two phase-median QPS values under the pinned replay,
not per-query latency gains or uniformly weighted query averages.

| Family        | Operation | Before QPS | After QPS | Luxir QPS | After / before | After / Luxir |
| ------------- | --------- | ---------: | --------: | --------: | -------------: | ------------: |
| regex         | TOP_10    |     17,719 |    70,309 |   118,017 |          3.97× |         0.60× |
| regex         | TOP_100   |     17,064 |    64,975 |   105,731 |          3.81× |         0.61× |
| regex         | COUNT     |      3,518 |     4,834 |    13,628 |          1.37× |         0.35× |
| wildcard_scan | TOP_10    |        307 |       304 |     1,832 |          0.99× |         0.17× |
| and_high_med  | TOP_10    |      4,201 |     4,229 |     9,005 |          1.01× |         0.47× |
| or_high_low   | TOP_10    |     19,073 |    19,016 |    37,524 |          1.00× |         0.51× |
| low_term      | TOP_10    |     67,321 |    66,938 |   116,099 |          0.99× |         0.58× |
| low_term      | TOP_100   |     39,661 |    39,812 |    35,410 |          1.00× |         1.12× |

Regex TOP_10 CPU/request falls from **1,525 to 362 µs**; TOP_100 from 1,601 to
399 µs. Other family/operation aggregate QPS changes range from −1.1% to +0.7%.
Cheap COUNT controls remain limited by the two driver CPUs. The result closes
a substantial part of the regex gap; broad scans and conjunction/OR gaps remain.

Separate instrumented CPU0 diagnostics explain the change:
`(www|http|https)` TOP_10 search drops from **80.87 ms to 12.32 µs**, while parse
time drops from 118.98 to 4.98 µs. The same three postings and one document block
are used. Weekday alternatives change 243.10→19.09 µs and `colou?r`
201.87→11.58 µs. Infinite `[jkqxz][a-z]*ess` stays near 104–105 ms.
These single-query stage timings are distinct from HTTP family throughput.
[Results, audits and diagnostics](finite-regex.json) and
[the exact source manifest](finite-regex-source.json) retain the evidence.

### Posting candidate: rejected

An isolated candidate consumed the matched right-hand prefix after every
posting-block match. Both variants pass all 677 corpus audits and the existing
partial-output/unsigned-ID regression. However, the extracted kernel screen
regresses dense inputs by **3.62× on ARM and 3.04× on x86**; medium inputs regress
1.42×/1.23× and sparse-left inputs 1.49×/1.19×. Sparse-right inputs improve, showing
that orientation matters rather than establishing a general replacement.

The uninstrumented CPU0 six-query A–B–B–A corpus screen confirms 1.5–2.8% slower
conjunction searches, with OR changes below 1%. Each phase has 100 samples after
20 warmups per query/limit. Keep the production kernel unchanged.
[Candidate results](posting-lanes.json), [patch](posting-lanes.patch) and
[extracted kernel harness](posting-lanes-source.rs.txt) preserve the rejected
experiment. No MaxScore or BMP speedup is claimed from this candidate.

### What accounts for the observed RSS

After the same untimed query/operation coverage and five-family timing, the new
Summa processes retain **3,802–3,804 MiB RSS**, including **242–244 MiB anonymous**
memory. Luxir retains 1,166–1,190 MiB RSS and about 57 MiB anonymous memory.
These are end-of-phase mapping snapshots, not peak or per-request allocations.

| Summa mapped component | Resident MiB, new-code phases |
| ---------------------- | ----------------------------: |
| Postings               |                       1,795.5 |
| Fast fields            |                       1,056.6 |
| Positions              |                         440.3 |
| Term dictionary        |                         210.6 |
| Chunk metadata         |                          19.3 |
| Store                  |                          11.0 |
| Row statistics         |                          10.1 |

All snapshots report zero locked pages. Mapped residency is evictable and is not
heap ownership or Rust `Pin`. The finite-regex change lowers observed term-file
residency by 10 MiB, but does not materially solve the overall memory gap.

Source inspection identifies two follow-ups: fast-field opening validates text
dictionary bytes, touching pages despite comments describing lazy dictionary
access; and first text access builds eight bytes of heap offsets per dictionary
entry. The current `fast_field_metadata_heap_bytes` explicitly excludes these
lazy text dictionaries, so canonical segment estimates cannot explain all
anonymous RSS. Preserve corruption rejection and ordinal semantics when improving
this path; do not simply remove validation. No attribution of the entire heap
gap to these offset tables is established. [Mapping totals](retrieval-memory.json)
retain RSS/PSS/anonymous/locked components without private paths.

## Real-corpus payloads under memory pressure

The standalone probe now exercises the unchanged **10M-document index**, using
its canonical store planner, decoder and query parser. A private physical copy
contains 18,693,711,229 bytes, including a 4,455,675,505-byte store. The address
trace is deterministic, not a production traffic recording: 65,536 distinct
addresses distributed across the corpus, replayed twice. A separate control
repeats 47 low-term queries 20 times, taking top 32 results, with and without
document hydration. The query working set is much smaller than the address trace.

The first 42 cells used 1/2/8-GiB trace limits and 1/8-GiB query/retrieval limits.
Explicit reads touched only about 743 MiB of payload pages, so their trace working
set could fit at 1 GiB. A **six-cell 512-MiB follow-up** then forced useful payload
pages to exceed the explicit-reader budget. Both method orders were retained
at every budget. All **48 corrected cells pass**, with **3,506,688 decoded-document
checks**, another **360,960 retrieval address/score-bit checks**, and 120,864
batch/query timing samples. No timed service errors or OOM events occurred.

Each cell uses a fresh memory cgroup with swap disabled and CPUs0–7 enforced.
Private-file advice and mincore show zero resident fixture pages before opening;
no host-wide cache flush is used. Header/metadata reads occur before operation
timing. Four futures share one current-thread runtime; they overlap explicit I/O,
not query CPU. Mmap remains its synchronous control and does not have equal
asynchronous depth. Pool and refill/grouped ring share eight in-flight reads and
eight MiB globally. The application store cache is disabled. Verification occurs
outside measured waves. The runtime kernel is Linux 7.0.0-1011-gcp; the build uses
the same Rust 1.98.1 release compiler and CPU flags as the retrieval comparison.

### Address-trace results

Seconds per complete 131,072-document two-pass sweep, averaged over forward and
reverse method orders. These are standalone execution sweeps, not HTTP QPS.

| Cgroup GiB | mmap seconds | Pool seconds | io_uring seconds | Operation CPU seconds, pool / ring |
| ---------- | -----------: | -----------: | ---------------: | ---------------------------------: |
| 0.5        |        93.53 |        10.05 |            10.78 |                        7.51 / 5.89 |
| 1          |        91.88 |         7.03 |             7.28 |                        6.38 / 4.84 |
| 2          |        83.94 |         6.98 |             7.22 |                        6.24 / 4.73 |
| 8          |        22.00 |         7.00 |             7.31 |                        6.26 / 4.76 |

At **512 MiB**, explicit reads remain much faster than the mmap control, but the
pool beats the ring: 10.05 versus 10.78 seconds. Ring CPU is 5.89 versus 7.51
seconds, **21.6% lower**, while elapsed time is **7.2% higher**. Across the 1/2/8-GiB
cells, ring CPU is 24–25% lower than the pool, while elapsed time is 3.4–4.4%
higher. This establishes a CPU/latency tradeoff, not an io_uring latency win.

Physical reads explain much of the mmap/control difference. At 512 MiB, mmap
reads **16,367 MiB** across both passes versus **1,486 MiB** for pool/ring.
The pool's passes take about 5.06 and 4.99 seconds; both remain cold under
reclaim. At 1 GiB, explicit physical reads fall to 743 MiB, and the pool's second
pass falls to 1.97 seconds. Thus the 1-GiB results alone would not establish
larger-than-working-set pressure for the explicit backends.

At 512 MiB all methods reach the 512-MiB charged-memory peak. Pool/ring record
roughly 174,000–181,000 file-page refaults and 553,000 reclaimed pages per cell;
these cgroup counters include startup. Timed physical reads/faults exclude
startup but include verification. None of these RSS or charge measurements is
an isolated heap-allocation total. At 8 GiB, mmap's **warm second pass is fastest**:
1.62 seconds versus pool 1.97 and ring 1.91. A universal backend switch is not
supported.

The 512-MiB batch median/p95 is 22.72/25.00 ms for mmap, 8.26/9.94 ms for the pool,
and 8.92/10.77 ms for the ring. Each pair covers 8,192 32-document calls across
two orders, starting at first poll; this excludes pre-poll queue wait and is not
a per-document or RPC latency percentile. Ring submissions average **1.44 SQEs**
under 512-MiB pressure and about **2.42–2.45** at larger budgets, with a maximum
of eight. The service refills live requests rather than waiting to force full
batches. Pool/ring reach eight concurrent reads without lost completions.

These gains combine bounded independent reads, targeted ranges and different
page-fault/readahead behavior. They cannot be attributed to the ring API alone;
the pool is the faster elapsed-time control, and mmap cannot overlap faults on
this single-thread diagnostic runtime.

### Smaller query and warm-path controls

At 1 GiB, all 940 query-plus-hydration requests take **2.302 s mmap, 1.577 s pool,
and 1.553 s ring**. At 8 GiB the values are 2.264/1.574/1.531 s.
Ring total time is only 1.5–2.7% below the pool, with about 25% less operation CPU.
These totals mix one cold sweep and 19 mostly warm sweeps of just 47 queries.

The first 1-GiB query sweep favors the pool (0.583 s) over the ring (0.599 s).
Subsequent warm 47-query sweeps favor mmap: **45.3 ms**, versus pool 52.3 and
ring 50.2 ms. Retrieval-only controls remain close, at 0.597–0.612 s under
1 GiB and 0.588–0.592 s under 8 GiB. Payload transport has not accelerated the
retrieval scorer. It cannot explain the separate Luxir HTTP comparison, whose
adapter returns IDs without stored-document hydration.

### Startup failure, repair and scope

The first attempt is preserved and entirely excluded. Pool/ring rejected the
store block-directory read because its contiguous response exceeded the
diagnostic eight-MiB single-read budget. The shared diagnostic adapter now
assembles larger logical reads sequentially through the same service, with a
separate **64-MiB returned-response cap**; in-flight reads remain limited to
eight MiB. Ordinary reads keep the direct path. The extra assembled response is
accounted separately from kernel-owned buffers; no parser or format is copied.

The behavior regression fails before the repair and passes afterward for all
five pool/drain/refill/handoff combinations, covering exact bytes, nonzero offsets,
short later chunks, in-flight future drop, subsequent reuse, response-budget
rejection before I/O, and permit recovery. Linux strict Clippy and formatting
pass. An extracted-source timestamp issue initially reused a stale binary; the
regression caught it, archive extraction now refreshes source mtimes, and the
fresh binary passed before transfer and timing. Successful transfer hashes and
temporary-key cleanup are retained.

[Results and controls](real-payload.json), [real-corpus driver](real-payload-source.rs.txt),
and [shared diagnostic service](real-payload-service-source.rs.txt) retain source
hashes, cgroup statistics, per-pass timings, distributions, CPU, physical reads,
mapping residency and the excluded startup errors. The original index and pinned
replay hashes match the previous campaign; private fixture hashes also match
after both I/O campaigns. No index writer, production Linux dependency or backend
default changes. Both cloud VMs were explicitly stopped and independently
confirmed terminated after artifact collection.

This is one CPU/storage platform, buffered I/O, a declared address trace and a
small query set. Production RPC latency, separate queue/service timing,
concurrent merges, ARM I/O and BMP speedups remain unmeasured. The evidence
supports the shared explicit-read seam and continued pool/ring comparison;
it does not yet justify shipping a default production ring backend.

The finite-regex implementation passes the five-stage search harness
(2,090 native tests, 25 ignored), native without sync, strict Clippy, the WASM
release build and 41 JavaScript tests. Lifecycle/RPC production code did not
change in this pass, so the full nine-stage suite was not rerun. Local ownership,
documentation, export and whitespace checks pass.

## Fresh Summa 2 baseline: same-host ABBA comparison

The current source was rebuilt with Rust 1.98.1, release optimization and
`-C target-cpu=native`. The same immutable 10M-document Unicode-word/impacts
index and the pinned Luxir 0.1.0 x86-64-v4 release ran in
**Summa–Luxir–Luxir–Summa** order on the same 32-logical-CPU host. Each server
received the same 30-CPU affinity; the native replay driver used the other two.
Summa used four HTTP runtime workers, 30 search workers and admission 64;
Luxir retained its release defaults within that affinity. Result caching was
disabled. There were no builds, profiles or artifact transfers during timing.

Fresh probes admit **677 of 826** expressions, with **640 identical counts**;
all admitted count differences are within 5%. The other **149** remain explicit
scan/expansion-budget errors, retained in [the coverage data](retrieval-coverage.json).
All 677 Summa queries pass exhaustive counts and top-100 ID/score-bit audits.
Both Summa deployments additionally pass all **4,062 explicit HTTP count/top-k
checks**. The **228 timing cells** (19 families × three operations × four runs)
contain **47,207,548 requests with zero request errors**.

Each cell has three three-second repetitions after a 40-second session warmup.
The table uses the mean of each engine's two phase medians; CPU/request is
weighted by completed requests. [The full results](retrieval.json) retain all
57 comparisons, individual phases, repetition ranges, CPU and memory.

| Family        | Operation | Summa QPS | Luxir QPS | Summa / Luxir | CPU µs/request, Summa / Luxir |
| ------------- | --------- | --------: | --------: | ------------: | ----------------------------: |
| regex         | TOP_10    |    17,365 |   122,170 |         0.14× |                   1,567 / 215 |
| wildcard_scan | TOP_10    |       304 |     1,753 |         0.17× |               96,153 / 17,020 |
| wildcard      | TOP_10    |    10,453 |    28,446 |         0.37× |                 2,503 / 1,040 |
| and_high_med  | TOP_10    |     3,931 |     8,544 |         0.46× |                 7,048 / 3,482 |
| or_high_low   | TOP_10    |    17,584 |    34,356 |         0.51× |                   1,446 / 811 |
| or_high_high  | COUNT     |     2,264 |     4,144 |         0.55× |                12,770 / 7,086 |
| low_term      | TOP_10    |    77,445 |   114,815 |         0.67× |                     320 / 231 |
| prefix3       | TOP_10    |    79,504 |   117,575 |         0.68× |                     321 / 227 |
| low_term      | TOP_100   |    37,725 |    32,696 |         1.15× |                     659 / 888 |
| med_term      | TOP_100   |    30,175 |    20,756 |         1.45× |                   826 / 1,421 |
| high_term     | TOP_100   |    24,785 |    15,423 |         1.61× |                 1,025 / 1,909 |

Second-phase median changes are −1.0% across Summa cells and −1.2% across Luxir
cells; individual cells move −4.3% to +3.1% and −6.5% to +2.7%, respectively.
The large gaps repeat; small differences should be read against those ranges.
Luxir's cheap term-count cells consume essentially both driver CPUs, so those
rows do not establish its unconstrained server capacity. These short runs do
not establish latency percentiles or equal cross-engine ranking semantics.

Peak process RSS across the measured cells is **3,849 MiB for Summa versus
1,229 MiB for Luxir**; peak anonymous RSS is **280 versus 80 MiB**. These are
observed process peaks with mapped files and different engine layouts, not
isolated per-query allocations. Memory pressure remains a separate measurement
requirement. The adapter returns IDs without stored-document hydration, so the
payload batching and io_uring experiments do not explain these results.

The [source manifest](retrieval-source.json) records all 689 packaged files.
The Summa binary SHA-256 is
`8410fac8ef2d87ff5a9464bd8072011777ce964b039887dbd38e4bfb69903cb7`;
the Luxir binary is
`d848ed884ff43788e8a05596006622c907912c8905f8314f59b08322b4150b89`.
This is a baseline/profile pass, with no production code or default change.

### CPU profiles and the next optimization targets

Collected September 26 from the completed September 25 run. All **ten CPU
captures** (five families × two engines) have zero lost samples, use the same
binaries as the throughput comparison, and aggregate across worker names.
[The profile export](retrieval-profiles.json) retains symbols, sample counts and
raw-data hashes. Each 199-Hz `task-clock` capture mixes warmup, TOP_10, TOP_100,
COUNT and idle intervals over 90 seconds; these are sampled CPU self percentages,
not per-operation wall-time fractions. Profiled throughput is excluded from the
ABBA results. Frame-pointer stacks can be incomplete, and the Luxir release has
no application symbols, so its addresses cannot identify matching algorithms.

| Summa family  | Largest observed CPU owners                                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| regex         | Vint decoding 15.64%; Zstd sequence decoding 14.06%; UTF-8 validation 9.02%; dictionary entry decoding 7.13%                  |
| wildcard_scan | Zstd sequence decoding 32.29%; vint decoding 20.16%; dictionary entry decoding 9.22%                                          |
| and_high_med  | Posting-block intersection 24.21%; 8-bit delta decoding 11.38%; document-window filling 9.46%; text bounds 7.16%              |
| or_high_low   | 8-bit delta decoding 9.81%; text-window execution 7.97%; block seeking 7.70%; text bounds 6.81%                               |
| low_term      | Document-length gathering 11.39%; collector insertion 4.74%; heap replacement 4.27%, alongside substantial scheduling samples |

These measurements narrow the work to existing owners. Regex and wildcard scans
spend much of their CPU decoding dictionary blocks; regex matching alone does
not explain the gap. Conjunctions primarily spend CPU intersecting and decoding
postings. Collector work is visible, but it is not the dominant owner in those
two families. Scheduling samples in the low-term capture include idle intervals;
an active-request-only profile is needed before attributing them to request cost.

[Native diagnostics](retrieval-diagnostics.json) cover 18 selected expressions
at TOP_10 and TOP_100, with 20 warmups and 100 samples each on one CPU. All 18
instrumented exhaustive audits match the uninstrumented results exactly. This
is a selected diagnostic sample, not another throughput comparison:

| Expression                                       | TOP_10 median search time | Postings opened | Document blocks decoded |
| ------------------------------------------------ | ------------------------: | --------------: | ----------------------: |
| `(www\|http\|https)`                             |                  81.64 ms |               3 |                       1 |
| `(mon\|tues\|wednes\|thurs\|fri\|satur\|sun)day` |                  0.244 ms |               7 |                       5 |
| `colou?r`                                        |                  0.201 ms |               2 |                       2 |
| `[jkqxz][a-z]*ess`                               |                 104.57 ms |             387 |                      14 |
| `h*band`                                         |                  41.57 ms |              66 |                       1 |
| `+has +please`                                   |                   3.89 ms |               2 |                  10,933 |

For `(www|http|https)`, parsing is only 0.125 ms while search takes 81.64 ms
and touches one document block. Together with the family profile and the current
prefix-scan implementation, this supports avoiding unrelated dictionary entries
before changing the scorer. For `+has +please`, the diagnostic records 22,376
conjunction candidates, 44,752 exact score units and 78 heap updates: reducing
posting work is a more direct target than another isolated heap rewrite.

The next experiments, in priority order, are:

1. **Finite regex alternatives:** preserve complete exact literals before prefix
   subsumption and use bounded dictionary point lookups, feeding the existing
   constant-score union. Require all extracted literals to be exact and no
   look assertions; keep the existing bounded scan for infinite or truncated
   extraction. Reuse the dictionary loader and expansion accounting in native
   and async execution. Test overlapping alternatives such as `http`/`https`,
   Unicode, budgets, corruption and exact scores. This is a proposal, not a
   measured improvement, and introduces no FST or second scorer.
2. **Conjunction and disjunction posting work:** profile block intersection,
   window refill and bounds on the same audited fixtures. Optimize their owning
   implementations, then rerun the paired comparison. Shared collector changes
   can reach BMP, but these text-specific hot paths do not establish a BMP gain.
3. **General dictionary scans and memory:** measure repeated block decoding,
   cache misses and allocation/residency before changing layout or cache budgets.
   The broad wildcard profile motivates reducing decoded work, not another
   unmeasured vint micro-optimization or a larger cache by default.
4. **Payload I/O under memory pressure:** run the
   [real-corpus pool/ring experiment](../../batched-payload-reads.md#next-experiment-real-corpus-under-memory-pressure)
   through the canonical planner and decoder with equal global budgets. Keep
   retrieval-only controls; this comparison does not exercise hydration.

[The final fixture inventory](retrieval-fixture.json) confirms all ten index files
and the pinned replay sources stayed byte-identical. The locally retained full
artifact archive, including raw profiles and failed-attempt logs, has SHA-256
`d55f90bf521cb92d126ebbb6fa5a099b08e7c9f7f88b6fda90bbea1f5d23b7b4`.
An early capture shutdown and a report-file ownership error were recovered before
export; the incomplete capture is excluded. Export checks require completed
replays, matching binary/raw-data hashes, zero errors and zero lost samples.
Ruff, contracts, documentation links, campaign unit tests and whitespace checks
pass. Production source is unchanged; the preceding full native and WASM checks
were not rerun for this reporting-only completion.

## Workload and controls

The immutable corpus contains 10,000,000 documents (13,579,510,045 input bytes; SHA-256 `b15e60ad32a0e9f09f3be5335db86ccd885e1be1418086a906d68529e6ec1c9d`). Paired Linux runs use the same Intel Xeon 2.80 GHz machine with 32 logical CPUs / 16 physical cores, Rust 1.98.1, release optimization and `-C target-cpu=native`. Servers receive 30 logical CPUs; the driver receives the other two. There are 32 clients, four HTTP workers, 30 search workers, admission 64, and no result cache. Each cell has three three-second repetitions after a 40-second session warmup. Tables show median repetition QPS; JSON retains CPU/request, RSS, repetition ranges, hashes and query counts. Short runs do not establish tail-latency percentiles.

Candidate comparisons require exact within-index exhaustive counts and top-100 IDs/score bits before timing, followed by HTTP count and top-10/top-100 ID verification. Cross-engine tables use identical query text; hit-count differences remain visible and scores are not assumed equivalent. Broad queries that exceed resource limits are reported as failures, never as successful empty results.

## Coverage and analyzer configuration

The opt-in `unicode_word` tokenizer uses Unicode word boundaries and lowercase. Existing analyzer defaults are unchanged. On all 826 expressions, the latest probe succeeds on **677**, of which **640 have exactly the reference count** and all 677 are within 5%. The other **149 fail explicit scan/expansion budgets**. Before this analyzer change, 684 succeeded, only 15 matched exactly, and 619 were within 5%. Thus count agreement improves while successful coverage decreases by seven; this is not an execution-only speedup.

The new ordinary index is 18,693,711,229 bytes. Its build took about 17m40s with peak RSS 28,730,716 KiB; there is no controlled indexing-throughput comparison. Quantized norms are a separate index experiment because they can change ranking. Neither impacts, quantized norms, larger dictionary caches nor RGB are enabled by default by this work.

## Same-index execution changes

The `baseline-plain` and `patterns-plain` binaries use the same original plain index and 684 expressions. This checkpoint combines borrowed L1 metadata, existing single-term block traversal admission, lazy constant-score union, bounded regex prefix ranges and single-star matching. It predates the subsequent owner and inline optimizations.

| Family        | Operation | Queries | Before QPS | After QPS | Ratio | CPU µs/request before → after |
| ------------- | --------- | ------: | ---------: | --------: | ----: | ----------------------------: |
| and_high_high | COUNT     |      47 |      2,201 |     2,260 | 1.03× |           13,139.9 → 12,816.2 |
| and_high_high | TOP_10    |      47 |      1,261 |     1,286 | 1.02× |           23,157.6 → 22,615.9 |
| and_high_high | TOP_100   |      47 |      1,101 |     1,123 | 1.02× |           26,529.4 → 25,948.9 |
| and_high_low  | COUNT     |      50 |     12,694 |    12,872 | 1.01× |             2,143.1 → 2,094.6 |
| and_high_low  | TOP_10    |      50 |      2,987 |     2,952 | 0.99× |             9,577.4 → 9,689.7 |
| and_high_low  | TOP_100   |      50 |      2,949 |     2,927 | 0.99× |             9,735.6 → 9,747.9 |
| and_high_med  | COUNT     |      50 |      4,571 |     4,615 | 1.01× |             6,128.7 → 6,115.5 |
| and_high_med  | TOP_10    |      50 |      2,849 |     2,940 | 1.03× |             9,848.8 → 9,560.4 |
| and_high_med  | TOP_100   |      50 |      2,581 |     2,654 | 1.03× |           10,920.6 → 10,560.5 |
| high_term     | COUNT     |      45 |     80,970 |    82,284 | 1.02× |                   75.3 → 74.5 |
| high_term     | TOP_10    |      45 |      5,480 |     8,764 | 1.60× |             5,057.5 → 3,089.9 |
| high_term     | TOP_100   |      45 |      3,068 |     4,504 | 1.47× |             9,207.7 → 6,120.7 |
| low_term      | COUNT     |      47 |     80,197 |    81,215 | 1.01× |                   77.5 → 77.3 |
| low_term      | TOP_10    |      47 |     32,381 |    56,792 | 1.75× |                 766.0 → 450.6 |
| low_term      | TOP_100   |      47 |     20,199 |    29,923 | 1.48× |               1,230.5 → 829.9 |
| med_term      | COUNT     |      49 |     81,385 |    81,179 | 1.00× |                   75.3 → 77.1 |
| med_term      | TOP_10    |      49 |     18,867 |    30,561 | 1.62× |               1,345.4 → 831.2 |
| med_term      | TOP_100   |      49 |     10,088 |    13,919 | 1.38× |             2,590.0 → 1,859.2 |
| prefix3       | COUNT     |       4 |      2,546 |     2,415 | 0.95× |           11,353.0 → 12,034.5 |
| prefix3       | TOP_10    |       4 |      1,504 |     4,808 | 3.20× |            19,431.6 → 5,916.5 |
| prefix3       | TOP_100   |       4 |      1,492 |     4,941 | 3.31× |            19,579.6 → 5,814.5 |
| wildcard      | COUNT     |      15 |      1,206 |     1,218 | 1.01× |           24,325.8 → 24,189.5 |
| wildcard      | TOP_10    |      15 |        677 |     5,548 | 8.20× |            43,465.1 → 4,965.9 |
| wildcard      | TOP_100   |      15 |        677 |     5,470 | 8.08× |            43,468.9 → 5,011.6 |
| wildcard_scan | COUNT     |      47 |        321 |       449 | 1.40× |           89,936.7 → 64,395.1 |
| wildcard_scan | TOP_10    |      47 |        324 |       475 | 1.47× |           90,089.7 → 60,224.1 |
| wildcard_scan | TOP_100   |      47 |        323 |       475 | 1.47× |           90,066.5 → 59,924.6 |

## Unicode-word checkpoint versus Luxir

The following is the complete 673-query timing checkpoint before the four newly admitted regex expressions. Prefix/wildcard reference cells were rerun on exactly the same 55 selected expressions; the remaining cells retain the matching reference query sets. Exact count agreement does not establish equal candidate work or scoring semantics. Summa here uses the opt-in impacts index. Later owner/inline gains must not be multiplied into these measurements.

| Family             | Operation | Queries | Summa QPS | Luxir QPS | Summa / Luxir |
| ------------------ | --------- | ------: | --------: | --------: | ------------: |
| and_high_high      | COUNT     |      47 |     2,269 |     4,221 |         0.54× |
| and_high_high      | TOP_10    |      47 |     2,225 |     2,279 |         0.98× |
| and_high_high      | TOP_100   |      47 |     1,400 |     1,733 |         0.81× |
| and_high_low       | COUNT     |      50 |    44,627 |    47,970 |         0.93× |
| and_high_low       | TOP_10    |      50 |    26,090 |    43,628 |         0.60× |
| and_high_low       | TOP_100   |      50 |    24,451 |    30,337 |         0.81× |
| and_high_med       | COUNT     |      50 |     4,767 |     6,204 |         0.77× |
| and_high_med       | TOP_10    |      50 |     4,060 |     8,794 |         0.46× |
| and_high_med       | TOP_100   |      50 |     3,201 |     6,088 |         0.53× |
| high_phrase        | COUNT     |      30 |        60 |        63 |         0.96× |
| high_phrase        | TOP_10    |      30 |     2,036 |     1,144 |         1.78× |
| high_phrase        | TOP_100   |      30 |       506 |       319 |         1.59× |
| high_sloppy_phrase | COUNT     |       7 |       191 |       150 |         1.27× |
| high_sloppy_phrase | TOP_10    |       7 |     7,088 |       374 |        18.94× |
| high_sloppy_phrase | TOP_100   |       7 |     2,731 |       328 |         8.33× |
| high_term          | COUNT     |      45 |    83,284 |   122,838 |         0.68× |
| high_term          | TOP_10    |      45 |    60,500 |    58,989 |         1.03× |
| high_term          | TOP_100   |      45 |    25,484 |    15,833 |         1.61× |
| low_phrase         | COUNT     |      50 |       372 |       369 |         1.01× |
| low_phrase         | TOP_10    |      50 |     2,338 |     1,407 |         1.66× |
| low_phrase         | TOP_100   |      50 |       896 |       764 |         1.17× |
| low_sloppy_phrase  | COUNT     |      37 |       362 |       304 |         1.19× |
| low_sloppy_phrase  | TOP_10    |      37 |       930 |       360 |         2.58× |
| low_sloppy_phrase  | TOP_100   |      37 |       522 |       310 |         1.68× |
| low_term           | COUNT     |      47 |    82,830 |   123,633 |         0.67× |
| low_term           | TOP_10    |      47 |    67,399 |   110,381 |         0.61× |
| low_term           | TOP_100   |      47 |    39,112 |    33,502 |         1.17× |
| med_phrase         | COUNT     |      46 |       167 |       154 |         1.08× |
| med_phrase         | TOP_10    |      46 |     1,709 |     1,003 |         1.70× |
| med_phrase         | TOP_100   |      46 |       573 |       510 |         1.12× |
| med_sloppy_phrase  | COUNT     |      27 |       280 |       217 |         1.29× |
| med_sloppy_phrase  | TOP_10    |      27 |     1,265 |       358 |         3.53× |
| med_sloppy_phrase  | TOP_100   |      27 |       803 |       284 |         2.83× |
| med_term           | COUNT     |      49 |    82,747 |   125,919 |         0.66× |
| med_term           | TOP_10    |      49 |    66,582 |    88,022 |         0.76× |
| med_term           | TOP_100   |      49 |    30,836 |    21,776 |         1.42× |
| or_high_high       | COUNT     |      42 |     2,184 |     4,267 |         0.51× |
| or_high_high       | TOP_10    |      42 |     2,004 |     2,340 |         0.86× |
| or_high_high       | TOP_100   |      42 |     1,468 |     1,773 |         0.83× |
| or_high_low        | COUNT     |      46 |    37,650 |     9,229 |         4.08× |
| or_high_low        | TOP_10    |      46 |    17,878 |    33,938 |         0.53× |
| or_high_low        | TOP_100   |      46 |    11,955 |    15,274 |         0.78× |
| or_high_med        | COUNT     |      45 |     4,469 |     6,556 |         0.68× |
| or_high_med        | TOP_10    |      45 |     5,061 |     7,771 |         0.65× |
| or_high_med        | TOP_100   |      45 |     3,753 |     5,361 |         0.70× |
| prefix3            | COUNT     |       2 |     2,905 |    10,013 |         0.29× |
| prefix3            | TOP_10    |       2 |     5,804 |   117,904 |         0.05× |
| prefix3            | TOP_100   |       2 |     5,797 |   107,381 |         0.05× |
| wildcard           | COUNT     |      11 |     1,481 |     6,654 |         0.22× |
| wildcard           | TOP_10    |      11 |     5,659 |    29,427 |         0.19× |
| wildcard           | TOP_100   |      11 |     5,648 |    28,470 |         0.20× |
| wildcard_scan      | COUNT     |      42 |       292 |     1,740 |         0.17× |
| wildcard_scan      | TOP_10    |      42 |       303 |     1,815 |         0.17× |
| wildcard_scan      | TOP_100   |      42 |       302 |     1,808 |         0.17× |

Four additional regex expressions now complete with exact reference counts: `(www|http|https)`, the weekday alternation, `colou?r`, and `[jkqxz][a-z]*ess`. Their separately timed four-query cells are in [regex-matrix.json](regex-matrix.json). Bounded literal-prefix extraction retains all scan, match and posting limits.

## Concurrent expansion ownership

An ABBA comparison on the same 55-query index gives each expansion a local byte owner, preserving mmap lifetime and the original shared corruption observer. No payload is copied or pinned. The native handle remains 32 bytes; the portable handle grows from 24 to 32 bytes. The later query-local integrity wrapper shares the segment-wide write-once error state; its completed matrix is in [final-matrix.json](final-matrix.json).

| Family        | Operation | Before QPS, two rounds | After QPS, two rounds | CPU µs/request before → after | Anonymous RSS MiB before → after |
| ------------- | --------- | ---------------------: | --------------------: | ----------------------------: | -------------------------------: |
| prefix3       | COUNT     |           2,892, 2,879 |          3,370, 3,415 |                10,003 → 8,476 |                    296.9 → 283.0 |
| prefix3       | TOP_10    |           5,829, 5,856 |        16,270, 17,672 |                 4,846 → 1,584 |                    202.6 → 188.3 |
| prefix3       | TOP_100   |           5,801, 5,763 |        16,153, 17,570 |                 4,877 → 1,591 |                    202.9 → 188.9 |
| wildcard      | COUNT     |           1,487, 1,480 |          1,658, 1,647 |               19,818 → 17,704 |                    623.6 → 609.4 |
| wildcard      | TOP_10    |           5,670, 5,643 |          8,601, 8,784 |                 4,881 → 3,045 |                    350.7 → 337.4 |
| wildcard      | TOP_100   |           5,708, 5,587 |          8,508, 8,579 |                 4,861 → 3,039 |                    356.3 → 343.0 |
| wildcard_scan | COUNT     |               295, 294 |              295, 296 |               97,572 → 97,173 |                    628.3 → 616.0 |
| wildcard_scan | TOP_10    |               305, 305 |              305, 304 |               93,926 → 94,083 |                    626.5 → 613.4 |
| wildcard_scan | TOP_100   |               305, 305 |              305, 304 |               93,664 → 93,545 |                    628.1 → 615.7 |

## ARM and density controls

The Apple M4 checks use release Rust 1.98.1 with native CPU flags and ABBA order. Plain ranked-term fixtures improve 1.25–1.75×, constant-score filters 32–42×, and conjunction controls remain near parity. These are bounded synthetic fixtures, not evidence for a global default change. Single-star matching improves about 1.35×.

The retained list-density gate improves dense membership windows 2.12–2.40× on ARM; sparse controls stay within 1.1%. On the real corpus, `+of +s` COUNT improves about 1.47× in latency and CPU, with five sparse/common controls within about 3.5%. Unconditional grouping was rejected for 20–38% CPU regressions. [Count evidence](count-density.json) and [ARM evidence](arm-unions.json).

Avoiding re-encoding one-to-three-document dictionary postings improves the small prefix fixture 7.18–7.54× versus the immediately preceding union checkpoint. Regex and single-star controls improve 7–12%. Exact membership is checked outside the timed loop. [ARM inline evidence](arm-inline.json) includes binary hashes, confidence intervals and whole-process peak memory; its later concurrent corpus matrix is reported below.

## Recovered concurrent results (September 25)

[deferred-matrix.json](deferred-matrix.json) contains five completed runs on the
same 55 expressions: projected, dense union, deferred views, repeated deferred
views, repeated projected. All 275 exhaustive count audits and 825 HTTP
count/top-10/top-100 checks pass, and all 45 timing cells have zero request
errors. The two outer/inner pairs give the following observed ranges; ratios use
the mean of each pair's median QPS. These are the previously queued Linux runs,
not measurements performed during the new local refactor.

| Family        | Operation | Projected QPS (two rounds) | Dense + deferred QPS (two rounds) | Ratio |     CPU µs/request before → after | Anonymous RSS MiB before → after |
| ------------- | --------- | -------------------------: | --------------------------------: | ----: | --------------------------------: | -------------------------------: |
| prefix3       | COUNT     |              4,879 / 4,855 |                     7,673 / 7,672 | 1.58× |     5,803 / 5,792 → 3,528 / 3,514 |                     276 → 99–100 |
| prefix3       | TOP_10    |            71,489 / 71,303 |                   87,991 / 88,449 | 1.24× |             360 / 360 → 294 / 293 |                  182–183 → 68–70 |
| prefix3       | TOP_100   |            65,751 / 65,917 |                   83,436 / 83,745 | 1.27× |             389 / 390 → 313 / 310 |                  183–184 → 69–70 |
| wildcard      | COUNT     |              2,453 / 2,456 |                     3,708 / 3,725 | 1.51× |   11,845 / 11,823 → 7,737 / 7,697 |                605–608 → 185–187 |
| wildcard      | TOP_10    |            10,964 / 10,943 |                   11,378 / 11,372 | 1.04× |     2,412 / 2,417 → 2,333 / 2,331 |                    332–334 → 151 |
| wildcard_scan | TOP_10    |                  318 / 320 |                         318 / 318 | 1.00× | 92,077 / 91,349 → 90,958 / 92,162 |                606–610 → 180–189 |

The intermediate dense-union run puts prefix COUNT at 7,780 QPS but leaves
prefix top-10 at 70,980 QPS. The retained bitmap explains the count improvement;
deferring posting views contributes the subsequent ranked-query improvement.
Memory values are observed process peaks per cell, not isolated scratch sizes.
The same-query historical Luxir prefix top-10 reference remains 117,904 QPS;
88,000 QPS is progress, not parity. Broad scans remain the largest pattern gap.

### Forward-FST feasibility result

[fst-probe.json](fst-probe.json) records all raw samples. Both dictionary probes
use the separate 8-vCPU validation host described below. The additional map
contains 51,874,769 keys, occupies 236,702,613 bytes (225.7 MiB), and took 23.17 s
to build. Whole-process peak RSS was 690,088 KiB, including the mapped source,
cache, construction and query phases; it is not isolated FST heap use. The
single-core scan/FST/FST/scan probe checks full matching `TermInfo` values before
timing and checks serialized-value digests outside every timed iteration.

The FST path is slower on every tested pattern: 3.80× for `g*itar`, 4.25× for
`h*band`, 4.23× for `q*lity`, 4.38× for `ht*p`, 6.28× for `wh*h`, 5.17× for
`ne*w`, 41.24× for `wou*`, and 1.95× for `*band`. **Reject this prototype.** Its
single-star automaton still traverses the prefix subtree, then performs point
lookups for matching values. This does not establish that every FST design is
slow: a reverse dictionary can use a selective literal suffix as its leading
range. That is a separate format experiment, with a bounded external sort and
additional index bytes to measure. Neither experiment changes live index files
or production expansion budgets; `*band` exceeds the ordinary expansion cap.

### Reversed-term feasibility result

The new probe ran on the preserved **8-vCPU / 4-core Intel Xeon 2.80 GHz**
validation host, pinned to one logical CPU, with Rust 1.98.1 and
`-C target-cpu=native`. This is separate from the 32-client HTTP host above.
It uses the unchanged pre-rename core snapshot and the same immutable Unicode
index, with the canonical reader limited to 1,024 cached dictionary blocks and
a 16 MiB cache byte budget. Baseline and candidate run scan/candidate/candidate/scan in one process,
40 iterations per phase with the first five excluded. Full matching values are
compared before timing; serialized-value checks run after each timed iteration.

The probe reverses term bytes for field 0, sorts hex keys with `LC_ALL=C`, GNU
sort's 256 MiB buffer and one worker, then builds a separate mmap-backed FST.
It searches the reversed literal suffix and applies the remaining prefix test,
then fetches values through the existing dictionary reader and restores lexical
order. The initial seven-pattern probe uses the ordinary scanner when the
literal prefix is at least as long as the suffix. That control does not test
reverse scanning for those three patterns.

The sidecar contains **17,430,381 terms / 125,431,309 bytes (119.6 MiB)** for
one field. Construction took 30.16 s from the previously constructed forward
FST, excluding its build; the temporary hex input was 439,651,019 bytes and the
sort also wrote a sorted copy. Whole-process peak RSS was 310,912 KiB, including
construction and queries. This size is not directly comparable with the
51.9-million-key forward FST covering all fields.

The follow-up covers **all 42 successful broad-wildcard patterns** in the shared
55-pattern HTTP matrix. Matching values agree for every pattern (at most 748
matched terms). **41 improve; `q*tion` regresses 4.77×.** The sum of paired-median
dictionary times across one evaluation of each query drops from 1,448.80 ms to
45.39 ms (31.92×). The median per-query ratio is 56.44×; neither statistic is an
HTTP QPS estimate. Whole-process peak RSS for this query-only run is 240,692 KiB.
The probe has no isolated allocation/scratch meter and uses warm data; it does
not establish cold-I/O or concurrent-cache behavior.

| Query                     | Matched terms |   Scan ms | Reverse ms | Scan / reverse |
| ------------------------- | ------------: | --------: | ---------: | -------------: |
| `g*itar` (initial screen) |            15 |    14.440 |      0.116 |        124.83× |
| `h*band` (initial screen) |            66 |    18.550 |      0.543 |         34.17× |
| `q*lity` (initial screen) |            32 |     2.092 |      1.013 |          2.07× |
| `*band` (initial screen)  |         2,271 | 1,429.143 |     75.130 |         19.02× |

`*band` is a format-feasibility stress case above the production term-expansion
cap. The probe bypasses scan budgets explicitly; it grants no new production
query capability. The broad follow-up's `q*tion` regression shows why literal
length alone is insufficient: suffix frequency and point-lookup cost matter.
A production design needs bounded cost estimates, canonical batched value
lookup, cold/concurrent measurements, native/async/WASM parity, version gates,
and builder/merge ownership before this can become a search implementation.
No live segment or production setting is changed.

Evidence: [initial samples](reverse-probe.json), [42-pattern samples](reverse-all.json),
[initial source](reverse-probe-source.rs.txt), and [42-pattern source](reverse-all-source.rs.txt).
The source archives intentionally retain the historical crate import to match
the captured source checksums. In an isolated checkout they require the same
diagnostic-only `AsyncSSTableReader<TermInfo>::prototype_pattern_values` shim as
the forward probe: delegate to canonical `prefix_scan_values_sync` with unlimited
scan/match budgets. The CLI takes the original index, an experimental output
directory, and the forward FST artifact. The second run reuses the immutable
reverse sidecar; its JSON therefore has no construction measurement. The exporter
checks completion, source hashes and sample counts. Raw private logs and archive
hashes are retained under the workspace's `.context/performance-20260925/`.

## September 25 continuation: I/O and shared scoring

The production changes remain small: dictionary batch lookup now holds one
block while searching all its requested keys, instead of preloading and relying
on cache retention; mapped MaxScore batch admission now respects virtual seeded
slots. Both reuse existing decoders/collectors. Directory range/byte limits are
defined once and reused by hydration planning. No new scorer, dictionary format,
production dependency or backend default is introduced.

### Known-offset Linux I/O

The standalone [probe](../../../scripts/experiments/io_uring/README.md) ran on the
same 8-vCPU Xeon host, Linux 7.0.0-1011-gcp, Rust 1.98.1, optimized release
profile (no target-cpu flag for this probe), io-uring 0.7.14. The main thread and
all positional-read pool workers are pinned to one logical CPU. It creates a
private, fully written 2 GiB deterministic fixture on the attached persistent
disk; the host has 62 GiB RAM. This is **cold-advised I/O, not sustained
larger-than-memory search**. No index files or global cache state are modified.

Each cell reads the same 4,096 unique offsets, at 4 KiB or 64 KiB. Allocation,
registration, warmup and full-byte checks are outside I/O timing; CPU figures
include byte verification. Every cold phase starts with zero resident fixture
pages according to `mincore`. Two passes reverse method order. Timings below
are milliseconds spent in read batches over the complete offset list,
**first / reverse-order pass**, at depth eight:

| Cache / read size | Persistent 8-worker pread |        io_uring | Registered io_uring |
| ----------------- | ------------------------: | --------------: | ------------------: |
| Warm / 4 KiB      |             17.28 / 16.51 |     3.87 / 3.51 |         3.80 / 3.67 |
| Cold / 4 KiB      |           342.70 / 265.33 | 294.09 / 293.24 |     285.80 / 288.26 |
| Warm / 64 KiB     |             53.98 / 54.06 |   38.63 / 37.52 |       38.44 / 38.45 |
| Cold / 64 KiB     |           551.00 / 372.27 | 428.03 / 436.78 |     419.37 / 440.18 |

Concurrency explains most of the large gain over serial positional reads:
registered rings and the persistent pool trade places across cold passes.
Registration provides no consistent warm advantage. The pool pays channel and
scheduler costs that a ring avoids, but it is a diagnostic control, not Tokio's
production pool. The first screen's cold 4 KiB depth-eight serial reads average
1,900.73 ms versus 290.39 ms registered; **do not attribute that entire ratio to
the io_uring API**. Depth one and 32 controls are in the complete export.

Default mmap-copy takes 3,679.75–4,201.30 ms for cold 4 KiB requests and reads
512 MiB for 16 MiB requested; cold 64 KiB requests read about 1,024 MiB for
256 MiB requested. With `MADV_RANDOM`, it avoids that read amplification but a
64 KiB copy can require sixteen separate page faults: this is why the initial
screen's mmap numbers are especially poor. Warm 64 KiB mmap-copy takes
22.41–32.63 ms, ahead of explicit reads. The warm-cache mmap control includes
minor faults/PTE installation, particularly its first pass; it is not a pure
resident-PTE memcpy measurement. Summa's normal mmap path also avoids this
probe's explicit output copy. These results support separate cold-payload
handles, not wholesale replacement of mapped metadata/postings.

Each ring reader preallocates 2 MiB (32 × 64 KiB); the fixed ring registers those
pages. The control pool owns another 512 KiB across eight workers. There are two
ring readers, no per-batch thread creation and no borrowed kernel buffers.
Exported RSS is cumulative process peak, including mapped pages, not an isolated
per-method allocation measurement. Per-cell physical bytes, faults, CPU and
p50/p95/p99 **batch** latencies are retained. The probe drains ordinary errors
and verifies reuse after a short read. No SQPOLL or direct-I/O result is claimed.

Evidence: [96-cell screen](io-results.json), [32-cell pool/default-mmap follow-up](io-followup.json),
[screen source](io-results-source.rs.txt), [follow-up source](io-followup-source.rs.txt).
Raw samples remain in the workspace; exported SHA-256 hashes identify them.
The final standalone source passes Linux release Clippy and verifies cancellation
with forced asynchronous submission: 256 ordinary and 256 registered reads
complete with cancellation, both CQE types are drained, and reused buffers return
correct bytes. Both readers keep working after their opened alias filename is
unlinked. The initial unforced run completed all 512 reads before cancellation;
it did not cover cancelled reads and is not used for that claim. This is a kernel
ownership experiment, not a production future-drop/deletion implementation.
[Lifecycle evidence and exact final-source hash](io-lifecycle.json) link the
check to the tracked standalone source. The source VM was briefly restarted
solely to collect the completed rank probe and finish cancellation validation;
no latency measurements from the restart are mixed into these tables.

### Reverse dictionary: batch values and direction cost

The repaired canonical batch API plus delayed key allocation reduces aggregate
paired-median dictionary time by **35.12×** over the forward scan across the same
42 patterns. Values agree for every query; this is still dictionary-only timing
on the preserved historical source. `q*tion` remains **4.26× slower**: its suffix
has 49,137 candidate terms, versus 5,359 for `q*lity`, despite similar literal
lengths. Batched point lookup cannot fix choosing the wrong direction.

A bounded early-enumeration screen visits at most four reverse keys per forward
block (clamped to 64–4,096), then falls back to the canonical scanner. It avoids
the large regression but rejects 17 useful reverse scans and reduces the
aggregate benefit to **2.87×**; fallback cases lose roughly 2–3% to setup. Reject
this selection rule. The next private experiment stores sorted ranks in the FST
values so two range endpoints give exact suffix cardinality without scanning.
The completed ranked sidecar is **148,197,134 bytes (141.3 MiB)** for
17,430,381 terms, 21.7 MiB larger than the earlier sidecar. Rebuilding it from
that sorted sidecar takes 14.29 seconds, excluding its original construction.
Whole-process peak RSS across build and queries is 247,232 KiB.

Using exact suffix cardinality and the experimental cutoff of 128 reverse keys
per estimated forward block retains the useful reverse scans and routes only
`q*tion` to the ordinary scanner. Aggregate paired-median dictionary time improves
**42.09×** over scanning across all 42 patterns. Forty-one improve substantially;
`q*tion` is at parity (1.01×, within measurement noise). Complete matching values
agree in every case. This ratio is calibrated on the observed corpus; it is not
a production selection policy, holdout result, HTTP QPS estimate or fresh Luxir
comparison. [Ranked samples and build cost](reverse-ranked.json) and
[exact source](reverse-ranked-source.rs.txt) preserve the experiment. A production
proposal still needs query-budget behavior, format/version/merge ownership,
held-out workloads, concurrency and cold/ARM measurements.

[Batch/cap samples and counts](reverse-batch.json) and [source](reverse-batch-source.rs.txt)
retain all 42 query results and ABBA-style paired samples. There are no new HTTP
QPS or fresh Luxir numbers from these dictionary probes.

### Shared MaxScore / BMP collector

Text MaxScore, sparse MaxScore and BMP use one `ScoreCollector` and one seeded
threshold protocol. Two comparator experiments extract that exact production
implementation into a standalone replay, changing only ordering code; there is
no alternative production collector. Full random-bit comparator checks and
seeded output comparisons preserve IDs, score bits and ordinals before timing.
Each experiment uses 100,000 candidates, k=10/100/1000, 25 samples per phase
(drop five), baseline/candidate/candidate/baseline order, Rust 1.98.1 and
`-C target-cpu=native` on Apple M4 and the pinned Xeon.

Packing score/document order into a 64-bit key regresses most cases on both
architectures. Hardware float comparison with the existing total-order fallback
improves random-score replay, especially on x86, but the ARM improving-score
k=1000 control regresses about 19%. Neither comparator is retained. These are
collector-only synthetic streams, not sparse query or Luxir throughput.
Both retain the same 12-byte entries and 48-byte collector, without extra cache
or query allocations. [Full samples](collector-followup.json),
[packed source](collector-packed-source.rs.txt), [float source](collector-float-source.rs.txt).

The retained seeded-batch fix uses the existing conceptual `len()` instead of
physical heap length. A 257-hit regression with a seeded floor now resolves only
the two competitive mapped IDs, including k=1000. An isolated 100,000-candidate
ARM replay reduces seeded admission time 9.49–10.43× (12.39–13.91× on x86); this is a deliberately
seed-heavy work screen, not an end-to-end search speedup. Unseeded controls are
near parity within the observed run variation. BMP's integer threshold screen
already follows this rule, so it receives no new performance claim from this
fix. [Exact replay source](collector-seeded-source.rs.txt) and the shared sample
export make that boundary explicit.

## Rejected experiments and remaining work

- Fewer search workers improve cheap term counts but materially hurt conjunction throughput; the 30-worker setting remains the reference. [Worker matrix](workers-matrix.json).
- Several alternative intersection kernels, wider sparse-seek admission and peeled variable-integer decoding failed matched corpus screens. They are absent from production code.
- Pre-scoring the rare conjunction side regressed representative top-100 probes by 16–29%; it is rejected. An incomplete ARM screen overlapped local compilation and is excluded from all performance claims.
- A 64 MiB dictionary-cache experiment roughly doubled two broad native wildcard probes, but increases memory and still needs its full concurrent matrix collected. No cache default changes.
- The matched 288-query quantized-norm experiment regresses high-term top-100 by about 13%, medium-term top-100 by about 13%, and high/low conjunction top-k by about 9%. It is rejected for the recommended benchmark configuration; defaults remain unchanged. [Matched norm matrix](norm-matrix-valid.json).
- A further blocked-intersection kernel has mixed ARM results and no consistent advantage on the important corpus controls; it remains outside production code.
- Broad dictionary scans and several conjunction/cheap-term cells remain below Luxir. Final renamed-source retiming and a fresh whole-suite Luxir comparison remain outstanding. Both preserved machines were found running during initial evidence collection. The later I/O continuation briefly restarts the source machine after its scheduled shutdown to collect results and finish validation, then stops it again; the 32-client machine is unchanged.

## Validation and reproducibility

The September 24 inline implementation passed the five-step search harness `20260924T223721.093990Z-check`: formatting, strict Clippy, native tests, native without synchronous execution, and standalone broker compilation. Two later decoder regression tests also pass in the nine-test inline selection. Its WASM release build and all 41 JavaScript tests pass. That checkpoint did not change lifecycle or RPC implementation.

The September 25 payload refactor passes all nine stages of `20260925T053741.643327Z-full`, including portable compilation, API docs and five real-server broker tests. The final consuming response-loop adjustment is covered by that server build and real-server run, then the complete final-code `20260925T054823.459132Z-check`: formatting, strict Clippy, 2,083 native tests (25 ignored), native without sync and standalone broker compilation. The final check uses `RUST_TEST_THREADS=1`: two parallel runs exposed an existing broker test-launcher ephemeral-port race (`Address already in use`); all 14 broker integration tests also pass in an isolated serial run. No test deadline or production setting changes. The final WASM release build and 41 JavaScript tests pass. This validates behavior and portability, not an io_uring backend or a hydration performance improvement.

The continued lookup/seeded-screen fixes pass `20260925T061755.141098Z-full`
(all nine stages): 2,084 native tests, 25 ignored, five real-server broker tests,
strict Clippy, native/portable feature checks and docs. The WASM release build
and all 41 JavaScript tests pass. The broker launcher workaround remains
`RUST_TEST_THREADS=1`; production concurrency is unchanged.

New regressions cover borrowed encoded-group bytes and unaligned seeks; local-owner and mmap lifetimes; segment-wide corruption visibility after an expansion is dropped; complete mixed inline/external unions; malformed inline payloads in async/sync expansion; exact inline serialized bytes; and integers above `u32` rejected without truncation. Existing deletion, RGB, cross-segment, score-bit and batch-tail tests remain in the harness.

[matrix.json](matrix.json), [union-matrix.json](union-matrix.json), [coverage.json](coverage.json), and the other linked exports contain only completed retrieved evidence. `summarize.py` validates completion/error markers and HTTP/audit gates while exporting private artifacts. `traversal.patch` is the earlier measured checkpoint, not the final source; [current-source.patch](current-source.patch) records the September 24 implementation snapshot. Later September 25 changes live in the branch; each standalone probe has its own exact source archive and hash.

## Whole-query reverse-dictionary experiment on Summa 2.0

The September 25 continuation moves the ranked reverse probe into a fresh
snapshot of the renamed **Summa 2.0.0** source. The experiment replaces only
bounded dictionary expansion. It reuses the canonical parser, `get_batch`,
posting readers, union scorer, collectors and searcher; no alternate scoring
implementation or production reverse-dictionary format was added. The snapshot
includes the preceding batch-lookup and seeded-collector fixes. The payload-open
refactor below was developed separately; these queries do not hydrate documents.

Same immutable 10-million-document, single-segment Unicode fixture, x86 host,
Rust 1.98.1 release and `-C target-cpu=native`, pinned to CPU 0. Both modes run
inside the same binary with a 1,024-block/16-MiB term cache. For each of the 55
shared queries and COUNT/top-10/top-100, phase order is scan/reverse/reverse/scan:
25 calls per phase, discarding five warmups. The timer includes parsing and
query execution; result serialization and assertions are outside it. Sidecar
opening is outside timing. The selector is still the calibrated 128×
forward-block estimate from the preceding experiment, not a held-out policy.

For each query and mode, average the two phase medians; then sum those
averages across the 42 `wildcard_scan` queries:

| Operation | Forward scan, ms per 42-query sweep | Reverse prototype, ms per sweep | Reduction |
| --------- | ----------------------------------: | ------------------------------: | --------: |
| COUNT     |                            1,622.92 |                          114.52 |    14.17× |
| Top 10    |                            1,545.90 |                           38.60 |    40.05× |
| Top 100   |                            1,547.87 |                           39.37 |    39.32× |

The smaller COUNT gain shows the remaining posting-union work after dictionary
expansion improves. The two prefix controls and eleven narrower wildcard queries
stay within 1.2% in aggregate for each operation. `q*tion`, selected for forward
fallback, stays within about 1% across its operations. These are warm,
single-caller **whole-query** timings, not 32-worker HTTP QPS. They establish that
the dictionary result survives query execution; they do not establish a fresh
throughput ratio against Luxir.

All 55 queries pass exact-count and top-1,000 comparison in both modes, plus
ranked top-10/100/1,000 comparison against complete collection. All 16,500 timed
results agree with the baseline; 13,200 samples are retained in 165 cells.
Counts also match the earlier exhaustive audit. Whole-process peak RSS is
1,811,796 KiB (1.73 GiB), including both modes and warmed index mappings. This is
not an incremental sidecar-residency measurement. The sidecar remains 148,197,134
bytes (141.3 MiB) for one field.

[Measurements, raw samples and hashes](whole-query.json),
[exact probe source](whole-query-source.rs.txt), and
[private instrumentation patch](whole-query-hook.patch) preserve the evidence.
The private hook uses process-global query selection for this serial experiment;
it is **not** suitable for concurrent serving. There is no new public core API in
the production tree for this experiment. The full source archive and resource
logs are retained privately with their hashes. Two earlier probe attempts failed
before completion (type signatures, then JSON serialization of 128-bit segment
IDs); only the completed, corrected run is exported.

Remaining production gates: attach validated sidecars to immutable reader
owners, carry suffix selection per query, preserve term/posting/scan budgets,
version and verify the field/generation association, and design compatible merge
without repeatedly rebuilding a monolithic corpus dictionary. Measure held-out
patterns, cold residency, ARM, concurrent queries and merge interference, then
repeat the HTTP comparison with Luxir on the same host/fixture. Defaults remain
unchanged.

## Explicit stored-payload boundary

`Directory::open_payload` now selects asynchronously consumed files independently
of sync-capable metadata/postings. Native/WASM defaults delegate to `open_lazy`;
only stored-document opening uses the new method. Both cache wrappers preserve
the selected backend. Slice caching retains the opened payload file on misses
and shares its existing cache-fill policy, with one outer directory-read metric.
A mixed mmap/positional regression verifies documents, sync/async scoring,
wrappers and open failures; an unlink/path-replacement regression verifies the
old handle keeps its file owner after the directory drops.

The [payload design](../../batched-payload-reads.md) now gives a concrete proposed
service lifecycle, including admission before allocation, separately tracked
read/cancel completions, byte accounting for completed replies and shutdown
draining. Ordinary owned buffers are the proposed first step; registered-pool
ownership is not added on the strength of the standalone measurements. No
production io_uring dependency or backend default changes.

Validation: full nine-stage harness passed with 2,086 native tests, five
real-server broker tests, strict Clippy, native-without-sync and portable builds,
and API docs; WASM release build and all 41 JavaScript tests passed. Tests ran
serially for the previously documented broker launcher port race. Changes remain
uncommitted.

The temporary source VM was stopped after completed-result collection; its
terminated state was confirmed separately. No timings overlap compilation or
artifact collection, and the other benchmark VM was left untouched.

## Actual Summa hydration: bounded pool and io_uring

The standalone optional hydration binary uses the current production Summa core
without reverse-dictionary instrumentation. All 238 core Rust source files in
the snapshot match the worktree. It uses the canonical writer, store batch
planner, decoder, document cache and vector hydration. The diagnostic directory
selects only stored-payload I/O; metadata and postings remain mapped.

The private fixture contains 4,096 documents in four segments, 112,147,164 bytes
of store files, multi-value text and stored vectors. Each of 40 cells requests
1,024 documents in 32-document windows. Two reversed method orders cover
zero/16 MiB document cache and warm/cold-advised store pages. Zero-cache cells
use 1,024 distinct documents; cache-enabled cells repeat a 256-document working
set four times. Warmup and full serialized-document/requested-address checks
are outside timing. All 40,960 timed document results agree.

Measurements use the same 8-vCPU Xeon host, Linux 7.0.0-1011-gcp, Rust 1.98.1,
release with `-C target-cpu=native`, io-uring 0.7.14 and CPU0 affinity inherited
by workers. The pool and ring share bounded admission/completion code: eight
in-flight reads, an 8 MiB byte budget, owned buffers/file handles and a bounded
queue. The ring uses ordinary buffered reads, without registration, SQPOLL,
forced async during timing or artificial batch delays.

The following values are mean total hydration time for 1,024 requests across
the two method orders, with the document cache disabled:

| Stored-payload reader      | Warm ms | Cold-advised ms | Cold hydration CPU ms | Cold physical read MiB |
| -------------------------- | ------: | --------------: | --------------------: | ---------------------: |
| mmap demand                |   32.82 |          846.06 |                 87.69 |                  96.80 |
| Filesystem demand          |   47.37 |          567.51 |                 89.84 |                  30.73 |
| Filesystem batch           |   48.43 |          117.30 |                 71.55 |                  30.73 |
| Persistent positional pool |   48.34 |          115.72 |                 67.84 |                  30.73 |
| io_uring                   |   56.03 |          181.79 |                 73.41 |                  30.73 |

Batched filesystem hydration reduces cold time 4.84× versus filesystem demand.
The positional pool and existing filesystem batching are close; this ring is
1.57× slower than the pool on cold reads and 1.16× slower warm. Warm mmap remains
fastest. These results support the shared batch planner, not a production
io_uring default. The larger cold mmap cost accompanies 3.15× physical read
amplification, so it cannot be attributed entirely to submission overhead.

Cold cells start with only 96 KiB of store pages resident for explicit readers
and 272 KiB for mmap, after opening metadata. Actual physical reads are recorded
for both orders. With the 16 MiB document cache warm, every method takes about
5.9–6.0 ms per sweep, and the pool/ring issue zero timed payload reads. Cold
cache-enabled sweeps take 238.82/149.75/36.89/37.13/53.33 ms in table order;
medians alone hide their first-pass misses. Whole-process peak RSS is 119,000 KiB
(116.21 MiB), including the untimed oracle and all reader modes. This is not an
isolated per-backend memory measurement.

The ring achieves 1,024 timed reads in 160–161 submission batches in the two
zero-cache cold cells: mostly depth eight, with 31–32 depth-seven and 32–33
single-read batches. Warm cells mostly submit single reads. The current worker
waits for a gathered batch's original completions before admitting its next
batch; continuous replenishment is a possible future experiment, not an
explanation established by this comparison.

Both lifecycle checks pass short-read errors, opened-file reads after unlink,
oversized rejection, deterministic in-flight future drop, subsequent reader
reuse, full completion drain and permit recovery. The forced completion pause
is outside timed runs. It validates future-drop buffer ownership, not kernel
`AsyncCancel` or production index deletion. Both timed services submit and
complete 7,232 reads, report zero errors/dropped replies, and stay within depth
eight. Both standalone binaries pass strict release Clippy; formatting,
documentation/contracts and exporter checks pass. Core/server code is unchanged
since the prior full 2,086-test harness and 41-test WASM validation; those larger
checks were not repeated for this standalone experiment.

[All samples, CPU/I/O/residency and source hashes](hydration.json),
[hydration driver](hydration-source.rs.txt), [shared service](hydration-service-source.rs.txt),
and [shared measurement helpers](hydration-measure-source.rs.txt) retain the
completed evidence. Raw logs and the verified source archive are retained under
`.context/performance-20260925/summa-hydration-20260925/`. This small synthetic
fixture fits RAM; it does not establish larger-than-memory serving, merge
interference, multi-caller throughput, ARM behavior or a fresh Luxir gap.

## Scheduling follow-up: completion-driven admission

The next experiment separates two costs in the ring: waiting for a whole
submission batch to drain, and handing each request to the worker individually.
One worker implementation now supports drain/refill policies. Grouped variants
use a diagnostic wrapper that hands off same-service requests produced during
one poll of canonical preparation. Its thread-local scope is cleared before
yielding, and requests for another service keep their own transport. Admission,
owned buffers/files, decoder, cache and completion validation remain shared.
The diagnostic adapter is not a new production API or backend default.

The investigation also reproduced head-of-line blocking in the production
`FileHandle::read_many`: an ordered future buffer stopped admitting reads after
its first eight when the first result was stalled, even after later requests
completed. The behavior-named regression failed on that implementation. The
reader now replenishes from any completion and writes results into bounded
input-indexed slots. Returned bytes, duplicate/order semantics, first-input-error
selection, error draining and the eight-read limit remain unchanged. Scratch
remains O(32) request/result metadata under the existing 8 MiB batch byte cap.
This is the shared directory implementation; no second decoder or scorer is added.

### Controlled comparisons

The same immutable four-segment fixture, host, Rust 1.98.1 release build and
`-C target-cpu=native` flags are used. Store-file SHA-256 values match before and
after every campaign. Each executable covers eight methods, zero/16 MiB cache,
warm/cold-advised pages and two reversed method orders. Four scenarios cover
one/four callers and CPU0 versus workers allowed CPUs0–7. Callers are futures
on one current-thread runtime: this overlaps I/O, not query CPU. The services
share eight slots; filesystem batching can have eight per caller, so its
four-caller result is not an equal-depth comparison with the pool/ring.

Both the original and changed directory schedulers complete this matrix.
A further before/after/after/before sequence uses the frozen binaries on CPU0
with one caller. No compilation or artifact transfer overlaps timing. In total,
768 cells, 24,576 timed hydration-call samples and 786,432 requested-document
checks pass. Results are verified outside each bounded wave; source and binary
hashes identify both variants. Per-call samples start at the future's first
poll and exclude time queued before it; summed `wave_ns` measures sweep time.

The alternating comparison below isolates the **directory scheduler** change.
Values are mean cold-advised, zero-cache milliseconds per 1,024 documents over
four sweeps per executable (two processes, two method orders):

| Reader                          | Ordered admission ms | Completion-driven admission ms | Time reduction |
| ------------------------------- | -------------------: | -----------------------------: | -------------: |
| mmap control                    |               971.08 |                         951.53 |           2.0% |
| Filesystem demand control       |               626.17 |                         619.27 |           1.1% |
| Filesystem batch                |               123.38 |                         117.95 |           4.4% |
| Positional pool                 |               122.57 |                         119.22 |           2.7% |
| Drain ring, individual handoff  |               175.17 |                         169.10 |           3.5% |
| Refill ring, individual handoff |               162.92 |                         153.55 |           5.8% |
| Drain ring, grouped handoff     |               154.92 |                         152.25 |           1.7% |
| Refill ring, grouped handoff    |               168.24 |                         153.37 |           8.8% |

The controls drift by 1–2%, so the smaller differences should not be presented
as decisive speedups. Cache-enabled cold results are mixed: filesystem batching
changes 38.19→39.81 ms, while grouped refill changes 50.44→46.29 ms. The broader
concurrent comparisons are also mixed. Keep the shared admission fix because
it removes the reproduced scheduling stall while preserving semantics; do not
claim a universal latency improvement or change a backend default from this run.

With completion-driven directory admission, grouped drain/refill rings take
48.92/48.88 ms warm in the alternating check, versus 55.06/54.86 ms for individual
handoff. The positional pool is still faster at 47.04 ms, and mmap takes 32.70 ms.
On cold reads, the best of these ring variants takes 152.25 ms versus 119.22 ms
for the pool. On the initial ordered-admission CPU0/four-caller screen, refill
reduces individual-handoff ring time from 170.05 to 131.65 ms, but the pool is
108.56 ms. Giving workers eight CPUs does not consistently improve refill.
These results support batching as an architectural requirement, not an io_uring
advantage over this positional control.

Peak process RSS across the initial matrix is 120,116 KiB; across the changed
matrix and alternating runs it is 119,220 KiB. These are cumulative high-water
marks including oracle setup, all backends and up to four returned batches;
there is no isolated backend-memory saving claim. The fixture still fits RAM,
and there is no sustained memory-pressure, merge-interference, ARM I/O or fresh
Luxir HTTP comparison. The existing Luxir adapter does not hydrate documents,
so these measurements do not close its retrieval-only gap.

### Validation and retained evidence

All 60 lifecycle cases pass, covering pool/drain/refill, individual/grouped
handoff, short reads, mixed-service isolation, reads through an unlinked opened
file, deterministic future drop after submission, subsequent reuse, original
CQE draining and permit recovery. No timed service reports a read error or
lost reply. The frozen after-snapshot matches all 238 production core Rust files
and every diagnostic runtime source file.

The full nine-stage harness `20260925T174728.281118Z-full` passes: strict Clippy,
2,088 native tests (25 ignored), native without sync, portable core, API docs,
server build and five real-server broker tests. Native tests run serially for
the previously documented launcher port race. The Mac linker repeats its known
compact-unwind-size warning; checks pass. The WASM release build and all 41
JavaScript tests pass. Both Linux diagnostic binaries pass strict release Clippy;
formatting, documentation/contracts, exporter and whitespace checks pass.

[Original scheduler matrix](scheduling.json),
[changed scheduler matrix and alternating runs](scheduling-unordered.json),
[exact core patch](scheduling-core.patch), [driver](scheduling-driver-source.rs.txt),
[shared service](scheduling-service-source.rs.txt), [ring worker](scheduling-ring-source.rs.txt),
and [measurement helpers](scheduling-measure-source.rs.txt) preserve the evidence.
Raw archives and regression/full/WASM logs are retained under
`.context/performance-20260925/scheduling/` and the two `summa-scheduling-*`
result directories. The reverse-FST track remains discontinued.
