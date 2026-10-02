# Warm MaxScore execution and ring notification experiment

Retain the shared cursor expansion in `query/scoring.rs`. It reduces CPU and
latency for both explicit backends without another scorer, allocation, cache,
configuration option, or format. Reject notification coalescing: it did not
improve capacity. All production I/O defaults and the ring implementation remain
unchanged. This does **not** establish parity with pool or close the direct Luxir
gap.

## Change and ownership

`Searcher::search` still selects the existing bounded async executor for explicit
sparse reads and polls it on the shared CPU pool. `MaxScoreExecutor` now expands
the existing advance/seek macros directly in its document loop. The same macros
serve synchronous cursor methods and direct async regression tests; the latter
wrappers compile only for tests because production no longer calls them.
Already-loaded advances do not construct an inner block-loading future.
Actual block reads still suspend through the existing directory-owned service.
Seek accounting, lazy ordinals, pruning, accumulation order, cancellation and
error propagation are preserved. There is no new unsafe code or public API.

The [retained patch](cursor.patch) also shares the encoded-block test fixture and
tests failure/cancellation **after** eight blocks have produced partial scores.
A failed refill must return an error, dropping a suspended query releases its
pending read, and a subsequent async execution matches synchronous score bits
and ordinals. Existing tests compare encoded bytes, positioned results, multiple
values, native/current-thread execution and explicit backend routing.

The [notification patch](notification-experiment.patch) was applied only to a
disposable source copy. It adds a pending flag to suppress redundant eventfd
writes. It was measured, not hardened or retained; it must not be treated as an
alternative production wakeup protocol.

## Protocol

Same immutable 262,144-document, two-segment, approximately 652-MiB hybrid fixture
and oracle as the preceding campaign: 32 queries per field, eight query
dimensions, top 32 hydrated documents. Rust 1.98.1, release,
`-C target-cpu=native`; all comparisons run on one Linux benchmark boot, CPUs
0–7, 256-MiB cgroup, no swap, no metadata pinning or idle buffer reuse. Compiler,
binary, source and fixture hashes are recorded in [sources](sources.json) and
[validation](validation.json). The baseline is the preceding validated binary;
both candidates were built on the same build VM with matching compiler/flags.

Three interleaved orders compare baseline/cursor mmap, pool and ring, plus the
disposable ring-notification candidate:

- 63 eight-second fixed-arrival cells: MaxScore at 800/1,200 offers/s and mixed
  MaxScore/BMP at 600/s, eight admitted requests, four Tokio workers.
- Seven mixed smoke cells, using the same oracle.
- 72 four-pass waves: baseline/cursor across all three backends, both fields,
  current-thread/sequential and four-worker/eight-request execution.
- Four separate 30-second CPU-clock profiles at 199 Hz, with DWARF call chains.
  Profiles include startup, verification and output; they are attribution
  evidence, not the throughput measurements.

Fixed-arrival CPU includes oracle verification. Wave CPU excludes verification.
Do not compare their absolute CPU values. Arrival latency includes generator
lag. “Later” means offers scheduled after the first second; wave “warm” means
passes 1–3. Full admission records a rejection rather than creating a queue.
Published load cells retain per-query acceptance counts. Cgroup peaks include
bounded measurement traces and their JSON export, so more accepted requests can
increase reported peak memory independently of search scratch.

## Profiles

Before the change, `TermCursor::advance` and `seek` async bodies account for
9.10% of pool and 9.09% of ring self samples. Those samples include necessary
navigation, not just removable wrapper cost. The ring profile additionally
attributes 2.21% to `ensure_block_loaded`. The candidate profile has no separate
advance/seek async wrappers: navigation is in the shared loop and
`seek_prepare` (3.72%); block loading accounts for 1.30%. Scheduling/futex work
remains substantial. See [profile attribution and raw hashes](profiles.json).
No samples were reported lost.

## Fixed arrivals

Means of three repetitions. CPU is microseconds per completed query; latency is
later-arrival p95 in milliseconds. Arrows show baseline → retained cursor change.

| Workload / offers/s | Backend |       CPU/query |       Completed/s |       Later p95 |  Later rejected |
| ------------------- | ------- | --------------: | ----------------: | --------------: | --------------: |
| MaxScore / 800      | mmap    |   2,773 → 2,671 |     775.0 → 774.8 |     3.55 → 3.46 |         0% → 0% |
| MaxScore / 800      | pool    |   5,400 → 5,113 |     794.7 → 795.3 |     6.83 → 6.06 |         0% → 0% |
| MaxScore / 800      | ring    |   5,226 → 4,931 |     794.7 → 795.0 |     6.64 → 5.97 |         0% → 0% |
| MaxScore / 1,200    | mmap    |   2,755 → 2,660 | 1,156.0 → 1,158.3 |     3.82 → 3.61 |         0% → 0% |
| MaxScore / 1,200    | pool    |   4,719 → 4,447 | 1,113.5 → 1,169.4 |     8.08 → 7.17 |   6.42% → 1.69% |
| MaxScore / 1,200    | ring    |   4,666 → 4,437 | 1,106.8 → 1,157.1 |     8.18 → 7.37 |   6.77% → 2.63% |
| Mixed / 600         | mmap    | 12,071 → 12,040 |       28.2 → 27.8 | 824.06 → 788.36 | 96.37% → 96.42% |
| Mixed / 600         | pool    |   6,459 → 6,264 |     576.6 → 577.0 |     8.63 → 8.64 |         0% → 0% |
| Mixed / 600         | ring    |   6,184 → 5,991 |     576.5 → 576.7 |     8.56 → 8.58 |         0% → 0% |

At 800 MaxScore offers/s, ring CPU/query falls 5.6% and p95 falls 10.0%; pool
improves 5.3% and 11.4%. All three paired repetitions improve. At 1,200/s, ring
completes 4.5% more queries with 4.9% less CPU per completion; pool improves 5.0%
and 5.8%. Surviving query mixes under overload are published, so CPU/completion
there is not an equal-work CPU comparison. Fixed-work waves below confirm the
CPU reduction. Cold-start rejection remains included in whole-cell throughput.

Mixed explicit CPU/query falls about 3%, with nearly unchanged later latency.
BMP scoring remains mapped and unchanged: this is a benefit to the mixed
workload, not evidence of an optimized or asynchronous BMP scorer. Mixed mmap
still saturates the memory cap and rejects almost all later offers. Its small
CPU/latency differences do not establish an improvement under thrashing.

Candidate ring peaks are 90.2 / 110.3 / 195.2 MiB for these three load points,
versus baseline 92.0 / 108.3 / 196.1 MiB. No OOM or service failure occurred.

Coalescing at 800/s changes CPU/query by less than 0.1%. At 1,200/s it reduces
completed throughput from 1,106.8 to 1,092.4/s, raises CPU/query from 4,666 to
4,718 microseconds and raises later rejection from 6.77% to 8.04%. Mixed CPU is
also slightly worse. There is no reason to add this synchronization to the
production reactor on this evidence. Full [load results](results.json) include
all controls and the rejected candidate.

## Fixed-work waves

All 24 paired explicit-backend comparisons have identical logical read counts
and bytes, as well as exact oracle matches. MaxScore ring CPU falls approximately
5.9% sequentially and 4.4% with eight requests (mean paired changes); warmed wall
falls 5.7% / 5.5%. Pool also improves. Sequential candidate ring uses about 13%
less CPU than candidate pool. At eight requests ring still takes about 7% longer
and uses about 1% more CPU than pool. Faster query execution has not removed
that reactor/scheduling gap.

Mixed explicit waves save approximately 2–3% CPU. Mapped MaxScore's cold phase
is noisy; warmed wall improves. Mixed mmap's mean wall is worse, with paired
eight-request differences ranging from −15.5% to +17.5% under the memory cap.
We do not claim a mapped mixed-workload gain. See all [wave cells](waves.json),
including per-pass timing, memory and counters.

## Validation, artifacts and limits

142 comparison/smoke cells verify **408,541 queries and 13,073,312 hydrated
documents**. The four profiles verify another 136,147 queries. All fixture
hashes match the preceding campaign; no I/O, notification, worker, quarantine or
OOM failures occurred. Fixed arrivals account for all 436,800 offers, including
42,531 rejections. The exporter reuses the prior per-cell validator and wave
exporter; extraction reproduces the prior fixed-arrival report byte-for-byte.

Final native `check` passes all five stages: **2,119 tests**, 25 ignored. Linux
strict Clippy and **147 tests** pass (73 service/RPC tests, 71 scoring tests,
three sparse-refill tests). WASM release and **41 JavaScript tests** pass. The
post-build source update is confined to native tests; all 533 final Rust/Cargo
files match the validated snapshot. Native `full` was not rerun because no
lifecycle/RPC implementation changed; Linux lifecycle and real RPC tests were
run. Initial unused-wrapper Clippy and isolated Git-provenance failures were
fixed and rerun. A standalone Mac test emitted a linker unwind-size warning;
the final strict harness passed.

Complete raw archives, including all four root-owned perf recordings, were
collected under `.context/warm-io-20260926/`. An initial unprivileged archive
could not read those recordings; the verified final archive includes them.
Both build and benchmark VMs were explicitly stopped after collection and
independently confirmed terminated. Checksums and stop records are in
[validation](validation.json).

This remains one short repeating sparse fixture on Linux x86. ARM performance,
larger cold sets, slower storage, concurrent merge pressure, HTTP and a direct
Luxir rerun remain unmeasured here. No production defaults changed. Next useful
experiment: amortize CPU-pool handoffs when preparing already-required initial
cursor blocks, within the existing service admission budget; then measure
larger working sets and merge interference. Owned BMP/Seismic preparation stays
a separate design task around their existing scorers.

Reproduce the exported evidence after restoring the complete raw archive:

```sh
python3 docs/benchmark-results/warm-io-2026-09-26/summarize.py \
  .context/warm-io-20260926/summa-warm-io-20260926
```

Raw archives contain the exact build, profile and campaign commands. Production
defaults must not be selected from this synthetic fixture or one architecture.
