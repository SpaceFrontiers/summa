# Fixed-arrival sparse I/O comparison

September 26, 2026. This extends the existing sparse probe after
[single-ring wakeup hardening](../ring-wakeup-2026-09-26/README.md). Production
search code and defaults are unchanged. The new mode tests continuous arrivals
without the wave probe's global verification pauses.

## Measured outcome

All **54 load cells** complete: **259,200 offers**, **221,260 accepted queries** verified against the oracle, **37,940 rejected offers**, and **7,080,320 decoded documents**. Three wave smoke cells verify another 192 queries / 6,144 documents. Fixture hashes are unchanged; I/O, notification, worker, quarantine and OOM failures are zero. Admission rejection is an explicit measured outcome, not a correctness pass for an unexecuted query.

**Ring saves CPU below overload; it does not win every workload.** At 400/800 offered MaxScore queries/s, ring uses 5.7% / 2.2% less process CPU per completed query than pool. At 200/400/600 mixed queries/s it saves 4.8% / 5.5% / 3.4%. Every paired repetition shows a CPU saving at those rates. Steady p95 is close to pool, within about 1%. CPU includes verification; these are three short observations on one machine, not a universal backend ranking.

Means of three observations follow. Completed/s and CPU/query cover the complete eight-second window plus drain. Rejection and percentiles use arrivals scheduled in seconds 1–8; rejected offers are excluded from percentiles. Peak memory covers the complete cell, including bounded observation export.

### MaxScore

| Offered/s | Backend | Completed/s | CPU/query, ms | Later rejection | Later p95, ms | Later p99, ms | Peak MiB |
| --------: | ------- | ----------: | ------------: | --------------: | ------------: | ------------: | -------: |
|       400 | mmap    |       391.3 |         2.966 |           0.00% |          3.46 |          3.67 |    186.0 |
|       400 | pool    |       399.0 |         6.394 |           0.00% |          6.83 |          7.35 |     74.3 |
|       400 | uring   |       398.7 |         6.027 |           0.00% |          6.78 |          7.26 |     72.7 |
|       800 | mmap    |       775.9 |         2.888 |           0.00% |          3.62 |          4.37 |    202.0 |
|       800 | pool    |       794.3 |         5.613 |           0.00% |          7.05 |          7.78 |     93.1 |
|       800 | uring   |       793.8 |         5.491 |           0.00% |          7.12 |          7.88 |     90.8 |
|     1,200 | mmap    |      1157.7 |         2.850 |           0.00% |          3.91 |          4.71 |    218.0 |
|     1,200 | pool    |      1066.9 |         5.000 |          10.28% |          8.70 |          9.63 |    108.9 |
|     1,200 | uring   |      1035.7 |         5.043 |          12.76% |          9.02 |         10.00 |    107.5 |

### Alternating MaxScore/BMP

| Offered/s | Backend | Completed/s | CPU/query, ms | Later rejection | Later p95, ms | Later p99, ms | Peak MiB |
| --------: | ------- | ----------: | ------------: | --------------: | ------------: | ------------: | -------: |
|       200 | mmap    |        29.5 |        14.752 |          87.43% |        721.26 |        901.21 |    256.0 |
|       200 | pool    |       194.2 |         7.570 |           0.00% |          9.66 |         10.10 |    178.5 |
|       200 | uring   |       194.2 |         7.205 |           0.00% |          9.71 |         10.12 |    177.2 |
|       400 | mmap    |        27.7 |        15.406 |          94.63% |        823.28 |        930.74 |    256.0 |
|       400 | pool    |       386.1 |         7.405 |           0.00% |          9.63 |         10.07 |    187.6 |
|       400 | uring   |       385.7 |         6.997 |           0.00% |          9.61 |         10.07 |    186.8 |
|       600 | mmap    |        29.1 |        14.666 |          96.21% |        833.98 |        971.85 |    256.0 |
|       600 | pool    |       575.5 |         7.065 |           0.00% |          9.88 |         10.54 |    196.9 |
|       600 | uring   |       574.9 |         6.825 |           0.00% |          9.92 |         10.66 |    196.1 |

At 1,200 offered MaxScore queries/s, pool completes about 1,067/s versus ring's 1,036/s. Ring rejects 12.76% of later arrivals versus pool's 10.28%, with slightly higher tail latency and 0.9% more CPU per completed request. Ring's lower total CPU at overload reflects fewer completions and must not be called an efficiency win. Mmap handles all later MaxScore offers at this rate with much lower latency/CPU, but its initial-second rejection is 28%, versus 16.2% pool and 19.8% ring. Warm-only success hides a cold-start tradeoff.

The mixed mapped control reaches the 256-MiB cap and rejects 87–96% of later offers. It reads about 1.7–1.8 GiB from storage per cell and records roughly 383,000–410,000 file refaults. Pool and ring read about 140 MiB, record zero file refaults, remain below 197 MiB, and accept all later offers through 600/s. These counters strongly support memory-pressure thrashing as the reason for the mapped control's poor result. This does **not** establish a 20× general ring speedup: it is a constrained mixed-workload comparison against an overloaded/reclaiming control. BMP scoring remains mapped, so the shared explicit-payload path benefits the mixed workload without a second BMP scorer.

Cold transients remain visible in explicit mixed reads: pool/ring reject about 23–33% of first-second offers across these rates. Overall completion rates therefore stay below offered rates even though later rejection is zero. Per-query acceptance is published; the overloaded mapped control can also change its surviving MaxScore/BMP mix. For example, at 400/s it accepts about 104 MaxScore and 131 BMP requests per cell instead of the offered 1,600 of each.

Generator p95 lag is about 1.3–2.0 ms and is included in response latency. The oracle-check p95 is about 0.15–0.18 ms. Do not attribute either to the storage backend alone or subtract overlapping verification time from process CPU. The mapped mixed tails (roughly 0.7–1.0 seconds) are far larger than generator lag.

The evidence supports retaining ring as an explicit CPU-saving option for these sub-overload workloads. It also preserves the reasons to keep mmap as the general default and pool available for peak explicit-read throughput. The next optimization target is the warm explicit MaxScore path and ring's high-rate capacity; larger working sets, merge interference and ARM remain unmeasured.

## Measurement contract

The same immutable 262,144-document, two-segment MaxScore/BMP fixture, canonical
query executor, hydration helper and exact score/document oracle are reused.
Each accepted query hydrates its top 32 documents. MaxScore cycles through 32
queries; mixed work alternates MaxScore/BMP through 64. BMP scoring remains
mapped for every backend. This is not asynchronous BMP payload preparation.

Each eight-second cell offers requests on a fixed monotonic schedule, with at
most eight admitted tasks and four Tokio workers. A full task budget rejects
an offer and records its sequence and intended/observed time. There is no
unbounded waiting queue. Generator delays retain the original deadlines, making
late generation and catch-up bursts visible. Response latency starts at intended
arrival and ends after canonical hydration. It includes generator and runtime
queue delay. Service time, first poll and verification completion are retained
separately for every accepted request.

Each task checks exact hit IDs/raw score bits and a digest of decoded documents
before releasing its slot. Consequently **process CPU and admission occupancy
include oracle checking**. Verification time is reported separately as elapsed
time; overlapping task times are never subtracted from process CPU. Do not
compare these CPU numbers directly with the preceding wave-mode report. Memory
includes bounded observations and final JSON output construction, as well as
the search engine and page cache. Rejection changes the surviving request mix;
per-query acceptance is retained so lower CPU/latency cannot silently be
attributed to less or easier work.

The first second after cache advice and the following seven seconds are reported
separately, based on intended arrival time. "Steady" below labels those later
arrivals, not a claim of long-term stationarity. CPU covers the complete arrival
window and final drain. Completed requests per second include drain time.
Accepted-offer rates use the fixed arrival window. Request percentiles exclude
rejected offers and must always be read alongside rejection rates.

## Protocol

Mmap, pool and the retained single ring run at 400/800/1,200 offered queries/s
for MaxScore, and 200/400/600 for mixed queries. Three interleaved orders
(forward, reverse, rotated) give 54 load cells. Three additional mixed-query
wave smoke cells verify the refactored shared executor against the existing
oracle. Probe settings bound rate to 10,000/s, observations to 65,536, tasks to
32 and the arrival window to 120 seconds before opening the fixture.

All cells use one VM boot, Rust 1.98.1 with `-C target-cpu=native`, CPUs 0–7,
a fresh 256-MiB cgroup, swap disabled, metadata pinning disabled and idle buffer
reuse disabled. Builds and tests run on a separate VM. Fresh CPU/disk inventory
is collected for this boot. Every cell advises only the private fixture and
verifies zero `.sparse`/`.store` residency before opening it. Fixture hashes
must match afterward. Sparse routing is enabled for pool/ring; mmap is the
mapped control. All methods use the same task admission limit; pool and ring
retain the shared eight-read/eight-MiB service budget.

This is an in-process generator over a repeating query set, not HTTP, a live
arrival trace, a larger cold working set, or long-duration production saturation.
It does not measure merge interference, ARM ring performance or the direct
Luxir gap. Initial arrivals can warm later ones. The short repeated workload
cannot justify changing backend, worker-count or residency defaults.

## Implementation and validation

The CLI `load` mode reuses `execute_query`, the existing document hydration
helper, fixture and oracle. The private `sparse/load.rs` module owns only bounded
arrival scheduling and timing. It drains accepted tasks on success; on a query
or task error it cancels and joins remaining futures before returning an error.
The core service retains ownership of submitted kernel I/O. No scorer, decoder,
cache, reactor or lifecycle implementation is duplicated.

Four Linux tests cover limits and fixed deadlines, overload accounting,
admission reuse and failure cancellation/drain. Strict standalone Clippy and the
release build pass. Native `check` passes all five stages and 2,118 tests.
The initial attempt to lint every test target encountered the older raw-pointer
microdiagnostic's intentional `panic=abort` guard. That guard remains; normal
probe binaries are linted and the new safe scheduler's test target is tested
explicitly. Production search sources match the prior validated implementation.
Native `full`, the production Linux lifecycle harness and WASM are not rerun
because this change affects only the standalone Linux diagnostic.

## Reproduction and evidence

See the [load-mode guide](../../../scripts/experiments/io_uring/README.md#fixed-arrival-load).
The exact source/binary/flags are recorded in [sources](sources.json). The
[results](results.json) include all compact cell summaries, acceptance by query,
resource counters, percentiles and hashes of raw observations. The
[validation record](validation.json) captures checks and artifact hashes.

Raw per-offer samples, cell controls, driver scripts and exact source archives
are retained under `.context/sustained-io-20260926/` in the originating workspace.
The exporter reuses the existing source-manifest and percentile helpers, checks
every offered sequence and timing relation, and verifies the cgroup and service
invariants before emitting compact summaries:

```sh
python3 docs/benchmark-results/sustained-io-2026-09-26/summarize.py \
  .context/sustained-io-20260926/summa-sustained-io-20260926 \
  --sources .context/sustained-io-20260926/source-evidence/summa-sustained-io-20260926
```

Both VMs were explicitly stopped after artifact collection and independently
confirmed terminated.
