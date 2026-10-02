# Concurrent sparse I/O: bounded cursor windows

September 26, 2026. Retained change: explicitly enabled MaxScore payload reads
coalesce at most eight adjacent blocks and 16 KiB in the existing cursor. The
reader owns range validation and reuses the canonical block decoder. There is
no new service, scorer, shared cache, on-disk format or lifecycle protocol.
Mmap, stored-document, synchronous and merge read policies remain unchanged.
Sparse payload reads remain opt-in; mmap and zero pin/reuse budgets remain defaults.

**128 cells passed**, verifying **23,552 queries and 753,664 decoded documents**
against the preceding implementation's exact score-bit/document oracle. Fixture
hashes are unchanged; all cells have zero I/O errors, quarantine and OOM events.
[Results](results.json) retain individual latency samples, process CPU, faults,
physical I/O, context switches, service counts and cgroup memory. [Sources](sources.json)
record source/binary hashes and all 537 Rust/Cargo source entries; the final
measured source matches the retained implementation.

## Interleaved confirmation

Means of two observations per implementation in **before–after–after–before**
order, with 256 MiB cgroups, swap off, pinning off and no idle buffer reuse.
Each cell runs 128 MaxScore queries and top-32 hydration. Four Tokio workers,
four search-pool workers, eight concurrent requests:

| Backend  | Version  | Batch wall, s | Process CPU, s | Queries/s | Request p95, ms | Peak cgroup, MiB |
| -------- | -------- | ------------: | -------------: | --------: | --------------: | ---------------: |
| Pool     | Before   |         0.302 |          1.465 |       424 |           26.98 |             57.1 |
| Pool     | Windowed |         0.170 |          0.694 |       753 |           17.77 |             57.7 |
| io_uring | Before   |         0.371 |          1.560 |       345 |           51.42 |             56.7 |
| io_uring | Windowed |         0.198 |          0.710 |       646 |           23.52 |             57.1 |

The windowed ring reduces elapsed time **46.6%** and CPU **54.5%** versus its
preceding implementation; pool improves **43.8% / 52.6%**. Ring remains about
17% slower than pool in this concurrent cell and consumes about 2% more CPU.
There is no universal ring-over-pool CPU win.

With a current-thread runtime and one request, ring changes from **0.927/0.702 s**
wall/CPU to **0.677/0.470 s** (27.0% / 33.1% reductions). The corresponding new
pool takes **0.622/0.563 s**: ring uses 16.5% less CPU but takes 8.9% longer.

The larger initial matrix includes current-thread/four-worker runtimes,
concurrency one/eight, and homogeneous MaxScore or alternating MaxScore/BMP work.
Its backend order is reversed for the second repetition. The initial before/after
campaigns were sequential; the interleaved confirmation above addresses campaign
order drift for the two endpoint workloads. Mixed four-worker/eight-request ring
work changes from **1.025/2.692 s** wall/CPU to **0.795/1.868 s** (22.4% / 30.6%
reductions). BMP scoring itself remains mapped: this mixed result includes the
MaxScore improvement and shared scheduling, not an asynchronous BMP implementation.

## Why it helps, and what it does not fix

For the 128-query MaxScore trace, document reads account for 4,100 submissions.
Additional sparse reads fall **33,428 → 5,116 (84.7%)**. Total service submissions
fall 37,528 → 9,216. Total requested bytes increase only **1,908 bytes**, from
69,982,076 to 69,983,984 (0.0027%). This fixture traverses most neighboring
blocks; selective workloads can have larger read amplification and need their
own measurements. The byte limit and block limit both apply, with no reads
crossing a dimension boundary. An individual oversized block retains demand-read
behavior and the existing service limits.

Each cursor owns one encoded window. At the 64-term query limit this adds at
most 1 MiB per active segment scorer, independent of corpus size. The previous
lazy-ordinal view and old window are released before a refill; cancellation,
short reads and failures cannot publish partial bytes. Returned block views
share the window allocation. Decoded cursor buffers keep their existing owner.

Separate `perf` CPU-clock recordings corroborate lower scheduling overhead.
In the four-worker/eight-request ring recording, approximate sampled CPU falls
9.58 → 4.32 s; the Tokio worker's `finish_task_switch` self share falls
11.65% → 5.12%. These are diagnostic samples, not benchmark timings or a complete
causal attribution. There are no lost samples; optimized stacks contain some
unresolved frames. Full reports: [before concurrent](before-t4-c8-profile.txt),
[after concurrent](after-t4-c8-profile.txt), [before sequential](before-t0-c1-profile.txt),
[after sequential](after-t0-c1-profile.txt). Raw perf data and commands are retained
with the private campaign artifacts. Profiles run separately after each matrix,
with 32 passes and without the matrix memory cgroup; do not compare their elapsed
or CPU totals directly to four-pass cell timings.

The new sparse path still costs more CPU than keeping scoring mapped and only
reading documents explicitly. In the initial concurrent MaxScore matrix,
document-only ring takes **0.159 s wall / 0.395 s CPU**, versus windowed sparse
ring **0.213 / 0.720 s**; peak memory is about **80 / 57 MiB**, respectively.
The separately interleaved sparse-only cells above should not be substituted into
that comparison. Sparse opt-in therefore stays off by default.

## Protocol and reproduction

The same immutable 262,144-document, two-segment, 652 MiB MaxScore/BMP fixture and
oracle from the [preceding campaign](../hybrid-sparse-io-2026-09-26/README.md) were
reused. Same Intel Xeon 2.80 GHz host, CPUs 0–7, Rust 1.98.1 and
`-C target-cpu=native`; compilation/validation run on a separate host. No network
server is included in timing. Private-fixture cache advice and zero `.sparse` /
`.store` residency checks precede every cell. Later passes reuse the process's
working set. Pinning and buffer reuse are disabled. The shared payload service
retains its eight-read/eight-MiB admission bound; CPU pools are unchanged.

The exact same extended probe source runs before and after. Configure it using
the [probe guide](../../../scripts/experiments/io_uring/README.md#hybrid-sparse-residency-probe):

```sh
SUMMA_PROBE_RUNTIME_THREADS=4 SUMMA_PROBE_CONCURRENCY=8 \
  SUMMA_PIN_METADATA_BUDGET_MB=0 \
  /path/to/summa-sparse-io-probe /private/fixture run /private/oracle.json uring maxscore sparse
```

Use `mixed` instead of `maxscore` for alternating formats, and `store` for the
document-only control. Runtime threads 0 selects current-thread. Requests run in
bounded waves; process CPU and batch wall are counted once per wave. Correctness
verification happens after the wave. QPS uses batch wall, not summed overlapping
latencies. Per-request latency starts at the task's first poll, excluding prior
queue delay; batch time includes task dispatch. This is a closed-batch test with
verification gaps, not sustained arrival-load or production p99 measurement.

The initial matrix has 80 baseline cells (five role/backend configurations),
32 windowed sparse cells and 16 interleaved confirmation cells. Two observations
per configuration are insufficient to characterize small percentage differences.
The 256 MiB mapped controls can reach the cap; explicit MaxScore peaks below
80 MiB, so those cells are not sustained-memory-pressure tests after switching.

Re-export collected evidence deterministically:

```sh
python3 docs/benchmark-results/sparse-concurrency-2026-09-26/summarize.py \
  .context/sparse-concurrency-20260926/summa-sparse-concurrency-20260926 \
  --sources .context/sparse-concurrency-20260926/source-evidence/summa-sparse-concurrency-20260926
```

## Validation and remaining work

Native full harness passes all nine stages: **2,118 tests**, 25 ignored, plus five
real-server tests. Linux io_uring harness passes strict core/server Clippy,
56 directory tests, four document-batch and two sparse-routing tests, and the
real RPC hydration/deletion regression. The standalone probe passes strict
Clippy; both window boundary tests also pass on Linux. WASM release build and all 41 JS tests pass. Window tests cover byte
identity, block/byte/dimension bounds, malformed metadata, short reads, failed
refills, cancellation and retry. The integration regression failed before the
optimization, then passed for pool/ring and current-thread/multithread execution.

Unfinished tracks: ring latency versus pool; BMP/Seismic asynchronous preparation;
process-wide pin admission across retained generations; concurrent merge/load,
a larger cold sparse working set and sustained mixed-client throughput. The
ownership prerequisites are documented in [payload design](../../batched-payload-reads.md)
and [pinning design](../../hot-metadata-pinning.md). This campaign does **not**
rerun or close the direct Luxir full-text gap; that comparison uses a different
corpus and HTTP workload. ARM correctness passed, but these performance numbers
are Linux x86 only. No backend, pinning or buffer-reuse defaults changed.

Both benchmark and build VMs were explicitly stopped after artifact collection
and independently confirmed terminated.
