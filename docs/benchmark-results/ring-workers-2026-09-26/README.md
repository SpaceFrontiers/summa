# Ring workers and queue wakeups

September 26, 2026. Experimental source snapshots only; the production payload
service, backend defaults and shared eight-read/eight-MiB admission limit remain
unchanged. This follows the [bounded sparse cursor experiment](../sparse-concurrency-2026-09-26/README.md).
The subsequent [wakeup hardening](../ring-wakeup-2026-09-26/README.md) retains a
tested single-ring notification path; extra workers remain experimental.

## Measured outcome

**Four rings match pool's aggregate throughput on this fixture**, but this is
not a universal backend win or a default change. The campaign completes
**96 total cells** across screening and confirmation, verifying **30,720 queries and 983,040
decoded documents**. All fixture hashes are unchanged; service errors, worker
failures, active leftovers, quarantined bytes and OOM events are zero.
[Results](results.json) retain every cell and request sample;
[sources](sources.json) identify the exact source snapshots, compiler, flags,
binaries and patches.

Means of three confirmation observations, 256 MaxScore queries with sparse
payload reads and top-32 hydration:

| Requests | Backend    | Batch wall, s | Process CPU, s | Queries/s | Request p95, ms | Peak cgroup, MiB |
| -------: | ---------- | ------------: | -------------: | --------: | --------------: | ---------------: |
|        1 | Pool       |         1.100 |          1.079 |       233 |            9.63 |             48.7 |
|        1 | One ring   |         1.126 |          0.920 |       227 |           11.43 |             48.2 |
|        1 | Four rings |         1.062 |          1.002 |       241 |            9.09 |             48.5 |
|        1 | Wake ring  |         1.039 |          0.929 |       246 |            8.66 |             47.9 |
|        8 | Pool       |         0.307 |          1.292 |       835 |           18.01 |             59.7 |
|        8 | One ring   |         0.334 |          1.305 |       765 |           22.98 |             58.6 |
|        8 | Four rings |         0.304 |          1.273 |       841 |           18.87 |             59.0 |
|        8 | Wake ring  |         0.312 |          1.302 |       821 |           18.65 |             58.2 |

At eight requests, four rings reduce wall time **9.0%** and CPU **2.4%** versus
one ring. Against pool, their wall/CPU means are **0.8% / 1.5% lower**: treat
that as approximate parity, not a meaningful lead. Pool wall times range
0.301–0.318 s and four rings 0.303–0.305 s. Four-ring request p95 is still about
4.8% higher than pool. Sequential four-ring CPU rises **8.8%** versus one ring;
spreading work across more workers has a cost.

Wakeups reduce sequential wall **7.8%** versus one ring with **1.0% more CPU**.
Against pool, their sequential means use **5.6% less wall and 13.9% less CPU**.
At eight requests they reduce single-ring wall **6.7%**, but still take **1.7%
longer than pool** with **0.7% more CPU**. They do not establish a concurrent CPU
saving over pool.

The phase split explains where the gains occur. Initial/warmed wall totals in
the eight-request sparse confirmation (32 initial + 224 subsequent queries):

| Backend    | Initial pass, s | Seven warmed passes, s |
| ---------- | --------------: | ---------------------: |
| Pool       |          0.0799 |                 0.2268 |
| One ring   |          0.1013 |                 0.2331 |
| Four rings |          0.0829 |                 0.2214 |
| Wake ring  |          0.0821 |                 0.2299 |

Wakeups remove most of the first-pass single-ring penalty but barely change
warmed time. Four rings improve both phases in the concurrent case. Sequential
wakeups likewise change initial wall from 0.359 to 0.273 s while warmed wall is
nearly unchanged, 0.767 to 0.765 s. This is consistent with queued jobs waiting
behind an earlier pending read in the single-ring worker.

Document-only confirmation, with sparse scoring still mapped:

| Requests | Backend    | Batch wall, s | Process CPU, s | Peak cgroup, MiB |
| -------: | ---------- | ------------: | -------------: | ---------------: |
|        1 | Pool       |         1.028 |          0.934 |             74.6 |
|        1 | One ring   |         1.019 |          0.821 |             74.0 |
|        1 | Four rings |         0.994 |          0.861 |             74.4 |
|        1 | Wake ring  |         1.036 |          0.846 |             74.2 |
|        8 | Pool       |         0.242 |          0.817 |             81.9 |
|        8 | One ring   |         0.244 |          0.770 |             79.8 |
|        8 | Four rings |         0.233 |          0.773 |             80.6 |
|        8 | Wake ring  |         0.242 |          0.771 |             80.0 |

Four rings improve concurrent document-only wall about 4.6% versus one ring,
with essentially unchanged CPU. Sequential document-only CPU is 4.9% higher.
Wake ring has no clear document-only advantage: its sequential mean is 1.7%
slower and uses 3.1% more CPU than one ring, with one slower observation.
These memory peaks are well below the 256 MiB cap; this is not sustained memory
pressure after the explicit-read path is selected. Startup cost is outside the
operation timings.

Every MaxScore sparse cell performs exactly **18,428 reads / 139,312,516 bytes**;
every document-only cell **8,196 / 113,140,236**. Thus scheduling gains do not
come from different logical work, extra aggregate depth or reduced correctness
checks. Mixed MaxScore/BMP screen differences are small: concurrent wall is
1.167 s pool, 1.197 one ring, 1.180 four rings and 1.188 wake ring. Two rings give an
intermediate concurrent MaxScore result, 0.316 s / 1.290 s CPU, and sequential
CPU above wake ring at similar wall time. The confirmation selects the strongest
concurrent-throughput and sequential-CPU candidates; it does not establish that
two rings are an inferior compromise for every workload. Cooperative task
scheduling does not remove the concurrent gap. Neither receives confirmation.

## Candidates

All candidates use the same service, admission, byte ownership, decoder and
cancellation machinery. Narrow patches against the retained implementation are
published here: [two](two.patch), [four](four.patch), [coop](coop.patch),
[wake](wake.patch); no alternate service modules are retained.

| Variant | Change                                            | Purpose                                               |
| ------- | ------------------------------------------------- | ----------------------------------------------------- |
| Base    | One ring worker                                   | Retained implementation and pool control              |
| Two     | Two ring workers, four outstanding reads each     | Reduce waiting behind unrelated I/O                   |
| Four    | Four ring workers, two outstanding reads each     | Test further isolation at the same total depth        |
| Coop    | One ring with `COOP_TASKRUN`                      | Test cooperative kernel task scheduling               |
| Wake    | One ring with eventfd notification and a poll SQE | Let a queued job interrupt a wait for an earlier read |

The worker variants construct every ring before launching consumers. Partial
thread-start failure closes admission and the queue and joins started workers.
The wake variant adds one control SQE, identified separately from payload CQEs;
it does not count control completions as reads. The eventfd owner survives ring
teardown. There is no busy polling, SQPOLL, registered-buffer pool, extra payload
concurrency or new cache. SINGLE_ISSUER/DEFER_TASKRUN are not tested: current ring
creation and submission occur on different threads and need an ownership refactor
before those flags can be considered.

## Protocol

The same immutable 262,144-document, two-segment, 652 MiB MaxScore/BMP fixture and
exact score/document oracle from the [hybrid campaign](../hybrid-sparse-io-2026-09-26/README.md)
are reused. All cells run within one boot of the same GCP `n2-highmem-32` VM
(minimum CPU platform: Intel Cascade Lake), Linux x86 kernel `7.0.0-1011-gcp`,
CPUs 0–7, Rust 1.98.1 and `-C target-cpu=native`; 256 MiB cgroup, swap off,
pinning off and idle buffer reuse off. Builds and validation run on a separate
host. The archived lscpu/disk inventory is carried over from the preceding
campaign, not refreshed after the restart; exact CPU model is therefore not
reasserted for this boot. Comparisons here are between candidates in this run. Every cell checks zero
`.sparse`/`.store` residency after private-fixture cache advice; before/after
fixture hashes must match.

Each cell runs eight passes: 256 MaxScore queries or 512 alternating MaxScore/BMP
queries, with top-32 hydration. The probe separates the initial pass from seven
subsequent warmed passes. The initial pass is not entirely cold: earlier queries
can warm data for later queries. Likewise, warm means repeated within the same
process, not guaranteed residency of every payload page.

The 48-cell screen runs all six backend/candidate pairs with sequential
current-thread and eight-request/four-worker execution, MaxScore and mixed work,
and two reversed orders. Four workers and wakeups advance to a 48-cell
confirmation: base pool, base ring, four rings and wake ring; MaxScore with sparse
reads enabled or document-only reads, both execution modes, three orders
(forward, reverse, half-rotated). Candidates were selected from the screen, so
confirmation is reported separately rather than pooling both campaigns.

CPU and wall are counted once per bounded wave, excluding subsequent oracle
verification; throughput never sums overlapping request latencies. Request
latency starts at the task's first poll and excludes earlier queue delay.
Per-pass totals sum to the aggregate. These are closed batches with verification
gaps, not sustained arrival-load or HTTP measurements. BMP scoring stays mapped;
mixed-work improvements do not establish asynchronous BMP support.

## Pending-read regression

The [test-only patch](wake-regression.patch), applied after the wake prototype,
uses a pipe read with no data and a ready file read. A test hook signals after
the worker has drained its queue and is about to enter the kernel, preventing
the later job from accidentally joining the first batch. With the wake poll
disabled, the ready job times out after five seconds behind the pending pipe
read. With the poll enabled, it completes before the pipe is released. Both
paths release the blocked read and drain the worker. The positive test also
checks zero active and quarantined bytes. The negative test fails for the
expected timeout; the positive test passes.

This isolates the scheduling problem without depending on disk timing. The
hook and test are absent from every measured binary and production file. The
source is restored after the regression, and candidate validation verifies its
Rust bytes against the original measured archive before rebuilding.

## Reproduction

Build each disposable source with its published patch and the same probe:

```sh
cargo build --locked --release \
  --manifest-path scripts/experiments/io_uring/Cargo.toml \
  --features hydration --bin summa-sparse-io-probe
SUMMA_PROBE_PASSES=8 SUMMA_PROBE_RUNTIME_THREADS=4 SUMMA_PROBE_CONCURRENCY=8 \
  SUMMA_PIN_METADATA_BUDGET_MB=0 \
  /path/to/summa-sparse-io-probe /private/fixture run /private/oracle.json uring maxscore sparse
```

Use `pool`, `mixed`, or `store` for the corresponding controls. Follow the
[probe guide](../../../scripts/experiments/io_uring/README.md#hybrid-sparse-residency-probe)
for private fixture preparation and memory/residency collection. The raw run
scripts, cell logs, exact source archives and validation evidence are retained
under `.context/ring-workers-20260926/` in the originating workspace.

Re-export collected artifacts:

```sh
python3 docs/benchmark-results/ring-workers-2026-09-26/summarize.py \
  .context/ring-workers-20260926/summa-ring-workers-20260926 \
  --sources .context/ring-workers-20260926/source-evidence/summa-ring-workers-20260926
```

The exporter reuses the preceding campaign's parser and integrity checks, then
validates pass totals, repeat counts and the shared admission bound. Small
percentage differences from two or three observations are not robust evidence
of a universal winner. This campaign does not measure the direct Luxir gap,
concurrent merge pressure, a larger cold sparse working set or ARM performance.

## Validation and cleanup

[Validation evidence](validation.json) records commands, log hashes and distinct
core test-binary hashes. Each of the five source variants passes a fresh Linux
io_uring harness: strict core/server Clippy, 56 directory tests, four document
batch tests, two sparse-routing/equivalence tests and one real RPC
hydration/deletion test. The standalone probe also passes strict Clippy. The
pending-read regression above fails and passes with notification disabled and
enabled, respectively.

Initial debug validation reused Cargo test binaries across source snapshots
sharing a target directory. Those runs are not used as candidate-specific proof.
The final validation cleans `summa-core` before every variant, verifies a fresh
compilation and records five distinct executable hashes. Release builds had
already compiled each variant independently; their source/binary hashes and the
benchmark results remain unchanged.

Native `check` passes **2,118 tests**, with 25 ignored. One initial broker
discovery test exceeded its ten-second deadline; the isolated retry and complete
check retry both pass. Formatting, report integrity and documentation checks
pass; the historical exporter still produces byte-identical prior results.
Native `full` and WASM were not rerun this turn: production Rust/Cargo sources
are unchanged from the preceding validated implementation. The sole retained
Rust change adds per-pass metrics to the standalone probe. These Linux
prototypes still need wider platform/load validation and startup/notification
failure coverage before production adoption.

The raw archives and source snapshots are retained in the originating workspace;
their hashes are included in the validation evidence.

Both benchmark and build VMs were explicitly stopped after artifact collection
and independently confirmed terminated.
