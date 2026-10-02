# Hardened single-ring queue wakeups

September 26, 2026. This retains the queue-wakeup optimization from the
[worker experiment](../ring-workers-2026-09-26/README.md) in the shared payload
service. Backend defaults, one ring worker, eight active reads and the eight-MiB
admission budget stay unchanged. Mapped reads remain the default; explicit sparse
reads, metadata pinning and idle buffer reuse remain opt-in.

## Measured outcome

All **72 cells** pass **27,648 exact query/score checks and 884,736 decoded-document checks**. Fixture hashes are unchanged. Service errors, notification failures, worker failures, active leftovers, quarantined bytes and cgroup OOM events are zero. [Results](results.json) retain every sample; [sources](sources.json) identify both builds.

Means of three observations, 256 MaxScore queries with sparse reads and top-32 hydration:

| Requests | Backend       | Batch wall, s | Process CPU, s | Queries/s | Request p95, ms | Peak cgroup, MiB |
| -------: | ------------- | ------------: | -------------: | --------: | --------------: | ---------------: |
|        1 | Pool          |         1.083 |          1.051 |       237 |            9.43 |             48.5 |
|        1 | Original ring |         1.107 |          0.899 |       231 |           11.35 |             48.3 |
|        1 | Hardened ring |         1.032 |          0.921 |       248 |            8.67 |             48.2 |
|        8 | Pool          |         0.303 |          1.272 |       844 |           17.73 |             59.3 |
|        8 | Original ring |         0.334 |          1.289 |       766 |           23.61 |             58.5 |
|        8 | Hardened ring |         0.319 |          1.319 |       803 |           19.31 |             58.9 |

Compared with the original ring, wakeups reduce sequential elapsed time **6.8%** and request p95 **23.6%**, with **2.4% more CPU**. At eight requests, elapsed time falls **4.6%** and p95 **18.2%**, again with **2.4% more CPU**. Memory remains about 48 MiB sequential and 59 MiB concurrent. The optimization improves progress and latency; it is not a CPU reduction relative to the original ring.

Against pool, the hardened ring uses **12.4% less CPU** and **4.7% less elapsed time** sequentially. Under eight-request sparse load it still takes **5.1% longer**, uses **3.8% more CPU**, and has **8.9% higher p95**. Concurrent wall observations are 0.315–0.322 s for the hardened ring, 0.328–0.337 s for the original ring and 0.297–0.313 s for pool. These three observations do not justify a universal backend or worker-policy change.

The reduction comes from the initial pass after cache advice:

| Requests | Backend       | Initial pass, s | Seven subsequent passes, s |
| -------: | ------------- | --------------: | -------------------------: |
|        1 | Pool          |          0.3034 |                     0.7796 |
|        1 | Original ring |          0.3578 |                     0.7491 |
|        1 | Hardened ring |          0.2744 |                     0.7575 |
|        8 | Pool          |          0.0803 |                     0.2231 |
|        8 | Original ring |          0.1029 |                     0.2311 |
|        8 | Hardened ring |          0.0835 |                     0.2353 |

Initial-pass wall falls 23.3% sequentially and 18.8% concurrently versus the original ring. Subsequent-pass wall is slightly higher (1.1% / 1.8%). This agrees with the blocked-pipe regression: a newly queued read can now start before unrelated I/O completes.

Alternating MaxScore/BMP sparse work improves elapsed time 3.3% sequentially and 1.6% concurrently versus the original ring. Sequential CPU is unchanged; concurrent CPU increases 0.9%. Against pool it saves 8.6% sequential CPU, while concurrent differences are small (0.6% more elapsed time, 1.1% less CPU). BMP scoring itself remains mapped.

Document-only MaxScore reads show no clear wakeup benefit: versus the original ring, sequential wall is 1.3% lower with unchanged CPU; concurrent wall is 0.4% higher with 1.3% more CPU. The hardened ring still saves 11.8% / 4.0% CPU versus pool in those two modes. Mixed document-only results are likewise close to the original ring. No extra buffer reuse or worker count is enabled.

Memory is stable across the other workloads too: hardened-ring document-only
MaxScore peaks average 74.0 / 80.0 MiB sequential/concurrent; mixed sparse
164.8 / 173.2 MiB; mixed document-only 190.6 / 198.3 MiB. The largest individual
cell is 199.5 MiB, below the 256-MiB cap. No sustained memory-pressure claim is
made. Every MaxScore sparse cell performs exactly 18,428 reads / 139,312,516
bytes; document-only cells perform 8,196 / 113,140,236. Mixed sparse cells perform
26,620 / 251,797,516 and mixed document-only 16,388 / 225,625,236. Logical work is
identical across candidates and execution modes.

## Implementation and failure behavior

A job published while the ring waits for another read now signals an owned,
nonblocking eventfd. One poll SQE wakes the existing reactor; control completions
have their own token and counters. Notification ownership lives through ring
teardown. Kernel read buffers still use the existing slots, completion handling,
synchronous cancellation and bounded quarantine. The notifier lives in
`payload/ring/wake.rs`; reactor tests live beside it. No second service, cache,
scorer or decoder is introduced.

A failed notification closes admission and the queue, records the failure and
poisons the reactor. An extended-enter timeout checks that flag every 100 ms
while I/O is pending; it is not a read deadline and does not cancel healthy slow
reads. Required read/poll opcodes and extended-enter support are checked before
worker startup. Unsupported kernels fail explicitly. Successful ring jobs add
one eventfd write; the ring has at most one control poll outstanding.

Hardening exposed two lifecycle bugs, both reproduced before their fixes:

- Closing the shared channel on worker failure did not release queued jobs while
  the service retained its sender. The common worker error/panic wrapper now
  drains those jobs, releasing their file owners, admission permits and directory
  leases. Pool and ring share this fix.
- A notification failure arriving between the reactor's health check and its idle
  receive could be mistaken for successful shutdown. The closed-queue exit now
  checks the failure flag again.

Tests cover notification before poll registration and after submission, a ready
file read making progress behind a blocked pipe, saturated counters, bounded
interruption retries, short transfers, notification failure with pending reads,
caller cancellation, startup/armed-poll ownership, healthy slow reads, and the
existing deletion, panic and cancellation/quarantine invariants.

## Measurement protocol

The immutable two-segment, 262,144-document MaxScore/BMP fixture and exact oracle
from the [hybrid campaign](../hybrid-sparse-io-2026-09-26/README.md) are reused.
Three interleaved orders compare original pool, original single ring and the
hardened single ring for MaxScore and alternating MaxScore/BMP queries, sparse
plus document reads and document-only reads, and sequential/current-thread or
eight-request/four-worker execution. Every cell has eight passes with top-32
hydration. Initial and subsequent passes are reported separately.

All comparisons run on one benchmark VM boot, CPUs 0–7, a 256-MiB cgroup with
swap disabled, no metadata pinning and no idle buffer reuse. CPU/disk inventory
is captured during this boot: Intel Xeon at 2.80 GHz, x86_64, Linux
`7.0.0-1011-gcp`, and GCP PersistentDisk. Builds and Linux validation run on a separate
host, with Rust 1.98.1 and `-C target-cpu=native`. Each cell checks zero initial
`.sparse`/`.store` residency after private-fixture cache advice; fixture hashes
must match afterward. Source and executable hashes identify both binaries.

These are bounded batches with oracle-verification gaps. Per-task latency starts
at first poll, excluding earlier scheduling delay; process CPU and elapsed time
are counted once per wave. Initial passes can warm later queries in that pass;
subsequent passes do not guarantee complete residency. This does not measure
sustained arrival load, merge interference, slower storage, ARM performance or
the direct Luxir HTTP/full-text gap. BMP scoring remains mapped; explicit
hydration benefits are shared, but asynchronous BMP/Seismic preparation remains
separate work.

## Reproduction and evidence

Build and run the [existing probe](../../../scripts/experiments/io_uring/README.md#hybrid-sparse-residency-probe).
The same source/fixture/flags apply to both implementations. Raw logs, driver
scripts, source archives and result samples are retained in the originating
workspace under `.context/ring-wakeup-20260926/`.

Export with the shared exporter (its original campaign output remains
byte-identical):

```sh
python3 docs/benchmark-results/ring-workers-2026-09-26/summarize.py \
  .context/ring-wakeup-20260926/summa-ring-wakeup-20260926 \
  --sources .context/ring-wakeup-20260926/source-evidence/summa-ring-wakeup-20260926 \
  --campaigns confirmation --repeats 3 \
  --output docs/benchmark-results/ring-wakeup-2026-09-26
```

## Validation and cleanup

[Validation evidence](validation.json) records commands, log hashes, exact source
checks and the freshly rebuilt Linux core test executable. Native `full` passes
all nine stages: **2,118 tests**, 25 ignored, and **five real-server tests**.
Linux `io-uring` passes strict core/server Clippy, **66 directory tests**, four
document-batch tests, two sparse-routing/equivalence tests and one real RPC
hydration/deletion test (**73 total**). The standalone probe passes strict Clippy.
The WASM release build and **41 JavaScript tests** pass. Payload modules are
native-only; later native cleanup and Linux style fixes do not change WASM
inputs. `npm ci` reports four existing audit findings (three moderate, one high);
this change does not modify dependencies.

The first Linux failure run exposed queued permits surviving worker exit; its
logs are preserved. A later strict-Clippy run caught a nested notification
condition; it was collapsed, then the final release binary was rebuilt and all
72 cells rerun. Earlier measurements are retained under `pre-lint-confirmation`
but excluded from the reported results. Final Linux validation cleans the core
build first, confirms a fresh compilation and records its executable hash.
The 27 focused payload regressions also pass; negative runs establish the
queued-lease and idle-exit failures before their fixes.

The final archive matches all **532 Rust/Cargo source files** it includes in the
workspace. Compared with the control, production/probe changes are restricted to
the payload owner, reactor, counters and tests; two reactor submodules are added.
The final archive excludes unrelated historical benchmark runners under
`docs/benchmark-results`, which are present in the older control archive.
No scoring, on-disk format, dependency or configuration-default changes occur.

Both benchmark and build VMs were explicitly stopped after artifact collection
and independently confirmed terminated.
