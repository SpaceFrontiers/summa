# Shared payload service: lifecycle, buffers and CPU

September 26, 2026. All 78 cells pass, with 7,013,376 decoded-document checks,
902,400 query address/raw-score-bit checks, unchanged fixture hashes,
and zero service errors, quarantine or cgroup OOM events.

The [results](results.json) retain individual cells, latency percentiles, per-pass
wall time, CPU, RSS, cgroup peak/pressure, physical reads and buffer counters.
[Source manifests](sources.json) identify the exact archives, binaries, compiler,
flags and Rust/Cargo files. `before` is the preceding core service; `final` is the
first server-lifetime/exact-capacity implementation; `classes` is the retained
size-class/eviction implementation with shared directory gates. The misleading
historical variant name `final` is preserved to match raw evidence.

## Protocol and limits

Both campaigns use the same private 10M-document index, oracle, Rust 1.98.1,
`-C target-cpu=native`, and 32-vCPU Intel Xeon benchmark VM. Processes are restricted
to CPUs 0–7, four concurrent hydration futures on a current-thread runtime,
32-document windows, zero decompressed-store cache, and an eight-read/eight-MiB
service budget. Compilation happens on a separate VM. Two reversed method orders
cover 512-MiB and 8-GiB cgroups with swap disabled. Cache advice applies only to
the private fixture; payload residency is verified zero before every cell.

Trace cells replay 65,536 fixed addresses twice. Query cells run 47 fixed queries
with top-32 hydration for 20 passes; their smaller working set is a different
control. Retrieval-only cells verify scoring without hydration. Hash checks run
outside operation timings. The raw probe's historical `verified_documents`
counter counts addresses even in retrieval-only cells; this export separates
actual decoded-document and score-bit checks. The first campaign has 52 cells; the revised buffer
campaign has 26. Results below average the two order repetitions.

These are canonical service/decoder measurements, not network RPC latency or
production workload coverage. Mmap's synchronous faults do not have equal I/O
depth to explicit reads. RSS includes mapped metadata; the cgroup cap includes
page cache. Separate campaigns show small timing drift, so their percentages
must not be attributed solely to individual code edits. No BMP scoring speedup,
Luxir gap closure, Windows performance, or mixed merge-load result is claimed.

## Final service comparison

These rows compare pool and ring **within the revised campaign**, with buffer
reuse disabled (the default). CPU is summed user+system operation CPU.

| Workload                 | Pool wall / CPU, s | Ring wall / CPU, s | Ring CPU saving | Ring wall change |
| ------------------------ | -----------------: | -----------------: | --------------: | ---------------: |
| Trace, 512 MiB           |      9.779 / 7.490 |     10.457 / 5.767 |           23.0% |            +6.9% |
| Trace, 8 GiB             |      6.851 / 6.158 |      7.184 / 4.704 |           23.6% |            +4.9% |
| Query + hydration, 8 GiB |      1.563 / 1.427 |      1.520 / 1.082 |           24.1% |            −2.7% |

CPU savings justify keeping explicit io_uring selection even where trace latency
trails the pool. The initial campaign's paired warm controls still favor mmap;
it remains the server default. Final no-reuse trace peak RSS is approximately
510 MiB under the 512-MiB cgroup and 1,152 MiB under 8 GiB for both explicit
backends. Full per-cell memory and latency evidence is in the JSON.

The first lifetime integration tracked the previous implementation closely:
512-MiB ring trace wall changed from 10.193 to 10.103 s and CPU from 5.674 to
5.691 s; pool changed from 9.390/7.283 to 9.295/7.249 s. The later revised-policy
campaign's no-reuse controls are modestly slower. This is not evidence that the
entire refactor improves latency or CPU relative to the preceding service.

## Buffer experiment and decision

The first pool retained an eight-MiB startup metadata buffer indefinitely, rejecting
smaller returned allocations. Every timed cell reported zero reuse. A failing
regression reproduces that sequence; this policy is removed from active code.

The retained recycler evicts oldest idle allocations and uses power-of-two classes.
Admission charges capacity, and byte views expose only the requested range. The
final owner returns storage; empty/sliced views and cancelled replies preserve that
lifetime. Idle retention is capped at eight allocations/eight MiB. Each returned range has backing smaller than twice its original requested length,
separately from the idle budget; further slices retain that same allocation.

With the eight-MiB optional budget, timed reuse reaches 12.8–20.2%, and end-of-cell
idle retention is only 64–88 KiB. Ring trace wall changes from 10.457 to 10.312 s
at 512 MiB and 7.184 to 7.056 s at 8 GiB; CPU changes are negligible. Query wall
moves from 1.520 to 1.532 s, with a small CPU increase. Pool results are similarly
mixed. This establishes functioning bounded reuse, **not a consistent performance
win**. The default idle budget stays zero. More retained buffers, queue wakeup/
coalescing, queue-versus-service timing and concurrent merge output remain measured
follow-ups; registration is not implemented.

## Correctness and reproducibility

Final validation: nine-stage native `full` harness (2,114 tests, 25 ignored, plus
five real-server broker tests); Linux strict core/server Clippy, 56 directory tests,
four canonical document-batch tests and RPC hydration/deletion with both pool and
real io_uring. WASM release + 41 JS tests cover the portable heap-accounting change;
subsequent buffer/lifecycle refinements are native-only.

Regression evidence covers stale idle eviction, capacity admission, final-view
return, cancelled requests, deletion after a cancelled unpublished open, fresh
index generations, and failed cancellation refusing unlink/shutdown. The exceptional
failure test injects a cancellation outcome; successful kernel cancellation also
runs against a real ring. An actually malfunctioning kernel and a Windows host were
not exercised. The heap-accounting regression also fails against the preceding
implementation. This changes accounting, not dictionary validation or index bytes.

Raw archives and excluded intermediate compile/test failures are retained privately
under `.context/payload-followup-20260926`. Regenerate this export with:

```sh
python3 docs/benchmark-results/payload-service-2026-09-26/summarize.py \
  .context/payload-followup-20260926/summa-payload-followup-20260926 \
  .context/payload-followup-20260926/summa-payload-classes-20260926
```

Both cloud VMs were explicitly stopped and independently confirmed terminated
after artifact collection; their shutdown timers were additional safeguards.
