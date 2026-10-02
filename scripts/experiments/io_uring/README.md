# Linux payload I/O experiment

The default standalone probe measures known-offset immutable payload reads.
The hydration binary exercises Summa’s first-class `PayloadReadService` through
`MmapDirectory::with_payload_reads`. Core’s optional Linux `io-uring` feature
is enabled for that binary. Neither probe changes server defaults.
The ownership design remains in [batched payload reads](../../../docs/batched-payload-reads.md).

The same deterministic offsets and bytes are used for mmap-copy, positional
reads and io_uring, with ordinary and registered resources. Allocation and
registration happen before timing. Every batch drains all read completions,
including errors, before reusing buffers. Registered buffers and the file stay
alive until the ring is closed. Unexpected submission failures abort this
diagnostic process, rather than risk freeing an in-flight buffer. This is not a
production failure-recovery design.

Use only a private fixture: cold phases advise its pages out of cache, never
global kernel caches. Record actual `/proc/self/io` read bytes, page faults,
CPU, peak RSS and pre-phase residency. A cold-advised fixture smaller than RAM
does not establish sustained larger-than-memory behavior. Buffered I/O,
registration, queue depth, and application batching are separate comparisons.
The `followup` mode adds a persistent eight-worker positional-read pool and
default mmap readahead at depth eight. The `lifecycle` mode unlinks a private
hard link after opening its reader, races explicit cancellation with reads,
drains both original and cancel CQEs, and verifies buffer reuse. This exercises
kernel ownership boundaries, not a production async cancellation protocol.
No SQPOLL, direct I/O or merge-interference claim.

On Linux, build with `cargo build --release --manifest-path
scripts/experiments/io_uring/Cargo.toml`. Run the binary with a new fixture path
and `create`, then that path and `run`. Fixture creation is outside timings;
JSONL results and the exact lockfile/compiler/flags should be retained together.
The CPU affinity of the launching process is inherited by the pool workers;
record it when comparing dispatch costs. The release and development profiles
abort on panic so assertions cannot unwind through live kernel buffer pointers.

## Actual Summa document hydration

Enable the optional `hydration` feature to build `summa-hydration-probe`:

```sh
cargo build --locked --release --manifest-path scripts/experiments/io_uring/Cargo.toml --features hydration --bin summa-hydration-probe
scripts/experiments/io_uring/target/release/summa-hydration-probe /private/new-fixture create
scripts/experiments/io_uring/target/release/summa-hydration-probe /private/new-fixture run
```

This separate binary uses Summa's canonical writer, store reader, batch planner,
decoder, document cache and vector hydration. Metadata/postings remain mapped.
Five controls (mmap, filesystem demand, filesystem batch, persistent positional
pool, and completion-driven io_uring) share a four-segment fixture with 4,096 documents, multi-value text and
stored vectors. Two reversed method orders cover zero/16 MiB document cache and
warm/cold-advised pages. Each cell hydrates 1,024 requests in 32-document windows;
full document and requested-address identity checks occur outside timing.

The pool and ring share admission and completion ownership: eight reads in
flight, an 8 MiB byte budget, bounded queue, owned file handles and buffers.
Dropped futures cannot reclaim submitted buffers. Lifecycle regressions now
live with the core owner, including admission cancellation, submitted/reply
ownership, exact bytes, short reads, shutdown and exceptional ring teardown.
Run `python3 scripts/check_search.py io-uring` on Linux 6.0+ with io_uring
available. Backend unavailability fails explicitly; tests do not silently skip.
The original low-level probe retains its separate kernel-cancellation control.
Timed hydration uses ordinary buffered reads without registration, SQPOLL or
artificial batching delays.

The fixture marker is required for `run`. Cold advice targets its `.store`
files only. Report achieved submission batch sizes, actual cache residency,
CPU, faults, disk bytes and cumulative process peak RSS alongside latency.
This small fixture cannot establish larger-than-memory throughput, production
latency superiority or a benefit over mmap on real workloads. Lower CPU cost
is a valid independent tradeoff. The server default remains mmap.

## Scheduling comparison

`run` accepts an optional caller count, `1` (default) or `4`. Four callers are
independent hydration futures on the same current-thread runtime. They overlap
I/O; they do not parallelize query CPU. Results are retained for at most one
wave (32 documents per caller) and verified after every wave. Use `wave_ns` for
throughput and `samples_ns` for execution time from each future's first poll to
completion. The latter excludes queuing before its first poll. Pin to CPU0 for the
single-CPU control or allow CPUs0–7 to give the persistent I/O workers separate
cores. The ring/pool share eight service slots across callers; filesystem batch
reads may use eight per caller, so their concurrent result is a different I/O
concurrency budget.

The maintained ring uses completion-driven refill. The duplicate diagnostic
service, drain-ring variant and poll-scoped grouped handoff have been removed.
Their measured source snapshots and results remain in the closing-gap report;
they are historical evidence, not measurements of the current implementation.
Current output reports core service counters and ring submission group sizes.

## Real-corpus memory-pressure probe

The hydration binary also accepts a private copy of an existing index. It requires
the `.summa-real-corpus-probe` marker containing `summa-real-corpus-v1` and a newline;
the experiment runner creates and hash-verifies the copy before setting the marker.
It must never target a live index with cache advice.

```sh
summa-hydration-probe /private/index real-oracle oracle.json terms.json 65536
summa-hydration-probe /private/index real-run oracle.json pool trace
```

`terms.json` is an array of query strings. Oracle creation uses the canonical
query parser, searcher and document decoder to record deterministic addresses,
raw score bits and BLAKE3 document hashes. The current diagnostic requires one
immutable segment. `real-run` accepts `mmap`, `pool` or `uring` and
`trace`, `query` or `retrieval`. Trace replays the address set twice; query and
retrieval repeat the declared query set 20 times, with or without hydration.
Four futures share one current-thread runtime and at most four returned batches.
Pool and ring share eight service slots and eight MiB; mmap is the existing
synchronous control. Application store caching is disabled.

An external runner must isolate each cell in a fresh memory cgroup, disable swap,
bound its CPU set, evict only private-fixture pages and verify payload residency
before opening the index. Record cgroup charges, reclaim, pressure and OOM events
alongside the JSON output. Failed cells must remain visible. Operation timing
excludes oracle verification; I/O/fault counters include it. Batch latency starts
at first poll and does not separate queue wait from service time. Summed wave
times measure execution sweeps, not HTTP throughput. The much smaller query
working set is a separate control from corpus-wide address replay.

## Idle-buffer comparison

The maintained probe uses the production service's optional bounded recycler.
`run 4 8388608` enables an 8-MiB idle budget; `run 4 0` is the allocation-only
control. Real-corpus mode takes the same optional byte budget after the workload,
e.g. `real-run oracle.json uring trace 8388608`. Both pool and ring use the same
byte-owner return protocol and limits. Results include idle budget, buffer hits,
allocations and retained idle bytes. Retain a binary/source snapshot from before
the change for a true before/after comparison; disabled reuse alone does not
remove the directory lifecycle leases added in the follow-up.

## Hybrid sparse residency probe

`summa-sparse-io-probe` uses the same directory-owned payload service, document
hydration helper and process counters as the other probes. It builds one private
262,144-document fixture with MaxScore and BMP fields, 4,096 dimensions, up to
64 deterministic dimensions per document and 2,048-byte stored text. Each run
executes 32 deterministic eight-dimension queries four times, with top-32
hydration and zero decoded-store cache. An oracle records exact document/segment
IDs, raw score bits and decoded-document digests separately for both formats.
Checks happen outside operation timings.

```sh
cargo build --locked --release --manifest-path scripts/experiments/io_uring/Cargo.toml --features hydration --bin summa-sparse-io-probe
# Use a new private directory; create never overwrites an existing fixture.
/path/to/summa-sparse-io-probe /private/fixture create
/path/to/summa-sparse-io-probe /private/fixture oracle /private/oracle.json
SUMMA_PIN_MODE=mlock SUMMA_PIN_METADATA_BUDGET_MB=64 /path/to/summa-sparse-io-probe /private/fixture run /private/oracle.json uring maxscore
```

Methods are `mmap`, `pool`, `uring`; fields are `maxscore`, `bmp`, or `mixed`
(alternating both formats). An optional
final argument selects `sparse` (default, also route MaxScore blocks) or `store`
(route only stored documents). Both use one service and admission budget. Run in
controlled memory cgroups with swap disabled and sufficient `LimitMEMLOCK`,
using the same fixture, binary, CPU affinity and cold-cache procedure for each
cell. Enabled pinning requires nonzero intended metadata, full successful
coverage and nonzero `VmLck`; failed locks fail the cell. Pinning stays budgeted
per segment, not per process. Capture cgroup peaks/reclaim/OOM separately from
RSS; record initial file residency after advice restricted to the private
fixture. Defaults remain current-thread execution and sequential queries.
`SUMMA_PROBE_RUNTIME_THREADS=4` enables four Tokio workers (0 selects
current-thread; maximum 32). `SUMMA_PROBE_CONCURRENCY=8` submits bounded waves
of eight tasks (1–32); `SUMMA_PROBE_PASSES` controls repetitions (1–128, default
4). The index retains four CPU workers and one process-shared payload service.

CPU and wall time are measured once per wave; verification occurs after all
tasks finish. Throughput uses batch wall time, not summed overlapping request
latencies. The `phases` array reports those same totals per pass, allowing the
first pass after cache advice to be separated from subsequent warmed passes;
it does not imply that every request in the first pass performs cold I/O.
Per-request timings start when a task starts running, exclude queue
time before that poll, and split search from hydration. Context switches include
the whole measured process interval. These are closed batches with idle gaps
for verification, not a sustained arrival stream or network latency. CPU
profiles must run separately from the measurement matrix.

### Fixed-arrival load

Use `load` in place of `run` to remove global wave/verification pauses:

```sh
SUMMA_PROBE_RUNTIME_THREADS=4 SUMMA_PROBE_CONCURRENCY=8 \
SUMMA_PROBE_RATE=800 SUMMA_PROBE_REQUESTS=6400 \
SUMMA_PIN_METADATA_BUDGET_MB=0 \
/path/to/summa-sparse-io-probe /private/fixture load /private/oracle.json uring mixed sparse
```

The same executor, hydration helper and oracle serve both modes. Rate is bounded
to 1–10,000 offers/second, count to 1–65,536, concurrency to 1–32 and the arrival
window to 120 seconds. Settings are checked before opening the fixture. The
schedule uses one monotonic origin; delayed generation catches up against that
origin rather than moving later arrivals. Full admission rejects an offer and
records its intended and observed time. No unbounded request queue is created.

Every accepted sample records scheduled arrival, admission, first poll, response
completion and verification completion. `latency_ns` includes generator and
runtime queue delay and ends after canonical hydration. `ns` retains execution
latency; `search_ns` separates search from hydration. `verification_ns` measures
elapsed oracle-check time. CPU is counted once across the arrival window and
final drain, **including verification and scheduling**. Do not compare that CPU
directly to wave-mode CPU or subtract summed overlapping verification durations.
Admission stays occupied through verification; this is part of the harness load.

Report rejected offers alongside latency to avoid treating surviving requests
as a complete workload. Check per-query/per-field acceptance because overload
can change the request mix. Separate the first second after cache advice from
later arrivals, and report generation lag rather than treating Tokio timers as
an exact external clock. This is an in-process load generator, not HTTP latency.
Peak process/cgroup memory includes bounded observations and final JSON output
construction. It is not a measurement of service buffers alone.

Accepted work drains before successful return. On query/task failure, the
scheduler cancels and joins remaining task futures, then returns an error;
kernel-buffer ownership stays with the core service. Validate the scheduler on
Linux with:

```sh
cargo test --locked --manifest-path scripts/experiments/io_uring/Cargo.toml \
  --features hydration --bin summa-sparse-io-probe
cargo clippy --locked --manifest-path scripts/experiments/io_uring/Cargo.toml \
  --features hydration --bins -- -D warnings
```

The raw-pointer microdiagnostic intentionally requires `panic=abort`, so its
unwinding test target is unsupported; do not disable that guard to lint the
sustained sparse probe.
