# Hybrid mapped metadata and explicit payload reads

September 26, 2026. The retained implementation combines budgeted mmap metadata
pinning with the shared pool/io_uring service. Stored-document selection remains
`--payload-io`; `--sparse-payload-reads` independently opts asynchronous MaxScore
blocks into the same service. Sparse opt-in defaults off. Explicit synchronous
block APIs, raw-dimension merge reads and BMP/Seismic scoring payloads stay mapped.
Both payload roles reuse the existing decoder, cache, admission and lifecycle owners.

The final 64 cells and preceding 60 exploratory cells all pass: **15,872 verified
queries and 507,904 decoded-document checks**, unchanged fixture hashes, zero
service errors, quarantine and cgroup OOM events. [Results](results.json) retain
individual cells, samples, per-pass times, faults, I/O, memory and service counters.
[Sources](sources.json) identify exact source archives, binaries, compiler, flags
and Rust/Cargo file hashes. The final `optin` source matches all 536 current
Rust/Cargo source entries at validation time.

## What the measurements support

The useful hybrid on this fixture keeps sparse scoring mapped and sends stored
fields through explicit reads. Expanding explicit reads to MaxScore blocks lowers
memory further, but increases CPU and elapsed time. That finding is why sparse
reads have a separate opt-in rather than changing the existing document setting.

These are means of two reversed-order repetitions, **128 MiB cgroup, pinning
disabled**, each timing 128 sequential MaxScore queries plus top-32 hydration:

| Scoring / document reads | Wall, s | CPU, s | Peak cgroup memory, MiB |
| ------------------------ | ------: | -----: | ----------------------: |
| Mmap / mmap              |   3.400 |  0.740 |                   128.0 |
| Mmap / pool              |   0.638 |  0.488 |                    74.3 |
| Mmap / io_uring          |   0.655 |  0.432 |                    74.0 |
| Pool / pool              |   0.865 |  0.846 |                    48.2 |
| io_uring / io_uring      |   1.018 |  0.715 |                    47.5 |

For documents alone, ring uses **11.6% less CPU** than pool and takes 2.7% longer.
With sparse reads enabled, ring uses **15.5% less CPU** than pool and takes 17.7%
longer. Compared with document-only ring, enabling sparse reads reduces peak
cgroup memory about 36%, but raises elapsed time about 55% and CPU about 66%.
The 512 MiB cells show the same tradeoff. Zero major faults on the explicit sparse
path means page faults were replaced by explicit reads, not that disk I/O vanished;
physical bytes remain in the exported counters.

BMP scoring itself still uses mmap. At 128 MiB without pinning, switching document
hydration from mmap to ring changes wall/CPU from 24.032/6.576 s to 12.106/2.639 s.
That is a hydration/cache-pressure result, not an asynchronous BMP scoring gain.
Ring versus pool CPU differences for this BMP workload are much smaller than for
MaxScore; a workload-wide 15% saving is not supported.

## Pinning result

Every enabled cell successfully pins all **5,615,440 logical metadata bytes**
across two segments and reports **5,516 KiB in VmLck**. The configured allowance
is 64 MiB **per segment**, not a process cap. Both sparse fields exist in the same
fixture, so pinning includes metadata for both even when only one is queried.

There is **no consistent latency improvement from pinning**. In the final
128 MiB mapped BMP cells, pinning lowers CPU from 6.576 to 6.267 s (4.7%) while
wall time rises from 24.032 to 24.529 s (2.1%). Explicit-document BMP cells do not
show the same CPU gain. MaxScore differences are small or mixed. The exploratory
campaign likewise does not establish a repeatable pinning latency win. Keep the
zero pin-budget default; successful locking establishes residency, not speed.

The 128 MiB mapped cases reach their cgroup cap. The document-only and explicit
sparse MaxScore cases peak around 74 and 48 MiB respectively, so those particular
working sets are not under sustained cgroup pressure after switching backends.
This limits conclusions about pinning on larger cold sparse working sets.

## Protocol and reproduction

The immutable synthetic fixture has 262,144 documents in two segments, a
4,096-dimension vocabulary, up to 64 deterministic dimensions per document, both
MaxScore and BMP fields, and 2,048-byte stored text. It occupies 683,023,808 bytes.
A preceding implementation creates the oracle; checks compare document/segment
IDs, raw score bits and decoded-document hashes. BMP retains its existing pruning
policy: oracle equality is backend equivalence, not an exhaustive-recall claim.

All cells use Rust 1.98.1, `-C target-cpu=native`, the same Intel Xeon 2.80 GHz VM,
CPUs 0–7, 128/512 MiB cgroups, swap disabled, zero decoded-store cache and zero
idle buffer reuse. Pinning is disabled or configured for mlock with sufficient OS
lock limits; incomplete locking fails the cell. Processes share the existing
maximum of eight payload reads/eight MiB of admitted buffers. No direct I/O,
registered buffers or SQPOLL is introduced.

Private-fixture cache advice precedes every cell, with zero `.sparse`/`.store`
residency verified before opening the index. Each process executes 32 deterministic
eight-dimension queries four times. The first pass starts with cold payloads;
subsequent passes reuse that process's working set. Operation timing includes
search and hydration, but excludes score/digest verification. Index opening and
pinning are outside the operation timing; memory peaks include startup and the
small cgroup control process. Percentiles describe these 128 samples per cell,
not production tail latency.

The final 64-cell campaign uses **one binary** for all role selections. MaxScore
compares mmap, document-only pool/ring, and document-plus-sparse pool/ring, with
both pin settings, both cgroup limits and two reversed method orders. BMP uses
mmap and document-only pool/ring controls. The exploratory campaign compares the
preceding `baseline` and initial implicit-sparse `hybrid` implementation. Its
results remain in the `exploratory` section; do not attribute cross-campaign
timing drift solely to code changes.

The [probe guide](../../../scripts/experiments/io_uring/README.md#hybrid-sparse-residency-probe)
contains build/run commands. The optional final probe argument is `store` or
`sparse`. Core/server configuration is:

```text
--payload-io io-uring --pin-mode mlock --pin-metadata-budget-mb <MiB-per-segment>
```

Add `--sparse-payload-reads` only when the extra memory saving justifies the measured
cost for the deployment. io_uring requires the Linux `io-uring` Cargo feature.
Actual pin failures remain observable; production configuration does not promise
full residency merely because a budget was requested.

Re-export collected evidence:

```sh
python3 docs/benchmark-results/hybrid-sparse-io-2026-09-26/summarize.py \
  .context/hybrid-io-20260926/summa-hybrid-optin-20260926 \
  --exploratory .context/hybrid-io-20260926/summa-hybrid-io-20260926
```

## Validation and limits

Native full harness: nine stages, **2,116 passing tests**, 25 ignored, plus five
real-server broker tests. Final Linux io_uring harness: strict core/server
all-target Clippy, 56 directory tests, four document-batch tests, two sparse
routing/cache tests and one RPC regression. The sparse regression runs both pool
and real ring on current-thread and multithread runtimes, checks normal/fused/
candidate/Boolean results and ordinal positions, preserves synchronous reads,
and rejects cold reads after directory retirement. WASM release build and all
41 JS tests pass. The diagnostic binaries pass strict Clippy with `--bins`;
`--all-targets` invokes an unwind test profile intentionally rejected by the
original raw-pointer diagnostic's panic=abort safety guard.

The first fixture attempt hit writer backpressure and was discarded from timing;
the retained builder uses bounded retries. The initial detached source checkout
needed git metadata for harness provenance; validation was rerun successfully.
Historical failed logs remain with the collected evidence.

This is one synthetic x86 fixture with sequential queries on a current-thread
runtime. Multithread correctness is tested, but server throughput and its CPU-pool
polling overhead are not measured here. A mixed index with explicit sparse reads
uses async search dispatch even for its other query types; mixed-query throughput
needs its own measurement before enabling the option broadly. Concurrent merge
load, Windows execution, registered-buffer gains, asynchronous BMP/Seismic payload
scoring, and the broader Luxir gap remain unmeasured or unimplemented. Backend,
pinning and buffer defaults are unchanged.

Both benchmark and build VMs were explicitly stopped after artifact collection
and independently confirmed terminated.
