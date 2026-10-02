# Bounded payload reads for asynchronous storage

September 25–26, 2026. The preparation refactor separates stored-document read
planning from decoding. The September 26 maintenance section below documents
the optional directory-owned pool/io_uring service; earlier proposal and
experiment sections describe the path to that implementation. Index bytes, ranking, field selection and response budgets stay
unchanged.

## Motivation and ownership

[Photon](https://www.perplexity.ai/el/hub/blog/photon) batches known record offsets
and sends cache misses to dedicated io_uring workers. This overlaps independent
cold reads. Its warm-cache policy also avoids shared LRU updates on every hit.
Those techniques do not accelerate a CPU-bound dictionary scan merely by
replacing the I/O API.

Before this refactor, Summa's public search RPC selected hits, then called
`Searcher::get_document_with_fields` one hit at a time. `SegmentReader` owns
visibility and vector hydration; `AsyncStoreReader` owns compressed block
directories, decompression, and the process-wide byte-bounded store cache.
`FileHandle` owns mapped/lazy byte access. The server's `MmapDirectory` currently
returns mapped handles even for `open_lazy`: an async method does not make cold
page faults asynchronous. The benchmark adapter returns document IDs and scores
without hydrating stored fields, so a hydration change cannot explain a win in
the existing Luxir comparison.

## First stage

`FileHandle::read_bytes_ranges` validates the complete batch before I/O and
returns exact-length owned ranges in input order. A batch has at most 32 ranges
and 8 MiB of requested bytes; lazy execution has at most eight reads in flight.
Completion order controls replenishment: a later finished read frees a slot even
while an earlier read is stalled. Bounded result slots restore input order before
returning, including choosing the first input error. All admitted operations
finish before an ordinary read error is returned.
The directory owner defines the range/byte limits once; store and searcher
planning reuse those constants rather than maintaining independent policies.
Dropping the future cancels unstarted reads; callbacks must own resources used
by already-started external work until its completion. Existing filesystem
callbacks move their file owner and allocation into `spawn_blocking`.

Stored-document preparation deduplicates compressed blocks within a bounded
32-hit window. All segment plans are collected before I/O; the directory layer
executes them together under one eight-read limit, including when hits are in
different files. It excludes cached blocks and requests at most 8 MiB of compressed
bytes across segments. This caps read-ahead, not total per-document I/O or decoded
size. Backend buffer pools and cache allocations require their own accounting.
Blocks beyond this read-ahead allowance use the ordinary demand-read path; no document is dropped. One document is decoded and charged
to the existing response budget at a time. The normal store decoder and shared
cache remain the sole owners of decompression and cache admission. Oversized
blocks keep the existing bounded demand-decompression behavior.

The RPC checks the core reader capability once and skips all batch-planning
allocations for mapped/RAM stores. Mapped handles bypass payload preparation:
a mapped slice does not complete disk I/O,
and a temporary slice collection would only add warm-path overhead. Lazy native
and WASM handles use the same bounded preparation protocol. No native threads,
Linux dependencies, format versions or configuration defaults are introduced.

The prepared batch borrows its searcher and segment readers. It cannot outlive
the generation that owns its files. Results retain input order, duplicates,
missing/deleted-document behavior and field/vector semantics.

Cost: O(32) metadata per window and up to 8 MiB of requested compressed ranges,
plus backend backing-allocation ownership and the existing single-block demand
read, decode and response memory. RPC search admission remains held through
hydration; standalone callers still own request admission. No
whole-result decoded document buffer, second persistent cache, or query-specific
thread pool. RPC `load_us` includes preparation; `store_get` only times the subsequent individual
get and can exclude I/O already performed by preparation. Use the complete load
phase and directory read metrics when comparing latency.

## Explicit payload handles

`Directory::open_payload` identifies files whose consumers permit asynchronous
range reads. Its default delegates to `open_lazy` on native and WASM backends;
all existing backend defaults and persisted bytes stay unchanged. The segment
reader uses this entry point only for the document store. Term dictionaries,
postings, positions and vector readers keep their current open methods and
synchronous capabilities. A directory can therefore opt stored blocks into an
explicit-read service without replacing the scorer or store decoder.

Caching wrappers forward the payload role to their inner directory. The slice
cache retains its existing cache owner and eviction policy, wrapping the opened
payload handle rather than reopening its pathname for each miss. This retains
the selected backend and immutable file owner through outstanding reads. No
second cache, reader pool, parser or file-extension dispatch is introduced.
The mixed mmap/positional test backend demonstrates this seam; it is not a
production io_uring service or a new runtime setting.

## Subsequent Linux backend

A directory-owned, process-shared service must provide bounded queue slots,
in-flight bytes, descriptor registrations and buffer pools. Reader generations
and buffers must remain owned through CQE completion after cancellation, panic,
retirement and deletion. A shared service can combine the independently polled
range futures into submission batches. Async cancellation acknowledgement alone
does not prove the original read no longer owns a buffer.

Select explicit reads for cold payload handles while keeping metadata and sync
scoring mapped. Replacing every `open_lazy` result would break synchronous
posting access. Start with stored blocks; selected ANN/Seismic extents are
separate consumers. Buffered reads, direct I/O and polling are separate
experiments. Capability errors and backend selection must be observable.

### Proposed first service: owned, unregistered reads

This historical proposal preceded the implemented service described below. The measured persistent
positional pool is competitive with the ring on cold storage, and registration
did not produce a consistent warm win. Start a future service with ordinary
buffered reads and transfer its completed `Vec<u8>` directly into `OwnedBytes`.
The current byte owner supports vectors and mappings; it has no safe return-to-
registered-pool owner. Adding registered buffers first would require a copy or
a new pooled lifetime protocol and a separate residency budget. The Rust
[registration safety contract](https://docs.rs/io-uring/0.7.14/io_uring/struct.Submitter.html#method.register_buffers)
and Linux [resource registration rules](https://man7.org/linux/man-pages/man2/io_uring_register.2.html)
require buffer/resource lifetimes beyond individual future polls.

One shared service should accept the same offset/length requests from
`open_payload` handles for either a positional control or an io_uring worker.
Keep the worker count fixed, with a bounded admission queue and byte permits
acquired before allocation. The current 32-range/8-MiB preparation limit is
per batch and **does not** provide a process-wide service limit. Each service
request must retain its open file, owned allocation, generation ownership and
admission permits through completion. No per-index pool, pathname reopen on a
miss, second store decoder, or process-wide unbounded completion cache.

| Request state           | Owner and release rule                                                                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Waiting for admission   | The caller owns metadata only; dropping it starts no I/O.                                                                                                                      |
| Admitted, not submitted | The service owns the request and permits; a cancelled receiver can be discarded before submission.                                                                             |
| Submitted               | The worker owns file and buffer; dropping the response receiver cannot release either.                                                                                         |
| Cancel requested        | Track cancel completion separately; only the original read completion ends the kernel's buffer ownership.                                                                      |
| Read completed          | Validate exact length and release the submission slot. Keep the byte permit with the completed reply until the caller takes it or drops it; a departed receiver discards both. |
| Draining                | Reject new work, settle all submitted reads, then release resources and join the worker through the existing deletion/shutdown lifecycle.                                      |

Completed replies must not form an uncharged buffer queue. Once a caller takes
the bytes, its existing batch/cache/response budgets own retained-memory
accounting; standalone callers still own their admission. The reactor must not
block while delivering a reply. Queue-full admission must remain cancellable. A request larger than the service
byte limit must fail before waiting for permits; demand blocks larger than the
8-MiB preparation allowance need an explicit bounded service policy, not a
permanently pending admission or a silently omitted document. Interrupted/partial submissions
must retain unsent entries and account for submitted entries exactly once.
Unexpected worker failure must close admission and reach a proven drain before
releasing live allocations; the standalone probe's process-abort policy is not
a production recovery implementation. Ordinary read errors are returned after
admitted peers settle. Backend unavailability must be an explicit error or an
observable, configured fallback. Runtime capability selection and lifecycle
integration remain work before a production backend can be enabled.

## Validation and performance gate

Test whole-batch admission before callbacks, slice offsets, duplicate/empty
ranges, short reads, out-of-order completions, errors and cancellation. Test
stored batches against ordinary hydration with multiple blocks/segments,
duplicates, field selection, deleted/missing documents, and cache disabled.
The search `full` harness and WASM build are required for RPC wiring.

Measure mmap demand reads, bounded positional reads, and io_uring on identical
larger-than-memory Linux data: cold/warm p50/p95/p99, QPS, CPU, faults, bytes read,
RSS, queue depth and merge interference. Include warm no-hydration controls.
Until that comparison exists, this refactor is enabling work, not a performance
claim or a reason to change the default backend.

### Hydration experiment gate

The optional standalone [hydration probe](../scripts/experiments/io_uring/README.md)
connects `open_payload` to a bounded owned-buffer service and exercises `Searcher::prepare_document_reads` plus the canonical
stored-document decoder. It compares mapped demand hydration, existing filesystem
demand/batch reads, a persistent positional pool and an ordinary (unregistered) ring.
Use identical document addresses and serialized document equality, warm/cold
advice, and cache-disabled/cache-hit controls. Include mixed files, dropped
receivers, short reads, impossible byte admission and draining shutdown.
Diagnostic process-abort-on-submission-failure remains distinct from production
failure recovery. Results must report achieved ring batch sizes; polling several
futures does not by itself guarantee one submission batch.

### Scheduling follow-up (diagnostic only)

The first ring waits for every read in a gathered batch before replenishing it.
The next experiment compares that drain policy with completion-driven reuse of
eight stable owned slots. A slot is reusable only after its original CQE; queue,
byte admission, exact-length validation and reply ownership remain shared.
Use one ring worker implementation with a scheduling policy, rather than two
buffer-lifetime protocols. Count submissions made while older reads remain live.

The current range callback exposes one request at a time. To isolate transport
batching without adding a production API solely for a benchmark, a diagnostic
future wrapper may collect same-service requests during one poll and hand them
to the worker together before returning from that poll. Its thread-local scope
must be cleared before yielding, must not gather other services' requests, and
must retain no unsubmitted jobs across polls. This neither waits for future
requests nor bypasses the canonical store planner or admission limits. Treat it
as an experimental adapter, not a production batch interface.

The historical experiment retained the positional pool and drain-ring controls; compare refill with and
without poll-scoped handoff. Check cancellation, short reads, mixed files,
queue shutdown and all permit recovery for each scheduling mode. Measure both
single-CPU and multi-CPU execution, then concurrent callers with bounded result
retention and verification outside timed hydration. No registered resources,
format changes or production backend selection belong to this ablation.

### Next experiment: real corpus under memory pressure

This follow-up was measured on September 26; the protocol below remains the
reference for its controls. Separate queue/service latency and concurrent merge
interference remain unmeasured. The retained 10M
document comparison index contains 4,455,675,505 bytes of stored payload within
18,693,711,229 total bytes. Its existing stored `body` field permits a realistic
hydration experiment without introducing another document writer or format.

Use an isolated copy with verified file hashes and a dedicated Linux memory
cgroup. Start with 1/2/8-GiB limits, recording actual metadata residency, page-cache
charges, reclaim and OOM events; these are experimental limits, not defaults.
Follow the kernel
[memory ownership rules](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-ownership):
ensure copied pages are charged to the measured cgroup. Reusing pages already
charged to another group would invalidate the intended memory-pressure control.
Any cold advice must target only the private fixture; never drop host-wide caches.

First replay fixed document-address traces through `prepare_document_reads` and
the canonical decoder. Then measure query plus hydration with the same query,
field selection, top-k, request concurrency and response limits for each backend.
Retain a retrieval-only control. Use the existing mapped metadata path and the
same `open_payload` service for pool/ring reads. Bound total outstanding service
reads and bytes equally across callers; per-batch filesystem limits alone are
not an equal-budget concurrent control.

Report sustained throughput, CPU, physical bytes, faults, reclaim, payload-cache
hits, achieved queue/submission sizes and memory alongside verified documents.
Collect enough completed requests for any reported latency percentiles. Separate
queue wait, service time and end-to-end time. Add concurrent cold merge output
only after the isolated baseline is repeatable. Preserve the existing lifecycle
checks and evaluate CPU cost as well as latency against the positional pool before
changing a backend default.

### September 26 experiment implementation

The standalone hydration probe now has a real-corpus mode. It
reuses the same `hydrate`, index configuration, payload directory and bounded
pool/refill service; it adds no production backend or decoder. An immutable
oracle records deterministic document addresses, BLAKE3 hashes of canonical
decoded documents, and query result addresses/raw score bits. Verification runs
outside operation timings. Each process uses four callers, batches of at most
32 documents and zero application store cache; pool/ring share the existing
eight-read/eight-MiB global admission limits. Mmap remains its synchronous
control, with no claim that page-fault concurrency equals asynchronous depth.

A private copy is required by an explicit fixture marker. The external Linux
runner owns cgroup limits and private-file cache advice, and records reclaim,
OOM/pressure and process memory alongside CPU and physical reads. Fixed address
replay spreads across the corpus; query-plus-hydration and retrieval-only
controls use a declared smaller set of existing queries and must not be described
as a corpus-wide payload working set. Per-file mappings and canonical segment
memory statistics also provide evidence for the separate memory investigation.
These are experiment controls, not production defaults. The
[measured outcomes](benchmark-results/closing-gap-2026-09-24/README.md#real-corpus-payloads-under-memory-pressure)
separate address replay, smaller query working sets, cold starts and warm passes.

The first real-corpus attempt exposed a diagnostic admission mismatch: the store
reader opens its canonical block directory with one contiguous read larger than
the service's eight-MiB in-flight budget. The diagnostic adapter satisfies
such reads sequentially through the same admitted read owner, with a separate
64-MiB maximum returned range. It retains an assembled response plus one admitted
chunk; the returned-response cap is not an increase to the in-flight I/O budget.
Ordinary reads at or below eight MiB keep their existing direct path. Both pool
and ring use this adapter, including short-read errors and cancellation ownership;
the core store format/parser and production backends remain unchanged.

### September 26 measured decision

All 48 corrected real-corpus cells pass, including a 512-MiB follow-up after the
original 1/2/8-GiB matrix showed that explicit reads touched only 743 MiB.
Under 512-MiB pressure the positional pool completes the two-pass trace in
10.05 seconds versus ring 10.78 and synchronous mmap 93.53. The ring uses 21.6%
less operation CPU than the pool but takes 7.2% longer. At larger budgets its
CPU advantage is 24–25%, with 3.4–4.4% longer total trace time. Warm mmap remains
fastest. These controls combine range targeting and concurrency changes;
they do not isolate an io_uring API speedup over mmap.

Keep the production payload seam and existing defaults. The next backend
decision needs production RPC/queue latency and mixed-workload evidence.
The positional pool remains the required elapsed-time control; the ring's CPU
saving is worth retaining as a measured tradeoff. No BMP scoring improvement
has been measured.

## September 26 maintenance: first-class payload service

The maintenance implementation moves payload scheduling into the directory owner.
`PayloadReadService` is shared explicitly between `MmapDirectory` instances;
metadata and synchronous scoring retain mapped handles. Backend selection is
explicit, with a persistent positional pool and an optional Linux io_uring
backend. Unsupported kernels return an error; there is no silent fallback. The
server default remains mapped. CPU savings are a valid reason to select the
ring even when its latency is slightly higher than the pool. Future comparisons
must report both, using the same service budgets.

One service owns admission (eight requests and 8 MiB), its bounded queue and
worker lifetime. Replies retain byte admission until consumed or discarded.
Larger contiguous demand reads assemble at most 64 MiB from bounded chunks;
caller/cache budgets own returned bytes. All jobs own their open file and buffer
through completion; cancelling a caller only discards the reply. Shutdown closes
admission, drains accepted jobs, and joins workers. No per-index worker creation
is implicit in opening a directory.

The ring requires Linux synchronous cancellation support (6.0+) for exceptional
teardown. Only the worker submits, with no SQPOLL or registered buffers. An
exceptional exit synchronously cancels submitted operations before destroying
slots; unsent SQ entries cannot execute once the ring is closed. If the kernel
cannot prove cancellation, the failed service retains its bounded live resources
for process lifetime and reports failure, rather than freeing kernel-owned
memory. This exceptional quarantine is observable and never reopens admission.
Normal shutdown drains original completions.

The service transfers owned vectors without copying. Optional reusable buffers
return after the final byte-view owner, not the completion or response future.
Registered buffers remain a separate, unimplemented optimization. The
service API must not force a second decoder, cache or scorer to enable that work.
Historical probe source snapshots remain immutable evidence. The maintained
probe uses the core service; old drain/poll-local scheduling variants are retired.

### Using and validating the shared service

Create one `PayloadReadService::new(PayloadReadBackend::IoUring)` and share its
`Arc` with `MmapDirectory::with_payload_reads` for each index. Compile core with
`--features io-uring` on Linux. `Pool` uses the same API on native platforms.
The server exposes `--payload-io mmap|pool|io-uring`. Its `io-uring` Cargo feature
forwards the core feature; unsupported selections fail at startup. Metadata and
explicit synchronous scoring reads remain mapped. `--sparse-payload-reads`
separately opts asynchronous MaxScore blocks into the same service; it requires
pool or io_uring and defaults off. BMP/Seismic scoring payloads remain mapped. `--payload-buffer-mb 0..8` opts into idle buffer reuse for
the explicit backends, with zero as the default. A nonzero budget with mmap is
rejected. The registry shares one service across create/open operations.

Existing segment cleanup waits for accepted directory reads before unlinking.
After the ordinary writer/manager drain, whole-index deletion calls
`directory.retire_payload_reads().await` before directory removal. Library
callers follow that same order. Process shutdown calls `service.shutdown().await`
after all index managers drain. Counters, buffer reuse/allocations and idle bytes
are available through `stats()` and logged at server shutdown.

`file_handle` owns range validation/scheduling; `owned_bytes` owns immutable
views; `payload` owns admission and lifecycle; its `pool`, `ring`, and `stats`
modules own backend execution and observation. Public directory reexports are
preserved. Both positional consumers use `local::read_exact_at`. Every lazy
range is checked for exact response length, including single reads. Invalid
views and reversed ranges cannot reach the backend or trigger payload allocation.

Run the ordinary full harness and WASM checks, plus
`python3 scripts/check_search.py io-uring` on a supported Linux host. That mode
runs strict Clippy, directory/lifecycle regressions and canonical multi-segment
document-batch tests with the ring enabled. It must fail if the requested backend
is unavailable, including container seccomp restrictions.

The exceptional drain uses the documented
[synchronous cancellation API](https://docs.rs/io-uring/0.7.14/io_uring/struct.Submitter.html#method.register_sync_cancel).
It applies only to submitted requests; the worker owns the unsent SQ entries and
never enables SQPOLL. A failed drain retains allocations and reports failure.

### HTTP transport ownership cleanup

HTTP range transport has one implementation for direct and lazy reads. The
transport no longer retains its own whole-file cache beneath the existing
bounded `SliceCachingDirectory`/`CachingDirectory` owner; standalone transport
reads may therefore issue a new request on each call. Production WASM callers
already use the bounded slice wrapper. Returned byte views retain their ordinary
ownership independently of cache eviction.

Network diagnostics retain the latest 256 operations in a deque, alongside exact
cumulative request/byte counters and an explicit omitted-operation count. One
lock owns recording, snapshots and reset so concurrent reset cannot split the
history from its totals. This bounds historical metadata by 256 request URLs,
not by index/query lifetime. The diagnostic JSON gains `omitted_operations`;
index bytes and search responses are unchanged. No search results are truncated.

## September 26 follow-up: server lifetime and reusable buffers

The server explicitly selects `mmap` (unchanged default), `pool`, or
`io-uring`. A registry owns one shared service. Directory clones share a read
lease gate: open operations and accepted jobs retain a read lease through actual
completion, including after caller cancellation. Existing segment cleanup takes
an exclusive lease before unlinking; it may briefly wait for reads of other
files in the same index. This adds no second segment retirement protocol.
Whole-index deletion retires that directory after its existing writer/manager
drain, rejects later opens/reads, and waits for its leases. Other indexes remain
usable. Process shutdown additionally drains the shared service. Failed kernel
cancellation forbids deletion, even after error replies have been delivered.

A bounded buffer recycler is shared by both backends. It retains at most
eight allocations and 8 MiB of idle capacity, separately from in-flight admission
and caller-owned bytes. Power-of-two capacity classes are reused; admission charges backing capacity
rather than hiding padding behind small requested reads. Idle entries are evicted
oldest first to make room for recent returns. New allocations remain initialized; reused buffers
need no clearing because only a successful exact read publishes them.
`OwnedBytes` retains an immutable recyclable owner; only its final clone/subview
drop may return storage. Returned views never retain a service or admission
permit. A weak recycler reference prevents service lifetime cycles. Shutdown
clears idle storage and disables further returns. A zero idle budget is the
allocation-only comparison; reuse stays opt-in pending matched measurements.
No file format, scorer, decoder or wire response changes.

### Measured follow-up: size classes and idle eviction

The first matched corpus cells found zero timed buffer reuse: startup metadata
left an eight-MiB allocation in the idle pool, and rejecting smaller returns
preserved that stale allocation indefinitely. A regression models that sequence.
The revised recycler evicts the oldest idle allocations to accept recent returns.
Power-of-two capacity classes permit reuse between nearby payload lengths; a
class larger than the configured idle budget bypasses recycling and keeps exact
allocation. Admission charges backing capacity before allocation. Successful
reads expose only the requested length, and reused initialized storage needs no
clearing. Short/error completions never expose old tail bytes.

Idle retention remains at most eight allocations/eight MiB. Admitted storage
remains at most eight MiB; class padding consumes that budget. Each returned range retains less than twice its original requested byte length,
separately from the idle budget; further slices retain that same allocation. A prepared eight-MiB logical window can therefore retain less
than sixteen MiB of raw backing when reuse is enabled. Zero reuse budget retains
the exact-sized allocation control. Compare the revised policy against both
that control and the recorded rejected exact-capacity policy; defaults stay zero.

Cancellation before registry publication also requires a drain: the cancelled
opening future can disappear while a blocking open/read job remains. The service
therefore shares directory gates by root path, using weak references to the gate
itself (owned read guards retain it). Expired entries are pruned when attaching a
directory; the map does not grow with past index names. Deletion acquires that
same gate even without a cached index handle. A retired gate is replaced only
when attaching a new directory generation, so stale handles remain closed while
recreated indexes can open. This is directory I/O ownership within the existing
per-name server delete transaction, not a second segment lifecycle.

### Final follow-up evidence

The [78-cell comparison](benchmark-results/payload-service-2026-09-26/README.md)
verifies 7,013,376 documents with unchanged fixture hashes and no service errors
or OOM events. The retained ring saves 23–24% operation CPU relative to the pool;
its address trace takes 4.9–6.9% longer without reuse, while query-plus-hydration
is 2.7% faster. Size-class reuse has mixed performance results and stays opt-in.
Server mmap and zero idle-budget defaults are unchanged. Both benchmark/build
VMs were explicitly stopped and independently confirmed terminated.

## Hybrid sparse reads (September 26)

The implementation keeps the sparse file's metadata and synchronous source
mapped while routing asynchronous MaxScore block reads through the same selected
payload service as stored documents. The loader opens one additional payload
handle per sparse component only when it has MaxScore fields and its directory
opts into `open_sparse_payload`. Inline payload handles retain the original mapped path;
explicit handles are shared across the component's MaxScore fields. Metadata
pinning continues to operate on the original skip-section views. No format,
scorer, block decoder, cache, or lifecycle protocol is added.

The async search entry points must honor this capability on both current-thread
and multithread runtimes, including fusion and candidate-list retrieval. Ready
CPU work uses the existing shared search pool; an I/O wait releases that worker.
Explicit synchronous block APIs continue reading the mapping. Bulk raw-dimension
reads for merge retain their original source and are not subject to demand-read
limits. The service's existing request/byte admission bounds allocations before
I/O. Existing per-cursor block ownership remains unchanged.

This first extension covers MaxScore blocks. BMP and Seismic still dereference
mapped payloads during scoring; changing them requires staged block preparation
around the existing scorer, not a copied async scorer. Their stored-document
hydration already uses the shared service.

Measure mmap, pool and ring with metadata pinning disabled and with successful
mlock under equal cgroup limits, swap disabled, identical queries and compiler.
Record logical intended/pinned bytes and actual VmLck; pin failure invalidates a
claimed pinned cell. Keep pinning's existing per-segment/generation budget and
defaults unchanged. Use sparse data with nonzero eligible metadata; the previous
text-only fixture cannot establish a metadata-pinning benefit. Include exact
query-score equality, memory, faults, CPU and elapsed time. No automatic residency
probing or hot/cold migration is introduced.

### Selection after the first hybrid measurements

The exploratory matrix found lower memory use but higher CPU and elapsed time
when explicit reads also covered MaxScore blocks instead of documents alone.
Consequently, preserve `--payload-io` as the stored-document selection and add
`--sparse-payload-reads` as a separate opt-in that reuses the same service. It
requires pool or io_uring. `Directory::open_sparse_payload` returns an optional
explicit handle; its default returns none without reopening the mapped source.
Cache wrappers forward this role using their existing cache owner. Mmap opts in
through `with_sparse_payload_reads`; ordinary `with_payload_reads` keeps sparse
blocks mapped. This is a role selection, not another I/O backend or budget.

### Concurrent sparse follow-up protocol

Before widening sparse coverage, measure the existing implementation on the same
immutable fixture with one and eight concurrent requests, current-thread and
four-worker runtimes, and both homogeneous MaxScore and mixed MaxScore/BMP work.
Keep the process-shared eight-read admission limit. Measure process CPU once per
batch, never sum overlapping per-request process counters. Verify results after
the timed batch, retain per-request latency and batch throughput separately, and
record service read counts, bytes, batch depths, memory and context switches.
Profile representative runs separately from latency measurements.

The optimization hypothesis is that small serialized block reads and repeated
CPU-pool handoffs dominate explicit MaxScore reads. Any preparation window must
be bounded per cursor and query, retain existing pruning and score order, and
reuse the reader's decoder and payload service. It must not load whole posting
lists or introduce a second cache/lifecycle owner. The measured window below is retained; broader preparation remains a proposal.
BMP/Seismic preparation
and process-wide pin admission remain separate follow-ups; neither should be
enabled as an incidental side effect of optimizing MaxScore reads.

The retained follow-up coalesces at most eight adjacent blocks and 16 KiB in the
existing sparse cursor. Only explicitly opted-in payload handles use this
lookahead; mmap and public single-block APIs retain demand reads. The reader
validates the range and uses the existing block decoder; the cursor owns one
encoded window, released on replacement/drop. At the 64-term limit, retained
lookahead is at most 1 MiB per active segment scorer, independent of corpus size.
Pruning can skip prefetched blocks, so bytes/read amplification must be reported
alongside fewer submissions. Oversize individual blocks keep the existing
single-read path and service limits. Failed or cancelled refill cannot publish
partially initialized bytes. The [concurrent comparison](benchmark-results/sparse-concurrency-2026-09-26/README.md)
validates this change with identical probe code and an interleaved confirmation.

### BMP/Seismic preparation boundary review

BMP enters `query/planner.rs` and `execute_bmp_inner`, which borrows thread-local
`BmpScratch` for the complete synchronous traversal. `BmpIndex` owns mapped
block bytes plus the D/E grids. An asynchronous extension must move scratch into
an owned execution state before suspension, separate bounded window selection
from payload acquisition, then feed the existing scoring kernels. It must also
prepare selected grid payloads; routing only block bytes leaves cold grid faults
in the scoring phase. Keep threshold updates and superblock visit order intact,
and do not hold a thread-local `RefMut` across an await.

Seismic's synchronous entry point owns nomination and then calls its shared
exact forward scorer. Stage bounded selected summaries/nomination runs and
forward rows around those existing phases; preserve candidate limits, cut/factor,
ordinal combination and approximation semantics. A MaxScore encoded window is
not a reason to preload complete BMP blobs or Seismic runs. These extensions
remain proposals, not capabilities provided by `--sparse-payload-reads`.

## Ring worker experiments (September 26, experimental only)

Keep the retained service unchanged while testing disposable source variants:
one ring (control), two/four rings sharing the existing admission and job queue,
eventfd notification of queued reads while a prior read is pending, and
cooperative kernel task processing. Multi-ring variants divide the eight active slots into per-worker
limits; aggregate reads/bytes, returned-byte ownership, cancellation and deletion
leases remain shared. Construct all rings before starting workers; any partial
worker-start failure closes admission, wakes receivers and joins started workers.
No per-request ring, unbounded polling, registered-buffer pool or new decoder is
introduced. Compare pool and ring on the same sparse fixture, compiler, CPU set
and memory cap. Record warmed and initially cold phases separately, batch wall,
request percentiles, process CPU, memory, service counts and exact oracle matches.
Only repeatable measured gains should be proposed for the production backend.

The [96-cell experiment](benchmark-results/ring-workers-2026-09-26/README.md)
finds four rings roughly level with pool throughput on this fixture, about 9%
faster than the retained single ring under eight-request MaxScore load. Queue
wakeups improve initially cold sequential work while preserving most of the
single-ring CPU saving. Four rings increase sequential CPU; wakeups have no
clear document-only benefit. Both were disposable-source experiments in that campaign; the single-ring
wakeup path is now hardened below, while extra workers remain experimental. The wake prototype uses a nonblocking eventfd and one
control poll SQE, handled separately from read completions and retaining its
descriptor through ring teardown. The retained version below adds notification-failure and startup-race
tests to the existing cancellation/deletion suite. Further measurements must include sustained load,
slower storage and merge interference before choosing a worker policy.

## Single-ring queue notification hardening (September 26)

The implementation keeps one worker and the current eight-read/eight-MiB
admission budget. Public entry remains `PayloadReadService::read`, called by
explicit payload handles attached to mmap directories; native synchronous and
WASM mapped reads do not use this Linux-only worker. No stored/wire format or
scoring implementation changes. The service owns the producer notification
handle; the reactor retains the same eventfd through ring teardown. One level
triggered poll SQE has a separate control token and never owns a payload buffer.
Publish each job before notifying, with no await between publication and notify.

A ready eventfd remains readable if notification precedes poll registration.
Control CQEs drain the eventfd and rearm on the next wait. Notification failures
close the queue and admission immediately, then poison the worker through a
shared atomic flag. Waiting uses the ring's extended enter timeout (100 ms) so
failed notification cannot leave it asleep behind an unrelated pending read.
This is a wakeup-health check, not a payload-read deadline: healthy slow reads
retain their ownership and continue waiting. Existing worker failure, synchronous
kernel cancellation, quarantine and join logic own exceptional cleanup. There
is no second cancellation protocol or automatic backend fallback.

Cost: one eventfd write per accepted ring job, at most one control SQE, and at
most ten timeout wakeups per second while reads remain pending without CQEs.
Idle workers still block on the shared job channel. Read/control completion and
notification-failure counters remain separate. Probe required read/poll opcodes
and extended-enter support before spawning the worker. Tests cover notices
before/after poll submission, saturation/interruption/short transfer, notification
failure with pending reads, cancelled callers, teardown with an armed poll and
startup resource ownership. The validation and comparison use identical
probe/fixture and compiler settings, with full native, Linux ring and WASM checks.

Failure regression uncovered during hardening: a closed `async_channel` can keep
queued jobs alive as long as the service's sender remains. The shared worker
error/panic wrapper must close admission and explicitly drain that queue before
joining completes. Unsubmitted jobs release their reply, byte/slot permits and
directory lease together; count them as skipped. This applies to both backends.
The reactor must also check notification failure when an idle closed queue ends
its receive, so a failure racing a fast completion cannot look like successful
shutdown. Neither change alters successful pool reads or payload formats.

The [hardening report](benchmark-results/ring-wakeup-2026-09-26/README.md) records
the regressions, final-source validation and comparison against the original
ring and pool. Backend, worker-count and sparse-read defaults remain unchanged.

## Sustained-load measurement (September 26)

### Follow-up: warm-path overhead

Profile the existing fixed-arrival workload before changing production code.
Separate the shared async scoring/pool handoff cost from ring control work.
Disposable notification coalescing is a candidate: publish the job first, then
write eventfd only on the first pending notification. The single consumer must
drain the descriptor and clear the pending flag **before** checking the job queue
again. Jobs published before that clear must be observed by that queue check;
jobs published afterward must signal. A failed write still poisons admission and
the worker; coalescing must not add an await or another lifecycle owner.
The proposed cost is one atomic operation per notification and one syscall per
notification group, with unchanged buffer, queue, and reader budgets. Retain only
after deterministic boundary/failure tests and same-boot comparisons show a
repeatable benefit. This paragraph describes an experiment, not current behavior.

The [warm-path comparison](benchmark-results/warm-io-2026-09-26/README.md)
retains a different optimization: expand the existing cursor advance/seek macros
directly in the shared MaxScore loop. Previously each call constructed a nested
async future even when the decoded block could answer immediately. Cursor
methods and MaxScore still share one navigation implementation, preserving seek
accounting, block transitions, lazy ordinals, errors and cancellation at actual
reads. Direct async wrappers now compile only for their regression tests. The
change removes a future boundary and adds no retained memory or prefetch.
Exact scores, result bytes and query results remain equal. Ring notification
coalescing did not improve performance and remains a discarded experiment;
production keeps one eventfd write per accepted ring job.

The existing sparse probe supports a fixed arrival schedule for evaluating
further reactor optimizations. It reuses the fixture, query execution, hydration
and exact oracle checks. The driver admits at most 32 tasks, records rejected offers
when full, and retains at most 65,536 compact observations. Bound the arrival
window to 120 seconds and validate settings before opening the fixture. Schedule
from one monotonic origin, so delayed generation remains visible and does not
silently lower the offered rate. Record intended arrival, admission, first poll,
response completion and verification completion separately. Drain accepted work
before shutdown; errors fail the run.

Unlike the wave probe, sustained operation includes per-request oracle checking
in its process CPU and occupancy. Report that cost separately as elapsed
verification time; do not subtract overlapping task times from process CPU.
Response latency ends after canonical hydration, before checking the oracle.
This is a bounded in-process load generator, not HTTP or a production arrival
trace. Compare mmap, pool and the retained ring at several offered rates with
identical fixture, binary, CPU/memory limits and repeated interleaved orders.
Track correctness, admission loss, generator delay, p95/p99, CPU, memory and
logical I/O. Keep all service, worker, buffer and residency defaults unchanged.

The [fixed-arrival report](benchmark-results/sustained-io-2026-09-26/README.md)
records the bounded load-generator protocol, acceptance and queue-delay checks,
and the same-binary comparison. This adds measurement capability only.
