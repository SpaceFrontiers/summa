# IResearch optimization and Linux I/O audit

Audit date: September 20, 2026. Summa source:
`15b8a3683fd563f03f1b4f8b14310c761303734e`. This checks the mechanisms in the
maintainer's Search Benchmark Game description against current Summa call paths.
It is not a new engine benchmark, an exhaustive IResearch review, or a claim that
implementing each mechanism would make Summa faster. No runtime/default changes
are made by this audit.

## Source identities and methodology

- SereneDB's benchmark fork: [`2e1b99be11526baff1af056bd8911cad220f26e5`](https://github.com/serenedb/search-benchmark-game/tree/2e1b99be11526baff1af056bd8911cad220f26e5).
  Its IResearch gitlink pins SereneDB `8030e488b871ca8bfd4f2e8115e3efa57be8f4dc`.
- Current SereneDB source inspected separately:
  [`fd6d6cacf2e79fd399ed03874c27e841138871f4`](https://github.com/serenedb/serenedb/tree/fd6d6cacf2e79fd399ed03874c27e841138871f4).
  IResearch has moved from `libs/iresearch/include/iresearch` to `iresearch`.
- The fork's [driver](https://github.com/serenedb/search-benchmark-game/blob/2e1b99be11526baff1af056bd8911cad220f26e5/src/client.py)
  defaults to 60 seconds warmup, ten workload passes, and seed 2. The
  [UI](https://github.com/serenedb/search-benchmark-game/blob/2e1b99be11526baff1af056bd8911cad220f26e5/web/src/index.js)
  computes the upper middle sample of sorted timings as its median. Older README
  language about selecting the best run is not the current driver/UI definition.
- The [benchmark adapter](https://github.com/serenedb/serenedb/blob/8030e488b871ca8bfd4f2e8115e3efa57be8f4dc/tests/bench/search-benchmark-game/executor.hpp)
  uses `1_5simd`, BM25, the segmentation analyzer, and `MMapDirectory`. It parses and
  prepares queries per execution; no answer-cache lookup appears in that path.
  A warm page cache is intentional and is different from an answer cache.
- IResearch's [build wrapper](https://github.com/serenedb/search-benchmark-game/blob/2e1b99be11526baff1af056bd8911cad220f26e5/engines/iresearch/Makefile)
  supports a normal benchmark build and separate PGO workflows. Their availability
  does not establish which flags produced a particular published result.

The interactive sites do not provide a plain-text leaderboard to the reader used
here. The quoted universal-win statement is not independently re-established.
Summa's [existing benchmark](search-benchmark-current.md) uses its own pinned
corpus, controls, and timing protocol; do not compare unrelated microsecond tables
as if measured on the same machine or scoring/analyzer configuration.

## Coverage summary

| Advertised mechanism         | Summa status                                                | Concrete finding                                                                                                                                             |
| ---------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Vectorized scoring / AVX2    | Batch structure present; production ISA coverage incomplete | Contiguous BM25 scoring and score masks exist; BM25 uses ordinary Rust loops, while portable release builds do not request AVX2                              |
| `nth_element` top-k          | Missing from main text collectors                           | Both `ScoreCollector` and `TopKCollector` retain binary heaps; fusion's partial selection does not cover text collection                                     |
| Adaptive posting compression | Partial                                                     | Four codecs and per-block widths exist; the writer does not choose among IResearch's constant-gap, bitmap, StreamVByte, and packed representations per block |
| Lazy sparse evaluation       | Substantially present                                       | Deferred TF, selective intersections, candidate-first phrases, prepared score bounds, and bounded score windows are active                                   |
| No JVM                       | Already true                                                | Summa executes native Rust; this is not a remaining implementation task                                                                                      |
| `io_uring`                   | Absent                                                      | Server search uses mmap, lazy filesystem ranges use blocking positional reads, writes use buffered/cold file writers                                         |

## Scoring and instruction selection

The pinned IResearch [BM25 implementation](https://github.com/serenedb/serenedb/blob/8030e488b871ca8bfd4f2e8115e3efa57be8f4dc/libs/iresearch/include/iresearch/search/bm25.cpp)
scores contiguous arrays using prepared coefficients. Its
[collector](https://github.com/serenedb/serenedb/blob/8030e488b871ca8bfd4f2e8115e3efa57be8f4dc/libs/iresearch/include/iresearch/index/iterators.hpp)
explicitly screens groups of eight scores with AVX2 before selection. These are
separate mechanisms: vectorized arithmetic and vectorized admission.

Summa's [score_text_run](../summa-core/src/query/scoring.rs) gathers lengths and
scores contiguous frequency arrays through the canonical
[Bm25Params](../summa-core/src/query/bm25.rs). Byte norms have a bounded lookup
table and batch path. Ranked collection screens eight scores before heap insertion;
the generic top-k collector screens a 64-score membership window. Posting decode,
integer search, and dense-vector kernels also have SIMD paths, but their presence
does not prove the BM25 arithmetic uses AVX2.

The BM25/admission loops rely on compiler vectorization. The measured benchmark
builds request native CPU code generation. [Dockerfile.release](../Dockerfile.release)
and [server Dockerfile](../summa-server/Dockerfile) run ordinary release Cargo
builds without `target-cpu` flags. Runtime AVX2 dispatch exists for other kernels;
there is no equivalent target-feature BM25 batch dispatch in this path.

Finding: batch scoring is implemented, but AVX2 BM25 in every distributed binary
is not established. Inspect deployed x86 assembly and profile before choosing
runtime multiversioning or an explicitly documented CPU baseline. Do not silently
build on a CI host with `target-cpu=native` and ship an accidental ISA requirement.
Preserve exact f32 scoring, zero-TF behavior, boosts, missing lengths, and tie
semantics. IResearch's algebraic expression is not a bit-identical replacement
for Summa's arithmetic. No deployed-binary disassembly was performed in this audit.

## Top-k selection

### Collector experiment

The subsequent [collector benchmark](collector-benchmark.md) compares the
production `ScoreCollector` heap with benchmark-only bounded 2k partial selection
and a loser tree. It preserves total float ordering and document/ordinal tie
breaks and compares ordered result bits against a full-sort oracle. It times
construction, eight-score screening, admission, final sorting, and destruction
on identical fixed-seed streams. Include canonical BM25 scores, improving and
deteriorating score order, ties, short streams, and clustered blocks whose upper
bounds feed back through each collector's threshold. Report retained allocation
capacity and candidate/block counts outside timing. The block experiment isolates
threshold staleness; it is not an end-to-end MaxScore replacement. Partial
selection wins at larger k on the measured ARM fixture, with additional candidate
visits under pruning. See the report for both runs and limitations. No production
collector, seeded-threshold behavior, API, storage format, or default changes.

The benchmark-pinned [document collector](https://github.com/serenedb/serenedb/blob/8030e488b871ca8bfd4f2e8115e3efa57be8f4dc/libs/iresearch/include/iresearch/search/doc_collector.hpp)
uses `NthPartitionScoreCollector` for both top-k and top-k-with-count. It supplies
a bounded buffer of 2k score/document entries. Accepted hits fill the spare half;
selection partitions the buffer and publishes a new score threshold. This avoids
maintaining heap order for every accepted candidate; it is not an all-hits vector.

Summa's [ScoreCollector](../summa-core/src/query/scoring.rs) and
[TopKCollector](../summa-core/src/query/collector.rs) instead update binary heaps,
with threshold screening and in-place root replacement. `select_nth_unstable_by`
in [fusion](../summa-core/src/query/fusion.rs) is a different call path.

Current IResearch has changed again: its
[document collector](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/search/detail/doc_collector.hpp)
uses [LoserScoreCollector](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/index/iterators.hpp).
The remaining `TopKHeap`/`nth_element` helper is used for term selection, not proof
that current document collection follows the older benchmark's algorithm.

Priority experiment: compare the current Summa heap, bounded buffered selection,
and the current IResearch-style tournament/loser-tree approach inside existing
collector ownership. Test k=10/100/1000, rising thresholds, seeded thresholds,
exact counts, stable document/ordinal ties, exceptional floats, and position
collection. A buffer doubles retained-entry storage and refreshes its threshold
less often; fewer selection operations can be offset by weaker pruning. Preserve
counted-search optimizations that avoid scoring every match. Measure complete
queries as well as collector CPU; no winner is assumed.

## Adaptive compression

The benchmark-pinned [format implementation](https://github.com/serenedb/serenedb/blob/8030e488b871ca8bfd4f2e8115e3efa57be8f4dc/libs/iresearch/include/iresearch/formats/formats_impl.hpp)
evaluates each document block/tail and chooses an encoding using its shape and
estimated byte size: raw values, repeated gaps, SIMD delta packing, StreamVByte
variants, or a bitmap over a compact document range. Frequency arrays also have
constant-value choices. Current [FormatBlock128](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/formats/posting/format_block_128.hpp)
retains this design and includes an explicit gap-one case.

Summa [posting codecs](posting-codecs.md) are `Rounded`, `Packed`, `Pfor`, and
`Simd4x`. Widths adapt per block; Pfor chooses its exception width; Simd4x uses a
different tail policy. The main codec family is configured at build time. A
merged list may contain mixed families because sources were copied, not because
the writer ran the same adaptive selection policy. The `adaptive` optimization
setting chooses Rounded and must not be read as equivalent to IResearch's scheme.

Priority experiment: collect block-shape statistics first, then evaluate constant
TF/gap cases, dense document-range bitmaps, and tail coding against the present
codecs. Existing rejected Roaring/Elias-Fano/seek experiments are not evidence for
this precise per-block policy, but they do prohibit treating a familiar codec
name as a demonstrated win. The header's two codec bits are already exhausted;
additional representations require explicit format design and copy-merge support.
Report decode/seek time, index bytes, warm/cold residency, and whole-query latency.

## Lazy phrase and Boolean evaluation

Summa already implements the central idea:

- Text cursors decode document IDs before TFs. Membership/count paths avoid
  frequency work where it is unnecessary.
- [Conjunctions](../summa-core/src/query/scoring/conjunction.rs) order cursors by
  document frequency, intersect decoded blocks, and score bounded surviving batches.
- [PhraseScorer](../summa-core/src/query/phrase.rs) exposes candidates separately
  from positional confirmation. Prepared TF/length bounds can reject losing
  ranked candidates before positions are loaded. Exact phrase matching intersects
  positions incrementally; phrase frequency continues only when scoring needs it.
- [Window execution](../summa-core/src/query/scoring/windows.rs) uses selective
  required candidates, deferred optional work, score bounds, and threshold-driven
  OR-tail transitions. Count remains separate from competitive ranking where needed.

Current IResearch's [pruned conjunction](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/search/top/pruned_conjunction.hpp)
and [pruned phrase](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/search/top/pruned_phrase.hpp)
use analogous selective batches and bound-before-confirmation structure. Summa's
prior [performance review](search-performance-review.md) already records adopting
ideas from those paths and rejecting other experiments after measurement.

Coverage is not universal. Sloppy phrases load their term-position lists before
their interval scan. Generic/cross-field/mapped compositions can take other paths;
the explicit compact conjunction pruning specialization asserts two terms. These
are targeted profiling opportunities, not grounds to replace the existing query
executor or describe all lazy evaluation as missing.

## io_uring: source and running-host findings

There is no `io_uring`, `io-uring`, `tokio-uring`, or `monoio` backend/dependency in
the audited core/server source and lockfile. The [server registry](../summa-server/src/registry.rs)
opens `Index<MmapDirectory>`. [MmapDirectory](../summa-core/src/directories/mmap.rs)
provides borrowed mapped byte views; touching cold pages invokes kernel demand
paging, not an application-submitted async read.

[FsDirectory::open_lazy](../summa-core/src/directories/directory.rs) retains an
open descriptor and performs positional reads in `spawn_blocking`. Ordinary small
writes use Tokio file APIs. Tokio [documents](https://docs.rs/tokio/latest/tokio/fs/index.html)
that its current filesystem implementation uses blocking workers, not io_uring.
Bulk output uses [cold writers](cold-io.md): buffered writes, Linux writeback/drop
advice, fsync, and `copy_file_range` for compatible byte ranges.

A read-only check of the index host found kernel `6.8.0-134-generic`,
`kernel.io_uring_disabled = 0`, and three Summa server processes with seccomp
mode 0. Each had zero open io_uring descriptors at the inspection instant. This
supports the source finding; it is not a syscall trace or proof about every prior
request. No service was restarted or configuration changed. Host details are in
the gitignored audit evidence.

The IResearch comparison does not justify replacing mmap to improve this warm
benchmark. Its pinned adapter uses `MMapDirectory` for both build and query, and
current [AsyncDirectory](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/store/async_directory.hpp)
inherits mmap reads. Its [implementation](https://github.com/serenedb/serenedb/blob/fd6d6cacf2e79fd399ed03874c27e841138871f4/iresearch/store/async_directory.cpp)
submits registered-buffer writes and fsync through io_uring. Backend availability
and benchmark-path use are different facts.

## Proposed Linux I/O work, not implemented

The [Linux API](https://man7.org/linux/man-pages/man7/io_uring.7.html) can submit
multiple asynchronous operations and receive their completions through shared
queues. It is not a switch that accelerates existing mmap dereferences. Evaluate
two concrete paths independently:

1. **Cold candidate payload reads:** after BMP/ANN/Seismic nomination, coalesce
   selected extents and submit bounded batches. A directory-owned service retains
   buffers and file/segment owners until completion, including after cancellation.
   The query consumes explicit owned ranges instead of expecting mapped faults to
   become asynchronous. Keep warm metadata/mapped reads cheap and bound copied
   cache data so a second payload cache does not defeat the RAM constraint.
2. **Bulk writes:** compare bounded queued registered-buffer writes/fsync with
   current cold writers. Preserve short-write handling, final durability ordering,
   cold-cache policy, and ownership through commit/cancellation. Retain efficient
   kernel range copying for compatible encoded blocks where it wins.

Use a shared bounded service rather than a ring per query. Charge queue depth,
in-flight bytes, buffer pools, descriptors, and completion work to process budgets.
Direct I/O and polling are separate choices: neither follows automatically from
using io_uring. Buffered io_uring can still use the page cache, and submitted read
buffers add copying/residency compared with a hot mapped view. Capability failure
must be reported, with the existing backend remaining available; Linux-specific
machinery must not break native async, macOS, or portable/WASM paths.

Acceptance requires a Linux fixture with the actual index larger than the allowed
RAM: warm and cold p50/p95/p99/QPS, bytes read, faults, I/O queue depth, CPU, RSS,
and concurrent merge/ingest interference. Also compare warm full-text on identical
index bytes. Verify actual `io_uring_setup`/submission activity and exported
submitted/completed counters; a Cargo dependency is not evidence of use. Kernel
availability alone does not establish performance. No such backend or measurement
was added by this audit, so io_uring enablement remains outstanding.

## Recommended order and checks

For the warm text benchmark: collector comparison, adaptive block/tail encodings,
then release-binary SIMD coverage and remaining lazy-path profiles. For the
memory-constrained 150M-document / 2–3B-vector deployment: explicit cold-read
batching/io_uring is a separate high-priority experiment; its priority should come
from I/O evidence, not the warm leaderboard.

`python3 scripts/check_search.py check` passed on the unchanged runtime source
during this audit: contracts, formatting, Clippy, core/server/broker/tool tests,
native-without-sync, and standalone broker compilation. Documentation checks run
after the report is added. No new latency benchmark, deployed-binary ISA audit,
Linux io_uring prototype, full RPC harness, or WASM rebuild is claimed.
