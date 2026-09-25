# BMP forward compression: query latency

The tables below record the original compressed scorer. The subsequent
[compact-accumulator optimization](accumulator-optimization.md) cuts its warm
rescoring time by about 2.5× while retaining the same compressed files.

## Query latency results

All times below are milliseconds for retrieval at k=1,000 followed by L1
backfill over every passage of the retrieved documents, then selection of at
most ten documents. Both formats use the same 64 query templates. Warm timings
exclude the first pass; constrained timings use the second pass.

| Cache condition | Raw mean | Raw p95 | Compressed mean | Compressed p95 |
| --------------- | -------: | ------: | --------------: | -------------: |
| Warm cache      |    53.22 |   81.76 |          143.38 |         213.17 |
| 1 GiB cap       |  1559.65 | 4704.47 |          145.17 |         217.65 |
| 512 MiB cap     |  5221.43 | 8936.47 |         3056.54 |        4848.53 |

Warm pipeline latency increases **2.69×** with packets. At 1 GiB the compressed
working set fits and the raw one keeps faulting: packets are **10.74× faster**.
At 512 MiB both layouts remain constrained; packets are **1.71× faster**, but
still take seconds. Compression does not remove the disk-access bottleneck.

Retrieval alone stays around 7.5–8.1 ms across these conditions. At k=100 with
L1, warm means are 18.17 ms raw and 33.00 ms compressed. At k=1,000 the warm
retrieval component is about 26.6 ms for both, while backfill is 26.56 ms raw
versus 116.78 ms compressed. The regression is in candidate scoring.

## Memory and disk evidence

At 1 GiB, compressed pipeline runs peak at **892 MiB** of cgroup memory. Raw
runs reach the 1 GiB cap and continue faulting. Both variants hit the 512 MiB
cap in the smaller-budget pipeline runs. All pressure runs completed with zero
OOM events. These counters include file cache, kernel memory and the worker;
they are not measurements of Rust heap allocations alone.

The first 64-query sweep is deliberately reported separately:

| Cap     | Raw first-sweep mean | Compressed first-sweep mean |
| ------- | -------------------: | --------------------------: |
| 1 GiB   |           1760.69 ms |                  1005.06 ms |
| 512 MiB |           5361.89 ms |                  3149.27 ms |

For the 512 MiB pipeline, whole-run counters below average the two trials;
each trial includes open and 128 requests across both sweeps. They cannot be
assigned to an individual pass.

| Layout     | Disk reads per run | Major faults during queries |
| ---------- | -----------------: | --------------------------: |
| Raw        |         11.126 GiB |                   1,018,416 |
| Compressed |          8.450 GiB |                     545,600 |

Compression removes about 24% of total pipeline disk-read bytes and 46% of
major faults here. The inverted representation is unchanged, so total I/O
does not fall by the full 45% forward-payload reduction. The raw run spends
about 47 seconds of CPU across roughly 678 seconds of wall time; waiting for
pages dominates.

## Locality measurements

These controls score exactly 1,000 candidate documents without retrieval.
Times are repeat-sweep means on the common 64-query subset (warm excludes the
first pass):

| Cache condition | Raw scattered | Raw nearby | Compressed scattered | Compressed nearby |
| --------------- | ------------: | ---------: | -------------------: | ----------------: |
| Warm            |       9.83 ms |    9.02 ms |             39.00 ms |          38.48 ms |
| 1 GiB           |      10.07 ms |    9.11 ms |             39.04 ms |          38.45 ms |
| 512 MiB         |    1826.26 ms | 1687.91 ms |             39.05 ms |          38.45 ms |

At 512 MiB, compression makes the scattered forward-only workload **46.76×
faster** because that working set fits. Nearby IDs improve raw latency only
about **7.6%** in this test. Membership and vector-length differences mean this
is an access-pattern probe, not a measured benefit from physically reordering
the same candidates.

Mean selected footprints make that difference visible:

| Pool                 | Vectors | Raw payload | Compressed payload | Raw / compressed distinct 4 KiB pages |
| -------------------- | ------: | ----------: | -----------------: | ------------------------------------: |
| Retrieved k=1,000    |  64,653 |   50.61 MiB |          27.74 MiB |                        13,422 / 7,570 |
| Scattered 1,000 docs |  20,841 |   16.19 MiB |           8.90 MiB |                         5,145 / 3,274 |
| Nearby 1,000 docs    |  21,173 |   16.47 MiB |           9.05 MiB |                         4,218 / 2,317 |

Page counts describe the union of selected forward-payload extents, not actual
disk reads or metadata pages. The current access path still faults cold pages
individually, as described below.

## CPU profile and remaining findings

A short resident `perf` sample places about 88% of compressed-run CPU samples
inside `CandidateBmpPreparation::score` and about 3% in the packet iterator.
The [raw](raw-profile.txt) and [compressed](gap-profile.txt) profiles contain 74
and 254 samples respectively, with no lost samples; these are hotspot probes,
not another latency estimate.

The [annotated hot-loop excerpt](gap-profile-loop.txt) shows repeated scalar and
vector stack copies around the per-coordinate accumulator. Together with the
source passing `Result<u32, Error>` through the fold, this suggests an avoidable
accumulator-copy cost. A primitive accumulator with error construction at the
boundary was the next experiment at the time of this measurement. It is now
implemented and measured in the [follow-up](accumulator-optimization.md). This
initial profile alone does not identify a specific hardware stall.

The other measured concern is demand paging without forward-range prefetch.
These two findings belong to the scorer and forward reader respectively; the
measurement did not add another runtime scorer, codec or compatibility reader.

## Fixture and query workload

Measured on 2026-09-19. The follow-up uses a separate isolated Linux machine and public `Searcher::search`
and `Searcher::score_candidates` APIs. Nine 100,000-row windows of genuine
retained passage vectors preserve document grouping (42,596 documents total).
No production query set was available: 216 deterministic 16-term queries come
from disjoint following rows, interleaved across all nine windows. They are synthetic query templates, not traffic.
The newly built BMPB fixture is converted offline to raw BMPA forward rows;
inverted blocks/grids/maps are copied exactly. The raw comparison uses the same
writer-trusted scoring policy as BMPB, avoiding a confounding validation pass.
Ordinary retrieval at k=10, retrieval plus backfill at depths 100/1000, and fixed
local/scattered candidate pools isolate retrieval, decode and locality costs.
Warm and bounded Linux memory trials record per-query timings, score bits,
page faults and memory. No production settings or caches are changed.

The fixture contains 900,000 vectors with 146,460,740 nonzero entries (162.73
per vector), in one segment. Vectors per document have mean 21.13, median 4,
p95 58, p99 311 and maximum 7,800; the maximum vector has 512 nonzeros.
Boundary documents may contain only their sampled passages; original passage order and grouping are retained, with compact document
IDs and renumbered ordinals. Raw forward payload is 732,303,700 bytes; packet
payload is 402,399,443 bytes (45.05% smaller). The 900,363,945-byte inverted/grid/
document-map prefix is identical. Entire sparse files are 1,647,067,806 bytes raw
and 1,317,163,549 bytes compressed (20.03% smaller).

Queries use the Max document combiner, U8 document impacts with scale 5, and a
105,879-dimension field with BMP blocks of 32. Backfill scores every passage of
each admitted document and retains at most 10 after document aggregation.
Retrieval k is passed directly to the public search API; overlapping passage
hits collapse to fewer unique documents. Measured candidate counts are reported
separately, so k=1,000 must not be read as 1,000 distinct retrieved documents.
On the common 64-query subset, it yields 476.09 unique documents on average
but expands to 64,653.36 passage vectors (p95 104,366; maximum 124,040).
The 96-template warm set yields 497.63 documents and 64,776.38 vectors on average.

Local pools are 1,000 consecutive document IDs; scattered pools use a coprime stride
of 104,729 over the same document space. These pools differ in membership and
row-length distribution; vector/page footprints accompany their timings. They
test physical access patterns, not a rewrite that reorders identical candidates. Both variants receive exactly the same
queries and candidate pools.

## Measurement protocol

Warm trials read the fixture before opening, run 96 queries for three passes,
and exclude the first pass from steady timings. Cross-memory comparisons use
the common first 64 query templates; the complete 96-template warm sample is
retained as supplemental evidence. Memory-limited trials run 64
queries twice in a fresh cgroup with swap disabled; report first and repeat
passes separately. Each case runs twice with raw/gap order reversed. Only
fixture files receive `POSIX_FADV_DONTNEED` between runs. No global cache drop
is used. One request is in flight; reciprocal mean latency is sequential QPS,
not saturated concurrent throughput. Per-query time includes query preparation,
retrieval where selected, candidate scoring and final sorting. It excludes index
open, correctness serialization, RPC, document hydration and model inference.

The benchmark uses an isolated 8-vCPU Intel Cascade Lake Linux machine and a
1,000 GiB Google Cloud pd-ssd disk, Rust 1.98.1, release thin LTO,
one codegen unit and `-C target-cpu=native`. Both binaries use the same dependency
lockfile and compiler. Metadata pin mode is `copy` with a 64 MiB per-segment
budget. Process peak RSS and cgroup memory are distinct: pressure-run cgroup
limits also charge file cache and kernel memory. I/O counters cover the whole
run, including open and all passes, and must not be attributed to an individual
pass. Warm cgroup counters belong to the shared host group and are not used as
isolated memory/I/O evidence.

The first memory trials had no `io.stat` because systemd's I/O controller was
not enabled for service cgroups. Their disk-read values are recorded as absent,
not zero. I/O accounting was enabled on the isolated machine during the reversed
1 GiB raw pipeline run; that partial interval is also excluded from I/O totals.
Subsequent runs expose counters before and after the workload. Page-fault and
memory measurements are available for every completed trial. The checked-in
matrix now requests `IOAccounting=yes` for every pressure service; the private
evidence archive preserves the exact orchestration source used for these runs.

## Locality finding

Code-path observation: `BmpForwardIndex` marks forward bytes `MADV_RANDOM`.
Candidate execution charges selected extents through `reserve_candidate_bmp_reads`
and then scores mapped bytes; it does not issue forward-payload range prefetch.
Thus nearby IDs do not automatically trigger normal mmap read-ahead. A next
experiment should coalesce bounded selected forward ranges after byte admission,
without fetching the entire span between the first and last candidate. This is
a proposed experiment, not an optimization implemented in this measurement.

## Reproduction and limits

The [Rust runner](query-runner/src/main.rs) calls the public core APIs. Its
[offline converter](query-runner/src/bin/raw_fixture.rs) reuses the production
packet decoder and copies the inverted prefix. It is not a compatibility path
in the application. The [matrix driver](query_matrix.py) manages isolated Linux
cgroups and cache conditions; [analysis](analyze_queries.py) checks paired hit
hashes and keeps first/repeat timings separate.

In an isolated copy of the source tree, build with:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --manifest-path docs/benchmark-results/bmp-forward/2026-09-19/query-runner/Cargo.toml
```

Save the BMPB executable and converter before making a second source copy for
the baseline. The raw baseline restores these six BMP files from commit
`782227808a54c560b231d19e0a4d46662ecca334`: `segment/bmp_forward.rs`,
`segment/bmp_forward/rewrite.rs`, `segment/builder/bmp.rs`, `segment/format.rs`,
`segment/reader/bmp.rs`, and `query/bmp.rs` (all under `summa-core/src`). Its
candidate scorer calls `vector_for_scoring` rather than `vector` so both formats
use the same writer-trusted policy. Rebuild after restoring files; old archive
mtimes can otherwise make Cargo reuse the wrong artifact. Opening the raw
fixture during preflight detected that cache issue in the initial setup;
all published timings use the correctly rebuilt, distinct binaries.

The runner accepts `build INDEX CORPUS` and
`query INDEX QUERIES DEPTH MODE PASSES COUNT HITS`. Corpus rows are
`[document: u32 LE][nnz: u32 LE][nnz × (dimension: u32 LE, impact: u8)]`.
Queries are JSON arrays of dimension/float-weight pairs. Result IDs and score
bits stream to NDJSON outside the timed section. The original query templates
and retained vectors remain private; hashes and aggregates identify the fixture.

The pipeline deliberately forces backfill of the same sparse field to exercise
forward storage. It does not imply that ordinary sparse retrieval needs this
additional scoring pass.

This measures one sparse field in one rebuilt segment, one request at a time,
on the stated machine/storage. It does not establish production-server latency,
concurrent QPS, relevance/recall, BP rebuild cost, or performance for other query
widths, ranking plans and candidate limits. The standalone ARM row benchmark
uses a different accumulator and omits the full candidate pipeline; its CPU
ratio should not be used to predict these end-to-end numbers.

## Evidence and validation

[Aggregates](query-summary.json) include mean, p50/p95/p99, sequential QPS,
per-run RSS/faults, valid I/O counters and hit hashes. The
[manifest](query-manifest.json) records source/input/binary/dependency hashes,
hardware, compiler and the private raw archive. [Candidate footprints](candidate-footprints.json)
record selected vectors, encoded bytes and distinct payload pages.

All **52 runs / 9,856 timed requests** completed, with **26 matching raw/packet
pairs** for every returned document ID and score bit. An independent offline
comparison matched all **900,000 source vectors / 146,460,740 entries**, their
logical document/ordinal mapping, and the unchanged **900,363,945-byte** inverted
prefix. The archive was downloaded and SHA-256 verified. The machine is confirmed
`TERMINATED`; production files and settings were only read.

This task added benchmark tooling and documentation, without changing the
runtime compression implementation. Both release benchmark binaries compiled;
Ruff, documentation/ownership checks and `git diff --check` pass. The preceding
runtime change's native/WASM validation is recorded in the [storage report](README.md).
