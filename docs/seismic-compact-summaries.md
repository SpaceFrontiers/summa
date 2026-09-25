# Compact Seismic summaries

Status: implemented and measured in the working tree, 2026-09-18. No schema,
query defaults or intended scoring behavior changed. The persisted Seismic
directory-only format was version 4; the cluster-ID experiment below advances
it to version 5. Existing older indexes require rebuilding. This extends
the [Seismic design](seismic-sparse-index.md)
under the [search contract](search-system-contract.md).

## Problem and measured baseline

Summaries already use UInt8 weights, per-cluster Float32 minimum/scale, and
bit-packed dimension IDs, occurrence ends, and cluster IDs. They are transposed:
for a query coordinate, the reader finds the clusters containing that coordinate
and accumulates their proxy scores. Replacing Float32 summary weights with UInt8
is therefore not a remaining opportunity.

The 1M-vector, four-source fixture has 2,138,175,719 summary-array bytes. Its
saved routing census further separates 563,398,111 dimension bytes and
505,785,242 occurrence-end bytes: **1.069 GB, or 50.0% of summary arrays, is
directories**. There are 300,453,433 directory entries across 109,870 term
payloads. These are repeated term-local coordinates, not distinct vocabulary
items. Evidence: `.context/seismic-copy-merge/summary-routing-census.json`;
the aggregate fixture census and pressure results are recorded in the
[performance review](search-performance-review.md).

The previous fixed widths use the largest absolute coordinate and occurrence
position within a term. Both arrays are monotone, so representing local ranges
can use fewer bits. The upstream [pinned summary implementation](https://github.com/TusKANNy/seismic/blob/3c267137e202748e69ada8cd093f4c0b7c04479c/src/quantized_summary.rs)
already uses Elias–Fano offsets; that is supporting precedent, not a measurement
of the Summa format measured below.

## First stage: lossless monotone directories

Keep clustering, summary cropping, UInt8 codes and quantizers unchanged. Replace
only the dimension and occurrence-end representation. The invariant is identical
decoded coordinates, occurrence ranges, cluster IDs, weights, arithmetic order,
proxy scores, nominations and final hits for a fixed execution environment.
Encoded directory bytes intentionally change; untouched forward and nomination
payloads do not.

The implementation uses independent blocks of at most 128 directory entries.
64- and 256-entry alternatives remain future benchmark controls. Each block
stores its absolute base and encodes offsets from that base, choosing between
two bounded representations:

- Local fixed-width packing: each offset uses the block's maximum required width.
- Local Elias–Fano: separate low bits and a unary high-bit sequence for monotone
  offsets. Store the bit extents explicitly and validate them before lookup.

Choose an array representation by its complete encoded size, including descriptors
and alignment. Retain the current packed array when the blocked array is larger,
especially for small lists. The new term footer adds eight bytes even to a term
whose arrays retain their old packing; charge that overhead in aggregate results.
Tie-break in favor of
the simpler fixed-width representation. Dense coordinate ranges retain implicit
IDs; single-cluster terms retain implicit ends and cluster IDs where applicable.
Do not force dimensions into UInt16: the public dimension space remains U32.

A term contains a versioned codec descriptor, a searchable block directory,
encoded dimension/end blocks, and the existing occurrence IDs and UInt8 codes.
The block directory records each block's base, maximum, payload offset, width
and codec. Entry positions follow from the fixed block size; a preceding end
can be read from the previous block. Byte offsets must cover the complete term
without truncation; use checked arithmetic and explicit little-endian fields.
Count every descriptor, restart value and padding byte in size comparisons.
The prototype's precise layout is described below.

Lookup binary-searches block maxima, then searches at most one block for the
coordinate. Access to its occurrence ends uses the corresponding end-array block and,
at a block boundary, the previous block's maximum. Local Elias–Fano selection scans only a block's bounded high-bit
region; it must not scan a term-wide bitmap from its beginning. With the usual
low-bit choice, high-bit storage is O(block entries), even for widely separated
U32 dimensions. Read words from borrowed bytes safely across alignment and tail
boundaries. No term-wide decode or per-query heap cache is introduced.

The existing `structures/postings/elias_fano.rs` owns related integer encoding,
but its current representation owns Vec buffers and its select implementation
scans from the beginning. Do not deserialize that object per summary or assume
its lookup is already suitable. A reusable bounded borrowed block view belongs
in structures; term assembly and codec dispatch remain in
`segment/seismic/term.rs`. Build and maintenance share its existing encoder.
`query/seismic.rs` continues consuming `TermRun::score_summaries` on native sync,
native async and WASM paths.

### Implemented wire layout

The outer Seismic version is 4. Term magic is `0x33544d53`, with a
40-byte footer: the first 27 bytes retain the old counts, coordinate limit and
three array-width tags; byte 27 is reserved zero; bytes 28 and 32 store U32
dimension/end array byte lengths; bytes 36–39 are reserved zero. Width tags
0–32 retain their prior meaning; 255 selects blocked monotone encoding.
Cluster metadata, nomination rows, cluster IDs and UInt8 weight streams retain
their old bytes. Dimension and end arrays choose their codec independently.

For each blocked array, count is supplied by the term. The initial directory
has one 16-byte entry per 128 values: base U32, maximum U32, payload offset U32,
bit width U8, codec U8 (0 local packing, 1 Elias–Fano), and two reserved zeros.
Count determines tail-block length. Low bits come first in each payload;
Elias–Fano then appends the bounded unary high bitmap. Payload lengths derive
from count, maximum minus base, bit width and codec. All fields are little-endian.
The two independently blocked arrays use the same logical entry boundaries;
an end crossing a boundary reads the preceding block's maximum. The prototype
currently obtains that value through the ordinary indexed accessor.

Encoding temporarily holds the alternative encoded directory to compare complete
sizes. This is bounded by one admitted term, not merely one 128-value scratch
block; include it in construction/maintenance memory measurements. Corpus-sized
decoded directories are never retained at query time. The version-4 reader checked encoded
monotonicity at admission using sequential block decoding. The current reader
trusts the writer and performs no summary-payload admission scans. The
version-4 measurements below retain their historical admission cost.

## Second stage: reduce pages touched

Measure directory compression independently before changing payload placement.
Then evaluate colocating a coordinate block's occurrence IDs and weight codes
with its lookup metadata, so finding a coordinate and reading its contributions
does not jump between several term-wide arrays. Bound work by encoded extents;
128 coordinates is not a fixed byte-size limit when many clusters contain them.
Do not pad every small block to a filesystem page. Keep cluster quantizers shared
per term and account for the additional pages they require.

This stage still preserves per-cluster accumulation order and existing scheduling.
It does not imply a new prefetch policy, a decoded summary cache, or pinning all
summaries. Previous query-I/O experiments regressed warm search; physical locality
must be evaluated separately from advice and concurrency changes.

## Optional lossy experiment

Four-bit weights can halve only the UInt8-code stream, not the entire summary.
Cluster IDs, dimensions, offsets and quantizers remain. Test this separately,
with explicit persisted precision and schema validation before exposing it.
Keep Float32 quantizer metadata initially. Changes to codes can alter proxy
ordering and pruning; exact scoring cannot recover a document never nominated.
Cropping already prevents these proxies from being conservative bounds, so
rounding quantized values upward does not establish exact-search correctness.
No recall-preservation claim or new default is justified without measurements.

## Capacity and execution costs

If directory bytes halve, the 1M fixture's summary arrays would fall from
2.138 GB to approximately 1.604 GB: a 25% summary reduction. If directories shrink
by 75%, summary arrays would be approximately 1.336 GB. These are arithmetic
scenarios, not measured compression ratios. The corresponding complete index
would still be approximately 3.003 GB or 2.736 GB, versus the 3.537 GB baseline.
Neither scenario makes this fixture fit in a 1 GiB cache.

Encoding scratch is bounded by admitted term construction and its encoded
directory alternatives. Query scratch stays bounded independently of corpus size;
borrowed decoding may use a fixed block buffer if measurements justify it.
The query cost becomes block-directory lookup plus bounded local decoding and
the unchanged matching occurrences. Count bytes touched and faults as well as
encoded bytes: a smaller codec can lose on CPU or touch the same number of pages.
At billions of vectors, forward-vector reads, multiple vectors per candidate
document, concurrency and fragmented runs remain separate capacity costs.

## Compatibility and lifecycle

The incompatible term codec and Seismic envelope version 4 make old readers
reject new indexes at open, and new readers reject old indexes. Default merge still
copies compatible encoded runs and remaps small row/document metadata, without
reclustering or requantizing. Mixed incompatible versions must fail clearly;
use explicit rebuild/migration, never silent conversion in ordinary merge.
Existing immutable generation claims, cancellation handling, cold output writers
and reader lifetimes remain the publication protocol.

Writer/codec tests establish codec tags, counts, ordered block ranges, monotone
directories, occurrence extents and legal widths. Normal summary reading trusts
those writer guarantees, with no format/length or payload integrity checks.
No server, broker, RPC or query limit changes are required for the lossless stage.

## Validation and decision gates

1. Census the actual current 100K and 1M fixtures: per-term counts, local spans,
   density and exact encoded sizes including fallback and block overhead. Record
   consolidated and fragmented indexes separately.
2. Compare old/new decoded arrays and proxy scores, then serialized query hits,
   against signed, duplicate, absent, empty, single-cluster and U32-edge fixtures.
   Test partial blocks, repeated dense ends, malformed descriptors and overflows.
   Preserve untouched payload bytes and copy-merge output blocks exactly.
3. Validate rebuild, maintenance, reopen, cancellation and held-reader publication
   through the existing owners. Run the search harness and portable/WASM matrix
   required by the contract for the actual implementation scope.
4. Measure warm and 512 MiB/1 GiB/4 GiB searches with the same fixture, compiler,
   machine, flags and query sets; include p50/p95/p99, QPS, RSS/cache, faults,
   bytes read, scratch and recall. Add a representative multi-vector sample.
5. Keep lossless compression only with query equivalence and a demonstrated
   size/latency tradeoff. Evaluate UInt4 at matched recall and report any extra
   candidate work. Do not extrapolate a production RAM minimum from file size.

## Measured codec results (2026-09-18)

The 128-entry codec preserves existing UInt8 proxies and nomination order. The
1M-vector four-source fixture was converted offline without rebuilding clusters;
all 300,453,433 decoded directory entries match, and SHA-256 comparisons preserve
2,459,230,762 bytes across 219,744 forward/row, cluster/nomination and occurrence
ID/weight-code sections. The separately converted Apple Silicon fixture has the
same complete-index size. All sampled approximate and exact results match their
controls.

| Encoded bytes              |        Before |       Compact | Reduction |
| -------------------------- | ------------: | ------------: | --------: |
| Dimension directories      |   563,398,111 |   200,823,278 |     64.4% |
| Occurrence-end directories |   505,785,242 |   144,729,209 |     71.4% |
| Both directories           | 1,069,183,353 |   345,552,487 |     67.7% |
| All summary arrays         | 2,138,175,719 | 1,414,544,853 |     33.8% |
| Complete live index        | 3,537,446,009 | 2,814,694,103 |     20.4% |

Warm timings use three alternating runs per version, 200 queries/top-100 and
three passes, excluding the first pass from latency summaries. Values below are
medians of per-run statistics. Both versions use the same source fixture,
compiler, machine, flags and four-worker settings within each architecture.
There was no compiler overlap. Linux uses the existing n2-highmem-8 machine; Apple M4
uses the local shared workstation. All runs use a 64 MiB copy-pin allowance;
summary arrays remain evictable.

| Warm search          |    Before |   Compact | Change |
| -------------------- | --------: | --------: | -----: |
| Linux mean           | 22.419 ms | 20.201 ms |  -9.9% |
| Linux p95            | 36.538 ms | 32.489 ms | -11.1% |
| Linux p99            | 44.442 ms | 39.424 ms | -11.3% |
| Linux process RSS    | 3.353 GiB | 2.681 GiB | -20.1% |
| Linux open           |   2.892 s |   3.273 s | +13.2% |
| Apple M4 mean        | 11.789 ms | 12.107 ms |  +2.7% |
| Apple M4 process RSS | 3.016 GiB | 2.343 GiB | -22.3% |
| Apple M4 open        |   1.117 s |   1.786 s | +59.9% |

Apple M4 had a 16.824 ms candidate run with a large scheduling tail; the other
two candidate runs were 11.839/12.107 ms, versus controls 11.709/11.789/12.572 ms.
Report the positive 2.7% median change and variability rather than claiming an
ARM speedup. Linux had one 9.867-second control open outlier; the other control
opens were about 2.89 seconds. No architecture-independent latency win is claimed.

The initial implementation repeatedly selected random entries during full
admission scans, increasing Linux open time to 33.3 seconds. Sequential block
decoding fixed that regression without changing the encoded bytes or query
algorithm. The residual open-time cost above remains a tradeoff.

### Enforced memory caps

These are single Linux cgroup-v2 capacity probes with swap disabled, identical
64 MiB pin allowances, fresh cold opening and three query passes. The 1/4 GiB
rows use the same 50 queries; 512 MiB uses ten queries and must not be directly
compared to the larger sample. Latency excludes the first pass.

| RAM cap |  Before mean | Compact mean |    Before p95 |   Compact p95 |
| ------- | -----------: | -----------: | ------------: | ------------: |
| 4 GiB   |    22.161 ms |    22.866 ms |     36.543 ms |     37.587 ms |
| 1 GiB   |   695.558 ms |   650.968 ms |  1,318.960 ms |  1,331.653 ms |
| 512 MiB | 7,063.370 ms | 6,524.871 ms | 13,104.254 ms | 12,135.474 ms |

All probes completed with identical hits and zero OOM/kill events. At 1 GiB,
mean latency falls 6.4%, but p95 is approximately unchanged/slightly higher.
Major faults fall 236,274 to 211,459 and file refaults 1,537,720 to 1,248,084.
At 512 MiB, mean falls 7.6%, while total filesystem input is only 2.6% lower
(68,298,184 to 66,524,432 512-byte blocks). These I/O counters include opening
and all three passes, not only timed warmed queries. The remaining summary
occurrences and exact-vector payloads still impose a severe cache-capacity
limit; component-specific query I/O attribution is not yet measured.

Compression is a demonstrated space improvement, with a mixed latency tradeoff.
It does not establish a low-RAM production configuration for 150M documents
with 2–3B vectors. This fixture has one vector per document, and no concurrent
production QPS, multi-vector workload, or optimal block-size claim is made.
64/256-entry tuning, further physical colocation and UInt4 remain unimplemented
experiments.

Validation after the sequential-admission fix: the native `check` harness passes
1,997 tests (24 intentionally ignored), strict lint and native-without-sync;
WASM build and all 36 tests pass. Search harness evidence is
`.context/search-harness/20260918T173418.407024Z-check/`. Benchmark scripts, raw
results, hashes and before/after binaries are retained under
`.context/seismic-compact/`.

### Copy merge and fresh build

Two alternating Linux trials per version on the four-source 1M fixture measured
median copy-merge time 11.776 → 10.707 seconds (9.1% lower). Maximum process RSS
fell 2,509,152 → 1,806,620 KiB (28.0% lower). Every source run's encoded bytes
were preserved through merge; no reclustering or requantization was introduced.
Controls varied 11.155–12.396 seconds, while candidates were 10.704–10.709; this
is a small paired sample, not a universal throughput claim.

One fresh 100K build per version measured 62.139 → 61.935 seconds and
652,660 → 652,280 KiB peak RSS: effectively similar in this single trial. The
complete index shrank 468,256,562 → 356,739,945 bytes (23.8%). Forward values,
cluster/nomination payloads, cluster IDs and weight codes match; all 50 sampled
approximate and five exact query results match. Initial construction still has
no hard peak-memory guarantee.

Both old-reader/new-index and new-reader/old-index combinations were explicitly
checked and fail with the incompatible-format error. The offline fixture
converter is benchmark tooling, not a supported live migration path. Existing
production indexes were not migrated.

## Cluster-ID compression and locality results

This implementation preserves every decoded cluster ID, UInt8 code, quantizer,
nomination row and accumulation order. It starts from the measured version-4
fixture above. The public entry remains `query/seismic.rs::execute` through
`TermRun::score_summaries`; `segment/seismic/term.rs` owns assembly, admission
and merge-preserved bytes. The same borrowed scorer serves native sync, async
and WASM. No query options, RPCs, candidate budgets or clustering change.

The initial census finds 650,174,079 occurrences and 418,818,287 ID bytes in
this fixture, with IDs using at most six bits. Ordinary cyclic-gap block
packing at 16/32/64/128 entries does not improve any complete term after its
restart directory. That proposal is rejected before changing the writer.

The candidate represents each coordinate's strictly increasing cluster-ID set
as a combinatorial rank, with cardinality supplied by the existing occurrence
ends. For C clusters and K occurrences, its width is ceil(log2(binomial(C,K))).
Limit this codec to C <= 64; larger terms and terms without a complete byte-size
saving retain fixed-width IDs. Independent 128-coordinate groups have U32 byte
checkpoints and byte-aligned rank payloads. An accessed group decodes at most
128 occurrence ends into fixed scratch to locate a coordinate's bit range.
A rank decodes into at most 64 IDs. The writer guarantees legal ranks, ordered
IDs, array extents and counts; normal readers do not revalidate them. This is a bounded CPU/space
tradeoff, not a claim of faster search.

Compare three layouts on identical immutable input: version-4 baseline;
compressed IDs with the existing separate weight stream; and the same
compressed IDs with each coordinate group's weight codes immediately following
its ranks. The two compressed layouts must have identical decoded content and
sizes, isolating placement from compression. Dimension/end directories and
cluster quantizers remain separate in this locality experiment. Query-time
scratch must not grow with corpus size; encoder scratch is bounded by an
admitted term. Default merge copies either encoded layout, without retraining.

An incompatible term descriptor and owning envelope version gate distinguish
the experiment. Persist the layout in the term, not a process-level reader
switch. No new public schema default is warranted before paired native/WASM
correctness, warm latency on x86 and ARM, memory-cap I/O and lifecycle checks.
Benchmark-only conversion must verify all decoded IDs and preserve untouched
bytes; it is not a supported migration path. Existing indexes need rebuilding
if the format is retained. Record rejected candidates and final evidence below.

### Revised read policy (user requested)

Normal summary readers trust immutable writer output. They no longer scan
summary payloads at open, validate term format/lengths, or validate occurrence
ranks while scoring. Writer/codec tests retain correctness assertions. This
replaces the eager-admission proposal above; the benchmark control receives
the same policy so storage/locality comparisons are not confounded by it.
Existing index-envelope compatibility and directory ownership are unchanged.

### Version-5 occurrence layout

The term footer remains 40 bytes; magic is `0x34544d53`. Byte 27 is the
occurrence layout: 0 retains fixed-width IDs, 1 uses subset ranks with a separate
weight stream, and 2 stores ranks and weights together per coordinate group.
The U32 at byte 36 stores encoded ID bytes, excluding the unchanged one-byte
weight codes. Counts, quantizers and directory tags otherwise retain their
version-4 meanings. The owning Seismic envelope advances to version 5.

For modes 1/2, the occurrence section starts with one U32 byte offset per
128-coordinate group. Each offset is relative to the payload after this small
directory. A group's coordinates contribute consecutive rank bits; only the
end of a group is padded to a byte. Its occurrence counts come from the existing
end directory, so each rank needs no separate header. Mode 1 concatenates rank
groups and then all codes; mode 2 puts each group's codes immediately after
its rank bytes. Both have exactly the same number of bytes. A query decodes
one accessed end-directory group into 128 U32s and computes 129 bit offsets;
these buffers are reused within the term. ID decoding needs at most 64 U32s.
A fixed 65x65 binomial-coefficient table is shared read-only, independent of
corpus size. There is no per-query heap cache or retained decoded corpus state.

For example, choosing 32 IDs from 64 clusters costs 61 rank bits instead of
192 fixed-width bits. Choosing one ID still costs six bits; choosing two costs
11 rather than 12. Sparse/small terms therefore retain their old ID packing
when checkpoint overhead would erase the saving. The measured 1M fixture
chooses subset encoding for 54,648 of 109,870 terms: ID bytes fall from
418,818,287 to 332,618,612 (20.6%). Weight codes remain 650,174,079 bytes.
Summary arrays become 1,328,345,178 bytes (another 6.1% reduction), and the
complete index becomes 2,728,494,428 bytes (another 3.1% reduction).

The offline converter uses the production occurrence encoder and checks all
650,174,079 decoded IDs/codes against its source. Dimension/end bytes,
quantizers, nomination rows and forward payloads are copied unchanged. It
is evidence tooling only; there is no silent conversion during ordinary merge.
The initial eager-admission benchmark is retained separately under `eager/`
and is superseded by the final writer-trusting reader measurements.

### Final reader and locality measurements

The final comparison rebuilds the fixed-ID control with the same writer-trusting
read policy. It therefore separates compression/locality from removing eager
scans. Three alternating warm trials per layout use 200 queries/top-100 and
three passes (first excluded), four workers and a 64 MiB copy-pin allowance.
All sampled approximate/exact hits match, with no compiler overlap.

| Mean query latency     |    Fixed IDs | Compressed IDs, separate codes | Compressed IDs, colocated codes |
| ---------------------- | -----------: | -----------------------------: | ------------------------------: |
| Linux warm             |    24.877 ms |                      25.186 ms |                       22.303 ms |
| Apple M4 warm          |    12.200 ms |                      11.968 ms |                       12.085 ms |
| Linux 1 GiB, top-100   |   667.399 ms |                     642.730 ms |                      628.797 ms |
| Linux 1 GiB, top-10    |   492.745 ms |                     468.405 ms |                      462.614 ms |
| Linux 512 MiB, top-100 | 6,093.037 ms |                   6,469.505 ms |                    6,506.735 ms |

The memory-cap cases are single probes with swap disabled and three passes.
1 GiB uses 50 queries; 512 MiB uses ten and is not directly comparable to the
larger sample. All complete with zero OOM events/kills. At 1 GiB/top-100,
filesystem input is 11,451,456 / 11,111,880 / 11,119,832 512-byte blocks;
at 512 MiB it is 61,112,800 / 60,943,920 / 61,028,480. These counters include
opening, all passes and result collection. Colocation produces essentially
no additional disk-volume reduction. Compression is 6.2% slower than the
control in the single 512 MiB probe, despite the smaller index; it does not
resolve severe cache thrashing.

Median warm open time is now 26.4 / 25.8 / 25.4 ms on Linux and
14.0 / 12.8 / 12.5 ms on M4. The initial compressed prototype with eager
checks took about 7.3 seconds on Linux and four seconds on M4. Removing
those scans is the startup improvement; it is not a compression speedup.
The [performance review](search-performance-review.md) records p95, residency,
refaults, lifecycle results and complete evidence locations.

The normal writer retains compressed IDs with separate weight codes. The
colocated layout remains an experiment: it improves Linux warm latency but
shows little additional benefit under memory pressure and no clear ARM win.
Both representations remain byte-copyable through ordinary merge. Further
layout/default changes need representative workloads, especially multiple
vectors per document; the current fixture has one vector/document.

Two paired copy-merge trials per layout preserve every encoded source run,
with medians 3.899 / 3.859 / 3.503 seconds and peak RSS below 64 MiB. A fresh
100K build is 2.9% smaller and has similar time/RSS; converting its control
produces identical encoded sparse components. All fresh approximate/exact
outputs match. Final native checks pass 1,997 tests (24 ignored), strict lint,
native-without-sync, and the WASM build plus 36 tests. Logs, scripts, binaries
and hashes are retained in `.context/seismic-occurrences/`.
