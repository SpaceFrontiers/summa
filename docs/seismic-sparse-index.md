# Seismic sparse indexing

## Status

BMP remains the default sparse-vector backend. Sparse MaxScore and Seismic are
explicit alternatives selected by `SparseVectorConfig::format` or SDL `format`.
Seismic is a separate approximate algorithm, not an optimization inside BMP.
All three dispatch through the existing schema, segment, query and lifecycle
owners; Seismic shares exact forward scoring for its approximate and exhaustive
paths. The earlier sole-backend integration is superseded by this design.

Version 3 introduced partitioned nomination storage and schedules copied runs
together during queries. Version 4 added lossless block compression of summary
directories. Version 5 added cardinality-aware cluster-ID sets and trusts immutable
writer-produced summary payloads during reads. Older Seismic envelopes require
an explicit rebuild. BMP/MaxScore
configuration and dispatch remain available. No benchmark
artifacts are production code.
See the [performance review](search-performance-review.md) for measured evidence
and remaining costs. The version-3 storage change below separates forward values
from independently replaceable nomination partitions.

Version 6 adds opt-in U16/U24/DotVByte forward dimension compression, with U32
reconstruction for vocabularies such as 100k tokens. Older envelopes require
rebuilding; no legacy read path is retained. See [forward compression](seismic-forward-compression.md).

## Configuration and recall controls

New SDL and programmatic schemas default to BMP; select `format: seismic`
explicitly. Serialized sparse configurations always write the backend name.
An omitted backend in older serialized metadata retains its historical MaxScore
meaning, independently of the new-schema constructor default. Reopening and
appending to an older index therefore preserve its encoded backend. The existing
schema configures both index construction and default query behavior:

```sdl
index example {
    field vector: sparse_vector<u32> [indexed<
        format: seismic,
        seismic_postings: 4096,
        seismic_cluster_size: 64,
        seismic_summary_energy: 0.4,
        query<seismic_cut: 10, seismic_factor: 0.85, exhaustive: false>
    >]
}
```

The values above are the current defaults, not a guaranteed recall target.

| Setting                       | Meaning and trade-off                                                                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seismic_postings`            | Retain at most 4,096 nominations per term in each newly built run. Higher values increase candidate coverage, build work and stored nominations. Copy merge preserves source runs; this is not a global post-merge cap. Valid range: 1–65,536.          |
| `seismic_cluster_size`        | Target 64 nominations per geometric cluster. Seed count is the ceiling of retained nominations divided by this target; actual cluster sizes vary. Smaller targets create more summaries and finer selection. Valid range: 1 through `seismic_postings`. |
| `seismic_summary_energy`      | Retain coordinates covering 40% of the cluster summary's total absolute magnitude, starting with the largest. This is not 40% of its coordinates. More retained magnitude makes larger summaries and changes candidate ranking. Valid range: (0, 1].    |
| `seismic_forward_compression` | Losslessly compress forward dimension IDs using adaptive widths and aligned gaps. Default: true; set false for raw U32. Weight precision and logical U32 IDs are unchanged. See [forward compression](seismic-forward-compression.md).                  |
| `seismic_cut`                 | Nominate through the 10 highest-absolute-weight eligible query dimensions. Exact scoring of nominees still uses the effective full query. Larger cuts explore more lists and cost more work. Valid range: 1–64.                                         |
| `seismic_factor`              | Skip a cluster when its summary proxy is below this factor times the full result heap's threshold. Lower values reduce this pruning and usually improve recall at higher cost; higher values prune more aggressively. Valid range: [0, 1].              |
| `exhaustive`                  | `true` bypasses nominations and scores the shared forward values exhaustively. Default: `false`.                                                                                                                                                        |

`lsp_gamma` controls BMP LSP traversal and has no effect on Seismic. For Seismic,
use `exhaustive: true` for complete coverage. `seismic_factor: 0` removes this
summary-threshold pruning but does not restore postings omitted at construction
or dimensions omitted by the cut. For multi-value combiners other than `Max`, the per-row proxy is not used to
prune document aggregates. Summary proxies are approximate, not exact
upper bounds, so changes to build/query knobs should be checked against an exact
reference on the intended workload. Shared query pruning and weight/precision
settings still apply; exhaustive search is exact for that effective stored query.

Text `SPARSE(...)` queries, RPC conversion and WASM query JSON apply the field's
schema query defaults. RPC and WASM request fields then override any explicitly
supplied values. Direct Rust `SparseVectorQuery::new` and `SparseTermQuery::new`
use the shared type defaults (`seismic_cut = 10`, `seismic_factor = 0.85`,
`exhaustive = false`); they do not inspect a schema. A direct Rust caller must
apply field-specific settings or overrides with the query builder methods.

Construction parameters describe persisted data and require rebuild/ALTER to
change existing encoded runs. None of these settings makes all summaries
resident: metadata pinning and query I/O policy are separate from recall
controls.

## Invariants

- Preserve signed weights, configured precision, U32 dimensions, logical document
  IDs, vector ordinals, missing values, filters, deletions and cross-segment scores.
- Store vector values once. Nominations contain row IDs and cropped cluster
  summaries, not another exact vector copy. Exact scoring, retrieval, exhaustive
  queries, multi-value aggregation and rebuild consume the same forward values.
- Own formats and build/merge code in `segment`; query owns nomination traversal
  and scoring. Adapt the Seismic clustering approach under its MIT attribution;
  do not persist upstream private Rust/serde layouts or add its native CLI stack
  as a portable-core dependency.
- Keep payloads in immutable `OwnedBytes` views. Budget compact directories and
  summary residency, candidate scratch, worker concurrency and maintenance units.
- Ordinary merges stream encoded runs and remap small row/document metadata.
  They do not cluster, prune or train. Explicit bounded maintenance may rebuild
  selected term lists from the shared forward representation, including deletion
  backfill when retained nominations cannot establish coverage.
- Reuse SegmentManager claims, replacement publication, cancellation ownership,
  deferred retirement and background permits. No second generation protocol.
- New format envelopes reject incompatible old bytes with a rebuild error.
  Search trusts admitted payload extents; integrity checks belong at format
  admission and explicit diagnostics, not repeated per-candidate scans.

The `reorder` field attribute belongs to indexed text (plain or chunked) and
indexed BMP sparse fields. Schema admission rejects it on other sparse backends,
dense and unindexed fields. Seismic and
binary ANN maintenance eligibility follows their persisted debt independently
of text flags. This avoids silently accepting a sparse flag with no effect.

## Cost model

Build groups retained term postings into geometric clusters and stores sparse
summaries plus row nominations. Exact values are written once. Search ranks
summaries and scores selected forward rows; explicit exhaustive requests scan
forward rows and use the same scorer. Filters apply before collection, and a
candidate document's relevant ordinals are aggregated consistently.

Merge cost is copied bytes plus directory remapping, with bounded scratch.
Maintenance cost is selected-term clustering plus affected immutable output.
The earlier monolithic layout rewrote the whole sparse file for each bounded
pass. Version 3 rewrites one nomination partition and reuses unchanged files.
Its sixteen extra files add open/fsync overhead; paired measurements must account
for build and copy-merge costs as well as maintenance savings.
Old readers can retain replaced payloads until their snapshots are released.

### Mapped query I/O

The retained reader keeps the operating system's normal mapping policy for
Seismic query payloads. A measured experiment using random-access advice and
explicit per-query prefetch improved some memory-pressure queries but regressed
warm search and exhaustive scans, so it was removed. Compact-directory pinning
is independent of that experiment. Opening no longer scans or prefetches
summary payloads; queries read them on demand.
See the performance review for the controls and limitations.

### Memory-limited operation

The server/tool `MmapDirectory` backend keeps large encoded payloads evictable.
The same statement does not apply to RAM, HTTP or `FsDirectory` readers that
materialize component buffers on the heap. Evictability prevents a requirement
to retain the whole index in RAM, but does not guarantee stable latency when a
query's working set exceeds available page cache.

The existing pin policy can retain compact term and logical-row directories.
The final memory-capped probes completed without OOM, including a 512 MiB
copy-merge. Before directory compression, search was about 35x slower at 1 GiB than at 4 GiB
on the same 50-query sample; pinning made little difference at 1 GiB. These
are workload-specific measurements, not a minimum-memory guarantee.
The canonical 1M-document four-source fixture has 28,394,800 pinnable directory
bytes (27.08 MiB). Its summary arrays shrink from 2.138 GB in version 3 to
1.415 GB in version 4 and 1.328 GB in version 5; they remain evictable. Compression does not eliminate the
pressure-induced latency cliff; see the [compact-summary measurements](seismic-compact-summaries.md). These are
logical metadata sizes, not total RSS or page-rounded lock costs. Budgets apply
per segment, so four active source segments have four allowances; a merged
segment has one. Old reader generations, concurrent query scratch, other fields
and index-global ANN metadata also contribute to process memory.

Configure residency with `SUMMA_PIN_METADATA_BUDGET_MB` and
`SUMMA_PIN_MODE=copy|mlock` (or the server's matching CLI flags). The default
budget is zero. `copy` retains eligible metadata on the heap;
`mlock` locks eligible pages subject to OS limits. Copy-mode memory can swap on a
swap-enabled host. Opening visits row and term directories, not summary payloads. Include open time, faults,
query latency and actual memory limits in measurements. See
[metadata pinning](hot-metadata-pinning.md) for accounting and priorities and the
[performance review](search-performance-review.md) for measured limits.

## Validation gate

The final implementation must pass the search harness `full`, portable native
checks, WASM build/tests, format admission and lifecycle failure regressions.
Compare build, copy merge, maintenance, search recall/latency distributions,
RSS and encoded bytes on identical fixtures/compiler/machine/flags. Check
quantized exhaustive truth and signed full-precision cases; include multi-value,
missing, repeated dimensions, filters/deletions, held readers and replacement.
Report environmental failures and unrun architectures explicitly. All benchmark
artifacts remain in ignored workspace storage; this document records the design.

## Partition storage design, version 3

The new default layout separates the existing single-copy forward representation
from sixteen fixed nomination partitions. Partition `dimension % 16` owns each
term. The segment owns one shared sparse-TOC sidecar per partition, covering all
Seismic fields; file count does not multiply with field count. Small bounded
writer buffers cap aggregate sidecar buffering. All sixteen partition envelopes
are present whenever Seismic data is present, including empty partitions.

The root stores only forward runs and row directories. Each partition retains
the existing encoded term payloads, forty-byte term entries, run directories
and build settings. Version 3 rejects earlier monolithic envelopes; no legacy
reader or alternate writer remains. Partition envelopes identify their partition
and validate row counts, settings and term routing against the root. Query
`term_runs` selects one partition and retains the same borrowed term views.

Ordinary merge copies root and partition runs and remaps their row/document
bases. It does not recluster or decode vectors. A bounded maintenance pass
selects one partition across the segment, prioritizes duplicated lists by run
count and then encoded payload bytes, rebuilds admitted terms in that
partition and copies its remaining encoded terms. Unchanged root and other
partition files are reused through the existing immutable replacement lifecycle.
Selected terms are processed in priority order so an expired time budget cannot
spend its first unit on a smaller selected term. The term directory remains
sorted by dimension; physical payload order is independent. Admission sorts
one temporary extent record per term in the current run to reject overlapping
or gapped payload ownership. It does not inspect the owned term payloads.

Thus output scales with one nomination partition, not all exact vectors and
summaries. Expired continuation publishes completed work; cancellation aborts
the claimed replacement. Pending debt counts only remaining fragmented terms.

Sixteen partitions bound file/fsync overhead while targeting approximately a
sixteenth of nomination bytes per pass. Skew can make partitions unequal; this
is a storage policy whose build, merge and maintenance cost must be measured
before claiming an improvement. Scratch is admitted for the selected partition
and one rebuilt term; no corpus-sized decoded array or parallel builder is added.

## Encoded storage and ownership

The [compact-summary design](seismic-compact-summaries.md) specifies the version-4
lossless directory codec, compatibility gate and measurement protocol. Weight
quantization and payload-colocation experiments described there remain proposals.

Each field retains the existing sparse-file TOC envelope. Root and partition
blobs contain independently copyable runs, a 32-byte entry per run, and a
40-byte versioned footer. Their combined representation contains:

- One forward payload in the configured Float32, Float16, UInt8 or UInt4
  precision, using the shared sparse weight codec. Version 6 optionally compresses
  dimension IDs with lossless U16/U24/DotVByte encoding; reconstructed IDs remain U32.
- A 24-byte row directory recording logical document ID, value ordinal,
  payload offset, encoded length and number of coordinates. Empty values keep
  their row and ordinal; missing values have no row.
- Geometric-cluster payloads: row nominations and energy-cropped sparse
  summaries of coordinate-wise absolute maxima. Summaries transpose dimensions
  and store UInt8 weights with
  per-cluster Float32 minimum and scale. Dimension/occurrence-end directories
  choose existing packing or independently readable 128-entry monotone blocks
  using local packing/Elias–Fano, including encoded block overhead in that choice.
  Cluster IDs remain bit-packed. They rank approximate candidates;
  they are not conservative bounds after cropping and quantization. Signed exact scores always
  use the original forward values.
- A 40-byte term entry recording dimension, cluster count, payload extent,
  row base, nomination coverage capacity and full nonzero-vector frequency.
  Full frequency remains independent of top-L nomination pruning for IDF.
- A 32-byte run footer locating the forward and term directories.

Format parsing rejects incompatible envelopes, out-of-range extents and
inconsistent logical directories. It visits forward-row metadata, compact
cluster directories and summary-directory entries to establish borrowed byte
slices (compressed directories use bounded sequential block decoding), but does not scan vector coordinates, nomination IDs or summary
codes during normal open. Runs remain `OwnedBytes` views; corpus
payloads are not deserialized into per-entry heap objects. Opening therefore
still touches O(rows + clusters + summary-directory entries) metadata and can
fault pages containing those directories. Compact run objects remain on the
heap; configured pinning can additionally retain directory copies.
Term-debt calculation temporarily sorts one dimension, payload
length and frequency per term entry in the partition being opened. It also verifies that
the combined nonzero frequency for a dimension does not exceed forward rows.

The portable builder uses deterministic sampled centroids and sparse dot-product
assignment over the strongest 15 coordinates, following Seismic's geometric
blocking approach. Cluster count is derived from the configured target cluster
size. This is an adaptation, not byte-identical reproduction of the upstream
builder's random-number generator or minimum-cluster reassignment policy. The
upstream MIT attribution is retained beside the implementation.

## Merge and maintenance implementation

Ordinary native merge uses the directory's existing range-copy helper for whole
encoded runs, with bounded byte-copy fallback. It rewrites run metadata to rebase
logical documents and forward rows. It neither scores nor clusters vectors.
Vocabulary bounds inferred independently at build are combined with `max`;
configured weight precision and Seismic build settings must agree.

Explicit maintenance selects a bounded number of duplicated term lists. For
compatible lists with complete top-L coverage and no deleted nominees, the
union of their retained nominations suffices for the new top-L selection. An
incomplete coverage marker or deleted nominee triggers a full-forward backfill
scan for that selected term. Live-document membership is supplied by the owning
segment replacement operation. Selected decoded vectors, clustering scratch and
term-directory scratch are admitted against the maintenance memory allowance;
unadmitted work remains visible as pending debt. The caller supplies cancellation
and a separate continuation predicate so expiration can publish completed work.

Maintenance reuses the root and unaffected partition files. Within the selected
partition it copies unaffected nomination payloads, remapping term-level row
bases. Completed terms become one list; copied terms keep their remaining
fragmentation. Publication and reader retention use the normal segment protocol.

Deletion compaction copies the retained forward values byte-for-byte and rebuilds
nomination lists from their decoded values. It admits its decoded working set
before writing, preserves empty ordinals, and reuses the initial builder's
nomination writer. Reader generations and output cleanup continue to belong to
SegmentManager rather than the storage codec.

## Query execution and exact operations

The query executor uses signed, decoded forward weights for retrieval and exact
candidate backfill. It scores all ordinals of a nominated document before the
shared combiner and document top-k heap, including zero contributions from empty
or nonmatching values. Entirely nonmatching documents remain absent. A full heap
may publish a cross-segment floor only when it covers the query result window.
Required Boolean scoring clauses use a complete document-order forward scorer;
their membership is independent of top-k and nomination pruning.

Nomination scratch admits at most 262,144 distinct documents and 262,144 visited
clusters per segment execution. Hitting either limit marks the shared query as
truncated and emits a warning; query diagnostics and metrics expose the event.
These are work limits, separate from the query's deadline and response budgets.

### Merged-run query scheduling

Merged runs share one candidate-document set, exact scorer and top-k heap. The
first nomination term orders clusters by descending proxy across all its runs,
so a strong cluster in a later run can establish the pruning threshold before
we visit weaker earlier runs. Later terms preserve encoded run/cluster order.
This changes approximate nomination order; exhaustive scoring is unchanged and
recall must be measured alongside latency.

Summary scoring is independent between runs. Native synchronous search may
split that work into at most four tasks on the existing search pool; direct
calls outside a Rayon worker and portable/async execution remain sequential.
All tasks write disjoint slices of one bounded summary-score buffer. Candidate
admission, filters, exact scoring and heap mutation remain on the caller, so
parallel work neither multiplies nomination budgets nor changes arithmetic or
heap ordering. The existing visited-cluster cap bounds the aggregate buffer and
borrowed cluster views across runs; cancellation is checked between runs and
before consuming a completed batch.

The 100K maintained-index sample found the coordinate iterator and exact forward
scoring dominating query time; independent summary work was only a small part.
Inlining the coordinate iterator alone moved that cost into an out-of-line
weight decoder and did not improve the paired 100K query measurement. The
iterator therefore specializes its full traversal by precision once per vector;
the shared scorer consumes it through `for_each`, which uses that specialized
`fold`. Each precision still calls the shared sparse-weight codec with a constant
format. Partial iteration and scalar `next` remain available to other callers.
Coordinate order, signed values and duplicate-query arithmetic stay unchanged;
there is no decoded buffer, second scorer or new representation. Release
assembly and cross-version exact-result identity verify the loop transformation,
and paired latency measurements determine whether it is retained.

Filters with at most 4,096 eligible bitmap IDs score those documents directly.
If a predicate/deletion-filtered nomination pass underfills its heap, the executor
scans remaining forward rows to recover qualifying documents omitted by top-L
pruning. This can increase work for restrictive predicates without a materialized
bitmap; `seismic_filter_scans` and debug output expose that choice. Explicit
exhaustive queries always use the same forward scorer without nomination.

Compaction's admission estimate additionally includes the whole-field term map
and output directory that coexist with clustering. On 64-bit hosts it charges
448 bytes per coordinate plus 248 per retained vector, versus 256 per coordinate
plus 248 per selected vector for term maintenance (including the assignment
coordinate cache described below). The additional allowance
accounts for BTreeMap node occupancy, candidate-vector allocation slack and term
metadata; it is conservative scratch accounting, not a measured RSS promise.

Document combination includes zero dot products from stored nonmatching or
empty ordinals. Reported matched positions contain only ordinals sharing a
query dimension, including an actual match whose positive and negative
contributions cancel to zero. This distinction keeps chunk-level fusion from
treating a nonmatching value as corroborating evidence.

### Reusing build-time centroid assignment coordinates

Centroid assignment prepares each decoded forward row's strongest fifteen
coordinates once, in descending absolute weight with dimension-ID ties. All
term clusters reuse that ordering. The transient cache is a flat array of
fifteen `(u32, f32)` entries per row (120 bytes per row); short rows use their
existing coordinate count, with no per-row heap allocation. Preparation uses a
fixed fifteen-entry stack buffer, avoiding a second full-vector allocation.
Term maintenance prepares the same cache only for its selected rows; compaction
and maintenance admission include its additional 120 bytes per decoded row.
The cache is never persisted or retained by readers.

Top-L nomination selection partitions candidates by the existing total order
before sorting the retained prefix. Both changes preserve forward and nomination
bytes: floating-point score accumulation order, centroid sampling, row order,
summary generation, and all format settings stay the same. Initial ingestion
still owns the complete decoded rows and inverted candidate lists during build;
this optimization does not turn initial construction into a bounded-memory
streaming build.

Each segment query prepares its scoring weights once. Unique dimensions below
65,536 use a query-local Float32 lookup table, capped at 256 KiB; exact forward
scoring reuses that table; summary proxies use the coordinate-transposed lists. Wider U32 dimensions
and duplicate query dimensions retain the sorted sparse walk. Duplicate terms
are not folded together because changing multiplication/addition order can
change scores or hide overflow. A stored zero still counts as a dimension match
when its query weight is nonzero. The lookup is scratch, not an index-resident
vocabulary table, and uses the existing bounded sparse execution concurrency.

## Compact summary encoding

The summary encoding introduced in version 2 replaces row-major Float32 pairs
with coordinate-transposed UInt8 summaries. Version 3 moved them into nomination
partitions. Version 4 compresses their monotone dimension and occurrence-end
directories without changing quantized values. The exact forward owner and
logical row IDs remain unchanged. Version 5 additionally compresses cluster-ID
sets, preserving each coordinate's sorted IDs and UInt8 codes. Version 5 rejects
older envelopes; there is
no fallback reader or automatic live migration.
Copy merge still copies complete runs byte-for-byte
and only adjusts existing run/term row bases. Maintenance and initial build use
one shared term writer.

Each term stores a compact cluster directory (cumulative nomination-row end,
Float32 scale and minimum), contiguous U32 nomination rows, a packed summary
dimension directory, cumulative posting ends, encoded cluster IDs, UInt8 summary
codes, and a 40-byte footer describing these extents. Cluster-ID arrays choose
fixed-width packing or subset ranks for at most 64 clusters, including every
128-coordinate checkpoint in the size comparison. The format also represents
ranks and weight codes colocated by group for the locality experiment. Dimension and end arrays
independently choose original packing or 128-entry monotone blocks; the
[compact-summary format](seismic-compact-summaries.md) defines the byte layout. The existing sparse
UInt8 encoder provides per-cluster min/scale/codes; existing fast-field bit-pack
primitives provide portable integer encoding and random access. The existing
packed-size comparison first chooses a sparse ordered dictionary or dense
implicit dimension directory. Block compression is then considered for its
explicit arrays; this does not claim globally optimal sparse/dense selection. Single-cluster sparse summaries need no cluster
IDs or posting-end array. Dimension IDs retain U32 width within the existing exclusive vocabulary bound
(the largest accepted ID is `u32::MAX - 1`); compactness
is determined by data rather than a schema restriction.

Query execution binary-searches only query dimensions (or directly addresses a
dense directory), then accumulates the matching summary postings into a caller-
owned Float32 array bounded by the term's admitted cluster count. Each transient term view
caches its decoded layout offsets without introducing resident per-term objects. Duplicate
query coordinates retain their individual operations in ascending dimension
order; negative query weights use their absolute magnitude for nomination only.
Positive infinity remains an eligible proxy. Exact document/ordinal scoring
continues to use the signed forward weights. The version-4 directory rewrite
preserves existing quantized proxies and nomination behavior. Changing weight
quantization or cropping would still require new recall measurements: these
proxies are not exact pruning bounds.

Normal summary views read writer-produced layout fields directly; they do not
check term format/lengths, scan directories or validate decoded IDs. Correctness
is established by the writer and codec tests. Existing outer-envelope
compatibility and directory ownership checks do not inspect summary contents. Corpus-sized arrays remain borrowed from OwnedBytes rather than expanded
into resident objects. Build scratch for one term additionally holds transposed
summary occurrences, bounded by the already admitted selected-row coordinate
budget. The dense directory is considered only when its dimension domain is at
most twice the number of occurring dimensions, bounding construction scratch.
Maintenance adds 64 bytes per selected coordinate for occurrence-array capacity
slack and simultaneous sparse/dense transpose buffers; maintenance accounts for the transposition buffers before publication.
No native-only codec or additional dependency is introduced. Build parallelism
is a separate follow-up after this format and query path are measured.

### Partitioned maintenance lifecycle

Version 3 separates immutable exact-forward runs in `.sparse` from sixteen
term partitions, routed by `dimension % 16`. Each partition is a segment-owned
`seg_<id>.seismic.<partition>` file shared by all Seismic fields. The existing
sparse field TOC and footer describe each field's partition payload; the root
format requires all sixteen files. File count is bounded independently of the
number of fields. Native and portable readers reject missing or incompatible
partitions instead of treating them as empty nominations.

Build and copy-merge use the same field codecs and sparse envelopes. Copy-merge
streams encoded forward and nomination runs and remaps only their directories.
Partition writers use bounded 64 KiB buffers (1 MiB total on local directories).
Maintenance chooses one partition for the segment, admits terms under its time
and scratch limits, and rewrites only that partition. There is no fixed term cap. Unchanged partitions
are hard-linked under the replacement segment ID, with the existing bounded
streaming fallback on directories without links. The root `.sparse` file is
retained the same way unless BMP maintenance also needs to rewrite it. Other
Seismic fields in the selected partition are copied if they have no admitted
work. No source-generation paths are persisted in replacement metadata.

Output claims, atomic metadata publication, reader-held retirement, cancellation
cleanup, and orphan ownership remain the existing segment lifecycle. The central
segment file inventory includes all partition names, including partial outputs.
Dense-only ALTER/rebuild retains the root and every required partition; deletion
compaction writes forward rows and nominations through the same partition writer.

## Productive maintenance and budgeted partition work

The full-consolidation measurement exposed two earlier policy limits: the server's
replacement-lineage pass limit stopped productive work, and a fixed 64-term cap
forced repeated partition rewrites. Successful maintenance passes remain tracked
for cooldown admission, while `seismic_no_progress_passes` separately persists
consecutive no-progress attempts. Both counters clear when debt reaches zero. Published maintenance
with lower term debt resets that count; a successful unchanged-debt pass
increments it. Failed/cancelled publication changes neither counter. New merge
inputs reset stale stall history, while unchanged single-source replacement
preserves it. BMP scheduling and existing failure backoff remain unchanged.
Seismic follow-up eligibility uses the no-progress threshold, preserving
shared concurrency, cooldown and per-pass scratch/time admission. Counter cost
is one compact integer per segment, without payload scans.

Within one chosen partition, maintenance continues through fragmented terms
in existing priority order until its time budget expires, rather than stopping
at 64 terms. One term is decoded/rebuilt at a time under the memory allowance;
the existing directory scratch is charged separately. A term already admitted
may finish across the deadline; remaining terms copy unchanged. Explicit work
with no deadline is still bounded by the selected partition and scratch limits.
Zero/expired budgets, memory refusal, cancellation and partial publication keep
their distinct existing semantics. Ordinary copy merges retain encoded runs.

Before claiming a copy-merge improvement, profile run copying, file lifecycle
and publication separately and compare identical prebuilt inputs with the same
binary settings. Validate repeated ingestion/merge/maintenance cycles, query
latency during maintenance, debt arrival/retirement, snapshot retention, restart
and cancellation boundaries. Lifecycle tests use shared owners; benchmark
orchestration remains ignored under `.context/`. The performance review records
validation status and measured evidence; no steady-state capacity claim follows
from a single completed consolidation.

### Mixed-backend maintenance intent

Automatic optimization must not rerun completed or retry-capped text/BMP work
merely because Seismic still has productive work. The scheduler invokes a
policy-aware wrapper around the same claimed replacement operation. Under the
source metadata lock, the owner decides whether BP is initially due or remains
eligible for deepening under its existing cap. A crate-private execution intent
gates text planning and BMP root rewriting; unrelated files retain the existing
hard-link/copy path. A maintenance-only replacement preserves BP ordering,
convergence and pass counters while updating independently observed ANN/Seismic
debt. Explicit manual reorder keeps its existing meaning. No second writer,
publication protocol or public schema flag is introduced.

### Reusing prior RGB/BMP and Seismic work

RGB/BMP reuse the published document or per-field virtual-ID ordering rather
than a separate optimizer checkpoint. Copy-merge preserves compatible ordered blocks, and
a later BP pass starts from the current order while rebuilding its temporary
graph. Seismic already reuses completed term payloads and untouched partitions;
it does not persist iterative clustering state because sampled assignment is
one pass. Its quantized, energy-truncated summaries are neither centroids nor
complete resume state.

A future finer-grained fast path could reuse an existing complete term if global
top-L selection yields exactly the same live logical rows, configuration and
forward precision match, and row addresses admit existing remapping. It must
still prove nomination coverage (or backfill from exact values), update aggregate
term frequency, and preserve deletion/ordinal semantics. This is a proposal,
not an implemented shortcut; its hit rate and recall implications need measuring.

## Implementation owners

- `segment/seismic/`: persisted forward values, nomination construction, summary
  encoding, copy merges and bounded maintenance.
- `query/seismic.rs`: nomination scheduling and top-k collection.
- `query/seismic/scoring.rs`: query preparation, exact row scoring and ordinal
  combination shared by nomination, candidate reranking and required clauses.
- `query/seismic/required.rs`: complete membership and document-order scoring.
- `query/planner.rs`: shared backend dispatch for sparse vector/term entry points.
- `segment/sparse_partitions.rs`: segment file writers and shared TOC integration.

Query unit and integration tests live beside the executor in separate modules.

Forward dimension compression is evaluated in [the codec design](seismic-forward-compression.md).

Components use envelope version 6; older versions require rebuilding. Default `seismic_forward_compression: true` uses
lossless adaptive U16/U24/DotVByte forward dimension encoding; it leaves configured
weight precision and nomination unchanged. Default is false. See the linked
codec design for layout, applicability, and measured tradeoffs.
