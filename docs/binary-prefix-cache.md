# Optional resident binary-IVF prefix cache

The native reader may retain primary prefixes once and u32 primary-row references
for SOAR prefix-only runs. The nomination tier also retains one u32 document ID
and u16 ordinal per primary row; secondary labels use the existing primary-row
references. These compact columns add six bytes per primary value. This is a
derived heap cache; persisted format,
quantizer identity, scoring and nomination budgets do not change. Default budget
is zero. It is not mlock or a swap-safe residency guarantee.

`IndexConfig.binary_prefix_cache_budget_bytes` is one index-reader resource budget
shared across segments, reloads and all retained historical payload owners. It is
not multiplied per segment and is not a process-wide limit across independent
Index instances. Reservations cover requested prefix/reference allocations, the
primary label columns, per-run address directory and cache owner; allocator overhead and existing index
metadata/mappings are separate. The server exposes `--binary-prefix-cache-budget-mb` (MiB per index), default 0,
with checked byte conversion and a startup scope log. There is no process cap
across independently opened indexes. No default changes.

Loading occurs on native reader creation/reload, never in the scoring loop.
Reserve before allocation. Copy primary prefixes and decoded/remapped primary
labels in bounded batches. Document IDs include existing run document bases;
resolve secondary (doc, ordinal) through the existing immutable exact-vector
lookup, then binary-search the existing primary run directory to determine the
resident primary row. This avoids any document/vector-count-sized temporary
lookup. Per-run descriptors are the persistent cache metadata, not a second
construction table. Loader and query gather buffers are bounded. Cooperative
yields allow cancellation to drop the unfinished payload and reservation; only
complete caches become visible. One build at a time per resource budget bounds
construction concurrency. Missing/mismatched locations are corruption, not cache
misses. Over-wide prefixes/reference overflow must be observably refused.

AnnDiskIndex clones share one immutable cache cell independently of visibility.
Unchanged and deletion-only reader generations reuse it. Replacement segment IDs
build new caches and references; old owners retain reservations until the last
query releases them. Budget refusal preserves mapped execution and logs its
reason. Explicit reader.reload() and background reload owners retry admission even when the index generation
has not changed, allowing admission after old readers release memory. There is no
unbounded background retry worker or forced eviction of active readers.

The existing Hamming block scorer obtains bytes from the resident primary pool
or gathers secondary references into a fixed 8-KiB scratch. Candidate ordering,
visibility filtering, suffix offsets, rescoring and all-passage completion remain
owned by the existing executor. With a complete resident nomination cache,
read labels from the primary label columns and suppress all nomination prefetch.
The mapped fallback retains the existing selected-label/prefix prefetch policy.
Corpus-sized suffix/full codes and exact lookup metadata remain evictable.

This opt-in cache includes the labels needed to nominate candidates without
reading mapped label columns. It changes no candidate/rerank/probe budget, format or default.
Correctness includes poisoning the persisted nomination prefixes/labels after
cache construction, comparing candidates under visibility and serial/parallel
execution, and full-query lifecycle comparisons. Loading, cancellation and
retained-reader admission charge these additional columns atomically with the
existing payload. Performance must compare contemporaneous mapped, prefix-only
and complete-nomination readers; no full-query gain is assumed.

Validate complete native hit/score equivalence for single/multi-value and max/
softmax searches, native sync/async-only, partial/full merges, visibility commits,
physical/empty compaction, old readers, constrained admission/retry, allocation
ownership, malformed lookup labels and cancellation. Check disabled default and
portable compilation. A private original-vector fixture supplies additional full
query controls. No production performance claims follow from correctness tests;
full-query I/O, startup/build I/O, memory pressure and large-pool CPU cost require
separate measurements before enabling on a production shard.

## Bounded suffix prefetch

Fully resident nomination removes selected-leaf prefetch as well as mapped label
reads. A private complete-query measurement reduced bytes but increased cold
latency and major faults. The cache-enabled executor resolves suffix locations in batches
of at most 64 nominees. It validates their byte ranges before issuing advice,
sorts only a bounded copy of offsets, and combines only overlapping or adjacent
requested OS pages, capped at 64 KiB per combined range. It does not fill unneeded
page gaps or replace demand reads with a new I/O backend.

Sample the first/middle/last suffix ranges in each batch with the existing
residency helper. When all samples are resident, skip prefetch; this is only an
optional hint decision, never a correctness or residency guarantee for the batch.
Otherwise issue mmap WILLNEED hints for selected ranges before scoring them in
the original candidate order. Stack offsets remain bounded; candidate selection,
suffix bytes, score reductions and passage completion do not change. Only mmap
readers with the explicitly enabled nomination cache use this path. The
records the positive private cold-query measurements and the remaining validation
limits. The small fixture does not establish a production benefit.


## Availability and sizing

Summa 2.2.1 exposes this as an opt-in native feature. Existing indexes with a
binary prefix layout can use it without rewriting their persisted files. An
unprefixed generation needs rebuilding with prefix scanning enabled before this
cache can help; upgrading the server alone does not change the layout.

For primary vector count N, prefix width P bytes, secondary posting count S and
run count R, the cache requests approximately N × (P + 6) + 4 × S +
R × size_of(RunAddress), plus its fixed owner. Include overlap with retained
readers when sizing reloads and merges. This is additional heap alongside
existing metadata, page cache, query scratch and indexing memory. The recorded
social counts project roughly 150 GiB for the nomination tier; this has not been
allocated or tested at production scale. Budget refusal falls back to mapped
execution and logs the reason. Validate startup, concurrent queries and memory
pressure on a private workload before enabling a live shard.
