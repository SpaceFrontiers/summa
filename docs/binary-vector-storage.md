# Exact binary vector storage

## Previous layout and measured duplication

Binary IVF stores exact packed codes, unlike float IVF-TQ and ScaNN AH, whose
ANN codes approximate the retained vectors. Binary ScaNN also retains exact
codes, with optional secondary assignments for SOAR. A binary field previously
stored the same codes twice: document order in flat storage and cluster order
in ANN storage. Each copy also has six bytes per vector for document ID and
value ordinal. Flat storage supports retrieval, candidate scoring, completing
multi-value scores, training, and vector-index ALTER/rebuild.

On the one-million-vector projected-SIFT fixture, the two representations use
76,012,384 bytes at 256 bits and 652,012,384 bytes at 2,560 bits, excluding the
outer container and shared quantizer. The corpus columns are evictable mappings;
these sizes are not resident heap requirements.

An exploratory lossless residual experiment XORed each code with its centroid
and compressed independent approximately 64 KiB leaf blocks using Zstd level 1.
It recovered the exact original bytes and reduced the code payload by 26.7%
at 256 bits and 25.2% at 2,560 bits. This excludes lookup/block metadata and uses
cluster order. Compressing only the flat duplicate therefore saves roughly
12–13% of total vector storage, with additional decode work. These fixture-specific
results do not justify a compression default.

## Selected direction: one exact copy, all operations preserved

New writes remove duplicate binary code payloads whenever a trained binary
ANN representation exists. This is the default behavior, without a flag, as
requested by the user. The implementation preserves retrieval, all multi-value combiners, point scoring, training, rebuild,
ALTER, deletion, merge, and reorder. It is not a search-only mode. Duplicate flat-plus-binary-ANN segments are unsupported and rejected on open;
recreate those indexes. Flat-only binary fields remain valid before training.

Keep exact codes in cluster order for sequential ANN scans. Replace the flat
code payload with a document-ordered location map into that field's ANN payload.
Reuse the existing exact-vector reader and ANN writers; do not add another
scorer, vector reconstruction scheme, or lifecycle publication protocol.

Each map record contains a document ID, ordinal, and 64-bit
span-relative address (14 bytes). At one vector per document, replacing
the flat payload with this map would reduce total vector bytes by approximately
32% at 256 bits and 48% at 2,560 bits. These are format estimates; measured implemented sizes are recorded in the
performance review. A narrower or implicit-address map may improve the small-code
case but requires explicit overflow and missing/multi-value rules. Codes shorter
than 64 bits can occupy more total bytes with this 14-byte lookup than with the
former six-byte label plus duplicate code. Single-copy remains the requested
default for every binary dimension.

Required implementation invariants:

- Exact codes and ordinals survive round trips bit-for-bit. Missing values and
  SOAR secondary assignments never create extra logical values.
- The location map is versioned, file-backed, and bounded by the segment's ANN
  payload. Reject incompatible/corrupt references instead of returning empty data.
- Native sync, async, and WASM use the same vector access semantics. Training
  can sample exact codes without decoding float representations.
- Before training, flat-only fields retain their sole exact copy. ALTER to a
  geometry whose training floor exceeds the corpus writes deferred flat
  storage; ALTER to a trainable binary ANN representation rebuilds its lookup
  atomically with the replacement generation. Direct ALTER to `flat` remains
  unsupported by the existing schema contract.
- Existing run writers generate locations from the offsets they actually write.
  Merge copies compatible codes and lookup rows, remapping only directories; it
  never retrains the codebook. Reorder must not assume ANN rows are document-sorted.
- Location construction, sorting, reads, and candidate gathers have bounded
  scratch and cancellation checks. Temporary outputs use existing ownership and
  cleanup rules. Publication remains the current immutable-generation protocol.
- Diagnostics distinguish exact code bytes, location bytes, secondary ANN
  assignments, file-backed residency, and heap metadata.

## Verification before shipping

Compare the two layouts for byte-exact vector retrieval and query results across
single/multi-value fields, missing values, every combiner, deletions, forced
merge, reorder, retraining, and ALTER in both directions. Hold old readers across
publication and exercise failure/cancellation cleanup. Test rejection of duplicate binary layouts, malformed locations and unsupported versions. Run the full search
harness and WASM suite.

Measure complete index bytes, build/merge scratch, retrieval and training I/O,
multi-value completion, and warm/cold ANN latency on the same fixtures. ANN
references into document-ordered flat codes are an alternative, but must first
demonstrate that scattered reads do not erase the sequential-scan advantage.

## Implemented format and merge cost

TOC type 11 replaces the flat entry whenever exact binary ANN codes exist.
The existing ANN format is unchanged. Old readers reject this unknown TOC type;
new readers reject duplicate flat-plus-binary-ANN segments. There is no schema flag.

| Region                         | Encoding                                                                                                        |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Header, 32 bytes               | magic `XVL1`, dimension u32, logical count u32, version u32, ANN length u64, block count u32, reserved zero u32 |
| Lookup rows, 14 bytes each     | local doc u32, ordinal u16, address u64 (span index in high 32 bits, row in low 32 bits)                        |
| Span directory, 12 bytes each  | ANN-relative code offset u64, row count u32                                                                     |
| Block directory, 16 bytes each | row count u32, document base u32, span count u32, reserved zero u32                                             |

Rows are ordered by effective document ID and ordinal. Each lookup block owns
consecutive rows and spans. Directory lengths and cumulative ranges follow from
the header and block directory; readers reject unsupported versions and invalid
extents. The metadata stays file-backed except for the small block directory.
An initial build uses one lookup block and one span per emitted ANN run. Existing
run writers report actual code positions; a bounded external metadata sort puts
rows in document order and deduplicates SOAR secondary assignments. Temporary
sort files are anonymous RAII-owned files, with bounded runs and merge fan-in.

**Normal merge keeps lookup rows verbatim and coalesces clusters.** When the
merged output would hold more than one run for some cluster (predicted exactly
from the source directories), the encoded-run compactor writes one run per
cluster: codes and ordinals are copied verbatim, and document IDs are rewritten
absolute with one `u32` add per posting, as float AH merge already does. Merges
whose sources cannot share a cluster keep the byte-copy writer. Neither path
reconstructs vectors, reassigns leaves, or retrains models.

The lookup is not rebuilt. Every lookup span lies inside one ANN run: builds and
rebuilds emit one span per run, compaction and deletion compaction one span per
copied or filtered run, and the reader rejects any other span at admission.
Coalescing moves each source run's codes as one contiguous range, so the
compactor records `(source code range → output offset)` per source run and the
lookup writer translates each span offset through that map. Rows, whose
addresses are `(span, row)` pairs, are copied verbatim; block document bases are
rebased as before. A span not wholly inside one copied run aborts the merge as
corrupt data rather than shifting an address. Relocation memory is one entry
per source run, the same order as the source directories the merge already
holds; there is no lookup sort and no per-row lookup rewrite. The span directory
keeps the sum of the source spans, exactly as a byte-copy merge does.

This replaces the earlier policy of copying binary extents unchanged, which let
runs per cluster multiply with every merge tier. That earlier policy existed
because coalescing was thought to require rebuilding the lookup with the external
sort (about 240 ns per row). Measured merge and query costs are in the
[performance review](search-performance-review.md).

Standalone reorder uses the same coalescing writer for segments that are still
fragmented, such as outputs of earlier versions. It keeps its reorder budget,
cancellation, cold writer, output claim, and atomic publication lifecycle, and
it also works on binary-only indexes. Already contiguous fields keep the copy
path. Deletion compaction and explicit training/rebuild construct new lookups
with the bounded external sort, because they write new rows.

End-to-end merge also includes opening and validating readers and publishing the
output; those costs must be measured rather than inferred from the writer alone.

The exact-vector reader shares the ANN reader's immutable byte owner, including
for remote readers, so it neither duplicates the payload in memory nor fetches
it again. It resolves span addresses for point retrieval, returns zero-copy views
for contiguous batches, and gathers scattered batches directly from that owner. Sequential document-order scans may
therefore require scattered ANN reads; the deleted flat payload formerly provided
sequential access for those operations. No ANN scoring kernel or candidate budget
changes in this storage work. See the [performance review](search-performance-review.md)
for validation and measurements; scan optimizations have separate evidence.

The public logical access methods on `LazyFlatVectorData` remain the common
interface. The former raw `handle()` / `vectors_byte_offset()` accessors are
removed because a single contiguous document-ordered region no longer exists
for ANN-backed fields. In-repository callers use logical reads or the internal
optional `flat_region()` copy view. External Rust callers using those raw
accessors must migrate to the logical read methods.

Reader admission resolves each lookup span to an ANN label slice once. It then
walks lookup blocks and rows sequentially, validating address bounds, matching
ANN labels, document limits, and ordering in the shared document-map validator.
This avoids a run-directory binary search and repeated lookup-block searches for
every logical value. Temporary span views scale with the small span directory;
no corpus-sized decoded address array is built or retained.

Native admission prefetches at most two 64K-row metadata windows and 8 MiB of
page-rounded ANN ordinal ranges. The shared flat-map validator uses the same
windowing. ANN document-column validation follows the already computed physical
payload order instead of alternating between distant source extents in cluster
order. These are bounded metadata reads; normal merge still never reconstructs,
reassigns, or sorts vectors.

## Implementation owners

`segment/vector_locations.rs` owns the immutable lookup view and directory.
Its private `vector_locations/writer.rs` owns bounded location sorting and
encoded-copy output. `vector_data.rs` provides the shared exact-vector access
interface; existing ANN writers emit addresses and the segment lifecycle owns
publication. Splitting these files introduces no second writer or scorer.
