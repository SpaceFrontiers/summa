# BMP forward values and candidate scoring

Status: implemented, 2026-09-05. L1 remains opt-in.

BMP owns both sparse retrieval and exact stored sparse values. L1 must score a
logical `(document, ordinal)` without an inverse physical-ID sidecar or a scan
of the corpus. BP should read these same values when building its graph and
rewriting records. The quantized impacts, pruning and query quantization remain
identical to inverted BMP scoring; this is not a second sparse scoring model.

## V20 representation

The V19 block payload, offset table, grids and physical document maps keep their
encodings. V20 appends a forward section before the existing 80-byte footer and
uses magic `BMPA`. The section contains:

- Quantized vector payload in ascending `(document, ordinal)` order. Each entry
  is dimension `u32 LE` followed by impact `u8`, with nondecreasing dims. Repeated dimensions retain every impact, preserving
  the existing additive posting semantics.
- A sorted directory: document `u32`, ordinal `u16`, reserved zero `u16`, and
  payload-relative byte offset `u64` per vector.
- A 16-byte trailer: vector count `u32`, reserved zero `u32`, payload bytes `u64`.
  Unknown reserved bits are rejected. The payload length terminates the final
  vector. The retired search prototype's uniqueness certificate is not part of
  this format; experimental blobs carrying it must be rebuilt.

Space overhead is 5 bytes per retained posting, 16 bytes per vector, and 16
bytes per field. The forward section is file-backed and evictable; opening
validates compact directory bounds/order and scoring validates selected payloads.
Neither forward directory nor payload is pinned; both remain evictable. Lookup uses binary search;
scoring visits only nominated vectors and intersects sorted query dimensions.

## Build, merge, reorder and compatibility

### Optional storage

The per-field `bmp_forward_index` setting defaults to `true`. In SDL:

```sdl
field sparse: sparse_vector [indexed<format: bmp, dims: 105879, bmp_forward_index: false>]
```

With storage disabled, ingestion emits the existing V19 representation and skips
forward-payload construction. Ordinary merges and both BP granularities omit the
forward section, including when their sources contain it. The inverted payload
and scoring semantics are preserved. Existing forward values may still accelerate
BP's graph/rewrite reads; disabled storage never triggers a legacy forward upgrade.
BP still builds its bounded transient graph from inverted postings when needed.
This option changes replacement output, not immutable live source files. A field
excluded from an explicit reorder remains a byte-identical copy, as before.

The setting is persisted in schema JSON and reported in server SDL. BMP search
always uses inverted scoring and has no forward-completion option. L1 candidate
backfill and BP use stored forward values whenever present, without a dispatch
heuristic or separate enable switch.
Enabling storage for existing V19 data requires explicit reorder/rebuild; ordinary
merge never synthesizes forward values. With storage enabled, mixed V19/V20 copy
merges still require an explicit upgrade. With storage disabled, both source
versions can copy their compatible inverted representations into V19 output.

L1 can backfill V19 by locating candidates in a logically ordered document map,
then probing only their blocks and query terms. After BP makes that map unordered,
missing-score backfill requires stored forward values; it fails with actionable
guidance rather than scanning the corpus or substituting a missing default for
an unsupported lookup. `backfill: false` still uses organic scores and learned
missing defaults. Full-text and sparse MaxScore backfill use their own postings
readers and are unaffected by this BMP-only storage option.

Ingestion emits the same retained quantized entries as the inverted builder.
It keeps the already-admitted per-dimension input until forward output, uses a
k-way cursor heap over dimensions and an 8-byte offset per retained vector,
and does not build another posting-sized heap representation. Peak accounting
includes input postings plus grids during inverted output, then input postings
plus offsets during forward output. This increases input lifetime, not its size.
Ordinary V20 merge concatenates payload bytes and streams directory entries,
adjusting document IDs and byte offsets with bounded scratch. BP changes only
physical inverted order, so existing forward payloads also copy through record
and block reorder without decoding or re-quantizing them.

V19 remains readable for retrieval. With forward storage enabled, explicit reorder can construct its missing
forward values under the existing reorder memory budget, using a sorted bounded
physical permutation and existing block decoders. This migration is allowed to
decode; ordinary merge is not. All-legacy ordinary merges preserve V19, and with
storage enabled mixed V19/V20 ordinary merges fail with migration guidance instead of discarding
capability or reconstructing values implicitly. Unknown versions fail loudly.
Generation claims, cold output, fsync, publication and cleanup remain owned by
the existing sparse segment lifecycle. No new sidecar or preparation RPC exists.

## Validation and cost evidence

Required evidence: quantized forward/inverted score equality, missing values and
ordinals, payload byte identity across merge and both BP modes, V19 reopen and
explicit migration, incompatible/corrupt input rejection, cancellation and
writer failure. Compare BP graph and record rewrite against the existing path.
Measure storage bytes, build and merge working memory, candidate scoring and BP
with the same fixtures/compiler before claiming a speedup. Record results in
