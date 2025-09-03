# BMP forward values and candidate scoring

Status: current-format implementation complete. Deploy against rebuilt indexes
or indexes whose BMP blobs already use the current envelope. L1 remains opt-in.

BMP owns sparse retrieval and exact stored sparse values. L1 addresses a logical
`(document, ordinal)` without an inverse physical-ID sidecar or a corpus scan.
BP reads the same values when building its graph and rewriting records. The
quantized impacts, pruning and query quantization match inverted BMP scoring.

## One representation with optional storage

Every BMP blob uses the current `BMPA` envelope: adaptive inverted blocks, their
offset table, compressed pruning grids, physical document maps, a forward-storage
section, and the 80-byte footer. Forward storage changes that section, never the
format version. Already-published forward-enabled bytes remain unchanged.

An enabled section contains:

- Quantized vector payload in ascending `(document, ordinal)` order. Each entry
  is dimension `u32 LE` followed by impact `u8`, with nondecreasing dimensions.
  Repeated dimensions retain every impact, preserving additive posting semantics.
- A sorted directory: document `u32`, ordinal `u16`, reserved zero `u16`, and
  payload-relative byte offset `u64` per vector.
- A 16-byte trailer: vector count `u32`, flags `u32 = 0`, payload bytes `u64`.
  The payload length terminates the final vector.

A disabled section consists of exactly one 16-byte trailer: zero vector count,
flags `u32 = 1`, and zero payload length. Unknown flags, missing trailers, hidden
payloads, invalid counts and invalid empty blobs are errors. Older BMP envelopes
are rejected before payload access. Hermes 1.8.125 can migrate old offline
indexes before they are opened by a current-format-only server.

Enabled storage costs 5 bytes per retained posting, 16 bytes per vector, and 16
bytes per field. Disabled storage costs 16 bytes per field. The directory and
payload stay file-backed and evictable; neither is pinned. Opening validates
compact directory bounds and order; scoring validates the selected vectors.
Lookup uses binary search and intersects only nominated vectors with query terms.

## Configuration and scoring

The per-field `bmp_forward_index` setting defaults to `true`. In SDL:

```sdl
field sparse: sparse_vector [indexed<format: bmp, dims: 105879, bmp_forward_index: false>]
```

The setting is persisted in schema JSON and reported in server SDL. With storage
disabled, ingestion, ordinary merges and both BP granularities omit the values
and write the disabled marker. A field excluded from explicit reorder remains
a byte-identical copy. A setting changes replacement output, not immutable live
source files.

BMP search always scores inverted blocks. L1 backfill and record BP use stored
forward values whenever present, without a heuristic or another enable switch.
When values are absent, BP builds its bounded transient graph from inverted
postings. L1 can locate candidates in a logically ordered document map and probe
only their blocks and query terms. An unordered map without forward values
cannot backfill missing scores; it fails with actionable guidance. It never
turns an unsupported lookup into a missing-value default.

`backfill: false` still uses organic scores and learned missing defaults.
Full-text and sparse MaxScore backfill use their own postings readers and are
unaffected by this BMP storage option.

## Build, merge and option transitions

Ingestion emits the same retained quantized entries as the inverted builder.
It retains the already-admitted per-dimension input through forward output,
uses a k-way cursor heap over dimensions and an 8-byte offset per retained
vector, and never builds another posting-sized heap representation. Peak
accounting includes input postings plus grids during inverted output, then
input postings plus offsets during forward output.

Ordinary merge concatenates existing forward payload bytes and streams directory
entries, adjusting only document IDs and byte offsets with bounded scratch. BP
changes physical inverted order, so both record and block reorder also copy
existing forward payloads without decoding or re-quantizing them.

Copy-only merge never materializes absent values. With storage enabled, mixed
presence requires explicit reorder with a uniform storage policy. With storage
disabled, compatible inverted representations can be copied regardless of the
source's storage setting. All-absent copy inputs retain the disabled marker.

Explicitly enabling storage later uses the same current envelope. Reorder can
materialize its absent values under the existing memory budget, using a sorted
physical permutation and block decoders. It admits 12 bytes per real vector of
directory scratch and streams payload output. Existing forward payloads always
use the copy path. There is no older-format reader, writer or merge dispatch.

Generation claims, cold output, fsync, publication and cleanup remain owned by
the existing sparse segment lifecycle. No sidecar or second preparation protocol
is introduced.

## Validation and cost evidence

Regression coverage includes enabled byte identity, old-envelope rejection,
strict optional/empty-section validation, quantized forward/inverted equality,
missing values and ordinals, copy merge and both BP modes, bounded option
transitions, cancellation and partial writer failure. Sync, native async and
WASM use the same format gate.

Compare BP graph and record rewrite against the inverted path with the same
fixture, compiler and machine. Storage, working memory and scoring evidence are
