# BMP forward values and candidate scoring

BMP owns sparse retrieval and exact stored sparse values. L1 addresses a logical
`(document, ordinal)` without an inverse physical-ID sidecar or a corpus scan.
BP reads the same values when building its graph and rewriting records. The
quantized impacts, pruning and query quantization match inverted BMP scoring.
L1 remains opt-in.

## Current format and cost

The `BMPB` envelope contains adaptive inverted blocks, their offset table,
compressed pruning grids, physical document maps, a forward-storage section,
and the 80-byte footer. BMPA and older envelopes require rebuilding. There is
no compatibility reader or merge dispatch for them.

An enabled forward section contains:

- Vector payload in ascending `(document, ordinal)` order. Each row contains
  independent packets of at most 128 entries, followed by its U32 LE total entry
  count. Dimensions are nondecreasing; repeated dimensions retain every impact.
- A sorted 16-byte directory entry per vector: document U32, ordinal U16,
  reserved zero U16, and payload-relative byte offset U64. Adjacent offsets bound
  each row; the payload length terminates the final row.
- A 16-byte trailer: vector count U32, flags U32 = 0, payload bytes U64.

Each packet stores `count: u8` (1..128), `encoding: u8`,
`dimension_bytes: u16 LE`, dimension bytes, then `count` unchanged U8 impacts.
The shared sparse dimension codec selects raw U16/U24/U32 or DotVByte, whichever
fits and is smaller. DotVByte stores an independent U32 base, one control bit per
one/two-byte gap, and up to seven absolute U32 tail IDs. Eight-lane prefix sums
retain the full U32 dimension range. Zero gaps preserve repeated coordinates.
See [Seismic forward compression](seismic-forward-compression.md) for the shared
codec and its SIMD/scalar implementations; BMP keeps its own existing U8 impacts
and segment quantization scale.

Storage is encoded IDs + one byte per posting + four bytes per packet + four
bytes per row + 16 bytes per directory entry, plus the per-field trailer.
Packet overhead can increase very short or uncompressible rows. The production sample reduces
payload by 44.7%, or 43.8% including the directory; this is a storage result,
not an equivalent resident-RAM or query-latency reduction.

A disabled section is exactly one 16-byte trailer: zero vector count,
flags U32 = 1, zero payload length. Unknown flags, missing trailers, hidden
payloads and incompatible envelopes are errors. Both directory and payload
remain file-backed and evictable; neither is pinned. Opening checks directory
bounds and order. Query scoring trusts writer-produced payloads and decodes once;
explicit integrity and BP maintenance validation remain separate.

## Configuration and scoring

`bmp_forward_index` defaults to `true`. In SDL:

```sdl
field sparse: sparse_vector [indexed<format: bmp, dims: 105879, bmp_forward_index: false>]
```

The setting is persisted in schema JSON and reported in server SDL. With storage
disabled, ingestion, ordinary merges and both BP granularities omit the values
and write the disabled marker. A field excluded from explicit reorder remains
a byte-identical copy. A setting affects replacement output, not immutable live
source files. Enabled BMPB storage uses adaptive packet compression directly;
`seismic_forward_compression` is specific to Seismic.

| Operation                              | Owner and behavior                                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ordinary BMP retrieval                 | `query/bmp.rs::score_superblock_blocks` reads adaptive inverted blocks; it does not decode forward rows                                                       |
| L1 candidate backfill                  | Candidate lookup resolves logical rows and admits encoded byte ranges; `CandidateBmpPreparation::score` streams their values into the existing integer scorer |
| BP graph construction                  | `builder/graph_bisection.rs` validates forward values explicitly, then reads them in graph passes                                                             |
| Record reorder                         | `segment/reorder.rs::forward_route_vector` supplies retained impacts to the inverted rewrite; row count is readable without decoding                          |
| Copy merge and BP forward preservation | `bmp_forward/rewrite.rs::write_forward_sources` copies encoded payload and remaps small directory entries                                                     |
| Deletion compaction                    | `bmp_forward.rs::write_compacted_forward` copies surviving encoded ranges unchanged                                                                           |

When values are absent, BP builds its bounded transient graph from inverted
postings. L1 can locate candidates in a logically ordered document map and probe
only their blocks and query terms. An unordered map without forward values
cannot backfill missing scores; it fails with actionable guidance. It never
turns unsupported lookup into a missing-value default.

`backfill: false` still uses organic scores and learned missing defaults.
Full-text and sparse MaxScore backfill use their own postings readers.
Forward compression does not change these execution policies.

## Rescoring accumulator

The query scorer carries `Option<u32>` through the packet fold and constructs the
existing overflow error once at the row boundary. Arithmetic remains checked;
overflow stays sticky. Duplicate query and document dimensions remain additive,
and final dequantization is unchanged. This avoids repeatedly copying the full
`Result<u32, Error>` state in the measured x86 inner loop.

Native, async and WASM share the scorer. It adds no allocation or decoded-row
buffer, changes no encoded bytes, and retains the same O(nonzeros + query terms

- duplicate matches) work and bounded decoder scratch. The
  measures the actual public candidate scorer with identical index files, queries,
  release flags and compiler. Codec, cache policy and defaults are unchanged.

## Build, merge and option transitions

Ingestion emits exactly the retained quantized entries used by the inverted
builder. It retains already-admitted per-dimension input through forward output,
uses a k-way cursor heap over dimensions and an 8-byte offset per vector, and
never builds another posting-sized representation. Peak accounting includes
input postings plus grids during inverted output, then postings plus offsets
during forward output.

`bmp_forward/codec.rs::RowWriter` is the single row encoder for ingestion and
explicit materialization. It buffers 128 `(dimension, impact)` pairs, a fixed
impact array, and bounded encoded-ID scratch, independent of row/corpus length.
The writer struct is 1,040 bytes on the measured ARM64 build; temporary encoding
scratch is a few additional KiB at most. Decoding uses eight U32 lanes, with no
full-vector allocation per candidate. The dimension codec lives in
`structures/postings/sparse/dimensions.rs` and is shared with Seismic without
changing Seismic's version-6 bytes.

Ordinary merge concatenates existing forward payload bytes and streams directory
entries, adjusting only document IDs and byte offsets with bounded scratch. BP
changes physical inverted order, so both record and block reorder also copy
existing forward payloads without decoding or re-quantizing them.

Copy-only merge never materializes absent values. With storage enabled, mixed
presence requires explicit reorder with a uniform storage policy. With storage
disabled, compatible inverted representations can be copied regardless of the
source's storage setting. All-absent copy inputs retain the disabled marker.

Explicitly enabling storage later materializes absent values under the existing
memory budget, using a sorted physical permutation and block decoders. It admits
12 bytes per real vector of directory scratch and streams the same packet writer.
Existing payloads always use the copy path. Generation claims, cold output,
fsync, publication and cleanup remain owned by the sparse segment lifecycle.

## Locality and evidence

Forward rows remain in logical document/ordinal order after physical BP reorder.
Compression packs nearby rows onto fewer pages. It does not make scattered
candidate IDs or physically reordered blocks contiguous. A top-10 response may
require scoring many more nominated candidates; forward bytes are charged for
those candidates, not just returned hits. Directory binary search also touches
metadata pages. No second physical forward copy or corpus-sized mapping is added.

Regression coverage includes a BMPB byte golden, BMPA rejection, packet tails
and boundaries, full-width IDs, duplicate dimensions, quantized forward/inverted
score equality, missing values and ordinals, copy merge, compaction, both BP
modes, storage transitions, cancellation and partial writer failure. Native,
async and WASM share the codec and format gate.

Read-only production sampling, storage estimates and resident decode measurements
not establish production cold-query latency or QPS under a constrained cache.
