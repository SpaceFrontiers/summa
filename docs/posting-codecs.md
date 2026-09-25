# Posting block codecs

Reference for the on-disk text posting format written and read by
`BlockPostingList` (`summa-core/src/structures/postings/posting.rs`) and its
submodules (`posting/{validation,reader,impacts,compact}.rs`).

## Status

Implemented and retained:

- Four per-block codecs, `Rounded` (default), `Packed`, `Pfor`, `Simd4x`
  (opt-in), sharing one container and copy-through merges.
- Extended `BPL2` footer with flags 1/2/4/8/16/32 (position cursors, packed
  length bounds, L1 bounds, ratio bounds, impact envelopes, group envelopes).
  The BM25 impact envelope is implemented (`posting/impacts.rs`), opt-in, and
  not a default.
- Query reads trust writer-produced directory and decoded-document invariants.
  Explicit deserialization and merge admission retain structural checks.
- In-block seeks: current/next-document probe, then a bounded binary search.
  Deferred term-frequency and position-prefix accounting on document movement.
- Bounded decoded-block intersections for conjunctions and phrase candidates;
  ordinary sparse seeks retain their existing policy. Fully overwriting decoders
  reuse initialized buffers. See [posting block execution](search-block-execution.md).
- Opt-in `IndexConfig::compact_text`: separate fixed-width posting descriptors,
  four-byte cursors where possible, and POS6 position directories. Opt-in
  `quantized_norms` applies byte4 norms to new ordinary text columns. These
  formats require metadata version 9; previous encodings remain readable.
  See [compact text storage](compact-text-format.md) for the experiment and limits.

Rejected; do not re-add:

- **Tiled in-block seek** (16-value tile maxima before a SIMD scan) and the
  earlier **SIMD linear scan** over the decoded block: correct, but their
  timings did not support selection over the probe + binary search
  ([review](search-benchmark-current.md)).
- **Exact-width Simd4x tails** for postings and positions (`simd4x-v1`):
  regressed on the merged ARM fixture; tails are written `Rounded` (codec 0,
  position tag 0). The reader still accepts the old exact tails.
- **Unbounded unaligned readers** (`read_unaligned::<u64>` past the slice end);
  every bit-packed array decodes through `horizontal_bp128::unpack_block_n`.
- **Elias-Fano, partitioned Elias-Fano, Roaring, vertical BP128** as index
  codecs (see [library codecs](#library-codecs-not-wired)).
- Silent decode fallbacks (unknown codec decoded as `Rounded`, zeroed TFs on a
  bad exception table). Post-admission decode failures are bugs and panic with
  a message naming the invariant.

## Container

```text
[stream][L0][optional descriptors][L1 docs][L1 bounds][cursors][ratios L0+L1][impact dir+records][footer]
```

| Section              | Size                                      | Present when                                                          |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------------- |
| stream               | `stream_len`                              | always                                                                |
| L0                   | `l0_count × 16`                           | always                                                                |
| L1 docs              | `l1_count × 4` (`last_doc` per 8 blocks)  | always                                                                |
| L1 bounds            | `l1_count × 4` (packed `max_tf, min_len`) | flag 4                                                                |
| cursors              | `l0_count × 8` (`u64` positions before)   | flag 1                                                                |
| ratios               | `(l0_count + l1_count) × 4` (`f32`)       | flag 8                                                                |
| impact dir + records | `(records + 1) × 4` offsets, then records | flag 16 (`records = l0_count`), flag 32 adds `l1_count` group records |
| footer               | 24 (legacy) or 44 (extended)              | always                                                                |

L0 entry: `first_doc u32, last_doc u32, offset u32, bounds u32`. With flag 2
the bounds word packs `max_tf` (low 16 bits, saturating; readers fall back to
the list `max_tf` at 65,535) and `min_len` (high 16 bits); legacy lists store an
`f32` max tf there. Block byte lengths are derived from neighbouring L0
offsets, never from the block header, so payloads may carry variable-length
exception tables.

### Footer

Legacy (24 bytes): `stream_len u64, l0_count u32, l1_count u32, doc_count u32,
max_tf u32`.

Extended (44 bytes): the legacy 24 bytes followed by `total_positions u64,
flags u32, min_len u32, magic u32` with magic `"BPL2"` = `0x324C_5042`
little-endian.

Detection: a list is extended iff it is at least 44 bytes long and its last
four bytes are the magic; otherwise it is legacy. The builder's `u16` term
frequencies keep a legacy `max_tf` far below the magic. A legacy list whose
`max_tf` equals the magic is ambiguous and is rejected by section-offset
validation rather than misread (pinned by
`legacy_footer_whose_max_tf_equals_the_magic_is_rejected_not_misread`).

| Flag | Meaning                                                                    |
| ---- | -------------------------------------------------------------------------- |
| 1    | position cursors: `u64` per L0 block                                       |
| 2    | packed `(max_tf, min_len)` L0 bounds; `min_len` in footer                  |
| 4    | L1 group bounds after L1 docs                                              |
| 8    | ratio bounds: `f32` minimum `length / tf` per L0 block and L1 group        |
| 16   | impact envelopes: offset directory + records, one per L0 block             |
| 32   | group envelopes appended to the impact table; requires 16 and 4            |
| 64   | four-byte block descriptors after L0; stream contains only packed arrays   |
| 128  | four-byte position cursors; requires flag 1 and total positions ≤ u32::MAX |

Unknown flags are rejected. Flag 16 requires flags 2 and 8.

With flag 64, descriptors store `count u16, doc_bits u8, tf_bits u8`; the first
document comes from L0. Admission checks fixed-width payload extents from this
metadata without reading payload pages. Pfor retains the legacy layout because
its exception framing is embedded in the payload. Decoded document content is
still checked against L0 before it is returned. Compatible compact copy merges
retain compact headers; mixed layouts may rebuild headers while copying arrays.

## Block header and codecs

Each block holds up to 128 postings: delta-coded doc ids then term
frequencies, each array packed by one codec. The 8-byte header is

```text
[count: u16][first_doc: u32][doc_bits: u8][tf_bits: u8]
```

with the codec id in the top two bits of `doc_bits` (low six bits = width,
values ≤ 32).

| codec     | id  | payload                                                                                                                                                            |
| --------- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Rounded` | 0   | `(count−1)` deltas at 0/8/16/32 bits, then `count` tfs at 0/8/16/32 bits (SIMD widening)                                                                           |
| `Packed`  | 1   | deltas bit-packed at the exact width, byte-padded; tfs likewise                                                                                                    |
| `Pfor`    | 2   | per array: `[n_exceptions: u8][packed low bits at width][exception (pos u8, high u32)…]`                                                                           |
| `Simd4x`  | 3   | full blocks: four-lane library packing of 128 gap-minus-one values (first gap zero) and 128 tfs; tails: horizontal exact packing of `count−1` gaps and `count` tfs |

`Pfor` uses the OptP4D width rule: minimise `n·b + exceptions·(8+32)` with at
most 10 % exceptions; exceptions store the high bits.

Merges copy blocks verbatim and patch only `first_doc` and the skip metadata,
so one list may mix codecs. `Simd4x` writers emit `Rounded` tails; readers
accept codec-3 tails from the first prototype (pinned by
`simd4x_exact_tail_blocks_decode_seek_and_copy_through_merge`).

**The codec id space is exhausted.** A fifth codec cannot be signalled in the
header. Adding one requires a new footer flag (rejected by current readers)
and an `INDEX_META_FORMAT_VERSION` bump so old binaries refuse the index
before touching a block.

Positions (`positions_v2.rs`) reuse the posting codec policy: header byte 3 is
a codec tag, 0 for rounded blocks and 1 for four-lane full blocks; unknown
tags are rejected at admission.

## Selection

| `-O` / `IndexOptimization` | posting codec | term dictionary zstd |
| -------------------------- | ------------- | -------------------- |
| `adaptive` (default)       | `Rounded`     | 9                    |
| `performance`              | `Rounded`     | 1                    |
| `size`                     | `Pfor`        | 22                   |

`IndexConfig::posting_codec` (`summa-tool index --posting-codec`) selects any
codec explicitly. Repository benchmark (`cargo bench --bench
posting_compression -- summary`, Apple M4): exact packing is ~1.8× smaller than
`Rounded` for ~10 % slower decoding, `Pfor` ~2.2× smaller for ~30 % slower.
Elias-Fano variants decode 25–100× slower and Roaring is larger and slower on
these posting shapes.

### Simd4x

Opt-in (`--posting-codec simd4x`), using the `bitpacking` 0.9.3 four-lane
kernels. Measured on the full corpus with the same writer, document order and
gates as the rounded control
([evidence](benchmark-results/simd4x-2026-09-14/manifest.json),
[review](search-performance-review.md)): index bytes 5,067,578,899 →
4,044,864,324 (about 20 % smaller); official TOP_10 geometric mean 803.338 →
824.067 µs (about 2.6 % slower). It is a space trade-off, not the default.

## Format gates

Metadata format **9** is required; formats 6–8 are upgraded on open (see
[row deletion](row-deletion.md)), older indexes must be rebuilt. See
[`INDEX_META_FORMAT_VERSION`](../summa-core/src/index/metadata.rs) and the
[SSTable format gates](../summa-core/src/structures/sstable.rs). The gate is
what protects older readers from the position codec tag; individual block
bytes are never converted.

## Query trust and explicit integrity checks

Normal segment queries trust Summa-produced postings and positions. The owning
`PostingListReader` parses the footer and constructs borrowed views without
scanning block headers, directory ordering, pruning metadata or position blocks.
Decoded document order and endpoints are not rechecked on this path. There is no
full-index audit at open and no validation cache, lock, budget or CLI option.

Explicit `BlockPostingList::deserialize` / `deserialize_zero_copy` still check
structure and decoded document order. `PositionStream::open` and merge admission
also retain their strict checks. Their malformed-input tests remain applicable to
those boundaries. Ordinary search is not an integrity audit of modified bytes.
I/O failures and unsupported envelopes still fail. Decoders establish slice
extents before unsafe kernels; detected posting decode errors retain the shared
constant-sized first-failure record so collectors do not hide a known failure.
Legacy document-indexed positions retain their existing materializing decoder.

<a id="bounded-reuse-of-posting-validation"></a>

### Footer flags describe stored layout

The unknown-bit mask is the compile-time constant `0xff`, one test per footer.
The flags are format descriptors, not eight independent query-time decisions:

| Flags                                | Why readers retain them                                                                                         |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Position cursors / short cursors     | Positions can be absent; four-byte cursors are valid only when totals fit u32.                                  |
| Length / L1 bounds                   | Existing lists encode bounds differently or omit group bounds.                                                  |
| Ratio / impact / group impact bounds | Optional pruning sections trade extra metadata and build cost for tighter bounds; short lists may omit impacts. |
| Compact headers                      | Compact streams relocate headers; Pfor currently retains inline exception framing.                              |

New writers already choose short cursors when eligible under compact encoding.
A default change cannot substitute for reading the flags of existing immutable
segments. Forcing every section present would add bytes and break compatibility.
No flag or writer default is changed by trusted query reads.

## Bounds metadata

<a id="experimental-ratio-bounds-2026-09-13-follow-up"></a>

### Length and ratio bounds (flags 2, 4, 8)

Per-block `(max_tf, min_len)` and per-group maxima/minima drive MaxScore
block skipping. Ratio bounds (`IndexConfig.posting_ratio_bounds`) add one
downward-rounded `f32` minimum `length / tf` per block and group (zero means
unknown), 4.5 bytes per full block. For `tf ≤ T` and `length / tf ≥ r`, BM25 is
bounded by `idf·(k1+1) / (1 + k1·((1−b)/T + b·r/avg))`; the query owner
evaluates it in f64 with the canonical scorer's rounding inflation. Lengths are
capped at `MAX_CHUNK_LENGTH` inside the constructor
(`from_posting_list_with_ratio_bounds`, `from_posting_list_with_impact_bounds`)
because persisted scoring lengths saturate there; callers cannot produce an
over-tight bound. Measured results: [ratio follow-up](search-benchmark-ratio-results.md).

<a id="proposed-bm25-impact-envelope-research-not-implemented"></a>
<a id="whole-vocabulary-storage-audit-and-narrower-first-experiment"></a>

### BM25 impact envelopes (flags 16, 32)

Implemented, opt-in (`IndexConfig.posting_impact_bounds`,
`--posting-impact-bounds`); implies flags 2 and 8. For positive tf the BM25
factor depends on the minimum nonnegative linear combination of
`(1/tf, length/tf)`, so each block stores the lower convex hull of its
`(tf, effective_length)` pairs: sorted by descending tf then ascending length,
dominated points removed, orientation computed with exact `i128` products.
Records are `count u8 (1–8)` followed by that many unsigned-vint pairs (at most
81 bytes); an empty record means unknown (also used when the hull exceeds eight
points, never a truncated hull). Flag 32 appends one record per L1 group, the
complete minimum over its blocks; any unknown constituent makes the group
unknown. Copying merges keep L0 records byte-for-byte and copy aligned group
records; regrouped groups are recomputed from at most 64 stored points. The
query owner converts the weighted reciprocal minimum to a conservative bound
with the canonical scorer's guards; unsupported parameters or absent records
fall back to the extrema/ratio bound. Evidence:
[impact audit](benchmark-results/impact-audit-2026-09-13/README.md),
[competitive impacts run](benchmark-results/competitive-impacts-2026-09-14/manifest.json).
The L0-only build was flat on the 100k ARM workload; group envelopes are an
opt-in experiment, not a default.

<a id="proposed-traversal-and-position-accounting-separation"></a>

## Cursor behaviour

- Opening a posting list borrows its L1 document ends and packed bounds from
  the validated `OwnedBytes` range. Four-byte little-endian views support
  unaligned files without allocating or copying two group arrays per query.
  Writers still construct owned bytes; the serialized representation is unchanged.
- `seek` compares with the current posting and the decoded block's last id
  first; leaving the block gallops over L1 group ends before a bounded L0
  search; inside a block it probes the next document, gallops to bound the
  remaining distance, then binary-searches that interval.
- The shared sorted-ID helper used by ranked term cursors scans eight IDs on
  AVX2-capable x86 CPUs with an unsigned comparison and packed match mask.
  SSE2 handles short tails and older x86 CPUs; ARM NEON and the portable
  fallback retain their existing kernels. Loads stay inside the supplied slice.
- Two-term ranked conjunctions intersect decoded blocks with local row offsets;
  cursor seek/decode still owns block transitions and validation. Compact
  128-document count/score batches use the same posting iterator kernels for
  copying IDs and retaining sorted candidates. Scoring opts into frequency
  decoding and the canonical batch formula; count-only batches do not.
- Document movement never decodes term frequencies. `term_freq`,
  `position_cursor` and window/run visitors initialise the block's frequencies
  on first use (inline `OnceLock` scratch, no per-cursor heap allocation
  beyond the 128-id buffer).
- Position cursors (flag 1) give the stream offset of a block; the per-posting
  offset is the block cursor plus the tf prefix, committed at most once per
  block by phrase consumers.

## Library codecs (not wired)

`horizontal_bp128`, `vertical_bp128`, `opt_p4d`, `elias_fano`,
`partitioned_ef` and `roaring` remain standalone codecs exercised by
`benches/posting_compression`. `horizontal_bp128::pack_block_n` /
`unpack_block_n` are the single little-endian bit packer/unpacker shared by
`Packed`, `Pfor` low bits, `Simd4x` tails and the OptP4D library codec.

## Validation

- `posting::tests`: round trip, seek/advance equality and byte-identical
  `Rounded` output for every codec; mixed-codec concatenation; legacy footer
  reads; cursor, window and run accounting; content-corruption regressions;
  codec-3 tail fixture; footer ambiguity; mixed-cursor rejection.
- `posting::validation::tests`, `posting/merge_admission_tests.rs`: malformed
  headers, directories, exception tables and trailers rejected before output.
- `horizontal_bp128::tests`: every width and tail against a bit oracle, with
  the input placed before a protected page to catch over-reads.
- `bitpacking4x::tests`: canonical four-lane bytes for every width and tail.
- Integration: `tests/corrupt_posting_content.rs`, `tests/corrupt_text_postings.rs`,
  `tests/corrupt_text_positions.rs`, `tests/posting_reader_resources.rs`,
  `tests/simd_posting_format.rs`, `tests/impact_ranked_collection.rs`,
  `tests/ratio_chunked_collection.rs`.
- `benches/core_structures.rs`: `block_postings/{rounded,packed,pfor}` size,
  sequential decode and skip-seek on the production container.

### Unique-position certificate for exact phrase bounds

A term stream whose positions are strictly increasing within every document
allows exact phrase frequency to be bounded by any phrase term's frequency,
provided the **original first term** has this property. Distinct first starts
map injectively to the required position in every other term. Sloppy phrases
and streams containing duplicate first positions retain the original-first TF
bound. This changes pruning only, never occurrence counting or BM25 scores.

Current position streams use footer magic `POS5` (interleaved blocks) or `POS6`
(compact directory). The high bit of the footer's block count is the unique
positions certificate; the remaining 31 bits are the block count. Both layouts
support duplicate positions with the bit clear. There are no extra bytes.
Earlier position formats are rejected; existing indexes must be rebuilt. There
is no legacy decoder or migration path in query, merge, compaction, or reorder.

The canonical writer observes equality while sorting/delta-encoding each
document. Raw delta appends cannot certify document boundaries and clear the
property. Encoded concatenation combines certificates with logical AND;
doc-aligned compaction/reordering preserves the source certificate while copying
payloads. No query-time scan or schema/tokenizer inference establishes it.

For certified exact phrases the scorer chooses the rarest term for score
admission, uses that term's existing block metadata and TF/length bounds, and
only aligns other terms for competitive candidates. Scratch stays bounded as in
first-term admission. Writer, copied merge, deletion compaction, reorder,
duplicate-position, and sloppy-phrase cases are covered by regression tests.
The [paired 10M benchmark](benchmark-results/searchbench-2026-09-23-phrases.md)
measures the retained optimization; impact metadata remains opt-in.
