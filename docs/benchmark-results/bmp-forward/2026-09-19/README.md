# BMP gap encoding on retained production vectors

Date: 2026-09-19. Three `documents_20260912` shards, read-only snapshots of
published metadata and immutable BMPA files. On production, no index files or
live settings were changed, and no caches were dropped. Raw vectors remain in the private
workspace; only aggregate results and measurement code are checked in.

## Population and sampling

Metadata describes 101,492,098 physical documents, including 208,230 marked
for deletion, and 2,338,119,515 retained sparse vectors across two fields.
The schema declares a 105,879-dimension multilingual sparse model. The
`sparse_vectors` field has nonzero ordinals despite `multi: false` in schema:
actual persisted logical units, rather than that flag, determine storage.
There are about 22 passage vectors per document plus one short-document vector.

The sample reads eight deterministic stratified windows of 16 consecutive rows
per eligible field/segment, seed 260919: 9,472 rows and 1,455,838 retained
postings. Footer/directory/payload reads total 7,451,538 bytes. These are bytes
requested by the sampler, not physical disk-read counters. No full-corpus
payload scan was performed. Segments too small for a window contribute only
45,685 unsampled payload bytes to the extrapolation.

| Field                 | Retained vectors | Exact mean NNZ from footer/payload sizes | Sample mean / p95 NNZ | Gaps ≤255 | Gaps ≤65535 |
| --------------------- | ---------------: | ---------------------------------------: | --------------------: | --------: | ----------: |
| Passage sparse        |    2,236,627,417 |                                   160.75 |          161.87 / 259 |    53.13% |    99.9995% |
| Short-document sparse |      101,492,098 |                                   150.14 |          145.53 / 237 |    48.89% |    99.9999% |

Sample windows cover 37 nontrivial segments, with equal sampling per segment.
The overall sample mean is therefore not a population-weighted mean. Full
payload estimates below apply each segment's sampled byte ratio to its actual
payload size; the tiny unsampled remainder uses its field's overall ratio.
These are estimates, not rewritten-file measurements or confidence intervals.

## Measured bytes and estimated storage

The Rust harness includes the production shared dimension codec and BMP row
writer directly. Every reconstructed `(dimension, impact)` tuple matches the
BMPA input, and integer scores match an independent raw-row oracle. U8 impacts
and their existing scale are unchanged. Packet tags: 15,268 DotVByte, 801 U24,
6 U16, zero raw-U32 packets in this sample. Full U32 fallback is covered by
regression tests independently of this vocabulary. Extracting the Seismic
codec into its shared owner was also compared against the archived pre-extraction
encoder: identical tags and bytes for all 9,472 rows in raw and compact modes,
plus 1,932 boundary cases (`seismic-extraction.txt`).

| Field                 | Sample BMPA payload | Sample BMPB payload | Payload reduction | Reduction including 16-byte row directory |
| --------------------- | ------------------: | ------------------: | ----------------: | ----------------------------------------: |
| Passage sparse        |         3,833,090 B |         2,100,883 B |            45.19% |                                    44.31% |
| Short-document sparse |         3,446,100 B |         1,923,801 B |            44.17% |                                    43.22% |
| Combined              |         7,279,190 B |         4,024,684 B |            44.71% |                                    43.80% |

Extrapolated forward payload: **1.874 TB → 1.028 TB**, saving **846 GB**
(decimal). Directory bytes remain 37.41 GB. Total sparse blobs would decrease
from about 3.894 TB to 3.048 TB (21.7%); all enumerated shard files total 8.542 TB,
so this alone is about a 9.9% overall disk reduction. Neither quantity is a
resident-RAM requirement: forward payload and directory remain evictable.

The row writer struct is 1,040 bytes on ARM64, independent of vector length;
its 128-entry scratch, fixed impacts, and bounded temporary dimension buffers
consume a few KiB. The decoder retains eight U32 lanes. Benchmark input/output
fixtures are held in heap for measurement, so their process RSS would not model
production mmap residency. No corpus-sized compression dictionary is introduced.

## Resident CPU

Same machine, compiler, flags, input rows, and integer-query semantics for both
layouts. The query has 16 deterministic sorted dimensions with U16 weights;
50 complete passes per timing sample, seven samples, alternating layout order.
The CPU test uses existing bytes in heap and excludes lookup, prefetch, I/O,
query preparation, and response hydration. It is not an end-to-end L1 latency
or QPS benchmark. All checksums match.

| Layout                             | Median time per row |
| ---------------------------------- | ------------------: |
| BMPA U32+U8 reference              |           164.66 ns |
| BMPB packets, fused dimension fold |           263.84 ns |

Compressed scoring is about **1.60×** the raw-row CPU cost on this ARM64 test
(about 99 ns more per row). An initial nested-iterator version measured about
420 ns; dispatching once per packet and feeding decoded groups directly into
the same integer scorer removed much of that overhead. BP's iterator remains
available; this test does not measure full graph construction/rewrite time.

Compression is justified here by production storage distribution, with a
measured resident CPU tradeoff. Ordinary BMP retrieval remains inverted and
has no direct forward decode cost. The [query-latency follow-up](query-latency.md)
now measures public core candidate latency, page faults and memory pressure on
x86. Production-server latency and concurrent QPS remain unmeasured.

## Locality limit

For uniformly aligned 4 KiB pages, a random row of length L needs on average
`1 + (L - 1) / 4096` pages. At the sample's mean row lengths (768.5 B raw,
424.9 B compressed), ten scattered rows require about 11.87 versus 11.03 pages
before caching, metadata, or filesystem read-ahead. Ten consecutive rows need
about 2.88 versus 2.04 pages. These are alignment-model calculations, not disk
measurements. They show why 44.7% fewer bytes does not imply 44.7% fewer random
I/Os: most isolated rows still touch at least one page. Candidate expansion and
page-cache misses remain more important than the final top-10 response size.

## Other fields and next opportunities

A second read-only inspection requested 35.1 MB of block/column metadata and
sampled 596 inverted BMP blocks. See `other-fields-summary.json`. A further 1.21 MB read sampled fast values
from the largest segment of each shard (`fast-values-summary.json`).

- **Duplicate binary codes (already addressed by the repository writer):** an
  additional 5,016 bytes of `.vectors` TOC/header reads found IVF sections
  (type 6) totaling 762.318 GB **and** flat binary sections (type 4) totaling
  762.227 GB for the same fields/vector counts. There are no exact-location
  sections (type 11) in these files. The current `segment/builder/dense.rs`
  already writes one exact ANN code copy plus `vector_locations.rs` lookup
  entries (14 bytes/vector) when a trained binary ANN is available. Rebuilding
  through that existing path suggests about **729 GB net additional savings**
  after row lookup bytes, before small span/block metadata. This is an existing
  code capability absent from these physical files, not a feature added by this
  patch or a deployed/rebuilt-file measurement. Preserve exact lookup, ordinals,
  global quantizers and merge ownership when applying it.
- **BMP inverted IDs:** U32 dimension arrays occupy 3.90 MB of 11.72 MB sampled
  block payload (33.3%). About 96.4% of adjacent dimension gaps fit one byte.
  U24 alone could save about 8.3% of these sampled block bytes while retaining
  direct indexed access. Gap encoding could save more, but its effect on binary
  search/random term lookup needs a separate design with restart points; do
  not transplant the sequential forward layout into this hot path unchanged.
- **Missing fast values:** language columns total 816.8 MB. Blocks covering
  100.4M rows use 64-bit bitpacking although block dictionaries are small.
  Issued-at columns total 749.7 MB, with 83.6M rows in 64-bit blocks. The existing
  `u64::MAX` missing sentinel explains width inflation for dictionary ordinals;
  separating validity from value codes is the useful next fast-column experiment.
  These counts describe rows in wide blocks, not counts of missing values.
  In the additional sample, language IDs range 0–39 with 37/3,072 missing
  values (1.2%). Issued-at has one missing value among 2,969 decoded values;
  103 values in a different codec were excluded. This is a small, biased
  three-segment sample, but it confirms that even rare missing values can
  inflate a whole block. Timestamp values also span from year 1 to 2026;
  compression must preserve that range rather than assume modern dates.
- **Primary-key dictionaries:** ID fast columns total 3.71 GB, including 3.55 GB
  of dictionary bytes. Prefix compression of independent dictionary blocks
  deserves testing against existing primary-key lookup requirements.
- **Impacts:** sampled entropy is about 5.45 bits, but about 63% of impacts exceed 15. Blind U4 packing would be lossy. An independent lossless entropy codec
  might save additional bytes at a decode cost; current U8 scoring stays exact.
- **Larger files:** `.post` is 1.818 TB, `.vectors` 1.525 TB, `.pos` 1.178 TB.
  Positions already use delta coding plus bitpacking, fast values already use
  adaptive numeric codecs, and binary vectors are already bit representations.
  Aside from the binary duplicate-copy finding above, file size alone does not
  justify another generic compressor. Their full payload distributions and
  access patterns were not scanned in this experiment.

## Reproduction and validation

`sample.py INDEX_DIR...` reads only BMPA files and emits private gzip JSON to
stdout. It is an offline measurement utility, not a legacy reader in Summa.
`measure.rs` accepts concatenated `[nnz: u32 LE][nnz × (dimension: u32 LE,
impact: u8)]` rows and an output filename for count-prefixed BMPB row bytes:

```sh
rustc --edition=2024 -O --cfg 'feature="native"' measure.rs -o measure
./measure rows.raw rows.gap
```

`production-summary.json`, `other-fields-summary.json`, `resident-arm64.txt`
and `manifest.json` record aggregates and provenance. The fixed input's SHA-256
identifies the privately retained fixture. Production format gates require
rebuilding BMPA indexes before a BMPB release can open them.

Final checks: 2,021 regular native tests and 38 WASM tests passed, along with
strict Clippy, feature-matrix checks, documentation and server build. The
parallel real-server suite encountered one local port collision; all five
integration tests passed on a serial rerun. The full harness evidence is
`.context/search-harness/20260919T080450.635887Z-full`; the serial rerun log is
`/tmp/summa-bmp-gap-e2e-serial.log`. No production deployment was performed.

The [query-latency follow-up](query-latency.md) uses 900,000 real vectors to
compare raw and packet formats through public search and candidate-scoring APIs.
Mean retrieval-plus-L1 latency is 53.22 → 143.38 ms warm, 1,559.65 → 145.17 ms
at 1 GiB, and 5,221.43 → 3,056.54 ms at 512 MiB on the common query subset.
All 26 pairs match IDs and score bits. The report includes locality, I/O and CPU
profiles; production-server latency remains outside this experiment.
