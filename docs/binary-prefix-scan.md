# Binary IVF prefix-first scan (proposal)

Status: proposal. Nothing here is implemented. Measurements below come from
read-only samples; adopting this needs the implementation and validation plan
at the end.

## Problem

Binary IVF query time is the exact Hamming scan of the probed leaves, and that
scan is memory-bandwidth bound ([geometry](binary-ivf-geometry.md)). A
production field (2,560-bit `pplx-embed-v1-4b` codes, `target_vectors` 3B,
4·sqrt(N) leaves, 128 probes) touches about 1.75M postings per query: 560 MB of
codes per query, warm or cold. Bytes read, not instructions, set the cost.

## Idea

Rank the probed postings by Hamming distance on the first `P` bits of each code,
keep the best `M`, and rescore only those on the full code. Matryoshka-trained
embeddings put their most informative dimensions first, so a short prefix
preserves neighbour order much better than the same number of arbitrary bits.
For 2,560-bit codes a 640-bit prefix reads 4× fewer bytes per posting; the
second stage adds `M` full-code reads per query.

## Evidence

Production sample: 404,086 binary `content` vectors from one hash partition of
the production embedding table (`pplx-embed-v1-4b`, 2,560 bits, packed MSB-first
like the index builder), and 2,358 real user query embeddings from the
search-api cache. Both were read without writes and analysed on the deploy
host. For each query the exact Hamming top-10 over the whole sample is the
ground truth (tie-aware). The cascade ranks the _whole sample_ by prefix
distance, keeps `M`, and rescores on full codes. That is a stricter pool than an
IVF probe set. The median nearest neighbour is 924 bits away and the 10th is 973.

Fraction of the exact top-10 retained:

| Prefix (leading bits) | Bytes per posting | M = 500 | M = 1,000 | M = 2,000 | M = 5,000 |
| --------------------- | ----------------- | ------- | --------- | --------- | --------- |
| 256                   | 32 (10× fewer)    | 0.575   | 0.674     | 0.764     | 0.866     |
| 512                   | 64 (5× fewer)     | 0.864   | 0.919     | 0.956     | 0.983     |
| 640                   | 80 (4× fewer)     | 0.926   | 0.961     | 0.981     | 0.995     |
| 1,024                 | 128 (2.5× fewer)  | 0.989   | 0.996     | 0.999     | 1.000     |
| 1,280                 | 160 (2× fewer)    | 0.997   | 0.999     | 1.000     | 1.000     |

Taking the same number of _trailing_ bits is consistently worse (640 bits:
0.889 / 0.932 / 0.961 / 0.984), which confirms the Matryoshka ordering. A
non-Matryoshka control (Cohere `embed-english-v3`, 1,024 bits) needed half the
code: [geometry](binary-ivf-geometry.md).

Bandwidth check (x86 AVX-512, one core, warm memory, synthetic codes): one
production-scale query over 1.75M postings takes 60 ms as a full 2,560-bit scan
and 22 ms as a 640-bit prefix scan with histogram selection of 2,000 candidates
and full-code rescoring (floor about 15 ms at the measured 9 GB/s).

Candidate operating points: 640 bits with `M` ≈ 2,000 (0.981 retained, 4×
fewer scan bytes), or 1,024 bits with `M` ≈ 1,000 (0.996, 2.5× fewer). `M` must
grow with the probed pool; at 3B vectors and 128 probes the pool (about 1.75M)
is four times this sample, so the retained fraction must be re-measured at that
pool size before choosing defaults.

## Proposed layout (option B: additional prefix column)

Keep today's run columns unchanged (`doc_ids`, `ordinals`, exact `codes`) and
add, per run, a contiguous `prefix` column of `count × P/8` bytes holding the
first `P` bits of each code in row order. The global quantizer, routing, exact
lookup and every existing reader path stay as they are; only the leaf scan
reads the new column. Storage grows by `P / dim` (25% at 640 of 2,560 bits).

Option A (splitting each code into prefix and suffix blocks with no
duplication) saves that storage but makes every exact-vector read a two-part
gather and changes lookup span semantics. Option B comes first; A can follow
if the storage matters.

Format and compatibility:

- A new ANN layout revision for `BinaryIvf` (and `ScannBinary` if adopted
  there) carries `prefix_bits` in the header; readers reject unknown revisions
  and payloads whose run directory lacks prefix offsets. Fields without
  `prefix_bits` keep the current layout byte for byte.
- The run record gains the prefix column offset. Build, rebuild, deletion
  compaction and the coalescing merge copy the prefix column exactly like the
  codes: each run's prefix rows move with its codes, so lookup relocation is
  unchanged.
- Schema: `indexed<ivf, prefix_bits: 640, prefix_rerank: 2000>`, with
  `prefix_bits` a positive multiple of 64 below `dim`. Changing `prefix_bits`
  rewrites payloads from retained exact codes (an ALTER without retraining);
  `prefix_rerank` is query-time metadata.

## Query algorithm

1. Route and select probed leaves exactly as today.
2. Scan the prefix columns of the probed runs with the existing resolved
   Hamming kernel (AVX-512 masked loads already cover 64–80-byte rows). Keep the
   best `M` by prefix distance. Prefix distances are small integers (at most
   `P`), so a histogram threshold selects `M` without a heap. Visibility
   filtering happens before selection.
3. Read the `M` candidates' full codes (one scattered read each from the
   existing codes column) and score them exactly. Multi-value combiners and
   deduplication then work exactly as today, on exact scores.
4. Expose `prefix_scanned`, `prefix_rerank_candidates` and the stage-2 bytes
   read as query diagnostics.

Cold storage: stage 2 is `M` random reads. On warm pages they are cheap; on
cold NVMe they are about `M` 4-KiB page reads per query, which must be
measured against the saved sequential bytes before choosing `M`.

## Validation plan

- Recall: offline study per production model (this document) and recall@10
  against exact full-code IVF on the real index, per `P` and `M`.
- Byte identity: fields without `prefix_bits` must produce byte-identical
  payloads; prefix columns must equal the leading bits of the exact codes.
- Merge, deletion compaction, reorder and ALTER keep prefix columns aligned
  with codes; corrupt or misaligned prefix columns are rejected at open.
- Latency: warm and cold, single segment and fan-out, x86 and ARM, against the
  same index without prefixes.
