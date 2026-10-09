# Binary IVF prefix-first scan

Status: implemented for binary IVF as a per-field SDL option
(`indexed<ivf, prefix_bits: N, prefix_rerank: M>`); off by default. Binary
ScaNN does not support it. Measurements below come from read-only samples.

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

### Inside a 4·sqrt(N) IVF, with and without SOAR

The same production sample, indexed as a 4·sqrt(N) = 2,543-leaf k-majority IVF
(a NumPy reimplementation of Summa's trainer: k-means++ seeding, 10 iterations)
and probed for each real query. SOAR adds one secondary leaf per vector chosen
by the Hamming form of the SOAR loss with λ = 1. Postings count spilled
duplicates, and bytes read per query are prefix bytes for every probed posting
plus full codes for the `M` rescored ones. Recall@10 is tie-aware against exact
Hamming over the whole sample.

| Configuration                             | nprobe | Recall@10 | MB per query |
| ----------------------------------------- | ------ | --------- | ------------ |
| 4·sqrt(N), full codes                     | 128    | 0.908     | 7.07         |
| 4·sqrt(N), full codes                     | 192    | 0.934     | 10.60        |
| 4·sqrt(N), full codes                     | 256    | 0.950     | 14.13        |
| 4·sqrt(N), prefix 1,024, M = 1,000        | 128    | 0.907     | 3.15         |
| 4·sqrt(N) + SOAR, full codes              | 64     | 0.936     | 7.17         |
| 4·sqrt(N) + SOAR, prefix 1,024, M = 1,000 | 64     | 0.935     | 3.19         |
| 4·sqrt(N) + SOAR, prefix 1,024, M = 1,000 | 96     | 0.952     | 4.60         |
| 4·sqrt(N) + SOAR, prefix 640, M = 2,000   | 128    | 0.957     | 4.20         |
| 4·sqrt(N) + SOAR, prefix 1,024, M = 1,000 | 128    | 0.963     | 6.02         |
| 4·sqrt(N) + SOAR, prefix 1,024, M = 1,000 | 256    | 0.981     | 11.63        |

At equal recall the combination reads 3.1–3.3× fewer bytes than 4·sqrt(N)
alone (0.935: 3.19 versus 10.60 MB; 0.95: 4.60 versus 14.13 MB) and reaches
recall that full-code 4·sqrt(N) does not reach within 256 probes. A 640-bit
prefix with `M` = 2,000 sits on the same frontier as 1,024 bits with
`M` = 1,000. Storage: with the split layout and prefix-only SOAR copies (below) this
configuration stores about 1.38× the bytes per vector of the plain index. SOAR for binary IVF is available as `soar: full`
([binary IVF SOAR](binary-ivf-soar.md)).

## Layout

A field with `prefix_bits` stores each leaf run's codes column split: the
leading `prefix_bits / 8` bytes of every row first, then the remaining bytes of
every row. The column has exactly the size of plain row-major codes, so the
prefix costs no storage. The run directory record is unchanged. Exact-vector
lookups keep addressing rows row-major (`codes start + row × code size`); the
exact-vector reader and the rescoring stage translate that virtual address into
the row's prefix and suffix and assemble the vector, so lookup files, span
validation and merge relocation need no new format. Full scans add the prefix
and suffix Hamming distances, which is exact because Hamming distance is
additive over disjoint bytes.

Format and compatibility:

- A prefixed payload is binary IVF layout revision 1, with the prefix byte
  width in the low 32 bits of the header tail. Earlier readers reject both, and
  readers reject a width of zero, a width at least the code size, or a width on
  any other kind. Fields without `prefix_bits` keep revision 0 and
  byte-identical payloads.
- Segment build and generation rebuilds write split columns. Byte-copy merges
  carry them verbatim; coalescing merges concatenate all prefix blocks and then
  all suffix blocks of a leaf; deletion compaction filters prefix and suffix
  rows. Sources with different prefix widths are incompatible generations, and
  readers reject payloads whose width differs from the schema.
- Schema: `prefix_bits` must be a positive multiple of 8 below `dim` and
  requires `ivf`; `prefix_rerank` (default 1,000) requires `prefix_bits` and is
  query-time only. Changing `prefix_bits` through ALTER retrains the field's
  quantizer and rewrites the field in every segment; changing `prefix_rerank`
  does not.

## Query algorithm

When the probed leaves hold more postings than `prefix_rerank`:

1. Route and select probed leaves exactly as for a full scan.
2. Scan only the prefix columns of the probed runs with the resolved Hamming
   kernel, prefetching labels and prefixes but not codes. Keep the best `M`
   visible postings in a bounded heap ordered by prefix distance, then
   document, ordinal and code offset. That total order makes serial and
   parallel selection agree. Doc IDs are read only for distances that can
   enter the heap.
3. Read only the `M` candidates' code suffixes (the prefix distance is already
   known) and add their distances. Candidates found only in a prefix-only SOAR
   run read the suffix of the vector's primary copy through the exact-vector
   lookup ([binary IVF SOAR](binary-ivf-soar.md)). Collectors, multi-value
   combiners and exact completion then run unchanged, so every returned score is
   an exact Hamming score.

With fewer probed postings than `M` the full scan runs instead.

Cold storage: stage 2 is `M` random reads. On warm pages they are cheap; on
cold NVMe they are about `M` 4-KiB page reads per query, which must be
measured against the saved sequential bytes before choosing `M`.

## Validation

Implemented tests cover the header round trip and rejection of invalid widths;
alignment of prefix rows with codes after byte-copy merge, coalescing merge and
deletion compaction; an independent oracle for the two-stage selection in
serial and parallel modes; SDL parsing and validation errors; and an
end-to-end index build, train, search, merge and ALTER. Remaining before
adopting a production default:

- Recall: offline study per production model (this document) and recall@10
  against exact full-code IVF on the real index, per `P` and `M`.
- Byte identity: fields without `prefix_bits` must produce byte-identical
  payloads; prefix columns must equal the leading bits of the exact codes.
- Merge, deletion compaction, reorder and ALTER keep prefix columns aligned
  with codes; corrupt or misaligned prefix columns are rejected at open.
- Latency: warm and cold, single segment and fan-out, x86 and ARM, against the
  same index without prefixes.
