# Bitmap posting blocks

September 28, 2026. Status: **default** (`PostingCodec::RoundedBitmap` for
the `adaptive` and `performance` optimization modes; `size` keeps `Pfor`).
Exact counts over frequent terms are 1.1–1.6× faster on x86 and aarch64;
term, conjunction and disjunction top-k are within ±2% on x86 and up to 10%
faster for conjunctions on aarch64 in a controlled comparison. In the
[September 28 campaign](benchmark-results/default-2026-09-28/README.md) the
default build answers frequent-term counts 1.09–1.56× faster than a
`Rounded` build from the same builder; phrases (−3% to −6%) and prefix3
top-k (−17%) moved the other way between those two builds, which also differ
in document order (see that report's caveats).

**Scope:** full-text posting lists (`BlockPostingList`: term, conjunction,
disjunction and phrase queries, counts, and wildcard/regex/prefix unions).
Learned-sparse fields (BMP, sparse MaxScore blocks, Seismic), dense vectors
and position streams keep their own encodings; `PostingCodec` does not reach
them (position streams treat `RoundedBitmap` as `Rounded`).

## Problem

Exact counts over frequent terms were 2.3–5.4× slower than Luxir on the 10M
Searchbench index (single segment, x86; Summa isolated single-thread medians
against Luxir's loaded latencies, which match its CPU per request):

| COUNT query          |   Matches |   Summa |  Luxir |
| -------------------- | --------: | ------: | -----: |
| `+as +by`            | 1,977,765 | 15.0 ms | 5.5 ms |
| `in on`              | 7,831,136 | 18.3 ms | 5.8 ms |
| `(www\|http\|https)` | 3,502,663 |  6.7 ms | 2.7 ms |
| `ht*p`               | 3,493,796 |  7.2 ms | 2.9 ms |

Profiles put 65–72% of that time in setting one membership bit per decoded
posting (`fill_doc_window`, the union bitset) and 11–12% in 8-bit delta
decoding. A branch-free bit setter (`set_sorted_doc_bits`) removes the
store-forwarding chain, but it still costs ~0.9 ns per posting on x86 and
improves these families by only 5–7%. Luxir answers in ~0.5 ns per posting
_including_ decoding, so dense blocks must not be processed per posting.

## Format

A block whose documents are dense stores them as a bitmap instead of deltas:
`8 × ceil((last − first + 1) / 64)` bytes against `(count − 1) ×` the rounded
delta width.

- **Header:** the existing 8-byte header (or compact descriptor) with
  `doc_bits = 0x3F`: codec bits `Rounded`, width 63, which no codec emits
  (widths are ≤ 32, and earlier readers reject anything larger).
- **Payload:** whole little-endian `u64` words, bit `i` ↔ document
  `first_doc + i`, then the term frequencies in the `Rounded` encoding. Bits 0
  and `last − first` are set, the population equals `count`, and no bit lies
  past the range; admission checks all three.
- **Extent:** derived from the L0 entry (`first_doc`, `last_doc`), so
  compact-descriptor admission stays metadata-only.
- **Merges:** bits are relative to `first_doc`, so copy merges patch the
  header exactly as for delta blocks. Range compaction and reordering
  re-encode through the same writer.
- **Selection:** `PostingCodec::RoundedBitmap` (`summa-tool
--posting-codec rounded-bitmap`) is `Rounded` plus a bitmap wherever it takes at most half
  the delta bytes, which for 8-bit deltas means one ID in four or denser.
  Sparser bitmaps expand into IDs slower than their deltas decode (below).
  `Packed`, `Pfor` and `Simd4x` are unchanged; a list may mix block forms.

Compatibility: introduced by `INDEX_META_FORMAT_VERSION` 10; builds that
predate the change refuse the index at open ([compatibility](compatibility.md)). On the 10M index the posting file shrinks
by 2% (7.17 → 7.05 GB): frequencies and positions dominate its size.

## Readers

1. **Membership windows** (`fill_doc_window`, union materialization, windowed
   conjunction counts): bitmap blocks that end inside the window are ORed in
   as shifted words, never decoded.
2. **Membership probes** (`retain_doc_batch`): candidates that land in bitmap
   blocks are bit tests. Only the block holding the batch's last probe is
   decoded, so the cursor rests there as the batch contract requires.
3. **Ranked conjunctions:** when the common term's current block is an
   undecoded bitmap, each candidate of the rarer term is a bit test, and its
   frequency slot is the rank (popcount of earlier bits). Frequencies decode
   through the cursor's existing deferred path; IDs never do.
4. **Decode to IDs** (all other consumers): eight lanes per byte from a
   bit-position table, widened in one 256-bit register with AVX2 (portable
   fixed-width chunks elsewhere), one bounds decision per word.

`vpcompressd` on 512-bit registers expands fastest in isolation (x86
Cascade Lake, ns per ID; the table column is the portable form):

| Block density | `vpcompressd` zmm | byte table | 8-bit deltas (scalar) |
| ------------- | ----------------: | ---------: | --------------------: |
| 13%           |              0.49 |       0.82 |                  0.36 |
| 20%           |              0.33 |       0.56 |                  0.36 |
| 35%           |              0.20 |       0.33 |                  0.36 |
| 60%           |              0.13 |       0.21 |                  0.36 |
| 90%           |              0.11 |       0.16 |                  0.36 |

but it is **rejected**: inside query execution, sporadic 512-bit instructions
downclocked the core. Decoding the whole list of `date`, 3% of whose blocks
are bitmaps, took 35% longer with it (12% with the table), and the
alternating `Rounded` control arm itself ran 8–10% slower beside it. The
256-bit table decodes the densest lists faster than 8-bit deltas (`the` −10%,
`in` −2%) and 25–50%-dense ones 35–47% slower (`ref`, `on`, `was`): at those
densities per-byte expansion costs about twice the uops of SIMD delta
decoding, which is why blocks sparser than one ID in four stay delta-coded.

## Measurements

**Controlled ranked comparison.** Parallel indexing assigns different
document IDs in every build, which alone moved untouched families by up to
±7% between two builds. The controlled harness therefore takes each term's
real postings from one index, encodes them both ways with the segment's
lengths and impact bounds, opens them as trusted query views and times the
same `MaxScoreExecutor` top-10 alternately (x86, one pinned core, medians of
21). All 593 term, conjunction and disjunction queries return bit-identical
IDs and scores.

| TOP_10 family       | Bitmap blocks |  Rounded |    RoundedBitmap |
| ------------------- | ------------: | -------: | ---------------: |
| high_term           |           37% |  3.70 ms |  3.81 ms (+2.8%) |
| med_term / low_term |            0% |  3.08 ms |          3.08 ms |
| and_high_high       |           20% | 366.8 ms | 356.2 ms (−2.9%) |
| and_high_med        |           13% | 195.5 ms | 193.0 ms (−1.3%) |
| and_high_low        |           15% |  14.2 ms |  14.5 ms (+1.7%) |
| or_high_high        |           20% | 338.7 ms | 341.6 ms (+0.9%) |
| or_high_med         |           11% | 144.0 ms |         143.8 ms |
| or_high_low         |           12% |  24.6 ms |          24.7 ms |

**Exact counts** (same binary against a `Rounded` and a `RoundedBitmap`
build of the corpus; sums of single-thread medians; every count agrees):

| COUNT family  | Rounded |   Bitmap ≥ 1/8 |   Bitmap ≥ 1/4 |
| ------------- | ------: | -------------: | -------------: |
| and_high_high |  316 ms | 157 ms (0.49×) | 203 ms (0.64×) |
| or_high_high  |  290 ms | 142 ms (0.49×) | 177 ms (0.61×) |
| and_high_low  |  7.4 ms | 4.9 ms (0.67×) | 6.0 ms (0.82×) |
| wildcard      |   40 ms |  35 ms (0.86×) |  36 ms (0.90×) |
| regex         |   35 ms |  31 ms (0.90×) |  30 ms (0.88×) |
| prefix3       |  3.6 ms | 3.6 ms (1.00×) | 3.3 ms (0.91×) |

Counting rarely expands bitmap blocks (windows copy words, probes test bits
and decode only the block a batch ends in), so these barely depend on the
decode kernel. The one-in-eight
threshold counts faster still, but its 13–25%-dense bitmaps decode about
twice as slowly as their deltas for ranked disjunctions. Audits of all 673
audit queries on the two builds agree on every count and on the IDs of every
score group except the last, cut by the top-100 boundary.

**Phrases.** med_phrase TOP_10 was 8% slower on the bitmap build than on a
same-code `Rounded` build (single-thread, 46 queries) and 7% slower in the
HTTP campaign, but the work differs, not the decoding: the bitmap build
performed 16% more phrase confirmations and position reads, 7% more bound
calls and 5% more seeks for the same queries, and decoding took a similar
share of both profiles (3.7% vs 4.4%). The two builds assign document IDs in
different orders (parallel indexing), which moves block boundaries, bounds
and how fast the threshold rises; the codec cannot change the number of
confirmed phrase candidates.

**Controlled comparison, current tree** (after per-frequency pruning in
ranked conjunctions, which the bitmap probe path applies too):

| TOP_10 family             | x86 Rounded → Bitmap | aarch64 Rounded → Bitmap |
| ------------------------- | -------------------: | -----------------------: |
| high_term                 |                +0.4% |                    −0.5% |
| and_high_high             |                +1.5% |                    −5.4% |
| and_high_med              |                +2.0% |                   −10.5% |
| and_high_low              |                −1.8% |                    −1.1% |
| or_high_high              |                +1.6% |                    −0.2% |
| or_high_med / or_high_low |        +0.9% / +0.4% |             0.0% / −0.1% |

aarch64 is an Apple M-series laptop with a 1M-document index built from the
same corpus; its exact counts on a `Rounded` and a `RoundedBitmap` build
move like x86's (and_high_high 0.62×, or_high_high 0.61×, and_high_low 0.84×,
wildcard and regex 0.90×, prefix3 0.99×). The [x86 campaign against Luxir](benchmark-results/bitmap-blocks-2026-09-27/README.md)
puts bitmap-build counts at 0.85× Luxir for and/or_high_high (from ~0.55×)
and above Luxir for and_high_low and and_high_med.
