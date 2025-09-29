# Bitmap posting blocks

September 28, 2026. Status: **default** (`PostingCodec::RoundedBitmap` for
the `adaptive` and `performance` optimization modes; `size` keeps `Pfor`).
Exact counts over frequent terms are 1.1–1.6× faster on x86 and aarch64;
term, conjunction and disjunction top-k are within ±2% on x86 and up to 10%
faster for conjunctions on aarch64 in a controlled comparison. In the
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

Compatibility: `INDEX_META_FORMAT_VERSION` 9 → 10 with the metadata-only
upgrade on open. Builds that predate the change refuse the index at open.
Old blocks are never converted. On the 10M index the posting file shrinks
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

