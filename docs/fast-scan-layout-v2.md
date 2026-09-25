# FastScan layout v2 for float ScaNN leaves

## Status

Implemented in `summa-core/src/structures/vector/scann/fast_scan.rs`. The
segment payload format version (`SCANN_SEGMENT_PAYLOAD_VERSION`) is bumped
from 1 to 2; version-1 payloads are refused with an actionable error.
Migration is a rebuild of every float ScaNN generation (`alter_vector_index`
to the same configuration, or a full reindex). Binary ScaNN payloads are not
affected: their leaves store exact packed codes, not FastScan groups.

## Why

The float leaf scan spends nearly all of its time in the 4-bit asymmetric
hashing lookup: for every row, every AH block contributes one 16-entry table
lookup. Layout v1 stored one AH block per 16-byte word, two rows per byte
(row `2p` in the low nibble, row `2p + 1` in the high nibble). That shape has
three costs:

1. On AVX2 a 16-byte word only feeds `_mm_shuffle_epi8`, so the machine does
   one 128-bit shuffle per 16 lookups and the upper half of every 256-bit
   register is idle.
2. Results come out even/odd interleaved and had to be de-interleaved into
   row order after accumulation on every kernel.
3. Lookup tables were signed `i8` clamped to `[-127, 127]`; a block whose
   sixteen scores are all positive (or all negative) used half of that range.

## Layout

A complete 32-row group of `B` AH blocks is stored as `ceil(B / 2)` 32-byte
words. Word `j` covers blocks `2j` and `2j + 1`; byte `lane` of word `j` is
the code of row `lane`:

```text
word j (32 bytes)
byte lane = code(block 2j, row lane)  |  code(block 2j + 1, row lane) << 4
            ^ low nibble                  ^ high nibble

group = [word 0][word 1] ... [word ceil(B/2) - 1]
```

An odd block count pads block `B` (which does not exist) with zero codes; its
lookup table is all zeros, so it never changes a score. Because of the
padding, `packed_block_bytes(B) = ceil(B / 2) * 32`, which differs from v1
(`B * 16`) for odd `B` only.

Rows that do not fill a complete 32-row group (a run's tail, at most 31 rows)
keep the v1 row-major nibble pairs (`ceil(B / 2)` bytes per row) and are scored
by the float `AhQuery::score_packed`. Only the group layout changed.

`fast_scan::packed_code_position(row_in_group, block)` is the single
definition of where a code lives; the compaction path
(`ann_disk::unpack_scann_ah_row`) and the packer use it.

## Query-side lookup table

`FastScanQuery::new` builds one unsigned byte table per block:

```text
min_b   = min over the 16 AH scores of block b
range   = max over blocks of (max_b - min_b)
mult    = 255 / range
lut_b[c] = round((score_b[c] - min_b) * mult)      in 0..=255
bias    = sum over blocks of min_b                  (f64 accumulate, then f32)
row score = centroid_dot + bias + acc_row / mult
```

Subtracting the per-block minimum lets every block use the full `0..=255`
range, so table quantization error is never worse than v1 and typically about
half of it. `FastScanQuery::is_degenerate()` reports the all-constant table
case (every block has zero range, e.g. a zero query); `FloatScannQuery::new`
logs a rate-limited warning for it instead of silently scoring with a unit
multiplier.

## Kernels

All three kernels accumulate per row in 16-bit lanes and fold into 32-bit
lanes every 64 words (128 blocks): a word adds at most `2 * 255 = 510` per
lane and `64 * 510 = 32,640 < 65,535`.

- **AVX2**: per word, one 32-byte load, `_mm256_broadcastsi128_si256` of each
  16-byte table, and two `_mm256_shuffle_epi8` (32 lookups each: one 256-bit
  shuffle per block instead of one 128-bit shuffle per half block).
- **NEON**: per word, two 16-byte loads and four `vqtbl1q_u8`; output lands in
  row order so the v1 de-interleave is gone.
- **Scalar**: the reference; `FastScanKernel::Scalar` is also the target for
  the equivalence tests. The kernel is resolved once per `FastScanQuery`
  (which is built once per query inside `FloatScannQuery`), never per block.

The `fast_scan_*` tests in `fast_scan.rs` pin: dispatch == scalar for even and
odd block counts, correctness past the 16-bit flush window with saturated
tables, the bound of the FastScan score against the exact f32 AH score, and
that the integer lane cutoff used for pruning is conservative.

## Version gate

`ScannSegmentPayload::from_bytes` refuses any version other than 2 with
"unsupported ScaNN segment payload version N; reader supports 2".

The ANN container (`segment/ann_disk.rs`) gates the leaf layout as well: the
header's former reserved `u32` now carries `FAST_SCAN_LAYOUT_VERSION` for
`AnnKind::ScannAh` payloads and stays zero for every other kind. A v1 build
left it at zero, and for even block counts a v1 code column has exactly the
v2 byte length, so this field is the only thing that distinguishes a stale
generation. `AnnDiskIndex::open` refuses revision 0 with "ScaNN AH payload
uses FastScan layout revision 0; this build reads revision 2 ... Rebuild the
float ScaNN generation", pinned by
`open_refuses_scann_ah_payload_with_pre_v2_fast_scan_layout`. Non-ScaNN kinds
keep the old "reserved field must be zero" check. Migration is a rebuild of
every float ScaNN generation before the field is searched.

## Measured

`scann_fast_scan/score_block` in `summa-core/benches/scann_vectors.rs`
(4,096 rows = 128 groups per iteration), Apple M4 aarch64/NEON, rustc 1.98,
`--warm-up-time 1 --measurement-time 2`; each cell is the best median of
three interleaved baseline/final runs:

| AH blocks | v1 baseline | v2 final | change |
| --------: | ----------: | -------: | -----: |
|        32 |     4.36 µs |  2.29 µs |   -47% |
|        96 |    12.48 µs |  6.43 µs |   -48% |
|       384 |    48.99 µs | 25.12 µs |   -49% |

On NEON a word still costs two `vqtbl1q_u8` per block; the measured gain comes
from hoisting kernel dispatch, accumulating safely in 16-bit lanes, and
removing the v1 even/odd de-interleave. The structural win is larger still on
AVX2, where the 256-bit shuffle halves the lookup instruction count. That
kernel was written against the scalar reference and is verified by
`fast_scan_dispatch_matches_scalar_scores` and friends in CI, but was **not**
benchmarked locally (no x86 machine).

Accuracy against the exact f32 AH score
(`fast_scan_v2_bias_tables_track_v1_signed_table_accuracy`, synthetic 2-dim
blocks, 32 rows, worst lane; v1 emulated with the old signed table):

| dim (blocks) | v1 max error | v2 max error |
| -----------: | -----------: | -----------: |
|       16 (8) |       0.0205 |       0.0116 |
|      96 (48) |       0.0682 |       0.0426 |
|    384 (192) |       0.0877 |       0.1070 |
|    768 (384) |       0.1757 |       0.0947 |

When some block's sixteen scores span both signs almost symmetrically the v2
step (`widest range / 255`) equals the v1 step (`max |score| / 127`), so the
two tables are equally precise and differ only by rounding luck (dim 384
above); when block ranges are one-sided v2 is roughly twice as precise. The
test pins that v2 is never systematically worse.
