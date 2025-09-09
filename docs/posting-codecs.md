# Posting block codecs

Status: design (2026-09-04), implemented.

## Problem

`BlockPostingList` (`structures/postings/posting.rs`) is the only posting
format the index writes: 128-posting blocks of delta-coded doc ids and term
frequencies, each array packed at a width rounded up to 0/8/16/32 bits so it
decodes with plain SIMD widening. The repository also carried six standalone
codecs (exact-width BP128, vertical BP128, OptP4D, Elias-Fano, partitioned
Elias-Fano, Roaring) plus an `IndexOptimization` knob (`-O adaptive | size |
performance` in `hermes-tool index`) whose documented effect — "OptP4D + zstd
22", "Roaring + zstd 3" — was never wired: `IndexConfig.optimization` was
stored and never read.

| codec            | size, % of raw (avg) | decode, M ints/s |
| ---------------- | -------------------- | ---------------- |
| rounded 8/16/32  | 27.3                 | 750              |
| exact BP128      | 15.1                 | 674              |
| OptP4D (patched) | 12.2                 | 534              |
| partitioned EF   | 15.7                 | 27               |
| Elias-Fano       | 16.6                 | 8                |
| Roaring          | 39.4                 | 336              |

Only exact bit-packing and OptP4D buy anything: 1.8–2.2× smaller postings for
10–30 % slower decoding. Elias-Fano variants decode 25–100× slower and Roaring
is both larger and slower on the posting shapes this engine sees.

## Design

### One container, per-block codec

The container stays: `[stream][L0 skip entries][L1][footer]`, 128-posting
blocks, zero-copy `OwnedBytes`, block-max `max_tf` per L0 entry, and merges
that copy blocks verbatim with an 8-byte header patch. What changes is the
block payload. The 8-byte block header

```text
[count: u16][first_doc: u32][doc_bits: u8][tf_bits: u8]
```

now stores the **codec in the top two bits of `doc_bits`** (low six bits =
width, values ≤ 32):

| codec     | id  | payload                                                                                  |
| --------- | --- | ---------------------------------------------------------------------------------------- |
| `Rounded` | 0   | unchanged: `(count−1)` deltas at 0/8/16/32 bits, then `count` tfs at 0/8/16/32 bits      |
| `Packed`  | 1   | deltas bit-packed at the exact width, byte-padded; tfs likewise (`tf_bits` exact)        |
| `Pfor`    | 2   | per array: `[n_exceptions: u8][packed low bits at width][exception (pos u8, high u32)…]` |

`Rounded` keeps codec id 0 and the existing 0/8/16/32 payload representation.
That makes it the low-overhead performance baseline; compatibility is governed
by the index-format gate below, not by the individual block representation.

`Packed` uses `bits_needed(max)` per array. `Pfor` is the OptP4D block
encoding already in the tree: the width minimising `n·b + exceptions·(8+32)`
with at most 10 % exceptions, exceptions storing the high bits.

Block sizes are derived from the L0 offsets (`next.offset − offset`, or
`stream_len − offset` for the last block) instead of from the header, so a
payload can carry variable-length exception tables. Decoding dispatches on the
codec id per block; a posting list — and, after a merge, a single list — may
mix codecs freely.

### Selection

`IndexConfig.optimization` finally does something, and `IndexConfig` also
gets an explicit `posting_codec`:

| `-O` / `IndexOptimization` | posting codec | term dictionary zstd |
| -------------------------- | ------------- | -------------------- |
| `adaptive` (default)       | `Rounded`     | 9                    |
| `performance`              | `Rounded`     | 1                    |
| `size`                     | `Pfor`        | 22                   |

`Packed` is selectable through `IndexConfig::posting_codec` (and
`hermes-tool index --posting-codec`) for deployments that want most of the
size win with SIMD-friendly single-array decode. The default stays `Rounded`:
changing the default codec is a cross-architecture performance decision that
the numbers above (one machine, synthetic lists) do not justify on their own.

The merger writes the index's codec on its decode–re-encode path (inline or
mixed sources) and copies blocks verbatim on its fast path, so a segment
merged from `size` and `adaptive` sources simply contains both kinds of
blocks.

### Format break (fail loud)

This layout originally shipped with metadata format **5**. The current
reader requires metadata format **7** (format **6** is upgraded on open, see
[row deletion](row-deletion.md)) and STB5 term dictionaries. These
version numbers belong to separate formats and need not advance together. Older indexes must be rebuilt; see
[`INDEX_META_FORMAT_VERSION`](../hermes-core/src/index/metadata.rs) and the
[SSTable format gates](../hermes-core/src/structures/sstable.rs).
Unknown per-block codec ids are rejected instead of being decoded as rounded
32-bit blocks.

### What is not wired

## Validation

- `posting::tests`: round trip, `seek`/`advance` equality and byte-identical
  `Rounded` output against the pre-change layout for every codec, including
  lists with delta and tf outliers that force exceptions.
- Streaming and in-memory concatenation of mixed-codec sources.
- Index-level: `-O size` writes Pfor blocks, survives decode/re-encode merges,
  and searches identically.
  serialized size, sequential decode, and skip-seek, on the production
  container rather than the standalone codecs.