# Index diagnostics

Hermes had no way to answer "is this index healthy?" short of reading payload
bytes off disk by hand. Two production incidents motivated this feature, both
of which were invisible until latency regressed:

1. **Zero-vector collapse.** 31% of one binary field's vectors were all-zero
   embeddings from an upstream producer bug. IVF ties resolve to the lowest
   cluster ID, so 20.1M codes (6.0 GiB) accumulated in leaf 0 of every
   segment, and any query probing that leaf scanned all of it (~1 s dense
   queries). Found only by parsing ANN payloads with `dd`.
2. **Extent fragmentation.** Byte-copy merges preserve each source payload as
   one physical extent, so runs for one logical cluster scatter across the
   file. Nothing counted extents per probed leaf, so the read amplification
   was invisible until `diskstats` showed the array IOPS-bound at 32 KB/read.

## Prior art

- **Lucene `CheckIndex`** reports per-segment identity, doc counts,
  deletions, per-part status, and file sizes cheaply, gating exhaustive
  verification behind an explicit `-slow` flag. We copy the tiering: cheap
  observations always, expensive scans opt-in.
- **Faiss** exposes `imbalance_factor()` on inverted lists — the relative
  variance of list sizes. It is the expected multiplier on distance
  computations at fixed `nprobe`: 1.0 is perfectly balanced, and a value of
  γ means probes cost γ× the balanced baseline on average. We adopt it
  unchanged as the canonical skew metric.
- **Elasticsearch's disk-usage API** breaks index footprint down per field,
  behind `run_expensive_tasks=true`. Our TOCs already carry per-field extents,
  so Hermes reports this for free.

## Coverage

Diagnostics span every index structure, not only dense ANN:

| structure                       | cheap (always)                                                                                     | expensive (opt-in)                                                                                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dense ANN (binary IVF / IVF-TQ) | run-directory health (below)                                                                       | `--sample`: zero/NaN scan of flat vectors; `--probe-cost`                                                                                                                     |
| sparse (BMP / MaxScore)         | vectors, postings, dims, blocks, postings per vector, block padding ratio                          | `--sparse-stats`: per-dimension posting distribution (p50/p99/max, top-1% share, hottest dims) and u8 impact saturation                                                       |
| full-text                       | per-field doc count and avg tokens/doc (BM25F stats), term-dict entry/block/bloom/dictionary sizes | `--terms N`: whole-dictionary scan — doc-frequency distribution (p50/p99/max), inline-term ratio, postings/positions bytes, top-1% postings share, field-attributed top terms |
| fast fields                     | per-column type, doc count, multi flag, disk bytes                                                 | —                                                                                                                                                                             |
| store                           | bytes, block count, docs/block, bytes/block, bytes/doc (from the block index, no decompression)    | —                                                                                                                                                                             |
| files                           | per-kind bytes (terms/postings/positions/store/sparse/vectors/fast)                                | `--residency`: page-cache residency per file                                                                                                                                  |

Reading the generic stats:

- **Inline-term ratio** — terms with ≤3 postings stored inside the dictionary
  entry. A high ratio on a large dictionary means many unique terms (IDs,
  numbers), which is normal; a _low_ ratio with huge top-1% postings share
  means stopword-shaped postings dominate and WAND/MaxScore upper bounds are
  loose.
- **doc_freq p50/p99/max** — the Zipf shape dynamic pruning depends on. A max
  near the corpus size identifies de-facto stopwords.
- **Sparse top-1% share and hottest dims** — SPLADE-style vocabularies are
  Zipfian; a few hot dimensions holding most postings make BMP block upper
  bounds loose. Candidates for `weight_threshold` or vocabulary pruning.
- **Impact saturation** — postings clipped at the u8 quantization ceiling.
  More than a fraction of a percent means `max_weight_scale` is compressing
  the model's weight range and ranking is losing resolution.
- **Store docs/block and bytes/doc** — retrieval cost per document; underfull
  blocks mean the configured block budget is not being reached.
- **BMP padding ratio** — virtual-doc grid slots that hold no document but
  are scanned by every query.

## Metric definitions (dense ANN)

For one segment's IVF payload (binary or IVF-TQ), computed entirely from the
in-memory run directory — never from payload bytes:

| metric                                 | definition                                           | reading it                                                                                                                                          |
| -------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vectors`                              | Σ run counts                                         | —                                                                                                                                                   |
| `clusters_nonempty` / `clusters_total` | distinct cluster IDs with ≥1 posting / codebook size | low occupancy ⇒ dead centroids (the zero-vector incident left ~30% of the codebook unassignable)                                                    |
| `runs`                                 | run-directory entries                                | —                                                                                                                                                   |
| `fragmentation`                        | `runs / clusters_nonempty`                           | extents a probe of one leaf must touch; 1.0 after a rebuild, grows with every byte-copy merge; each extent is a potential disk seek on a cold index |
| `imbalance`                            | `K_nonempty · Σ nᵢ² / N²` (Faiss)                    | expected distance-computation multiplier vs a balanced codebook                                                                                     |
| `largest_leaf_share`                   | max nᵢ / N                                           | a scan cliff regardless of cause; the incident value was 0.31                                                                                       |
| `payload_bytes`                        | codes-column extent                                  | bytes a full probe of every leaf would read                                                                                                         |

Thresholds used for load-time warnings (chosen from the incident values with
headroom; both would have fired months early):

- `largest_leaf_share ≥ 0.05` **and** vectors ≥ 100k → warn (incident: 0.31)
- `fragmentation ≥ 8` → warn (a rebuilt segment is 1.0; 32-way merges can
  reach 32 in one step)

## Run compaction

Fragmentation now heals itself instead of only being reported:

- **Merges** compact binary ANN runs when the byte-copy output would reach
  fragmentation ≥ 4 (`ANN_COMPACTION_FRAGMENTATION_THRESHOLD`): the payload is
  rewritten cluster-major — one extent per cluster, document IDs made absolute
  — restoring the freshly-built layout. Below the threshold, merges keep the
  cheap byte-copy, so shallow merge trees pay nothing.
- **The external `reorder` API** (the optimize pass) compacts at _any_
  fragmentation above 1.0, alongside its BP reorder of sparse fields. It
  requires the trained artifacts to be loaded; without them the vectors file
  is cloned unchanged and a warning names why.

Compaction streams the same payload bytes the byte-copy merge already
writes; the only extra work is one `u32` add per posting for the doc-ID
rewrite. Peak extra memory is a 256 KiB scratch buffer plus one 48-byte
directory record per non-empty cluster — nothing scales with vectors.
Measured on 0.32 GiB across 4 sources (aarch64, warm cache): byte-copy
9.8 GiB/s, compaction 20.1 GiB/s — the cluster-major writer is _faster_
because it streams larger sequential extents. TQ payloads are exempt: their
codes are block-packed and cannot be concatenated without re-packing.

`hermes_ann_fragmentation` and `hermes-tool diagnose` observe the outcome.

## Tiers

### Passive: segment open (always on)

`AnnDiskIndex::health()` is O(runs) over data already resident — the run
directory is parsed during `open()` regardless. For the production shape
(163k clusters, ≤ a few hundred k runs) this is tens of microseconds per
segment, paid once at open, zero allocations beyond the report struct.

Emitted at every segment open:

```
[ann_health] index=documents_20260724 field=7 segment=019f…: vectors=65434704 \
  clusters=114716/163342 runs=114716 fragmentation=1.00 imbalance=41.68 \
  largest_leaf=30.7% payload=19.5 GiB
```

plus gauges (`hermes_ann_imbalance`, `hermes_ann_fragmentation`,
`hermes_ann_largest_leaf_share`, labelled `index`/`field`) so dashboards see
drift without log scraping. Threshold breaches log at `warn`.

After all segments load, the searcher logs one per-field aggregate line, so
an operator reads index health from N_fields lines instead of
N_segments × N_fields.

### Active: `hermes-tool diagnose` (on demand)

```
hermes-tool diagnose -i ./my_index [--json] \
    [--sample N] [--probe-cost NPROBE] [--residency]
```

With no flags: the passive report for every segment and field (per-field
disk usage from the TOCs, doc counts, ANN health), no payload reads. This is
safe to run against a live index — everything is read-only.

Expensive opt-ins, one flag each (the `run_expensive_tasks` idea):

- `--sample N` — read N deterministically sampled vectors per field per
  segment from flat storage and report: all-zero count, popcount mean/min/max for binary
  codes (healthy sign-quantized embeddings sit near 0.5), NaN/∞ rows for
  float vectors. This is the check that would have
  caught the zero-embedding regression the week it started. Cost: N reads of
  one vector each, sequential in the sampled order.
- `--probe-cost NPROBE` — simulate the per-query I/O of an average probe:
  expected leaves, bytes and extents touched per segment at the given
  `nprobe`, using leaf sizes from the run directory. Extents ≈ seeks on a
  cold index; this is the number that explains "why is p99 1 s".
- `--residency` — `mincore(2)` over each mmapped index file, reporting the
  resident fraction of every file kind. Answers "is this index actually in
  page cache" without touching prod tooling. Unix only; the flag reports
  unsupported elsewhere.

`--json` emits the same structure machine-readably for trending in CI or
cron.

## What is deliberately not here

- No corruption checking — Hermes validates checksums and structure at open;
  duplicating Lucene's exhaustive `-slow` verify adds cost without new
  information.
- No automatic remediation. The tool reports; retraining or re-embedding are
  operator decisions.
- No background scan thread in the server. Segment-open reporting plus a
  cron'd `diagnose --json` covers the periodic case without a new scheduler
  in hermes-core.

## Usage

```bash
# Cheap report, safe against a live index
hermes-tool diagnose -i ./my_index

# Everything, machine-readable, for cron/CI trending
hermes-tool diagnose -i ./my_index --json \
    --sample 1000 --probe-cost 64 --residency > health.json

# Stopword/tokenization bloat and sparse vocabulary shape
hermes-tool diagnose -i ./my_index --terms 20 --sparse-stats
```

The `--sample` positions come from a deterministic splitmix64 sequence, not an
even stride: flat order is a BP permutation of ingestion order, and a fixed
stride can alias with that structure (an even stride missed 100/401 zero
vectors in testing exactly this way).

## Files

- `hermes-core/src/segment/ann_disk.rs` — `AnnHealth`, `health()`
- `hermes-core/src/index/searcher.rs` — per-field aggregate at load
- `hermes-core/src/observe.rs` — gauges
- `hermes-tool/src/diagnose.rs` — the subcommand