# Seismic versus BMP: memory, merge and memory pressure (2026-10-06)

Decision: BMP stays the default sparse backend. Seismic is 2-5x faster on warm
fresh segments, but after copy merge it is 1.8x BMP+BP's size and latency
advantage mostly disappears until 16 maintenance passes (12x BMP's BP CPU)
restore it; its default Float32 forward values also fall off a latency cliff
once they no longer fit in page cache. Two measured query optimizations below
were retained. See [Seismic sparse indexing](../../seismic-sparse-index.md).

Fixture: big-ann SPLADE sparse-1M (1,000,000 vectors, 126,320,094 postings,
30,109-dim vocab, 27,646 active terms), first 1,000 dev queries with <=64 dims,
recall vs exact Float32 brute force. Source: origin/main 1f53719, Rust 1.98.1,
`-C target-cpu=native`, thin LTO. Timings on a dedicated GCE c3-standard-8
(Xeon Platinum 8481C, 32 GB, pd-ssd), 4 search threads, warm, 1 warm-up + 2
timed passes, medians of 3 alternating rounds. Byte sizes are deterministic
(identical on the shared sandbox and the VM). Defaults: Seismic
postings=4096, cluster=64, energy=0.4, cut=10, factor=0.85, forward
compression on; BMP block 32, grid 4 bits, max_weight 5, forward index on.

## On-disk size and warm resident set

| Index                               |         Bytes | vs BMP+BP | Warm RSS (file pages) | top-10 ms / recall | top-100 ms / recall |
| ----------------------------------- | ------------: | --------: | --------------------: | -----------------: | ------------------: |
| BMP + BP reorder (production shape) | 1,053,761,902 |     1.00x |               616 MiB |      12.9 / 98.78% |       25.1 / 99.25% |
| BMP, no BP                          | 1,231,931,494 |     1.17x |               735 MiB |      25.4 / 98.77% |       42.0 / 99.18% |
| Seismic, forward UInt8              | 1,144,941,988 |     1.09x |               703 MiB |       5.0 / 98.98% |        9.0 / 98.70% |
| Seismic, forward Float16            | 1,263,218,679 |     1.20x |               815 MiB |       5.3 / 99.31% |        9.9 / 98.93% |
| Seismic, forward Float32 (default)  | 1,515,883,146 |     1.44x |             1,057 MiB |       5.6 / 99.39% |        9.6 / 98.95% |

Seismic breakdown: root `.sparse` (forward values + 24 B/row directory) is
336.4 MB u8 / 454.7 MB f16 / 707.3 MB f32; the 16 nomination partitions are
808.6 MB in every variant (42,937,566 nominations -> 18.8 B per nomination:
4 B row id + ~14.8 B cropped u8 summaries/directories). BMP forward section is
322.9 MB (2.56 B/posting); inverted blocks + grids + maps 909.0 MB (7.2 B/posting).

Build (VM): BMP 53 s (+32 s BP), Seismic 263-272 s (serial clustering).
Peak build RSS minus the 0.98 GB input: BMP ~2.6 GB, Seismic ~5.0 GB.
A 1M-vector Seismic flush exceeds the 300 s `prepare_commit` flush timeout on
the shared sandbox (306 s); the RPC retries forever, the shutdown path retries once.

## Production-scale model (per segment)

Seismic bytes ~= F*P + 18.8*M, BMP+BP ~= 8.34*P (1M fixture; production BMPB
measured ~8.4 B/posting), where P = postings, M = sum_t min(df_t, lambda) nominations
summed over every copied source run, F = 2.66 (u8) / 3.60 (f16) / 5.60 (f32) B/posting.
Seismic is smaller than BMP+BP when M/P < 0.30 (u8), 0.25 (f16), 0.15 (f32). At 1M,
M/P = 0.34. Larger consolidated segments with fixed lambda lower M/P; copy-merged
fragments raise it (each source keeps its own top-lambda lists until maintenance).

## Measured implementation gaps (dedicated VM, identical results)

| Change                                                                                                                  |                 top-10 |                 top-100 |
| ----------------------------------------------------------------------------------------------------------------------- | ---------------------: | ----------------------: |
| CPU prefetch of row entry + first 256 B of each nominated vector per surviving cluster (upstream two-pass prefetch), u8 | 4.95 -> 3.71 ms (-25%) |  8.87 -> 6.55 ms (-26%) |
| same, f32                                                                                                               | 5.52 -> 3.87 ms (-30%) |  9.52 -> 6.77 ms (-29%) |
| Dense query lookup up to 2^17 dims (now 2^16), dims shifted past 65,536                                                 | 7.53 -> 5.36 ms (-29%) | 13.73 -> 9.04 ms (-34%) |

Both are retained in `query/seismic.rs`, `query/seismic/scoring.rs` and
`segment/seismic/mod.rs`; prefetch is a no-op on targets other than x86-64 and
AArch64. Only x86-64 (Sapphire Rapids) was measured; ARM latency is unmeasured.

## Merge lifecycle (1M as 4 x 250K commits, current code, dedicated VM)

| Step                         |                                                 BMP + BP |                                                                  Seismic u8 |
| ---------------------------- | -------------------------------------------------------: | --------------------------------------------------------------------------: |
| 4-segment build              |                                 56 s + 27 s BP; 1.109 GB |                                           453 s (vs 266 s single); 2.030 GB |
| 4 segments: top-10 / top-100 |                                           10.7 / 19.4 ms |                                                5.6 / 8.7 ms (recall 99.50%) |
| Copy merge                   |                      8.0 s, 0.26 GB peak RSS -> 1.097 GB |                            11.6 s, 0.04 GB peak RSS -> 2.030 GB (no shrink) |
| Merged: top-10 / top-100     |                                           19.7 / 31.8 ms |                                                              13.7 / 25.4 ms |
| Merged warm RSS              |                                                  666 MiB |                                                                   1,497 MiB |
| Restore fresh layout         | 1 BP pass, 31 s, 1.36 GB RSS -> 1.054 GB, 12.8 / 24.8 ms | 16 passes (one per partition), 381 s, 0.46 GB RSS -> 1.146 GB, 5.3 / 9.2 ms |

## Memory-capped cold queries (cgroup v2, 200 queries, k=10, 64 MiB metadata pin)

| Cap   |   BMP | BMP+BP | Seismic u8 | u8 + fwd mlock (321 MiB) | Seismic f32 | f32 + fwd mlock (675 MiB) |
| ----- | ----: | -----: | ---------: | -----------------------: | ----------: | ------------------------: |
| 1024M |  25.2 |   12.8 |        4.7 |                      4.8 |         5.1 |                       4.9 |
| 768M  |  25.0 |   12.6 |        4.6 |                      4.6 |        33.1 |                      10.1 |
| 512M  |  86.3 |   37.7 |        7.6 |                      9.1 |         480 |          n/a (lock > cap) |
| 384M  | 102.5 |   47.8 |       25.9 |                     11.3 |         970 |                       n/a |

Seismic Float32 falls off a cliff once its 675 MiB of randomly read forward
values exceed the cap; u8 forward values (321 MiB) stay ahead of BMP+BP at every
cap, and locking only the forward values (simulated with an external `mlock`
inside the same cgroup) holds 11.3 ms at 384M. Locking was measured on the fresh
layout only; a merged segment has roughly twice the working set.

## Literature and upstream comparison

Matches upstream Seismic (TusKANNy/seismic @3c26713): transposed u8 summaries
with compressed directories, top-15 centroid assignment, first-list-only block
ordering, full-query summary scoring, `dot < factor * threshold` skip rule,
hash-set dedup, DotVByte forward dimensions. Differences: upstream prefetches
each surviving block's vectors (now adopted here), defaults to f16 forward
values (Summa defaults to Float32), uses about 10-document blocks
(`centroid_fraction` 0.1) versus 64 here, and selects nominations with a global
weight threshold capped per list instead of a fixed per-term top-lambda. The
kappa-NN graph (SeismicWave, CIKM'24) is disabled in every upstream best
configuration; per-segment graphs would cost tens of GB at production scale
and only cover their source run after copy merge. No newer SeismicWave paper
exists as of this date.

## Reproduction

The evaluator was an out-of-tree Rust binary linking `summa-core`: one
`IndexWriter` builder, two compression threads, `NoMergePolicy`, MmapDirectory,
query latency excludes document hydration, recall uses a stored ID for
multi-segment indexes. Memory caps used `systemd-run --scope -p MemoryMax=...
-p MemorySwapMax=0` with caches dropped before each process. Raw logs were kept
outside the repository.
