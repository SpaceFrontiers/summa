# Range block scans: follow-up experiments

## Implementation and invariants

The public entry point remains `RangeQuery::as_doc_bitset`. The column reader
owns copied block boundaries, decoding and ordinal remapping. The codec owns
header interpretation. The query owns comparisons and document membership.
Native sync, native async and WASM share these owners. No wire or persisted
format changes, new metadata cache, writer or schema option are introduced.

1. **Reject disjoint blocks using existing headers.** Constant, bitpacked and
   linear headers can imply conservative raw-value intervals without payload
   reads. Unknown, wrapping or unsupported intervals must scan normally.
   Local text ordinals must never be pruned before global remapping. Signed
   ranges initially use only exact constant bounds; other signed blocks scan.
   Missing sentinels never match. Query ranges still inspect the first value
   for multi-value fields, using their existing fallback. This adapts block
   summary pruning without adding a summary format or derived resident index.
2. **Carry a sequential BlockwiseLinear cursor.** Each 256-value read previously
   restarts the variable-length header walk at byte eight. A full scan therefore
   repeats prefixes of its 512-value records. Carry the current record index
   and byte offset across batches, resetting at each copied column block.
   Random reads must keep their existing behavior through the same decoder.
   Preserve residual interpolation, out-of-range zero fill where supported,
   ordinal remapping and first-error callback termination.

The first experiment adds O(copied blocks) header work and can avoid complete
payloads. The second reduces header traversal from quadratic to linear in the
number of codec records. Both keep constant scratch and the existing output
bitset. The cursor adds two `usize` fields (16 bytes of logical state on this arm64 host),
reused across batches and reset for each copied block. It retains no payload
and allocates no heap memory. Decode arithmetic and encoded bytes remain unchanged. Neither changes
an architecture-sensitive codec, planner or configuration default.

## Findings

Header pruning is useful for clustered bitpacked values spanning multiple copied
blocks. At 65,536 documents with sixteen blocks and a selective range, pruning
alone reduces materialization from 63.219 to 4.092 microseconds. The one-block
control selects BlockwiseLinear and cannot use these cheap bounds: 192.61 versus
193.83 microseconds, with no significant change. Whole missing blocks improve
from 103.22 to 91.921 microseconds. Ordered fixtures select BlockwiseLinear too;
header pruning does not explain their timing differences. The ordered sixteen-
block baseline has wide confidence intervals and is unsuitable for a precise
speedup claim. No codec selection or block sizing defaults were changed.

The cursor reduces the million-document piecewise case from 5.236 to 2.863 ms
relative to pruning alone (45% less time). At 64K documents it improves 191.723
to 178.728 microseconds; at 1K, 2.905 to 2.841 microseconds. These results support
eliminating repeated header traversal, with the gain increasing with record
count. The shared decoder still uses the original interpolation arithmetic.

| Fixture                 | Documents / copied blocks | Before (µs) | Pruning only (µs) | Pruning + cursor (µs) |
| ----------------------- | ------------------------- | ----------: | ----------------: | --------------------: |
| Clustered               | 65,536 / 16               |      63.219 |             4.092 |                 4.083 |
| Clustered               | 65,536 / 1                |     192.605 |           193.829 |               183.082 |
| Ordered                 | 65,536 / 1                |     195.826 |           192.950 |               180.102 |
| Ordered, noisy baseline | 65,536 / 16               |     217.174 |           184.040 |               180.560 |
| Half missing blocks     | 65,536 / 16               |     103.217 |            91.921 |                89.755 |
| All match               | 65,536 / 16               |     186.193 |           184.515 |               179.773 |
| Piecewise               | 1,024 / 1                 |       2.929 |             2.905 |                 2.841 |
| Piecewise               | 65,536 / 1                |     194.348 |           191.723 |               178.728 |
| Piecewise               | 1,048,576 / 1             |   5,171.634 |         5,235.662 |             2,862.518 |

The existing shuffled-data controls also pass exact membership checks and do
not regress: the four cases measure 21.35–21.67 microseconds after both changes,
versus the earlier word-materialization measurements of 24.29–27.86 microseconds.
These are longer-separated controls; their improvement is not attributed solely
to the cursor, which does not accelerate their bitpacked codec.

Encoded column sizes are identical in all runs: for example 98,464 bytes for the
clustered sixteen-block case and 452,617 bytes for the million-document case.
Output bitsets remain 8,192 and 131,072 bytes respectively. `ColumnBlock` (208 B)
and `FastFieldReader` (136 B) stay unchanged. Pruning allocates nothing; cursor
state is constant per scan. Three alternating process RSS runs of the complete
fixture/test driver measure 40.1–40.9 MiB with pruning and 39.3–40.0 MiB with the
cursor too. Setup and allocator noise dominate these small differences; this is
not evidence of lower retained heap or mmap residency. The original unpruned
process RSS was not measured in this follow-up.

These fixtures measure warm range-bitset construction, including allocation
and destruction, on Apple M4 / Rust 1.98.1 with ordinary release flags. They do
not establish cold-storage, full-search, concurrent throughput or x86 speedups.
No new persistence format, block summaries, codec default or reorder policy is
justified by these measurements.

## Validation

Focused codec tests pass, including conservative header bounds across all 65
bit widths and wrapping/descending linear headers; sequential versus random
reads for every codec at nine batch sizes; record boundaries, tails, empty and
backward reads, and existing byte-compatible serializer checks. The range
regression compares merged global text ordinals with independent scorer output.
The complete `check` harness passes 2,029 tests (25 normally ignored), strict
Clippy, native-without-sync and standalone broker compilation. All three
async-only range regressions pass. The WASM release build and all 38 JavaScript tests pass. The
existing fallible/cancellable scan tests pass in the full suite.
Validation records and logs are alongside the measurement evidence.