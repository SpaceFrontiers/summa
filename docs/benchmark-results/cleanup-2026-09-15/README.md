# Cleanup validation — September 15, 2026

The selected cleanup fixes corruption propagation and compaction bounds, with
regressions reproduced before each fix. Byte-gap validation is retained for a
1.3% official COUNT improvement against the corrected control (six of seven
x86 passes). Ranked differences are inconclusive; negated top-10 is 2.5% slower.
Current official top-10 remains 2.7% slower than pre-cleanup and 1.25× slower
than Tantivy. No cache, codec or approximation default changes.

See the [current comparison](../../search-benchmark-current.md) and
[review](../../search-performance-review.md#cleanup-full-corpus-result-and-selection)
for results and remaining work.

## Evidence

- results.zip (local archive `results.zip`): frozen source overlays, runner scripts, raw samples,
  score/count oracles, index manifests, disassembly and process RSS measurements.
- [manifest.json](manifest.json): every logical file's length, SHA-256 and stored
  ZIP path. Identical files share an `archive_path`; extracting the ZIP alone
  does not reconstruct all logical aliases. Read each logical file from its
  `archive_path` and verify its length and SHA-256 before restoring it.
- native-check.zip (local archive `native-check.zip`) and
  [native-check-manifest.json](native-check-manifest.json): final full native
  harness logs and source/binary provenance. All eight stages passed: 1,786
  standard tests plus four real-server broker tests; 21 tests remain ignored.

The `arm/` prefix contains local continuation evidence, including summaries of
cloud results. The `cloud/` prefix preserves downloaded remote evidence.
Executable binaries and corpus/index payloads are omitted from this compact
archive; their hashes are retained. The full downloaded remote archive was
verified against its external SHA-256 and all 62 manifest entries before the
owned machine was stopped. Its disks and fixtures remain preserved.

## Reproducing the comparisons

Use the recorded source overlays on base
`ce2c96b945fccc4bac58ccc45bb4bc23b809773e`, verifying every source-manifest hash.
The archives carry exact runner commands, build flags, environment and index
manifests; the full corpus and previously frozen benchmark engines must be
restored separately from the preserved benchmark workspace.

- `cloud/inputs/cloud-run.py` builds the corrected source, then overlays
  `posting-byte-gap.rs` for the candidate. It verifies the existing control and
  Tantivy binary hashes, tests the candidate, checks both variants' raw oracles,
  and launches `cloud/inputs/cloud-paired.py`. Its workspace paths and create-only
  output directories require adapting for a fresh rerun.
- Cloud latency uses the same Cascade Lake machine, Rust 1.98.1, native CPU flags,
  release LTO, CPU 2, unchanged 5,032,104-document indexes, 962 official queries
  and 714 supplemental terms. Seven complete passes rotate engine order after
  at least ten seconds of command warmup. Reports use geometric means of
  per-query medians. All four commands are measured.
- `arm/paired-arm-run.py` records its CLI in its opening docstring. It compares
  frozen binaries with 11 rotated samples per query and 15-second warmup on
  canonical and merged Apple M4 fixtures. Restore its referenced `.context`
  query and index fixtures and verify the captured manifests before rerunning.
- Exact oracles cover all 1,676 queries on both full-corpus layouts and four ARM
  layouts. Ordered top-1000 document IDs, raw score bits and counts agree with
  the saved traversal baseline; pruned top-k results agree with exhaustive
  collection. Index manifests are unchanged.
- `arm/final-source.tar.gz` and its manifest pin the final source.
  `arm/final-binary-proof.json` proves its rebuilt ARM release is byte-identical
  to the timed candidate after the final private helper rename and test-only
  strengthening. Build logs record compiler and flags.

RSS measurements are separate process high-water marks over all four commands,
including mapped pages and heap; they do not isolate heap residency. Summa RSS
is essentially flat across controls and candidate. Checked decoder code grows
from 2,996 to 3,752 bytes on ARM and 3,304 to 4,006 bytes on x86.

## Limits

These are warm, single-query-CPU measurements. Cold-cache, concurrent-ingest,
tail latency and original upstream-runner confirmation of changes after the
OR optimization remain unrun. WASM was not rebuilt under the standing user
instruction; portable-core and native-without-sync checks passed. Corruption
checks reject observed invalid posting order and propagate that failure; they
are not cryptographic payload authentication.
