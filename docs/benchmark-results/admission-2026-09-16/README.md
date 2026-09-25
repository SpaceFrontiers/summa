# Validation reuse and position lookup — September 16, 2026

## Outcome

RGB off. The final cache-plus-position build is compared directly with the
preserved September 15 traversal build, a cache-only parent and Tantivy in one
run. Official top-1000 improves 4.0%, top-100 plus exact count 4.5%, and exact
count 3.6%, all seven passes faster. Top-10 is nearly flat (-0.5%, five passes).
Final Summa/Tantivy ratios are **1.207/1.153/1.305/1.102** respectively for
top-10/top-1000/top-100 plus count/count. Parity remains unmet.

Tradeoffs: union top-10 +1.2% (five slower passes), standalone metadata COUNT
+1.7% (all seven slower). Standalone top-10 remains 2.725× Tantivy; standalone
top-100 plus count is 0.979×. The length-gather experiment is rejected: x86
ranked workloads regress despite correct results and small ARM improvements.

Official RSS after three complete passes per command is
1008.24/1007.56/574.09 MiB for original/selected/Tantivy. No meaningful memory
reduction is claimed. Postings and positions still dominate mapped residency;
quantized norms and compact formats are not implemented by this change.

[Current tables and limitations](../../search-benchmark-current.md) ·
[Engineering review](../../search-performance-review.md) ·
[Existing format accounting](../../text-format-comparison.md).

## Archive and reproduction

results.zip (local archive `results.zip`) is content-deduplicated. [manifest.json](manifest.json)
maps every logical file to its stored `archive_path`, byte length and SHA-256;
it also records the completed ZIP's size and hash. Resolve this mapping when
extracting: identical logical files may share one physical archive member.
Every member and the final archive are verified.

- `admission-ways/`: retained four-way proof cache, failing-before/passing-after
  regression, source overlay, paired timings, exact oracles and native checks.
  Its separate memory audit contains 401 verified members; CPU profiles contain
  45, with unsupported hardware counters explicitly recorded.
- `length-admission/`: rejected maximum-ID precheck for length gathers, complete
  correctness and performance results, source and selection rationale.
- `position-directory/`: initial ARM prototype based on that rejected gather.
  Its queued cloud run was cancelled before creating any output folder. The
  prototype's results are not a final performance claim. A test-call signature
  compile error and its correction are retained.
- `position-directory-clean/`: final cache-plus-position source, rebased without
  the rejected gather; full final comparisons, native/portable checks, four ARM
  and two x86 raw-score oracles, byte manifests, final main/cloud source and
  baseline-binary verification. Its cloud benchmark archive has 42 verified
  members and its separate memory audit 401.
- `inputs/`: exact workloads, independent exhaustive oracle helper, original ARM
  byte manifest and starting source manifest.

The final overlay has 332 files and changes only
`summa-core/src/structures/postings/posting/reader.rs` and
`summa-core/src/structures/postings/positions_v2.rs` relative to the starting
build. Source archive SHA-256:
`722875a6e1a771a600e0edd61187c71cf6a1d0fac4bd3862366c84997feed921`.
The frozen source matches the main workspace and cloud build exactly.
Binaries and large raw perf.data are excluded; hashes, complete source overlays,
build flags, scripts, raw samples, logs and text profiles are retained.

Linux: Cascade Lake, Rust 1.98.1 / LLVM 22.1.8, native CPU flags and release LTO,
four build jobs, one query CPU (CPU 2). Same 5,032,104 documents and document
order, 962 official queries and 714 supplemental terms. Seven rotated complete
passes after at least ten seconds of warmup per engine/command. Proof-cache
budget 262,144 bytes; term-cache limits 8,192 blocks / 4,194,304 bytes. These
runtime budgets and every index byte match the controls; defaults are unchanged.
No build, full byte scan or memory audit overlaps latency on the same host.

ARM: 262,144 documents, canonical/copy-merged Rounded timing and corresponding
grouped-impact correctness fixtures. Same compiler/flags, fresh builds and
preserved controls. Shared Mac load limits interpretation of small movements.

## Validation

Final native harness: **1,806 passed, 25 ignored**, formatting/Clippy/native
no-sync checks pass. Portable core check passes with the existing
`set_document_units` dead-code warning. Final core tests: ARM 1,589 and x86
1,590 passed, 16 ignored on each. All 1,676 queries match independent exhaustive
ordered-ID/raw-score/count oracles across four ARM and two x86 layouts, including
optimized exact-count top-k at k=10/100/1000. All 40 ARM files and all full-corpus
index hashes are unchanged.

Documentation checks pass with the declared `uv run scripts/check_docs.py`
runner. The initial plain-Python attempt lacked `markdown_it`; its failure and
the corrected passing run are retained.

WASM was not rebuilt, following the user's instruction. Cold/concurrent workload
and full lifecycle/RPC checks were not rerun. Hardware cycle/instruction/branch/
cache counters are unsupported on this machine; CPU-clock profiling works. Quantiles
in the archive summarize per-query medians, not concurrent-service tail latency.
The owned machine is stopped after verified evidence download. Changes are uncommitted.
