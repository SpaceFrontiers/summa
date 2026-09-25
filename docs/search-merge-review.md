# Search merge review — September 17, 2026

## Scope and base

Reviewed the pending search changes against `origin/main`, including untracked
implementation modules and tests. The workspace originally lagged main by 11
commits. It was fast-forwarded to `b6212939`, then the working changes were
reapplied and the four conflicting files resolved. Main's content-hash dedup,
staged admission, resource logging and lifecycle changes are retained.

The format stamp is **9**: it covers SIMD blocks, compact posting/position
directories and byte norms. Format 8 also carries main's content-hash schema
marker. Metadata migration preserves both capabilities and does not rewrite
segment payloads. Historical benchmark archives describe their measured source;
they have not been relabeled as measurements of this review revision.

## Corrections and reuse

- Row compaction now preserves each field's norm encoding independently when
  exact and byte columns coexist; it no longer expands all columns because one
  field is exact. The mixed-column regression reproduced this before the fix.
- Standalone RGB now copies untouched byte/exact norm payloads through the CHNK
  owner. Previously it expanded unrelated byte norms into u16 columns. The new
  and copied column paths share one header/table/payload writer.
- Text BP uses BMP's existing frequency selection and budget-fitting helpers.
  The planner charges vocabulary, encoded input, retained field permutations,
  pair/CSR construction and graph scratch. It releases construction buffers
  before BP and releases text plans before starting BMP reordering.
- Explicit text rewriting reuses `PostingBlockSource`, `PostingStreamWriter`
  and `PositionRangeSource`, already used for row compaction. It retains one
  bounded permutation record per posting, rather than positions in a separate
  vector for every document, another decoded posting list and a full serialized
  output buffer. Posting/position directories have individual limits within
  the admitted term budget. Corrupt counts/addresses and cancellation fail the
  rewrite through the existing publication owner. A before/after measurement
  caught source-block copying fragmenting the reordered position stream. Explicit
  reorder now requests repacking through the same encoder; ordinary compaction
  still copies intact encoded blocks. A byte-for-byte regression covers all four
  codecs and both directory layouts.
- Posting builder configuration is passed as the existing typed config rather
  than a six-element tuple. Impact-implies-ratio policy has one implementation.
- Removed the redundant term-frequency accessor and the duplicate decompression
  capacity helper exposed by integrating main. Retained upstream and branch
  decompression regressions. Compact-layout inspection helpers are test-only.
- Merge publication distinguishes BMP-only BP from completion of all requested
  fields. Copied text maps remain pending and retain BP debt, so later standalone
  work can finish them. Legacy plain text without a document map now fails with
  migration guidance instead of silently skipping a requested reorder.
- WASM's old experimental SIMD fixture contained short blocks with a full-block
  SIMD tag. It was regenerated through the current native writer; validation
  remains strict. Added compact-directory/byte-norm native-to-WASM coverage.

The initial failing regressions and commands are retained under
`.context/merge-review/`. The existing RGB regression compares raw score bits,
counts, positions, untouched posting/position bytes, and now norm encoding/bytes.

## Ownership and correctness audit

| Area                 | Decision / invariant                                                                                                                                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scoring and batching | Keep one `Bm25Params` owner; byte tables and batches retain its f32 arithmetic. Two-term contribution elision stays restricted to admitted finite nonnegative mapped BM25 cursors. Longer queries retain input-order reduction. |
| Physical IDs         | Same-field physical traversal is opt-in. Stable IDs are resolved before heap ties, deletions and external results. Independently permuted fields and chunked ordinals retain their existing fallbacks.                          |
| Ranked composition   | Complete child membership, query-owned statistics, negative/zero boosts, exact counts and position requirements survive native and async planning. Child top-k handoff is not used where it could hide matches.                 |
| Encodings            | Existing legacy/compact/SIMD format owners remain authoritative. Ordinary merges copy compatible payloads and remap directories; reorder and partial row compaction are explicit rebuild cases.                                 |
| Validation           | Structural checks precede byte-view admission; decoded content failures propagate through the reader integrity record. Validation proofs retain no decoded corpus payload and are disabled for mutable/lazy callbacks.          |
| Byte views           | `OwnedBytes` retains immutable Arc storage for the entire pointer-view lifetime; subviews are checked against their parent view. Existing move, clone, mmap, thread and empty-slice tests remain relevant.                      |
| Residency            | Dictionary retention and validation tables are bounded per segment. Decoded scoring scratch is bounded and reused. No new query cache or corpus-sized decoded norm table is introduced.                                         |
| Lifecycle            | Reuse existing output claims, cold writers, cancellation and replacement publication. No second full segment generation is written to fake integrated merge-time RGB.                                                           |

## Applying the optimizations to BMP

| Text change                                     | BMP conclusion                                                                                                                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Resolve logical IDs only for competitive hits   | Already implemented in `query/bmp.rs::score_superblock_blocks`, including stable ties and ordinal resolution.                                                                                                             |
| Shared heap admission                           | BMP already uses `ScoreCollector`. Its integer `ThresholdUnits` filter avoids conversion and map access for losing candidates. Adding the text f32 screening pass would duplicate that work.                              |
| Query-constant preparation and reusable scratch | BMP already has `PreparedBmpQuery`, integer query weights, prepared grid locations and `BmpScratch`.                                                                                                                      |
| Quantized BM25 norms / lookup tables            | Not applicable to BMP's integer sparse-weight accumulation. Moving normalization there would change scoring semantics.                                                                                                    |
| Document-gap/position codecs                    | BMP D/E payloads and grids have different addressing and pruning semantics. Copying text codecs is not a compatible format optimization.                                                                                  |
| BP selection and budgets                        | Shared in this review. BMP's existing 4-byte CSR posting cost and selection policy are preserved; text supplies its 12-byte construction cost.                                                                            |
| OR-to-conjunction tail                          | Not transplanted: BMP has block-local term masks and integer partial-score bounds, not the text executor's required-term membership contract. A new BMP experiment requires fixed-recall and concurrent I/O measurements. |

No BMP speedup is claimed from this review. The existing bounded CPU pool and
BMP I/O gate remain in place.

## Remaining scope and evidence

Integrated merge-time text RGB is implemented in the September 17 follow-up
below. The Lucene RGB top-10 target remains open: a fresh matched x86 run puts
Summa **6.5%** behind Lucene (426.918 versus 400.954 µs), consistent with the
earlier 6.7% gap. Scoring and codec defaults are unchanged.

There are approximately 733 MiB of historical raw benchmark artifacts under
`docs/benchmark-results/`. Per the user's instruction, benchmark artifacts must stay out of commits.
They remain intact as local, untracked evidence; no commit was made. Review evidence and backup patches are gitignored in
`.context/merge-review/`.

## Validation

The final `full` harness is recorded at
`.context/search-harness/20260917T040610.439686Z-full/`: **all eight stages passed**.
Its workspace test stage passes **1,856 tests**, with **26 ignored**; formatting,
Clippy with warnings denied, native without sync, and portable compilation pass.
The harness includes the standard `check` stages and adds docs, server build,
and real server/broker integration coverage (**4 passed**).

Additional checks:

- `cargo test --locked -p summa-core --features query-diagnostics --test
query_work_diagnostics --test text_reordering`: **6 passed**.
- `cd summa-wasm && bash build.sh`, dependency installation with `npm ci`,
  and `CI=1 npm test`: **32 passed across 7 files**. Native-generated SIMD,
  compact directories, byte norms, impact layouts and phrases decode correctly
  through WASM. wasm-pack reports its existing missing-license-file notice.
- The RGB preservation regression covers all four posting codecs with both
  legacy and compact requests, including impact and ratio bounds. Pfor retains
  its supported legacy posting headers. It compares raw score bits and
  unchanged field bytes, not only result counts.
- New regressions cover mixed norm compaction, tiny-budget admission,
  cancellation before planning, missing legacy maps, and merge completion
  metadata. Existing lifecycle tests cover failure/cancellation publication.

The 116 changed or new source/test/fixture files are fingerprinted in
`.context/merge-review/validated-source-manifest.json`; this includes untracked
implementation files, unlike `git diff` alone. Documentation completion follows
check results. Before-fix failures remain separate from the final passing runs.

## Reorder measurement

A local ARM comparison used the previously delivered immutable binary and the
review build, Rust 1.98.1, native CPU flags, release LTO, and the same frozen
131,072-document positions-heavy fixture. Runs alternated before/after/after/before;
compilation and tests had finished before timing.

| Metric                      | Before          | Review build  |
| --------------------------- | --------------- | ------------- |
| Peak process RSS            | 119.2–122.0 MiB | 35.5–36.3 MiB |
| Reorder wall time, two runs | 0.24 / 0.24 s   | 0.43 / 0.22 s |
| Positions bytes             | 3,484,756       | 3,484,756     |
| Posting bytes               | 263,716         | 263,716       |

Peak RSS falls about **70%** on this fixture. This is process residency, not an
isolated allocator measurement. The short, variable timings do **not** establish
a speedup. This compares the whole review plus the main update; it does not
isolate each fix or measure query latency, BMP, x86, or Lucene parity.

All seven payload components (`.post`, `.pos`, `.terms`, `.chunks`, `.store`,
`.fast`, `.rowstats`) are byte-identical across all four outputs, including the
same physical permutation. Generation metadata is excluded. Each output passes
11 query checks comparing pruned top-10/100/1000 scores and IDs against its
exhaustive oracle, with matching exact counts. Query counts also agree with the
frozen source. The initial 32,768-document measurement exposed position-file
fragmentation; that failure and its fix are retained in local evidence.

Raw results, corpus and binary hashes, and commands are under
`.context/merge-review/repacked-measurement/` and
`.context/merge-review/measure-reorder-packed.py`. They stay out of commits.

## Follow-up: integrated merge-time text RGB

An explicit merge reorder now plans each opted-in text field over its source
segments and writes the final physical order directly. It shares k-way term
traversal with copy merge, BP selection and scheduling with BMP, term rewriting
with standalone RGB, and the existing posting, position and CHNK writers.
Unselected fields retain ordinary merge behavior. Text planning, term scratch
and map migration share the configured budget; text releases its plan and CPU
permit before BMP work. Publication still belongs to the segment manager.

The public regression first reproduced a merge that only reordered BMP and left
text pending. It now checks physical text order and completed metadata. A second
public regression forces budget failure and checks that source metadata/files
remain intact, queries still work, and unpublished output is removed. Map tests
reject invalid permutations before writing and interrupt during column output.

Across all four posting codecs, fused merge matches copy-merge followed by
standalone RGB byte-for-byte for postings, positions, dictionary, chunk maps,
stored fields and fast fields. The fixture mixes legacy unmapped and mapped
sources, plain/chunked text, missing/multiple values, compact layouts and impact
bounds. Scores and IDs match exactly; position/ordinal lists are sorted before
comparison because chunk ordinals are emitted in physical encounter
order by the existing collector. Ranked top-10 agrees with exhaustive scoring.

The follow-up `full` harness at
`.context/search-harness/20260917T043453.487192Z-full/` passes all eight stages:
**1,859 workspace tests**, 26 ignored, and **4 real server/broker tests**.
The fresh WASM build passes **32 tests in 7 files**. Targeted public merge/RGB
integration passes five tests. Logs are under `.context/merge-text-rgb/`.

A same-binary ARM comparison uses four source segments totaling 131,072
documents, native CPU flags, release LTO and alternating separate/fused runs:

| Metric              | Copy merge + standalone RGB | Integrated merge-time RGB |
| ------------------- | --------------------------- | ------------------------- |
| Wall time, two runs | 0.30 / 0.31 s               | 0.22 / 0.22 s             |
| Peak process RSS    | 43.31 / 44.80 MiB           | 38.19 / 31.56 MiB         |

The fused lifecycle is about 28% shorter on this small fixture. Every emitted
payload component is byte-identical, excluding generation metadata; all 11
query checks retain exact top-10/100/1000 and counts. This measures maintenance,
not query latency or large-corpus scaling. The executable, corpus/output hashes,
commands and raw results stay local in
`.context/merge-text-rgb/measurement-final/` and its parent directory.
