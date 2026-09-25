# RGB execution repair — September 16, 2026

This repairs the query-execution regression diagnosed in the
[standalone RGB benchmark](search-rgb-benchmark.md). Reader-only measurements use
unchanged index files. A later writer repair preserves compact layouts when
rebuilding RGB indexes. Stable logical IDs, raw scores, positions and exact
counts are preserved in both cases.

## Implementation

- Query implementations explicitly admit compatible same-field plain-text trees
  into one physical address space. Existing term, Boolean and phrase scorers
  compose there; collection maps candidates to stable IDs before heap comparisons
  and position callbacks. No corpus-sized per-query permutation or second Boolean
  executor is added.
- Ranked conjunctions use the existing typed block executor, with physical lengths
  and stable-ID heap ties. Only retained hits use the validated inverse slot column to re-enter the
  physical stream in O(k). Standalone collection takes the bounded precomputed
  list, restores stable tie order before truncating, and avoids a second heap.
  Complete children never use a ranked cutoff.
- One-to-one document maps retain dictionary term cardinalities and same-field
  two-term union counts. Membership-only traversal does not probe frequencies just
  to reconstruct logical order or compute unused scores.
- Deletion predicates resolve physical candidates to logical IDs. Deleted readers
  retain complete conjunction traversal before filtering. Cross-field, chunked,
  opaque and fast-column query trees retain the existing logical fallback.
- Existing ranked term/OR pruning remains active. A guarded query-wide proof
  switches an OR tail to the existing conjunction executor once no document
  missing a term can enter the heap. The heap and canonical score order survive
  the transition, including exhaustion and cancellation.
- Mapped term/conjunction batches resolve stable IDs only after the existing
  eight-score admission screen. Equal scores still use stable-ID tie breaking.
- Mapped two-term windows reuse their accumulated scores when validated finite,
  nonnegative scoring makes that sum bit-identical to canonical reduction.
  They omit duplicate contribution writes and the second reduction pass.
  Longer queries and unsupported scoring retain the existing ordered reduction.
- Explicit reordering preserves compact posting headers and position directories
  using the existing serializers. Legacy layouts remain legacy; untouched fields
  retain byte-identical payloads. There is no new format version or norm default.

The [design](maxscore-text-reordering.md#collection-boundary-implementation)
records address-space, count, cancellation and tie-order constraints. Direct
logical scorer APIs and unsupported compositions still use the logical fallback;
this does not claim to remove every mapping cost from every query shape.

## Performance target

The latest requirement is RGB top-10 faster than Lucene BP/RGB. Tantivy is still
a useful control, but beating it alone does not meet the updated target. A
matched Lucene 10.4.0 BP run uses the pinned upstream adapter and the same
5,032,104-document corpus on the same x86 machine. The upstream Lucene analyzer
and BM25 parameters differ from Summa, as recorded in the
[benchmark contract](search-benchmark-game.md#analyzer-and-scoring-differences).

## Historical physical-traversal repair measurements

Same frozen indexes, Rust 1.98.1 / LLVM 22.1.8, release LTO and
`-C target-cpu=native`. ARM uses the first 100,000 canonical Wikipedia documents;
x86 Cascade Lake uses all 5,032,104. x86 engines are pinned to CPU 2, driver unpinned.
962 official queries have seven rotated passes and at least ten seconds of
whole-workload warmup per engine/command. The 714 supplemental terms have five
passes and at least three seconds of warmup. No builds, diagnostics or memory
sampling overlap latency on the same host.

The initial paired run includes the frozen old RGB reader. Its counted modes run
on ARM only: the prior full-corpus experiment established prohibitively slow
replay, so x86 old-RGB counted latency is unmeasured. The handoff run below measures the physical-traversal reader against fresh
RGB-off controls and Tantivy on all four commands.
Old RGB times below are retained from the initial run on the same host/compiler/
flags/index bytes; the contemporaneous RGB-off controls expose run-to-run drift.
The two intermediate repaired readers and all their results remain identifiable in the
evidence; the historical tables below use the handoff reader for “repaired”, before
the selected OR-tail, admission and compact-layout changes.

Times are geometric means of per-query median microseconds. Distribution summaries
in the evidence are across query medians, not production request-tail estimates.
These are warm serial measurements; cold-cache and concurrent-ingestion behavior
are not measured. No index-format or policy defaults are changed.

## Results

### x86: official

| Operation       |    Old RGB | Repaired RGB | RGB-off control | Tantivy | Repaired / Tantivy |
| --------------- | ---------: | -----------: | --------------: | ------: | -----------------: |
| Top 10          |   2444.065 |      440.615 |         506.238 | 524.345 |             0.840× |
| Top 1000        |   3873.167 |      885.654 |         962.152 | 987.633 |             0.897× |
| Top 100 + count | unmeasured |      725.922 |         797.661 | 869.111 |             0.835× |
| Count           | unmeasured |      342.566 |         403.197 | 424.969 |             0.806× |

### x86: supplemental

| Operation       |    Old RGB | Repaired RGB | RGB-off control | Tantivy | Repaired / Tantivy |
| --------------- | ---------: | -----------: | --------------: | ------: | -----------------: |
| Top 10          |     98.110 |       99.598 |         128.338 |  54.446 |             1.829× |
| Top 1000        |    856.881 |      869.661 |         569.050 | 445.526 |             1.952× |
| Top 100 + count | unmeasured |      310.783 |         245.859 | 266.912 |             1.164× |
| Count           | unmeasured |       19.168 |          19.089 |  15.635 |             1.226× |

### ARM: official

| Operation       | Old RGB | Repaired RGB | RGB-off control | Repaired / old RGB |
| --------------- | ------: | -----------: | --------------: | -----------------: |
| Top 10          |  52.685 |       39.378 |          42.457 |             0.747× |
| Top 1000        |  74.832 |       45.637 |          47.975 |             0.610× |
| Top 100 + count | 158.663 |       47.346 |          49.092 |             0.298× |
| Count           | 145.636 |       37.185 |          38.484 |             0.255× |

### ARM: supplemental

| Operation       | Old RGB | Repaired RGB | RGB-off control | Repaired / old RGB |
| --------------- | ------: | -----------: | --------------: | -----------------: |
| Top 10          |  17.189 |       26.293 |          28.580 |             1.530× |
| Top 1000        |  37.998 |       38.136 |          38.226 |             1.004× |
| Top 100 + count | 123.665 |       27.461 |          31.920 |             0.222× |
| Count           |  10.538 |       19.452 |          18.621 |             1.846× |

### ARM host drift

The unchanged RGB-off baseline moved between runs. Fresh controls, rotated within each pass, are the primary estimate of RGB benefit; old-RGB ratios also include this host drift.

| Command         | Initial unchanged control µs | Final unchanged control µs |
| --------------- | ---------------------------: | -------------------------: |
| Top 10          |                       32.285 |                     41.964 |
| Top 1000        |                       47.369 |                     47.462 |
| Top 100 + count |                       35.663 |                     47.711 |
| Count           |                       25.242 |                     39.064 |

### x86 query families

Repaired RGB / contemporaneous RGB-off control; below one is faster.

| Family             | Top 10 | Top 1000 | Top 100 + count |  Count |
| ------------------ | -----: | -------: | --------------: | -----: |
| intersection       | 0.863× |   0.985× |          0.910× | 0.828× |
| phrase             | 0.981× |   0.821× |          0.807× | 0.808× |
| union              | 0.803× |   0.958× |          1.047× | 0.898× |
| negated            | 0.964× |   1.057× |          1.009× | 0.841× |
| intersection_union | 0.712× |   0.944× |          0.792× | 1.006× |
| term               | 0.059× |   0.295× |          0.110× | 0.846× |

The official `term` family is one stop-word query; the supplemental table covers the 714 standalone terms.

## Work removed

Separate instrumented ARM runs, second pass of all 962 official queries. No latency claims use this build.

| Command         | Old RGB doc blocks | Repaired doc blocks | Old TF blocks | Repaired TF blocks |
| --------------- | -----------------: | ------------------: | ------------: | -----------------: |
| Top 10          |            339,303 |              31,791 |       139,083 |             21,513 |
| Top 1000        |            363,426 |              56,195 |       165,969 |             48,773 |
| Top 100 + count |         15,011,241 |              60,270 |     5,395,031 |             52,756 |
| Count           |         14,851,059 |              60,157 |     5,194,781 |              5,197 |

Decoder bytes include repeated processing of resident bytes; they are not disk I/O. Count phrases still legitimately read frequencies for positional matching.

## Memory and index bytes

Separate fresh processes run one complete pass per command. Ranked RSS is the snapshot after top-1000; all-command RSS is the maximum snapshot after all four commands. These are observed working sets, not long-run plateaus. Old RGB has ranked-only samples.

| Reader / index  | Ranked RSS MiB | All-command RSS MiB | Final anonymous MiB |
| --------------- | -------------: | ------------------: | ------------------: |
| Old RGB         |        1012.89 |          unmeasured |                5.80 |
| Repaired RGB    |        1019.58 |             1020.04 |                6.24 |
| RGB-off control |         760.20 |              760.69 |                6.20 |
| Tantivy         |         574.12 |              574.14 |                0.75 |

All compact, RGB and Tantivy index file hashes match the frozen controls. This repair removes repeated work; it does not shrink the persisted maps or position files. Resident mapped payloads continue to dominate the remaining memory difference.

## Handoff validation and remaining limits

The behavior-named work regression failed before the repair: a two-term count
decoded 278 blocks although the lists contained 16. It now bounds block work and
requires zero TF decoding for term-only counts. Regression coverage includes
stable-ID ties, sync/async agreement, phrase positions, multivalue/missing fields,
independent field permutations, boosts, exclusions, deletion, merge, compaction
and an already-expired mapped request.

The handoff-stage native harness includes formatting, Clippy, 1,830 passing tests and
native-without-sync compilation. Four focused reorder/diagnostic tests pass.
Portable compilation passes with the pre-existing unused `set_document_units`
warning in the no-native build. WASM is not rebuilt, per the standing instruction.
The exact oracle checks ordered top-10/100/1000 IDs/raw score bits, complete top-100
and exact counts for all 1,676 queries on ARM compact/identity/RGB and x86
compact/RGB. New and old index formats are not rewritten by these reader changes.

Standalone reorder's existing merge-time planning, migration and writer-budget
limitations remain separate work. The repair does not claim that every query
family now beats Tantivy or that every arbitrary cross-field composition is fast.

## Initial matched Lucene BP/RGB top-10

Pinned upstream Lucene 10.4.0 BP adapter, Java 21, vector module and native access enabled, ParallelGC, query cache disabled. Same canonical full corpus, Cascade Lake host and engine CPU 2. Lucene force-merges to one BP-reordered segment. All 1,676 query counts match the preserved reference; cross-engine score bits/rankings are not equated.

Four fresh engines receive at least 60 seconds of whole-workload warmup before seven rotated official passes. Supplemental terms use ten seconds and five passes. Indexing, build, exact-count checks and residency sampling finish outside the timing window. This run is separate from the shorter-warmup repair comparison above.

| Workload     | Summa RGB µs | Summa RGB-off µs | Lucene BP/RGB µs | Tantivy µs | Summa RGB / Lucene |
| ------------ | -----------: | ---------------: | ---------------: | ---------: | -----------------: |
| official     |      444.841 |          510.738 |          392.235 |    534.525 |             1.134× |
| supplemental |      101.493 |          134.319 |           86.781 |     55.645 |             1.170× |

| Official family    | Queries | Summa RGB µs | Lucene BP/RGB µs |  Ratio |
| ------------------ | ------: | -----------: | ---------------: | -----: |
| intersection       |     300 |      325.480 |          306.187 | 1.063× |
| phrase             |     300 |      486.759 |          505.129 | 0.964× |
| union              |     301 |      499.622 |          325.063 | 1.537× |
| negated            |      19 |      432.208 |          361.973 | 1.194× |
| intersection_union |      40 |      908.803 |         1394.824 | 0.652× |
| term               |       1 |      312.672 |          912.448 | 0.343× |

[Raw evidence and reproduction](benchmark-results/rgb-repair-2026-09-16/README.md).

## Selected reader and compact-layout follow-up

The selected OR-tail and mapped-admission changes improve official top-10 from
441.718 to 424.943 µs on the original RGB index, but Lucene RGB is 393.534 µs:
**the updated top-10 target remains unmet (8.0% slower)**. This is the same
5,032,104-document fixture, compiler, machine and native release flags. Seven
rotated official passes follow ten seconds of native warmup per command;
Lucene official top-10 receives sixty seconds. Supplemental terms use five
passes and ten seconds. All eight engines run serial requests; builds, exact
reference checks and memory sampling are outside latency.

“Selected/original” uses the selected reader on the original frozen RGB files.
“Compact RGB” uses that same reader on a fresh RGB rewrite preserving compact
headers/directories. Its `.chunks`, stored fields, fast fields and row statistics
are byte-identical to the original RGB fixture. It retains the same permutation,
ranked ID/score-bit/count references and norm policy. “Before” is the repaired
handoff reader, not the originally broken logical-replay reader.

### Official 962 queries

| Operation             | Before RGB | Selected/original | Compact RGB | Lucene RGB | Tantivy |
| --------------------- | ---------: | ----------------: | ----------: | ---------: | ------: |
| Top 10                |    441.718 |           424.943 |     429.560 |    393.534 | 532.048 |
| Top 1000              |    831.648 |           839.102 |     842.470 |    854.396 | 927.447 |
| Top 100 + exact count |    724.655 |           724.421 |     733.098 |   1078.629 | 875.129 |
| Exact count           |    344.500 |           349.461 |     347.905 |    378.291 | 428.260 |

### Supplemental 714 terms

| Operation             | Before RGB | Selected/original | Compact RGB | Lucene RGB | Tantivy |
| --------------------- | ---------: | ----------------: | ----------: | ---------: | ------: |
| Top 10                |    103.642 |            69.277 |      71.207 |     86.756 |  56.148 |
| Top 1000              |    863.846 |           763.532 |     775.068 |    478.015 | 440.568 |
| Top 100 + exact count |    300.937 |           208.634 |     211.154 |    471.224 | 263.356 |
| Exact count           |     18.794 |            18.682 |      18.853 |     18.628 |  15.116 |

The same-reader compact rebuild saves **192.2 MiB** in postings and positions,
but official top-10 is 1.1% slower than the original representation in this run.
It repairs representation loss; smaller files are not presented as a latency win.
The frequent-term graph and finer-partition variants are rejected for query
performance: top-10 is 432.040/432.673 µs versus compact RGB's 429.560 µs, while
ARM ranked results are essentially flat. Their smaller graph makes partitioning
cheaper, but no production graph default changes.

### Where top-10 still differs

| Family             | Selected/original µs | Lucene RGB µs |
| ------------------ | -------------------: | ------------: |
| intersection       |              303.566 |       306.915 |
| phrase             |              491.741 |       504.860 |
| union              |              455.484 |       326.813 |
| negated            |              460.342 |       362.718 |
| intersection_union |              921.105 |      1433.569 |

Union execution remains the main family deficit. Standalone terms improve
substantially with mapped batch admission, but that supplemental result does not
establish an official-workload win. Adaptive windows, local required driving and
window-bounded SIMD intersections were not selected from their isolated probes.
The retained general OR tail reuses the existing typed conjunction executor.

### Resident memory is separate from index size

Fresh processes execute one official pass per command after a known count query
establishes that opening is complete. The following first top-10 snapshots precede
the other commands. RSS includes mapped pages; “anonymous” includes allocator and
runtime memory. No pages are locked. These are warm-query residency measurements,
not cold-cache latency or concurrent-workload results.

| Engine/layout         | RSS MiB | Anonymous MiB |
| --------------------- | ------: | ------------: |
| Selected/original RGB | 1019.39 |          5.58 |
| Compact RGB           | 1055.73 |          5.56 |
| RGB off               |  745.95 |          5.57 |
| Tantivy               |  573.92 |          0.68 |
| Lucene RGB            |  860.11 |        309.27 |

Original RGB residency is concentrated in positions (637.94 MiB), postings
(268.56 MiB), and the document map (57.09 MiB), with only 5.58 MiB anonymous.
Lucene maps 362.02 MiB of positions and 135.84 MiB of document postings in this
snapshot, while its JVM contributes substantially more anonymous memory.

Compact RGB reduces position residency by 95.88 MiB, but posting residency rises
114.69 MiB and term-dictionary residency rises 17.68 MiB, leaving total RSS
36.34 MiB higher. The measurements establish which mappings differ; they do not
prove a particular page-fault mechanism. Position-file residency also does not
by itself explain OR CPU cost: ordinary term unions do not read positions.
The existing SIMD codec is measured separately on the same permutation to test
encoding without changing graph or scoring policy.

### Validation and measurement recovery

That stage passes 1,833 native tests (25 ignored), formatting, Clippy,
and native-without-sync compilation. The expanded existing two-term regression
covers mapped IDs, all posting codecs, gaps, ties, filters and late winners.
Its production scorer is byte-identical to the measured compact-layout source;
the stage's test-coverage patch records this test-only difference explicitly. Earlier
portable compilation passes for the unchanged production code, with the existing
unused-method warning. WASM remains skipped under the standing instruction.

All 1,676 exact references pass for every measured Summa layout on ARM and x86;
Lucene/Tantivy counts match the reference. Cross-engine score bits are not equated.
The complete indexes remain hash-audited. A memory-inventory harness error tried
to hash `java` without resolving PATH; it failed before launching memory queries.
After correction, the completed latency data was reused, the memory audit passed,
and all forty timing-file hashes were independently verified unchanged.

The evidence retains source overlays, query-level samples, immutable fixture
hashes, rejected experiments and recovered harness logs. Results are geometric
means of query medians and apply to this warm serial benchmark only.

### Remaining work identified by the comparison

The union deficit grows with clause count: selected/original RGB takes 1.293×
Lucene's time for the 198 two-term queries, 1.575× for 83 three-term queries,
and 1.669× for 13 four-term queries. These ratios use the same per-query medians
as the family table. They locate a workload deficit; they do not establish that
any one loop or codec causes it. The isolated candidate-join, two-term reduction,
local-boundary and combined probes test separate parts of this execution cost.

Byte norms and lookup-based normalization already exist for ordinary document
length columns, but explicitly reordered fields currently retain exact chunk-map
lengths. This RGB comparison therefore does not measure quantized normalization.
Extending byte norms to mapped plain fields needs a separate scoring-norm column
in physical order, with bounds derived from the same representative lengths;
physical geometry and old index semantics must remain intact. Substituting
quantized lengths only in the scorer would invalidate existing pruning bounds.

### Same-permutation SIMD codec result

The completed four-engine x86 comparison uses the same reader and RGB
permutation for all Summa layouts. All 1,676 exact references pass, and stored
fields, fast fields, row statistics and document maps remain byte-identical.
No format version or default changes. The existing Simd4x option is exercised
through the standalone reorder writer after compact-format preservation.

| Operation             | Original RGB µs | Compact RGB µs | SIMD RGB µs | Lucene RGB µs |
| --------------------- | --------------: | -------------: | ----------: | ------------: |
| Top 10                |         475.696 |        474.245 |     473.595 |       440.889 |
| Top 1000              |         867.476 |        850.180 |     863.530 |       873.304 |
| Top 100 + exact count |         754.616 |        762.874 |     760.468 |      1113.546 |
| Exact count           |         386.129 |        389.516 |     383.373 |       420.001 |

SIMD is flat for x86 top-10 and still 7.4% behind Lucene. Its ARM top-10 gain
(30.043 → 28.549 µs against compact RGB) does not transfer. Supplemental x86
top-10 is 70.200/70.339/71.642/88.012 µs respectively. Compare engines within
this table: absolute times drift relative to the earlier eight-engine run.

The memory result is useful independently of latency. Compact/SIMD index sizes
are 4,289.08/3,117.26 MiB, a 27.3% reduction. Posting bytes fall from 1,840.49
to 1,268.96 MiB; positions fall from 2,297.07 to 1,696.72 MiB.

| Layout       | Fresh top-10 RSS MiB | Anonymous MiB | Posting RSS MiB | Position RSS MiB |
| ------------ | -------------------: | ------------: | --------------: | ---------------: |
| Original RGB |              1018.90 |          5.59 |          268.56 |           637.94 |
| Compact RGB  |              1055.71 |          5.56 |          383.25 |           542.06 |
| SIMD RGB     |               855.82 |          5.54 |          273.50 |           451.56 |
| Lucene RGB   |               859.91 |        309.04 |          135.84 |           362.02 |

SIMD lowers RSS 16.0% against original RGB and 18.9% against compact RGB in
this workload, reaching approximately Lucene's total RSS with a different
heap/mapping split. This does not imply equal cold-cache or concurrent memory
behavior. Keep the codec opt-in: storage and residency improve, while the
requested top-10 latency target remains unmet. The complete 141-file export
and the separate 11-file fixture export are locally hash-verified.

### Selected two-term reduction repair

The subsequent reader-only change omits duplicate contribution storage and
reduction for mapped two-term windows under validated finite, nonnegative scoring.
All seven paired x86 passes improve; ARM top-10 improves 30.120 → 29.580 µs.
This source is selected in the main tree, with the expanded mapped regression.

| Workload/operation    | Before µs | Selected µs | Lucene RGB µs |
| --------------------- | --------: | ----------: | ------------: |
| Official top-10       |   421.535 |     414.659 |       392.170 |
| Official top-1000     |   833.542 |     819.923 |       860.651 |
| Supplemental top-10   |    71.609 |      71.214 |        87.790 |
| Supplemental top-1000 |   752.208 |     754.747 |       477.232 |

Official two-term unions improve 301.912 → 285.075 µs, explaining most of the
overall gain. All unions improve 451.284 → 432.612 µs, but still trail Lucene's
326.564 µs. The overall top-10 target remains unmet by **5.7%**.
All 1,676 exact references pass on RGB and RGB-off indexes on both architectures.
The delivered production scorer is byte-identical to the measured pair-sums
source; the source overlay adds only the expanded existing test. Index formats,
encoded bytes and configuration defaults are unchanged.

The selected source passes 1,833 native tests (25 ignored), formatting, Clippy,
native without sync and portable compilation. WASM remains skipped. A fresh
root release build is byte-identical to the measured ARM binary and passes all
1,676 pruned-versus-exhaustive checks on RGB-off, original RGB and SIMD RGB.

The separate block-sized candidate-join experiment leaves x86 top-10 flat
(427.854 → 429.181 µs), although top-1000 improves 830.612 → 817.965 µs.
It is not selected for the top-10 target. Both 30-file exports are locally
hash-verified. The local-window-boundary probe is also rejected: x86 top-10
is 440.348 → 441.180 µs and top-1000 is 847.646 → 849.651 µs. ARM is likewise
flat/slightly worse. Its 30-file export is verified. The combined-window probe
improves x86 top-10 only 424.858 → 421.053 µs, with flat ARM top-10, and is not
selected over the simpler two-term repair. Transposing canonical score reduction
regresses x86 top-10 419.165 → 422.230 µs and top-1000 826.648 → 834.807 µs;
it is also rejected. All these probes preserve exact references and retain
separate verified source/results exports.

## Final selected reader and codec confirmation

The selected production reader retains the physical-traversal repairs, guarded
OR tail, mapped batch admission and two-term contribution elision. The target
remains unmet: official top-10 is **6.7% slower than Lucene RGB**. It is
1.3% faster than its matched predecessor; the SIMD layout adds 2.0% latency
on this run. No rejected reader, graph or scheduling experiment is selected.

Seven official passes and five supplemental passes, same immutable indexes and
compiler/flags/host; native warmup is ten seconds per command, Lucene official
top-10 sixty seconds. Geometric means of query medians in microseconds:

| Workload/operation    | Before RGB | Selected RGB | Selected SIMD RGB | Lucene RGB |
| --------------------- | ---------: | -----------: | ----------------: | ---------: |
| official TOP_10       |    422.282 |      416.881 |           425.249 |    390.625 |
| official TOP_1000     |    835.406 |      824.472 |           835.657 |    848.283 |
| supplemental TOP_10   |     71.228 |       70.099 |            72.239 |     88.379 |
| supplemental TOP_1000 |    749.313 |      752.598 |           764.611 |    475.309 |

Fresh-process memory snapshots follow latency, with one official top-10 pass:

| Reader/layout     | RSS MiB | Anonymous MiB |
| ----------------- | ------: | ------------: |
| before-rgb        | 1019.14 |          5.58 |
| selected-rgb      | 1019.28 |          5.58 |
| selected-simd-rgb |  856.00 |          5.54 |
| lucene-rgb        |  864.87 |        314.05 |

SIMD remains an opt-in space/residency tradeoff. The original and compact index
byte counts, component attribution and exact-permutation checks above remain
applicable. The selected reader's scratch capacity is unchanged; the two-term
repair removes work, not an index-sized allocation.

The native harness passes 1,833 tests (25 ignored), formatting, Clippy and
native-without-sync compilation; portable compilation passes with the existing
unused-method warning. WASM is skipped under the standing instruction. The fresh
root release binary is byte-identical to the measured ARM binary and passes
1,676 pruned/exhaustive score/count checks on RGB-off, original RGB and SIMD RGB.
The source overlay differs from the measured pair-sums source only in expanded
existing test coverage. The selected x86 verifier also passes all 1,676
canonical references on the SIMD index after timing and memory collection.

The final comparison has four engines. A proposed addition of RGB-off controls
was refused by the runner guard because measurement had started, before any
process stop or script mutation. Those controls were not added and this run was
not restarted. The latest pair-sums source has paired ARM RGB-off timings and
x86 exact references; its x86 RGB-off latency was not remeasured. The earlier
eight-engine x86 comparison remains a separate result. All immutable-index
audits and the complete timing/memory export are verified before machine shutdown.

The machine is independently verified **TERMINATED** after all required downloads
and correctness checks. Changes remain uncommitted. Shutdown polling lost its
connection; the subsequent independent status check confirms completion.
