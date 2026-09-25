# Posting block execution and Tantivy comparison

## Implementation

This follow-up keeps the existing query executors and immutable index formats.
RGB is disabled. The current selected source is identified as `packed` in the evidence. It
adds score-only heap ordering to the cumulative `inline` reader candidate.

- Conjunctions intersect the two rarest decoded posting blocks with one bounded
  SIMD primitive, then probe other required terms. A 128-byte origin map preserves
  the original TF rows and canonical score reduction order. Phrase candidates use
  the same primitive and retain the original position identities.
- Overlapping blocks align inside the intersection loop. Directory seeking is
  reserved for blocks that cannot overlap. The L0 directory search reads entries
  directly instead of gathering 32 strided last-document IDs into scratch.
- Fully overwriting document, frequency and position decoders reuse initialized
  buffers. Removing `clear` before `resize` avoids a redundant zero-store pass;
  growth still initializes new elements, and errors still clear output.
- Score-only collectors pack the exact float total order and document ID into
  one eight-byte integer key, preserving signed zero, NaN payloads and tie order.
  They screen batches before serial heap admission. Complete
  ranked conjunctions reuse the existing intersection and report its exact
  cardinality. Required-plus-optional text queries can reuse an exact required
  term count when the existing planner's physical and cardinality gates hold.
- Prepared phrase bounds reuse query constants. Quantized normalization gathers
  lookup results into bounded scratch before the canonical scoring loop.
- [Owned byte views](owned-byte-views.md) retain a checked direct view alongside
  the existing Arc owner, removing repeated enum/range resolution on hot reads.

No additional executor, scorer, full-document bitmap or unbounded cache is added.
The counted-result handoff is explicit: arbitrary wrappers cannot request a
truncated child stream and lose its omitted count. Budgeted plans do not promise
an exact count. Tests cover wrappers, filters, deadlines, duplicates and codecs.

## Final confirmation

**Summa beats Tantivy on all four commands of the official 962-query workload
in this seven-pass comparison, with RGB disabled.** This is a claim about this
warm serial benchmark. Supplemental standalone queries and memory remain behind.

Full x86 corpus: 5,032,104 Wikipedia documents, Cascade Lake, CPU 2 for engine
processes. ARM uses the first 100,000 documents on Apple M4. Rust 1.98.1 /
LLVM 22.1.8, release LTO and native CPU flags. Seven rotated official passes
follow at least ten seconds of warmup per engine/command; supplemental terms use
five passes and at least three seconds. The Python protocol driver is not pinned.
Times include protocol overhead and are geometric means of per-query median µs.
Builds, profiling, full-file audits and memory runs are outside timed runs.

`packed` names the cumulative selected reader on unchanged compact postings with
exact norms. `packed-norm` uses byte norms; `packed-combined` uses existing compact
directories, Simd4x gaps and impact bounds with exact norms. No format defaults
change. The fastest official configuration is compact/exact, with RGB off.

### Official 962 queries, x86

| Operation             |   before |  packed | tantivy | packed-norm | packed-combined |
| --------------------- | -------: | ------: | ------: | ----------: | --------------: |
| Top 10                |  613.554 | 507.672 | 528.746 |     522.479 |         544.659 |
| Top 1000              | 1097.240 | 916.958 | 934.137 |     966.181 |         941.286 |
| Top 100 + exact count | 1166.682 | 800.381 | 875.478 |     815.863 |         830.820 |
| Exact count           |  477.260 | 406.881 | 431.714 |     405.058 |         410.488 |

Selected Summa is 4.0% faster for top 10, 1.8% faster for top 1000, 8.6% faster for top 100 + exact count, 5.8% faster for exact count.

Against starting Summa on the same index, elapsed time falls 17.3%, 16.4%, 31.4%, 14.7% in command order. These are same-run comparisons, not multiplied stage gains.

### Supplemental 714 standalone queries, x86

| Operation             |  before |  packed | tantivy | packed-norm | packed-combined |
| --------------------- | ------: | ------: | ------: | ----------: | --------------: |
| Top 10                | 137.696 | 131.789 |  56.391 |     124.505 |          66.868 |
| Top 1000              | 584.455 | 560.453 | 436.359 |     598.824 |         554.660 |
| Top 100 + exact count | 256.898 | 235.051 | 260.321 |     229.333 |         183.836 |
| Exact count           |  19.041 |  18.816 |  15.357 |      19.080 |          19.037 |

The selected reader/Tantivy ratios are 2.337×, 1.284×, 0.903×, 1.225× in command order. The combined layout changes these to 1.186×, 1.271×, 0.706×, 1.240×. These terms do not enter the official-workload headline.

### ARM official workload

| Operation             | before | packed | packed-norm | packed-combined |
| --------------------- | -----: | -----: | ----------: | --------------: |
| Top 10                | 33.111 | 31.086 |      31.183 |          30.978 |
| Top 1000              | 48.573 | 46.119 |      46.851 |          45.933 |
| Top 100 + exact count | 38.248 | 35.240 |      34.570 |          34.901 |
| Exact count           | 26.235 | 24.721 |      24.694 |          24.383 |

The selected cumulative reader improves ARM by 6.1%, 5.1%, 7.9%, 5.8% in command order. Small ARM differences remain sensitive to shared Mac load. The isolated packed-heap screen regressed ARM top-1000 by 1.5%; the complete final change above improves it.

### Official query families

Ratios below are selected Summa time / Tantivy time; below one is faster.

| Family                                                     | Top 10 | Top 1000 | Top 100 + count |  Count |
| ---------------------------------------------------------- | -----: | -------: | --------------: | -----: |
| intersection/global/intersection:num_tokens_2              | 0.968× |   0.971× |          0.925× | 0.960× |
| intersection/global/intersection:num_tokens_3              | 1.008× |   0.980× |          0.945× | 1.261× |
| intersection/global/intersection:num*tokens*>3             | 0.964× |   0.938× |          0.905× | 1.351× |
| intersection/intersection:num*tokens*>3                    | 1.429× |   1.391× |          1.384× | 0.797× |
| intersection_union/global/intersection_union:num_tokens_2  | 0.567× |   1.150× |          0.702× | 0.249× |
| intersection_union/global/intersection_union:num_tokens_3  | 0.540× |   0.936× |          0.729× | 0.375× |
| intersection*union/global/intersection_union:num_tokens*>3 | 0.484× |   0.965× |          0.752× | 0.255× |
| negated/global/negated:num_tokens_2                        | 1.448× |   0.952× |          1.193× | 2.029× |
| negated/global/negated:num_tokens_3                        | 1.290× |   0.986× |          1.066× | 1.736× |
| phrase/phrase:num_tokens_2                                 | 0.983× |   1.124× |          1.082× | 1.118× |
| phrase/phrase:num_tokens_3                                 | 1.016× |   1.077× |          1.042× | 1.053× |
| phrase/phrase:num*tokens*>3                                | 0.943× |   1.052× |          1.036× | 0.946× |
| term                                                       | 2.724× |   2.498× |          0.517× | 1.315× |
| two-phase-critic                                           | 1.382× |   1.373× |          1.348× | 1.394× |
| union/global/union:num_tokens_2                            | 1.079× |   0.974× |          0.756× | 0.689× |
| union/global/union:num_tokens_3                            | 0.787× |   0.738× |          0.842× | 1.116× |
| union/global/union:num*tokens*>3                           | 0.757× |   0.624× |          0.828× | 1.056× |
| union/union:num*tokens*>3                                  | 0.479× |   0.503× |          1.102× | 1.298× |

Aggregate parity does not mean every family or individual query is faster. The raw per-query samples preserve these tails.

### Memory and index size

Fresh x86 processes run all four commands sequentially, three official passes
each. RSS below is the observed maximum; anonymous is the final snapshot and a
subset of RSS. No locked pages were observed. Later commands retain pages
accessed by earlier commands. These runs do not measure latency.

| Configuration   | Index MiB | Max RSS MiB | Final anonymous MiB |
| --------------- | --------: | ----------: | ------------------: |
| before          |   4550.15 |      760.26 |                6.20 |
| packed          |   4550.15 |      760.03 |                6.23 |
| tantivy         |   2890.67 |      574.24 |                0.78 |
| packed-norm     |   4545.35 |      755.67 |                6.28 |
| packed-combined |   3639.34 |      623.60 |                6.21 |

The file-level comparison isolates the larger representations (MiB):

| Representation                               | Compact exact | Combined | Tantivy |
| -------------------------------------------- | ------------: | -------: | ------: |
| Document postings and TFs (`.post` / `.idx`) |       2005.28 |  1606.73 | 1005.72 |
| Positions (`.pos`)                           |       2440.27 |  1927.97 | 1764.83 |
| Terms (`.terms` / `.term`)                   |         53.00 |    53.04 |   44.94 |
| Lengths (`.chunks` / `.fieldnorm`)           |          9.60 |     9.60 |    4.80 |

These are whole-file roles, not an assertion that their internal metadata is
identical. Rounded payloads use byte-rounded widths; the Simd4x option packs full
blocks at exact widths and keeps Rounded tails. Directories, pruning envelopes
and position addressing also occupy space. The file totals establish where the
remaining size gap is; they do not attribute every byte to one codec decision.
See [posting codecs](posting-codecs.md) and [compact formats](compact-text-format.md).

Memory parity remains unmet. The reader changes remove CPU overhead without reducing the immutable payloads. Byte norms save 5,032,104 encoded bytes (about 4.8 MiB), so they cannot close the much larger postings/positions gap.

## What changed in measured work

The diagnostic binaries are separate from latency binaries. This comparison
isolates the retained traversal changes in `inline`, before the final heap
representation change. On a warm official 962-query ARM pass with compact exact
norms:

| Work                            | Starting reader | Selected reader |
| ------------------------------- | --------------: | --------------: |
| Top-10 posting seeks            |         319,056 |          85,109 |
| Top-10 document blocks          |          41,951 |          42,105 |
| Top-10 position blocks          |          11,855 |          11,855 |
| Top-100 + count posting seeks   |         387,065 |         143,438 |
| Top-100 + count document blocks |          72,586 |          68,932 |
| Count-only posting seeks        |         259,377 |         105,475 |
| Count-only normalization tables |               0 |               0 |

Most of this follow-up removes work around decoded blocks; it does not claim a
large reduction in top-10 decoded bytes. Strict corruption rejection remains.
The earlier [pruning fixes](search-pruning-fixes.md) removed unused count-only
normalization tables; this follow-up preserves that behavior.

## Validation and scope

The final native search harness passes 1,828 tests, with 25 intentionally ignored
and zero failures. Formatting, focused Clippy, native-without-sync and portable
core compilation pass. The portable build has the existing unused
`DocLengths::set_document_units` warning. WASM is not rebuilt, following the
standing user instruction. Lifecycle/RPC behavior is unchanged; the `full`
RPC suite and cold/concurrent workload tests are not part of this measurement.

All 1,676 queries match frozen references for ordered top-10/100/1000 IDs and raw
score bits, complete top-100 counts and count-only results on ARM and x86, across
compact exact, compact quantized, impact, Simd4x and combined layouts. Byte norms
use their own scoring reference. Hashes verify unchanged compact and quantized
index files. Direct byte views additionally pass native heap/mmap/thread lifetime
tests and three isolated strict-provenance Miri tests.

A global fixed-128 lower-bound replacement was rejected: it regressed official
x86 top-10 by 9.6% and ARM by 2.6%. The retained use is confined to candidate-batch
membership. Intermediate stage numbers are experiments, not multiplicative gains.

See the [verified evidence archive](benchmark-results/block-execution-2026-09-16/README.md) for exact source/binary/index hashes, raw samples, profiles, memory mappings and reproduction scripts.
