# Whole-term regex queries and escaped literals

## Implementation

Public search enters the query-language parser or a typed `RegexQuery`, then
uses the segment reader's existing bounded dictionary matching and term-union
scorer. The benchmark HTTP adapter translates its declared `regex` family into
that core type. Native sync, native async and WASM share compilation and union
execution. No schema, posting, protocol or persisted format changes are needed.

The result invariant is the deduplicated union of documents containing matching
whole terms, each with constant score 1.0. Patterns are case-sensitive and are
not analyzed or lowercased: changing regex source text can change its language.
Matching uses Unicode scalar values. The supported regular-expression subset
includes literals, character classes/ranges, grouping, alternation, `.`, `?`,
`*`, `+` and bounded repetition. It covers all 13 Searchbench regex expressions.
Extended Lucene operators, lookaround, backreferences, inline flags and shorthand
escape classes are rejected explicitly. Escapes quote literal punctuation.

Regex and wildcard share the existing dictionary/union execution owner. Regex and wildcard restrict scans to proven literal-prefix ranges;
patterns without a finite nonempty prefix set scan the field vocabulary. Compilation is limited to 1,024 pattern bytes and 2 MiB for the
compiled expression and DFA cache. The existing per-segment limits remain:
1,000,000 examined terms, 1,024 matched terms and 5,000,000 postings. Exhaustion
returns an error rather than partial results. Work is bounded dictionary
matching plus the existing posting union; this does not establish full-corpus
support for broad expressions. Chunked-field rejection and RGB ID translation
remain shared with existing expanded-term queries.

### Bounded prefix extraction

The implementation uses the regex parser's proven literal-prefix set
to restrict dictionary access. Extraction retains at most 64 literals of 64
bytes, collapses covered ranges, and falls back to the entire field when the
prefix set is infinite or includes the empty prefix. The original expression
still confirms every term. Alternation and optional literals must never drop
matches. Scan, matched-term and posting budgets apply to the union of all
ranges, not independently to each range. No dictionary or posting format
changes, auxiliary index, cache, or raised default limit are involved.

### Finite alternatives

For a proven finite language, the query retains complete exact literals before collapsing
prefix ranges. Extraction remains capped at 64 literals of 64 bytes, and look
assertions exclude this specialization. Sorted, deduplicated full terms use
bounded SSTable point lookups grouped by dictionary block. All other expressions
retain the current prefix scan and matcher. Empty literals are complete terms,
not an instruction to scan the vocabulary.

The existing segment expansion owner validates matched-term/posting budgets and
loads canonical inline/deferred postings; the existing constant-score union owns
deduplication, scoring and RGB translation. Sync and async share lookup planning
and block decoding. No writer, auxiliary index, cache, schema or format changes.
Scratch is bounded by extracted literals plus ordinary expansion metadata; each
distinct dictionary block is loaded at most once per lookup batch. A regression
with 10,000 unrelated prefix-sharing terms verifies that exact alternatives avoid
those blocks while retaining overlapping terms such as `http` and `https`.
Native sync and async results preserve IDs and score bits. This specialization
does not accelerate infinite expressions or general wildcard scans.

### Dictionary decoder code generation (September 26)

The remaining broad wildcard gap is dominated by decompression and entry
parsing on the unchanged 10M-document dictionary. The canonical variable-integer
decoder is kept at its call sites, allowing the compiler to eliminate small-call
and result-construction overhead while retaining its byte consumption,
truncation, and overflow checks. This is a code-generation change; the decoder,
cache, dictionary layout and scan budgets remain shared and unchanged.
[The paired corpus and architecture comparison](benchmark-results/dictionary-decoding-2026-09-26/README.md)
records exact query audits, native/portable validation, instruction-size and
memory costs, CPU and throughput. Shared posting readers are regression controls.

### Exact counts for dominated expansions (October 2)

Field admission is shared by prefix, wildcard and regex execution: unknown,
unindexed and non-text fields are errors, including estimates and count-only
collection. Each query type has one async expansion helper used by its scorer,
estimate and exact count; the native synchronous helper uses the same validation
and dictionary plan. This keeps admission consistent and avoids instantiating
the generic dictionary matcher separately for each collector path.

A pattern often expands to one frequent term and a small tail. The count owner
in `query/term_union` can use `|A union B| = |A| + |B minus A|`: take the largest
posting list's dictionary document frequency, materialize the deduplicated
union of the other lists using the existing bounded union, and probe only those
IDs through the largest list's canonical posting cursor. A single matched term
needs only its document frequency. Balanced expansions retain materialization.
The cost gate requires the dominant list to have at least 64 times as many
postings as all other lists combined. This bounds probe work; it does not change
semantics or expansion budgets. The [paired measurements](benchmark-results/pattern-count-2026-10-02/README.md)
record both architectures, rejected candidates and the retained implementation.

The query exposes an optional exact-count future, separate from its estimate.
The collector may use it only without scores, positions or deleted rows, and
only if it accepts count-only collection. Identity Boolean wrappers may forward
it; filters, exclusions and multi-clause Booleans must retain ordinary execution.
Chunked fields remain rejected. RGB document maps are bijections, so physical
IDs preserve cardinality. Native sync/async collection and WASM share this path.
Dictionary matching and all scan/term/posting limits run before the shortcut;
errors still propagate through the normal query and posting-integrity checks.

Scratch is the existing bounded materialization of the smaller lists plus one
posting cursor and one document batch. There is no new cache, storage format,
writer, merge rule or ranking path. The earlier direct-bitmap-accumulation
experiment was not adopted: it helped RGB counts but regressed default-index
wildcards. Measurements and rejected candidates belong in the performance review.

### Query syntax

Expose explicit syntax `field:regex("pattern")`, with JSON string escaping.
Bare regex operators are not reinterpreted as regex queries. Ordinary term
syntax additionally accepts dots, apostrophes and backslash-escaped literals.
Unescape once in the parser, then use the field's existing tokenizer; a colon
escaped in a term must not become a field separator, nor may escaped wildcard
operators become patterns. Boolean modifiers and explicit fields retain their
current meaning. Analyzer compatibility is separate from accepting syntax.

## Validation

Regression tests first reproduced rejected regex and punctuated expressions through
public query parsing/search. Tests verify exact IDs/counts, whole-term anchoring,
Unicode, duplicate matching terms, no-match cases, bounded expansion errors,
malformed/unsupported syntax, Boolean composition, RGB mapping, and native/async
agreement. The WASM search API exercises regex and escaped terms. The probe submits all
826 expressions to the real HTTP adapter on the same capability fixture; this
measures acceptance, not 10M-corpus count or ranking parity. Search harness and portable build outcomes are recorded in the performance review.

The [HTTP capability probe](benchmark-results/query-support-2026-09-24/README.md)
now accepts **826/826 expressions**, compared with 801/826 before. All 801
previously accepted responses remain identical on the unchanged four-document
fixture. This is capability evidence only. The later 10M-document campaign admits
677 of 826 expressions, retaining 149 explicit budget errors. The
[finite-alternative follow-up](benchmark-results/closing-gap-2026-09-24/README.md#september-26-finite-regex-alternatives-and-posting-investigation)
audits all 677 counts and top-100 IDs/score bits, then repeats five selected
families against the unchanged baseline and Luxir. Its four admitted regex
expressions improve aggregate TOP_10/TOP_100 throughput by 3.97×/3.81× in that
replay; infinite expressions retain the original scan cost. Equal cross-engine
ranking and a general regex speedup are not asserted.
