# Wildcard queries

## Core API

`WildcardQuery` is added to the core query API. It matches whole indexed UTF-8 terms:
`*` matches zero or more Unicode scalar values, `?` matches one, and backslash
escapes the following character. A dangling escape is an error. Patterns are
not tokenized or stemmed. The explicit text constructor lowercases like
`PrefixQuery::text`; the raw constructor preserves case. Regex metacharacters
other than wildcard operators are literal.

The invariant is the union of matching terms' documents, with one constant 1.0
score per document. Prefix, wildcard and regex share union execution. Ranked
plain segments lazily merge posting heads in a bounded heap and can stop when
the collector proves every later equal-score document loses the stable-ID tie.
Ordinary scorer iteration remains exhaustive, including nested Boolean callers.
Complete collection and mapped RGB segments retain union materialization.
Individual posting iterators are deferred too: validated
skip metadata supplies a lower bound for each unopened list, and the smallest
heap entry is opened before exposing its document. Seeking opens a list directly
at the requested block through the existing candidate iterator. This keeps
ordering exact while avoiding a decoded block and frequency scratch for every
expanded term when only a few terms contribute to top-k. Pending metadata and
the heap remain bounded by the existing expansion limits.
Complete unions consume decoded-ID batches from the existing
posting iterator. Sorted-vector or bitmap accumulation and RGB mapping remain
unchanged; batching removes the per-document iterator dispatch without
decoding frequencies or changing the encoded format.
Materialized sorted unions also expose the existing 128-document membership
batch protocol, so count collectors add batch lengths and deletion filters
still inspect the returned IDs. This avoids per-document virtual calls without
introducing a separate count or deletion path.
Inline expansion retains the dictionary decoder's bounded inline
postings directly. It avoids rebuilding encoded block lists for one-to-three
document terms. The segment reader still owns expansion limits and external
reads; the union owns membership and the shared dictionary decoder owns inline
parsing. Public prefix-posting APIs keep their existing return type by converting
through the canonical writer only when that representation is requested.
Malformed inline payloads must produce an error rather than disappear from a union.
This is a filter query, not a sum of term BM25 scores. Existing
chunked-field rejection remains explicit until logical-document mapping is
implemented for multi-term filters. No persisted format or legacy branch is added.

Term dictionary scans belong to structures; segment readers own field-key
composition, term metadata and posting reads; queries own pattern compilation
and document unions. An initial literal prefix restricts the dictionary scan.
Leading wildcards scan the selected field's vocabulary, never stored documents.
The existing SSTable block index is not a full-vocabulary FST: efficient arbitrary
automaton traversal would require a separate measured dictionary enhancement.

The September 24 profile of `g*itar` attributes about 40% of sampled CPU to regex
execution, 7% to UTF-8 validation, and 25% to dictionary decompression. A measured
implementation specializes exactly one unescaped star with no question marks into
literal prefix/suffix checks plus a non-overlap length check. UTF-8 validation
still rejects malformed terms, but only after the cheap byte checks pass. All
other wildcard forms retain regex execution. This changes neither matching
semantics nor expansion budgets and adds no dictionary cache.

Patterns are limited to 1,024 bytes and compiled-regex/DFA caches to 2 MiB each before dictionary access.
At most 1,000,000 dictionary terms are examined per segment. Matched terms and
posting expansion retain the prefix limits of 1,024 terms and 5,000,000 postings; budget exhaustion is an error, never partial results. Matching adds
bounded work per visited term, plus the existing posting-union cost. Async,
native sync and WASM use the same pattern and union implementation. Existing
prefix defaults and scoring stay unchanged. Both expanded-term filters translate
RGB physical IDs through the existing document map before returning logical IDs;
the previous prefix path omitted this mapping.

The type is exposed through the public core API, explicit query-language syntax
`field:wildcard("pattern")` or `field:foo*bar?`, and the benchmark's declared wildcard families.
A dedicated production protobuf variant is a separate interface.
[RegexQuery](regex-query.md) now supports whole-term regular expressions through
the core API and explicit query-language syntax. This does not make the 826
benchmark queries comparable: analyzer rebuilding, sloppy-phrase matching and
escaped query syntax are still required. Broad patterns may exhaust explicit
expansion budgets, as prefixes already do.

## Validation

Use a raw-text fixture to test full-term versus substring matching, interior and
leading stars, Unicode `?`, escaped operators, empty/no-match patterns, overlap
and exact counts, Boolean composition, and native/async equivalence. Compare a
trailing-star wildcard with the existing prefix query. Exercise scan and match
budget boundaries without constructing large corpora. Native checks and the
portable build must pass; runtime matching never scans or rewrites index metadata.

```rust
use summa_core::WildcardQuery;
let query = WildcardQuery::text(body_field, "foo*bar?")?;
```

The benchmark HTTP adapter accepts `wildcard`, `wildcard_scan` and
`wildcard_lead` through this API. Its regex family now uses the separate `RegexQuery`.
The query-string API accepts `field:wildcard("pattern")` or unqualified
`wildcard("pattern")` over the default fields. Pattern arguments use JSON string
escaping, so a literal star is written as `wildcard("a\\*b")`. Bare patterns are
consumed as one query: `th*e` is no longer split into prefix `th*` OR term `e`.
A single trailing star retains `PrefixQuery` execution. No full-corpus
wildcard throughput or count agreement is claimed yet.

The initial post-wildcard capability check submitted all 826 published expressions to the real
HTTP adapter over a four-document fixture: **801 accepted, 25 explicit errors**
(13 regex and 12 escaped-query syntax cases). All 145 wildcard expressions are
accepted. This fixture deliberately does not establish 10M-corpus count agreement
or test full-vocabulary expansion budgets; the throughput gate remains 15/826.
Raw responses and binary/query hashes are retained in
`.context/yonik-benchmark/gap/wildcard-http-capabilities-826.json`.
The native-only preflight separately accepts 730 expressions; its 83 syntax
rejections include 71 sloppy phrases handled by the HTTP adapter's existing
`PhraseQuery` translation, plus those same 12 escaped expressions.

The subsequent [regex and escaped-literal follow-up](regex-query.md) accepts
826/826 expressions on the same small fixture. This does not remove expansion
limits or establish full-corpus parity.

## Matched-value projection experiment

Expansion consumes dictionary values but currently clones each accepted term
key before immediately discarding it. The next candidate lets the canonical
bounded scanner project accepted entries directly to values; public key/value
scan APIs retain their output. One parser, scan order, result/scan budgets,
truncation boundary and error path remain shared across both projections and
native/async execution. Scratch remains bounded by the same expansion caps.
Measure allocations, CPU, throughput and memory before retaining the change.

## Dense union retention experiment

Complete unions already allocate a document bitmap when it is smaller than the
posting-ID vector, then unnecessarily expand that bitmap into an ID vector.
The next candidate retains the bitmap and exposes the existing forward-only
DocSet window protocol. Count collectors popcount those windows; deletion and
Boolean wrappers keep their existing filtering protocol. Sparse unions retain
sorted vectors and ranked plain unions retain lazy posting heads. No second
counter, query cache, payload format or expansion limit is introduced. Window
copies must preserve unaligned bases, consumed prefixes, tail padding and seeks.
The cost is bounded bitmap construction plus word scanning, avoiding one u32
allocation/write/read per matching document. Measure CPU and memory separately.

## Deferred posting-view construction experiment

Ranked expansion still constructs every optional bound/position byte view before
knowing which posting lists will supply top-k documents. A private candidate
keeps the owning reader's already-validated footer and single byte range until
the union opens that list. The canonical footer parser still validates every
matched external envelope before returning a scorer; the canonical list
constructor and decoder still own materialization. The first L0 document bound
orders unopened lists. Shared corruption observation, immutable byte lifetime,
posting-open diagnostics and public prefix return types stay unchanged. The
expected saving is fewer Arc operations and smaller pending-list storage;
complete unions must show no material regression. No payload cache or second
posting decoder is introduced.

The initial matched screens retain all three candidates for concurrent validation:
matched-value projection improves the corpus prefix probe about 9%; retained
bitmaps improve four dense count probes 1.62–1.83×; deferred list views improve
prefix top-k 1.70–1.76×. ARM fixtures independently support each direction. These
ratios compare successive checkpoints and must not be multiplied into a claimed
end-to-end result. Complete unions preserve exact count/score/delete semantics,
and the format, default configuration and query budgets remain unchanged.
