# Whole-term regex queries and escaped literals

## Implementation

Regex and wildcard share the existing dictionary/union execution owner. Regex
currently scans the field vocabulary; wildcard retains its literal-prefix
restriction. Compilation is limited to 1,024 pattern bytes and 2 MiB for the
compiled expression and DFA cache. The existing per-segment limits remain:
1,000,000 examined terms, 1,024 matched terms and 5,000,000 postings. Exhaustion
returns an error rather than partial results. Work is bounded dictionary
matching plus the existing posting union; this does not establish full-corpus
support for broad expressions. Chunked-field rejection and RGB ID translation
remain shared with existing expanded-term queries.

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

now accepts **826/826 expressions**, compared with 801/826 before. All 801
previously accepted responses remain identical on the unchanged four-document
fixture. This is capability evidence only; the performance comparison remains
the original 15-query subset until full-corpus semantic agreement is established.
