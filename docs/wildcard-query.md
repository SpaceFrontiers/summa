# Wildcard queries

## Core API

`WildcardQuery` is added to the core query API. It matches whole indexed UTF-8 terms:
`*` matches zero or more Unicode scalar values, `?` matches one, and backslash
escapes the following character. A dangling escape is an error. Patterns are
not tokenized or stemmed. The explicit text constructor lowercases like
`PrefixQuery::text`; the raw constructor preserves case. Regex metacharacters
other than wildcard operators are literal.

The invariant is the union of matching terms' documents, with one constant 1.0
score per document. Prefix and wildcard share union materialization and scorer
execution. This is a filter query, not a sum of term BM25 scores. Existing
chunked-field rejection remains explicit until logical-document mapping is
implemented for multi-term filters. No persisted format or legacy branch is added.

Term dictionary scans belong to structures; segment readers own field-key
composition, term metadata and posting reads; queries own pattern compilation
and document unions. An initial literal prefix restricts the dictionary scan.
Leading wildcards scan the selected field's vocabulary, never stored documents.
The existing SSTable block index is not a full-vocabulary FST: efficient arbitrary
automaton traversal would require a separate measured dictionary enhancement.

Patterns are limited to 1,024 bytes and compiled-regex/DFA caches to 2 MiB each before dictionary access.
At most 1,000,000 dictionary terms are examined per segment. Matched terms and
posting expansion retain the prefix limits of 1,024 terms and 5,000,000 postings; budget exhaustion is an error, never partial results. Matching adds
bounded work per visited term, plus the existing posting-union cost. Async,
native sync and WASM use the same pattern and union implementation. Existing
prefix defaults and scoring stay unchanged. Both expanded-term filters translate
RGB physical IDs through the existing document map before returning logical IDs;
the previous prefix path omitted this mapping.

## Validation

Use a raw-text fixture to test full-term versus substring matching, interior and
leading stars, Unicode `?`, escaped operators, empty/no-match patterns, overlap
and exact counts, Boolean composition, and native/async equivalence. Compare a
trailing-star wildcard with the existing prefix query. Exercise scan and match
budget boundaries without constructing large corpora. Native checks and the
portable build must pass; runtime matching never scans or rewrites index metadata.

```rust
use hermes_core::WildcardQuery;
let query = WildcardQuery::text(body_field, "foo*bar?")?;
```

The subsequent [regex and escaped-literal follow-up](regex-query.md) accepts
826/826 expressions on the same small fixture. This does not remove expansion
limits or establish full-corpus parity.
