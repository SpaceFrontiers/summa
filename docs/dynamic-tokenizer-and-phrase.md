# Dynamic per-document stemming and wire-level phrase queries

Status: design (2026-09-03), implemented.

## Problem

A text field has exactly one tokenizer. Multilingual corpora therefore either
stem everything with one language (wrong stems for the rest), stem nothing, or
split the corpus into one field per language (many fields, query fan-out,
fragmented BM25 statistics). Separately, `hermes-core` has a positional
`PhraseQuery` with a BM25 scorer, but the gRPC protocol exposes only
`MatchQuery` (OR of tokens) and the query language's `"..."` syntax degraded
to an AND of terms.

## Dynamic tokenizer

`Tokenizer::tokenize_hinted(text, hint)` extends the tokenizer trait with an
optional caller-supplied hint; static tokenizers ignore it (default method
body), `DynamicStemmer` interprets it.

### Schema

```
field languages: text<raw_ci> [fast]
field content: text<stem(by: languages, default: simple)> [indexed<token_position>]
```

`stem(by: <field>, default: <language|simple>)` is a parameterized tokenizer
spec. Its canonical string form is stored in the existing
`FieldEntry.tokenizer`, so `metadata.json` has no new field and old indexes
load unchanged. `parse_sdl` fails loudly when the `by` field is missing or not
a text field, when the default language is unknown, and (new for all schemas)
when a plain tokenizer name is not registered — previously such fields
silently fell back to the builder's inline lowercasing.

### Indexing

`SegmentBuilder` resolves, per document, all text values of the `by` field,
trims and lower-cases them and joins them with commas ("ru,en"), then calls
`tokenize_hinted` for the hinted field. Documents without any hint value use
the spec's default and are counted in
`SegmentBuilderStats::unhinted_dynamic_docs`.

### Hint semantics (`DynamicStemmer`)

The hint is a comma-separated list of language codes or names. Every token is
classified by the script of its first alphabetic character (Latin, Cyrillic,
Greek, Arabic, Tamil, other) and stemmed with the first hinted language whose
Snowball algorithm operates on that script; tokens no hinted language covers
are kept as cleaned text. Snowball stemmers are script-local, so a Russian
paper with an English abstract tagged `["ru","en"]` stems both parts
correctly; same-script multilingual text (en+de) uses the first listed
language. No hint (or only unrecognised codes) applies the spec default;
`default: simple` means plain cleaning.

### Query side

`TermQuery`, `MatchQuery` and the new `PhraseQuery` carry `string
tokenizer_hint`. The server resolves the field's tokenizer as before and calls
`tokenize_hinted`, so index-time and query-time stemming agree whenever the
client passes the query language. Static tokenizers ignore the hint, so
clients may always send it.

### Merges

Segment merges union term dictionaries and never re-tokenize, so per-document
tokenizer choices are durable. BM25 statistics (IDF, average field length) are
shared across languages within the field; accepted.

## PhraseQuery on the wire

```
message PhraseQuery { string field = 1; string text = 2; uint32 slop = 3; string tokenizer_hint = 4; }
```

`text` is tokenized server-side with the field's tokenizer (and hint), terms
must occur consecutively (`slop` 0) or within `slop` positions. Requires
`indexed<token_position>` or `indexed<positions>`; without positions
`hermes_core::PhraseQuery` degrades to a MUST of the terms, and a single
token collapses to a `TermQuery`. Scored with BM25. The query-language parser
now builds the same `PhraseQuery` for `"quoted spans"`.

Clients: Python (`{"phrase": {...}}`, `tokenizer_hint`), TypeScript
(`{ phrase: {...} }`, `tokenizerHint`), WASM (`phrase`, `tokenizerHint`).
The broker forwards `SearchRequest` opaquely but must be rebuilt against the
new proto: a stale decoder drops the unknown oneof field and the server then
rejects the request as having no query.

## Not in scope

- JSON (`SchemaFieldConfig`) schemas: the dynamic spec is SDL-only.
- Highlighting / matched-position export.
- Stop-word filtering for dynamic fields.