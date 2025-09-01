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
field content: text<lex(by: languages, default: en, stop_words: true)> [indexed<token_position>]
```

`lex(by: <field>, default: <language|none>, stop_words: <bool>, segmenter: <icu|unicode|simple>, stem: <light|snowball|none>, variants: <bool>, fold: <bool>, max_token_length: <n>, han: <as_written|simplified>, cjk: <icu|dictionary>)`
is a parameterized tokenizer spec. Its canonical string form is stored in the
existing `FieldEntry.tokenizer`, so `metadata.json` has no new field and old
indexes load unchanged; `stop_words` is omitted from the canonical form when
false, so specs written before the parameter existed render identically. `parse_sdl` fails loudly when the `by` field is missing or not
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

### Stop words and phrase gaps

`stop_words: true` drops the stop words of the language a token is routed to
(the same script routing as the stemmer). The lists are the NLTK ones from
the `stop_words` crate (150–250 words per language, all 18 stemmer languages
covered); the crate's default ISO lists are several times larger and remove
content words such as `state`, which a scholarly corpus cannot afford. Filtering happens after cleaning and before stemming, so
inflected stop words are matched by their surface form. Dropped words produce
no term, posting, term frequency or position, and do not count toward the
BM25 field or chunk length.

Filtering does not renumber the survivors (Tantivy semantics). Indexing
`quantum of the art` records `quantum@0 art@3`; the query side tokenizes the
phrase with the same spec and hint and passes the token positions as phrase
offsets, so `"quantum of the art"` matches that document and `"quantum art"`
does not. The gap proves distance and order, not the identity of the removed
words: `"quantum in an art"` matches the same document. `slop` is measured
against the gapped expectation.

A phrase made only of stop words has no postings to check. As a standalone
query it is rejected like any token-less query; inside a `BooleanQuery`
(where clients put quoted spans as MUST constraints, possibly wrapped in a
SHOULD-only Boolean over several fields or hints) the clause is dropped with
a warning so the rest of the query still runs. `MatchQuery` text made only of
stop words is still rejected as token-less.

### Word segmentation (`segmenter`)

`simple` (default) splits on whitespace and strips every non-alphanumeric
character, so `float-zero` becomes `floatzero` and CJK text is one token per
whitespace run. `unicode` uses UAX #29 word boundaries (`float`, `zero`;
`p53`, `co2` and `10.1007` stay whole, punctuation inside a word is
stripped), turns each contiguous run of Han, Hiragana or Katakana into
character bigrams (`東京都` → `東京@0 京都@1`, a lone ideograph stays a
unigram), and folds diacritics of Latin, Cyrillic and Greek tokens after
stemming (`résumé` → `resume`, `ёлка` → `елка`). Stop lists and stemmers see
the unfolded letters. Query text goes through the same segmenter, so a quoted
`"東京都"` becomes the bigram phrase and `"float-zero"` the phrase
`float zero`.

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

`text` is tokenized server-side with the field's tokenizer (and hint), and
the token positions become the phrase offsets (`hermes_core::PhraseQuery::
with_offsets`), so terms must occur at their original distances (`slop` 0)
or within `slop` positions of them; with no stop words dropped that is the
usual consecutive match. The score is BM25 over the phrase frequency (the
number of occurrences of the whole phrase in the unit) with the summed idf of
the terms and the unit's real length, Lucene semantics. Requires
`indexed<token_position>` or `indexed<positions>`; without positions
`hermes_core::PhraseQuery` degrades to a MUST of the terms, and a single
token collapses to a `TermQuery`. Scored with BM25. The query-language parser
now builds the same `PhraseQuery` for `"quoted spans"`.

Clients: Python (`{"phrase": {...}}`, `tokenizer_hint`), TypeScript
(`{ phrase: {...} }`, `tokenizerHint`), WASM (`phrase`, `tokenizerHint`).
The broker forwards `SearchRequest` opaquely but must be rebuilt against the
new proto: a stale decoder drops the unknown oneof field and the server then
rejects the request as having no query.

## Performance notes (2026-09-03)

Measured on an M-series laptop with a 1.1 MB English text (`tokenize_hinted`,
release build, single thread): plain cleaning runs at ~340 MiB/s; adding the
English Snowball stemmer brings it to ~55 MiB/s, and mixed Russian/English
with `"ru,en"` to ~54 MiB/s. The stemmer itself (~120 ns per word) is the
cost; the tokenizer around it now fetches the thread-local stemmers once per
call instead of once per token and keeps a token's allocation when stemming
leaves it unchanged (+8% over the first implementation). Query-time
resolution of a `lex(...)` spec is cached per spec string in
`TokenizerRegistry` (~1 µs per query for a five-word query).

## Not in scope

- JSON (`SchemaFieldConfig`) schemas: the dynamic spec is SDL-only.
- Highlighting / matched-position export.