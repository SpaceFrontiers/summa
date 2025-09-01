# Chunked text fields: BM25 over passages with ordinals

Status: design (2026-09-03), implemented.

## Problem

Sparse and dense vector fields are chunk-level: every value of a multi-valued
field is one scoring unit, results carry per-ordinal scores, and hybrid
fusion (`FusionQuery`) keys on `(segment, doc, ordinal)` so a chunk found by
two verticals compounds. Text fields were document-level end to end:

- postings store `(doc_id, tf)` with term frequencies **summed across all
  values** of a multi-valued field, so a chunk never had its own BM25 score;
- the only place an ordinal existed was the high 12 bits of an optional
  position list, and text scorers exposed raw encoded token positions as
  `ScoredPosition`s. Fusion turned every token occurrence into a pseudo-chunk
  key that could never collide with a vector ordinal, and the response
  serialiser leaked those encoded positions as `ordinal_scores`;
- BM25 had no length normalisation: the engine used `tf` as the document
  length proxy because no per-document field length was persisted;
- Block-Max MaxScore was disabled whenever positions were requested — which
  is the default server search path and every fusion sub-query — so
  multi-term BM25 ran through the exhaustive `BooleanScorer`.

## Design

A text field declared `chunked` treats **every value as its own document for
the purposes of the inverted index**. Postings, positions, IDF and average
length are all computed over chunks; results are folded back to documents with
per-ordinal scores exactly like sparse vectors.

```
field content: text<stem(by: languages, default: simple)> [indexed<chunked, token_position>]
```

### Virtual ids

Each chunked field owns a segment-local, dense **virtual id** space. The
`n`-th chunk value indexed for the field (across all documents, in indexing
order) gets virtual id `n`; documents are added in doc-id order, so virtual
ids start out sorted by `(doc_id, ordinal)`. A reorder pass on a field with
the `reorder` attribute permutes them (BP over the field's postings,
`docs/lexical-vertical.md`); no query path depends on the order. Term
postings and position lists of a
chunked field are keyed by virtual id and use the existing
`BlockPostingList` format and the cursor-addressed position stream of
`docs/lexical-vertical.md` ("Position list format v2"). Token positions
restart at 0 in every chunk, so a `PhraseQuery` can never match across a chunk
boundary.

A sidecar file `seg_<id>.chunks` maps virtual ids back:

```text
[magic "CHNK"][version u32 = 1][num_fields u32]
TOC × num_fields: [field_id u32][num_chunks u32][total_tokens u64][data_offset u64]
per field:        doc_ids u32 × n | ordinals u16 × n | lengths u16 × n
```

`lengths` is the token count of the chunk (saturating at 65 535) and gives
BM25 its real length normalisation: `score = idf · tf · (k1 + 1) /
(tf + k1 · (1 − b + b · len / avg_len))`. Plain fields score with the per-document
length columns of the same file (`.chunks` version 2, section kind 1) and
block bounds use each block's minimum length (`docs/lexical-vertical.md`).

`FieldStats` of a chunked field count chunks: `doc_count` is the number of
chunks and `total_tokens / doc_count` is the average chunk length. IDF uses
the segment's chunk count as `N` and a posting's `doc_count` (number of
chunks containing the term) as `df`.

### Query execution

Every query on a chunked field runs an executor over virtual ids, resolves
each hit through the chunk map and combines per document:

| query                                                            | path                                                                                                                                                 |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| multi-term OR (`MatchQuery`, Boolean SHOULD of terms, one field) | Block-Max MaxScore over chunk postings, over-fetched like sparse (`2 × limit`, capped by the chunk count), then `combine_ordinal_results` with `Max` |
| single `TermQuery`                                               | same executor with one cursor                                                                                                                        |
| `PhraseQuery`                                                    | positional `PhraseScorer` over virtual ids, drained, then combined with `Max`                                                                        |
| `PrefixQuery`                                                    | rejected: prefix unions materialise doc-id sets and cannot be re-keyed cheaply                                                                       |

The combined result is a `VectorTopKResultScorer`, so `matched_positions`
carries `(ordinal, chunk score)` pairs, `SearchHit.ordinal_scores` holds true
ordinals, and fusion keys line up with the sparse/dense ordinals of the same
chunk when the client sends chunk texts in the same order as the vectors.
Boolean composition with other clauses (filters, other fields) happens at the
document level on the combined scorer, as it already does for sparse fields.

Cross-segment threshold seeding is skipped for chunked MaxScore groups: the
k-th _chunk_ score of a full heap is not a valid document-level floor after
`Max` folding.

### MaxScore and position collection

`collect_positions` no longer disables text MaxScore wholesale. It stays
disabled only for fields whose position mode tracks element ordinals
(`positions` / `ordinal`), because those callers expect the raw encoded
positions. Fields with no positions or `token_position` — phrase support
only — take the MaxScore path and report no positions; a chunked field takes
the MaxScore path and reports ordinals. Multi-term BM25 on the default server
search path and inside every fusion sub-query is therefore pruned again.

### Merges and reorder

Postings and positions of a chunked field are stacked with a **per-field
virtual-id offset** (the sum of the source segments' chunk counts) instead of
the document offset; the term-dictionary key carries the field id, so the
merger picks the offset per term. The `.chunks` sections are concatenated
with the document offset added to `doc_ids`; ordinals and lengths are copied
verbatim. BP reorder identity-copies `.chunks` like the other text files.

### Schema rules (fail loudly)

- `chunked` is valid only on text fields.
- A chunked field may declare `token_position` (per-chunk phrases) or no
  positions; `positions` and `ordinal` are rejected because the chunk is the
  ordinal.
- A chunked field is implicitly multi-valued (`multi`) so stored values
  round-trip as arrays.

## Wire and client impact

No protocol change. `GetIndexInfo` renders `chunked` in the SDL so clients can
detect the capability. `SearchHit.ordinal_scores` for chunked text fields are
chunk ordinals; the existing per-token leak for positioned text fields is
unchanged for `positions`/`ordinal` fields and gone for `token_position`
fields (they now report no positions).

## Not in scope

- Highlighting / matched-token export.
- Per-query combiner selection for text (fixed to `Max`; fusion works at
  chunk granularity anyway).
- Re-encoding existing indexes: a chunked field needs a rebuilt index
  generation.