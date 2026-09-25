# Lexical vertical: positions, pruning, reordering, tokenization

Status: implementation ledger and remaining research, reviewed 2026-09-05.
Cursor-addressed positions, block bounds, windowed MaxScore, dynamic tokenization, phrase
queries, and cross-shard BM25 statistics are implemented. Sections below
identify remaining proposals explicitly; the initial audit records the
pre-change baseline.

Companion documents: [dynamic tokenization and phrases](dynamic-tokenizer-and-phrase.md),
[chunked text](chunked-text-fields.md), [posting codecs](posting-codecs.md),
[BMP compression](bmp-grid-compression.md), [merge-time reorder](merge-time-reorder.md),
[budgeted reorder](budgeted-reorder.md), [metadata pinning](hot-metadata-pinning.md),
and [Seismic research](seismic-research.md). The original investigation also
used private Azeroth design notes, which are not part of this repository.

## Why a separate vertical

The sparse vertical (SPLADE over BMP V19) and the lexical vertical (BM25 over
stemmed tokens) share executors but not data shapes:

|                              | SPLADE sparse                                                       | BM25 lexical                                                        |
| ---------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| vocabulary                   | 105,879 fixed dims                                                  | open, tens of millions of stems and identifiers per corpus, Zipfian |
| non-zeros per unit           | learned, roughly 100–300 per chunk, bounded by FLOPS regularisation | every token of the chunk; df ranges from 1 to N                     |
| weight                       | learned float, quantised u8                                         | integer tf plus a length norm; score needs k1, b, avg length        |
| positions                    | none                                                                | required (phrases, proximity, highlighting)                         |
| forward grid (dims × blocks) | feasible (`bmp_grid_compression.md`)                                | impossible at a 50M-term vocabulary                                 |
| pruning unit                 | block and superblock maxima over a dense grid                       | per-list block maxima on doc-ordered postings (MaxScore, BMW)       |
| query                        | 30–100 weighted dims, no phrases                                    | 2–15 tokens, quotes, per-query tokenizer hint                       |

Everything in this document is therefore inverted-list based and doc-ordered,
and reuses the sparse machinery only where the shape allows it (executor,
codecs, BP, pinning, cold IO, merge streaming).

## Initial audit of the text path (2026-09-03)

Historical baseline before the changes described below. Current posting
codecs and format gates are documented in [posting codecs](posting-codecs.md).

- Doc postings (`structures/postings/posting.rs`): 128-posting blocks, header
  `[count u16][first_doc u32][doc_bits u8][tf_bits u8]`, rounded-width bit
  packing of doc deltas and tfs, L0 skip 16 B per block `(first_doc,
last_doc, offset, max_weight f32 = block max tf)`, L1 `last_doc` per 8
  blocks, 24 B footer. Read zero-copy from the mmap (`deserialize_zero_copy`),
  blocks decoded lazily with deferred tf decode. Good.
- Block-Max MaxScore for text exists (`query/scoring.rs`, `TermCursor::text`,
  `current_block_max_score`), with conjunction skipping and block skipping;
  the upper bound is `bm25_upper_bound(max_tf, idf)` with the shortest
  possible length (`1 - b`). Since chunked fields (`chunked-text-fields.md`)
  MaxScore stays on when positions are requested.
- Positions (`structures/postings/positions.rs`): a separate list per term,
  blocks of 128 documents, block prefix `count u32 + first_doc u32`, then per
  document a vint doc delta, a vint position count and **absolute vint
  positions** encoded `(ordinal << 20) | token_position`. Skip entries are
  20 B per block. `SegmentReader::get_positions` reads the term's whole
  position range and `PositionPostingList::deserialize` copies it into the
  heap; `get_positions_into` then linearly vint-decodes a 128-document block
  to reach one document. Every phrase term pays a whole-list read and copy.
- Phrase (`query/phrase.rs`): conjunction over doc postings, then position
  check with `expected_pos += 1` per term (no gaps), score =
  `BM25(sum of term tfs) × 1.5` (not phrase frequency). One term collapses
  to `TermQuery`; no positions collapse to a MUST of terms.
- Server conversion (`summa-server/src/converters.rs::field_tokens`): phrase
  text is tokenized with the field tokenizer and hint, `Token.position` is
  discarded, so the query side cannot express gaps.
- Tokenizer (`tokenizer/mod.rs`): `tokenize_and_clean` splits on whitespace,
  strips every non-alphanumeric character, lower-cases; `DynamicStemmer`
  routes each token to the first hinted Snowball language of its script
  (Latin, Cyrillic, Greek, Arabic, Tamil). `StopWordTokenizer` exists as a
  wrapper but is not reachable from a schema spec: `TokenizerSpec::parse`
  accepts only `by` and `default`. CJK text is not segmented; hyphenated
  identifiers collapse (`float-zero` → `floatzero`); no diacritic folding.
- Lengths: chunked fields persist real chunk lengths (`.chunks`); non-chunked
  text fields persist only the segment average (`SegmentMeta::avg_field_len`)
  and score with tf as the length.
- Term dictionary (`structures/sstable.rs`): `TermInfo::Inline` for df ≤ 3
  without positions, otherwise `External { posting_offset, posting_len,
doc_freq, position_offset, position_len }`. FST or raw mmap index. Not in
  `PinPolicy`.
- Reordering (`segment/reorder.rs`, `merge-time-reorder.md`): BP permutes
  only BMP sparse blobs through their own doc maps; postings, positions,
  store and fast fields are copied unchanged, so text postings are in
  insertion order.
- IDF (`query/global_stats.rs`): lazily aggregated over the segments of one
  searcher; no cross-shard statistics (broker phase 2 merges by RRF).

## Position list format v3

Status: implemented (2026-09-03), updated to POS5/POS6 on 2026-09-23,
`structures/postings/positions_v2.rs`. The [posting codec contract](posting-codecs.md)
defines the current formats and unique-position certificate.

Goal: positions cost bytes only where they exist, and a phrase query touches
only the positions of documents that survived the doc-level conjunction.

### Layout

One position stream per term, addressed through the doc postings instead of
through its own skip list:

```text
.pos  per term:  [block 0]...[block n-1]
                 [block index: (byte offset u32, value start u64) × n]
                 [footer: n_and_flags u32, total_positions u64, magic "POS5"]
block:           [count u16][bits u8][pad u8][packed values: count × bytes(bits)]
                 at most 128 values per block
```

The high bit of `n_and_flags` certifies unique positions within each document;
the low 31 bits are the block count. POS6 separates compact block metadata from
the payload. The certificate enables rare-term score bounds for exact phrases.

- Values are **deltas**: a document's positions are sorted, the first is
  stored as is, the rest relative to the previous one. For a chunked field
  positions restart per chunk and the virtual id already carries the
  ordinal, so there are no ordinal bits. A non-chunked multi-valued field
  keeps the `(ordinal << 20) | position` values; the ordinal step is one
  large delta that widens that block only.
- The packer is the rounded-width codec of the doc blocks (`simd::
pack_rounded`, 0/8/16/32 bits), so decode is the same SIMD widening.
  OptPFD-style exception packing would shave another 30–40% and is the
  next codec step once sizes are measured on a real generation.
- Doc postings gain one `u64` cursor per L0 block: the number of values
  before the block's first document (cumulative tf). Inside a block the
  position of document _j_ is `cursor + Σ tf[0..j]`, which the iterator
  keeps as a running prefix (`BlockPostingIterator::position_cursor`).
  The position block index maps that logical cursor to a physical block and
  offset. Fresh streams have full interior blocks; merged streams may retain
  a short source tail as an interior block.
- The doc-posting footer grew by `total_positions u64 + flags u32 + magic
u32` ("BPL2"); a list ending in the magic has the extension, a legacy
  list ends with its `max_tf` (≤ 65 535 by the builder's u16 tf), so both
  stay readable. `TermInfo` is unchanged.

### Query execution

`PhraseScorer` runs the conjunction over doc postings as before and, for a
candidate, asks each term's `TermPositions` for
`[cursor, cursor + tf)`: one or two blocks decoded from the mmap into a
reused scratch buffer, delta-summed into the term's buffer, then the
offset-aware sliding match. Nothing is copied to the heap up front.
Still to do: `MADV_RANDOM` on `.pos` and a bounded `WILLNEED` on the
candidate's ranges.

### Size

With stop words removed and per-chunk restarts, in-document deltas are
roughly `chunk_len / tf`, so most blocks use 8-bit values: about 1 byte per
position plus 12 bytes per 128, against 2–3 bytes of absolute vint plus a
count per document and 20 B of skip entry per 128 documents before. The
`u64` cursor costs 8 B per 128 postings.

### Merge

Values are doc-agnostic, so a merge copies every encoded source block
verbatim and rebuilds only the `(byte offset, value start)` index and footer.
The block copy of the doc postings shifts each source's cursors by the values
of the sources before it (`Footer::total_positions`). Inline doc postings
that no longer fit inline are promoted to tiny blocks; existing external
blocks are never decoded or re-encoded.

### Compatibility

POS5/POS6 are a rebuild boundary. The position readers, merging, compaction,
and reordering reject earlier formats. No legacy position decoder or migration
branch remains.

Metadata format 6 introduced this layout. Current writers upgrade metadata
formats 6–8 to 9 without rewriting segments; older segment encodings can still
require a rebuild. See [compatibility](row-deletion.md).

## Doc postings and skip metadata

- **Block bounds with real lengths** (implemented 2026-09-03). The fourth
  L0 word packs `(max_tf u16, min_len u16)` and the footer carries the
  list minimum (`FLAG_LEN_BOUNDS`), so a bound is
  `bm25(max_tf, idf, min_len, avg)` instead of the `1 - b` floor; k1/b stay
  a query-time choice. The builder derives `min_len` from chunk lengths or
  the document norms; merges copy the words and take the minimum; a list
  rebuilt without lengths stores 1, which every real unit satisfies.
  Legacy lists keep their `f32` word and the old bound. A cursor uses the
  stored minimum only when it also scores with real lengths: a `tf`-as-
  length score is not bounded by a real-length bound.
- **Norms for plain fields** (implemented 2026-09-03). `.chunks` version 2
  adds per-field sections of kind 1: one `u16` token count per document of
  the segment (0 = no value), written by the builder for every plain
  indexed text field with tokens and concatenated on merge with zero fill
  for sources lacking the column. `short_document` now scores with its
  real length in MaxScore, `TermScorer` and `PhraseScorer`; multi-valued
  fields sum their values' lengths.
- **L1 superblock maxima** (implemented 2026-09-03). Every L1 group of
  8 blocks stores a packed `(max_tf, min_len)` word, giving two-level
  block-max skipping (Mallia & Porciani, "Faster BlockMax WAND with Longer
  Skipping", ECIR 2019), the text analogue of the BMP D/E hierarchy, at
  4 B per 1,024 postings.
- **Codec.** Keep 128-block OptPFD by default; after BP reordering (below)
  measure partitioned Elias-Fano (`structures/postings/partitioned_ef.rs`)
  for lists longer than a few hundred thousand postings, where PEF wins on
  clustered gaps (Ottaviano & Venturini, SIGIR 2014; Pibiri & Venturini,
  ACM CSUR 2020 survey).
- **Inline positions.** Terms with df ≤ 3 are external today only because
  positions force it; allow inline positions when the packed bytes fit the
  16 B inline slot (most tail identifiers), saving a `.pos` read for the
  rare-token queries the needle body bucket is made of.

## Pruning

- Keep Block-Max MaxScore as the rank-safe default. Fixed on the way
  (2026-09-03): the block-skip branch moved every cursor at the minimum
  document to its next block even when another essential cursor still held
  a document inside that block, so that document lost the skipped cursors'
  scores and could drop out of the top-k (regression test
  `block_max_skip_never_jumps_over_another_essential_cursor`; the same loop
  serves sparse MaxScore). The skip is now bounded by the next essential
  document, the Block-Max MaxScore rule. Superblock skips landed with it:
  every L1 group (eight blocks) stores a packed `(max_tf, min_len)` word
  (`FLAG_L1_BOUNDS`, 4 B per 1,024 postings, rebuilt from the L0 words on
  merge), and when the at-minimum cursors' group bounds cannot reach the
  threshold either, each jumps to its next group, bounded by the same next
  essential document. Sparse cursors keep single-block skips.
- **Windowed execution for text** (implemented 2026-09-03). Text cursors
  (in-memory lists, synchronous decode) run `MaxScoreExecutor::execute_windowed`
  instead of the document-at-a-time loop, which the sparse cursors keep. The
  id space is walked in windows of at most 4,096 ids that start at the first
  id a globally essential cursor can reach and end at the smallest current
  block end among them. Per window: every cursor's bound over the window is
  read from its L0/L1 words without decoding (`window_upper_bound`), the
  cursors are re-partitioned by those bounds (the block-max partition of
  Lucene's `partitionScorers`), a window whose bounds cannot reach the
  threshold is skipped by all cursors (`skip_past_sync`, no decode), each
  essential cursor scores all of its postings in the window in one pass over
  its decoded block into a dense score buffer plus a match bitset
  (`score_window_sync`; turbopuffer's batched iterator advancement), the
  candidates are filtered branch-free against the threshold minus the
  remaining bounds (`filter_competitive`, Lucene `filterByScore`), and each
  non-essential cursor is sought to the survivors in id order, strongest
  bound first. Rank-safe by the same argument as the loop; the parity test
  `windowed_text_maxscore_matches_exhaustive_and_doc_at_a_time` checks it
  against an exhaustive scorer and the loop over random corpora, lengths on
  and off, predicates, and seeded floors.
- **Filters and phrases as executor predicates** (implemented 2026-09-03).
  The query shape clients send is `MUST [phrases, filters] + SHOULD
[terms]`. When every SHOULD clause is a text term and the MUST/MUST_NOT
  clauses combine into one document bitset (`build_combined_bitset`:
  narrowest clause materialised, the rest probed; `PhraseQuery::as_doc_bitset`
  drains the phrase, resolving chunk ids to documents), the text MaxScore
  executors run with the bitset as their predicate (`finish_text_maxscore`,
  `finish_chunked_text_maxscore`), one group per field. The top-k is exact
  over the filtered documents because a filter never changes a bound, and
  documents matching only the filters fill the tail with score 0
  (`BitsetFillScorer`) when fewer than `limit` scored documents survive.
  Before this, text SHOULD went through an over-fetched unfiltered top-k and
  a `PredicatedScorer`, which could lose phrase matches that ranked below the
  candidate budget. A chunked phrase now keeps every matching document
  instead of its top `limit`, which a MUST constraint requires.
- **Phrase as a scoring clause.** Still open: a `PhraseQuery` in SHOULD is a
  verifier; its block bound would be the minimum of its terms' bounds.
- **Approximate and anytime modes** (implemented 2026-09-03).
  `MatchQuery.heap_factor` (0 < factor < 1) uses the sparse threshold scaling for the
  text executors: the collector's floor is divided by the factor, so
  candidates that cannot beat `threshold / heap_factor` are skipped and the
  retained candidates have exact scores, but exact top-k recall is not guaranteed.
  Zero/unset or 1 selects exact search; non-finite, negative and >1 RPC values
  are rejected (contract unified on 2026-09-05, without legacy translation). An approximate
  pass neither seeds from nor publishes to the segment-shared threshold, so
  it can never lower an exact clause's floor. `SearchRequest.time_budget_ms`
  is the anytime knob: the deadline travels on `SharedThreshold`, every text
  MaxScore executor reads the clock once per 4,096 loop iterations and stops
  at expiry with the results collected so far, and `SearchResponse.truncated`
  reports that it fired. Because the field-level BP pass clusters similar
  chunks, document order already approximates "best ranges first" (Mackenzie,
  Petri, Moffat, "Anytime Ranking on Document-Ordered Indexes", TOIS 2022);
  ordering the traversal by superblock bound instead of position is the
  remaining refinement, to be decided from measurements. The ~4–10% corpus
  budget in arXiv:2608.00229 (Hierarchical BM25) recovers 0.85–0.92 of the
  exhaustive score on a topically ordered corpus; that is the same knob
  without a second cluster index.
- **Long queries** (implemented 2026-09-03). `MatchQuery.max_terms` caps a
  match at its highest-idf terms (static query pruning, query order kept;
  `cap_terms` runs before each text finisher). Beyond ~24 terms MaxScore's
  essential set collapses; whether the residual deserves Block-Max WAND
  (Ding & Suel, SIGIR 2011; variable-sized blocks, Mallia et al., SIGIR 2017)
  is a measurement question on a real generation.
- Query term de-duplication with query tf weighting (done 2026-09-04: the
  server collapses repeated match tokens into one clause boosted by the
  count; `TermQueryInfo.weight` keeps a boosted term on the MaxScore path).
  Per-term boosts from the request (needed by the search-api field
  weighting) ride on the same weight.

## Field-level reordering

Status: implemented for the standalone reorder pass (2026-09-03,
`segment/text_reorder.rs`, driven by `reorder_segment`, i.e. the optimizer
and `IndexWriter::reorder`); merge-time BP for text fields is still open.

There is no document-level permutation in Summa and this design keeps it
that way: doc ids, the store, the fast fields and the dense vector maps
never move. A field that wants locality owns its permutation and a map back
to `(doc_id, ordinal)`, exactly as a BMP sparse field does today
(`merge-time-reorder.md`, `block-level-reorder.md`).

For text the map already exists. A chunked text field keys postings and
positions by a field-local virtual id and resolves hits through `.chunks`
(`chunked-text-fields.md`). Today virtual ids are assigned in indexing
order, which makes `doc_ids` in the map non-decreasing; nothing in query
execution depends on that order (`ChunkMap::resolve`, `length` are array
lookups). Reordering a text field therefore means:

- Schema: the existing `reorder` attribute on a chunked text field,
  `field content: text<...> [indexed<chunked, token_position>, reorder]`.
  Fields without it keep indexing order and block-copy on merge, as sparse
  fields without the attribute do.
- Objective: BP (`graph_bisection`, `BpBudget`, optimizer tiering) over the
  field's own postings, terms restricted to a mid-df band. Each reordered
  field computes its own order; several text or sparse fields never have
  to agree.
- Output: postings re-encoded in the permuted virtual-id order (the same
  rewrite a BP pass does for a BMP blob), `.chunks` written in that order
  with `(doc_id, ordinal, length)` per virtual id, and the position stream
  re-packed with each chunk's run of values moved to its new place. The
  runs are intra-chunk deltas, so no delta decoding is needed, but the
  128-value blocks and the cursors are rebuilt.
- Merge: with `reorder_on_merge`, BP runs inside the merge over the
  concatenated field; otherwise virtual ids are concatenated with per-source
  offsets as now.
- Predicates: doc-id filters resolve through the map before the check, the
  path the predicate-aware sparse MaxScore executor already takes; the
  chunked text executor gets the same hook. Fusion keys on
  `(segment, doc, ordinal)` after resolve and is unaffected.
- `short_document` becomes `chunked` too (one value per document), so it
  gains real lengths, ordinals and the same reorder path; non-chunked
  positioned fields stay in doc-id order and are not reorderable.

Expected gains per reordered field (Dhulipala et al., KDD 2016; Mackenzie
et al., ECIR 2019 reproducibility): 10–30% smaller delta-coded postings,
tighter block maxima, superblock skipping that actually fires. The cost is
the map lookup per hit that chunked fields already pay.

## Tokenization, stemming, stop words

Implemented 2026-09-04 as one dynamic tokenizer with three layers, identical
at index and query time (`summa-core/src/tokenizer/lex.rs`):

```
text<lex(by: <field>, default: <language|none>, stop_words: <bool>,
         segmenter: <icu|unicode|simple>, stem: <light|snowball|none>,
         variants: <bool>, fold: <bool>, max_token_length: <n>,
         han: <as_written|simplified>, cjk: <icu|dictionary>)>
```

Defaults are the recommended configuration (`icu`, `light`, variants on,
folding on, 64-character tokens), so `lex()` is a complete language-agnostic
tokenizer and only non-default options are rendered
(`summa-core/src/tokenizer/lex.rs`, one `LexOptions` struct shared by the
spec, the tokenizer and the renderer).

- **Segmentation and normalisation.** `icu` uses ICU4X's word segmenter
  (Unicode-3.0, compiled data): dictionary words for Chinese and Japanese,
  LSTM word breaks for Thai, Lao, Khmer and Burmese, UAX #29 elsewhere. A
  dictionary word of three or more characters also indexes its character
  bigrams as same-position variants, and runs of characters the dictionary
  does not know fall back to the bigram stream, so a query segmented
  differently still matches and a phrase over dictionary words stays exact.
  Every token is NFKC-normalised and lowercased; Arabic tokens get the Lucene
  orthographic normalisation (alef and yeh forms, teh marbuta, tatweel and
  harakat), Cyrillic `ё` becomes `е`; with `han: simplified`, Han characters are
  converted to simplified Chinese with OpenCC's character table
  (`tokenizer/han_t2s.rs`, 3,222 entries, Apache-2.0), index and query
  alike, so traditional and simplified spellings meet (no Rust crate fits:
  `opencc-fmmseg` pins an old rayon, `zhconv` is GPL). Tokens longer than `max_token_length`
  characters (default 64: hashes, sequences, URLs) are dropped and keep
  their position. `unicode` (UAX #29 + bigrams) and `simple` remain.
- **Morphology.** `stem: light` is the inflection-only family ported from
  Lucene's light and minimal stemmers (`tokenizer/light_stem.rs`, Savoy /
  Dolamic rules; English is Harman's S-stemmer): English, French, German,
  Spanish, Italian, Portuguese, Russian, Finnish, Hungarian, Swedish,
  Norwegian, Arabic. Languages without a light stemmer (Danish, Dutch, Greek,
  Romanian, Tamil, Turkish) keep the written word. `snowball` is the full
  algorithm, `none` keeps every word. Routing is unchanged: the first hinted
  language of the token's script; `by` reads the hint from a sibling field
  at index time and the query passes `tokenizer_hint`; without `by` the
  `default` applies to everything and hints are ignored. `GetIndexInfo`
  reports every text field's tokenization (`TextFieldInfo`: hint field,
  variants, positions, chunked) so clients never parse the SDL. No Rust crate
  provides light stemmers or a multilingual lemmatizer (tantivy-stemmers
  carries a Czech one; simplemma is Python), which is why the port exists.
- **Variants.** With `variants: true` (the default), the written word is the indexed token
  and its light stem and diacritic-folded form (Latin, Cyrillic, Greek) are
  variants at the same position (Lucene `KeywordRepeatFilter` pattern).
  Variants are ordinary postings but `Token::variant` keeps them out of the
  field length. Query tokenization (`Tokenizer::tokenize_with` and
  `Purpose::{Index, Match, Exact}`) emits one
  form per word and never a variant: the stem for a match query (the written
  form when the language is unknown), the written form for a phrase or an
  exact term. Because a stem shares its word's position, a phrase term also
  matches words that stem to it (`"cell membrane"` finds `cell membranes`)
  while an inflected phrase term matches only that inflection; an exact
  written form for every term would need a second term namespace and twice
  the postings, which is not worth it. Without variants the (folded) stem replaces the word
  and every query form is that stem. The server converter passes the purpose
  per clause (`MatchQuery` → `Match`, `PhraseQuery` and `TermQuery` →
  `Exact`).
- Stop words: NLTK lists (`stop-words` crate, `nltk` feature) routed like
  the stemmers; dropped words keep their positions as gaps.
- **Japanese and Korean morphology** (`cjk: dictionary`, `cjk-dict` feature,
  `tokenizer/cjk_morph.rs`): lindera with UniDic and ko-dic embedded in the
  binary (about 200 MB; summa-server builds with the feature, the broker and
  wasm do not, and a `cjk: dictionary` spec fails to parse in a binary without it so
  index and query tokenization can never diverge). Hangul runs always go
  through ko-dic, kana runs through UniDic, Han runs through UniDic only
  when the document or query is hinted `ja` (otherwise ICU's Chinese
  dictionary). Particles, auxiliaries, endings and suffixes are dropped and
  keep their positions; content morphemes are indexed as written with the
  dictionary base form (`食べ` → `食べる`) as a same-position variant and,
  for three or more characters, their bigrams; match queries use the base
  form, phrases the surface. Korean gains the most: UAX #29 leaves particles
  attached to nouns (`학교에서`), which the dictionary separates (`학교`).
- Open: dictionary lemmatization for the long tail (no Rust source;
  simplemma's data is Python-side and CC BY-SA, and the literature finds
  no significant retrieval difference to light stemming), decompounding
  (needs per-language word lists).

## Scoring

- BM25 constants are per-field schema options (`indexed<k1: .., b: ..>`,
  implemented 2026-09-03; `Bm25Params::for_field`) with the Lucene
  formulation as default (Kamphuis et al., "Which BM25 Do You Mean?", ECIR
  2020: the variants are practically equivalent, so keep the common one).
  The stored `(max_tf, min_len)` bounds are parameter-free, so k1/b change
  without a rebuild.
- Phrase score = BM25 over the **phrase frequency** (number of matched
  occurrences) with the phrase's summed idf, not `1.5 × BM25(Σ tf)`
  (implemented 2026-09-03).
- **Proximity rescoring** (implemented 2026-09-03, `query/proximity.rs`).
  `MatchQuery.proximity_weight` (and `proximity_window`, default 8) turn on
  a second stage over an over-fetched top-k of the MaxScore pass:
  sequential-dependence ordered and unordered window counts for adjacent
  query term pairs (Metzler & Croft, SIGIR 2005), each saturated with the
  field's k1/b and length and weighted by the pair's mean idf, added to the
  BM25 score. Positions come from the v2 stream through the term cursors.
  Approximate by design (candidate pool of 4× the limit, no cross-segment
  threshold seeding for the pass); works on plain and chunked fields and
  under filters.
- **Fields.** `short_document` and `content` are fused by RRF today. Inside
  the lexical channel a per-field boost with a doc-level max/sum
  (`bm25f_score` already exists) is the better combination once norms exist
  for `short_document`.
- Global IDF for chunked fields must use chunk counts consistently across
  segments (`LazyGlobalStats::text_idf` uses document totals; verify before
  the first multi-segment chunked index).

## Statistics, sharding, residency

- Searcher-wide IDF (implemented 2026-09-03). Text scorers used the segment's
  own `df` and document count, so the same document scored differently in a
  small and a large segment and the cross-segment threshold compared
  incomparable scores. `Query::text_terms` lists the BM25 terms of a query;
  for a multi-segment searcher `Searcher::query_text_stats` sums their
  document frequencies over the segments through the term dictionary
  (`text_doc_freq_sync`, no posting bytes read, cached per searcher) and
  passes `GlobalStats` (per-field corpus size in scoring units, average
  length, term df) through `ScorerOptions::global_stats`; a query's own
  `with_global_stats` still wins. Phrase scoring keeps local idf (verifier).
- Cross-shard IDF (implemented, including broker fan-out).
  `GetTextStats(index, query)` returns the statistics of the query's terms on
  one backend; `SearchRequest.text_stats` carries a total back and replaces
  the backend's own statistics. The partitioned broker sums
  `GetTextStats` over the shards of an index before fanning the search out;
  see [partitioned indexes](broker.md#phase-2-partitioned-indexes). The periodic merged DF table for
  terms above a df threshold (the "global DF table" of arXiv:2608.00229)
  stays the fallback if the extra round trip is measured to matter.
- `PinPolicy`: pin the term dictionary index (FST or raw index) and the
  L0/L1 skip sections of terms above a df threshold; `.pos` stays evictable
  with random-access advice.
- Segment size: virtual ids and doc ids are u32 per segment; `pos_cursor`
  is u64 so one term may exceed 4G positions in a segment.

## What is reused from the sparse vertical

| sparse component                                                      | lexical use                                  |
| --------------------------------------------------------------------- | -------------------------------------------- |
| `MaxScoreExecutor` / `TermCursor`                                     | unchanged; gains phrase cursors and L1 skips |
| OptPFD / rounded-width packers, SIMD unpack                           | doc deltas, tfs, and now positions           |
| `graph_bisection`, `BpBudget`, optimizer tiering, `reorder` attribute | per-field permutation over the chunk map     |
| `PinPolicy`, cold-IO writers, `MADV_*` helpers                        | term index, skips, `.pos`                    |
| streaming merge with offset rebasing                                  | pos stream concatenation                     |
| `heap_factor` approximate threshold                                   | approximate BM25                             |
| chunk maps and ordinal fusion keys                                    | unchanged                                    |

Not reused: BMP grids, LSP hierarchy, forward index, Seismic-style
per-list clustering (all need a bounded vocabulary).

## Sequencing

1. `stop_words` spec + gap-preserving positions + `PhraseQuery` offsets.
   Small; unblocks azeroth's templates.
2. Position format v3 + lazy phrase scorer, `(max_tf, min_len)` block
   bounds, norms for plain fields (all done). Must precede the first
   text-bearing index generation.
3. UAX #29 tokenizer, folding, CJK bigrams (done, opt-in `segmenter:
unicode`).
4. Filters and phrases as executor predicates, phrase-frequency scoring,
   L1 superblock maxima, per-field k1/b, proximity rescoring (all done).
   Block-Max WAND as an alternative executor for short queries: possible on
   the same cursors, to be decided from measurements on a real generation.
5. Field-level BP reordering of chunked text fields through their chunk maps
   (done for the standalone/optimizer pass; merge-time pass open).
6. Anytime/budgeted BM25 and long-query handling (done: `heap_factor`,
   `max_terms`, `time_budget_ms` + `truncated`; bound-ordered traversal open).
7. Searcher-wide DF (done) and the cross-shard statistics contract (done);
   broker scatter-gather is implemented in [phase 2](broker.md#phase-2-partitioned-indexes).

## Lucene and turbopuffer parity checklist

What Lucene 10's `MaxScoreBulkScorer`/`Lucene104PostingsReader` and
turbopuffer's FTS v2 (blog "fts-v2-maxscore", batched iterator advancement)
do, and where the text vertical stands (2026-09-03):

| Technique                                                                                                                                                                                          | Source      | Summa                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Block-Max MaxScore with a two-level skip (Lucene: 256-doc level 0, level 1 = `LEVEL1_NUM_DOCS`)                                                                                                    | Lucene      | done: 128-doc L0 blocks, 8-block L1 groups, both with bounds                                                                                                                             |
| Window-at-a-time scoring: essential lists bulk-scored into a dense `windowScores[]` + `windowMatches` bitset, non-essential lists applied to the surviving candidates (`scoreNonEssentialClauses`) | Lucene      | done: `execute_windowed` for text cursors (see below)                                                                                                                                    |
| Batched iterator advancement: each postings iterator advances many times in a row before the next one (cache prefetch, branch prediction, SIMD)                                                    | turbopuffer | done: same windowed executor                                                                                                                                                             |
| Per-window essential/non-essential partition from block maxima (`partitionScorers`, `maxWindowScore`)                                                                                              | Lucene      | done: window bounds from L0 entries (`window_max_score`)                                                                                                                                 |
| Branch-free competitive filtering of a candidate buffer (`VectorUtil.filterByScore`)                                                                                                               | Lucene      | done: `filter_competitive`                                                                                                                                                               |
| Adaptive window size from candidate density (`minWindowSize` doubling up to `INNER_WINDOW_SIZE = 4096`)                                                                                            | Lucene      | done: window ends at the smallest current block among essential cursors, capped at 4096 ids                                                                                              |
| Bulk SIMD block decode (`ForUtil`/`PForUtil` on the Panama vector API)                                                                                                                             | Lucene      | done: `simd::unpack_rounded`, prefix sum as a plain loop                                                                                                                                 |
| Lazy frequency decode (freqs decoded only when scored)                                                                                                                                             | Lucene      | done: `deferred_tf`                                                                                                                                                                      |
| BM25 norm cache: 1-byte `SmallFloat` norm, 256-entry `1 / (k1 * ((1 - b) + b * len / avgdl))` table, `weight - weight / (1 + freq * normInverse)`                                                  | Lucene      | equivalent: exact u16 lengths, per-cursor precomputed coefficients, one division per posting; no lossy norm                                                                              |
| Impacts as a Pareto list of `(freq, norm)` pairs per block                                                                                                                                         | Lucene      | open: one `(max_tf, min_len)` pair per block/group is a looser bound; measure before widening the L0 word                                                                                |
| Dense blocks stored as bitsets (`docBitSet`, `intoBitSet`, `docIDRunEnd`)                                                                                                                          | Lucene      | partial: width-0/8 rounded blocks; no bitset blocks, no run detection (matters for conjunctions, not for MaxScore)                                                                       |
| Dynamic minimum competitive score from the collector, shared across slices (`MaxScoreAccumulator`)                                                                                                 | Lucene      | done: heap threshold + `SharedThreshold` across segments                                                                                                                                 |
| Conjunction optimisation inside MaxScore; MUST clauses as cheap predicates                                                                                                                         | Lucene      | done                                                                                                                                                                                     |
| MaxScore rather than BMW for long queries (turbopuffer: several times faster at tens of terms)                                                                                                     | turbopuffer | done: MaxScore default, `max_terms` cap; BMW only if measured                                                                                                                            |
| Terms ordered by upper bound                                                                                                                                                                       | both        | done (global order; per-window order in the windowed executor)                                                                                                                           |
| Time-limited bulk scoring                                                                                                                                                                          | Lucene      | done: `time_budget_ms` / `truncated`                                                                                                                                                     |
| Query term de-duplication with query-tf weighting                                                                                                                                                  | both        | done: the server collapses a repeated token into one clause boosted by its query term frequency; `TermQueryInfo.weight` scales the idf so a boosted term stays on the text MaxScore path |
| Bimorphic call sites for JIT inlining                                                                                                                                                              | Lucene      | not applicable (monomorphised Rust)                                                                                                                                                      |

## Evaluation

- Rank-safety differential tests: MaxScore vs exhaustive Boolean on
  synthetic Zipfian corpora with real lengths, for every pruning change.
- Phrase semantics tests: gaps, stop-word-only phrases, chunk boundaries,
  multi-valued ordinals, slop.
- Size accounting per file (`summa-tool diagnose`): `.post`, `.pos`,
  `.terms`, `.chunks` bytes per document before and after each step.
- Latency by query length (1, 2, 4, 8, 16, 32 terms) and by phrase count,
  cold and warm cache.
- azeroth `needle scholar` (`body` bucket) after each index-time change.

## References

- Deshpande & Sundararaman, Hierarchical BM25, arXiv:2608.00229 (2026).
- Ding & Suel, Faster top-k document retrieval using block-max indexes,
  SIGIR 2011. Mallia et al., Faster BlockMax WAND with variable-sized
  blocks, SIGIR 2017. Mallia & Porciani, Faster BlockMax WAND with longer
  skipping, ECIR 2019.
- Mackenzie, Petri, Moffat, Anytime ranking on document-ordered indexes,
  TOIS 2022 (arXiv:2104.08976).
- Dhulipala et al., Compressing graphs and indexes with recursive graph
  bisection, KDD 2016; Mackenzie et al., reproducibility, ECIR 2019.
- Ottaviano & Venturini, Partitioned Elias-Fano indexes, SIGIR 2014;
  Pibiri & Venturini, Techniques for inverted index compression, ACM CSUR
  2020; Yan, Ding, Suel, Inverted index compression and query processing
  with optimized document ordering, WWW 2009 (OptPFD, positions).
- Metzler & Croft, A Markov random field model for term dependencies,
  SIGIR 2005; Tao & Zhai, An exploration of proximity measures in IR,
  SIGIR 2007.
- Kamphuis, de Vries, Boytsov, Lin, Which BM25 do you mean?, ECIR 2020.
- Mallia, Suel, Tonellotto, Faster learned sparse retrieval with Block-Max
  Pruning, SIGIR 2024; Carlson et al., Dynamic superblock pruning, SIGIR
  2025; Bruch et al., Seismic, SIGIR 2024 (sparse-side context only).
- Williams, Zobel, Bahle, Fast phrase querying with combined indexes, TOIS
  2004 (next-word indexes; not planned while stop words are dropped).

## Phrase competitive scans (September 23 follow-up)

Ranked phrases retain exact BM25 scores, stable-ID ties, duplicate-start
multiplicity, slop, and RGB mapping semantics. The collector passes its local
heap floor into the phrase scorer. Strict score admission is allowed only when
subsequent physical IDs also follow stable result-ID order; reordered fields
retain equality. Count traversal never receives this ranked cutoff.

The owning phrase module caches integer length cutoffs for TF 1–32. An inverse
BM25 estimate seeds a bounded search, and canonical neighboring score comparisons
establish the actual cutoff, including floating-point rounding boundaries.
Certified singleton phrases use the canonical singleton score; larger TFs retain
conservative bounds. Posting-block and optional impact-group bounds can skip
whole ranges. SIMD scans test TF eligibility before gathering exact u16 lengths;
AVX2 and AVX-512 implementations retain scalar admission for unknown frequencies
and tails, with explicit gather extent and CPU-feature preconditions. Portable
execution uses the same cutoff semantics. Scratch remains constant per scorer;
no index-format or default change is required by this follow-up.

Position decoding caches a bounded prefix of per-document offsets and specializes
singleton and two-term matching without a second position codec. The intended
cost reduction is fewer candidate comparisons, indexed length loads and repeated
position-directory traversals. Same-index exhaustive IDs/score-bit comparisons,
scalar/SIMD lane-and-tail tests, native/async tests, and separate ordinary/RGB
measurements are required before interpreting throughput gains. The [completed paired comparison](benchmark-results/searchbench-2026-09-23-gap.md)
records the retained speedups, regressions, memory and remaining gaps. The
32-vCPU results are a separate campaign in the performance review.
