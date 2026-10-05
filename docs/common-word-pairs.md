# Common word pairs for exact phrases

September 29, 2026. Status: **implemented** (opt-in per field). On a
1M-document index with the 128 most common words, high_phrase top-10 runs 65×
faster and its exact counts ~790×, for 2.7% more index bytes; every count is
unchanged ([measurements](#measurements)).

## Problem

An exact phrase of frequent words (`"in a"`, `"of the"`, `"ref the"`) has
no rare term to drive it: the executor walks the postings of a frequent term,
seeks the other, and checks positions for every candidate whose term-based
bound can still compete. On the 10M-document benchmark `"ref the"` takes
152 ms for a top-10 and `"in a"` 13.5 ms; high_phrase top-10 as a family costs
343 ms per pass on an ordinary build and 552 ms on an RGB build, where
clustering documents rich in both words keeps the term-based bounds high
([measurement](benchmark-results/rgb-2026-09-29/README.md)). Exact phrase
counts over frequent words (high_phrase COUNT 0.92× Luxir) enumerate every
position match.

Phrases with at least one rare word are already fast: the rare word drives.

## Idea

Index, for a configured set of common words, every adjacent pair of them as
one extra term of the same field (Lucene's common grams, restricted to pairs
of common words). A two-word exact phrase of common words then reads one
posting list whose term frequency is the phrase frequency: ranked top-k runs
the single-term MaxScore executor with block bounds and impacts, and the exact
count is the pair's document frequency.

Measured on 100,000 documents of the benchmark corpus (approximate word
tokenizer; common words = the N most document-frequent):

|       N | Extra token occurrences | Extra postings | Distinct pairs | Phrases covered (high / med / low) |
| ------: | ----------------------: | -------------: | -------------: | ---------------------------------- |
|      64 |                    8.4% |          10.5% |          3,244 | 26/30, 17/46, 6/50                 |
| **128** |               **12.5%** |      **16.2%** |     **11,923** | **29/30**, 28/46, 13/50            |
|     256 |                   18.3% |          24.2% |         40,687 | 29/30, 37/46, 21/50                |
|   1,024 |                   35.2% |          47.8% |        284,823 | 30/30, 46/46, 35/50                |

Pairs carry no positions, so the extra postings add document IDs and
frequencies only; positions, which dominate the index, do not grow.

## Design

### Configuration

A text field may declare `common_grams`: an explicit list of words, stored in
`FieldEntry` (`#[serde(default)]`, omitted when empty) and set with
`SchemaBuilder::set_common_grams`. Schema admission rejects non-text,
unindexed and chunked fields, fields without token positions, duplicate or
empty words and more than 4,096 words; the segment builder rejects a word the
field's tokenizer does not produce as itself (`Purpose::Exact`). In SDL it is
`indexed<token_position, common_grams: ["of", "the"]>`. The list is fixed for the life of
the index: segments never disagree about which pairs exist. The benchmark
harness derives the list from the corpus (the top 128 by document frequency);
choosing it automatically at index time is left open (below).

### Index

`index_text_field` already sees every token and its same-position variants.
For each position `p` it emits one pair term `(x, y)` for every common term `x`
at `p` and common term `y` at `p + 1` (variants included, so a pair exists
exactly where the phrase would match, since phrase terms match variants by
bytes). The pair's term frequency in a document is the number of such `p`,
which is how the phrase executor counts exact phrase frequency (every start,
overlapping matches included: `"the the"` in `the the the` is 2). Pairs are derived from the
encoded positions the phrase matcher reads, collected across all values of the
field in a document, so cross-value adjacency in `TokenPosition` mode and
duplicate starts match exactly. Pair terms add nothing to the field length or
to `total_tokens`. In the builder a pair is interned as `first\0second` under
a flagged field id and only its serialized key takes the `0xFF` form.

A pair term is encoded as `0xFF x 0xFF y` in the field's own term space.
`0xFF` never occurs in UTF-8, so pair terms cannot collide with words, decode
unambiguously, and sort after every word of the field. Pair postings are
written without positions (`TermInfo` without a position stream).

Because pairs are ordinary terms of the same field, everything keyed by the
field applies unchanged: document lengths and length-based block bounds and
impacts at build, merge and reorder time; merge copy and re-encoding; RGB
reordering (the pair postings follow the field's permutation and chunk map);
deletions. RGB planning leaves pair terms out of its document graph: they are
derived from the words and would only perturb the permutation.

### Queries

A `PhraseQuery` uses the pair when all of these hold, per segment: two terms,
adjacent offsets (`is_adjacent`), slop 0, the field declares `common_grams`
containing both terms' bytes, and the collector does not ask for match
positions. It then scores the pair's postings as the phrase scores itself:
BM25 over the pair frequency, with idf the sum of the two words' idfs (global
statistics when present), the field's document lengths and average length,
and the field's k1 and b. The phrase hands the pair to a `TermQuery` carrying
these statistics, so no weight rescaling enters the arithmetic and the scores
are bit-identical to the position path. One exception: like every term
frequency, a pair's saturates at 65,535 per document, while the position path
counts on. A missing pair term means no document contains the phrase.

- **Ranked:** the pair runs the term query's plans (single-term MaxScore
  with block bounds and impacts). On a reordered field a pair phrase opts
  into physical traversal only for complete streams, like a term: it is
  scored, so a ranked physical stream would have to visit every posting to
  keep stable-ID ties.
  Inside a Boolean query the phrase stays an opaque clause whose scorer is
  the pair's.
- **Exact counts:** a new per-segment `Query::word_pair_term(reader)` (the
  count-only hint `count_equivalent_term` stays unchanged, since other
  planners treat it as the query's membership everywhere) lets a
  deletion-free segment answer from the dictionary's document frequency;
  single-clause Boolean wrappers forward it.
- **Positions requested, slop, three or more words, gaps from dropped
  stopwords, fields without `common_grams`:** the existing phrase executor,
  unchanged.

### Dictionary scans

Prefix, wildcard and regex expansions must not see pair terms, including in
their scan budgets. Non-empty prefixes are valid UTF-8 and never reach them; an
empty prefix (a whole-field pattern) scans with an exclusive end key
`field 0xFF`, which caps the blocks to visit and ends the scan at the first
pair. Pattern matchers already reject invalid UTF-8, but the bound keeps pair
terms out of `MAX_SCANNED_TERMS` accounting.

### Compatibility

`INDEX_META_FORMAT_VERSION` 10 → 11 with the usual metadata-only upgrade on
open, so earlier builds, which would treat pair terms as words in unbounded
scans, refuse any new index. Segments without pairs are byte-identical. The
word list is fixed at index creation.

## Validation

`common_word_pairs_answer_two_word_phrases_like_positions` builds the same
random documents (duplicate and overlapping words, stemming variants,
two-value documents) into an index with and without `common_grams`, in
`TokenPosition` and `Full` modes, merged, with and without deletions and RGB
reordering, and requires identical exact counts and top-1/10/1,000 IDs and
score bits for every two-word phrase over the vocabulary; that a whole-field
expansion returns the same terms; and, with `query-diagnostics`, that the
paired index confirms no phrase positions. Removing the scan bound, counting
a duplicated second word twice or disabling the routing each fails it.
`common_grams_are_validated_at_schema_admission_and_by_the_tokenizer` covers
the schema and tokenizer errors.

## Measurements

1M documents of the benchmark corpus, `unicode_word`, the 128 most
document-frequent words of the first 200,000 documents
(`searchbench_http common-words`), one binary building both indexes (Apple
M-series, single-thread medians, A B A B; every count identical):

| Family                                   | TOP_10 without → with pairs | COUNT without → with pairs |
| ---------------------------------------- | --------------------------: | -------------------------: |
| high_phrase                              |       60.29 → 0.92 ms (65×) |    497.7 → 0.63 ms (~790×) |
| med_phrase                               |       56.36 → 2.43 ms (23×) |     265.3 → 39.3 ms (6.8×) |
| low_phrase                               |      30.87 → 5.84 ms (5.3×) |     121.6 → 42.1 ms (2.9×) |
| and_high_med, high_term, wildcard, regex |                         ±3% |                        ±3% |

The index grows 2.7% (postings +6.5%, positions unchanged) and the build
takes 15% longer (37.7 → 43.5 s).

## Open questions

- **Automatic word lists.** Choosing the top N words at index time needs
  document frequencies before the first document; a per-segment list would
  let segments disagree and requires merges to rebuild pairs from positions.
- **Longer phrases** of common words (`"one of the"`) could intersect pair
  lists and verify adjacency with pair positions, at the cost of storing them.
- **Sloppy phrases** are out of scope.
- The QL phrase parser tokenizes with `Purpose::Index`, which emits variants
  at repeated positions (`dsl/ql/mod.rs:760`); unrelated to pairs, but found
  while reviewing phrase construction.
