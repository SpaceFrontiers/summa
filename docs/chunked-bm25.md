# BM25 over equal-length chunks

Status: research note + decisions (2026-09-04). Mechanism: `docs/chunked-text-fields.md`.

Hermes splits a document's text into fixed-size chunks before building the
inverted index of a `chunked` text field. Each chunk is a scoring unit with
its own posting entry and length; the document score is the maximum over its
chunks (MaxP). This note records what the literature says about that setup
and which of it Hermes applies.

## What is applied

1. **Chunk length floor.** The BM25 length of a chunk is
   `max(len, nominal_len)`, where `nominal_len` is the 90th-percentile chunk
   length of the field in the segment — the chunk size the corpus was split
   at, computed from the `.chunks` length column when the segment opens
   (`ChunkMap::length_floor`). With equal-length chunks the normaliser
   `1 − b + b · len / avg` is a constant for every full chunk; `b` only acts on
   the short last chunk of a document, which it _rewards_ (at `tf = 1`,
   `k1 = 1.2`, `b = 0.75` a half-length tail scores ≈ 22 % higher per term).
   Nothing in the literature says a short tail is likelier to be relevant, and
   Kaszkiel & Zobel found the opposite: under plain length normalisation
   "over 95 % of the top-ranked passages were of 50 words", which they fixed
   with a minimum passage length (SIGIR 1997) and, for fixed windows, by
   dropping length normalisation altogether (JASIST 2001). The floor keeps
   `b` meaningful for the (rare) chunk longer than nominal and removes the
   tail reward; every full chunk and every tail chunk now share one
   normaliser. (Flooring at the _average_ is not enough: tails pull the
   average below the chunk size, so full chunks would be penalised relative
   to tails.) Implemented in `TermCursor` (MaxScore) and `PhraseScorer`.

2. **MaxP stays the aggregation.** Best-passage ranking is the robust choice
   across classical and neural studies: Callan (SIGIR 1994; best passage beat
   summing passages), Kaszkiel & Zobel (1997, 2001), Dai & Callan (SIGIR 2019,
   MaxP > FirstP > SumP), Zhang, Yates & Lin (ECIR 2021: "MaxP almost always
   significantly outperforms the other approaches"), Nguyen, MacAvaney & Yates
   (2023: Sum "bias towards longer documents", Mean corrects it but
   underperforms Max). Sum is length-biased and mean dilutes (Bendersky &
   Kurland 2008). Hermes' `MultiValueCombiner::Max` for chunked text is kept.

3. **`k1`, `b` unchanged** (`1.2`, `0.75`). Within a 200–300-token chunk
   `tf` is small and saturation matters little. Anserini's tuned MS MARCO
   passage run uses `k1 = 0.82`, `b = 0.68`; that is evidence that tuning can
   help a specific collection, not a portable replacement for the defaults.
   With the length floor `b` is almost inert among full chunks. A Hermes
   change therefore needs a labelled set (see "Measure").

## What is not applied (and why)

- **Document-level IDF.** Chunk-level `df` counts every chunk of a long
  document that repeats a term, lowering that term's IDF. Kaszkiel & Zobel and
  Callan used document-level statistics for passage ranking; Anserini's
  passage-indexed MS MARCO run uses passage-level IDF and matches the
  document index after tuning. No controlled comparison exists. Switching
  needs a document-level `df` per term in the term dictionary; it is planned
  for the next term-dictionary format revision, not done here.
- **Chunk-count prior.** A document with `n` chunks has `n` draws at a high
  chunk score. No paper quantifies this for BM25 MaxP; BM25 caps each term's
  contribution, so the effect is bounded (unlike log-sum-exp, which adds
  exactly `ln n / t`, the bias that `docs/…` and the combiner history record
  for the vector path). Diagnose before correcting: bin documents by chunk
  count and compare retrieved-vs-relevant rates (Singhal 1996; Lv & Zhai
  2011); only then apply `score − c · log n`.
- **Hybrid document + passage score** (`α · BM25_doc + (1 − α) · max_chunk`,
  Callan 1994 D+W: always better than either alone; Bendersky & Kurland
  2008; Sheetrit et al. 2020). Best-supported classical option, but it needs
  a document-level `tf` (sum over chunks) per candidate, i.e. a second index
  or a per-query scan of every chunk of every candidate. Deferred.

## Measure

## Primary references

- James P. Callan, [“Passage-Level Evidence in Document Retrieval”](https://www.sigmod.org/publications/dblp/db/conf/sigir/Callan94.html), SIGIR 1994.
- Marcin Kaszkiel and Justin Zobel, [“Effective Ranking with Arbitrary Passages”](<https://doi.org/10.1002/1532-2890(2000)9999:9999%3C::AID-ASI1075%3E3.0.CO;2-%23>), JASIST 2001.
- Michael Bendersky and Oren Kurland, [“Utilizing Passage-Based Language Models for Document Retrieval”](https://doi.org/10.1007/978-3-540-78646-7_17), ECIR 2008.
- Zhuyun Dai and Jamie Callan, [“Deeper Text Understanding for IR with Contextual Neural Language Modeling”](https://doi.org/10.1145/3331184.3331303), SIGIR 2019.
- Xinyu Zhang, Andrew Yates, and Jimmy Lin, [“Comparing Score Aggregation Approaches for Document Retrieval with Pretrained Transformers”](https://doi.org/10.1007/978-3-030-72240-1_11), ECIR 2021.
- Thong Nguyen, Sean MacAvaney, and Andrew Yates, [“Adapting Learned Sparse Retrieval for Long Documents”](https://doi.org/10.1145/3539618.3591943), SIGIR 2023.
- Amit Singhal, Chris Buckley, and Mandar Mitra, [“Pivoted Document Length Normalization”](https://doi.org/10.1145/243199.243206), SIGIR 1996.
- Yuanhua Lv and ChengXiang Zhai, [“Lower-Bounding Term Frequency Normalization”](https://doi.org/10.1145/2063576.2063584), CIKM 2011.
- Eilon Sheetrit, Anna Shtok, and Oren Kurland, [“A Passage-Based Approach to Learning to Rank Documents”](https://doi.org/10.1007/s10791-020-09369-x), Information Retrieval Journal 2020.
- Anserini, [MS MARCO passage reproduction settings](https://github.com/castorini/anserini/blob/master/docs/reproduce/from-document-collection/msmarco-v1-passage.docTTTTTquery.md).