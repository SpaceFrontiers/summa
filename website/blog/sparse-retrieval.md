---
title: Sparse Retrieval — From BM25 to Learned Representations
parent: Blog
nav_order: 4
---

# Sparse Retrieval: From BM25 to Learned Representations

<i>[@PashaPodolsky](https://github.com/ppodolsky)</i>

_September 25, 2026_

<script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>

Let's return to the [inverted index](how-search-engines-work.md). We already know how to find documents containing a word. Now imagine that a model could decide which words describe a document, including useful words the author never wrote. We would get semantic expansion without throwing away the inverted index.

This leads us to learned sparse retrieval. More generally, sparse retrieval represents a query and a document using a large coordinate space in which most weights are zero. The important property is the pattern of
nonzero coordinates, not whether a neural network produced them. Classical
lexical retrieval and learned sparse retrieval can both exploit inverted
indexes, although their scores and workload distributions differ.

A lexical system can retrieve an error code exactly. A learned sparse model
can additionally give weight to related vocabulary that was absent from the
original sentence. That provides a route to semantic matching while retaining
the ability to inspect influential vocabulary coordinates. It does not make
the representation a complete explanation of the model's reasoning.

## The shared operation

We will denote the nonzero coordinates of vector $$x$$ by $$supp(x)$$. For sparse query weights $$q$$ and document weights $$d$$, a common score is the dot product:

$$score(q,d)=\sum_{t\in supp(q)\cap supp(d)}q_t d_t.$$

Only matching
nonzero coordinates contribute. Query and document representations must use
the same vocabulary mapping and compatible model revision. An integer
dimension ID has no meaning independently of that mapping.

An inverted list for coordinate t stores the documents that have a nonzero
weight there, together with their weights or encoded impacts. Searching can
then visit the lists for active query coordinates rather than examine every
coordinate of every document. A forward representation stores each document's
sparse vector and is useful for scoring a nominated candidate completely.

For a small numerical example, suppose a query gives `password` weight 2 and
`reset` weight 1. A document with corresponding weights 0.5 and 3 scores
2 × 0.5 + 1 × 3 = 4. We can keep the arithmetic visible:

```text
query:     password → 2,   reset → 1
document:  password → 0.5, reset → 3, account → 0.8
score:     2 × 0.5 + 1 × 3 = 4
```

The `account` coordinate contributes nothing unless it is also active in the query. Learned expansion changes which coordinates are active;
the index still performs matching and arithmetic on those coordinates.

## BM25 and learned expansion solve different parts of the problem

BM25 uses term frequency, document length and collection statistics. Its
saturation and length normalization produce term contributions suited to an
inverted index. It remains a useful baseline, especially when exact names and
identifiers matter. It is not simply the raw dot product of two bags of term
counts. The [search-engine introduction](how-search-engines-work.md) develops
these scoring and indexing ideas.

[SPLADE v2](https://arxiv.org/abs/2109.10086) learns sparse vocabulary weights,
including expansion terms. Its pooling, sparsity control and training objective
shape both relevance and index cost.
[SPLADE-v3](https://arxiv.org/abs/2403.06789) improves the training recipe and
reports results across many query sets. Those results describe particular
models and evaluations; they do not establish that every learned sparse model
beats every dense model or lexical baseline.

Domain-specific behavior remains an active research question. For example,
a [2026 study of learned sparse retrieval for code](https://arxiv.org/abs/2603.22008)
examines challenges beyond ordinary natural-language passage retrieval.
Vocabulary coverage, identifiers and the distribution of queries matter.
Test the actual model, not just the label “sparse.”

## Skipping Work Without Losing Answers

Naively visiting all matching postings can be expensive, especially when a
learned query activates many coordinates. MaxScore and WAND reduce work using
upper bounds on possible contributions. Block-level bounds can be tighter
than list-wide bounds because they describe a smaller range of documents.

For an additive nonnegative score, if the sum of all applicable bounds is
strictly below the current top-k threshold, the range cannot contain a winning
document. Equality needs the collector's tie-breaking rule. Bounds must remain
conservative after boosts, quantization and any score transformation; arbitrary
negative weights or a changed scorer cannot inherit a proof automatically.

So far, so good: we avoid scoring documents that cannot possibly win. These methods can preserve the exact top-k for the represented score. That
does not mean the representation itself is lossless. Removing document
coordinates or quantizing weights changes the score being searched. Pruning
query coordinates likewise changes the effective query. Measure representation
loss separately from traversal loss.

Also distinguish ranked retrieval from exact match counting. A document that
cannot enter the top-k may still need to contribute to a count or aggregation.
Score pruning alone is not permission to omit that work.

## Approximate sparse retrieval

Learned representations can produce posting distributions for which classical
exact evaluation is expensive. Approximate systems trade candidate coverage
for less work.

[Seismic](https://arxiv.org/abs/2404.18812) organizes sparse postings into
geometric clusters and uses compact summaries to guide retrieval. Such
summaries help choose promising work, but a heuristic proxy is not the same as
a certified score upper bound. A candidate's final dot product can be exact
even when the candidate selection is approximate.

This distinction gives two separate tuning questions: did the system nominate
the right documents, and did it score those nominees correctly? Increasing
the final result count cannot recover documents discarded during construction
or skipped during nomination. Compare candidate coverage with an exhaustive
reference, using the same stored vectors and effective query.

## Sparse retrieval in Summa

Summa 2 exposes three sparse backends through the same schema and query owners:
BMP, sparse MaxScore and Seismic. BMP is the default for newly declared sparse
fields; Seismic is an explicit approximate alternative. Existing serialized
metadata keeps its recorded semantics. See the
[backend and configuration contract](https://github.com/SpaceFrontiers/summa/blob/main/docs/seismic-sparse-index.md).

BMP uses block-oriented metadata and compressed sparse payloads. Its
`lsp_gamma` setting controls traversal approximation; `lsp_gamma: 0` provides
the exhaustive traversal reference described in the
[BMP documentation](https://github.com/SpaceFrontiers/summa/blob/main/docs/bmp-grid-compression.md).
That reference still uses the configured stored precision and query
preprocessing. It is not a promise to reproduce an unquantized model with
all original coordinates restored.

Seismic separates nomination data from forward vectors used for final scoring.
Its controls include retained postings, cluster size, summary energy, query
dimension cut and a summary-threshold factor. In Summa, `exhaustive: true`
bypasses nomination and scores the forward values. Setting `seismic_factor: 0`
alone does not restore postings or query dimensions omitted earlier. The
defaults are operating settings, not a guaranteed recall target.

Compatible segment merges preserve encoded representations where their format
allows it. Changes that need nomination maintenance use an explicit bounded
operation. This keeps ordinary merging distinct from silently rebuilding the
retrieval model or changing its candidate policy.

## Putting It to Work

Start with a lexical baseline and a judged query set. Measure a learned sparse
model's relevance, query-encoding cost and index size before tuning its
traversal. Record query and document nonzero counts, the lengths of commonly
visited lists, and the cost of final forward-vector reads. Sparse does not
necessarily mean small: expansion can add a large number of postings.

Then compare exact and approximate execution at matched quality. Include tail
latency, resident memory, cold reads, build and maintenance costs, filters and
multi-value aggregation. Keep query pruning, weight quantization and search
budgets fixed when attributing a speedup to an indexing algorithm.

Finally, compare with [dense retrieval](dense-retrieval.md) and a hybrid pipeline
on the same corpus. The two representations can make different mistakes.
Combining their candidates can help, but the result depends on candidate depth,
fusion, reranking and the application. A state-of-the-art result is evidence
for a defined evaluation setting, not an exemption from measuring yours.
