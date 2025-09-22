---
title: Sparse Retrieval — From BM25 to Learned Representations
parent: Blog
nav_order: 4
---

# Sparse Retrieval: From BM25 to Learned Representations

<i>[@PashaPodolsky](https://github.com/ppodolsky)</i>

_September 25, 2026_

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

### How a language model produces an inverted list

In the max-pooled SPLADE formulation, the model produces a vocabulary logit
$$z_{it}$$ for each input position i and vocabulary coordinate t. A sequence
weight is

$$w_t=\max_i\log(1+\max(0,z_{it})).$$

Negative logits become zero, the logarithm dampens large activations, and max
pooling retains the strongest signal across positions. A coordinate can become
active even when its token was absent from the input: that is learned expansion.
The ranking objective is combined with separate query and document sparsity
penalties. One FLOPS penalty uses batch-average activations:

$$R=\sum_t\left(\frac{1}{B}\sum_{b=1}^{B}w_t(x_b)\right)^2.$$

Unlike merely counting nonzeros, it penalizes repeatedly activating the same
coordinates across examples. This connects training to posting-list workload;
it is a differentiable surrogate, not a measurement of CPU instructions or
p99 latency. See [SPLADE v2, equations 4–6](https://arxiv.org/pdf/2109.10086).

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

### A block bound we can calculate by hand

Suppose a block B contains two documents with weights `(password=3, reset=0)`
and `(password=0, reset=4)`. Our query still has weights `(2, 1)`. The document
scores are 6 and 4, but the coordinate-wise block maxima give

$$U(B,q)=\sum_{t\in supp(q)}q_t\max_{d\in B}d_t=2\cdot3+1\cdot4=10.$$

The bound is safe for nonnegative query weights, but loose: no document
achieves both maxima. If the current heap threshold is 7, this bound forces us
to inspect a block whose documents cannot win. Smaller or more coherent blocks
can tighten bounds, at the cost of more metadata and bound computations.

[Block-Max Pruning (BMP)](https://arxiv.org/abs/2405.01117) builds bounds over
aligned document ranges and processes promising blocks first. A high threshold
found early makes later blocks easier to skip. With conservative bounds, we
can stop when the largest remaining bound is below the threshold. Restricting
the number of processed blocks or making the threshold more aggressive changes
that exactness argument.

```text
bounds = block_bounds(query)
for block in descending_bound_order(bounds):
    if heap_is_full and bound[block] < kth_score:
        break
    score_block_and_update_heap(block, query)
```

This is conceptual pseudocode. Real implementations also handle live-document
masks, ties, bounds rounded conservatively during quantization, and the cost of
ordering blocks. It is not a license to sort or materialize unlimited metadata.

## Approximate sparse retrieval

Learned representations can produce posting distributions for which classical
exact evaluation is expensive. Approximate systems trade candidate coverage
for less work.

[Seismic](https://arxiv.org/abs/2404.18812) organizes sparse postings into
geometric clusters and uses compact summaries to guide retrieval. Such
summaries help choose promising work, but a heuristic proxy is not the same as
a certified score upper bound. A candidate's final dot product can be exact
even when the candidate selection is approximate.

Let's unpack the approximation. Seismic retains high-impact postings, groups
nearby sparse vectors within each list, and attaches summaries to the groups.
For nonnegative weights, the full coordinate-wise maximum vector is an upper
bound: its dot product with the query dominates every member's score. Trimming
that summary to save space can discard a query-relevant coordinate and removes
this general guarantee. Static list pruning and visiting only selected query
lists introduce additional ways to miss a document.

Once a block is selected, forward vectors supply complete candidate scores.
The original [Seismic paper](https://arxiv.org/abs/2404.18812) is useful precisely
because it separates this nomination structure from final scoring. Exact
arithmetic at the end does not turn approximate nomination into exact search.

This distinction gives two separate tuning questions: did the system nominate
the right documents, and did it score those nominees correctly? Increasing
the final result count cannot recover documents discarded during construction
or skipped during nomination. Compare candidate coverage with an exhaustive
reference, using the same stored vectors and effective query.

### What newer work changes

[SeismicWave](https://doi.org/10.1145/3627673.3679977) orders blocks in the first
visited list by summary score so strong candidates raise the heap threshold
earlier. It also expands the retrieved set through a precomputed document kNN
graph before rescoring. That creates a new way to recover missed candidates,
but adds graph storage, construction and maintenance costs. Its reported
latencies exclude query encoding, a relevant detail when comparing complete
applications.

The 2026 preprint [Efficiency Optimizations for Superblock-based Sparse Retrieval](https://doi.org/10.48550/arxiv.2602.02883)
instead studies another level of pruning above document blocks. Its lightweight
scheme removes average-score bookkeeping and guarantees visitation of a fixed
number of promising superblocks while using approximate thresholds. Guaranteed
visitation is not guaranteed exact top-k. The paper's comparisons also change
with retrieval depth: performance at k=1000 should not be transferred to k=10.

Finally, [Forward Index Compression for Learned Sparse Retrieval (2026)](https://doi.org/10.48550/arxiv.2602.05445)
asks what happens after nomination. Its DotVByte codec specializes integer
component decoding and fuses it with inner-product computation, avoiding an
intermediate decoded buffer. Lossless dimension-ID compression and lower
precision weights are different choices: the latter can change nearest-neighbor
rankings. This is why a smaller index should be evaluated for score fidelity,
decoding cost and memory traffic together. These are research results, not
claims that every technique is present in Summa.

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

## Papers to Read Next

- [Formal et al., SPLADE v2 (2021)](https://arxiv.org/abs/2109.10086): vocabulary expansion, pooling and sparsity control.
- [Lassance et al., SPLADE-v3 (2024)](https://arxiv.org/abs/2403.06789): training and distillation.
- [Mallia, Suel and Tonellotto, Block-Max Pruning (2024)](https://arxiv.org/abs/2405.01117): block bounds and execution order.
- [Bruch et al., Seismic (2024)](https://arxiv.org/abs/2404.18812) and [SeismicWave (2024)](https://doi.org/10.1145/3627673.3679977): geometric nomination and graph expansion.
- [Efficiency Optimizations for Superblock-based Sparse Retrieval (2026)](https://doi.org/10.48550/arxiv.2602.02883): two-level pruning and retrieval-depth trade-offs.
- [Bruch et al., Forward Index Compression (2026)](https://doi.org/10.48550/arxiv.2602.05445): the storage and scoring costs after nomination.

Read each result with its embedding model, corpus, candidate depth and memory
budget. These choices determine which bottleneck the proposed method removes.
