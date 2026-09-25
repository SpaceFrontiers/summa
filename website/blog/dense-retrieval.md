---
title: Dense Retrieval — Embeddings, Candidates, and Exact Scores
parent: Blog
nav_order: 3
---

# Dense Retrieval: Embeddings, Candidates, and Exact Scores

<i>[@PashaPodolsky](https://github.com/ppodolsky)</i>

_September 25, 2026_

Imagine that we have a collection of help pages and a user types “how to recover an account.” The useful page is called “Resetting your password.” We would like to find it without first writing a dictionary of every possible paraphrase.

Let's make the computer learn this correspondence. In dense retrieval, an encoder maps each query and document into vectors,
and a search engine finds documents whose vectors receive high similarity
scores. The model defines which relationships are useful; the index makes
searching those representations affordable.

There is a small trap here. Finding nearby vectors and finding useful documents are two different problems. We can build a perfectly accurate index for a poor embedding and return the wrong answers very quickly. We can also choose a good model and lose its best answers because the index never visits them. Let's keep these two sources of error separate as we build the system.

## What the vectors mean

In a single-vector retriever, a document or passage has a vector with a fixed
number of coordinates. The query encoder produces a compatible vector. The
encoders may share weights, or use different query and document instructions;
they must belong to the same trained retrieval system. A generic language
model's hidden state is not automatically a good search embedding.

The simplest useful score is the inner product. For query vector $$q$$ and document vector $$d$$ with $$D$$ coordinates, it is

$$score(q,d)=\sum_{i=1}^{D}q_i d_i.$$

Other common choices are cosine similarity and squared Euclidean distance. For unit-length vectors, maximizing inner product also
maximizes cosine similarity and minimizes squared Euclidean distance. Without
normalization, these objectives can rank documents differently. Changing the
metric or normalization after indexing is therefore a model compatibility
change, not merely a tuning knob.

Chunking is part of this contract too. A long document compressed into one
vector may lose a small but important passage. Passage vectors improve the
opportunity to retrieve that passage, while increasing index size and requiring
an explicit rule for combining passage hits into document results. Training,
tokenization, truncation and query instructions all need to be versioned with
the embeddings.

## Teaching the geometry what matters

Let's look at the training step. Suppose we have a query, a relevant passage
$$d^+$$, and a set of negative passages $$\mathcal N$$. One common contrastive
objective is

$$\mathcal L=-\log\frac{\exp(s(q,d^+)/T)}{\exp(s(q,d^+)/T)+\sum_{d^-\in\mathcal N}\exp(s(q,d^-)/T)}.$$

Here $$T>0$$ is a temperature; setting it to one gives the unscaled softmax.
The loss asks the positive passage to win against the chosen alternatives.
[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) made this dual-encoder
approach practical for open-domain question answering, including the use of
other examples in a training batch as negatives. Passage encoding can happen
offline because the passage encoder does not need to see the query.

What should count as a negative? A random page about gardening is an easy
alternative to a password-reset guide. A page about resetting a two-factor
authenticator is harder and may teach a useful distinction. But an unlabelled
page can also answer the question. [RocketQA](https://doi.org/10.48550/arxiv.2010.08191)
addresses this training problem with cross-batch negatives, denoised hard
negatives and data augmentation. Harder examples are not automatically better
labels. The model learns the distinctions represented by its training data;
changing the index cannot teach a missing distinction afterward.

## Begin with an exact reference

A flat search scores every stored vector. For N vectors of dimension D, the
scoring work is proportional to N × D, plus result selection. This can be a
reasonable production choice for a small collection or a highly selective
filter. It is also the reference for evaluating an approximate nearest-neighbor
index, usually abbreviated ANN.

Index recall@k measures how many of the exact vector top-k an ANN search
recovers. It does not measure relevance to a person. Evaluate relevance
separately, with judgments and metrics such as nDCG or MRR. The
[BEIR benchmark](https://arxiv.org/abs/2104.08663) illustrates why testing across
domains matters; a result on one collection is not a universal model ranking.

## How to Avoid Scoring Everything

Several major families make different exchanges between memory, search work
and recall:

| Method                     | How it finds candidates                                 | Costs to measure                          |
| -------------------------- | ------------------------------------------------------- | ----------------------------------------- |
| Flat scan                  | Scores every vector                                     | Bytes read, dimensions, batch size        |
| IVF                        | Routes to selected centroid partitions                  | Training, partitions probed, skewed lists |
| HNSW                       | Traverses a hierarchy of proximity graphs               | Graph memory, exploration budget, updates |
| Disk-oriented graph search | Traverses candidates with SSD-aware storage and caching | I/O latency, cache residency, concurrency |

[HNSW](https://arxiv.org/abs/1603.09320) uses a graph hierarchy to navigate the
space. [DiskANN](https://www.microsoft.com/en-us/research/?p=634449) demonstrates
that high-recall graph search can use SSD storage rather than keeping the
entire index in RAM. We now have several ways of avoiding a full scan. Their names do not tell us which one will be fastest on our data, and not all of them are implemented in Summa.

IVF exposes a particularly clear trade-off. Probing more partitions normally
improves coverage but reads and scores more candidates. Balanced partitions
help; average list size alone hides the long lists that can dominate tail
latency. A selective filter can also change which execution strategy is best.
Retrieving an unfiltered top-k and discarding forbidden rows afterward can
leave too few eligible results.

### Following the graph, or choosing a partition

In HNSW, a search starts in sparse upper graph layers and descends toward a
more detailed neighborhood. At the bottom layer, it maintains a set of
promising candidates and explores their neighbors. A wider exploration set
usually improves recall but requires more distance computations and reads.
Graph connectivity and construction effort affect which routes exist in the
first place. Raising the query budget cannot create an absent edge.
The [HNSW paper](https://arxiv.org/abs/1603.09320) describes the hierarchy and
neighbor-selection heuristic; it does not make every query logarithmic under
arbitrary data, filters and update patterns.

For IVF, let the corpus contain N vectors divided into C lists. Under the
simplifying assumption of balanced lists, probing P lists examines about
$$NP/C$$ vectors, in addition to routing work. This is a cost estimate, not a
recall guarantee: the true neighbors may fall across a partition boundary.
A long list or a filter selecting only a few documents can invalidate the
average-case intuition. Measure the actual visited candidates and eligible
results, not just the number of partitions.

A useful back-of-the-envelope memory calculation is just as concrete. One
million 768-dimensional float32 vectors occupy 3.072 GB before IDs, graph
edges or metadata. Four bits per coordinate would occupy 384 MB for the codes
alone. If exact reranking retains the original float32 values, those 3.072 GB
still exist somewhere. Moving them to SSD changes residency and read latency;
it does not remove them from the storage bill.

## Compression is a second source of approximation

Quantization replaces full-precision vectors with smaller codes. Those codes
reduce memory traffic, but their scores are estimates. A practical pipeline
often retrieves more than k candidates using compressed scores, then scores
those candidates with the retained original representation.

We can write the whole query path in a few lines:

```text
q = encode_query(text)
candidates = approximate_search(q, candidate_budget)
for d in candidates:
    score[d] = similarity(q, retained_vector[d])
return top_k(score)
```

The innocent-looking `candidate_budget` is doing real work: it determines how many documents get a chance at the final comparison. This final scoring stage can correct the ordering of candidates it received.
It cannot recover a document omitted during partition routing or candidate
selection. A larger reranking budget and a larger routing budget address
different losses.

[ScaNN](https://proceedings.mlr.press/v119/guo20h.html) combines candidate search
with compression designed for maximum inner-product search. Its anisotropic
quantization objective treats errors differently according to their effect on
important inner products. This is more specific than simply minimizing vector
reconstruction error. If a stored vector $$x$$ becomes $$\hat x$$, its score
error is $$q^T(x-\hat x)$$. A small Euclidean reconstruction error is useful,
but it treats every error direction alike. ScaNN's objective decomposes the
residual into components parallel and perpendicular to $$x$$ and weights them
differently to protect the high inner products that matter for retrieval.
The practical consequence is that codec quality should be judged by recovered
neighbors and score errors on queries, not only by reconstruction loss.

[TurboQuant, published at ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html),
is a newer example of data-oblivious quantization. It uses randomized rotation,
scalar quantization and a residual correction for inner-product estimation.
Training-free quantization does not make an entire search index training-free:
an IVF routing layer can still require learned centroids. Nor does compression
turn a full scan into sublinear search. These distinctions matter more than
the algorithm's name when estimating operating costs.

## Dense retrieval in Summa

Summa 2 supports flat dense search, compressed `tq` scans, trained `ivf_tq`
routing and `scann`. Its TQ implementation uses a fixed four-bit code per
padded coordinate plus per-vector metadata; it is not a claim to reproduce
every bit width or experimental result in the paper. `tq` scans the compressed
codes, while `ivf_tq` adds partition selection. See the
[implementation and compatibility contract](https://github.com/SpaceFrontiers/summa/blob/main/docs/turboquant-quantization.md).

Trained routing and codebooks belong to an immutable index generation shared
by its segments. Ordinary commits encode against that generation, and compatible
merges copy encoded runs rather than retraining the model. Incompatible
generations are rejected. The
[ScaNN design](https://github.com/SpaceFrontiers/summa/blob/main/docs/scann-streaming-index.md)
documents this lifecycle, its memory budgets and its separate binary-vector
path. Binary Hamming search should not be confused with float inner-product
quantization.

## Beyond one vector per passage

Late-interaction retrievers retain multiple token-level vectors and aggregate
their interactions at query time. For example,
[ColBERTv2](https://arxiv.org/abs/2112.01488) combines late interaction with
compression and improved training. This represents another quality, storage
and compute trade-off; it is not interchangeable with a single-vector ANN
index. Support for multiple values in a field alone does not establish that an
engine implements a complete ColBERT retrieval pipeline.

A simplified late-interaction score makes the distinction explicit:

$$s(q,d)=\sum_{i=1}^{|q|}\max_{j=1}^{|d|}q_i^T d_j.$$

Each query token selects its best document-token match, then the matches are
summed. Averaging all document tokens into one vector generally cannot
preserve those query-dependent maxima. ColBERTv2 combines this interaction
with residual compression and improved supervision. Candidate generation and
the final MaxSim calculation remain separate costs; a multi-vector field
needs the correct aggregation semantics as well as storage for repeated vectors.

Dense and [sparse retrieval](sparse-retrieval.md) can also contribute different
candidates to a hybrid system. Rank-based fusion avoids directly comparing
unrelated raw score scales, but fusion depth, deduplication and subsequent
reranking still need evaluation. Neither fusion nor a reranker can rescue a
relevant document that every first-stage retriever missed.

## Making the Numbers Mean Something

Keep the corpus, embedding revision, metric, filters and hardware fixed when
comparing indexes. Measure index recall against exact search and relevance
against judgments. Include embedding latency, p50/p95/p99 query latency,
throughput under concurrency, build time, index bytes, resident memory and
bytes read. Report warm-cache and cold-cache behavior separately.

Repeat those measurements after realistic updates, deletions and merges.
Treat a published result as evidence for its measured setting. The useful
choice is the configuration that meets your quality and operating budget on
your data, rather than whichever system most recently described itself as
state of the art.

## Papers to Read Next

- [Karpukhin et al., Dense Passage Retrieval (2020)](https://arxiv.org/abs/2004.04906): the dual encoder and its training examples.
- [Qu et al., RocketQA (2021)](https://doi.org/10.48550/arxiv.2010.08191): negative sampling and the problem of unlabelled positives.
- [Malkov and Yashunin, HNSW](https://arxiv.org/abs/1603.09320): graph construction and search.
- [Guo et al., Anisotropic Vector Quantization (2020)](https://proceedings.mlr.press/v119/guo20h.html): score-aware compression behind ScaNN.
- [Santhanam et al., ColBERTv2 (2022)](https://arxiv.org/abs/2112.01488): multi-vector interaction and residual compression.

These papers answer different questions. Use the training papers to understand
representation quality, the index papers to understand candidate coverage,
and end-to-end experiments to decide whether the complete system helps users.
