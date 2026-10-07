# Documentation

[Quick start](../README.md#quick-start) · [Packages](../README.md#packages) ·
[Contributing](../CONTRIBUTING.md) · [Benchmark commands](benchmarks.md)

## Search and schema

- [Schema (SDL)](schema.md)
- [Query language](query-language.md)
- [Wildcard queries and expansion limits](wildcard-query.md)
- [Regex queries and escaped literals](regex-query.md)
- [Tokenization and phrase queries](dynamic-tokenizer-and-phrase.md)
- [Chunked text](chunked-text-fields.md) and [chunked BM25](chunked-bm25.md)
- [Formula ranking, backfill, and diagnostics](candidate-rescoring.md)
- [Passage evidence for document nominations](document-nominated-passages.md)
- [Web UI configuration](ux-config.md)

## Operations

- [Publishing packages](publishing.md)
- [Summa 2 migration](summa-2-migration.md)
- [Index compatibility (2.1)](compatibility.md)
- [Server](../summa-server/README.md) and [broker](broker.md)
- [Segment lifecycle and recovery](segment-lifecycle.md)
- [Deletion, upserts, and compaction](row-deletion.md)
- [Content-hash deduplication](content-deduplication.md)
- [Index diagnostics](diagnostics.md), [query work counters](query-work-diagnostics.md), and [metrics](metrics.md)
- [Metadata residency](hot-metadata-pinning.md) and [cold I/O](cold-io.md)

## Search architecture

- [Engineering contract and validation matrix](search-system-contract.md)
- [Lexical execution and formats](lexical-vertical.md)
- [Posting codecs](posting-codecs.md) and [block execution](search-block-execution.md)
- [Compact text and norms](compact-text-format.md)
- [Bitmap posting blocks](bitmap-posting-blocks.md) (default `RoundedBitmap`)
- [Document store v3](document-store-v3.md) and [batched payload reads](batched-payload-reads.md)
- [Payload service CPU, lifetime and buffer measurements](benchmark-results/payload-service-2026-09-26/README.md)
- [Hybrid metadata pinning and sparse I/O measurements](benchmark-results/hybrid-sparse-io-2026-09-26/README.md)
- [Concurrent sparse I/O and bounded cursor windows](benchmark-results/sparse-concurrency-2026-09-26/README.md)
- [Ring workers, queue wakeups and pool parity experiments](benchmark-results/ring-workers-2026-09-26/README.md)
- [Hardened single-ring wakeups and lifecycle validation](benchmark-results/ring-wakeup-2026-09-26/README.md)
- [Fixed-arrival sparse I/O and overload measurements](benchmark-results/sustained-io-2026-09-26/README.md)
- [Warm MaxScore execution and ring notification comparison](benchmark-results/warm-io-2026-09-26/README.md)
- [Dictionary decoding and the direct Luxir comparison](benchmark-results/dictionary-decoding-2026-09-26/README.md)
- [Dictionary scans, term cache and FST feasibility](benchmark-results/dict-scan-2026-09-27/README.md) and the [FST term dictionary note](fst-term-dictionary.md)
- [Frequent-term counts, regex term checks and bitmap posting blocks](benchmark-results/bitmap-blocks-2026-09-27/README.md)
- [Ranked conjunctions, disjunction windows and bitmap blocks by default](benchmark-results/default-2026-09-28/README.md)
- [Dictionary entry offsets, rarer-block skips and per-candidate bounds; the JCC erratum](benchmark-results/next-2026-09-29/README.md)
- [RGB-reordered body field against Luxir](benchmark-results/rgb-2026-09-29/README.md)
- [Common word pairs against Luxir](benchmark-results/pairs-2026-09-29/README.md)
- [Physical tie order against Luxir](benchmark-results/ties-2026-09-30/README.md)
- [Exact pattern counts](benchmark-results/pattern-count-2026-10-02/README.md)
- [Full search and I/O review](benchmark-results/review-2026-10-02/README.md)
- [Process-wide term dictionary cache](term-dictionary-cache.md)
- [Suffix filters for cached dictionary blocks](dictionary-suffix-filters.md)
- [Entry offsets for cached dictionary blocks](dictionary-entry-offsets.md)
- [Common word pairs for exact phrases](common-word-pairs.md)
- [Physical tie order for constant-score queries](physical-tie-order.md)
- [Owned byte views](owned-byte-views.md)
- [Text reordering](maxscore-text-reordering.md)
- [Merge-time BP](merge-time-reorder.md), [budgeted BP](budgeted-reorder.md), and [block reorder](block-level-reorder.md)

## Vector retrieval

- [TurboQuant](turboquant-quantization.md)
- [ScaNN](scann-streaming-index.md) and [FastScan layout](fast-scan-layout-v2.md)
- [Single-copy binary storage](binary-vector-storage.md)
- [Binary IVF geometry](binary-ivf-geometry.md)
- [Binary IVF prefix-first scan](binary-prefix-scan.md)
- [Binary IVF SOAR spill](binary-ivf-soar.md)
- [BMP grid compression](bmp-grid-compression.md) and [forward storage](bmp-forward-index.md)
- [Seismic](seismic-sparse-index.md), [compact summaries](seismic-compact-summaries.md), and [forward compression](seismic-forward-compression.md)
- [Algebraic float reductions](algebraic-float-reductions.md)

## Models and training

- [Code map](llm-code-map.md) and [compute architecture](uni-stack-inference.md)
- [MAL parser](mal-single-parser.md)
- [Tokenizer compatibility](tokenizer-backends.md)
- [Dependency pins](upstream-dependencies.md)
- [Model Lab](llm-visualization-lab.md)
- [Training workflows and task contracts](training-objectives-and-curricula.md)
- [Generation evaluation](generation-eval.md) and [retrieval-pool evaluation](retrieval-pool-eval.md)
- [MoE design](moe-design.md)
- [Attention](fused-attention.md), [cross-entropy](fused-cross-entropy.md), and [selective scan](segmented-selective-scan.md)
- [BF16 residual stream](bf16-residual-stream.md)
- [Kernel tuning](kernel-tuning-surface.md)

## Reviews, measurements, and research

These retain dated experiments, rejected approaches, and proposals. Read each
status and fixture before treating a result as current behavior or performance.

- [Full-text comparison](search-benchmark-current.md)
- [Search review ledger](search-performance-review.md)
- [BMP and quantized ANN review](bmp-ann-optimization-review.md) and [complete measurements](benchmark-results/bmp-ann-2026-10-04/README.md)
- [Yonik Searchbench comparison](searchbench-comparison.md)
- [Collector selection benchmark](collector-benchmark.md)
- [IResearch optimization and Linux I/O audit](iresearch-optimization-audit.md)
- [Topic-aware placement and redistribution proposal](topic-aware-placement.md)
- [Range bitset word materialization](range-word-materialization.md): source comparison and measured filter-construction gains.
- [Range block scan experiments](range-block-scans.md): header-based pruning and sequential decoding.
- [Merge review](search-merge-review.md), [reorder review](reordering-performance-review.md), and [Rust hot paths](rust-hot-path-review.md)
- [Search Benchmark Game](search-benchmark-game.md) and [Wikipedia results](search-benchmark-results.md)
- [Score bounds](search-benchmark-ratio-results.md), [bulk scoring](search-benchmark-bulk-results.md), and [dictionary experiments](search-benchmark-dictionary-results.md)
- [Posting validation cache experiment (superseded)](search-benchmark-validation-results.md)
- [Text formats and memory](text-format-comparison.md)
- [Query-work diagnosis](search-work-diagnosis.md) and [pruning fixes](search-pruning-fixes.md)
- [Ranked conjunction and phrase follow-up](ranked-pruning-followup.md)
- [Score-guided posting traversal research](score-guided-traversal.md)
- [RGB diagnosis](search-rgb-benchmark.md) and [repair results](search-rgb-repair.md)
- [Lucene research](lucene-11-performance-research.md)
- [BMP forward-search experiment (removed)](bmp-forward-search.md)
- [Former IVF-PQ design](unified-vector-quantization.md)
- [Seismic research](seismic-research.md)
- [L1 scoring handoff](handoffs/2026-09-05-l1-candidate-scoring.md)
- [LLM and retrieval research](llm-design-and-rag-embeddings.md)
- [RL search training research](rl-search-training.md)
- [MoE A100 results](moe-performance.md) and [cuBLAS dispatch experiment](cublas-gemm-dispatch.md)
- [Raw benchmark evidence](benchmark-results/README.md)

## Documentation checks

Run `uv run scripts/check_docs.py` to validate local links, anchors, guide
coverage, and benchmark targets. Keep current contracts separate from dated
results; follow the [reporting rules](benchmarks.md#recorded-results-and-reporting).

- [Latest performance table for posts](search-performance-post.md)
