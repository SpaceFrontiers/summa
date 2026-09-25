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

- [Summa 2 migration](summa-2-migration.md)
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
- [Document store v3](document-store-v3.md)
- [Owned byte views](owned-byte-views.md)
- [Text reordering](maxscore-text-reordering.md)
- [Merge-time BP](merge-time-reorder.md), [budgeted BP](budgeted-reorder.md), and [block reorder](block-level-reorder.md)

## Vector retrieval

- [TurboQuant](turboquant-quantization.md)
- [ScaNN](scann-streaming-index.md) and [FastScan layout](fast-scan-layout-v2.md)
- [Single-copy binary storage](binary-vector-storage.md)
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
