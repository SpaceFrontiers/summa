# Documentation

[Quick start](../README.md#quick-start) · [Packages](../README.md#packages) ·

## Search and schema

- [Schema (SDL)](schema.md)
- [Query language](query-language.md)
- [Tokenization and phrase queries](dynamic-tokenizer-and-phrase.md)
- [Chunked text](chunked-text-fields.md) and [chunked BM25](chunked-bm25.md)
- [Formula ranking, backfill, and diagnostics](candidate-rescoring.md)
- [Passage evidence for document nominations](document-nominated-passages.md)
- [Web UI configuration](ux-config.md)

## Operations

- [Server](../hermes-server/README.md) and [broker](broker.md)
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

## Documentation checks

## Additional guides

- [Contributing](../CONTRIBUTING.md)
- [Range bitset word materialization](range-word-materialization.md)
- [Range block scan experiments](range-block-scans.md)
- [Text formats and memory](text-format-comparison.md)
- [pruning fixes](search-pruning-fixes.md)
- [Lucene research](lucene-11-performance-research.md)
- [BMP forward-search experiment (removed)](bmp-forward-search.md)
- [Former IVF-PQ design](unified-vector-quantization.md)
- [Seismic research](seismic-research.md)
- [LLM and retrieval research](llm-design-and-rag-embeddings.md)
- [RL search training research](rl-search-training.md)
- [MoE A100 results](moe-performance.md)
- [cuBLAS dispatch experiment](cublas-gemm-dispatch.md)
