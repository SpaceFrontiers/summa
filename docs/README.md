# Hermes documentation

Start with the [repository quick start](../README.md#quick-start) and
[contribution guide](../CONTRIBUTING.md). This index covers every guide in this
directory. Package READMEs below document each executable or library.

## Getting started and interfaces


## Schema, text search, and query behavior

- [Hermes Schema Definition Language (SDL)](schema.md)
- [Dynamic per-document stemming and wire-level phrase queries](dynamic-tokenizer-and-phrase.md)
- [Chunked text fields: BM25 over passages with ordinals](chunked-text-fields.md)
- [BM25 over equal-length chunks](chunked-bm25.md)
- [Lexical vertical: positions, pruning, reordering, tokenization](lexical-vertical.md)
- [Posting block codecs](posting-codecs.md)
- [Hermes Web UX Configuration DSL](ux-config.md)

## Storage, operations, and distributed search

- [Hermes broker](broker.md)
- [Segment Lifecycle and Recovery](segment-lifecycle.md)
- [Index diagnostics](diagnostics.md)
- [Prometheus Metrics](metrics.md)
- [Document store v3](document-store-v3.md)
- [Hot-Metadata Pinning (meta/data residency split)](hot-metadata-pinning.md)
- [Cold IO for merges (hot-metadata-pinning Phase 2)](cold-io.md)
- [Merge-Time BP Reordering](merge-time-reorder.md)
- [Budgeted (Partial) BP Reordering](budgeted-reorder.md)
- [Block-Level Reorder with Stats-Guided Granularity](block-level-reorder.md)

## Vector retrieval and compression

- [Streaming ScaNN index](scann-streaming-index.md)
- [FastScan layout v2 for float ScaNN leaves](fast-scan-layout-v2.md)
- [TurboQuant (TQ) — training-free dense ANN codec](turboquant-quantization.md)
- [Unified Dense IVF Architecture](unified-vector-quantization.md)
- [BMP LSP/0 and Maximum-Grid Compression](bmp-grid-compression.md)
- [Algebraic float reductions](algebraic-float-reductions.md)
- [Seismic: research assessment (2026-07-09)](seismic-research.md)

## Language models, training, and evaluation

- [LLM inference and training code map](llm-code-map.md)
- [LLM compute architecture](uni-stack-inference.md)
- [MAL parser architecture](mal-single-parser.md)
- [Tokenizer backend compatibility](tokenizer-backends.md)
- [Upstream dependency pins](upstream-dependencies.md)
- [LLM Visualization Lab](llm-visualization-lab.md)
- [Training workflows and task contracts](training-objectives-and-curricula.md)
- [Generation evaluation](generation-eval.md)
- [Large-candidate-pool retrieval evaluation](retrieval-pool-eval.md)
- [Configurable MoE design](moe-design.md)
- [SOTA LLM design (2024–2026) + shared retrieval embeddings — research notes](llm-design-and-rag-embeddings.md)
- [RL Training Pipeline for Agentic Search over Hermes](rl-search-training.md)

## Package guides

- [hermes-broker](../hermes-broker/README.md)
- [hermes-client-python](../hermes-client-python/README.md)
- [hermes-client-typescript](../hermes-client-typescript/README.md)
- [hermes-core](../hermes-core/README.md)
- [hermes-llm](../hermes-llm/README.md)
- [hermes-mal](../hermes-mal/README.md)
- [hermes-mal-python](../hermes-mal-python/README.md)
- [hermes-model-lab](../hermes-model-lab/README.md)
- [hermes-proto](../hermes-proto/README.md)
- [hermes-server](../hermes-server/README.md)
- [hermes-tokenizer](../hermes-tokenizer/README.md)
- [hermes-tool](../hermes-tool/README.md)
- [hermes-train](../hermes-train/README.md)
- [hermes-wasm](../hermes-wasm/README.md)
- [hermes-web](../hermes-web/README.md)

## Keeping documentation current

## Additional guides

- [MoE A100 performance](moe-performance.md)
- [Attention kernels](fused-attention.md)
- [Fused chunked cross-entropy (GPU)](fused-cross-entropy.md)
- [Segment-parallel selective scan](segmented-selective-scan.md)
- [BF16 residual stream (CUDA training)](bf16-residual-stream.md)
- [Native cuBLAS GEMM dispatch: proof results and verdict](cublas-gemm-dispatch.md)
- [Kernel size-generality and tuning surface](kernel-tuning-surface.md)
