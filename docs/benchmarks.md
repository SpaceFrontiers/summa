# Benchmark guide

Run commands from the repository root unless stated otherwise. Use the pinned
Rust toolchain and `Cargo.lock`. Choose a target explicitly: `cargo bench
--workspace` includes CUDA-only executables and data-dependent workloads.

## Target inventory

Criterion targets accept a name filter after `--` and write statistical reports
under `target/criterion/` (or `$CARGO_TARGET_DIR/criterion/`). Standalone targets
have their own CLI or environment variables and print reports to stdout.

| Package       | Target                | Harness                | Measures                                                                           |
| ------------- | --------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| `summa-core`  | `compression`         | Criterion              | Zstd encode/decode across sizes and levels                                         |
| `summa-core`  | `indexing`            | Criterion              | Text ingestion, commit, and document storage                                       |
| `summa-core`  | `posting_compression` | Criterion              | Standalone posting codecs, distributions, seek, decode, and summary                |
| `summa-core`  | `vector_indexing`     | Criterion              | Coarse training, TQ block scoring, and IVF-TQ query plans                          |
| `summa-core`  | `binary_vectors`      | Criterion              | Hamming kernels and binary coarse routing/build assignment                         |
| `summa-core`  | `scann_vectors`       | Criterion              | Persisted float/binary ScaNN routing and FastScan scoring                          |
| `summa-core`  | `dense_ann`           | Criterion              | Float routing, on-disk IVF-TQ search, and exact scoring kernels                    |
| `summa-core`  | `bmp_vs_maxscore`     | Criterion              | Synthetic sparse retrieval, recall, latency, and build cost                        |
| `summa-core`  | `bmp_hot_path`        | Criterion              | Wide-query BMP executor hot paths                                                  |
| `summa-core`  | `bmp_reorder`         | Standalone             | Sparse latency/quality before and after reorder                                    |
| `summa-core`  | `bmp_payload_layout`  | Criterion              | Sparse payload layouts under diffuse/clustered locality                            |
| `summa-core`  | `core_structures`     | Criterion              | Production posting containers, collectors, fast fields, and directories            |
| `summa-core`  | `collector_selection` | Criterion              | Heap, bounded partial selection, and loser-tree collection with threshold feedback |
| `summa-core`  | `search_pipeline`     | Criterion              | Multi-segment text/vector/fusion search plumbing                                   |
| `summa-core`  | `rust_hot_paths`      | Criterion              | Range materialization, closures, and code-generation probes                        |
| `summa-core`  | `segment_merge`       | Criterion              | RAM segment merges, deleted-row compaction, and PK deletion commits                |
| `summa-core`  | `summa_benchmark`     | Standalone             | Dataset-driven dense MRL/nprobe, sparse, and single-term BM25 evaluation           |
| `summa-llm`   | `moe_layer`           | Standalone; Linux CUDA | MoE forward/backward with and without router losses                                |
| `summa-llm`   | `moe_primitives`      | Standalone; Linux CUDA | Routing, packing, and expert-kernel costs                                          |
| `summa-llm`   | `memory_reserve`      | Standalone             | Paired static/dormant and active-slot memory overhead                              |
| `summa-train` | `wake_tier_step`      | Standalone             | Complete wake step at due/non-due memory-tier clocks                               |

Sources: [core benches](../summa-core/benches/),
[LLM benches](../summa-llm/benches/), and [training benches](../summa-train/benches/).
`uv run scripts/check_docs.py` checks this inventory against the Cargo manifests.

## CPU microbenchmarks

The standalone example `bm25_execution_benchmark` builds 20,000 documents with
four positioned text chunks each, then measures warm top-40 match, phrase and
phrase-bonus searches. It prints ordered doc/score/ordinal bit signatures and
ten samples after warmup. The deliberately high term frequencies stress phrase
intersection; it is not a production-query or recall benchmark.

```bash
cargo build --locked --release -p summa-core --example bm25_execution_benchmark
# macOS: process peak RSS includes fixture construction, not only query scratch.
/usr/bin/time -l target/release/examples/bm25_execution_benchmark
```

```bash
# Compile all core benchmark targets without running their workloads.
cargo bench --locked -p summa-core --no-run

# Production posting containers and ScaNN FastScan kernels.
cargo bench --locked -p summa-core --bench core_structures -- block_postings
cargo bench --locked -p summa-core --bench scann_vectors -- scann_fast_scan/score_block

# A short exploratory run; use longer repeated runs for published evidence.
cargo bench --locked -p summa-core --bench vector_indexing -- \
  ivf_ --warm-up-time 1 --measurement-time 3 --noplot
```

To compare revisions, run the same Criterion command on both builds using
`--save-baseline before` and then `--baseline before`. Keep the same target
output directory, toolchain, features, CPU settings, and input seeds. Preserve
both commands and raw reports with the comparison.

Sparse workload controls include `BMP_BENCH_DOCS` (`bmp_vs_maxscore`),
`BMP_BENCH_WIDE_DOCS` (`bmp_hot_path`, default 200,000), and `BMP_REORDER_DOCS`
(`bmp_reorder`, default 100,000). Payload-layout experiments accept
`BMP_LAYOUT_BLOCKS`, `BMP_LAYOUT_TERMS_PER_DOC`, and `BMP_LAYOUT_TOPIC_VOCAB`.
These are synthetic workload measurements; they do not establish production
recall or throughput on a real SPLADE corpus.

Current BMP/Seismic corpus comparisons include initial build, compatible merge,
bounded maintenance, warm search, recall, live disk bytes and peak RSS. See
[Seismic sparse indexing](seismic-sparse-index.md). Report BMP UInt8 versus
Seismic Float32 precision alongside recall; keep generated fixtures and logs
in ignored workspace evidence.

## Dense ANN quality and latency

The ignored release-mode tests in [tq_bench.rs](../summa-core/src/index/tests/tq_bench.rs)
compare flat exact search, TQ, IVF-TQ, and selective SOAR using synthetic
clustered vectors. They resolve stored corpus IDs after merging and report
recall, p50/p95, sequential query throughput, vector bytes, and build/train time.

```bash
cargo test --locked --release -p summa-core tq_dense_ann_benchmark -- \
  --ignored --nocapture
cargo test --locked --release -p summa-core ivf_tq_selective_soar_benchmark -- \
  --ignored --nocapture
```

The TQ default is 100,000 documents, 768 dimensions, and 100 queries, with
1,024 IVF clusters and 64 probes. Scale with `TQ_BENCH_DOCS`,
`TQ_BENCH_CLUSTERS`, `TQ_BENCH_QUERIES`, `TQ_BENCH_IVF_CLUSTERS`,
`TQ_BENCH_NPROBE`, and `TQ_BENCH_RERANK`. The SOAR test has a smaller default
and equivalent `SOAR_BENCH_*` controls. Record the reported ANN kind: a corpus
below the training floor can remain flat.

## Dataset-driven retrieval

The [unified harness](../summa-core/benches/summa_benchmark/main.rs) is a
Cargo **bench target**, not an installable binary. Its required inputs are:

- `dense_embeddings.bin` and `dense_queries.bin`: little-endian `u32` row count,
  `u32` dimension, followed by row-major `f32` vectors;
- `ground_truth_dense_full.bin`: little-endian `u32` query count, `u32` neighbor
  count, followed by ranked `u32` corpus row IDs for full-dimensional exact search.

Optional sparse, text, and qrels files add other evaluation rows. Their corpus
and query ordering must match the dense files; count validation cannot detect
a permutation. The [loaders](../summa-core/benches/summa_benchmark/data.rs)
and [generator](../summa-core/benches/generate_benchmark_data.py) define their formats.

The generator downloads datasets/models and uses a configured Triton Jina-v3
embedding service. Set `TRITON_URL` and `TRITON_API_KEY` in the environment.
Install its dependencies in a dedicated environment first:

```bash
uv venv .context/benchmark-venv
uv pip install --python .context/benchmark-venv/bin/python \
  'tritonclient[grpc]' transformers datasets numpy tqdm torch
.context/benchmark-venv/bin/python summa-core/benches/generate_benchmark_data.py \
  --use-beir --num-docs 100000 --num-queries 1000 \
  --output-dir "$PWD/.context/benchmark-data"

BENCHMARK_DATA="$PWD/.context/benchmark-data" DENSE_DIM=256 NUM_QUERIES=1000 \
  cargo bench --locked -p summa-core --bench summa_benchmark
```

Pin and record the Python packages, dataset revision, model revisions, service
configuration, and generated file checksums for a published run. The generator's
`--dense-dim` controls an additional truncated ground-truth file; this harness
uses `ground_truth_dense_full.bin`. `generate_embeddings.py` creates a separate
synthetic embedding fixture and is not a substitute for this dataset generator.

`BENCHMARK_DATA` defaults to `summa-core/benches/benchmark_data`, anchored to
the crate directory. `DENSE_DIM` defaults to 256 (clamped to the input dimension)
and `NUM_QUERIES` defaults to all queries; the latter applies to all modalities.
Missing or unreadable required data makes the harness fail.

Quality reports resolve stored corpus row IDs outside the timed search window.
Recall@10 uses only the first ten exact neighbors, and MRR@10 excludes hits
below rank ten. Earlier versions used segment-local IDs, counted any neighbor
in the ground-truth file, and computed unrestricted MRR under an MRR@10 label;
rerun those experiments before comparing their quality numbers. These fixes
apply to the unified harness, not the separate TQ benchmark above.

This harness reports mean single-pass query latency without an explicit warmup.
Its BM25 row uses only the first query word and is labeled accordingly. Qrels
are binary relevance over the sampled corpus, not a full official MS MARCO
leaderboard evaluation. Use the dedicated ANN tests for warmed latency
percentiles, and a full-query held-out protocol for production relevance claims.

## GPU and memory benchmarks

The two MoE executables require Linux CUDA and exit unsuccessfully on CPU or
Metal. Use `training-fusion` for the published MoE training configuration:

```bash
cargo bench --locked -p summa-llm --bench moe_layer --features training-fusion -- \
  --tokens 8192 --warmup 20 --iterations 100
cargo bench --locked -p summa-llm --bench moe_primitives --features training-fusion -- \
  --tokens 8192 --warmup 20 --iterations 100
```

[MoE A100 results](moe-performance.md) include the PyTorch comparison and
historical environment. Both layer benchmarks emit JSON. Save stdout and stderr
separately, and compare identical model geometry, dtype, and loss modes.

The embedded memory models support local CPU smoke runs:

```bash
cargo bench --locked -p summa-llm --bench memory_reserve -- \
  --tokens 16 --warmup 3 --iterations 10
cargo bench --locked -p summa-train --bench wake_tier_step -- \
  --batch-size 2 --sequence-length 8 --warmup 3 --iterations 10
```

For CUDA acceptance, use the production models and explicitly require CUDA.
Cargo runs benchmark executables from their package directory. The absolute
model and output paths below are expanded by your shell at the repository
root, so the benchmark finds the shared model definitions:

```bash
cargo bench --locked -p summa-llm --bench memory_reserve --features training-fusion -- \
  --model "$PWD/summa-mal/well-known/retriever_300m_moe_sleep.mal" \
  --baseline-model "$PWD/summa-mal/well-known/retriever_300m_moe.mal" \
  --tokens 8192 --tier 0 --max-active 2 --require-cuda --enforce \
  --output "$PWD/.context/memory-reserve-cuda.json"

cargo bench --locked -p summa-train --bench wake_tier_step --features cuda -- \
  --model "$PWD/summa-mal/well-known/retriever_300m_moe_sleep.mal" \
  --batch-size 4 --sequence-length 1024 --periods 100,400,3200 \
  --non-due-clock 99 --due-clock 100 --require-cuda \
  --output "$PWD/.context/wake-tier-step-cuda.json"
```

Memory-reserve acceptance pairs at least three unique seeds. `--enforce`
requires at least three warmups and ten timed iterations per seed and applies
the overhead gate. CPU/Metal reports are smoke evidence only. Wake-tier timing
includes the optimizer path; memory-reserve timing measures model-only cost.

## Server load and training evaluation

The [Python stress runner](../summa-client-python/stress_test/main.py) measures
concurrent indexing and search against a running gRPC server:

```bash
(cd summa-client-python && uv run --locked python -m stress_test.main --help)
```

Use a dedicated test server and index: the runner creates/indexes data and can
clean it up. Record worker counts, target QPS, achieved QPS, failures, latency,
server resource controls, and concurrent optimizer work.

Training's verified benchmark workers are a separate acceptance/evaluation
surface. See [training contracts](training-objectives-and-curricula.md),
[retrieval-pool evaluation](retrieval-pool-eval.md), and
[generation evaluation](generation-eval.md).

## Recorded results and reporting

Historical results live with the design they informed:
[posting codecs](posting-codecs.md), [FastScan layout](fast-scan-layout-v2.md),
[algebraic reductions](algebraic-float-reductions.md),
[TurboQuant](turboquant-quantization.md), [MoE](moe-performance.md),
[tokenizer evaluation](tokenizer-backends.md), [cuBLAS](cublas-gemm-dispatch.md),
[attention](fused-attention.md), [selective scan](segmented-selective-scan.md),
[cross-entropy](fused-cross-entropy.md), and [BF16](bf16-residual-stream.md).
Their dates, baselines, and rejected experiments remain historical evidence;
editing documentation does not refresh those measurements.

For new numbers, retain the commit (and dirty diff), exact command, toolchain
and dependency versions, OS/CPU/GPU/driver, feature flags, seeds, dataset/model
checksums, index format and segment count, warmup/iteration counts, and raw
samples. Report quality alongside speed, specify cache state and timing
boundaries, and repeat comparisons in interleaved processes. A best-of-three
microbenchmark or CPU smoke run does not establish production GPU throughput.

### Row compaction copy paths

`segment_merge` includes `row_compaction/{pattern}/{rows}` at 4096 and 65536
physical rows. `mixed_fast_columns` deletes alternating rows from one segment
with eight numeric columns (including missing and multi-values) and a primary
key. `clustered_mixed_fields` and `scattered_mixed_fields` add indexed/stored text,
merge 1024-row source segments, and delete respectively one contiguous quarter
or every fourth row. Setup, deletion publication and count validation occur
outside timing. Compaction uses a 32 MiB scratch budget and a RAM directory.

Set `SUMMA_COMPACTION_BYTES=/path/prefix` to capture each validated `.fast`
output as `prefix-{pattern}-{rows}.fast`. Different legal block boundaries can
change complete-file bytes; compare copied source blocks byte-for-byte and all
decoded values in regression tests. Measure process peak RSS separately from
map scratch and output size. These fixtures do not measure remote I/O or ANN
throughput; ANN correctness tests cover encoded survivor bytes across all formats.

## Search Benchmark, the Game

The [comparison protocol](search-benchmark-game.md) describes the full Wikipedia
fixture, fairness rules, exact-count semantics, and measurement requirements.
See the [measured results](search-benchmark-results.md) and
[raw samples](benchmark-results/search-game-2026-09-13/README.md).
Build the native adapter and validate it before attaching it to upstream:

```bash
cargo build --locked --release -p summa-core --example search_benchmark_game
# Protocol smoke gate: exact counts against direct token matching plus the
# pruned-vs-exhaustive VERIFY check. Run it after any adapter or query change.
python3 scripts/search_benchmark/smoke.py target/release/examples/search_benchmark_game
# Download/transform the corpus with the pinned upstream repository first.
target/release/examples/search_benchmark_game index /path/to/new-summa-index < /path/to/corpus.json
# Copy scripts/search_benchmark/Makefile to upstream engines/summa/Makefile.
# Run from the upstream checkout, with absolute paths:
SUMMA_BENCH_BINARY=/path/to/search_benchmark_game SUMMA_BENCH_INDEX=/path/to/summa-index \
  make bench ENGINES='summa tantivy-0.26' NUM_ITER=10 WARMUP_TIME=60 \
    COMMANDS='TOP_10 TOP_100 TOP_1000 TOP_100_COUNT COUNT'
```

`COUNT` and `TOP_*_COUNT` disable pruning through Summa' exhaustive collector.
`VERIFY` checks pruned top-10, top-100, and top-1000 against exhaustive results
outside timing. All
operations require one merged segment for the comparison. Unsupported syntax
fails; unsupported protocol commands return `UNSUPPORTED`.

For paired runs, build both Rust engines with the same compiler and release
settings. The upstream Tantivy example enables LTO, so use
`CARGO_PROFILE_RELEASE_LTO=true RUSTFLAGS='-C target-cpu=native'` for Summa too.
The adapter accepts `--indexing-threads`, `--indexing-memory-bytes`,
`--posting-codec rounded|packed|pfor|simd4x`, `--term-dict-block-bytes`, and
`--no-background-merges` when building an index. The opt-in
`--posting-ratio-bounds` and `--posting-impact-bounds` options write tighter,
score-independent block bounds on new segments. Serving accepts
`--term-cache-blocks`, `--term-cache-bytes`, `--posting-validation-cache-bytes`,
and `--exhaustive`; pass serving options with
`SUMMA_BENCH_ARGS`. Defaults and limits are documented in the
[protocol](search-benchmark-game.md#runtime-controls).

Preserve the upstream `results.json` after each run; the next run overwrites it.
The comparison tools use all samples and check query sets and exact-count
agreement. Run `scripts/search_benchmark/verify.py --help` for the full-corpus
pruned/exhaustive verification gate, and `analyze.py --help` for grouped results.

See [posting block execution](search-block-execution.md) for the latest official
workload comparison, separate standalone-term results, and exact-reference gates.
