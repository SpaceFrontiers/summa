# Search Benchmark, the Game — 2026-09-13 evidence

See the [results and limitations](../../search-benchmark-results.md) and the
[protocol](../../search-benchmark-game.md). `raw-results.zip` contains the complete
selected timing samples, verification records/logs, memory measurements and
measurement metadata. It also includes the public-reader position-size diagnostic
and the earlier pruning-window counters. The manifest records SHA-256 hashes for every included
file. Intermediate experiments are labeled with their original engine names;
they are not pooled into the final `summa-optimized` results.

Extract and recompute the paired baseline comparison from the repository root:

```bash
python3 -m zipfile -e docs/benchmark-results/search-game-2026-09-13/raw-results.zip .context/search-game-results
python3 scripts/search_benchmark/analyze.py .context/search-game-results/native-baseline.json \
  --candidate-results .context/search-game-results/optimized.json \
  --baseline summa --candidate summa-optimized
```

For the final Tantivy comparison, use `optimized.json` as the input, with
`--baseline tantivy-0.26 --candidate summa-optimized`. All official runs use the
same 962 queries, 60-second warmup and ten repetitions. The separately generated
714-term supplement uses ten seconds and five repetitions. Geometric means are
computed over per-query median latency ratios; raw samples are microseconds.

The much larger local `.context/summa-benchmark-evidence.tar.gz` additionally
preserves full CPU profiles, builds, failed/interrupted experiments, pipeline
scripts, frozen source overlays and runtime details. Its checksum is saved beside
it. The paired base Git bundle and baseline parser overlay remain in `.context`.
These local artifacts are not part of this small versioned results bundle.
