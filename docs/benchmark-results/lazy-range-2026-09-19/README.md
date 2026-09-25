# Lazy range scorer measurements

The retained implementation and tradeoffs are described in the
[range scan report](../../range-block-scans.md#lazy-range-scorer-batching).
[summary.json](summary.json) contains the final M4 runs and validation;
[cloud-summary.json](cloud-summary.json) contains isolated x86 measurements.
All times are nanoseconds. Each run retains its confidence interval. The selected
estimate is the slope for linear Criterion sampling and the mean for flat sampling.
There are 20 samples per case, one-second warmups and two-second measurements.

The base is Summa 1.8.147, Rust 1.98.1, ordinary release flags. Branch revision
`05150b9ba26eeb4d3f8e1bdba5572b816412f8b9` and main revision
`717141880763b6ea60c62af46b210ef3a77474c8` have identical trees. Benchmark fixtures
were added before building both sides; package versions and compiler flags match.

- [local-evidence.tar.gz](local-evidence.tar.gz) contains final `before-0`,
  `after-0`, `after-1`, `before-1` samples/logs, memory pairs, commands and binary
  hashes. Earlier helper experiments and noisy exploratory harness runs are
  explicitly distinguished in `trial-notes.txt`. Only the final matrix supports
  the retained implementation's numbers.
- [cloud-evidence.tar.gz](cloud-evidence.tar.gz) contains the exact measured
  patches, benchmark sources, drivers, compiler/CPU metadata, build/test logs,
  per-run CPU counters, samples, memory runs and binary hashes. `results/boundary`
  measures the retained runtime (`candidate-boundary.patch`); `results/edges`
  repeats the added multi-value and complete-miss controls with matched expanded
  benchmark binaries. Root `results` measurements are the earlier helper-only
  experiment, not the retained runtime. The final memory pairs are in `results/edges`.
- [check.json](check.json), [async-only.txt](async-only.txt) and
  [wasm.txt](wasm.txt) record validation of the integrated source, including the
  additional exact-hit compressed-tail regression added after benchmarking.

To replay, use a fresh checkout of the base revision, copy the archived benchmark
source to `summa-core/benches/rust_hot_paths.rs`, and build with
`cargo bench --locked -p summa-core --bench rust_hot_paths --no-run`. Preserve
that binary, apply the archived candidate patch, and rebuild. The archived drivers
record exact filters, alternating order and affinity. Expanded fixtures require
`expanded-bench.rs` on both sides. The current benchmark uses an equivalent
`is_multiple_of` expression for lint compliance and gates the lazy group on `sync`;
these do not change the measured fixture or default-feature operation.

Encoded formats and writers are unchanged. The benchmark uses warm in-memory
segments and public scorer/bitset APIs; it does not measure cold storage,
concurrent full-query throughput or RPC latency. The machine archive hash was verified
before deleting the temporary machine and boot disk. Archive hashes are in the summaries.
