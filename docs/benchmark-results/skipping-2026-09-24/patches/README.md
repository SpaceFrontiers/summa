# Isolated skipping variants

Apply each patch independently to Summa `46f4e294` (1.9.1). An empty
`current.patch` denotes the unchanged control. Never stack these patches.
The constant-false conditions are compile-time ablations for this experiment;
they are not proposed production configuration or code changes.

Build all variants with the same locked dependencies and compiler:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --locked --release \
  -p summa-server --example searchbench_http
```

The build must recompile `summa-core` after each patch. Save distinct binary
hashes. Run the phrase and skewed-conjunction regressions, then the full-corpus
`searchbench_http audit` and HTTP response comparisons before timing.
See [the design and invariant](../../../score-guided-traversal.md#expanded-workload-experiment-september-24).

These files record the experiment definitions. They do not imply successful
execution or a measured improvement; the campaign report owns that evidence.
