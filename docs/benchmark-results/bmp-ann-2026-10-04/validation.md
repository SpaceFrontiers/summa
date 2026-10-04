# Validation — 2026-10-04

| Check                                                          | Result                                                                                                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `python3 scripts/check_search.py check`                        | All five stages pass: formatting, strict Clippy, native tests, native without sync, standalone broker. 2,173 tests pass; 26 ignored.       |
| WASM `bash build.sh && npm ci && npm test`                     | Release build passes; 41 JavaScript tests across nine files pass.                                                                          |
| Linux release library tests                                    | 1,916 pass, 16 ignored in both C and E measured executables. E has the final binary scorer plus the subsequently rejected BMP E candidate. |
| Independent persisted binary scan oracle                       | All measured queries agree with scalar exhaustive Hamming results; all 12 byte/result hash keys agree across builds/architectures.         |
| Persisted BMP replay                                           | 864 unique query/depth/gamma cases agree bit-for-bit; 10,368 recorded executions across ABBA.                                              |
| Synthetic BMP audit                                            | All 48 shape/distribution/depth/gamma hash keys agree across their compared variants.                                                      |
| `python3 docs/benchmark-results/bmp-ann-2026-10-04/analyze.py` | Validates evidence and regenerates 149 paired cells plus memory tables.                                                                    |
| `uv run scripts/check_docs.py`                                 | Passes after final report links are present.                                                                                               |
| `git diff --check`                                             | Passes.                                                                                                                                    |

The native harness includes the new BMP touched-slot, signed TQ accumulator and
unsigned ScaNN accumulator boundary tests, along with existing binary tie,
deletion, parallel and ordinal-completeness regressions. Measurements ran before
the native/WASM validation builds, not concurrently with them on the same host.

The nine-stage `full` lifecycle/RPC harness was not rerun: no lifecycle, wire,
publication or RPC behavior changed. No GPU, trained-routing recall, cold I/O,
service concurrency or WASM performance claim is made. The portable WASM binary
scan retains its original loop. The native harness ran against the retained
source after all BMP runtime experiments were removed.

Original logs remain under `.context/bmp-ann-20261004/` and the native harness
run `.context/search-harness/20261004T163537.763322Z-check/`. Compact validation
and build provenance are included in `validation-logs.json.gz`. Both cloud
machines (`hermes-validation-moroni`, `hermes-benchmark-32-moroni`) were verified
`TERMINATED`; their disks were preserved.
