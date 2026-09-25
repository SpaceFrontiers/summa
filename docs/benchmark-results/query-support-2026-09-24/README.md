# Query capability evidence — September 24, 2026

`before.json` retains the preceding four-document HTTP capability probe:
801 expressions accepted, 25 rejected (13 regex and 12 punctuation/escaping
cases). `queries.jsonl` contains the unchanged 826 expressions from the pinned
[Searchbench workload](../../searchbench-comparison.md#pinned-workload).
The [design](../../regex-query.md) describes the new query support and its limits.

Reproduce the new probe with:

```sh
cargo build -p summa-server --example searchbench_http
python3 docs/benchmark-results/query-support-2026-09-24/verify.py
```

The script creates an isolated four-document index, submits all expressions to
the real HTTP adapter, retains every response and executable/query hashes, and
fails if any expression is rejected. It always stops its local server.
The regex integration tests separately check nonempty matches for all 13
benchmark regex expressions, exact IDs, scores and counts.

Acceptance on this fixture does not establish full-corpus count agreement,
ranking equivalence or throughput. Existing dictionary and posting limits still
apply; broad queries may exceed them on a large vocabulary. Analyzer and sloppy
phrase differences remain. The prior performance tables still cover 15 queries.

`after.json` records **826 accepted / zero rejected**, with all 801 previously
accepted responses unchanged. The exact query bytes match the original probe's
recorded SHA-256. Results contain no throughput measurements.
