# Dictionary decoding and the Luxir gap — September 26

The retained shared-decoder change improves wildcard TOP_10 throughput by
**6.9% (287 → 307 QPS)** and reduces server CPU per request by **7.2%** on the
unchanged 10M-document corpus. Regex TOP_10 improves 6.0%. This closes part of
the gap: Luxir still reaches 1,767 wildcard QPS, about **5.75×** Summa. No format,
cache, expansion-budget or backend defaults change.

This continuation measures dictionary decoding on the same immutable 10M-document
index and pinned Searchbench replay as the
[finite-regex campaign](../closing-gap-2026-09-24/README.md#september-26-finite-regex-alternatives-and-posting-investigation).
The production candidate changes the inlining policy of the existing unsigned
variable-integer reader. It retains the decoder's implementation, byte
consumption and validation; there is no second codec, new index, larger cache,
raised expansion limit, or changed backend default.

## Paired HTTP results

The 90 cells complete **23,587,209 timed requests with zero errors**. The two
native Summa builds each pass 677 exact audits; untimed HTTP checks total 8,124
Summa and 4,062 Luxir responses. The 826-query probe preserves every count and
error, including the 149 excluded budget errors. The fixture and replay hashes
are identical before/after. [All phase metrics and memory](results.json),
[source/compiler hashes](sources.json), and [profiles/native diagnostics](profiles.json)
are retained.

Rates below are means of the two phase medians. CPU change uses total server CPU
per completed request across both phases. Small changes in cheap or Boolean
controls should not be treated as broad algorithmic gains; the selected native
Boolean diagnostics mostly remain within 1%.

| Family        | Operation | Before QPS | After QPS | Luxir QPS | Throughput change | CPU/request change |
| ------------- | --------- | ---------: | --------: | --------: | ----------------: | -----------------: |
| and_high_med  | TOP_10    |      3,949 |     4,060 |     8,523 |             +2.8% |              -1.7% |
| and_high_med  | TOP_100   |      3,118 |     3,142 |     5,927 |             +0.8% |              -0.2% |
| and_high_med  | COUNT     |      4,799 |     4,839 |     6,066 |             +0.8% |              -0.9% |
| low_term      | TOP_10    |     59,898 |    60,024 |   105,463 |             +0.2% |              -0.4% |
| low_term      | TOP_100   |     34,452 |    35,617 |    32,386 |             +3.4% |              -1.9% |
| low_term      | COUNT     |     75,586 |    75,563 |   118,644 |             -0.0% |              -0.8% |
| or_high_low   | TOP_10    |     16,829 |    17,145 |    32,616 |             +1.9% |              -2.5% |
| or_high_low   | TOP_100   |     11,295 |    11,557 |    14,920 |             +2.3% |              -2.9% |
| or_high_low   | COUNT     |     36,835 |    37,213 |     9,037 |             +1.0% |              -1.5% |
| regex         | TOP_10    |     61,182 |    64,849 |   106,338 |             +6.0% |              -3.7% |
| regex         | TOP_100   |     56,859 |    57,667 |    99,276 |             +1.4% |              -1.2% |
| regex         | COUNT     |      4,576 |     4,596 |    13,062 |             +0.4% |              -0.6% |
| wildcard_scan | TOP_10    |        287 |       307 |     1,767 |             +6.9% |              -7.2% |
| wildcard_scan | TOP_100   |        288 |       308 |     1,764 |             +6.9% |              -6.3% |
| wildcard_scan | COUNT     |        280 |       299 |     1,681 |             +6.5% |              -6.5% |

Wildcard TOP_10 CPU falls from 99.44 to 92.33 ms/request. Separate CPU-0 ABBA
diagnosis confirms 6.4–7.8% lower search time for the selected wildcard
expressions and 5.3–5.5% lower time for the infinite regex. The two selected
prefix expressions improve 9.5–12.4% in native diagnosis; prefix HTTP throughput
was not retimed in this campaign. Finite-regex and single-term native cases are
mixed/small, so these measurements do not establish a uniform query speedup.

The wildcard profile attributes 19.38% of control self samples to the integer
reader and 8.90% to the surrounding entry decoder. After inlining, 23.32% is
attributed to entry decoding; the standalone reader symbol disappears. The
integer decoding work still exists inside its caller. Zstd sequence decoding
remains the largest individual symbol, at 31.23% before and 33.12% after; its
larger share is not evidence of more absolute decompression work.

There is **no memory reduction**. End-of-phase Summa RSS is 3,802–3,807 MiB
before and 3,809–3,815 MiB after; anonymous RSS is 242–246 and 249–254 MiB.
The small increase is not attributed to a specific allocation. Luxir reaches
1,170–1,190 MiB total and about 57–58 MiB anonymous RSS under equal untimed
coverage. Locked bytes remain zero; most Summa residency is evictable mapped
payload. Configured cache/scratch budgets remain unchanged.

The remaining work is to reduce decompression and dictionary scan work under
the existing memory budgets, then revisit the larger conjunction/OR and
metadata-residency gaps. Another blanket inlining change is not supported by
the rejected entry experiment below. These warm retrieval measurements do not
establish cold-storage or io_uring performance.

## Protocol and limits

Both Summa executables are fresh builds from snapshots differing only in
`summa-core/src/structures/vint.rs`, using Rust 1.98.1 release and
`-C target-cpu=native`. [Host details](platform.json) record the 32-vCPU x86 VM.
It runs A1/B1/L1/B2/A2/L2, where A/B are
control/candidate and L is the unchanged Luxir release. Every phase restarts the
server, validates equal untimed query/operation coverage, warms for 40 seconds,
then runs three three-second repetitions per cell. There are 32 clients, 30
server CPUs and two driver CPUs. Result caching remains disabled.

Five families cover 42 wildcard scans, four admitted regexes, 50 high/medium
conjunctions, 46 high/low disjunctions and 47 low-frequency terms. Every family
runs TOP_10, TOP_100 and COUNT. These are weighted repeated-query workloads, not
uniform per-expression costs. Cheap counts can be driver-limited, and these
short cells do not establish tail-latency guarantees. HTTP responses contain IDs
and scores; they do not read stored documents. The io_uring hydration experiments
therefore do not explain these results.

The exact Summa audit compares all 677 admitted counts and top-100 IDs/raw score
bits with the retained oracle. The HTTP probe also verifies all 826 responses,
including 149 visible expansion-budget errors. Cross-engine ranking equivalence
is not asserted. Before/after file hashes verify the entire 18,693,711,229-byte
index and replay remain unchanged. No index rebuild is involved.

## Code generation and architecture controls

The x86 control contains out-of-line calls to the canonical reader. Those
symbols disappear in the candidate. The executable's `.text` section grows from
12,445,336 to 12,458,584 bytes: 13,248 bytes, or about 0.106%. Heap and cache
budgets are unchanged. End-of-phase mapping residency and peak RSS are recorded
separately from instruction size.

The ARM control uses the existing `search_pipeline::bench_pattern_filters`
fixture: 50,000 terms, 16 cached blocks, regex/wildcard/prefix queries at top-10
and top-100, and exact result checks before sampling. Initial wall-time ABBA
measurements on the busy workstation fluctuate substantially and are excluded
from speed claims. A separate ABBA uses a thread CPU clock with the same fixture
and query code; [the clock adapter](cpu-clock.patch) is experiment-only. This
controls scheduler descheduling, not all effects of frequency changes, cache
contention or heterogeneous cores. It is not a second large-corpus benchmark.

An initial CPU-clock build reused the same executable for both source labels;
its results were discarded. The replacement cleans the core release artifacts
before each build, records distinct executable hashes, and recompiles both
variants. An initial cloud launcher had a stale source path after the successful
audits/probe and failed before any timing cell; its logs are retained separately.

The clean ARM CPU-clock ABBA shows 2.3–3.6% lower wildcard CPU time,
4.2–6.9% lower regex CPU time and 3.3–4.0% lower prefix CPU time on that
fixture. The original wall-time run remains excluded. Full phase estimates,
confidence intervals and executable hashes are in [ARM evidence](arm.json).

## Rejected surrounding-entry inlining

A separate build additionally forced the whole SSTable entry decoder into its
callers. The eight-vCPU x86 build VM verified byte-identical corpus files and
all 677 exact audit responses, then ran CPU-0 ABBA diagnostics on 18 expressions
(two limits, 20 warmup and 100 measured iterations). This is a separate host
screen, not another Summa/Luxir throughput phase.

The extra change reduces selected wildcard search time by only 0.1–0.7%, while
the infinite regex is 6.0–6.2% slower, conjunctions 1.8–3.2% slower and
disjunctions 1.0–3.2% slower. Its `.text` grows by another 17,344 bytes. It is
**rejected**; production keeps only the smaller variable-integer change.
[All paired native rows](entry-experiment.json) are retained, including favorable
small cases. This screen does not establish general memory or cold-I/O effects.

## Validation and reproduction

`python3 scripts/check_search.py check` passes all five stages, including strict
Clippy, 2,121 native tests (25 ignored), native without sync, and standalone broker
compilation. The WASM release build and all 41 JavaScript tests pass. New tests
pin encoded bytes, consecutive values, reader positions, every truncation length,
non-minimal encodings and overflow consumption. The full lifecycle/RPC and
optional io_uring suites were not rerun because this change affects neither
lifecycle nor the payload service. See [validation metadata](validation.json).

The exporter reuses the existing Searchbench metric and profile parsers; the ten
historical profile exports remain identical after extracting the shared helper.
Run `python3 summarize.py /path/to/summa-dictionary-20260926` on the complete
private capture. Raw captures retain source archives, binary hashes, compiler
metadata, HTTP responses, profiles and failed launcher attempts. Timed HTTP runs,
profiled runs and single-CPU native diagnosis are separate evidence.

Both VMs were explicitly stopped after collection and independently confirmed
terminated.
