# BM25 impact audit evidence

See the [proposal and limits](../../posting-codecs.md#whole-vocabulary-storage-audit-and-narrower-first-experiment).
This is offline research: no new posting format or query algorithm is enabled.
The whole-vocabulary audit uses the 100k ARM fixture and canonical Summa readers;
the earlier 5,032,104-document audit covers only the 714 query terms. Do not
substitute the latter for full-vocabulary storage costs. The integer comparison
uses the existing `write_vint`; coordinate estimates use two f32 values per point.
Neither storage estimate includes a new footer or alignment.

The archive includes 87 hash-verified files: probe sources and raw
outputs, exact-rational and canonical-f32 checks, process resource measurements,
and changed build inputs over base `ce2c96b945fccc4bac58ccc45bb4bc23b809773e`.
Its source snapshot predates the supplemental deletion assertion. Source overlays
are complete changed build inputs, not a complete repository checkout. The
unchanged Cargo lock and full-corpus machine/corpus metadata are retained.

`raw-results.zip`: 624,800 bytes; SHA-256 `853bd5a75b65bd7d4f9c2e595d9527b666b56dc574c417a615966e5be199355f`.
Per-file sizes and hashes appear in `manifest.json`.

The subsequent ARM metadata-kernel comparison is captured separately in
`layout-kernel-results.zip`: 13 verified files, 14,307 bytes,
SHA-256 `88d2e643e67ea1c2c84f83907ddfe4fa4ff4e3f6db3ebf8be37a9a5f408718d3`. It contains raw nine-round alternating samples (32 full-record
passes each), source, compiler/CPU metadata, RSS, and the fixture hashes verified
unchanged afterward. The same source overlay in `raw-results.zip` builds this
helper: copy `impact-layout-kernel.rs` to an exclusive temporary example path and
build with the native LTO command recorded in `impact-layout-kernel-build.py`.
Invoke the executable with the prepared index directory as its sole argument.

Coordinate records average 4.539 ns/block versus 16.517 ns/block for existing
vint pairs across all 60,216 multi-block records. Pair payloads are 1,530,832 and
498,337 bytes respectively, before the common directory. Bounds are checked
against the canonical scorer for the retained integer points at the reported
parameters. This warm scalar kernel excludes parsing, query execution, posting
I/O and heap collection; it is not a whole-query benchmark. Shared Mac results
do not justify choosing a format without the corresponding x86/public workload
measurements. No production format is enabled.
