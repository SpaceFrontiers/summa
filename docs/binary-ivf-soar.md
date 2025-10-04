# Binary IVF SOAR spill

Status: implemented as `indexed<ivf, soar: full>` on binary dense vector fields;
off by default. Binary ScaNN keeps its existing nearest-alternate spill.

## Why

SOAR (Sun et al., NeurIPS 2023) stores each vector in a second leaf chosen so
that its residual to that leaf is close to orthogonal to its residual to the
primary leaf. A query that lies along the primary residual, where the primary
leaf is a poor fit, then finds the vector through the secondary leaf. For ±1
codes Hamming distance is affine in the inner product, so the SOAR loss carries
over exactly with Hamming quantities:

    loss(c') = h(x, c') + λ · popcount((x ⊕ c) & (x ⊕ c'))² / h(x, c)

where `c` is the primary leaf, `x ⊕ c` its residual, and λ = 1. On production
`pplx-embed-v1-4b` codes in a 4·sqrt(N) IVF this reads 3.1–3.3× fewer bytes per
query than the plain index at equal recall when combined with the prefix scan,
and reaches recall the plain index does not reach within 256 probes
([measurements](binary-prefix-scan.md)). λ = 0 (plain nearest alternate, what
binary ScaNN does) gave only 1.0–1.08× in the 1M Cohere study, so the
orthogonality term is what pays.

## Behaviour

- Every vector gets its primary leaf as today, plus one secondary leaf chosen
  among the 32 nearest leaves (under the field's routing) by the loss above
  (ties: lower leaf ID). On the production sample, restricting candidates to the
  16 nearest leaves gave recall@10 0.9364 / 0.9537 / 0.9643 at 64 / 96 / 128
  probes against 0.9363 / 0.9536 / 0.9647 for the loss minimised over all 2,543
  leaves, so the bounded pool costs no recall.
  Vectors that coincide with their primary centroid (`h(x, c) = 0`) are not
  spilled. Only `soar: full` is accepted on binary IVF: selective spilling
  measured 0.99–1.17× for its 30–50% storage, so it is rejected loudly rather
  than silently ignored.
- Storage: codes, doc IDs, ordinals and prefix rows are stored for both
  postings, roughly doubling the ANN payload. Exact-vector lookups keep one
  location per vector (the existing SOAR deduplication in the lookup writer),
  so exact reads and lookup size are unchanged.
- Queries deduplicate `(doc, ordinal)` keys: the heap-only collector is used only
  for payloads without spill, the general collector keeps the best (identical)
  score of both copies, and the prefix stage drops repeated keys before
  full-code rescoring.

## Format and lifecycle

- A spilled payload is binary IVF layout revision 1 with bit 32 of the header
  tail set (the low 32 bits remain the prefix width). Earlier readers reject
  revision 1. The header's vector count is the physical posting count, in
  `logical..=2 × logical`.
- The schema must match the payload: readers and merges reject spilled
  payloads for fields without `soar`, and unspilled payloads for fields with it.
  Changing `soar` through ALTER rebuilds payloads from retained exact codes
  without retraining the quantizer.
- Build and generation rebuilds compute secondary leaves with the global
  quantizer's own routing for candidates, so cost is one extra routed probe per
  vector. Byte-copy merges, coalescing merges, reorder and deletion compaction
  move postings without reassigning leaves, so both copies of a vector survive
  together or are deleted together.

## Validation

Tests cover: header round trip and rejection of spill flags on other kinds;
SOAR-loss selection against a brute-force oracle; deduplication in every query
path (serial, parallel, prefix, combined multi-value); merge, coalescing,
deletion and ALTER keeping posting counts within bounds; and an end-to-end
index with exact scores identical to brute-force Hamming for covering probes.
