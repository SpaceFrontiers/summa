# Binary retrieval eligibility and bounded nomination

Status: implemented and locally validated on 2026-10-29; production rollout and
matched quality/cost evaluation remain pending. Existing query defaults stay
unchanged. Production motivation is from Azeroth's sparse-removal audit.

Binary scorers previously discarded common-filter eligibility, selecting ANN/flat
top-k before the wrapper removed ineligible documents. Eligibility now applies
before the prefix shortlist and document heap. Intersect the segment's immutable
live-document mask with the request mask once per scorer; borrow the ANN payload
through a shallow immutable view. Never mutate shared reader visibility. Flat,
IVF (including prefix/SOAR), and ScaNN paths must agree. Filtering does not turn a
partial IVF probe into exact search and does not widen its probe budget.

Expose opt-in binary query controls for nprobe, prefix_rerank, and
max_ordinals_per_doc. Zero retains the schema/default behavior. Explicit limits
are validated before admission/conversion and again at the core boundary.
Explicit probe and prefix limits cannot exceed 65,536, and the ordinal limit
cannot exceed 64. Probe and prefix limits are independent of returned document depth.
The probe-plan fingerprint already includes the effective nprobe.

The ordinal limit selects the highest exact passage scores (ties by ordinal),
after computing the document's original combiner score and before fusion union,
trace, or feature backfill. It is an explicit nomination policy, not truncation of
an already-produced response. It bounds downstream binary nomination rows to
documents times max_ordinals_per_doc. It does not cap exact document completion
work, change score-only backfill, or promise constant query I/O; existing exact
scoring and request budgets still apply. Other branches may nominate additional
ordinals. No caps are raised to accommodate long documents.

This permits MAX or weighted-top-k document nomination without exporting every
passage of long documents. Azeroth must include these controls and the binary
combiner in its candidate/model contract before an online policy change. Older
servers ignore unknown protobuf fields, so clients must verify the deployed
engine's `binary_retrieval_version >= 1` capability before sending nondefault
controls. Brokers report the minimum across partitions. No production default or
model is promoted from synthetic tests or historical identity retention alone.

Validation covers filtered top-k with stronger ineligible documents, deletion
mask intersection, sync/async equivalence, exact scores and retained ordinals,
unchanged default output, invalid/oversized controls, and real RPC conversion.
Replay production queries at fixed nomination budgets, including the previously
failing long-document query, then compare relevance and cost before promotion.

The full search harness passed all nine stages: 2,197 native tests (27 ignored)
and seven real-server broker tests, strict Clippy, native without sync,
standalone broker compilation, portable core compilation, and API docs. The
WASM release build and 41 JavaScript tests passed; Python and TypeScript clients
passed 17 and 18 tests. These establish functional behavior and compatibility,
not production latency or relevance gains. The filtered-top-k regression failed
before the fix and passed afterward. A 2,000-passage fusion export retains only
three nominated ordinals while preserving the document score.
