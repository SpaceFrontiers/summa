# L1 candidate scoring: handoff to the Summa agent

Status: work in progress, handed off on 2026-09-05. The user rejected the new
inverse lookup maps in this implementation. The code is preserved for review;
it is not an accepted storage design or a release-ready change.

## Start here

- Handoff branch: `handoff/l1-scoring-2026-09-05` in `SpaceFrontiers/summa`.
- Code at handoff: `7fa5d4cd7743f33f822e284a4636ca92cd601151`, followed by this
  documentation commit. The handoff does not change engine code.
- Original branch: `feature/l1-candidate-scoring`;
  PR #169 in the archived development repository remains open.
- Upstream incorporated: `98c86059`, version 1.8.123. Review with
  `git diff 98c86059...HEAD`, preserving the independent upstream text-pruning fixes.
- Integration repository: `SpaceFrontiers/azeroth`, branch
  `handoff/ai-search-2026-09-05`; its handoff is
  `docs/plans/active/2026-09-05-ai-search-handoff.md`.
- Read `AGENTS.md`, `CLAUDE.md`, and the
  [search system contract](../search-system-contract.md) before editing.
  Rust 1.98.1 is the toolchain.

## The user's requirement

L0 nominates documents/passages through named lexical, phrase, sparse,
dense/binary, and document-profile branches. Every nominated item needs the
raw score of every requested branch, including branches that did not retrieve
it. A missing top-K result must not become a fabricated zero score.

Summa should optionally apply a portable linear formula over those features
before selecting the pool sent to the external cross-encoder. RRF remains an
alternative. Apply the formula on shards and the broker, preserve query and
document combiners, and export raw scores for training and inspection.

Search API owns teacher labeling, model fitting, feature/query contracts and
activation. The teacher is the configured cross-encoder. Train on larger frozen
candidate pools and evaluate on held-out query groups. Do not change extraction
chunk limits or AI context caps. Ordinary Telegram stays in document/title
discovery; shared smart passage policy serves API/RAG/MCP/Spacefrontiers/Cybrex.

## Rejected design and exact working-tree state

The user's latest design instruction is: **"we do not want to have such maps,
it flaw of your design"**, referring to the new inverse lookup maps.

The committed feature currently adds per-segment `.lookup` files with sorted
`(document, ordinal, physical ID)` rows, 12 bytes per row plus headers. It uses
these to address BP-reordered BMP/text fields during cross-branch point scoring.
These files and their lifecycle integration **still exist in the code on this
branch**. Removing/replacing them was not completed before handoff.

Existing index maps are present: BMP V19 stores physical/virtual ID to document
and ordinal in its blob; chunked text has its existing chunk map; flat vectors
have document-to-vector range access. Do not interpret this handoff as a claim
that the existing index has no maps. Review the existing representations and
query execution before choosing how to score nominated candidates without
adding the rejected maps. No replacement design has been implemented or measured.

An additional `PrepareCandidateScoring` RPC was briefly implemented locally to
derive lookup files without rerunning BP. It was **reverted before this handoff**:
no preparation RPC, client method, writer/manager wrapper, or preparation test
from that experiment is included. Its prototype compiled, but its first test
run failed to compile because a test compared `ScoredPosition` directly; it was
never validated. Do not restore that experiment as the proposed solution.

Older documents recommending Reorder/preparation describe the rejected design.
Their preparation instructions are superseded by this handoff. No production
reorder, lookup preparation, index rewrite, or L1 rollout was performed here.

## Implemented code to review and retain where appropriate

| Area                      | Implementation                                                                                                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Core scoring              | `summa-core/src/query/candidate_scoring/{mod,execution,model,tests}.rs`: named features, document/chunk scopes, score completion, fixed transforms, linear inference, presence versus zero, nominated passages and bounded diagnostics |
| Shared scoring primitives | `query/term.rs`, `phrase.rs`, `bmp.rs`, `reranker.rs`: reuse text statistics/positions and stored vector scoring; preserve quantization and negative dense scores                                                                      |
| Existing addressing       | `segment/reader/bmp.rs`, `segment/chunk_map.rs`, flat-vector readers: inspect existing maps and layouts before redesign                                                                                                                |
| Rejected additions        | `segment/ordinal_lookup.rs`, `segment/ordinal_lookup/lifecycle.rs`, `segment/reader/candidate_lookup.rs`; references in reader open, merger, reorder, segment types, diagnostics and tests                                             |
| Nomination/eligibility    | `query/filtered.rs`, `fusion.rs`, `planner.rs`, `index/searcher.rs`: bounded union, score-only branches, shared hard filters, logical passage deduplication                                                                            |
| Server                    | `summa-server/src/search_service/candidate_scoring.rs`, validation/conversion/response modules: limits, branch conversion, complete raw exports and ranking markers                                                                    |
| Broker                    | `summa-broker/src/ranking.rs`, `search_service.rs`, `partition.rs`: global statistics, full branch union/global RRF, shared L1 formula, exact final selection and bounded export                                                       |
| Protocol/clients          | `summa-proto/summa.proto`, Python and TypeScript clients/generated bindings: named scopes, `score_only`, `candidate_depth`, `l1`, `score_export`, candidate features and capability reporting                                          |

Useful commit landmarks:

- `6c9cdbc7`: initial cross-vertical backfill and linear L1 feature, including
  the now-rejected lookup design.
- `c13337b1`: native lookup-writer boundary and typed client score responses.
- `e1c821dd`: query/document combiner preservation.
- `fd199b0e`: broker/global fusion and shared formula on both levels.
- `80854214`: integration of upstream text-pruning fixes and scoring review.
- `8166940c`, `d2f2667b`: x86 minimal-core feature-boundary fixes; the second
  corrects the first and gates only the BMI2 writer.
- `7fa5d4cd`: independent `GetIndexInfo` schema-rendering fix.

The [candidate scoring document](../candidate-rescoring.md) describes the current
feature's API and semantics, but its lookup/preparation section is rejected.
The [performance review](../search-performance-review.md) records prior fixes
and tests; passing those tests does not validate the rejected architecture.

## Production facts and the incorrect diagnosis

Last verified during this continuation: all four shards and the broker were
ready on `ghcr.io/spacefrontiers/summa/summa-server:1.8.123`. The API-side
integration was already deployed, but candidate scoring capability on the old
Summa binary was version 0 and no learned model was enabled.

Production sparse fields `sparse_vectors` and
`short_document_sparse_embedding` use **BMP**, with reordering enabled.
Full-text intentionally uses **MaxScore**. This was checked in actual
`metadata.json` on `summa-server-fin`, `summa-server-fin2-s2`,
`summa-server-fin2-s3`, and `summa-server-fin2-s4`.

The previous agent incorrectly inferred sparse MaxScore from `GetIndexInfo` SDL,
which omitted BMP storage/reorder settings. Commit `7fa5d4cd` fixes that lossy
report and adds `index_info_schema_preserves_bmp_storage_and_reordering`.
It changes diagnostics/schema fingerprints, not stored indexes or ranking.
There is no production sparse-format migration to perform.

## Validation already completed

The eight-stage `full` harness passed with `RUST_TEST_THREADS=1` for the code
later committed as `7fa5d4cd`: formatting, focused Clippy, core/server/broker/tool
tests, native-without-sync and portable core checks, docs, server build, and
real-server broker E2E. Counts include 1,309 core unit tests, 63 server tests,
49 broker unit tests, 13 mock-broker integration tests and two real-server tests.
The [saved run manifest](2026-09-05-l1/full-harness.json) includes commands,
return codes, host/compiler, and the pre-commit dirty-diff identity. Full logs
remain in `.context/search-harness/20260905T150826.682593Z-full/` in the original
Summa checkout.

Two preceding parallel runs timed out in different mock-broker discovery tests.
The recovery test passed alone, then the complete serial run passed. Do not
report those failed runs as clean passes.

A separate fixture with two real local shards and a broker nominated only
through dense search and exported BM25, phrase, sparse and document features.
Zero/nonmatch and negative dense values survived. MAX, AVG and SUM top-1 matched
an independent full-union formula oracle. The winning scores were approximately
1.7098403, 1.1583292 and 2.3166585. The
[saved synthetic results](2026-09-05-l1/cross-vertical-smoke.json) are included.
This fixture used newly built BMP data; it is not production performance evidence
or validation of a replacement for the lookup design.

At handoff, CI run 33974102658 in the archived development repository
for `7fa5d4cd` had nine successful jobs, including Python, TypeScript and WASM,
while Rust CI was still in progress. Its final status was not assumed.

## Remaining Summa work

1. Review/redesign score completion to meet the user's no-new-lookup-maps
   constraint, using the existing index representations. Remove the rejected
   storage, lifecycle, readiness and preparation assumptions together.
2. Preserve logical passage alignment, missing versus zero, exact scoring of
   the nominated union, fixed/global statistics, query expression order,
   document combiners, hard eligibility and bounded resource use.
3. Validate the redesigned path on existing reordered BMP/text data as well as
   fresh data. Measure addressing/scoring cost, memory and IO on a representative
   corpus. No latency or scalability result exists for a replacement design.
4. Update docs and generated clients if the capability/protocol contract changes;
   coordinate the mirrored proto and capability checks in Azeroth.
5. Run the repository's required full/portable/RPC checks and complete CI/review.
   The original task includes PR, merge and `publish.yml`, but this handoff is
   not authorization to merge the rejected design unchanged.

No Summa release/publish, production rollout, training dataset collection,
teacher-labeling campaign, model fit or learned-model activation was completed.
The user is transferring engine ownership to a dedicated Summa agent.
