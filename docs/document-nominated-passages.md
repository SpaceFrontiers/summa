# Passage evidence for document nominations

Opt-in candidate-scoring capability 4. No storage-format or default
retrieval change; capability 3 requests retain their existing results.

Combining document-profile and body nominations must not leave a document-only
candidate without real passage features. `ScoreExport.seed_document_passages`
requests missing-only passage seeding, and requires backfill plus a chunk-scoped
feature. It is independent of response passage limits and `all_passages`.

For candidates with no organically nominated body ordinal, enumerate the real
ordinals of the plan's chunk fields, then score those rows with the existing
candidate scorer. Never manufacture ordinal zero from a document-profile hit.
Candidates already carrying body ordinals retain exactly their nominated union.
Missing body fields remain missing; callers must handle genuinely bodyless rows.
Seeded rows do not acquire organic RRF votes.

Enumeration and scoring use the existing logical-address owner, request-wide
feature/value, sparse-payload and vector-byte budgets. Enumerate and charge
locations before retaining feature rows. A budget failure errors, never silently
cuts a document's evidence. Cost is proportional to the stored passages of
document-only nominees, not the corpus; payload remains evictable. The existing
model scores every admitted row before reducing documents and exporting top
passages. Raw exports retain the complete admitted union.

The server validates the option before admission, advertises capability 4, and
the broker forwards it unchanged while computing global RRF. Search API must
gate use on that capability and bind the option into its new retrieval contract.
Legacy models must not be relabeled as trained on the expanded union.

`SearchResponse.seeded_document_passages` acknowledges the requested policy.
The broker rejects shards omitting the acknowledgment and merges it only when
every shard confirms. Clients require it when requesting seeding: old adapters
that drop new protobuf fields fail closed during a mixed rollout.

Validation covers a profile-only nominee with its best evidence at ordinal one,
preserved organic passage restrictions, missing fields, disabled-backfill
rejection, raw export/formula parity, and native/current-thread execution.
