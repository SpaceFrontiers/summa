# Summa broker

`summa-broker` is a stateless gRPC service that fronts many `summa-server`
instances behind one address. It serves the exact `summa-proto/summa.proto`
`SearchService` and `IndexService`, so every existing client — the Rust,
Python, and TypeScript clients alike — switches to it by re-pointing its
endpoint, nothing else. A broker-only control surface lives in a separate
proto (`summa-proto/summa-broker.proto`) so the shared wire contract and
its generated clients never churn for broker concerns.

## Problem

One summa-server process serves all indexes from one data directory on one
machine. Large deployments need indexes on different hosts (two big indexes
that no longer fit one box), later partitions of one index across hosts, and
replicas for read scaling — all without teaching every client about topology.

## Topology model

- **Backend**: one summa-server process, discovered as a Kubernetes pod or a
  static `--backend` entry.
- **Shard**: the unit of placement, identified by the pod label
  `summa.spacefrontiers.org/shard-id` (`--shard-label`). Backends sharing a
  shard id are **replicas** of the same data.
- **Role**: `summa.spacefrontiers.org/role` = `master` | `follower`
  (`--role-label`). Writes go to the master only, never fan out. A shard
  whose only member is unlabeled is implicitly master — today's
  single-pod-per-shard world needs no labels. A multi-member shard with zero
  or several labeled masters refuses writes with `FAILED_PRECONDITION`
  (fail loud, no guessing).
- **Index → shard mapping is learned, not configured**: the broker polls
  `ListIndexes` on every ready backend (15s steady state, 5s while a backend
  is unhealthy, immediately after a broker-issued `CreateIndex`/
  `DeleteIndex`). What IS configured are **placement rules**:
  `--placement "documents*=0"` — glob → shard, first match wins — which
  govern where `CreateIndex` lands (dated names follow their family) and pin
  reads/writes when an index name transiently exists on several shards
  during a migration.

## Backend health

```
Healthy --poll failure--> Suspect --grace (60s) elapsed--> Evicted
Suspect --success--> Healthy
Evicted --2 consecutive successful probes--> Healthy
```

A Suspect backend keeps serving reads off its last-known index map (better a
possibly-stale answer than none; counted by
`summa_broker_stale_topology_serves_total`). An Evicted backend drops out of
every route: its indexes vanish from `ListIndexes` and reads return
`NOT_FOUND` if no other backend advertises them. Snapshots are immutable and
swapped atomically; request handlers never take a lock on the hot path.

## Routing (phase 1 — index-level)

| RPC                                                                                                               | Behavior                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Search`, `GetDocument`, `GetIndexInfo`                                                                           | Exact index → its shard → a healthy replica (rotating), request and response forwarded verbatim                                                                                                                                                                            |
| `ListIndexes`                                                                                                     | Union across routable backends, served from the cached topology (never fans out — this is every client's health probe and must answer fast)                                                                                                                                |
| `BatchIndexDocuments`, `Commit`, `ForceMerge`, `Reorder`, `RetrainVectorIndex`, `AlterVectorIndex`, `DeleteIndex` | Whole request to the **master** of the shard hosting the index; response verbatim                                                                                                                                                                                          |
| `IndexDocuments` (client-streaming)                                                                               | Buffered per index (512 docs / 4 MiB), forwarded as `BatchIndexDocuments`, re-routed on mid-stream `index_name` switches. `DocumentError.index` positions are flush-relative — the server's own stream handling already numbers per internal batch, so no fidelity is lost |
| `CreateIndex`                                                                                                     | Placement rule (or `--placement-default single`: the shard hosting the fewest indexes; `reject`: refuse) → that shard's master                                                                                                                                             |
| Unknown index                                                                                                     | `NOT_FOUND("index '…' is not present on any healthy backend")`                                                                                                                                                                                                             |
| Index on several shard ids without a rule                                                                         | Reads: lexicographically-first shard, deterministic, counted by `summa_broker_ambiguous_index_total`; writes: `FAILED_PRECONDITION` until a placement rule pins the writable shard                                                                                         |

Contract guarantees clients rely on:

- **Byte-faithful responses** on the write path: duplicate-primary-key and
  backpressure `DocumentError`s pass through untouched (client retry loops
  string-match them), `indexed_count`/`error_count` and error indices are the
  backend's own.
- **No broker-imposed deadlines.** The incoming `grpc-timeout` header is
  propagated minus a 50ms epsilon (floor 10ms); an absent header means the
  outbound RPC carries none. Untimed index-builder channels and 24h admin
  `Reorder`/`ForceMerge` deadlines work unchanged.
- **Admission mirrors the backend.** Per-backend in-flight Search permits
  (`--backend-max-searches`, default 16 = the production
  `--max-concurrent-searches`) plus an optional broker-global cap. Rejection
  is `RESOURCE_EXHAUSTED` with the server's exact message, so client backoff
  logic cannot tell broker and backend apart. Only genuine unavailability
  surfaces as `UNAVAILABLE` (it trips client circuit breakers).
- Transport limits and tuning mirror summa-server by default (search
  4 MiB decode / 256 MiB encode, index 256 MiB decode / 64 MiB encode,
  gzip+zstd). All six message caps are startup flags
  (`--search-max-decode-mb`, `--search-max-encode-mb`,
  `--index-max-decode-mb`, `--index-max-encode-mb`, plus
  `--backend-max-decode-mb` / `--backend-max-encode-mb` for the
  broker→backend channels); inconsistent combinations warn loudly at
  startup.

Coordinated searches (fusion, L1, score exports and traces) additionally use
`--coordinator-max-transfer-mb` (default **64 MiB**). This is the total
uncompressed response allowance for a search, not a per-shard allowance.
The decoder divides it by the number of target shards, even on compressed
connections. The same setting bounds combined responses, selected output and
RRF diagnostic retention. Requests exceeding a bound fail with
`RESOURCE_EXHAUSTED`; results are never silently truncated to fit.

The value must be nonzero, fit in bytes, and not exceed either
`--backend-max-decode-mb` or `--search-max-encode-mb`; invalid combinations fail
at startup. Clients and shard encoders must also accept the required messages.
Increasing generic gRPC limits alone does not raise the coordinator limit.
Startup logs report the coordinator allowance and both admission caps.

For a bounded increase, use `--coordinator-max-transfer-mb 128
--max-concurrent-searches 8`: three shards get about 42.7 MiB each. Eight
simultaneous coordinated searches have up to 1 GiB of incoming protobuf
payload, before decoded objects, ranking buffers and response copies. This
is a transfer bound, not a process RSS limit; measure broker memory and
admission rejections before expanding concurrency. This setting does not
change retrieval depth, formulas, candidate completeness or shard concurrency.

Pass-through responses are proto-equal, not always byte-equal: protobuf map
fields (`SearchHit.fields`) may re-serialize entries in a different order.

## Discovery

Kubernetes mode watches **Pods** (not EndpointSlices — shard identity and
role are pod labels, and the pod carries labels, IP, and readiness in one
object) in `--namespace` with a label-existence selector on the shard label.
Readiness = PodReady ∧ has IP ∧ not terminating; unready pods are visible in
the admin surface but never routed or polled. RBAC: `get/list/watch pods` in
the summa namespace. Static mode (`--discovery static --backend
"id=..,addr=..,shard=..[,role=..]"`) feeds the identical machinery and is
what local development and the integration tests use.

## Phase 2: partitioned indexes

One logical index across several shards, declared by a multi-shard placement
rule: `--placement "documents*=2,3,4"`. Partition order = rule order (an
immutable contract: repartitioning or reordering = full rebuild). Every
partition must host the index; a partition without it fails the request
with `FAILED_PRECONDITION` instead of serving a partial view.

Writes:

- `CreateIndex` creates the index on every partition master (the schema is
  sent verbatim to each).
- `BatchIndexDocuments` and streaming `IndexDocuments` route each document
  to the partition of a pinned FNV-1a 64 hash of its primary key (the field
  declared `primary` in the schema, read once via `GetIndexInfo` and cached
  per index). A document without the primary key is refused at the broker
  with its request position; `DocumentError.index` values from a partition
  are mapped back to request positions. The stream's 512-message / 4 MiB
  flushes are split per partition.
- `Commit`, `ForceMerge`, `Reorder`, `DeleteIndex`, `RetrainVectorIndex`
  and `AlterVectorIndex` go to every partition master; counts are summed,
  `success` is the conjunction.

Reads:

- `Search` first asks every partition for `GetTextStats` of the query's
  text terms (skipped for queries without BM25 terms or when the caller
  already supplied `text_stats`), sums them, and sends the sum as
  `SearchRequest.text_stats` so every partition scores with corpus-wide
  document frequencies and lengths. Plain pointwise searches query partitions
  with `offset=0, limit=offset+limit` and merge by score descending, ties by
  `(segment_id, doc_id)`. Top-level fusion without the legacy vector reranker
  exports the per-branch shard nomination lists and computes global RRF (or
  normalized weighted sum) using core fusion at the broker. L1 applies the same
  core model on shards and broker, retaining sufficient passage rows for the
  requested combiner. `formula_v1` compiles `l1.formula` once, preserves organic scores and supports optional
  missing-only backfill plus learned raw missing defaults; defaults never fill
  raw export maps. A formula referencing `rrf` requires the full bounded union
  and all scored passage rows from every shard; the broker applies the formula
  with global votes before selection. See [candidate scoring](candidate-rescoring.md) for
  transport bounds and mixed-version checks. Legacy fusion nested inside
  another query or combined with the vector reranker retains its existing
  shard execution; it does not claim global RRF candidate selection.
  `total_hits` = saturating sum, timings = maximum plus coordinator work,
  `truncated` = any (incomplete candidate exports are rejected). Admission
  takes one permit per partition backend and retains it through coordinator CPU work.
- `GetDocument` asks every partition; the one holding the segment answers.
- `GetIndexInfo` sums document/segment/memory counts and per-field stats;
  `GetTextStats` merges like the search prepass.

Partial partition failure fails the request (`partition '<shard>' of index
'<name>': <status>`) — a silently-partial result set is a wrong answer.
The admin `GetTopology` reports `merge_policy = "score"` and the cached
primary-key field for partitioned indexes.

## Phase 3 (designed, not yet built): master/follower replication

A new `ReplicationService` on summa-server (separate proto):
`GetIndexState` (metadata generation + segment metas), `FetchSegmentFile`
(chunk stream). A follower (`--replicate-from`) polls the master after
commits, pulls missing write-once segment files, atomically installs the new
`metadata.json`, and hot-reloads — which requires adding a reload-from-disk
path (`IndexReader::do_reload_check` currently only consults in-memory
segment-manager state). Followers are read-only; the broker already routes
writes to masters only, and spreads reads across master + fresh followers
with per-replica lag surfaced from `GetIndexState` generations. There is no
oplog: a follower that diverges beyond the client's retry horizon is rebuilt.

## Operations

- Health: `grpc.health.v1` on the broker itself — `SERVING` once the first
  topology snapshot has ≥1 healthy backend, `NOT_SERVING` while draining.
  Kubernetes gRPC probes work out of the box.
- Admin: `summa.broker.BrokerService` — `GetTopology` (per-replica live
  `num_docs`/`num_segments`, for migration verification), `GetBackends`,
  `RefreshTopology`.
- Metrics: `summa_broker_*`, documented in [metrics.md](metrics.md).
- Shutdown mirrors summa-server: SIGTERM → refuse new RPCs with
  `UNAVAILABLE("Summa broker is shutting down")`, drain, stop.

## Testing

- Unit: topology assembly, placement globs, master validation, ambiguity
  rules, health transitions, `grpc-timeout` parsing (all pure functions).
- Integration (`tests/broker_integration.rs`): the real broker binary with
  static discovery against in-process mock backends — pass-through equality,
  index routing, ambiguity + placement pinning, stream re-grouping, deadline
  presence/absence at the backend, eviction and two-probe recovery.
- End-to-end (`tests/e2e_real_server.rs`, `--ignored`, CI runs it after
  building summa-server): two real summa-servers, placement-routed
  `CreateIndex`, batch write + commit, duplicate-primary-key pass-through,
  search + `GetDocument` by address, cross-shard isolation.
