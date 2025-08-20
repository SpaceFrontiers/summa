# Hermes broker

`hermes-broker` is a stateless gRPC service that fronts many `hermes-server`
instances behind one address. It serves the exact `hermes-proto/hermes.proto`
`SearchService` and `IndexService`, so every existing client — the Rust,
Python, and TypeScript clients alike — switches to it by re-pointing its
endpoint, nothing else. A broker-only control surface lives in a separate
proto (`hermes-proto/hermes-broker.proto`) so the shared wire contract and
its generated clients never churn for broker concerns.

## Problem

One hermes-server process serves all indexes from one data directory on one
machine. Large deployments need indexes on different hosts (two big indexes
that no longer fit one box), later partitions of one index across hosts, and
replicas for read scaling — all without teaching every client about topology.

## Topology model

- **Backend**: one hermes-server process, discovered as a Kubernetes pod or a
  static `--backend` entry.
- **Shard**: the unit of placement, identified by the pod label
  `hermes.spacefrontiers.org/shard-id` (`--shard-label`). Backends sharing a
  shard id are **replicas** of the same data.
- **Role**: `hermes.spacefrontiers.org/role` = `master` | `follower`
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
`hermes_broker_stale_topology_serves_total`). An Evicted backend drops out of
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
| Index on several shard ids without a rule                                                                         | Reads: lexicographically-first shard, deterministic, counted by `hermes_broker_ambiguous_index_total`; writes: `FAILED_PRECONDITION` until a placement rule pins the writable shard                                                                                        |

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
- Transport limits and tuning mirror hermes-server by default (search
  4 MiB decode / 256 MiB encode, index 256 MiB decode / 64 MiB encode,
  gzip+zstd). All six message caps are startup flags
  (`--search-max-decode-mb`, `--search-max-encode-mb`,
  `--index-max-decode-mb`, `--index-max-encode-mb`, plus
  `--backend-max-decode-mb` / `--backend-max-encode-mb` for the
  broker→backend channels); inconsistent combinations warn loudly at
  startup.

Pass-through responses are proto-equal, not always byte-equal: protobuf map
fields (`SearchHit.fields`) may re-serialize entries in a different order.

## Discovery

Kubernetes mode watches **Pods** (not EndpointSlices — shard identity and
role are pod labels, and the pod carries labels, IP, and readiness in one
object) in `--namespace` with a label-existence selector on the shard label.
Readiness = PodReady ∧ has IP ∧ not terminating; unready pods are visible in
the admin surface but never routed or polled. RBAC: `get/list/watch pods` in
the hermes namespace. Static mode (`--discovery static --backend
"id=..,addr=..,shard=..[,role=..]"`) feeds the identical machinery and is
what local development and the integration tests use.

## Phase 2 (designed, not yet built): partitioned indexes

One logical index across several shards. Fan out per-shard with
`offset=0, limit=offset+limit` (candidate_limit forwards unchanged — the
per-shard window keeps it valid; multi-shard windows above the server's
10 000 limit cap are rejected). Merge policy: RRF over shard-local ranks when
the query is rank-shaped (top-level `FusionQuery`, `reranker.rrf_k > 0`,
`PrefixQuery` — their scores are shard-local), score merge for pure
dense/binary similarity (corpus-independent). Dedup key `(segment_id,
doc_id)` — segment ids are UUIDv7-like, collision-safe across shards.
`total_hits` = saturating sum ("documents scored", as ever, not matched).
Writes route documents by a pinned FNV-1a 64 hash of the primary key;
partition order = placement-rule order (an immutable contract);
repartitioning = full rebuild. Partial shard failure fails the request —
a silently-partial result set is a wrong answer.

## Phase 3 (designed, not yet built): master/follower replication

A new `ReplicationService` on hermes-server (separate proto):
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
- Admin: `hermes.broker.BrokerService` — `GetTopology` (per-replica live
  `num_docs`/`num_segments`, for migration verification), `GetBackends`,
  `RefreshTopology`.
- Metrics: `hermes_broker_*`, documented in [metrics.md](metrics.md).
- Shutdown mirrors hermes-server: SIGTERM → refuse new RPCs with
  `UNAVAILABLE("Hermes broker is shutting down")`, drain, stop.

## Testing

- Unit: topology assembly, placement globs, master validation, ambiguity
  rules, health transitions, `grpc-timeout` parsing (all pure functions).
- Integration (`tests/broker_integration.rs`): the real broker binary with
  static discovery against in-process mock backends — pass-through equality,
  index routing, ambiguity + placement pinning, stream re-grouping, deadline
  presence/absence at the backend, eviction and two-probe recovery.
- End-to-end (`tests/e2e_real_server.rs`, `--ignored`, CI runs it after
  building hermes-server): two real hermes-servers, placement-routed
  `CreateIndex`, batch write + commit, duplicate-primary-key pass-through,
  search + `GetDocument` by address, cross-shard isolation.