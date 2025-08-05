# hermes-broker

A stateless gRPC broker that fronts many `hermes-server` instances behind
one address. It serves the exact `hermes.proto` `SearchService` and
`IndexService` — existing clients switch by re-pointing their endpoint —
plus the broker-only `hermes.broker.BrokerService` (topology inspection) and
`grpc.health.v1`.

Design doc: [docs/broker.md](../docs/broker.md). Metrics:
[docs/metrics.md](../docs/metrics.md).

## What it does

- Discovers hermes-server backends: Kubernetes pod watch (shard identity from
  the `hermes.spacefrontiers.org/shard-id` pod label, replica role from
  `hermes.spacefrontiers.org/role`) or a static `--backend` list.
- Learns which backend hosts which index by polling `ListIndexes`.
- Routes every RPC for an index to the backend hosting it; writes go to the
  shard's master only. Responses are forwarded verbatim.
- Places new indexes by glob rules: `--placement "documents*=0"` (first match
  wins), so dated index families stay on their shard. The same rules pin
  reads/writes when an index name transiently exists on two shards during a
  host-to-host migration.
- Evicts unreachable backends after a grace period and recovers them after
  two successful probes; propagates client deadlines untouched (no
  broker-imposed timeouts); mirrors hermes-server's admission, message-size
  limits, and transport tuning.

## Run locally

```sh
cargo run -p hermes-broker -- \
  --discovery static \
  --backend "id=local,addr=127.0.0.1:50051,shard=0" \
  --placement "documents*=0"
```

## Tests

```sh
cargo test -p hermes-broker                      # unit + mock-backend integration
cargo build -p hermes-server --bin hermes-server # e2e prerequisite
cargo test -p hermes-broker --test e2e_real_server -- --ignored
```
