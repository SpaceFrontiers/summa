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
- Partitions an index across shards with a shard list:
  `--placement "documents*=2,3,4"`. Documents hash by primary key to one
  partition; searches fan out with shared BM25 statistics and merge by score;
  every other RPC fans out to all partitions. See `docs/broker.md`.
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

## Container image

The published Hermes image contains both the server and broker binaries. It
starts `hermes-server` by default; select the broker with a command override:

```bash
docker run --rm -p 50051:50051 \
  ghcr.io/spacefrontiers/hermes/hermes-server:latest \
  hermes-broker --discovery static \
  --backend "id=local,addr=10.0.0.10:50051,shard=0"
```
