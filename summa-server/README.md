# Summa Server

gRPC search, indexing, and maintenance for multiple local indexes.

## Run

```bash
cargo install summa-server
summa-server --addr 127.0.0.1:50051 --data-dir ./data
summa-server --help
```

Defaults: `0.0.0.0:50051`, data directory `./data`. See
[Python](../summa-client-python/README.md) and
[TypeScript](../summa-client-typescript/README.md) for client examples,
[metrics](../docs/metrics.md) for monitoring, and
[the broker](../summa-broker/README.md) for sharding.

## Operations

### Search resource controls

`--search-threads` bounds the shared CPU pool, including nested and cross-index
work. Default: detected CPUs / 4, minimum 1.

`--max-concurrent-searches` bounds admitted search pipelines. Default: CPUs / 8,
clamped to 1–8. Overload fails immediately with `RESOURCE_EXHAUSTED`; retry with
bounded backoff. Completion/cancellation releases admission. Document lookup and
metadata RPCs do not consume search permits.

Request limits:

| Request component                           |            Limit |
| ------------------------------------------- | ---------------: |
| Final search results                        |          `10000` |
| Pagination window (`offset + limit`)        |          `50000` |
| L1 reranker candidates                      |          `50000` |
| Fusion candidates fetched per sub-query     |          `50000` |
| Fusion sub-queries                          |             `16` |
| Fusion fetch depth x number of sub-queries  |         `200000` |
| Query nesting depth                         |             `32` |
| Query nodes / aggregate clauses             |    `256` / `512` |
| Clauses in one Boolean query                |            `128` |
| Aggregate query text                        |         `64 KiB` |
| Aggregate query vector payload              |          `1 MiB` |
| Dense dimensions / sparse input dimensions  | `65536` / `4096` |
| Binary query bytes                          |        `256 KiB` |
| Stored fields requested                     |             `64` |
| Aggregate requested-field name bytes        |         `16 KiB` |
| Retained response / encoded response (each) |         `48 MiB` |

Zero-valued defaults remain supported. Derived reranker and fusion defaults are
checked and capped at the corresponding limit; explicit values over a limit
return gRPC `INVALID_ARGUMENT`.

The structural limits are checked iteratively before query conversion and
before a search permit is acquired. Requested stored fields are resolved and
deduplicated once, and response hydration is charged before field values are
cloned into protobuf objects. A response that would exceed its memory/encoding
budget fails with `RESOURCE_EXHAUSTED`; request fewer hits or fields.

### Commit completion and timeouts

Once a `Commit` acquires the index writer, the server owns its completion even
if the client disconnects or its RPC deadline expires. A slow worker flush is
reported every five minutes and automatically retried against the same paused
generation; it never resumes workers early or publishes only the fast workers.
Concurrent commits serialize on that writer. Cancellation while waiting to
acquire the writer does not start a commit.

A client deadline is therefore an unknown outcome, not an abort. Retry `Commit`
to observe completion. Publication/build errors remain errors and are logged
even if the original client is gone; only worker-flush timeouts are retried
automatically. Backpressure messages distinguish a full queue from a paused
commit. Graceful shutdown waits for an accepted commit to finish, but a forced
process kill cannot preserve that guarantee: allow sufficient termination grace
and stop ingestion/commit outstanding work before a deployment restart.

### Background merge and reorder

The server uses one BP CPU pool and one whole-pass gate across all indexes.
These are deliberately separate controls:

| Option                                   |   Default | Meaning                                                                                 |
| ---------------------------------------- | --------: | --------------------------------------------------------------------------------------- |
| `--optimizer-threads`                    |       `0` | Shared BP threads; `0` disables periodic scans. Manual/merge BP uses the fallback pool. |
| `--optimizer-concurrent-passes`          |       `2` | Shared whole-pass limit, clamped to 1–2; automatic merges use at most one slot.         |
| `--optimizer-scan-interval-secs`         |      `60` | Background scan interval.                                                               |
| `--optimizer-large-segment-docs`         | `5000000` | Large-segment threshold for budgeted first passes.                                      |
| `--optimizer-time-budget-secs`           |     `600` | Large-segment pass time budget.                                                         |
| `--optimizer-partial-min-partition-docs` |     `256` | Initial partition floor for large segments.                                             |
| `--optimizer-unconverged-cooldown-secs`  |     `600` | Completion-to-retry delay for deepening.                                                |
| `--optimizer-max-unconverged-passes`     |       `3` | Pass limit per truncated lineage, including the first pass; `0` disables follow-up.     |
| `--merge-bp-budget-secs`                 |     `600` | Merge BP time budget; `0` means unbudgeted.                                             |
| `--bp-memory-budget-mb`                  |   `24576` | Per-pass scratch bound, not reserved memory or a process RSS cap.                       |

Each pass shares the CPU pool but has its own memory budget. Budget for
`concurrent-passes * bp-memory-budget`, plus readers, indexing, merge state,
and page-cache residency. Over-budget BP uses blockwise reorder or a bounded
graph and reports incomplete convergence; stored postings are never truncated.
Force merge pauses new background BP and reserves foreground capacity after
existing merges drain.

Merge failures use exponential retry backoff (30 seconds through 30 minutes).
A deterministic missing/corrupt source is quarantined for the process lifetime
so the same candidate cannot consume all cores in an immediate loop. The
metadata entry remains visible—Summa never silently removes documents. To
explicitly remove corrupt entries and their files, stop normal traffic and run:

```bash
summa-server --data-dir ./data --doctor
```

`--doctor` removes metadata entries and files that cannot be opened. Normal
cleanup removes only unowned files. Reorder failures back off; truncated outputs
stop optimizer retries at `--optimizer-max-unconverged-passes`. See the
[segment lifecycle contract](../docs/segment-lifecycle.md).

## gRPC API

The [protocol](../summa-proto/summa.proto) defines the complete wire API:

| Service         | Operations                                                                                                                        |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `SearchService` | Search, retrieve documents, inspect indexes, export BM25 statistics                                                               |
| `IndexService`  | Create/list/delete indexes; batch/stream ingest; delete/upsert rows; commit; merge/compact; reorder; retrain/alter vector indexes |

Search supports text, phrase, Boolean, range, prefix, sparse/dense/binary vector,
fusion, formula ranking, and reranking. Use [SDL](../docs/schema.md) or the
shared JSON schema to create indexes. Document addresses are snapshot-local;
use primary keys for durable identity.

`RetrainVectorIndex` rebuilds ANN segments and publishes the new global artifacts
atomically. Training samples are bounded per field by both
`--vector-training-max-samples` (10,000,000) and
`--vector-training-memory-mb` (4096). `AlterVectorIndex` switches compatible
fields between IVF and ScaNN; see [vector configuration](../docs/schema.md#dense-vectors).

## Row compaction

`ForceMerge` retains tombstones by default. Set `compact: true` to physically
remove deleted rows from its final output, including a singleton. `GetIndexInfo`
returns `physical_num_docs`, `num_deleted_docs`, and `deleted_ratio`; `num_docs`
counts live rows. Broker responses aggregate counts before computing the ratio.

The existing optimizer (`--optimizer-threads > 0`) also compacts segments with
at least `--optimizer-compaction-deleted-ratio` deleted rows (default 0.30;
0 disables compaction). It uses the existing task slots, maintenance capacity,
whole-pass gate and CPU pool. Only one automatic compaction can run globally;
`--optimizer-compaction-cooldown-secs` (default 60) starts at completion.
`--compaction-memory-budget-mb` bounds scratch for manual/API and automatic
compaction (default 256 MiB). Busy/foreground-owned segments are skipped; failures
use optimizer backoff. Segments selected for compaction are excluded from BP work
in the same scan. This applies to indexes without reorder fields too.

Compaction preserves surviving row/BMP record order and the `reordered` flag,
but changes BMP block membership, so it invalidates convergence on previously
reordered nonempty BMP layouts. The BP attempt count is retained. See
[the compaction contract](../docs/row-deletion.md) for details.

## Primary-key deletion and upserts

IndexService exposes `DeleteDocuments { index_name, primary_keys }` and
`UpsertDocuments { index_name, documents }`. Both stage mutations and return
`{ accepted_count, errors }`; `Commit` atomically publishes accepted work and
reloads the reader. Deletion hides every chunk of a matching document. Upserts
replace the complete document and insert missing keys. Batch errors keep their
original positions; missing deletes are accepted. Requests require a primary-key
schema and are bounded before conversion/admission. The broker routes both
operations to the same partition as ingestion. See
[the mutation contract](../docs/row-deletion.md#mutation-surfaces) for limits and
failure/cancellation semantics.

## Docker

Build the server-only image from the repository root:

```bash
docker build -t summa-server -f summa-server/Dockerfile .
docker run --rm -p 50051:50051 -v "$PWD/data:/data" summa-server --data-dir /data
```

The published image contains both server and broker binaries:

```bash
docker run --rm -p 50051:50051 -v "$PWD/data:/data" \
  ghcr.io/spacefrontiers/summa/summa-server:latest \
  summa-server --data-dir /data
```

## Development checks

```bash
python3 scripts/check_search.py full
```
