# Hermes Server

A high-performance gRPC search server for Hermes indexes.

## Features

- **Index Management**: Create, delete, and manage search indexes
- **Document Indexing**: Stream or batch index documents
- **Full-Text Search**: Term queries, boolean queries, and boosting
- **Document Retrieval**: Get documents by ID
- **Segment Management**: Commit changes and force merge segments

## Installation

```bash
cargo install hermes-server
```

Or build from source:

```bash
cargo build --release -p hermes-server
```

## Usage

### Starting the Server

```bash
hermes-server --addr 0.0.0.0:50051 --data-dir ./data
```

Options:

- `-a, --addr`: Address to bind to (default: `0.0.0.0:50051`)
- `-d, --data-dir`: Directory for storing indexes (default: `./data`)

### Background merge and reorder

The server uses one BP CPU pool and one whole-pass gate across all indexes.
These are deliberately separate controls:

| Option                                   |   Default | Meaning                                                                                                                                  |
| ---------------------------------------- | --------: | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `--optimizer-threads`                    |       `0` | Threads in the shared BP pool. `0` disables periodic optimizer scans; merge-time and manual BP still use the process-wide fallback pool. |
| `--optimizer-concurrent-passes`          |       `2` | Maximum simultaneous whole-segment BP passes across background, merge-time, and manual reorder. `0` is invalid and is clamped to `1`.    |
| `--optimizer-scan-interval-secs`         |      `60` | Interval between background scans.                                                                                                       |
| `--optimizer-large-segment-docs`         | `5000000` | Document threshold for partial/budgeted first passes.                                                                                    |
| `--optimizer-time-budget-secs`           |     `600` | Wall-clock budget for an optimizer pass on a large segment.                                                                              |
| `--optimizer-partial-min-partition-docs` |    `4096` | Initial depth floor for large segments.                                                                                                  |
| `--optimizer-unconverged-cooldown-secs`  |     `600` | Delay after a rewrite finishes before another deepening pass.                                                                            |
| `--optimizer-max-unconverged-passes`     |       `3` | Optimizer follow-up eligibility limit per truncated lineage, including the initial partial pass. `0` disables follow-up deepening.       |
| `--merge-bp-budget-secs`                 |     `600` | Wall-clock budget for BP performed inside a merge; `0` means unbudgeted.                                                                 |
| `--bp-memory-budget-mb`                  |   `24576` | Per-pass algorithmic working-set bound; not a reservation or a total-process RSS limit.                                                  |

An active BP pass is CPU-bound and is expected to occupy up to
`--optimizer-threads` cores. Concurrent passes share that same pool, so raising
the pass limit primarily raises simultaneous working sets and outstanding IO;
it does not create another pool per pass or per index. For predictable service
latency, start with one pass and a CPU width that leaves capacity for query and
indexing work.

The dominant graph representation is roughly `4 bytes/posting + 32
bytes/document`. The limit also accounts for record maps, vocabulary-sized
degree arrays, and record-rewrite grid/encode windows. If the record
representation cannot fit, Hermes performs a valid blockwise rewrite and marks
it unconverged; if only the graph is too large, it retains a bounded set of
low-frequency dimensions. Stored postings are never truncated. At the process
level, still budget for up to `concurrent-passes * bp-memory-budget`, plus
indexing builders, merge state, mmap/page-cache residency, output buffering,
and open readers.

Merge failures use exponential retry backoff (30 seconds through 30 minutes).
A deterministic missing/corrupt source is quarantined for the process lifetime
so the same candidate cannot consume all cores in an immediate loop. The
metadata entry remains visible—Hermes never silently removes documents. To
explicitly remove corrupt entries and their files, stop normal traffic and run:

```bash
hermes-server --data-dir ./data --doctor
```

`--doctor` is destructive recovery: it validates every metadata-live segment
and removes entries that cannot be opened. Normal startup/writer-open cleanup
only deletes true unowned files and cannot delete metadata-live, actively
written, or reader-retained segments. Standalone reorder failures are also
backed off per source from pass completion, so a pass that outlasts the scan
interval cannot restart continuously. Successful but budget-truncated outputs
carry a retry count across replacement IDs and stop being optimizer candidates
at `--optimizer-max-unconverged-passes`, so a lineage that never converges
cannot consume the optimizer pool forever. Explicitly configured merge-time BP
still runs when that lineage later participates in a real merge. Details and
invariants are in [Segment lifecycle and recovery](../docs/segment-lifecycle.md).

## gRPC API

The server exposes two services: `SearchService` and `IndexService`.

### IndexService

#### CreateIndex

Create a new index with a schema definition (SDL or JSON format).

```protobuf
rpc CreateIndex(CreateIndexRequest) returns (CreateIndexResponse);

message CreateIndexRequest {
  string index_name = 1;
  string schema = 2;  // SDL or JSON schema
}
```

**SDL Schema Example:**

```
index articles {
    title: text indexed stored
    body: text indexed stored
    author: text indexed stored
    published_at: u64 indexed stored
    tags: text indexed stored
}
```

**JSON Schema Example:**

```json
{
  "fields": [
    { "name": "title", "type": "text", "indexed": true, "stored": true },
    { "name": "body", "type": "text", "indexed": true, "stored": true },
    { "name": "score", "type": "f64", "indexed": false, "stored": true }
  ]
}
```

#### BatchIndexDocuments

Index multiple documents in a single request.

```protobuf
rpc BatchIndexDocuments(BatchIndexDocumentsRequest) returns (BatchIndexDocumentsResponse);

message BatchIndexDocumentsRequest {
  string index_name = 1;
  repeated NamedDocument documents = 2;
}

message NamedDocument {
  map<string, FieldValue> fields = 1;
}
```

#### IndexDocuments (Streaming)

Stream documents for indexing.

```protobuf
rpc IndexDocuments(stream IndexDocumentRequest) returns (IndexDocumentsResponse);
```

#### Commit

Commit pending changes to make them searchable.

```protobuf
rpc Commit(CommitRequest) returns (CommitResponse);
```

#### ForceMerge

Merge all segments into one for optimal search performance.

```protobuf
rpc ForceMerge(ForceMergeRequest) returns (ForceMergeResponse);
```

#### DeleteIndex

Delete an index and all its data.

```protobuf
rpc DeleteIndex(DeleteIndexRequest) returns (DeleteIndexResponse);
```

### SearchService

#### Search

Search for documents matching a query.

```protobuf
rpc Search(SearchRequest) returns (SearchResponse);

message SearchRequest {
  string index_name = 1;
  Query query = 2;
  uint32 limit = 3;
  uint32 offset = 4;
  repeated string fields_to_load = 5;
}
```

**Query Types:**

- **TermQuery**: Match a specific term in a field
- **BooleanQuery**: Combine queries with must/should/must_not
- **BoostQuery**: Boost the score of a query

#### GetDocument

Retrieve a document by its ID.

```protobuf
rpc GetDocument(GetDocumentRequest) returns (GetDocumentResponse);
```

#### GetIndexInfo

Get information about an index (document count, segments, schema).

```protobuf
rpc GetIndexInfo(GetIndexInfoRequest) returns (GetIndexInfoResponse);
```

## Field Types

| Type            | Description                       |
| --------------- | --------------------------------- |
| `text`          | Full-text searchable string       |
| `u64`           | Unsigned 64-bit integer           |
| `i64`           | Signed 64-bit integer             |
| `f64`           | 64-bit floating point             |
| `bytes`         | Binary data                       |
| `json`          | JSON object (stored as string)    |
| `sparse_vector` | Sparse vector for semantic search |
| `dense_vector`  | Dense vector for semantic search  |

## Example: Python Client

```python
import grpc
from hermes_pb2 import *
from hermes_pb2_grpc import IndexServiceStub, SearchServiceStub

channel = grpc.insecure_channel('localhost:50051')
index_service = IndexServiceStub(channel)
search_service = SearchServiceStub(channel)

# Create index
schema = '''
index articles {
    title: text indexed stored
    body: text indexed stored
}
'''
index_service.CreateIndex(CreateIndexRequest(
    index_name="articles",
    schema=schema
))

# Index documents
docs = [
    NamedDocument(fields={
        "title": FieldValue(text="Hello World"),
        "body": FieldValue(text="This is my first article")
    }),
    NamedDocument(fields={
        "title": FieldValue(text="Goodbye World"),
        "body": FieldValue(text="This is my last article")
    })
]
index_service.BatchIndexDocuments(BatchIndexDocumentsRequest(
    index_name="articles",
    documents=docs
))

# Commit
index_service.Commit(CommitRequest(index_name="articles"))

# Search
response = search_service.Search(SearchRequest(
    index_name="articles",
    query=Query(term=TermQuery(field="title", term="hello")),
    limit=10,
    fields_to_load=["title", "body"]
))

for hit in response.hits:
    print(f"Doc {hit.doc_id}: {hit.score} - {hit.fields}")
```

## Docker

Build and run with Docker:

```bash
docker build -t hermes-server -f hermes-server/Dockerfile .
docker run -p 50051:50051 -v ./data:/data hermes-server --data-dir /data
```

Or pull from GitHub Container Registry:

```bash
docker pull ghcr.io/spacefrontiers/hermes/hermes-server:latest
docker run -p 50051:50051 -v ./data:/data ghcr.io/spacefrontiers/hermes/hermes-server:latest --data-dir /data
```

## License

MIT
