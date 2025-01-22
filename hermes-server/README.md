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
