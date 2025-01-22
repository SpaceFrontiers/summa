# Hermes Client

Async Python client for [Hermes](https://github.com/SpaceFrontiers/hermes) search server.

## Installation

```bash
pip install hermes-client-python
```

## Quick Start

```python
import asyncio
from hermes_client_python import HermesClient

async def main():
    async with HermesClient("localhost:50051") as client:
        # Create index with SDL schema
        await client.create_index("articles", '''
            index articles {
                title: text indexed stored
                body: text indexed stored
                score: f64 stored
            }
        ''')

        # Index documents
        await client.index_documents("articles", [
            {"title": "Hello World", "body": "First article", "score": 1.5},
            {"title": "Goodbye World", "body": "Last article", "score": 2.0},
        ])

        # Commit changes
        await client.commit("articles")

        # Search
        results = await client.search("articles", term=("title", "hello"), limit=10)
        for hit in results.hits:
            print(f"Doc {hit.doc_id}: score={hit.score}, fields={hit.fields}")

        # Get document by ID
        doc = await client.get_document("articles", 0)
        print(doc.fields)

        # Delete index
        await client.delete_index("articles")

asyncio.run(main())
```

## API Reference

### HermesClient

```python
client = HermesClient(address="localhost:50051")
```

#### Connection

```python
# Using context manager (recommended)
async with HermesClient("localhost:50051") as client:
    ...

# Manual connection
client = HermesClient("localhost:50051")
await client.connect()
# ... use client ...
await client.close()
```

#### Index Management

```python
# Create index with SDL schema
await client.create_index("myindex", '''
    index myindex {
        title: text indexed stored
        body: text indexed stored
    }
''')

# Create index with JSON schema
await client.create_index("myindex", '''
{
    "fields": [
        {"name": "title", "type": "text", "indexed": true, "stored": true},
        {"name": "body", "type": "text", "indexed": true, "stored": true}
    ]
}
''')

# Get index info
info = await client.get_index_info("myindex")
print(f"Documents: {info.num_docs}, Segments: {info.num_segments}")

# Delete index
await client.delete_index("myindex")
```

#### Document Indexing

```python
# Index multiple documents (batch)
indexed, errors = await client.index_documents("myindex", [
    {"title": "Doc 1", "body": "Content 1"},
    {"title": "Doc 2", "body": "Content 2"},
])

# Index single document
await client.index_document("myindex", {"title": "Doc", "body": "Content"})

# Stream documents (for large datasets)
async def doc_generator():
    for i in range(10000):
        yield {"title": f"Doc {i}", "body": f"Content {i}"}

count = await client.index_documents_stream("myindex", doc_generator())

# Commit changes (required to make documents searchable)
num_docs = await client.commit("myindex")

# Force merge segments (for optimization)
num_segments = await client.force_merge("myindex")
```

#### Searching

```python
# Term query
results = await client.search("myindex", term=("title", "hello"), limit=10)

# Boolean query
results = await client.search("myindex", boolean={
    "must": [("title", "hello")],
    "should": [("body", "world")],
    "must_not": [("title", "spam")],
})

# With pagination
results = await client.search("myindex", term=("title", "hello"), limit=10, offset=20)

# With field loading
results = await client.search(
    "myindex",
    term=("title", "hello"),
    fields_to_load=["title", "body"]
)

# Access results
for hit in results.hits:
    print(f"Doc {hit.doc_id}: {hit.score}")
    print(f"  Title: {hit.fields.get('title')}")

print(f"Total hits: {results.total_hits}")
print(f"Took: {results.took_ms}ms")
```

#### Document Retrieval

```python
# Get document by ID
doc = await client.get_document("myindex", doc_id=42)
if doc:
    print(doc.fields["title"])
```

## Field Types

| Type            | Python Type     | Description                           |
| --------------- | --------------- | ------------------------------------- |
| `text`          | `str`           | Full-text searchable string           |
| `u64`           | `int` (>= 0)    | Unsigned 64-bit integer               |
| `i64`           | `int`           | Signed 64-bit integer                 |
| `f64`           | `float`         | 64-bit floating point                 |
| `bytes`         | `bytes`         | Binary data                           |
| `json`          | `dict` / `list` | JSON object (auto-serialized)         |
| `dense_vector`  | `list[float]`   | Dense vector for semantic search      |
| `sparse_vector` | `dict`          | Sparse vector with indices and values |

## Error Handling

```python
import grpc

try:
    await client.search("nonexistent", term=("field", "value"))
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.NOT_FOUND:
        print("Index not found")
    else:
        raise
```

## Development

Generate protobuf stubs:

```bash
pip install grpcio-tools
python generate_proto.py
```

## License

MIT
