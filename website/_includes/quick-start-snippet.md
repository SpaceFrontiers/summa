Let's create a small searchable collection with Summa 2. We need the server,
which owns the index, and a client that sends indexing and search requests.

### Start the server

The current container image is published by SpaceFrontiers in GitHub Container
Registry. It listens for gRPC requests on port 50051 and stores its indexes in
`/data`:

```bash
docker pull ghcr.io/spacefrontiers/summa/summa-server:2.0.0
mkdir -p data
docker run --rm -p 50051:50051 -v "$PWD/data:/data" \
  ghcr.io/spacefrontiers/summa/summa-server:2.0.0 summa-server \
  --data-dir /data --addr 0.0.0.0:50051
```

Alternatively, build and run the Rust package:

```bash
cargo install summa-server --version 2.0.0
summa-server --data-dir ./data --addr 127.0.0.1:50051
```

### Install the Python client

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install summa-client-python==2.0.0
```

### Index and search

Save the following as `search.py`. The schema declares a stored, searchable text
field. Indexing stages documents; `commit` makes them visible to searches.

```python
import asyncio

from summa_client_python import SummaClient


async def main():
    async with SummaClient("localhost:50051") as client:
        await client.create_index(
            "articles",
            """
            index articles {
                field id: text<raw> [primary, stored]
                field title: text<simple> [indexed, stored]
                field body: text<simple> [indexed, stored]
            }
            """,
        )

        indexed, error_count, errors = await client.index_documents(
            "articles",
            [
                {"id": "1", "title": "Hello World", "body": "First article"},
                {"id": "2", "title": "Summa Search", "body": "Fast retrieval"},
            ],
        )
        if error_count:
            raise RuntimeError(errors)
        print(f"Indexed {indexed} documents")

        await client.commit("articles")

        results = await client.search(
            "articles",
            query={"match": {"field": "title", "text": "hello"}},
            fields_to_load=["title", "body"],
        )
        for hit in results.hits:
            print(hit.address, hit.score, hit.fields)

        if results.hits:
            document = await client.get_document("articles", results.hits[0].address)
            print(document.fields if document else "document not found")


asyncio.run(main())
```

Run it with `python search.py`. The client connects to the local server and
prints the matching article. Check indexing errors before committing: an RPC
completing does not mean every supplied document was accepted.

See the [Python client guide](https://github.com/SpaceFrontiers/summa/blob/main/summa-client-python/README.md)
for updates, deletions and streaming, and [server operations](https://github.com/SpaceFrontiers/summa/blob/main/summa-server/README.md)
for resource limits. Existing Summa 0.x applications should follow the
[migration guide](https://github.com/SpaceFrontiers/summa/blob/main/docs/summa-2-migration.md);
the old YAML configuration and `aiosumma` CLI are not the Summa 2 interface.
