"""Async Hermes client implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import grpc
from grpc import aio

from . import hermes_pb2 as pb
from . import hermes_pb2_grpc as pb_grpc
from .types import Document, IndexInfo, SearchHit, SearchResponse


class HermesClient:
    """Async client for Hermes search server.

    Example:
        async with HermesClient("localhost:50051") as client:
            # Create index
            await client.create_index("articles", '''
                index articles {
                    title: text indexed stored
                    body: text indexed stored
                }
            ''')

            # Index documents
            await client.index_documents("articles", [
                {"title": "Hello", "body": "World"},
                {"title": "Foo", "body": "Bar"},
            ])
            await client.commit("articles")

            # Search
            results = await client.search("articles", term=("title", "hello"))
            for hit in results.hits:
                print(hit.doc_id, hit.score)
    """

    def __init__(self, address: str = "localhost:50051"):
        """Initialize client.

        Args:
            address: Server address in format "host:port"
        """
        self.address = address
        self._channel: aio.Channel | None = None
        self._index_stub: pb_grpc.IndexServiceStub | None = None
        self._search_stub: pb_grpc.SearchServiceStub | None = None

    async def connect(self) -> None:
        """Connect to the server."""
        # Increase message size limits for large responses (e.g., loading content fields)
        options = [
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
        ]
        # Enable gzip compression for smaller message sizes over the wire
        self._channel = aio.insecure_channel(
            self.address,
            options=options,
            compression=grpc.Compression.Gzip,
        )
        self._index_stub = pb_grpc.IndexServiceStub(self._channel)
        self._search_stub = pb_grpc.SearchServiceStub(self._channel)

    async def close(self) -> None:
        """Close the connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._index_stub = None
            self._search_stub = None

    async def __aenter__(self) -> HermesClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _ensure_connected(self) -> None:
        if self._index_stub is None or self._search_stub is None:
            raise RuntimeError(
                "Client not connected. Use 'async with' or call connect() first."
            )

    # =========================================================================
    # Index Management
    # =========================================================================

    async def create_index(self, index_name: str, schema: str) -> bool:
        """Create a new index.

        Args:
            index_name: Name of the index
            schema: Schema definition in SDL or JSON format

        Returns:
            True if successful

        Example SDL schema:
            index myindex {
                title: text indexed stored
                body: text indexed stored
                score: f64 stored
            }

        Example JSON schema:
            {
                "fields": [
                    {"name": "title", "type": "text", "indexed": true, "stored": true}
                ]
            }
        """
        self._ensure_connected()
        request = pb.CreateIndexRequest(index_name=index_name, schema=schema)
        response = await self._index_stub.CreateIndex(request)
        return response.success

    async def delete_index(self, index_name: str) -> bool:
        """Delete an index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if successful
        """
        self._ensure_connected()
        request = pb.DeleteIndexRequest(index_name=index_name)
        response = await self._index_stub.DeleteIndex(request)
        return response.success

    async def get_index_info(self, index_name: str) -> IndexInfo:
        """Get information about an index.

        Args:
            index_name: Name of the index

        Returns:
            IndexInfo with document count, segments, and schema
        """
        self._ensure_connected()
        request = pb.GetIndexInfoRequest(index_name=index_name)
        response = await self._search_stub.GetIndexInfo(request)
        return IndexInfo(
            index_name=response.index_name,
            num_docs=response.num_docs,
            num_segments=response.num_segments,
            schema=response.schema,
        )

    # =========================================================================
    # Document Indexing
    # =========================================================================

    async def index_documents(
        self, index_name: str, documents: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """Index multiple documents in batch.

        Args:
            index_name: Name of the index
            documents: List of documents (dicts with field names as keys)

        Returns:
            Tuple of (indexed_count, error_count)
        """
        self._ensure_connected()

        named_docs = []
        for doc in documents:
            fields = _to_field_entries(doc)
            named_docs.append(pb.NamedDocument(fields=fields))

        request = pb.BatchIndexDocumentsRequest(
            index_name=index_name, documents=named_docs
        )
        response = await self._index_stub.BatchIndexDocuments(request)
        return response.indexed_count, response.error_count

    async def index_document(self, index_name: str, document: dict[str, Any]) -> None:
        """Index a single document.

        Args:
            index_name: Name of the index
            document: Document as dict with field names as keys
        """
        await self.index_documents(index_name, [document])

    async def index_documents_stream(
        self, index_name: str, documents: AsyncIterator[dict[str, Any]]
    ) -> int:
        """Stream documents for indexing.

        Args:
            index_name: Name of the index
            documents: Async iterator of documents

        Returns:
            Number of indexed documents
        """
        self._ensure_connected()

        async def request_iterator():
            async for doc in documents:
                fields = _to_field_entries(doc)
                yield pb.IndexDocumentRequest(index_name=index_name, fields=fields)

        response = await self._index_stub.IndexDocuments(request_iterator())
        return response.indexed_count

    async def commit(self, index_name: str) -> int:
        """Commit pending changes.

        Args:
            index_name: Name of the index

        Returns:
            Total number of documents in the index
        """
        self._ensure_connected()
        request = pb.CommitRequest(index_name=index_name)
        response = await self._index_stub.Commit(request)
        return response.num_docs

    async def force_merge(self, index_name: str) -> int:
        """Force merge all segments.

        Args:
            index_name: Name of the index

        Returns:
            Number of segments after merge
        """
        self._ensure_connected()
        request = pb.ForceMergeRequest(index_name=index_name)
        response = await self._index_stub.ForceMerge(request)
        return response.num_segments

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        index_name: str,
        *,
        term: tuple[str, str] | None = None,
        boolean: dict[str, list[tuple[str, str]]] | None = None,
        sparse_vector: tuple[str, list[int], list[float]] | None = None,
        sparse_text: tuple[str, str] | None = None,
        dense_vector: tuple[str, list[float]] | None = None,
        nprobe: int = 0,
        rerank_factor: int = 0,
        heap_factor: float = 1.0,
        combiner: str = "sum",
        limit: int = 10,
        offset: int = 0,
        fields_to_load: list[str] | None = None,
    ) -> SearchResponse:
        """Search for documents.

        Args:
            index_name: Name of the index
            term: Term query as (field, term) tuple
            boolean: Boolean query with "must", "should", "must_not" keys
            sparse_vector: Sparse vector query as (field, indices, values) tuple
            sparse_text: Sparse vector query with server-side tokenization as (field, text) tuple
            dense_vector: Dense vector query as (field, vector) tuple
            nprobe: Number of clusters to probe for dense vector (IVF indexes)
            rerank_factor: Re-ranking factor for dense vector search
            heap_factor: Approximate search factor for sparse vectors (1.0=exact, 0.8=faster)
            combiner: Score combiner for multi-value fields: "sum", "max", or "avg"
            limit: Maximum number of results
            offset: Offset for pagination
            fields_to_load: List of fields to include in results

        Returns:
            SearchResponse with hits

        Examples:
            # Term query
            results = await client.search("articles", term=("title", "hello"))

            # Boolean query
            results = await client.search("articles", boolean={
                "must": [("title", "hello")],
                "should": [("body", "world")],
            })

            # Sparse vector query (pre-tokenized)
            results = await client.search("docs",
                sparse_vector=("embedding", [1, 5, 10], [0.5, 0.3, 0.2]),
                fields_to_load=["title", "body"]
            )

            # Sparse text query (server-side tokenization)
            results = await client.search("docs",
                sparse_text=("embedding", "what is machine learning?"),
                fields_to_load=["title", "body"]
            )

            # Dense vector query
            results = await client.search("docs",
                dense_vector=("embedding", [0.1, 0.2, 0.3, ...]),
                fields_to_load=["title"]
            )
        """
        self._ensure_connected()

        query = _build_query(
            term=term,
            boolean=boolean,
            sparse_vector=sparse_vector,
            sparse_text=sparse_text,
            dense_vector=dense_vector,
            nprobe=nprobe,
            rerank_factor=rerank_factor,
            heap_factor=heap_factor,
            combiner=combiner,
        )

        request = pb.SearchRequest(
            index_name=index_name,
            query=query,
            limit=limit,
            offset=offset,
            fields_to_load=fields_to_load or [],
        )

        response = await self._search_stub.Search(request)

        hits = [
            SearchHit(
                doc_id=hit.doc_id,
                score=hit.score,
                fields={k: _from_field_value(v) for k, v in hit.fields.items()},
            )
            for hit in response.hits
        ]

        return SearchResponse(
            hits=hits,
            total_hits=response.total_hits,
            took_ms=response.took_ms,
        )

    async def get_document(self, index_name: str, doc_id: int) -> Document | None:
        """Get a document by ID.

        Args:
            index_name: Name of the index
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        self._ensure_connected()
        request = pb.GetDocumentRequest(index_name=index_name, doc_id=doc_id)
        try:
            response = await self._search_stub.GetDocument(request)
            fields = {k: _from_field_value(v) for k, v in response.fields.items()}
            return Document(fields=fields)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise


# =============================================================================
# Helper functions
# =============================================================================


def _is_sparse_vector(value: list) -> bool:
    """Check if list is a sparse vector: list of (int, float) pairs."""
    if not value:
        return False
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        idx, val = item
        if not isinstance(idx, int) or not isinstance(val, (int, float)):
            return False
    return True


def _is_multi_sparse_vector(value: list) -> bool:
    """Check if list is a multi-value sparse vector: list of sparse vectors."""
    if not value:
        return False
    # All items must be lists and each must be a valid sparse vector
    if not all(isinstance(item, list) for item in value):
        return False
    return all(_is_sparse_vector(item) for item in value)


def _is_dense_vector(value: list) -> bool:
    """Check if list is a dense vector: flat list of numeric values."""
    if not value:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)


def _is_multi_dense_vector(value: list) -> bool:
    """Check if list is a multi-value dense vector: list of dense vectors."""
    if not value:
        return False
    # All items must be lists and each must be a valid dense vector
    if not all(isinstance(item, list) for item in value):
        return False
    return all(_is_dense_vector(item) for item in value)


def _to_field_entries(doc: dict[str, Any]) -> list[pb.FieldEntry]:
    """Convert document dict to list of FieldEntry for multi-value field support.

    Multi-value fields (list of sparse vectors or list of dense vectors) are
    expanded into multiple FieldEntry with the same name.
    """
    entries = []
    for name, value in doc.items():
        if isinstance(value, list):
            # Check for multi-value sparse vectors: [[( idx, val), ...], ...]
            if _is_multi_sparse_vector(value):
                for sv in value:
                    indices = [int(item[0]) for item in sv]
                    values = [float(item[1]) for item in sv]
                    fv = pb.FieldValue(
                        sparse_vector=pb.SparseVector(indices=indices, values=values)
                    )
                    entries.append(pb.FieldEntry(name=name, value=fv))
                continue
            # Check for multi-value dense vectors: [[f1, f2, ...], ...]
            if _is_multi_dense_vector(value):
                for dv in value:
                    fv = pb.FieldValue(
                        dense_vector=pb.DenseVector(values=[float(v) for v in dv])
                    )
                    entries.append(pb.FieldEntry(name=name, value=fv))
                continue
        # Single value - use standard conversion
        entries.append(pb.FieldEntry(name=name, value=_to_field_value(value)))
    return entries


def _to_field_value(value: Any) -> pb.FieldValue:
    """Convert Python value to protobuf FieldValue.

    Special handling for vector types:
    - list[(int, float)] -> SparseVector (list of (index, value) tuples)
    - list[float] -> DenseVector (flat list of numeric values)
    - Other lists/dicts -> JSON
    """
    if isinstance(value, str):
        return pb.FieldValue(text=value)
    elif isinstance(value, bool):
        return pb.FieldValue(u64=1 if value else 0)
    elif isinstance(value, int):
        if value >= 0:
            return pb.FieldValue(u64=value)
        else:
            return pb.FieldValue(i64=value)
    elif isinstance(value, float):
        return pb.FieldValue(f64=value)
    elif isinstance(value, bytes):
        return pb.FieldValue(bytes_value=value)
    elif isinstance(value, dict):
        # Dicts are always JSON
        return pb.FieldValue(json_value=json.dumps(value))
    elif isinstance(value, list):
        # Check if it's a sparse vector: list of (index, value) pairs
        if _is_sparse_vector(value):
            indices = [int(item[0]) for item in value]
            values = [float(item[1]) for item in value]
            return pb.FieldValue(
                sparse_vector=pb.SparseVector(indices=indices, values=values)
            )
        # Check if it's a dense vector: flat list of numeric values
        if _is_dense_vector(value):
            return pb.FieldValue(
                dense_vector=pb.DenseVector(values=[float(v) for v in value])
            )
        # Otherwise treat as JSON
        return pb.FieldValue(json_value=json.dumps(value))
    else:
        return pb.FieldValue(text=str(value))


def _from_field_value(fv: pb.FieldValue) -> Any:
    """Convert protobuf FieldValue to Python value."""
    which = fv.WhichOneof("value")
    if which == "text":
        return fv.text
    elif which == "u64":
        return fv.u64
    elif which == "i64":
        return fv.i64
    elif which == "f64":
        return fv.f64
    elif which == "bytes_value":
        return fv.bytes_value
    elif which == "json_value":
        return json.loads(fv.json_value)
    elif which == "sparse_vector":
        return {
            "indices": list(fv.sparse_vector.indices),
            "values": list(fv.sparse_vector.values),
        }
    elif which == "dense_vector":
        return list(fv.dense_vector.values)
    return None


def _combiner_to_proto(combiner: str) -> int:
    """Convert combiner string to proto enum value."""
    return {"sum": 0, "max": 1, "avg": 2}.get(combiner.lower(), 0)


def _build_query(
    *,
    term: tuple[str, str] | None = None,
    boolean: dict[str, list[tuple[str, str]]] | None = None,
    sparse_vector: tuple[str, list[int], list[float]] | None = None,
    sparse_text: tuple[str, str] | None = None,
    dense_vector: tuple[str, list[float]] | None = None,
    nprobe: int = 0,
    rerank_factor: int = 0,
    heap_factor: float = 1.0,
    combiner: str = "sum",
) -> pb.Query:
    """Build a protobuf Query from parameters."""
    if term is not None:
        field, value = term
        return pb.Query(term=pb.TermQuery(field=field, term=value))

    if boolean is not None:
        must = [
            pb.Query(term=pb.TermQuery(field=f, term=t))
            for f, t in boolean.get("must", [])
        ]
        should = [
            pb.Query(term=pb.TermQuery(field=f, term=t))
            for f, t in boolean.get("should", [])
        ]
        must_not = [
            pb.Query(term=pb.TermQuery(field=f, term=t))
            for f, t in boolean.get("must_not", [])
        ]
        return pb.Query(
            boolean=pb.BooleanQuery(must=must, should=should, must_not=must_not)
        )

    combiner_value = _combiner_to_proto(combiner)

    if sparse_vector is not None:
        field, indices, values = sparse_vector
        return pb.Query(
            sparse_vector=pb.SparseVectorQuery(
                field=field,
                indices=indices,
                values=values,
                combiner=combiner_value,
                heap_factor=heap_factor,
            )
        )

    if sparse_text is not None:
        field, text = sparse_text
        return pb.Query(
            sparse_vector=pb.SparseVectorQuery(
                field=field,
                text=text,
                combiner=combiner_value,
                heap_factor=heap_factor,
            )
        )

    if dense_vector is not None:
        field, vector = dense_vector
        return pb.Query(
            dense_vector=pb.DenseVectorQuery(
                field=field,
                vector=vector,
                nprobe=nprobe,
                rerank_factor=rerank_factor,
                combiner=combiner_value,
            )
        )

    # Default: match all (empty boolean query)
    return pb.Query(boolean=pb.BooleanQuery())
