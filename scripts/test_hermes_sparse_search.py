#!/usr/bin/env python3
"""
Test script for Hermes sparse vector search with timing diagnostics.

Usage:
    # First port-forward:
    kubectl port-forward -n hermes svc/hermes-server 50051:50051

    # Then run:
    python scripts/test_hermes_sparse_search.py "machine learning"
"""

import asyncio
import sys
import time

from hermes_client_python import HermesClient

HERMES_ENDPOINT = "localhost:50051"


def extract_text_from_span(content: str, span: dict) -> str:
    """Extract text from content using span info (start_offset, length)."""
    if not content or not span:
        return ""
    start = span.get("start_offset", 0)
    length = span.get("length", 0)
    return content[start : start + length]


async def search_sparse(query: str, limit: int = 10):
    """Search documents using sparse vector search."""
    total_start = time.perf_counter()

    print(f"Connecting to Hermes at {HERMES_ENDPOINT}...")

    connect_start = time.perf_counter()
    client = HermesClient(HERMES_ENDPOINT)
    await client.connect()
    connect_time = time.perf_counter() - connect_start
    print(f"[TIMING] Connect: {connect_time*1000:.1f}ms")

    try:
        # Get index info
        info_start = time.perf_counter()
        try:
            info = await client.get_index_info("documents")
            info_time = time.perf_counter() - info_start
            print(f"Index has {info.num_docs} documents, {info.num_segments} segments")
            print(f"[TIMING] Get index info: {info_time*1000:.1f}ms\n")
        except Exception as e:
            print(f"Error getting index info: {e}")
            return

        print(f"Searching for: '{query}'\n")

        # === Search on title_sparse_vectors ===
        print("=" * 80)
        print("TITLE SPARSE SEARCH (title_sparse_vectors)")
        print("=" * 80)

        title_start = time.perf_counter()
        title_results = await client.search(
            "documents",
            sparse_text=("title_sparse_vectors", query),
            limit=limit,
            fields_to_load=["id", "title", "abstract", "uris"],
            heap_factor=0.7,
        )
        title_time = time.perf_counter() - title_start

        print(
            f"Found {title_results.total_hits} results, returned {len(title_results.hits)} hits"
        )
        print(f"Server reported: {title_results.took_ms}ms")
        print(f"[TIMING] Title search (client-side): {title_time*1000:.1f}ms\n")

        for i, hit in enumerate(title_results.hits, 1):
            title = hit.fields.get("title", "N/A")
            doc_id = hit.fields.get("id", "N/A")
            print(f"{i}. [score: {hit.score:.4f}] {title}")
            print(f"   ID: {doc_id}")

        # === Search on sparse_vectors (content/abstract) ===
        print("\n" + "=" * 80)
        print("CONTENT SPARSE SEARCH (sparse_vectors)")
        print("=" * 80)

        content_start = time.perf_counter()
        content_results = await client.search(
            "documents",
            sparse_text=("sparse_vectors", query),
            limit=limit,
            fields_to_load=[
                "id",
                "title",
                "content",
                "abstract",
                "sparse_span",
                "uris",
            ],
            heap_factor=0.7,
        )
        content_time = time.perf_counter() - content_start

        print(
            f"Found {content_results.total_hits} results, returned {len(content_results.hits)} hits"
        )
        print(f"Server reported: {content_results.took_ms}ms")
        print(f"[TIMING] Content search (client-side): {content_time*1000:.1f}ms\n")

        for i, hit in enumerate(content_results.hits, 1):
            title = hit.fields.get("title", "N/A")
            doc_id = hit.fields.get("id", "N/A")
            content = hit.fields.get("content", "")
            abstract = hit.fields.get("abstract", "")
            sparse_spans = hit.fields.get("sparse_span", [])
            ordinal = getattr(hit, "ordinal", None)
            uris = hit.fields.get("uris", [])

            print(f"{i}. [score: {hit.score:.4f}] {title}")
            print(f"   ID: {doc_id}")
            if uris:
                print(f"   URI: {uris[0] if isinstance(uris, list) else uris}")

            # Extract text from the best matching span
            if sparse_spans and ordinal is not None and ordinal < len(sparse_spans):
                span = sparse_spans[ordinal]
                field_name = span.get("field_name", "")
                source_text = content if field_name == "content" else abstract
                matched_text = extract_text_from_span(source_text, span)
                print(f"   [{field_name}] {matched_text}")
            elif sparse_spans:
                span = sparse_spans[0]
                field_name = span.get("field_name", "")
                source_text = content if field_name == "content" else abstract
                matched_text = extract_text_from_span(source_text, span)
                print(f"   [{field_name}] {matched_text}")

            print()

    finally:
        close_start = time.perf_counter()
        await client.close()
        close_time = time.perf_counter() - close_start
        print(f"\n[TIMING] Client close: {close_time*1000:.1f}ms")

    total_time = time.perf_counter() - total_start
    print(f"[TIMING] Total script time: {total_time*1000:.1f}ms")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hermes_sparse_search.py <query>")
        print("Example: python test_hermes_sparse_search.py 'machine learning'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    await search_sparse(query)


if __name__ == "__main__":
    asyncio.run(main())
