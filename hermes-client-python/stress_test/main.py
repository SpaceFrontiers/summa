#!/usr/bin/env python3
"""CLI entry point for Hermes stress testing."""

import argparse
import asyncio
import sys

from .config import IndexType, StressTestConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hermes stress testing tool for indexing and search performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server connection
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Server address (host:port)",
    )
    parser.add_argument(
        "--index",
        default="stress_test",
        help="Index name to use for testing",
    )

    # Index type
    parser.add_argument(
        "--index-type",
        choices=["sparse", "dense", "fulltext", "mixed"],
        default="sparse",
        help="Type of index to test",
    )

    # Document generation
    parser.add_argument(
        "--docs",
        type=int,
        default=50_000,
        help="Number of documents to index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Documents per batch",
    )

    # Worker configuration
    parser.add_argument(
        "--index-workers",
        type=int,
        default=4,
        help="Number of parallel indexing workers",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=4,
        help="Number of parallel search workers",
    )

    # Sparse vector parameters
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30_000,
        help="Vocabulary size for sparse vectors",
    )
    parser.add_argument(
        "--avg-nnz",
        type=int,
        default=100,
        help="Average non-zero entries per sparse vector",
    )
    parser.add_argument(
        "--sparse-vectors-per-doc",
        type=int,
        default=3,
        help="Number of sparse vectors per document",
    )

    # Dense vector parameters
    parser.add_argument(
        "--dense-dim",
        type=int,
        default=128,
        help="Dimension of dense vectors",
    )
    parser.add_argument(
        "--dense-vectors-per-doc",
        type=int,
        default=1,
        help="Number of dense vectors per document",
    )

    # Fulltext parameters
    parser.add_argument(
        "--avg-title-words",
        type=int,
        default=10,
        help="Average words in title field",
    )
    parser.add_argument(
        "--avg-body-words",
        type=int,
        default=200,
        help="Average words in body field",
    )

    # Search parameters
    parser.add_argument(
        "--search-qps",
        type=float,
        default=50.0,
        help="Target queries per second",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per query",
    )

    # Test timing
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Maximum test duration in seconds",
    )
    parser.add_argument(
        "--warmup-docs",
        type=int,
        default=10_000,
        help="Documents to index before starting searches",
    )

    # Memory monitoring
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=2048,
        help="Expected server memory limit in MB for verification",
    )
    parser.add_argument(
        "--no-memory-monitor",
        action="store_true",
        help="Disable memory monitoring",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Map string index type to enum
    index_type_map = {
        "sparse": IndexType.SPARSE,
        "dense": IndexType.DENSE,
        "fulltext": IndexType.FULLTEXT,
        "mixed": IndexType.MIXED,
    }
    index_type = index_type_map[args.index_type]

    config = StressTestConfig(
        server_address=args.server,
        index_name=args.index,
        index_type=index_type,
        doc_count=args.docs,
        batch_size=args.batch_size,
        index_workers=args.index_workers,
        search_workers=args.search_workers,
        vocab_size=args.vocab_size,
        avg_nnz=args.avg_nnz,
        sparse_vectors_per_doc=args.sparse_vectors_per_doc,
        dense_dim=args.dense_dim,
        dense_vectors_per_doc=args.dense_vectors_per_doc,
        avg_title_words=args.avg_title_words,
        avg_body_words=args.avg_body_words,
        search_qps=args.search_qps,
        search_top_k=args.top_k,
        duration_seconds=args.duration,
        warmup_docs=args.warmup_docs,
        memory_limit_mb=args.memory_limit,
    )

    print("=" * 60)
    print("Hermes Stress Test")
    print("=" * 60)
    print(f"Server:          {config.server_address}")
    print(f"Index:           {config.index_name}")
    print(f"Index type:      {config.index_type.value}")
    print(f"Documents:       {config.doc_count:,}")
    print(f"Batch size:      {config.batch_size}")
    print(f"Index workers:   {config.index_workers}")
    print(f"Search workers:  {config.search_workers}")
    print(f"Memory limit:    {config.memory_limit_mb} MB")

    # Print type-specific parameters
    if index_type == IndexType.SPARSE:
        print(
            f"Sparse vectors:  {config.sparse_vectors_per_doc}/doc, {config.avg_nnz} avg nnz"
        )
    elif index_type == IndexType.DENSE:
        print(
            f"Dense vectors:   {config.dense_vectors_per_doc}/doc, dim={config.dense_dim}"
        )
    elif index_type == IndexType.FULLTEXT:
        print(
            f"Text fields:     title (~{config.avg_title_words} words), body (~{config.avg_body_words} words)"
        )
    elif index_type == IndexType.MIXED:
        print("Mixed:           sparse + dense + fulltext")

    print(f"Target QPS:      {config.search_qps}")
    print("=" * 60)

    try:
        from .runner import StressTestRunner

        runner = StressTestRunner(config, monitor_memory=not args.no_memory_monitor)
        asyncio.run(runner.run())
        runner.print_report()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
