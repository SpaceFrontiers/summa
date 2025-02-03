"""Async worker coroutines for stress testing."""

import asyncio
import time

from .config import IndexType, StressTestConfig
from .generators import generate_batch, generate_query_vector
from .metrics import StressMetrics

# Import client lazily to avoid import errors
HermesClient: type | None = None


def _get_client_class():
    """Lazy import of HermesClient."""
    global HermesClient
    if HermesClient is None:
        import sys

        sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")
        from hermes_client_python import HermesClient as HC

        HermesClient = HC
    return HermesClient


async def document_producer(
    queue: asyncio.Queue,
    config: StressTestConfig,
    metrics: StressMetrics,
    stop_event: asyncio.Event,
) -> None:
    """Produce document batches for indexing workers.

    Args:
        queue: Queue to put batches into
        config: Test configuration
        metrics: Metrics collector
        stop_event: Event to signal shutdown
    """
    doc_id = 0
    total_docs = config.doc_count

    while doc_id < total_docs and not stop_event.is_set():
        remaining = total_docs - doc_id
        batch_size = min(config.batch_size, remaining)

        batch = generate_batch(doc_id, batch_size, config)
        await queue.put(batch)

        doc_id += batch_size

    # Signal completion to workers
    for _ in range(config.index_workers):
        await queue.put(None)


async def indexing_worker(
    worker_id: int,
    queue: asyncio.Queue,
    config: StressTestConfig,
    metrics: StressMetrics,
    stop_event: asyncio.Event,
    warmup_complete: asyncio.Event,
) -> None:
    """Worker that indexes batches from the queue.

    Args:
        worker_id: Worker identifier
        queue: Queue to get batches from
        config: Test configuration
        metrics: Metrics collector
        stop_event: Event to signal shutdown
        warmup_complete: Event to signal warmup is done
    """
    HC = _get_client_class()

    async with HC(config.server_address) as client:
        docs_by_worker = 0
        warmup_threshold = config.docs_before_search() // config.index_workers

        while not stop_event.is_set():
            try:
                batch = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if batch is None:
                break

            # Index batch
            start = time.perf_counter()
            try:
                indexed, errors = await client.index_documents(config.index_name, batch)
                elapsed_ms = (time.perf_counter() - start) * 1000

                metrics.add_batch_latency(elapsed_ms)
                metrics.docs_indexed += indexed
                metrics.index_errors += errors
                docs_by_worker += indexed

            except Exception as e:
                metrics.index_errors += len(batch)
                print(f"[Worker {worker_id}] Batch error: {e}")

            # Commit periodically (every 10 batches)
            if (
                docs_by_worker > 0
                and docs_by_worker % (config.batch_size * 10) < config.batch_size
            ):
                commit_start = time.perf_counter()
                try:
                    await client.commit(config.index_name)
                    commit_elapsed = (time.perf_counter() - commit_start) * 1000
                    metrics.add_commit_latency(commit_elapsed)
                except Exception as e:
                    print(f"[Worker {worker_id}] Commit error: {e}")

            # Signal warmup complete after threshold
            if docs_by_worker >= warmup_threshold and not warmup_complete.is_set():
                warmup_complete.set()

        # Final commit
        try:
            commit_start = time.perf_counter()
            await client.commit(config.index_name)
            commit_elapsed = (time.perf_counter() - commit_start) * 1000
            metrics.add_commit_latency(commit_elapsed)
        except Exception as e:
            print(f"[Worker {worker_id}] Final commit error: {e}")


async def search_worker(
    worker_id: int,
    config: StressTestConfig,
    metrics: StressMetrics,
    stop_event: asyncio.Event,
    warmup_complete: asyncio.Event,
) -> None:
    """Worker that performs continuous searches based on index type.

    Args:
        worker_id: Worker identifier
        config: Test configuration
        metrics: Metrics collector
        stop_event: Event to signal shutdown
        warmup_complete: Event to wait for before starting
    """
    HC = _get_client_class()

    # Wait for warmup to complete
    await warmup_complete.wait()

    # Calculate delay between queries for this worker
    target_qps_per_worker = config.search_qps / config.search_workers
    delay_between_queries = (
        1.0 / target_qps_per_worker if target_qps_per_worker > 0 else 0.1
    )

    async with HC(config.server_address) as client:
        while not stop_event.is_set():
            query_start = time.perf_counter()

            # Generate and execute query based on index type
            query_data = generate_query_vector(config)

            try:
                start = time.perf_counter()

                if config.index_type == IndexType.DENSE:
                    _result = await client.search(
                        config.index_name,
                        dense_vector=("embedding", query_data),
                        limit=config.search_top_k,
                    )
                elif config.index_type == IndexType.FULLTEXT:
                    _result = await client.search(
                        config.index_name,
                        term=("body", query_data),
                        limit=config.search_top_k,
                    )
                else:  # SPARSE or MIXED
                    indices, values = query_data
                    field = (
                        "sparse_emb"
                        if config.index_type == IndexType.MIXED
                        else "embedding"
                    )
                    _result = await client.search(
                        config.index_name,
                        sparse_vector=(field, indices, values),
                        limit=config.search_top_k,
                    )

                elapsed_ms = (time.perf_counter() - start) * 1000

                metrics.add_search_latency(elapsed_ms)
                metrics.queries_executed += 1

            except Exception as e:
                metrics.search_errors += 1
                if metrics.search_errors <= 5:  # Only log first few errors
                    print(f"[Search {worker_id}] Error: {e}")

            # Rate limiting
            elapsed = time.perf_counter() - query_start
            sleep_time = delay_between_queries - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


async def segment_monitor(
    config: StressTestConfig,
    metrics: StressMetrics,
    stop_event: asyncio.Event,
    start_time: float,
) -> None:
    """Monitor segment count over time.

    Args:
        config: Test configuration
        metrics: Metrics collector
        stop_event: Event to signal shutdown
        start_time: Test start time for relative timestamps
    """
    HC = _get_client_class()
    poll_interval = config.segment_poll_interval_ms / 1000.0

    async with HC(config.server_address) as client:
        while not stop_event.is_set():
            try:
                info = await client.get_index_info(config.index_name)
                timestamp = time.perf_counter() - start_time
                metrics.add_segment_count(timestamp, info.num_segments)
            except Exception:
                pass  # Index may not exist yet

            await asyncio.sleep(poll_interval)
