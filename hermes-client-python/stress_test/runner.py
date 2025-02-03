"""Stress test orchestrator."""

import asyncio
import time

from .config import StressTestConfig
from .memory_monitor import MemoryStats, memory_monitor, print_memory_report
from .metrics import StressMetrics, print_report
from .workers import (
    document_producer,
    indexing_worker,
    search_worker,
    segment_monitor,
)

# Import client lazily
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


class StressTestRunner:
    """Orchestrates stress test execution."""

    def __init__(self, config: StressTestConfig, monitor_memory: bool = True):
        """Initialize runner with configuration.

        Args:
            config: Test configuration
            monitor_memory: Whether to monitor server memory usage
        """
        self.config = config
        self.metrics = StressMetrics()
        self.memory_stats = MemoryStats()
        self.monitor_memory = monitor_memory

    async def setup_index(self) -> None:
        """Create the test index if it doesn't exist."""
        HC = _get_client_class()

        async with HC(self.config.server_address) as client:
            # Try to delete existing index
            try:
                await client.delete_index(self.config.index_name)
                print(f"Deleted existing index: {self.config.index_name}")
            except Exception:
                pass  # Index didn't exist

            # Create new index
            await client.create_index(self.config.index_name, self.config.schema)
            print(f"Created index: {self.config.index_name}")

    async def run(self) -> StressMetrics:
        """Run the stress test.

        Returns:
            Collected metrics
        """
        # Setup
        await self.setup_index()

        # Create synchronization primitives
        stop_event = asyncio.Event()
        warmup_complete = asyncio.Event()
        doc_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.index_workers * 2)

        self.metrics.start_time = time.perf_counter()

        # Create all tasks
        tasks = []

        # Document producer
        producer_task = asyncio.create_task(
            document_producer(doc_queue, self.config, self.metrics, stop_event),
            name="producer",
        )
        tasks.append(producer_task)

        # Indexing workers
        for i in range(self.config.index_workers):
            task = asyncio.create_task(
                indexing_worker(
                    i, doc_queue, self.config, self.metrics, stop_event, warmup_complete
                ),
                name=f"index_worker_{i}",
            )
            tasks.append(task)

        # Search workers
        for i in range(self.config.search_workers):
            task = asyncio.create_task(
                search_worker(
                    i, self.config, self.metrics, stop_event, warmup_complete
                ),
                name=f"search_worker_{i}",
            )
            tasks.append(task)

        # Segment monitor
        monitor_task = asyncio.create_task(
            segment_monitor(
                self.config, self.metrics, stop_event, self.metrics.start_time
            ),
            name="monitor",
        )
        tasks.append(monitor_task)

        # Memory monitor (if enabled)
        if self.monitor_memory:
            mem_monitor_task = asyncio.create_task(
                memory_monitor(
                    self.memory_stats,
                    stop_event,
                    self.metrics.start_time,
                    self.config.memory_poll_interval_ms,
                ),
                name="memory_monitor",
            )
            tasks.append(mem_monitor_task)

        # Wait for duration or indexing completion
        print(f"\nRunning stress test for up to {self.config.duration_seconds}s...")
        print(f"  Target: {self.config.doc_count:,} docs, {self.config.search_qps} QPS")
        print()

        try:
            # Wait for producer to complete (all docs generated)
            await asyncio.wait_for(producer_task, timeout=self.config.duration_seconds)

            # Wait a bit for indexing workers to finish
            remaining_timeout = max(
                10,
                self.config.duration_seconds
                - (time.perf_counter() - self.metrics.start_time),
            )

            # Wait for all indexing workers
            index_tasks = [t for t in tasks if t.get_name().startswith("index_worker")]
            done, pending = await asyncio.wait(
                index_tasks,
                timeout=remaining_timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

        except asyncio.TimeoutError:
            print("Test duration reached, stopping...")

        finally:
            # Signal all workers to stop
            stop_event.set()

            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)

        self.metrics.end_time = time.perf_counter()

        return self.metrics

    def print_report(self) -> None:
        """Print the test report."""
        config_summary = (
            f"Documents: {self.config.doc_count:,} | "
            f"Batch: {self.config.batch_size} | "
            f"Index Workers: {self.config.index_workers} | "
            f"Search Workers: {self.config.search_workers}"
        )
        print_report(self.metrics, config_summary)

        # Print memory report if we collected data
        if self.monitor_memory and self.memory_stats.samples:
            print_memory_report(self.memory_stats, self.config.memory_limit_mb)


async def run_stress_test(config: StressTestConfig) -> StressMetrics:
    """Convenience function to run a stress test.

    Args:
        config: Test configuration

    Returns:
        Collected metrics
    """
    runner = StressTestRunner(config)
    metrics = await runner.run()
    runner.print_report()
    return metrics
