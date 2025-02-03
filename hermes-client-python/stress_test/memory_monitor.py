"""Memory monitoring for stress tests."""

import asyncio
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class MemoryStats:
    """Memory statistics collected during test."""

    # RSS memory samples: (timestamp, rss_mb)
    samples: list[tuple[float, float]] = field(default_factory=list)

    # Peak values
    peak_rss_mb: float = 0.0

    # Server PID
    server_pid: int | None = None

    def add_sample(self, timestamp: float, rss_mb: float) -> None:
        """Add a memory sample."""
        self.samples.append((timestamp, rss_mb))
        if rss_mb > self.peak_rss_mb:
            self.peak_rss_mb = rss_mb

    def avg_rss_mb(self) -> float:
        """Average RSS memory."""
        if not self.samples:
            return 0.0
        return sum(rss for _, rss in self.samples) / len(self.samples)

    def min_rss_mb(self) -> float:
        """Minimum RSS memory."""
        if not self.samples:
            return 0.0
        return min(rss for _, rss in self.samples)


def find_hermes_server_pid() -> int | None:
    """Find the PID of the running hermes-server process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "hermes-server"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            if pids and pids[0]:
                return int(pids[0])
    except Exception:
        pass
    return None


def get_process_memory_mb(pid: int) -> float | None:
    """Get RSS memory usage of a process in MB."""
    try:
        # Use ps command (works on macOS and Linux)
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            rss_kb = int(result.stdout.strip())
            return rss_kb / 1024.0  # Convert to MB
    except Exception:
        pass
    return None


async def memory_monitor(
    stats: MemoryStats,
    stop_event: asyncio.Event,
    start_time: float,
    poll_interval_ms: int = 500,
) -> None:
    """Monitor server memory usage.

    Args:
        stats: MemoryStats to populate
        stop_event: Event to signal shutdown
        start_time: Test start time for relative timestamps
        poll_interval_ms: How often to sample memory
    """
    poll_interval = poll_interval_ms / 1000.0

    # Find server PID
    pid = find_hermes_server_pid()
    if pid is None:
        print("[memory_monitor] Warning: Could not find hermes-server process")
        return

    stats.server_pid = pid

    while not stop_event.is_set():
        rss_mb = get_process_memory_mb(pid)
        if rss_mb is not None:
            timestamp = time.perf_counter() - start_time
            stats.add_sample(timestamp, rss_mb)

        await asyncio.sleep(poll_interval)


def print_memory_report(
    stats: MemoryStats,
    memory_limit_mb: float,
) -> None:
    """Print memory usage report.

    Args:
        stats: Collected memory statistics
        memory_limit_mb: Configured memory limit
    """
    print()
    print("=" * 60)
    print("MEMORY USAGE REPORT")
    print("=" * 60)
    print(f"Server PID:      {stats.server_pid}")
    print(f"Samples:         {len(stats.samples)}")
    print(f"Memory Limit:    {memory_limit_mb:.0f} MB")
    print()
    print(f"Peak RSS:        {stats.peak_rss_mb:.1f} MB")
    print(f"Average RSS:     {stats.avg_rss_mb():.1f} MB")
    print(f"Min RSS:         {stats.min_rss_mb():.1f} MB")
    print()

    # Check if we stayed within limits
    # Note: RSS includes all memory, not just indexing buffer
    # So we check if peak is reasonable relative to limit
    overhead_mb = 200  # Base server overhead estimate
    effective_limit = memory_limit_mb + overhead_mb

    if stats.peak_rss_mb <= effective_limit:
        print(
            f"[OK] Memory stayed within limits ({stats.peak_rss_mb:.1f} <= {effective_limit:.0f} MB)"
        )
    else:
        excess = stats.peak_rss_mb - effective_limit
        print(f"[WARNING] Memory exceeded limit by {excess:.1f} MB")
        print(
            f"          Peak: {stats.peak_rss_mb:.1f} MB, Limit: {effective_limit:.0f} MB"
        )

    print("=" * 60)
