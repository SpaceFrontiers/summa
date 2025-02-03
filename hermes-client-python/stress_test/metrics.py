"""Metrics collection and analysis for stress testing."""

import statistics
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class StressMetrics:
    """Collected metrics from stress test run."""

    # Indexing metrics
    index_batch_latencies_ms: list[float] = field(default_factory=list)
    commit_latencies_ms: list[float] = field(default_factory=list)
    docs_indexed: int = 0
    index_errors: int = 0

    # Search metrics
    search_latencies_ms: list[float] = field(default_factory=list)
    search_errors: int = 0
    queries_executed: int = 0

    # Segment tracking
    segment_counts: list[tuple[float, int]] = field(
        default_factory=list
    )  # (timestamp, count)

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    def add_batch_latency(self, latency_ms: float) -> None:
        """Record a batch indexing latency."""
        self.index_batch_latencies_ms.append(latency_ms)

    def add_commit_latency(self, latency_ms: float) -> None:
        """Record a commit latency."""
        self.commit_latencies_ms.append(latency_ms)

    def add_search_latency(self, latency_ms: float) -> None:
        """Record a search latency."""
        self.search_latencies_ms.append(latency_ms)

    def add_segment_count(self, timestamp: float, count: int) -> None:
        """Record segment count at a point in time."""
        self.segment_counts.append((timestamp, count))

    def duration_seconds(self) -> float:
        """Total test duration in seconds."""
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def indexing_throughput(self) -> float:
        """Documents indexed per second."""
        duration = self.duration_seconds()
        if duration > 0:
            return self.docs_indexed / duration
        return 0.0

    def search_qps(self) -> float:
        """Achieved queries per second."""
        duration = self.duration_seconds()
        if duration > 0:
            return self.queries_executed / duration
        return 0.0


def compute_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute latency percentiles.

    Returns:
        Dict with p50, p95, p99, max, min, mean keys
    """
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0, "min": 0, "mean": 0}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(p: float) -> float:
        idx = int(p / 100 * n)
        idx = min(idx, n - 1)
        return sorted_latencies[idx]

    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": max(latencies),
        "min": min(latencies),
        "mean": statistics.mean(latencies) if latencies else 0,
    }


def analyze_reload_frequency(
    segment_counts: list[tuple[float, int]],
) -> dict[str, float | int]:
    """Analyze segment count changes to estimate reload frequency.

    Args:
        segment_counts: List of (timestamp, count) tuples

    Returns:
        Dict with analysis results
    """
    if len(segment_counts) < 2:
        return {
            "min_segments": 0,
            "max_segments": 0,
            "final_segments": 0,
            "estimated_reloads": 0,
            "avg_reload_interval_s": 0,
        }

    counts = [c for _, c in segment_counts]
    changes = sum(1 for i in range(1, len(counts)) if counts[i] != counts[i - 1])

    duration = segment_counts[-1][0] - segment_counts[0][0]
    avg_interval = duration / changes if changes > 0 else 0

    return {
        "min_segments": min(counts),
        "max_segments": max(counts),
        "final_segments": counts[-1],
        "estimated_reloads": changes,
        "avg_reload_interval_s": avg_interval,
    }


def print_report(
    metrics: StressMetrics,
    config_summary: str,
    output: TextIO | None = None,
) -> None:
    """Print formatted stress test report.

    Args:
        metrics: Collected metrics
        config_summary: Short config description
        output: Output file (default: stdout)
    """
    import sys

    out = output or sys.stdout

    batch_stats = compute_percentiles(metrics.index_batch_latencies_ms)
    commit_stats = compute_percentiles(metrics.commit_latencies_ms)
    search_stats = compute_percentiles(metrics.search_latencies_ms)
    segment_analysis = analyze_reload_frequency(metrics.segment_counts)

    lines = [
        "=" * 80,
        "                        HERMES STRESS TEST REPORT",
        "=" * 80,
        "",
        "Configuration:",
        f"  {config_summary}",
        "",
        "Indexing:",
        f"  Documents:    {metrics.docs_indexed:,}",
        f"  Errors:       {metrics.index_errors}",
        f"  Throughput:   {metrics.indexing_throughput():,.0f} docs/sec",
        f"  Batch Latency:  p50={batch_stats['p50']:.0f}ms  p95={batch_stats['p95']:.0f}ms  p99={batch_stats['p99']:.0f}ms",
        f"  Commit Latency: p50={commit_stats['p50']:.0f}ms  p95={commit_stats['p95']:.0f}ms  p99={commit_stats['p99']:.0f}ms",
        "",
        "Search:",
        f"  Queries:      {metrics.queries_executed:,}",
        f"  Errors:       {metrics.search_errors}",
        f"  Achieved QPS: {metrics.search_qps():.1f}",
        f"  Latency:      p50={search_stats['p50']:.0f}ms  p95={search_stats['p95']:.0f}ms  p99={search_stats['p99']:.0f}ms",
        "",
        "Segment Analysis:",
        f"  Range: {segment_analysis['min_segments']}-{segment_analysis['max_segments']} segments | Final: {segment_analysis['final_segments']}",
        f"  Estimated Reloads: {segment_analysis['estimated_reloads']} | Avg Interval: {segment_analysis['avg_reload_interval_s']:.2f}s",
    ]

    # Add warning if reload frequency is high
    if (
        segment_analysis["estimated_reloads"] > 50
        and segment_analysis["avg_reload_interval_s"] < 2.0
    ):
        lines.extend(
            [
                "",
                "  [!] HIGH RELOAD FREQUENCY - consider increasing reload_interval_ms",
            ]
        )

    lines.extend(["=" * 80, ""])

    for line in lines:
        print(line, file=out)
