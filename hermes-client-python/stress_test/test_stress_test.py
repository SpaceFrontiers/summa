"""Unit tests for the stress test framework."""

import inspect
import io

from .config import StressTestConfig
from .generators import (
    generate_batch,
    generate_document,
    generate_query_vector,
    generate_sparse_vector,
)
from .metrics import (
    StressMetrics,
    analyze_reload_frequency,
    compute_percentiles,
    print_report,
)
from .runner import StressTestRunner, run_stress_test
from .workers import (
    document_producer,
    indexing_worker,
    search_worker,
    segment_monitor,
)


class TestConfig:
    """Tests for StressTestConfig."""

    def test_default_config(self):
        config = StressTestConfig()
        assert config.doc_count == 50_000
        assert config.batch_size == 500
        assert config.index_workers == 4
        assert config.search_workers == 4
        assert config.vocab_size == 30_000
        assert config.avg_nnz == 100

    def test_docs_before_search(self):
        config = StressTestConfig(doc_count=100_000, warmup_docs=10_000)
        assert config.docs_before_search() == 10_000

        config = StressTestConfig(doc_count=100, warmup_docs=10_000)
        assert config.docs_before_search() == 25  # doc_count // 4

    def test_batches_total(self):
        config = StressTestConfig(doc_count=1000, batch_size=100)
        assert config.batches_total() == 10

        config = StressTestConfig(doc_count=1050, batch_size=100)
        assert config.batches_total() == 11  # Ceiling division


class TestGenerators:
    """Tests for document and vector generators."""

    def test_sparse_vector_generation(self):
        sv = generate_sparse_vector(vocab_size=30000, avg_nnz=100)
        assert len(sv) > 0
        assert all(isinstance(idx, int) and isinstance(val, float) for idx, val in sv)
        assert all(0 <= idx < 30000 for idx, _ in sv)
        assert all(0 <= val <= 1.0 for _, val in sv)

    def test_sparse_vector_uniqueness(self):
        sv = generate_sparse_vector(vocab_size=30000, avg_nnz=100)
        indices = [idx for idx, _ in sv]
        assert len(indices) == len(set(indices)), "Indices should be unique"

    def test_document_generation(self):
        config = StressTestConfig(sparse_vectors_per_doc=3)
        doc = generate_document(42, config)
        assert doc["id"] == "doc_42"
        assert "content" in doc
        assert len(doc["embedding"]) == 3

    def test_query_vector_generation(self):
        config = StressTestConfig(vocab_size=30000, avg_nnz=100)
        indices, values = generate_query_vector(config)
        assert len(indices) == len(values)
        assert len(indices) > 0
        # Query vectors should be sparser than doc vectors
        assert len(indices) < config.avg_nnz

    def test_batch_generation(self):
        config = StressTestConfig()
        batch = generate_batch(0, 10, config)
        assert len(batch) == 10
        assert all("id" in doc and "embedding" in doc for doc in batch)
        # Check IDs are sequential
        assert [doc["id"] for doc in batch] == [f"doc_{i}" for i in range(10)]


class TestMetrics:
    """Tests for metrics collection and analysis."""

    def test_compute_percentiles(self):
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        stats = compute_percentiles(latencies)
        assert 40 <= stats["p50"] <= 60
        assert stats["p95"] >= 90
        assert stats["max"] == 100
        assert stats["min"] == 10
        assert 50 <= stats["mean"] <= 60

    def test_compute_percentiles_empty(self):
        stats = compute_percentiles([])
        assert stats["p50"] == 0
        assert stats["p95"] == 0
        assert stats["max"] == 0

    def test_analyze_reload_frequency(self):
        segment_counts = [
            (0.0, 1),
            (1.0, 2),
            (2.0, 3),
            (3.0, 4),
            (4.0, 2),  # merge
            (5.0, 3),
        ]
        analysis = analyze_reload_frequency(segment_counts)
        assert analysis["min_segments"] == 1
        assert analysis["max_segments"] == 4
        assert analysis["final_segments"] == 3
        assert analysis["estimated_reloads"] == 5

    def test_stress_metrics(self):
        metrics = StressMetrics()
        metrics.start_time = 0.0
        metrics.end_time = 10.0
        metrics.docs_indexed = 1000
        metrics.queries_executed = 100

        assert metrics.duration_seconds() == 10.0
        assert metrics.indexing_throughput() == 100.0
        assert metrics.search_qps() == 10.0

    def test_print_report(self):
        metrics = StressMetrics()
        metrics.start_time = 0.0
        metrics.end_time = 10.0
        metrics.docs_indexed = 1000
        metrics.queries_executed = 100
        for i in range(10):
            metrics.add_batch_latency(10 + i)
            metrics.add_search_latency(5 + i)
        metrics.add_segment_count(0.0, 1)
        metrics.add_segment_count(10.0, 5)

        output = io.StringIO()
        print_report(metrics, "Test config", output)
        report = output.getvalue()
        assert "HERMES STRESS TEST REPORT" in report
        assert "Test config" in report


class TestWorkerStructure:
    """Tests for worker module structure."""

    def test_workers_are_async(self):
        assert inspect.iscoroutinefunction(document_producer)
        assert inspect.iscoroutinefunction(indexing_worker)
        assert inspect.iscoroutinefunction(search_worker)
        assert inspect.iscoroutinefunction(segment_monitor)

    def test_runner_initialization(self):
        config = StressTestConfig(doc_count=100, batch_size=10)
        runner = StressTestRunner(config)
        assert runner.config == config
        assert runner.metrics is not None

    def test_run_stress_test_is_async(self):
        assert inspect.iscoroutinefunction(run_stress_test)
