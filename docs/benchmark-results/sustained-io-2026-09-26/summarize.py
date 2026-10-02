#!/usr/bin/env python3
"""Validate every offer before exporting compact fixed-arrival summaries."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "sparse_export", HERE.parent / "sparse-concurrency-2026-09-26/summarize.py"
)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    assert values
    return {
        "mean": statistics.mean(values),
        "p50": shared.percentile(values, 0.5),
        "p95": shared.percentile(values, 0.95),
        "p99": shared.percentile(values, 0.99),
        "max": max(values),
    }


def phase(samples, rejected, seconds):
    offers = len(samples) + len(rejected)
    assert offers
    generation = [(s["admitted_ns"] - s["scheduled_ns"]) / 1e6 for s in samples]
    generation += [(s["observed_ns"] - s["scheduled_ns"]) / 1e6 for s in rejected]
    return {
        "offered": offers,
        "accepted": len(samples),
        "rejected": len(rejected),
        "reject_fraction": len(rejected) / offers,
        "accepted_offers_per_second": len(samples) / seconds,
        "latency_ms": distribution([s["latency_ns"] / 1e6 for s in samples])
        if samples
        else None,
        "generation_lag_ms": distribution(generation),
    }


def read_cell(campaign, record):
    """Validate protocol/accounting before returning a compact cell."""
    path = campaign / record["label"]
    row = shared.read(path / "result.jsonl")
    control = shared.read(path / "control.json")
    assert record["exit_code"] == control["exit_code"] == 0
    assert row["complete"] and shared.read(path / "completed.json")["complete"]
    assert row["method"] == record["method"]
    assert (
        row["concurrency"] == 8 and row["runtime_threads"] == 4 and row["sparse_reads"]
    )
    group = control["cgroup_after"]
    assert group["cpuset.cpus.effective"].strip() == "0-7"
    assert int(group["memory.max"]) == 256 * 1024 * 1024
    assert int(group["memory.swap.max"]) == 0
    events = dict(line.split() for line in group["memory.events"].splitlines())
    assert int(events["oom"]) == int(events["oom_kill"]) == 0
    residency = shared.read(campaign / (record["label"] + "-residency.json"))
    assert all(
        v["resident_bytes"] == 0
        for k, v in residency.items()
        if k.endswith((".store", ".sparse"))
    )
    stats = row["stats_after"]
    if stats:
        assert all(
            stats[k] == 0
            for k in [
                "errors",
                "worker_failures",
                "active",
                "quarantined_bytes",
                "notification_failures",
            ]
        )
        assert stats["completed"] == stats["submitted"] and stats["max_active"] <= 8
        if record["method"] == "uring":
            assert stats["wakeups"] > 0
    samples = row["samples"]
    assert all(s["documents"] == 32 for s in samples)
    if record["mode"] == "run":
        assert row["load"] is None and len(samples) == 64
        return {
            **record,
            "queries": len(samples),
            "documents": 32 * len(samples),
            "raw_sha256": digest(path / "result.jsonl"),
        }
    load = row["load"]
    assert load["cpu_includes_verification"]
    assert load["rate"] == record["rate"]
    assert load["offered"] == record["rate"] * 8
    assert load["arrival_window_seconds"] == 8
    assert row["operation_wall_seconds"] >= 8
    assert load["accepted"] == len(samples) and load["max_in_flight"] <= 8
    rejected = load["rejected"]
    assert len(samples) + len(rejected) == load["offered"]
    offers = sorted([s["sequence"] for s in samples + rejected])
    assert offers == list(range(load["offered"]))
    for s in samples + rejected:
        assert s["scheduled_ns"] == s["sequence"] * 1_000_000_000 // load["rate"]
    for s in rejected:
        assert s["observed_ns"] >= s["scheduled_ns"]
    for s in samples:
        assert (
            s["scheduled_ns"]
            <= s["admitted_ns"]
            <= s["started_ns"]
            <= s["response_ns"]
            <= s["completed_ns"]
        )
        assert s["completed_ns"] <= row["operation_wall_seconds"] * 1e9 + 1
        assert s["latency_ns"] == s["response_ns"] - s["scheduled_ns"]
        assert s["search_ns"] <= s["ns"] <= s["response_ns"] - s["started_ns"]
        assert s["verification_ns"] <= s["completed_ns"] - s["response_ns"]
        count = 64 if record["field"] == "mixed" else 32
        index = s["sequence"] % count
        expected_field = (
            "bmp" if record["field"] == "mixed" and index % 2 else "maxscore"
        )
        assert s["field"] == expected_field
        assert s["query"] == (index // 2 if record["field"] == "mixed" else index)
    coverage = defaultdict(int)
    for s in samples:
        coverage[f"{s['field']}/{s['query']}"] += 1
    phases = {}
    for name, initial in [("initial", True), ("steady", False)]:
        chosen = [s for s in samples if (s["scheduled_ns"] < 1e9) == initial]
        dropped = [s for s in rejected if (s["scheduled_ns"] < 1e9) == initial]
        phases[name] = phase(chosen, dropped, 1 if initial else 7)
    cpu, wall = row["operation_cpu_seconds"], row["operation_wall_seconds"]
    return {
        **record,
        "offered": load["offered"],
        "accepted": len(samples),
        "rejected": len(rejected),
        "documents": 32 * len(samples),
        "operation_cpu_seconds": cpu,
        "operation_wall_seconds": wall,
        "completed_per_second": len(samples) / wall,
        "cpu_us_per_completed": cpu * 1e6 / len(samples),
        "reject_fraction": len(rejected) / load["offered"],
        "phases": phases,
        "execution_ms": distribution([s["ns"] / 1e6 for s in samples]),
        "verification_ms": distribution([s["verification_ns"] / 1e6 for s in samples]),
        "runtime_queue_ms": distribution(
            [(s["started_ns"] - s["admitted_ns"]) / 1e6 for s in samples]
        ),
        "coverage": dict(sorted(coverage.items())),
        "peak_cgroup_bytes": int(group["memory.peak"]),
        "cgroup_after": group,
        "stats_before": row["stats_before"],
        "stats_after": stats,
        "kernel_read_bytes": row["kernel_read_bytes"],
        "max_in_flight": load["max_in_flight"],
        "minor_faults": row["minor_faults"],
        "major_faults": row["major_faults"],
        "peak_rss_kib": row["peak_rss_kib"],
        "raw_sha256": digest(path / "result.jsonl"),
        "control_sha256": digest(path / "control.json"),
    }


def export(root):
    campaign = root / "load"
    done = shared.read(campaign / "completed.json")
    assert done["complete"] and done["fixture_unchanged"] and done["failures"] == 0
    records = shared.read(campaign / "cells.json")
    assert len(records) == done["cells"] == 57
    cells, smoke = [], []
    for record in records:
        cell = read_cell(campaign, record)
        (smoke if record["mode"] == "run" else cells).append(cell)
    assert len(cells) == 54 and len(smoke) == 3
    groups = defaultdict(list)
    for c in cells:
        groups[c["field"], c["rate"], c["method"]].append(c)
    summaries = []
    for (field, rate, method), rows in sorted(groups.items()):
        assert len(rows) == 3
        summary = {"field": field, "rate": rate, "method": method, "repeats": 3}
        for key in [
            "accepted",
            "rejected",
            "reject_fraction",
            "completed_per_second",
            "operation_cpu_seconds",
            "cpu_us_per_completed",
            "peak_cgroup_bytes",
        ]:
            summary[key] = statistics.mean(r[key] for r in rows)
        for name in ["initial", "steady"]:
            summary[name] = {
                "reject_fraction": statistics.mean(
                    r["phases"][name]["reject_fraction"] for r in rows
                ),
                "latency_p95_ms": statistics.mean(
                    r["phases"][name]["latency_ms"]["p95"] for r in rows
                ),
                "latency_p99_ms": statistics.mean(
                    r["phases"][name]["latency_ms"]["p99"] for r in rows
                ),
                "generation_p95_ms": statistics.mean(
                    r["phases"][name]["generation_lag_ms"]["p95"] for r in rows
                ),
            }
        summaries.append(summary)
    return {
        "schema": 1,
        "fixture": shared.read(root / "fixture.json"),
        "cells": cells,
        "summaries": summaries,
        "smoke": smoke,
        "offered": sum(c["offered"] for c in cells),
        "verified_queries": sum(c["accepted"] for c in cells),
        "decoded_documents": sum(c["documents"] for c in cells),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    args = parser.parse_args()
    assert shared.read(args.sources / "provenance.json") == shared.read(
        args.root / "provenance.json"
    )
    for name, data in [
        ("results", export(args.root)),
        ("sources", shared.sources(args.sources)),
    ]:
        (HERE / (name + ".json")).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
