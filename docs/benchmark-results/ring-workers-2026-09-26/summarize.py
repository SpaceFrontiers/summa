#!/usr/bin/env python3
"""Reuse the concurrent sparse exporter and add initially cold/warmed phase totals."""

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--campaigns", nargs="+", default=("screen", "confirmation"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "sparse_export",
        Path(__file__).resolve().parent.parent
        / "sparse-concurrency-2026-09-26/summarize.py",
    )
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    data = shared.export(
        args.root, campaigns=args.campaigns, expected_repeats=args.repeats
    )
    grouped = defaultdict(list)
    for cell in data["cells"]:
        group = cell["cgroup_after"]
        assert group["cpuset.cpus.effective"].strip() == "0-7"
        assert int(group["memory.max"]) == 256 * 1024 * 1024
        assert int(group["memory.swap.max"]) == 0
        assert cell["stats_after"]["max_active"] <= 8
        assert cell["stats_after"].get("notification_failures", 0) == 0
        if cell["method"] == "uring" and "wakeups" in cell["stats_after"]:
            assert cell["stats_after"]["wakeups"] > 0
        phases = cell["phases"]
        assert [p["pass"] for p in phases] == list(range(8))
        assert sum(p["queries"] for p in phases) == cell["queries"]
        for phase in phases:
            assert phase["queries"] == sum(
                sample["pass"] == phase["pass"] for sample in cell["samples"]
            )
        assert (
            abs(sum(p["cpu_seconds"] for p in phases) - cell["operation_cpu_seconds"])
            < 1e-6
        )
        assert (
            abs(sum(p["wall_seconds"] for p in phases) - cell["operation_wall_seconds"])
            < 1e-6
        )
        for name, selected in (("initial", phases[:1]), ("warm", phases[1:])):
            wall = sum(p["wall_seconds"] for p in selected)
            cpu = sum(p["cpu_seconds"] for p in selected)
            queries = sum(p["queries"] for p in selected)
            times = [
                s["ns"] / 1e6
                for s in cell["samples"]
                if (s["pass"] == 0) == (name == "initial")
            ]
            cell[name] = {
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "queries": queries,
                "qps": queries / wall,
                "cpu_us_per_query": cpu * 1e6 / queries,
                "p95_ms": shared.percentile(times, 0.95),
            }
        keys = (
            "campaign",
            "variant",
            "method",
            "field",
            "scope",
            "runtime_threads",
            "concurrency",
        )
        grouped[tuple(cell[k] for k in keys)].append(cell)
    # Scheduling variants must perform the same logical reads for a workload.
    work = defaultdict(set)
    for cell in data["cells"]:
        stats = cell["stats_after"]
        work[(cell["field"], cell["scope"])].add((stats["submitted"], stats["bytes"]))
    assert all(len(counts) == 1 for counts in work.values())
    for summary in data["summaries"]:
        rows = grouped[tuple(summary[k] for k in keys)]
        expected = args.repeats or (2 if summary["campaign"] == "screen" else 3)
        assert len(rows) == expected
        for phase in ("initial", "warm"):
            summary[phase] = {
                key: statistics.mean(row[phase][key] for row in rows)
                for key in rows[0][phase]
            }
    sources = shared.sources(args.sources)
    sources["patches"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.sources.glob("*.patch"))
    }
    for name, value in (("results", data), ("sources", sources)):
        (output / (name + ".json")).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
