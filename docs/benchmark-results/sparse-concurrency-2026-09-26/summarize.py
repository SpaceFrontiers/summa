#!/usr/bin/env python3
"""Export the concurrent sparse campaign; never infer throughput from summed latency."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def export(root, campaigns=("before", "after", "confirm"), expected_repeats=2):
    cells = []
    for variant in campaigns:
        campaign = root / variant
        done = read(campaign / "completed.json")
        assert done["complete"] and done["fixture_unchanged"] and done["failures"] == 0
        records = read(campaign / "cells.json")
        assert len(records) == done["cells"]
        for record in records:
            path = campaign / record["label"]
            control = read(path / "control.json")
            row = read(path / "result.jsonl")
            assert record["exit_code"] == control["exit_code"] == 0
            assert row["complete"] and read(path / "completed.json")["complete"]
            assert row["concurrency"] == record["concurrency"]
            assert row["runtime_threads"] == record["runtime_threads"]
            assert row["sparse_reads"] == (record["scope"] == "sparse")
            events = dict(
                line.split()
                for line in control["cgroup_after"]["memory.events"].splitlines()
            )
            assert int(events["oom"]) == int(events["oom_kill"]) == 0
            residency = read(campaign / (record["label"] + "-residency.json"))
            assert all(
                info["resident_bytes"] == 0
                for name, info in residency.items()
                if name.endswith((".store", ".sparse"))
            )
            stats = row["stats_after"]
            if stats:
                assert all(
                    stats[key] == 0
                    for key in (
                        "errors",
                        "worker_failures",
                        "active",
                        "quarantined_bytes",
                    )
                )
                assert stats["completed"] == stats["submitted"]
            times = [s["ns"] / 1e6 for s in row["samples"]]
            searches = [s["search_ns"] / 1e6 for s in row["samples"]]
            cells.append(
                {
                    **record,
                    **row,
                    "campaign": variant,
                    "queries": len(times),
                    "documents": sum(s["documents"] for s in row["samples"]),
                    "qps": len(times) / row["operation_wall_seconds"],
                    "p50_ms": percentile(times, 0.5),
                    "p95_ms": percentile(times, 0.95),
                    "p99_ms": percentile(times, 0.99),
                    "search_p95_ms": percentile(searches, 0.95),
                    "peak_cgroup_bytes": int(control["cgroup_after"]["memory.peak"]),
                    "cgroup_after": control["cgroup_after"],
                }
            )
    groups = defaultdict(list)
    keys = (
        "campaign",
        "variant",
        "field",
        "runtime_threads",
        "concurrency",
        "method",
        "scope",
    )
    for cell in cells:
        groups[tuple(cell[key] for key in keys)].append(cell)
    summaries = []
    for key, rows in sorted(groups.items()):
        assert len(rows) >= 2
        if expected_repeats is not None:
            assert len(rows) == expected_repeats
        summaries.append(
            {
                **dict(zip(keys, key, strict=True)),
                "repeats": len(rows),
                **{
                    name: statistics.mean(row[name] for row in rows)
                    for name in (
                        "operation_wall_seconds",
                        "operation_cpu_seconds",
                        "qps",
                        "p50_ms",
                        "p95_ms",
                        "p99_ms",
                        "search_p95_ms",
                        "voluntary_switches",
                        "involuntary_switches",
                        "peak_cgroup_bytes",
                        "kernel_read_bytes",
                    )
                },
            }
        )
    return {
        "schema": 1,
        "fixture": read(root / "fixture.json"),
        "summaries": summaries,
        "cells": cells,
        "verified_queries": sum(c["queries"] for c in cells),
        "decoded_documents": sum(c["documents"] for c in cells),
    }


def sources(source_root):
    result = read(source_root / "provenance.json")
    for variant, info in result["variants"].items():
        archive = source_root / (variant + ".tar.gz")
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["source_sha256"]
        with tarfile.open(archive) as tar:
            info["files"] = {
                entry.name: hashlib.sha256(tar.extractfile(entry).read()).hexdigest()
                for entry in tar.getmembers()
                if entry.isfile()
                and (
                    entry.name.endswith(".rs")
                    or Path(entry.name).name in ("Cargo.toml", "Cargo.lock")
                )
            }
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    for name, data in (
        ("results", export(args.root)),
        ("sources", sources(args.sources)),
    ):
        (output / (name + ".json")).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
