#!/usr/bin/env python3
"""Reuse the fixed-arrival validator for the warm-path candidate comparison."""

import argparse
import importlib.util
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "load_export", HERE.parent / "sustained-io-2026-09-26/summarize.py"
)
loads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loads)
waves = loads.shared


def export(root):
    campaign = root / "load"
    done = waves.read(campaign / "completed.json")
    assert done["complete"] and done["fixture_unchanged"] and done["failures"] == 0
    records = waves.read(campaign / "cells.json")
    assert len(records) == done["cells"] == 70
    cells, smoke = [], []
    for record in records:
        assert record["variant"] in ["base", "cursor", "wake"]
        cell = loads.read_cell(campaign, record)
        (smoke if record["mode"] == "run" else cells).append(cell)
    assert len(cells) == 63 and len(smoke) == 7
    groups = defaultdict(list)
    for c in cells:
        groups[c["field"], c["rate"], c["variant"], c["method"]].append(c)
    summaries = []
    for (field, rate, variant, method), rows in sorted(groups.items()):
        assert sorted(r["order"] for r in rows) == [0, 1, 2]
        summary = {"field": field, "rate": rate, "variant": variant, "method": method}
        for key in [
            "accepted",
            "rejected",
            "reject_fraction",
            "completed_per_second",
            "cpu_us_per_completed",
            "operation_cpu_seconds",
            "peak_cgroup_bytes",
        ]:
            summary[key] = statistics.mean(r[key] for r in rows)
        for phase in ["initial", "steady"]:
            summary[phase] = {
                "reject_fraction": statistics.mean(
                    r["phases"][phase]["reject_fraction"] for r in rows
                ),
                "p95_ms": statistics.mean(
                    r["phases"][phase]["latency_ms"]["p95"] for r in rows
                ),
                "p99_ms": statistics.mean(
                    r["phases"][phase]["latency_ms"]["p99"] for r in rows
                ),
            }
        summaries.append(summary)
    return {
        "schema": 1,
        "fixture": waves.read(root / "fixture.json"),
        "cells": cells,
        "summaries": summaries,
        "smoke": smoke,
        "offered": sum(c["offered"] for c in cells),
        "verified_queries": sum(c["accepted"] for c in cells),
        "decoded_documents": sum(c["documents"] for c in cells),
    }


def export_waves(root):
    data = waves.export(root, campaigns=("waves",), expected_repeats=3)
    assert len(data["cells"]) == 72
    for cell in data["cells"]:
        path = root / "waves" / cell["label"]
        group = cell["cgroup_after"]
        assert group["cpuset.cpus.effective"].strip() == "0-7"
        assert int(group["memory.max"]) == 256 * 1024 * 1024
        assert int(group["memory.swap.max"]) == 0
        assert cell["load"] is None and cell["pin_budget_bytes"] == 0
        assert (
            cell["stats_after"] is None
            or cell["stats_after"]["notification_failures"] == 0
        )
        count = 64 if cell["field"] == "mixed" else 32
        samples = cell.pop("samples")
        assert len(samples) == count * 4
        assert sorted((s["pass"], s["field"], s["query"]) for s in samples) == sorted(
            (p, f, q)
            for p in range(4)
            for q in range(32)
            for f in (["maxscore", "bmp"] if count == 64 else ["maxscore"])
        )
        assert all(s["documents"] == 32 for s in samples)
        cell["pass_latency_ms"] = [
            loads.distribution([s["ns"] / 1e6 for s in samples if s["pass"] == p])
            for p in range(4)
        ]
        cell["raw_sha256"] = loads.digest(path / "result.jsonl")
        cell["control_sha256"] = loads.digest(path / "control.json")
        cell.pop("memory_before")
        cell.pop("memory_after")
    return data


def export_profiles(root):
    profiles = {}
    for name in ["mmap", "pool", "uring", "cursor-uring"]:
        path = root / ("profile-" + name)
        report = path.with_name(path.name + "-self.txt")
        text = report.read_text()
        assert "# Total Lost Samples: 0" in text
        row = waves.read(path.with_suffix(".jsonl"))
        assert row["complete"]
        if row["stats_after"]:
            assert all(
                row["stats_after"][k] == 0
                for k in [
                    "errors",
                    "worker_failures",
                    "active",
                    "quarantined_bytes",
                    "notification_failures",
                ]
            )
        entries = []
        for line in text.splitlines():
            match = re.match(
                r"^\s*(\d+\.\d+)%\s+(\S+)\s+(\S+)\s+\[([.k])\]\s+(.*)", line
            )
            if match:
                percent, thread, obj, kind, symbol = match.groups()
                entries.append(
                    {
                        "self_percent": float(percent),
                        "thread": thread,
                        "object": obj,
                        "kind": kind,
                        "symbol": symbol,
                    }
                )
        assert entries
        profiles[name] = {
            "entries": entries,
            "verified_queries": len(row["samples"]),
            "raw_sha256": loads.digest(path.with_suffix(".data")),
            "report_sha256": loads.digest(report),
            "result_sha256": loads.digest(path.with_suffix(".jsonl")),
        }
    return profiles


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for name, data in [
        ("results", export(args.root)),
        ("waves", export_waves(args.root)),
        ("profiles", export_profiles(args.root)),
    ]:
        (HERE / (name + ".json")).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
