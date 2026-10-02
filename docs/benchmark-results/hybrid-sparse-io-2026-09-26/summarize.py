#!/usr/bin/env python3
"""Export verified hybrid sparse cells without copying fixture data or binaries."""

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


def sha(data):
    return hashlib.sha256(data).hexdigest()


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def export(root):
    completed = read(root / "completed.json")
    assert completed["complete"] and completed["fixture_unchanged"]
    assert completed["failures"] == 0
    cells = []
    for cell in read(root / "cells.json"):
        cell = {
            **cell,
            "sparse_reads": cell.get("sparse_reads", cell["variant"] == "hybrid"),
        }
        directory = root / cell["label"]
        control = read(directory / "control.json")
        assert cell["exit_code"] == control["exit_code"] == 0
        assert read(directory / "completed.json")["complete"]
        rows = [
            json.loads(line)
            for line in (directory / "result.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1 and rows[0]["complete"]
        assert rows[0].get("sparse_reads", cell["sparse_reads"]) == cell["sparse_reads"]
        row = rows[0]
        events = dict(
            line.split()
            for line in control["cgroup_after"]["memory.events"].splitlines()
        )
        assert int(events["oom"]) == int(events["oom_kill"]) == 0
        residency = read(root / (cell["label"] + "-residency.json"))
        assert all(
            info["resident_bytes"] == 0
            for name, info in residency.items()
            if name.endswith((".store", ".sparse"))
        )
        intended = sum(
            segment["intended"] for segment in row["memory_before"]["segments"]
        )
        pinned = sum(segment["pinned"] for segment in row["memory_before"]["segments"])
        locked_kib = int(
            next(
                line.split()[1]
                for line in row["memory_before"]["status"].splitlines()
                if line.startswith("VmLck:")
            )
        )
        if cell["pin_mib"]:
            assert intended == pinned and pinned > 0 and locked_kib > 0
        else:
            assert pinned == 0 and locked_kib == 0
        delta = {}
        if row["stats_after"] is not None:
            for name in (
                "submitted",
                "completed",
                "bytes",
                "errors",
                "quarantined_bytes",
                "worker_failures",
            ):
                delta[name] = row["stats_after"][name] - row["stats_before"][name]
            assert (
                delta["errors"]
                == delta["quarantined_bytes"]
                == delta["worker_failures"]
                == 0
            )
            assert row["stats_after"]["active"] == 0
        times = [sample["ns"] / 1e6 for sample in row["samples"]]
        passes = defaultdict(float)
        for sample in row["samples"]:
            passes[sample["pass"]] += sample["ns"] / 1e9
        cells.append(
            {
                **cell,
                "queries": len(times),
                "decoded_documents": sum(
                    sample["documents"] for sample in row["samples"]
                ),
                "wall_seconds": sum(times) / 1000,
                "cpu_seconds": row["operation_cpu_seconds"],
                "p50_ms": percentile(times, 0.5),
                "p95_ms": percentile(times, 0.95),
                "p99_ms": percentile(times, 0.99),
                "passes_seconds": dict(passes),
                "samples": row["samples"],
                "kernel_read_bytes": row["kernel_read_bytes"],
                "major_faults": row["major_faults"],
                "minor_faults": row["minor_faults"],
                "peak_rss_kib": row["peak_rss_kib"],
                "cgroup_peak_bytes": int(control["cgroup_after"]["memory.peak"]),
                "cgroup_after": control["cgroup_after"],
                "pin_intended_bytes": intended,
                "pinned_metadata_bytes": pinned,
                "locked_kib": locked_kib,
                "segments": row["memory_before"]["segments"],
                "stats_delta": delta,
                "stats_after": row["stats_after"],
            }
        )
    assert len(cells) == completed["cells"]
    grouped = defaultdict(list)
    keys = ("field", "budget_mib", "variant", "method", "pin_mib", "sparse_reads")
    for cell in cells:
        grouped[tuple(cell[key] for key in keys)].append(cell)
    groups = [
        {
            **dict(zip(keys, key, strict=True)),
            "repeats": len(rows),
            **{
                f"mean_{name}": statistics.mean(row[name] for row in rows)
                for name in (
                    "wall_seconds",
                    "cpu_seconds",
                    "p95_ms",
                    "peak_rss_kib",
                    "cgroup_peak_bytes",
                    "major_faults",
                    "kernel_read_bytes",
                )
            },
        }
        for key, rows in grouped.items()
    ]
    return {
        "schema": 1,
        "completed": completed,
        "fixture": read(root / "fixture.json"),
        "cpu": read(root / "environment-cpu.json"),
        "verified_queries": sum(cell["queries"] for cell in cells),
        "decoded_documents": sum(cell["decoded_documents"] for cell in cells),
        "groups": groups,
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--exploratory", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).parent
    result = export(args.root)
    sources = read(args.root / "provenance.json")
    roots = dict.fromkeys(sources["variants"], args.root)
    if args.exploratory:
        result["exploratory"] = export(args.exploratory)
        previous = read(args.exploratory / "provenance.json")
        assert (
            previous["compiler"] == sources["compiler"]
            and previous["flags"] == sources["flags"]
        )
        sources["variants"].update(previous["variants"])
        roots.update(dict.fromkeys(previous["variants"], args.exploratory))
    for variant, metadata in sources["variants"].items():
        archive = roots[variant] / (variant + ".tar.gz")
        assert sha(archive.read_bytes()) == metadata["source_sha256"]
        with tarfile.open(archive) as tar:
            metadata["files"] = {
                member.name: sha(tar.extractfile(member).read())
                for member in tar.getmembers()
                if member.isfile()
                and (
                    member.name.endswith(".rs")
                    or Path(member.name).name in ("Cargo.toml", "Cargo.lock")
                )
            }
    for name, value in (("results.json", result), ("sources.json", sources)):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
