#!/usr/bin/env python3
"""Export verified service campaigns; raw corpus/oracle and smaps remain private."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def counters(text):
    return {
        key: int(value) for key, value in (line.split() for line in text.splitlines())
    }


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def distributions(rows):
    values = [row["ns"] / 1e6 for row in rows]
    return {f"p{int(p * 100)}_ms": percentile(values, p) for p in (0.5, 0.95, 0.99)}


def export_campaign(root):
    completed = load(root / "completed.json")
    assert completed["complete"] and completed["fixture_unchanged"]
    assert completed["failures"] == 0
    records = load(root / "cells.json")
    assert len(records) == completed["cells"]
    cells = []
    for record in records:
        assert record["exit_code"] == 0
        cell = root / record["label"]
        result = load(cell / "result.jsonl")
        control = load(cell / "control.json")
        assert result["complete"] and control["exit_code"] == 0
        stats = result.get("stats_after") or {}
        before = result.get("stats_before") or {}
        for key in ("errors", "worker_failures", "quarantined_bytes", "active"):
            assert stats.get(key, 0) == 0
        group = control["cgroup_after"]
        events = counters(group["memory.events"])
        assert events["oom"] == events["oom_kill"] == 0
        residency = load(root / (record["label"] + "-residency.json"))
        assert (
            sum(
                v["resident_bytes"]
                for k, v in residency.items()
                if k.endswith(".store")
            )
            == 0
        )
        passes = []
        for n in sorted({wave["pass"] for wave in result["waves"]}):
            waves = [wave for wave in result["waves"] if wave["pass"] == n]
            samples = [sample for sample in result["samples"] if sample["pass"] == n]
            passes.append(
                {
                    "pass": n,
                    "wall_seconds": sum(w["ns"] for w in waves) / 1e9,
                    **distributions(samples),
                }
            )
        memory = counters(group["memory.stat"])
        cells.append(
            {
                **record,
                "wall_seconds": sum(w["ns"] for w in result["waves"]) / 1e9,
                "cpu_seconds": result["operation_cpu_seconds"],
                "kernel_read_bytes": result["kernel_read_bytes"],
                "peak_rss_kib": result["peak_rss_kib"],
                "cgroup_peak_bytes": int(group["memory.peak"]),
                "cgroup_limit_bytes": int(group["memory.max"]),
                "cgroup_memory_events": events,
                "cgroup_file_refaults": memory.get("workingset_refault_file", 0),
                "verified_addresses": result["verified_documents"],
                "decoded_documents": 0
                if record["workload"] == "retrieval"
                else result["verified_documents"],
                "score_bit_checks": 0
                if record["workload"] == "trace"
                else result["verified_documents"],
                "samples": len(result["samples"]),
                "latency": distributions(result["samples"]),
                "passes": passes,
                "stats_delta": {
                    k: stats.get(k, 0) - before.get(k, 0)
                    for k in (
                        "submitted",
                        "completed",
                        "bytes",
                        "buffer_reuses",
                        "buffer_allocations",
                        "refills",
                    )
                },
                "idle_buffer_bytes": stats.get("idle_buffer_bytes", 0),
            }
        )
    metadata = {
        "name": root.name,
        **completed,
        "oracle_sha256": hashlib.sha256(
            (root / "oracle.json").read_bytes()
        ).hexdigest(),
    }
    return metadata, cells


def source_manifest(roots):
    variants = {}
    for root in roots:
        provenance = load(root / "provenance.json")
        for variant, info in provenance["variants"].items():
            archive = root / (variant + ".tar.gz")
            if not archive.exists():
                continue
            assert (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                == info["source_sha256"]
            )
            files = {}
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    if (
                        member.isfile()
                        and not Path(member.name).name.startswith("._")
                        and (
                            member.name.endswith(".rs")
                            or member.name.endswith("Cargo.toml")
                            or member.name.endswith("Cargo.lock")
                        )
                    ):
                        files[member.name] = hashlib.sha256(
                            source.extractfile(member).read()
                        ).hexdigest()
            variants[variant] = {
                **info,
                "compiler": provenance["compiler"],
                "flags": provenance["flags"],
                "files": files,
            }
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    campaigns, cells = [], []
    for root in args.campaigns:
        metadata, rows = export_campaign(root)
        campaigns.append(metadata)
        cells.extend(rows)
    assert len({c["oracle_sha256"] for c in campaigns}) == 1
    grouped = defaultdict(list)
    for cell in cells:
        grouped[
            (
                cell["workload"],
                cell["budget_gib"],
                cell["variant"],
                cell["method"],
                cell["idle_bytes"],
            )
        ].append(cell)
    groups = []
    for key, rows in grouped.items():
        groups.append(
            {
                **dict(
                    zip(
                        ("workload", "budget_gib", "variant", "method", "idle_bytes"),
                        key,
                        strict=True,
                    )
                ),
                "repeats": len(rows),
                **{
                    f"mean_{name}": statistics.mean(row[name] for row in rows)
                    for name in (
                        "wall_seconds",
                        "cpu_seconds",
                        "peak_rss_kib",
                        "cgroup_peak_bytes",
                        "kernel_read_bytes",
                    )
                },
                "reuses": sum(row["stats_delta"]["buffer_reuses"] for row in rows),
                "allocations": sum(
                    row["stats_delta"]["buffer_allocations"] for row in rows
                ),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "campaigns": campaigns,
        "verified_addresses": sum(c["verified_addresses"] for c in cells),
        "decoded_documents": sum(c["decoded_documents"] for c in cells),
        "score_bit_checks": sum(c["score_bit_checks"] for c in cells),
        "groups": groups,
        "cells": cells,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "sources.json").write_text(
        json.dumps(source_manifest(args.campaigns), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
