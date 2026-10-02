"""Validate the complete October 2 sweep and export the total comparison table.

Timing and memory calculations reuse the established exporter. This validator
also checks fixture identity, all admissions, exact same-index rankings, replay
membership, wire-request identity, server binaries and CPU placement.
"""

import argparse
import csv
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
COMMON = runpy.run_path(str(HERE.parent / "closing-gap-2026-09-24/summarize.py"))
read, write = COMMON["read"], COMMON["write"]
PHASES = [
    ("d1", "default"),
    ("p1", "pairs"),
    ("g1", "rgbpairs"),
    ("l1", "luxir"),
    ("l2", "luxir"),
    ("g2", "rgbpairs"),
    ("p2", "pairs"),
    ("d2", "default"),
]
INDEXES = {
    "default": "fresh-default-index",
    "pairs": "fresh-default-pairs-index",
    "rgbpairs": "fresh-rgb-pairs-index",
}
BINARY = "dc508dc0c32929268600834bc1eedbb8d7c29e8105c4df27ada565a322de74de"
LUXIR = "d848ed884ff43788e8a05596006622c907912c8905f8314f59b08322b4150b89"


def audit(path):
    return {
        r["class"] + "\t" + r["query"]: r
        for r in map(json.loads, path.read_text().splitlines())
    }


def summarize(folder, previous, output):
    assert read(folder / "all-completed.json") == {"complete": True}
    assert not (folder / "failed.txt").exists()
    assert read(folder / "isolation-exit.json") == {"returncode": 0}
    inventory = read(folder / "inventory-before.json")
    assert inventory == read(folder / "inventory-after.json")
    retained = read(previous / "fresh-inventory-before.json")
    for variant, name in INDEXES.items():
        assert inventory[variant] == retained[name], variant
    rows = read(folder / "agreement.json")
    selected = {r["key"]: r for r in rows if r["include"]}
    counts = Counter(r["query"]["query_class"] for r in selected.values())
    assert len(rows) == 826 and len(selected) == 677 and len(counts) == 19
    audits = {}
    for variant in INDEXES:
        probes = {
            r["key"]: r
            for r in read(folder / ("probes-" + variant) / "summa-counts.json")
        }
        assert set(probes) == {r["key"] for r in rows}
        for row in rows:
            for field in ["count", "error"]:
                assert probes[row["key"]].get(field) == row["observed"]["summa"].get(
                    field
                )
        audits[variant] = audit(folder / (variant + "-audit.jsonl"))
        assert audits[variant] == audit(previous / (variant + "-audit.jsonl"))
        assert set(audits[variant]) == set(selected)
        assert all(r["exhaustive"] == r["optimized"] for r in audits[variant].values())
    runs, memory, wire = [], [], {}
    reference_queries = None
    for phase, variant in PHASES:
        directory = folder / phase
        engine = "luxir" if variant == "luxir" else "summa"
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        server = read(directory / "server.json")
        assert server["binary_sha256"] == (LUXIR if engine == "luxir" else BINARY)
        assert server["cpus"] == "0-14,16-30"
        context = read(directory / engine / "context.json")
        args = context["args"]
        assert context["topology"] == read(directory / engine / "topology-after.json")
        assert args["clients"] == [32] and args["client_cores"] == "15,31"
        assert args["duration"] == 3 and args["repetitions"] == 3
        assert args["session_warmup"] == 40 and args["comparison"] == "shared-input"
        assert args["families"] is None and context["comparison_modes"] == [
            "shared-input"
        ]
        assert (
            context["agreement_sha256"]
            == hashlib.sha256((directory / "agreement.json").read_bytes()).hexdigest()
        )
        if reference_queries is None:
            reference_queries = context["queries"]
        assert context["queries"] == reference_queries
        if engine == "summa":
            index = "/mnt/scratch/" + INDEXES[variant]
            assert index in server["command"]
            assert read(directory / "variant.json") == {
                "variant": variant,
                "index": index,
            }
            responses = read(directory / "responses.json")
            assert set(responses) == set(selected)
            for key, expected in audits[variant].items():
                assert (
                    responses[key]["0"]["found"]
                    == expected["exhaustive"]
                    == selected[key]["observed"]["summa"]["count"]
                )
                for limit in [10, 100]:
                    assert [d["id"] for d in responses[key][str(limit)]["docs"]] == [
                        d["id"] for d in expected["ranked"][:limit]
                    ]
        cells = []
        for path in sorted((directory / engine).glob("*-c32.json")):
            summary, raw = read(path), read(path.with_suffix(".raw.json"))
            cell = COMMON["metrics"](path)
            family, op = cell["family"], cell["operation"]
            assert cell["query_count"] == counts[family] and summary["clients"] == 32
            keys = {
                k for k, r in selected.items() if r["query"]["query_class"] == family
            }
            assert set(raw["workload"]["buckets"].values()) == keys
            wire_key = engine, family, op
            if wire_key in wire:
                assert summary["requests"] == wire[wire_key]
            wire[wire_key] = summary["requests"]
            for repetition in raw["repetitions"]:
                assert set(repetition["per_bucket"]) == keys
                assert all(r["errors"] == 0 for r in repetition["per_bucket"].values())
                assert (
                    sum(r["requests"] for r in repetition["per_bucket"].values())
                    == repetition["requests"]
                )
            # Keep the three observed quantiles; never pool or average them as a
            # supposed campaign-wide percentile without the raw histograms.
            cell["latency_us_by_repetition"] = [
                r["overall"]["latency_us"] for r in raw["repetitions"]
            ]
            cells.append(cell)
        assert {(c["family"], c["operation"]) for c in cells} == {
            (f, op) for f in counts for op in ["TOP_10", "TOP_100", "COUNT"]
        }
        assert len(cells) == 57
        runs.append(
            {
                "phase": phase,
                "variant": variant,
                "binary_sha256": server["binary_sha256"],
                "cells": cells,
            }
        )
        memory.append(
            {
                "phase": phase,
                "variant": variant,
                "mapping_kib": COMMON["mapping_memory"](
                    (directory / "smaps.txt").read_text(), ("reviewed-bin",)
                ),
            }
        )
    comparisons = []
    for family in sorted(counts):
        for op in ["TOP_10", "TOP_100", "COUNT"]:
            row = {"family": family, "operation": op, "queries": counts[family]}
            for variant in [*INDEXES, "luxir"]:
                cells = [
                    c
                    for r in runs
                    if r["variant"] == variant
                    for c in r["cells"]
                    if (c["family"], c["operation"]) == (family, op)
                ]
                assert len(cells) == 2
                row[variant] = COMMON["phase_metrics"](cells)
                for percentile in ["p50", "p90", "p99", "p999"]:
                    values = [
                        q[percentile]
                        for c in cells
                        for q in c["latency_us_by_repetition"]
                    ]
                    row[variant][percentile + "_us_observed_range"] = [
                        min(values),
                        max(values),
                    ]
            for variant in INDEXES:
                row[variant]["over_luxir_qps"] = (
                    row[variant]["mean_phase_median_qps"]
                    / row["luxir"]["mean_phase_median_qps"]
                )
            comparisons.append(row)
    result = {
        "binary_sha256": BINARY,
        "luxir_sha256": LUXIR,
        "admitted_queries": len(selected),
        "excluded_budget_errors": len(rows) - len(selected),
        "requests": sum(c["requests"] for r in runs for c in r["cells"]),
        "errors": 0,
        "runs": runs,
        "comparisons": comparisons,
        "memory": memory,
    }
    output.mkdir(exist_ok=True)
    write(output / "campaign-results.json", result)
    lines = [
        "| Family (queries) | Operation | Default QPS (× Luxir) | Pairs QPS (× Luxir) | RGB + pairs QPS (× Luxir) | Luxir QPS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        values = [
            f"{row[v]['mean_phase_median_qps']:,.0f} ({row[v]['over_luxir_qps']:.3f}×)"
            for v in INDEXES
        ]
        values.append(f"{row['luxir']['mean_phase_median_qps']:,.0f}")
        lines.append(
            "| "
            + " | ".join(
                [f"{row['family']} ({row['queries']})", row["operation"], *values]
            )
            + " |"
        )
    (output / "table.md").write_text("\n".join(lines) + "\n")
    with (output / "total.csv").open("w", newline="") as target:
        fields = [
            "family",
            "operation",
            "queries",
            "variant",
            "qps",
            "over_luxir",
            "server_cpu_us",
            "peak_rss_mib",
            "peak_anonymous_rss_mib",
            "p99_us_min",
            "p99_us_max",
        ]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            for variant in [*INDEXES, "luxir"]:
                cell = row[variant]
                writer.writerow(
                    {
                        "family": row["family"],
                        "operation": row["operation"],
                        "queries": row["queries"],
                        "variant": variant,
                        "qps": cell["mean_phase_median_qps"],
                        "over_luxir": cell.get("over_luxir_qps", 1),
                        "server_cpu_us": cell["server_cpu_us_per_request"],
                        "peak_rss_mib": cell["peak_rss_mib"],
                        "peak_anonymous_rss_mib": cell["peak_anonymous_rss_mib"],
                        "p99_us_min": cell["p99_us_observed_range"][0],
                        "p99_us_max": cell["p99_us_observed_range"][1],
                    }
                )
    print("Validated", result["requests"], "requests, zero errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    summarize(args.artifacts, args.previous, args.output)
