"""Validate the matched four-arm pattern replay, including the old regression."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).parent
COMMON = runpy.run_path(str(HERE.parent / "closing-gap-2026-09-24/summarize.py"))
read, write = COMMON["read"], COMMON["write"]
BINARIES = {
    "old_before": "32a317b050d2cd564a3a22023e92f54ccfc98f65a203e48d5e6166a1a78fce2e",
    "old_count": "0e7c14c3c6364363f2d7bdb5c7864cacccdbed75366481e1af5d55d70b92be83",
    "before": "70a97015fc96cfac073db733e1af8e8d15b72c3eaa2d9c7f6667b83f8776dbb8",
    "reviewed": "dc508dc0c32929268600834bc1eedbb8d7c29e8105c4df27ada565a322de74de",
}
FAMILIES = {
    "prefix3",
    "regex",
    "wildcard",
    "wildcard_scan",
    "high_term",
    "and_high_med",
}


def summarize(root):
    folder = root / "control"
    assert read(folder / "completed.json") == {"complete": True}
    assert read(folder / "isolation-exit.json") == {"returncode": 0}
    assert (
        read(folder / "fixture-after.json")
        == read(root / "inventory-after.json")["rgbpairs"]
    )
    assert read(folder / "host-restored.json") == {
        "apparmor_restrict_unprivileged_userns": 1,
        "control_fixture_unchanged": True,
    }
    assert read(folder / "binaries.json") == BINARIES
    assert not (folder / "failed.txt").exists()
    phases = [
        (arm + str(round), arm)
        for round, order in [(1, list(BINARIES)), (2, list(BINARIES)[::-1])]
        for arm in order
    ]
    assert read(folder / "phases.json") == [list(p) for p in phases]
    rows = [
        r
        for r in read(root / "agreement.json")
        if r["query"]["query_class"] in FAMILIES
    ]
    selected = {r["key"]: r for r in rows if r["include"]}
    assert len(selected) == 154
    audits = {
        r["class"] + "\t" + r["query"]: r
        for r in map(
            json.loads, (root / "rgbpairs-audit.jsonl").read_text().splitlines()
        )
    }
    runs, memory, requests = [], [], {}
    for phase, arm in phases:
        directory = folder / phase
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        server = read(directory / "server.json")
        assert (
            server["binary_sha256"] == BINARIES[arm] and server["cpus"] == "0-14,16-30"
        )
        assert "/mnt/scratch/fresh-rgb-pairs-index" in server["command"]
        context = read(directory / "summa/context.json")
        args = context["args"]
        assert context["topology"] == read(directory / "summa/topology-after.json")
        assert args["clients"] == [32] and args["client_cores"] == "15,31"
        assert (
            args["duration"] == 3
            and args["repetitions"] == 3
            and args["session_warmup"] == 40
        )
        assert (
            set(args["families"]) == FAMILIES and args["comparison"] == "shared-input"
        )
        assert (
            context["agreement_sha256"]
            == hashlib.sha256((directory / "agreement.json").read_bytes()).hexdigest()
        )
        responses = read(directory / "responses.json")
        assert set(responses) == set(selected)
        for key, row in selected.items():
            assert (
                responses[key]["0"]["found"]
                == audits[key]["exhaustive"]
                == row["observed"]["summa"]["count"]
            )
            for limit in [10, 100]:
                assert [d["id"] for d in responses[key][str(limit)]["docs"]] == [
                    d["id"] for d in audits[key]["ranked"][:limit]
                ]
        cells = []
        for path in sorted((directory / "summa").glob("*-c32.json")):
            cell = COMMON["metrics"](path)
            summary, raw = read(path), read(path.with_suffix(".raw.json"))
            key = cell["family"], cell["operation"]
            if key in requests:
                assert requests[key] == summary["requests"]
            requests[key] = summary["requests"]
            chosen = {
                k
                for k, row in selected.items()
                if row["query"]["query_class"] == cell["family"]
            }
            assert set(raw["workload"]["buckets"].values()) == chosen and cell[
                "query_count"
            ] == len(chosen)
            for repetition in raw["repetitions"]:
                assert set(repetition["per_bucket"]) == chosen
                assert all(v["errors"] == 0 for v in repetition["per_bucket"].values())
            cells.append(cell)
        assert len(cells) == 18
        assert {(c["family"], c["operation"]) for c in cells} == {
            (f, op) for f in FAMILIES for op in ["TOP_10", "TOP_100", "COUNT"]
        }
        runs.append({"phase": phase, "variant": arm, "cells": cells})
        memory.append(
            {
                "phase": phase,
                "variant": arm,
                "mapping_kib": COMMON["mapping_memory"](
                    (directory / "smaps.txt").read_text(),
                    ("before-bin", "count-batch-bin", "reviewed-bin"),
                ),
            }
        )
    comparisons = []
    for family in sorted(FAMILIES):
        for operation in ["TOP_10", "TOP_100", "COUNT"]:
            row = {"family": family, "operation": operation}
            for arm in BINARIES:
                cells = [
                    c
                    for r in runs
                    if r["variant"] == arm
                    for c in r["cells"]
                    if (c["family"], c["operation"]) == (family, operation)
                ]
                assert len(cells) == 2
                row[arm] = COMMON["phase_metrics"](cells)
            for arm in BINARIES:
                qps = row[arm]["mean_phase_median_qps"]
                row[arm]["over_old_before"] = (
                    qps / row["old_before"]["mean_phase_median_qps"]
                )
                row[arm]["over_before"] = qps / row["before"]["mean_phase_median_qps"]
            comparisons.append(row)
    return {
        "binaries": BINARIES,
        "admitted_queries": len(selected),
        "excluded_budget_errors": len(rows) - len(selected),
        "requests": sum(c["requests"] for r in runs for c in r["cells"]),
        "errors": 0,
        "runs": runs,
        "comparisons": comparisons,
        "memory": memory,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    result = summarize(args.artifacts)
    write(HERE / "control-results.json", result)
    lines = [
        "| Family | Operation | Old baseline QPS | Old count QPS | Review baseline QPS | Reviewed QPS | Reviewed / baseline | Old count / old baseline | Reviewed CPU / baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["comparisons"]:
        qps = [f"{row[arm]['mean_phase_median_qps']:,.0f}" for arm in BINARIES]
        cpu_ratio = (
            row["reviewed"]["server_cpu_us_per_request"]
            / row["before"]["server_cpu_us_per_request"]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["operation"],
                    *qps,
                    f"{row['reviewed']['over_before']:.3f}×",
                    f"{row['old_count']['over_old_before']:.3f}×",
                    f"{cpu_ratio:.3f}×",
                ]
            )
            + " |"
        )
    (HERE / "control-table.md").write_text("\n".join(lines) + "\n")
    print(result["requests"], "requests, zero errors")
    for row in result["comparisons"]:
        print(
            row["family"],
            row["operation"],
            "review / before",
            round(row["reviewed"]["over_before"], 3),
            "old count / old before",
            round(row["old_count"]["over_old_before"], 3),
            "review / old before",
            round(row["reviewed"]["over_old_before"], 3),
        )
