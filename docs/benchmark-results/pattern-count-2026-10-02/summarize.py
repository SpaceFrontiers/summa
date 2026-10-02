"""Validate the October 2 paired RGB + pairs HTTP campaign.

Usage: summarize.py <artifact directory> [--output directory]
Artifacts contain campaign/, batch/, campaign.done and rgbpairs-audit.jsonl.
Native count and top-100 digests must match in all four phases, index byte
inventories must match, and HTTP counts and IDs must match the retained
September 30 audit of this exact RGB + pairs index. Cross-engine ranking
agreement is not asserted: this is the canonical shared-input comparison.
"""

import argparse
import collections
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).parent
COMMON = runpy.run_path(str(HERE.parent / "closing-gap-2026-09-24/summarize.py"))
NATIVE = runpy.run_path(str(HERE / "summarize_screen.py"))
read, write = COMMON["read"], COMMON["write"]
PHASES = [
    ("a1", "before"),
    ("b1", "after"),
    ("l1", "luxir"),
    ("l2", "luxir"),
    ("b2", "after"),
    ("a2", "before"),
]
FAMILIES = {
    "prefix3",
    "regex",
    "wildcard",
    "wildcard_scan",
    "and_high_med",
    "or_high_med",
    "high_term",
    "high_phrase",
}
BINARIES = {
    "before": "32a317b050d2cd564a3a22023e92f54ccfc98f65a203e48d5e6166a1a78fce2e",
    "after": "0e7c14c3c6364363f2d7bdb5c7864cacccdbed75366481e1af5d55d70b92be83",
}
INDEX = "/mnt/scratch/fresh-rgb-pairs-index"


def summarize(folder, output):
    assert (folder / "campaign.done").read_text() == "done"
    assert not (folder / "campaign.failed").exists()
    assert read(folder / "isolation-exit.json") == {"returncode": 0}
    native = NATIVE["summarize"](folder / "batch", ["default", "rgb"], 200)
    assert native["binary_sha256"] == BINARIES
    assert (
        read(folder / "http-inventory-after.json")
        == read(folder / "batch/inventory-after.json")["rgb"]
    )
    rows = read(folder / "campaign/a1/agreement.json")
    selected = {r["key"]: r for r in rows if r["include"]}
    counts = collections.Counter(r["query"]["query_class"] for r in selected.values())
    assert set(counts) == FAMILIES
    audits = {
        r["class"] + "\t" + r["query"]: r
        for r in map(
            json.loads, (folder / "rgbpairs-audit.jsonl").read_text().splitlines()
        )
    }
    runs, memory = [], []
    reference_queries, reference_binary = None, None
    requests_by_arm = {}
    for phase, arm in PHASES:
        directory = folder / "campaign" / phase
        engine = "luxir" if arm == "luxir" else "summa"
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        assert read(directory / "variant.json") == {"arm": arm, "index": INDEX}
        server = read(directory / "server.json")
        assert server["cpus"] == "0-14,16-30"
        context = read(directory / engine / "context.json")
        args = context["args"]
        assert context["topology"] == read(directory / engine / "topology-after.json")
        assert args["clients"] == [32] and args["client_cores"] == "15,31"
        assert args["duration"] == 3 and args["repetitions"] == 3
        assert args["session_warmup"] == 40 and args["comparison"] == "shared-input"
        assert set(args["families"]) == FAMILIES
        assert context["comparison_modes"] == ["shared-input"]
        assert (
            context["agreement_sha256"]
            == hashlib.sha256((directory / "agreement.json").read_bytes()).hexdigest()
        )
        if reference_queries is None:
            reference_queries = context["queries"]
        assert context["queries"] == reference_queries
        if engine == "summa":
            assert server["binary_sha256"] == BINARIES[arm]
            assert INDEX in server["command"]
            responses = read(directory / "responses.json")
            assert set(responses) == set(selected)
            for key, row in selected.items():
                expected = audits[key]
                assert (
                    responses[key]["0"]["found"]
                    == expected["exhaustive"]
                    == row["observed"]["summa"]["count"]
                )
                for limit in [10, 100]:
                    assert [d["id"] for d in responses[key][str(limit)]["docs"]] == [
                        d["id"] for d in expected["ranked"][:limit]
                    ]
        else:
            assert (
                server["binary_sha256"]
                == "d848ed884ff43788e8a05596006622c907912c8905f8314f59b08322b4150b89"
            )
            if reference_binary is None:
                reference_binary = server["binary_sha256"]
            assert server["binary_sha256"] == reference_binary
        cells = []
        for path in sorted((directory / engine).glob("*-c32.json")):
            summary, raw = read(path), read(path.with_suffix(".raw.json"))
            cell = COMMON["metrics"](path)
            assert cell["query_count"] == counts[cell["family"]]
            assert summary["clients"] == 32
            bucket_keys = set(raw["workload"]["buckets"].values())
            assert bucket_keys == {
                k
                for k, r in selected.items()
                if r["query"]["query_class"] == cell["family"]
            }
            request_key = (arm, cell["family"], cell["operation"])
            if request_key in requests_by_arm:
                assert summary["requests"] == requests_by_arm[request_key]
            requests_by_arm[request_key] = summary["requests"]
            cells.append(cell)
        assert len(cells) == 24
        assert {(c["family"], c["operation"]) for c in cells} == {
            (f, op) for f in FAMILIES for op in ["TOP_10", "TOP_100", "COUNT"]
        }
        runs.append(
            {
                "phase": phase,
                "variant": arm,
                "binary_sha256": server["binary_sha256"],
                "cells": cells,
            }
        )
        memory.append(
            {
                "phase": phase,
                "variant": arm,
                "mapping_kib": COMMON["mapping_memory"](
                    (directory / "smaps.txt").read_text(),
                    ("before-bin", "count-batch-bin"),
                ),
            }
        )
    for family in FAMILIES:
        for op in ["TOP_10", "TOP_100", "COUNT"]:
            assert (
                requests_by_arm["before", family, op]
                == requests_by_arm["after", family, op]
            )
    comparisons = []
    for family in sorted(FAMILIES):
        for op in ["TOP_10", "TOP_100", "COUNT"]:
            row = {"family": family, "operation": op}
            for arm in ["before", "after", "luxir"]:
                cells = [
                    c
                    for r in runs
                    if r["variant"] == arm
                    for c in r["cells"]
                    if c["family"] == family and c["operation"] == op
                ]
                assert len(cells) == 2
                row[arm] = COMMON["phase_metrics"](cells)
            for arm in ["before", "after", "luxir"]:
                qps = row[arm]["mean_phase_median_qps"]
                row[arm]["over_before_qps"] = (
                    qps / row["before"]["mean_phase_median_qps"]
                )
                row[arm]["over_luxir_qps"] = qps / row["luxir"]["mean_phase_median_qps"]
            comparisons.append(row)
    output.mkdir(exist_ok=True)
    write(output / "native-x86.json", native)
    result = {
        "audited_queries_per_summa_phase": len(selected),
        "excluded_budget_errors": len(rows) - len(selected),
        "runs": runs,
        "comparisons": comparisons,
        "memory": memory,
    }
    write(output / "campaign-results.json", result)
    print("requests", sum(c["requests"] for r in runs for c in r["cells"]))
    for row in comparisons:
        print(
            row["family"],
            row["operation"],
            "speedup",
            round(row["after"]["over_before_qps"], 3),
            "Luxir",
            round(row["after"]["over_luxir_qps"], 3),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    summarize(args.artifacts, args.output)
