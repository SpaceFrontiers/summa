"""Validate and summarize the September 29 JCC campaign.

The first September 29 campaign (`summarize.py`) found frequent-term exact
counts 3-7% slower with the new binary although their code is unchanged: in
the new binary the loop closing `BlockPostingIterator::fill_doc_window`
fuses a compare and branch across a 32-byte boundary (the Skylake-family JCC
erratum), which costs little on one thread and about 7% with 30 workers on 16
hyperthreaded cores. This campaign adds `jcc`: the same tree built with
`-C llvm-args=-x86-branches-within-32B-boundaries`. Variants `before` (the
September 28 binary), `code` and `jcc` serve the same fresh default build as
the first campaign. Phases b1 c1 j1 l1 j2 c2 b2, all 19 families, 32
clients. Audits follow `summarize.py`, and all three binaries must agree
exactly.

Usage: summarize-jcc.py <campaign folder> <folder with final.tar.gz>
"""

import argparse
import collections
import hashlib
import json
import runpy
from pathlib import Path

COMMON = runpy.run_path(
    str(Path(__file__).parent.parent / "closing-gap-2026-09-24/summarize.py")
)
read, write = COMMON["read"], COMMON["write"]
PHASES = [
    ("b1", "before"),
    ("c1", "code"),
    ("j1", "jcc"),
    ("l1", "luxir"),
    ("j2", "jcc"),
    ("c2", "code"),
    ("b2", "before"),
]
FRESH = "/mnt/scratch/fresh-default-index"
INDEXES = {"before": FRESH, "code": FRESH, "jcc": FRESH}
BEFORE_SHA256 = "73ada083ae59e5758fa39dae2cc450292d3bcc709986f5dc39ceaabd244d948f"
BINARY_SHA256 = "cb0fc90b24b465895c50582c9b408cd5e6ef6c1f21e88aa3e3f351573712e651"
ARCHIVE_SHA256 = "931113f3b48c36e3dd53a1e3d96286e5a49c965341f182df03ab907eda9540dc"
JCC_SHA256 = "84ba77a626e7fbd66233c611f01cbb0a2995822ac3c9908dd54394416ac2c2c2"
BINARIES = {"before": BEFORE_SHA256, "code": BINARY_SHA256, "jcc": JCC_SHA256}


def audit(path):
    return {
        r["class"] + "\t" + r["query"]: r
        for r in map(json.loads, path.read_text().splitlines())
    }


def groups(ranked):
    if not ranked:
        return []
    last = ranked[-1]["score_bits"]
    return sorted(
        collections.Counter(
            (r["score_bits"], r["id"]) for r in ranked if r["score_bits"] != last
        ).items()
    )


def summarize(folder, archives, output):
    assert read(folder / "all-completed.json") == {"complete": True}
    assert read(folder / "inventory-before.json") == read(
        folder / "inventory-after.json"
    )
    assert read(folder / "fresh-inventory-before.json") == read(
        folder / "fresh-inventory-after.json"
    )
    archive = archives / "final.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256
    rows = read(folder / "agreement.json")
    selected = {r["key"] for r in rows if r["include"]}
    probes = {r["key"]: r for r in read(folder / "probes/summa-counts.json")}
    assert set(probes) == {r["key"] for r in rows}
    for row in rows:
        expected, actual = row["observed"]["summa"], probes[row["key"]]
        assert actual.get("count") == expected.get("count")
        assert actual.get("error") == expected.get("error")

    baseline = audit(folder / "baseline-audit.jsonl")
    assert set(baseline) == selected
    audits, tie_order_only = {}, {}
    for variant in INDEXES:
        audits[variant] = audit(folder / (variant + "-audit.jsonl"))
        assert set(audits[variant]) == selected
        tie_order_only[variant] = 0
        for key, row in audits[variant].items():
            base = baseline[key]
            assert row["exhaustive"] == row["optimized"] == base["exhaustive"], key
            assert row["plan"] == base["plan"], key
            ranked, base_ranked = row.get("ranked", []), base.get("ranked", [])
            assert [r["score_bits"] for r in ranked] == [
                r["score_bits"] for r in base_ranked
            ]
            assert groups(ranked) == groups(base_ranked), key
            tie_order_only[variant] += ranked != base_ranked
    assert audits["before"] == audits["code"] == audits["jcc"]

    runs, memory = [], []
    for phase, variant in PHASES:
        directory = folder / phase
        engine = "luxir" if variant == "luxir" else "summa"
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        server = read(directory / "server.json")
        if engine == "summa":
            assert server["binary_sha256"] == BINARIES[variant]
            recorded = read(directory / "variant.json")
            assert (
                recorded["variant"] == variant and recorded["index"] == INDEXES[variant]
            )
            assert INDEXES[variant] in server["command"]
            responses = read(directory / "responses.json")
            assert set(responses) == selected
            for key, expected in audits[variant].items():
                assert responses[key]["0"]["found"] == expected["exhaustive"]
                for limit in [10, 100]:
                    assert [d["id"] for d in responses[key][str(limit)]["docs"]] == [
                        d["id"] for d in expected["ranked"][:limit]
                    ]
        else:
            validation = read(directory / "validation.json")
            assert not validation["errors"] and validation["queries"] == len(selected)
        cells = [
            COMMON["metrics"](p)
            for p in sorted((directory / engine).glob("*-c32.json"))
        ]
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
                    (directory / "smaps.txt").read_text(), ("final-bin",)
                ),
            }
        )

    variants = ["before", "code", "jcc", "luxir"]
    comparisons = []
    for family in sorted({c["family"] for c in runs[0]["cells"]}):
        for operation in ["TOP_10", "TOP_100", "COUNT"]:
            comparison = {"family": family, "operation": operation}
            for variant in variants:
                cells = [
                    c
                    for r in runs
                    if r["variant"] == variant
                    for c in r["cells"]
                    if c["family"] == family and c["operation"] == operation
                ]
                assert len(cells) == sum(v == variant for _, v in PHASES)
                comparison[variant] = COMMON["phase_metrics"](cells)
            for variant in variants:
                qps = comparison[variant]["mean_phase_median_qps"]
                comparison[variant]["over_before_qps"] = (
                    qps / comparison["before"]["mean_phase_median_qps"]
                )
                comparison[variant]["over_luxir_qps"] = (
                    qps / comparison["luxir"]["mean_phase_median_qps"]
                )
            comparisons.append(comparison)

    write(
        output / "jcc-campaign-results.json",
        {
            "indexes": INDEXES,
            "build": {
                "archive_sha256": ARCHIVE_SHA256,
                "binary_sha256": BINARY_SHA256,
                "before_binary_sha256": BEFORE_SHA256,
                "jcc_binary_sha256": JCC_SHA256,
            },
            "audited_queries_per_variant": len(baseline),
            "fresh_audits_differing_only_in_tie_order": tie_order_only,
            "excluded_budget_errors": len(rows) - len(selected),
            "runs": runs,
            "comparisons": comparisons,
            "memory": memory,
        },
    )
    print("requests", sum(c["requests"] for r in runs for c in r["cells"]))
    for c in comparisons:
        print(
            c["family"],
            c["operation"],
            *(
                f"{v}={c[v]['mean_phase_median_qps']:.1f}"
                f"(luxir {c[v]['over_luxir_qps']:.3f}x)"
                for v in variants
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("archives", type=Path, help="folder with final.tar.gz")
    args = parser.parse_args()
    summarize(args.artifacts, args.archives, Path(__file__).parent)
