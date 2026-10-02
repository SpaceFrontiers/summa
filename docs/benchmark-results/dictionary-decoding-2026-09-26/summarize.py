"""Validate and summarize the paired dictionary decoder experiment."""

import argparse
import hashlib
import json
import runpy
import tarfile
from pathlib import Path

COMMON = runpy.run_path(
    str(Path(__file__).parent.parent / "closing-gap-2026-09-24/summarize.py")
)
read, write = COMMON["read"], COMMON["write"]


def summarize(folder, output):
    assert read(folder / "all-completed.json") == {"complete": True}
    assert read(folder / "inventory-before.json") == read(
        folder / "inventory-after.json"
    )
    rows = read(folder / "agreement.json")
    selected = {r["key"] for r in rows if r["include"]}
    probes = {r["key"]: r for r in read(folder / "probes/summa-counts.json")}
    assert set(probes) == {r["key"] for r in rows}
    for row in rows:
        expected, actual = row["observed"]["summa"], probes[row["key"]]
        assert actual.get("count") == expected.get("count")
        assert actual.get("error") == expected.get("error")
    audits = {
        r["class"] + "\t" + r["query"]: r
        for r in map(
            json.loads, (folder / "baseline-audit.jsonl").read_text().splitlines()
        )
    }
    assert set(audits) == selected
    provenance = {
        v: read(folder / (v + "-provenance.json")) for v in ["before", "after"]
    }
    for variant in provenance:
        archive = folder / (variant + ".tar.gz")
        manifest = provenance[variant]
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == manifest["archive_sha256"]
        )
        with tarfile.open(archive) as source:
            captured = {
                member.name: hashlib.sha256(
                    source.extractfile(member).read()
                ).hexdigest()
                for member in source
                if member.isfile() and member.name in manifest["sources"]
            }
        assert captured == manifest["sources"]
        actual = {
            r["class"] + "\t" + r["query"]: r
            for r in map(
                json.loads,
                (folder / (variant + "-audit.jsonl")).read_text().splitlines(),
            )
        }
        assert actual == audits
    assert provenance["before"]["compiler"] == provenance["after"]["compiler"]
    assert provenance["before"]["flags"] == provenance["after"]["flags"]
    assert (
        provenance["before"]["sources"].keys() == provenance["after"]["sources"].keys()
    )
    assert provenance["before"]["binary_sha256"] != provenance["after"]["binary_sha256"]
    changed = [
        k
        for k in provenance["before"]["sources"]
        if provenance["before"]["sources"][k] != provenance["after"]["sources"][k]
    ]
    assert changed == ["summa-core/src/structures/vint.rs"], changed
    runs, memory = [], []
    phases = [
        ("a1", "before"),
        ("b1", "after"),
        ("l1", "luxir"),
        ("b2", "after"),
        ("a2", "before"),
        ("l2", "luxir"),
    ]
    for phase, variant in phases:
        directory = folder / phase
        engine = "luxir" if variant == "luxir" else "summa"
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        server = read(directory / "server.json")
        if engine == "summa":
            assert server["binary_sha256"] == provenance[variant]["binary_sha256"]
            responses = read(directory / "responses.json")
            assert set(responses) == selected
            for key, audit in audits.items():
                assert responses[key]["0"]["found"] == audit["exhaustive"]
                for limit in [10, 100]:
                    assert [d["id"] for d in responses[key][str(limit)]["docs"]] == [
                        d["id"] for d in audit["ranked"][:limit]
                    ]
        else:
            validation = read(directory / "validation.json")
            assert not validation["errors"] and validation["queries"] == len(selected)
        cells = [
            COMMON["metrics"](p)
            for p in sorted((directory / engine).glob("*-c32.json"))
        ]
        assert len(cells) == 15
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
                    (directory / "smaps.txt").read_text(), ("before-bin", "after-bin")
                ),
            }
        )
    comparisons = []
    for family in sorted({c["family"] for c in runs[0]["cells"]}):
        for operation in ["TOP_10", "TOP_100", "COUNT"]:
            comparison = {"family": family, "operation": operation}
            for variant in ["before", "after", "luxir"]:
                cells = [
                    c
                    for r in runs
                    if r["variant"] == variant
                    for c in r["cells"]
                    if c["family"] == family and c["operation"] == operation
                ]
                assert len(cells) == 2
                comparison[variant] = COMMON["phase_metrics"](cells)
            for control in ["before", "luxir"]:
                comparison["after_over_" + control + "_qps"] = (
                    comparison["after"]["mean_phase_median_qps"]
                    / comparison[control]["mean_phase_median_qps"]
                )
            comparisons.append(comparison)
    assert read(folder / "profiles-completed.json") == {"complete": True}
    profiles = []
    for variant in ["before", "after"]:
        directory = folder / ("profile-" + variant)
        assert read(directory / "completed.json") == {"complete": True}
        assert (
            read(directory / "server.json")["binary_sha256"]
            == provenance[variant]["binary_sha256"]
        )
        report = (directory / "profile.txt").read_text()
        event, symbols = COMMON["parse_profile"](report)
        summaries = sorted((directory / "summa").glob("*-c32.json"))
        assert len(summaries) == 3
        for path in summaries:
            summary, raw = read(path), read(path.with_suffix(".raw.json"))
            assert not summary["errors"] and not raw["errors"]
            assert summary["memory"]["error"] is None
            assert len(raw["repetitions"]) == 1 and raw["repetitions"][0]["errors"] == 0
        profiles.append(
            {
                "variant": variant,
                "samples": event[1],
                "event": event[2],
                "symbols": symbols,
                "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
            }
        )
    diagnostics = {
        phase: list(
            map(
                json.loads,
                (folder / (phase + "-diagnose.jsonl")).read_text().splitlines(),
            )
        )
        for phase in ["a1", "b1", "b2", "a2"]
    }

    def keys(rows):
        return [
            (r["class"], r["query"], r["limit"], r["returned"], r["response_bytes"])
            for r in rows
        ]

    assert all(
        keys(rs) == keys(diagnostics["a1"]) and all(not r["instrumented"] for r in rs)
        for rs in diagnostics.values()
    )
    write(
        output / "profiles.json",
        {
            "captures": profiles,
            "diagnostics": diagnostics,
            "limitations": [
                "Flat self CPU attribution mixes warmup and TOP_10/TOP_100/COUNT operations.",
                "Profile throughput is excluded from the paired comparison.",
                "Native diagnostics run separately on CPU 0 and include 20 warmup plus 100 measured iterations per query/limit.",
            ],
        },
    )
    write(
        output / "results.json",
        {
            "audited_queries_per_variant": len(audits),
            "excluded_budget_errors": len(rows) - len(selected),
            "http_checks": {"summa": 4 * 3 * len(audits), "luxir": 2 * 3 * len(audits)},
            "runs": runs,
            "comparisons": comparisons,
            "memory": memory,
        },
    )
    # Compiler metadata and hashes are sufficient; private machine paths in build commands stay in raw artifacts.
    write(
        output / "sources.json",
        {
            v: {
                k: p[k]
                for k in [
                    "archive_sha256",
                    "binary_sha256",
                    "compiler",
                    "flags",
                    "sources",
                ]
            }
            for v, p in provenance.items()
        },
    )
    print("requests", sum(c["requests"] for r in runs for c in r["cells"]))
    for c in comparisons:
        print(
            c["family"],
            c["operation"],
            *(
                round(c[v]["mean_phase_median_qps"], 2)
                for v in ["before", "after", "luxir"]
            ),
            "gain",
            round(c["after_over_before_qps"], 4),
            "relative",
            round(c["after_over_luxir_qps"], 4),
            "cpu",
            *(
                round(c[v]["server_cpu_us_per_request"], 2)
                for v in ["before", "after", "luxir"]
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    summarize(args.artifacts, Path(__file__).parent)
