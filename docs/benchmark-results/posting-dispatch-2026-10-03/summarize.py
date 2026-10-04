#!/usr/bin/env python3
"""Verify retained evidence and regenerate the complete comparison tables."""

import collections
import csv
import hashlib
import io
import json
import re
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def median(values):
    return statistics.median(values)


def samples(archive, name):
    return [json.loads(line) for line in archive.read(name).decode().splitlines()]


def table(title, headers, rows):
    return [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        "",
    ]


def screen(archive, stem, group_column):
    result = []
    for host in ("arm", "x86"):
        groups = collections.defaultdict(list)
        raw = archive.read(f"{stem}-{host}.csv").decode()
        for row in csv.DictReader(io.StringIO(raw)):
            cell = row[group_column], row.get("capacity", "")
            groups[cell, row["kernel"]].append(
                int(row["elapsed_ns"]) / int(row["calls"])
            )
        for (cell, kernel), values in groups.items():
            baseline = median(groups[cell, "production"])
            result.append(
                {
                    "host": host,
                    "shape": cell[0],
                    "capacity": cell[1],
                    "kernel": kernel,
                    "samples": len(values),
                    "ns": median(values),
                    "speedup": baseline / median(values),
                }
            )
    return result


def resource(text, host):
    if host == "arm":
        cpu = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
        rss = re.search(r"(\d+)\s+maximum resident set size", text)
        return float(cpu[2]) + float(cpu[3]), int(rss[1]) / 2**20
    user = re.search(r"User time \(seconds\): ([\d.]+)", text)
    system = re.search(r"System time \(seconds\): ([\d.]+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    return float(user[1]) + float(system[1]), int(rss[1]) / 1024


def integrated(archive, suite):
    result, resources = [], []
    query_file = {
        "integrated": "queries.jsonl",
        "multi": "multi-queries.jsonl",
        "selected": "selected-queries.jsonl",
    }[suite]
    queries = samples(archive, query_file)
    families = {q["query"]: q["family"] for q in queries}
    assert len(families) == len(queries)
    phases = ("a1", "b1", "b2", "a2")
    repetitions = 7 if suite == "selected" else 15
    if suite == "multi":
        phases = ("a1", "b1", "c1", "c2", "b2", "a2")
        repetitions = 21
    hosts = (
        [("x86", ("default", "rgb-pairs"))]
        if suite == "selected"
        else [("arm", ("bitmap", "rgb")), ("x86", ("default", "rgb-pairs"))]
    )
    for host, indexes in hosts:
        folder = f"{host}-{suite}"
        assert json.loads(archive.read(f"{folder}/done.json"))["status"] == "pass"
        assert json.loads(
            archive.read(f"{folder}/inventory-before.json")
        ) == json.loads(archive.read(f"{folder}/inventory-after.json"))
        for index in indexes:
            variants = (
                ("baseline", "candidate")
                if suite != "multi"
                else ("baseline", "candidate", "refined")
            )
            audit_reference = None
            for variant in variants:
                audit = samples(archive, f"{folder}/{index}-{variant}-audit.jsonl")
                assert len(audit) == len(queries)
                assert all(row["optimized"] == row["exhaustive"] for row in audit)
                if audit_reference is None:
                    audit_reference = audit
                assert audit == audit_reference
            for limit in (10, 100, 0):
                grouped = collections.defaultdict(lambda: collections.defaultdict(list))
                reference = None
                process = {}
                for phase in phases:
                    label = f"{folder}/{index}-{phase}-{limit}"
                    rows = samples(archive, f"{label}.jsonl")
                    assert len(rows) == len(queries), label
                    actual = [
                        [
                            r[k]
                            for k in (
                                "query",
                                "class",
                                "limit",
                                "count",
                                "hits",
                                "plan",
                            )
                        ]
                        for r in rows
                    ]
                    if reference is None:
                        reference = actual
                    assert actual == reference, label
                    for row in rows:
                        assert len(row["samples_ns"]) == repetitions, label
                        value = median(s[0] for s in row["samples_ns"]) / 1000
                        grouped[families[row["query"]]][phase].append(value)
                        grouped["all"][phase].append(value)
                    process[phase] = resource(
                        archive.read(f"{label}.resources").decode(), host
                    )
                for candidate in ("b",) if suite != "multi" else ("b", "c"):
                    for family, timings in grouped.items():
                        totals = {p: sum(values) for p, values in timings.items()}
                        a = (totals["a1"] + totals["a2"]) / 2
                        b = (totals[candidate + "1"] + totals[candidate + "2"]) / 2
                        result.append(
                            {
                                "host": host,
                                "index": index,
                                "limit": limit,
                                "family": family,
                                "candidate": "D"
                                if suite == "selected"
                                else candidate.upper(),
                                "queries": len(timings["a1"]),
                                "baseline_sum_us": a,
                                "candidate_sum_us": b,
                                "speedup": a / b,
                                "paired": [
                                    totals[f"a{i}"] / totals[f"{candidate}{i}"]
                                    for i in (1, 2)
                                ],
                            }
                        )
                    a = [process[f"a{i}"] for i in (1, 2)]
                    b = [process[f"{candidate}{i}"] for i in (1, 2)]
                    resources.append(
                        {
                            "host": host,
                            "index": index,
                            "suite": suite,
                            "limit": limit,
                            "candidate": "D"
                            if suite == "selected"
                            else candidate.upper(),
                            "baseline_cpu_s": statistics.mean(v[0] for v in a),
                            "candidate_cpu_s": statistics.mean(v[0] for v in b),
                            "baseline_rss_mib": statistics.mean(v[1] for v in a),
                            "candidate_rss_mib": statistics.mean(v[1] for v in b),
                        }
                    )
    return result, resources


def main():
    data = b"".join(
        p.read_bytes() for p in sorted(ROOT.glob("evidence.zip.[0-9][0-9][0-9]"))
    )
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert hashlib.sha256(data).hexdigest() == provenance["evidence_zip_sha256"]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads((ROOT / "evidence-manifest.json").read_text())
        assert set(archive.namelist()) == set(manifest)
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        micro = screen(archive, "screen", "shape")
        replay = screen(archive, "replay", "trace")
        broad, broad_resources = integrated(archive, "integrated")
        multi, multi_resources = integrated(archive, "multi")
        selected, selected_resources = integrated(archive, "selected")
    resources = broad_resources + multi_resources + selected_resources
    result = {
        "screen": micro,
        "replay": replay,
        "broad": broad,
        "multi": multi,
        "selected": selected,
        "resources": resources,
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    output = [
        "# Complete conditional-dispatch comparison",
        "",
        "Speedup = baseline / candidate elapsed time; above 1 is faster. B is the",
        "density gate; C adds inlining and x86 galloping; D is the cleaned generic-build x86 hybrid.",
        "Family times sum each query's median, then average the two phases.",
        "Paired ratios retain forward/reverse-order drift; small ratios are not",
        "automatically statistically significant. Count is an unchanged-path control",
        "for ordinary AND, but phrase counts do exercise the shared kernel.",
        "",
    ]
    for title, rows in (
        ("Synthetic bounded blocks", micro),
        ("Sampled real-block replay", replay),
    ):
        output += table(
            title,
            ["Host", "Shape/trace", "Capacity", "Kernel", "ns/call", "Speedup"],
            (
                [
                    r["host"],
                    r["shape"],
                    r["capacity"] or "recorded",
                    r["kernel"],
                    f"{r['ns']:.2f}",
                    f"{r['speedup']:.3f}×",
                ]
                for r in rows
            ),
        )
    for title, rows in (
        ("Selected generic release, 199 queries", selected),
        ("Broad queries", broad),
        ("Thirty multi-term conjunctions", multi),
    ):
        output += table(
            title,
            [
                "Host/index",
                "Limit",
                "Family",
                "Variant",
                "A sum µs",
                "Candidate sum µs",
                "Speedup",
                "Forward/reverse",
            ],
            (
                [
                    f"{r['host']}/{r['index']}",
                    r["limit"] or "count",
                    r["family"],
                    r["candidate"],
                    f"{r['baseline_sum_us']:.2f}",
                    f"{r['candidate_sum_us']:.2f}",
                    f"{r['speedup']:.3f}×",
                    "/".join(f"{v:.3f}" for v in r["paired"]),
                ]
                for r in rows
            ),
        )
    output += [
        "Process resources include startup, warmups, all queries and output. They",
        "are not per-kernel heap measurements. RSS includes mapped index pages.",
        "Raw caller-thread CPU clocks exclude search-pool workers and are not used",
        "as total query CPU. The process user+system counters below include them.",
        "",
    ]
    output += table(
        "Whole-process resources (mean of paired phases)",
        [
            "Suite",
            "Host/index",
            "Limit",
            "Variant",
            "A CPU s",
            "Candidate CPU s",
            "A RSS MiB",
            "Candidate RSS MiB",
        ],
        (
            [
                r["suite"],
                f"{r['host']}/{r['index']}",
                r["limit"] or "count",
                r["candidate"],
                f"{r['baseline_cpu_s']:.3f}",
                f"{r['candidate_cpu_s']:.3f}",
                f"{r['baseline_rss_mib']:.2f}",
                f"{r['candidate_rss_mib']:.2f}",
            ]
            for r in resources
        ),
    )
    (ROOT / "table.md").write_text("\n".join(output))


if __name__ == "__main__":
    main()
