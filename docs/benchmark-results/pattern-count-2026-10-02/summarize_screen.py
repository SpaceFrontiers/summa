"""Validate paired native probes; timings are sums across each family's queries."""

import argparse
import json
import math
import re
from pathlib import Path


def summarize(root, indexes, iterations):
    queries = [
        json.loads(line) for line in (root / "patterns.jsonl").read_text().splitlines()
    ]
    result = {
        "iterations": iterations,
        "warmup": 5,
        "queries": len(queries),
        "indexes": {},
        "peak_rss_mib": {},
    }
    for index in indexes:
        result["indexes"][index] = {}
        for limit in [0, 100]:
            phases = {}
            for phase in ["a1", "b1", "b2", "a2"]:
                rows = [
                    line.split("\t")
                    for line in (root / f"{index}-{phase}-{limit}.tsv")
                    .read_text()
                    .splitlines()
                ]
                assert len(rows) == len(queries), (index, phase, limit, len(rows))
                for query, row in zip(queries, rows, strict=True):
                    assert len(row) == 4 and json.loads(row[0]) == query["query"]
                    assert int(row[1]) >= 0
                    assert all(
                        math.isfinite(float(v)) and float(v) > 0 for v in row[2:]
                    )
                resources = (root / f"{index}-{phase}-{limit}.resources").read_text()
                linux = re.search(
                    r"Maximum resident set size \(kbytes\): (\d+)", resources
                )
                if linux:
                    assert "Exit status: 0" in resources
                    rss_mib = int(linux[1]) / 1024
                else:
                    mac = re.search(r"(\d+)  maximum resident set size", resources)
                    assert mac
                    rss_mib = int(mac[1]) / 1048576
                result["peak_rss_mib"][f"{index}-{phase}-{limit}"] = rss_mib
                phases[phase] = rows
            assert all(
                [row[:2] for row in rows] == [row[:2] for row in phases["a1"]]
                for rows in phases.values()
            ), (index, limit, "result mismatch")
            families = {}
            for family in sorted({query["family"] for query in queries}):
                ids = [
                    i for i, query in enumerate(queries) if query["family"] == family
                ]
                sums = {
                    phase: {
                        "median_us": sum(float(rows[i][2]) for i in ids),
                        "cpu_us": sum(float(rows[i][3]) for i in ids),
                    }
                    for phase, rows in phases.items()
                }
                before = {
                    key: (sums["a1"][key] + sums["a2"][key]) / 2
                    for key in ["median_us", "cpu_us"]
                }
                after = {
                    key: (sums["b1"][key] + sums["b2"][key]) / 2
                    for key in ["median_us", "cpu_us"]
                }
                families[family] = {
                    "queries": len(ids),
                    "before": before,
                    "after": after,
                    "speedup": {key: before[key] / after[key] for key in before},
                    "phases": sums,
                }
            result["indexes"][index][str(limit)] = families
    if (root / "inventory-before.json").exists():
        assert json.loads((root / "inventory-before.json").read_text()) == json.loads(
            (root / "inventory-after.json").read_text()
        )
        result["inventory_unchanged"] = True
        result["binary_sha256"] = json.loads((root / "binary-hashes.json").read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--indexes", nargs="+", default=["default", "rgb"])
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    result = summarize(args.root, args.indexes, args.iterations)
    (args.root / "screen-results.json").write_text(json.dumps(result, indent=2) + "\n")
    for index, limits in result["indexes"].items():
        for limit, families in limits.items():
            print(
                index,
                limit,
                {
                    family: round(values["speedup"]["median_us"], 3)
                    for family, values in families.items()
                },
            )
