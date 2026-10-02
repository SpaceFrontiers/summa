"""Validate the normal diagnostic binary's ARM A B B A ranked control."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(folder):
    assert (folder / "done").read_text() == "done"
    assert not (folder / "failed.txt").exists()
    result = {
        "timing_status": "inconclusive: shared desktop and large phase drift in both attempts",
        "binaries": json.loads((folder / "binaries.json").read_text()),
        "indexes": {},
    }
    for index in ["bitmap", "rgb"]:
        phases = {}
        expected = None
        for phase in ["a1", "b1", "b2", "a2"]:
            label = index + "-" + phase
            audit = read_rows(folder / (label + "-audit.jsonl"))
            if expected is None:
                expected = audit
            assert audit == expected and len(audit) == 59
            assert all(r["exhaustive"] == r["optimized"] for r in audit)
            rows = read_rows(folder / (label + "-diagnose.jsonl"))
            assert len(rows) == 118
            assert {(r["class"], r["query"], r["limit"]) for r in rows} == {
                (r["class"], r["query"], limit) for r in audit for limit in [10, 100]
            }
            assert all(
                r["samples"] == 100 and r["warmup"] == 20 and not r["instrumented"]
                for r in rows
            )
            resources = (
                (folder / (label + "-diagnose.resources")).read_text().splitlines()
            )
            peak = int(
                next(
                    line.split()[0]
                    for line in resources
                    if "maximum resident set size" in line
                )
            )
            phases[phase] = {
                "rows": rows,
                "peak_rss_bytes": peak,
                "audit_sha256": hashlib.sha256(
                    (folder / (label + "-audit.jsonl")).read_bytes()
                ).hexdigest(),
            }
        comparisons = []
        for family in sorted({r["class"] for r in expected}):
            for limit in [10, 100]:
                row = {"family": family, "limit": limit}
                for stage in ["parse", "search", "project", "serialize"]:
                    sums = {
                        phase: sum(
                            r["median_us"][stage]
                            for r in data["rows"]
                            if r["class"] == family and r["limit"] == limit
                        )
                        for phase, data in phases.items()
                    }
                    before = statistics.mean(sums[phase] for phase in ["a1", "a2"])
                    after = statistics.mean(sums[phase] for phase in ["b1", "b2"])
                    row[stage] = {
                        "before_sum_medians_us": before,
                        "reviewed_sum_medians_us": after,
                        "speedup": before / after,
                        "phase_sum_medians_us": sums,
                    }
                comparisons.append(row)
        result["indexes"][index] = {"phases": phases, "comparisons": comparisons}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "arm-results.json"
    )
    args = parser.parse_args()
    result = summarize(args.artifacts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for index, values in result["indexes"].items():
        for row in values["comparisons"]:
            print(
                index, row["family"], row["limit"], f"{row['search']['speedup']:.3f}x"
            )
