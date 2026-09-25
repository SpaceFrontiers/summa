"""Validate and summarize the paired plain-index pruning campaign."""

import argparse
import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "skipping-2026-09-24/summarize.py"
spec = importlib.util.spec_from_file_location("previous_summary", SOURCE)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def summarize(source, output):
    rows, audits = [], []
    for layout in ["plain", "rgb"]:
        baseline = summary.read(source / f"1-current-{layout}" / "responses.json")
        oracle = (
            (source / f"1-current-{layout}" / "audit.jsonl").read_text().splitlines()
        )
        assert len(oracle) == 334
        for variant in ["current", "plain-bounds"]:
            first = source / f"1-{variant}-{layout}"
            actual = (first / "audit.jsonl").read_text().splitlines()
            assert list(map(json.loads, actual)) == list(map(json.loads, oracle))
            assert all(
                row["exhaustive"] == row["optimized"] for row in map(json.loads, actual)
            )
            for round_id in [1, 2]:
                folder = source / f"{round_id}-{variant}-{layout}"
                responses = summary.read(folder / "responses.json")
                assert responses == baseline
                assert len(responses) == 334
                for row in map(json.loads, actual):
                    response = responses[row["class"] + "\t" + row["query"]]
                    assert response["0"]["found"] == row["exhaustive"]
                    for limit in [10, 100]:
                        assert [doc["id"] for doc in response[str(limit)]["docs"]] == [
                            doc["id"] for doc in row["ranked"][:limit]
                        ]
            audits.append(
                {
                    "layout": layout,
                    "variant": variant,
                    "exhaustive_queries": 334,
                    "http_checks": 334 * 3 * 2,
                }
            )
            paths = summary.measured_paths(first / "summa")
            assert len(paths) == 21
            for path in paths:
                cell = summary.read(path)
                value = summary.metrics(
                    [
                        source / f"{r}-{variant}-{layout}" / "summa" / path.name
                        for r in [1, 2]
                    ]
                )
                before = summary.metrics(
                    [
                        source / f"{r}-current-{layout}" / "summa" / path.name
                        for r in [1, 2]
                    ]
                )
                rows.append(
                    {
                        "layout": layout,
                        "variant": variant,
                        "family": cell["family"],
                        "operation": cell["operation"],
                        **value,
                        "qps_ratio": value["median_qps"] / before["median_qps"],
                        "cpu_ratio": value["server_cpu_ms_per_request"]
                        / before["server_cpu_ms_per_request"],
                        "round_qps_ratios": [
                            a / b
                            for a, b in zip(
                                value["round_median_qps"],
                                before["round_median_qps"],
                                strict=True,
                            )
                        ],
                    }
                )
    output.mkdir(exist_ok=True)
    summary.write(output / "results.json", rows)
    summary.write(output / "audits.json", audits)
    write_tables(rows, output)
    return rows


def write_tables(rows, output):
    lines = [
        "# Paired throughput results",
        "",
        "QPS, higher is better. Median of six repetitions across two rounds;",
        "CPU change is server CPU time per completed request. See",
        "[protocol and limitations](README.md) and [all samples](results.json).",
        "",
    ]
    for layout in ["plain", "rgb"]:
        lines += [
            f"## {layout.upper()}",
            "",
            "| Family | Operation | Before | After | QPS change | CPU/request change |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            if row["layout"] != layout or row["variant"] != "plain-bounds":
                continue
            before = row["median_qps"] / row["qps_ratio"]
            lines.append(
                f"| {row['family']} | {row['operation']} | {before:,.0f} | "
                f"{row['median_qps']:,.0f} | {(row['qps_ratio'] - 1) * 100:+.1f}% | "
                f"{(row['cpu_ratio'] - 1) * 100:+.1f}% |"
            )
        lines.append("")
    (output / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    summarize(args.raw, args.out)
