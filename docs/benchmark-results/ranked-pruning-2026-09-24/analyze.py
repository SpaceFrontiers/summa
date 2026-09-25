"""Recompute measurements and correctness gates from the retained results.zip."""

import json
import statistics
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent


def analyze(archive):
    source = archive
    if not archive.exists():
        parts = sorted(archive.parent.glob(archive.name + ".[0-9][0-9][0-9]"))
        if not parts:
            raise FileNotFoundError(archive)
        source = BytesIO(b"".join(part.read_bytes() for part in parts))
    with ZipFile(source) as bundle:

        def read(name):
            return bundle.read(name)

        def obj(name):
            return json.loads(read(name))

        def lines(name):
            return [json.loads(line) for line in read(name).splitlines()]

        def measure(paths):
            summaries = [obj(path) for path in paths]
            raw = [obj(path.removesuffix(".json") + ".raw.json") for path in paths]
            for summary, detail in zip(summaries, raw, strict=True):
                assert not summary["errors"] and not summary["memory"]["error"]
                assert not detail["errors"]
                assert len(summary["repetitions"]) == len(detail["repetitions"]) == 3
                assert all(r["errors"] == 0 for r in summary["repetitions"])
                assert all(
                    r["errors"] == 0 and r["requests"] > 0
                    for r in detail["repetitions"]
                )
                assert summary["clients"] == detail["config"]["concurrency"] == 32
            reps = [r for s in summaries for r in s["repetitions"]]
            cpu = [r for s in raw for r in s["repetitions"]]
            qps = [r["qps"] for r in reps]
            return {
                "qps": statistics.median(qps),
                "qps_range": [min(qps), max(qps)],
                "session_medians": [s["median_qps"] for s in summaries],
                "repetitions": len(reps),
                "cpu_us_request": statistics.median(
                    r["server_cpu_s"] * 1e6 / r["requests"] for r in cpu
                ),
                "cpu_cores": statistics.median(
                    r["server_cpu_s"] / r["elapsed_s"] for r in cpu
                ),
                "peak_rss_mib": max(s["memory"]["peak_vmrss_kb"] for s in summaries)
                / 1024,
                "peak_anonymous_mib": max(
                    s["memory"]["peak_rssanon_kb"] for s in summaries
                )
                / 1024,
                "errors": 0,
            }

        result = {
            "posture": obj("cloud/pruning/final/posture.json"),
            "cloud": {},
            "luxir": {},
        }
        assert read("cloud/pruning/final/status.txt") == b"complete"
        for layout in ["plain", "rgb"]:
            expected = obj(f"cloud/pruning/final/1-before-{layout}/responses.json")
            assert len(expected) == 45
            audit = read(f"cloud/pruning/final/1-before-{layout}/audit.jsonl")
            for mode in ["before", "hybrid"]:
                assert (
                    read(f"cloud/pruning/final/1-{mode}-{layout}/audit.jsonl") == audit
                )
                for run in [1, 2]:
                    assert (
                        obj(f"cloud/pruning/final/{run}-{mode}-{layout}/responses.json")
                        == expected
                    )
            cells = {}
            for family in ["and_high_low", "low_phrase", "med_phrase"]:
                for operation in ["TOP_10", "TOP_100", "COUNT"]:
                    key = f"{family}-{operation}"
                    row = {
                        mode: measure(
                            [
                                f"cloud/pruning/final/{run}-{mode}-{layout}/summa/{key}-c32.json"
                                for run in [1, 2]
                            ]
                        )
                        for mode in ["before", "hybrid"]
                    }
                    row["qps_change_pct"] = 100 * (
                        row["hybrid"]["qps"] / row["before"]["qps"] - 1
                    )
                    row["cpu_change_pct"] = 100 * (
                        row["hybrid"]["cpu_us_request"]
                        / row["before"]["cpu_us_request"]
                        - 1
                    )
                    cells[key] = row
                    if layout == "plain":
                        result["luxir"][key] = measure(
                            [f"cloud/pruning/luxir-fresh/luxir/{key}-c32.json"]
                        )
            result["cloud"][layout] = cells
        result["stages"] = {}
        result["work"] = {}
        for layout in ["plain", "rgb"]:
            for mode in ["before", "hybrid"]:
                key = f"{mode}-{layout}"
                result["stages"][key] = lines(
                    f"cloud/pruning/final/1-{key}/stages.jsonl"
                )
                counters = lines(f"cloud/pruning/final/{key}-diagnostic.jsonl")
                assert len(counters) == 30 and all(
                    r["instrumented"] and r["work"] for r in counters
                )
                result["work"][key] = counters
        arm = {
            mode: [lines(f"arm/arm-final-{mode}-{run}.jsonl") for run in [1, 2]]
            for mode in ["before", "after"]
        }
        rows = []
        for i, first in enumerate(arm["before"][0]):
            signature = first["signature"]
            assert all(
                r[i]["signature"] == signature for mode in arm.values() for r in mode
            )
            row = {key: first[key] for key in ["rgb", "query", "limit"]}
            for mode in ["before", "after"]:
                row[mode + "_run_medians_us"] = [r[i]["median_us"] for r in arm[mode]]
                row[mode + "_us"] = statistics.mean(row[mode + "_run_medians_us"])
            row["change_pct"] = 100 * (row["after_us"] / row["before_us"] - 1)
            rows.append(row)
        result["arm"] = rows
        result["correctness"] = {
            "audits_byte_identical_per_layout": True,
            "http_responses_byte_identical": 8 * 45,
            "arm_score_bit_signatures_identical": True,
        }
        return result


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.zip"
    output = analyze(source)
    (HERE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    for layout, cells in output["cloud"].items():
        print(layout)
        for name, row in cells.items():
            print(
                f"  {name}: {row['before']['qps']:.1f} -> {row['hybrid']['qps']:.1f} QPS ({row['qps_change_pct']:+.1f}%), CPU {row['cpu_change_pct']:+.1f}%"
            )
