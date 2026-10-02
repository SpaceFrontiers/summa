"""Reuse the fixed-arrival validator and compare the retained immutable fixture."""

import argparse
import json
import runpy
from pathlib import Path

HERE = Path(__file__).parent
SHARED = runpy.run_path(str(HERE.parent / "sustained-io-2026-09-26/summarize.py"))


def summarize(folder):
    result = SHARED["export"](folder)
    previous = json.loads(
        (HERE.parent / "sustained-io-2026-09-26/results.json").read_text()
    )
    assert result["fixture"] == previous["fixture"]
    result["provenance"] = json.loads((folder / "provenance.json").read_text())
    source = json.loads((HERE / "source.json").read_text())
    assert result["provenance"]["binary_sha256"] == source["io_binary_sha256"]
    assert result["provenance"]["flags"] == source["flags"]
    assert (
        result["provenance"]["compiler"].replace("\\n", "\n").splitlines()[0]
        == source["compiler"]
    )
    result["errors"] = 0
    (HERE / "io-results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "| Workload | Offered/s | Backend | Completed/s | CPU/query ms | First-second rejected | Later rejected | Later p95 ms | Later p99 ms | Mean cgroup peak MiB |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summaries"]:
        steady = row["steady"]
        lines.append(
            f"| {row['field']} | {row['rate']:,} | {row['method']} | {row['completed_per_second']:,.1f} | {row['cpu_us_per_completed'] / 1000:.3f} | {row['initial']['reject_fraction']:.2%} | {steady['reject_fraction']:.2%} | {steady['latency_p95_ms']:.2f} | {steady['latency_p99_ms']:.2f} | {row['peak_cgroup_bytes'] / 2**20:.1f} |"
        )
    (HERE / "io-table.md").write_text("\n".join(lines) + "\n")
    print(
        result["offered"],
        "offered,",
        result["verified_queries"],
        "verified,",
        result["decoded_documents"],
        "documents",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    summarize(args.artifacts)
