"""Reproduce the method tables from the retained raw screening samples."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def intersections():
    rows = []
    for arch in ("arm", "arm-repeat", "x86"):
        groups = defaultdict(list)
        with (ROOT / f"{arch}.csv").open() as stream:
            for r in csv.DictReader(stream):
                groups[r["shape"], int(r["capacity"]), r["kernel"]].append(
                    int(r["elapsed_ns"]) / int(r["calls"])
                )
        for (shape, cap, method), samples in groups.items():
            assert len(samples) == 9
            control = statistics.median(groups[shape, cap, "production"])
            rows.append(
                {
                    "host": arch,
                    "shape": shape,
                    "capacity": cap,
                    "method": method,
                    "median_ns": statistics.median(samples),
                    "min_ns": min(samples),
                    "max_ns": max(samples),
                    "speedup": control / statistics.median(samples),
                }
            )
    return rows


def unions():
    rows = []
    for arch in ("arm", "x86"):
        groups = defaultdict(list)
        scratch = {}
        with (ROOT / f"{arch}-union.csv").open() as stream:
            for r in csv.DictReader(stream):
                key = r["shape"], r["method"]
                groups[key].append(int(r["elapsed_ns"]) / int(r["iterations"]) / 1000)
                scratch[key] = int(r["scratch_bound_bytes"])
        for (shape, method), samples in groups.items():
            assert len(samples) == 7
            rows.append(
                {
                    "host": arch,
                    "shape": shape,
                    "method": method,
                    "median_us": statistics.median(samples),
                    "min_us": min(samples),
                    "max_us": max(samples),
                    "scratch_payload_bound_bytes": scratch[shape, method],
                }
            )
    return rows


def main():
    inter, union = intersections(), unions()
    (ROOT / "results.json").write_text(
        json.dumps({"intersections": inter, "unions": union}, indent=2) + "\n"
    )
    table = [
        "# Complete method screen tables",
        "",
        "Intersection speedup = production median / candidate median. Above 1 is faster.",
        "",
        "| Host | Shape | Output capacity | Production ns/call | Scalar | Gallop | 4×4 SIMD | NEON prefix8 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped = defaultdict(dict)
    for r in inter:
        grouped[r["host"], r["shape"], r["capacity"]][r["method"]] = r
    for (host, shape, cap), v in grouped.items():
        ratios = [
            f"{v[m]['speedup']:.3f}×" if m in v else "—"
            for m in ("scalar", "gallop", "cross4", "prefix8")
        ]
        table.append(
            f"| {host} | {shape} | {cap} | {v['production']['median_ns']:.2f} | "
            + " | ".join(ratios)
            + " |"
        )
    table += [
        "",
        "Union times include scratch allocation and exact cardinality, but exclude decoding.",
        "",
        "| Host | Shape | Sort µs | Bitmap µs | Heap µs | Window µs | Sort/bitmap/heap/window scratch bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    grouped = defaultdict(dict)
    for r in union:
        grouped[r["host"], r["shape"]][r["method"]] = r
    for (host, shape), v in grouped.items():
        methods = ("sort", "bitmap", "heap", "windows")
        times = [f"{v[m]['median_us']:.2f}" for m in methods]
        memory = "/".join(str(v[m]["scratch_payload_bound_bytes"]) for m in methods)
        table.append(f"| {host} | {shape} | " + " | ".join(times) + f" | {memory} |")
    (ROOT / "table.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
