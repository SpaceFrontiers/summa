"""Export completed benchmark evidence without machine names or private paths."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def parse_profile(report):
    assert "Total Lost Samples: 0" in report
    event = re.search(r"# Samples: (.*?) of event '(.*?)'", report)
    assert event and event[2].startswith("task-clock")
    symbols = []
    for line in report.splitlines():
        match = re.match(r"\s*([\d.]+)%\s+(\S+)\s+(\[.\])\s+(.*)", line)
        if match:
            symbols.append(
                {
                    "self_percent": float(match[1]),
                    "object": match[2],
                    "mode": match[3],
                    # perf appends padded IPC columns even for task-clock.
                    "symbol": re.split(r"\s{2,}", match[4].strip())[0],
                }
            )
    assert symbols
    return event, symbols


def metrics(path):
    summary = read(path)
    raw = read(path.with_suffix(".raw.json"))
    assert read(path.parent / "complete.json") == {"complete": True}
    assert not summary["errors"] and not raw["errors"]
    assert summary["memory"]["error"] is None
    reps = raw["repetitions"]
    assert len(reps) == 3 and all(rep["errors"] == 0 for rep in reps)
    requests = sum(rep["requests"] for rep in reps)
    elapsed = sum(rep["elapsed_s"] for rep in reps)
    qps = [rep["requests"] / rep["elapsed_s"] for rep in reps]
    return {
        "family": summary["family"],
        "operation": summary["operation"],
        "query_count": summary["query_count"],
        "median_qps": statistics.median(qps),
        "min_qps": min(qps),
        "max_qps": max(qps),
        "server_cpu_us_per_request": 1e6
        * sum(r["server_cpu_s"] for r in reps)
        / requests,
        "driver_cpu_us_per_request": 1e6
        * sum(r["client_cpu_s"] for r in reps)
        / requests,
        "mean_server_cores": sum(r["server_cpu_s"] for r in reps) / elapsed,
        "mean_driver_cores": sum(r["client_cpu_s"] for r in reps) / elapsed,
        "peak_rss_mib": summary["memory"]["peak_vmrss_kb"] / 1024,
        "peak_anonymous_rss_mib": summary["memory"]["peak_rssanon_kb"] / 1024,
        "requests": requests,
        "repetitions": len(reps),
        "errors": 0,
    }


def matrix(folder):
    runs = []
    for run in sorted(folder.iterdir()):
        # Profile-artifact transfer at 03:56 UTC overlapped this round.
        if folder.name == "close-gap-inline-matrix" and run.name == "inline-unicode":
            continue
        if not run.is_dir() or not (run / "completed.json").exists():
            continue
        assert read(run / "completed.json") == {"complete": True}
        engine = "luxir" if run.name.startswith("luxir") else "summa"
        # Historical captures keep their original engine directory name.
        # Rename exported labels, never rewrite the captured measurements.
        engine_folder = run / engine
        if engine == "summa" and not engine_folder.exists():
            engine_folder = run / "hermes"
        selected = [row for row in read(run / "agreement.json") if row["include"]]
        query_keys = sorted(row["key"] for row in selected)
        entry = {
            "label": run.name,
            "engine": engine,
            "query_count": len(selected),
            "query_set_sha256": hashlib.sha256(
                "\n".join(query_keys).encode()
            ).hexdigest(),
            "cells": [
                metrics(path) for path in sorted(engine_folder.glob("*-c32.json"))
            ],
        }
        assert entry["cells"], f"No timing cells in {engine_folder}"
        if engine == "summa":
            entry["binary_sha256"] = read(run / "provenance.json")["binary_sha256"]
            audits = [
                json.loads(line)
                for line in (run / "audit.jsonl").read_text().splitlines()
            ]
            assert len(audits) == len(selected)
            assert all(row["optimized"] == row["exhaustive"] for row in audits)
            responses = read(run / "responses.json")
            assert set(responses) == set(query_keys)
            for row in audits:
                response = responses[row["class"] + "\t" + row["query"]]
                assert response["0"]["found"] == row["exhaustive"]
                for limit in [10, 100]:
                    assert [d["id"] for d in response[str(limit)]["docs"]] == [
                        d["id"] for d in row["ranked"][:limit]
                    ]
            entry["audited_queries"] = len(audits)
        runs.append(entry)
    return runs


def coverage(artifacts, out):
    previous = read(out / "tokenizer-counts.json")
    probes = {
        row["class"] + "\t" + row["query"]: row
        for row in map(
            json.loads,
            (artifacts / "close-gap-unicode-probe-ranges/counts.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    for row in previous["queries"]:
        probe = probes[row["family"] + "\t" + row["query"]]
        found = probe["response"].get("found") if probe["status"] == 200 else None
        row["unicode_word"] = found
        row["error"] = None if found is not None else probe["response"]
        row["exact"] = found is not None and found == row["reference"]
        row["within_5_percent"] = found is not None and abs(
            found - row["reference"]
        ) <= 0.05 * max(1, row["reference"])
    previous["summary"] = {
        "successful": sum(r["unicode_word"] is not None for r in previous["queries"]),
        "exact": sum(r["exact"] for r in previous["queries"]),
        "within_5_percent": sum(r["within_5_percent"] for r in previous["queries"]),
    }
    write(out / "coverage.json", previous)


def fst_probe(artifacts, out):
    folder = artifacts / "close-gap-fst-probe"
    if not folder.exists():
        return
    assert read(folder / "completed.json") == {"complete": True}
    rows = [
        json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()
    ]
    resources = (folder / "resources.txt").read_text()
    assert "Exit status: 0" in resources

    def resource(label):
        return int(
            next(
                line.split(":", 1)[1]
                for line in resources.splitlines()
                if label in line
            )
        )

    assert len(rows) == 9
    result = {
        "status": "rejected",
        "build": rows[0],
        "whole_process_peak_rss_kib": resource("Maximum resident set size"),
        "whole_process_major_faults": resource("Major (requiring I/O) page faults"),
        "queries": [],
        "capture_sha256": hashlib.sha256(
            (artifacts / "close-gap-fst-probe.tar.gz").read_bytes()
        ).hexdigest(),
    }
    for row in rows[1:]:
        assert all(len(timing["samples_ns"]) == 35 for timing in row["timings"])
        medians = {
            timing["label"]: statistics.median(timing["samples_ns"])
            for timing in row["timings"]
        }
        assert set(medians) == {"scan1", "fst1", "fst2", "scan2"}
        row["median_ns"] = medians
        row["scan_over_fst_ratio"] = statistics.mean(
            [medians["scan1"], medians["scan2"]]
        ) / statistics.mean([medians["fst1"], medians["fst2"]])
        result["queries"].append(row)
    write(out / "fst-probe.json", result)


def reverse_probe(artifacts, out):
    for name, output, count in [
        ("summa-reverse-probe-20260925", "reverse-probe", 7),
        ("summa-reverse-all-20260925", "reverse-all", 42),
    ]:
        folder = artifacts / name
        if not folder.exists():
            continue
        assert read(folder / "completed.json") == {"complete": True}
        rows = [
            json.loads(line)
            for line in (folder / "results.jsonl").read_text().splitlines()
        ]
        resources = (folder / "resources.txt").read_text()
        assert "Exit status: 0" in resources
        build = rows.pop(0) if "build_terms" in rows[0] else None
        assert len(rows) == count
        provenance = read(folder / "provenance.json")
        assert (
            hashlib.sha256((folder / "probe.rs").read_bytes()).hexdigest()
            == provenance["source_sha256"]
        )
        result = {
            "status": "experimental-dictionary-only",
            "build": build,
            "provenance": provenance,
            "whole_process_peak_rss_kib": int(
                next(
                    line.split(":", 1)[1]
                    for line in resources.splitlines()
                    if "Maximum resident set size" in line
                )
            ),
            "queries": [],
            "capture_sha256": hashlib.sha256(
                (artifacts / (name + ".tar.gz")).read_bytes()
            ).hexdigest(),
        }
        for row in rows:
            assert all(len(timing["samples_ns"]) == 35 for timing in row["timings"])
            medians = {
                timing["label"]: statistics.median(timing["samples_ns"])
                for timing in row["timings"]
            }
            assert set(medians) == {"scan1", "fst1", "fst2", "scan2"}
            prefix, suffix = row["query"].split("*")
            row["strategy"] = (
                "reverse_suffix"
                if len(suffix.encode()) > len(prefix.encode())
                else "ordinary_prefix"
            )
            row["median_ns"] = medians
            row["scan_over_candidate_ratio"] = statistics.mean(
                [medians["scan1"], medians["scan2"]]
            ) / statistics.mean([medians["fst1"], medians["fst2"]])
            result["queries"].append(row)
        write(out / (output + ".json"), result)
        (out / (output + "-source.rs.txt")).write_bytes(
            (folder / "probe.rs").read_bytes()
        )


def followup_probes(artifacts, out):
    """Export only completed probes; do not infer success from partial logs."""

    def lines(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    folder = artifacts / "summa-io-20260925"
    for capture, count in [("results", 96), ("followup", 32)]:
        path = folder / (capture + ".jsonl")
        if not path.exists():
            continue
        rows = lines(path)
        assert rows.pop() == {"complete": True, "short_read_drain": True}
        assert len(rows) == count
        cells = []
        for row in rows:
            samples = sorted(row.pop("samples_ns"))
            assert len(samples) == row["reads"] // row["depth"]
            assert row["verified"] == row["reads"] == 4096
            if row["cold_advised"]:
                assert row["resident_bytes_before"] == 0
            row["io_ms"] = sum(samples) / 1e6
            row["batch_latency_us"] = {
                str(percentile): samples[round((len(samples) - 1) * percentile / 100)]
                / 1000
                for percentile in [50, 95, 99]
            }
            cells.append(row)
        write(
            out / ("io-" + capture + ".json"),
            {
                "scope": "standalone-buffered-io-not-search-throughput",
                "fixture_bytes": 2147483648,
                "machine_ram_gib": 62,
                "cpu_affinity": "one logical Xeon CPU, including pool workers",
                "mmap_advice": "random" if capture == "results" else "normal",
                "raw_sha256": sha(path),
                "cells": cells,
            },
        )
        source = folder / (
            "initial-source.rs" if capture == "results" else "followup-source.rs"
        )
        if not source.exists():
            source = folder / "summa-io-probe/src/main.rs"
        (out / ("io-" + capture + "-source.rs.txt")).write_bytes(source.read_bytes())

    for name, filename, output in [
        ("summa-followup-20260925", "reverse.jsonl", "reverse-batch"),
        ("summa-ranked-20260925", "results.jsonl", "reverse-ranked"),
    ]:
        folder = artifacts / name
        if not (folder / "completed.json").exists():
            continue
        assert read(folder / "completed.json")["complete"]
        assert "Exit status: 0" in (folder / "resources.txt").read_text()
        rows = lines(folder / filename)
        build = rows.pop(0) if "ranked_terms" in rows[0] else None
        assert len(rows) == 42
        source = folder / (
            "reverse_ranked_probe.rs" if build else "reverse_batch_probe.rs"
        )
        assert sha(source) == read(folder / "completed.json")["probe_sha256"]
        for row in rows:
            assert all(len(t["samples_ns"]) == 35 for t in row["timings"])
            medians = {
                t["label"]: statistics.median(t["samples_ns"]) for t in row["timings"]
            }
            assert set(medians) == {
                "scan1",
                "scan2",
                "batch1",
                "batch2",
                "adaptive1",
                "adaptive2",
            }
            row["paired_median_ns"] = {
                label: statistics.mean([medians[label + "1"], medians[label + "2"]])
                for label in ["scan", "batch", "adaptive"]
            }
        write(
            out / (output + ".json"),
            {
                "scope": "experimental-dictionary-only",
                "build": build,
                "queries": rows,
                "source_sha256": sha(source),
                "raw_sha256": sha(folder / filename),
            },
        )
        (out / (output + "-source.rs.txt")).write_bytes(source.read_bytes())

    folder = artifacts / "summa-ranked-20260925"
    if (folder / "lifecycle-final.json").exists():
        lifecycle = read(folder / "lifecycle-final.json")
        assert lifecycle["complete"] and lifecycle["slot_reuse_verified"]
        assert lifecycle["unlinked_file_reads"]
        for backend in ["ordinary", "registered"]:
            assert lifecycle[backend + "_cancelled"] > 0
            assert (
                lifecycle[backend + "_cancelled"] + lifecycle[backend + "_completed"]
                == 256
            )
        source = folder / "main-lifecycle-final.rs"
        lifecycle["source_sha256"] = sha(source)
        assert (
            lifecycle["source_sha256"]
            == read(folder / "lifecycle-source.json")["sha256"]
        )
        lifecycle["submission"] = "IOSQE_ASYNC to exercise actual cancellation"
        lifecycle["scope"] = (
            "kernel CQE ownership and unlinked filename; not production deletion/cancellation"
        )
        write(out / "io-lifecycle.json", lifecycle)

    captures = [
        (
            "packed",
            "arm64",
            "continuation/collector-arm.jsonl",
            "continuation/collector_probe.rs",
        ),
        (
            "packed",
            "x86_64",
            "summa-followup-20260925/collector-x86.jsonl",
            "summa-followup-20260925/collector_probe.rs",
        ),
        (
            "float",
            "arm64",
            "continuation/collector-float/collector-arm.jsonl",
            "continuation/collector-float/collector_probe.rs",
        ),
        (
            "float",
            "x86_64",
            "summa-followup-20260925/collector-float-x86.jsonl",
            "summa-followup-20260925/collector-float.rs",
        ),
        (
            "seeded",
            "arm64",
            "continuation/seed-arm.jsonl",
            "continuation/seed_probe.rs",
        ),
        (
            "seeded",
            "x86_64",
            "summa-ranked-20260925/seed-x86.jsonl",
            "summa-ranked-20260925/seed_probe.rs",
        ),
    ]
    collected = []
    for experiment, architecture, capture, source in captures:
        path = artifacts / capture
        if not path.exists():
            continue
        rows = lines(path)
        completion = rows.pop()
        assert completion["complete"]
        assert len(rows) == (16 if experiment == "seeded" else 36)
        assert all(
            len(row["samples_ns"]) == (100 if experiment == "seeded" else 20)
            for row in rows
        )
        source = artifacts / source
        collected.append(
            {
                "experiment": experiment,
                "architecture": architecture,
                "completion": completion,
                "raw_sha256": sha(path),
                "source_sha256": sha(source),
                "rows": rows,
            }
        )
        target = out / ("collector-" + experiment + "-source.rs.txt")
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            target.write_bytes(source.read_bytes())
    if collected:
        write(out / "collector-followup.json", collected)


def whole_query_probe(artifacts, out):
    folder = artifacts / "summa-whole-query-20260925"
    if not folder.exists():
        return
    assert read(folder / "completed.json") == {"complete": True}
    resources = (folder / "resources.txt").read_text()
    assert "Exit status: 0" in resources
    queries = read(folder / "queries.json")
    rows = [
        json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()
    ]
    assert len(queries) == 55 and len(rows) == 165
    assert {(r["query"], r["k"]) for r in rows} == {
        (q["query"], k) for q in queries for k in [0, 10, 100]
    }
    previous = {
        row["query"]: row["exhaustive"]
        for row in map(
            json.loads,
            (artifacts / "close-gap-deferred-matrix/audit-unicode.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    assert all(row["exact_count"] == previous[row["query"]] for row in rows)
    result_rows = []
    for row in rows:
        phases = row["timings"]
        assert [p["label"] for p in phases] == [
            "scan1",
            "reverse1",
            "reverse2",
            "scan2",
        ]
        assert all(len(p["samples_ns"]) == 20 for p in phases)
        before = statistics.mean(
            statistics.median(p["samples_ns"])
            for p in phases
            if p["label"].startswith("scan")
        )
        after = statistics.mean(
            statistics.median(p["samples_ns"])
            for p in phases
            if p["label"].startswith("reverse")
        )
        result_rows.append(
            {
                **row,
                "scan_median_ns": before,
                "reverse_median_ns": after,
                "ratio": before / after,
            }
        )
    groups = []
    for family in sorted({r["class"] for r in rows}):
        for k in [0, 10, 100]:
            selected = [r for r in result_rows if r["class"] == family and r["k"] == k]
            before = sum(r["scan_median_ns"] for r in selected)
            after = sum(r["reverse_median_ns"] for r in selected)
            groups.append(
                {
                    "class": family,
                    "k": k,
                    "queries": len(selected),
                    "scan_sum_ns": before,
                    "reverse_sum_ns": after,
                    "ratio": before / after,
                }
            )
    source = folder / "source/summa-core/examples/whole_query.rs"
    (out / "whole-query-source.rs.txt").write_bytes(source.read_bytes())
    provenance = read(folder / "provenance.json")
    write(
        out / "whole-query.json",
        {
            "schema_version": 1,
            "core_version": "2.0.0",
            "mode": "private whole-query expansion experiment, canonical executors; no HTTP or hydration",
            "fixture_docs": 10_000_000,
            "segments": 1,
            "warmup_per_phase": 5,
            "retained_samples_per_phase": 20,
            "phases": ["scan1", "reverse1", "reverse2", "scan2"],
            "term_cache_blocks": 1024,
            "term_cache_budget_bytes": 16 * 1024 * 1024,
            "sidecar_bytes": 148197134,
            "peak_process_rss_kib": int(
                next(
                    line.split(":", 1)[1]
                    for line in resources.splitlines()
                    if "Maximum resident set size" in line
                )
            ),
            "source_archive_sha256": provenance["archive_sha256"],
            "instrumented_sstable_sha256": hashlib.sha256(
                (folder / "source/summa-core/src/structures/sstable.rs").read_bytes()
            ).hexdigest(),
            "hook_patch_sha256": hashlib.sha256(
                (out / "whole-query-hook.patch").read_bytes()
            ).hexdigest(),
            "binary_sha256": provenance["binary_sha256"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(
                (folder / "results.jsonl").read_bytes()
            ).hexdigest(),
            "compiler": provenance["compiler"],
            "flags": provenance["flags"],
            "query_set_sha256": hashlib.sha256(
                (folder / "queries.json").read_bytes()
            ).hexdigest(),
            "correctness": {
                "exact_count_and_top1000_queries": 55,
                "ranked_limits": [10, 100, 1000],
                "modes": 2,
                "all_timed_results_checked": True,
                "counts_match_previous_exhaustive_audit": True,
            },
            "limitations": [
                "warm single-caller x86 CPU0",
                "selector calibrated on these queries",
                "peak RSS includes both modes and shared page cache",
                "no fresh Luxir HTTP comparison",
                "not a production dictionary format",
            ],
            "groups": groups,
            "rows": result_rows,
        },
    )


def batch_latency(samples):
    samples = sorted(samples)
    return {
        "median_batch_ms": statistics.median(samples) / 1e6,
        "p95_batch_ms": samples[int(len(samples) * 0.95)] / 1e6,
    }


def hydration_summary(cells):
    return {
        **batch_latency(n for row in cells for n in row["samples_ns"]),
        "mean_sweep_ms": statistics.mean(
            sum(r.get("wave_ns", r["samples_ns"])) / 1e6 for r in cells
        ),
        "mean_hydration_cpu_ms": statistics.mean(
            r["hydration_cpu_seconds"] * 1e3 for r in cells
        ),
        "kernel_read_bytes": [r["kernel_read_bytes"] for r in cells],
        "resident_at_timing_start": [r["resident_at_timing_start"] for r in cells],
    }


def scheduling_probe(artifacts, out):
    import difflib

    baseline = artifacts / "summa-scheduling-20260925"
    for variant in ["scheduling", "scheduling-unordered"]:
        folder = artifacts / f"summa-{variant}-20260925"
        if not (folder / "completed.json").exists():
            continue
        assert read(folder / "completed.json") == {"complete": True}
        provenance = read(folder / "provenance.json")
        assert (
            provenance["source_archive_sha256"]
            == read(artifacts / variant / "local-provenance.json")["archive_sha256"]
        )
        sources = {}
        for name, relative in {
            "driver": "hydration.rs",
            "service": "payload_io.rs",
            "ring": "payload_io/ring.rs",
            "measure": "measure.rs",
        }.items():
            relative = "scripts/experiments/io_uring/src/" + relative
            data = (folder / "source" / relative).read_bytes()
            assert data == (baseline / "source" / relative).read_bytes()
            target = out / f"scheduling-{name}-source.rs.txt"
            target.write_bytes(data)
            sources[name] = hashlib.sha256(data).hexdigest()
        scenarios = []
        for path in sorted(folder.glob("*.jsonl")):
            captured = [json.loads(line) for line in path.read_text().splitlines()]
            complete = captured.pop()
            assert complete["complete"]
            for name in ["pool", "uring", "refill"]:
                stats = complete[name]
                assert stats["submitted"] == stats["completed"]
                assert stats["errors"] == stats["dropped"] == 0
                assert stats["max_active"] <= 8
            lifecycle = [r for r in captured if "lifecycle" in r]
            assert len(lifecycle) == 5 and all(
                r["in_flight_drop_verified"] for r in lifecycle
            )
            rows = [r for r in captured if "method" in r]
            assert len(rows) == 64
            assert all(
                len(r["samples_ns"]) == 32 and len(r["wave_ns"]) == 32 // r["callers"]
                for r in rows
            )
            groups = {}
            for row in rows:
                key = (row["method"], row["cache_budget_bytes"], row["cold_advised"])
                groups.setdefault(key, []).append(row)
            summarized = []
            for (method, cache, cold), cells in groups.items():
                assert len(cells) == 2
                summarized.append(
                    {
                        "method": method,
                        "cache_budget_bytes": cache,
                        "cold_advised": cold,
                        **hydration_summary(cells),
                        "sweeps_ms": [sum(r["wave_ns"]) / 1e6 for r in cells],
                        "refills": sum(
                            r["stats_after"]["refills"] - r["stats_before"]["refills"]
                            for r in cells
                            if r["stats_after"]
                        ),
                    }
                )
            scenarios.append(
                {
                    "label": path.stem,
                    "binary_sha256": (
                        read(baseline / "provenance.json")["binary_sha256"]
                        if path.stem.startswith("abba-before")
                        else provenance["binary_sha256"]
                    ),
                    "directory_scheduler": (
                        "ordered"
                        if variant == "scheduling"
                        or path.stem.startswith("abba-before")
                        else "completion-driven"
                    ),
                    "callers": rows[0]["callers"],
                    "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "lifecycle": lifecycle,
                    "complete": complete,
                    "groups": summarized,
                    "rows": rows,
                }
            )
        assert len(scenarios) == (4 if variant == "scheduling" else 8)
        directory = "summa-core/src/directories/directory.rs"
        core = (folder / "source" / directory).read_bytes()
        if variant == "scheduling-unordered":
            patch = difflib.unified_diff(
                (baseline / "source" / directory).read_text().splitlines(keepends=True),
                core.decode().splitlines(keepends=True),
                fromfile="a/" + directory,
                tofile="b/" + directory,
            )
            (out / "scheduling-core.patch").write_text("".join(patch))
            assert (
                provenance["store_sha256"]
                == read(baseline / "provenance.json")["store_sha256"]
            )
        write(
            out / f"{variant}.json",
            {
                "provenance": provenance,
                "source_sha256": sources,
                "directory_sha256": hashlib.sha256(core).hexdigest(),
                "scenarios": scenarios,
                "limitations": [
                    "small cold-advised fixture fits RAM; no sustained memory pressure",
                    "current-thread query runtime; concurrent callers overlap I/O, not parallel query CPU",
                    "call samples exclude queue time before the first poll; wave time measures throughput",
                    "services share eight slots; filesystem batching can use eight per caller",
                    "CPU0 or workers allowed CPUs0-7; two reversed method orders per process",
                    "cumulative process peak RSS includes oracle and all methods",
                    "poll-scoped grouped handoff is a diagnostic adapter, not a production API",
                    "no registered resources, direct I/O, SQPOLL, or merge interference",
                ],
            },
        )


def hydration_probe(artifacts, out):
    folder = artifacts / "summa-hydration-20260925"
    if not (folder / "completed.json").exists():
        return
    assert read(folder / "completed.json") == {"complete": True}
    provenance = read(folder / "provenance.json")
    local = read(artifacts / "next-stage/local-provenance.json")
    assert provenance["source_archive_sha256"] == local["archive_sha256"]
    sources = {
        "hydration-measure-source.rs.txt": "scripts/experiments/io_uring/src/measure.rs",
        "hydration-source.rs.txt": "scripts/experiments/io_uring/src/hydration.rs",
        "hydration-service-source.rs.txt": "scripts/experiments/io_uring/src/payload_io.rs",
    }
    hashes = {}
    for name, relative in sources.items():
        data = (folder / "source" / relative).read_bytes()
        (out / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    common = {
        "compiler": provenance["compiler"],
        "flags": provenance["flags"],
        "panic": provenance["panic"],
        "source_archive_sha256": provenance["source_archive_sha256"],
        "source_sha256": hashes,
    }
    rows = [
        json.loads(line)
        for line in (folder / "hydration.jsonl").read_text().splitlines()
    ]
    complete = rows.pop()
    assert complete["complete"]
    lifecycle = [row for row in rows if "lifecycle" in row]
    assert len(lifecycle) == 2 and all(
        row["in_flight_drop_verified"] for row in lifecycle
    )
    rows = [row for row in rows if "method" in row]
    assert len(rows) == 40 and all(len(row["samples_ns"]) == 32 for row in rows)
    groups = []
    for cache in [0, 16 * 1024 * 1024]:
        for cold in [False, True]:
            for method in ["mmap", "fs-demand", "fs-batch", "pool", "uring"]:
                cells = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["cache_budget_bytes"] == cache
                    and r["cold_advised"] == cold
                ]
                assert len(cells) == 2
                groups.append(
                    {
                        "method": method,
                        "cache_budget_bytes": cache,
                        "cold_advised": cold,
                        **hydration_summary(cells),
                    }
                )
    write(
        out / "hydration.json",
        {
            **common,
            "complete": complete,
            "lifecycle": lifecycle,
            "groups": groups,
            "rows": rows,
            "raw_sha256": hashlib.sha256(
                (folder / "hydration.jsonl").read_bytes()
            ).hexdigest(),
            "binary_sha256": provenance["binaries"]["payload_hydration"],
            "limitations": [
                "small four-segment synthetic fixture fits RAM",
                "x86 CPU0",
                "two method orders, 32 samples per cell",
                "cold advice targets private store only",
                "RSS is cumulative process high-water mark",
                "no registered buffers, SQPOLL or direct I/O",
                "future-drop ownership check is not kernel AsyncCancel or production deletion",
            ],
        },
    )


def phase_metrics(cells):
    """Summarize repeated phases without mixing unlike per-query protocols."""
    requests = sum(cell["requests"] for cell in cells)
    return {
        "mean_phase_median_qps": statistics.mean(cell["median_qps"] for cell in cells),
        "phase_median_qps": [cell["median_qps"] for cell in cells],
        "min_repetition_qps": min(cell["min_qps"] for cell in cells),
        "max_repetition_qps": max(cell["max_qps"] for cell in cells),
        "server_cpu_us_per_request": sum(
            cell["server_cpu_us_per_request"] * cell["requests"] for cell in cells
        )
        / requests,
        "peak_rss_mib": max(cell["peak_rss_mib"] for cell in cells),
        "peak_anonymous_rss_mib": max(cell["peak_anonymous_rss_mib"] for cell in cells),
    }


def retrieval_probe(artifacts, out):
    """Fresh renamed-source ABBA evidence, using the existing HTTP metrics owner."""
    folder = artifacts / "summa-retrieval-20260925"
    if not (folder / "matrix-completed.json").exists():
        return
    assert read(folder / "matrix-completed.json") == {"complete": True}
    local = read(artifacts / "retrieval/local-provenance.json")
    provenance = read(folder / "provenance.json")
    assert provenance["source_archive_sha256"] == local["archive_sha256"]
    for name, digest in local["files"].items():
        assert (
            hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest()
            == digest
        )
    rows = read(folder / "agreement.json")
    chosen = [row for row in rows if row["include"]]
    keys = {row["key"] for row in chosen}
    audits = {
        row["class"] + "\t" + row["query"]: row
        for row in map(json.loads, (folder / "audit.jsonl").read_text().splitlines())
    }
    assert set(audits) == keys
    assert all(row["optimized"] == row["exhaustive"] for row in audits.values())
    runs = []
    for phase, engine in [
        ("a1", "summa"),
        ("b1", "luxir"),
        ("b2", "luxir"),
        ("a2", "summa"),
    ]:
        directory = folder / phase
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        if engine == "summa":
            responses = read(directory / "responses.json")
            assert set(responses) == keys
            for key, audit in audits.items():
                assert responses[key]["0"]["found"] == audit["exhaustive"]
                for limit in [10, 100]:
                    assert [
                        doc["id"] for doc in responses[key][str(limit)]["docs"]
                    ] == [doc["id"] for doc in audit["ranked"][:limit]]
        runs.append(
            {
                "phase": phase,
                "engine": engine,
                "binary_sha256": read(directory / "server.json")["binary_sha256"],
                "cells": [
                    metrics(path)
                    for path in sorted((directory / engine).glob("*-c32.json"))
                ],
            }
        )
    families = {row["query"]["query_class"] for row in chosen}
    assert all(len(run["cells"]) == 3 * len(families) for run in runs)
    comparisons = []
    for family in sorted(families):
        for operation in ["TOP_10", "TOP_100", "COUNT"]:
            pair = {"family": family, "operation": operation}
            for engine in ["summa", "luxir"]:
                cells = [
                    cell
                    for run in runs
                    if run["engine"] == engine
                    for cell in run["cells"]
                    if cell["family"] == family and cell["operation"] == operation
                ]
                assert len(cells) == 2
                pair[engine] = phase_metrics(cells)
            pair["summa_over_luxir_qps"] = (
                pair["summa"]["mean_phase_median_qps"]
                / pair["luxir"]["mean_phase_median_qps"]
            )
            comparisons.append(pair)
    write(
        out / "retrieval.json",
        {
            "compiler": provenance["compiler"],
            "flags": provenance["flags"],
            "source_archive_sha256": local["archive_sha256"],
            "query_set_sha256": hashlib.sha256(
                "\n".join(sorted(keys)).encode()
            ).hexdigest(),
            "queries": len(chosen),
            "exhaustive_audits": len(audits),
            "http_checks": 2 * 3 * len(audits),
            "runs": runs,
            "comparisons": comparisons,
            "limitations": [
                "Warm single-segment 10M-document shared-input HTTP comparison; no stored document hydration",
                "Cross-engine count differences retained; scores are audited only within Summa",
                "Three 3-second repetitions per phase; no tail-latency claim",
                "No production-default change or BMP throughput claim",
            ],
        },
    )
    write(
        out / "retrieval-coverage.json",
        {
            "total": len(rows),
            "admitted": len(chosen),
            "exact_counts": sum(row["counts_agree"] for row in chosen),
            "queries": rows,
        },
    )
    write(out / "retrieval-source.json", local)
    if (folder / "inventory-after.json").exists():
        before = read(folder / "inventory-before.json")
        after = read(folder / "inventory-after.json")
        assert before == after
        files = before["index"]["files"]
        write(
            out / "retrieval-fixture.json",
            {
                "files": files,
                "bytes": sum(value["bytes"] for value in files.values()),
                "unchanged_after_campaign": True,
                "searchbench": before["searchbench"],
            },
        )
    if (folder / "profiles-completed.json").exists():
        profiles = read(folder / "profiles-completed.json")
        assert profiles["complete"]
        aggregation = read(folder / "profiles-aggregation.json")
        assert aggregation == {
            "sort": ["dso", "symbol"],
            "captures": 2 * len(profiles["families"]),
            "complete": True,
        }
        raw_hashes = read(folder / "profile-raw-sha256.json")
        binaries = {run["engine"]: run["binary_sha256"] for run in runs}
        captured = []
        for family in profiles["families"]:
            for engine in ["summa", "luxir"]:
                directory = folder / f"profile-{family}-{engine}"
                assert read(directory / "completed.json") == {"complete": True}
                assert read(directory / engine / "complete.json") == {"complete": True}
                assert (
                    read(directory / "server.json")["binary_sha256"] == binaries[engine]
                )
                data = directory / "cpu.perf"
                data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
                assert raw_hashes[str(data.relative_to(folder))] == data_hash
                report = (directory / "profile.txt").read_text()
                event, symbols = parse_profile(report)
                paths = list((directory / engine).glob("*-c32.json"))
                assert len(paths) == 3
                for path in paths:
                    summary = read(path)
                    assert not summary["errors"] and summary["memory"]["error"] is None
                    replay = read(path.with_suffix(".raw.json"))
                    assert not replay["errors"]
                    assert len(replay["repetitions"]) == 1
                    assert replay["repetitions"][0]["errors"] == 0
                captured.append(
                    {
                        "family": family,
                        "engine": engine,
                        "symbols": symbols,
                        "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
                        "binary_sha256": read(directory / "server.json")[
                            "binary_sha256"
                        ],
                        "samples": event[1],
                        "event": event[2],
                        "raw_sha256": data_hash,
                    }
                )
        write(
            out / "retrieval-profiles.json",
            {
                "captures": captured,
                "frequency_hz": 199,
                "sort": aggregation["sort"],
                "limitations": [
                    "Each family profile mixes warmup and top-10/top-100/count operations",
                    "The 90-second process captures include idle intervals; scheduling attribution is not active-request-only",
                    "Self percentages are sampled CPU attribution, not wall-time fractions",
                    "Profiled throughput is excluded from the ABBA comparison",
                    "Frame-pointer callgraphs may be incomplete; flat self attribution is primary",
                    "The Luxir release lacks application symbols; addresses do not establish corresponding internal algorithms",
                ],
            },
        )
    if (folder / "diagnostics-completed.json").exists():
        complete = read(folder / "diagnostics-completed.json")
        assert complete["complete"]
        diagnostic_audits = list(
            map(
                json.loads, (folder / "diagnostic-audit.jsonl").read_text().splitlines()
            )
        )
        assert len(diagnostic_audits) == complete["queries"]
        assert all(
            row == audits[row["class"] + "\t" + row["query"]]
            for row in diagnostic_audits
        )
        diagnostics = list(
            map(
                json.loads,
                (folder / "diagnostic-diagnose.jsonl").read_text().splitlines(),
            )
        )
        assert len(diagnostics) == 2 * complete["queries"]
        assert all(row["instrumented"] for row in diagnostics)
        write(
            out / "retrieval-diagnostics.json",
            {
                "provenance": read(folder / "diagnostics-provenance.json"),
                "complete": complete,
                "rows": diagnostics,
                "limitations": [
                    "Instrumented single-CPU native diagnosis; excluded from HTTP throughput",
                    "First three exactly count-matched queries per selected family, plus all four admitted regexes",
                ],
            },
        )


def mapping_memory(source, binary_names=()):
    """Aggregate measured mapping residency without exposing private file paths."""
    groups = {}
    group = None
    for line in source.splitlines():
        header = re.match(r"[0-9a-f]+-[0-9a-f]+\s+\S+\s+\S+\s+\S+\s+(\d+)\s*(.*)", line)
        if header:
            inode, name = int(header[1]), header[2]
            if inode == 0:
                group = (
                    "kernel mappings"
                    if name in ["[vdso]", "[vvar]", "[vvar_vclock]", "[vsyscall]"]
                    else "anonymous"
                )
            elif ".so" in name:
                group = "shared libraries"
            elif any(
                binary in name
                for binary in (
                    "summa-bin",
                    "hydration-bin",
                    "luxir-0.1.0",
                    *binary_names,
                )
            ):
                group = "executable"
            else:
                suffix = Path(name.removesuffix(" (deleted)")).suffix
                group = suffix if suffix else "other files"
            groups.setdefault(group, {})
        elif group is not None:
            value = re.match(
                r"(Size|Rss|Pss|Anonymous|Private_Clean|Private_Dirty|Locked|Swap):\s+(\d+) kB",
                line,
            )
            if value:
                groups[group][value[1]] = groups[group].get(value[1], 0) + int(value[2])
    assert groups and sum(g.get("Rss", 0) for g in groups.values()) > 0
    return groups


def finite_regex_probe(artifacts, out):
    folder = artifacts / "summa-retrieval-20260926"
    if not (folder / "completed.json").exists():
        return
    assert read(folder / "completed.json") == {"complete": True}
    assert read(folder / "finalized.json")["original_index_and_replay_unchanged"]
    inventory = read(folder / "inventory-after.json")
    previous_fixture = read(out / "retrieval-fixture.json")
    assert inventory["index"]["files"] == previous_fixture["files"]
    assert inventory["searchbench"] == previous_fixture["searchbench"]
    manifest = read(artifacts / "retrieval-source.json")
    provenance = read(folder / "provenance.json")
    assert provenance["source_archive_sha256"] == manifest["archive_sha256"]
    for name, digest in manifest["files"].items():
        assert (
            hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest()
            == digest
        )
    rows = read(folder / "agreement.json")
    audits = {
        r["class"] + "\t" + r["query"]: r
        for r in map(
            json.loads, (folder / "baseline-audit.jsonl").read_text().splitlines()
        )
    }
    assert len(audits) == sum(r["include"] for r in rows)
    for candidate in ["regex", "posting"]:
        observed = {
            r["class"] + "\t" + r["query"]: r
            for r in map(
                json.loads,
                (folder / f"{candidate}-audit.jsonl").read_text().splitlines(),
            )
        }
        assert observed == audits
    probes = read(folder / "probes/summa-counts.json")
    assert len(probes) == len(rows)
    probes_by_key = {r["key"]: r for r in probes}
    for row in rows:
        if row["include"]:
            assert (
                probes_by_key[row["key"]]["count"] == row["observed"]["summa"]["count"]
            )
    phases = [
        ("a1", "before", "summa"),
        ("b1", "after", "summa"),
        ("l1", "luxir", "luxir"),
        ("b2", "after", "summa"),
        ("a2", "before", "summa"),
        ("l2", "luxir", "luxir"),
    ]
    runs = []
    memory = []
    for phase, variant, engine in phases:
        directory = folder / phase
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        if engine == "summa":
            responses = read(directory / "responses.json")
            assert set(responses) == set(audits)
            for key, audit in audits.items():
                assert responses[key]["0"]["found"] == audit["exhaustive"]
                for limit in [10, 100]:
                    assert [d["id"] for d in responses[key][str(limit)]["docs"]] == [
                        d["id"] for d in audit["ranked"][:limit]
                    ]
        else:
            validation = read(directory / "validation.json")
            assert (
                validation["queries"] == len(audits) and validation["operations"] == 3
            )
            assert not validation["errors"]
        cells = [metrics(p) for p in sorted((directory / engine).glob("*-c32.json"))]
        assert len(cells) == 15
        binary = read(directory / "server.json")["binary_sha256"]
        if variant == "after":
            assert binary == provenance["binary_sha256"]
        runs.append(
            {
                "phase": phase,
                "variant": variant,
                "binary_sha256": binary,
                "cells": cells,
            }
        )
        memory.append(
            {
                "phase": phase,
                "variant": variant,
                "mapping_kib": mapping_memory((directory / "smaps.txt").read_text()),
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
                comparison[variant] = phase_metrics(cells)
            comparison["after_over_before_qps"] = (
                comparison["after"]["mean_phase_median_qps"]
                / comparison["before"]["mean_phase_median_qps"]
            )
            comparison["after_over_luxir_qps"] = (
                comparison["after"]["mean_phase_median_qps"]
                / comparison["luxir"]["mean_phase_median_qps"]
            )
            comparisons.append(comparison)
    diagnostics = {}
    for label in ["before", "after"]:
        diagnostics[label] = list(
            map(
                json.loads,
                (folder / f"{label}-diagnose.jsonl").read_text().splitlines(),
            )
        )
        assert len(diagnostics[label]) == 36 and all(
            r["instrumented"] for r in diagnostics[label]
        )
    write(
        out / "finite-regex.json",
        {
            "provenance": provenance,
            "queries_audited_per_candidate": len(audits),
            "runtime": read(folder / "runtime-environment.json"),
            "original_index_and_replay_unchanged": True,
            "http_checks": 4 * 3 * len(audits),
            "luxir_structure_and_count_checks": 2 * 3 * len(audits),
            "runs": runs,
            "comparisons": comparisons,
            "probe": probes,
            "diagnostics": diagnostics,
            "limitations": [
                "Five selected families, not a new full-workload comparison",
                "Old/new Summa phases follow ABBA with interleaved Luxir controls",
                "Native instrumented diagnostics are separate from HTTP throughput",
                "Cross-engine ranking equivalence is not asserted",
            ],
        },
    )
    write(out / "finite-regex-source.json", manifest)
    write(
        out / "retrieval-memory.json",
        {
            "snapshots": memory,
            "limitations": [
                "End-of-phase mapping residency, not peak or per-request allocation",
                "Anonymous memory and canonical metadata accounting have different scope",
                "PSS depends on shared pages; no PSS capacity comparison",
            ],
        },
    )
    posting = {
        phase: list(
            map(
                json.loads,
                (folder / f"{phase}-diagnose.jsonl").read_text().splitlines(),
            )
        )
        for phase in ["p1", "q1", "q2", "p2"]
    }
    assert all(
        len(rows) == 12 and all(not r["instrumented"] for r in rows)
        for rows in posting.values()
    )
    kernels = {
        "arm": list(
            map(json.loads, (artifacts / "pair-arm.jsonl").read_text().splitlines())
        ),
        "x86": list(
            map(json.loads, (folder / "pair-x86.jsonl").read_text().splitlines())
        ),
    }
    assert all(len(rows) == 48 for rows in kernels.values())
    write(
        out / "posting-lanes.json",
        {
            "provenance": read(folder / "posting-provenance.json"),
            "audits": len(audits),
            "native_phases": posting,
            "kernel_screens": kernels,
            "decision": "Reject general consume-match replacement; production posting kernel unchanged",
            "limitations": [
                "Kernel screen extracts the existing functions into a standalone harness; it is not an end-to-end workload",
                "Native six-query CPU0 ABBA screen is separate from HTTP throughput",
                "No BMP performance improvement is established",
            ],
        },
    )
    (out / "posting-lanes.patch").write_text(
        (folder / "posting-candidate.patch").read_text()
    )
    (out / "posting-lanes-source.rs.txt").write_text(
        (artifacts / "pair-probe.rs").read_text()
    )


def real_payload_probe(artifacts, out):
    folder = artifacts / "summa-io-real-20260926"
    if not (folder / "completed.json").exists():
        return
    complete = read(folder / "completed.json")
    assert complete["complete"] and complete["fixture_unchanged"]
    assert (
        read(folder / "fixture.json") == read(out / "retrieval-fixture.json")["files"]
    )
    provenance = read(artifacts / "summa-retrieval-20260926/io-provenance.json")
    build = artifacts / "summa-retrieval-20260926"
    large_reads = read(build / "io-large-green-completed.json")
    assert large_reads == {
        "complete": True,
        "service_handoff_variants": 5,
        "large_range_cases": 5,
    }
    assert read(build / "io-large-red-completed.json")["reproduced"]
    lifecycle = list(
        map(json.loads, (build / "io-large-green.log").read_text().splitlines())
    )
    assert len(lifecycle) == 10
    initial = artifacts / "summa-io-real-20260926-attempt1"
    assert (initial / "aborted.json").exists()
    failed_attempt = []
    for cell in read(initial / "cells.json"):
        if cell["exit_code"]:
            error = (initial / cell["label"] / "stderr.log").read_text()
            assert "payload exceeds diagnostic service byte budget" in error
            failed_attempt.append(
                {**cell, "stderr_sha256": hashlib.sha256(error.encode()).hexdigest()}
            )
    assert read(folder / "started.json")["binary_sha256"] == provenance["binary_sha256"]
    assert (
        hashlib.sha256((artifacts / "io-source.tar.gz").read_bytes()).hexdigest()
        == provenance["source_archive_sha256"]
    )
    oracle_bytes = (folder / "oracle.json").read_bytes()
    oracle_sha = hashlib.sha256(oracle_bytes).hexdigest()
    assert oracle_sha == (folder / "oracle-sha256.txt").read_text()
    assert (
        oracle_sha == hashlib.sha256((initial / "oracle.json").read_bytes()).hexdigest()
    )
    oracle = json.loads(oracle_bytes)
    assert len(oracle["traces"]) == 2048 and all(
        len(t["addresses"]) == 32 for t in oracle["traces"]
    )
    sources = read(artifacts / "io-source.json")
    import tarfile

    with tarfile.open(artifacts / "io-source.tar.gz") as archive:
        for name, digest in sources.items():
            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            )
        (out / "real-payload-source.rs.txt").write_bytes(
            archive.extractfile("scripts/experiments/io_uring/src/real.rs").read()
        )
        (out / "real-payload-service-source.rs.txt").write_bytes(
            archive.extractfile("scripts/experiments/io_uring/src/payload_io.rs").read()
        )
    cells = read(folder / "cells.json")
    assert len(cells) == complete["cells"] == 42
    assert sum(c["exit_code"] != 0 for c in cells) == complete["failures"]
    campaigns = [{"label": "initial", **complete}]
    captured = [(folder, cell) for cell in cells]
    pressure = artifacts / "summa-io-real-20260926-pressure"
    if (pressure / "completed.json").exists():
        pressure_complete = read(pressure / "completed.json")
        pressure_cells = read(pressure / "cells.json")
        assert pressure_complete["complete"] and pressure_complete["fixture_unchanged"]
        assert len(pressure_cells) == pressure_complete["cells"] == 6
        assert (
            sum(c["exit_code"] != 0 for c in pressure_cells)
            == pressure_complete["failures"]
        )
        assert (
            read(pressure / "started.json")["binary_sha256"]
            == provenance["binary_sha256"]
        )
        assert (
            hashlib.sha256((pressure / "oracle.json").read_bytes()).hexdigest()
            == oracle_sha
        )
        assert read(pressure / "fixture.json") == read(folder / "fixture.json")
        campaigns.append({"label": "512-MiB follow-up", **pressure_complete})
        captured.extend((pressure, cell) for cell in pressure_cells)
    complete = {
        "complete": True,
        "fixture_unchanged": True,
        "cells": len(captured),
        "failures": sum(c["failures"] for c in campaigns),
    }
    exported = []
    normalized = []
    for captured_folder, cell in captured:
        directory = captured_folder / cell["label"]
        residency = read(captured_folder / (cell["label"] + "-residency.json"))
        payload_resident = sum(
            v["resident_bytes"] for n, v in residency.items() if n.endswith(".store")
        )
        assert payload_resident == 0
        result = {
            **cell,
            "payload_resident_before_open": payload_resident,
            "fixture_resident_before_open": sum(
                v["resident_bytes"] for v in residency.values()
            ),
        }
        control_path = directory / "control.json"
        if control_path.exists():
            control = read(control_path)
            assert (
                int(control["cgroup_before"]["memory.max"])
                == cell["budget_gib"] * 1024**3
            )
            assert int(control["cgroup_before"]["memory.swap.max"]) == 0
            assert control["command"][:3] == ["taskset", "-c", "0-7"]
            assert control["cgroup_before"]["cpuset.cpus.effective"].strip() == "0-7"
            result["cgroup_before"] = control["cgroup_before"]
            result["cgroup_after"] = control["cgroup_after"]
            if cell["exit_code"] == 0:
                for stage in ["cgroup_before", "cgroup_after"]:
                    events = dict(
                        line.split()
                        for line in control[stage]["memory.events"].splitlines()
                    )
                    assert int(events["oom"]) == int(events["oom_kill"]) == 0
            result["peak_sampled_cgroup_bytes"] = max(
                (s["memory_current"] for s in control["samples"]), default=None
            )
        if cell["exit_code"] != 0:
            result["complete"] = False
            result["failure_evidence_sha256"] = hashlib.sha256(
                (captured_folder / (cell["label"] + "-unit.log")).read_bytes()
            ).hexdigest()
            exported.append(result)
            continue
        assert read(directory / "completed.json") == {"complete": True}
        data_path = directory / "result.jsonl"
        rows = list(map(json.loads, data_path.read_text().splitlines()))
        assert len(rows) == 1
        row = rows[0]
        assert (
            row["complete"]
            and row["workload"] == cell["workload"]
            and row["method"] == cell["method"]
        )
        assert (
            row["callers"] == 4
            and row["batch_documents"] == 32
            and row["cache_budget_bytes"] == 0
        )
        requests = 4096 if cell["workload"] == "trace" else 20 * len(oracle["queries"])
        expected_docs = (
            131072
            if cell["workload"] == "trace"
            else 20 * sum(len(q["addresses"]) for q in oracle["queries"])
        )
        assert (
            row["verified_documents"] == expected_docs
            and len(row["samples"]) == requests
        )
        if row["stats_after"] is not None:
            stats = row["stats_after"]
            assert (
                stats["submitted"] == stats["completed"]
                and stats["errors"] == 0
                and stats["max_active"] <= 8
            )
        passes = []
        for number in sorted({s["pass"] for s in row["samples"]}):
            samples = [s["ns"] for s in row["samples"] if s["pass"] == number]
            passes.append(
                {
                    "pass": number,
                    "requests": len(samples),
                    "operation_wave_ms": sum(
                        w["ns"] for w in row["waves"] if w["pass"] == number
                    )
                    / 1e6,
                    **batch_latency(samples),
                }
            )
        result.update(
            {
                key: value
                for key, value in row.items()
                if key not in ["samples", "waves", "memory_before", "memory_after"]
            }
        )
        result["passes"] = passes
        result["raw_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        for stage in ["memory_before", "memory_after"]:
            result[stage] = {
                "segments": row[stage]["segments"],
                "mapping_kib": mapping_memory(row[stage]["smaps"]),
            }
        exported.append(result)
        normalized.append(
            {
                **cell,
                "samples_ns": [s["ns"] for s in row["samples"]],
                "wave_ns": [s["ns"] for s in row["waves"]],
                "hydration_cpu_seconds": row["operation_cpu_seconds"],
                "kernel_read_bytes": row["kernel_read_bytes"],
                "resident_at_timing_start": None,
            }
        )
    comparisons = []
    for workload, budgets in [
        (
            "trace",
            sorted({c["budget_gib"] for c in normalized if c["workload"] == "trace"}),
        ),
        ("query", [1, 8]),
        ("retrieval", [1, 8]),
    ]:
        for budget in budgets:
            methods = {}
            for method in ["mmap", "pool", "uring-refill-batch"]:
                matching = [
                    c
                    for c in normalized
                    if c["workload"] == workload
                    and c["budget_gib"] == budget
                    and c["method"] == method
                ]
                if len(matching) == 2:
                    summary = hydration_summary(matching)
                    summary["mean_operation_cpu_ms"] = summary.pop(
                        "mean_hydration_cpu_ms"
                    )
                    summary.pop("resident_at_timing_start")
                    summary["order_sweep_ms"] = [
                        sum(c["wave_ns"]) / 1e6 for c in matching
                    ]
                    methods[method] = summary
            comparisons.append(
                {"workload": workload, "budget_gib": budget, "methods": methods}
            )
    write(
        out / "real-payload.json",
        {
            "complete": complete,
            "campaigns": campaigns,
            "runtime": read(build / "runtime-environment.json"),
            "large_read_regression": large_reads,
            "lifecycle": lifecycle,
            "excluded_attempt": {
                "reason": "Explicit-read startup exceeded single-read admission; entire attempt excluded and matrix restarted after bounded chunking repair",
                "failed_cells": failed_attempt,
            },
            "provenance": provenance,
            "source_hashes": sources,
            "oracle_sha256": oracle_sha,
            "fixture": read(folder / "fixture.json"),
            "oracle_documents": oracle["documents"],
            "query_terms": oracle["terms"],
            "cells": exported,
            "comparisons": comparisons,
            "limitations": [
                "Diagnostic backend only; no production io_uring default",
                "Payload pages are cold before index open; header reads occur before timing",
                "Address replay and the much smaller repeated query working set are separate workloads",
                "Verification is outside operation timing; fault/read counters cover the whole phase",
                "Mmap uses its existing synchronous hydration path; only pool and ring share equal asynchronous service budgets",
                "Reported latency is per batch/query, not per document or separated queue/service latency",
                "No concurrent merge workload or BMP speedup claim",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    for name in [
        "matrix",
        "regex-matrix",
        "union-matrix",
        "final-matrix",
        "norm-matrix",
        "norm-matrix-valid",
        "workers-matrix",
        "inline-matrix",
        "deferred-matrix",
        "complete-matrix",
    ]:
        folder = args.artifacts / ("close-gap-" + name)
        if folder.exists():
            write(args.out / (name + ".json"), matrix(folder))
    if (args.artifacts / "close-gap-unicode-probe-ranges").exists():
        coverage(args.artifacts, args.out)
    fst_probe(args.artifacts, args.out)
    reverse_probe(args.artifacts, args.out)
    followup_probes(args.artifacts, args.out)
    whole_query_probe(args.artifacts, args.out)
    hydration_probe(args.artifacts, args.out)
    scheduling_probe(args.artifacts, args.out)
    retrieval_probe(args.artifacts, args.out)
    finite_regex_probe(args.artifacts, args.out)
    real_payload_probe(args.artifacts, args.out)


if __name__ == "__main__":
    main()
