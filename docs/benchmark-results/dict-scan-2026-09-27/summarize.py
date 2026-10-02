"""Validate and summarize the paired term-dictionary campaigns.

`cache`: one binary; `default` phases use the default `IndexConfig` term cache
and `cache` phases set the experiment-only `SEARCHBENCH_TERM_CACHE_*` override
(see term-cache.patch). `tight`: the same override crossed with the baseline
binary and the tight-scan candidate, as `<binary>-<cache>` variants.
`shared`: the previous binary with old defaults (`old`), the process-wide
dictionary cache build pinned to per-segment caches (`per-segment`), and the
same build with its new defaults (`new`), see shared-policy.patch.
`tight2`: the shared-cache default build versus the same tree plus the tight
key-only scan, with CPU profiles of both under the broad-wildcard family.
`matcher`: that tight build versus the same tree plus word-sized literal checks
in the single-star matcher, with profiles.
`filter`: that matcher build versus the same tree plus dictionary-block suffix
filters (`docs/dictionary-suffix-filters.md`), with profiles.
`final`: all 19 Searchbench families with stock defaults: the final build
(`new`), the September 26 baseline binary (`old`) and Luxir.
`dispatch`: one binary serving with `blocking` versus `pool` dispatch (the
`serve` argument), all 19 families, Luxir, and per-family CPU profiles.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

COMMON = runpy.run_path(
    str(Path(__file__).parent.parent / "closing-gap-2026-09-24/summarize.py")
)
read, write = COMMON["read"], COMMON["write"]
OVERRIDE = {
    "SEARCHBENCH_TERM_CACHE_BLOCKS": "16384",
    "SEARCHBENCH_TERM_CACHE_BYTES": str(256 << 20),
}
PROCESS_KEY = "SEARCHBENCH_TERM_CACHE_PROCESS_BYTES"
ENV_KEYS = [*OVERRIDE, PROCESS_KEY]
CACHE_LOG = "term cache override: 16384 blocks"
CAMPAIGNS = {
    "cache": {
        "phases": [
            ("a1", "default"),
            ("c1", "cache"),
            ("l1", "luxir"),
            ("c2", "cache"),
            ("a2", "default"),
            ("l2", "luxir"),
        ],
        "binaries": {"default": "cache", "cache": "cache"},
        "environment": {"default": {}, "cache": OVERRIDE},
        "logs": {"default": [], "cache": [CACHE_LOG]},
        "audits": ["default", "cache"],
        "baseline": "default",
        "output": "results.json",
    },
    "tight": {
        "phases": [
            ("a1", "base-default"),
            ("b1", "tight-default"),
            ("c1", "base-cache"),
            ("d1", "tight-cache"),
            ("l1", "luxir"),
            ("d2", "tight-cache"),
            ("c2", "base-cache"),
            ("b2", "tight-default"),
            ("a2", "base-default"),
            ("l2", "luxir"),
        ],
        "binaries": {
            "base-default": "cache",
            "base-cache": "cache",
            "tight-default": "tight",
            "tight-cache": "tight",
        },
        "environment": {
            "base-default": {},
            "base-cache": OVERRIDE,
            "tight-default": {},
            "tight-cache": OVERRIDE,
        },
        "logs": {
            "base-default": [],
            "base-cache": [CACHE_LOG],
            "tight-default": [],
            "tight-cache": [CACHE_LOG],
        },
        "audits": ["tight-default", "tight-cache"],
        "baseline": "base-default",
        "output": "tight-results.json",
    },
    "shared": {
        "phases": [
            ("o1", "old"),
            ("p1", "per-segment"),
            ("n1", "new"),
            ("l1", "luxir"),
            ("n2", "new"),
            ("p2", "per-segment"),
            ("o2", "old"),
            ("l2", "luxir"),
        ],
        "binaries": {"old": "cache", "per-segment": "shared", "new": "shared"},
        "environment": {"old": {}, "per-segment": {PROCESS_KEY: "0"}, "new": {}},
        "logs": {
            "old": [],
            "per-segment": ["term cache policy: process_bytes=0 "],
            "new": [f"term cache policy: process_bytes={256 << 20} "],
        },
        "audits": ["per-segment", "new"],
        "baseline": "old",
        "output": "shared-results.json",
    },
    "tight2": {
        "phases": [
            ("s1", "shared"),
            ("t1", "tight"),
            ("l1", "luxir"),
            ("t2", "tight"),
            ("s2", "shared"),
            ("l2", "luxir"),
        ],
        "binaries": {"shared": "shared", "tight": "tight2"},
        "environment": {"shared": {}, "tight": {}},
        "logs": {
            "shared": [f"term cache policy: process_bytes={256 << 20} "],
            "tight": [f"term cache policy: process_bytes={256 << 20} "],
        },
        "audits": ["tight"],
        "baseline": "shared",
        "output": "tight2-results.json",
        "profiles": {
            "shared": ("profile-shared", "shared"),
            "tight": ("profile-tight", "tight2"),
        },
    },
    "matcher": {
        "phases": [
            ("t1", "tight"),
            ("m1", "matcher"),
            ("l1", "luxir"),
            ("m2", "matcher"),
            ("t2", "tight"),
            ("l2", "luxir"),
        ],
        "binaries": {"tight": "tight2", "matcher": "matcher"},
        "environment": {"tight": {}, "matcher": {}},
        "logs": {
            "tight": [f"term cache policy: process_bytes={256 << 20} "],
            "matcher": [f"term cache policy: process_bytes={256 << 20} "],
        },
        "audits": ["matcher"],
        "baseline": "tight",
        "output": "matcher-results.json",
        "profiles": {
            "tight": ("profile-tight", "tight2"),
            "matcher": ("profile-matcher", "matcher"),
        },
    },
    "filter": {
        "phases": [
            ("m1", "matcher"),
            ("f1", "filter"),
            ("l1", "luxir"),
            ("f2", "filter"),
            ("m2", "matcher"),
            ("l2", "luxir"),
        ],
        "binaries": {"matcher": "matcher", "filter": "filter"},
        "environment": {"matcher": {}, "filter": {}},
        "logs": {
            "matcher": [f"term cache policy: process_bytes={256 << 20} "],
            "filter": [f"term cache policy: process_bytes={256 << 20} "],
        },
        "audits": ["filter"],
        "baseline": "matcher",
        "output": "filter-results.json",
        "profiles": {
            "matcher": ("profile-matcher", "matcher"),
            "filter": ("profile-filter", "filter"),
        },
    },
    "final": {
        "phases": [
            ("n1", "new"),
            ("o1", "old"),
            ("l1", "luxir"),
            ("l2", "luxir"),
            ("o2", "old"),
            ("n2", "new"),
        ],
        "binaries": {"old": "cache", "new": "final"},
        "environment": {"old": {}, "new": {}},
        "logs": {"old": [], "new": []},
        "audits": ["old", "new"],
        "baseline": "old",
        "output": "final-results.json",
        "cells": 57,
    },
    "dispatch": {
        "phases": [
            ("b1", "blocking"),
            ("p1", "pool"),
            ("l1", "luxir"),
            ("p2", "pool"),
            ("b2", "blocking"),
        ],
        "binaries": {"blocking": "dispatch", "pool": "dispatch"},
        "environment": {"blocking": {}, "pool": {}},
        "logs": {"blocking": ["dispatch: Blocking"], "pool": ["dispatch: Pool"]},
        "audits": ["blocking"],
        "baseline": "blocking",
        "output": "dispatch-results.json",
        "cells": 57,
        "profiles": {
            family: ("profile-" + family, "dispatch")
            for family in ["regex", "and_high_med", "or_high_low", "low_term"]
        },
    },
}


def environment(directory):
    lines = (directory / "environ.txt").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def summarize(campaign, folder, archives, output):
    spec = CAMPAIGNS[campaign]
    assert read(folder / "all-completed.json") == {"complete": True}
    assert read(folder / "inventory-before.json") == read(
        folder / "inventory-after.json"
    )
    provenance = {}
    for label in sorted(set(spec["binaries"].values())):
        provenance[label] = read(folder / (label + "-provenance.json"))
        archive = archives / (label + ".tar.gz")
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == provenance[label]["archive_sha256"]
        )
    rows = read(folder / "agreement.json")
    selected = {r["key"] for r in rows if r["include"]}
    probes = {r["key"]: r for r in read(folder / "probes/summa-counts.json")}
    assert set(probes) == {r["key"] for r in rows}
    for row in rows:
        expected, actual = row["observed"]["summa"], probes[row["key"]]
        assert actual.get("count") == expected.get("count")
        assert actual.get("error") == expected.get("error")

    def audit(name):
        return {
            r["class"] + "\t" + r["query"]: r
            for r in map(json.loads, (folder / name).read_text().splitlines())
        }

    audits = audit("baseline-audit.jsonl")
    assert set(audits) == selected
    for label in spec["audits"]:
        assert audit(label + "-audit.jsonl") == audits

    runs, memory = [], []
    for phase, variant in spec["phases"]:
        directory = folder / phase
        engine = "luxir" if variant == "luxir" else "summa"
        assert read(directory / "completed.json") == {"complete": True}
        assert read(directory / "agreement.json") == rows
        server = read(directory / "server.json")
        if engine == "summa":
            binary = provenance[spec["binaries"][variant]]["binary_sha256"]
            assert server["binary_sha256"] == binary
            env = environment(directory)
            expected_env = spec["environment"][variant]
            assert {k: env[k] for k in ENV_KEYS if k in env} == expected_env
            log = (directory / "server.log").read_text()
            expected_logs = spec["logs"][variant]
            assert all(line in log for line in expected_logs), (phase, expected_logs)
            if not expected_logs:
                assert "term cache override" not in log
            responses = read(directory / "responses.json")
            assert set(responses) == selected
            for key, expected in audits.items():
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
        assert len(cells) == spec.get("cells", 15)
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
                    (directory / "smaps.txt").read_text(),
                    ("cache-bin", "tight-bin", "shared-bin", "final-bin"),
                ),
            }
        )

    variants = list(dict.fromkeys(v for _, v in spec["phases"]))
    baseline = spec["baseline"]
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
                assert len(cells) == sum(v == variant for _, v in spec["phases"])
                comparison[variant] = COMMON["phase_metrics"](cells)
            for variant in variants:
                comparison[variant]["over_" + baseline + "_qps"] = (
                    comparison[variant]["mean_phase_median_qps"]
                    / comparison[baseline]["mean_phase_median_qps"]
                )
                comparison[variant]["over_luxir_qps"] = (
                    comparison[variant]["mean_phase_median_qps"]
                    / comparison["luxir"]["mean_phase_median_qps"]
                )
            comparisons.append(comparison)

    profiles = []
    for variant, (name, build) in spec.get("profiles", {}).items():
        directory = folder / name
        assert read(folder / "profiles-completed.json") == {"complete": True}
        assert read(directory / "completed.json") == {"complete": True}
        binary = provenance[build]["binary_sha256"]
        assert read(directory / "server.json")["binary_sha256"] == binary
        report = (directory / "profile.txt").read_text()
        event, symbols = COMMON["parse_profile"](report)
        profiles.append(
            {
                "variant": variant,
                "samples": event[1],
                "event": event[2],
                "symbols": symbols,
                "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
            }
        )

    compilers = {p["compiler"] for p in provenance.values()}
    flags = {p["flags"] for p in provenance.values()}
    assert len(compilers) == 1 and len(flags) == 1
    write(
        output / spec["output"],
        {
            "environment": spec["environment"],
            "builds": {
                label: {k: p[k] for k in ["archive_sha256", "binary_sha256"]}
                for label, p in provenance.items()
            },
            "compiler": compilers.pop(),
            "flags": flags.pop(),
            "audited_queries_per_variant": len(audits),
            "excluded_budget_errors": len(rows) - len(selected),
            "http_checks": {
                engine: 3
                * len(audits)
                * sum(
                    1 for r in runs if (r["variant"] == "luxir") == (engine == "luxir")
                )
                for engine in ["summa", "luxir"]
            },
            "runs": runs,
            "comparisons": comparisons,
            "memory": memory,
            **({"profiles": profiles} if profiles else {}),
        },
    )
    print("requests", sum(c["requests"] for r in runs for c in r["cells"]))
    for c in comparisons:
        print(
            c["family"],
            c["operation"],
            *(
                f"{v}={c[v]['mean_phase_median_qps']:.1f}"
                f"({c[v]['over_' + baseline + '_qps']:.3f}x,"
                f" luxir {c[v]['over_luxir_qps']:.3f}x,"
                f" cpu {c[v]['server_cpu_us_per_request']:.0f}us)"
                for v in variants
            ),
        )
    for m in memory:
        anonymous = m["mapping_kib"]["anonymous"]["Rss"] / 1024
        print(m["phase"], m["variant"], f"anonymous RSS {anonymous:.1f} MiB")
    for profile in profiles:
        print(profile["variant"], profile["samples"])
        for symbol in profile["symbols"][:15]:
            print(f"  {symbol['self_percent']:6.2f}% {symbol['symbol'][:110]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=sorted(CAMPAIGNS))
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("archives", type=Path, help="folder with <label>.tar.gz")
    args = parser.parse_args()
    summarize(args.campaign, args.artifacts, args.archives, Path(__file__).parent)
