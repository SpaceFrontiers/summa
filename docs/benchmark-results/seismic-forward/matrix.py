"""Linux comparison. Environment: FORWARD_EVIDENCE, FORWARD_FIXTURE, FORWARD_QUERIES.
Binaries before-bin/after-bin and converter are prepared in FORWARD_EVIDENCE.
Run without simultaneous compilers; pressure probes run in a swap-free cgroup.
"""

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

E = Path(os.environ["FORWARD_EVIDENCE"])
SOURCE = Path(os.environ["FORWARD_FIXTURE"])
QUERIES = Path(os.environ["FORWARD_QUERIES"])
ENV = os.environ | {
    "RAYON_NUM_THREADS": "4",
    "TOKIO_WORKER_THREADS": "4",
    "SUMMA_PIN_MODE": "copy",
    "SUMMA_PIN_METADATA_BUDGET_MB": "64",
}


def compilers():
    return [
        p
        for p in subprocess.check_output(
            ["ps", "-A", "-o", "comm="], text=True
        ).splitlines()
        if p.strip() in {"rustc", "cargo", "clippy-driver", "cc1", "cc1plus"}
    ]


def run(label, variant, count=200, k=100, kind="ann", cold=False):
    path = SOURCE if variant == "before" else E / (variant + "-index")
    assert not compilers()
    for p in path.iterdir():
        if p.is_file():
            with p.open("rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                if not cold:
                    while f.read(8 * 1024 * 1024):
                        pass
    done = threading.Event()
    overlap = []

    def watch():
        while not done.is_set():
            active = compilers()
            if active:
                overlap.append(active)
            done.wait(0.5)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        with (
            (E / (label + ".log")).open("w") as out,
            (E / (label + ".time")).open("w") as err,
        ):
            subprocess.run(
                [
                    "/usr/bin/time",
                    "-v",
                    str(E / ("before-bin" if variant == "before" else "after-bin")),
                    "search",
                    str(path),
                    str(QUERIES),
                    str(count),
                    str(k),
                    kind,
                    str(E / (label + ".hits.json")),
                ],
                env=ENV,
                stdout=out,
                stderr=err,
                check=True,
                timeout=1200,
            )
    finally:
        done.set()
        watcher.join()
    assert not overlap
    value = json.loads(
        next(
            s
            for s in reversed((E / (label + ".log")).read_text().splitlines())
            if s.startswith("{")
        )
    )
    value.update(label=label, variant=variant, compiler_overlap=overlap)
    timing = (E / (label + ".time")).read_text()
    for key, pattern in [
        ("rss_kib", r"Maximum resident set size \(kbytes\): (\d+)"),
        ("major_faults", r"Major \(requiring I/O\) page faults: (\d+)"),
        ("fs_input_blocks", r"File system inputs: (\d+)"),
    ]:
        value[key] = int(re.search(pattern, timing).group(1))
    hits = json.loads((E / (label + ".hits.json")).read_text())
    reference = E / f"reference-{kind}-{k}.json"
    if not reference.exists():
        assert variant == "before"
        reference.write_text(json.dumps(hits))
    assert hits == json.loads(reference.read_text())[:count]
    value["hits_identical"] = True
    return value


if len(sys.argv) > 1:
    variant = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "pressure-" + variant
    value = run(label, variant, count=20, cold=True)
    cgroup = Path(
        "/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split("::")[-1]
    )
    value["cgroup"] = {
        p: (cgroup / p).read_text()
        for p in [
            "memory.max",
            "memory.swap.max",
            "memory.peak",
            "memory.events",
            "memory.stat",
        ]
    }
    assert "oom 0\n" in value["cgroup"]["memory.events"]
    (E / (label + ".json")).write_text(json.dumps(value, indent=2))
    raise SystemExit()

subprocess.run(["sync"], check=True)
result = {"warm": [], "exact": [], "top10": [], "pressure": []}


def save():
    (E / "results.json").write_text(json.dumps(result, indent=2))


# Balanced orders; same binary/fixture/flags for format comparisons, plus main baseline.
for i, variant in enumerate(
    [
        "before",
        "raw",
        "u24",
        "dot",
        "dot",
        "u24",
        "raw",
        "before",
        "u24",
        "before",
        "dot",
        "raw",
    ]
):
    result["warm"].append(run(f"warm-{i}-{variant}", variant))
    save()
for variant in ["before", "raw", "u24", "dot"]:
    result["exact"].append(run("exact-" + variant, variant, 5, 100, "exact"))
    save()
    result["top10"].append(run("top10-" + variant, variant, 50, 10))
    save()
if os.environ.get("FORWARD_PRESSURE") == "1":
    for i, variant in enumerate(["raw", "u24", "dot", "dot", "u24", "raw"]):
        label = "pressure-" + ("repeat-" if i >= 3 else "") + variant
        subprocess.run(["sync"], check=True)
        command = [
            "sudo",
            "systemd-run",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
            "--unit=forward-" + variant,
            "-p",
            "MemoryMax=1024M",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "MemoryAccounting=yes",
            "-p",
            "User=pasha",
            "/usr/bin/env",
            *[f"{k}={v}" for k, v in os.environ.items() if k.startswith("FORWARD_")],
            "python3",
            str(Path(__file__).resolve()),
            variant,
            label,
        ]
        with (E / (label + "-service.log")).open("w") as f:
            subprocess.run(
                command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1500
            )
        result["pressure"].append(json.loads((E / (label + ".json")).read_text()))
        save()
(E / "complete").touch()
