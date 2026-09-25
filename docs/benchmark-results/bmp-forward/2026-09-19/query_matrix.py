"""Serial latency comparison on an isolated Linux machine; never point at live indexes."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

E = Path(os.environ.get("BMP_QUERY_EVIDENCE", "/mnt/summa-copy-merge/bmp-query"))
env = os.environ | {"SUMMA_PIN_MODE": "copy", "SUMMA_PIN_METADATA_BUDGET_MB": "64"}


def cgroup():
    return Path(
        "/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    )


def counters():
    root = cgroup()
    return {
        p: (root / p).read_text()
        for p in [
            "memory.current",
            "memory.peak",
            "memory.stat",
            "memory.events",
            "io.stat",
        ]
        if (root / p).exists()
    }


def worker(variant, mode, depth, passes, count, label):
    cmd = [
        str(E / (variant + "-bin")),
        "query",
        str(E / (variant + "-index")),
        str(E / "queries.json"),
        depth,
        mode,
        passes,
        count,
        str(E / (label + ".hits.jsonl")),
    ]
    before = counters()
    start = time.monotonic()
    with (E / (label + ".stderr")).open("w") as f:
        r = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=f,
            text=True,
            timeout=900,
            check=True,
        )
    result = json.loads(r.stdout)
    result.update(
        {
            "wall_s": time.monotonic() - start,
            "cgroup_before": before,
            "cgroup_after": counters(),
            "variant": variant,
            "label": label,
            "hits_sha256": hashlib.sha256(
                (E / (label + ".hits.jsonl")).read_bytes()
            ).hexdigest(),
        }
    )
    (E / (label + ".json")).write_text(json.dumps(result))
    print(label, flush=True)


def files(variant):
    return [p for p in (E / (variant + "-index")).iterdir() if p.is_file()]


def cool(variant):
    subprocess.run(["sync"], check=True)
    for p in files(variant):
        with p.open("rb") as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def warm(variant):
    for p in files(variant):
        with p.open("rb") as f:
            while f.read(8 * 1024 * 1024):
                pass


if len(sys.argv) > 1 and sys.argv[1] == "worker":
    worker(*sys.argv[2:])
    sys.exit()
results = []
for budget in ["warm", "1024M", "512M"]:
    cases = [
        ("retrieval", 10),
        ("pipeline", 1000),
        ("scattered", 1000),
        ("local", 1000),
    ]
    if budget == "warm":
        cases.insert(1, ("pipeline", 100))
    for mode, depth in cases:
        for trial in range(2):
            pair = []
            for variant in ["raw", "gap"] if trial == 0 else ["gap", "raw"]:
                label = f"{budget}-{mode}-{depth}-{trial}-{variant}"
                cool(variant)
                args = [
                    variant,
                    mode,
                    str(depth),
                    "3" if budget == "warm" else "2",
                    "96" if budget == "warm" else "64",
                    label,
                ]
                if budget == "warm":
                    warm(variant)
                    worker(*args)
                else:
                    cmd = [
                        "sudo",
                        "systemd-run",
                        "--quiet",
                        "--wait",
                        "--pipe",
                        "--collect",
                        "--unit=bmp-query-" + label,
                        "-p",
                        "MemoryMax=" + budget,
                        "-p",
                        "MemorySwapMax=0",
                        "-p",
                        "MemoryAccounting=yes",
                        "-p",
                        "IOAccounting=yes",
                        "-p",
                        "User=pasha",
                        "/usr/bin/env",
                        "BMP_QUERY_EVIDENCE=" + str(E),
                        "python3",
                        str(Path(__file__).resolve()),
                        "worker",
                        *args,
                    ]
                    with (E / (label + ".service.log")).open("w") as f:
                        subprocess.run(
                            cmd,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=950,
                        )
                result = json.loads((E / (label + ".json")).read_text())
                results.append(result)
                pair.append(result["hits_sha256"])
                (E / "query-results.json").write_text(json.dumps(results))
                print("DONE", label, flush=True)
            assert pair[0] == pair[1], (
                f"Ranking/score-bit mismatch: {budget}/{mode}/{depth}/{trial}"
            )
(E / "matrix-complete").touch()
