"""Compare accumulator implementations against the same immutable BMPB fixture."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FIXTURE = Path(os.environ.get("BMP_QUERY_EVIDENCE", "/mnt/summa-copy-merge/bmp-query"))
OUTPUT = FIXTURE / "optimization"
env = os.environ | {"SUMMA_PIN_MODE": "copy", "SUMMA_PIN_METADATA_BUDGET_MB": "64"}


def counters():
    cg = Path(
        "/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    )
    return {
        p: (cg / p).read_text()
        for p in ["memory.peak", "memory.events", "memory.stat", "io.stat"]
        if (cg / p).exists()
    }


def worker(variant, mode, depth, passes, count, label):
    before = counters()
    start = time.monotonic()
    with (OUTPUT / (label + ".stderr")).open("w") as err:
        result = subprocess.run(
            [
                str(OUTPUT / (variant + "-bin")),
                "query",
                str(FIXTURE / "gap-index"),
                str(FIXTURE / "queries.json"),
                depth,
                mode,
                passes,
                count,
                str(OUTPUT / (label + ".hits.jsonl")),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            check=True,
            timeout=900,
        )
    data = json.loads(result.stdout)
    data.update(
        variant=variant,
        label=label,
        wall_s=time.monotonic() - start,
        cgroup_before=before,
        cgroup_after=counters(),
        hits_sha256=hashlib.sha256(
            (OUTPUT / (label + ".hits.jsonl")).read_bytes()
        ).hexdigest(),
    )
    (OUTPUT / (label + ".json")).write_text(json.dumps(data))
    print(label, flush=True)


def fixture_hashes():
    result = {}
    for p in (FIXTURE / "gap-index").iterdir():
        if p.is_file():
            with p.open("rb") as f:
                result[p.name] = hashlib.file_digest(f, "sha256").hexdigest()
    return result


if len(sys.argv) > 1 and sys.argv[1] == "worker":
    worker(*sys.argv[2:])
    sys.exit()

hashes = fixture_hashes()
results = []
for budget, cases in [
    (
        "warm",
        [("retrieval", 10), ("pipeline", 100), ("pipeline", 1000), ("scattered", 1000)],
    ),
    ("1024M", [("pipeline", 1000)]),
    ("512M", [("scattered", 1000)]),
]:
    for mode, depth in cases:
        for trial in range(2):
            pair = []
            for variant in ["before", "after"] if trial == 0 else ["after", "before"]:
                for p in (FIXTURE / "gap-index").iterdir():
                    if p.is_file():
                        with p.open("rb") as f:
                            if budget == "warm":
                                while f.read(8 * 1024 * 1024):
                                    pass
                            else:
                                os.posix_fadvise(
                                    f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                                )
                label = f"{budget}-{mode}-{depth}-{trial}-{variant}"
                args = [
                    variant,
                    mode,
                    str(depth),
                    "3" if budget == "warm" else "2",
                    "64",
                    label,
                ]
                if budget == "warm":
                    worker(*args)
                else:
                    subprocess.run(
                        [
                            "sudo",
                            "systemd-run",
                            "--quiet",
                            "--wait",
                            "--collect",
                            "--unit=bmp-opt-" + label,
                            "-p",
                            "MemoryMax=" + budget,
                            "-p",
                            "MemorySwapMax=0",
                            "-p",
                            "IOAccounting=yes",
                            "python3",
                            str(Path(__file__).resolve()),
                            "worker",
                            *args,
                        ],
                        check=True,
                        timeout=960,
                    )
                data = json.loads((OUTPUT / (label + ".json")).read_text())
                data.update(budget=budget, trial=trial)
                results.append(data)
                pair.append(data["hits_sha256"])
                (OUTPUT / "results.json").write_text(json.dumps(results))
            assert pair[0] == pair[1], (budget, mode, trial, "scores differ")
assert fixture_hashes() == hashes
(OUTPUT / "fixture-hashes.json").write_text(json.dumps(hashes, indent=2))
(OUTPUT / "matrix-complete").touch()
print("MATRIX COMPLETE", flush=True)
