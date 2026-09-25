#!/usr/bin/env python3
"""Check the adapter against direct token matching and exhaustive ranking."""

import json
import pathlib
import subprocess
import sys
import tempfile

binary = pathlib.Path(sys.argv[1]).resolve()
texts = [
    "alpha beta gamma",
    "alpha padding beta",
    "beta alpha beta alpha",
    "gamma delta",
    "alpha alpha alpha beta",
    "delta epsilon",
    "beta beta gamma",
] * 200
queries = {
    "alpha": lambda t: "alpha" in t,
    "alpha beta": lambda t: "alpha" in t or "beta" in t,
    "+alpha +beta": lambda t: "alpha" in t and "beta" in t,
    "+alpha -padding": lambda t: "alpha" in t and "padding" not in t,
    '"alpha beta"': lambda t: any(
        t[i : i + 2] == ["alpha", "beta"] for i in range(len(t) - 1)
    ),
    '"alpha alpha"': lambda t: any(
        t[i : i + 2] == ["alpha", "alpha"] for i in range(len(t) - 1)
    ),
    '+"alpha beta" +gamma': lambda t: (
        "gamma" in t
        and any(t[i : i + 2] == ["alpha", "beta"] for i in range(len(t) - 1))
    ),
    "alpha +gamma": lambda t: "gamma" in t,
    "absent": lambda t: "absent" in t,
}
with tempfile.TemporaryDirectory(prefix="summa-bench-smoke-") as tmp:
    index = pathlib.Path(tmp) / "idx"
    corpus = "".join(
        json.dumps({"id": str(i), "text": t, "sort_field": i}) + "\n"
        for i, t in enumerate(texts)
    )
    subprocess.run(
        [binary, "index", index, *sys.argv[2:]], input=corpus, text=True, check=True
    )
    lines, expected = [], []
    for query, match in queries.items():
        count = sum(match(t.split()) for t in texts)
        for command in (
            "COUNT",
            "TOP_10_COUNT",
            "TOP_100_COUNT",
            "TOP_1000_COUNT",
            "VERIFY",
        ):
            lines.append(f"{command}\t{query}\n")
            expected.append(str(count))
        for command in ("TOP_10", "TOP_100", "TOP_1000"):
            lines.append(f"{command}\t{query}\n")
            expected.append("1")
    result = subprocess.run(
        [binary, "serve", index],
        input="".join(lines),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = result.stdout.splitlines()
    assert actual == expected, (actual, expected, result.stderr)
    exhaustive = subprocess.run(
        [binary, "serve", index, "--exhaustive"],
        input="".join(lines),
        text=True,
        capture_output=True,
        check=True,
    )
    assert exhaustive.stdout.splitlines() == expected
    invalid = subprocess.run(
        [binary, "serve", index],
        input='TOP_10\t+"unterminated\n',
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0, "malformed syntax must fail"
print(f"PASS: {len(lines) * 2} requests; independent counts and exhaustive top-k")
