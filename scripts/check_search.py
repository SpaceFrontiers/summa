#!/usr/bin/env python3
"""Focused search-stack checks and reproducible Criterion runs (Python 3.11+)."""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("summa-core", "summa-server", "summa-broker", "summa-tool")
BENCHES = (
    "segment_merge",
    "search_pipeline",
    "core_structures",
    "collector_selection",
    "bmp_hot_path",
    "rust_hot_paths",
)


def contracts(root=ROOT):
    """Enforce crate-level ownership without pretending to parse Rust with regex."""
    allowed = {
        "summa-core": set(),
        "summa-server": {"summa-core", "summa-proto"},
        "summa-broker": {"summa-core", "summa-proto"},
        "summa-proto": set(),
        "summa-tool": {"summa-core"},
        "summa-wasm": {"summa-core"},
    }
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    workspace_deps = workspace["workspace"]["dependencies"]
    errors = []
    for package, permitted in allowed.items():
        manifest = tomllib.loads((root / package / "Cargo.toml").read_text())
        # Dev dependencies may cross boundaries for integration testing.
        sections = [manifest, *manifest.get("target", {}).values()]
        for section in sections:
            for kind in ("dependencies", "build-dependencies"):
                for alias, spec in section.get(kind, {}).items():
                    if isinstance(spec, dict) and spec.get("workspace"):
                        spec = workspace_deps[alias]
                    name = (
                        spec.get("package", alias) if isinstance(spec, dict) else alias
                    )
                    if name.startswith("summa-") and name not in permitted:
                        errors.append(
                            f"{package}: {kind} on {name} violates search ownership"
                        )
                    if package == "summa-core" and name in {"tonic", "prost", "clap"}:
                        errors.append(
                            f"{package}: transport/CLI dependency {name} belongs in an adapter"
                        )

    for path in ("AGENTS.md", "docs/search-system-contract.md"):
        source = root / path
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            if not (source.parent / target.split("#", 1)[0]).is_file():
                errors.append(f"{path}: missing linked file {target}")
    if errors:
        raise ValueError("\n".join(errors))


def commands(args):
    packages = [arg for package in PACKAGES for arg in ("-p", package)]
    if args.mode == "contracts":
        return []
    if args.mode == "bench":
        command = [
            "cargo",
            "bench",
            "--locked",
            "-p",
            "summa-core",
            "--bench",
            args.bench,
            "--",
        ]
        if args.save_baseline:
            command += ["--save-baseline", args.save_baseline]
        if args.baseline:
            command += ["--baseline", args.baseline]
        if args.filter:
            command.append(args.filter)
        return [command]

    steps = [
        ["cargo", "fmt", "--all", "--", "--check"],
        [
            "cargo",
            "clippy",
            "--locked",
            *packages,
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        ["cargo", "test", "--locked", *packages, "--features", "summa-core/metrics"],
        [
            "cargo",
            "check",
            "--locked",
            "-p",
            "summa-core",
            "--no-default-features",
            "--features",
            "native",
            "--all-targets",
        ],
    ]
    # Check the broker alone: workspace feature unification enables core writers.
    steps.append(["cargo", "check", "--locked", "-p", "summa-broker", "--all-targets"])
    if args.mode == "full":
        steps += [
            [
                "cargo",
                "check",
                "--locked",
                "-p",
                "summa-core",
                "--no-default-features",
                "--lib",
            ],
            ["cargo", "doc", "--locked", *packages, "--no-deps"],
            [
                "cargo",
                "build",
                "--locked",
                "-p",
                "summa-server",
                "--bin",
                "summa-server",
            ],
            [
                "cargo",
                "test",
                "--locked",
                "-p",
                "summa-broker",
                "--test",
                "e2e_real_server",
                "--",
                "--ignored",
            ],
        ]
    return steps


def capture(command):
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def environment():
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    # Include untracked source: git diff alone misses newly added benchmarks.
    digest = hashlib.sha256(diff)
    untracked = capture(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    for name in sorted(filter(None, untracked.split("\0"))):
        path = ROOT / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return {
        "revision": capture(["git", "rev-parse", "HEAD"]),
        "status": capture(["git", "status", "--short"]),
        "working_diff_sha256": digest.hexdigest(),
        "host": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": capture(["sysctl", "-n", "machdep.cpu.brand_string"])
        if sys.platform == "darwin"
        else platform.processor(),
        "logical_cpus": os.cpu_count(),
        "rustc": capture(["rustc", "-vV"]),
        "cargo": capture(["cargo", "--version"]),
        "flags": {
            name: os.environ.get(name, "")
            for name in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_BUILD_TARGET",
                "CARGO_TARGET_DIR",
                "CARGO_BUILD_JOBS",
                "CRITERION_HOME",
                "RAYON_NUM_THREADS",
                "SUMMA_PIN_MODE",
                "SUMMA_PIN_METADATA_BUDGET_MB",
            )
        },
    }


def run_command(command, log, timeout, env):
    """Own the process group so timeout/interruption also drains child processes."""
    with log.open("w") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Cargo and the integration tests can spawn descendants. Killing
            # only the wrapper leaves builds or test servers running.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contracts", "check", "full", "bench"))
    parser.add_argument(
        "--plan", action="store_true", help="print commands without running checks"
    )
    parser.add_argument("--bench", choices=BENCHES, default="segment_merge")
    parser.add_argument("--filter", help="Criterion benchmark name filter (bench mode)")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--save-baseline")
    baseline.add_argument("--baseline")
    parser.add_argument(
        "--timeout", type=int, default=3600, help="maximum seconds per command"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.mode != "bench" and (args.save_baseline or args.baseline or args.filter):
        parser.error("baseline and filter options require bench mode")
    if args.filter and args.filter.startswith("-"):
        parser.error("benchmark filter must not start with a hyphen")
    for name in (args.save_baseline, args.baseline):
        if name is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            parser.error(
                "baseline names must use letters, digits, underscores or hyphens"
            )
    steps = commands(args)
    if args.plan:
        print("Check search ownership and documentation contracts")
        for command in steps:
            print(shlex.join(command))
        return 0

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / ".context" / "search-harness" / f"{stamp}-{args.mode}"
    output.mkdir(parents=True)
    report = {"mode": args.mode, "steps": [], "success": False}
    print(f"Evidence: {output}", flush=True)
    try:
        contracts()
        print("Search ownership and documentation contracts passed", flush=True)
        if steps:
            report["environment"] = environment()
        env = os.environ.copy()
        if args.mode == "bench":
            # Baselines are evidence, not disposable compiler outputs. Respect
            # an explicit directory while keeping the default across cargo clean.
            env.setdefault(
                "CRITERION_HOME", str(ROOT / ".context/search-harness/criterion")
            )
            report["criterion_directory"] = env["CRITERION_HOME"]
        if args.mode != "bench":
            env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} -Dwarnings".strip()
        env["RUSTDOCFLAGS"] = f"{env.get('RUSTDOCFLAGS', '')} -D warnings".strip()
        for index, command in enumerate(steps, 1):
            print(f"[{index}/{len(steps)}] {shlex.join(command)}", flush=True)
            started = time.monotonic()
            log = output / f"{index:02d}-{command[1]}.log"
            step = {"command": command, "log": log.name}
            report["steps"].append(step)
            try:
                code = run_command(command, log, args.timeout, env)
                step["returncode"] = code
            finally:
                step["seconds"] = time.monotonic() - started
            if code:
                print(
                    "\n".join(log.read_text(errors="replace").splitlines()[-60:]),
                    file=sys.stderr,
                )
                print(f"Failed; full log: {log}", file=sys.stderr)
                return 1
            print(f"Passed ({step['seconds']:.1f}s); {log.name}", flush=True)
        report["success"] = True
        return 0
    except KeyboardInterrupt:
        report["error"] = "Interrupted; command process group terminated"
        print(report["error"], file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(f"Harness failed: {error}", file=sys.stderr)
        return 1
    finally:
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
