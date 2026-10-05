"""Tag a version already reviewed and merged; never write a protected branch."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run(
        args, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def release_version(root, requested=""):
    def toml(path):
        return tomllib.loads((root / path).read_text())

    workspace = toml("Cargo.toml")["workspace"]
    version = workspace["package"]["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Expected a numeric major.minor.patch release version")
    if requested and requested != version:
        raise ValueError(f"Requested {requested}, but checked-out version is {version}")
    versions = {
        path: toml(path)["project"]["version"]
        for path in (
            "summa-client-python/pyproject.toml",
            "summa-mal-python/pyproject.toml",
        )
    }
    versions["TypeScript client"] = json.loads(
        (root / "summa-client-typescript/package.json").read_text()
    )["version"]
    for name, spec in workspace["dependencies"].items():
        if isinstance(spec, dict) and "path" in spec and name.startswith("summa-"):
            versions[f"workspace dependency {name}"] = spec["version"]
    for package in toml("Cargo.lock")["package"]:
        if "source" not in package and package["name"].startswith("summa-"):
            versions[f"Cargo.lock {package['name']}"] = package["version"]
    client = next(
        package
        for package in toml("summa-client-python/uv.lock")["package"]
        if package["name"] == "summa-client-python"
    )
    versions["Python client lockfile"] = client["version"]
    mismatches = {name: value for name, value in versions.items() if value != version}
    if mismatches:
        raise ValueError(f"Release versions must all match {version}: {mismatches}")
    return version


def publish(version, run=command):
    if run("git", "status", "--porcelain"):
        raise ValueError("Release must use a clean, committed checkout")
    head = run("git", "rev-parse", "HEAD")
    tag = f"v{version}"
    ref = f"refs/tags/{tag}"
    if run("git", "ls-remote", "--tags", "origin", ref):
        run("git", "fetch", "origin", f"{ref}:{ref}")
        if run("git", "rev-parse", f"{tag}^{{commit}}") != head:
            raise ValueError(
                f"{tag} already tags another commit; bump versions through a PR first"
            )
    else:
        run("git", "tag", tag, head)
        run("git", "push", "origin", ref)
    return head


if __name__ == "__main__":
    os.chdir(ROOT)
    version = release_version(ROOT, sys.argv[1] if len(sys.argv) > 1 else "")
    sha = publish(version)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"new_version={version}\nrelease_sha={sha}\n")
