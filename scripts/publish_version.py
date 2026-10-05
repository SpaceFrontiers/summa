"""Merge a release-only PR before tagging its exact merge commit.

Requires contents/pull-requests write permissions and the repository setting
allowing Actions to create PRs. Normal branch protection applies: a required
review or check stops publication; this helper never approves or bypasses it.
"""

import json
import os
import re
import subprocess
import sys

RELEASE_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "summa-client-python/pyproject.toml",
    "summa-client-python/uv.lock",
    "summa-mal-python/pyproject.toml",
    "summa-client-typescript/package.json",
)


def command(*args, input=None):
    return subprocess.run(
        args, input=input, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def publish(version, branch, base, repo, run=command):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Expected a numeric major.minor.patch release version")
    if not branch.startswith(f"release/v{version}-") or branch == base:
        raise ValueError("Release changes require a separate release branch")
    changed = set(run("git", "diff", "--name-only", "HEAD").splitlines())
    if changed - set(RELEASE_FILES):
        raise ValueError(
            f"Unexpected release changes: {sorted(changed - set(RELEASE_FILES))}"
        )
    run("git", "switch", "-c", branch)
    run("git", "add", "--", *RELEASE_FILES)
    run("git", "commit", "-m", f"chore: bump version to {version}")
    head = run("git", "rev-parse", "HEAD")
    run("git", "push", "origin", f"HEAD:refs/heads/{branch}")
    pr = run(
        "gh",
        "pr",
        "create",
        "--repo",
        repo,
        "--base",
        base,
        "--head",
        branch,
        "--title",
        f"chore: bump version to {version}",
        "--body-file",
        "-",
        input=f"Prepare version {version} for publish.yml. Only package versions and "
        "lockfiles change. Publishing starts only after this PR is merged; "
        "the release tag points to its exact merge commit.\n",
    )
    print(f"Release pull request: {pr}", flush=True)
    run(
        "gh", "pr", "merge", pr, "--repo", repo, "--squash", "--match-head-commit", head
    )
    merged = json.loads(
        run(
            "gh",
            "pr",
            "view",
            pr,
            "--repo",
            repo,
            "--json",
            "state,headRefOid,mergeCommit",
        )
    )
    if merged["state"] != "MERGED":
        raise ValueError(f"Release PR is not merged; no tag was created: {pr}")
    if merged["headRefOid"] != head:
        raise ValueError("Release PR head changed; refusing to tag another revision")
    merge_sha = merged["mergeCommit"]["oid"]
    run("git", "fetch", "origin", merge_sha)
    # A merge may include newer main commits, but all prepared version files
    # must still match the release we are about to publish.
    run("git", "diff", "--exit-code", head, merge_sha, "--", *RELEASE_FILES)
    run("git", "tag", f"v{version}", merge_sha)
    run("git", "push", "origin", f"refs/tags/v{version}")
    return merge_sha


if __name__ == "__main__":
    version = sys.argv[1]
    branch = f"release/v{version}-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    sha = publish(
        version, branch, os.environ["BASE_BRANCH"], os.environ["GITHUB_REPOSITORY"]
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"release_sha={sha}\n")
