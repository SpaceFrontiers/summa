# Publishing packages

`publish.yml` publishes a version already merged into the default branch.
It does not bump versions or push to `main`: branch protection requires a PR,
and organization policy forbids GitHub Actions from creating/approving PRs.

Before a new release, submit a maintainer PR updating these declarations:

- `Cargo.toml`: workspace package version and internal `summa-*` dependency versions.
- `summa-client-python/pyproject.toml` and `summa-mal-python/pyproject.toml`.
- `summa-client-typescript/package.json`.

Refresh the lockfiles with `cargo update --workspace` and
`uv lock --project summa-client-python`, then commit them in the same PR.
Use `python3 -m unittest discover -s scripts -p test_publish_version.py` to
check release orchestration. Merge the version PR before dispatching:

```sh
gh workflow run publish.yml --ref main -f version=2.0.1
```

The optional `version` input asserts the expected version; when blank, the
workflow reads `Cargo.toml`. All Rust/Python/TypeScript declarations and local
lockfile entries must agree. The workflow tags the exact dispatch commit, and
every package/image job checks out that same SHA even if `main` later advances.
It never moves an existing tag to a different commit.

For a partial publication failure, rerun **failed jobs** on the original run.
An existing tag at the same commit can be reused. Do not dispatch a newer main
commit with the same version: bump versions through a new PR for a new release.
The publication checkboxes still select crates.io, NPM, PyPI and Docker jobs.
