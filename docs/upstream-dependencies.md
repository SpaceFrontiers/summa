# Upstream dependency pins

Hermes vendors no source repositories and Cargo contains no personal-fork
URLs. GPU fixes that have not yet merged are fetched from their official
repositories at the immutable heads of the corresponding upstream pull
requests.

| Dependency | Official repository revision                                | Upstream submission                                                                                                              | Return to a release when                                                                              |
| ---------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| CubeK      | `tracel-ai/cubek@7b26a4bdc68c092ecd77ee6ee8a2cd39d61d4a90`  | [PR #428](https://github.com/tracel-ai/cubek/pull/428)                                                                           | The consolidated forward softmax-LSE API is in a compatible release.                                  |
| CubeCL     | `tracel-ai/cubecl@c0efe74d3d4b820fbf992dbbe443ed7125c5205c` | [PR #1440](https://github.com/tracel-ai/cubecl/pull/1440)                                                                        | The allocation-retry and cuBLASLt changes are in a compatible release.                                |
| Burn       | `tracel-ai/burn@973605c4be470d8beaed21e24a7e7010d4101068`   | [PR #5190](https://github.com/tracel-ai/burn/pull/5190) and prerequisite [PR #5166](https://github.com/tracel-ai/burn/pull/5166) | The integration is rebased on current upstream `main` and uses compatible CubeCL and CubeK revisions. |

The Apache Arrow `object_store` Git revision is an official-repository
security-fix pin. It can return to crates.io after a release containing the
pinned `quick-xml` update is available.

## Update procedure

1. Verify the linked upstream submissions and their exact head revisions.
2. Advance only to commits reachable from the official repository URL; never
   add a Cargo patch or dependency URL for a contributor repository.
3. Run the host test, lint, documentation, and CUDA-feature build gates, then
   run CUDA parity and steady-state throughput evidence before promotion.
4. Replace Git revisions with released versions once all required changes are
   available upstream.
