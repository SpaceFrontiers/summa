# Forked dependency register

Hermes vendors no source repositories. It temporarily pins three Git forks
for GPU changes that are not yet available from upstream. Every pin below has
an upstream submission and an explicit removal condition.

| Dependency | Hermes pin                  | Upstream status                                                                                                                                                                                                                                                                  | Remove the fork when                                                                                                                                                         |
| ---------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CubeK      | `ppodolsky/cubek@b4fe9788`  | [consolidated forward softmax-LSE and CubeCL compatibility PR #428](https://github.com/tracel-ai/cubek/pull/428)                                                                                                                                                                 | The LSE API is merged and Burn/Hermes can pin an upstream revision containing it. Do not include fork commit `ee578923`: its tensor-core backward was measured and rejected. |
| CubeCL     | `ppodolsky/cubecl@bda6a68d` | [consolidated allocation-retry and cuBLASLt PR #1440](https://github.com/tracel-ai/cubecl/pull/1440)                                                                                                                                                                             | Both runtime changes are merged, or Hermes drops the corresponding feature.                                                                                                  |
| Burn       | `ppodolsky/burn@e2fe6651`   | [integration PR #5190](https://github.com/tracel-ai/burn/pull/5190), with the [pre-PR discussion #5189](https://github.com/tracel-ai/burn/issues/5189) required by Burn's contribution guide. The foreign-stream ordering prerequisite already merged upstream as Burn PR #5166. | The integration is merged against upstream CubeCL/CubeK revisions and Hermes passes its CUDA parity and throughput gates on that upstream stack.                             |

The Apache Arrow `object_store` Git revision is an upstream security-fix pin,
not a fork. It can return to crates.io after a release containing the pinned
quick-xml update is available.

## Update procedure

1. Check the linked submissions before advancing any fork revision.
2. Rebase only the still-required commits onto the dependency's current main;
   exclude changes already merged or recorded as rejected experiments.
3. Run CPU tests and clippy for `hermes-mal`, `hermes-llm`, and
   `hermes-train`, then the CUDA parity and end-to-end training gates.
4. Replace fork URLs with upstream URLs as soon as the last required commit is
   upstream. Keep this register until the lockfile no longer contains the fork.
