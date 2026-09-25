# MAL parser architecture

The Model Architecture Language has one parser and one schema: the Rust
`summa-mal` crate.

| Consumer      | Integration                                                    |
| ------------- | -------------------------------------------------------------- |
| `summa-llm`   | Re-exports `summa-mal` as `summa_llm::mal`                     |
| `summa-train` | Uses the `summa-llm` re-export and shared `ModelDef`           |
| Python tools  | Optional `summa-mal-python` PyO3 wrapper around the same crate |

The grammar, AST, serde representation, embedded well-known models, and computed
properties live under `summa-mal/`. Neither training nor inference has a
parallel parser or copied configuration type.

`summa-mal-python` is a general binding for external Python tools. It exposes
`parse_mal(source) -> JSON`; it is not part of the training path.

When changing MAL, update the grammar/schema and Rust tests in `summa-mal`, then
verify both direct consumers:

```bash
cargo test -p summa-mal -p summa-llm -p summa-train
```
