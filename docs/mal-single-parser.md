# MAL parser architecture

The Model Architecture Language has one parser and one schema: the Rust
`hermes-mal` crate.

| Consumer       | Integration                                                 |
| -------------- | ----------------------------------------------------------- |
| `hermes-llm`   | Re-exports `hermes-mal` as `hermes_llm::mal`                |
| `hermes-train` | Uses the `hermes-llm` re-export and shared `ModelDef`       |
| Python tools   | Optional `hermes-mal-py` PyO3 wrapper around the same crate |

The grammar, AST, serde representation, embedded well-known models, and computed
properties live under `hermes-mal/`. Neither training nor inference has a
parallel parser or copied configuration type.

`hermes-mal-py` is a general binding for external Python tools. It exposes
`parse_mal(source) -> JSON`; it is not part of the training path.

When changing MAL, update the grammar/schema and Rust tests in `hermes-mal`, then
verify both direct consumers:

```bash
cargo test -p hermes-mal -p hermes-llm -p hermes-train
```
