# Generation evaluation

Implemented by [`summa-train/src/generate_eval.rs`](../summa-train/src/generate_eval.rs).
The percentages below describe the motivating 300M experiment, not a fresh
evaluation of the current checkout.

## Why

`summa-train eval` measures teacher-forced cross-entropy: given a correct
target prefix, how well the model predicts the next token. That is the right
number for tracking training, and it is not a measure of whether the model can
produce text.

The 300M MoE run made the gap concrete. A corrective SFT pass improved held-out
loss on four of five supervised objectives — grounded QA fell 17%, planning 19% —
while free-running greedy decoding on the same objective still collapsed into
repetition on 5 of 24 records. Earlier, the reverse happened: loss caught
retrieval decay that generations hid entirely. **Perplexity and generation
measure different things and both are required.** Neither alone is sufficient,
and only one of them was automated.

## What is measured

For each held-out record, frame the prompt exactly as training frames it, decode
greedily, and score the decode. Reported as means over scored records:

| metric                                                            | meaning                                                                                                                             |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `repeated_trigram_rate`                                           | share of the decode's word trigrams taken by its most frequent one, excluding that trigram's first occurrence; zero for unique text |
| `degenerate_fraction`                                             | share of decodes above a 0.10 repeated-trigram rate                                                                                 |
| `target_containment`                                              | share whose lowercase alphanumeric word sequence contains the complete gold target word sequence                                    |
| `source_overlap`                                                  | share of the decode's word 4-grams that also occur in the prompt source                                                             |
| `stopped_at_eos`                                                  | share that ended by emitting EOS rather than exhausting `--max-new-tokens`                                                          |
| `empty_fraction`, `mean_generated_tokens`, `mean_generated_words` | shape of the output                                                                                                                 |

Two are deliberately reported without a verdict:

- `source_overlap` is the goal for grounded QA (extract from the passage) and the
  failure mode for summarization (copy sentence one). The task decides, not the
  metric.
- `target_containment` is a correctness _floor_, not accuracy. It counts exact
  inclusion, so a correct answer phrased differently scores zero — the observed
  "Yadkin County" for gold "Yadkin River" counts as a miss.

`stopped_at_eos` low together with `repeated_trigram_rate` high is the signature
of a model that cannot stop, which is the specific failure this project hit.

## Correctness constraints

**The decode prompt is the training prompt by construction.** Both come from
`TaskConfig::construct_supervised_prompt` followed by
`data::structured::frame_supervised`, one function with two consumers. Framing
was previously inlined in `make_supervised_sample`; it was extracted so decoding
could not drift from it.

This is not a stylistic preference. A prompt assembled by hand — passages before
the question, no instruction line, no `Response:` suffix — is out of distribution,
and a continuation-pretrained model correctly continues it instead of answering.
That produced a false "generation is still broken" conclusion on this project,
reported twice, and an exposure-bias theory built on top of it. In the real
format the same checkpoint answers correctly and stops.

**The target's length is reserved, then decoding replaces it.** Training truncates
the source to fit prompt + target + EOS inside `--sequence-length`. Decoding
supplies the gold target too, purely so the same budget is reserved and the
prompt is identical to the one the loss was measured against. The target is never
shown to the model. Source-overlap is computed against only the retained source
token range; text truncated out of the prompt cannot count as grounding.

**Greedy by default.** `--temperature 0` is the default because it is
reproducible and because degenerate copying is visible under it; sampling can
mask a collapsed argmax, which is how one earlier failure stayed hidden. The seed
is always explicit so a report is reproducible even when sampling is enabled.

**Oversized records are skipped and counted**, matching `eval`: one over-long
held-out record must not void an evaluation, but the drop is never silent.

## Shape

```
summa-train generate-eval \
  --config <config.json> --tokenizer <tokenizer.json> --checkpoint <weights.safetensors> \
  --data <shard.jsonl.zst> --objective qa_reasoning \
  --sequence-length 2048 --max-new-tokens 60 --max-records 200 \
  --samples samples.json -o report.json
```

`--samples` writes every prompt, decode, and gold target for review. Metrics
summarize; reading decodes is what caught the failure metrics missed. When
`--samples` is given, the decodes live only in that file — duplicating them into
the metrics report would silently double a large artifact.

Decoding is autoregressive and therefore far slower than `eval`'s single forward
pass; `--max-records` exists to bound it, and what it excludes is reported.

## Tests

- The prompt tokens `frame_supervised` yields equal the training sample's tokens
  up to the target, pinning the shared-framing guarantee.
- `repeated_trigram_rate` and `source_overlap` on hand-computed inputs, including
  the fully-degenerate and fully-unique ends.
- Metrics are finite, in range, and deterministic under greedy decoding.
- Oversized records are skipped, counted, and warned about.
- `--samples` writes decodes to its own file and omits them from the report.
- `--require-reasoning` is rejected for objectives other than `qa_reasoning`.
