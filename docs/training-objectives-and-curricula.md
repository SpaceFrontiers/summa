# Training objectives and curricula

Hermes training uses a versioned, sequential curriculum. Each stage has one
explicit objective, sequence length, batch geometry, epoch/step bound, and
deterministic shuffle configuration. Stage boundaries are also optimizer-step
boundaries.

## Objectives and data contracts

| Objective               | JSONL fields                                           | Loss                                                                 |
| ----------------------- | ------------------------------------------------------ | -------------------------------------------------------------------- |
| `causal_lm`             | `text` (plain text is also accepted)                   | Next-token cross-entropy over EOS-packed documents.                  |
| `summarization`         | `document`, `summary`                                  | Causal cross-entropy over summary tokens and its EOS only.           |
| `retrieval_planning`    | `request`, `plan`, optional `context`                  | Causal cross-entropy over plan/action-trace tokens and its EOS only. |
| `contrastive_retrieval` | `query`, `positive`, optional string array `negatives` | Temperature-scaled query-to-document in-batch cross-entropy.         |

Structured examples are padded with EOS for fixed-shape execution, but padding
never contributes to the generative losses. A summarization document or the
optional retrieval-planning context is right-truncated only after reserving
room for the instruction/request, target marker, complete target, and EOS.
Every truncated-token count is written to metrics. A required request or target
that cannot fit is an error rather than being silently shortened.

Retrieval embeddings are L2-normalized last-meaningful-token states. The stage
may select a one-based Transformer layer; the default is the final layer. This
lets hybrid models read after a global-attention layer without adding a second
model or changing existing checkpoint tensors. Every explicit negative and all
other documents in the batch are candidates. A one-document batch is rejected
because its contrastive loss cannot train anything.

## Curriculum format and resume

The top-level `version` is currently `1`; newer versions fail loudly. Relative
data paths resolve against the curriculum file.

```json
{
  "version": 1,
  "stages": [
    {
      "name": "foundation-512",
      "data": "data/pretrain.jsonl.zst",
      "objective": { "type": "causal_lm" },
      "sequence_length": 512,
      "batch_size": 32,
      "gradient_accumulation": 4,
      "epochs": 1,
      "steps": 10000
    },
    {
      "name": "retrieval",
      "data": "data/retrieval.jsonl.zst",
      "objective": {
        "type": "contrastive_retrieval",
        "temperature": 0.05,
        "layer": 24
      },
      "sequence_length": 512,
      "batch_size": 8,
      "gradient_accumulation": 4,
      "epochs": 1,
      "steps": 2000,
      "learning_rate_scale": 0.25
    }
  ]
}
```

`sequence_length`, `batch_size`, and `gradient_accumulation` are required for
every stage. `epochs` defaults to one and `shuffle_buffer` to 8192. `steps`
limits that stage; without it, every complete optimizer step from all
configured epochs is used. `learning_rate_scale` multiplies the global
schedule, and `loss_weight` multiplies the stage loss before backpropagation.
Both default to one.

Training state stores the curriculum signature, position, stage-local step,
and cumulative model tokens. Resume refuses a changed curriculum, so
reordering data or changing sequence/batch geometry cannot silently replay the
wrong examples. Earlier safetensors can warm-start a curriculum with
`--checkpoint`; optimizer/corpus cursor state without a curriculum signature is
rejected.

Each metrics row includes the active objective and its named loss, stage name
and step, raw and weighted loss, effective learning rates, gradient norm,
compute/supervised tokens, examples, truncation count, and throughput.
Contrastive stages additionally report candidate count and top-1 retrieval
accuracy. The existing W&B sidecar forwards these fields without special-case
configuration.
