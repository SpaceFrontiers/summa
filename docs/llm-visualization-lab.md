# LLM Visualization Lab

## Purpose

Hermes needs one inspectable view of the model described by MAL, the values that
flow through a real checkpoint during inference, and the metrics produced by a
training run. The lab is for education and debugging; it is not a profiler and
does not participate in model execution.

The implementation has three boundaries:

1. `hermes-llm trace` runs the existing `hermes_llm::Transformer`, captures a
   bounded diagnostic pass, and writes a portable JSON bundle.
2. `hermes-llm lab` keeps that same model resident, serves the model-lab page on
   loopback, and accepts bounded prompt requests through `POST /api/trace`.
3. The standalone model-lab entry in `hermes-web` can submit a query to that
   local session or load a saved bundle, then renders architecture, inference,
   and training views. It does not parse MAL or load model weights in
   JavaScript.

MAL remains the architecture source of truth, and tracing does not introduce a
second model or checkpoint format.

## Views

- **Architecture** shows the resolved layer sequence, including cyclic MAL
  patterns, attention/Mamba mixer details, FFN shape, normalization, residuals,
  and estimated parameter count. Selecting a layer connects it to its captured
  values.
- **Inference** shows token-by-channel residual-stream heatmaps at embedding,
  every block, and final norm; per-token RMS and change across layers; captured
  attention heads; and final Mamba recurrent state. Token and layer selectors
  make the same trace explorable without rerunning the model.
- **Training** plots loss, learning rate, gradient norm, throughput, and
  curriculum stages from `metrics.jsonl`. A normalized metric heatmap makes
  correlated changes visible. When opt-in per-layer gradient norms are present,
  it also shows a layer-by-step gradient heatmap. The W&B sidecar expands those
  rows into one scalar series per layer for the existing remote dashboard.

## Trace bundle

Bundles are JSON with `kind: "hermes_model_trace"` and `version: 1`. Readers
must reject an unknown kind or a version newer than they support. Required top
level sections are:

```text
model       resolved MAL metadata and one descriptor per concrete layer
inference   prompt/output tokens and bounded diagnostic tensors
training    optional, bounded rows copied from a metrics JSONL file
capture     requested limits, original sizes, retained sizes, and truncation
```

Dense values use a shared heatmap object:

```text
{ rows, cols, row_labels, col_labels, values, min, max, value_kind }
```

`values` is row-major and its length must equal `rows * cols`. Residual streams
are reduced to signed means over contiguous hidden-channel bins. Attention is
not channel-binned, but only the requested leading heads are retained. Mamba
state is reduced over contiguous inner-channel bins. Every reduction records
the original and captured dimensions in `capture`; nothing is silently dropped.

For the selected token, the inference view also renders a compact signal-flow
graph across model depth. Its log-scaled vertical position reports the exact RMS
captured from the full hidden vector, and the thickness of each incoming edge
reports the exact RMS of that block's residual update. Attention, Mamba, and
endpoint stages use different node shapes. Small positive/negative bars
summarize the squared energy of the captured signed channel-bin means; this sign
balance is explicitly approximate because values were aggregated before they
reached the browser.
Dense models retain every graph segment but reduce stage labels and glyphs to
fit the available width. Playback moves one probe over the graph instead of
rebuilding it, and the visible readout reports the interpolated display value
between exact captured endpoints.

Training rows retain the original JSON objects. If their count exceeds the
configured limit, deterministic evenly spaced sampling keeps the first and last
row and records total rows, retained rows, and sampling stride.

## Capture path and cost

Tracing is off by default. The normal `generate` and trainer paths allocate
nothing for it.

The trace command first performs ordinary cached generation. It then runs one
full-sequence diagnostic pass over the retained tokens with dropout-disabled
inference weights. Each block output is copied to the CPU before aggregation.
Attention weights are recomputed from the actual normalized mixer input because
the optimized attention kernel intentionally does not materialize them. Mamba
uses the same stateful mixer path and captures its final recurrent state. This
extra work is acceptable only in the explicit diagnostic command and is called
out in CLI output.

Defaults bound a bundle to 128 tokens, 64 residual channels, four attention
heads per layer, and 2,000 metric rows. The CLI validates non-zero limits and
prints every token, channel, head, or metric-row reduction.

## Live local session

`hermes-llm lab` loads the checkpoint and tokenizer once, then moves them to one
dedicated inference worker. Requests are serialized: one request may wait in a
bounded queue and further concurrent requests receive an explicit busy response.
The HTTP runtime never accesses model tensors directly and a failed request does
not reload the checkpoint.

The server binds to `127.0.0.1` by default and refuses a non-loopback address
unless `--allow-remote` is explicitly supplied. Prompt bytes, generated tokens,
temperature, top-k, and JSON request size are validated against server-side
limits. Static assets and the API share an origin; no CORS permission is added.

After a live response, the page selects the generated answer token and plays the
captured sequence from embedding through every block to final norm. Play/pause,
previous/next, and speed controls update the existing stage selector, heatmap,
attention/Mamba view, and selected layer. A labeled flow rail makes that route
explicit and lets a user jump to any captured stage.

Playback interpolates the displayed residual values between adjacent captured
stages with a bounded ease curve. Those animation frames are presentation only:
the endpoints are the exact captured tensors and no interpolated frame is
reported as a model checkpoint. The activation canvas remains allocated while
values move so stage changes do not flash through an empty chart; controls and
secondary charts update once at the captured endpoint. Reduced-motion clients
switch stages immediately without interpolation.

## Failure and safety behavior

- Configuration is loaded through the existing MAL parser for `.mal`, or the
  existing exported-JSON reader; the browser never has a competing parser.
- Vocabulary mismatches, malformed metric lines, non-finite values, invalid
  matrix sizes, and unsupported bundle versions are hard errors with context.
- The bundle loader reads only the file the user selects. Live mode talks only
  to the same-origin local server; the lab does not upload prompts, tokens,
  metrics, or activations to an external service.
- Large tensors are aggregated during export, not after creating a huge JSON
  document in the browser.

## Extensibility

New optional fields may be added with reader defaults within version 1.
Semantic or incompatible shape changes require a version bump. Future profiler
timings or optimizer-state views should extend the bundle rather than add a
second instrumentation channel.
