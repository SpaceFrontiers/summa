const tokenPieces = ['Retrieval', '·planning', '·works', '·best', '·when', '·the', '·model', '·first', '·identifies', '·the', '·missing', '·evidence', ',', '·then', '·asks', '·for', '·it', '.']

function seededValue(index, seed, scale = 1) {
  return (Math.sin(index * 0.37 + seed * 1.71) * 0.65 + Math.cos(index * 0.11 - seed) * 0.35) * scale
}

function activationHeatmap(stageIndex) {
  const rows = tokenPieces.length
  const cols = 32
  const values = Array.from({ length: rows * cols }, (_, index) => {
    const row = Math.floor(index / cols)
    const channel = index % cols
    const signal = seededValue(index, stageIndex + 1, 0.45 + stageIndex * 0.035)
    const semanticBand = row > 7 && row < 13 && channel > 10 && channel < 20 ? 0.35 : 0
    return Number((signal + semanticBand).toFixed(5))
  })
  return {
    rows,
    cols,
    row_labels: tokenPieces.map((piece, index) => `${index}:${piece}`),
    col_labels: Array.from({ length: cols }, (_, index) => `${index * 4}..${index * 4 + 3}`),
    values,
    min: Math.min(...values),
    max: Math.max(...values),
    value_kind: 'signed_mean',
    original_rows: rows,
    original_cols: 128,
  }
}

function attentionHead(head, stageIndex) {
  const size = tokenPieces.length
  const values = []
  for (let query = 0; query < size; query += 1) {
    const scores = []
    for (let key = 0; key <= query; key += 1) {
      const recency = Math.exp(-(query - key) / (2.4 + head))
      const anchor = [0, 1, 8, 10, 11].includes(key) ? 0.55 : 0
      scores.push(recency + anchor + Math.abs(seededValue(query * size + key, stageIndex + head, 0.12)))
    }
    const total = scores.reduce((sum, value) => sum + value, 0)
    for (let key = 0; key < size; key += 1) values.push(key <= query ? Number((scores[key] / total).toFixed(6)) : 0)
  }
  return {
    rows: size,
    cols: size,
    row_labels: tokenPieces.map((piece, index) => `${index}:${piece}`),
    col_labels: tokenPieces.map((piece, index) => `${index}:${piece}`),
    values,
    min: 0,
    max: Math.max(...values),
    value_kind: 'probability',
    original_rows: size,
    original_cols: size,
  }
}

function mambaState(stageIndex) {
  const rows = 12
  const cols = 24
  const values = Array.from({ length: rows * cols }, (_, index) => Number(seededValue(index, stageIndex, 0.8).toFixed(5)))
  return {
    rows,
    cols,
    row_labels: Array.from({ length: rows }, (_, index) => `state ${index}`),
    col_labels: Array.from({ length: cols }, (_, index) => `${index * 11}..${index * 11 + 10}`),
    values,
    min: Math.min(...values),
    max: Math.max(...values),
    value_kind: 'signed_mean',
    original_rows: rows,
    original_cols: 256,
  }
}

const mixers = ['mamba', 'mamba', 'attention', 'mamba', 'mamba', 'attention']
const layers = mixers.map((kind, index) => ({
  index: index + 1,
  name: kind === 'mamba' ? 'local_memory' : 'global_routing',
  pattern_index: index % 3,
  mixer: kind === 'attention'
    ? { kind, num_heads: 4, num_kv_heads: 2, head_dim: 32, causal: true, window_size: null, qk_norm: true, position_encoding: 'rope(theta=10000)' }
    : { kind, state_dim: 12, conv_kernel: 4, expand: 2, dt_rank: 8, inner_size: 256 },
  ffn_hidden_size: 384,
  ffn_activation: 'swiglu',
  ffn_gated: true,
  norm: 'rms_norm',
  norm_position: 'pre',
  residual: true,
  dropout: 0,
}))

function stage(stageIndex, kind, layerIndex = null) {
  const activation = activationHeatmap(stageIndex)
  const tokenRms = Array.from({ length: tokenPieces.length }, (_, token) => Number((0.82 + stageIndex * 0.08 + Math.abs(seededValue(token, stageIndex, 0.16))).toFixed(5)))
  const values = activation.values
  const stageTrace = {
    stage: kind,
    label: kind === 'embedding' ? 'Token embedding' : kind === 'final_norm' ? 'Final norm' : `Layer ${layerIndex} · ${mixers[layerIndex - 1]}`,
    layer_index: layerIndex,
    mixer: layerIndex ? mixers[layerIndex - 1] : null,
    activation,
    stats: {
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length),
      min: Math.min(...values),
      max: Math.max(...values),
      abs_max: Math.max(...values.map(Math.abs)),
    },
    token_rms: tokenRms,
    token_delta_rms: stageIndex === 0 ? null : tokenRms.map((value, token) => Number((0.08 + value * 0.03 + token * 0.001).toFixed(5))),
  }
  if (layerIndex && mixers[layerIndex - 1] === 'attention') {
    stageTrace.attention = { total_heads: 4, captured_heads: 2, heads: [attentionHead(0, stageIndex), attentionHead(1, stageIndex)] }
  }
  if (layerIndex && mixers[layerIndex - 1] === 'mamba') stageTrace.mamba_state = mambaState(stageIndex)
  return stageTrace
}

const stages = [stage(0, 'embedding'), ...layers.map((_, index) => stage(index + 1, 'block', index + 1)), stage(7, 'final_norm')]
const trainingRows = Array.from({ length: 84 }, (_, index) => {
  const step = (index + 1) * 50
  const stageIndex = index < 42 ? 1 : index < 68 ? 2 : 3
  const baseLoss = stageIndex === 1 ? 4.7 : stageIndex === 2 ? 2.3 : 1.45
  const localStep = stageIndex === 1 ? index : stageIndex === 2 ? index - 42 : index - 68
  const row = {
    step,
    stage: stageIndex,
    stage_name: ['language foundation', 'retrieval grounding', 'planning instructions'][stageIndex - 1],
    objective: ['causal_lm', 'retrieval', 'instruction'][stageIndex - 1],
    loss: Number((baseLoss * Math.exp(-localStep / 32) + 0.08 * Math.abs(Math.sin(index * 0.8))).toFixed(5)),
    weighted_loss: Number((baseLoss * Math.exp(-localStep / 32) + 0.08 * Math.abs(Math.sin(index * 0.8))).toFixed(5)),
    lr: index < 8 ? 0.0003 * (index + 1) / 8 : 0.0003 * (1 - index / 105),
    grad_norm: Number((0.72 + Math.abs(Math.sin(index * 0.42)) * 0.48 + (index === 43 ? 1.2 : 0)).toFixed(5)),
    tokens_per_second: Math.round(41800 + Math.sin(index * 0.17) * 2200 - stageIndex * 350),
    tokens: step * 32768,
  }
  if (index % 4 === 0) {
    row.layer_grad_norms = layers.map((_, layer) => Number((0.18 + layer * 0.035 + Math.abs(seededValue(index + layer, 4, 0.24))).toFixed(5)))
  }
  return row
})

export const demoTrace = {
  kind: 'hermes_model_trace',
  version: 1,
  model: {
    name: 'hermes-hybrid-debug-demo',
    description: 'Synthetic but structurally faithful hybrid trace. Open a trace bundle to inspect checkpoint values.',
    vocab_size: 32000,
    max_seq_len: 1024,
    hidden_size: 128,
    num_layers: layers.length,
    estimated_parameters: 12480768,
    tied_embeddings: true,
    layers,
  },
  inference: {
    prompt: 'How should a small model plan retrieval?',
    generated_text: 'Retrieval planning works best when the model first identifies the missing evidence, then asks for it.',
    full_text: tokenPieces.join('').replaceAll('·', ' '),
    original_token_count: tokenPieces.length,
    prompt_token_count: 7,
    generated_token_count: tokenPieces.length - 7,
    token_offset: 0,
    tokens: tokenPieces.map((piece, index) => ({ original_index: index, id: 120 + index * 7, piece: piece.replaceAll('·', ' '), display: piece, source: index < 7 ? 'prompt' : 'generated' })),
    sampling: { max_new_tokens: 32, temperature: 0.7, top_k: 40, seed: 42, stop_at_eos: true },
    stages,
  },
  training: {
    source: 'synthetic/demo/metrics.jsonl',
    total_rows: trainingRows.length,
    captured_rows: trainingRows.length,
    dropped_rows: 0,
    sampling_stride: 1,
    rows: trainingRows,
  },
  capture: {
    requested_token_limit: 128,
    original_tokens: tokenPieces.length,
    captured_tokens: tokenPieces.length,
    dropped_leading_tokens: 0,
    tokens_truncated: false,
    original_hidden_channels: 128,
    captured_hidden_channels: 32,
    channels_reduced: true,
    requested_attention_head_limit: 2,
    metrics_total_rows: trainingRows.length,
    metrics_captured_rows: trainingRows.length,
  },
}
