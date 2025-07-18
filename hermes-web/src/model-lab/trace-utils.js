export const TRACE_KIND = 'hermes_model_trace'
export const TRACE_VERSION = 1
export const MAX_BUNDLE_BYTES = 64 * 1024 * 1024

const fail = (message) => { throw new Error(message) }
const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value)
const finite = (value) => typeof value === 'number' && Number.isFinite(value)
const integer = (value) => Number.isInteger(value) && value >= 0

export function validateHeatmap(heatmap, path) {
  if (!isObject(heatmap)) fail(`${path} must be an object`)
  if (!integer(heatmap.rows) || heatmap.rows === 0) fail(`${path}.rows must be a positive integer`)
  if (!integer(heatmap.cols) || heatmap.cols === 0) fail(`${path}.cols must be a positive integer`)
  if (!Array.isArray(heatmap.values) || heatmap.values.length !== heatmap.rows * heatmap.cols) {
    fail(`${path}.values has ${heatmap.values?.length ?? 'no'} entries; expected ${heatmap.rows * heatmap.cols}`)
  }
  if (!heatmap.values.every(finite)) fail(`${path}.values contains a non-finite number`)
  if (!Array.isArray(heatmap.row_labels) || heatmap.row_labels.length !== heatmap.rows) {
    fail(`${path}.row_labels must contain ${heatmap.rows} labels`)
  }
  if (!Array.isArray(heatmap.col_labels) || heatmap.col_labels.length !== heatmap.cols) {
    fail(`${path}.col_labels must contain ${heatmap.cols} labels`)
  }
  return heatmap
}

export function interpolateHeatmap(from, to, progress) {
  if (from.rows !== to.rows || from.cols !== to.cols || from.values.length !== to.values.length) {
    fail('Cannot interpolate heatmaps with different shapes')
  }
  const amount = Math.max(0, Math.min(1, progress))
  return {
    ...from,
    values: from.values.map((value, index) => value + (to.values[index] - value) * amount),
    min: Math.min(from.min, to.min),
    max: Math.max(from.max, to.max),
  }
}

export function tokenSignalSeries(inference, tokenIndex) {
  if (!isObject(inference) || !Array.isArray(inference.tokens) || !Array.isArray(inference.stages)) {
    fail('Inference trace is malformed')
  }
  if (!integer(tokenIndex) || tokenIndex >= inference.tokens.length) {
    fail(`Token index ${tokenIndex} is outside the captured trace`)
  }
  return inference.stages.map((stage, stageIndex) => {
    const start = tokenIndex * stage.activation.cols
    const bins = stage.activation.values.slice(start, start + stage.activation.cols)
    const totalEnergy = bins.reduce((sum, value) => sum + value * value, 0)
    const positiveEnergy = bins.reduce((sum, value) => value > 0 ? sum + value * value : sum, 0)
    const negativeEnergy = bins.reduce((sum, value) => value < 0 ? sum + value * value : sum, 0)
    return {
      stageIndex,
      stage: stage.stage,
      label: stage.label,
      layerIndex: stage.layer_index,
      mixer: stage.mixer,
      rms: stage.token_rms[tokenIndex],
      updateRms: finite(stage.token_delta_rms?.[tokenIndex]) ? stage.token_delta_rms[tokenIndex] : null,
      binMean: bins.reduce((sum, value) => sum + value, 0) / bins.length,
      binRms: Math.sqrt(totalEnergy / bins.length),
      positiveBinEnergy: totalEnergy === 0 ? 0 : positiveEnergy / totalEnergy,
      negativeBinEnergy: totalEnergy === 0 ? 0 : negativeEnergy / totalEnergy,
      capturedBins: bins.length,
    }
  })
}

export function validateTrace(trace) {
  if (!isObject(trace)) fail('Trace bundle must be a JSON object')
  if (trace.kind !== TRACE_KIND) fail(`Unsupported trace kind: ${String(trace.kind)}`)
  if (!integer(trace.version) || trace.version === 0) fail('Trace version must be a positive integer')
  if (trace.version > TRACE_VERSION) {
    fail(`Trace version ${trace.version} is newer than this lab supports (${TRACE_VERSION})`)
  }
  if (!isObject(trace.model)) fail('Trace is missing model metadata')
  if (!integer(trace.model.num_layers) || trace.model.num_layers === 0) fail('model.num_layers must be positive')
  if (!integer(trace.model.hidden_size) || trace.model.hidden_size === 0) fail('model.hidden_size must be positive')
  if (!Array.isArray(trace.model.layers) || trace.model.layers.length !== trace.model.num_layers) {
    fail(`model.layers must contain ${trace.model.num_layers} resolved layers`)
  }
  trace.model.layers.forEach((layer, index) => {
    if (!isObject(layer) || !isObject(layer.mixer)) fail(`model.layers[${index}] is malformed`)
    if (!['attention', 'mamba'].includes(layer.mixer.kind)) {
      fail(`model.layers[${index}].mixer.kind is unsupported`)
    }
  })
  if (!isObject(trace.inference) || !Array.isArray(trace.inference.tokens)) {
    fail('Trace is missing inference tokens')
  }
  if (trace.inference.tokens.length === 0) fail('Inference trace contains no captured tokens')
  if (!Array.isArray(trace.inference.stages) || trace.inference.stages.length < 2) {
    fail('Inference trace must contain at least embedding and final-norm stages')
  }
  trace.inference.stages.forEach((stage, index) => {
    validateHeatmap(stage?.activation, `inference.stages[${index}].activation`)
    if (stage.activation.rows !== trace.inference.tokens.length) {
      fail(`inference.stages[${index}] has ${stage.activation.rows} token rows; expected ${trace.inference.tokens.length}`)
    }
    if (!Array.isArray(stage.token_rms) || stage.token_rms.length !== trace.inference.tokens.length || !stage.token_rms.every(finite)) {
      fail(`inference.stages[${index}].token_rms is malformed`)
    }
    if (stage.attention) {
      if (!integer(stage.attention.total_heads) || !Array.isArray(stage.attention.heads)) {
        fail(`inference.stages[${index}].attention is malformed`)
      }
      stage.attention.heads.forEach((head, headIndex) => validateHeatmap(head, `inference.stages[${index}].attention.heads[${headIndex}]`))
    }
    if (stage.mamba_state) validateHeatmap(stage.mamba_state, `inference.stages[${index}].mamba_state`)
  })
  if (trace.training !== undefined && trace.training !== null) {
    if (!isObject(trace.training) || !Array.isArray(trace.training.rows)) fail('training must contain rows')
    if (trace.training.rows.length === 0) fail('training.rows is empty')
    trace.training.rows.forEach((row, index) => {
      if (!isObject(row) || !integer(row.step)) fail(`training.rows[${index}] has no valid step`)
    })
  }
  return trace
}

export function parseTraceJson(text) {
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    throw new Error(`Trace is not valid JSON: ${error.message}`)
  }
  return validateTrace(parsed)
}

export const metricDefinitions = [
  { key: 'loss', label: 'Loss', unit: '', better: 'lower' },
  { key: 'weighted_loss', label: 'Weighted loss', unit: '', better: 'lower' },
  { key: 'lr', label: 'Learning rate', unit: '', better: null },
  { key: 'grad_norm', label: 'Gradient norm', unit: '', better: null },
  { key: 'tokens_per_second', label: 'Throughput', unit: ' tok/s', better: 'higher' },
  { key: 'retrieval_accuracy', label: 'Retrieval accuracy', unit: '%', better: 'higher', percent: true },
]

export function availableMetrics(rows) {
  return metricDefinitions.filter((definition) => rows.some((row) => finite(row[definition.key])))
}

export function metricSeries(rows, key) {
  return rows
    .filter((row) => finite(row[key]))
    .map((row) => ({ step: row.step, value: row[key], stage: row.stage_name ?? `Stage ${row.stage ?? 1}` }))
}

export function normalizedMetricHeatmap(rows, definitions = availableMetrics(rows)) {
  const selected = definitions.filter((definition) => rows.every((row) => finite(row[definition.key])))
  const usable = selected.length > 0 ? selected : definitions.filter((definition) => rows.some((row) => finite(row[definition.key])))
  const values = []
  for (const definition of usable) {
    const series = rows.map((row) => finite(row[definition.key]) ? row[definition.key] : Number.NaN)
    const present = series.filter(finite)
    const min = Math.min(...present)
    const max = Math.max(...present)
    const range = max - min
    let last = present[0] ?? 0
    for (const value of series) {
      if (finite(value)) last = value
      values.push(range === 0 ? 0.5 : (last - min) / range)
    }
  }
  return {
    rows: usable.length,
    cols: rows.length,
    row_labels: usable.map((definition) => definition.label),
    col_labels: rows.map((row) => String(row.step)),
    values,
    min: 0,
    max: 1,
    value_kind: 'normalized',
    original_rows: usable.length,
    original_cols: rows.length,
  }
}

export function layerGradientHeatmap(rows, layerCount) {
  const samples = rows.filter((row) => Array.isArray(row.layer_grad_norms))
  if (samples.length === 0) return null
  samples.forEach((row, index) => {
    if (row.layer_grad_norms.length !== layerCount || !row.layer_grad_norms.every(finite)) {
      fail(`layer_grad_norms at sampled row ${index} must contain ${layerCount} finite values`)
    }
  })
  const values = []
  for (let layer = 0; layer < layerCount; layer += 1) {
    for (const row of samples) values.push(row.layer_grad_norms[layer])
  }
  return {
    heatmap: {
      rows: layerCount,
      cols: samples.length,
      row_labels: Array.from({ length: layerCount }, (_, index) => `Layer ${index + 1}`),
      col_labels: samples.map((row) => String(row.step)),
      values,
      min: Math.min(...values),
      max: Math.max(...values),
      value_kind: 'gradient_norm',
      original_rows: layerCount,
      original_cols: samples.length,
    },
    samples: samples.length,
  }
}

export function formatCount(value) {
  if (!finite(value)) return '—'
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return String(value)
}

export function formatValue(value, definition = null) {
  if (!finite(value)) return '—'
  if (definition?.percent) return `${(value * 100).toFixed(1)}%`
  if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(2)
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 })
}
