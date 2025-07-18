import { demoTrace } from './demo-trace.js'
import {
  MAX_BUNDLE_BYTES,
  availableMetrics,
  formatCount,
  formatValue,
  interpolateHeatmap,
  layerGradientHeatmap,
  metricSeries,
  normalizedMetricHeatmap,
  parseTraceJson,
  tokenSignalSeries,
  validateTrace,
} from './trace-utils.js'

const $ = (id) => document.getElementById(id)
const css = (name) => window.getComputedStyle(document.documentElement).getPropertyValue(name).trim()
const svgNamespace = 'http://www.w3.org/2000/svg'
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

const state = {
  trace: null,
  source: 'Synthetic example',
  currentView: 'architecture',
  selectedLayer: 0,
  selectedStage: 0,
  selectedToken: 0,
  selectedHead: 0,
  selectedMetric: 'loss',
  liveReady: false,
  liveRunning: false,
  liveStatus: null,
  playing: false,
  playbackDelay: 700,
  playbackTimer: null,
  animationFrame: null,
  animating: false,
  animationTarget: null,
}

function showError(message) {
  const banner = $('error-banner')
  banner.textContent = message
  banner.hidden = false
}

function clearError() {
  $('error-banner').hidden = true
  $('error-banner').textContent = ''
}

function setText(id, value) {
  $(id).textContent = value ?? ''
}

function setTrace(trace, source, options = {}) {
  stopPlayback()
  const validated = validateTrace(trace)
  state.trace = validated
  state.source = source
  state.selectedLayer = 0
  const defaultStage = Math.max(0, validated.inference.stages.findIndex((stage) => stage.attention))
  state.selectedStage = Math.max(0, Math.min(validated.inference.stages.length - 1, options.initialStage ?? defaultStage))
  const defaultToken = validated.inference.prompt_token_count - validated.inference.token_offset - 1
  state.selectedToken = Math.max(0, Math.min(validated.inference.tokens.length - 1, options.selectedToken ?? defaultToken))
  state.selectedHead = 0
  state.selectedMetric = availableMetrics(validated.training?.rows ?? [])[0]?.key ?? 'loss'
  syncLayerFromStage()
  clearError()
  renderSummary()
  renderArchitecture()
  populateInferenceControls()
  renderInference()
  renderTraining()
}

function renderSummary() {
  const { trace } = state
  const capture = trace.capture ?? {}
  setText('trace-source', state.source)
  setText('model-name', trace.model.name)
  setText('model-description', trace.model.description ?? 'No model description was included in the trace.')
  setText('parameter-count', formatCount(trace.model.estimated_parameters))
  setText('model-shape', `${trace.model.num_layers} layers × ${formatCount(trace.model.hidden_size)} hidden`)
  setText('capture-shape', `${trace.inference.tokens.length} tokens × ${trace.inference.stages[0].activation.cols} bins`)
  setText('bundle-version', `v${trace.version}`)

  const reductions = []
  if (capture.tokens_truncated) reductions.push(`${capture.dropped_leading_tokens} leading tokens omitted`)
  if (capture.channels_reduced) reductions.push(`${capture.original_hidden_channels} channels → ${capture.captured_hidden_channels} bins`)
  if (trace.training?.dropped_rows) reductions.push(`${trace.training.dropped_rows} metric rows sampled out`)
  const headReduction = trace.inference.stages.find((stage) => stage.attention?.captured_heads < stage.attention?.total_heads)
  if (headReduction) reductions.push(`${headReduction.attention.captured_heads}/${headReduction.attention.total_heads} heads per attention layer`)
  setText('trace-reductions', reductions.length ? `Capture: ${reductions.join(' · ')}` : 'Full requested capture retained')
}

function renderArchitecture() {
  const grid = $('layer-grid')
  grid.replaceChildren()
  state.trace.model.layers.forEach((layer, index) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'layer-button'
    button.dataset.mixer = layer.mixer.kind
    button.dataset.layerIndex = String(index)
    button.setAttribute('aria-pressed', String(index === state.selectedLayer))

    const number = document.createElement('span')
    number.className = 'layer-number'
    number.textContent = `LAYER ${layer.index}`
    const mixer = document.createElement('span')
    mixer.className = 'layer-mixer'
    const dot = document.createElement('i')
    dot.className = 'mixer-dot'
    dot.setAttribute('aria-hidden', 'true')
    mixer.append(dot, document.createTextNode(layer.mixer.kind === 'attention' ? 'Attention' : 'Mamba memory'))
    const ffn = document.createElement('span')
    ffn.className = 'layer-ffn'
    ffn.textContent = `${layer.ffn_activation} · ${formatCount(layer.ffn_hidden_size)} FFN`
    button.append(number, mixer, ffn)
    button.addEventListener('click', () => {
      stopPlayback()
      state.selectedLayer = index
      const stageIndex = state.trace.inference.stages.findIndex((stage) => stage.layer_index === layer.index)
      if (stageIndex >= 0) commitStage(stageIndex, { render: false })
      else syncArchitectureSelection()
    })
    grid.append(button)
  })
  renderLayerInspector()
}

function syncLayerFromStage() {
  if (!state.trace) return
  const stage = state.trace.inference.stages[state.selectedStage]
  if (!Number.isInteger(stage?.layer_index)) return
  const layerIndex = state.trace.model.layers.findIndex((layer) => layer.index === stage.layer_index)
  if (layerIndex >= 0) state.selectedLayer = layerIndex
}

function syncArchitectureSelection() {
  document.querySelectorAll('.layer-button').forEach((button) => {
    button.setAttribute('aria-pressed', String(Number(button.dataset.layerIndex) === state.selectedLayer))
  })
  renderLayerInspector()
}

function renderLayerInspector() {
  const layer = state.trace.model.layers[state.selectedLayer]
  const mixer = layer.mixer
  setText('layer-kind', mixer.kind === 'attention' ? 'Global routing' : 'Recurrent memory')
  setText('pattern-position', layer.pattern_index === undefined || layer.pattern_index === null ? 'Homogeneous block' : `Pattern slot ${layer.pattern_index + 1}`)
  setText('layer-title', `Layer ${layer.index} · ${layer.name}`)
  setText('mixer-label', mixer.kind === 'attention' ? 'attention' : 'selective scan')

  const details = mixer.kind === 'attention'
    ? [
        ['Mixer', `${mixer.num_heads} query / ${mixer.num_kv_heads} KV heads`],
        ['Head width', mixer.head_dim],
        ['Position', mixer.position_encoding],
        ['Mask', mixer.window_size ? `causal · window ${mixer.window_size}` : mixer.causal ? 'causal · global' : 'bidirectional'],
        ['QK norm', mixer.qk_norm ? 'enabled' : 'disabled'],
      ]
    : [
        ['Mixer', 'Mamba selective state space'],
        ['Memory state', mixer.state_dim],
        ['Inner width', mixer.inner_size],
        ['Convolution', `causal width ${mixer.conv_kernel}`],
        ['Δ rank', mixer.dt_rank],
      ]
  details.push(
    ['Feed-forward', `${layer.ffn_activation}${layer.ffn_gated ? ' gated' : ''} · ${formatCount(layer.ffn_hidden_size)}`],
    ['Normalization', `${layer.norm_position}-${layer.norm}`],
    ['Residual', layer.residual ? 'enabled' : 'disabled'],
  )
  const list = $('layer-details')
  list.replaceChildren()
  for (const [label, value] of details) {
    const row = document.createElement('div')
    const term = document.createElement('dt')
    const description = document.createElement('dd')
    term.textContent = label
    description.textContent = value
    row.append(term, description)
    list.append(row)
  }

  const stage = state.trace.inference.stages.find((candidate) => candidate.layer_index === layer.index)
  setText('selected-layer-rms', stage ? formatValue(stage.stats.rms) : 'Not captured')
  setText('selected-layer-max', stage ? formatValue(stage.stats.abs_max) : 'Not captured')
}

function populateInferenceControls() {
  const stageSelect = $('stage-select')
  stageSelect.replaceChildren()
  state.trace.inference.stages.forEach((stage, index) => {
    const option = document.createElement('option')
    option.value = String(index)
    option.textContent = stage.label
    stageSelect.append(option)
  })
  stageSelect.value = String(state.selectedStage)

  const tokenSelect = $('token-select')
  tokenSelect.replaceChildren()
  state.trace.inference.tokens.forEach((token, index) => {
    const option = document.createElement('option')
    option.value = String(index)
    option.textContent = `${token.original_index}: ${token.display}`
    tokenSelect.append(option)
  })
  tokenSelect.value = String(state.selectedToken)
  renderFlowRail()
}

function renderFlowRail() {
  const rail = $('flow-rail')
  rail.replaceChildren()
  const stages = state.trace.inference.stages
  stages.forEach((stage, index) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'flow-node'
    button.dataset.stageIndex = String(index)
    button.dataset.mixer = stage.mixer ?? (index === 0 || index === stages.length - 1 ? 'endpoint' : 'block')
    button.setAttribute('aria-label', `Stage ${index + 1} of ${stages.length}: ${stage.label}`)

    const marker = document.createElement('span')
    marker.className = 'flow-node-marker'
    marker.setAttribute('aria-hidden', 'true')
    const label = document.createElement('span')
    label.className = 'flow-node-label'
    label.textContent = index === 0 ? 'Emb' : index === stages.length - 1 ? 'Norm' : `L${stage.layer_index ?? index}`
    button.append(marker, label)
    button.addEventListener('click', () => transitionToStage(index))
    rail.append(button)
  })
  updateFlowRail(state.selectedStage)
}

function updateFlowRail(progress = state.selectedStage) {
  const rail = $('flow-rail')
  const current = Math.round(progress)
  rail.dataset.animating = String(state.animating)
  rail.querySelectorAll('.flow-node').forEach((button) => {
    const index = Number(button.dataset.stageIndex)
    button.dataset.state = index === current ? 'current' : index < progress ? 'past' : 'future'
    if (index === current) button.setAttribute('aria-current', 'step')
    else button.removeAttribute('aria-current')
    const incomingFill = Math.max(0, Math.min(1, progress - index + 1))
    const segmentFill = Math.max(0, Math.min(1, progress - index))
    button.style.setProperty('--flow-incoming', `${incomingFill * 100}%`)
    button.style.setProperty('--flow-fill', `${segmentFill * 100}%`)
  })
}

function updatePlaybackControls(progress = state.selectedStage) {
  if (!state.trace) return
  const count = state.trace.inference.stages.length
  const atStart = state.selectedStage === 0
  const atEnd = state.selectedStage === count - 1
  $('flow-prev').disabled = state.animating || atStart
  $('flow-next').disabled = state.animating || atEnd
  $('flow-play').textContent = state.playing ? 'Pause' : 'Play'
  $('flow-play').setAttribute('aria-pressed', String(state.playing))
  $('flow-progress').max = Math.max(1, count - 1)
  $('flow-progress').value = Math.max(0, Math.min(count - 1, progress))
  const position = state.animating && Number.isInteger(state.animationTarget)
    ? `${state.selectedStage + 1} → ${state.animationTarget + 1} / ${count}`
    : `${state.selectedStage + 1} / ${count}`
  setText('flow-position', position)
  updateFlowRail(progress)
  updateSignalFlow(progress)
}

function cancelAnimation() {
  const wasAnimating = state.animating
  if (state.animationFrame !== null) window.cancelAnimationFrame(state.animationFrame)
  state.animationFrame = null
  state.animating = false
  state.animationTarget = null
  $('view-inference').removeAttribute('aria-busy')
  return wasAnimating
}

function stopPlayback({ settle = true } = {}) {
  if (state.playbackTimer !== null) window.clearTimeout(state.playbackTimer)
  state.playbackTimer = null
  state.playing = false
  const wasAnimating = cancelAnimation()
  if (settle && wasAnimating && state.trace) renderInference({ renderTokens: false })
  else updatePlaybackControls()
}

function commitStage(index, { render = true } = {}) {
  const lastStage = state.trace.inference.stages.length - 1
  state.selectedStage = Math.max(0, Math.min(lastStage, index))
  state.selectedHead = 0
  syncLayerFromStage()
  syncArchitectureSelection()
  if (render) renderInference({ renderTokens: false })
  else {
    $('stage-select').value = String(state.selectedStage)
    updatePlaybackControls()
  }
}

function easeInOutCubic(progress) {
  return progress < 0.5
    ? 4 * progress * progress * progress
    : 1 - Math.pow(-2 * progress + 2, 3) / 2
}

function animateAdjacentStage(target, duration, onComplete) {
  const source = state.selectedStage
  const from = state.trace.inference.stages[source]
  const to = state.trace.inference.stages[target]
  if (reducedMotion.matches || duration <= 0) {
    commitStage(target)
    onComplete()
    return
  }

  const scale = interpolateHeatmap(from.activation, to.activation, 0)
  renderScale(scale, $('activation-scale'))
  setText('activation-title', `${from.label} → ${to.label}`)
  state.animating = true
  state.animationTarget = target
  $('view-inference').setAttribute('aria-busy', 'true')
  updatePlaybackControls(source)
  const startedAt = window.performance.now()

  const frame = (now) => {
    const linear = Math.min(1, (now - startedAt) / duration)
    const eased = easeInOutCubic(linear)
    const progress = source + (target - source) * eased
    const heatmap = interpolateHeatmap(from.activation, to.activation, eased)
    renderHeatmap($('activation-heatmap'), heatmap, {
      divergent: true,
      minHeight: 320,
      label: `${from.label} to ${to.label} residual-stream transition`,
    })
    updatePlaybackControls(progress)
    if (linear < 1) {
      state.animationFrame = window.requestAnimationFrame(frame)
      return
    }

    state.animationFrame = null
    state.animating = false
    state.animationTarget = null
    $('view-inference').removeAttribute('aria-busy')
    commitStage(target)
    onComplete()
  }
  state.animationFrame = window.requestAnimationFrame(frame)
}

function runStageSequence(target, duration, onComplete) {
  if (state.selectedStage === target) {
    onComplete?.()
    return
  }
  const next = state.selectedStage + Math.sign(target - state.selectedStage)
  animateAdjacentStage(next, duration, () => runStageSequence(target, duration, onComplete))
}

function transitionToStage(index, { pause = true, onComplete = null } = {}) {
  if (pause) stopPlayback()
  const lastStage = state.trace.inference.stages.length - 1
  const target = Math.max(0, Math.min(lastStage, index))
  if (target === state.selectedStage) {
    updatePlaybackControls()
    onComplete?.()
    return
  }
  const distance = Math.abs(target - state.selectedStage)
  const duration = distance === 1
    ? Math.max(220, Math.min(650, state.playbackDelay * 0.72))
    : Math.max(70, Math.min(180, 1000 / distance))
  runStageSequence(target, duration, onComplete)
}

function schedulePlayback() {
  if (!state.playing || state.animating) return
  const delay = reducedMotion.matches ? state.playbackDelay : Math.max(70, state.playbackDelay * 0.18)
  state.playbackTimer = window.setTimeout(() => {
    state.playbackTimer = null
    if (state.selectedStage >= state.trace.inference.stages.length - 1) {
      stopPlayback()
      return
    }
    transitionToStage(state.selectedStage + 1, { pause: false, onComplete: schedulePlayback })
  }, delay)
}

function startPlayback() {
  if (state.trace.inference.stages.length < 2) return
  stopPlayback()
  if (state.selectedStage >= state.trace.inference.stages.length - 1) commitStage(0)
  state.playing = true
  updatePlaybackControls()
  schedulePlayback()
}

function renderInference({ renderTokens = true, renderSignal = renderTokens } = {}) {
  const { inference } = state.trace
  const stage = inference.stages[state.selectedStage]
  $('stage-select').value = String(state.selectedStage)
  $('token-select').value = String(state.selectedToken)
  setText('activation-title', stage.label)
  renderScale(stage.activation, $('activation-scale'))
  renderHeatmap($('activation-heatmap'), stage.activation, { divergent: true, minHeight: 320, label: `${stage.label} residual-stream activation` })
  if (renderTokens) renderTokenStrip()
  if (renderSignal) renderSignalFlowChart()
  renderMixerTrace(stage)
  setText('generated-text', inference.generated_text || 'The generation stopped without a decoded continuation.')
  setText('prompt-token-count', inference.prompt_token_count)
  setText('generated-token-count', inference.generated_token_count)
  const sampling = inference.sampling
  const method = sampling.temperature <= 0 ? 'greedy' : `T ${sampling.temperature}${sampling.top_k ? ` · top-${sampling.top_k}` : ''}`
  setText('sampling-description', `${method} · seed ${sampling.seed}`)
  updatePlaybackControls()
}

function renderTokenStrip() {
  const strip = $('token-strip')
  strip.replaceChildren()
  const tokens = state.trace.inference.tokens
  const start = Math.max(0, Math.min(tokens.length - 11, state.selectedToken - 5))
  const end = Math.min(tokens.length, start + 11)
  for (let index = start; index < end; index += 1) {
    const token = tokens[index]
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'token-chip'
    button.dataset.source = token.source
    button.setAttribute('aria-pressed', String(index === state.selectedToken))
    button.textContent = token.display
    button.setAttribute('aria-label', `Token ${token.original_index}, ${token.display}, ${token.source}`)
    button.addEventListener('click', () => {
      stopPlayback()
      state.selectedToken = index
      renderInference()
    })
    strip.append(button)
  }
}

function renderMixerTrace(stage) {
  const attentionPanel = $('attention-panel')
  const mambaPanel = $('mamba-panel')
  attentionPanel.hidden = !stage.attention
  mambaPanel.hidden = !stage.mamba_state
  if (stage.attention) {
    const select = $('head-select')
    select.replaceChildren()
    stage.attention.heads.forEach((_, index) => {
      const option = document.createElement('option')
      option.value = String(index)
      option.textContent = `${index + 1} of ${stage.attention.total_heads}`
      select.append(option)
    })
    state.selectedHead = Math.min(state.selectedHead, stage.attention.heads.length - 1)
    select.value = String(state.selectedHead)
    renderHeatmap($('attention-heatmap'), stage.attention.heads[state.selectedHead], { sequential: true, minHeight: 420, label: `${stage.label} attention head ${state.selectedHead + 1}` })
  }
  if (stage.mamba_state) {
    renderHeatmap($('mamba-heatmap'), stage.mamba_state, { divergent: true, minHeight: 320, label: `${stage.label} final Mamba recurrent state` })
  }
}

function renderTraining() {
  const training = state.trace.training
  $('training-empty').hidden = Boolean(training)
  $('training-content').hidden = !training
  if (!training) return
  const definitions = availableMetrics(training.rows)
  if (definitions.length === 0) {
    $('training-content').hidden = true
    $('training-empty').hidden = false
    return
  }
  if (!definitions.some((definition) => definition.key === state.selectedMetric)) state.selectedMetric = definitions[0].key

  const metricSelect = $('metric-select')
  metricSelect.replaceChildren()
  definitions.forEach((definition) => {
    const option = document.createElement('option')
    option.value = definition.key
    option.textContent = definition.label
    metricSelect.append(option)
  })
  metricSelect.value = state.selectedMetric
  const definition = definitions.find((candidate) => candidate.key === state.selectedMetric)
  const series = metricSeries(training.rows, state.selectedMetric)
  setText('training-source', `${training.captured_rows}/${training.total_rows} rows · ${training.source}`)
  setText('metric-title', definition.label)
  setText('metric-latest', formatValue(series.at(-1)?.value, definition) + (definition.unit && !definition.percent ? definition.unit : ''))
  renderLineChart($('training-chart'), series, { xLabel: 'optimizer step', yLabel: definition.label, stages: true })

  const metricMap = normalizedMetricHeatmap(training.rows, definitions)
  renderHeatmap($('metric-heatmap'), metricMap, { sequential: true, minHeight: 300, label: 'Normalized training metrics by optimizer step' })
  const gradient = layerGradientHeatmap(training.rows, state.trace.model.num_layers)
  $('layer-gradient-panel').hidden = !gradient
  if (gradient) {
    renderHeatmap($('gradient-heatmap'), gradient.heatmap, { sequential: true, minHeight: 300, label: 'Per-layer gradient norms by optimizer step' })
    setText('gradient-note', `${gradient.samples} sampled steps include layer gradients. Enable with --layer-metrics-every N.`)
  }
}

function renderScale(heatmap, target) {
  target.replaceChildren()
  const min = document.createElement('span')
  min.textContent = formatValue(heatmap.min)
  const gradient = document.createElement('i')
  gradient.className = 'scale-gradient'
  gradient.setAttribute('aria-hidden', 'true')
  const max = document.createElement('span')
  max.textContent = formatValue(heatmap.max)
  target.append(min, gradient, max)
}

function parseColor(value) {
  const hex = value.match(/^#([\da-f]{6})$/i)
  if (hex) return [0, 2, 4].map((offset) => Number.parseInt(hex[1].slice(offset, offset + 2), 16))
  const rgb = value.match(/[\d.]+/g)
  return rgb ? rgb.slice(0, 3).map(Number) : [128, 128, 128]
}

function mixColor(left, right, amount) {
  const t = Math.max(0, Math.min(1, amount))
  return `rgb(${left.map((value, index) => Math.round(value + (right[index] - value) * t)).join(' ')})`
}

function heatPalette() {
  return {
    negative: parseColor(css('--heat-negative')),
    zero: parseColor(css('--heat-zero')),
    positive: parseColor(css('--heat-positive')),
    sequentialLow: parseColor(css('--heat-sequential-low')),
    sequentialHigh: parseColor(css('--heat-sequential-high')),
  }
}

function heatColor(value, heatmap, options, palette) {
  if (options.sequential) {
    const range = heatmap.max - heatmap.min
    const t = range === 0 ? 0.5 : (value - heatmap.min) / range
    return mixColor(palette.sequentialLow, palette.sequentialHigh, t)
  }
  const maximum = Math.max(Math.abs(heatmap.min), Math.abs(heatmap.max), Number.EPSILON)
  const t = Math.min(1, Math.abs(value) / maximum)
  return value < 0
    ? mixColor(palette.zero, palette.negative, t)
    : mixColor(palette.zero, palette.positive, t)
}

function renderHeatmap(canvas, heatmap, options = {}) {
  const wrap = canvas.parentElement
  const cssWidth = Math.max(280, wrap.clientWidth)
  const cssHeight = Math.max(options.minHeight ?? 280, Math.min(520, heatmap.rows * 18 + 82))
  const ratio = Math.min(window.devicePixelRatio || 1, 2)
  const pixelWidth = Math.round(cssWidth * ratio)
  const pixelHeight = Math.round(cssHeight * ratio)
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight
  if (canvas.style.height !== `${cssHeight}px`) canvas.style.height = `${cssHeight}px`
  canvas.setAttribute('role', 'img')
  canvas.setAttribute('aria-label', options.label ?? `${heatmap.rows} by ${heatmap.cols} heatmap`)
  const context = canvas.getContext('2d')
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, cssWidth, cssHeight)

  const margins = { left: Math.min(98, Math.max(54, cssWidth * 0.15)), right: 12, top: 10, bottom: 48 }
  const plotWidth = cssWidth - margins.left - margins.right
  const plotHeight = cssHeight - margins.top - margins.bottom
  const cellWidth = plotWidth / heatmap.cols
  const cellHeight = plotHeight / heatmap.rows
  const palette = heatPalette()
  for (let row = 0; row < heatmap.rows; row += 1) {
    for (let column = 0; column < heatmap.cols; column += 1) {
      const value = heatmap.values[row * heatmap.cols + column]
      context.fillStyle = heatColor(value, heatmap, options, palette)
      context.fillRect(margins.left + column * cellWidth, margins.top + row * cellHeight, Math.ceil(cellWidth + 0.3), Math.ceil(cellHeight + 0.3))
    }
  }
  context.strokeStyle = css('--line-strong')
  context.lineWidth = 1
  context.strokeRect(margins.left, margins.top, plotWidth, plotHeight)
  context.fillStyle = css('--muted')
  context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace'
  context.textBaseline = 'middle'
  const rowStep = Math.max(1, Math.ceil(heatmap.rows / 8))
  for (let row = 0; row < heatmap.rows; row += rowStep) {
    const label = String(heatmap.row_labels[row]).slice(0, 13)
    context.textAlign = 'right'
    context.fillText(label, margins.left - 7, margins.top + (row + 0.5) * cellHeight)
  }
  const columnStep = Math.max(1, Math.ceil(heatmap.cols / 7))
  context.textAlign = 'center'
  context.textBaseline = 'top'
  for (let column = 0; column < heatmap.cols; column += columnStep) {
    const label = String(heatmap.col_labels[column]).slice(0, 12)
    context.save()
    context.translate(margins.left + (column + 0.5) * cellWidth, margins.top + plotHeight + 6)
    context.rotate(-0.45)
    context.fillText(label, 0, 0)
    context.restore()
  }
  canvas.__heatmap = { heatmap, margins, plotWidth, plotHeight, cssWidth, cssHeight }
  attachHeatmapTooltip(canvas)
}

function attachHeatmapTooltip(canvas) {
  if (canvas.__tooltipAttached) return
  canvas.__tooltipAttached = true
  const tooltip = $('heatmap-tooltip')
  canvas.addEventListener('pointermove', (event) => {
    const rendered = canvas.__heatmap
    const bounds = canvas.getBoundingClientRect()
    const x = event.clientX - bounds.left
    const y = event.clientY - bounds.top
    const column = Math.floor((x - rendered.margins.left) / rendered.plotWidth * rendered.heatmap.cols)
    const row = Math.floor((y - rendered.margins.top) / rendered.plotHeight * rendered.heatmap.rows)
    if (row < 0 || row >= rendered.heatmap.rows || column < 0 || column >= rendered.heatmap.cols) {
      tooltip.hidden = true
      return
    }
    const value = rendered.heatmap.values[row * rendered.heatmap.cols + column]
    tooltip.textContent = `${rendered.heatmap.row_labels[row]} × ${rendered.heatmap.col_labels[column]} · ${formatValue(value)}`
    tooltip.hidden = false
    const width = tooltip.offsetWidth
    const height = tooltip.offsetHeight
    tooltip.style.left = `${Math.min(window.innerWidth - width - 8, event.clientX + 13)}px`
    tooltip.style.top = `${Math.min(window.innerHeight - height - 8, event.clientY + 13)}px`
  })
  canvas.addEventListener('pointerleave', () => { tooltip.hidden = true })
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name)
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value))
  return element
}

function renderSignalFlowChart() {
  const svg = $('signal-flow-chart')
  const series = tokenSignalSeries(state.trace.inference, state.selectedToken)
  svg.replaceChildren()
  if (series.length < 2) return

  const measuredWidth = svg.getBoundingClientRect().width
  const width = Math.max(300, Math.round(measuredWidth || 960))
  const height = 330
  const margin = { left: width < 480 ? 42 : 54, right: 18, top: 42 }
  const rmsBottom = 190
  const signBaseline = 242
  const labelY = 286
  const plotWidth = width - margin.left - margin.right
  const positiveRms = series.map((point) => point.rms).filter((value) => value > 0)
  const rmsMin = Math.max(Math.min(...positiveRms, 1) * 0.82, Number.EPSILON)
  const rmsMax = Math.max(...series.map((point) => point.rms), rmsMin) * 1.08
  const logMin = Math.log10(rmsMin)
  const logMax = Math.max(Math.log10(rmsMax), logMin + 0.1)
  const updateMax = Math.max(...series.map((point) => point.updateRms ?? 0), Number.EPSILON)
  const x = (stageIndex) => margin.left + stageIndex / (series.length - 1) * plotWidth
  const y = (rms) => {
    const normalized = (Math.log10(Math.max(rms, rmsMin)) - logMin) / (logMax - logMin)
    return rmsBottom - normalized * (rmsBottom - margin.top)
  }

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`)
  const title = svgElement('title')
  title.textContent = 'Selected-token signal flow across model depth'
  const description = svgElement('desc')
  description.textContent = 'Log-scaled vertical position reports exact token RMS, edge width reports residual update RMS, node shape is mixer type, and diverging bars approximate positive and negative energy in captured channel bins.'
  svg.append(title, description)

  const encoding = svgElement('text', {
    x: margin.left,
    y: 17,
    class: 'signal-flow-encoding',
  })
  encoding.textContent = width < 590
    ? 'RMS(log) line · Δ edge · +/− bins'
    : 'height: token RMS (log) · edge width: residual update · bars: +/− captured-bin energy'
  svg.append(encoding)
  if (width >= 700) {
    const mixers = svgElement('text', {
      x: width - margin.right,
      y: 17,
      class: 'signal-flow-encoding',
      'text-anchor': 'end',
    })
    mixers.textContent = '○ attention · ◇ Mamba'
    svg.append(mixers)
  }

  const rmsTicks = [rmsMin, Math.sqrt(rmsMin * rmsMax), rmsMax]
  for (const value of rmsTicks) {
    const position = y(value)
    svg.append(svgElement('line', {
      x1: margin.left,
      x2: width - margin.right,
      y1: position,
      y2: position,
      class: 'signal-flow-grid',
    }))
    const label = svgElement('text', {
      x: margin.left - 8,
      y: position + 4,
      class: 'signal-flow-axis-label',
      'text-anchor': 'end',
    })
    label.textContent = formatValue(value)
    svg.append(label)
  }

  const yLabel = svgElement('text', {
    x: 13,
    y: (margin.top + rmsBottom) / 2,
    class: 'signal-flow-axis-label',
    transform: `rotate(-90 13 ${(margin.top + rmsBottom) / 2})`,
    'text-anchor': 'middle',
  })
  yLabel.textContent = 'token RMS · log'
  const signLine = svgElement('line', {
    x1: margin.left,
    x2: width - margin.right,
    y1: signBaseline,
    y2: signBaseline,
    class: 'signal-flow-sign-axis',
  })
  const signLabel = svgElement('text', {
    x: margin.left - 8,
    y: signBaseline + 4,
    class: 'signal-flow-axis-label',
    'text-anchor': 'end',
  })
  signLabel.textContent = '+/−'
  const xLabel = svgElement('text', {
    x: margin.left + plotWidth / 2,
    y: height - 5,
    class: 'signal-flow-axis-label',
    'text-anchor': 'middle',
  })
  xLabel.textContent = 'embedding → transformer depth → final norm'
  svg.append(yLabel, signLine, signLabel, xLabel)

  const edges = []
  for (let index = 1; index < series.length; index += 1) {
    const from = series[index - 1]
    const to = series[index]
    const left = x(from.stageIndex)
    const right = x(to.stageIndex)
    const middle = (left + right) / 2
    const update = to.updateRms ?? 0
    const edge = svgElement('path', {
      d: `M ${left.toFixed(2)} ${y(from.rms).toFixed(2)} C ${middle.toFixed(2)} ${y(from.rms).toFixed(2)}, ${middle.toFixed(2)} ${y(to.rms).toFixed(2)}, ${right.toFixed(2)} ${y(to.rms).toFixed(2)}`,
      class: 'signal-flow-edge',
      'stroke-width': 1.4 + 7 * Math.log1p(update) / Math.log1p(updateMax),
    })
    const edgeTitle = svgElement('title')
    edgeTitle.textContent = `${from.label} → ${to.label}: residual update RMS ${formatValue(to.updateRms)}`
    edge.append(edgeTitle)
    svg.append(edge)
    edges.push({ stageIndex: index, element: edge })
  }

  const probeGuide = svgElement('line', {
    y1: margin.top,
    y2: signBaseline + 31,
    class: 'signal-flow-probe-guide',
  })
  svg.append(probeGuide)

  const glyphTarget = Math.max(8, Math.floor(width / 34))
  const glyphStride = Math.max(1, Math.ceil(series.length / glyphTarget))
  const labelTarget = Math.max(4, Math.floor(width / 72))
  const labelStride = Math.max(1, Math.ceil(series.length / labelTarget))
  const nodes = []
  series.forEach((point, index) => {
    const showGlyph = index === 0 || index === series.length - 1 || index % glyphStride === 0
    const showLabel = index === 0 || index === series.length - 1 || index % labelStride === 0
    if (showGlyph) {
      const group = svgElement('g', {
        class: 'signal-flow-stage',
        'data-stage-index': index,
        'data-mixer': point.mixer ?? 'endpoint',
      })
      const stageTitle = svgElement('title')
      stageTitle.textContent = `${point.label}: RMS ${formatValue(point.rms)}, update ${formatValue(point.updateRms)}, positive-bin energy ${(point.positiveBinEnergy * 100).toFixed(1)}%`
      group.append(stageTitle)

      const positiveHeight = point.positiveBinEnergy * 30
      const negativeHeight = point.negativeBinEnergy * 30
      group.append(
        svgElement('rect', {
          x: x(index) - 3,
          y: signBaseline - positiveHeight,
          width: 6,
          height: positiveHeight,
          class: 'signal-flow-positive',
        }),
        svgElement('rect', {
          x: x(index) - 3,
          y: signBaseline,
          width: 6,
          height: negativeHeight,
          class: 'signal-flow-negative',
        }),
      )

      const yPosition = y(point.rms)
      let shape
      if (point.mixer === 'attention') {
        shape = svgElement('circle', { cx: x(index), cy: yPosition, r: 5.5, class: 'signal-flow-node-shape' })
      } else if (point.mixer === 'mamba') {
        shape = svgElement('polygon', {
          points: `${x(index)},${yPosition - 6} ${x(index) + 6},${yPosition} ${x(index)},${yPosition + 6} ${x(index) - 6},${yPosition}`,
          class: 'signal-flow-node-shape',
        })
      } else {
        shape = svgElement('rect', { x: x(index) - 7, y: yPosition - 5, width: 14, height: 10, rx: 3, class: 'signal-flow-node-shape' })
      }
      group.append(shape)
      svg.append(group)
      nodes.push({ stageIndex: index, element: group })
    }

    if (showLabel) {
      const label = svgElement('text', {
        x: x(index),
        y: labelY,
        class: 'signal-flow-stage-label',
        'text-anchor': 'middle',
      })
      label.textContent = index === 0 ? 'Emb' : index === series.length - 1 ? 'Norm' : `L${point.layerIndex ?? index}`
      svg.append(label)
    }
  })

  const probeHalo = svgElement('circle', { r: 10, class: 'signal-flow-probe-halo' })
  const probe = svgElement('circle', { r: 4.5, class: 'signal-flow-probe' })
  const probeValue = svgElement('text', { class: 'signal-flow-probe-value' })
  svg.append(probeHalo, probe, probeValue)
  svg.__signalFlow = { series, width, x, y, probe, probeHalo, probeGuide, probeValue, nodes, edges }

  const hiddenChannels = state.trace.capture?.original_hidden_channels ?? state.trace.model.hidden_size
  setText('signal-flow-compression', `${formatCount(hiddenChannels)} hidden values → ${series[0].capturedBins} signed bins · RMS/Δ exact`)
  updateSignalFlow(state.selectedStage)
}

function updateSignalFlow(progress = state.selectedStage) {
  const svg = $('signal-flow-chart')
  const rendered = svg.__signalFlow
  if (!rendered) return
  const last = rendered.series.length - 1
  const bounded = Math.max(0, Math.min(last, progress))
  const leftIndex = Math.floor(bounded)
  const rightIndex = Math.ceil(bounded)
  const amount = bounded - leftIndex
  const left = rendered.series[leftIndex]
  const right = rendered.series[rightIndex]
  const interpolate = (from, to) => from + (to - from) * amount
  const rms = interpolate(left.rms, right.rms)
  const positive = interpolate(left.positiveBinEnergy, right.positiveBinEnergy)
  const negative = interpolate(left.negativeBinEnergy, right.negativeBinEnergy)
  const binMean = interpolate(left.binMean, right.binMean)
  const xPosition = rendered.x(bounded)
  const yPosition = rendered.y(rms)
  for (const element of [rendered.probe, rendered.probeHalo]) {
    element.setAttribute('cx', xPosition)
    element.setAttribute('cy', yPosition)
  }
  rendered.probeGuide.setAttribute('x1', xPosition)
  rendered.probeGuide.setAttribute('x2', xPosition)
  const labelOnLeft = xPosition > rendered.width - 92
  rendered.probeValue.setAttribute('x', xPosition + (labelOnLeft ? -10 : 10))
  rendered.probeValue.setAttribute('y', Math.max(34, yPosition - 11))
  rendered.probeValue.setAttribute('text-anchor', labelOnLeft ? 'end' : 'start')
  rendered.probeValue.textContent = `RMS ${formatValue(rms)}`

  const current = Math.round(bounded)
  rendered.nodes.forEach((node) => {
    node.element.dataset.state = node.stageIndex === current ? 'current' : node.stageIndex < bounded ? 'past' : 'future'
  })
  const activeEdge = Number.isInteger(bounded) ? -1 : Math.ceil(bounded)
  rendered.edges.forEach((edge) => {
    edge.element.dataset.state = edge.stageIndex === activeEdge ? 'current' : edge.stageIndex <= bounded ? 'past' : 'future'
  })

  const token = state.trace.inference.tokens[state.selectedToken]
  const stageLabel = leftIndex === rightIndex ? left.label : `${left.label} → ${right.label}`
  const update = leftIndex === rightIndex
    ? left.updateRms
    : (right.updateRms ?? 0) * amount
  const signedMean = `${binMean > 0 ? '+' : ''}${formatValue(binMean)}`
  setText(
    'signal-flow-description',
    `Token ${token.original_index} “${token.display}” · ${stageLabel} · RMS ${formatValue(rms)} · ${update === null ? 'input embedding' : `Δ ${formatValue(update)}`} · bins +${(positive * 100).toFixed(0)}% / −${(negative * 100).toFixed(0)}% · mean ${signedMean}`,
  )
}

function renderLineChart(svg, series, options = {}) {
  svg.replaceChildren()
  if (series.length === 0) return
  const width = 760
  const height = options.compact ? 300 : 340
  const margin = { left: 62, right: 22, top: 28, bottom: 46 }
  const plotWidth = width - margin.left - margin.right
  const plotHeight = height - margin.top - margin.bottom
  const xMin = series[0].step
  const xMax = series.at(-1).step
  let yMin = Math.min(...series.map((point) => point.value))
  let yMax = Math.max(...series.map((point) => point.value))
  const padding = Math.max((yMax - yMin) * 0.12, Math.abs(yMax || 1) * 0.03)
  yMin -= padding
  yMax += padding
  const x = (value) => margin.left + (xMax === xMin ? plotWidth / 2 : (value - xMin) / (xMax - xMin) * plotWidth)
  const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * plotHeight

  const title = svgElement('title')
  title.textContent = `${options.yLabel ?? 'Metric'} from ${formatValue(series[0].value)} to ${formatValue(series.at(-1).value)}`
  svg.append(title)
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`)

  if (options.stages) {
    let start = 0
    while (start < series.length) {
      let end = start
      while (end + 1 < series.length && series[end + 1].stage === series[start].stage) end += 1
      const left = start === 0 ? margin.left : (x(series[start - 1].step) + x(series[start].step)) / 2
      const right = end === series.length - 1 ? margin.left + plotWidth : (x(series[end].step) + x(series[end + 1].step)) / 2
      const rect = svgElement('rect', { x: left, y: margin.top, width: Math.max(1, right - left), height: plotHeight, fill: start % 2 === 0 ? css('--accent-soft') : css('--surface-raised'), opacity: 0.42 })
      const label = svgElement('text', { x: left + 6, y: margin.top + 14, fill: css('--muted'), 'font-size': 10 })
      label.textContent = series[start].stage
      svg.append(rect, label)
      start = end + 1
    }
  }

  for (let index = 0; index <= 4; index += 1) {
    const value = yMin + (yMax - yMin) * index / 4
    const yPosition = y(value)
    svg.append(svgElement('line', { x1: margin.left, x2: width - margin.right, y1: yPosition, y2: yPosition, stroke: css('--grid'), 'stroke-width': 1 }))
    const label = svgElement('text', { x: margin.left - 9, y: yPosition + 4, fill: css('--muted'), 'font-size': 11, 'text-anchor': 'end' })
    label.textContent = formatValue(value)
    svg.append(label)
  }
  for (let index = 0; index <= 4; index += 1) {
    const value = xMin + (xMax - xMin) * index / 4
    const label = svgElement('text', { x: x(value), y: height - 18, fill: css('--muted'), 'font-size': 11, 'text-anchor': 'middle' })
    label.textContent = Math.round(value).toLocaleString()
    svg.append(label)
  }
  const path = svgElement('path', {
    d: series.map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(point.step).toFixed(2)} ${y(point.value).toFixed(2)}`).join(' '),
    fill: 'none',
    stroke: css('--chart-1'),
    'stroke-width': 2.5,
    'stroke-linejoin': 'round',
    'stroke-linecap': 'round',
  })
  svg.append(path)
  const pointStride = Math.max(1, Math.ceil(series.length / 40))
  series.forEach((point, index) => {
    if (index % pointStride !== 0 && index !== series.length - 1 && index !== options.selectedIndex) return
    svg.append(svgElement('circle', {
      cx: x(point.step), cy: y(point.value), r: index === options.selectedIndex ? 5 : 2.7,
      fill: index === options.selectedIndex ? css('--chart-2') : css('--chart-1'),
      stroke: css('--surface'), 'stroke-width': 1.5,
    }))
  })
  const yLabel = svgElement('text', { x: 14, y: margin.top + plotHeight / 2, fill: css('--muted'), 'font-size': 11, transform: `rotate(-90 14 ${margin.top + plotHeight / 2})`, 'text-anchor': 'middle' })
  yLabel.textContent = options.yLabel ?? ''
  const xLabel = svgElement('text', { x: margin.left + plotWidth / 2, y: height - 2, fill: css('--muted'), 'font-size': 11, 'text-anchor': 'middle' })
  xLabel.textContent = options.xLabel ?? ''
  svg.append(yLabel, xLabel)
}

function switchView(viewName) {
  state.currentView = viewName
  document.querySelectorAll('.tabs button').forEach((button) => {
    button.setAttribute('aria-selected', String(button.dataset.view === viewName))
  })
  document.querySelectorAll('.view').forEach((view) => {
    view.hidden = view.id !== `view-${viewName}`
  })
  if (viewName === 'inference') renderInference()
  if (viewName === 'training') renderTraining()
}

function setLiveSessionStatus(message, status) {
  const target = $('live-session-status')
  target.textContent = message
  target.dataset.state = status
}

async function readJsonResponse(response) {
  const text = await response.text()
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`Model Lab returned ${response.status} without a JSON response`)
  }
}

async function checkLiveSession() {
  try {
    const response = await window.fetch('/api/status', { headers: { Accept: 'application/json' } })
    const status = await readJsonResponse(response)
    if (!response.ok) throw new Error(status.error ?? `Model Lab returned ${response.status}`)
    if (status.status !== 'ready' || !Number.isInteger(status.max_new_tokens) || status.max_new_tokens < 1) {
      throw new Error('Model Lab status response is malformed')
    }
    state.liveReady = true
    state.liveStatus = status
    $('live-max-tokens').max = String(status.max_new_tokens)
    $('live-max-tokens').value = String(Math.min(Number($('live-max-tokens').value), status.max_new_tokens))
    if (Number.isInteger(status.max_prompt_bytes)) $('live-prompt').maxLength = status.max_prompt_bytes
    $('run-query').disabled = false
    setLiveSessionStatus(`Ready · ${status.model} · ${status.num_layers} layers`, 'ready')
  } catch {
    state.liveReady = false
    state.liveStatus = null
    $('run-query').disabled = true
    setLiveSessionStatus('Static mode · start hermes-llm lab for live queries', 'static')
  }
}

async function runLiveQuery(event) {
  event.preventDefault()
  if (!state.liveReady || state.liveRunning) return
  const form = $('live-query-form')
  if (!form.reportValidity()) return

  const prompt = $('live-prompt').value.trim()
  const maxTokens = Number($('live-max-tokens').value)
  const temperature = Number($('live-temperature').value)
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > state.liveStatus.max_new_tokens) {
    showError(`New tokens must be between 1 and ${state.liveStatus.max_new_tokens}.`)
    return
  }
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 5) {
    showError('Temperature must be between 0 and 5.')
    return
  }

  stopPlayback()
  clearError()
  state.liveRunning = true
  form.setAttribute('aria-busy', 'true')
  const button = $('run-query')
  button.disabled = true
  button.textContent = 'Running…'
  const startedAt = Date.now()
  const updateElapsed = () => {
    const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
    setLiveSessionStatus(`Running inference · ${seconds}s`, 'running')
  }
  updateElapsed()
  const elapsedTimer = window.setInterval(updateElapsed, 1000)

  try {
    const response = await window.fetch('/api/trace', {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature }),
    })
    const trace = await readJsonResponse(response)
    if (!response.ok) throw new Error(trace.error ?? `Model Lab returned ${response.status}`)
    setTrace(trace, 'Live query', { initialStage: 0, selectedToken: trace.inference.tokens.length - 1 })
    switchView('inference')
    const seconds = ((Date.now() - startedAt) / 1000).toFixed(1)
    setLiveSessionStatus(`Ready · ${trace.inference.generated_token_count} tokens in ${seconds}s`, 'ready')
    startPlayback()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    showError(message)
    setLiveSessionStatus('Request failed · model remains loaded', 'error')
  } finally {
    window.clearInterval(elapsedTimer)
    state.liveRunning = false
    form.removeAttribute('aria-busy')
    button.disabled = !state.liveReady
    button.textContent = 'Run & play'
  }
}

$('trace-file').addEventListener('change', async (event) => {
  const file = event.target.files?.[0]
  if (!file) return
  try {
    if (file.size > MAX_BUNDLE_BYTES) throw new Error(`Trace is ${formatCount(file.size)}B; the browser limit is ${formatCount(MAX_BUNDLE_BYTES)}B`)
    const trace = parseTraceJson(await file.text())
    setTrace(trace, file.name)
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error))
  } finally {
    event.target.value = ''
  }
})

$('load-demo').addEventListener('click', () => setTrace(demoTrace, 'Synthetic example'))
$('stage-select').addEventListener('change', (event) => {
  transitionToStage(Number(event.target.value))
})
$('token-select').addEventListener('change', (event) => {
  stopPlayback()
  state.selectedToken = Number(event.target.value)
  renderInference()
})
$('head-select').addEventListener('change', (event) => {
  stopPlayback()
  state.selectedHead = Number(event.target.value)
  renderMixerTrace(state.trace.inference.stages[state.selectedStage])
})
$('metric-select').addEventListener('change', (event) => {
  state.selectedMetric = event.target.value
  renderTraining()
})
$('live-query-form').addEventListener('submit', runLiveQuery)
$('live-prompt').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
    event.preventDefault()
    $('live-query-form').requestSubmit()
  }
})
$('flow-prev').addEventListener('click', () => transitionToStage(state.selectedStage - 1))
$('flow-next').addEventListener('click', () => transitionToStage(state.selectedStage + 1))
$('flow-play').addEventListener('click', () => {
  if (state.playing) stopPlayback()
  else startPlayback()
})
$('flow-speed').addEventListener('change', (event) => {
  state.playbackDelay = Number(event.target.value)
  if (state.playing && !state.animating) {
    if (state.playbackTimer !== null) window.clearTimeout(state.playbackTimer)
    schedulePlayback()
  }
})

document.querySelectorAll('.tabs button').forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.view))
})

let resizeQueued = false
window.addEventListener('resize', () => {
  if (resizeQueued) return
  resizeQueued = true
  window.requestAnimationFrame(() => {
    resizeQueued = false
    const visibleView = document.querySelector('.view:not([hidden])')?.id
    if (visibleView === 'view-inference' && !state.animating) {
      renderInference({ renderTokens: false, renderSignal: true })
    }
    if (visibleView === 'view-training') renderTraining()
  })
})

setTrace(demoTrace, 'Synthetic example')
checkLiveSession()
