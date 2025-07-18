import assert from 'node:assert/strict'
import test from 'node:test'

import { demoTrace } from './demo-trace.js'
import { interpolateHeatmap, layerGradientHeatmap, normalizedMetricHeatmap, tokenSignalSeries, validateTrace } from './trace-utils.js'

const clone = (value) => JSON.parse(JSON.stringify(value))

test('demo trace satisfies the versioned bundle contract', () => {
  assert.equal(validateTrace(clone(demoTrace)).version, 1)
})

test('reader rejects a trace newer than its supported contract', () => {
  const trace = clone(demoTrace)
  trace.version = 2
  assert.throws(() => validateTrace(trace), /newer than this lab supports/)
})

test('reader rejects heatmaps whose dense shape does not match values', () => {
  const trace = clone(demoTrace)
  trace.inference.stages[0].activation.values.pop()
  assert.throws(() => validateTrace(trace), /entries; expected/)
})

test('heatmap interpolation moves values while keeping a stable endpoint scale', () => {
  const from = { rows: 1, cols: 2, values: [-2, 2], min: -2, max: 2 }
  const to = { rows: 1, cols: 2, values: [4, -4], min: -4, max: 4 }
  const middle = interpolateHeatmap(from, to, 0.5)

  assert.deepEqual(middle.values, [1, -1])
  assert.equal(middle.min, -4)
  assert.equal(middle.max, 4)
  assert.deepEqual(interpolateHeatmap(from, to, -1).values, from.values)
  assert.deepEqual(interpolateHeatmap(from, to, 2).values, to.values)
})

test('heatmap interpolation rejects incompatible captures', () => {
  const from = { rows: 1, cols: 1, values: [0], min: 0, max: 0 }
  const to = { rows: 2, cols: 1, values: [0, 1], min: 0, max: 1 }
  assert.throws(() => interpolateHeatmap(from, to, 0.5), /different shapes/)
})

test('selected-token signal flow preserves exact RMS and summarizes captured bins', () => {
  const series = tokenSignalSeries(demoTrace.inference, 0)
  assert.equal(series.length, demoTrace.inference.stages.length)
  assert.equal(series[0].rms, demoTrace.inference.stages[0].token_rms[0])
  assert.equal(series[0].updateRms, null)
  assert.equal(series[0].capturedBins, demoTrace.inference.stages[0].activation.cols)
  for (const point of series) {
    assert.ok(Number.isFinite(point.binMean))
    assert.ok(Number.isFinite(point.binRms))
    assert.ok(Math.abs(point.positiveBinEnergy + point.negativeBinEnergy - 1) < 1e-12)
  }
  assert.throws(() => tokenSignalSeries(demoTrace.inference, -1), /outside the captured trace/)
})

test('training heatmaps retain metric and layer axes', () => {
  const rows = demoTrace.training.rows
  const metrics = normalizedMetricHeatmap(rows)
  assert.equal(metrics.cols, rows.length)
  assert.ok(metrics.rows >= 4)
  assert.equal(metrics.values.length, metrics.rows * metrics.cols)

  const gradients = layerGradientHeatmap(rows, demoTrace.model.num_layers)
  assert.equal(gradients.heatmap.rows, demoTrace.model.num_layers)
  assert.equal(gradients.heatmap.cols, gradients.samples)
})
