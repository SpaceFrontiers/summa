import assert from 'node:assert/strict'
import test from 'node:test'

import { demoTrace } from './demo-trace.js'
import { layerGradientHeatmap, normalizedMetricHeatmap, validateTrace } from './trace-utils.js'

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
