import assert from 'node:assert/strict'
import test from 'node:test'

import { parseContentPath } from './ipfsUrl.js'

test('parseContentPath accepts URI, gateway, and local path forms', () => {
  const cases = [
    [' ipfs://bafy/index ', { type: 'ipfs', path: 'bafy/index', original: 'ipfs://bafy/index' }],
    ['ipns://docs.example/v1', { type: 'ipns', path: 'docs.example/v1', original: 'ipns://docs.example/v1' }],
    ['/ipfs/bafy/index', { type: 'ipfs', path: 'bafy/index', original: '/ipfs/bafy/index' }],
    ['/ipns/docs.example/v1', { type: 'ipns', path: 'docs.example/v1', original: '/ipns/docs.example/v1' }],
    ['/index', { type: 'local', path: '/index', original: '/index' }],
    ['https://example.test/index', {
      type: null,
      path: 'https://example.test/index',
      original: 'https://example.test/index'
    }]
  ]

  for (const [input, expected] of cases) {
    assert.deepEqual(parseContentPath(input), expected)
  }
})

test('protocol matching is case-sensitive like URL scheme handling was previously', () => {
  assert.deepEqual(parseContentPath('IPFS://bafy'), {
    type: null,
    path: 'IPFS://bafy',
    original: 'IPFS://bafy'
  })
})
