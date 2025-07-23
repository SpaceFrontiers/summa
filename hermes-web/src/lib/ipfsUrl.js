const CONTENT_PREFIXES = [
  ['ipfs://', 'ipfs'],
  ['ipns://', 'ipns'],
  ['/ipfs/', 'ipfs'],
  ['/ipns/', 'ipns']
]

/**
 * Parse an IPFS/IPNS URI, gateway path, or local absolute path.
 *
 * Keeping this protocol-only parser independent from Vue makes the URL rules
 * reusable by both the composable and the download manager.
 *
 * @param {string} url
 * @returns {{ type: 'ipfs' | 'ipns' | 'local' | null, path: string, original: string }}
 */
export function parseContentPath(url) {
  const original = url.trim()

  for (const [prefix, type] of CONTENT_PREFIXES) {
    if (original.startsWith(prefix)) {
      return { type, path: original.slice(prefix.length), original }
    }
  }

  if (original.startsWith('/')) {
    return { type: 'local', path: original, original }
  }

  return { type: null, path: original, original }
}
