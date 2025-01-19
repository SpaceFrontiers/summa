import { ref } from 'vue'

// Configuration
const MAX_RETRIES = 10
const RETRY_DELAY = 1000
const MAX_503_RETRIES = 3
const RETRY_503_DELAY = 2000

// Remote IPFS gateways
const REMOTE_GATEWAYS = [
  'https://ipfs.io',
  'https://dweb.link',
  'https://cloudflare-ipfs.com',
]

/**
 * Download Manager with resumable downloads and gateway fallback
 *
 * Handles:
 * - Resumable downloads that continue after interruption
 * - Multiple gateway fallback
 * - Progress tracking
 * - Network stats
 */
export function createDownloadManager(appConfig) {
  // Network stats (tracks actual network activity)
  const stats = ref({
    requestCount: 0,
    bytesDownloaded: 0,
    currentDownload: null  // { file, bytesDownloaded, totalBytes }
  })

  const resetStats = () => {
    stats.value = {
      requestCount: 0,
      bytesDownloaded: 0,
      currentDownload: null
    }
  }

  /**
   * Parse IPFS/IPNS URL or local path
   */
  const parseIpfsUrl = (url) => {
    const trimmed = url.trim()

    if (trimmed.startsWith('ipfs://')) {
      return { type: 'ipfs', path: trimmed.slice(7) }
    }
    if (trimmed.startsWith('ipns://')) {
      return { type: 'ipns', path: trimmed.slice(7) }
    }
    if (trimmed.startsWith('/ipfs/')) {
      return { type: 'ipfs', path: trimmed.slice(6) }
    }
    if (trimmed.startsWith('/ipns/')) {
      return { type: 'ipns', path: trimmed.slice(6) }
    }

    // Local path starting with / (e.g., /s/file.bin)
    if (trimmed.startsWith('/')) {
      return { type: 'local', path: trimmed }
    }

    return { type: null, path: trimmed }
  }

  /**
   * Build gateway URL for a path
   * Local daemon uses path format: localhost:8080/ipfs/cid or localhost:8080/ipns/name
   * Remote gateways use path format: gateway.io/ipfs/cid
   * Local paths use current origin: /s/file.bin -> origin/s/file.bin
   * On IPFS path gateway: local paths are relative to /ipfs/CID/
   */
  const buildGatewayUrl = (path, gateway = null) => {
    const { type, path: ipfsPath } = parseIpfsUrl(path)
    if (!type) return null

    // Local paths - needs special handling on IPFS gateways
    if (type === 'local') {
      // On IPFS path gateway (e.g., https://gateway.io/ipfs/CID/),
      // local paths must be prefixed with /ipfs/CID
      if (appConfig.isIpfsGateway.value && appConfig.ipfsCid.value) {
        const baseUrl = appConfig.getBaseUrl()
        return `${baseUrl}${ipfsPath}`
      }
      return ipfsPath  // Regular server: browser resolves relative to origin
    }

    // Use specific gateway if provided
    if (gateway) {
      return `${gateway}/${type}/${ipfsPath}`
    }

    // Local daemon - use localDaemonUrl (root gateway, not subdomain)
    if (appConfig.isLocalDaemon.value && appConfig.localDaemonUrl.value) {
      return `${appConfig.localDaemonUrl.value}/${type}/${ipfsPath}`
    }

    // IPFS gateway - use path format
    if (appConfig.isIpfsGateway.value) {
      return `${appConfig.gatewayUrl.value}/${type}/${ipfsPath}`
    }

    return null
  }

  /**
   * Get list of gateway URLs to try for a path
   */
  const getGatewayUrls = (path) => {
    const { type, path: ipfsPath } = parseIpfsUrl(path)
    if (!type) return []

    const urls = []

    // Local paths - use buildGatewayUrl to handle IPFS gateway prefix
    if (type === 'local') {
      const localUrl = buildGatewayUrl(path)
      if (localUrl) urls.push({ url: localUrl, name: 'local' })
      return urls
    }

    if (appConfig.isLocalDaemon.value) {
      const localUrl = buildGatewayUrl(path)
      if (localUrl) urls.push({ url: localUrl, name: 'local' })
    } else {
      const primaryUrl = buildGatewayUrl(path)
      if (primaryUrl) urls.push({ url: primaryUrl, name: 'primary' })
      for (const gateway of REMOTE_GATEWAYS) {
        urls.push({ url: `${gateway}/${type}/${ipfsPath}`, name: gateway })
      }
    }

    return urls
  }

  /**
   * Resumable fetch - continues interrupted downloads using Range requests
   */
  const resumableFetch = async (url, options = {}) => {
    const { headers = {}, onProgress = null } = options
    let downloadedBytes = new Uint8Array(0)
    let attempt = 0
    let totalSize = null

    while (attempt < MAX_RETRIES) {
      attempt++
      const startByte = downloadedBytes.length

      try {
        const requestHeaders = { ...headers }

        // Resume from where we left off
        if (startByte > 0) {
          requestHeaders['Range'] = `bytes=${startByte}-`
          console.log(`Resuming download from byte ${startByte}`)
        }

        const response = await fetch(url, { headers: requestHeaders })

        // Already complete
        if (response.status === 416) {
          console.log('Download already complete')
          break
        }

        if (!response.ok && response.status !== 206) {
          throw new Error(`HTTP ${response.status}`)
        }

        // Get total size from headers
        if (!totalSize) {
          const contentRange = response.headers.get('Content-Range')
          if (contentRange) {
            const match = contentRange.match(/\/(\d+)$/)
            if (match) totalSize = parseInt(match[1], 10)
          } else {
            const contentLength = response.headers.get('Content-Length')
            if (contentLength) totalSize = parseInt(contentLength, 10) + startByte
          }
        }

        // Stream the response
        const reader = response.body.getReader()
        const chunks = []
        let receivedLength = 0

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value)
          receivedLength += value.length

          if (onProgress) {
            onProgress(startByte + receivedLength, totalSize)
          }
        }

        // Combine chunks
        const newData = new Uint8Array(receivedLength)
        let position = 0
        for (const chunk of chunks) {
          newData.set(chunk, position)
          position += chunk.length
        }

        // Append to existing data
        if (startByte > 0) {
          const combined = new Uint8Array(downloadedBytes.length + newData.length)
          combined.set(downloadedBytes, 0)
          combined.set(newData, downloadedBytes.length)
          downloadedBytes = combined
        } else {
          downloadedBytes = newData
        }

        // Check completion
        if (totalSize && downloadedBytes.length >= totalSize) {
          console.log(`Download complete: ${downloadedBytes.length} bytes`)
          break
        }

        if (!totalSize) {
          console.log(`Download complete (no size header): ${downloadedBytes.length} bytes`)
          break
        }

      } catch (e) {
        const isNetworkError = e.message.includes('Load failed') ||
                               e.message.includes('network') ||
                               e.name === 'TypeError'

        if (isNetworkError && downloadedBytes.length > 0) {
          const delay = RETRY_DELAY * Math.min(attempt, 5)
          console.log(`Download interrupted at ${downloadedBytes.length} bytes (attempt ${attempt}/${MAX_RETRIES}). Retrying in ${delay}ms...`)
          await new Promise(resolve => setTimeout(resolve, delay))
          continue
        }

        if (attempt >= MAX_RETRIES) {
          throw new Error(`Download failed after ${MAX_RETRIES} attempts: ${e.message}`)
        }

        const delay = RETRY_DELAY * Math.min(attempt, 5)
        console.log(`Fetch error (attempt ${attempt}/${MAX_RETRIES}): ${e.message}. Retrying in ${delay}ms...`)
        await new Promise(resolve => setTimeout(resolve, delay))
      }
    }

    if (downloadedBytes.length === 0) {
      throw new Error('Download failed: no data received')
    }

    return downloadedBytes
  }

  /**
   * Simple fetch for range requests (no resumption needed)
   * Includes retry logic for 503 errors
   */
  const simpleFetch = async (url, headers) => {
    let lastError

    for (let attempt = 0; attempt < MAX_503_RETRIES; attempt++) {
      const response = await fetch(url, { headers })

      // 206 Partial Content is expected for range requests
      if (response.ok || response.status === 206) {
        const buffer = await response.arrayBuffer()
        return new Uint8Array(buffer)
      }

      // Retry on 503 Service Unavailable
      if (response.status === 503 && attempt < MAX_503_RETRIES - 1) {
        const delay = RETRY_503_DELAY * (attempt + 1)
        console.log(`503 error, retrying in ${delay}ms (attempt ${attempt + 1}/${MAX_503_RETRIES})...`)
        await new Promise(resolve => setTimeout(resolve, delay))
        continue
      }

      lastError = new Error(`HTTP ${response.status}`)
    }

    throw lastError
  }

  /**
   * Download a file with gateway fallback and resumable support
   */
  const download = async (path, options = {}) => {
    const { rangeStart, rangeEnd } = options
    const hasRange = rangeStart !== undefined && rangeEnd !== undefined
    const fileName = path.split('/').pop()

    const headers = {}
    if (hasRange) {
      headers['Range'] = `bytes=${rangeStart}-${rangeEnd - 1}`
    }

    const urls = getGatewayUrls(path)
    if (urls.length === 0) {
      throw new Error('No gateways available')
    }

    // Progress callback (only for full file downloads)
    const onProgress = hasRange ? null : (bytesDownloaded, totalBytes) => {
      stats.value = {
        ...stats.value,
        currentDownload: { file: fileName, bytesDownloaded, totalBytes }
      }
    }

    // Try each gateway with 503 retry logic
    const errors = []
    for (const { url, name } of urls) {
      let lastError

      for (let attempt = 0; attempt < MAX_503_RETRIES; attempt++) {
        try {
          console.log(`Trying gateway: ${url}${hasRange ? ` [${rangeStart}-${rangeEnd}]` : ''}${attempt > 0 ? ` (retry ${attempt})` : ''}`)

          // Use simple fetch for range requests, resumable fetch for full files
          const data = hasRange
            ? await simpleFetch(url, headers)
            : await resumableFetch(url, { headers, onProgress })

          console.log(`Success: ${name} (${url})`)

          // Update stats
          stats.value = {
            requestCount: stats.value.requestCount + 1,
            bytesDownloaded: stats.value.bytesDownloaded + data.length,
            currentDownload: null
          }

          return data
        } catch (e) {
          lastError = e

          // Retry on 503 errors
          if (e.message.includes('503') && attempt < MAX_503_RETRIES - 1) {
            const delay = RETRY_503_DELAY * (attempt + 1)
            console.log(`503 error from ${name}, retrying in ${delay}ms (attempt ${attempt + 1}/${MAX_503_RETRIES})...`)
            await new Promise(resolve => setTimeout(resolve, delay))
            continue
          }

          break
        }
      }

      errors.push(`${name}: ${lastError.message}`)
      console.log(`Gateway failed: ${name} - ${lastError.message}`)
    }

    const notFoundError = errors.find(err => err.includes('File not found') || err.includes('HTTP 404'))
    if (notFoundError) {
      throw new Error(`File not found: ${path}`)
    }
    throw new Error(`All gateways failed: ${errors.join(', ')}`)
  }

  /**
   * Get file size via HEAD request
   */
  const getSize = async (path) => {
    const urls = getGatewayUrls(path)

    for (const { url, name } of urls) {
      try {
        const response = await fetch(url, {
          method: 'HEAD',
          signal: AbortSignal.timeout(10000)
        })
        if (response.ok) {
          const contentLength = response.headers.get('Content-Length')
          if (contentLength) {
            const size = parseInt(contentLength, 10)
            console.log(`Size for ${path}: ${size} (from ${name} HEAD)`)
            return size
          } else {
            console.log(`HEAD ${name}: no Content-Length header`)
          }
        } else {
          console.log(`HEAD ${name}: HTTP ${response.status}`)
        }
      } catch (e) {
        console.log(`HEAD ${name} failed: ${e.message}`)
      }
    }

    // Fallback: download full file (this is expensive!)
    console.warn(`HEAD failed for ${path}, fetching full content (fallback)`)
    const data = await download(path)
    return data.length
  }

  return {
    stats,
    resetStats,
    download,
    getSize,
    parseIpfsUrl,
    buildGatewayUrl
  }
}
