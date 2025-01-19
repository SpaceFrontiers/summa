import { ref } from 'vue'
import { useAppConfig } from './useAppConfig'
import { createDownloadManager } from '../lib/downloadManager'

// Singleton state
const isInitialized = ref(false)
const isInitializing = ref(false)
const error = ref(null)
let downloadManager = null

/**
 * IPFS composable using HTTP gateway fetching
 *
 * Uses download manager for resumable downloads with gateway fallback.
 *
 * Supports:
 * - /ipfs/<CID> - Direct IPFS content addressing
 * - /ipns/<name> - IPNS name resolution
 * - ipfs://<CID> - IPFS URI scheme
 * - ipns://<name> - IPNS URI scheme
 */
export function useIpfs() {
  const appConfig = useAppConfig()

  // Download manager handles all network operations (singleton)
  const getDownloadManager = () => {
    if (!downloadManager) {
      downloadManager = createDownloadManager(appConfig)
    }
    return downloadManager
  }

  /**
   * Initialize IPFS HTTP gateway fetcher
   */
  const init = async () => {
    if (isInitialized.value || isInitializing.value) return isInitialized.value

    isInitializing.value = true
    error.value = null

    try {
      // Ensure app config detection has run
      if (!appConfig.isDetected.value) {
        appConfig.detect()
      }

      // Initialize download manager
      getDownloadManager()

      // Log gateway configuration
      if (appConfig.isLocalDaemon.value) {
        console.log('IPFS: Using local daemon at', appConfig.gatewayUrl.value)
      } else if (appConfig.isIpfsGateway.value) {
        console.log('IPFS: Using serving gateway at', appConfig.gatewayUrl.value)
      } else {
        console.log('IPFS: Using remote gateways')
      }

      isInitialized.value = true
      return true
    } catch (e) {
      error.value = `Failed to initialize IPFS: ${e.message}`
      console.error('Failed to initialize IPFS:', e)
      return false
    } finally {
      isInitializing.value = false
    }
  }

  /**
   * Parse an IPFS/IPNS URL or local path and extract the path
   *
   * @param {string} url - URL to parse
   * @returns {{ type: 'ipfs'|'ipns'|'local'|null, path: string, original: string }}
   */
  const parseIpfsUrl = (url) => {
    const trimmed = url.trim()

    // Handle ipfs:// and ipns:// URI schemes
    if (trimmed.startsWith('ipfs://')) {
      return { type: 'ipfs', path: trimmed.slice(7), original: trimmed }
    }
    if (trimmed.startsWith('ipns://')) {
      return { type: 'ipns', path: trimmed.slice(7), original: trimmed }
    }

    // Handle /ipfs/ and /ipns/ paths
    if (trimmed.startsWith('/ipfs/')) {
      return { type: 'ipfs', path: trimmed.slice(6), original: trimmed }
    }
    if (trimmed.startsWith('/ipns/')) {
      return { type: 'ipns', path: trimmed.slice(6), original: trimmed }
    }

    // Local path starting with / (e.g., /s)
    if (trimmed.startsWith('/')) {
      return { type: 'local', path: trimmed, original: trimmed }
    }

    // Not an IPFS URL
    return { type: null, path: null, original: trimmed }
  }

  /**
   * Check if a URL is an IPFS/IPNS URL
   */
  const isIpfsUrl = (url) => {
    const { type } = parseIpfsUrl(url)
    return type !== null
  }

  /**
   * Normalize IPFS path to URI format (ipfs://... or ipns://...)
   */
  const toIpfsUri = (url) => {
    const { type, path } = parseIpfsUrl(url)
    if (!type) return url
    return `${type}://${path}`
  }

  /**
   * Parse IPFS error messages into user-friendly format
   */
  const parseIpfsError = (error, path) => {
    const msg = error.message || String(error)

    if (msg.includes('no providers found')) {
      return `Content not available: No peers found serving "${path}". The content may not be pinned or the CID may be incorrect.`
    }
    if (msg.includes('timeout')) {
      return `IPFS timeout: Could not retrieve "${path}" within the time limit. Try again or check if the content is available.`
    }
    if (msg.includes('invalid CID') || msg.includes('invalid path')) {
      return `Invalid IPFS path: "${path}" is not a valid CID or IPFS path.`
    }
    if (msg.includes('not found') || msg.includes('404')) {
      return `Content not found: "${path}" does not exist on IPFS.`
    }
    if (msg.includes('500') || msg.includes('Internal Server Error')) {
      return `IPFS gateway error for "${path}". The gateway may be overloaded. Try again later.`
    }
    if (msg.includes('501') || msg.includes('Not Implemented')) {
      return `IPFS gateway does not support this request type for "${path}".`
    }

    return `IPFS error for "${path}": ${msg}`
  }


  /**
   * Create fetch function for IpfsIndex using HTTP gateways
   * Returns a function: (path: string, rangeStart?: number, rangeEnd?: number) => Promise<Uint8Array>
   */
  const createFetchFn = () => {
    if (!isInitialized.value) {
      throw new Error('IPFS not initialized. Call init() first.')
    }

    const dm = getDownloadManager()

    return async (path, rangeStart, rangeEnd) => {
      const hasRange = rangeStart !== undefined && rangeEnd !== undefined
      console.log('IPFS fetch:', path, hasRange ? `[${rangeStart}-${rangeEnd}]` : '(full)')

      try {
        const data = await dm.download(path, { rangeStart, rangeEnd })
        return data
      } catch (e) {
        const friendlyError = parseIpfsError(e, path)
        console.error('IPFS fetch error:', friendlyError)
        throw new Error(friendlyError)
      }
    }
  }

  /**
   * Create size function for IpfsIndex using HTTP gateways
   * Returns a function: (path: string) => Promise<number>
   */
  const createSizeFn = () => {
    if (!isInitialized.value) {
      throw new Error('IPFS not initialized. Call init() first.')
    }

    const dm = getDownloadManager()
    const sizeCache = new Map()

    return async (path) => {
      if (sizeCache.has(path)) {
        return sizeCache.get(path)
      }

      console.log('IPFS size:', path)

      try {
        const size = await dm.getSize(path)
        sizeCache.set(path, size)
        return size
      } catch (e) {
        const friendlyError = parseIpfsError(e, path)
        console.error('IPFS size error:', friendlyError)
        throw new Error(friendlyError)
      }
    }
  }

  /**
   * Get network stats from download manager
   */
  const getStats = () => {
    return getDownloadManager().stats
  }

  /**
   * Reset network stats
   */
  const resetStats = () => {
    getDownloadManager().resetStats()
  }

  return {
    isInitialized,
    isInitializing,
    error,
    init,
    parseIpfsUrl,
    isIpfsUrl,
    toIpfsUri,
    createFetchFn,
    createSizeFn,
    getStats,
    resetStats
  }
}
