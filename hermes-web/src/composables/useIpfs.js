import { ref } from 'vue'

let verifiedFetch = null

/**
 * IPFS composable for handling IPFS/IPNS URLs with Helia verified-fetch
 * 
 * Supports:
 * - /ipfs/<CID> - Direct IPFS content addressing
 * - /ipns/<name> - IPNS name resolution
 * - ipfs://<CID> - IPFS URI scheme
 * - ipns://<name> - IPNS URI scheme
 */
export function useIpfs() {
  const isInitialized = ref(false)
  const isInitializing = ref(false)
  const error = ref(null)
  
  /**
   * Initialize Helia verified-fetch
   */
  const init = async () => {
    if (isInitialized.value || isInitializing.value) return isInitialized.value
    
    isInitializing.value = true
    error.value = null
    
    try {
      const { createVerifiedFetch } = await import('@helia/verified-fetch')
      verifiedFetch = await createVerifiedFetch()
      isInitialized.value = true
      console.log('Helia verified-fetch initialized')
      return true
    } catch (e) {
      error.value = `Failed to initialize IPFS: ${e.message}`
      console.error('Failed to initialize Helia:', e)
      return false
    } finally {
      isInitializing.value = false
    }
  }
  
  /**
   * Parse an IPFS/IPNS URL and extract the path
   * 
   * @param {string} url - URL to parse
   * @returns {{ type: 'ipfs'|'ipns'|null, path: string, original: string }}
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
    
    return `IPFS error for "${path}": ${msg}`
  }

  /**
   * Create fetch function for IpfsIndex
   * Returns a function: (path: string) => Promise<Uint8Array>
   */
  const createFetchFn = () => {
    if (!verifiedFetch) {
      throw new Error('IPFS not initialized. Call init() first.')
    }
    
    return async (path) => {
      console.log('IPFS fetch:', path)
      const uri = toIpfsUri(path)
      
      try {
        const response = await verifiedFetch(uri)
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
        
        const buffer = await response.arrayBuffer()
        return new Uint8Array(buffer)
      } catch (e) {
        const friendlyError = parseIpfsError(e, path)
        console.error('IPFS fetch error:', friendlyError)
        throw new Error(friendlyError)
      }
    }
  }
  
  /**
   * Create size function for IpfsIndex
   * Returns a function: (path: string) => Promise<number>
   * 
   * Note: IPFS doesn't have a native HEAD-like operation, so we fetch
   * the content and return its length. The SliceCachingDirectory will
   * cache this for subsequent reads.
   */
  const createSizeFn = () => {
    if (!verifiedFetch) {
      throw new Error('IPFS not initialized. Call init() first.')
    }
    
    // Cache sizes to avoid re-fetching
    const sizeCache = new Map()
    
    return async (path) => {
      if (sizeCache.has(path)) {
        return sizeCache.get(path)
      }
      
      console.log('IPFS size:', path)
      const uri = toIpfsUri(path)
      
      try {
        const response = await verifiedFetch(uri)
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }
        
        const buffer = await response.arrayBuffer()
        const size = buffer.byteLength
        sizeCache.set(path, size)
        return size
      } catch (e) {
        const friendlyError = parseIpfsError(e, path)
        console.error('IPFS size error:', friendlyError)
        throw new Error(friendlyError)
      }
    }
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
    createSizeFn
  }
}
