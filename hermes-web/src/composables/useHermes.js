import { ref, shallowRef } from 'vue'
import { useIpfs } from './useIpfs'
import { useUxConfig } from './useUxConfig'
import { parseUxConfig } from '../lib/uxConfigParser'
import { useConnectionsStore } from '../stores/connections'

let wasmModule = null
let RemoteIndex = null
let IpfsIndex = null

// Singleton promise for WASM initialization
const wasmInitPromise = (async () => {
  try {
    const module = await import('hermes-wasm')
    await module.default()
    module.setup_logging()
    wasmModule = module
    RemoteIndex = module.RemoteIndex
    IpfsIndex = module.IpfsIndex
    console.log('WASM module initialized')
    return true
  } catch (e) {
    console.error('Failed to load WASM module:', e.message)
    return false
  }
})()

/**
 * Initialize WASM module (can be called early for preloading)
 * Returns a singleton promise that resolves to true on success, false on failure
 */
export function initWasm() {
  return wasmInitPromise
}

export function useHermes() {
  const index = shallowRef(null)
  const isLoading = ref(false)
  const isConnected = ref(false)
  const error = ref(null)
  const indexInfo = ref(null)
  const networkStats = ref(null)
  const cacheStats = ref(null)
  const connectionType = ref(null) // 'http' or 'ipfs'
  const currentUrl = ref(null)
  const connectionProgress = ref(null) // Progress status during connection
  let abortController = null // AbortController for cancelling connection

  const ipfs = useIpfs()
  const uxConfig = useUxConfig()
  const connectionsStore = useConnectionsStore()

  const setProgress = (message, details = null) => {
    connectionProgress.value = { message, details }
    console.log(`[Connect] ${message}`, details || '')
  }

  const connect = async (serverUrl) => {
    error.value = null
    isLoading.value = true
    connectionProgress.value = null
    abortController = new AbortController()

    try {
      // Check if aborted
      const checkAborted = () => {
        if (abortController?.signal.aborted) {
          throw new Error('Connection aborted')
        }
      }

      // Helper to race a promise against abort signal
      const withAbort = (promise) => {
        return Promise.race([
          promise,
          new Promise((_, reject) => {
            abortController.signal.addEventListener('abort', () => {
              reject(new Error('Connection aborted'))
            })
          })
        ])
      }

      setProgress('Initializing WASM module...')
      const wasmReady = await withAbort(initWasm())
      if (!wasmReady) {
        error.value = 'Failed to load WASM module. Please refresh the page and try again.'
        isLoading.value = false
        connectionProgress.value = null
        return false
      }

      const normalizedUrl = serverUrl.replace(/\/+$/, '')
      let newIndex

      // Check if this is an IPFS URL
      if (ipfs.isIpfsUrl(normalizedUrl)) {
        // Initialize IPFS if needed
        if (!ipfs.isInitialized.value) {
          setProgress('Initializing IPFS gateway...')
          const ipfsReady = await withAbort(ipfs.init())
          if (!ipfsReady) {
            throw new Error(ipfs.error.value || 'Failed to initialize IPFS')
          }
        }

        // Create IPFS index with verified-fetch callbacks
        setProgress('Resolving IPNS name...')
        newIndex = new IpfsIndex(normalizedUrl)

        // Reset download stats
        ipfs.resetStats()

        const fetchFn = ipfs.createFetchFn()
        const sizeFn = ipfs.createSizeFn()

        setProgress('Loading index files...')
        await withAbort(newIndex.load(fetchFn, sizeFn))
        connectionType.value = 'ipfs'
      } else {
        // Standard HTTP connection
        setProgress('Connecting to server...')
        newIndex = new RemoteIndex(normalizedUrl)
        await withAbort(newIndex.load())
        connectionType.value = 'http'
      }

      // Try to restore cache from IndexedDB
      setProgress('Restoring cache...')
      try {
        const restored = await newIndex.load_cache_from_idb()
        if (restored) {
          console.log('Restored slice cache from IndexedDB')
        }
      } catch (e) {
        console.warn('Failed to restore cache from IndexedDB:', e)
      }

      index.value = newIndex
      isConnected.value = true
      currentUrl.value = normalizedUrl

      indexInfo.value = {
        numDocs: newIndex.num_docs(),
        numSegments: newIndex.num_segments(),
        fieldNames: newIndex.field_names() || [],
        defaultFields: newIndex.default_fields() || []
      }

      // Load UX config (optional, won't fail if not present)
      if (ipfs.isIpfsUrl(normalizedUrl)) {
        await uxConfig.load(ipfs.createFetchFn(), normalizedUrl)
      } else {
        // For HTTP, try to fetch ux.dsl
        try {
          const response = await fetch(`${normalizedUrl}/ux.dsl`)
          if (response.ok) {
            const text = await response.text()
            uxConfig.config.value = parseUxConfig(text)
            uxConfig.isLoaded.value = true
          }
        } catch {
          // ux.dsl is optional
        }
      }

      // Update connection label from ux.dsl short_name or title
      if (uxConfig.isLoaded.value && uxConfig.config.value) {
        const label = uxConfig.config.value.short_name || uxConfig.config.value.title
        if (label && label !== 'Hermes Search') {
          connectionsStore.updateLabel(normalizedUrl, label)
        }
      }

      setProgress('Connected!')
      updateStats()
      connectionProgress.value = null
      return true
    } catch (e) {
      const msg = e.message || String(e)

      // Parse error to give user-friendly feedback
      let userError
      if (msg.includes('schema.json') && msg.includes('does not exist')) {
        userError = 'This IPFS path does not contain a valid Hermes index. A Hermes index must have a schema.json file.'
      } else if (msg.includes('segments.json') && msg.includes('does not exist')) {
        userError = 'This IPFS path does not contain a valid Hermes index. Missing segments.json file.'
      } else if (msg.includes('Invalid SSTable magic')) {
        userError = 'The index files are corrupted or not in the correct format.'
      } else if (msg.includes('Content not found') || msg.includes('does not exist on IPFS')) {
        userError = 'Content not found on IPFS. The CID may be incorrect or the content is not pinned/available.'
      } else if (msg.includes('No peers found') || msg.includes('no providers')) {
        userError = 'No IPFS peers found serving this content. The content may not be pinned or available on the network.'
      } else if (msg.includes('timeout')) {
        userError = 'Connection timed out. The IPFS network may be slow or the content unavailable.'
      } else if (msg.includes('Failed to initialize IPFS')) {
        userError = 'Failed to initialize IPFS client. Check your network connection.'
      } else if (msg.includes('500') || msg.includes('Internal Server Error')) {
        userError = 'IPFS gateway error. The gateway may be overloaded or the content format is not supported. Try again later.'
      } else if (msg.includes('501') || msg.includes('Not Implemented')) {
        userError = 'IPFS gateway does not support this content type. The content may need to be accessed differently.'
      } else if (msg.includes('Connection aborted')) {
        // User cancelled - don't show error
        userError = null
      } else {
        userError = `Failed to connect: ${msg}`
      }

      if (userError) {
        error.value = userError
      }
      isConnected.value = false
      index.value = null
      connectionType.value = null
      connectionProgress.value = null
      return false
    } finally {
      isLoading.value = false
    }
  }

  const abortConnect = () => {
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    isLoading.value = false
    connectionProgress.value = null
  }

  const disconnect = () => {
    index.value = null
    isConnected.value = false
    indexInfo.value = null
    networkStats.value = null
    cacheStats.value = null
    currentUrl.value = null
    uxConfig.reset()
    connectionType.value = null
    error.value = null
  }

  const search = async (query, limit = 10, offset = 0) => {
    if (!index.value) {
      throw new Error('Index not connected')
    }

    const results = await index.value.search_offset(query, limit, offset)
    updateStats()

    // Save cache to IndexedDB after search (non-blocking)
    saveCache()

    return results
  }

  const saveCache = async () => {
    if (!index.value) return
    try {
      await index.value.save_cache_to_idb()
      console.log('Saved slice cache to IndexedDB')
    } catch (e) {
      console.warn('Failed to save cache to IndexedDB:', e)
    }
  }

  const clearCache = async () => {
    if (!index.value) return
    try {
      await index.value.clear_idb_cache()
      console.log('Cleared IndexedDB cache')
    } catch (e) {
      console.warn('Failed to clear IndexedDB cache:', e)
    }
  }

  const getDocument = async (segmentId, docId) => {
    if (!index.value) {
      throw new Error('Index not connected')
    }

    const doc = await index.value.get_document(segmentId, docId)
    updateStats()
    return doc
  }

  const updateStats = () => {
    if (!index.value) return

    networkStats.value = index.value.network_stats()
    cacheStats.value = index.value.cache_stats()
  }

  const resetNetworkStats = () => {
    if (index.value) {
      index.value.reset_network_stats()
      updateStats()
    }
  }

  return {
    index,
    isLoading,
    isConnected,
    connectionType,
    currentUrl,
    connectionProgress,
    downloadStats: ipfs.getStats(),
    error,
    indexInfo,
    networkStats,
    cacheStats,
    uxConfig,
    connect,
    abortConnect,
    disconnect,
    search,
    getDocument,
    updateStats,
    resetNetworkStats,
    saveCache,
    clearCache
  }
}
