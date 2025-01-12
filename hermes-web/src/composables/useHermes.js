import { ref, shallowRef } from 'vue'
import { useIpfs } from './useIpfs'

let wasmModule = null
let RemoteIndex = null
let IpfsIndex = null

export function useHermes() {
  const index = shallowRef(null)
  const isLoading = ref(false)
  const isConnected = ref(false)
  const error = ref(null)
  const indexInfo = ref(null)
  const networkStats = ref(null)
  const cacheStats = ref(null)
  const connectionType = ref(null) // 'http' or 'ipfs'
  
  const ipfs = useIpfs()

  const initWasm = async () => {
    if (wasmModule) return true
    
    try {
      const module = await import('hermes-wasm')
      await module.default()
      module.setup_logging()
      wasmModule = module
      RemoteIndex = module.RemoteIndex
      IpfsIndex = module.IpfsIndex
      return true
    } catch (e) {
      error.value = `Failed to load WASM module: ${e.message}`
      return false
    }
  }

  const connect = async (serverUrl) => {
    error.value = null
    isLoading.value = true
    
    try {
      const wasmReady = await initWasm()
      if (!wasmReady) {
        isLoading.value = false
        return false
      }

      const normalizedUrl = serverUrl.replace(/\/+$/, '')
      let newIndex
      
      // Check if this is an IPFS URL
      if (ipfs.isIpfsUrl(normalizedUrl)) {
        // Initialize IPFS if needed
        if (!ipfs.isInitialized.value) {
          console.log('Initializing IPFS...')
          const ipfsReady = await ipfs.init()
          if (!ipfsReady) {
            throw new Error(ipfs.error.value || 'Failed to initialize IPFS')
          }
        }
        
        // Create IPFS index with verified-fetch callbacks
        newIndex = new IpfsIndex(normalizedUrl)
        const fetchFn = ipfs.createFetchFn()
        const sizeFn = ipfs.createSizeFn()
        await newIndex.load(fetchFn, sizeFn)
        connectionType.value = 'ipfs'
      } else {
        // Standard HTTP connection
        newIndex = new RemoteIndex(normalizedUrl)
        await newIndex.load()
        connectionType.value = 'http'
      }
      
      // Try to restore cache from IndexedDB
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
      
      indexInfo.value = {
        numDocs: newIndex.num_docs(),
        numSegments: newIndex.num_segments(),
        fieldNames: newIndex.field_names() || [],
        defaultFields: newIndex.default_fields() || []
      }
      
      updateStats()
      return true
    } catch (e) {
      error.value = `Failed to connect: ${e.message || e}`
      isConnected.value = false
      index.value = null
      connectionType.value = null
      return false
    } finally {
      isLoading.value = false
    }
  }

  const disconnect = () => {
    index.value = null
    isConnected.value = false
    indexInfo.value = null
    networkStats.value = null
    cacheStats.value = null
    connectionType.value = null
    error.value = null
  }

  const search = async (query, limit = 10) => {
    if (!index.value) {
      throw new Error('Index not connected')
    }
    
    const results = await index.value.search(query, limit)
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
    error,
    indexInfo,
    networkStats,
    cacheStats,
    connect,
    disconnect,
    search,
    getDocument,
    updateStats,
    resetNetworkStats,
    saveCache,
    clearCache
  }
}
